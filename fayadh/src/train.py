"""
train.py — Training loop with cross-validation, experiment logging, and model comparison.
"""

import numpy as np
import pandas as pd
import time

from src.config import (
    TARGET_COL, DATETIME_COL, STATION_COL, FULL_TRAIN_HOLDOUT_FRAC,
    RAINY_SEASON_WEIGHT, CLIMATOLOGY_MIN_SAMPLES,
)
from src.validation import TimeSeriesCV
from src.models import BaseModel, get_model
from src.utils import (
    evaluate, print_metrics, log_experiment, plot_learning_curves
)

CLIMATOLOGY_COL = "station_month_climatology"


class StationMonthClimatology:
    """
    Leak-safe seasonal prior: historical mean tma_mdpl per (station, month).

    Must be fit ONLY on the training portion of a fold/split — never on
    validation or test targets. Falls back to the station's overall mean
    (then the global mean) when a (station, month) cell has too few
    observations, so it never produces NaN.

    This exists because target-lag/rolling features are disabled (see
    config.USE_TARGET_HISTORY_FEATURES) — the model otherwise has zero
    autoregressive signal for the ~8 month blind forecast horizon. This
    feature IS available for every future timestamp since it only needs
    station + calendar month, not recent target history.
    """

    def __init__(self, min_samples: int = CLIMATOLOGY_MIN_SAMPLES):
        self.min_samples = min_samples
        self.station_month_mean_ = None
        self.station_mean_ = None
        self.global_mean_ = None

    def fit(self, df: pd.DataFrame):
        d = df[[STATION_COL, DATETIME_COL, TARGET_COL]].dropna(subset=[TARGET_COL]).copy()
        d["month"] = pd.to_datetime(d[DATETIME_COL]).dt.month

        counts = d.groupby([STATION_COL, "month"])[TARGET_COL].transform("count")
        grp_mean = d.groupby([STATION_COL, "month"])[TARGET_COL].mean()
        # Only keep cells with enough samples; sparse cells fall back downstream
        grp_mean = grp_mean[
            d.groupby([STATION_COL, "month"])[TARGET_COL].count() >= self.min_samples
        ]
        self.station_month_mean_ = grp_mean.to_dict()

        self.station_mean_ = d.groupby(STATION_COL)[TARGET_COL].mean().to_dict()
        self.global_mean_ = d[TARGET_COL].mean()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        month = pd.to_datetime(df[DATETIME_COL]).dt.month

        def lookup(station, m):
            val = self.station_month_mean_.get((station, m))
            if val is not None:
                return val
            val = self.station_mean_.get(station)
            if val is not None:
                return val
            return self.global_mean_

        df[CLIMATOLOGY_COL] = [
            lookup(s, m) for s, m in zip(df[STATION_COL].values, month.values)
        ]
        return df


def _drop_leaky_target_history_cols(feature_cols: list) -> list:
    """Defensively strip tma_lag_/tma_roll_/tma_diff_ columns (see
    config.USE_TARGET_HISTORY_FEATURES) even if a caller passed a stale
    feature list that still contains them."""
    return [
        c for c in feature_cols
        if not c.startswith("tma_lag_")
        and not c.startswith("tma_roll_")
        and not c.startswith("tma_diff_")
    ]


def make_sample_weight(df: pd.DataFrame) -> np.ndarray:
    """Upweight rainy-season rows (see config.RAINY_SEASON_WEIGHT)."""
    if "musim_hujan" in df.columns:
        return np.where(df["musim_hujan"].values == 1, RAINY_SEASON_WEIGHT, 1.0)
    return np.ones(len(df))


class TargetScaler:
    """Scales target variable (tma_mdpl) per station using Z-score.

    Falls back to the GLOBAL (all-stations-pooled) mean/std for any station
    not present in the fit data. This matters for stations with a short
    history (e.g. "Gunungsari" only starts reporting 2024-08-13): in CV
    fold 0 (train ends 2024-06-30) it has zero training rows, so it has no
    per-station stats. The old code silently fell back to treating the
    RAW target/scaled-prediction as already-scaled/already-raw (a no-op
    passthrough), which is wrong on both ends and corrupted both the
    validation eval metric and inverse-transformed predictions for that
    station by ~10x during that fold (predicted ~0.6 vs actual ~13.5).
    """
    def __init__(self):
        self.stats = {}
        self.global_mean = 0.0
        self.global_std = 1.0

    def fit(self, y, stations):
        df = pd.DataFrame({'y': y, 'station': stations})
        self.stats = df.groupby('station')['y'].agg(['mean', 'std']).to_dict(orient='index')
        self.global_mean = float(df['y'].mean())
        self.global_std = float(df['y'].std())
        if not self.global_std or np.isnan(self.global_std):
            self.global_std = 1.0
        return self

    def _mean_std(self, st):
        if st in self.stats and self.stats[st]['std'] and not np.isnan(self.stats[st]['std']) and self.stats[st]['std'] > 0:
            return self.stats[st]['mean'], self.stats[st]['std']
        return self.global_mean, self.global_std

    def transform(self, y, stations):
        res = y.copy() if isinstance(y, np.ndarray) else y.values.copy()
        res_scaled = np.zeros_like(res, dtype=float)
        for i, (val, st) in enumerate(zip(res, stations)):
            mean, std = self._mean_std(st)
            res_scaled[i] = (val - mean) / std
        return res_scaled

    def inverse_transform(self, y_scaled, stations):
        res = np.zeros_like(y_scaled, dtype=float)
        for i, (val, st) in enumerate(zip(y_scaled, stations)):
            mean, std = self._mean_std(st)
            res[i] = (val * std) + mean
        return res


def perform_feature_selection(model, X_train, y_train, feature_cols, categorical_features=None,
                               sample_weight=None):
    """Perform a fast fit to select the most important features."""
    print(f"  [Feature Selection] Running fast fit for {model.name}...")
    # Fast fit parameters
    fast_params = model.params.copy()
    if 'n_estimators' in fast_params: fast_params['n_estimators'] = 100
    if 'iterations' in fast_params: fast_params['iterations'] = 100
    if 'max_epochs' in fast_params: fast_params['max_epochs'] = 5

    fast_model = get_model(model.name, params=fast_params)
    fast_model.fit(X_train, y_train, categorical_features=categorical_features, sample_weight=sample_weight)

    feat_imp = fast_model.get_feature_importance(feature_cols)
    if feat_imp is not None and not feat_imp.empty:
        # Keep top 70% features or features with importance > 0
        threshold = feat_imp['importance'].quantile(0.3) 
        if threshold == 0:
            threshold = 1e-6 # Drop strict zeros
        selected = feat_imp[feat_imp['importance'] > threshold]['feature'].tolist()
        print(f"  [Feature Selection] Selected {len(selected)} / {len(feature_cols)} features.")
        return selected
    return feature_cols

def run_cv(
    model: BaseModel,
    df: pd.DataFrame,
    feature_cols: list,
    cv: TimeSeriesCV = None,
    categorical_features: list = None,
    verbose: bool = True,
    select_features: bool = True,
    use_residual_target: bool = False,
) -> dict:
    """
    use_residual_target : bool
        If True, the model is trained to predict (tma_mdpl - station_month_
        climatology) instead of raw tma_mdpl, with the climatology added
        back after inverse-scaling to get the final raw-scale prediction
        used for evaluation. Motivation: per-station prediction dispersion
        was found to be 2-8x lower than the true historical dispersion —
        the model was leaning on the climatology feature as an "easy"
        answer for the station/season baseline and under-fitting the
        weather-driven day-to-day signal. Separating the two explicitly
        (climatology handled additively, model only predicts the
        deviation) frees model capacity to chase that residual signal.
    """
    if cv is None:
        cv = TimeSeriesCV()

    # Defensively strip tma_lag_/tma_roll_/tma_diff_ columns (see
    # config.USE_TARGET_HISTORY_FEATURES) and make sure the leak-safe
    # seasonal climatology feature is always considered.
    feature_cols = _drop_leaky_target_history_cols(feature_cols)
    if CLIMATOLOGY_COL not in feature_cols:
        feature_cols = feature_cols + [CLIMATOLOGY_COL]

    fold_metrics = []
    oof_preds = []
    evals_results = []
    station_rmses = []
    start_time = time.time()

    # Store the first fold scaler
    master_scaler = None
    selected_features = feature_cols.copy()

    # Drop rows with NaN target globally
    df = df.dropna(subset=[TARGET_COL]).copy()

    for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(df)):
        train_df = df.iloc[train_idx].copy()
        valid_df = df.iloc[valid_idx].copy()

        # Seasonal climatology — fit ONLY on this fold's train rows, then
        # apply to both train and valid. Never fit on validation targets.
        climatology = StationMonthClimatology().fit(train_df)
        train_df = climatology.transform(train_df)
        valid_df = climatology.transform(valid_df)

        sample_weight_train = make_sample_weight(train_df)

        X_train_full = train_df[feature_cols]
        y_train_raw = train_df[TARGET_COL]
        y_train = (
            train_df[TARGET_COL] - train_df[CLIMATOLOGY_COL]
            if use_residual_target else y_train_raw
        )
        stations_train = train_df[STATION_COL].values

        # Feature Selection on Fold 0
        if select_features and fold_idx == 0:
            scaler = TargetScaler().fit(y_train, stations_train)
            y_train_scaled = scaler.transform(y_train, stations_train)
            selected_features = perform_feature_selection(
                model, X_train_full, y_train_scaled, feature_cols, categorical_features,
                sample_weight=sample_weight_train,
            )
            master_scaler = scaler

        if categorical_features:
            active_cats = [c for c in categorical_features if c in selected_features]
        else:
            active_cats = None

        X_train = train_df[selected_features]
        X_valid = valid_df[selected_features]
        y_valid_raw = valid_df[TARGET_COL]
        y_valid = (
            valid_df[TARGET_COL] - valid_df[CLIMATOLOGY_COL]
            if use_residual_target else y_valid_raw
        )
        stations_valid = valid_df[STATION_COL].values

        # Cold-start rows: stations with ZERO rows in this fold's training
        # window (e.g. "Gunungsari" only starts reporting 2024-08-13, so
        # fold 0's train_end=2024-06-30 excludes it entirely). The model
        # never saw this station category during training, so its
        # predictions here are structurally meaningless -- no scaler
        # trick can fix that. This never happens for the real production
        # model (train_full uses all data through the actual train/test
        # cutoff, by which every station has history), so counting these
        # rows in the CV metric only adds fold-specific noise unrelated
        # to real generalization. Excluded from metrics, kept in oof_all
        # (flagged) for transparency.
        train_stations_set = set(np.unique(stations_train))
        is_cold_start = ~pd.Series(stations_valid).isin(train_stations_set).values
        if is_cold_start.any() and verbose:
            cold_stations = sorted(set(stations_valid[is_cold_start]))
            print(f"  [Cold-start] {is_cold_start.sum()} valid rows excluded from metrics "
                  f"(station absent from fold train): {cold_stations}")

        if verbose:
            print(f"\n{'='*60}")
            print(f"Fold {fold_idx} — {model.name}")
            print(f"  Train: {len(X_train)} rows, Valid: {len(X_valid)} rows")

        # Target Scaling (applied to the residual when use_residual_target=True)
        scaler = TargetScaler().fit(y_train, stations_train)
        y_train_scaled = scaler.transform(y_train, stations_train)
        y_valid_scaled = scaler.transform(y_valid, stations_valid)

        # Fit
        model.fit(
            X_train, y_train_scaled,
            X_valid=X_valid, y_valid=y_valid_scaled,
            categorical_features=active_cats,
            sample_weight=sample_weight_train,
        )
        if hasattr(model, "evals_result_") and model.evals_result_ is not None:
            evals_results.append(model.evals_result_)

        # Predict (Returns scaled values)
        preds_scaled = model.predict(X_valid)

        # Unscale predictions (still a residual, if use_residual_target=True)
        preds_raw = scaler.inverse_transform(preds_scaled, stations_valid)

        # Add climatology back to get the final raw tma_mdpl-scale prediction
        if use_residual_target:
            preds_raw = preds_raw + valid_df[CLIMATOLOGY_COL].values

        # Evaluate on raw MDPL (excluding cold-start rows -- see above)
        eval_mask = ~is_cold_start
        metrics = evaluate(y_valid_raw.values[eval_mask], preds_raw[eval_mask])
        fold_metrics.append(metrics)

        if verbose:
            print_metrics(metrics, prefix=f"Fold {fold_idx}")

        # Per-station RMSE for this fold
        fold_station_df = pd.DataFrame({
            "station": stations_valid,
            "actual": y_valid_raw.values,
            "predicted": preds_raw,
        })
        station_rmse = fold_station_df.groupby("station").apply(
            lambda g: np.sqrt(((g["actual"] - g["predicted"])**2).mean()),
            include_groups=False,
        ).reset_index()
        station_rmse.columns = ["station", f"fold_{fold_idx}_rmse"]
        station_rmses.append(station_rmse)

        if verbose:
            worst_5 = station_rmse.nlargest(5, f"fold_{fold_idx}_rmse")
            print(f"  Worst 5 stations:")
            for _, row in worst_5.iterrows():
                print(f"    {row['station']}: RMSE={row[f'fold_{fold_idx}_rmse']:.4f}")

        # Store OOF predictions (cold-start rows flagged, not dropped)
        oof_df = pd.DataFrame({
            DATETIME_COL: valid_df[DATETIME_COL].values,
            STATION_COL: stations_valid,
            "actual": y_valid_raw.values,
            "predicted": preds_raw,
            "cold_start": is_cold_start,
        })
        oof_preds.append(oof_df)

    elapsed = time.time() - start_time

    # Combine OOF
    oof_all = pd.concat(oof_preds, ignore_index=True)

    # Aggregate metrics.
    # IMPORTANT: the primary "rmse"/"mae"/"r2" MUST be computed on the
    # pooled OOF predictions (all folds concatenated), not as a naive
    # unweighted average of per-fold RMSE values. Folds here are very
    # unequal in size (e.g. 16425 vs 7184 rows) and difficulty, and RMSE
    # does not average linearly (it's a quadratic mean) -- naively
    # averaging per-fold RMSE systematically UNDERESTIMATES the true
    # pooled RMSE, which is what the actual Kaggle leaderboard metric
    # computes (RMSE over every submitted row at once, not per-fold).
    # The unweighted fold-average is still kept under *_fold_avg for
    # comparison against notebooks that report CV this (biased) way.
    oof_scored = oof_all[~oof_all["cold_start"]]
    n_cold = int(oof_all["cold_start"].sum())
    mean_metrics = evaluate(oof_scored["actual"].values, oof_scored["predicted"].values)
    mean_metrics["n_cold_start_excluded"] = n_cold
    for key in fold_metrics[0]:
        values = [m[key] for m in fold_metrics]
        mean_metrics[f"{key}_fold_avg"] = np.mean(values)
        mean_metrics[f"{key}_std"] = np.std(values)

    # Season-weighted estimate: the real test set (2025-09-19 -> 2026-05-18)
    # is 62.4% rainy-season days, but our CV folds are only 0%/33%/50% wet
    # -- so pooled CV RMSE above is systematically optimistic. Reweighting
    # wet/dry-season squared error to match the real test's composition
    # gives a much closer proxy to actual leaderboard RMSE (empirically
    # verified: 1.47 estimated vs 1.64 actual LB, vs 1.32 naive pooled).
    oof_scored = oof_scored.copy()
    oof_scored["_month"] = pd.to_datetime(oof_scored[DATETIME_COL]).dt.month
    is_wet = oof_scored["_month"].isin([11, 12, 1, 2, 3])
    if is_wet.any() and (~is_wet).any():
        wet_mse = ((oof_scored.loc[is_wet, "actual"] - oof_scored.loc[is_wet, "predicted"]) ** 2).mean()
        dry_mse = ((oof_scored.loc[~is_wet, "actual"] - oof_scored.loc[~is_wet, "predicted"]) ** 2).mean()
        TEST_WET_FRAC = 151 / 242  # real test period composition, see config note
        mean_metrics["rmse_wet"] = float(np.sqrt(wet_mse))
        mean_metrics["rmse_dry"] = float(np.sqrt(dry_mse))
        mean_metrics["rmse_season_weighted"] = float(
            np.sqrt(TEST_WET_FRAC * wet_mse + (1 - TEST_WET_FRAC) * dry_mse)
        )

    if verbose:
        print(f"\n{'='*60}")
        print(f"MEAN CV — {model.name}")
        print_metrics(
            {k: v for k, v in mean_metrics.items() if "_std" not in k and "_fold_avg" not in k},
            prefix="Mean (pooled)",
        )
        print(f"  (unweighted fold-avg RMSE: {mean_metrics['rmse_fold_avg']:.6f} — for reference only, biased low)")
        if "rmse_season_weighted" in mean_metrics:
            print(f"  >>> rmse_season_weighted: {mean_metrics['rmse_season_weighted']:.6f} "
                  f"— BEST proxy for real leaderboard RMSE (wet={mean_metrics['rmse_wet']:.4f}, "
                  f"dry={mean_metrics['rmse_dry']:.4f}, test is 62.4% wet-season) <<<")
        print(f"Time: {elapsed:.1f}s")

    # Combine station metrics across folds
    station_metrics = station_rmses[0]
    for sr in station_rmses[1:]:
        station_metrics = station_metrics.merge(sr, on="station", how="outer")
    rmse_cols = [c for c in station_metrics.columns if "rmse" in c]
    station_metrics["mean_rmse"] = station_metrics[rmse_cols].mean(axis=1)
    station_metrics = station_metrics.sort_values("mean_rmse", ascending=False)

    if verbose:
        print(f"\n--- Station-Level RMSE Summary (Top 10 worst) ---")
        for _, row in station_metrics.head(10).iterrows():
            print(f"  {row['station']}: mean_rmse={row['mean_rmse']:.4f}")

    # Feature importance from last fold
    feat_imp = model.get_feature_importance(selected_features)

    return {
        "fold_metrics": fold_metrics,
        "oof_predictions": oof_all,
        "mean_metrics": mean_metrics,
        "feature_importance": feat_imp,
        "elapsed_seconds": elapsed,
        "evals_results": evals_results,
        "station_metrics": station_metrics,
        "scaler": master_scaler,
        "selected_features": selected_features,
    }


def run_experiment(
    exp_id: str,
    model_name: str,
    df: pd.DataFrame,
    feature_cols: list,
    feature_desc: str = "",
    cv: TimeSeriesCV = None,
    categorical_features: list = None,
    model_params: dict = None,
    save_model: bool = True,
    verbose: bool = True,
    select_features: bool = True,
    use_residual_target: bool = False,
) -> dict:
    model = get_model(model_name, params=model_params)

    results = run_cv(
        model=model,
        df=df,
        feature_cols=feature_cols,
        cv=cv,
        categorical_features=categorical_features,
        verbose=verbose,
        select_features=select_features,
        use_residual_target=use_residual_target,
    )

    # Log experiment
    cv_rmse_scores = [m["rmse"] for m in results["fold_metrics"]]
    log_experiment(
        exp_id=exp_id,
        features=feature_desc,
        model_name=model_name,
        cv_scores=cv_rmse_scores,
        notes=f"params={model.params}",
    )

    # Save model
    if save_model:
        model.save()

    # Plot learning curves if available
    if "evals_results" in results and results["evals_results"]:
        plot_learning_curves(results["evals_results"], model_name)

    results["model"] = model
    return results


def train_full(
    model_name: str,
    df: pd.DataFrame,
    feature_cols: list,
    categorical_features: list = None,
    model_params: dict = None,
    select_features: bool = True,
    use_residual_target: bool = False,
) -> tuple:
    """
    Returns (trained_model, scaler, selected_features, climatology, use_residual_target)

    `climatology` is a StationMonthClimatology fit on ALL available training
    rows (not just the pre-holdout slice) — safe to do since it is the object
    that will later be applied to the test set, which has no target to leak
    from. A SEPARATE climatology (fit on the pre-holdout slice only) is used
    internally so the printed holdout metric stays honest.

    See run_cv's docstring for what use_residual_target does and why.
    """
    model = get_model(model_name, params=model_params)

    # Drop NaN targets
    df = df.dropna(subset=[TARGET_COL]).copy()

    # IMPORTANT: Sort by datetime before taking the last N% rows!
    # Otherwise, if df is grouped by station, we will just hold out the last 3 stations
    # instead of the last 10% of time across all stations!
    df = df.sort_values(DATETIME_COL).reset_index(drop=True)

    # Defensively strip tma_lag_/tma_roll_/tma_diff_ columns and make sure
    # the leak-safe seasonal climatology feature is always considered.
    feature_cols = _drop_leaky_target_history_cols(feature_cols)
    if CLIMATOLOGY_COL not in feature_cols:
        feature_cols = feature_cols + [CLIMATOLOGY_COL]

    # Use last N% as holdout for early stopping
    n_holdout = int(len(df) * FULL_TRAIN_HOLDOUT_FRAC)
    train_idx = slice(None, -n_holdout)
    valid_idx = slice(-n_holdout, None)

    train_part = df.iloc[train_idx]
    valid_part = df.iloc[valid_idx]

    # Honest climatology for the internal holdout metric: fit on the
    # pre-holdout slice only, so the holdout rows never inform their own
    # seasonal baseline.
    holdout_climatology = StationMonthClimatology().fit(train_part)
    train_part = holdout_climatology.transform(train_part)
    valid_part = holdout_climatology.transform(valid_part)

    sample_weight_train = make_sample_weight(train_part)

    X_train_full = train_part[feature_cols]
    y_train_raw = train_part[TARGET_COL]
    y_train = (
        train_part[TARGET_COL] - train_part[CLIMATOLOGY_COL]
        if use_residual_target else y_train_raw
    )
    stations_train = train_part[STATION_COL].values

    X_valid_full = valid_part[feature_cols]
    y_valid_raw = valid_part[TARGET_COL]
    y_valid = (
        valid_part[TARGET_COL] - valid_part[CLIMATOLOGY_COL]
        if use_residual_target else y_valid_raw
    )
    stations_valid = valid_part[STATION_COL].values

    # Target Scaling (applied to the residual when use_residual_target=True)
    scaler = TargetScaler().fit(y_train, stations_train)
    y_train_scaled = scaler.transform(y_train, stations_train)

    selected_features = feature_cols.copy()
    if select_features:
        selected_features = perform_feature_selection(
            model, X_train_full, y_train_scaled, feature_cols, categorical_features,
            sample_weight=sample_weight_train,
        )

    active_cats = [c for c in categorical_features if c in selected_features] if categorical_features else None

    X_train = train_part[selected_features]
    X_valid = valid_part[selected_features]
    y_valid_scaled = scaler.transform(y_valid, stations_valid)

    print(f"Training {model_name} on full data with holdout:")
    print(f"  Train: {len(X_train)} rows, Holdout: {len(X_valid)} rows")

    model.fit(
        X_train, y_train_scaled,
        X_valid=X_valid, y_valid=y_valid_scaled,
        categorical_features=active_cats,
        sample_weight=sample_weight_train,
    )

    # Report holdout score
    preds_scaled = model.predict(X_valid)
    preds_raw = scaler.inverse_transform(preds_scaled, stations_valid)
    if use_residual_target:
        preds_raw = preds_raw + valid_part[CLIMATOLOGY_COL].values

    holdout_metrics = evaluate(y_valid_raw.values, preds_raw)
    print(f"  Holdout RMSE: {holdout_metrics['rmse']:.6f}")
    print(f"  Holdout MAE: {holdout_metrics['mae']:.6f}")
    print(f"  Holdout R²: {holdout_metrics['r2']:.6f}")

    # Final climatology for real test-time inference: fit on ALL training
    # rows (including the holdout slice) for maximum robustness. Safe
    # because the actual test set has no target to leak from.
    final_climatology = StationMonthClimatology().fit(df)

    # Refit on 100% of the data (train + holdout) for the actually-deployed
    # model. The holdout above was only used to find the early-stopping
    # iteration count -- without this refit, the deployed model would
    # never learn from its most recent ~FULL_TRAIN_HOLDOUT_FRAC of history,
    # which is exactly the slice temporally closest to the real test period
    # and likely carries the most relevant near-term seasonal signal.
    best_iteration = getattr(model, "best_iteration_", None)
    df_full = final_climatology.transform(df)
    sample_weight_full = make_sample_weight(df_full)
    X_full = df_full[selected_features]
    y_full_raw = df_full[TARGET_COL]
    y_full = (
        df_full[TARGET_COL] - df_full[CLIMATOLOGY_COL]
        if use_residual_target else y_full_raw
    )
    stations_full = df_full[STATION_COL].values

    final_scaler = TargetScaler().fit(y_full, stations_full)
    y_full_scaled = final_scaler.transform(y_full, stations_full)

    final_model = get_model(model_name, params=model.params)
    if best_iteration:
        final_model.set_fixed_iterations(best_iteration)
    final_model.fit(
        X_full, y_full_scaled,
        categorical_features=active_cats,
        sample_weight=sample_weight_full,
    )

    print(f"Refit on 100% of data ({len(X_full)} rows, best_iteration={best_iteration}).")
    print("Done.")
    return final_model, final_scaler, selected_features, final_climatology, use_residual_target


def compare_models(results_dict: dict) -> pd.DataFrame:
    rows = []
    for name, res in results_dict.items():
        row = {"model": name}
        row.update(res["mean_metrics"])
        row["time_s"] = round(res["elapsed_seconds"], 1)
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("rmse").reset_index(drop=True)
    return df
