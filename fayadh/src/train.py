"""
train.py — Training loop with cross-validation, experiment logging, and model comparison.
"""

import numpy as np
import pandas as pd
import time

from src.config import TARGET_COL, DATETIME_COL, STATION_COL
from src.validation import TimeSeriesCV
from src.models import BaseModel, get_model
from src.utils import (
    evaluate, print_metrics, log_experiment, plot_learning_curves
)


def run_cv(
    model: BaseModel,
    df: pd.DataFrame,
    feature_cols: list,
    cv: TimeSeriesCV = None,
    categorical_features: list = None,
    verbose: bool = True,
) -> dict:
    """
    Run walk-forward cross-validation for a model.

    Parameters
    ----------
    model : BaseModel
        A model wrapper instance.
    df : pd.DataFrame
        Master train DataFrame with features already built.
    feature_cols : list
        List of feature column names.
    cv : TimeSeriesCV
        Cross-validation splitter. Defaults to TimeSeriesCV().
    categorical_features : list
        Categorical feature names (for LightGBM/CatBoost).
    verbose : bool
        Whether to print fold-level results.

    Returns
    -------
    dict with keys:
        - 'fold_metrics': list of per-fold metric dicts
        - 'oof_predictions': DataFrame with oof predictions
        - 'mean_metrics': dict with averaged metrics
        - 'feature_importance': DataFrame (from last fold)
        - 'elapsed_seconds': total time
    """
    if cv is None:
        cv = TimeSeriesCV()

    fold_metrics = []
    oof_preds = []
    evals_results = []
    start_time = time.time()

    for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(df)):
        X_train = df.iloc[train_idx][feature_cols]
        y_train = df.iloc[train_idx][TARGET_COL]
        X_valid = df.iloc[valid_idx][feature_cols]
        y_valid = df.iloc[valid_idx][TARGET_COL]

        # Drop rows with NaN target
        train_mask = ~y_train.isna()
        valid_mask = ~y_valid.isna()
        X_train, y_train = X_train[train_mask], y_train[train_mask]
        X_valid, y_valid = X_valid[valid_mask], y_valid[valid_mask]

        if verbose:
            print(f"\n{'='*60}")
            print(f"Fold {fold_idx} — {model.name}")
            print(f"  Train: {len(X_train)} rows, Valid: {len(X_valid)} rows")

        # Fit
        model.fit(
            X_train, y_train,
            X_valid=X_valid, y_valid=y_valid,
            categorical_features=categorical_features,
        )
        if hasattr(model, "evals_result_") and model.evals_result_ is not None:
            evals_results.append(model.evals_result_)

        # Predict
        preds = model.predict(X_valid)

        # Evaluate
        metrics = evaluate(y_valid.values, preds)
        fold_metrics.append(metrics)

        if verbose:
            print_metrics(metrics, prefix=f"Fold {fold_idx}")

        # Store OOF predictions
        oof_df = pd.DataFrame({
            DATETIME_COL: df.iloc[valid_idx[valid_mask]][DATETIME_COL].values,
            STATION_COL: df.iloc[valid_idx[valid_mask]][STATION_COL].values,
            "actual": y_valid.values,
            "predicted": preds,
        })
        oof_preds.append(oof_df)

    elapsed = time.time() - start_time

    # Aggregate metrics
    mean_metrics = {}
    for key in fold_metrics[0]:
        values = [m[key] for m in fold_metrics]
        mean_metrics[key] = np.mean(values)
        mean_metrics[f"{key}_std"] = np.std(values)

    if verbose:
        print(f"\n{'='*60}")
        print(f"MEAN CV — {model.name}")
        print_metrics(
            {k: v for k, v in mean_metrics.items() if "_std" not in k},
            prefix="Mean",
        )
        print(f"Time: {elapsed:.1f}s")

    # Combine OOF
    oof_all = pd.concat(oof_preds, ignore_index=True)

    # Feature importance from last fold
    feat_imp = model.get_feature_importance(feature_cols)

    return {
        "fold_metrics": fold_metrics,
        "oof_predictions": oof_all,
        "mean_metrics": mean_metrics,
        "feature_importance": feat_imp,
        "elapsed_seconds": elapsed,
        "evals_results": evals_results,
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
) -> dict:
    """
    Run a full experiment: CV + logging + optional model save.

    Parameters
    ----------
    exp_id : str
        Experiment identifier (e.g. "001").
    model_name : str
        One of 'lightgbm', 'catboost', 'xgboost', 'random_forest'.
    df : pd.DataFrame
        Master train with features.
    feature_cols : list
        Feature columns.
    feature_desc : str
        Human-readable feature description for logging.
    cv : TimeSeriesCV
        CV splitter.
    categorical_features : list
        Categorical feature names.
    model_params : dict
        Override model parameters.
    save_model : bool
        Save model after last fold.
    verbose : bool
        Print progress.

    Returns
    -------
    dict with CV results.
    """
    model = get_model(model_name, params=model_params)

    results = run_cv(
        model=model,
        df=df,
        feature_cols=feature_cols,
        cv=cv,
        categorical_features=categorical_features,
        verbose=verbose,
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
) -> BaseModel:
    """
    Train a model on the FULL training data (no validation split).
    Used for final submission.

    Returns
    -------
    Trained BaseModel wrapper.
    """
    model = get_model(model_name, params=model_params)

    X = df[feature_cols]
    y = df[TARGET_COL]

    # Drop NaN targets
    mask = ~y.isna()
    X, y = X[mask], y[mask]

    print(f"Training {model_name} on full data: {len(X)} rows...")
    model.fit(X, y, categorical_features=categorical_features)
    print("Done.")

    return model


def compare_models(results_dict: dict) -> pd.DataFrame:
    """
    Create a comparison table from multiple experiment results.

    Parameters
    ----------
    results_dict : dict
        {model_name: results_from_run_cv}

    Returns
    -------
    pd.DataFrame sorted by mean RMSE ascending.
    """
    rows = []
    for name, res in results_dict.items():
        row = {"model": name}
        row.update(res["mean_metrics"])
        row["time_s"] = round(res["elapsed_seconds"], 1)
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("rmse").reset_index(drop=True)
    return df

