"""
inference.py — Prediction pipeline, ensemble, and submission generation.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.config import (
    TARGET_COL, STATION_COL, DATETIME_COL, ID_COL,
    SAMPLE_SUB_CSV, OUTPUT_DIR,
)
from src.models import BaseModel


def predict_test(
    model: BaseModel,
    master_test: pd.DataFrame,
    feature_cols: list,
    scaler=None,
    climatology=None,
    use_residual_target: bool = False,
) -> pd.DataFrame:
    """
    Generate predictions for test set.

    Parameters
    ----------
    model : BaseModel
        Trained model wrapper.
    master_test : pd.DataFrame
        Master test DataFrame with features built.
    feature_cols : list
        Feature columns (must match training).
    scaler : TargetScaler
        Scaler fitted on training data to inverse transform predictions.
    climatology : StationMonthClimatology, optional
        Seasonal (station x month) prior fitted on ALL training data
        (train_full's `final_climatology` return value). Required whenever
        "station_month_climatology" is among feature_cols — the raw test
        set has no tma_mdpl to compute it from directly.
    use_residual_target : bool
        Must match the value passed to train_full for this model. If True,
        the model predicts (tma_mdpl - climatology), so climatology is
        added back here to recover the final raw-scale prediction.

    Returns
    -------
    pd.DataFrame with columns: id, tma_mdpl
    """
    master_test = master_test.copy()

    if climatology is not None:
        master_test = climatology.transform(master_test)

    # Ensure all feature columns exist; fill missing with NaN (GBDT handles it)
    for col in feature_cols:
        if col not in master_test.columns:
            master_test[col] = np.nan

    X_test = master_test[feature_cols]
    preds = model.predict(X_test)

    if scaler is not None:
        stations = master_test[STATION_COL].values
        preds = scaler.inverse_transform(preds, stations)

    if use_residual_target:
        from src.train import CLIMATOLOGY_COL
        preds = preds + master_test[CLIMATOLOGY_COL].values

    result = master_test[[ID_COL]].copy()
    result[TARGET_COL] = preds

    # Station-aware clipping: use minimum 0 (TMA can be near-zero but not negative)
    result[TARGET_COL] = result[TARGET_COL].clip(lower=0)

    return result


def ensemble_predictions(
    predictions_dict: dict,
    weights: dict = None,
) -> pd.DataFrame:
    """
    Weighted average ensemble of multiple model predictions.

    Parameters
    ----------
    predictions_dict : dict
        {model_name: pd.DataFrame with columns [id, tma_mdpl]}
    weights : dict
        {model_name: weight}. If None, equal weights.

    Returns
    -------
    pd.DataFrame with ensemble predictions.
    """
    model_names = list(predictions_dict.keys())

    if weights is None:
        w = {name: 1.0 / len(model_names) for name in model_names}
    else:
        total = sum(weights.values())
        w = {name: weights[name] / total for name in model_names}

    # Use the first prediction as base
    base = predictions_dict[model_names[0]][[ID_COL]].copy()
    base[TARGET_COL] = 0.0

    for name in model_names:
        df = predictions_dict[name]
        base[TARGET_COL] += df[TARGET_COL].values * w[name]

    base[TARGET_COL] = base[TARGET_COL].clip(lower=0)

    print("Ensemble weights:")
    for name, weight in w.items():
        print(f"  {name}: {weight:.3f}")

    return base


def ensemble_predictions_by_rmse(
    predictions_dict: dict,
    cv_rmses: dict,
) -> pd.DataFrame:
    """
    Ensemble weighted by inverse CV RMSE.

    Parameters
    ----------
    predictions_dict : dict
        {model_name: prediction DataFrame}
    cv_rmses : dict
        {model_name: mean_cv_rmse}

    Returns
    -------
    Ensemble predictions DataFrame.
    """
    # Inverse RMSE weighting: lower RMSE → higher weight
    inv_rmses = {name: 1.0 / rmse for name, rmse in cv_rmses.items()}
    return ensemble_predictions(predictions_dict, weights=inv_rmses)


def stacking_ensemble(
    oof_predictions: dict,
    test_predictions: dict,
    y_true: pd.Series,
) -> pd.DataFrame:
    """
    Stacking ensemble: train a Ridge meta-model on OOF predictions.

    Parameters
    ----------
    oof_predictions : dict
        {model_name: oof_pred_array (aligned with y_true)}
    test_predictions : dict
        {model_name: test_pred_DataFrame with [id, tma_mdpl]}
    y_true : pd.Series
        Actual target values aligned with OOF predictions.

    Returns
    -------
    Stacked prediction DataFrame.
    """
    from sklearn.linear_model import Ridge

    # Build meta-features from OOF
    model_names = list(oof_predictions.keys())
    meta_train = pd.DataFrame({name: oof_predictions[name] for name in model_names})
    meta_test = pd.DataFrame({
        name: test_predictions[name][TARGET_COL].values for name in model_names
    })

    # Drop rows where OOF has NaN
    valid_mask = ~meta_train.isna().any(axis=1) & ~y_true.isna()
    meta_train_clean = meta_train[valid_mask]
    y_clean = y_true[valid_mask]

    # Fit Ridge meta-model
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(meta_train_clean, y_clean)

    print("Stacking meta-model coefficients:")
    for name, coef in zip(model_names, meta_model.coef_):
        print(f"  {name}: {coef:.4f}")
    print(f"  intercept: {meta_model.intercept_:.4f}")

    # Predict on test
    stacked_preds = meta_model.predict(meta_test)

    base = test_predictions[model_names[0]][[ID_COL]].copy()
    base[TARGET_COL] = stacked_preds
    base[TARGET_COL] = base[TARGET_COL].clip(lower=0)

    # Evaluate stacking on OOF
    oof_stacked = meta_model.predict(meta_train_clean)
    oof_rmse = np.sqrt(((y_clean.values - oof_stacked) ** 2).mean())
    print(f"Stacking OOF RMSE: {oof_rmse:.6f}")

    return base


def build_submission(
    predictions: pd.DataFrame,
    filename: str = "submission.csv",
) -> Path:
    """
    Format and save submission file.

    Parameters
    ----------
    predictions : pd.DataFrame
        Must have columns [id, tma_mdpl].
    filename : str
        Output filename.

    Returns
    -------
    Path to saved submission file.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load sample submission for reference
    sample = pd.read_csv(SAMPLE_SUB_CSV)

    # Ensure correct order & all IDs present
    submission = sample[[ID_COL]].merge(
        predictions[[ID_COL, TARGET_COL]],
        on=ID_COL,
        how="left",
    )

    # Check for missing predictions
    n_missing = submission[TARGET_COL].isna().sum()
    if n_missing > 0:
        print(f"WARNING: {n_missing} missing predictions! Filling with 0.")
        submission[TARGET_COL] = submission[TARGET_COL].fillna(0)

    # Validate shape
    assert submission.shape[0] == sample.shape[0], (
        f"Submission shape mismatch: {submission.shape[0]} vs {sample.shape[0]}"
    )
    assert submission.shape[1] == 2, (
        f"Submission should have 2 columns, got {submission.shape[1]}"
    )

    # Save
    filepath = OUTPUT_DIR / filename
    submission.to_csv(filepath, index=False)
    print(f"Submission saved -> {filepath}")
    print(f"  Shape: {submission.shape}")
    print(f"  TMA stats: mean={submission[TARGET_COL].mean():.3f}, "
          f"std={submission[TARGET_COL].std():.3f}, "
          f"min={submission[TARGET_COL].min():.3f}, "
          f"max={submission[TARGET_COL].max():.3f}")

    return filepath


def build_test_features_with_history(
    master_train: pd.DataFrame,
    master_test: pd.DataFrame,
    build_features_fn,
) -> pd.DataFrame:
    """
    Build features for test set using historical train data.

    For lag/rolling features on the test set, we need the train data
    as history. This function:
    1. Concatenates train (ALL data) + test
    2. Builds features on the combined DataFrame
    3. Returns only the test portion with features

    The target column in train provides the history for target lag features.
    Test rows will have NaN target lags beyond what train history covers,
    which GBDT models handle natively.

    Parameters
    ----------
    master_train : pd.DataFrame
        Master train (with env + spatial merged, BEFORE feature engineering).
    master_test : pd.DataFrame
        Master test (with env + spatial merged, BEFORE feature engineering).
    build_features_fn : callable
        The feature engineering function to apply.

    Returns
    -------
    pd.DataFrame — test portion with all features built.
    """
    # Mark source
    master_train = master_train.copy()
    master_test = master_test.copy()
    master_train["_source"] = "train"
    master_test["_source"] = "test"

    # Ensure test has target column (NaN) for lag computation
    if TARGET_COL not in master_test.columns:
        master_test[TARGET_COL] = np.nan

    # Concat ALL train + test to preserve full history for lags
    combined = pd.concat([master_train, master_test], ignore_index=True)
    combined = combined.sort_values([STATION_COL, DATETIME_COL]).reset_index(drop=True)

    # Build features on combined (use is_train=False to reuse encoder)
    combined = build_features_fn(combined, is_train=False)

    # Split back
    test_featured = combined[combined["_source"] == "test"].copy()
    test_featured = test_featured.drop(columns=["_source"])

    # Report lag coverage
    tma_lag_cols = [c for c in test_featured.columns if c.startswith("tma_lag_")]
    if tma_lag_cols:
        print(f"\nTarget lag coverage in test set:")
        for col in tma_lag_cols:
            n_valid = test_featured[col].notna().sum()
            pct = 100.0 * n_valid / len(test_featured)
            print(f"  {col}: {n_valid}/{len(test_featured)} ({pct:.1f}%) non-NaN")

    print(f"Test features built: {test_featured.shape}")
    return test_featured
