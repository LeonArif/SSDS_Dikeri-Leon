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

    Returns
    -------
    pd.DataFrame with columns: id, tma_mdpl
    """
    X_test = master_test[feature_cols]
    preds = model.predict(X_test)

    result = master_test[[ID_COL]].copy()
    result[TARGET_COL] = preds

    # Clip negative predictions (TMA should be >= 0)
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
    1. Concatenates train (end portion) + test
    2. Builds features on the combined DataFrame
    3. Returns only the test portion with features

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

    # Concat
    combined = pd.concat([master_train, master_test], ignore_index=True)
    combined = combined.sort_values([STATION_COL, DATETIME_COL]).reset_index(drop=True)

    # Build features on combined (use is_train=False to reuse encoder)
    combined = build_features_fn(combined, is_train=False)

    # Split back
    test_featured = combined[combined["_source"] == "test"].copy()
    test_featured = test_featured.drop(columns=["_source"])

    print(f"Test features built: {test_featured.shape}")
    return test_featured

