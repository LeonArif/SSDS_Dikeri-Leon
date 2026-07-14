"""
validation.py — Time-series cross-validation for walk-forward strategy.
"""

import numpy as np
import pandas as pd

from src.config import CV_SPLITS, DATETIME_COL


class TimeSeriesCV:
    """
    Walk-forward cross-validation for time-series data.

    Uses predefined date-based splits from config.CV_SPLITS.
    Each fold trains on all data up to train_end and validates on valid_start -> valid_end.
    """

    def __init__(self, splits: list = None):
        self.splits = splits or CV_SPLITS

    @property
    def n_splits(self) -> int:
        return len(self.splits)

    def split(self, df: pd.DataFrame):
        """
        Generate (train_idx, valid_idx) tuples for each fold.

        Parameters
        ----------
        df : pd.DataFrame
            Must have a datetime column.

        Yields
        ------
        (train_indices, valid_indices) : tuple of numpy arrays
        """
        dt = pd.to_datetime(df[DATETIME_COL])

        for i, fold in enumerate(self.splits):
            train_end = pd.Timestamp(fold["train_end"])
            valid_start = pd.Timestamp(fold["valid_start"])
            valid_end = pd.Timestamp(fold["valid_end"])

            train_mask = dt <= train_end
            valid_mask = (dt >= valid_start) & (dt <= valid_end)

            train_idx = np.where(train_mask)[0]
            valid_idx = np.where(valid_mask)[0]

            if len(train_idx) == 0 or len(valid_idx) == 0:
                print(f"WARNING: Fold {i} has empty train ({len(train_idx)}) "
                      f"or valid ({len(valid_idx)}) set. Skipping.")
                continue

            print(f"Fold {i}: train={len(train_idx)} "
                  f"({dt.iloc[train_idx[0]].date()} -> {dt.iloc[train_idx[-1]].date()}), "
                  f"valid={len(valid_idx)} "
                  f"({dt.iloc[valid_idx[0]].date()} -> {dt.iloc[valid_idx[-1]].date()})")

            yield train_idx, valid_idx

    def __repr__(self):
        lines = [f"TimeSeriesCV with {self.n_splits} folds:"]
        for i, fold in enumerate(self.splits):
            lines.append(
                f"  Fold {i}: train -> {fold['train_end']}, "
                f"valid: {fold['valid_start']} -> {fold['valid_end']}"
            )
        return "\n".join(lines)

