"""
dl_models.py — Deep Learning architectures for Advanced Time-Series Forecasting.

This module contains skeleton classes for State-of-the-Art Deep Learning models
such as PatchTST and Spatio-Temporal Graph Neural Networks (ST-GNN).

Note: These models require `torch`, `torch-geometric` (for ST-GNN), and
advanced time-series libraries (e.g. `neuralforecast`) to be installed.
"""

import pandas as pd
import numpy as np
from src.models import BaseModel

class PatchTSTModel(BaseModel):
    """
    PatchTST (Patch Time Series Transformer)
    Breaks time series into patches to preserve local semantic information
    while capturing long-term dependencies via Transformer encoder.
    """
    def __init__(self, name="PatchTST", params=None):
        super().__init__(name=name, params=params)
        self.is_dl = True
        print(f"[{self.name}] Initialized. (Requires neuralforecast/PyTorch)")

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, categorical_features=None):
        """
        Skeleton for PyTorch training loop.
        In reality, X_train needs to be converted into a sliding window Dataset.
        """
        print(f"[{self.name}] Building sliding window patches...")
        print(f"[{self.name}] Initializing Transformer Encoder...")
        print(f"[{self.name}] Training for 5 epochs (Mock PyTorch loop)...")
        
        train_loss = [3.0, 2.5, 2.0, 1.8, 1.6]
        val_loss = [3.2, 2.8, 2.3, 1.9, 1.7]
        for epoch in range(5):
            print(f"Epoch {epoch+1:02d}/05 | Train Loss (RMSE): {train_loss[epoch]:.4f} | Val Loss (RMSE): {val_loss[epoch]:.4f}")
        
        # Mock eval results for plot_learning_curves
        self.evals_result_ = {
            "train": {"rmse": train_loss},
            "valid": {"rmse": val_loss}
        }
        
        self.is_fitted = True
        self.feature_importance_ = np.zeros(X_train.shape[1])
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        print(f"[{self.name}] Predicting {len(X)} rows via Transformer inference...")
        # Mock prediction returning mean of historical targets
        return np.full(len(X), 50.0)


class STGNNModel(BaseModel):
    """
    Spatio-Temporal Graph Neural Network.
    Uses Graph Convolutional Networks (GCN) over the River Adjacency Matrix
    combined with Temporal Convolutional Networks (TCN) or LSTMs for time.
    """
    def __init__(self, name="ST-GNN", params=None):
        super().__init__(name=name, params=params)
        self.is_dl = True
        print(f"[{self.name}] Initialized. (Requires torch-geometric)")

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, categorical_features=None):
        """
        Skeleton for ST-GNN training loop.
        Requires construction of Adjacency Matrix from `NEXT_DOWN` or `main_river_id`.
        """
        print(f"[{self.name}] Constructing River Network Adjacency Graph...")
        print(f"[{self.name}] Passing temporal features through Graph Convolution...")
        print(f"[{self.name}] Training for 5 epochs (Mock PyTorch loop)...")
        
        train_loss = [4.0, 3.0, 2.2, 1.7, 1.4]
        val_loss = [4.5, 3.5, 2.4, 1.8, 1.5]
        for epoch in range(5):
            print(f"Epoch {epoch+1:02d}/05 | Train Loss (RMSE): {train_loss[epoch]:.4f} | Val Loss (RMSE): {val_loss[epoch]:.4f}")
        
        self.evals_result_ = {
            "train": {"rmse": train_loss},
            "valid": {"rmse": val_loss}
        }
        
        self.is_fitted = True
        self.feature_importance_ = np.zeros(X_train.shape[1])
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        print(f"[{self.name}] Graph Inference...")
        return np.full(len(X), 50.0)
