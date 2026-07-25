"""
dl_models.py — Deep Learning model wrappers for tabular data.

TabNet: A practical deep learning model for tabular data that uses
sequential attention to choose which features to reason from at each step.
Falls back gracefully if pytorch-tabnet is not installed.
"""

import pandas as pd
import numpy as np
from src.models import BaseModel


class TabNetModel(BaseModel):
    """
    TabNet Regressor wrapper.
    Uses pytorch_tabnet which is a practical DL model for tabular data.
    
    Requires: pip install pytorch-tabnet
    """
    
    TABNET_DEFAULT_PARAMS = {
        "n_d": 32,
        "n_a": 32,
        "n_steps": 5,
        "gamma": 1.5,
        "n_independent": 2,
        "n_shared": 2,
        "lambda_sparse": 1e-4,
        "momentum": 0.3,
        "clip_value": 2.0,
        "optimizer_params": {"lr": 2e-2},
        "scheduler_params": {"step_size": 50, "gamma": 0.9},
        "scheduler_fn": None,  # Will set to StepLR in fit
        "mask_type": "entmax",
        "seed": 42,
        "verbose": 10,
    }
    
    def __init__(self, name="tabnet", params=None):
        super().__init__(name=name, params=params or self.TABNET_DEFAULT_PARAMS)
        self.is_dl = True
    
    def fit(self, X_train, y_train, X_valid=None, y_valid=None, 
            categorical_features=None, **kwargs):
        """
        Train TabNet model.
        """
        try:
            from pytorch_tabnet.tab_model import TabNetRegressor
            import torch
        except ImportError:
            raise ImportError(
                "pytorch-tabnet is required for TabNet. "
                "Install with: pip install pytorch-tabnet"
            )
        
        params = self.params.copy()
        
        # Handle scheduler
        from torch.optim.lr_scheduler import StepLR
        if params.get("scheduler_fn") is None:
            params["scheduler_fn"] = StepLR
        
        # Prepare categorical feature indices
        cat_idxs = []
        cat_dims = []
        if categorical_features and isinstance(X_train, pd.DataFrame):
            for c in categorical_features:
                if c in X_train.columns:
                    idx = X_train.columns.get_loc(c)
                    cat_idxs.append(idx)
                    # Get cardinality
                    n_unique = max(
                        X_train[c].nunique(),
                        X_valid[c].nunique() if X_valid is not None else 0,
                    )
                    cat_dims.append(n_unique + 1)  # +1 for unknown
        
        # Convert to numpy
        X_train_np = X_train.values.astype(np.float32) if isinstance(X_train, pd.DataFrame) else X_train.astype(np.float32)
        y_train_np = y_train.values.reshape(-1, 1).astype(np.float32) if isinstance(y_train, pd.Series) else y_train.reshape(-1, 1).astype(np.float32)
        
        # Replace NaN with 0 for TabNet (it doesn't handle NaN natively)
        X_train_np = np.nan_to_num(X_train_np, nan=0.0)
        
        # Build model
        self.model = TabNetRegressor(
            cat_idxs=cat_idxs if cat_idxs else [],
            cat_dims=cat_dims if cat_dims else [],
            cat_emb_dim=1,
            **{k: v for k, v in params.items() if k not in ["seed"]},
        )
        
        fit_kwargs = {
            "X_train": X_train_np,
            "y_train": y_train_np,
            "max_epochs": 200,
            "patience": 30,
            "batch_size": 1024,
            "virtual_batch_size": 256,
            "num_workers": 0,
            "drop_last": False,
        }
        
        if X_valid is not None and y_valid is not None:
            X_valid_np = X_valid.values.astype(np.float32) if isinstance(X_valid, pd.DataFrame) else X_valid.astype(np.float32)
            y_valid_np = y_valid.values.reshape(-1, 1).astype(np.float32) if isinstance(y_valid, pd.Series) else y_valid.reshape(-1, 1).astype(np.float32)
            X_valid_np = np.nan_to_num(X_valid_np, nan=0.0)
            fit_kwargs["eval_set"] = [(X_valid_np, y_valid_np)]
            fit_kwargs["eval_metric"] = ["rmse"]
        
        self.model.fit(**fit_kwargs)
        
        # Feature importance from attention masks
        self.feature_importance_ = self.model.feature_importances_
        
        # Store eval results
        self.evals_result_ = {
            "train": {"rmse": self.model.history["loss"]},
        }
        if X_valid is not None:
            self.evals_result_["valid"] = {"rmse": self.model.history["val_0_rmse"]}
        
        return self
    
    def predict(self, X):
        X_np = X.values.astype(np.float32) if isinstance(X, pd.DataFrame) else X.astype(np.float32)
        X_np = np.nan_to_num(X_np, nan=0.0)
        preds = self.model.predict(X_np)
        return preds.flatten()
    
    def save(self, filepath=None):
        """TabNet has its own save method."""
        from src.config import MODEL_DIR
        if filepath is None:
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            filepath = str(MODEL_DIR / self.name)
        self.model.save_model(filepath)
        print(f"TabNet model saved -> {filepath}")
    
    def load(self, filepath=None):
        """TabNet has its own load method."""
        from pytorch_tabnet.tab_model import TabNetRegressor
        from src.config import MODEL_DIR
        if filepath is None:
            filepath = str(MODEL_DIR / self.name)
        self.model = TabNetRegressor()
        self.model.load_model(filepath + ".zip")
        print(f"TabNet model loaded <- {filepath}")
