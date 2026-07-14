"""
models.py — Model wrappers with a unified interface for the ML pipeline.

Each model wrapper provides: fit, predict, get_feature_importance, save, load.
"""

import numpy as np
import pandas as pd

try:
    from src.dl_models import PatchTSTModel, STGNNModel
except ImportError:
    PatchTSTModel, STGNNModel = None, None
import joblib
from pathlib import Path
from abc import ABC, abstractmethod

from src.config import (
    LGBM_PARAMS, CATBOOST_PARAMS, XGBOOST_PARAMS, RF_PARAMS, MODEL_DIR,
)


class BaseModel(ABC):
    """Abstract base class for all model wrappers."""

    def __init__(self, name: str, params: dict):
        self.name = name
        self.params = params.copy()
        self.model = None
        self.feature_importance_ = None
        self.evals_result_ = None

    @abstractmethod
    def fit(self, X_train, y_train, X_valid=None, y_valid=None, **kwargs):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def get_feature_importance(self, feature_names: list = None) -> pd.DataFrame:
        """Return feature importance as a sorted DataFrame."""
        if self.feature_importance_ is None:
            return pd.DataFrame()
        df = pd.DataFrame({
            "feature": feature_names or list(range(len(self.feature_importance_))),
            "importance": self.feature_importance_,
        })
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    def save(self, filepath: str = None):
        """Save model to disk."""
        if filepath is None:
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            filepath = MODEL_DIR / f"{self.name}.joblib"
        joblib.dump(self.model, filepath)
        print(f"Model saved -> {filepath}")

    def load(self, filepath: str = None):
        """Load model from disk."""
        if filepath is None:
            filepath = MODEL_DIR / f"{self.name}.joblib"
        self.model = joblib.load(filepath)
        print(f"Model loaded ← {filepath}")


# ============================================================
# LightGBM
# ============================================================
class LGBModel(BaseModel):

    def __init__(self, params: dict = None, name: str = "lightgbm"):
        super().__init__(name, params or LGBM_PARAMS)

    def fit(self, X_train, y_train, X_valid=None, y_valid=None,
            categorical_features=None, **kwargs):
        import lightgbm as lgb

        params = self.params.copy()
        n_estimators = params.pop("n_estimators", 2000)

        evals_result = {}
        callbacks = [lgb.log_evaluation(period=200), lgb.record_evaluation(evals_result)]
        if X_valid is not None:
            callbacks.append(lgb.early_stopping(stopping_rounds=100))

        self.model = lgb.LGBMRegressor(n_estimators=n_estimators, **params)

        fit_params = {
            "X": X_train,
            "y": y_train,
            "callbacks": callbacks,
        }

        if categorical_features:
            fit_params["categorical_feature"] = categorical_features

        if X_valid is not None and y_valid is not None:
            fit_params["eval_set"] = [(X_train, y_train), (X_valid, y_valid)]
            fit_params["eval_names"] = ["train", "valid"]

        self.model.fit(**fit_params)
        self.feature_importance_ = self.model.feature_importances_
        self.evals_result_ = evals_result
        return self

    def predict(self, X):
        return self.model.predict(X)


# ============================================================
# CatBoost
# ============================================================
class CatBoostModel(BaseModel):

    def __init__(self, params: dict = None, name: str = "catboost"):
        super().__init__(name, params or CATBOOST_PARAMS)
        self.cat_features = None

    def _preprocess_cat(self, X):
        if self.cat_features and isinstance(X, pd.DataFrame):
            X = X.copy()
            for c in self.cat_features:
                if c in X.columns:
                    X[c] = (X[c].astype(str)
                            .str.replace(r'\.0$', '', regex=True)
                            .replace(['nan', 'NaN', '<NA>', 'None'], 'Missing'))
            return X
        return X

    def fit(self, X_train, y_train, X_valid=None, y_valid=None,
            categorical_features=None, **kwargs):
        from catboost import CatBoostRegressor, Pool

        params = self.params.copy()
        self.cat_features = categorical_features

        X_train_proc = self._preprocess_cat(X_train)
        X_valid_proc = self._preprocess_cat(X_valid) if X_valid is not None else None

        cat_features_idx = []
        if self.cat_features and isinstance(X_train_proc, pd.DataFrame):
            cat_features_idx = [
                X_train_proc.columns.get_loc(c) for c in self.cat_features
                if c in X_train_proc.columns
            ]

        train_pool = Pool(X_train_proc, y_train, cat_features=cat_features_idx)

        eval_pool = None
        if X_valid_proc is not None and y_valid is not None:
            eval_pool = Pool(X_valid_proc, y_valid, cat_features=cat_features_idx)

        self.model = CatBoostRegressor(**params)
        self.model.fit(train_pool, eval_set=eval_pool, use_best_model=True)
        self.feature_importance_ = self.model.get_feature_importance()
        self.evals_result_ = self.model.get_evals_result()
        return self

    def predict(self, X):
        X_proc = self._preprocess_cat(X)
        return self.model.predict(X_proc)


# ============================================================
# XGBoost
# ============================================================
class XGBModel(BaseModel):

    def __init__(self, params: dict = None, name: str = "xgboost"):
        super().__init__(name, params or XGBOOST_PARAMS)

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, **kwargs):
        import xgboost as xgb

        params = self.params.copy()
        n_estimators = params.pop("n_estimators", 2000)

        self.model = xgb.XGBRegressor(n_estimators=n_estimators, **params)

        fit_params = {
            "X": X_train,
            "y": y_train,
            "verbose": False,
        }

        if X_valid is not None and y_valid is not None:
            fit_params["eval_set"] = [(X_train, y_train), (X_valid, y_valid)]
            params_copy = params.copy()
            fit_params["verbose"] = False

        self.model.fit(**fit_params)
        self.feature_importance_ = self.model.feature_importances_
        
        # XGBoost evals result
        if X_valid is not None and y_valid is not None:
            self.evals_result_ = self.model.evals_result()
            
        return self

    def predict(self, X):
        return self.model.predict(X)


# ============================================================
# Random Forest
# ============================================================
class RFModel(BaseModel):

    def __init__(self, params: dict = None, name: str = "random_forest"):
        super().__init__(name, params or RF_PARAMS)

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, **kwargs):
        from sklearn.ensemble import RandomForestRegressor

        self.model = RandomForestRegressor(**self.params)
        self.model.fit(X_train, y_train)
        self.feature_importance_ = self.model.feature_importances_
        return self

    def predict(self, X):
        return self.model.predict(X)


# ============================================================
# Factory function
# ============================================================
def get_model(name: str, params: dict = None) -> BaseModel:
    """
    Get a model wrapper by name.

    Parameters
    ----------
    name : str
        One of: 'lightgbm', 'catboost', 'xgboost', 'random_forest'
    params : dict, optional
        Override default parameters.
    """
    models = {
        "lightgbm": LGBModel,
        "catboost": CatBoostModel,
        "xgboost": XGBModel,
        "random_forest": RFModel,
    }
    if PatchTSTModel is not None:
        models["patchtst"] = PatchTSTModel
    if STGNNModel is not None:
        models["st-gnn"] = STGNNModel

    if name not in models:
        raise ValueError(f"Unknown model '{name}'. Available: {list(models.keys())}")

    return models[name](params=params)

