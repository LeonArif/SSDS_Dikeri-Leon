"""
utils.py — Helper functions: metrics, logging, experiment tracking.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from src.config import OUTPUT_DIR, SEED


def set_seed(seed: int = SEED):
    """Set random seed for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae(y_true, y_pred):
    """Mean Absolute Error."""
    return mean_absolute_error(y_true, y_pred)


def r2(y_true, y_pred):
    """R² Score."""
    return r2_score(y_true, y_pred)


def evaluate(y_true, y_pred):
    """Compute all metrics and return as dict."""
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }


def print_metrics(metrics: dict, prefix: str = ""):
    """Pretty-print metrics dict."""
    parts = [f"{k.upper()}: {v:.6f}" for k, v in metrics.items()]
    line = " | ".join(parts)
    if prefix:
        line = f"[{prefix}] {line}"
    print(line)


# ============================================================
# Experiment tracking
# ============================================================
EXPERIMENT_LOG_PATH = OUTPUT_DIR / "experiment_log.csv"


def log_experiment(
    exp_id: str,
    features: str,
    model_name: str,
    cv_scores: list,
    lb_score: float = None,
    notes: str = "",
):
    """
    Append one experiment record to the CSV tracker.

    Parameters
    ----------
    exp_id : str        e.g. "001"
    features : str      e.g. "Basic+Lag+Rolling"
    model_name : str    e.g. "LightGBM"
    cv_scores : list    per-fold RMSE values
    lb_score : float    leaderboard score (optional)
    notes : str         free-text notes
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mean_cv = np.mean(cv_scores)
    std_cv = np.std(cv_scores)

    record = {
        "exp_id": exp_id,
        "features": features,
        "model": model_name,
        "cv_mean": round(mean_cv, 6),
        "cv_std": round(std_cv, 6),
        "cv_folds": str([round(s, 6) for s in cv_scores]),
        "lb_score": lb_score if lb_score is not None else "",
        "notes": notes,
    }

    if EXPERIMENT_LOG_PATH.exists():
        df = pd.read_csv(EXPERIMENT_LOG_PATH)
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    else:
        df = pd.DataFrame([record])

    df.to_csv(EXPERIMENT_LOG_PATH, index=False)
    print(f"Experiment {exp_id} logged -> {EXPERIMENT_LOG_PATH}")


def load_experiment_log():
    """Load experiment tracker as DataFrame."""
    if EXPERIMENT_LOG_PATH.exists():
        return pd.read_csv(EXPERIMENT_LOG_PATH)
    return pd.DataFrame()


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Reduce memory usage of a DataFrame by downcasting numeric types.
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and str(col_type) != "category":
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type).startswith("int"):
                if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            elif str(col_type).startswith("float"):
                if c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    if verbose:
        print(f"Memory: {start_mem:.1f} MB -> {end_mem:.1f} MB "
              f"({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)")
    return df


def plot_learning_curves(evals_results: list, model_name: str, show: bool = True):
    """
    Plot train vs validation loss across all CV folds.

    Parameters
    ----------
    evals_results : list
        List of eval_result dictionaries from each model fold.
    model_name : str
        Name of the model (e.g. 'lightgbm').
    show : bool
        If True, call plt.show() (only safe in an interactive/notebook
        session — a GUI matplotlib backend running headless, e.g. in a
        background/scripted job, will block forever waiting for a window
        that nobody can close). Set False for scripted/background runs;
        the figure is always saved to OUTPUT_DIR regardless.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    if not evals_results or evals_results[0] is None:
        print("No evaluation results available to plot.")
        return
        
    n_folds = len(evals_results)
    fig, axes = plt.subplots(1, n_folds, figsize=(5 * n_folds, 4), sharey=True)
    if n_folds == 1:
        axes = [axes]
        
    for i, (ax, res) in enumerate(zip(axes, evals_results)):
        if model_name == "lightgbm":
            train_loss = list(res.values())[0]['rmse']
            val_loss = list(res.values())[1]['rmse']
        elif model_name == "catboost":
            train_loss = res['learn']['RMSE']
            val_loss = res['validation']['RMSE']
        elif model_name == "xgboost":
            train_loss = res['validation_0']['rmse']
            val_loss = res['validation_1']['rmse']
        else:
            continue
            
        ax.plot(train_loss, label='Train Loss')
        ax.plot(val_loss, label='Val Loss')
        ax.set_title(f"Fold {i+1}")
        ax.set_xlabel("Iterations")
        ax.legend()
        
    axes[0].set_ylabel("RMSE")
    plt.suptitle(f"{model_name.upper()} Learning Curves", y=1.05)
    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"learning_curves_{model_name}.png", dpi=100)

    if show and matplotlib.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(fig)

