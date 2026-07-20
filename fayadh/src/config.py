"""
config.py — Central configuration for the Water Level Prediction Pipeline.

All paths, constants, feature lists, and hyperparameters in one place.
"""

import os
from pathlib import Path

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SUPPORT_DIR = DATA_DIR / "data_pendukung"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"
CONFIG_DIR = PROJECT_ROOT / "configs"

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
SAMPLE_SUB_CSV = DATA_DIR / "sample_submission.csv"
ENV_CSV = SUPPORT_DIR / "data_lingkungan.csv"
COORD_CSV = SUPPORT_DIR / "koordinat_pos.csv"
RIVER_SHP = SUPPORT_DIR / "HydroRIVERS_v10_au_shp" / "HydroRIVERS_v10_au.shp"

# ============================================================
# Random seed & reproducibility
# ============================================================
SEED = 42

# ============================================================
# Target & ID columns
# ============================================================
TARGET_COL = "tma_mdpl"
STATION_COL = "nama_pos"
DATETIME_COL = "datetime"
ID_COL = "id"

# ============================================================
# Observation hours (train/test are 3x daily)
# ============================================================
OBS_HOURS = [6, 12, 18]

# ============================================================
# Sentinel value used in raw environment data
# ============================================================
INVALID_VALUE = -999

# ============================================================
# Environment feature groups
# ============================================================
WEATHER_FEATURES = [
    "rainfall_mm",
    "humidity_pct",
    "wind_direction_deg",
    "dew_point_c",
    "cloud_cover_pct",
    "temperature_c",
    "wind_speed_kmh",
    # "rainfall_openmeteo_mm" intentionally excluded — verified identical to
    # rainfall_mm in every row (correlation = 1); dropped in load_env().
    "rainfall_max_24h_mm",
    "solar_radiation_mj_m2",
]

SOIL_FEATURES = [
    "soil_moisture_0_7cm",
    "soil_moisture_7_28cm",
    "soil_moisture_28_100cm",
    "soil_moisture_100_255cm",
]

PRESSURE_FEATURES = [
    "surface_pressure_hpa",
    "pressure_msl_hpa",
]

CLIMATE_FEATURES = [
    "rmm1",
    "rmm2",
    "mjo_phase",
    "mjo_amplitude",
    "mjo_active",
    "nino_34",
]

LAND_FEATURES = [
    "built_surface_m2",
    "landcover_class",
]

# All numeric environment features (excluding nama_pos, datetime, landcover_name)
ALL_ENV_NUMERIC = (
    WEATHER_FEATURES + SOIL_FEATURES + PRESSURE_FEATURES + CLIMATE_FEATURES + LAND_FEATURES
)

# ============================================================
# Aggregation rules for hourly -> 6-hourly
# Sum-based features (rainfall) vs mean-based (everything else)
# ============================================================
AGG_SUM_FEATURES = [
    "rainfall_mm",
    "solar_radiation_mj_m2",
]

AGG_MEAN_FEATURES = [
    f for f in ALL_ENV_NUMERIC if f not in AGG_SUM_FEATURES
]

# ============================================================
# Target lag configurations (in observation steps, 1 step = ~6 hours)
# These use ONLY past target values, no leakage
# ============================================================
TARGET_LAG_STEPS = [1, 2, 3, 6, 9, 12]   # 6h, 12h, 18h, 36h, 54h, 72h

# Target rolling windows (in observation steps)
TARGET_ROLLING_WINDOWS = [3, 6, 12, 24]   # 18h, 36h, 72h, 6 days

# Target diff steps
TARGET_DIFF_STEPS = [1, 3, 6]              # 6h, 18h, 36h

# ============================================================
# Lag configurations
# ============================================================
RAIN_LAG_STEPS = [1, 3, 6]
HUMIDITY_LAG_STEPS = [1, 3]
SOIL_LAG_STEPS = [1]

# ============================================================
# Rolling window sizes (in observation steps)
# ============================================================
RAIN_ROLLING_WINDOWS = [6, 12, 24]
SOIL_ROLLING_WINDOWS = [6]

# ============================================================
# Cumulative rainfall windows (in observation steps)
# 8 obs ≈ 24h, 16 obs ≈ 48h, 24 obs ≈ 72h
# 28 obs = 7 days, 56 obs = 14 days
# (each obs is 6 hours apart)
# ============================================================
RAIN_CUM_WINDOWS = [8, 16, 24, 28, 56]

# ============================================================
# Antecedent Precipitation Index (API) decay factors
# API_t = rain_t + k * API_{t-1}  (exponentially weighted)
# ============================================================
API_DECAY_FACTORS = [0.85, 0.95]

# ============================================================
# Climate rolling windows (in observation steps)
# 120 obs = 30 days, 240 obs = 60 days
# ============================================================
CLIMATE_ROLLING_WINDOWS = [120, 240]

# ============================================================
# Spatial features
# ============================================================
N_STATION_CLUSTERS = 5

# ============================================================
# Target-history features (tma_lag_*, tma_roll_*, tma_diff_*)
# ============================================================
# The competition test set starts 242 days / 726 six-hourly steps
# after the last train observation, with NO ground-truth tma_mdpl
# ever available in between. A single-shot (non-recursive) pipeline
# therefore has valid target-lag/rolling/diff features for <2% of
# test rows (only the first ~1-12 steps per station, computed off
# the last known train value); the other 98%+ are NaN.
#
# CV_SPLITS below has zero gap between train and validation, so
# during cross-validation these features ARE fully available for
# every validation row -> CV score becomes wildly optimistic and
# does not reflect real leaderboard performance (train/serving skew,
# not classical leakage, but just as harmful). Keep the builder
# functions in feature_engineering.py for reference/experimentation,
# but do not include them in the default feature set.
USE_TARGET_HISTORY_FEATURES = False

# ============================================================
# Rainy-season sample weighting (Bengawan Solo wet season)
# ============================================================
RAINY_SEASON_MONTHS = [11, 12, 1, 2, 3]
RAINY_SEASON_WEIGHT = 1.0  # swept 1.0-6.0 against rmse_season_weighted: higher
# weights do NOT reduce wet-season RMSE (stays ~1.63-1.65 regardless) and hurt
# dry-season fit -- wet-season error is a feature/capacity ceiling, not an
# optimization-focus problem. Weighting disabled (weight=1.0 is a no-op).

# ============================================================
# Station x month climatology (seasonal prior)
# ============================================================
# With target-lag features disabled (see above), the model has no
# autoregressive signal at all for the ~8 month blind horizon. A
# per-station, per-calendar-month historical mean/median of tma_mdpl
# (fit ONLY on the training fold, never on validation/test targets)
# gives a strong, leakage-safe seasonal baseline that IS available
# for every future timestamp, since only station + month is needed.
CLIMATOLOGY_MIN_SAMPLES = 5

# ============================================================
# Validation — Walk-forward splits
# ============================================================
CV_SPLITS = [
    {
        "train_end": "2024-06-30 18:00:00",
        "valid_start": "2024-07-01 06:00:00",
        "valid_end": "2024-12-31 18:00:00",
    },
    {
        "train_end": "2024-12-31 18:00:00",
        "valid_start": "2025-01-01 06:00:00",
        "valid_end": "2025-06-30 18:00:00",
    },
    {
        "train_end": "2025-06-30 18:00:00",
        "valid_start": "2025-07-01 06:00:00",
        "valid_end": "2025-09-18 18:00:00",
    },
]

# ============================================================
# Holdout fraction for train_full early stopping
# ============================================================
FULL_TRAIN_HOLDOUT_FRAC = 0.1   # last 10% as validation

# ============================================================
# Default model hyperparameters (tuned for generalization)
# ============================================================
LGBM_PARAMS = {
    "objective": "huber",          # Robust to outlier stations (empirically beats plain rmse objective — see experiment 001_v3_rmse_obj)
    "metric": "rmse",
    "boosting_type": "gbdt",
    "n_estimators": 3000,          # was 25000 — early stopping (200 rounds) governs actual convergence; a lower cap just keeps iteration cycles fast (Leon/Ivant use 300-2000)
    "learning_rate": 0.03,         # was 0.01 — bumped so the model can still converge within the lower cap
    "max_depth": 10,
    "num_leaves": 63,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 20,
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": -1,
}

CATBOOST_PARAMS = {
    "iterations": 3000,            # was 10000 — see LGBM_PARAMS note above
    "learning_rate": 0.03,
    "depth": 8,
    "l2_leaf_reg": 5.0,
    "random_seed": SEED,
    "verbose": 200,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "early_stopping_rounds": 200,
    "subsample": 0.8,
    "rsm": 0.7,                   # random subspace method (colsample)
    "min_data_in_leaf": 30,
    "border_count": 254,
    "bootstrap_type": "MVS",      # Minimum Variance Sampling
}

XGBOOST_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "n_estimators": 3000,          # was 20000 — see LGBM_PARAMS note above
    "learning_rate": 0.03,         # was 0.015
    "max_depth": 10,
    "subsample": 0.7,
    "colsample_bytree": 0.6,
    "reg_alpha": 0.5,
    "reg_lambda": 5.0,
    "min_child_weight": 10,
    "gamma": 0.1,
    "random_state": SEED,
    "n_jobs": -1,
    "verbosity": 0,
    "tree_method": "hist",
}

RF_PARAMS = {
    "n_estimators": 1000,
    "max_depth": 20,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "max_features": 0.5,
    "random_state": SEED,
    "n_jobs": 2,  # Changed from -1 to avoid WinError 1455 (joblib multiprocessing RAM issue on Windows)
}

# ============================================================
# Bagging hyperparameters (wraps a base estimator)
# ============================================================
BAGGING_PARAMS = {
    "n_estimators": 20,
    "max_samples": 0.8,
    "max_features": 0.8,
    "random_state": SEED,
    "n_jobs": 2,  # Changed from -1 to avoid WinError 1455 (joblib multiprocessing RAM issue on Windows)
}
