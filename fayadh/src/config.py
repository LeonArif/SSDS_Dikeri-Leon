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
    "rainfall_openmeteo_mm",
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
    "rainfall_openmeteo_mm",
    "solar_radiation_mj_m2",
]

AGG_MEAN_FEATURES = [
    f for f in ALL_ENV_NUMERIC if f not in AGG_SUM_FEATURES
]

# ============================================================
# Lag configurations
# ============================================================
RAIN_LAG_STEPS = [1, 3, 6]
HUMIDITY_LAG_STEPS = [1]
SOIL_LAG_STEPS = [1]

# ============================================================
# Rolling window sizes (in observation steps)
# ============================================================
RAIN_ROLLING_WINDOWS = [6, 12]
SOIL_ROLLING_WINDOWS = [6]

# ============================================================
# Cumulative rainfall windows (in observation steps)
# 8 obs ≈ 24h, 16 obs ≈ 48h, 24 obs ≈ 72h
# 28 obs = 7 days, 56 obs = 14 days
# (each obs is 6 hours apart)
# ============================================================
RAIN_CUM_WINDOWS = [8, 16, 24, 28, 56]

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
# Default model hyperparameters
# ============================================================
LGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "n_estimators": 10000,
    "learning_rate": 0.01,
    "max_depth": 8,
    "num_leaves": 63,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "min_child_samples": 20,
    "random_state": SEED,
    "verbose": -1,
    "n_jobs": -1,
}

CATBOOST_PARAMS = {
    "iterations": 2000,
    "learning_rate": 0.05,
    "depth": 8,
    "l2_leaf_reg": 3.0,
    "random_seed": SEED,
    "verbose": 100,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "early_stopping_rounds": 100,
}

XGBOOST_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": SEED,
    "n_jobs": -1,
    "verbosity": 0,
}

RF_PARAMS = {
    "n_estimators": 500,
    "max_depth": 15,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "max_features": "sqrt",
    "random_state": SEED,
    "n_jobs": -1,
}

