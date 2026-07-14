"""
feature_engineering.py — All feature builders for the Water Level Prediction Pipeline.

All features are built per-station (groupby nama_pos) to respect time-series structure.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.config import (
    STATION_COL, DATETIME_COL, TARGET_COL,
    RAIN_LAG_STEPS, HUMIDITY_LAG_STEPS, SOIL_LAG_STEPS,
    RAIN_ROLLING_WINDOWS, SOIL_ROLLING_WINDOWS,
    RAIN_CUM_WINDOWS, CLIMATE_ROLLING_WINDOWS, CLIMATE_FEATURES
)


# ============================================================
# A. Time Features
# ============================================================
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract calendar & time features from datetime column."""
    df = df.copy()
    dt = df[DATETIME_COL]

    df["hour"] = dt.dt.hour
    df["day"] = dt.dt.day
    df["month"] = dt.dt.month
    df["week"] = dt.dt.isocalendar().week.astype(int)
    df["quarter"] = dt.dt.quarter
    df["dayofyear"] = dt.dt.dayofyear
    df["dayofweek"] = dt.dt.dayofweek
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)

    return df


# ============================================================
# B. Cyclic Encoding
# ============================================================
def add_cyclic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode periodic features (hour, month, dayofyear) as sin/cos."""
    df = df.copy()

    # Hour (period = 24)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Month (period = 12)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Day of year (period = 365)
    df["dayofyear_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
    df["dayofyear_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365)

    return df



# ============================================================
# D. Lag Features (Environment)
# ============================================================
def add_env_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Add lagged environment features per station."""
    df = df.copy()

    # Rainfall lags
    rain_cols = [c for c in ["rainfall_mm", "rainfall_openmeteo_mm"] if c in df.columns]
    for col in rain_cols:
        for lag in RAIN_LAG_STEPS:
            df[f"{col}_lag_{lag}"] = df.groupby(STATION_COL)[col].shift(lag)

    # Humidity lags
    if "humidity_pct" in df.columns:
        for lag in HUMIDITY_LAG_STEPS:
            df[f"humidity_lag_{lag}"] = df.groupby(STATION_COL)["humidity_pct"].shift(lag)

    # Soil moisture lags
    soil_cols = [c for c in df.columns if c.startswith("soil_moisture") and "_lag_" not in c and "_roll_" not in c]
    for col in soil_cols:
        for lag in SOIL_LAG_STEPS:
            df[f"{col}_lag_{lag}"] = df.groupby(STATION_COL)[col].shift(lag)

    return df



# ============================================================
# F. Rolling Features (Environment)
# ============================================================
def add_env_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling aggregates for key environment features per station."""
    df = df.copy()

    # Rainfall rolling sum
    rain_cols = [c for c in ["rainfall_mm", "rainfall_openmeteo_mm"] if c in df.columns]
    for col in rain_cols:
        for w in RAIN_ROLLING_WINDOWS:
            df[f"{col}_roll_sum_{w}"] = df.groupby(STATION_COL)[col].transform(
                lambda x: x.shift(1).rolling(window=w, min_periods=1).sum()
            )

    # Soil moisture rolling mean
    soil_cols = [c for c in df.columns if c.startswith("soil_moisture") and "_lag_" not in c and "_roll_" not in c]
    for col in soil_cols:
        for w in SOIL_ROLLING_WINDOWS:
            df[f"{col}_roll_mean_{w}"] = df.groupby(STATION_COL)[col].transform(
                lambda x: x.shift(1).rolling(window=w, min_periods=1).mean()
            )

    return df


# ============================================================
# G. Difference Features
# ============================================================
def add_difference_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add first-order differences for key env features."""
    df = df.copy()

    # Rainfall diff
    if "rainfall_mm" in df.columns:
        df["delta_rain_1"] = df.groupby(STATION_COL)["rainfall_mm"].diff(1)

    # Pressure diff
    if "pressure_msl_hpa" in df.columns:
        df["delta_pressure_1"] = df.groupby(STATION_COL)["pressure_msl_hpa"].diff(1)

    return df


# ============================================================
# H. Cumulative Rainfall
# ============================================================
def add_cumulative_rainfall(df: pd.DataFrame) -> pd.DataFrame:
    """Add cumulative rainfall over 24h, 48h, 72h windows."""
    df = df.copy()

    rain_col = "rainfall_mm" if "rainfall_mm" in df.columns else None
    if rain_col is None:
        return df

    for w in RAIN_CUM_WINDOWS:
        hours_label = w * 6  # each obs step is ~6 hours
        df[f"rain_cumsum_{hours_label}h"] = df.groupby(STATION_COL)[rain_col].transform(
            lambda x: x.shift(1).rolling(window=w, min_periods=1).sum()
        )

    return df



# ============================================================
# I. Advanced Interactions & Climate Lags
# ============================================================
def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add advanced interaction features (e.g. Rainfall * Upstream Area)."""
    df = df.copy()
    
    if "upstream_area" in df.columns:
        if "rainfall_mm" in df.columns:
            df["rain_x_upstream"] = df["rainfall_mm"] * df["upstream_area"]
        if "rain_cumsum_48h" in df.columns: # 8 obs = 48h
            df["rain_48h_x_upstream"] = df["rain_cumsum_48h"] * df["upstream_area"]
        if "rain_cumsum_168h" in df.columns: # 28 obs = 168h (7 days)
            df["rain_7d_x_upstream"] = df["rain_cumsum_168h"] * df["upstream_area"]
        if "rain_cumsum_336h" in df.columns: # 56 obs = 336h (14 days)
            df["rain_14d_x_upstream"] = df["rain_cumsum_336h"] * df["upstream_area"]
            
    return df

def add_climate_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """Add long-term rolling means for global climate indices (Nino, MJO)."""
    df = df.copy()
    
    climate_cols = [c for c in CLIMATE_FEATURES if c in df.columns]
    for col in climate_cols:
        for w in CLIMATE_ROLLING_WINDOWS:
            # 120 = 30 days, 240 = 60 days
            df[f"{col}_roll_mean_{w}"] = df.groupby(STATION_COL)[col].transform(
                lambda x: x.shift(1).rolling(window=w, min_periods=1).mean()
            )
            
    return df


# ============================================================
# I. Baseline & Regional Features
# ============================================================
_station_baselines = None

def add_station_baseline(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    global _station_baselines
    df = df.copy()
    
    if is_train:
        if TARGET_COL in df.columns:
            _station_baselines = df.groupby(STATION_COL)[TARGET_COL].agg(
                station_mean_tma='mean',
                station_min_tma='min',
                station_max_tma='max'
            ).reset_index()
        else:
            raise ValueError(f"Target column {TARGET_COL} missing during train.")
            
    if _station_baselines is not None:
        cols_to_drop = [c for c in _station_baselines.columns if c != STATION_COL and c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
        df = df.merge(_station_baselines, on=STATION_COL, how="left")
    
    return df

def add_regional_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "station_cluster" in df.columns and "rainfall_mm" in df.columns:
        df["regional_rainfall"] = df.groupby(["station_cluster", DATETIME_COL])["rainfall_mm"].transform("mean")
    if "soil_moisture_0_7cm" in df.columns and "rainfall_mm" in df.columns:
        df["runoff_potential"] = df["rainfall_mm"] * df["soil_moisture_0_7cm"]
    return df


# ============================================================
# J. Station Encoding
# ============================================================
_station_encoder = None  # module-level cache


def add_station_encoding(df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
    """
    Label-encode station names.

    Parameters
    ----------
    fit : bool
        If True, fit a new encoder. If False, use the cached one (for test set).
    """
    global _station_encoder
    df = df.copy()

    if fit:
        _station_encoder = LabelEncoder()
        df["station_encoded"] = _station_encoder.fit_transform(df[STATION_COL])
    else:
        if _station_encoder is None:
            raise ValueError("Station encoder not fitted. Call with fit=True first.")
        df["station_encoded"] = _station_encoder.transform(df[STATION_COL])

    return df


# ============================================================
# Master function: build all features
# ============================================================
def build_all_features(
    df: pd.DataFrame,
    is_train: bool = True,
) -> pd.DataFrame:
    """
    Apply all feature engineering steps to a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: datetime, nama_pos, and (for train) tma_mdpl.
        Should already be merged with env and spatial features.
    is_train : bool
        If True, fits the station encoder. If False, uses cached encoder.

    Returns
    -------
    pd.DataFrame with all features added.
    """
    print("Building features...")

    # Time
    df = add_time_features(df)
    print("  ✓ Time features")

    # Cyclic
    df = add_cyclic_features(df)
    print("  ✓ Cyclic features")

    # Station encoding
    df = add_station_encoding(df, fit=is_train)
    print("  ✓ Station encoding")

    # Station baselines
    df = add_station_baseline(df, is_train=is_train)
    print("  ✓ Station baselines")

    # Regional features
    df = add_regional_features(df)
    print("  ✓ Regional features")

    # Difference (Environment)
    df = add_difference_features(df)
    print("  ✓ Difference features")

    # Cumulative rainfall
    df = add_cumulative_rainfall(df)
    print("  ✓ Cumulative rainfall")

    # Environment lags
    df = add_env_lags(df)
    print("  ✓ Environment lags")

    # Environment rolling
    df = add_env_rolling(df)
    print("  ✓ Environment rolling")

    # Climate rolling
    df = add_climate_rolling(df)
    print("  ✓ Climate rolling")
    
    # Interactions
    df = add_interaction_features(df)
    print("  ✓ Interaction features")

    print(f"Final shape: {df.shape}")
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Return list of feature column names (excluding target, id, datetime, station name).
    """
    exclude = {TARGET_COL, DATETIME_COL, STATION_COL, "id", "date", "landcover_name"}
    features = [c for c in df.columns if c not in exclude]
    return features


def get_categorical_features(feature_cols: list) -> list:
    """Return list of categorical feature names from feature columns."""
    cat_feats = []
    for col in feature_cols:
        if col in ["station_encoded", "station_cluster", "landcover_class",
                    "hour", "month", "quarter", "dayofweek", "is_weekend",
                    "main_river_id", "hydrobasin_id"]:
            cat_feats.append(col)
    return cat_feats

