"""
feature_engineering.py — All feature builders for the Water Level Prediction Pipeline.

All features are built per-station (groupby nama_pos) to respect time-series structure.
Target lag features use ONLY past values — no leakage.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.config import (
    STATION_COL, DATETIME_COL, TARGET_COL,
    RAIN_LAG_STEPS, HUMIDITY_LAG_STEPS, SOIL_LAG_STEPS,
    RAIN_ROLLING_WINDOWS, SOIL_ROLLING_WINDOWS,
    RAIN_CUM_WINDOWS, CLIMATE_ROLLING_WINDOWS, CLIMATE_FEATURES,
    TARGET_LAG_STEPS, TARGET_ROLLING_WINDOWS, TARGET_DIFF_STEPS,
    API_DECAY_FACTORS, RAINY_SEASON_MONTHS, USE_TARGET_HISTORY_FEATURES,
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

    # Rainy season flag (Bengawan Solo wet season, Nov-Mar) — used both as a
    # feature and to build sample weights (see train.py). Calendar-only, so
    # it is available for every future timestamp with zero leakage risk.
    df["musim_hujan"] = dt.dt.month.isin(RAINY_SEASON_MONTHS).astype(int)

    # Days since start of dataset (captures long-term trend)
    df["days_since_start"] = (dt - dt.min()).dt.total_seconds() / 86400.0

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

    # Wonogiri Dam is a human-regulated reservoir, not a natural river
    # gauge -- its TMA dynamics (controlled releases, operational rules)
    # differ fundamentally from the other 29 stations. It is also the
    # single largest contributor to total squared error in both fayadh's
    # and Leon's per-station breakdowns. is_dam x seasonal-cycle lets the
    # model learn a distinct seasonal response for this one station.
    df["is_dam"] = (df[STATION_COL] == "Wonogiri Dam").astype(int)
    df["dayofyear_sin_x_dam"] = df["dayofyear_sin"] * df["is_dam"]
    df["dayofyear_cos_x_dam"] = df["dayofyear_cos"] * df["is_dam"]

    # Week (period = 52)
    df["week_sin"] = np.sin(2 * np.pi * df["week"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week"] / 52)

    return df


# ============================================================
# C. Target Lag Features (NO LEAKAGE — uses only past values)
# ============================================================
def add_target_lags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lagged target values per station.
    These are the most powerful features for time-series prediction.
    shift(n) ensures we only use past values.
    """
    df = df.copy()

    if TARGET_COL not in df.columns:
        return df

    for lag in TARGET_LAG_STEPS:
        df[f"tma_lag_{lag}"] = df.groupby(STATION_COL)[TARGET_COL].shift(lag)

    return df


def add_target_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling statistics of target per station.
    Uses shift(1) to prevent including current value.
    """
    df = df.copy()

    if TARGET_COL not in df.columns:
        return df

    for w in TARGET_ROLLING_WINDOWS:
        # Rolling mean
        df[f"tma_roll_mean_{w}"] = df.groupby(STATION_COL)[TARGET_COL].transform(
            lambda x: x.shift(1).rolling(window=w, min_periods=1).mean()
        )
        # Rolling std
        df[f"tma_roll_std_{w}"] = df.groupby(STATION_COL)[TARGET_COL].transform(
            lambda x: x.shift(1).rolling(window=w, min_periods=1).std()
        )
        # Rolling min/max
        df[f"tma_roll_min_{w}"] = df.groupby(STATION_COL)[TARGET_COL].transform(
            lambda x: x.shift(1).rolling(window=w, min_periods=1).min()
        )
        df[f"tma_roll_max_{w}"] = df.groupby(STATION_COL)[TARGET_COL].transform(
            lambda x: x.shift(1).rolling(window=w, min_periods=1).max()
        )

    return df


def add_target_diff(df: pd.DataFrame) -> pd.DataFrame:
    """Add rate-of-change features for target."""
    df = df.copy()

    if TARGET_COL not in df.columns:
        return df

    for step in TARGET_DIFF_STEPS:
        df[f"tma_diff_{step}"] = df.groupby(STATION_COL)[TARGET_COL].diff(step)

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
# E. Rolling Features (Environment)
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
            # Also add rolling max for rainfall (captures peak events)
            df[f"{col}_roll_max_{w}"] = df.groupby(STATION_COL)[col].transform(
                lambda x: x.shift(1).rolling(window=w, min_periods=1).max()
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
# F. Difference Features
# ============================================================
def add_difference_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add first-order differences for key env features."""
    df = df.copy()

    # Rainfall diff
    if "rainfall_mm" in df.columns:
        df["delta_rain_1"] = df.groupby(STATION_COL)["rainfall_mm"].diff(1)

    # Pressure diff — critical for weather fronts
    if "pressure_msl_hpa" in df.columns:
        df["delta_pressure_1"] = df.groupby(STATION_COL)["pressure_msl_hpa"].diff(1)
        df["delta_pressure_3"] = df.groupby(STATION_COL)["pressure_msl_hpa"].diff(3)
        # Pressure falling flag (storm indicator)
        df["pressure_falling"] = (df["delta_pressure_3"] < -2.0).astype(int)

    # Humidity diff
    if "humidity_pct" in df.columns:
        df["delta_humidity_1"] = df.groupby(STATION_COL)["humidity_pct"].diff(1)
        df["delta_humidity_3"] = df.groupby(STATION_COL)["humidity_pct"].diff(3)

    # Temperature diff
    if "temperature_c" in df.columns:
        df["delta_temp_1"] = df.groupby(STATION_COL)["temperature_c"].diff(1)

    # Soil moisture diff (saturation trend)
    if "soil_moisture_0_7cm" in df.columns:
        df["delta_soil_0_7_1"] = df.groupby(STATION_COL)["soil_moisture_0_7cm"].diff(1)

    return df


# ============================================================
# G. Cumulative Rainfall
# ============================================================
def add_cumulative_rainfall(df: pd.DataFrame) -> pd.DataFrame:
    """Add cumulative rainfall over various windows."""
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
# H. Antecedent Precipitation Index (API)
# ============================================================
def add_api_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Antecedent Precipitation Index — exponentially weighted rainfall.
    API_t = rain_t + k * API_{t-1}
    Better than raw cumulative because it captures recency effect.
    """
    df = df.copy()

    rain_col = "rainfall_mm" if "rainfall_mm" in df.columns else None
    if rain_col is None:
        return df

    for k in API_DECAY_FACTORS:
        col_name = f"api_{str(k).replace('.', '')}"

        def compute_api(rain_series, decay=k):
            api = np.zeros(len(rain_series))
            vals = rain_series.values
            for i in range(len(vals)):
                if i == 0:
                    api[i] = vals[i] if not np.isnan(vals[i]) else 0
                else:
                    r = vals[i] if not np.isnan(vals[i]) else 0
                    api[i] = r + decay * api[i - 1]
            return pd.Series(api, index=rain_series.index)

        df[col_name] = df.groupby(STATION_COL)[rain_col].transform(
            lambda x: compute_api(x.shift(1).fillna(0))
        )

    return df


# ============================================================
# I. Advanced Interactions & Climate Lags
# ============================================================
def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add advanced interaction features."""
    df = df.copy()

    if "upstream_area" in df.columns:
        if "rainfall_mm" in df.columns:
            df["rain_x_upstream"] = df["rainfall_mm"] * df["upstream_area"]
        if "rain_cumsum_48h" in df.columns:
            df["rain_48h_x_upstream"] = df["rain_cumsum_48h"] * df["upstream_area"]
        if "rain_cumsum_168h" in df.columns:
            df["rain_7d_x_upstream"] = df["rain_cumsum_168h"] * df["upstream_area"]
        if "rain_cumsum_336h" in df.columns:
            df["rain_14d_x_upstream"] = df["rain_cumsum_336h"] * df["upstream_area"]

    # Soil saturation × rainfall (runoff potential)
    if "soil_moisture_0_7cm" in df.columns and "rainfall_mm" in df.columns:
        df["runoff_potential"] = df["rainfall_mm"] * df["soil_moisture_0_7cm"]

    # Rainfall × humidity (combined wet indicator)
    if "humidity_pct" in df.columns and "rainfall_mm" in df.columns:
        df["rain_x_humidity"] = df["rainfall_mm"] * (df["humidity_pct"] / 100.0)

    # API × upstream (combining antecedent wetness with catchment size)
    for k in API_DECAY_FACTORS:
        api_col = f"api_{str(k).replace('.', '')}"
        if api_col in df.columns and "upstream_area" in df.columns:
            df[f"{api_col}_x_upstream"] = df[api_col] * df["upstream_area"]

    return df


def add_climate_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """Add long-term rolling means for global climate indices (Nino, MJO)."""
    df = df.copy()

    climate_cols = [c for c in CLIMATE_FEATURES if c in df.columns]
    for col in climate_cols:
        for w in CLIMATE_ROLLING_WINDOWS:
            df[f"{col}_roll_mean_{w}"] = df.groupby(STATION_COL)[col].transform(
                lambda x: x.shift(1).rolling(window=w, min_periods=1).mean()
            )

    return df


# ============================================================
# J. Wet/Dry Spell Features
# ============================================================
def add_wet_dry_spell(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute consecutive wet/dry spell length per station.
    Wet = rainfall_mm > 0.5 mm
    """
    df = df.copy()

    rain_col = "rainfall_mm" if "rainfall_mm" in df.columns else None
    if rain_col is None:
        return df

    def spell_length(series, threshold=0.5):
        """Count consecutive steps above/below threshold."""
        is_wet = (series > threshold).astype(int)
        wet_spell = pd.Series(0, index=series.index)
        dry_spell = pd.Series(0, index=series.index)

        for i in range(1, len(series)):
            if is_wet.iloc[i] == 1:
                wet_spell.iloc[i] = wet_spell.iloc[i - 1] + 1
                dry_spell.iloc[i] = 0
            else:
                wet_spell.iloc[i] = 0
                dry_spell.iloc[i] = dry_spell.iloc[i - 1] + 1

        return wet_spell, dry_spell

    wet_spells = []
    dry_spells = []
    for station, group in df.groupby(STATION_COL):
        rain = group[rain_col].fillna(0)
        ws, ds = spell_length(rain)
        wet_spells.append(ws)
        dry_spells.append(ds)

    df["wet_spell_length"] = pd.concat(wet_spells).reindex(df.index)
    df["dry_spell_length"] = pd.concat(dry_spells).reindex(df.index)

    return df


# ============================================================
# K. Regional Features
# ============================================================
def add_regional_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cross-station regional aggregates."""
    df = df.copy()

    if "station_cluster" in df.columns and "rainfall_mm" in df.columns:
        df["regional_rainfall"] = df.groupby(["station_cluster", DATETIME_COL])["rainfall_mm"].transform("mean")

    if "station_cluster" in df.columns and "humidity_pct" in df.columns:
        df["regional_humidity"] = df.groupby(["station_cluster", DATETIME_COL])["humidity_pct"].transform("mean")

    # Regional soil moisture
    if "station_cluster" in df.columns and "soil_moisture_0_7cm" in df.columns:
        df["regional_soil_moisture"] = df.groupby(["station_cluster", DATETIME_COL])["soil_moisture_0_7cm"].transform("mean")

    return df


# ============================================================
# L. Station Encoding (no target leakage)
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
        # Handle unseen stations gracefully (use max+1, not -1, to avoid LGB warnings)
        known = set(_station_encoder.classes_)
        mask = df[STATION_COL].isin(known)
        df.loc[mask, "station_encoded"] = _station_encoder.transform(df.loc[mask, STATION_COL])
        df.loc[~mask, "station_encoded"] = len(_station_encoder.classes_)

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
    print("  [OK] Time features")

    # Cyclic
    df = add_cyclic_features(df)
    print("  [OK] Cyclic features")

    # Station encoding (label only - no target stats)
    df = add_station_encoding(df, fit=is_train)
    print("  [OK] Station encoding")

    # Regional features
    df = add_regional_features(df)
    print("  [OK] Regional features")

    # Difference (Environment)
    df = add_difference_features(df)
    print("  [OK] Difference features")

    # Cumulative rainfall
    df = add_cumulative_rainfall(df)
    print("  [OK] Cumulative rainfall")

    # Antecedent Precipitation Index
    df = add_api_features(df)
    print("  [OK] Antecedent Precipitation Index")

    # Environment lags
    df = add_env_lags(df)
    print("  [OK] Environment lags")

    # Environment rolling
    df = add_env_rolling(df)
    print("  [OK] Environment rolling")

    # Climate rolling
    df = add_climate_rolling(df)
    print("  [OK] Climate rolling")

    # Wet/dry spells
    df = add_wet_dry_spell(df)
    print("  [OK] Wet/dry spells")

    # Interactions
    df = add_interaction_features(df)
    print("  [OK] Interaction features")

    # Target lag/rolling/diff features — OFF by default (see config.py:
    # USE_TARGET_HISTORY_FEATURES). They are technically leak-free (shift-
    # based) but are only non-NaN for the first 1-12 steps of the ~726-step
    # test horizon, causing severe CV/leaderboard mismatch since CV_SPLITS
    # has no matching gap. Kept here for controlled experimentation only.
    if USE_TARGET_HISTORY_FEATURES:
        df = add_target_lags(df)
        print("  [OK] Target lags (EXPERIMENTAL — mostly NaN on real test)")

        df = add_target_rolling(df)
        print("  [OK] Target rolling (EXPERIMENTAL — mostly NaN on real test)")

        df = add_target_diff(df)
        print("  [OK] Target diff (EXPERIMENTAL — mostly NaN on real test)")

    print(f"Final shape: {df.shape}")
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Return list of feature column names (excluding target, id, datetime, station name).

    Defensively excludes tma_lag_/tma_roll_/tma_diff_ columns even if
    USE_TARGET_HISTORY_FEATURES was left on by mistake — those columns are
    ~98%+ NaN on the real test set (see config.py) and must never leak into
    the default feature set for the submitted model.
    """
    exclude = {TARGET_COL, DATETIME_COL, STATION_COL, "id", "date",
               "landcover_name", "_source", "days_since_start"}
    features = [
        c for c in df.columns
        if c not in exclude
        and not c.startswith("tma_lag_")
        and not c.startswith("tma_roll_")
        and not c.startswith("tma_diff_")
    ]
    return features


def get_categorical_features(feature_cols: list) -> list:
    """Return list of categorical feature names from feature columns."""
    cat_feats = []
    for col in feature_cols:
        if col in ["station_encoded", "station_cluster", "landcover_class",
                    "hour", "month", "quarter", "dayofweek", "is_weekend",
                    "main_river_id", "hydrobasin_id", "pressure_falling"]:
            cat_feats.append(col)
    return cat_feats
