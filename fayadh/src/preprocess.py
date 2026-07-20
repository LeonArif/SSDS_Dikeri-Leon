"""
preprocess.py — Load, clean, aggregate, and merge all datasets into master DataFrames.
"""

import numpy as np
import pandas as pd
import warnings

from src.config import (
    TRAIN_CSV, TEST_CSV, ENV_CSV, COORD_CSV, RIVER_SHP,
    INVALID_VALUE, STATION_COL, DATETIME_COL, TARGET_COL,
    OBS_HOURS, AGG_SUM_FEATURES, AGG_MEAN_FEATURES, ALL_ENV_NUMERIC,
    CLIMATE_FEATURES, N_STATION_CLUSTERS,
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. Load raw data
# ============================================================
def load_train() -> pd.DataFrame:
    """Load train.csv with datetime parsed."""
    df = pd.read_csv(TRAIN_CSV, parse_dates=[DATETIME_COL])
    df = df.sort_values([STATION_COL, DATETIME_COL]).reset_index(drop=True)
    return df


def detect_and_fix_glitches(
    df: pd.DataFrame,
    window: int = 4,
    mad_threshold: float = 8.0,
    min_abs_jump: float = 0.5,
) -> pd.DataFrame:
    """
    Detect and correct isolated sensor glitches in tma_mdpl (train only —
    test has no target to clean). Only touches the training target, so
    this carries zero leakage risk.

    Flags row i as a glitch only if ALL of the following hold, using each
    station's own local (rolling, per-station) median/MAD so the threshold
    adapts to that station's natural scale and volatility:
      - |value_i - local_median| is large relative to the local MAD
        (mad_threshold x)
      - value_i jumps by >= min_abs_jump from BOTH immediate neighbors
      - neighbors i-1 and i+1 are close to each other (spike-and-revert
        shape). This deliberately does NOT flag genuine flood rises
        (e.g. Gunungsari's Nov-Dec swings), which move step-by-step and
        stay elevated for many consecutive observations rather than
        spiking and immediately reverting.

    Flagged points are replaced with the local median (safe, local
    imputation). Mirrors the glitch-cleaning step used in the best
    (Leon's) submission on this dataset, which fayadh's pipeline was
    missing entirely.
    """
    df = df.sort_values([STATION_COL, DATETIME_COL]).reset_index(drop=True)
    df[TARGET_COL] = df[TARGET_COL].astype(float)
    n_glitches = 0

    for pos, g in df.groupby(STATION_COL):
        idx = g.index.to_numpy()
        val = df.loc[idx, TARGET_COL].to_numpy()
        n = len(val)
        if n < 2 * window + 3:
            continue
        for i in range(window, n - window):
            if np.isnan(val[i]) or np.isnan(val[i - 1]) or np.isnan(val[i + 1]):
                continue
            neigh = np.concatenate([val[i - window:i], val[i + 1:i + 1 + window]])
            neigh = neigh[~np.isnan(neigh)]
            if len(neigh) < window:
                continue
            local_median = np.median(neigh)
            mad = np.median(np.abs(neigh - local_median)) + 1e-6
            dev = abs(val[i] - local_median)
            jump_prev = abs(val[i] - val[i - 1])
            jump_next = abs(val[i] - val[i + 1])
            neighbor_gap = abs(val[i - 1] - val[i + 1])

            is_glitch = (
                dev > max(min_abs_jump, mad_threshold * mad)
                and jump_prev > min_abs_jump
                and jump_next > min_abs_jump
                and neighbor_gap < 0.5 * (jump_prev + jump_next)
            )
            if is_glitch:
                df.loc[idx[i], TARGET_COL] = local_median
                n_glitches += 1

    print(f"Glitch detection: corrected {n_glitches} anomalous tma_mdpl points out of {len(df)}")
    return df


def load_test() -> pd.DataFrame:
    """Load test.csv, parse the composite 'id' column into datetime + station."""
    df = pd.read_csv(TEST_CSV)
    split = df["id"].str.split(" - ", n=1, expand=True)
    df[DATETIME_COL] = pd.to_datetime(split[0])
    df[STATION_COL] = split[1]
    df = df.sort_values([STATION_COL, DATETIME_COL]).reset_index(drop=True)
    return df


def load_env() -> pd.DataFrame:
    """Load data_lingkungan.csv with datetime parsed."""
    df = pd.read_csv(ENV_CSV, parse_dates=[DATETIME_COL])
    df = df.sort_values([STATION_COL, DATETIME_COL]).reset_index(drop=True)

    # rainfall_openmeteo_mm is identical to rainfall_mm for every row
    # (correlation = 1, verified exactly) -- pure duplicate that doubles
    # every rainfall lag/rolling/cumulative feature for zero new signal.
    if "rainfall_openmeteo_mm" in df.columns:
        df = df.drop(columns=["rainfall_openmeteo_mm"])

    return df


def load_coordinates() -> pd.DataFrame:
    """Load koordinat_pos.csv."""
    return pd.read_csv(COORD_CSV)


def load_river():
    """Load HydroRIVERS shapefile. Returns GeoDataFrame or None if geopandas unavailable."""
    try:
        import geopandas as gpd
        return gpd.read_file(RIVER_SHP)
    except ImportError:
        print("WARNING: geopandas not installed. River features will be skipped.")
        return None
    except Exception as e:
        print(f"WARNING: Could not load river shapefile: {e}")
        return None


def load_all_data():
    """Load all datasets and return as a dict."""
    print("Loading datasets...")
    data = {
        "train": load_train(),
        "test": load_test(),
        "env": load_env(),
        "coord": load_coordinates(),
        "river": load_river(),
    }
    for k, v in data.items():
        if v is not None:
            shape = v.shape if hasattr(v, "shape") else "loaded"
            print(f"  {k}: {shape}")
    return data


# ============================================================
# 2. Clean environment data
# ============================================================
def clean_env(env: pd.DataFrame) -> pd.DataFrame:
    """
    Replace sentinel -999 values with NaN, then interpolate.
    Also drop duplicate rows.
    """
    env = env.copy()

    # Replace sentinel
    numeric_cols = env.select_dtypes(include=np.number).columns
    env[numeric_cols] = env[numeric_cols].replace(INVALID_VALUE, np.nan)

    # Remove exact duplicates
    env = env.drop_duplicates().reset_index(drop=True)

    # Sort properly
    env = env.sort_values([STATION_COL, DATETIME_COL]).reset_index(drop=True)

    # Interpolate per station (linear, limit forward fill to 6 hours)
    env[numeric_cols] = (
        env.groupby(STATION_COL)[numeric_cols]
        .transform(lambda x: x.interpolate(method="linear", limit=6))
    )

    # Fill remaining NaN with station median
    env[numeric_cols] = (
        env.groupby(STATION_COL)[numeric_cols]
        .transform(lambda x: x.fillna(x.median()))
    )

    # Global median fallback
    for col in numeric_cols:
        if env[col].isna().any():
            env[col] = env[col].fillna(env[col].median())

    print(f"Cleaned env: {env.shape}, remaining NaN: {env[numeric_cols].isna().sum().sum()}")
    return env


# ============================================================
# 3. Aggregate hourly env -> 6-hourly aligned to obs hours
# ============================================================
def aggregate_env_to_6h(env: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate hourly environment data into windows aligned to each
    observation time (06:00, 12:00, 18:00), where each window spans the
    time SINCE THE PREVIOUS OBSERVATION — not a fixed 6 hours.

    TMA is only measured 3x/day, so the gap before the 06:00 reading is
    actually 12 hours (back to 18:00 the PREVIOUS day), while the gaps
    before the 12:00 and 18:00 readings are 6 hours each. A fixed 6-hour
    window for every slot (the previous implementation) silently drops
    the overnight 19:00-24:00 rainfall/weather from every single 06:00
    reading's aggregation — a systematic error affecting 1/3 of all rows,
    not just a few rows near data gaps.

    Uses SUM for rainfall/solar, MEAN for everything else.
    """
    env = env.copy()
    hour = env[DATETIME_COL].dt.hour
    date = env[DATETIME_COL].dt.normalize()

    conditions = [hour < 6, hour < 12, hour < 18]
    choices = [
        date + pd.Timedelta(hours=6),
        date + pd.Timedelta(hours=12),
        date + pd.Timedelta(hours=18),
    ]
    default = date + pd.Timedelta(days=1, hours=6)
    env["window_end"] = np.select(conditions, choices, default=default)

    agg_dict = {}
    for feat in AGG_SUM_FEATURES:
        if feat in env.columns:
            agg_dict[feat] = "sum"
    for feat in AGG_MEAN_FEATURES:
        if feat in env.columns:
            agg_dict[feat] = "mean"

    env_agg = env.groupby([STATION_COL, "window_end"]).agg(agg_dict).reset_index()
    env_agg = env_agg.rename(columns={"window_end": DATETIME_COL})
    env_agg = env_agg.sort_values([STATION_COL, DATETIME_COL]).reset_index(drop=True)

    print(f"Aggregated env: {env_agg.shape}")
    return env_agg


# ============================================================
# 4. Extract river features
# ============================================================
def extract_river_features(river_gdf, coord: pd.DataFrame) -> pd.DataFrame:
    """
    For each station, find the nearest river segment and extract:
    - river_order (ORD_FLOW)
    - river_length (LENGTH_KM)
    - upstream_area (UPLAND_SKM)
    - nearest_river_dist (km)

    Returns DataFrame with one row per station.
    """
    if river_gdf is None:
        print("River data not available. Returning empty spatial features.")
        result = coord[[STATION_COL]].copy()
        result["river_order"] = np.nan
        result["river_length"] = np.nan
        result["upstream_area"] = np.nan
        result["nearest_river_dist"] = np.nan
        return result

    try:
        import geopandas as gpd
        from shapely.geometry import Point
        from scipy.spatial import cKDTree

        # Create GeoDataFrame for stations
        stations = gpd.GeoDataFrame(
            coord,
            geometry=gpd.points_from_xy(coord["longitude"], coord["latitude"]),
            crs="EPSG:4326",
        )

        # Ensure same CRS
        if river_gdf.crs != stations.crs:
            stations = stations.to_crs(river_gdf.crs)

        # Get river centroids for nearest search
        river_centroids = river_gdf.geometry.centroid
        river_coords = np.array(list(zip(river_centroids.x, river_centroids.y)))
        station_coords = np.array(list(zip(stations.geometry.x, stations.geometry.y)))

        tree = cKDTree(river_coords)
        distances, indices = tree.query(station_coords, k=1)

        result = coord[[STATION_COL]].copy()
        result["river_order"] = river_gdf.iloc[indices]["ORD_FLOW"].values
        result["nearest_river_dist"] = distances

        if "LENGTH_KM" in river_gdf.columns:
            result["river_length"] = river_gdf.iloc[indices]["LENGTH_KM"].values
        else:
            result["river_length"] = np.nan

        if "UPLAND_SKM" in river_gdf.columns:
            result["upstream_area"] = river_gdf.iloc[indices]["UPLAND_SKM"].values
        else:
            result["upstream_area"] = np.nan
            
        # Graph-topology features (New)
        if "DIS_AV_CMS" in river_gdf.columns:
            result["river_discharge_avg"] = river_gdf.iloc[indices]["DIS_AV_CMS"].values
        if "DIST_DN_KM" in river_gdf.columns:
            result["dist_to_ocean"] = river_gdf.iloc[indices]["DIST_DN_KM"].values
        if "DIST_UP_KM" in river_gdf.columns:
            result["dist_to_source"] = river_gdf.iloc[indices]["DIST_UP_KM"].values
        if "MAIN_RIV" in river_gdf.columns:
            result["main_river_id"] = river_gdf.iloc[indices]["MAIN_RIV"].values
        if "CATCH_SKM" in river_gdf.columns:
            result["catchment_local"] = river_gdf.iloc[indices]["CATCH_SKM"].values
        if "HYBAS_L12" in river_gdf.columns:
            result["hydrobasin_id"] = river_gdf.iloc[indices]["HYBAS_L12"].values
        if "ORD_STRA" in river_gdf.columns:
            result["river_strahler"] = river_gdf.iloc[indices]["ORD_STRA"].values

        # main_river_id / hydrobasin_id come from HydroRIVERS as huge raw
        # integer codes (hydrobasin_id ~5.12e9, past int32 max 2.15e9).
        # LightGBM's categorical handling casts category codes to int32,
        # so hydrobasin_id silently overflows to negative and gets dropped
        # to NaN for every single row ("Met negative value in categorical
        # features" warning) -- the feature was pure noise for LightGBM.
        # Factorize to small dense codes (only ~30 stations / a handful of
        # rivers & basins here) so the category identity survives intact.
        for id_col in ("main_river_id", "hydrobasin_id"):
            if id_col in result.columns:
                result[id_col] = pd.factorize(result[id_col])[0]

        print(f"River features extracted for {len(result)} stations.")
        return result

    except Exception as e:
        print(f"WARNING: River feature extraction failed: {e}")
        result = coord[[STATION_COL]].copy()
        result["river_order"] = np.nan
        result["river_length"] = np.nan
        result["upstream_area"] = np.nan
        result["nearest_river_dist"] = np.nan
        result["river_discharge_avg"] = np.nan
        result["dist_to_ocean"] = np.nan
        result["dist_to_source"] = np.nan
        result["main_river_id"] = np.nan
        result["catchment_local"] = np.nan
        result["hydrobasin_id"] = np.nan
        result["river_strahler"] = np.nan
        return result


# ============================================================
# 5. Build spatial features (coordinates + clusters + nearest station)
# ============================================================
def build_spatial_features(coord: pd.DataFrame, river_features: pd.DataFrame) -> pd.DataFrame:
    """
    Combine coordinate features with river features and add:
    - station clusters (KMeans)
    - nearest station distance
    """
    from sklearn.cluster import KMeans
    from scipy.spatial.distance import cdist

    spatial = coord.copy()

    # Merge river features
    spatial = spatial.merge(river_features, on=STATION_COL, how="left")

    # KMeans clustering on lat/lon
    coords_arr = spatial[["latitude", "longitude"]].values
    kmeans = KMeans(n_clusters=N_STATION_CLUSTERS, random_state=42, n_init=10)
    spatial["station_cluster"] = kmeans.fit_predict(coords_arr)

    # Nearest station distance
    dist_matrix = cdist(coords_arr, coords_arr, metric="euclidean")
    np.fill_diagonal(dist_matrix, np.inf)
    spatial["nearest_station_dist"] = dist_matrix.min(axis=1)

    print(f"Spatial features: {spatial.shape}")
    return spatial


# ============================================================
# 5b. Dense 6-hourly grid reindex (CRITICAL — see docstring)
# ============================================================
def reindex_dense_grid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reindex to a complete 6-hourly grid (obs hours 06/12/18) per station,
    inserting explicit rows for any missing timestamp (marked _observed=0).

    train.csv/test.csv have substantial real gaps: most stations are
    missing hundreds of expected 6-hourly observations, and several
    stations have gaps spanning multiple WEEKS (e.g. Floodway Bridge C
    has a 163-day gap; most stations have a shared ~25-day gap around
    Feb 2025). Without this step, a plain groupby().shift(n)/rolling(n)
    silently treats consecutive ROWS as if they were exactly n*6 hours
    apart, regardless of the true calendar gap between them — corrupting
    every lag/rolling/cumulative/API feature computed across a gap
    (not just at CV fold boundaries, but throughout the whole dataset).
    Reindexing first makes shift/rolling operate on true calendar time;
    the synthetic gap-filler rows are dropped again after feature
    engineering (see drop_phantom_rows).
    """
    frames = []
    for pos, g in df.groupby(STATION_COL):
        full_range = pd.date_range(g[DATETIME_COL].min(), g[DATETIME_COL].max(), freq="6h")
        full_range = full_range[full_range.hour.isin(OBS_HOURS)]
        frames.append(pd.DataFrame({STATION_COL: pos, DATETIME_COL: full_range}))
    grid = pd.concat(frames, ignore_index=True)

    merged = grid.merge(df, on=[STATION_COL, DATETIME_COL], how="left", indicator=True)
    merged["_observed"] = (merged["_merge"] == "both").astype(int)
    merged = merged.drop(columns=["_merge"])
    merged = merged.sort_values([STATION_COL, DATETIME_COL]).reset_index(drop=True)

    n_phantom = int((merged["_observed"] == 0).sum())
    print(f"Dense grid reindex: {len(df)} -> {len(merged)} rows "
          f"({n_phantom} phantom/gap-filler rows added for correct lag/rolling spacing)")
    return merged


def drop_phantom_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the synthetic gap-filler rows added by reindex_dense_grid, after
    features have been computed on the dense grid."""
    out = df[df["_observed"] == 1].drop(columns=["_observed"]).reset_index(drop=True)
    return out


# ============================================================
# 6. Merge into master dataset
# ============================================================
def merge_master(
    target_df: pd.DataFrame,
    env_agg: pd.DataFrame,
    spatial: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge target (train or test) with aggregated environment and spatial features.
    """
    # Merge environment
    master = target_df.merge(
        env_agg,
        on=[STATION_COL, DATETIME_COL],
        how="left",
    )

    # Merge spatial
    master = master.merge(
        spatial,
        on=STATION_COL,
        how="left",
    )

    print(f"Master shape: {master.shape}")
    return master


# ============================================================
# 7. Split combined dataframe back into train/test after feature engineering
# ============================================================
def split_train_test(df: pd.DataFrame) -> tuple:
    """
    Drop phantom (gap-filler) rows and split the combined dataframe back
    into train and test portions using the `_source` marker set in
    run_preprocessing(). Call this AFTER feature engineering has been run
    once on the full combined dataframe.
    """
    df = drop_phantom_rows(df)
    master_train = df[df["_source"] == "train"].drop(columns=["_source"]).reset_index(drop=True)
    master_test = df[df["_source"] == "test"].drop(columns=["_source"]).reset_index(drop=True)
    return master_train, master_test


# ============================================================
# 8. Full preprocessing pipeline
# ============================================================
def run_preprocessing():
    """
    Full pipeline: load -> clean -> combine train+test -> reindex to a
    dense gap-free 6-hourly grid -> merge env/spatial features.

    Train and test are combined and reindexed together BEFORE any lag/
    rolling feature is computed, and BEFORE the env/spatial merge, so that
    (a) shift/rolling operations always operate on true calendar-spaced
    rows even across the many real gaps in train.csv/test.csv (see
    reindex_dense_grid), and (b) feature history carries correctly across
    the train/test boundary without a separate concat step later.

    Returns
    -------
    master_combined : pd.DataFrame
        Single combined dataframe with `_source` ("train"/"test") and
        `_observed` (1 = real row, 0 = phantom gap-filler) markers.
        Run feature engineering on this ONCE, then call split_train_test()
        to get the final master_train / master_test matrices.
    """
    # Load
    data = load_all_data()
    train = data["train"]
    test = data["test"]
    env = data["env"]
    coord = data["coord"]
    river = data["river"]

    # Clean target (train only — sensor glitches inflate RMSE disproportionately
    # since errors are squared; test has no target so nothing to clean there)
    print("\nCleaning target (glitch detection)...")
    train = detect_and_fix_glitches(train)

    # Clean env
    print("\nCleaning environment data...")
    env = clean_env(env)

    # Aggregate env to 6-hourly
    print("\nAggregating environment data to 6-hourly...")
    env_agg = aggregate_env_to_6h(env)

    # Extract river features
    print("\nExtracting river features...")
    river_features = extract_river_features(river, coord)

    # Build spatial features
    print("\nBuilding spatial features...")
    spatial = build_spatial_features(coord, river_features)

    # Combine train + test BEFORE reindexing/gap-filling, so lag/rolling
    # history is continuous and correctly calendar-spaced across the
    # train/test boundary too.
    train = train.copy()
    test = test.copy()
    train["_source"] = "train"
    test["_source"] = "test"
    if TARGET_COL not in test.columns:
        test[TARGET_COL] = np.nan

    combined = pd.concat([train, test], ignore_index=True, sort=False)
    combined = combined.sort_values([STATION_COL, DATETIME_COL]).reset_index(drop=True)

    print("\nReindexing to dense 6-hourly grid (fills real data gaps)...")
    combined = reindex_dense_grid(combined)

    print("\nMerging environment + spatial features onto dense grid...")
    combined = merge_master(combined, env_agg, spatial)

    return combined

