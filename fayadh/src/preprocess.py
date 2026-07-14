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
    For each observation hour (06, 12, 18), aggregate environment data
    from the preceding 6-hour window.

    For obs at hour H -> aggregate env from (H-5) to H inclusive.
    E.g., obs at 12:00 -> aggregate env hours 07, 08, 09, 10, 11, 12.

    Uses SUM for rainfall/solar, MEAN for everything else.
    """
    env = env.copy()
    env["hour"] = env[DATETIME_COL].dt.hour
    env["date"] = env[DATETIME_COL].dt.date

    results = []

    for station in env[STATION_COL].unique():
        station_env = env[env[STATION_COL] == station].sort_values(DATETIME_COL)

        for obs_hour in OBS_HOURS:
            # Define the 6-hour window for each day
            # For obs_hour=6 -> hours 1-6, obs_hour=12 -> hours 7-12, obs_hour=18 -> hours 13-18
            start_hour = obs_hour - 5
            end_hour = obs_hour

            # Filter to relevant hours
            mask = station_env["hour"].between(start_hour, end_hour)
            window_data = station_env[mask].copy()

            if window_data.empty:
                continue

            # Build aggregation dict
            agg_dict = {}
            for feat in AGG_SUM_FEATURES:
                if feat in window_data.columns:
                    agg_dict[feat] = "sum"
            for feat in AGG_MEAN_FEATURES:
                if feat in window_data.columns:
                    agg_dict[feat] = "mean"

            # Group by date and aggregate
            grouped = window_data.groupby("date").agg(agg_dict).reset_index()
            grouped[STATION_COL] = station
            grouped[DATETIME_COL] = pd.to_datetime(
                grouped["date"].astype(str) + f" {obs_hour:02d}:00:00"
            )
            grouped = grouped.drop(columns=["date"])

            results.append(grouped)

    env_agg = pd.concat(results, ignore_index=True)
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
# 7. Full preprocessing pipeline
# ============================================================
def run_preprocessing():
    """
    Full pipeline: load -> clean -> aggregate -> extract spatial -> merge.

    Returns
    -------
    master_train : pd.DataFrame
    master_test  : pd.DataFrame
    """
    # Load
    data = load_all_data()
    train = data["train"]
    test = data["test"]
    env = data["env"]
    coord = data["coord"]
    river = data["river"]

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

    # Merge
    print("\nMerging master train...")
    master_train = merge_master(train, env_agg, spatial)

    print("\nMerging master test...")
    master_test = merge_master(test, env_agg, spatial)

    return master_train, master_test

