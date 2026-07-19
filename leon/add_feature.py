import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2

def add_features(full, koor):

    full.drop(['rainfall_openmeteo_mm'], axis=1, inplace=True, errors="ignore")
    full = full.sort_values(['nama_pos','datetime']).reset_index(drop=True)

    # ===== reindex grid 6-jaman (msh perlu, krn rain/soil_moisture lag-roll butuh alignment waktu) =====
    grids = []
    for pos, g in full.groupby('nama_pos'):
        full_range = pd.date_range(g['datetime'].min(), g['datetime'].max(), freq='6h')
        full_range = full_range[full_range.hour.isin([6,12,18])]
        grids.append(pd.DataFrame({'nama_pos': pos, 'datetime': full_range}))
    grid = pd.concat(grids, ignore_index=True)
    full = grid.merge(full, on=['nama_pos','datetime'], how='left')
    full['is_phantom'] = full['is_train'].isna().astype(int)
    full = full.sort_values(['nama_pos','datetime']).reset_index(drop=True)

    # ===== FITUR WAKTU (lebih lengkap) =====
    full['hour'] = full['datetime'].dt.hour
    full['day'] = full['datetime'].dt.day
    full['month'] = full['datetime'].dt.month
    full['week'] = full['datetime'].dt.isocalendar().week.astype(int)
    full['quarter'] = full['datetime'].dt.quarter
    full['dayofyear'] = full['datetime'].dt.dayofyear
    full['dayofweek'] = full['datetime'].dt.dayofweek
    full['is_weekend'] = full['dayofweek'].isin([5,6]).astype(int)
    full['hour_sin'] = np.sin(2*np.pi*full['hour']/24)
    full['hour_cos'] = np.cos(2*np.pi*full['hour']/24)
    full['month_sin'] = np.sin(2*np.pi*full['month']/12)
    full['month_cos'] = np.cos(2*np.pi*full['month']/12)
    full['dayofyear_sin'] = np.sin(2*np.pi*full['dayofyear']/365)
    full['dayofyear_cos'] = np.cos(2*np.pi*full['dayofyear']/365)

    # ===== FITUR CUACA: lag & rolling (SEMUA berbasis cuaca, BUKAN target -> gak butuh recursion) =====
    g = full.groupby('nama_pos')

    def rollsum(col, n): return g[col].transform(lambda x: x.shift(1).rolling(n).sum())
    def rollmean(col, n): return g[col].transform(lambda x: x.shift(1).rolling(n).mean())

    full['rainfall_mm_lag_1'] = g['rainfall_mm'].shift(1)
    full['rainfall_mm_lag_3'] = g['rainfall_mm'].shift(3)
    full['rainfall_mm_lag_6'] = g['rainfall_mm'].shift(6)
    full['humidity_lag_1'] = g['humidity_pct'].shift(1)
    full['soil_moisture_0_7cm_lag_1'] = g['soil_moisture_0_7cm'].shift(1)
    full['soil_moisture_7_28cm_lag_1'] = g['soil_moisture_7_28cm'].shift(1)
    full['soil_moisture_28_100cm_lag_1'] = g['soil_moisture_28_100cm'].shift(1)
    full['soil_moisture_100_255cm_lag_1'] = g['soil_moisture_100_255cm'].shift(1)

    full['rainfall_mm_roll_sum_6'] = rollsum('rainfall_mm', 6)
    full['rainfall_mm_roll_sum_12'] = rollsum('rainfall_mm', 12)
    full['soil_moisture_0_7cm_roll_mean_6'] = rollmean('soil_moisture_0_7cm', 6)
    full['soil_moisture_7_28cm_roll_mean_6'] = rollmean('soil_moisture_7_28cm', 6)
    full['soil_moisture_28_100cm_roll_mean_6'] = rollmean('soil_moisture_28_100cm', 6)
    full['soil_moisture_100_255cm_roll_mean_6'] = rollmean('soil_moisture_100_255cm', 6)

    # akumulasi hujan multi-window (48h=8 periode, 96h=16, 144h=24, 168h=28, 336h=56 -- satuan 6 jam-an)
    full['rain_cumsum_48h']  = rollsum('rainfall_mm', 8)
    full['rain_cumsum_96h']  = rollsum('rainfall_mm', 16)
    full['rain_cumsum_144h'] = rollsum('rainfall_mm', 24)
    full['rain_cumsum_168h'] = rollsum('rainfall_mm', 28)
    full['rain_cumsum_336h'] = rollsum('rainfall_mm', 56)

    # indeks iklim: rolling mean jangka panjang (120 & 240 periode 6-jaman = ~30 & ~60 hari)
    for col in ['rmm1','rmm2','mjo_amplitude','nino_34']:
        full[f'{col}_roll_mean_120'] = rollmean(col, 120)
        full[f'{col}_roll_mean_240'] = rollmean(col, 240)

    full['delta_rain_1'] = g['rainfall_mm'].diff(1)
    full['delta_pressure_1'] = g['surface_pressure_hpa'].diff(1)

    # =========================================================
    # SHOCK / INTENSITY FEATURES
    # =========================================================

    # lag tambahan (2 hari, 3 hari, 4 hari)
    full['rainfall_mm_lag_8']  = g['rainfall_mm'].shift(8)
    full['rainfall_mm_lag_12'] = g['rainfall_mm'].shift(12)
    full['rainfall_mm_lag_16'] = g['rainfall_mm'].shift(16)

    # rolling pendek (lebih responsif)
    full['rain_roll_sum_2'] = rollsum('rainfall_mm', 2)
    full['rain_roll_sum_4'] = rollsum('rainfall_mm', 4)

    # rolling rata-rata pendek
    full['rain_roll_mean_2'] = rollmean('rainfall_mm', 2)
    full['rain_roll_mean_4'] = rollmean('rainfall_mm', 4)

    # maksimum hujan terakhir
    full['rain_roll_max_4'] = g['rainfall_mm'].transform(
        lambda x: x.shift(1).rolling(4).max()
    )

    full['rain_roll_max_8'] = g['rainfall_mm'].transform(
        lambda x: x.shift(1).rolling(8).max()
    )

    # intensitas hujan
    full['rain_intensity'] = (
        full['rainfall_mm'] /
        (full['rainfall_mm_roll_sum_12'] + 1)
    )

    # perubahan hujan
    full['delta_rain_3'] = g['rainfall_mm'].diff(3)
    full['delta_rain_6'] = g['rainfall_mm'].diff(6)

    # perubahan soil moisture
    full['delta_sm_0_7'] = g['soil_moisture_0_7cm'].diff(1)
    full['delta_sm_7_28'] = g['soil_moisture_7_28cm'].diff(1)

    # interaksi hujan × kelembapan tanah
    full['rain_x_soil'] = (
        full['rainfall_mm'] *
        full['soil_moisture_0_7cm']
    )

    # interaksi hujan × humidity
    full['rain_x_humidity'] = (
        full['rainfall_mm'] *
        full['humidity_pct']
    )

    # end of shock

    full['wind_dir_sin'] = np.sin(np.radians(full['wind_direction_deg']))
    full['wind_dir_cos'] = np.cos(np.radians(full['wind_direction_deg']))

    # new features

    # Rolling STD
    def rollstd(col, n):
        return g[col].transform(lambda x: x.shift(1).rolling(n).std())

    full["rain_roll_std_4"] = rollstd("rainfall_mm",4)
    full["rain_roll_std_8"] = rollstd("rainfall_mm",8)

    full["sm_roll_std_4"] = rollstd("soil_moisture_0_7cm",4)

    # Rolling MAX soil moisture

    full["sm_max_4"] = g["soil_moisture_0_7cm"].transform(
        lambda x:x.shift(1).rolling(4).max()
    )

    full["sm_max_8"] = g["soil_moisture_0_7cm"].transform(
        lambda x:x.shift(1).rolling(8).max()
    )

    # Rain anomaly

    full["rain_anomaly"] = (
        full["rainfall_mm"] - full["rain_roll_mean_4"]
    )

    # Rain acceleration

    full["rain_acceleration"] = (
        full["delta_rain_1"] - full["delta_rain_3"]
    )


    # Rain × Climate

    full["rain_x_nino"] = (
        full["rainfall_mm"] * full["nino_34"]
    )

    full["rain_x_mjo"] = (
        full["rainfall_mm"] * full["mjo_amplitude"]
    )

    # Soil saturation ratio

    full["soil_saturation"] = (
        full["soil_moisture_0_7cm"] / (full["soil_moisture_100_255cm"] + 1e-6)
    )

    # Soil anomaly

    full["sm_anomaly"] = (
        full["soil_moisture_0_7cm"] - full["soil_moisture_0_7cm_roll_mean_6"]
    )

    # Rolling median

    full["rain_roll_median_4"] = g["rainfall_mm"].transform(
        lambda x:x.shift(1).rolling(4).median()
    )

    # =========================================================
    # Raining Features
    # =========================================================

    # API

    def api(series, k=0.9):
        out = np.zeros(len(series))
        for i in range(1,len(series)):
            if np.isnan(series.iloc[i-1]):
                rain = 0
            else:
                rain = series.iloc[i-1]

            out[i] = rain + k*out[i-1]

        return out

    full["api"] = (
        g["rainfall_mm"]
        .transform(api)
    )

    # Rain duration

    def consecutive_rain(x):

        out=[]
        c=0

        for r in x:

            if pd.isna(r):
                out.append(np.nan)

            elif r>0:

                c+=1
                out.append(c)

            else:

                c=0
                out.append(0)

        return pd.Series(out,index=x.index)

    full["rain_duration"] = (
        g["rainfall_mm"]
        .transform(consecutive_rain)
    )

    # Heavy Rain indicator

    thr = full["rainfall_mm"].quantile(0.95)

    full["heavy_rain"] = (
        full["rainfall_mm"] > thr
    ).astype(int)

    # Fraction short vs long rain

    full["recent_rain_ratio"] = (
        full["rain_roll_sum_2"] /
        (full["rainfall_mm_roll_sum_12"]+1)
    )

    # Rain Persistence

    full["rain_events_4"] = (
        g["rainfall_mm"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(4)
            .apply(lambda y:(y>0).sum())
        )
    )


    # ===== buang phantom, gabung lokasi =====
    full = full[full['is_phantom']==0].drop(columns=['is_phantom']).reset_index(drop=True)
    full = full.merge(koor, on='nama_pos', how='left')

    # ===== FITUR STASIUN STATIS (KUNCI: gantikan tma_lag tanpa butuh recursion) =====
    train_mask = full['is_train']==1
    station_stats = full[train_mask].groupby('nama_pos')['tma_mdpl'].agg(
        station_mean_tma='mean', station_min_tma='min', station_max_tma='max').reset_index()
    full = full.merge(station_stats, on='nama_pos', how='left')

    # ===== nearest_station_dist (haversine, dari koordinat) =====
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        dlat, dlon = radians(lat2-lat1), radians(lon2-lon1)
        a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
        return 2*R*atan2(sqrt(a), sqrt(1-a))

    pos_coords = koor.set_index('nama_pos')[['latitude','longitude']]
    nearest_dist = {p1: min(haversine(*pos_coords.loc[p1], *pos_coords.loc[p2])
                            for p2 in pos_coords.index if p2 != p1)
                    for p1 in pos_coords.index}
    full['nearest_station_dist'] = full['nama_pos'].map(nearest_dist)

    # ===== encoding kategorikal =====
    from sklearn.preprocessing import LabelEncoder
    for col in ['nama_pos','landcover_name','mjo_phase','mjo_active']:
        le = LabelEncoder()
        full[col+'_enc'] = le.fit_transform(full[col].astype(str))

    train = full[full['is_train']==1].copy()
    test = full[full['is_train']==0].copy()
    non_feature = ['datetime','nama_pos','is_train','id','tma_mdpl','landcover_name','mjo_phase','mjo_active']
    feature_cols = [c for c in train.columns if c not in non_feature]
    print(train.shape, test.shape, len(feature_cols), 'fitur')
    return full, feature_cols