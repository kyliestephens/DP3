#!/usr/bin/env python3
import os
import math
import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


DB_PATH = "../Extract/flights.duckdb"
OUT_DIR = "."
BBOX = { "lamin": 38.5, "lomin": -80.0, "lamax": 47.5, "lomax": -66.0 }

AIRPORTS = {
    "JFK": (40.6413, -73.7781),
    "LGA": (40.7769, -73.8740),
    "EWR": (40.6895, -74.1745),
    "BOS": (42.3656, -71.0096),
    "PHL": (39.8744, -75.2424),
    "BWI": (39.1754, -76.6684),
    "DCA": (38.8521, -77.0377),
    "IAD": (38.9531, -77.4565),
    "CHO": (38.1386, -78.4529), 
    "RIC": (37.5052, -77.3190)
}

RAD_TO_DEG = 180 / math.pi


def ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)

def haversine(lat1, lon1, lat2, lon2):
    """Scalar haversine (km)."""
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
    return R * 2 * math.asin(math.sqrt(a))

def haversine_array(lat_arr, lon_arr, lat0, lon0):
    """Vectorized haversine distances in km."""
    lat1 = np.radians(lat_arr.astype(float))
    lon1 = np.radians(lon_arr.astype(float))
    lat2 = math.radians(lat0)
    lon2 = math.radians(lon0)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * c

def connect_and_load():
    conn = duckdb.connect(DB_PATH)
    #Pull relevant columns, filter out null lat/lon (you asked to drop)
    df = conn.execute(f"""
        SELECT
            icao24,
            callsign,
            origin_country,
            time_position,
            last_contact,
            longitude,
            latitude,
            baro_altitude,
            geo_altitude,
            on_ground,
            velocity,
            heading,
            vertical_rate,
            squawk
        FROM flight_states
        WHERE latitude IS NOT NULL AND longitude IS NOT NULL
    """).df()
    conn.close()
    return df

def preprocess(df):
    df = df.copy()
    #ensure numeric types
    for c in ["latitude", "longitude", "velocity", "geo_altitude", "baro_altitude", "last_contact"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    #timestamp (assuming seconds)
    if "last_contact" in df.columns:
        df["ts_utc"] = pd.to_datetime(df["last_contact"], unit="s", utc=True, errors="coerce")
        df["hour_utc"] = df["ts_utc"].dt.hour
    else:
        df["ts_utc"] = pd.NaT
        df["hour_utc"] = np.nan
    return df

def hotspot_and_clusters(df):
    #DBSCAN clustering
    try:
        from sklearn.cluster import DBSCAN
        coords = df[['latitude','longitude']].to_numpy()
        coords_rad = np.radians(coords)
        #eps in radians corresponds to ~5km -> eps = 5/6371
        eps = 5.0 / 6371.0
        clusterer = DBSCAN(eps=eps, min_samples=8, metric='haversine')
        labels = clusterer.fit_predict(coords_rad)
        dfc = df.copy(); dfc['cluster'] = labels
        plt.figure(figsize=(10,8))
        unique = np.unique(labels)
        palette = sns.color_palette("hsv", len(unique))
        for lab, col in zip(unique, palette):
            subset = dfc[dfc['cluster']==lab]
            if lab == -1:
                plt.scatter(subset['longitude'], subset['latitude'], c=[(0.5,0.5,0.5)], s=6, alpha=0.3, label='noise')
            else:
                plt.scatter(subset['longitude'], subset['latitude'], s=10, alpha=0.6, label=f'cluster {lab}')
        plt.title("DBSCAN clusters (approx 5km radius) — install scikit-learn to run")
        plt.xlabel("Longitude"); plt.ylabel("Latitude")
        plt.legend(markerscale=3, bbox_to_anchor=(1.05,1), loc='upper left')
        out_clusters = os.path.join(OUT_DIR, "clusters_dbscan.png")
        plt.tight_layout(); plt.savefig(out_clusters); plt.close()
        print(f"Saved DBSCAN clusters: {out_clusters}")
    except Exception as e:
        print("sklearn not available or DBSCAN failed — skipping clustering. Install scikit-learn to enable.")

def nearest_airport_analysis(df):
    #compute distance to each airport and assign nearest
    airport_names = []
    for k,(lat,lon) in AIRPORTS.items():
        dist_col = f"dist_{k}"
        df[dist_col] = haversine_array(df['latitude'].values, df['longitude'].values, lat, lon)
        airport_names.append(k)
    #find nearest
    dist_cols = [f"dist_{k}" for k in airport_names]
    df['nearest_airport'] = df[dist_cols].idxmin(axis=1).str.replace("dist_","")
    #counts
    counts = df['nearest_airport'].value_counts().reindex(airport_names).fillna(0).astype(int)
    counts_df = counts.reset_index(); counts_df.columns = ['airport','count']
    csv_counts = os.path.join(OUT_DIR, "nearest_airport_counts.csv")
    counts_df.to_csv(csv_counts, index=False)
    print(f"Saved nearest-airport counts CSV: {csv_counts}")

    # bar plot
    plt.figure(figsize=(10,6))
    sns.barplot(data=counts_df, x='airport', y='count', order=counts_df['airport'])
    plt.title("Flights nearest to each major airport (within bbox)")
    plt.xlabel("Airport"); plt.ylabel("Number of flights")
    out = os.path.join(OUT_DIR, "nearest_airport_counts.png")
    plt.tight_layout(); plt.savefig(out); plt.close()
    print(f"Saved nearest-airport bar chart: {out}")

def crossing_side(lat, lon):
    if pd.isna(lat) or pd.isna(lon):
        return "unknown"
    top = abs(lat - BBOX['lamax'])
    bottom = abs(lat - BBOX['lamin'])
    left = abs(lon - BBOX['lomin'])
    right = abs(lon - BBOX['lomax'])
    minv = min(top,bottom,left,right)
    if minv == top:
        return 'N'
    if minv == bottom:
        return 'S'
    if minv == left:
        return 'W'
    if minv == right:
        return 'E'
    return 'unknown'

def main():
    ensure_out_dir()
    print("Loading data from DuckDB...")
    df = connect_and_load()
    if df.empty:
        print("No rows with lat/lon found. Exiting.")
        return
    print(f"Loaded {len(df)} rows with valid coordinates.")
    df = preprocess(df)
    #Hotspot / clustering
    hotspot_and_clusters(df)
    #Nearest airport & plots
    nearest_airport_analysis(df)
    print("All analyses complete. Check PNGs and CSVs in the current directory.")

if __name__ == "__main__":
    main()
