#!/usr/bin/env python3
"""
CHO radius analysis:
- Reads flight_states from DuckDB
- Drops rows with null lat/lon
- Computes haversine distance to CHO (38.1386, -78.4529)
- Produces visuals and prints counts for several radii
"""

import os
import math
import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timezone

sns.set(style="whitegrid")

#CHO coords
CHO_LAT = 38.1386
CHO_LON = -78.4529

DB_PATH = "../Extract/flights.duckdb"
OUT_DIR = "."  #output PNGs written here

#Radii in kilometers we care about
RADII_KM = [10, 25, 50, 100]

def haversine_array(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance (km) between two arrays/scalars of lat/lon (degrees)."""
    #convert degrees to radians
    lat1_r = np.radians(lat1)
    lat2_r = np.radians(lat2)
    lon1_r = np.radians(lon1)
    lon2_r = np.radians(lon2)

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    R = 6371.0  #Earth radius in kilometers
    return R * c

def ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)

def load_data():
    conn = duckdb.connect(DB_PATH)
    df = conn.execute("""
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
    #Ensure numeric types
    df = df.copy()
    #convert to numeric gracefully
    for c in ["latitude", "longitude", "velocity", "geo_altitude", "baro_altitude", "last_contact"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    #drop rows missing lat/lon after coercion
    df = df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)

    #compute datetime (assume last_contact is unix epoch seconds)
    if "last_contact" in df.columns:
        #some rows might have last_contact null -> leave as NaT
        df["timestamp_utc"] = pd.to_datetime(df["last_contact"], unit="s", utc=True, errors="coerce")
        #add hour column (0-23) in UTC
        df["hour_utc"] = df["timestamp_utc"].dt.hour
    else:
        df["timestamp_utc"] = pd.NaT
        df["hour_utc"] = np.nan

    #compute distance to CHO
    df["distance_km"] = haversine_array(df["latitude"].values, df["longitude"].values,
                                        CHO_LAT, CHO_LON)
    return df

def summary_counts(df):
    print("\nCounts within radii of CHO (km):")
    for r in RADII_KM:
        c = (df["distance_km"] <= r).sum()
        print(f"  <= {r} km : {c} flights")

def scatter_with_radii(df):
    plt.figure(figsize=(10,10))
    ax = plt.gca()
    sc = ax.scatter(df["longitude"], df["latitude"], c=df["distance_km"], cmap="viridis", s=8, alpha=0.6)
    plt.colorbar(sc, label="Distance to CHO (km)")
    #plot CHO
    ax.scatter(CHO_LON, CHO_LAT, c="red", s=80, marker="x", label="CHO")
    #draw circles for radii (approx: degrees - draw as circles in lon/lat using haversine-based circle approximation)
    theta = np.linspace(0, 2 * np.pi, 200)
    for r, color in zip(RADII_KM, ["#ff9999", "#ffcc99", "#99ccff", "#c0c0c0"]):
        #compute lat/lon offsets by small steps: create points at given distance r
        #approximate: for each angle compute destination using great-circle formula
        lat0 = math.radians(CHO_LAT)
        lon0 = math.radians(CHO_LON)
        R = 6371.0
        lat_pts = []
        lon_pts = []
        for t in theta:
            lat_pt = math.asin(math.sin(lat0) * math.cos(r / R) + math.cos(lat0) * math.sin(r / R) * math.cos(t))
            lon_pt = lon0 + math.atan2(math.sin(t) * math.sin(r / R) * math.cos(lat0),
                                       math.cos(r / R) - math.sin(lat0) * math.sin(lat_pt))
            lat_pts.append(math.degrees(lat_pt))
            lon_pts.append(math.degrees(lon_pt))
        ax.plot(lon_pts, lat_pts, color=color, linewidth=1.2, alpha=0.8, label=f"{r} km")
    ax.set_title("Flights around CHO with radius overlays")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="upper right")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "cho_scatter_radii.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved scatter with radii: {out}")

def velocity_distribution_nearby(df, radius_km=50):
    subset = df[df["distance_km"] <= radius_km]
    if subset.empty:
        print(f"No flights within {radius_km} km to analyze velocity.")
        return
    plt.figure(figsize=(10,6))
    sns.histplot(subset["velocity"].dropna(), bins=50, kde=True)
    plt.xlabel("Velocity (m/s)")
    plt.title(f"Velocity distribution for flights within {radius_km} km of CHO")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"cho_velocity_distribution_{radius_km}km.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved velocity distribution: {out}")

def density_kde(df):
    if df.empty:
        print("No spatial points to plot KDE.")
        return
    plt.figure(figsize=(10,8))
    sns.kdeplot(x=df["longitude"], y=df["latitude"], fill=True, thresh=0.05, levels=100, cmap="magma")
    plt.scatter([CHO_LON], [CHO_LAT], c="white", s=80, marker="x", label="CHO")
    plt.title("Spatial KDE of flights around CHO")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "cho_density_kde.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved KDE density plot: {out}")

def try_dbscan_clusters(df):
    try:
        from sklearn.cluster import DBSCAN
    except Exception as e:
        print("sklearn not available, skipping DBSCAN clustering. (Install scikit-learn to enable clustering.)")
        return

    coords = df[["latitude", "longitude"]].to_numpy()
    # Haversine-aware clustering by converting to radians and using haversine metric if desired
    #DBSCAN with haversine metric on radian coords
    coords_rad = np.radians(coords)
    clustering = DBSCAN(eps=0.05, min_samples=10, metric="haversine").fit(coords_rad)
    labels = clustering.labels_
    dfc = df.copy()
    dfc["cluster"] = labels

    plt.figure(figsize=(10,8))
    unique_labels = np.unique(labels)
    palette = sns.color_palette("hsv", len(unique_labels))
    for lab, col in zip(unique_labels, palette):
        sub = dfc[dfc["cluster"] == lab]
        if lab == -1:
            #Noise
            plt.scatter(sub["longitude"], sub["latitude"], c=[(0.5,0.5,0.5)], s=8, alpha=0.3, label="noise")
        else:
            plt.scatter(sub["longitude"], sub["latitude"], s=8, alpha=0.6, label=f"cluster {lab}")
    plt.scatter([CHO_LON], [CHO_LAT], c="black", s=80, marker="x")
    plt.title("DBSCAN clusters around CHO (if sklearn available)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend(markerscale=3, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "cho_clusters.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved clustering plot: {out}")

def main():
    ensure_out_dir()
    print("Loading data from DuckDB...")
    df = load_data()
    if df.empty:
        print("No rows with lat/lon found in flight_states. Exiting.")
        return
    print(f"Loaded {len(df)} rows with lat/lon.")
    df = preprocess(df)
    summary_counts(df)
    scatter_with_radii(df)
    velocity_distribution_nearby(df, radius_km=50)
    density_kde(df)
    #clustering optional
    try_dbscan_clusters(df)
    print("Done. Output PNGs saved in current directory.")

if __name__ == "__main__":
    main()
