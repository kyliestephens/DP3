#!/usr/bin/env python3
"""
analysis_all.py
Runs three analyses on the flight data restricted to the Northeast bounding box:
  2) Velocity distribution & outlier detection
  3) Spatial density (binned heatmap + KDE)
Saves plots to PNG files in the current directory.
"""

import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Config ---
DB_PATH = "../Extract/flights.duckdb"  
BBOX = {
    "lamin": 25.0,   #Florida
    "lomin": -90.0,  #Louisiana-ish
    "lamax": 47.5,   #Maine
    "lomax": -66.0   #Atlantic
}
OUT_DIR = "."   

sns.set(style="whitegrid")

def ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)

def connect_db():
    return duckdb.connect(DB_PATH)

def create_staging(conn):
    conn.execute(f"""
    CREATE OR REPLACE TABLE flight_staging AS
    SELECT *
    FROM flight_states
    WHERE icao24 IS NOT NULL
      AND latitude BETWEEN {BBOX['lamin']} AND {BBOX['lamax']}
      AND longitude BETWEEN {BBOX['lomin']} AND {BBOX['lomax']}
    """)
    print("Staging table created (flight_staging).")


def velocity_analysis(conn):
    #Pull needed columns
    df = conn.execute("""
    SELECT velocity, geo_altitude, last_contact
    FROM flight_staging
    WHERE velocity IS NOT NULL
    """).df()

    if df.empty:
        print("No velocity data available in flight_staging.")
        return

    #Basic cleaning: drop non-positive velocities, and handle NaNs
    df = df[df['velocity'] > 0].dropna(subset=['velocity'])

    #Histogram + KDE of velocity
    plt.figure(figsize=(10,6))
    sns.histplot(df['velocity'], bins=60, kde=True)
    plt.xlabel("Velocity (m/s)")
    plt.title("Velocity distribution (Eastcoast US)")
    plt.tight_layout()
    out_hist = os.path.join(OUT_DIR, "velocity_distribution.png")
    plt.savefig(out_hist)
    plt.close()
    print(f"Saved velocity distribution: {out_hist}")

    #Scatter velocity vs altitude (if altitude present)
    if 'geo_altitude' in df.columns and not df['geo_altitude'].isna().all():
        plt.figure(figsize=(10,6))
        sns.scatterplot(x='velocity', y='geo_altitude', data=df, alpha=0.6, s=10)
        plt.xlabel("Velocity (m/s)")
        plt.ylabel("Geo Altitude (m)")
        plt.title("Velocity vs Geo Altitude (Eastcoast US)")
        plt.tight_layout()
        out_scatter = os.path.join(OUT_DIR, "velocity_vs_altitude.png")
        plt.savefig(out_scatter)
        plt.close()
        print(f"Saved velocity vs altitude scatter: {out_scatter}")

    #Outlier detection: IQR method + z-score top outliers
    q1 = df['velocity'].quantile(0.25)
    q3 = df['velocity'].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    iqr_outliers = df[(df['velocity'] < lower) | (df['velocity'] > upper)].copy()
    #z-score for top extreme velocities
    df['v_z'] = (df['velocity'] - df['velocity'].mean()) / df['velocity'].std()
    z_outliers = df[df['v_z'].abs() > 3]
    print("\nVelocity outliers (IQR method) - sample:")
    print(iqr_outliers.head(10))
    print("\nVelocity outliers (z-score |z|>3) - sample:")
    print(z_outliers.head(10))

#Spatial
def spatial_density(conn):
    df = conn.execute("""
    SELECT latitude, longitude
    FROM flight_staging
    WHERE latitude IS NOT NULL AND longitude IS NOT NULL
    """).df()

    if df.empty:
        print("No spatial points to plot.")
        return

    #KDE heatmap (continuous density)
    plt.figure(figsize=(10,8))
    sns.kdeplot(x=df['longitude'], y=df['latitude'], fill=True, thresh=0.05, levels=100, cmap='Reds')
    plt.title("Flight Density (KDE) for Eastcoast US")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    out_kde = os.path.join(OUT_DIR, "density_kde.png")
    plt.savefig(out_kde)
    plt.close()
    print(f"Saved KDE density plot: {out_kde}")

def main():
    ensure_out_dir()
    conn = connect_db()
    try:
        create_staging(conn)
        velocity_analysis(conn)
        spatial_density(conn)
    finally:
        conn.close()
        print("Done. Connection closed.")

if __name__ == "__main__":
    main()




