import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import numpy as np
from sklearn.cluster import DBSCAN
from prefect import flow, task

#Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("flight_analysis.log"),
        logging.StreamHandler()
    ]
)


#Eastcoast bounds
BBOX = {"lamin": 38.5, "lomin": -80.0, "lamax": 47.5, "lomax": -66.0}

#Major airports: JFK, BOS, MIA, ATL, DCA
AIRPORTS = {
    "JFK": {"lat": 40.6413, "lon": -73.7781},
    "BOS": {"lat": 42.3656, "lon": -71.0096},
    "MIA": {"lat": 25.7959, "lon": -80.2870},
    "ATL": {"lat": 33.6407, "lon": -84.4277},
    "DCA": {"lat": 38.8512, "lon": -77.0402}
}


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2*np.arcsin(np.sqrt(a))
    return R * c

#Prefect Tasks
@task
def load_data(db_path: str):
    logging.info(f"Loading data from {db_path}...")
    conn = duckdb.connect(db_path, read_only=True)
    df = conn.execute("SELECT * FROM flight_states").df()
    conn.close()
    logging.info(f"Loaded {len(df)} rows")
    return df

@task
def filter_bbox(df: pd.DataFrame):
    logging.info("Filtering flights to East Coast bounding box...")
    df_bbox = df[
        (df['latitude'].between(BBOX['lamin'], BBOX['lamax'])) &
        (df['longitude'].between(BBOX['lomin'], BBOX['lomax']))
    ]
    logging.info(f"{len(df_bbox)} flights in bounding box")
    return df_bbox

@task
def compute_temporal_features(df: pd.DataFrame):
    logging.info("Computing temporal features...")
    df['last_contact_ts'] = pd.to_datetime(df['last_contact'], unit='s')
    df['hour'] = df['last_contact_ts'].dt.hour
    df['minute'] = df['last_contact_ts'].dt.minute
    return df

@task
def compute_altitude_velocity_stats(df: pd.DataFrame):
    logging.info("Computing altitude and velocity stats...")
    stats = df[['geo_altitude', 'velocity']].describe()
    logging.info(f"\n{stats}")
    return stats

@task
def plot_flights(df: pd.DataFrame):
    logging.info("Generating plots...")

    #Scatter map colored by altitude
    plt.figure(figsize=(10,8))
    sns.scatterplot(data=df, x='longitude', y='latitude', hue='geo_altitude', palette='viridis', s=20)
    plt.title("Flights over East Coast (colored by altitude)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.savefig("flights_scatter_altitude.png")
    logging.info("Saved flights_scatter_altitude.png")

    #Altitude histogram
    plt.figure(figsize=(8,6))
    sns.histplot(df['geo_altitude'].dropna(), bins=30, kde=True)
    plt.title("Altitude Distribution")
    plt.xlabel("Altitude (meters)")
    plt.ylabel("Flight Count")
    plt.savefig("altitude_histogram.png")
    logging.info("Saved altitude_histogram.png")

@task
def detect_anomalies(df: pd.DataFrame):
    logging.info("Detecting anomalies...")
    threshold_altitude = df['geo_altitude'].mean() + 2*df['geo_altitude'].std()
    anomalies = df[df['geo_altitude'] > threshold_altitude]
    logging.info(f"Detected {len(anomalies)} high-altitude anomalies")
    return anomalies

@task
def flights_within_airports(df: pd.DataFrame):
    logging.info("Calculating flights within radii of major airports...")
    radii = [10, 25, 50, 100]
    results = {}
    for airport, coords in AIRPORTS.items():
        airport_counts = {}
        for r in radii:
            count = df.apply(lambda row: haversine(row['latitude'], row['longitude'], coords['lat'], coords['lon']) <= r, axis=1).sum()
            airport_counts[r] = count
        results[airport] = airport_counts
        logging.info(f"{airport}: {airport_counts}")
    return results

@task
def cluster_flights(df: pd.DataFrame):
    logging.info("Clustering dense flight regions using DBSCAN...")
    df_coords = df[['latitude', 'longitude']].dropna()
    if len(df_coords) == 0:
        logging.info("No data to cluster")
        return None
    kms_per_radian = 6371.0088
    epsilon = 10 / kms_per_radian
    db = DBSCAN(eps=epsilon, min_samples=5, algorithm='ball_tree', metric='haversine')
    coords_rad = np.radians(df_coords)
    db.fit(coords_rad)
    df_coords['cluster'] = db.labels_
    logging.info(f"Number of clusters detected: {len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)}")
    
    #Save cluster visualization
    plt.figure(figsize=(10,8))
    sns.scatterplot(data=df_coords, x='longitude', y='latitude', hue='cluster', palette='tab10', s=20)
    plt.title("Clusters of Flights over East Coast")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.savefig("flight_clusters.png")
    logging.info("Saved flight_clusters.png")

    return df_coords


#Prefect Flow
@flow
def flight_analysis_flow(db_path: str):
    df = load_data(db_path)
    df = filter_bbox(df)
    df = compute_temporal_features(df)
    compute_altitude_velocity_stats(df)
    plot_flights(df)
    anomalies = detect_anomalies(df)
    flights_within_airports(df)
    cluster_df = cluster_flights(df)

    #Save anomalies and clusters
    if anomalies is not None and len(anomalies) > 0:
        anomalies.to_csv("anomalies.csv", index=False)
        logging.info("Saved anomalies.csv")
    if cluster_df is not None:
        cluster_df.to_csv("clusters.csv", index=False)
        logging.info("Saved clusters.csv")

    logging.info("Flow completed!")


if __name__ == "__main__":
    flight_analysis_flow("../Extract/flights.duckdb")
