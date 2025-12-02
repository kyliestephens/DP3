import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import logging
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

#Prefect Tasks
@task
def load_data(db_path: str):
    logging.info(f"Loading data from {db_path}...")
    conn = duckdb.connect(db_path)
    df = conn.execute("SELECT * FROM flight_states").df()
    conn.close()
    logging.info(f"Loaded {len(df)} rows")
    return df

@task
def filter_bbox(df: pd.DataFrame):
    logging.info("Filtering flights to East Coast bounding box...")
    BBOX = {"lamin": 38.5, "lomin": -80.0, "lamax": 47.5, "lomax": -66.0}
    df_bbox = df[
        (df['latitude'].between(BBOX['lamin'], BBOX['lamax'])) &
        (df['longitude'].between(BBOX['lomin'], BBOX['lomax']))
    ]
    logging.info(f"{len(df_bbox)} flights in bounding box")
    return df_bbox

@task
def compute_temporal_features(df: pd.DataFrame):
    logging.info("Computing temporal features...")
    df['hour'] = pd.to_datetime(df['last_contact'], unit='s').dt.hour
    df['minute'] = pd.to_datetime(df['last_contact'], unit='s').dt.minute
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

    #Histogram of altitudes
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

#Prefect flow
@flow
def flight_analysis_flow(db_path: str):
    df = load_data(db_path)
    df = filter_bbox(df)
    df = compute_temporal_features(df)
    compute_altitude_velocity_stats(df)
    plot_flights(df)
    anomalies = detect_anomalies(df)
    if len(anomalies) > 0:
        anomalies.to_csv("anomalies.csv", index=False)
        logging.info("Saved anomalies.csv")
    logging.info("Flow completed!")

if __name__ == "__main__":
    flight_analysis_flow("../Extract/flights.duckdb")

