import duckdb
import pandas as pd
import numpy as np
import logging
from prefect import flow, task, get_run_logger  

#Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


@task
def load_raw(db_path):
    logging.info(f"Loading raw flight data from {db_path}...")
    conn = duckdb.connect(db_path)
    df = conn.execute("SELECT * FROM flight_states").df()
    conn.close()
    logging.info(f"Loaded {len(df):,} rows and {len(df.columns)} columns.")
    return df

#Basic Dataset Description
@task
def describe_data(df):
    logger = get_run_logger()
    
    logger.info(f"DataFrame loaded with {len(df)} rows and {len(df.columns)} columns.")
    logger.info(f"Columns: {list(df.columns)}")

    #stat description
    desc = df.describe(include="all")
    print(desc)
    logger.info(f"Head of dataset:\n{df.head()}")

    return desc

@task
def summarize_data(df):
    logging.info("Summarizing dataset...")

    #Column list
    logging.info(f"Columns ({len(df.columns)}): {list(df.columns)}")

    #Missing values
    missing = df.isna().sum()
    logging.info("Missing values per column:")
    for col, val in missing.items():
        logging.info(f"  {col}: {val}")

    #Basic stats for numeric columns
    logging.info("Basic numeric statistics:")
    logging.info(df.describe(include='all').to_string())

    return df 

#Cleaning
@task
def clean_data(df):
    logging.info("Cleaning data: dropping invalid coordinates & standardizing callsigns...")

    before = len(df)
    df = df[(df["latitude"].between(-90, 90)) & (df["longitude"].between(-180, 180))]
    df["callsign"] = df["callsign"].fillna("").str.strip().str.upper()

    logging.info(f"Removed {before - len(df):,} rows with invalid coordinates.")
    return df

#Temporal Features
@task
def add_temporal_features(df):
    logging.info("Adding temporal features (timestamp, hour, day_of_week)...")

    df["timestamp"] = pd.to_datetime(df["last_contact"], unit="s", errors="coerce")
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    return df

#Feature Engineering
@task
def add_feature_engineering(df):
    logging.info("Adding feature engineering (altitude_change, speed_change)...")

    df = df.sort_values(["icao24", "last_contact"])

    df["altitude_change"] = df.groupby("icao24")["geo_altitude"].diff()
    df["speed_change"] = df.groupby("icao24")["velocity"].diff()

    return df

@task
def save_transformed(df, out_path):
    logging.info(f"Saving transformed dataset to {out_path}...")
    conn = duckdb.connect(out_path)
    conn.execute("CREATE OR REPLACE TABLE flight_states_transformed AS SELECT * FROM df")
    conn.close()
    logging.info("Saved transformed dataset successfully.")

#Prefect Flow
@flow(name="Flight Data Transformation Flow")
def transform_flow():

    df = load_raw("../Extract/flights.duckdb")

    describe_data(df) 

    df = summarize_data(df)

    df = clean_data(df)

    df = add_temporal_features(df)

    df = add_feature_engineering(df)

    save_transformed(df, "../Extract/transformed_flights.duckdb")

    logging.info("Transformation flow complete!")

if __name__ == "__main__":
    transform_flow()
