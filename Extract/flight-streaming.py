import os
import logging
import json
import sys
import requests
import pandas as pd
from tornado.ioloop import IOLoop, PeriodicCallback
from quixstreams import Application
import time

# ---------------------------------------------------
# Configuration
# ---------------------------------------------------
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:19092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "flight_states")

BBOX = {
    "lamin": 25.0,   # Florida
    "lomin": -90.0,  # Louisiana-ish
    "lamax": 47.5,   # Maine
    "lomax": -66.0   # Atlantic
}

# Polling interval (ms) - increase to avoid OpenSky rate limits
POLL_INTERVAL_MS = int(os.getenv("POLL_INTERVAL_MS", 60000))  # 60 seconds

# OpenSky API credentials (optional)
OPENSKY_USERNAME = os.getenv("OPENSKY_USER")  # None if not set
OPENSKY_PASSWORD = os.getenv("OPENSKY_PASS")

OPENSKY_URL = "https://opensky-network.org/api/states/all"

# ---------------------------------------------------
# Logging setup
# ---------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------
# OpenSky flight columns
# ---------------------------------------------------
COLUMNS = [
    "icao24", "callsign", "origin_country", "time_position", "last_contact",
    "longitude", "latitude", "baro_altitude", "on_ground", "velocity",
    "heading", "vertical_rate", "sensors", "geo_altitude", "squawk", "spi", "position_source"
]

# ---------------------------------------------------
# Kafka / Quix Application
# ---------------------------------------------------
app = Application(broker_address=KAFKA_BROKER, consumer_group="opensky-producer")
producer = app.get_producer()

# ---------------------------------------------------
# Fetch, format, and send flights
# ---------------------------------------------------
def fetch_flights():
    try:
        # Only use auth if both username and password exist
        auth = (OPENSKY_USERNAME, OPENSKY_PASSWORD) if OPENSKY_USERNAME and OPENSKY_PASSWORD else None

        response = requests.get(OPENSKY_URL, params=BBOX, auth=auth, timeout=10)
        response.raise_for_status()
        data = response.json()
        states = data.get("states")

        if not states:
            logger.warning("No flight data received from OpenSky.")
            return

        # Convert to pandas DataFrame
        df = pd.DataFrame(states, columns=COLUMNS)
        logger.info(f"Fetched {len(df)} flights.")

        # Convert each row to JSON and send to Kafka
        for _, row in df.iterrows():
            flight_json = row.to_dict()
            producer.produce(topic=KAFKA_TOPIC, value=json.dumps(flight_json))

        producer.flush()
        logger.info(f"Sent {len(df)} flights to Kafka topic '{KAFKA_TOPIC}'.")

    except requests.exceptions.HTTPError as e:
        if response.status_code == 429:
            logger.warning("Rate limit reached. Sleeping 60 seconds before retry...")
            time.sleep(60)
        else:
            logger.error(f"HTTP error fetching flight data: {e}")

    except Exception as e:
        logger.error(f"Error fetching/sending flight data: {e}")

# ---------------------------------------------------
# Main loop
# ---------------------------------------------------
if __name__ == "__main__":
    logger.info("Starting OpenSky → Kafka producer")
    logger.info(f"Kafka broker: {KAFKA_BROKER}")
    logger.info(f"Kafka topic: {KAFKA_TOPIC}")
    logger.info(f"Polling every {POLL_INTERVAL_MS / 1000} seconds")

    poller = PeriodicCallback(fetch_flights, POLL_INTERVAL_MS)
    poller.start()

    try:
        IOLoop.current().start()
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
        IOLoop.current().stop()
