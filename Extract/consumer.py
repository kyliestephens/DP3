import duckdb
import json
from quixstreams import Application
from time import sleep


# Kafka Config

KAFKA_BROKER = "localhost:19092"
KAFKA_TOPIC = "flight_states"


# DuckDB Setup

conn = duckdb.connect("flights.duckdb")

conn.execute("""
CREATE TABLE IF NOT EXISTS flight_states (
    icao24 TEXT,
    callsign TEXT,
    origin_country TEXT,
    time_position BIGINT,
    last_contact BIGINT,
    longitude DOUBLE,
    latitude DOUBLE,
    baro_altitude DOUBLE,
    on_ground BOOLEAN,
    velocity DOUBLE,
    heading DOUBLE,
    vertical_rate DOUBLE,
    sensors TEXT,
    geo_altitude DOUBLE,
    squawk TEXT,
    spi BOOLEAN,
    position_source INT
)
""")


# Kafka Consumer

app = Application(broker_address=KAFKA_BROKER, consumer_group="duckdb-consumer")
consumer = app.get_consumer()
consumer.subscribe([KAFKA_TOPIC])

print("Listening for Kafka messages...")


# Infinite consume loop

while True:
    for msg in consumer.consume():
        if msg is None:
            continue

        try:
            flight_json = msg.value()  #get the message payload
            flight = json.loads(flight_json)

            conn.execute("""
                INSERT INTO flight_states VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, [
                flight.get("icao24"),
                flight.get("callsign"),
                flight.get("origin_country"),
                flight.get("time_position"),
                flight.get("last_contact"),
                flight.get("longitude"),
                flight.get("latitude"),
                flight.get("baro_altitude"),
                flight.get("on_ground"),
                flight.get("velocity"),
                flight.get("heading"),
                flight.get("vertical_rate"),
                json.dumps(flight.get("sensors")),
                flight.get("geo_altitude"),
                flight.get("squawk"),
                flight.get("spi"),
                flight.get("position_source")
            ])
            print(f"Inserted flight {flight.get('icao24')}")

        except Exception as e:
            print("Error inserting flight:", e)

    #Prevent tight loop when there are no new messages
    sleep(.1) #.1 so it goes fast
