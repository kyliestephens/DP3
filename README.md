# DP3
## Kylie Stephens

## Data Source: 
OPENSKY_URL = "https://opensky-network.org/api/states/all"
(live flight data)

## Overview 
For my DS3022 project, I built a full data pipeline using Kafka, Prefect, and DuckDB to collect, store, and analyze flight data from OPENSKY. I chose this dataset because flight information is fast-moving, structured, and realistic for demonstrating modern data engineering patterns- streaming ingestion, transformation, cleaning, analysis, and visualizations. It was interesting to see the density of flights and what hubs had the most flights within their vicinities. It was also interesting to see that not many flights passed within range of CHO, which is why I ran a specific analysis script for our home airport. It was a miniscule number compared to the over 50,000 flights ingested. 
Main skills used: Kafka streaming and consumer, DuckDB, transformations + basic data descriptions, and analysis in python scripts (some using prefect for workflow)

## Main Challenges:
- Rate limit- had to change from 50s to 60s in order to avoid getting locked out
- Choosing range for flight data (geographic range)
- Consumer speed - initially had 1 second for sleep but changed to .1 for speed purposes (considered doing a batch but it was going fast enough)
- getting enough data for plots to be significant
- Most data did not require crazy transformations
- DBT was giving me issues so I used prefect for flows instead
- Needed to run "docker compose up -d" twice in order to get kafka running (only zookeeper would run the first time)


## Github Repository Link: 
https://github.com/kyliestephens/DP3
