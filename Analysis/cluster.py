import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

df_coords = pd.read_csv("clusters.csv")
#Airport coordinates (major airports)
AIRPORTS = {
    "JFK": {"lat": 40.6413, "lon": -73.7781},
    "BOS": {"lat": 42.3656, "lon": -71.0096},
    "MIA": {"lat": 25.7959, "lon": -80.2870},
    "ATL": {"lat": 33.6407, "lon": -84.4277},
    "DCA": {"lat": 38.8512, "lon": -77.0402}
}

#Mark noise points
df_coords['cluster_label'] = df_coords['cluster'].apply(lambda x: 'Noise' if x == -1 else f'Cluster {x}')

#Scatter plot
plt.figure(figsize=(12,10))
palette = sns.color_palette('tab10', n_colors=len(df_coords['cluster'].unique()))
sns.scatterplot(
    data=df_coords, 
    x='longitude', y='latitude', 
    hue='cluster_label', 
    palette=palette,
    legend=False,
    s=30, 
    alpha=0.7
)

#Add airports
for airport, coords in AIRPORTS.items():
    plt.scatter(coords['lon'], coords['lat'], marker='X', s=200, color='black')
    plt.text(coords['lon'] + 0.1, coords['lat'] + 0.1, airport, fontsize=12, weight='bold')

plt.title("Clusters of Flights over East Coast (DBSCAN)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.grid(True)
plt.savefig("flight_clusters_annotated.png", bbox_inches='tight')
plt.show()
