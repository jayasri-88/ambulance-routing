# Cell 1: Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
import folium
from folium.plugins import HeatMap
import networkx as nx
import joblib

# Cell 2: Generate synthetic emergency calls dataset
num_records = 500
timestamps = pd.date_range('2025-01-01', periods=num_records, freq='H')
latitudes = np.random.uniform(17.70, 17.75, num_records)
longitudes = np.random.uniform(83.30, 83.35, num_records)
emergency_types = np.random.choice(['Medical', 'Fire', 'Accident'], num_records)
traffic_levels = np.random.randint(1, 6, num_records)
hospitals = np.random.choice(['Government Hospital', 'City Hospital', 'Private Hospital'], num_records)

df = pd.DataFrame({
    'Timestamp': timestamps,
    'Latitude': latitudes,
    'Longitude': longitudes,
    'Emergency_Type': emergency_types,
    'Traffic_Level': traffic_levels,
    'Hospital': hospitals
})

df.to_csv('emergency_calls.csv', index=False)
print(df.head())

# Cell 3: Preprocess dataset
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Hour'] = df['Timestamp'].dt.hour
df['Day'] = df['Timestamp'].dt.day
df['Weekday'] = df['Timestamp'].dt.weekday
df['Traffic_Level'] = df['Traffic_Level'] / df['Traffic_Level'].max()
df['Emergency_Count'] = np.random.randint(0, 5, size=len(df))
print(df.head())

# Cell 4: Train RandomForest model with categorical features
X = df[['Hour', 'Day', 'Weekday', 'Traffic_Level', 'Emergency_Type', 'Hospital']]
y = df['Emergency_Count']

# Preprocessing: OneHotEncode categorical vars
categorical_features = ['Emergency_Type', 'Hospital']
numeric_features = ['Hour', 'Day', 'Weekday', 'Traffic_Level']

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)

# Build pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# Evaluate
predictions = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Save the pipeline (model + preprocessing)
joblib.dump(pipeline, "ambulance_model.pkl")
print("✅ Model saved as ambulance_model.pkl")

# Cell 5: KMeans clustering for zones
from sklearn.cluster import KMeans
coords = df[['Latitude','Longitude']]
num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df['Zone'] = kmeans.fit_predict(coords)
zone_centers = df.groupby('Zone')[['Latitude','Longitude']].mean().reset_index()
print(zone_centers)

# Cell 6: Generate routes map
hospitals_dict = {
    'Government Hospital': (17.7320, 83.3140),
    'City Hospital': (17.7345, 83.3180),
    'Private Hospital': (17.7360, 83.3200)
}

G = nx.Graph()
for h_name, coords_h in hospitals_dict.items():
    G.add_node(h_name, pos=coords_h)
for idx, row in zone_centers.iterrows():
    zone_name = f"Zone_{int(row['Zone'])}"
    G.add_node(zone_name, pos=(row['Latitude'], row['Longitude']))
for h_name, h_coords in hospitals_dict.items():
    for idx, row in zone_centers.iterrows():
        zone_name = f"Zone_{int(row['Zone'])}"
        distance = np.sqrt((h_coords[0]-row['Latitude'])**2 + (h_coords[1]-row['Longitude'])**2)
        traffic_factor = np.random.uniform(1, 2)
        G.add_edge(h_name, zone_name, weight=distance*traffic_factor)

# Print routes from each hospital
for h_name in hospitals_dict.keys():
    print(f"Routes from {h_name}:")
    for idx, row in zone_centers.iterrows():
        zone_name = f"Zone_{int(row['Zone'])}"
        path = nx.dijkstra_path(G, source=h_name, target=zone_name, weight='weight')
        print(f"  {zone_name} -> {path}")

# Cell 7: Visualize routes on Folium map
m_routes = folium.Map(location=[17.7333, 83.3167], zoom_start=13)
for h_name, coords_h in hospitals_dict.items():
    folium.Marker(coords_h, popup=h_name, icon=folium.Icon(color='green')).add_to(m_routes)
for idx, row in zone_centers.iterrows():
    zone_name = f"Zone_{int(row['Zone'])}"
    folium.Marker([row['Latitude'], row['Longitude']], popup=zone_name, icon=folium.Icon(color='red')).add_to(m_routes)
for edge in G.edges():
    start_coords = G.nodes[edge[0]]['pos']
    end_coords = G.nodes[edge[1]]['pos']
    folium.PolyLine([start_coords, end_coords], color='blue', weight=2.5, opacity=0.7).add_to(m_routes)

m_routes.save("ambulance_routes_map.html")

# Cell 8: Generate emergency heatmap
if 'Predicted_Emergency' not in df.columns:
    df['Predicted_Emergency'] = pipeline.predict(X)  # Use model predictions

heat_map = folium.Map(location=[17.7333, 83.3167], zoom_start=13)
heat_data = [[row['Latitude'], row['Longitude'], row['Predicted_Emergency']] for idx, row in df.iterrows()]
HeatMap(heat_data, radius=15, max_zoom=13).add_to(heat_map)
heat_map.save("ambulance_demand_heatmap.html")
