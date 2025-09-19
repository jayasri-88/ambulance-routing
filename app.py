# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from folium.plugins import HeatMap
from streamlit.components.v1 import html
from sklearn.cluster import KMeans
import networkx as nx

st.set_page_config(layout="wide")
st.title("🚑 Ambulance Demand & Routing System")

# --- Load ML model and data ---
model = joblib.load("ambulance_model.pkl")
df = pd.read_csv("emergency_calls.csv")

# --- Ensure Predicted_Emergency column exists ---
if 'Predicted_Emergency' not in df.columns:
    df['Predicted_Emergency'] = np.random.randint(0,5,len(df))

# --- Clip coordinates to city bounds ---
CITY_LAT_MIN, CITY_LAT_MAX = 17.70, 17.75
CITY_LON_MIN, CITY_LON_MAX = 83.30, 83.35

df = df[(df['Latitude'] >= CITY_LAT_MIN) & (df['Latitude'] <= CITY_LAT_MAX) &
        (df['Longitude'] >= CITY_LON_MIN) & (df['Longitude'] <= CITY_LON_MAX)]

# --- Sidebar: Predict Emergency Count ---
st.sidebar.header("📊 Predict Emergency Count")
hour = st.sidebar.number_input("Hour (0-23)", 0, 23, 12)
day = st.sidebar.number_input("Day (1-31)", 1, 31, 15)
weekday = st.sidebar.number_input("Weekday (0=Mon,6=Sun)", 0, 6, 2)
traffic = st.sidebar.slider("Traffic Level (0–1)", 0.0, 1.0, 0.5)

if st.sidebar.button("Predict"):
    X_new = np.array([[hour, day, weekday, traffic]])
    prediction = model.predict(X_new)
    st.sidebar.success(f"Predicted Emergency Count: {int(prediction[0])}")
    df['Predicted_Emergency'] = int(prediction[0])

# --- Function: Create dynamic heatmap ---
@st.cache_data
def create_heatmap(df):
    m = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=13)
    heat_data = [[row['Latitude'], row['Longitude'], row['Predicted_Emergency']] for _, row in df.iterrows()]
    HeatMap(heat_data, radius=15).add_to(m)
    return m._repr_html_()

# --- Function: Create dynamic routes map ---
@st.cache_data
def create_routes_map(df, hospitals, num_clusters=5):
    # KMeans zones
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    df['Zone'] = kmeans.fit_predict(df[['Latitude','Longitude']])
    zone_centers = df.groupby('Zone')[['Latitude','Longitude']].mean().reset_index()

    # Clip zone centers to city bounds
    zone_centers = zone_centers[(zone_centers['Latitude'] >= CITY_LAT_MIN) & (zone_centers['Latitude'] <= CITY_LAT_MAX) &
                                (zone_centers['Longitude'] >= CITY_LON_MIN) & (zone_centers['Longitude'] <= CITY_LON_MAX)]

    # Dynamically assign zone names
    zone_centers = zone_centers.reset_index(drop=True)
    zone_centers['Zone_Name'] = ["Zone_" + str(i) for i in range(len(zone_centers))]

    # Build graph
    G = nx.Graph()
    for h_name, coords_h in hospitals.items():
        G.add_node(h_name, pos=coords_h)
    for _, row in zone_centers.iterrows():
        G.add_node(row['Zone_Name'], pos=(row['Latitude'], row['Longitude']))
    for h_name, coords_h in hospitals.items():
        for _, row in zone_centers.iterrows():
            distance = np.sqrt((coords_h[0]-row['Latitude'])**2 + (coords_h[1]-row['Longitude'])**2)
            traffic_factor = np.random.uniform(1,2)
            G.add_edge(h_name, row['Zone_Name'], weight=distance*traffic_factor)

    # Folium map
    m = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=13)
    # Hospitals
    for h_name, coords_h in hospitals.items():
        folium.Marker(coords_h, popup=h_name, icon=folium.Icon(color='green')).add_to(m)
    # Zones
    for _, row in zone_centers.iterrows():
        folium.Marker([row['Latitude'], row['Longitude']], popup=row['Zone_Name'],
                      icon=folium.Icon(color='red')).add_to(m)
    # Edges
    for edge in G.edges():
        start_coords = G.nodes[edge[0]]['pos']
        end_coords = G.nodes[edge[1]]['pos']
        folium.PolyLine([start_coords, end_coords], color='blue', weight=2.5, opacity=0.7).add_to(m)

    return m._repr_html_(), zone_centers, G

# --- Sidebar: Map options ---
st.sidebar.header("🗺️ Map Options")
map_choice = st.sidebar.radio("Choose Map", ["Emergency Heatmap", "Zones & Routes"])
num_clusters = st.sidebar.slider("Number of Zones", 3, 8, 5)

# Hospitals dictionary
hospitals = {
    'Government Hospital': (17.7320, 83.3140),
    'City Hospital': (17.7345, 83.3180),
    'Private Hospital': (17.7360, 83.3200)
}

# --- Display selected map ---
if map_choice == "Emergency Heatmap":
    st.header("🌡️ Emergency Hotspot Heatmap")
    heatmap_html = create_heatmap(df)
    html(heatmap_html, height=500, width=700)

else:
    st.header("🚦 Zones & Shortest Routes")
    selected_hospital = st.selectbox("Select Hospital for Routes", list(hospitals.keys()))
    routes_map_html, zone_centers, G = create_routes_map(df, hospitals, num_clusters=num_clusters)

    # Display shortest paths
    st.subheader(f"Routes from {selected_hospital}")
    for _, row in zone_centers.iterrows():
        zone_name = row['Zone_Name']
        path = nx.dijkstra_path(G, source=selected_hospital, target=zone_name, weight='weight')
        st.write(f"{selected_hospital} → {zone_name}: {path}")

    st.header("🗺️ Routes Map")
    html(routes_map_html, height=500, width=700)
