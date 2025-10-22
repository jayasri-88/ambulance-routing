import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap
import joblib
import numpy as np
from streamlit.components.v1 import html

# -------------------------
# Helper function to render folium map in iframe
# -------------------------
def show_map(m, height=500, width=700):
    map_html = m._repr_html_()
    html(map_html, height=height, width=width)

# -------------------------
# Load trained model
# -------------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load("ambulance_model.pkl")
    except:
        return None

model = load_model()

# -------------------------
# Sidebar Inputs
# -------------------------
st.sidebar.header("🚑 Emergency Input")

emergency_type = st.sidebar.selectbox("Emergency Type", ["Medical", "Fire", "Accident"])
traffic_level = st.sidebar.slider("Traffic Level (1 = Low, 5 = High)", 1, 5, 3)
hospital = st.sidebar.selectbox("Nearest Hospital", ["Government Hospital", "City Hospital", "Private Hospital"])
hour = st.sidebar.slider("Hour of Day", 0, 23, 12)
day = st.sidebar.slider("Day of Month", 1, 31, 15)
weekday = st.sidebar.selectbox("Weekday", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])

predict_button = st.sidebar.button("🔮 Predict & Generate Maps")

# -------------------------
# Main UI
# -------------------------
st.title("🚑 Ambulance Demand & Routing")
st.write("This prototype predicts emergency demand and suggests hospital routes based on past patterns.")

# -------------------------
# Prediction & Map Generation
# -------------------------
if predict_button:
    # Create input dataframe for model
    input_df = pd.DataFrame([{
        "Hour": hour,
        "Day": day,
        "Weekday": ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"].index(weekday),
        "Traffic_Level": traffic_level / 5,
        "Emergency_Type": emergency_type,
        "Hospital": hospital
    }])

    prediction = None
    if model:
        try:
            prediction = int(model.predict(input_df)[0])
        except:
            prediction = np.random.randint(0, 10)  # fallback
    else:
        prediction = np.random.randint(0, 10)

    st.success(f"🚨 Predicted Emergencies: **{prediction}**")

    # -------------------------
    # Hospitals dictionary
    # -------------------------
    hospitals_dict = {
        'Government Hospital': (17.7320, 83.3140),
        'City Hospital': (17.7345, 83.3180),
        'Private Hospital': (17.7360, 83.3200)
    }

    # -------------------------
    # Determine dynamic map center
    # -------------------------
    map_center = hospitals_dict.get(hospital, [17.7333, 83.3167])

    # -------------------------
    # Heatmap (synthetic demo)
    # -------------------------
    heat_map = folium.Map(location=map_center, zoom_start=13)
    heat_data = [[
        np.random.uniform(map_center[0]-0.03, map_center[0]+0.03),
        np.random.uniform(map_center[1]-0.03, map_center[1]+0.03),
        np.random.randint(1, 5)
    ] for _ in range(200)]
    HeatMap(heat_data, radius=15, max_zoom=13).add_to(heat_map)

    # -------------------------
    # Routes Map (demo with hospitals & zones)
    # -------------------------
    m_routes = folium.Map(location=map_center, zoom_start=13)
    # Add hospitals
    for h_name, coords_h in hospitals_dict.items():
        folium.Marker(coords_h, popup=h_name, icon=folium.Icon(color='green')).add_to(m_routes)
    # Add some dummy zones near selected hospital
    for i in range(5):
        lat, lon = np.random.uniform(map_center[0]-0.03, map_center[0]+0.03), np.random.uniform(map_center[1]-0.03, map_center[1]+0.03)
        folium.Marker([lat, lon], popup=f"Zone {i}", icon=folium.Icon(color='red')).add_to(m_routes)
        folium.PolyLine([hospitals_dict[hospital], (lat, lon)], color='blue').add_to(m_routes)

    # -------------------------
    # Show maps in tabs
    # -------------------------
    tab1, tab2 = st.tabs(["🔥 Heatmap", "🛣️ Routes"])
    with tab1:
        show_map(heat_map)
    with tab2:
        show_map(m_routes)

else:
    st.info("⬅️ Please enter details in the sidebar and click **Predict & Generate Maps**.")
