import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------

# Load Dataset

# ----------------------------------

@st.cache_data
def load_data():
import os

```
# Works both locally and Streamlit Cloud
if os.path.exists("cleaned_vehicle_data.csv"):
    df = pd.read_csv("cleaned_vehicle_data.csv")
else:
    df = pd.read_csv("/mnt/data/cleaned_vehicle_data.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])
return df
```

df = load_data()

# ----------------------------------

# Dashboard Title

# ----------------------------------

st.set_page_config(page_title="Automotive Intelligence", layout="wide")
st.title("🚗 Automotive Data Intelligence Dashboard")
st.write("Analyze driving behavior, engine performance & vehicle health.")

# ----------------------------------

# Sidebar Filters

# ----------------------------------

st.sidebar.header("Filters")

min_time = df["timestamp"].min().to_pydatetime()
max_time = df["timestamp"].max().to_pydatetime()

time_range = st.sidebar.slider(
"Select Time Range",
min_value=min_time,
max_value=max_time,
value=(min_time, max_time)
)

df_f = df[(df["timestamp"] >= time_range[0]) & (df["timestamp"] <= time_range[1])]

# ----------------------------------

# KPI Metrics

# ----------------------------------

st.subheader("📊 Key Metrics")

avg_speed = df_f["speed"].mean()
max_rpm = df_f["rpm"].max()
fuel_used = df_f["fuel_level"].iloc[0] - df_f["fuel_level"].iloc[-1]

col1, col2, col3 = st.columns(3)
col1.metric("Average Speed", f"{avg_speed:.2f} km/h")
col2.metric("Max RPM", f"{max_rpm:.0f}")
col3.metric("Fuel Consumed", f"{fuel_used:.2f} %")

# ----------------------------------

# Time-Series Visualizations

# ----------------------------------

st.subheader("📈 Driving and Engine Performance")

# Speed

fig1, ax1 = plt.subplots(figsize=(10, 3))
ax1.plot(df_f["timestamp"], df_f["speed"])
ax1.set_xlabel("Time")
ax1.set_ylabel("Speed (km/h)")
ax1.grid(True)
st.pyplot(fig1)

# Coolant Temperature

fig2, ax2 = plt.subplots(figsize=(10, 3))
ax2.plot(df_f["timestamp"], df_f["coolant_temp"], color="red")
ax2.set_xlabel("Time")
ax2.set_ylabel("Coolant Temp (°C)")
ax2.grid(True)
st.pyplot(fig2)

# Fuel Level

fig3, ax3 = plt.subplots(figsize=(10, 3))
ax3.plot(df_f["timestamp"], df_f["fuel_level"], color="green")
ax3.set_xlabel("Time")
ax3.set_ylabel("Fuel Level (%)")
ax3.grid(True)
st.pyplot(fig3)

# ----------------------------------

# Scatter Plot

# ----------------------------------

st.subheader("⚙️ RPM vs Speed")
fig4, ax4 = plt.subplots(figsize=(6, 4))
ax4.scatter(df_f["speed"], df_f["rpm"], s=12)
ax4.set_xlabel("Speed")
ax4.set_ylabel("RPM")
ax4.grid(True)
st.pyplot(fig4)

# ----------------------------------

# GPS Map

# ----------------------------------

st.subheader("🗺 Vehicle Movement Map")

if "latitude" in df_f.columns and "longitude" in df_f.columns:
st.map(df_f[["latitude", "longitude"]])
else:
st.info("⚠️ GPS data not found.")

# ----------------------------------

# Intelligent Analytics

# ----------------------------------

st.subheader("🧠 Smart Analytics")

speed_var = df_f["speed"].std()
acceleration = df_f["speed"].diff().abs().mean()

if speed_var < 5 and acceleration < 2:
driving_score = "Excellent"
elif speed_var < 10:
driving_score = "Good"
else:
driving_score = "Poor"

st.write(f"**Driving Behavior Score:** {driving_score}")

temp_spikes = df_f[df_f["coolant_temp"] > 95]
st.write(f"**Engine Temperature Spikes:** {len(temp_spikes)} detected")

# ----------------------------------

# Machine Learning – Predictive Stub (Extend Later)

# ----------------------------------

st.subheader("🤖 ML Prediction (Prototype)")

st.write("Model could predict:")
st.write("- Component failure risk")
st.write("- Fuel efficiency trend")
st.write("- Driver safety rating")

st.info("ML model integration placeholder – ready for future enhancement.")

# ----------------------------------

# Alerts

# ----------------------------------

st.subheader("🚨 Alerts")

alerts = []

if df_f["coolant_temp"].max() > 100:
alerts.append("🔥 **High engine temperature detected!**")

if df_f["speed"].max() > 120:
alerts.append("⚠️ **Overspeeding detected.**")

if df_f["rpm"].max() > 4500:
alerts.append("⚙️ **High RPM – aggressive driving.**")

if df_f["fuel_level"].iloc[-1] < 15:
alerts.append("⛽ **Low fuel warning.**")

if alerts:
for alert in alerts:
st.error(alert)
else:
st.success("No alerts – vehicle operating normally.")

# ----------------------------------

# Footer

# ----------------------------------

st.write("---")
st.write("Made with ❤️ using Streamlit.")
