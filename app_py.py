import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Load Dataset (with upload support)
# -----------------------------

@st.cache_data
def load_default_data():
    df = pd.read_csv("cleaned_vehicle_data.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

st.sidebar.header("📁 Upload Your Data")

uploaded_file = st.sidebar.file_uploader("Upload CSV (Optional)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if "timestamp" not in df.columns:
        st.error("❌ Uploaded file must contain a 'timestamp' column.")
        st.stop()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    st.success("✔ Using uploaded file!")
else:
    df = load_default_data()
    st.info("📌 Using default dataset: cleaned_vehicle_data.csv")

# -----------------------------
# Dashboard Title
# -----------------------------

st.title("🚗 Automotive Data Intelligence Dashboard")
st.write("Analyze driving behavior, engine performance & vehicle health.")

# -----------------------------
# Sidebar Filters
# -----------------------------

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

# -----------------------------
# KPI Cards
# -----------------------------

st.subheader("📊 Key Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Average Speed (km/h)", f"{df_f['speed'].mean():.2f}")
col2.metric("Max RPM", f"{df_f['rpm'].max():.0f}")
col3.metric("Fuel Used (%)", 
            f"{df_f['fuel_level'].iloc[0] - df_f['fuel_level'].iloc[-1]:.2f}")

# -----------------------------
# Visualizations
# -----------------------------

st.subheader("📈 Speed Over Time")
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(df_f["timestamp"], df_f["speed"])
ax1.set_xlabel("Time")
ax1.set_ylabel("Speed (km/h)")
ax1.grid(True)
st.pyplot(fig1)

st.subheader("🌡 Coolant Temperature Over Time")
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(df_f["timestamp"], df_f["coolant_temp"], color="red")
ax2.set_xlabel("Time")
ax2.set_ylabel("Coolant Temp (°C)")
ax2.grid(True)
st.pyplot(fig2)

st.subheader("⛽ Fuel Level Trend")
fig3, ax3 = plt.subplots(figsize=(10, 4))
ax3.plot(df_f["timestamp"], df_f["fuel_level"], color="green")
ax3.set_xlabel("Time")
ax3.set_ylabel("Fuel Level (%)")
ax3.grid(True)
st.pyplot(fig3)

st.subheader("⚙️ RPM vs Speed Scatter")
fig4, ax4 = plt.subplots(figsize=(7, 5))
ax4.scatter(df_f["speed"], df_f["rpm"], s=10)
ax4.set_xlabel("Speed")
ax4.set_ylabel("RPM")
ax4.grid(True)
st.pyplot(fig4)

# -----------------------------
# Map Visualization 
# -----------------------------
st.subheader("🗺️ GPS Position Map")

if "latitude" in df_f.columns and "longitude" in df_f.columns:
    st.map(df_f[["latitude", "longitude"]])
else:
    st.info("⚠️ GPS coordinates not available in this dataset.")

# -----------------------------
# Phase 3 – Intelligent Analytics
# -----------------------------

st.subheader("🧠 Intelligent Analytics")

speed_var = df_f["speed"].std()
acceleration = df_f["speed"].diff().abs().mean()

if speed_var < 5 and acceleration < 2:
    driving_score = "Excellent"
elif speed_var < 10:
    driving_score = "Good"
else:
    driving_score = "Poor"

st.write(f"**Driving Behavior Score:** {driving_score}")

spikes = df_f[df_f["coolant_temp"] > 95]
st.write(f"**Engine Temperature Spikes:** {len(spikes)} detected")

# -----------------------------
# Phase 4 – Alerts
# -----------------------------

st.subheader("🚨 Alerts")

alerts = []

if df_f["coolant_temp"].max() > 100:
    alerts.append("🔥 High Coolant Temperature!")

if df_f["speed"].max() > 120:
    alerts.append("⚠️ Overspeeding Detected")

if df_f["rpm"].max() > 4500:
    alerts.append("⚙️ High RPM – Aggressive Driving")

if df_f["fuel_level"].iloc[-1] < 15:
    alerts.append("⛽ Low Fuel Warning")

if len(alerts) == 0:
    st.success("No alerts! Everything looks normal.")
else:
    for alert in alerts:
        st.error(alert)

st.write("---")
st.write("Made with ❤️ using Streamlit.")
