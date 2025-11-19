import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Load Dataset
# -----------------------------

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("cleaned_vehicle_data.csv")
    except FileNotFoundError:
        st.error("❌ CSV not found. Upload cleaned_vehicle_data.csv to Streamlit Cloud.")
        return None
    except pd.errors.EmptyDataError:
        st.error("❌ CSV is empty or malformed!")
        return None
    except Exception as e:
        st.error(f"❌ Error loading CSV: {e}")
        return None

    if df is None or df.empty:
        st.error("❌ CSV file is empty!")
        return None

    st.write("Columns found in CSV:", list(df.columns))

    if "timestamp" not in df.columns:
        st.error("❌ 'timestamp' column not found in CSV!")
        st.write("Available columns:", list(df.columns))
        return None

    # Ensure timestamp is datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if df["timestamp"].isnull().all():
        st.error("❌ All timestamp values are missing or invalid!")
        return None
    return df

df = load_data()

# Stop execution if dataset failed to load
if df is None:
    st.stop()

st.write("Dataset loaded successfully!")
st.write("Shape:", df.shape)

# -----------------------------
# Dashboard Title
# -----------------------------
st.title("🚗 Automotive Data Intelligence Dashboard")
st.write("Analyze driving behavior, engine performance & vehicle health.")

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.header("Filters")

# Filter only valid timestamps
df = df[df["timestamp"].notnull()]

# Safety: Only run if sufficient data
if df.empty:
    st.error("❌ Dataset has no valid rows after filtering!")
    st.stop()

min_time = df["timestamp"].min()
max_time = df["timestamp"].max()

time_range = st.sidebar.slider(
    "Select Time Range",
    min_value=min_time,
    max_value=max_time,
    value=(min_time, max_time)
)

df_f = df[(df["timestamp"] >= time_range[0]) & (df["timestamp"] <= time_range[1])]

# Safety: Check for required columns in filtered set
required_columns = ["speed", "rpm", "fuel_level", "coolant_temp"]
missing_cols = [col for col in required_columns if col not in df_f.columns]
if missing_cols:
    st.error(f"❌ Missing columns for analysis: {missing_cols}")
    st.stop()

# -----------------------------
# KPI Cards
# -----------------------------
st.subheader("📊 Key Metrics")
col1, col2, col3 = st.columns(3)

# Guard: ensure enough data for metrics
if not df_f.empty:
    col1.metric("Average Speed (km/h)", f"{df_f['speed'].mean():.2f}")
    col2.metric("Max RPM", f"{df_f['rpm'].max():.0f}")
    # Calculate fuel used only if two points exist
    if len(df_f["fuel_level"]) > 1:
        fuel_used = df_f["fuel_level"].iloc[0] - df_f["fuel_level"].iloc[-1]
        col3.metric("Fuel Used (%)", f"{fuel_used:.2f}")
    else:
        col3.metric("Fuel Used (%)", "N/A")
else:
    col1.metric("Average Speed (km/h)", "N/A")
    col2.metric("Max RPM", "N/A")
    col3.metric("Fuel Used (%)", "N/A")

# -----------------------------
# Visualizations
# -----------------------------
# Protect against plotting with empty data
if not df_f.empty:

    st.subheader("📈 Speed Over Time")
    fig1, ax1 = plt.subplots(figsize=(10,4))
    ax1.plot(df_f["timestamp"], df_f["speed"])
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Speed (km/h)")
    ax1.grid(True)
    st.pyplot(fig1)

    st.subheader("🌡 Coolant Temperature Over Time")
    fig2, ax2 = plt.subplots(figsize=(10,4))
    ax2.plot(df_f["timestamp"], df_f["coolant_temp"], color="red")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Coolant Temp (°C)")
    ax2.grid(True)
    st.pyplot(fig2)

    st.subheader("⛽ Fuel Level Trend")
    fig3, ax3 = plt.subplots(figsize=(10,4))
    ax3.plot(df_f["timestamp"], df_f["fuel_level"], color="green")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Fuel Level (%)")
    ax3.grid(True)
    st.pyplot(fig3)

    st.subheader("⚙️ RPM vs Speed Scatter")
    fig4, ax4 = plt.subplots(figsize=(7,5))
    ax4.scatter(df_f["speed"], df_f["rpm"], s=10)
    ax4.set_xlabel("Speed")
    ax4.set_ylabel("RPM")
    ax4.grid(True)
    st.pyplot(fig4)
else:
    st.warning("Not enough data to display visualizations.")

# -----------------------------
# Phase 3 – Intelligent Analytics
# -----------------------------
st.subheader("🧠 Intelligent Analytics")

if not df_f.empty:
    speed_var = df_f["speed"].std()
    acceleration = df_f["speed"].diff().abs().mean()

    if pd.isna(speed_var) or pd.isna(acceleration):
        driving_score = "N/A"
    elif speed_var < 5 and acceleration < 2:
        driving_score = "Excellent"
    elif speed_var < 10:
        driving_score = "Good"
    else:
        driving_score = "Poor"

    st.write(f"**Driving Behavior Score:** {driving_score}")

    # Engine temp spikes
    spikes = df_f[df_f["coolant_temp"] > 95]
    st.write(f"**Engine Temperature Spikes:** {len(spikes)} detected")
else:
    st.write("No analytics available for empty filter/data.")

# -----------------------------
# Phase 4 – Alerts
# -----------------------------
st.subheader("🚨 Alerts")

alerts = []
if not df_f.empty:
    if df_f["coolant_temp"].max() > 100:
        alerts.append("🔥 High Coolant Temperature!")
    if df_f["speed"].max() > 120:
        alerts.append("⚠️ Overspeeding Detected")
    if df_f["rpm"].max() > 4500:
        alerts.append("⚙️ High RPM – Aggressive Driving")
    if df_f["fuel_level"].iloc[-1] < 15:
        alerts.append("⛽ Low Fuel Warning")

if not alerts:
    st.success("No alerts! Everything looks normal.")
else:
    for alert in alerts:
        st.error(alert)

st.write("---")
st.write("Made with ❤️ using Streamlit.")

