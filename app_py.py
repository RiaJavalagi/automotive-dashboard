import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

# ---------- Page config ----------
st.set_page_config(
    page_title="Automotive Data Intelligence",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Helpers ----------
@st.cache_data
def load_data():
    df = pd.read_csv("/mnt/data/cleaned_vehicle_data.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

df = load_data()

    # Normalize column names: strip whitespace
    df.columns = [c.strip() for c in df.columns]

    # If timestamp-like column exists under any name, rename to 'timestamp'
    timestamp_candidates = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()]
    if len(timestamp_candidates) > 0:
        # Prefer exact 'timestamp' if present
        preferred = None
        for c in timestamp_candidates:
            if c.lower().strip() == "timestamp":
                preferred = c
                break
        if preferred is None:
            preferred = timestamp_candidates[0]
        df = df.rename(columns={preferred: "timestamp"})

    # Force timestamp conversion if present
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df, "OK"

def safe_to_numeric(df, col):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "."), errors="coerce")
    return df

@st.cache_data
def train_fuel_model(df):
    """Train a simple linear regression model to predict fuel_level from speed & rpm (if available)."""
    if not {"speed", "rpm", "fuel_level"}.issubset(df.columns):
        return None, None

    df_ml = df[["speed", "rpm", "fuel_level"]].dropna()
    if len(df_ml) < 20:
        return None, None

    X = df_ml[["speed", "rpm"]]
    y = df_ml["fuel_level"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    return model, score

@st.cache_data
def train_anomaly_detector(df, col="coolant_temp"):
    """Train IsolationForest on numeric features to detect anomalies in coolant temp or others."""
    if col not in df.columns:
        return None, None

    # Use only numeric columns + the target col
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        return None, None

    X = df[numeric_cols].fillna(0)
    # If not enough rows, skip
    if len(X) < 30:
        return None, None

    iso = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    iso.fit(X)
    preds = iso.predict(X)  # -1 = anomaly, 1 = normal
    return iso, preds

def driving_score_calc(df):
    """Compute a simple driving score (0-100) using std and events."""
    score = 100
    if "speed" in df.columns:
        score -= df["speed"].std() * 1.2
    if "rpm" in df.columns:
        score -= df["rpm"].std() * 0.7
    # harsh events penalty (acceleration spikes)
    if "speed" in df.columns:
        acc = df["speed"].diff().fillna(0)
        harsh_brakes = (acc < -6).sum()
        sudden_acc = (acc > 6).sum()
        score -= harsh_brakes * 2
        score -= sudden_acc * 1
    score = int(max(0, min(100, score)))
    return score

# ---------- Sidebar: Data source ----------
st.sidebar.header("Data Source")

uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
use_default = False
if uploaded is None:
    st.sidebar.write("Using default uploaded file (if present):")
    st.sidebar.code(DEFAULT_CSV_PATH)
    use_default = st.sidebar.button("Load default CSV")

# ---------- Load data ----------
if uploaded is not None:
    df, status = load_dataset(uploaded)
else:
    if use_default:
        df, status = load_dataset(DEFAULT_CSV_PATH)
    else:
        # initial message, app won't crash
        df, status = None, "No file selected"

if df is None:
    st.sidebar.error(status)
    st.write("# No dataset loaded")
    st.info("Upload a CSV or click 'Load default CSV' in the sidebar.")
    st.stop()

# ---------- Data cleaning: ensure numeric types for common columns ----------
for c in ["speed", "rpm", "coolant_temp", "fuel_level", "latitude", "longitude", "throttle"]:
    df = safe_to_numeric(df, c)

# ---------- Auto-fill small missing timestamp values if all NaT -->
if "timestamp" in df.columns:
    # If timestamp exists but many NaT, create synthetic monotonic timestamps
    if df["timestamp"].isna().mean() > 0.6:
        df = df.reset_index(drop=True)
        df["timestamp"] = pd.date_range(start="2023-01-01", periods=len(df), freq="S")
else:
    # create synthetic if not present (keeps app working)
    df = df.reset_index(drop=True)
    df["timestamp"] = pd.date_range(start="2023-01-01", periods=len(df), freq="S")

# Sort by time
df = df.sort_values("timestamp").reset_index(drop=True)

# ---------- Sidebar: Filters ----------
st.sidebar.header("Filters & Model Controls")
min_time = df["timestamp"].min().to_pydatetime()
max_time = df["timestamp"].max().to_pydatetime()

time_range = st.sidebar.slider(
    "Select Time Range",
    min_value=min_time,
    max_value=max_time,
    value=(min_time, max_time),
    format="YYYY-MM-DD HH:mm:ss"
)

df_f = df[(df["timestamp"] >= pd.to_datetime(time_range[0])) & (df["timestamp"] <= pd.to_datetime(time_range[1]))].copy()

# Sampling for faster UI (optional)
sample_n = st.sidebar.slider("Sampling for plots (0=no sample, >0 = n rows)", 0, 20000, 5000)
if sample_n > 0 and len(df_f) > sample_n:
    df_vis = df_f.sample(sample_n, random_state=42)
else:
    df_vis = df_f.copy()

# ---------- Top KPIs ----------
st.markdown("## 🚗 Overview")
k1, k2, k3, k4 = st.columns(4)
avg_speed = df_f["speed"].mean() if "speed" in df_f.columns else np.nan
k1.metric("Average Speed (km/h)", f"{avg_speed:.2f}" if not np.isnan(avg_speed) else "N/A")
avg_rpm = df_f["rpm"].mean() if "rpm" in df_f.columns else np.nan
k2.metric("Average RPM", f"{avg_rpm:.0f}" if not np.isnan(avg_rpm) else "N/A")
latest_temp = df_f["coolant_temp"].iloc[-1] if "coolant_temp" in df_f.columns else np.nan
k3.metric("Latest Coolant Temp (°C)", f"{latest_temp:.1f}" if not np.isnan(latest_temp) else "N/A")
drive_score = driving_score_calc(df_f)
k4.metric("Driving Score (0-100)", f"{drive_score}")

# ---------- Layout: Tabs ----------
tab_kpis, tab_charts, tab_analytics, tab_alerts, tab_map = st.tabs(
    ["📊 KPIs", "📈 Charts", "🧠 Analytics (ML)", "🚨 Alerts", "🗺️ Map"]
)

# ---------- Tab: Charts ----------
with tab_charts:
    st.header("Visualizations")
    col1, col2 = st.columns(2)
    with col1:
        if "speed" in df_vis.columns:
            fig = px.line(df_vis, x="timestamp", y="speed", title="Speed vs Time", labels={"speed":"Speed (km/h)"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No 'speed' column to plot.")

        if "coolant_temp" in df_vis.columns:
            fig = px.line(df_vis, x="timestamp", y="coolant_temp", title="Coolant Temp vs Time", labels={"coolant_temp":"Coolant Temp (°C)"})
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "fuel_level" in df_vis.columns:
            fig = px.line(df_vis, x="timestamp", y="fuel_level", title="Fuel Level Over Time", labels={"fuel_level":"Fuel Level (%)"})
            st.plotly_chart(fig, use_container_width=True)
        if {"speed", "rpm"}.issubset(df_vis.columns):
            fig = px.scatter(df_vis.sample(min(len(df_vis),20000)), x="speed", y="rpm", title="RPM vs Speed (sampled)")
            st.plotly_chart(fig, use_container_width=True)

# ---------- Tab: Analytics (ML) ----------
with tab_analytics:
    st.header("Machine Learning Analytics")

    # Fuel prediction model
    model, score = train_fuel_model(df_f)
    if model is not None:
        st.subheader("🔮 Fuel Level Prediction (Linear Regression)")
        st.write(f"Model R² score (test): **{score:.3f}**")
        spd = st.number_input("Speed (km/h)", min_value=0.0, max_value=300.0, value=float(avg_speed if not np.isnan(avg_speed) else 40.0))
        rpm_val = st.number_input("RPM", min_value=0.0, max_value=20000.0, value=float(avg_rpm if not np.isnan(avg_rpm) else 1500.0))
        pred = model.predict(np.array([[spd, rpm_val]]))[0]
        st.metric("Predicted Fuel Level (%)", f"{pred:.2f}")
    else:
        st.info("Fuel prediction requires 'speed', 'rpm', and 'fuel_level' columns with sufficient data.")

    # Anomaly detection for coolant_temp
    st.subheader("🚨 Anomaly Detection")
    anomaly_model, preds = train_anomaly_detector(df_f, col="coolant_temp")
    if anomaly_model is not None and preds is not None:
        df_f["anomaly_flag"] = (preds == -1)
        anomalies = df_f[df_f["anomaly_flag"]]
        st.write(f"Detected anomalies: **{len(anomalies)}**")
        if not anomalies.empty:
            st.dataframe(anomalies[["timestamp"] + [c for c in ["coolant_temp","speed","rpm","fuel_level"] if c in anomalies.columns]].head(50))
            fig_anom = px.scatter(df_f, x="timestamp", y="coolant_temp", color="anomaly_flag", title="Coolant Temp with Anomalies")
            st.plotly_chart(fig_anom, use_container_width=True)
    else:
        # fallback: simple Z-score anomalies if coolant_temp exists
        if "coolant_temp" in df_f.columns:
            z = (df_f["coolant_temp"] - df_f["coolant_temp"].mean()) / df_f["coolant_temp"].std()
            anomalies = df_f[z.abs() > 3]
            st.write(f"Detected z-score anomalies: **{len(anomalies)}** (fallback)")
            if not anomalies.empty:
                st.dataframe(anomalies[["timestamp","coolant_temp"]].head(50))
        else:
            st.info("No 'coolant_temp' column available for anomaly detection.")

    # Driving classification
    st.subheader("🧭 Driving Classification")
    if {"speed","rpm"}.issubset(df_f.columns):
        msp = df_f["speed"].mean()
        mrp = df_f["rpm"].mean()
        if msp < 40 and mrp < 2000:
            style = "Eco Driving"
        elif msp < 80:
            style = "Normal Driving"
        else:
            style = "Aggressive Driving"
        st.write(f"Detected driving style: **{style}**")
    else:
        st.info("Need both 'speed' and 'rpm' columns to classify driving style.")

# ---------- Tab: Alerts ----------
with tab_alerts:
    st.header("Alerts & Events")

    # Engine overheat threshold
    if "coolant_temp" in df_f.columns:
        thr = st.number_input("Overheat threshold (°C)", value=float(df_f["coolant_temp"].mean() + 2*df_f["coolant_temp"].std()))
        df_f["engine_overheat"] = df_f["coolant_temp"] > thr
        overheat_events = df_f[df_f["engine_overheat"]]
        st.metric("Overheat Events", int(overheat_events.shape[0]))
        if not overheat_events.empty:
            st.dataframe(overheat_events[["timestamp","coolant_temp"]].head(50))
    else:
        st.info("No 'coolant_temp' column to monitor for overheat.")

    # Speeding events
    if "speed" in df_f.columns:
        speed_limit = st.number_input("Speed limit (km/h)", min_value=20, max_value=250, value=120)
        speeding = df_f[df_f["speed"] > speed_limit]
        st.metric("Speeding events", int(speeding.shape[0]))
        if not speeding.empty:
            st.dataframe(speeding[["timestamp","speed"]].head(50))
    else:
        st.info("No 'speed' column to monitor for speeding.")

    # Harsh braking & accel
    if "speed" in df_f.columns:
        acc = df_f["speed"].diff().fillna(0)
        harsh_brake = df_f[acc < -6]
        sudden_acc = df_f[acc > 6]
        st.metric("Harsh Brakes", int(harsh_brake.shape[0]))
        st.metric("Sudden Accelerations", int(sudden_acc.shape[0]))
        if not harsh_brake.empty:
            st.dataframe(harsh_brake[["timestamp","speed"]].head(30))
    else:
        st.info("No 'speed' column to compute acceleration events.")

# ---------- Tab: Map ----------
with tab_map:
    st.header("Map / GPS (if available)")
    lat_col = None
    lon_col = None
    # attempt to detect latitude/longitude columns
    for candidate in ["latitude","lat","Latitude","LAT","lon","longitude","lng","long"]:
        if candidate in df_f.columns:
            if candidate.lower().startswith("lat"):
                lat_col = candidate
            if candidate.lower().startswith("lon") or candidate.lower().startswith("lng") or candidate.lower().startswith("long"):
                lon_col = candidate

    if lat_col and lon_col:
        # ensure numeric
        df_f[lat_col] = pd.to_numeric(df_f[lat_col], errors="coerce")
        df_f[lon_col] = pd.to_numeric(df_f[lon_col], errors="coerce")
        st.map(df_f[[lat_col, lon_col]].dropna())
    else:
        st.info("GPS coordinates not detected. If your columns are named differently, rename them to 'latitude' and 'longitude' or upload a file with GPS columns.")

# ---------- Footer: Export ----------
st.markdown("---")
st.header("Export / Download")

@st.cache_data
def to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")

csv_bytes = to_csv_bytes(df_f)
st.download_button("Download filtered CSV", data=csv_bytes, file_name="filtered_vehicle_data.csv", mime="text/csv")

st.caption("Made with ❤️ — Automotive Data Intelligence Dashboard")
