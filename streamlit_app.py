"""
=============================================================================
 TEXTILE MACHINE FAILURE PREDICTION — Streamlit Dashboard
=============================================================================
 Pages:
   1. Dashboard / Overview   – live KPIs & sensor gauges from data.csv
   2. Prediction Engine      – user inputs → 4 model predictions
   3. Model Analytics        – training results, comparison charts
   4. Data Explorer          – browse, filter, correlate the dataset
=============================================================================
"""

import os, joblib, datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results")

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Textile Machine Failure Prediction",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS — SCADA / Industrial Dark Theme
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── main theme ─────────────────────────────────────────────────────────── */
.stApp {
    background: linear-gradient(135deg, #0a0a0f 0%, #0d1117 50%, #0a0a14 100%);
    font-family: 'Inter', sans-serif;
    zoom: 1.15;
}
html, body, [data-testid="stAppViewContainer"] {
    font-size: 16px !important;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    border-right: 1px solid rgba(0, 212, 255, 0.15);
}
[data-testid="stSidebar"] .stRadio > label {
    color: #e6edf3 !important;
}

/* ── headings ───────────────────────────────────────────────────────────── */
h1, h2, h3 { color: #e6edf3 !important; font-family: 'Inter', sans-serif !important; }
h1 { font-weight: 800 !important; letter-spacing: -0.5px; font-size: 2.2rem !important; }
h2 { font-weight: 700 !important; font-size: 1.6rem !important; }
h3 { font-weight: 600 !important; font-size: 1.3rem !important; }

/* ── metric cards ───────────────────────────────────────────────────────── */
.metric-card {
    background: linear-gradient(145deg, rgba(22,27,34,0.95), rgba(13,17,23,0.9));
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    backdrop-filter: blur(12px);
    transition: all 0.3s ease;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}
.metric-card:hover {
    border-color: rgba(0,212,255,0.5);
    box-shadow: 0 8px 32px rgba(0,212,255,0.12);
    transform: translateY(-2px);
}
.metric-value {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00d4ff, #00ff88);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.2;
}
.metric-label {
    font-size: 1rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 10px;
    font-weight: 600;
}

/* ── prediction result cards ────────────────────────────────────────────── */
.pred-card {
    border-radius: 16px;
    padding: 32px;
    text-align: center;
    backdrop-filter: blur(12px);
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    transition: all 0.3s ease;
    min-height: 160px;
}
.pred-card:hover { transform: translateY(-3px); }
.pred-healthy {
    background: linear-gradient(145deg, rgba(0,255,136,0.08), rgba(0,200,100,0.03));
    border: 1px solid rgba(0,255,136,0.3);
}
.pred-warning {
    background: linear-gradient(145deg, rgba(255,200,0,0.08), rgba(255,165,0,0.03));
    border: 1px solid rgba(255,200,0,0.3);
}
.pred-danger {
    background: linear-gradient(145deg, rgba(255,60,60,0.08), rgba(200,0,0,0.03));
    border: 1px solid rgba(255,60,60,0.3);
}
.pred-info {
    background: linear-gradient(145deg, rgba(0,212,255,0.08), rgba(0,150,200,0.03));
    border: 1px solid rgba(0,212,255,0.3);
}
.pred-title {
    font-size: 1rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 600;
    margin-bottom: 14px;
}
.pred-value {
    font-size: 2.2rem;
    font-weight: 800;
    line-height: 1.2;
}
.pred-sub {
    font-size: 0.9rem;
    color: #8b949e;
    margin-top: 10px;
}

/* ── scenario buttons ───────────────────────────────────────────────────── */
.scenario-btn-row {
    display: flex;
    gap: 16px;
    margin-bottom: 24px;
}
.scenario-card {
    flex: 1;
    border-radius: 14px;
    padding: 20px 24px;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s ease;
}
.scenario-normal {
    background: linear-gradient(145deg, rgba(0,255,136,0.1), rgba(0,200,100,0.04));
    border: 2px solid rgba(0,255,136,0.35);
}
.scenario-failure {
    background: linear-gradient(145deg, rgba(255,60,60,0.1), rgba(200,0,0,0.04));
    border: 2px solid rgba(255,60,60,0.35);
}
.scenario-card h4 {
    margin: 0 0 6px 0 !important;
    font-size: 1.15rem !important;
}
.scenario-card p {
    margin: 0;
    font-size: 0.85rem;
    color: #8b949e;
}

/* ── section dividers ───────────────────────────────────────────────────── */
.section-header {
    font-size: 1.25rem;
    font-weight: 700;
    color: #00d4ff;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin: 36px 0 18px 0;
    padding-bottom: 10px;
    border-bottom: 2px solid rgba(0,212,255,0.2);
}

/* ── page banner ────────────────────────────────────────────────────────── */
.page-banner {
    background: linear-gradient(135deg, rgba(0,212,255,0.06), rgba(0,255,136,0.04));
    border: 1px solid rgba(0,212,255,0.15);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 32px;
}
.page-banner h2 {
    margin: 0 0 6px 0;
    font-size: 1.8rem !important;
}
.page-banner p {
    color: #8b949e;
    margin: 0;
    font-size: 1.05rem;
}

/* ── sidebar styling ────────────────────────────────────────────────────── */
[data-testid="stSidebar"] h1 {
    font-size: 1.35rem !important;
    text-align: center;
    padding: 10px 0 18px;
    border-bottom: 1px solid rgba(0,212,255,0.15);
    background: linear-gradient(135deg, #00d4ff, #00ff88);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
[data-testid="stSidebar"] .stRadio > div > label {
    font-size: 1.05rem !important;
    padding: 6px 0 !important;
}

/* ── tables ──────────────────────────────────────────────────────────────── */
.stDataFrame { border-radius: 12px; overflow: hidden; }

/* ── plotly charts bg ───────────────────────────────────────────────────── */
.js-plotly-plot .plotly .main-svg { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# LOAD DATA & MODELS (cached)
# ═════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, parse_dates=["Timestamp"])

@st.cache_resource
def load_models():
    models = {}
    for target in ["Failure_Mode_Code", "Failure_Imminent_Flag",
                    "Target_Failure_24H", "Target_RUL_Hours"]:
        path = os.path.join(MODEL_DIR, f"{target}_best_model.pkl")
        if os.path.exists(path):
            models[target] = joblib.load(path)
    return models

@st.cache_resource
def load_scaler():
    return joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

@st.cache_resource
def load_label_encoders():
    return joblib.load(os.path.join(MODEL_DIR, "label_encoders.pkl"))

@st.cache_data
def load_results():
    results = {}
    for fname in os.listdir(RESULT_DIR):
        if fname.endswith("_results.csv"):
            key = fname.replace("_results.csv", "")
            results[key] = pd.read_csv(os.path.join(RESULT_DIR, fname))
    summary_path = os.path.join(RESULT_DIR, "final_summary.csv")
    if os.path.exists(summary_path):
        results["final_summary"] = pd.read_csv(summary_path)
    return results


# ═════════════════════════════════════════════════════════════════════════════
# PLOTLY THEME HELPER
# ═════════════════════════════════════════════════════════════════════════════
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(13,17,23,0.8)",
    font=dict(family="Inter", color="#e6edf3"),
    margin=dict(l=40, r=20, t=50, b=40),
    xaxis=dict(gridcolor="rgba(139,148,158,0.1)", zerolinecolor="rgba(139,148,158,0.1)"),
    yaxis=dict(gridcolor="rgba(139,148,158,0.1)", zerolinecolor="rgba(139,148,158,0.1)"),
)
COLORS = ["#00d4ff", "#00ff88", "#ff6b6b", "#ffc107", "#a855f7", "#f472b6"]


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("#  MACHINE PREDICTOR")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["Dashboard", "Prediction Engine", "Model Analytics", "Data Explorer"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; color:#8b949e; font-size:0.7rem;'>"
        "Textile Machine Failure Prediction<br>ML Pipeline v2.0</p>",
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD / OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
def page_dashboard():
    df = load_data()

    st.markdown(
        '<div class="page-banner">'
        '<h2> Plant Operations Dashboard</h2>'
        '<p>Real-time overview of textile machine fleet health, energy & production KPIs</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── KPI row ──────────────────────────────────────────────────────────────
    total_machines = df["Machine_ID"].nunique()
    avg_rul = df["RUL_Hours"].mean()
    avg_eff = df["Efficiency_Index"].mean()
    critical_pct = (df["Degradation_Phase"] == "Critical").mean() * 100
    avg_energy = df["Energy_kWh"].mean()
    total_output = df["Output_kg"].sum()

    cols = st.columns(6)
    kpis = [
        (f"{total_machines}", "Machines"),
        (f"{avg_rul:.0f} h", "Avg RUL"),
        (f"{avg_eff:.2f}", "Avg Efficiency"),
        (f"{critical_pct:.1f}%", "Critical Rate"),
        (f"{avg_energy:.1f} kWh", "Avg Energy"),
        (f"{total_output/1000:.0f} T", "Total Output"),
    ]
    for col, (val, lbl) in zip(cols, kpis):
        col.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{val}</div>'
            f'<div class="metric-label">{lbl}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("")  # spacer

    # ── charts row 1 ─────────────────────────────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-header">Degradation Phase Distribution</div>', unsafe_allow_html=True)
        phase_counts = df["Degradation_Phase"].value_counts().reset_index()
        phase_counts.columns = ["Phase", "Count"]
        color_map = {"Healthy": "#00ff88", "Early Wear": "#00d4ff",
                     "Moderate Wear": "#ffc107", "Critical": "#ff6b6b"}
        fig = px.pie(phase_counts, names="Phase", values="Count",
                     color="Phase", color_discrete_map=color_map, hole=0.55)
        fig.update_layout(**PLOTLY_LAYOUT, showlegend=True, height=380,
                          legend=dict(font=dict(size=11)))
        fig.update_traces(textfont_size=12, textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">RUL Distribution (hours)</div>', unsafe_allow_html=True)
        fig = px.histogram(df, x="RUL_Hours", nbins=50,
                           color_discrete_sequence=["#00d4ff"])
        fig.update_layout(**PLOTLY_LAYOUT, height=380,
                          xaxis_title="RUL (hours)", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    # ── charts row 2 ─────────────────────────────────────────────────────────
    c3, c4 = st.columns(2)

    with c3:
        st.markdown('<div class="section-header">Energy vs Output by Machine Type</div>', unsafe_allow_html=True)
        fig = px.scatter(df, x="Energy_kWh", y="Output_kg",
                         color="Machine_Type", opacity=0.6,
                         color_discrete_sequence=COLORS)
        fig.update_layout(**PLOTLY_LAYOUT, height=380,
                          xaxis_title="Energy (kWh)", yaxis_title="Output (kg)")
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        st.markdown('<div class="section-header">Temperature vs Friction Index</div>', unsafe_allow_html=True)
        fig = px.scatter(df, x="Temp_C", y="Mechanical_Friction_Index",
                         color="Degradation_Phase", opacity=0.6,
                         color_discrete_map=color_map)
        fig.update_layout(**PLOTLY_LAYOUT, height=380,
                          xaxis_title="Temperature (°C)",
                          yaxis_title="Mechanical Friction Index")
        st.plotly_chart(fig, use_container_width=True)

    # ── sensor gauges (latest reading) ───────────────────────────────────────
    st.markdown('<div class="section-header">Latest Sensor Readings (most recent row)</div>', unsafe_allow_html=True)
    latest = df.sort_values("Timestamp").iloc[-1]

    g_cols = st.columns(5)
    gauges = [
        ("Temperature", latest["Temp_C"], "°C", 30, 120),
        ("Humidity", latest["Humidity_%"], "%", 20, 100),
        ("Motor Current", latest["Motor_Current_A"], "A", 0, 50),
        ("Machine Speed", latest["Machine_Speed_RPM"], "RPM", 500, 2500),
        ("Component Health", latest["Component_Health_%"], "%", 0, 100),
    ]
    for col, (name, val, unit, lo, hi) in zip(g_cols, gauges):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=val,
            title=dict(text=name, font=dict(size=13, color="#8b949e")),
            number=dict(suffix=f" {unit}", font=dict(size=18, color="#e6edf3")),
            gauge=dict(
                axis=dict(range=[lo, hi], tickcolor="#8b949e"),
                bar=dict(color="#00d4ff"),
                bgcolor="rgba(13,17,23,0.9)",
                borderwidth=1,
                bordercolor="rgba(0,212,255,0.3)",
                steps=[
                    dict(range=[lo, lo + (hi - lo) * 0.33], color="rgba(0,255,136,0.15)"),
                    dict(range=[lo + (hi - lo) * 0.33, lo + (hi - lo) * 0.66], color="rgba(255,193,7,0.15)"),
                    dict(range=[lo + (hi - lo) * 0.66, hi], color="rgba(255,60,60,0.15)"),
                ],
            ),
        ))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e6edf3"),
                          height=200, margin=dict(l=20, r=20, t=40, b=10))
        col.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PREDICTION ENGINE
# ═════════════════════════════════════════════════════════════════════════════
def page_prediction():
    df = load_data()
    models = load_models()
    scaler = load_scaler()
    label_encoders = load_label_encoders()

    st.markdown(
        '<div class="page-banner">'
        '<h2>Failure Prediction Engine</h2>'
        '<p>Enter machine parameters below — or use a quick scenario preset — to predict '
        'failure mode, imminent failure, 24-hour risk, and remaining useful life</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    if not models:
        st.error("No trained models found in the `models/` directory. "
                 "Please run `ml_pipeline.py` first.")
        return

    # ── scenario presets ─────────────────────────────────────────────────────
    # Define default values for Normal vs Failure
    NORMAL_PRESET = dict(
        p_speed=1500, p_temp=65.0, p_humidity=55.0, p_energy=35.0,
        p_power_factor=0.92, p_motor_current=18.0, p_voltage_var=2.0,
        p_output_kg=550, p_yarn_breaks=2, p_waste_kg=8,
        p_safety=0, p_auto_shut=0, p_emerg_stop=0,
        p_current_ratio=1.0, p_friction=0.3, p_wear=5.0,
        p_buffer_level=120.0, p_yarn_count=30.0, p_speed_override=0,
        p_torque=0.65, p_efficiency=0.9,
    )
    FAILURE_PRESET = dict(
        p_speed=2200, p_temp=105.0, p_humidity=85.0, p_energy=78.0,
        p_power_factor=0.6, p_motor_current=42.0, p_voltage_var=12.0,
        p_output_kg=180, p_yarn_breaks=35, p_waste_kg=70,
        p_safety=1, p_auto_shut=1, p_emerg_stop=4,
        p_current_ratio=1.8, p_friction=1.3, p_wear=42.0,
        p_buffer_level=15.0, p_yarn_count=50.0, p_speed_override=1,
        p_torque=1.8, p_efficiency=0.25,
    )

    # Initialize session state defaults if not set
    if "p_speed" not in st.session_state:
        for k, v in NORMAL_PRESET.items():
            st.session_state[k] = v

    st.markdown('<div class="section-header">Quick Scenario Presets</div>', unsafe_allow_html=True)
    sc1, sc2, sc3 = st.columns([1, 1, 1])
    with sc1:
        if st.button("Normal / Healthy Machine", use_container_width=True, type="secondary"):
            for k, v in NORMAL_PRESET.items():
                st.session_state[k] = v
            st.rerun()
    with sc2:
        if st.button("Failure / Critical Machine", use_container_width=True, type="primary"):
            for k, v in FAILURE_PRESET.items():
                st.session_state[k] = v
            st.rerun()
    with sc3:
        st.markdown(
            '<div style="padding:10px; color:#8b949e; font-size:0.9rem;">'
            'Click a preset to auto-fill all parameters, or manually adjust below.</div>',
            unsafe_allow_html=True,
        )

    # ── input parameters ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Machine Parameters</div>', unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("**Machine Identity**")
        machine_ids = sorted(df["Machine_ID"].unique().tolist())
        machine_id = st.selectbox("Machine ID", machine_ids, index=0)
        machine_types = sorted(df["Machine_Type"].unique().tolist())
        machine_type = st.selectbox("Machine Type", machine_types, index=0)
        sections = sorted(df["Section"].unique().tolist())
        section = st.selectbox("Section", sections, index=0)
        states = sorted(df["Machine_State"].unique().tolist())
        machine_state = st.selectbox("Machine State", states, index=0)
        shifts = sorted(df["Shift"].unique().tolist())
        shift = st.selectbox("Shift", shifts, index=0)

    with col_b:
        st.markdown("**Sensor Readings**")
        speed = st.slider("Machine Speed (RPM)", 500, 2500, key="p_speed")
        temp = st.slider("Temperature (°C)", 30.0, 120.0, step=0.5, key="p_temp")
        humidity = st.slider("Humidity (%)", 20.0, 100.0, step=0.5, key="p_humidity")
        energy = st.slider("Energy (kWh)", 0.0, 100.0, step=0.5, key="p_energy")
        power_factor = st.slider("Power Factor", 0.5, 1.0, step=0.01, key="p_power_factor")
        motor_current = st.slider("Motor Current (A)", 0.0, 50.0, step=0.5, key="p_motor_current")
        voltage_var = st.slider("Voltage Variation (%)", 0.0, 15.0, step=0.1, key="p_voltage_var")

    with col_c:
        st.markdown("**Production & Maintenance**")
        output_kg = st.slider("Output (kg)", 100, 1000, step=10, key="p_output_kg")
        yarn_breaks = st.slider("Yarn Breaks Count", 0, 50, key="p_yarn_breaks")
        waste_kg = st.slider("Waste (kg)", 0, 100, key="p_waste_kg")
        safety = st.selectbox("Safety Interlock Triggered", [0, 1], key="p_safety")
        auto_shut = st.selectbox("Auto Shutdown Flag", [0, 1], key="p_auto_shut")
        emerg_stop = st.slider("Emergency Stop Count", 0, 10, key="p_emerg_stop")
        current_ratio = st.slider("Current Ratio", 0.5, 2.0, step=0.01, key="p_current_ratio")
        friction = st.slider("Mechanical Friction Index", 0.0, 1.5, step=0.01, key="p_friction")
        wear = st.slider("Wear Score", 0.0, 50.0, step=0.5, key="p_wear")
        buffer_level = st.slider("Material Buffer Level (kg)", 0.0, 200.0, step=1.0, key="p_buffer_level")
        yarn_count = st.slider("Yarn Count (Ne)", 5.0, 60.0, step=0.5, key="p_yarn_count")
        speed_override = st.selectbox("Speed Override Flag", [0, 1], key="p_speed_override")
        torque = st.slider("Torque Load Index", 0.0, 2.0, step=0.01, key="p_torque")
        efficiency = st.slider("Efficiency Index", 0.0, 1.0, step=0.01, key="p_efficiency")

    st.markdown("---")

    # ── predict button ───────────────────────────────────────────────────────
    if st.button("Run Prediction", use_container_width=True, type="primary"):
        # ── replicate EXACT feature engineering from ml_pipeline.py ───────
        now = datetime.datetime.now()

        # raw numeric values (same order as pipeline feature_cols)
        raw = {
            "Machine_Speed_RPM": speed,
            "Temp_C": temp,
            "Humidity_%": humidity,
            "Energy_kWh": energy,
            "Power_Factor": power_factor,
            "Motor_Current_A": motor_current,
            "Voltage_Variation_%": voltage_var,
            "Output_kg": output_kg,
            "Yarn_Breaks_Count": yarn_breaks,
            "Waste_kg": waste_kg,
            "Safety_Interlock_Triggered": safety,
            "Auto_Shutdown_Flag": auto_shut,
            "Emergency_Stop_Count": emerg_stop,
            "Current_Ratio": current_ratio,
            "Mechanical_Friction_Index": friction,
            "Wear_Score": wear,
            "Material_Buffer_Level_kg": buffer_level,
            "Yarn_Count_Ne": yarn_count,
            "Speed_Override_Flag": speed_override,
            "Torque_Load_Index": torque,
            "Efficiency_Index": efficiency,
            # time features
            "Hour": now.hour,
            "DayOfWeek": now.weekday(),
            "DayOfMonth": now.day,
            # interaction features
            "Temp_x_Friction": temp * friction,
            "Speed_x_Torque": speed * torque,
            "Current_x_Voltage": motor_current * voltage_var,
            "Energy_per_Output": energy / (output_kg + 1),
            "Waste_Ratio": waste_kg / (output_kg + 1),
        }

        # categorical encoding
        cat_cols = ["Machine_ID", "Machine_Type", "Section", "Machine_State", "Shift"]
        cat_vals = [machine_id, machine_type, section, machine_state, shift]
        for col_name, val in zip(cat_cols, cat_vals):
            le = label_encoders[col_name]
            if val in le.classes_:
                raw[col_name + "_enc"] = le.transform([str(val)])[0]
            else:
                raw[col_name + "_enc"] = 0  # safe fallback

        # build feature vector in the same column order the scaler expects
        feature_names = scaler.feature_names_in_
        x_row = pd.DataFrame([{f: raw.get(f, 0) for f in feature_names}])
        x_scaled = pd.DataFrame(scaler.transform(x_row), columns=feature_names)

        # ── run all 4 models ─────────────────────────────────────────────
        results = {}
        for tname, model in models.items():
            results[tname] = model.predict(x_scaled)[0]

        # decode Failure_Mode_Code
        mode_map = {0: "Healthy", 1: "Early Wear", 2: "Moderate Wear", 3: "Critical"}
        mode_code = int(results.get("Failure_Mode_Code", 0))
        mode_label = mode_map.get(mode_code, "Unknown")

        imminent_flag = int(results.get("Failure_Imminent_Flag", 0))
        fail_24h = int(results.get("Target_Failure_24H", 0))
        rul = float(results.get("Target_RUL_Hours", 0))

        # ── derive CONSISTENT overall health from RUL ────────────────────
        # RUL is the anchor truth — everything else must agree with it.
        rul_days = rul / 24  # convert hours to days for human readability

        if rul >= 1250:
            # SAFE ZONE — healthy machine, no concerns
            overall_status = "HEALTHY"
            overall_icon = ""
            overall_color = "#00ff88"
            overall_bg = "rgba(0,255,136,0.08)"
            overall_border = "rgba(0,255,136,0.3)"
            overall_summary = (
                f"Machine is operating normally with <b>{rul:.0f} hours (~{rul_days:.0f} days)</b> "
                f"of useful life remaining. No maintenance action required."
            )
            maint_label = "Not Required"
            maint_color = "#00ff88"
            maint_cls = "pred-healthy"
            maint_sub = "Machine within safe operating range"
            risk_label = "Low Risk"
            risk_color = "#00ff88"
            risk_cls = "pred-healthy"
            risk_sub = "No risk of near-term failure"
        elif rul >= 1100:
            # CAUTION ZONE — degrading, needs attention soon
            overall_status = "NEEDS ATTENTION"
            overall_icon = ""
            overall_color = "#ffc107"
            overall_bg = "rgba(255,193,7,0.08)"
            overall_border = "rgba(255,193,7,0.3)"
            overall_summary = (
                f"Machine is showing wear with <b>{rul:.0f} hours (~{rul_days:.0f} days)</b> "
                f"of useful life remaining. <b>Schedule preventive maintenance</b> within "
                f"the next {max(1, int(rul_days - 46))}-{int(rul_days)} days."
            )
            maint_label = "Schedule Soon"
            maint_color = "#ffc107"
            maint_cls = "pred-warning"
            maint_sub = f"Plan maintenance within ~{int(rul_days)} days"
            risk_label = "Moderate"
            risk_color = "#ffc107"
            risk_cls = "pred-warning"
            risk_sub = "Approaching critical wear zone"
        else:
            # DANGER ZONE — critical, immediate action needed
            overall_status = "CRITICAL"
            overall_icon = ""
            overall_color = "#ff6b6b"
            overall_bg = "rgba(255,60,60,0.08)"
            overall_border = "rgba(255,60,60,0.3)"
            if rul > 100:
                overall_summary = (
                    f"Machine is in critical condition with only <b>{rul:.0f} hours "
                    f"(~{rul_days:.0f} days)</b> of useful life remaining. "
                    f"<b>Immediate maintenance required</b> to prevent unplanned failure."
                )
            else:
                overall_summary = (
                    f"Machine is at imminent risk of failure with only <b>{rul:.0f} hours</b> "
                    f"remaining. <b>Stop the machine and perform emergency maintenance NOW.</b>"
                )
            maint_label = " Immediate"
            maint_color = "#ff6b6b"
            maint_cls = "pred-danger"
            maint_sub = "Critical — act now to prevent failure"
            risk_label = " High Risk"
            risk_color = "#ff6b6b"
            risk_cls = "pred-danger"
            risk_sub = "Failure expected without intervention"

        # Mode card styling
        mode_cls = "pred-healthy" if mode_code == 0 else (
            "pred-info" if mode_code == 1 else (
                "pred-warning" if mode_code == 2 else "pred-danger"))
        mode_color = "#00ff88" if mode_code == 0 else (
            "#00d4ff" if mode_code == 1 else (
                "#ffc107" if mode_code == 2 else "#ff6b6b"))

        rul_color = overall_color  # RUL card matches the overall status

        # ── OVERALL HEALTH BANNER ────────────────────────────────────────
        st.markdown(
            f'<div style="background:{overall_bg}; border:2px solid {overall_border}; '
            f'border-radius:16px; padding:28px 36px; margin-bottom:24px;">'
            f'<div style="display:flex; align-items:center; gap:16px; margin-bottom:12px;">'
            f'<span style="font-size:2.5rem;">{overall_icon}</span>'
            f'<span style="font-size:1.8rem; font-weight:800; color:{overall_color};">'
            f'OVERALL STATUS: {overall_status}</span>'
            f'</div>'
            f'<p style="color:#c9d1d9; font-size:1.1rem; margin:0; line-height:1.6;">'
            f'{overall_summary}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── 4 prediction cards ───────────────────────────────────────────
        st.markdown('<div class="section-header">Detailed Predictions</div>', unsafe_allow_html=True)
        r1, r2, r3, r4 = st.columns(4)

        with r1:
            st.markdown(
                f'<div class="pred-card {mode_cls}">'
                f'<div class="pred-title">Machine Condition</div>'
                f'<div class="pred-value" style="color:{mode_color}">{mode_label}</div>'
                f'<div class="pred-sub">Degradation phase: {mode_code}/3</div>'
                f'</div>', unsafe_allow_html=True)

        with r2:
            st.markdown(
                f'<div class="pred-card {maint_cls}">'
                f'<div class="pred-title">Maintenance Alert</div>'
                f'<div class="pred-value" style="color:{maint_color}">{maint_label}</div>'
                f'<div class="pred-sub">{maint_sub}</div>'
                f'</div>', unsafe_allow_html=True)

        with r3:
            st.markdown(
                f'<div class="pred-card {risk_cls}">'
                f'<div class="pred-title">Failure Risk Level</div>'
                f'<div class="pred-value" style="color:{risk_color}">{risk_label}</div>'
                f'<div class="pred-sub">{risk_sub}</div>'
                f'</div>', unsafe_allow_html=True)

        with r4:
            st.markdown(
                f'<div class="pred-card {mode_cls}">'
                f'<div class="pred-title">Remaining Useful Life</div>'
                f'<div class="pred-value" style="color:{rul_color}">{rul:.0f} h</div>'
                f'<div class="pred-sub">~{rul_days:.0f} days until failure</div>'
                f'</div>', unsafe_allow_html=True)

        # ── RUL gauge ────────────────────────────────────────────────────
        st.markdown("")
        gc1, gc2 = st.columns([2, 1])
        with gc1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=rul,
                title=dict(text="Remaining Useful Life (hours)", font=dict(size=18, color="#8b949e")),
                number=dict(suffix=" hrs", font=dict(size=40, color="#e6edf3")),
                gauge=dict(
                    axis=dict(range=[0, 2000], tickcolor="#8b949e",
                              tickvals=[0, 500, 1000, 1100, 1250, 1500, 2000]),
                    bar=dict(color=rul_color),
                    bgcolor="rgba(13,17,23,0.9)",
                    borderwidth=1,
                    bordercolor="rgba(0,212,255,0.3)",
                    steps=[
                        dict(range=[0, 1100], color="rgba(255,60,60,0.2)"),
                        dict(range=[1100, 1250], color="rgba(255,193,7,0.2)"),
                        dict(range=[1250, 2000], color="rgba(0,255,136,0.12)"),
                    ],
                    threshold=dict(line=dict(color="#ff6b6b", width=3), thickness=0.8, value=1100),
                ),
            ))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e6edf3"),
                              height=320, margin=dict(l=40, r=40, t=60, b=20))
            st.plotly_chart(fig, use_container_width=True)

        with gc2:
            st.markdown(
                '<div style="padding:24px; background:rgba(22,27,34,0.9); border:1px solid '
                'rgba(0,212,255,0.15); border-radius:14px; height:100%;">'
                '<h4 style="color:#00d4ff; margin:0 0 16px 0; font-size:1.1rem;"> Zone Legend</h4>'
                '<div style="margin-bottom:14px;">'
                '<span style="display:inline-block;width:14px;height:14px;background:rgba(0,255,136,0.5);'
                'border-radius:3px;margin-right:8px;vertical-align:middle;"></span>'
                '<span style="color:#c9d1d9;font-size:0.95rem;"><b>Safe Zone</b> (≥ 1250 h)</span><br>'
                '<span style="color:#8b949e;font-size:0.85rem;margin-left:22px;">No action needed</span>'
                '</div>'
                '<div style="margin-bottom:14px;">'
                '<span style="display:inline-block;width:14px;height:14px;background:rgba(255,193,7,0.5);'
                'border-radius:3px;margin-right:8px;vertical-align:middle;"></span>'
                '<span style="color:#c9d1d9;font-size:0.95rem;"><b>Caution Zone</b> (1100–1250 h)</span><br>'
                '<span style="color:#8b949e;font-size:0.85rem;margin-left:22px;">Schedule maintenance</span>'
                '</div>'
                '<div>'
                '<span style="display:inline-block;width:14px;height:14px;background:rgba(255,60,60,0.5);'
                'border-radius:3px;margin-right:8px;vertical-align:middle;"></span>'
                '<span style="color:#c9d1d9;font-size:0.95rem;"><b>Critical Zone</b> (&lt; 1100 h)</span><br>'
                '<span style="color:#8b949e;font-size:0.85rem;margin-left:22px;">Immediate action required</span>'
                '</div>'
                '</div>',
                unsafe_allow_html=True,
            )

        # ── raw prediction table ─────────────────────────────────────────
        with st.expander(" Raw Model Outputs (for technical review)"):
            pred_df = pd.DataFrame([{
                "Failure_Mode_Code": mode_code,
                "Failure_Mode_Label": mode_label,
                "Failure_Imminent_Flag (RUL<1250)": "Yes" if imminent_flag else "No",
                "Target_Failure_24H (RUL≤1100)": "Yes" if fail_24h else "No",
                "Target_RUL_Hours": round(rul, 2),
            }])
            st.dataframe(pred_df, use_container_width=True)
            st.caption(
                "**Note:** 'Failure Imminent' means RUL is below 1250 hours (caution zone). "
                "'Failure 24H' means RUL is below 1100 hours (critical zone). "
                "These are model-trained thresholds, not literal time-to-failure."
            )

        with st.expander("Feature Vector (scaled)"):
            st.dataframe(x_scaled, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL ANALYTICS
# ═════════════════════════════════════════════════════════════════════════════
def page_analytics():
    results = load_results()
    models = load_models()

    st.markdown(
        '<div class="page-banner">'
        '<h2>Model Performance Analytics</h2>'
        '<p>Training results, model comparisons, and feature importance analysis</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── final summary ────────────────────────────────────────────────────────
    if "final_summary" in results:
        st.markdown('<div class="section-header">Final Summary — Best Models</div>', unsafe_allow_html=True)
        summary = results["final_summary"]
        st.dataframe(summary, use_container_width=True, hide_index=True)

    # ── classification targets ───────────────────────────────────────────────
    clf_targets = ["Failure_Mode_Code", "Failure_Imminent_Flag", "Target_Failure_24H"]
    for tname in clf_targets:
        if tname not in results:
            continue
        st.markdown(f'<div class="section-header">{tname.replace("_"," ")} — Model Comparison</div>',
                    unsafe_allow_html=True)
        rdf = results[tname]

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(rdf, x="Model", y=["Accuracy", "Precision", "Recall", "F1"],
                         barmode="group", color_discrete_sequence=COLORS,
                         title="Classification Metrics")
            fig.update_layout(**PLOTLY_LAYOUT, height=380, legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = px.bar(rdf, x="Model", y="CV_F1_mean",
                         error_y="CV_F1_std" if "CV_F1_std" in rdf.columns else None,
                         color_discrete_sequence=["#00ff88"],
                         title="Cross-Validated F1 Score (5-fold)")
            fig.update_layout(**PLOTLY_LAYOUT, height=380)
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(rdf, use_container_width=True, hide_index=True)

    # ── regression target ────────────────────────────────────────────────────
    if "Target_RUL_Hours" in results:
        st.markdown('<div class="section-header">Target RUL Hours — Model Comparison</div>',
                    unsafe_allow_html=True)
        rdf = results["Target_RUL_Hours"]

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(rdf, x="Model", y=["MAE", "RMSE"],
                         barmode="group", color_discrete_sequence=["#00d4ff", "#ff6b6b"],
                         title="Error Metrics (lower is better)")
            fig.update_layout(**PLOTLY_LAYOUT, height=380, legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = px.bar(rdf, x="Model", y="R2",
                         color_discrete_sequence=["#00ff88"],
                         title="R² Score (higher is better)")
            fig.update_layout(**PLOTLY_LAYOUT, height=380)
            fig.update_layout(yaxis=dict(
                range=[min(rdf["R2"].min() - 0.001, 0.996), 1.0001],
                gridcolor="rgba(139,148,158,0.1)"))
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(rdf, use_container_width=True, hide_index=True)

    # ── feature importance ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">Feature Importance (Best Models)</div>', unsafe_allow_html=True)

    # We need the feature columns to display importance
    # Rebuild them from the scaler
    scaler = load_scaler()
    feature_cols = list(scaler.feature_names_in_)

    tabs = st.tabs([t.replace("_", " ") for t in models.keys()])
    for tab, (tname, model) in zip(tabs, models.items()):
        with tab:
            if hasattr(model, "feature_importances_"):
                imp = pd.Series(model.feature_importances_, index=feature_cols)
                imp = imp.sort_values(ascending=True).tail(15)
                fig = px.bar(x=imp.values, y=imp.index, orientation="h",
                             color=imp.values,
                             color_continuous_scale=["#0d1117", "#00d4ff", "#00ff88"],
                             title=f"Top 15 Features — {tname}")
                fig.update_layout(**PLOTLY_LAYOUT, height=450, showlegend=False,
                                  coloraxis_showscale=False,
                                  xaxis_title="Importance", yaxis_title="")
                st.plotly_chart(fig, use_container_width=True)
            elif hasattr(model, "coef_"):
                coef = model.coef_.flatten() if model.coef_.ndim > 1 else model.coef_
                if len(coef) == len(feature_cols):
                    imp = pd.Series(np.abs(coef), index=feature_cols)
                    imp = imp.sort_values(ascending=True).tail(15)
                    fig = px.bar(x=imp.values, y=imp.index, orientation="h",
                                 color=imp.values,
                                 color_continuous_scale=["#0d1117", "#a855f7", "#f472b6"],
                                 title=f"Top 15 Coefficients (abs) — {tname}")
                    fig.update_layout(**PLOTLY_LAYOUT, height=450, showlegend=False,
                                      coloraxis_showscale=False,
                                      xaxis_title="|Coefficient|", yaxis_title="")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"Coefficient shape does not match feature count for {tname}.")
            else:
                st.info(f"Feature importance is not available for {tname}.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — DATA EXPLORER
# ═════════════════════════════════════════════════════════════════════════════
def page_data_explorer():
    df = load_data()

    st.markdown(
        '<div class="page-banner">'
        '<h2>Data Explorer</h2>'
        '<p>Browse, filter, and analyze the raw dataset (5,001 rows × 30 columns)</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── filters ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Filters</div>', unsafe_allow_html=True)
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        sel_types = st.multiselect("Machine Type", df["Machine_Type"].unique().tolist(),
                                   default=df["Machine_Type"].unique().tolist())
    with fc2:
        sel_sections = st.multiselect("Section", df["Section"].unique().tolist(),
                                      default=df["Section"].unique().tolist())
    with fc3:
        sel_phases = st.multiselect("Degradation Phase", df["Degradation_Phase"].unique().tolist(),
                                    default=df["Degradation_Phase"].unique().tolist())

    filtered = df[
        (df["Machine_Type"].isin(sel_types)) &
        (df["Section"].isin(sel_sections)) &
        (df["Degradation_Phase"].isin(sel_phases))
    ]
    st.caption(f"Showing **{len(filtered):,}** of {len(df):,} rows")

    # ── data table ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Dataset</div>', unsafe_allow_html=True)
    st.dataframe(filtered, use_container_width=True, height=400)

    # ── column statistics ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Column Statistics</div>', unsafe_allow_html=True)
    numeric_df = filtered.select_dtypes(include=[np.number])
    st.dataframe(numeric_df.describe().T.style.format("{:.2f}"),
                 use_container_width=True, height=400)

    # ── correlation heatmap ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">Correlation Matrix</div>', unsafe_allow_html=True)
    sel_corr = st.multiselect(
        "Select columns for correlation",
        numeric_df.columns.tolist(),
        default=["Temp_C", "Energy_kWh", "Motor_Current_A", "Output_kg",
                 "Wear_Score", "RUL_Hours", "Efficiency_Index",
                 "Mechanical_Friction_Index"],
    )
    if len(sel_corr) >= 2:
        corr = filtered[sel_corr].corr()
        fig = px.imshow(corr, text_auto=".2f",
                        color_continuous_scale=["#ff6b6b", "#0d1117", "#00ff88"],
                        aspect="auto")
        fig.update_layout(**PLOTLY_LAYOUT, height=500)
        st.plotly_chart(fig, use_container_width=True)

    # ── distribution histograms ──────────────────────────────────────────────
    st.markdown('<div class="section-header">Distribution Analysis</div>', unsafe_allow_html=True)
    hist_col = st.selectbox("Select column", numeric_df.columns.tolist(), index=0)
    fig = px.histogram(filtered, x=hist_col, nbins=50,
                       color_discrete_sequence=["#00d4ff"],
                       marginal="box")
    fig.update_layout(**PLOTLY_LAYOUT, height=400,
                      xaxis_title=hist_col, yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# ROUTER
# ═════════════════════════════════════════════════════════════════════════════
if "Dashboard" in page:
    page_dashboard()
elif "Prediction" in page:
    page_prediction()
elif "Analytics" in page:
    page_analytics()
elif "Data Explorer" in page:
    page_data_explorer()
