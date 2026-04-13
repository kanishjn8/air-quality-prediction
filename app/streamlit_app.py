"""
AirMind — Next-Day PM2.5 Forecast for Indian Cities
Streamlit Dashboard (v4 — Clean Black, No Emoji)

Run: uv run streamlit run app/streamlit_app.py
"""

import os
import inspect
import sys
import json
import subprocess

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ────────────────────────────────────────────────────────────────
# Page Config
# ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AirMind — PM2.5 Forecast",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def st_image(path: str, **kwargs):
    """Compatibility wrapper for Streamlit image API."""
    try:
        params = inspect.signature(st.image).parameters
        if "use_container_width" in kwargs and "use_container_width" not in params:
            kwargs["use_column_width"] = kwargs.pop("use_container_width")
    except Exception:
        if "use_container_width" in kwargs:
            kwargs["use_column_width"] = kwargs.pop("use_container_width")
    return st.image(path, **kwargs)


def fmt(val, decimals=1):
    """Format a numeric value to fixed decimals, return '—' for None."""
    if val is None:
        return "—"
    try:
        return f"{float(val):.{decimals}f}"
    except (TypeError, ValueError):
        return str(val)


# ────────────────────────────────────────────────────────────────
# CSS
# ────────────────────────────────────────────────────────────────
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* ── Black Canvas ── */
[data-testid="stAppViewContainer"] {
    background: #060606 !important;
    color: #e4e4e7 !important;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] {
    background: #0a0a0a !important;
    border-right: 1px solid #161616 !important;
}
.stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
.stMarkdown h4, .stMarkdown li, .stText, .stCaption p {
    color: #a1a1aa !important;
}
hr { border-color: #1a1a1a !important; }

/* ── Hero ── */
.hero-title {
    font-size: 2.4rem;
    font-weight: 900;
    letter-spacing: -1.2px;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff;
    margin: 0 0 0.15rem 0;
}
.hero-sub {
    font-size: 0.9rem;
    color: #52525b !important;
    margin: 0 0 1.5rem 0;
}

/* ── Buttons ── */
.stButton > button {
    background: #18181b !important;
    color: #d4d4d8 !important;
    border: 1px solid #27272a !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    padding: 0.5rem 1.1rem !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.2px !important;
}
.stButton > button:hover {
    background: #27272a !important;
    border-color: #3f3f46 !important;
    color: #fff !important;
}
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
    background: #fafafa !important;
    color: #09090b !important;
    border: 1px solid #e4e4e7 !important;
    font-weight: 700 !important;
}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {
    background: #d4d4d8 !important;
}

/* ── City Cards ── */
.city-card {
    background: #0e0e10;
    border: 1px solid #1c1c20;
    border-radius: 14px;
    padding: 1.6rem 1.5rem;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
    margin-bottom: 0.5rem;
}
.city-card:hover {
    border-color: #2a2a2e;
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.5);
}
.accent-bar {
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}
.card-city-name {
    font-size: 1.4rem;
    font-weight: 800;
    color: #ffffff !important;
    margin: 0.2rem 0 1.4rem 0;
    letter-spacing: -0.3px;
}
.card-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.5rem;
}
.card-stat-label {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #52525b;
    margin-bottom: 0.3rem;
}
.card-stat-val {
    font-size: 2.2rem;
    font-weight: 900;
    line-height: 1;
    margin-bottom: 0.35rem;
    letter-spacing: -1px;
}
.card-badge {
    display: inline-block;
    font-size: 0.6rem;
    font-weight: 700;
    padding: 0.18rem 0.5rem;
    border-radius: 4px;
    letter-spacing: 0.8px;
    text-transform: uppercase;
}
.card-meta {
    display: flex;
    gap: 1.5rem;
    margin-top: 1.2rem;
    padding-top: 1rem;
    border-top: 1px solid #1c1c20;
    font-size: 0.88rem;
    color: #71717a;
    font-weight: 500;
}
.card-meta strong {
    color: #e4e4e7;
    font-weight: 700;
}
.meta-label {
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #52525b;
    display: block;
    margin-bottom: 0.1rem;
}

/* ── Section Headers ── */
.section-header {
    font-size: 1.4rem;
    font-weight: 800;
    color: #ffffff !important;
    letter-spacing: -0.5px;
    margin: 2rem 0 0.3rem 0;
}
.section-sub {
    font-size: 0.85rem;
    color: #52525b !important;
    margin: 0 0 1.5rem 0;
}

/* ── Tabs (pill style) ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: #0e0e10;
    border-radius: 12px;
    padding: 4px;
    border: 1px solid #1c1c20;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 0.5rem 1.1rem;
    font-size: 0.85rem;
    font-weight: 600;
    color: #52525b;
    background: transparent;
    border: none !important;
    transition: all 0.15s ease;
}
.stTabs [data-baseweb="tab"]:hover { color: #a1a1aa; }
.stTabs [aria-selected="true"] {
    color: #fafafa !important;
    background: #27272a !important;
    border: none !important;
}
div[data-baseweb="tab-border"] { display: none !important; }

/* ── Plots ── */
.stPlotContainer {
    background: #0e0e10 !important;
    border-radius: 12px;
    border: 1px solid #1c1c20;
    padding: 0.3rem;
}

/* ── Dataframes ── */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
}

.timestamp {
    font-size: 0.75rem;
    color: #3f3f46;
    font-family: 'SF Mono', ui-monospace, monospace;
    margin-top: 1rem;
}
</style>""", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────
CITIES = ["Delhi", "Bengaluru", "Kolkata", "Hyderabad"]

CITY_ACCENT = {
    "Delhi":     "#ef4444",
    "Bengaluru": "#22c55e",
    "Kolkata":   "#a855f7",
    "Hyderabad": "#38bdf8",
}


def pm25_category(val):
    if val is None:
        return "Unknown", "#3f3f46"
    v = float(val)
    if v <= 30:   return "Good",         "#22c55e"
    if v <= 60:   return "Satisfactory", "#84cc16"
    if v <= 90:   return "Moderate",     "#eab308"
    if v <= 120:  return "Poor",         "#f97316"
    if v <= 250:  return "Very Poor",    "#ef4444"
    return "Severe", "#b91c1c"


def load_live_readings():
    path = "data/live/latest_reading.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def load_live_predictions():
    path = "data/live/predictions.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ────────────────────────────────────────────────────────────────
# Header
# ────────────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">AirMind</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Next-day PM2.5 forecast for Indian cities</p>',
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns([1, 1, 5])
with c1:
    refresh = st.button("Refresh Data", type="primary", use_container_width=True)
with c2:
    forecast = st.button("Run Forecast", use_container_width=True)

if refresh:
    with st.spinner("Fetching from WAQI API..."):
        try:
            r = subprocess.run(
                [sys.executable, "src/05_live_ingest.py"],
                capture_output=True, text=True, timeout=30,
            )
            if r.returncode == 0:
                st.success("Live data refreshed!")
                st.rerun()
            else:
                st.error(f"Error: {r.stderr[:300]}")
        except Exception as e:
            st.error(f"Failed: {e}")

if forecast:
    with st.spinner("Running model inference..."):
        try:
            r = subprocess.run(
                [sys.executable, "src/06_predict_live.py"],
                capture_output=True, text=True, timeout=30,
            )
            if r.returncode == 0:
                st.success("Predictions updated!")
                st.rerun()
            else:
                st.error(f"Error: {r.stderr[:300]}")
        except Exception as e:
            st.error(f"Failed: {e}")

st.markdown("")

# ────────────────────────────────────────────────────────────────
# Tabs
# ────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "Live Forecast",
    "Historical",
    "Explainability",
])


# ═══════════════════════════════════════════════════════════════
# Tab 1: Live Forecast
# ═══════════════════════════════════════════════════════════════
with tab1:
    readings = load_live_readings()
    predictions = load_live_predictions()

    if readings is None:
        st.info("No live data yet. Click **Refresh Data** to fetch current readings.")
    else:
        cols = st.columns(2, gap="medium")

        for i, city in enumerate(CITIES):
            col = cols[i % 2]
            with col:
                reading = readings.get(city)
                pred_val = None
                if predictions:
                    pred_val = (predictions.get("predictions") or {}).get(city)

                if reading is None:
                    st.warning(f"**{city}** — No data")
                    continue

                current_pm = reading.get("pm2_5")
                current_aqi = reading.get("aqi")
                temp = reading.get("temperature")
                wind = reading.get("wind_speed")

                cat_curr, clr_curr = pm25_category(current_pm)
                cat_pred, clr_pred = pm25_category(pred_val)
                accent = CITY_ACCENT.get(city, "#3f3f46")

                st.markdown(f"""<div class="city-card">
<div class="accent-bar" style="background:{accent};"></div>
<div class="card-city-name">{city}</div>
<div class="card-grid">
<div>
<div class="card-stat-label">Current PM2.5</div>
<div class="card-stat-val" style="color:{clr_curr};">{fmt(current_pm)}</div>
<span class="card-badge" style="background:{clr_curr}15;color:{clr_curr};">{cat_curr}</span>
</div>
<div>
<div class="card-stat-label">Tomorrow</div>
<div class="card-stat-val" style="color:{clr_pred};">{fmt(pred_val)}</div>
<span class="card-badge" style="background:{clr_pred}15;color:{clr_pred};">{cat_pred}</span>
</div>
</div>
<div class="card-meta">
<div>
<span class="meta-label">AQI</span>
<strong>{fmt(current_aqi, 0)}</strong>
</div>
<div>
<span class="meta-label">Temp</span>
<strong>{fmt(temp, 1)} C</strong>
</div>
<div>
<span class="meta-label">Wind</span>
<strong>{fmt(wind, 1)} m/s</strong>
</div>
</div>
</div>""", unsafe_allow_html=True)

        if predictions:
            ts = predictions.get("generated_at", "")
            st.markdown(f'<p class="timestamp">Last forecast: {ts}</p>',
                        unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# Tab 2: Historical
# ═══════════════════════════════════════════════════════════════
with tab2:
    if os.path.exists("outputs/predictions.csv"):
        preds_df = pd.read_csv("outputs/predictions.csv")

        st.markdown('<p class="section-header">Actual vs Predicted</p>',
                    unsafe_allow_html=True)
        st.markdown('<p class="section-sub">Test period performance across cities</p>',
                    unsafe_allow_html=True)

        plt.rcParams.update({
            "figure.facecolor": "#0e0e10",
            "axes.facecolor": "#0e0e10",
            "axes.edgecolor": "#1c1c20",
            "axes.labelcolor": "#71717a",
            "text.color": "#71717a",
            "xtick.color": "#52525b",
            "ytick.color": "#52525b",
            "grid.color": "#1c1c20",
            "grid.alpha": 0.8,
        })

        for city in CITIES:
            city_preds = preds_df[preds_df["city"] == city].copy()
            if city_preds.empty:
                continue

            city_preds = city_preds.sort_values("date").reset_index(drop=True)
            accent = CITY_ACCENT.get(city, "#71717a")

            fig, ax = plt.subplots(figsize=(12, 3))
            ax.plot(city_preds.index, city_preds["actual_pm2_5"],
                    label="Actual", alpha=0.8, color="#71717a", linewidth=1.2)
            ax.plot(city_preds.index, city_preds["predicted_pm2_5"],
                    label="Predicted", alpha=0.9, color=accent, linewidth=1.5,
                    linestyle="--")
            ax.fill_between(city_preds.index,
                            city_preds["actual_pm2_5"],
                            city_preds["predicted_pm2_5"],
                            alpha=0.06, color=accent)
            ax.set_ylabel("PM2.5", fontsize=8)
            ax.legend(loc="upper right", fontsize=7, framealpha=0.2)
            ax.grid(True)
            ax.set_title(city, fontsize=11, fontweight="bold",
                         color="#ffffff", pad=10)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown("---")

        m1, m2 = st.columns(2)
        with m1:
            st.markdown('<p class="section-header" style="font-size:1.1rem;">Per-City Metrics</p>',
                        unsafe_allow_html=True)
            if os.path.exists("outputs/per_city_metrics.csv"):
                st.dataframe(pd.read_csv("outputs/per_city_metrics.csv"),
                             use_container_width=True, hide_index=True)
        with m2:
            st.markdown('<p class="section-header" style="font-size:1.1rem;">Model Comparison</p>',
                        unsafe_allow_html=True)
            if os.path.exists("outputs/model_comparison.csv"):
                st.dataframe(pd.read_csv("outputs/model_comparison.csv"),
                             use_container_width=True, hide_index=True)
    else:
        st.info("No prediction data. Run training pipeline first.")


# ═══════════════════════════════════════════════════════════════
# Tab 3: Explainability
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-header">Explainability</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="section-sub">SHAP feature attribution for the XGBoost model</p>',
                unsafe_allow_html=True)

    x1, x2, x3, x4 = st.tabs([
        "Feature Importance", "Cross-City Influence", "Spike Events", "Lag Analysis",
    ])

    with x1:
        if os.path.exists("outputs/shap_summary.png"):
            st_image("outputs/shap_summary.png", use_container_width=True)
        else:
            st.info("Run 04_explain.py to generate SHAP plots.")

    with x2:
        st.caption(
            "Mean |SHAP| of neighbor PM2.5 lag features. "
            "Higher values indicate stronger cross-city influence."
        )
        if os.path.exists("outputs/cross_city_influence.png"):
            st_image("outputs/cross_city_influence.png", use_container_width=True)
        else:
            st.info("Run 04_explain.py to generate heatmap.")

    with x3:
        found = False
        for i in range(1, 4):
            path = f"outputs/shap_waterfall_spike{i}.png"
            if os.path.exists(path):
                st_image(path, caption=f"Spike Event {i}", use_container_width=True)
                found = True
        if not found:
            st.info("Run 04_explain.py to generate waterfall plots.")

    with x4:
        st.caption("PM2.5 lag feature contributions — 1, 2, 3, 7 days back.")
        if os.path.exists("outputs/lag_importance.png"):
            st_image("outputs/lag_importance.png", use_container_width=True)
        else:
            st.info("Run 04_explain.py to generate lag chart.")
