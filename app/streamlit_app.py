"""
AirMind — Next-Day PM2.5 Forecast for Indian Cities
Streamlit Dashboard

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
    initial_sidebar_state="expanded",
)


def st_image(path: str, **kwargs):
    """Compatibility wrapper for Streamlit image API.

    Streamlit versions prior to introducing/removing certain kwargs may not
    accept `use_container_width`. This helper maps it to `use_column_width`
    when needed.
    """
    try:
        params = inspect.signature(st.image).parameters
        if "use_container_width" in kwargs and "use_container_width" not in params:
            kwargs["use_column_width"] = kwargs.pop("use_container_width")
    except Exception:
        # If signature inspection fails for any reason, fall back to the older kwarg.
        if "use_container_width" in kwargs:
            kwargs["use_column_width"] = kwargs.pop("use_container_width")

    return st.image(path, **kwargs)

# ────────────────────────────────────────────────────────────────
# Styling
# ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }

    /* Force Dark Theme regardless of Streamlit user settings */
    [data-testid="stAppViewContainer"] {
        background-color: #0f172a !important;
        background-image: 
            radial-gradient(circle at 10% 20%, rgba(59, 130, 246, 0.05) 0%, transparent 40%),
            radial-gradient(circle at 90% 80%, rgba(16, 185, 129, 0.05) 0%, transparent 40%) !important;
        color: #f8fafc !important;
    }

    [data-testid="stHeader"] {
        background-color: transparent !important;
    }
    
    /* Ensure all default Streamlit text is visible against the dark background */
    .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown li, .stText {
        color: #e2e8f0 !important;
    }

    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00C6FF 0%, #0072FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        letter-spacing: -1.5px;
        line-height: 1.2;
    }
    .sub-header {
        font-size: 1.15rem;
        color: #94a3b8;
        margin-bottom: 3rem;
        font-weight: 400;
        letter-spacing: 0.2px;
    }
    
    /* Modern Premium Metric Cards */
    .metric-card {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 1.8rem;
        margin-bottom: 1.5rem;
        transition: all 0.4s ease;
        color: #f8fafc !important;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 25px 50px -12px rgba(0,0,0,0.7);
        border-color: rgba(255, 255, 255, 0.25);
        background: linear-gradient(145deg, #24334a, #111a30);
    }
    
    /* Floating gradient glow effect */
    .metric-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 4px;
        background: var(--card-glow, linear-gradient(90deg, #3b82f6, #8b5cf6));
        opacity: 0.9;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.8rem 1rem;
        font-size: 1.05rem;
        font-weight: 500;
        color: #64748b;
        transition: color 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #f8fafc;
    }
    .stTabs [aria-selected="true"] {
        color: #00f2fe !important;
        border-bottom-color: #00f2fe !important;
    }
    
    /* Sleeker Sidebar styling overrides */
    [data-testid="stSidebar"] {
        background-color: #0b1120 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0;
    }
    
    hr {
        border-color: rgba(255,255,255,0.05);
        margin: 2rem 0;
    }
    
    .stat-label {
        color: #94a3b8; 
        font-size: 0.85rem; 
        font-weight: 600; 
        text-transform: uppercase; 
        letter-spacing: 0.5px;
        margin-bottom: 0.4rem;
    }
    .stat-value {
        font-size: 2.2rem; 
        font-weight: 800; 
        line-height: 1.1;
        margin-bottom: 0.4rem;
    }
    .stat-category {
        font-size: 0.85rem; 
        font-weight: 700;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        display: inline-block;
        letter-spacing: 0.3px;
    }
    .meta-data {
        margin-top: 1.5rem; 
        color: #64748b; 
        font-size: 0.85rem;
        display: flex;
        gap: 1.2rem;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }
</style>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────
# Helper functions
# ────────────────────────────────────────────────────────────────
def pm25_category(val):
    """Classify PM2.5 into India NAQI categories."""
    if val is None:
        return "Unknown", "#808080", "❓"
    if val <= 30:   return "Good", "#00b050", "🟢"
    elif val <= 60: return "Satisfactory", "#92d050", "🟡"
    elif val <= 90: return "Moderate", "#03fff7", "🟠"
    elif val <= 120: return "Poor", "#ff9900", "🔴"
    elif val <= 250: return "Very Poor", "#ff0000", "🟣"
    else:            return "Severe", "#c00000", "⚫"


def load_live_readings():
    """Load latest live readings from JSON."""
    path = "data/live/latest_reading.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def load_live_predictions():
    """Load live predictions from JSON."""
    path = "data/live/predictions.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ────────────────────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌫️ AirMind Controls")
    st.markdown("---")

    if st.button("🔄 Refresh Live Data", use_container_width=True):
        with st.spinner("Fetching live data from WAQI API..."):
            try:
                result = subprocess.run(
                        [sys.executable, "src/05_live_ingest.py"],
                    capture_output=True, text=True, timeout=30,
                )
                if result.returncode == 0:
                    st.success("Live data refreshed!")
                else:
                    st.error(f"Error: {result.stderr}")
            except Exception as e:
                st.error(f"Failed: {e}")

    if st.button("🔮 Run Predictions", use_container_width=True):
        with st.spinner("Running model inference..."):
            try:
                result = subprocess.run(
                        [sys.executable, "src/06_predict_live.py"],
                    capture_output=True, text=True, timeout=30,
                )
                if result.returncode == 0:
                    st.success("Predictions updated!")
                else:
                    st.error(f"Error: {result.stderr}")
            except Exception as e:
                st.error(f"Failed: {e}")

    st.markdown("---")

    cities = ["Delhi", "Bengaluru", "Kolkata", "Hyderabad"]
    selected_cities = st.multiselect(
        "🏙️ Select Cities",
        cities,
        default=cities,
    )

    st.markdown("---")
    st.markdown("### 📊 Data Range")
    if os.path.exists("data/processed/features.parquet"):
        df = pd.read_parquet("data/processed/features.parquet")
        min_date = df["date"].min().date()
        max_date = df["date"].max().date()
        date_range = st.date_input(
            "Historical range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
    else:
        date_range = None
        st.warning("No feature data found. Run the pipeline first.")

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#7f8c8d; font-size:0.85rem;'>"
        "AirMind vNext<br>XGBoost + cross-city features</div>",
        unsafe_allow_html=True,
    )


# ────────────────────────────────────────────────────────────────
# Header
# ────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🌫️ AirMind</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Next-Day PM2.5 Forecast for Indian Cities — '
    'Powered by XGBoost + cross-city features</p>',
    unsafe_allow_html=True,
)


# ────────────────────────────────────────────────────────────────
# Tabs
# ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📡 Live Forecast",
    "📈 Historical Trends",
    "🔍 Explainability",
    "🏗️ Model Info",
])


# ═══════════════════════════════════════════════════════════════
# Tab 1: Live Forecast
# ═══════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Real-Time Air Quality & Tomorrow's Prediction")

    readings = load_live_readings()
    predictions = load_live_predictions()

    if readings is None:
        st.info("No live data available. Click **Refresh Live Data** in the sidebar.")
    else:
        cols = st.columns(min(len(selected_cities), 3))

        for i, city in enumerate(selected_cities):
            col = cols[i % len(cols)]
            with col:
                reading = readings.get(city)
                pred_val = None
                if predictions:
                    pred_val = (predictions.get("predictions") or {}).get(city)

                if reading is None:
                    st.error(f"**{city}** — Data unavailable")
                    continue

                current_pm = reading.get("pm2_5")
                current_aqi = reading.get("aqi")

                cat_curr, color_curr, emoji_curr = pm25_category(current_pm)
                cat_pred, color_pred, emoji_pred = pm25_category(pred_val)

                st.markdown(f"""
                <div class="metric-card" style="--card-glow: linear-gradient(90deg, {color_curr}, {color_pred});">
                    <h3 style="margin:0 0 1.2rem 0; font-size: 1.4rem; font-weight: 800; color: #f8fafc; display: flex; align-items: center; gap: 0.5rem; letter-spacing: -0.5px;">
                        <span style="font-size: 1.6rem; filter: drop-shadow(0 2px 4px rgba(0,0,0,0.2));">{emoji_curr}</span> {city}
                    </h3>
                    
                    <div style="display:grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                        <div>
                            <div class="stat-label">Current PM2.5</div>
                            <div class="stat-value" style="color: {color_curr}; text-shadow: 0 0 20px {color_curr}40;">{current_pm or 'N/A'}</div>
                            <div class="stat-category" style="background-color: {color_curr}1A; color: {color_curr}; border: 1px solid {color_curr}4D;">{cat_curr}</div>
                        </div>
                        <div>
                            <div class="stat-label">Tomorrow's Forecast</div>
                            <div class="stat-value" style="color: {color_pred}; text-shadow: 0 0 20px {color_pred}40;">{pred_val or 'N/A'}</div>
                            <div class="stat-category" style="background-color: {color_pred}1A; color: {color_pred}; border: 1px solid {color_pred}4D;">{cat_pred}</div>
                        </div>
                    </div>
                    
                    <div class="meta-data">
                        <span title="Air Quality Index">AQI: <strong style="color:#e2e8f0">{current_aqi or 'N/A'}</strong></span>
                        <span>Temp: <strong style="color:#e2e8f0">{reading.get('temperature', 'N/A')}°C</strong></span>
                        <span>Wind: <strong style="color:#e2e8f0">{reading.get('wind_speed', 'N/A')} m/s</strong></span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        if predictions:
            st.caption(f"Last prediction: {predictions.get('generated_at', 'Unknown')}")


# ═══════════════════════════════════════════════════════════════
# Tab 2: Historical Trends
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Actual vs Predicted PM2.5 (Test Period)")

    if os.path.exists("outputs/predictions.csv"):
        preds_df = pd.read_csv("outputs/predictions.csv")

        for city in selected_cities:
            city_preds = preds_df[preds_df["city"] == city].copy()
            if city_preds.empty:
                continue

            st.markdown(f"#### {city}")
            city_preds = city_preds.sort_values("date").reset_index(drop=True)

            fig, ax = plt.subplots(figsize=(10, 3.5))
            ax.plot(city_preds.index, city_preds["actual_pm2_5"],
                    label="Actual", alpha=0.8, color="#2980b9", linewidth=1.5)
            ax.plot(city_preds.index, city_preds["predicted_pm2_5"],
                    label="Predicted", alpha=0.8, color="#e74c3c", linewidth=1.5, linestyle="--")
            ax.fill_between(city_preds.index,
                            city_preds["actual_pm2_5"], city_preds["predicted_pm2_5"],
                            alpha=0.1, color="#e74c3c")
            ax.set_ylabel("PM2.5 (µg/m³)")
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)
            ax.set_title(f"{city} — Test Period Predictions", fontsize=11)
            st.pyplot(fig)
            plt.close()

        # Per-city metrics
        st.markdown("---")
        st.markdown("#### Per-City Test Metrics")
        if os.path.exists("outputs/per_city_metrics.csv"):
            metrics_df = pd.read_csv("outputs/per_city_metrics.csv")
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        # Model comparison
        if os.path.exists("outputs/model_comparison.csv"):
            st.markdown("#### Model Comparison (XGBoost vs RF Baseline)")
            comp_df = pd.read_csv("outputs/model_comparison.csv")
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
    else:
        st.info("No prediction data found. Run the training pipeline first (`02 → 03`).")


# ═══════════════════════════════════════════════════════════════
# Tab 3: Explainability
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Model Explainability (XAI)")

    xai_tab1, xai_tab2, xai_tab3, xai_tab4 = st.tabs([
        "SHAP Summary", "Cross-City Influence", "Spike Events", "Temporal Lags",
    ])

    with xai_tab1:
        st.markdown("#### Feature Importance (SHAP Beeswarm Plot)")
        if os.path.exists("outputs/shap_summary.png"):
            st_image("outputs/shap_summary.png", use_container_width=True)
        else:
            st.info("Run `04_explain.py` to generate SHAP plots.")

    with xai_tab2:
        st.markdown("#### Cross-City PM2.5 Influence Matrix")
        st.markdown(
            "Each cell shows the mean |SHAP value| of the neighbor's lagged PM2.5 "
            "feature when predicting the target city. Higher values mean stronger influence."
        )
        if os.path.exists("outputs/cross_city_influence.png"):
            st_image("outputs/cross_city_influence.png", use_container_width=True)
        else:
            st.info("Run `04_explain.py` to generate the influence heatmap.")

    with xai_tab3:
        st.markdown("#### SHAP Waterfall for Top PM2.5 Spike Events")
        found_any = False
        for i in range(1, 4):
            path = f"outputs/shap_waterfall_spike{i}.png"
            if os.path.exists(path):
                st_image(path, caption=f"Spike Event #{i}", use_container_width=True)
                found_any = True
        if not found_any:
            st.info("Run `04_explain.py` to generate waterfall plots.")

    with xai_tab4:
        st.markdown("#### Temporal Lag Importance")
        st.markdown(
            "Shows how much each PM2.5 lag feature (1, 2, 3, 7 days back) "
            "contributes to the model's predictions on average."
        )
        if os.path.exists("outputs/lag_importance.png"):
            st_image("outputs/lag_importance.png", use_container_width=True)
        else:
            st.info("Run `04_explain.py` to generate lag importance chart.")


# ═══════════════════════════════════════════════════════════════
# Tab 4: Model Info
# ═══════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Model Architecture & Training Details")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Architecture")
        st.markdown("""
        ```
        AirMind — XGBoost + Cross-City Features
        ═══════════════════════════════════════

        ┌───────────────────────────┐
        │  Raw Pollutant Features   │
        │  aqi, co, no, no2, o3,   │
        │  so2, pm2_5, pm10, nh3   │
        └───────────┬───────────────┘
                    │
        ┌───────────┴───────────────┐
        │  Temporal Lag Features    │
        │  pm2_5/aqi lag 1,2,3,7   │
        │  rolling mean/std 3,7    │
        └───────────┬───────────────┘
                    │
        ┌───────────┴───────────────┐
        │  Cross-City Features      │
        │  neighbor_*_pm2_5_lag1    │
        │  neighbor_*_wind_lag1     │
        │  (distance-weighted)      │
        └───────────┬───────────────┘
                    │
        ┌───────────┴───────────────┐
        │  Calendar + Meteorology   │
        │  dayofweek, month,        │
        │  season flags, temp,      │
        │  wind, rainfall, pressure │
        └───────────┬───────────────┘
                    │
        ┌───────────┴───────────────┐
        │  XGBoost Regressor        │
        │  2000 gradient-boosted    │
        │  trees (depth 8)          │
        │  log1p → predict → expm1  │
        │  → PM2.5 (next day)       │
        └───────────────────────────┘
        ```
        """)

    with col2:
        st.markdown("#### XGBoost Hyperparameters")
        st.markdown("""
        | Parameter | Value |
        |-----------|-------|
        | n_estimators | 2000 |
        | learning_rate | 0.03 |
        | max_depth | 8 |
        | min_child_weight | 3 |
        | subsample | 0.8 |
        | colsample_bytree | 0.7 |
        | gamma | 0.1 |
        | reg_alpha (L1) | 0.05 |
        | reg_lambda (L2) | 0.5 |
        | objective | reg:squarederror |
        | target transform | log1p(PM2.5) |
        | seed | 42 |
        """)

        st.markdown("#### RF Baseline")
        st.markdown("""
        | Parameter | Value |
        |-----------|-------|
        | n_estimators | 300 |
        | max_depth | 12 |
        | target transform | log1p(PM2.5) |
        | Features | No cross-city |
        """)

    st.markdown("---")

    # Feature columns info
    if os.path.exists("models/feature_columns.json"):
        with open("models/feature_columns.json") as f:
            feat_cols = json.load(f)
        neighbor_count = sum(1 for c in feat_cols if c.startswith("neighbor_"))
        core_count = len(feat_cols) - neighbor_count
        st.markdown(f"#### Feature Summary: **{len(feat_cols)} total** ({core_count} core + {neighbor_count} cross-city)")

    st.markdown("---")
    st.markdown("#### Data Provenance")

    # Read date range from features if available
    date_info = "2020–2023"
    if os.path.exists("data/processed/features.parquet"):
        feat_df = pd.read_parquet("data/processed/features.parquet", columns=["date", "split"])
        date_info = f"{feat_df['date'].min().date()} to {feat_df['date'].max().date()}"
        split_counts = feat_df["split"].value_counts().to_dict()
        split_str = " / ".join(f"{k}: {v}" for k, v in sorted(split_counts.items()))
        st.markdown(f"**Split sizes:** {split_str}")

    st.markdown(f"""
    | Source | Details |
    |--------|---------|
    | **Historical Data** | `merged.csv` — 5 Indian cities |
    | **Date Range** | {date_info} |
    | **Cities** | Delhi, Bengaluru, Kolkata, Hyderabad |
    | **Target** | Next-day PM2.5 (shifted by -1 day) |
    | **Live API** | WAQI (World Air Quality Index) |
    | **Research Gaps** | Cross-city dependencies (explicit spatial features), missing data bias (MCAR/MNAR) |
    """)
