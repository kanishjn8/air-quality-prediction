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
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(120deg, #1a5276, #2980b9, #48c9b0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #7f8c8d;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        border-radius: 12px;
        padding: 1.2rem;
        border-left: 4px solid;
        margin-bottom: 0.8rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px 8px 0 0;
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
    path = "outputs/live_predictions.json"
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
                    [sys.executable, "src/06_live_ingest.py"],
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
                    [sys.executable, "src/07_predict_live.py"],
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
        "AirMind v1.0<br>GNN + LSTM + MLP Fusion</div>",
        unsafe_allow_html=True,
    )


# ────────────────────────────────────────────────────────────────
# Header
# ────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🌫️ AirMind</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Next-Day PM2.5 Forecast for Indian Cities — '
    'Powered by Spatiotemporal GNN + LSTM</p>',
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
                pred_val = (predictions or {}).get("predictions", {}).get(city)

                if reading is None:
                    st.error(f"**{city}** — Data unavailable")
                    continue

                current_pm = reading.get("pm2_5")
                current_aqi = reading.get("aqi")

                cat_curr, color_curr, emoji_curr = pm25_category(current_pm)
                cat_pred, color_pred, emoji_pred = pm25_category(pred_val)

                st.markdown(f"""
                <div class="metric-card" style="border-left-color: {color_curr};">
                    <h3 style="margin:0 0 0.5rem 0;color:{color_curr};">{emoji_curr} {city}</h3>
                    <div style="display:flex; justify-content:space-between;">
                        <div>
                            <div style="color:#7f8c8d; font-size:0.8rem;">Current PM2.5</div>
                            <div style="font-size:1.8rem; font-weight:700; color:{color_curr};">
                                {current_pm or 'N/A'}
                            </div>
                            <div style="font-size:0.85rem; color:{color_curr};">{cat_curr}</div>
                        </div>
                        <div>
                            <div style="color:#7f8c8d; font-size:0.8rem;">Tomorrow PM2.5</div>
                            <div style="font-size:1.8rem; font-weight:700; color:{color_pred};">
                                {pred_val or 'N/A'}
                            </div>
                            <div style="font-size:0.85rem; color:{color_pred};">{cat_pred}</div>
                        </div>
                    </div>
                    <div style="margin-top:0.5rem; color:#7f8c8d; font-size:0.75rem;">
                        AQI: {current_aqi or 'N/A'} | 
                        Temp: {reading.get('temperature', 'N/A')}°C | 
                        Wind: {reading.get('wind_speed', 'N/A')} m/s
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
            city_preds = preds_df[preds_df["city"] == city]
            if city_preds.empty:
                continue

            st.markdown(f"#### {city}")

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

        # Overall metrics
        st.markdown("---")
        st.markdown("#### Test Set Metrics")
        if os.path.exists("outputs/city_metrics.csv"):
            metrics_df = pd.read_csv("outputs/city_metrics.csv")
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        if os.path.exists("outputs/model_comparison.csv"):
            st.markdown("#### Model Comparison")
            comp_df = pd.read_csv("outputs/model_comparison.csv")
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
    else:
        st.info("No prediction data found. Run the training pipeline first.")


# ═══════════════════════════════════════════════════════════════
# Tab 3: Explainability
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Model Explainability (XAI)")

    xai_tab1, xai_tab2, xai_tab3, xai_tab4 = st.tabs([
        "SHAP Summary", "City Influence", "Spike Events", "Temporal Lags",
    ])

    with xai_tab1:
        st.markdown("#### Feature Importance (SHAP Beeswarm Plot)")
        if os.path.exists("outputs/shap_summary.png"):
            st_image("outputs/shap_summary.png", use_container_width=True)
        else:
            st.info("Run `05_explain.py` to generate SHAP plots.")

        if os.path.exists("outputs/shap_dependence.png"):
            st.markdown("#### SHAP Dependence: PM2.5 Lag-1 × Wind Speed")
            st_image("outputs/shap_dependence.png", use_container_width=True)

    with xai_tab2:
        st.markdown("#### Cross-City PM2.5 Influence Matrix")
        if os.path.exists("outputs/gnn_edge_importance.png"):
            st_image("outputs/gnn_edge_importance.png", use_container_width=True)
        else:
            st.info("Run `05_explain.py` to generate the influence heatmap.")

    with xai_tab3:
        st.markdown("#### SHAP Waterfall for Top PM2.5 Spike Events")
        for i in range(1, 4):
            path = f"outputs/shap_waterfall_{i}.png"
            if os.path.exists(path):
                st_image(path, caption=f"Spike Event #{i}", use_container_width=True)
        if not any(os.path.exists(f"outputs/shap_waterfall_{i}.png") for i in range(1, 4)):
            st.info("Run `05_explain.py` to generate waterfall plots.")

    with xai_tab4:
        st.markdown("#### Temporal Lag Importance")
        if os.path.exists("outputs/temporal_lag_importance.png"):
            st_image("outputs/temporal_lag_importance.png", use_container_width=True)
        else:
            st.info("Run `05_explain.py` to generate lag importance chart.")


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
        AirMind Fusion Model
        ═══════════════════════════

        ┌─────────────────────┐
        │  GNN Branch         │
        │  GraphSAGE-style    │
        │  2 conv layers      │
        │  16 → 64 → 64      │
        │  + Adjacency matrix │
        └──────────┬──────────┘
                   │ 64
        ┌──────────┴──────────┐
        │  Temporal Branch    │
        │  2-layer LSTM       │
        │  hidden = 128       │
        │  7-day sequence     │
        └──────────┬──────────┘
                   │ 128
        ┌──────────┴──────────┐
        │  Met Branch (MLP)   │
        │  8 → 32 → 32       │
        │  Calendar + weather │
        └──────────┬──────────┘
                   │ 32
        ┌──────────┴──────────┐
        │  Fusion MLP Head    │
        │  224 → 64 → 1      │
        │  Huber Loss (δ=10)  │
        └─────────────────────┘
        ```
        """)

    with col2:
        st.markdown("#### Training Configuration")
        st.markdown("""
        | Parameter | Value |
        |-----------|-------|
        | Optimizer | AdamW |
        | Learning Rate | 1e-3 |
        | Weight Decay | 1e-4 |
        | Batch Size | 64 |
        | Max Epochs | 100 |
        | Early Stopping | Patience 15 |
        | Gradient Clipping | max_norm=1.0 |
        | Loss | HuberLoss (δ=10) |
        | LR Scheduler | ReduceLROnPlateau |
        | Seed | 42 |
        """)

    st.markdown("---")
    st.markdown("#### Training Curves")
    if os.path.exists("outputs/training_curves.png"):
        st_image("outputs/training_curves.png", use_container_width=True)
    else:
        st.info("Training curves will appear after running `04_train.py`.")

    st.markdown("---")
    st.markdown("#### Data Provenance")
    st.markdown("""
    | Source | Details |
    |--------|---------|
    | **Historical Data** | `merged.csv` — 5 Indian cities, 2020–2024 |
    | **Cities** | Delhi, Bengaluru, Kolkata, Hyderabad |
    | **Features** | 14 raw + 16 engineered = 30 total features |
    | **Target** | Next-day PM2.5 (shifted by -1 day) |
    | **Live API** | WAQI (World Air Quality Index) |
    | **Train/Val/Test** | Before Oct 2022 / Oct 2022–Jan 2023 / Feb 2023 onwards |
    | **Research Gaps** | Cross-city dependencies (GNN), missing data bias (MCAR/MNAR) |
    """)
