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
from pathlib import Path
from typing import Optional, Dict, Any, List

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
CITIES = ["Delhi", "Mumbai", "Bengaluru", "Kolkata", "Hyderabad"]

CITY_ACCENT = {
    "Delhi":     "#ef4444",
    "Mumbai":    "#f59e0b",
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


def _safe_float(v):
    try:
        if v is None:
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _discover_output_images(outputs_dir: str = "outputs") -> List[str]:
    root = Path(outputs_dir)
    if not root.exists():
        return []
    return sorted([str(p) for p in root.glob("*.png")])


def _load_predictions_csv(path: str = "outputs/predictions.csv") -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def _compute_metrics(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Compute metrics from a predictions dataframe.

    Expected columns: city, actual_pm2_5, predicted_pm2_5 (date optional)
    """
    required = {"city", "actual_pm2_5", "predicted_pm2_5"}
    if df is None or df.empty or not required.issubset(df.columns):
        return None

    rows = []
    for city, g in df.groupby("city"):
        y = g["actual_pm2_5"].astype(float).to_numpy()
        p = g["predicted_pm2_5"].astype(float).to_numpy()
        if len(y) < 2:
            continue
        err = p - y
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err**2)))
        denom = float(np.sum((y - float(np.mean(y))) ** 2))
        r2 = float(1.0 - (np.sum(err**2) / denom)) if denom > 0 else 0.0
        mape = float(np.mean(np.abs(err) / np.maximum(np.abs(y), 1e-6)) * 100.0)
        rows.append({"city": city, "MAE": mae, "RMSE": rmse, "R2": r2, "MAPE_%": mape, "n": int(len(y))})

    m = pd.DataFrame(rows).sort_values("R2", ascending=False)
    if not m.empty:
        # Global row
        y = df["actual_pm2_5"].astype(float).to_numpy()
        p = df["predicted_pm2_5"].astype(float).to_numpy()
        err = p - y
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err**2)))
        denom = float(np.sum((y - float(np.mean(y))) ** 2))
        r2 = float(1.0 - (np.sum(err**2) / denom)) if denom > 0 else 0.0
        mape = float(np.mean(np.abs(err) / np.maximum(np.abs(y), 1e-6)) * 100.0)
        m = pd.concat(
            [pd.DataFrame([{"city": "__overall__", "MAE": mae, "RMSE": rmse, "R2": r2, "MAPE_%": mape, "n": int(len(y))}]), m],
            ignore_index=True,
        )
    return m


def _error_by_bins(df: pd.DataFrame, n_bins: int = 8) -> Optional[pd.DataFrame]:
    required = {"actual_pm2_5", "predicted_pm2_5"}
    if df is None or df.empty or not required.issubset(df.columns):
        return None

    tmp = df.copy()
    tmp["actual_pm2_5"] = tmp["actual_pm2_5"].astype(float)
    tmp["predicted_pm2_5"] = tmp["predicted_pm2_5"].astype(float)
    tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna(subset=["actual_pm2_5", "predicted_pm2_5"])
    if tmp.empty:
        return None

    tmp["abs_err"] = np.abs(tmp["predicted_pm2_5"] - tmp["actual_pm2_5"])
    try:
        tmp["bin"] = pd.qcut(tmp["actual_pm2_5"], q=n_bins, duplicates="drop")
    except Exception:
        return None

    out = (
        tmp.groupby("bin", as_index=False)
        .agg(n=("abs_err", "size"), mean_abs_err=("abs_err", "mean"), median_abs_err=("abs_err", "median"))
    )
    return out


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
        # Snapshot table (quick 'current' + 'tomorrow' view)
        snap_rows = []
        for city in CITIES:
            r = (readings or {}).get(city) or {}
            p = ((predictions or {}).get("predictions") or {}).get(city) if predictions else None
            snap_rows.append(
                {
                    "city": city,
                    "current_pm2_5": _safe_float(r.get("pm2_5")),
                    "tomorrow_pm2_5": _safe_float(p),
                    "aqi": _safe_float(r.get("aqi")),
                    "temp_c": _safe_float(r.get("temperature")),
                    "wind_mps": _safe_float(r.get("wind_speed")),
                }
            )
        snap_df = pd.DataFrame(snap_rows)
        st.markdown(
            '<p class="section-header" style="margin-top:0.25rem;">Current snapshot</p>',
            unsafe_allow_html=True,
        )
        st.dataframe(snap_df, use_container_width=True, hide_index=True)
        if predictions is None:
            st.caption("Forecast not run yet. Click **Run Forecast** to generate tomorrow's PM2.5.")

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
    preds_df = _load_predictions_csv("outputs/predictions.csv")
    if preds_df is not None and not preds_df.empty:

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

        available_cities = (
            sorted([c for c in preds_df["city"].dropna().unique().tolist()])
            if "city" in preds_df.columns
            else []
        )
        if not available_cities:
            st.warning("`outputs/predictions.csv` exists, but it doesn't have a `city` column.")
        else:
            f1, f2, f3 = st.columns([2, 2, 3])
            with f1:
                selected_cities = st.multiselect(
                    "Cities",
                    options=available_cities,
                    default=[c for c in CITIES if c in available_cities] or available_cities,
                )
            with f2:
                max_points = st.number_input(
                    "Max points per city",
                    min_value=50,
                    max_value=5000,
                    value=600,
                    step=50,
                )
            with f3:
                date_range = None
                if "date" in preds_df.columns and preds_df["date"].notna().any():
                    dmin = preds_df["date"].min().date()
                    dmax = preds_df["date"].max().date()
                    date_range = st.date_input("Date range", value=(dmin, dmax))

            plot_df = preds_df.copy()
            if selected_cities:
                plot_df = plot_df[plot_df["city"].isin(selected_cities)].copy()
            if (
                date_range
                and isinstance(date_range, (list, tuple))
                and len(date_range) == 2
                and "date" in plot_df.columns
            ):
                start_d, end_d = date_range
                start_ts = pd.Timestamp(start_d)
                end_ts = pd.Timestamp(end_d) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
                plot_df = plot_df[(plot_df["date"] >= start_ts) & (plot_df["date"] <= end_ts)].copy()

            required_cols = {"city", "actual_pm2_5", "predicted_pm2_5"}
            missing = [c for c in required_cols if c not in plot_df.columns]
            if missing:
                st.warning(f"Missing columns in predictions file: {missing}")
            else:
                for city in selected_cities:
                    city_preds = plot_df[plot_df["city"] == city].copy()
                    if city_preds.empty:
                        continue

                    if "date" in city_preds.columns and city_preds["date"].notna().any():
                        city_preds = city_preds.sort_values("date")
                        x = city_preds["date"]
                        x_label = "Date"
                    else:
                        city_preds = city_preds.reset_index(drop=True)
                        x = city_preds.index
                        x_label = "Index"

                    if max_points and len(city_preds) > int(max_points):
                        city_preds = city_preds.tail(int(max_points)).copy()
                        if isinstance(x, pd.Series):
                            x = x.tail(int(max_points))
                        else:
                            x = range(len(city_preds))

                    accent = CITY_ACCENT.get(city, "#71717a")

                    fig, ax = plt.subplots(figsize=(12, 3))
                    ax.plot(
                        x,
                        city_preds["actual_pm2_5"],
                        label="Actual",
                        alpha=0.8,
                        color="#71717a",
                        linewidth=1.2,
                    )
                    ax.plot(
                        x,
                        city_preds["predicted_pm2_5"],
                        label="Predicted",
                        alpha=0.9,
                        color=accent,
                        linewidth=1.5,
                        linestyle="--",
                    )
                    try:
                        ax.fill_between(
                            x,
                            city_preds["actual_pm2_5"].to_numpy(),
                            city_preds["predicted_pm2_5"].to_numpy(),
                            alpha=0.06,
                            color=accent,
                        )
                    except Exception:
                        pass
                    ax.set_ylabel("PM2.5", fontsize=8)
                    ax.set_xlabel(x_label, fontsize=8)
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

        with st.expander("Preview historical predictions data", expanded=False):
            st.dataframe(preds_df.head(200), use_container_width=True, hide_index=True)

        # Metrics + diagnostics
        st.markdown('<p class="section-header" style="font-size:1.1rem;">Metrics</p>',
                    unsafe_allow_html=True)

        cm1, cm2 = st.columns([2, 1])
        with cm1:
            # Prefer training-produced metrics when present; otherwise compute directly.
            if os.path.exists("outputs/per_city_metrics.csv"):
                mdf = pd.read_csv("outputs/per_city_metrics.csv")
            else:
                mdf = _compute_metrics(preds_df)

            if mdf is not None and not mdf.empty:
                st.dataframe(mdf, use_container_width=True, hide_index=True)
            else:
                st.info("Metrics not available.")

        with cm2:
            if os.path.exists("outputs/model_comparison.csv"):
                st.markdown('<p class="section-header" style="font-size:1.1rem;">Model Comparison</p>',
                            unsafe_allow_html=True)
                st.dataframe(pd.read_csv("outputs/model_comparison.csv"),
                             use_container_width=True, hide_index=True)

        st.markdown('<p class="section-header" style="font-size:1.1rem; margin-top:1.25rem;">Diagnostics</p>',
                    unsafe_allow_html=True)

        d1, d2 = st.columns(2)
        with d1:
            st.caption("How well predictions track the 45° line (all cities)")
            if {"actual_pm2_5", "predicted_pm2_5"}.issubset(preds_df.columns):
                tmp = preds_df[["actual_pm2_5", "predicted_pm2_5"]].copy().dropna()
                tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna()
                if len(tmp) > 2:
                    fig, ax = plt.subplots(figsize=(5.5, 4.5))
                    ax.scatter(tmp["actual_pm2_5"], tmp["predicted_pm2_5"], s=8, alpha=0.15, color="#a1a1aa")
                    mn = float(np.nanmin(tmp[["actual_pm2_5", "predicted_pm2_5"]].to_numpy()))
                    mx = float(np.nanmax(tmp[["actual_pm2_5", "predicted_pm2_5"]].to_numpy()))
                    ax.plot([mn, mx], [mn, mx], linestyle="--", linewidth=1.2, color="#22c55e", alpha=0.8)
                    ax.set_xlabel("Actual")
                    ax.set_ylabel("Predicted")
                    ax.grid(True)
                    ax.set_title("Actual vs Predicted")
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close()

        with d2:
            st.caption("Residual distribution (all cities)")
            if {"actual_pm2_5", "predicted_pm2_5"}.issubset(preds_df.columns):
                tmp = preds_df[["actual_pm2_5", "predicted_pm2_5"]].copy().dropna()
                tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna()
                if len(tmp) > 2:
                    res = (tmp["predicted_pm2_5"] - tmp["actual_pm2_5"]).astype(float).to_numpy()
                    fig, ax = plt.subplots(figsize=(5.5, 4.5))
                    ax.hist(res, bins=60, color="#38bdf8", alpha=0.55)
                    ax.axvline(0, color="#e5e7eb", linewidth=1)
                    ax.set_xlabel("Residual (Pred - Actual)")
                    ax.set_ylabel("Count")
                    ax.grid(True)
                    ax.set_title("Residuals")
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close()

        st.caption("Error by actual PM2.5 quantile (higher bars = model struggles more)")
        bin_df = _error_by_bins(preds_df)
        if bin_df is not None and not bin_df.empty:
            fig, ax = plt.subplots(figsize=(12, 3.5))
            x = np.arange(len(bin_df))
            ax.bar(x, bin_df["mean_abs_err"], color="#f59e0b", alpha=0.65)
            ax.set_xticks(x)
            ax.set_xticklabels([str(b) for b in bin_df["bin"].tolist()], rotation=15, ha="right", fontsize=8)
            ax.set_ylabel("Mean |Error|")
            ax.grid(True, axis="y")
            ax.set_title("Mean absolute error by PM2.5 bin")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()
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

    all_pngs = _discover_output_images("outputs")
    png_by_name = {Path(p).name: p for p in all_pngs}

    with x1:
        path = png_by_name.get("shap_summary.png")
        if path and os.path.exists(path):
            st_image(path, use_container_width=True)
        else:
            st.info("No SHAP summary plot found yet. Run `src/04_explain.py`.")

        extras = [p for p in all_pngs if "shap_" in Path(p).name and p != path]
        if extras:
            with st.expander("Other SHAP plots", expanded=False):
                for p in extras:
                    st_image(p, caption=Path(p).name, use_container_width=True)

    with x2:
        st.caption(
            "Mean |SHAP| of neighbor PM2.5 lag features. "
            "Higher values indicate stronger cross-city influence."
        )
        path = png_by_name.get("cross_city_influence.png")
        if path and os.path.exists(path):
            st_image(path, use_container_width=True)
        else:
            st.info("Run 04_explain.py to generate heatmap.")

    with x3:
        spike_paths = [p for p in all_pngs if "shap_waterfall_spike" in Path(p).name]
        if spike_paths:
            for p in spike_paths:
                st_image(p, caption=Path(p).stem, use_container_width=True)
        else:
            st.info("No spike waterfall plots found yet. Run `src/04_explain.py`.")

    with x4:
        st.caption("PM2.5 lag feature contributions — 1, 2, 3, 7 days back.")
        path = png_by_name.get("lag_importance.png")
        if path and os.path.exists(path):
            st_image(path, use_container_width=True)
        else:
            st.info("Run 04_explain.py to generate lag chart.")

        lag_like = [p for p in all_pngs if "lag" in Path(p).name.lower() and p != path]
        if lag_like:
            with st.expander("Other lag-related plots", expanded=False):
                for p in lag_like:
                    st_image(p, caption=Path(p).name, use_container_width=True)
