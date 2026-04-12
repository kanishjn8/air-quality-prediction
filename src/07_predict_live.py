"""
Step 07 — Live Prediction
Input:  data/live/latest_reading.json, models/best_model.pt,
    models/feature_scaler.pkl, models/temporal_scaler.pkl,
    models/feature_columns.json, models/city_last7.json,
    data/processed/adj_matrix.npy
Output: Predicted next-day PM2.5 for each city (printed + saved to outputs/)

Builds feature vectors from live readings, handles missing fields via
imputation, and runs the trained model for inference.

This follows plan.md Option A: inference reconstructs lag/rolling features
from a saved per-city last-7 window (no need to load features.parquet).
"""

import os
import sys
import json
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import joblib


def aqi_to_training_scale(aqi_val):
    """Convert WAQI AQI (0–500-ish) to the project's historical 1–5 scale."""
    if aqi_val is None:
        return None
    try:
        aqi = float(aqi_val)
    except (TypeError, ValueError):
        return None
    if aqi <= 50:
        return 1
    if aqi <= 100:
        return 2
    if aqi <= 200:
        return 3
    if aqi <= 300:
        return 4
    return 5


def sanitize_live_value(name: str, value):
    """Sanitize a few live fields that are commonly out-of-distribution."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None

    if name == "pressure":
        return None if (v < 850 or v > 1100) else v
    if name == "wind_speed":
        return None if (v < 0 or v > 40) else v
    if name == "temperature":
        return max(-5.0, min(55.0, v))
    return v


def _safe_float(x):
    try:
        if x is None:
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def co_to_training_units(co_live):
    """Heuristic: WAQI CO is often reported in mg/m³ while training data looks like µg/m³.

    Training `co` median is ~881 and can go to 20k+. Live values we see are ~1–4.
    If CO looks like a small mg/m³ number, convert to µg/m³ by *1000.

    This keeps us from feeding extreme negative z-scores into the model.

    Returns: (co_converted, note)
    """
    co = _safe_float(co_live)
    if co is None:
        return None, "missing"

    # If already in a training-like magnitude, keep as-is.
    if co >= 50:
        return co, "as_is"

    # If it's a small positive number (typical mg/m³), convert.
    if 0 < co < 50:
        return co * 1000.0, "mg_to_ug"

    return co, "as_is"


def normalize_lag_window_units(lag_window, reading):
        """Normalize units inside the lag window for consistency with model training.

        Why:
            - When lag_window comes from `models/city_last7.json` (historical tail), it may
                be in different units than the live API for certain pollutants.
            - If we only convert the *current* live CO but keep previous 6 days CO in a
                different unit system, rolling/sequence inputs become inconsistent.

        Currently we normalize CO across the whole window using the same heuristic used
        for the live reading.

        Returns: (new_lag_window, debug)
        """
        out = [dict(r) for r in lag_window]
        dbg = {"co": {"converted_days": 0, "note": None}}

        # Pick heuristic based on today's live CO if available; otherwise default.
        co_today, co_note = co_to_training_units(reading.get("co"))
        dbg["co"]["note"] = co_note

        # If conversion isn't applicable, do nothing.
        if co_note != "mg_to_ug":
                return out, dbg

        for r in out:
                co = _safe_float(r.get("co"))
                if co is None:
                        continue
                # Apply same mg->ug conversion
                if 0 < co < 50:
                        r["co"] = co * 1000.0
                        dbg["co"]["converted_days"] += 1

        return out, dbg


def clip_ood_feature(feature_dict, feature_cols, scaler, z_max=8.0):
    """Clip out-of-distribution features using training scaler statistics.

    For each numeric feature present in the scaler, compute z = (x-mean)/scale.
    If |z| > z_max, replace x with mean +/- z_max*scale.

    Returns: (new_feature_dict, debug_info)
    """
    debug = {"clipped": {}, "z_max": z_max}
    out = dict(feature_dict)

    for c in feature_cols:
        if c not in out:
            continue
        if c == "city_idx":
            continue

        x = _safe_float(out.get(c))
        if x is None:
            continue

        try:
            idx = feature_cols.index(c)
        except ValueError:
            continue

        mean = float(scaler.mean_[idx])
        scale = float(scaler.scale_[idx])
        if scale <= 1e-12:
            continue

        z = (x - mean) / scale
        if abs(z) > z_max:
            new_x = mean + (z_max if z > 0 else -z_max) * scale
            out[c] = float(new_x)
            debug["clipped"][c] = {"old": float(x), "new": float(new_x), "z": float(z)}

    return out, debug

# ────────────────────────────────────────────────────────────────
# Reproducibility
# ────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ────────────────────────────────────────────────────────────────
# Paths
# ────────────────────────────────────────────────────────────────
LIVE_PATH     = "data/live/latest_reading.json"
MODEL_PATH    = "models/best_model.pt"
SCALER_PATH   = "models/feature_scaler.pkl"
TEMPORAL_SCALER_PATH = "models/temporal_scaler.pkl"
FEATURE_COLS_PATH = "models/feature_columns.json"
CITY_LAST7_PATH = "models/city_last7.json"
LIVE_HISTORY_PATH = "data/live/live_history.json"
ADJ_PATH      = "data/processed/adj_matrix.npy"

DEVICE = torch.device("cpu")

print("=" * 60)
print("Step 07 — Live Prediction")
print("=" * 60)

# ═══════════════════════════════════════════════════════════════
# Load resources
# ═══════════════════════════════════════════════════════════════
if not os.path.exists(LIVE_PATH):
    print(f"ERROR: {LIVE_PATH} not found. Run 06_live_ingest.py first.")
    sys.exit(1)

with open(LIVE_PATH) as f:
    live = json.load(f)

scaler = joblib.load(SCALER_PATH)
temporal_scaler = joblib.load(TEMPORAL_SCALER_PATH)
adj = np.load(ADJ_PATH)

if not os.path.exists(FEATURE_COLS_PATH):
    print(f"ERROR: {FEATURE_COLS_PATH} not found. Run 02_feature_engineering.py first.")
    sys.exit(1)
if not os.path.exists(CITY_LAST7_PATH):
    print(f"ERROR: {CITY_LAST7_PATH} not found. Run 04_train.py first.")
    sys.exit(1)

with open(FEATURE_COLS_PATH) as f:
    FEATURE_COLS = json.load(f)

with open(CITY_LAST7_PATH) as f:
    city_last7 = json.load(f)

# Optional: rolling live history buffer (prefer for inference once available)
live_history = {}
if os.path.exists(LIVE_HISTORY_PATH):
    try:
        with open(LIVE_HISTORY_PATH) as f:
            live_history = json.load(f) or {}
    except Exception:
        live_history = {}

CITY_TO_IDX = {"Delhi": 0, "Bengaluru": 1, "Kolkata": 2, "Hyderabad": 3}

# Temporal sequence uses pm2_5 and aqi (scaled via temporal_scaler),
# plus temperature, wind_speed, pressure (scaled via feature_scaler).
TEMPORAL_RAW_COLS = ["pm2_5", "aqi", "temperature", "wind_speed", "pressure"]
MET_RAW_COLS = ["temperature", "wind_speed", "pressure", "rainfall",
                "dayofweek", "month", "crop_burning_season", "monsoon_season"]

SEQ_LEN = 7

# Inference-time defaults (Option A in plan): missing live fields are filled with
# values from the saved historical lag window.

# ═══════════════════════════════════════════════════════════════
# Import model architecture (same as 04_train.py)
# ═══════════════════════════════════════════════════════════════
# We import the model class inline to avoid circular dependencies
import torch.nn as nn

class GraphBranch(nn.Module):
    def __init__(self, num_cities=4, embed_dim=16, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.city_embed = nn.Embedding(num_cities, embed_dim)
        self.conv1_w = nn.Linear(embed_dim, hidden_dim)
        self.conv2_w = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, city_idx, adj_matrix):
        all_embed = self.city_embed.weight
        deg = adj_matrix.sum(dim=1, keepdim=True).clamp(min=1e-6)
        adj_norm = adj_matrix / deg
        h = adj_norm @ all_embed
        h = self.relu(self.conv1_w(h))
        h = self.dropout(h)
        h = adj_norm @ h
        h = self.relu(self.conv2_w(h))
        h = self.dropout(h)
        return h[city_idx]


class TemporalBranch(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True, dropout=dropout)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]


class MetBranch(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class FeatureBranch(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class AirMindModel(nn.Module):
    def __init__(self, num_cities=4, num_temporal_feats=5, num_met_feats=8,
                 num_features=30, gnn_hidden=64, lstm_hidden=128,
                 met_hidden=32, feat_hidden=64):
        super().__init__()
        self.gnn_branch = GraphBranch(num_cities, embed_dim=16, hidden_dim=gnn_hidden)
        self.temporal_branch = TemporalBranch(num_temporal_feats, hidden_dim=lstm_hidden)
        self.met_branch = MetBranch(num_met_feats, hidden_dim=met_hidden)
        self.feat_branch = FeatureBranch(num_features, hidden_dim=feat_hidden)
        fusion_dim = gnn_hidden + lstm_hidden + met_hidden + feat_hidden
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, features, temporal_seq, met_features, city_idx, adj_matrix):
        gnn_out = self.gnn_branch(city_idx, adj_matrix)
        lstm_out = self.temporal_branch(temporal_seq)
        met_out = self.met_branch(met_features)
        feat_out = self.feat_branch(features)
        fused = torch.cat([gnn_out, lstm_out, met_out, feat_out], dim=1)
        return self.fusion_head(fused).squeeze(-1)


# ═══════════════════════════════════════════════════════════════
# Load trained model
# ═══════════════════════════════════════════════════════════════
TEMPORAL_SCALED_COLS = [f"{c}_scaled" for c in TEMPORAL_RAW_COLS]

model = AirMindModel(
    num_cities=4,
    num_temporal_feats=len(TEMPORAL_RAW_COLS),
    num_met_feats=len(MET_RAW_COLS),
    num_features=len(FEATURE_COLS),
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print(f"\n  ✓ Model loaded from {MODEL_PATH}")

adj_tensor = torch.tensor(adj, dtype=torch.float32)


# ═══════════════════════════════════════════════════════════════
# Build predictions
# ═══════════════════════════════════════════════════════════════
def pm25_category(val):
    """Classify PM2.5 into India NAQI categories."""
    if val <= 30:   return "Good", "🟢"
    elif val <= 60: return "Satisfactory", "🟡"
    elif val <= 90: return "Moderate", "🟠"
    elif val <= 120: return "Poor", "🔴"
    elif val <= 250: return "Very Poor", "🟣"
    else:            return "Severe", "⚫"


def _lag_window_median(lag_window, key, fallback=0.0):
    """Compute median of a key across the lag window rows (ignoring None/0)."""
    vals = []
    for r in lag_window:
        v = _safe_float(r.get(key))
        if v is not None and v > 0:
            vals.append(v)
    return float(np.median(vals)) if vals else fallback


def _scale_temporal_value(col_name, raw_value):
    """Scale a temporal column value using the appropriate scaler.

    pm2_5 and aqi use temporal_scaler; others use feature_scaler.
    """
    TEMPORAL_EXTRA = ["pm2_5", "aqi"]
    if col_name in TEMPORAL_EXTRA:
        idx = TEMPORAL_EXTRA.index(col_name)
        return (raw_value - temporal_scaler.mean_[idx]) / max(temporal_scaler.scale_[idx], 1e-8)
    else:
        # This column is in FEATURE_COLS
        idx = FEATURE_COLS.index(col_name)
        return (raw_value - scaler.mean_[idx]) / max(scaler.scale_[idx], 1e-8)


print("\n" + "─" * 60)
print("Predictions for tomorrow's PM2.5")
print("─" * 60)

predictions = {}
debug_by_city = {}

for city, reading in live.items():
    if reading is None:
        predictions[city] = None
        print(f"  ✗ {city:<12s}  No live data available")
        continue

    city_idx = CITY_TO_IDX.get(city)
    if city_idx is None:
        continue

    # Choose lag window source:
    # 1) Prefer recent live-history (if we have enough collected).
    # 2) Otherwise fall back to the static training-derived window.
    if city in live_history and isinstance(live_history[city], list) and len(live_history[city]) >= SEQ_LEN:
        lag_window = list(live_history[city])[-SEQ_LEN:]
    else:
        if city not in city_last7 or len(city_last7[city]) < SEQ_LEN:
            print(f"  ✗ {city:<12s}  Missing lag window (need live history or models/city_last7.json)")
            predictions[city] = None
            continue
        lag_window = list(city_last7[city])  # 7 dicts, oldest first

    debug_city = {
        "used_live_history": bool(city in live_history and isinstance(live_history[city], list) and len(live_history[city]) >= SEQ_LEN),
        "co_unit": None,
        "clipped": {},
    }

    # Normalize units across the lag window (keeps sequences/rollings consistent)
    lag_window, units_dbg = normalize_lag_window_units(lag_window, reading)

    # Sanitize live values and overwrite the latest row in the window
    live_row = dict(lag_window[-1])

    # Live → training scale conversions
    aqi_train = reading.get("aqi")
    if aqi_train is None and reading.get("aqi_raw") is not None:
        aqi_train = aqi_to_training_scale(reading.get("aqi_raw"))
    if isinstance(aqi_train, (int, float)) and aqi_train > 5:
        aqi_train = aqi_to_training_scale(aqi_train)

    live_row["aqi"] = aqi_train if aqi_train is not None else live_row.get("aqi")

    # CO unit normalization (heuristic)
    co_conv, co_note = co_to_training_units(reading.get("co"))
    debug_city["co_unit"] = co_note
    if co_conv is not None:
        live_row["co"] = co_conv
    debug_city["co_window"] = units_dbg.get("co")

    # NOTE: skip 'co' here because we normalized it above via `co_to_training_units()`
    for k in ["pm2_5", "pm10", "no2", "o3", "so2", "no", "nh3", "rainfall"]:
        if k in reading and reading.get(k) is not None:
            try:
                live_row[k] = float(reading.get(k))
            except (TypeError, ValueError):
                pass

    pressure = sanitize_live_value("pressure", reading.get("pressure"))
    wind_speed = sanitize_live_value("wind_speed", reading.get("wind_speed"))
    temperature = sanitize_live_value("temperature", reading.get("temperature"))
    if pressure is not None:
        live_row["pressure"] = pressure
    if wind_speed is not None:
        live_row["wind_speed"] = wind_speed
    if temperature is not None:
        live_row["temperature"] = temperature

    lag_window[-1] = live_row

    # ──────────────────────────────────────────────────────────
    # Build the lag/rolling series
    # ──────────────────────────────────────────────────────────
    # The lag window has 7 rows [t-7, t-6, ..., t-1] where t-1 is the
    # last historical day. The current live reading has been merged into
    # lag_window[-1]. For predicting TOMORROW, the live reading IS "today"
    # and should be treated as lag1 (1 day before the prediction target).
    #
    # Lag semantics at prediction time (predicting t+1):
    #   lag1 = today = live reading   → lag_window[-1] (already updated)
    #   lag2 = yesterday              → lag_window[-2]
    #   lag3 = two days ago           → lag_window[-3]
    #   lag7 = six days ago           → lag_window[-7] = lag_window[0]
    pm_series = [float(r.get("pm2_5", 0.0) or 0.0) for r in lag_window]
    aqi_series = [float(r.get("aqi", 0.0) or 0.0) for r in lag_window]

    now = datetime.now()

    # For fields that might be None in live data (no, nh3, rainfall),
    # use the lag window's median as imputation instead of 0
    no_val = _safe_float(live_row.get("no"))
    if no_val is None:
        no_val = _lag_window_median(lag_window, "no", fallback=0.0)

    nh3_val = _safe_float(live_row.get("nh3"))
    if nh3_val is None:
        nh3_val = _lag_window_median(lag_window, "nh3", fallback=0.0)

    rainfall_val = _safe_float(live_row.get("rainfall"))
    if rainfall_val is None:
        rainfall_val = _lag_window_median(lag_window, "rainfall", fallback=0.0)

    # Build feature vector (must match FEATURE_COLS exactly)
    # NOTE: pm2_5 and aqi at t=0 are NOT in FEATURE_COLS (to prevent leakage).
    feature_dict = {
        "city_idx":       city_idx,
        "co":             float(live_row.get("co", 0) or 0),
        "no":             no_val,
        "no2":            float(live_row.get("no2", 0) or 0),
        "o3":             float(live_row.get("o3", 0) or 0),
        "so2":            float(live_row.get("so2", 0) or 0),
        "pm10":           float(live_row.get("pm10", 0) or 0),
        "nh3":            nh3_val,
        "temperature":    float(live_row.get("temperature", 25) or 25),
        "wind_speed":     float(live_row.get("wind_speed", 5) or 5),
        "rainfall":       rainfall_val,
        "pressure":       float(live_row.get("pressure", 1013) or 1013),
        "dayofweek":      now.weekday(),
        "month":          now.month,
        "is_weekend":     1 if now.weekday() >= 5 else 0,
        "quarter":        (now.month - 1) // 3 + 1,
        "crop_burning_season": 1 if now.month in [10, 11] else 0,
        "monsoon_season": 1 if now.month in [6, 7, 8, 9] else 0,
    }

    # Lags & rolling from lag window
    # lag1 = today's reading (lag_window[-1]), lag2 = yesterday, etc.
    feature_dict["pm2_5_lag1"] = pm_series[-1]
    feature_dict["pm2_5_lag2"] = pm_series[-2]
    feature_dict["pm2_5_lag3"] = pm_series[-3]
    feature_dict["pm2_5_lag7"] = pm_series[0]  # oldest row in the 7-day window
    feature_dict["aqi_lag1"]   = aqi_series[-1]
    feature_dict["aqi_lag2"]   = aqi_series[-2]
    feature_dict["aqi_lag3"]   = aqi_series[-3]
    feature_dict["aqi_lag7"]   = aqi_series[0]

    # Rolling features computed from the lag window (shift(1) in training means
    # we exclude today and look at the last N days before today)
    feature_dict["pm2_5_roll3mean"] = float(np.mean(pm_series[-4:-1]))  # days t-3, t-2, t-1
    feature_dict["pm2_5_roll7mean"] = float(np.mean(pm_series[:-1]))    # all 6 days before today
    feature_dict["pm2_5_roll3std"]  = float(np.std(pm_series[-4:-1]))
    feature_dict["pm2_5_roll7std"]  = float(np.std(pm_series[:-1]))

    # Clip extreme OOD values based on training scaler stats
    feature_dict, clip_debug = clip_ood_feature(feature_dict, FEATURE_COLS, scaler, z_max=8.0)
    debug_city["clipped"] = clip_debug.get("clipped", {})

    # Scale features (use the saved column order + preserve feature names)
    feat_df = pd.DataFrame([{c: feature_dict[c] for c in FEATURE_COLS}], columns=FEATURE_COLS)
    feat_scaled = scaler.transform(feat_df)

    # Build temporal sequence from the lag window
    # pm2_5 and aqi are scaled using temporal_scaler;
    # temperature, wind_speed, pressure are scaled using feature_scaler.
    seq_raw = np.array(
        [[float(lag_window[i].get(c, 0.0) or 0.0) for c in TEMPORAL_RAW_COLS] for i in range(SEQ_LEN)],
        dtype=np.float32,
    )
    seq_scaled = np.zeros_like(seq_raw)
    for t in range(SEQ_LEN):
        for fi, col_name in enumerate(TEMPORAL_RAW_COLS):
            seq_scaled[t, fi] = _scale_temporal_value(col_name, seq_raw[t, fi])

    # Met features (scaled)
    met_indices = [FEATURE_COLS.index(c) for c in MET_RAW_COLS]
    met_scaled = feat_scaled[0, met_indices]

    # Inference
    feat_tensor = torch.tensor(feat_scaled, dtype=torch.float32)
    seq_tensor  = torch.tensor(seq_scaled, dtype=torch.float32).unsqueeze(0)
    met_tensor  = torch.tensor(met_scaled, dtype=torch.float32).unsqueeze(0)
    cidx_tensor = torch.tensor([city_idx], dtype=torch.long)

    with torch.no_grad():
        pred = model(feat_tensor, seq_tensor, met_tensor, cidx_tensor, adj_tensor).item()

    # Model predicts in log1p space — invert to get real PM2.5
    pred = float(np.expm1(pred))
    pred = max(0, round(pred, 2))  # PM2.5 can't be negative
    predictions[city] = pred

    current_pm = reading.get("pm2_5", "N/A")
    current_pm_str = "N/A" if current_pm is None else str(current_pm)
    cat, emoji = pm25_category(pred)
    print(f"  {emoji} {city:<12s}  Current={current_pm_str:<8s}  "
          f"Tomorrow={pred:<8.1f} µg/m³  ({cat})")

    debug_by_city[city] = debug_city


# ═══════════════════════════════════════════════════════════════
# Save predictions
# ═══════════════════════════════════════════════════════════════
output = {
    "generated_at": datetime.now().isoformat(),
    "predictions": predictions,
    "debug": debug_by_city,
}
with open("outputs/live_predictions.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\n{'=' * 60}")
print(f"✓ Live predictions saved to outputs/live_predictions.json")
print(f"{'=' * 60}")
