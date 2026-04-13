"""Step 06 — Live prediction (Plan vNext)

Inputs:
  - data/live/latest_reading.json
  - models/best_model.pkl
  - models/feature_scaler.pkl
  - models/feature_columns.json
  - models/city_last7.json

Output:
  - data/live/predictions.json

Approach:
  - Build the same feature row as training for each city, using:
      * today's live reading (raw)
      * lag/rolling features from city_last7.json (raw)
      * neighbor lag features from other cities in the same live batch
  - Scale with feature_scaler, then predict with trained XGBoost.

Notes:
  - For missing live readings, we fall back to last known values from city_last7.
"""

from __future__ import annotations

import json
import os
from math import atan2, cos, radians, sin, sqrt

import joblib
import numpy as np
import pandas as pd

LIVE_PATH = "data/live/latest_reading.json"
OUT_PATH = "data/live/predictions.json"

MODEL_PATH = "models/best_model.pkl"
SCALER_PATH = "models/feature_scaler.pkl"
FEATURE_COLS_PATH = "models/feature_columns.json"
CITY_LAST7_PATH = "models/city_last7.json"

CITIES = ["Delhi", "Bengaluru", "Kolkata", "Hyderabad"]
CITY_TO_IDX = {c: i for i, c in enumerate(CITIES)}

CITY_COORDS: dict[str, tuple[float, float]] = {
    "Delhi": (28.7041, 77.1025),
    "Bengaluru": (12.9716, 77.5946),
    "Kolkata": (22.5726, 88.3639),
    "Hyderabad": (17.3850, 78.4867),
}


def haversine_km(c1: tuple[float, float], c2: tuple[float, float]) -> float:
    r = 6371.0
    lat1, lon1 = map(radians, c1)
    lat2, lon2 = map(radians, c2)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return r * 2 * atan2(sqrt(a), sqrt(1 - a))


def _safe_float(x):
    try:
        if x is None:
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def _rolling_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _rolling_std(values: list[float]) -> float:
    return float(np.std(values)) if values else 0.0


def main() -> None:
    for p in [LIVE_PATH, MODEL_PATH, SCALER_PATH, FEATURE_COLS_PATH, CITY_LAST7_PATH]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing {p}. Run the training pipeline first.")

    os.makedirs("data/live", exist_ok=True)

    with open(LIVE_PATH) as f:
        live = json.load(f)

    with open(CITY_LAST7_PATH) as f:
        city_last7 = json.load(f)

    with open(FEATURE_COLS_PATH) as f:
        feature_cols = json.load(f)

    scaler = joblib.load(SCALER_PATH)
    model = joblib.load(MODEL_PATH)

    # Distance weights (same as Step 02)
    dist_weights: dict[tuple[str, str], float] = {}
    for c1 in CITIES:
        for c2 in CITIES:
            if c1 == c2:
                continue
            dist = haversine_km(CITY_COORDS[c1], CITY_COORDS[c2])
            dist_weights[(c1, c2)] = float(np.exp(-dist / 500.0))

    # Helper: get last values for lags from stored window
    def last_pm25(city: str, k_back: int) -> float | None:
        win = city_last7.get(city) or []
        if len(win) < k_back:
            return None
        return _safe_float(win[-k_back].get("pm2_5"))

    def last_aqi(city: str, k_back: int) -> float | None:
        win = city_last7.get(city) or []
        if len(win) < k_back:
            return None
        return _safe_float(win[-k_back].get("aqi"))

    def last7(city: str, field: str) -> list[float]:
        win = city_last7.get(city) or []
        vals = [_safe_float(r.get(field)) for r in win]
        return [v for v in vals if v is not None]

    # Build one row per city
    rows = []
    for city in CITIES:
        reading = live.get(city) or {}

        # If live missing, fall back to last record in last7
        if not reading:
            win = city_last7.get(city) or []
            reading = win[-1] if win else {}

        row: dict[str, float | int] = {}
        # Raw core fields
        row["aqi"] = _safe_float(reading.get("aqi")) or float(_rolling_mean(last7(city, "aqi")))
        row["co"] = _safe_float(reading.get("co")) or float(_rolling_mean(last7(city, "co")))
        # 'no' is not in the WAQI API — default to historical mean from last7
        row["no"] = _safe_float(reading.get("no")) or float(_rolling_mean(last7(city, "no")))
        row["no2"] = _safe_float(reading.get("no2")) or float(_rolling_mean(last7(city, "no2")))
        row["o3"] = _safe_float(reading.get("o3")) or float(_rolling_mean(last7(city, "o3")))
        row["so2"] = _safe_float(reading.get("so2")) or float(_rolling_mean(last7(city, "so2")))
        row["pm2_5"] = _safe_float(reading.get("pm2_5")) or float(_rolling_mean(last7(city, "pm2_5")))
        row["pm10"] = _safe_float(reading.get("pm10")) or float(_rolling_mean(last7(city, "pm10")))
        # 'nh3' is not in the WAQI API — default to historical mean from last7
        row["nh3"] = _safe_float(reading.get("nh3")) or float(_rolling_mean(last7(city, "nh3")))
        row["temperature"] = _safe_float(reading.get("temperature")) or float(_rolling_mean(last7(city, "temperature")))
        row["wind_speed"] = _safe_float(reading.get("wind_speed")) or float(_rolling_mean(last7(city, "wind_speed")))
        row["rainfall"] = _safe_float(reading.get("rainfall"))
        if row["rainfall"] is None:
            row["rainfall"] = 0.0
        row["pressure"] = _safe_float(reading.get("pressure")) or float(_rolling_mean(last7(city, "pressure")))

        # Lags
        for lag in [1, 2, 3, 7]:
            row[f"pm2_5_lag{lag}"] = last_pm25(city, lag) or float(_rolling_mean(last7(city, "pm2_5")))
            row[f"aqi_lag{lag}"] = last_aqi(city, lag) or float(_rolling_mean(last7(city, "aqi")))

        # Rolling (shifted by 1 day in training: use last days excluding today)
        pm_hist = last7(city, "pm2_5")
        pm_hist_excl_today = pm_hist[:-1] if len(pm_hist) > 1 else pm_hist
        row["pm2_5_roll3mean"] = _rolling_mean(pm_hist_excl_today[-3:])
        row["pm2_5_roll7mean"] = _rolling_mean(pm_hist_excl_today[-7:])
        row["pm2_5_roll3std"] = _rolling_std(pm_hist_excl_today[-3:])
        row["pm2_5_roll7std"] = _rolling_std(pm_hist_excl_today[-7:])

        # Calendar features: use today in UTC
        ts = pd.Timestamp.utcnow()
        row["dayofweek"] = int(ts.dayofweek)
        row["month"] = int(ts.month)
        row["is_weekend"] = int(ts.dayofweek >= 5)
        row["quarter"] = int(ts.quarter)
        row["crop_burning_season"] = int(ts.month in [10, 11])
        row["monsoon_season"] = int(ts.month in [6, 7, 8, 9])

        row["city_idx"] = int(CITY_TO_IDX[city])

        # Neighbor lag1 features from live batch, fallback to neighbor last7 lag1
        for neighbor in CITIES:
            if neighbor == city:
                continue
            w = dist_weights[(city, neighbor)]
            key = neighbor.lower()

            n_reading = live.get(neighbor) or {}
            n_pm = _safe_float(n_reading.get("pm2_5"))
            if n_pm is None:
                n_pm = last_pm25(neighbor, 1)
            n_wind = _safe_float(n_reading.get("wind_speed"))
            if n_wind is None:
                n_wind = _safe_float((city_last7.get(neighbor) or [{}])[-1].get("wind_speed"))

            row[f"neighbor_{key}_pm2_5_lag1"] = (n_pm or 0.0) * w
            row[f"neighbor_{key}_wind_lag1"] = (n_wind or 0.0) * w

        rows.append({"city": city, **row})

    feat_df = pd.DataFrame(rows).set_index("city")

    # Ensure schema/order
    missing = [c for c in feature_cols if c not in feat_df.columns]
    if missing:
        # Neighbor columns are part of schema; if missing, set to 0.
        for c in missing:
            feat_df[c] = 0.0

    feat_df = feat_df[feature_cols].astype(float)
    X_scaled = scaler.transform(feat_df)

    # Model is trained on log1p(target) — back-transform with expm1
    preds_log = model.predict(X_scaled)
    preds = np.maximum(np.expm1(preds_log), 0.0)

    out = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "predictions": {city: round(float(preds[i]), 2) for i, city in enumerate(feat_df.index.tolist())},
    }

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)

    print("=" * 60)
    print("Step 06 — Live prediction complete")
    print("=" * 60)
    for city, val in out["predictions"].items():
        print(f"  {city:<10s} {val:.2f} µg/m³")
    print(f"\n✓ Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
