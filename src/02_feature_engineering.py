"""Step 02 — Feature engineering (Plan vNext: XGBoost + cross-city features)

Input:
  - data/raw/merged.csv

Output:
  - data/processed/features.parquet
  - models/feature_scaler.pkl
  - models    # Time split: keep the most recent year as validation; train on all earlier years.
    # This is useful when you want maximum training history and you're not holding out a test set.
    max_dt = df["date"].max()
    val_start = (max_dt - pd.Timedelta(days=365)).normalize()

    df["split"] = "train"
    df.loc[df["date"] >= val_start, "split"] = "val"

    print("\nSplit windows:")
    print(f"  train: < {val_start.date()}")
    print(f"  val:   >= {val_start.date()} (last ~365 days)")

    print("\nSplit counts:")
    print(df["split"].value_counts().to_string())
olumns.json

Contract:
  - target_pm2_5 is next-day PM2.5: pm2_5 shifted by -1 per city.
  - The feature scaler is fit ONLY on train split, then applied to val/test.
  - Cross-city dependencies are encoded explicitly via neighbor_* lag features.
"""

from __future__ import annotations

import json
import os
from math import atan2, cos, radians, sin, sqrt

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

SEED = 42
np.random.seed(SEED)

# Source merged dataset.
# Prefer the newer post-EDA merged file if present.
RAW_CSV = os.getenv("AIR_MERGED_CSV", "data/processed/merged.csv")
OUT_PARQUET = "data/processed/features.parquet"
SCALER_PATH = "models/feature_scaler.pkl"
FEATURE_COLS_PATH = "models/feature_columns.json"


# NOTE: Keep this list in sync with plan.md.
CITIES = ["Delhi", "Mumbai", "Bengaluru", "Kolkata", "Hyderabad"]
CITY_TO_IDX = {c: i for i, c in enumerate(CITIES)}

CITY_COORDS: dict[str, tuple[float, float]] = {
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
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


def _ensure_dirs() -> None:
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)


def main() -> None:
    _ensure_dirs()

    print("=" * 60)
    print("Step 02 — Feature Engineering (XGBoost + cross-city)")
    print("=" * 60)

    df = pd.read_csv(RAW_CSV)

    # Common cleanup from EDA merges
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")

    if "date" not in df.columns or "city" not in df.columns:
        raise ValueError("merged.csv must include 'date' and 'city' columns")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()

    # Normalize city names (handle Bangalore alias)
    df["city"] = (
        df["city"]
        .astype(str)
        .str.strip()
        .replace({"Bangalore": "Bengaluru", "bangalore": "Bengaluru"})
        .str.title()
    )

    df = df[df["city"].isin(CITIES)].copy()
    df = df.sort_values(["city", "date"]).reset_index(drop=True)
    if df.empty:
        raise ValueError("No rows left after filtering to configured CITIES")

    print(f"\n✓ Loaded {len(df)} rows across {df['city'].nunique()} cities")
    print(f"  Date range: {df['date'].min().date()} → {df['date'].max().date()}")

    # Basic sensor failure rule
    df.loc[df["pm2_5"] < 1.0, "pm2_5"] = np.nan

    # Some merged datasets won't contain all pollutant channels.
    # Keep a stable schema by creating missing channels as 0.0 (so training/inference match).
    numeric_cols = [
        "aqi",
        "co",
        "no",
        "no2",
        "o3",
        "so2",
        "pm2_5",
        "pm10",
        "nh3",
        "temperature",
        "wind_speed",
        "rainfall",
        "pressure",
    ]
    for c in numeric_cols:
        if c not in df.columns:
            df[c] = 0.0

    # KNN imputation per city
    imputed_parts: list[pd.DataFrame] = []
    for _, grp in df.groupby("city", sort=True):
        grp = grp.copy().sort_values("date")
        imp = KNNImputer(n_neighbors=5)
        grp[numeric_cols] = imp.fit_transform(grp[numeric_cols])
        imputed_parts.append(grp)
    df = pd.concat(imputed_parts, ignore_index=True).sort_values(["city", "date"]).reset_index(
        drop=True
    )

    # Temporal lag/rolling features (per city)
    for lag in [1, 2, 3, 7, 14, 21, 28]:
        df[f"pm2_5_lag{lag}"] = df.groupby("city")["pm2_5"].shift(lag)
        df[f"aqi_lag{lag}"] = df.groupby("city")["aqi"].shift(lag)

    for window in [3, 7, 14]:
        df[f"pm2_5_roll{window}mean"] = df.groupby("city")["pm2_5"].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        df[f"pm2_5_roll{window}std"] = df.groupby("city")["pm2_5"].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).std().fillna(0.0)
        )

    # Calendar / seasonality features
    df["dayofweek"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["dayofyear"] = df["date"].dt.dayofyear
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["quarter"] = df["date"].dt.quarter
    df["crop_burning_season"] = df["month"].isin([10, 11]).astype(int)
    df["monsoon_season"] = df["month"].isin([6, 7, 8, 9]).astype(int)

    # Smooth yearly seasonality often boosts generalization.
    df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)

    # Neighbor weights based on distance
    dist_weights: dict[tuple[str, str], float] = {}
    for c1 in CITIES:
        for c2 in CITIES:
            if c1 == c2:
                continue
            dist = haversine_km(CITY_COORDS[c1], CITY_COORDS[c2])
            dist_weights[(c1, c2)] = float(np.exp(-dist / 500.0))

    # Prepare per-date lookup for neighbors (date-aligned mapping)
    pivot_pm = df.pivot_table(index="date", columns="city", values="pm2_5", aggfunc="first").sort_index()
    pivot_wind = (
        df.pivot_table(index="date", columns="city", values="wind_speed", aggfunc="first")
        .sort_index()
    )
    pivot_pm_lag1 = pivot_pm.shift(1)
    pivot_pm_lag7 = pivot_pm.shift(7)
    pivot_wind_lag1 = pivot_wind.shift(1)

    for target_city in CITIES:
        mask = df["city"] == target_city
        dates = df.loc[mask, "date"]
        for neighbor in CITIES:
            if neighbor == target_city:
                continue
            w = dist_weights[(target_city, neighbor)]
            key = neighbor.lower()

            pm_lag1_map = pivot_pm_lag1[neighbor]
            pm_lag7_map = pivot_pm_lag7[neighbor]
            wind_map = pivot_wind_lag1[neighbor]

            df.loc[mask, f"neighbor_{key}_pm2_5_lag1"] = dates.map(pm_lag1_map).to_numpy() * w
            df.loc[mask, f"neighbor_{key}_pm2_5_lag7"] = dates.map(pm_lag7_map).to_numpy() * w
            df.loc[mask, f"neighbor_{key}_wind_lag1"] = dates.map(wind_map).to_numpy() * w

    # Encode city + label
    df["city_idx"] = df["city"].map(CITY_TO_IDX).astype(int)
    df["target_pm2_5"] = df.groupby("city")["pm2_5"].shift(-1)
    df = df.dropna(subset=["target_pm2_5"]).reset_index(drop=True)

    # Time split: keep the most recent year as validation; train on all earlier years.
    # (No separate test split in this mode.)
    max_dt = df["date"].max()
    val_start = (max_dt - pd.Timedelta(days=365)).normalize()

    df["split"] = "train"
    df.loc[df["date"] >= val_start, "split"] = "val"

    print("\nSplit windows:")
    print(f"  train: < {val_start.date()}")
    print(f"  val:   >= {val_start.date()} (last ~365 days)")

    print("\nSplit counts:")
    print(df["split"].value_counts().to_string())

    neighbor_cols = sorted([c for c in df.columns if c.startswith("neighbor_")])
    # Neighbor features can be missing when a neighbor city doesn't have a reading for that date.
    # Missing neighbor contribution should behave like 0 influence.
    if neighbor_cols:
        df[neighbor_cols] = df[neighbor_cols].fillna(0.0)

    core_feature_columns = [
        "aqi",
        "co",
        "no",
        "no2",
        "o3",
        "so2",
        "pm2_5",
        "pm10",
        "nh3",
        "temperature",
        "wind_speed",
        "rainfall",
        "pressure",
        "pm2_5_lag1",
        "pm2_5_lag2",
        "pm2_5_lag3",
        "pm2_5_lag7",
        "pm2_5_lag14",
        "pm2_5_lag21",
        "pm2_5_lag28",
        "aqi_lag1",
        "aqi_lag2",
        "aqi_lag3",
        "aqi_lag7",
        "aqi_lag14",
        "aqi_lag21",
        "aqi_lag28",
        "pm2_5_roll3mean",
        "pm2_5_roll7mean",
        "pm2_5_roll14mean",
        "pm2_5_roll3std",
        "pm2_5_roll7std",
        "pm2_5_roll14std",
        "dayofweek",
        "month",
        "dayofyear",
        "is_weekend",
        "quarter",
        "crop_burning_season",
        "monsoon_season",
        "doy_sin",
        "doy_cos",
        "city_idx",
    ]

    feature_columns = core_feature_columns + neighbor_cols

    before = len(df)
    # Warmup NaNs only come from in-city lag features; neighbor columns are already filled with 0.
    df = df.dropna(subset=core_feature_columns).reset_index(drop=True)
    dropped = before - len(df)
    print(f"\nDropped {dropped} warmup rows")
    if df.empty:
        raise ValueError(
            "All rows were dropped after lag/neighbor feature creation; check date alignment and input coverage."
        )

    # Scale on train only
    train_mask = df["split"] == "train"
    # Scale into new float columns to avoid dtype-mismatch warnings when overwriting ints.
    scaler = StandardScaler()
    scaler.fit(df.loc[train_mask, feature_columns].astype(float))

    scaled_cols = [f"{c}_scaled" for c in feature_columns]
    df[scaled_cols] = np.nan
    for split in ["train", "val", "test"]:
        m = df["split"] == split
        if not m.any():
            continue
        df.loc[m, scaled_cols] = scaler.transform(df.loc[m, feature_columns].astype(float))

    joblib.dump(scaler, SCALER_PATH)
    with open(FEATURE_COLS_PATH, "w") as f:
        json.dump(feature_columns, f, indent=2)

    df.to_parquet(OUT_PARQUET, index=False)
    print("\nSaved:")
    print(f"  - {OUT_PARQUET} (rows={len(df)})")
    print(f"  - {SCALER_PATH}")
    print(f"  - {FEATURE_COLS_PATH}")


if __name__ == "__main__":
    main()
