"""
Step 02 — Feature Engineering
Input:  data/raw/merged.csv
Output: data/processed/features.parquet, models/feature_scaler.pkl

Performs: data cleaning, missingness diagnosis, imputation, temporal/calendar
feature engineering, target creation, time-based splitting, and feature scaling.
"""

import os
import sys
import random
import warnings
import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import joblib

# ────────────────────────────────────────────────────────────────
# Reproducibility
# ────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

warnings.filterwarnings("ignore", category=FutureWarning)

# ────────────────────────────────────────────────────────────────
# Paths (run from project root)
# ────────────────────────────────────────────────────────────────
RAW_CSV     = "data/raw/merged.csv"
OUT_PARQUET = "data/processed/features.parquet"
SCALER_PATH = "models/feature_scaler.pkl"
FEATURE_COLS_PATH = "models/feature_columns.json"

os.makedirs("data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# 2.1  Load and Parse
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("Step 02 — Feature Engineering")
print("=" * 60)

df = pd.read_csv(RAW_CSV, index_col=0)
df["date"] = pd.to_datetime(df["date"], infer_datetime_format=True)

# Normalize city names (handle 'Bangalore' → 'Bengaluru' if present)
df["city"] = df["city"].replace({"Bangalore": "Bengaluru", "bangalore": "Bengaluru"})

# Capitalize city names for consistency
df["city"] = df["city"].str.strip().str.title()

df = df.sort_values(["city", "date"]).reset_index(drop=True)
print(f"\n✓ Loaded {len(df)} rows, {df['city'].nunique()} cities: {df['city'].unique().tolist()}")
print(f"  Date range: {df['date'].min()} → {df['date'].max()}")

# Option: temporarily drop a city from the pipeline
# (keeps the rest of the code consistent when a city's live feed is unstable)
EXCLUDE_CITIES = {"Mumbai"}
if EXCLUDE_CITIES:
    before = len(df)
    df = df[~df["city"].isin(EXCLUDE_CITIES)].copy()
    df = df.sort_values(["city", "date"]).reset_index(drop=True)
    print(f"\n  ✓ Excluded cities: {sorted(EXCLUDE_CITIES)}")
    print(f"    Rows removed: {before - len(df)}")
    print(f"    Remaining cities: {sorted(df['city'].unique().tolist())}")


# ═══════════════════════════════════════════════════════════════
# 2.2  Missing Data Diagnosis
# ═══════════════════════════════════════════════════════════════
NUMERIC_COLS = [
    "aqi", "co", "no", "no2", "o3", "so2",
    "pm2_5", "pm10", "nh3",
    "temperature", "wind_speed", "rainfall", "pressure",
]

print("\n" + "─" * 60)
print("Missingness Report")
print("─" * 60)

# Treat pm2_5 < 1.0 as missing (sensor failure)
df.loc[df["pm2_5"] < 1.0, "pm2_5"] = np.nan

missingness_report = []
for city in sorted(df["city"].unique()):
    city_df = df[df["city"] == city]
    for col in NUMERIC_COLS:
        n_total = len(city_df)
        n_miss = city_df[col].isna().sum()
        rate = n_miss / n_total * 100

        # Simple MCAR approximation: compare mean of present rows vs overall
        # If data is MNAR, missingness correlates with high pollution
        if n_miss > 0 and n_miss < n_total:
            present = city_df[col].dropna()
            # Check if missingness in this column correlates with pm2_5 being high
            miss_mask = city_df[col].isna().astype(int)
            pm25_avail = city_df["pm2_5"].notna()
            if pm25_avail.sum() > 10 and miss_mask.sum() > 2:
                try:
                    corr, pval = stats.pointbiserialr(
                        miss_mask[pm25_avail], city_df.loc[pm25_avail.index[pm25_avail], "pm2_5"]
                    )
                    mechanism = "MNAR" if pval < 0.05 and abs(corr) > 0.1 else "MCAR"
                except Exception:
                    mechanism = "MCAR"
            else:
                mechanism = "MCAR"
        else:
            mechanism = "—"

        flag = " ⚠ HIGH" if rate > 30 else ""
        missingness_report.append({
            "city": city, "column": col,
            "missing": n_miss, "rate_pct": round(rate, 1),
            "mechanism": mechanism, "flag": flag,
        })

report_df = pd.DataFrame(missingness_report)
for city in sorted(df["city"].unique()):
    subset = report_df[report_df["city"] == city]
    if subset["missing"].sum() == 0:
        print(f"  {city}: no missing values")
    else:
        print(f"\n  {city}:")
        for _, r in subset[subset["missing"] > 0].iterrows():
            print(f"    {r['column']:>14s}: {r['rate_pct']:5.1f}% missing  "
                  f"({r['mechanism']}){r['flag']}")


# ═══════════════════════════════════════════════════════════════
# 2.3  Imputation
# ═══════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("Imputing missing values …")
print("─" * 60)

# Separate MNAR and MCAR columns per city
mnar_cols_by_city = {}
for city in df["city"].unique():
    mnar_cols = report_df[
        (report_df["city"] == city) & (report_df["mechanism"] == "MNAR")
    ]["column"].tolist()
    mnar_cols_by_city[city] = mnar_cols

# Apply imputation per city
for city in df["city"].unique():
    mask = df["city"] == city
    city_idx = df[mask].index

    mnar_cols = mnar_cols_by_city.get(city, [])

    # MNAR columns: forward-fill → backward-fill → city median
    for col in mnar_cols:
        if col in NUMERIC_COLS:
            df.loc[city_idx, col] = (
                df.loc[city_idx, col]
                .ffill()
                .bfill()
                .fillna(df.loc[city_idx, col].median())
            )

    # MCAR columns: KNN imputation (k=5) within city
    mcar_cols = [c for c in NUMERIC_COLS if c not in mnar_cols and df.loc[city_idx, c].isna().any()]
    if mcar_cols:
        imputer = KNNImputer(n_neighbors=5)
        df.loc[city_idx, mcar_cols] = imputer.fit_transform(df.loc[city_idx, mcar_cols])

# Fill any remaining NaN in numeric cols with global median
for col in NUMERIC_COLS:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

remaining = df[NUMERIC_COLS].isna().sum().sum()
print(f"  ✓ Imputation complete — remaining NaN in features: {remaining}")


# ═══════════════════════════════════════════════════════════════
# 2.4  Temporal Feature Engineering
# ═══════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("Creating temporal features …")
print("─" * 60)

for lag in [1, 2, 3, 7]:
    df[f"pm2_5_lag{lag}"] = df.groupby("city")["pm2_5"].shift(lag)
    df[f"aqi_lag{lag}"]   = df.groupby("city")["aqi"].shift(lag)

for window in [3, 7]:
    df[f"pm2_5_roll{window}mean"] = df.groupby("city")["pm2_5"].transform(
        lambda x: x.shift(1).rolling(window).mean()
    )
    df[f"pm2_5_roll{window}std"] = df.groupby("city")["pm2_5"].transform(
        lambda x: x.shift(1).rolling(window).std()
    )

print("  ✓ Lag features: pm2_5_lag{1,2,3,7}, aqi_lag{1,2,3,7}")
print("  ✓ Rolling features: pm2_5_roll{3,7}{mean,std}")


# ═══════════════════════════════════════════════════════════════
# 2.5  Calendar Features
# ═══════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("Creating calendar features …")
print("─" * 60)

df["dayofweek"]  = df["date"].dt.dayofweek          # 0=Monday
df["month"]      = df["date"].dt.month
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
df["quarter"]    = df["date"].dt.quarter

# Indian festival / harvest season proxy
df["crop_burning_season"] = df["month"].isin([10, 11]).astype(int)
df["monsoon_season"]      = df["month"].isin([6, 7, 8, 9]).astype(int)

print("  ✓ dayofweek, month, is_weekend, quarter, crop_burning_season, monsoon_season")


# ═══════════════════════════════════════════════════════════════
# 2.6  City Encoding
# ═══════════════════════════════════════════════════════════════
CITY_TO_IDX = {
    "Delhi": 0, "Bengaluru": 1, "Kolkata": 2, "Hyderabad": 3,
}
df["city_idx"] = df["city"].map(CITY_TO_IDX)

# Verify no unmapped cities
unmapped = df["city_idx"].isna().sum()
if unmapped > 0:
    print(f"  ⚠ {unmapped} rows have unmapped city names!")
    print(f"    Unique cities: {df['city'].unique().tolist()}")
    sys.exit(1)
df["city_idx"] = df["city_idx"].astype(int)
print(f"\n  ✓ City encoding: {CITY_TO_IDX}")


# ═══════════════════════════════════════════════════════════════
# 2.7  Target Creation
# ═══════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("Creating target (next-day PM2.5) …")
print("─" * 60)

df["target_pm2_5"] = df.groupby("city")["pm2_5"].shift(-1)

before = len(df)
df = df.dropna(subset=["target_pm2_5"])
after = len(df)
print(f"  ✓ Dropped {before - after} rows with no next-day label")
print(f"  ✓ Remaining rows: {after}")


# ═══════════════════════════════════════════════════════════════
# 2.8  Train / Val / Test Split (time-based)
# ═══════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("Time-based train/val/test split …")
print("─" * 60)

df["split"] = "train"
df.loc[df["date"] >= "2022-10-01", "split"] = "val"
df.loc[df["date"] >= "2023-02-01", "split"] = "test"

for s in ["train", "val", "test"]:
    n = (df["split"] == s).sum()
    if n > 0:
        date_range = f"{df.loc[df['split']==s, 'date'].min().date()} → {df.loc[df['split']==s, 'date'].max().date()}"
    else:
        date_range = "N/A"
    print(f"  {s:>5s}: {n:>5d} rows  ({date_range})")


# ═══════════════════════════════════════════════════════════════
# 2.9  Feature Scaling (fit on train only)
# ═══════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("Scaling features …")
print("─" * 60)

FEATURE_COLS = [
    "city_idx",
    # NOTE: pm2_5 and aqi at t=0 are EXCLUDED to prevent data leakage.
    # target_pm2_5 = pm2_5.shift(-1), so including pm2_5(t) leaks the target.
    # aqi(t) is also strongly correlated with pm2_5(t) and thus leaks.
    # Use only lagged versions (pm2_5_lag1, aqi_lag1, etc.).
    "co", "no", "no2", "o3", "so2",
    "pm10", "nh3",
    "temperature", "wind_speed", "rainfall", "pressure",
    "pm2_5_lag1", "pm2_5_lag2", "pm2_5_lag3", "pm2_5_lag7",
    "aqi_lag1", "aqi_lag2", "aqi_lag3", "aqi_lag7",
    "pm2_5_roll3mean", "pm2_5_roll7mean",
    "pm2_5_roll3std", "pm2_5_roll7std",
    "dayofweek", "month", "is_weekend", "quarter",
    "crop_burning_season", "monsoon_season",
]

# Drop rows with NaN in feature cols (from lag features at start of each city)
before = len(df)
df = df.dropna(subset=FEATURE_COLS)
after = len(df)
print(f"  Dropped {before - after} rows with NaN in feature columns (lag warmup)")

# Fit scaler on train split only
train_mask = df["split"] == "train"
scaler = StandardScaler()
scaler.fit(df.loc[train_mask, FEATURE_COLS])

# Transform all splits — store scaled values alongside originals
scaled_cols = [f"{c}_scaled" for c in FEATURE_COLS]
df[scaled_cols] = scaler.transform(df[FEATURE_COLS])

joblib.dump(scaler, SCALER_PATH)
print(f"  ✓ Scaler fitted on {train_mask.sum()} train rows, saved to {SCALER_PATH}")

# Also scale pm2_5 and aqi for the LSTM temporal branch.
# These columns are NOT in FEATURE_COLS (to prevent leakage in the main branch),
# but the temporal sequence still needs their scaled values.
TEMPORAL_EXTRA_COLS = ["pm2_5", "aqi"]
temporal_scaler = StandardScaler()
temporal_scaler.fit(df.loc[train_mask, TEMPORAL_EXTRA_COLS])
for c in TEMPORAL_EXTRA_COLS:
    idx = TEMPORAL_EXTRA_COLS.index(c)
    df[f"{c}_scaled"] = (df[c] - temporal_scaler.mean_[idx]) / temporal_scaler.scale_[idx]

TEMPORAL_SCALER_PATH = "models/temporal_scaler.pkl"
joblib.dump(temporal_scaler, TEMPORAL_SCALER_PATH)
print(f"  ✓ Temporal scaler (pm2_5, aqi) saved to {TEMPORAL_SCALER_PATH}")

# Save feature column order for inference reproducibility
with open(FEATURE_COLS_PATH, "w") as f:
    json.dump(FEATURE_COLS, f, indent=2)
print(f"  ✓ Feature column order saved to {FEATURE_COLS_PATH}")


# ═══════════════════════════════════════════════════════════════
# 2.10  Save
# ═══════════════════════════════════════════════════════════════
df.to_parquet(OUT_PARQUET, index=False)
print(f"\n{'=' * 60}")
print(f"✓ Features saved to {OUT_PARQUET}")
print(f"  Shape: {df.shape}")
print(f"  Columns: {df.columns.tolist()}")
print(f"{'=' * 60}")
