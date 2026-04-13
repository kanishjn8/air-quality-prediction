# AirMind: Spatiotemporal PM2.5 Forecasting — Agent Execution Plan

## Project Identity

**Project name:** AirMind
**Goal:** Predict next-day PM2.5 concentrations for 5 Indian cities using a spatiotemporally aware XGBoost model with explicit cross-city features, with full XAI interpretability via SHAP.
**Research gaps addressed:**
1. Cross-city pollutant dependency — encoded as explicit neighbor features derived from geospatial and wind-correlation analysis
2. Missing data bias from structured sensor failure — MCAR vs MNAR diagnosis and appropriate imputation

---

## Architecture Philosophy — Training vs Prediction

Two completely separate pipelines sharing only the saved model artefacts:

```
TRAINING PIPELINE  (run once, or retrain periodically)
────────────────────────────────────────────────────────
merged.csv  ──►  feature engineering  ──►  features.parquet
city coords ──►  cross-city features  ──►  (embedded in features.parquet)
                        ↓
              XGBoost regressor training
                        ↓
              best_model.pkl  +  feature_scaler.pkl
              feature_columns.json  +  city_last7.json


PREDICTION PIPELINE  (run daily at inference time)
────────────────────────────────────────────────────────
WAQI Live API  ──►  05_live_ingest.py  ──►  latest_reading.json
best_model.pkl  +  feature_scaler.pkl
                        ↓
              06_predict_live.py
                        ↓
              next-day PM2.5 per city  ──►  Streamlit dashboard
```

**Key design rules:**

1. **Live data is the input at inference — not a training signal.** Today's live API reading is fed to the trained model to predict tomorrow's PM2.5. No retraining at prediction time.
2. **Cross-city features replace the GNN.** Instead of a graph neural network, each city's feature row explicitly includes its neighbor cities' lagged PM2.5 values, weighted by geospatial distance. Same spatial signal, no deep learning needed.
3. **Feature schema must match exactly.** `feature_columns.json` defines the canonical ordered column list. Both training and inference must use it — any mismatch causes silent prediction errors.
4. **The scaler is fit on train split only.** Applied to training features and inference features. Target (`target_pm2_5`) is never scaled.
5. **`rainfall` defaults to 0.0 at inference** — not available from live API.
6. **`city_last7.json` must store RAW unscaled values** — used to build the lag window at inference before scaling.

---

## Repository Layout (create exactly this structure)

```
airmind/
├── data/
│   ├── raw/
│   │   └── merged.csv                   # provided — DO NOT modify
│   ├── processed/
│   │   └── features.parquet             # output of 02_feature_engineering.py
│   └── live/
│       ├── latest_reading.json          # written by 05_live_ingest.py
│       └── predictions.json             # written by 06_predict_live.py
├── src/
│   ├── 02_feature_engineering.py
│   ├── 03_train.py
│   ├── 04_explain.py
│   ├── 05_live_ingest.py
│   └── 06_predict_live.py
├── models/
│   ├── best_model.pkl                   # saved by 03_train.py
│   ├── feature_scaler.pkl               # saved by 02_feature_engineering.py
│   ├── feature_columns.json             # saved by 02_feature_engineering.py
│   └── city_last7.json                  # saved by 03_train.py (raw unscaled values)
├── outputs/
│   ├── shap_summary.png
│   ├── shap_waterfall_spike1.png
│   ├── shap_waterfall_spike2.png
│   ├── shap_waterfall_spike3.png
│   ├── cross_city_influence.png
│   ├── lag_importance.png
│   ├── per_city_metrics.csv
│   └── model_comparison.csv
├── app/
│   └── streamlit_app.py
├── .env.example
└── plan.md
```

---

## Environment Setup

### `.env.example`
```
WAQI_TOKEN=your_token_here
```

### Dependencies (uv)

```bash
uv init airmind
cd airmind
uv add pandas==2.2.2 numpy==1.26.4 scikit-learn==1.4.2 xgboost==2.0.3 \
        shap==0.45.1 matplotlib==3.8.4 seaborn==0.13.2 streamlit==1.35.0 \
        requests==2.32.2 python-dotenv==1.0.1 pyarrow==16.0.0 joblib==1.4.2
```

Run any script:
```bash
uv run python src/02_feature_engineering.py
```

---

## Data Specification

### Existing dataset — `data/raw/merged.csv`

| Column | Type | Notes |
|--------|------|-------|
| (index) | int | drop |
| city | str | one of: `Delhi`, `Bengaluru`, `Kolkata`, `Hyderabad` |
| date | str→datetime | parse with `pd.to_datetime` |
| aqi | float | Air Quality Index (composite) |
| co | float | Carbon monoxide |
| no2 | float | Nitrogen dioxide |
| o3 | float | Ozone |
| so2 | float | Sulphur dioxide |
| pm2_5 | float | **Primary target variable** |
| pm10 | float | Particulate matter 10µm |
| temperature | float | °C |
| wind_speed | float | m/s |
| rainfall | float | mm |
| pressure | float | hPa |

**Target:** `pm2_5` shifted by -1 day per city group — predicting tomorrow's PM2.5 from today's features.

### Live API

**Endpoint pattern:**
```
https://api.waqi.info/feed/{city_slug}/?token={WAQI_TOKEN}
```

**City slugs:**
```python
CITY_SLUGS = {
    "Delhi":     "delhi",
    "Mumbai":    "mumbai",
    "Bengaluru": "bengaluru",
    "Kolkata":   "kolkata",
    "Hyderabad": "hyderabad"
}
```

**API response field mapping** (from `data.iaqi`):

| API key | Our column | Notes |
|---------|-----------|-------|
| `data.aqi` | aqi | top-level, not in iaqi |
| `pm25.v` | pm2_5 | note: no underscore in API key |
| `pm10.v` | pm10 | |
| `co.v` | co | |
| `no2.v` | no2 | |
| `o3.v` | o3 | |
| `so2.v` | so2 | |
| `t.v` | temperature | |
| `w.v` | wind_speed | |
| `p.v` | pressure | |
| `h.v` | humidity | live only — not used in model |
| `dew.v` | dew_point | live only — not used in model |

**Fields NOT in live API:**
- `rainfall` → default to `0.0`
- Cross-city neighbor features → computed from other cities' readings in the same fetch batch

---

## Step-by-Step Agent Instructions

---

### Step 02 — Feature Engineering (`src/02_feature_engineering.py`)

**Input:** `data/raw/merged.csv`
**Output:** `data/processed/features.parquet`, `models/feature_scaler.pkl`, `models/feature_columns.json`

#### 2.1 Load and parse
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import joblib, json, os
from math import radians, sin, cos, sqrt, atan2

SEED = 42
np.random.seed(SEED)

df = pd.read_csv("data/raw/merged.csv", index_col=0)
df["date"] = pd.to_datetime(df["date"], infer_datetime_format=True)
df = df.sort_values(["city", "date"]).reset_index(drop=True)

# Normalize city names
df["city"] = df["city"].str.strip().replace({"Bangalore": "Bengaluru"})

# Treat pm2_5 < 1.0 as sensor failure → missing
df.loc[df["pm2_5"] < 1.0, "pm2_5"] = np.nan
```

#### 2.2 Missing data diagnosis — print report, do not drop rows
```python
for city in df["city"].unique():
    city_df = df[df["city"] == city]
    missing = city_df.isnull().mean().round(3)
    high = missing[missing > 0.30].index.tolist()
    print(f"{city}: missing={missing.to_dict()}")
    if high:
        print(f"  WARNING: >30% missing in {high}")
```

#### 2.3 KNN imputation per city (input features only — never impute target)
```python
NUMERIC_COLS = ["aqi","co","no2","o3","so2","pm2_5","pm10",
                "temperature","wind_speed","rainfall","pressure"]

imputed = []
for city, grp in df.groupby("city"):
    grp = grp.copy().sort_values("date")
    imp = KNNImputer(n_neighbors=5)
    grp[NUMERIC_COLS] = imp.fit_transform(grp[NUMERIC_COLS])
    imputed.append(grp)

df = pd.concat(imputed).sort_values(["city","date"]).reset_index(drop=True)
```

#### 2.4 Temporal lag and rolling features (past values only — no leakage)
```python
for lag in [1, 2, 3, 7]:
    df[f"pm2_5_lag{lag}"] = df.groupby("city")["pm2_5"].shift(lag)
    df[f"aqi_lag{lag}"]   = df.groupby("city")["aqi"].shift(lag)

for window in [3, 7]:
    df[f"pm2_5_roll{window}mean"] = df.groupby("city")["pm2_5"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )
    df[f"pm2_5_roll{window}std"] = df.groupby("city")["pm2_5"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).std().fillna(0)
    )
```

#### 2.5 Calendar features
```python
df["dayofweek"]           = df["date"].dt.dayofweek
df["month"]               = df["date"].dt.month
df["is_weekend"]          = (df["dayofweek"] >= 5).astype(int)
df["quarter"]             = df["date"].dt.quarter
df["crop_burning_season"] = df["month"].isin([10, 11]).astype(int)
df["monsoon_season"]      = df["month"].isin([6, 7, 8, 9]).astype(int)
```

#### 2.6 Cross-city spatial features (spatiotemporal novelty — replaces GNN)

For each city row, attach the previous day's PM2.5 from every other city, weighted by geospatial proximity. This encodes pollutant transport between cities explicitly as tabular features.

```python
CITIES = ["Delhi", "Bengaluru", "Kolkata", "Hyderabad"]
CITY_TO_IDX = {c: i for i, c in enumerate(CITIES)}

CITY_COORDS = {
    "Delhi":     (28.6139, 77.2090),
    "Mumbai":    (19.0760, 72.8777),
    "Bengaluru": (12.9716, 77.5946),
    "Kolkata":   (22.5726, 88.3639),
    "Hyderabad": (17.3850, 78.4867),
}

def haversine(c1, c2):
    R = 6371
    lat1, lon1 = map(radians, c1)
    lat2, lon2 = map(radians, c2)
    dlat, dlon = lat2-lat1, lon2-lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

dist_weights = {
    (c1, c2): round(np.exp(-haversine(CITY_COORDS[c1], CITY_COORDS[c2]) / 500), 4)
    for c1 in CITIES for c2 in CITIES if c1 != c2
}

# Pivot to get neighbor values by date
pivot_pm   = df.pivot_table(index="date", columns="city", values="pm2_5",   aggfunc="first")
pivot_wind = df.pivot_table(index="date", columns="city", values="wind_speed", aggfunc="first")

for target_city in CITIES:
    mask = df["city"] == target_city
    for neighbor in CITIES:
        if neighbor == target_city:
            continue
        w = dist_weights[(target_city, neighbor)]
        key = neighbor.lower()
        # Lag-1 neighbor PM2.5, distance-weighted
        lag1_pm   = pivot_pm[neighbor].shift(1)
        lag1_wind = pivot_wind[neighbor].shift(1)
        df.loc[mask, f"neighbor_{key}_pm2_5_lag1"] = \
            df.loc[mask, "date"].map(lag1_pm).values * w
        df.loc[mask, f"neighbor_{key}_wind_lag1"] = \
            df.loc[mask, "date"].map(lag1_wind).values * w
```

#### 2.7 City encoding and target creation
```python
df["city_idx"] = df["city"].map(CITY_TO_IDX)

# Target: next day's PM2.5 — shift -1 within each city
df["target_pm2_5"] = df.groupby("city")["pm2_5"].shift(-1)
df = df.dropna(subset=["target_pm2_5"])  # removes last row per city only
```

#### 2.8 Time-based train/val/test split
```python
df["split"] = "train"
df.loc[df["date"] >= "2024-01-01", "split"] = "val"
df.loc[df["date"] >= "2024-07-01", "split"] = "test"

print(df["split"].value_counts())
# If test < 100 rows total, move the val cutoff earlier and adjust accordingly
```

#### 2.9 Build feature column list, fit scaler on train only
```python
neighbor_cols = sorted([c for c in df.columns if c.startswith("neighbor_")])

FEATURE_COLUMNS = [
    "aqi","co","no2","o3","so2","pm2_5","pm10",
    "temperature","wind_speed","rainfall","pressure",
    "pm2_5_lag1","pm2_5_lag2","pm2_5_lag3","pm2_5_lag7",
    "aqi_lag1","aqi_lag2","aqi_lag3","aqi_lag7",
    "pm2_5_roll3mean","pm2_5_roll7mean","pm2_5_roll3std","pm2_5_roll7std",
    "dayofweek","month","is_weekend","quarter",
    "crop_burning_season","monsoon_season","city_idx",
] + neighbor_cols

train_df = df[df["split"] == "train"]
scaler = StandardScaler()
scaler.fit(train_df[FEATURE_COLUMNS])

for split in ["train", "val", "test"]:
    mask = df["split"] == split
    df.loc[mask, FEATURE_COLUMNS] = scaler.transform(df.loc[mask, FEATURE_COLUMNS])
```

#### 2.10 Save
```python
os.makedirs("data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)

df.to_parquet("data/processed/features.parquet", index=False)
joblib.dump(scaler, "models/feature_scaler.pkl")
with open("models/feature_columns.json", "w") as f:
    json.dump(FEATURE_COLUMNS, f)

print(f"Saved {len(df)} rows, {len(FEATURE_COLUMNS)} features.")
```

---

### Step 03 — Model Training (`src/03_train.py`)

**Input:** `data/processed/features.parquet`, `models/feature_columns.json`
**Output:** `models/best_model.pkl`, `models/city_last7.json`, `outputs/model_comparison.csv`, `outputs/per_city_metrics.csv`

#### 3.1 Load
```python
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib, json, os

SEED = 42
df = pd.read_parquet("data/processed/features.parquet")
with open("models/feature_columns.json") as f:
    FEATURE_COLUMNS = json.load(f)

train = df[df["split"] == "train"]
val   = df[df["split"] == "val"]
test  = df[df["split"] == "test"]

X_train, y_train = train[FEATURE_COLUMNS].values, train["target_pm2_5"].values
X_val,   y_val   = val[FEATURE_COLUMNS].values,   val["target_pm2_5"].values
X_test,  y_test  = test[FEATURE_COLUMNS].values,  test["target_pm2_5"].values
```

#### 3.2 Train XGBoost with early stopping
```python
model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=SEED,
    early_stopping_rounds=30,
    eval_metric="mae",
    n_jobs=-1,
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)
```

#### 3.3 Baseline — Random Forest without cross-city features
```python
BASE_COLS = [c for c in FEATURE_COLUMNS if not c.startswith("neighbor_")]
rf = RandomForestRegressor(n_estimators=200, max_depth=10,
                            random_state=SEED, n_jobs=-1)
rf.fit(train[BASE_COLS].values, y_train)
rf_preds = rf.predict(test[BASE_COLS].values)
```

#### 3.4 Evaluate and save comparison
```python
os.makedirs("outputs", exist_ok=True)

def metrics(y_true, y_pred, label):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2   = r2_score(y_true, y_pred)
    print(f"{label:40s}  MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.3f}")
    return {"model": label, "MAE": round(mae,2), "RMSE": round(rmse,2), "R2": round(r2,3)}

xgb_preds = model.predict(X_test)
results = [
    metrics(y_test, xgb_preds, "XGBoost (with cross-city features)"),
    metrics(y_test, rf_preds,  "RandomForest (no cross-city features)"),
]
pd.DataFrame(results).to_csv("outputs/model_comparison.csv", index=False)
```

#### 3.5 Per-city metrics
```python
test_copy = test.copy()
test_copy["predicted"] = xgb_preds
city_rows = []
for city, grp in test_copy.groupby("city"):
    m = metrics(grp["target_pm2_5"], grp["predicted"], city)
    m["city"] = city
    city_rows.append(m)
pd.DataFrame(city_rows).to_csv("outputs/per_city_metrics.csv", index=False)
```

#### 3.6 Save model and raw city_last7 window
```python
joblib.dump(model, "models/best_model.pkl")

# Load RAW data to save unscaled lag window — critical for correct inference
raw_df = pd.read_csv("data/raw/merged.csv", index_col=0)
raw_df["date"] = pd.to_datetime(raw_df["date"], infer_datetime_format=True)
raw_df["city"] = raw_df["city"].str.strip().replace({"Bangalore": "Bengaluru"})
raw_df.loc[raw_df["pm2_5"] < 1.0, "pm2_5"] = np.nan

RAW_COLS = ["aqi","co","no2","o3","so2","pm2_5","pm10",
            "temperature","wind_speed","rainfall","pressure"]

last7 = {}
for city, grp in raw_df.groupby("city"):
    rows = grp.sort_values("date").tail(7)[RAW_COLS + ["date"]]
    last7[city] = rows.fillna(0).to_dict("records")

with open("models/city_last7.json", "w") as f:
    json.dump(last7, f, default=str)

print("Training complete.")
```

---

### Step 04 — Explainability (`src/04_explain.py`)

**Input:** `models/best_model.pkl`, `data/processed/features.parquet`, `models/feature_columns.json`
**Output:** all PNGs and CSVs in `outputs/`

#### 4.1 SHAP summary beeswarm
```python
import shap, joblib, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

model = joblib.load("models/best_model.pkl")
df    = pd.read_parquet("data/processed/features.parquet")
with open("models/feature_columns.json") as f:
    FEATURE_COLUMNS = json.load(f)

test    = df[df["split"] == "test"]
X_test  = test[FEATURE_COLUMNS].values

explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, feature_names=FEATURE_COLUMNS,
                  show=False, max_display=25)
plt.tight_layout()
plt.savefig("outputs/shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved shap_summary.png")
```

#### 4.2 Waterfall plots for top 3 predicted spike events
```python
preds = model.predict(X_test)
top3  = np.argsort(preds)[-3:][::-1]

for i, idx in enumerate(top3, 1):
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[idx],
            base_values=explainer.expected_value,
            data=X_test[idx],
            feature_names=FEATURE_COLUMNS,
        ),
        show=False, max_display=15
    )
    plt.tight_layout()
    plt.savefig(f"outputs/shap_waterfall_spike{i}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved shap_waterfall_spike{i}.png  (predicted={preds[idx]:.1f} µg/m³)")
```

#### 4.3 Cross-city influence heatmap (5×5 SHAP-based)
```python
CITIES = ["Delhi", "Bengaluru", "Kolkata", "Hyderabad"]
influence = np.zeros((5, 5))

for i, target_city in enumerate(CITIES):
    city_mask  = (test["city"] == target_city).values
    city_shap  = shap_values[city_mask]
    for j, neighbor in enumerate(CITIES):
        if i == j:
            continue
        col = f"neighbor_{neighbor.lower()}_pm2_5_lag1"
        if col in FEATURE_COLUMNS:
            col_idx = FEATURE_COLUMNS.index(col)
            influence[i, j] = np.abs(city_shap[:, col_idx]).mean()

plt.figure(figsize=(7, 5))
sns.heatmap(influence, xticklabels=CITIES, yticklabels=CITIES,
            annot=True, fmt=".2f", cmap="YlOrRd")
plt.title("Cross-city PM2.5 influence (mean |SHAP|)")
plt.xlabel("Neighbor city (source of influence)")
plt.ylabel("Target city (being predicted)")
plt.tight_layout()
plt.savefig("outputs/cross_city_influence.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved cross_city_influence.png")
```

#### 4.4 Lag importance bar chart
```python
lag_cols = ["pm2_5_lag1","pm2_5_lag2","pm2_5_lag3","pm2_5_lag7"]
lag_imp  = {
    col: np.abs(shap_values[:, FEATURE_COLUMNS.index(col)]).mean()
    for col in lag_cols if col in FEATURE_COLUMNS
}

plt.figure(figsize=(6, 4))
plt.bar(lag_imp.keys(), lag_imp.values(), color="#e07b54")
plt.title("PM2.5 lag feature importance (mean |SHAP|)")
plt.ylabel("Mean |SHAP value|")
plt.tight_layout()
plt.savefig("outputs/lag_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved lag_importance.png")
```

---

### Step 05 — Live Data Ingestion (`src/05_live_ingest.py`)

**Output:** `data/live/latest_reading.json`

```python
import os, requests, json
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
TOKEN = os.getenv("WAQI_TOKEN")
if not TOKEN:
    raise EnvironmentError("WAQI_TOKEN not set in .env")

CITY_SLUGS = {
    "Delhi":     "delhi",
    "Mumbai":    "mumbai",
    "Bengaluru": "bengaluru",
    "Kolkata":   "kolkata",
    "Hyderabad": "hyderabad"
}

def fetch_city(slug):
    url = f"https://api.waqi.info/feed/{slug}/?token={TOKEN}"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    d = r.json()
    if d["status"] != "ok":
        raise ValueError(f"API error for {slug}: {d}")
    iaqi = d["data"]["iaqi"]
    return {
        "aqi":         d["data"]["aqi"],
        "pm2_5":       iaqi.get("pm25",  {}).get("v"),   # no underscore in API
        "pm10":        iaqi.get("pm10",  {}).get("v"),
        "co":          iaqi.get("co",    {}).get("v"),
        "no2":         iaqi.get("no2",   {}).get("v"),
        "o3":          iaqi.get("o3",    {}).get("v"),
        "so2":         iaqi.get("so2",   {}).get("v"),
        "temperature": iaqi.get("t",     {}).get("v"),
        "wind_speed":  iaqi.get("w",     {}).get("v"),
        "pressure":    iaqi.get("p",     {}).get("v"),
        "rainfall":    0.0,
        "fetched_at":  datetime.utcnow().isoformat(),
        "api_timestamp": d["data"]["time"]["s"],
    }

os.makedirs("data/live", exist_ok=True)
readings = {}
for city, slug in CITY_SLUGS.items():
    try:
        readings[city] = fetch_city(slug)
        print(f"  ✓ {city}: PM2.5={readings[city]['pm2_5']}")
    except Exception as e:
        print(f"  ✗ {city}: {e}")
        readings[city] = None

with open("data/live/latest_reading.json", "w") as f:
    json.dump(readings, f, indent=2)
print("Saved data/live/latest_reading.json")
```

---

### Step 06 — Live Prediction (`src/06_predict_live.py`)

**Input:** `data/live/latest_reading.json`, `models/best_model.pkl`, `models/feature_scaler.pkl`, `models/feature_columns.json`, `models/city_last7.json`
**Output:** `data/live/predictions.json`

```python
import joblib, json, os
import numpy as np
from datetime import date
from math import radians, sin, cos, sqrt, atan2

model  = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/feature_scaler.pkl")
with open("models/feature_columns.json") as f:
    FEATURE_COLUMNS = json.load(f)
with open("models/city_last7.json") as f:
    city_last7 = json.load(f)
with open("data/live/latest_reading.json") as f:
    live = json.load(f)

CITIES    = ["Delhi", "Bengaluru", "Kolkata", "Hyderabad"]
CITY_TO_IDX = {c: i for i, c in enumerate(CITIES)}
CITY_COORDS = {
    "Delhi":     (28.6139, 77.2090),
    "Bengaluru": (12.9716, 77.5946),
    "Kolkata":   (22.5726, 88.3639),
    "Hyderabad": (17.3850, 78.4867),
}

def haversine(c1, c2):
    R = 6371
    lat1, lon1 = map(radians, c1)
    lat2, lon2 = map(radians, c2)
    dlat, dlon = lat2-lat1, lon2-lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

dist_weights = {
    (c1, c2): round(np.exp(-haversine(CITY_COORDS[c1], CITY_COORDS[c2]) / 500), 4)
    for c1 in CITIES for c2 in CITIES if c1 != c2
}

today = date.today()
predictions = {}

for city in CITIES:
    reading = live.get(city)
    if reading is None:
        predictions[city] = None
        continue

    # Lag window: last 7 raw historical rows, replace most recent with today's live
    window = city_last7[city].copy()
    window[-1] = reading
    pm_series  = [float(r.get("pm2_5")  or 0.0) for r in window]
    aqi_series = [float(r.get("aqi")    or 0.0) for r in window]

    def g(key, fallback=0.0):
        v = reading.get(key)
        return float(v) if v is not None else fallback

    features = {
        "aqi":              g("aqi"),
        "co":               g("co"),
        "no2":              g("no2"),
        "o3":               g("o3"),
        "so2":              g("so2"),
        "pm2_5":            g("pm2_5"),
        "pm10":             g("pm10"),
        "temperature":      g("temperature", 25.0),
        "wind_speed":       g("wind_speed"),
        "rainfall":         0.0,
        "pressure":         g("pressure", 1013.0),
        "pm2_5_lag1":       pm_series[-1],
        "pm2_5_lag2":       pm_series[-2],
        "pm2_5_lag3":       pm_series[-3],
        "pm2_5_lag7":       pm_series[0],
        "aqi_lag1":         aqi_series[-1],
        "aqi_lag2":         aqi_series[-2],
        "aqi_lag3":         aqi_series[-3],
        "aqi_lag7":         aqi_series[0],
        "pm2_5_roll3mean":  np.mean(pm_series[-3:]),
        "pm2_5_roll7mean":  np.mean(pm_series),
        "pm2_5_roll3std":   float(np.std(pm_series[-3:])),
        "pm2_5_roll7std":   float(np.std(pm_series)),
        "dayofweek":        today.weekday(),
        "month":            today.month,
        "is_weekend":       int(today.weekday() >= 5),
        "quarter":          (today.month - 1) // 3 + 1,
        "crop_burning_season": int(today.month in [10, 11]),
        "monsoon_season":      int(today.month in [6, 7, 8, 9]),
        "city_idx":         CITY_TO_IDX[city],
    }

    # Cross-city neighbor features from today's live readings
    for neighbor in CITIES:
        if neighbor == city:
            continue
        w  = dist_weights[(city, neighbor)]
        nr = live.get(neighbor)
        key = neighbor.lower()
        features[f"neighbor_{key}_pm2_5_lag1"] = \
            float(nr.get("pm2_5") or 0.0) * w if nr else 0.0
        features[f"neighbor_{key}_wind_lag1"] = \
            float(nr.get("wind_speed") or 0.0) * w if nr else 0.0

    X        = np.array([[features.get(col, 0.0) for col in FEATURE_COLUMNS]])
    X_scaled = scaler.transform(X)
    pred     = float(model.predict(X_scaled)[0])
    predictions[city] = round(pred, 2)

# Print results
print("\nNext-day PM2.5 predictions:")
for city, pred in predictions.items():
    current = live.get(city, {}) or {}
    print(f"  {city:12s}  Current={current.get('pm2_5','N/A')}  Tomorrow={pred} µg/m³")

os.makedirs("data/live", exist_ok=True)
with open("data/live/predictions.json", "w") as f:
    json.dump({"date": str(today), "predictions": predictions}, f, indent=2)
```

---

### Step 07 — Streamlit App (`app/streamlit_app.py`)

**Run:** `uv run streamlit run app/streamlit_app.py`

**Page layout:**

```
Title: AirMind — Next-Day PM2.5 Forecast for Indian Cities

Sidebar:
  - [Refresh Live Data] button → subprocess: 05_live_ingest.py then 06_predict_live.py
  - City multiselect (default: all 5)

Main area:
  Tab 1: Live Forecast
    - 5 metric cards: city name | current PM2.5 | predicted tomorrow | AQI category badge
    - Colour-coded by India NAQI standard

  Tab 2: Historical Performance
    - Line chart: actual vs predicted PM2.5 on test split, per selected city
    - Metric row: MAE | RMSE | R² loaded from outputs/per_city_metrics.csv

  Tab 3: Explainability
    - st.image("outputs/shap_summary.png")
    - st.image("outputs/cross_city_influence.png")
    - st.image("outputs/lag_importance.png")
    - Selectbox: spike event 1/2/3 → shows matching waterfall PNG

  Tab 4: Model Info
    - st.dataframe(outputs/model_comparison.csv)
    - Feature count, training date range, architecture note
```

**India NAQI PM2.5 category mapping:**
```python
def pm25_category(val):
    if val is None:      return "N/A",        "#888888"
    if val <= 30:        return "Good",        "#00b050"
    elif val <= 60:      return "Satisfactory","#92d050"
    elif val <= 90:      return "Moderate",    "#ffff00"
    elif val <= 120:     return "Poor",        "#ff9900"
    elif val <= 250:     return "Very Poor",   "#ff0000"
    else:                return "Severe",      "#c00000"
```

---

## Execution Order

```
1.  uv init airmind && cd airmind
2.  uv add pandas numpy scikit-learn xgboost shap matplotlib seaborn \
        streamlit requests python-dotenv pyarrow joblib
3.  cp .env.example .env   # add WAQI_TOKEN
4.  uv run python src/02_feature_engineering.py
5.  uv run python src/03_train.py
6.  uv run python src/04_explain.py
7.  uv run python src/05_live_ingest.py
8.  uv run python src/06_predict_live.py
9.  uv run streamlit run app/streamlit_app.py
```

---

## Critical Implementation Rules

1. **Never use random train/test split** — always split by date cutoff
2. **Never fit the scaler on val or test data** — fit on train only, transform all splits
3. **Never scale the target** (`target_pm2_5`) — only scale input features
4. **Never use `pm2_5` at time t to predict time t** — only lags (t-1 and earlier)
5. **Never hardcode the API token** — always load from `.env` via `python-dotenv`
6. **Always use `.get()` with fallback when reading live API fields** — any field can be absent
7. **`city_last7.json` must contain raw unscaled values** — loaded from `merged.csv`, not `features.parquet`

### Reproducibility
```python
import random, numpy as np
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
```

---

## Evaluation Thresholds (Pass/Fail)

| Metric | Minimum | Target |
|--------|---------|--------|
| Test MAE | < 20 µg/m³ | < 10 µg/m³ |
| Test R² | > 0.65 | > 0.82 |
| XGBoost vs RF MAE improvement | > 5% | > 15% |
| Per-city R² worst case | > 0.55 | > 0.70 |

If thresholds not met: check for data leakage first (random split used? target scaled?). Then tune `max_depth` and `n_estimators` before changing the feature set.

---

## Known Data Quirks to Handle

- `pressure` values ~730 from the live API are reduced atmospheric pressure — different from historical dataset. Note the discrepancy but use consistently without converting.
- `Bengaluru` may appear as `Bangalore` in some rows — normalize on load.
- `pm2_5` values < 1.0 are sensor failure not true zero — treat as missing.
- Date column may have mixed formats — always use `pd.to_datetime(..., infer_datetime_format=True)`.
- Live API key is `pm25` (no underscore), internal column is `pm2_5` (with underscore) — map carefully in Step 05.

---

## XAI Deliverables (Required for Research Report)

1. **Figure 1:** `outputs/shap_summary.png` — beeswarm, all features, all test samples
2. **Figure 2–4:** `outputs/shap_waterfall_spike[1-3].png` — top 3 predicted spike events
3. **Figure 5:** `outputs/cross_city_influence.png` — 5×5 SHAP mean |value| heatmap
4. **Figure 6:** `outputs/lag_importance.png` — lag day importance bar chart
5. **Table 1:** `outputs/per_city_metrics.csv` — MAE, RMSE, R² per city
6. **Table 2:** `outputs/model_comparison.csv` — XGBoost vs RandomForest baseline

---

*AirMind project. Dataset: merged.csv, 5 Indian cities. Model: XGBoost with explicit cross-city spatial features. Live API: WAQI. Last updated: April 2026.*