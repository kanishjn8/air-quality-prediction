# 🌫️ AirMind — Spatiotemporal PM2.5 Forecasting

**Next-day PM2.5 prediction for Indian cities using an XGBoost model with explicit cross-city neighbor features and SHAP-based interpretability.**

> Research gaps addressed:
> 1. Cross-city pollutant dependency (encoded as explicit neighbor features)
> 2. Missing data bias from structured sensor failure (MCAR vs MNAR handling)

---

## 🏙️ Cities Covered

| City | Coordinates |
|------|-------------|
| Delhi | 28.6139°N, 77.2090°E |
| Mumbai | 19.0760°N, 72.8777°E |
| Bengaluru | 12.9716°N, 77.5946°E |
| Kolkata | 22.5726°N, 88.3639°E |
| Hyderabad | 17.3850°N, 78.4867°E |

> Note: Both training and live inference run on all 5 cities.

---

## 📁 Project Structure

```
airmind/
├── data/
│   ├── raw/
│   │   └── merged.csv                  # historical air quality data (DO NOT modify)
│   ├── processed/
│   │   ├── features.parquet            # output of 02_feature_engineering.py
│   │   └── (no graph artifacts in plan-vNext)
│   └── live/
│       ├── latest_reading.json         # written by 05_live_ingest.py
│       └── predictions.json            # written by 06_predict_live.py
├── src/
│   ├── 02_feature_engineering.py       # data cleaning, imputation, feature creation
│   ├── 03_train.py                     # XGBoost training
│   ├── 04_explain.py                   # SHAP explainability (XGBoost)
│   ├── 05_live_ingest.py               # WAQI API data fetching
│   └── 06_predict_live.py              # live inference
├── models/
│   ├── best_model.pkl                  # trained XGBoost model
│   ├── feature_scaler.pkl              # StandardScaler (fitted on train)
│   ├── feature_columns.json            # ordered feature schema saved by 02_feature_engineering.py
│   └── city_last7.json                 # last-7 per-city raw rows saved by 03_train.py (inference)
├── outputs/
│   ├── shap_summary.png                # SHAP feature importance
│   ├── cross_city_influence.png        # cross-city influence heatmap
│   ├── lag_importance.png              # lag day attribution
│   ├── predictions.csv                 # test set predictions
│   ├── per_city_metrics.csv            # per-city evaluation
│   └── model_comparison.csv            # XGBoost vs RF comparison
├── app/
│   └── streamlit_app.py                # interactive dashboard
├── notebooks/
│   └── 01_eda.ipynb                    # (optional) exploratory analysis
├── pyproject.toml                      # uv-managed dependencies
├── .env.example                        # environment variable template
├── .gitignore
├── plan.md                             # full project specification
└── README.md                           # ← you are here
```

---

## ⚠️ Deprecated legacy scripts

The canonical runnable pipeline is:

- `src/02_feature_engineering.py`
- `src/03_train.py`
- `src/04_explain.py`
- `src/05_live_ingest.py`
- `src/06_predict_live.py`

Older torch-era scripts still exist only as **deprecated stubs** (they exit immediately if run):

- `src/03_graph_construction.py`
- `src/04_train.py`
- `src/05_explain.py`
- `src/06_live_ingest.py`
- `src/07_predict_live.py`

Archived historical versions live in `legacy/src_torch/`.

## 🚀 Quick Start

### Prerequisites

- **Python** ≥ 3.11
- **uv** (Python package manager) — [install guide](https://docs.astral.sh/uv/getting-started/installation/)

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 1. Clone & Setup Environment

```bash
git clone <your-repo-url>
cd Pm_monitor

# Install all dependencies (uv reads pyproject.toml automatically)
uv sync
```

### 2. Configure Environment Variables

```bash
# Copy the template
cp .env.example .env

# Edit .env and add your WAQI API token
# Get a free token at: https://aqicn.org/data-platform/token/
```

Your `.env` file should contain:
```
WAQI_TOKEN=your_actual_token_here
```

### 3. Run the Pipeline (in order)

Each step reads the previous step's output. **Run them sequentially:**

```bash
# Step 1: Feature Engineering (clean data, impute, create features)
uv run python src/02_feature_engineering.py

# Step 2: Model Training (train XGBoost)
uv run python src/03_train.py

# Step 3: Explainability (generate SHAP plots)
uv run python src/04_explain.py

# Step 4: Live Data Ingestion (fetch current readings from WAQI API)
uv run python src/05_live_ingest.py

# Step 5: Live Prediction (predict tomorrow's PM2.5)
uv run python src/06_predict_live.py
```

### 4. Launch the Dashboard

```bash
uv run streamlit run app/streamlit_app.py
```

The dashboard will open at **http://localhost:8501** with 4 tabs:
- **📡 Live Forecast** — Current readings + tomorrow's prediction per city
- **📈 Historical Trends** — Actual vs predicted PM2.5 during test period
- **🔍 Explainability** — SHAP plots, city influence matrix, temporal lags
- **🏗️ Model Info** — Architecture, training config, data provenance

---

## 🧠 Model Architecture

**Plan vNext model:** XGBoost regressor over tabular features:

- In-city lag/rolling features (PM2.5, AQI)
- Calendar/seasonality features
- Explicit cross-city neighbor features (`neighbor_*_pm2_5_lag1`, `neighbor_*_wind_lag1`)

### Training strategy (performance-focused, ML-only)

- `log1p(target_pm2_5)` training target to stabilize heavy right-skew
- Time-series cross-validation (`TimeSeriesSplit`) on train+val to tune XGBoost hyperparameters
- Stable XGBoost objective: `reg:squarederror` with outlier-robust target clipping
- Recency sample weighting so newer periods influence the model more than older periods
- Fold-wise target clipping at 99.5th percentile during tuning to avoid overfitting rare outliers
- Final model fit on combined train+val with tuned params; test split remains untouched

---

## 📊 Evaluation Targets

| Metric | Minimum | Target |
|--------|---------|--------|
| Test MAE | < 20 µg/m³ | < 10 µg/m³ |
| Test R² | > 0.65 | > 0.80 |
| XGBoost vs RF MAE improvement | > 5% | > 15% |
| Per-city R² (worst) | > 0.55 | > 0.70 |

---

## 📡 Live API

This project uses the **WAQI (World Air Quality Index)** API for real-time data.

- **Endpoint:** `https://api.waqi.info/feed/{city}/?token={WAQI_TOKEN}`
- **Free token:** [Sign up here](https://aqicn.org/data-platform/token/)
- **Fields used:** pm25, pm10, co, no2, o3, so2, temperature, wind_speed, pressure, humidity, dew_point
- **Missing from API (imputed):** `no`, `nh3`, `rainfall`

### Live lag features (important)

The model uses lag/rolling features (e.g., `pm2_5_lag2`, `pm2_5_roll7mean`).

In plan-vNext, inference uses `models/city_last7.json` (saved during training) to build lag/rolling
features, plus the current live batch to compute cross-city neighbor features.

---

## 🔬 XAI Outputs

After running `04_explain.py`, the following artifacts are generated in `outputs/`:

1. **`shap_summary.png`** — SHAP beeswarm plot (all features ranked by importance)
2. **`shap_waterfall_spike1.png`** to **`shap_waterfall_spike3.png`** — Top 3 PM2.5 spike event explanations
3. **`cross_city_influence.png`** — 5x5 cross-city influence heatmap
4. **`lag_importance.png`** — Lag feature contribution chart

---

## ⚠️ Known Data Quirks

- `pressure` may be in reduced form (~730 hPa instead of ~1013 hPa) — noted in preprocessing
- `Bengaluru` may appear as `Bangalore` in some rows — normalized during loading
- `pm2_5 < 1.0` treated as sensor failure → set to NaN and imputed
- Date column may have mixed formats — parsed with `infer_datetime_format=True`
- API's `pm25` key ≠ our `pm2_5` column — mapped carefully in Step 06
- Mumbai remains in the production city set for both training and inference to keep the feature graph and deployment schema consistent.

---

## 🗂️ Dependencies

All managed via `uv` (see `pyproject.toml`):

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | 2.2.2 | Data manipulation |
| numpy | 1.26.4 | Numerical computing |
| scikit-learn | 1.4.2 | KNN imputation, RF baseline, metrics |
| xgboost | 2.0.3 | XGBoost regressor |
| shap | 0.45.1 | Explainability |
| matplotlib | 3.8.4 | Plotting |
| seaborn | 0.13.2 | Statistical visualization |
| streamlit | 1.35.0 | Interactive dashboard |
| requests | 2.32.2 | WAQI API calls |
| python-dotenv | 1.0.1 | Environment variable management |
| pyarrow | 16.0.0 | Parquet I/O |
| joblib | 1.4.2 | Model serialization |
| scipy | ≥1.12.0 | Statistical tests (Spearman, etc.) |

---

## 📋 Troubleshooting

### `uv sync` fails with dependency conflicts
```bash
# Try removing the lock file and re-syncing
rm uv.lock
uv sync
```

### `WAQI_TOKEN not set` error
```bash
# Make sure .env exists and has your token
cat .env
# Should show: WAQI_TOKEN=your_actual_token
```

### Training is slow
- The model uses CPU by default. Training ~100 epochs on CPU takes ~5–15 minutes depending on your machine.
- (Optional) If you have a CUDA GPU, XGBoost may use it if configured; this repo defaults to CPU.

### SHAP takes too long
- `04_explain.py` subsamples the evaluation set for speed. Adjust the `max_samples` constant in the script if needed.

### Streamlit won't start
```bash
# Make sure you're running from the project root
uv run streamlit run app/streamlit_app.py

# If port 8501 is in use:
uv run streamlit run app/streamlit_app.py --server.port 8502
```

---

## 📜 License

This project is for research and educational purposes.

---

*Built with ❤️ for cleaner air — AirMind v1.0*
