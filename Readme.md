# 🌫️ AirMind — Spatiotemporal PM2.5 Forecasting

**Next-day PM2.5 prediction for Indian cities using a Graph Neural Network + LSTM + MLP fusion model with full XAI interpretability.**

> Research gaps addressed:
> 1. Cross-city pollutant dependency (modelled via graph edges)
> 2. Missing data bias from structured sensor failure (MCAR vs MNAR handling)

---

## 🏙️ Cities Covered

| City | Coordinates |
|------|-------------|
| Delhi | 28.6139°N, 77.2090°E |
| Bengaluru | 12.9716°N, 77.5946°E |
| Kolkata | 22.5726°N, 88.3639°E |
| Hyderabad | 17.3850°N, 78.4867°E |

> Note: Mumbai is currently excluded from the pipeline due to unstable live feed / data quality.

---

## 📁 Project Structure

```
airmind/
├── data/
│   ├── raw/
│   │   └── merged.csv                  # historical air quality data (DO NOT modify)
│   ├── processed/
│   │   ├── features.parquet            # output of 02_feature_engineering.py
│   │   ├── graph_edges.csv             # output of 03_graph_construction.py
│   │   └── adj_matrix.npy             # adjacency matrix for GNN
│   └── live/
│       ├── latest_reading.json         # written by 06_live_ingest.py
│       └── live_history.json           # rolling 7-day buffer built from live ingests (used for lag features)
├── src/
│   ├── 02_feature_engineering.py       # data cleaning, imputation, feature creation
│   ├── 03_graph_construction.py        # spatial + correlation graph
│   ├── 04_train.py                     # model training + RF baseline
│   ├── 05_explain.py                   # SHAP, GNN edges, temporal attribution
│   ├── 06_live_ingest.py               # WAQI API data fetching
│   └── 07_predict_live.py              # live inference
├── models/
│   ├── best_model.pt                   # trained GNN+LSTM model
│   ├── feature_scaler.pkl              # StandardScaler (fitted on train)
│   ├── feature_columns.json            # ordered feature schema saved by 02_feature_engineering.py
│   ├── city_last7.json                 # last-7 per-city raw rows saved by 04_train.py (fallback)
│   └── baseline_rf.pkl                 # Random Forest baseline
├── outputs/
│   ├── shap_summary.png                # SHAP feature importance
│   ├── gnn_edge_importance.png         # cross-city influence heatmap
│   ├── temporal_lag_importance.png      # lag day attribution
│   ├── training_curves.png             # loss + MAE curves
│   ├── predictions.csv                 # test set predictions
│   ├── city_metrics.csv                # per-city evaluation
│   └── model_comparison.csv            # GNN vs RF comparison
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

# Step 2: Graph Construction (build spatial + correlation graph)
uv run python src/03_graph_construction.py

# Step 3: Model Training (train GNN+LSTM model + RF baseline)
uv run python src/04_train.py

# Step 4: Explainability (generate SHAP plots, heatmaps)
uv run python src/05_explain.py

# Step 5: Live Data Ingestion (fetch current readings from WAQI API)
uv run python src/06_live_ingest.py

# Step 6: Live Prediction (predict tomorrow's PM2.5)
uv run python src/07_predict_live.py
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

```
GNN Branch (GraphSAGE-style, 2 layers)
    ↓ city embedding [4 × 64]
    
Temporal Branch (LSTM, 2 layers, hidden=128)
    ↓ sequence embedding [batch × 128]
    
Met Branch (MLP, 2 layers)
    ↓ met embedding [batch × 32]
    
→ Concatenate [224] → MLP Head → scalar PM2.5 prediction
```

**Key design choices:**
- **HuberLoss (δ=10):** Robust to PM2.5 spike outliers
- **Time-based splitting:** Train (before Oct 2022), Val (Oct 2022 – Jan 2023), Test (Feb 2023+) — no temporal leakage
- **Per-city KNN imputation (k=5):** MCAR missing data handled within each city group
- **Forward-fill for MNAR:** Structured sensor failures during pollution spikes

---

## 📊 Evaluation Targets

| Metric | Minimum | Target |
|--------|---------|--------|
| Test MAE | < 20 µg/m³ | < 10 µg/m³ |
| Test R² | > 0.65 | > 0.80 |
| GNN vs RF MAE improvement | > 5% | > 15% |
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

To keep these features *time-consistent* during live inference, each run of `src/06_live_ingest.py` appends today's live readings into:

- `data/live/live_history.json`

Then `src/07_predict_live.py`:

1. **Prefers** the most recent 7 entries from `data/live/live_history.json` for each city's lag window.
2. **Falls back** to `models/city_last7.json` only until you have collected 7 live observations.

This avoids mixing a 2026 “current” reading with lag windows computed from the tail of the historical (2023) dataset.

---

## 🔬 XAI Outputs

After running `05_explain.py`, the following artifacts are generated in `outputs/`:

1. **`shap_summary.png`** — SHAP beeswarm plot (all features ranked by importance)
2. **`shap_waterfall_1..3.png`** — Detailed SHAP for top 3 PM2.5 spike events
3. **`shap_dependence.png`** — pm2_5_lag1 × wind_speed interaction
4. **`gnn_edge_importance.png`** — cross-city influence heatmap
5. **`temporal_lag_importance.png`** — Which lag days matter most (high pollution vs normal)

---

## ⚠️ Known Data Quirks

- `pressure` may be in reduced form (~730 hPa instead of ~1013 hPa) — noted in preprocessing
- `Bengaluru` may appear as `Bangalore` in some rows — normalized during loading
- `pm2_5 < 1.0` treated as sensor failure → set to NaN and imputed
- Date column may have mixed formats — parsed with `infer_datetime_format=True`
- API's `pm25` key ≠ our `pm2_5` column — mapped carefully in Step 06

---

## 🗂️ Dependencies

All managed via `uv` (see `pyproject.toml`):

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | 2.2.2 | Data manipulation |
| numpy | 1.26.4 | Numerical computing |
| scikit-learn | 1.4.2 | KNN imputation, RF baseline, metrics |
| torch | 2.3.0 | GNN + LSTM model |
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
- If you have a CUDA GPU, PyTorch will auto-detect it.

### SHAP takes too long
- `05_explain.py` subsamples to 500 test points for speed. Adjust `max_samples` in the script if needed.

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
