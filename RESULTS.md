# RESULTS — AirMind (Air Quality Prediction)

**Repository:** `kanishjn/air-quality-prediction`

**Branch:** `joler`

**Date:** 16 Apr 2026

This document captures the results obtained during the latest end-to-end pipeline runs (feature engineering → training → explainability → dashboard integration). It also records important context about dataset versions, split strategies, overfitting checks, and the final artifacts generated.

---

## 0) What changed vs earlier runs

### Dataset update (new `merged.csv`)
A new merged dataset was created from pollution + weather sources:
- **File used for training:** `data/processed/merged.csv`
- **Shape:** 11824 rows × 14 columns
- **Cities:** 5 (`bengaluru`, `delhi`, `hyderabad`, `kolkata`, `mumbai`)
- **Date range:** 2015-01-01 → 2023-05-25
- **Duplicates:** 0 duplicate `(city, date)` rows

Columns present in the new merge:
- `city`, `date`
- pollutants: `aqi`, `co`, `no2`, `o3`, `so2`, `pm2_5`, `pm10`
- weather: `temperature`, `wind_speed`, `rainfall`, `pressure`
- (plus an extra index-like column sometimes: `Unnamed: 0`)

**Important:** This new merged file is *not identical* to the older Plan vNext expected schema (which included `no` and `nh3`). The feature engineering step was updated to tolerate missing pollutant channels by creating them as constant zeros so training/inference schema stays stable.

---

## 1) Data EDA highlights (quick, actionable)

### 1.1 Coverage per city
From `data/processed/merged.csv`:
- Bengaluru: 2767 rows, 2767 unique dates, span ~2989 days
- Delhi: 2911 rows, 2911 unique dates, span ~3067 days
- Hyderabad: 2795 rows, 2795 unique dates, span ~2978 days
- Kolkata: 1663 rows, 1663 unique dates, span ~1872 days
- Mumbai: 1688 rows, 1688 unique dates, span ~1856 days

### 1.2 PM2.5 distribution + spikes
PM2.5 is heavy-tailed:
- median (p50): ~51.37
- p90: ~232.27
- p95: ~354.70
- p99: ~832.34
- max: ~2203.55

### 1.3 Temporal predictability proxy
PM2.5 autocorrelation proxy (corr with lag-1 per city):
- bengaluru: ~0.616
- delhi: ~0.564
- hyderabad: ~0.656
- kolkata: ~0.625
- mumbai: ~0.745

Interpretation: there is usable next-day signal from recent history; extreme spikes still dominate absolute-error metrics.

---

## 2) Pipeline configuration used for the latest runs

### 2.1 Feature engineering (`src/02_feature_engineering.py`)
**Input:** `data/processed/merged.csv` (overridable via env: `AIR_MERGED_CSV`)

Key transformations:
- City normalization (case/title, Bangalore→Bengaluru)
- KNN imputation per city (`KNNImputer(n_neighbors=5)`) for numeric channels
- Lag features (per city):
  - PM2.5 lags: 1,2,3,7,14,21,28
  - AQI lags: 1,2,3,7,14,21,28
- Rolling features (shifted by 1 day):
  - PM2.5 rolling mean/std windows: 3,7,14
- Seasonality features:
  - `dayofweek`, `month`, `dayofyear`, `quarter`, `is_weekend`
  - `crop_burning_season` (Oct–Nov)
  - `monsoon_season` (Jun–Sep)
  - `doy_sin`, `doy_cos`
- Cross-city neighbor features (distance-decay weighted):
  - For each neighbor city:
    - `neighbor_{city}_pm2_5_lag1`
    - `neighbor_{city}_pm2_5_lag7`
    - `neighbor_{city}_wind_lag1`

Target definition:
- `target_pm2_5` = next-day PM2.5 = `pm2_5.shift(-1)` per city

Scaling:
- `StandardScaler` fit **only** on the train split
- Writes `*_scaled` feature columns for model input

Artifacts written:
- `data/processed/features.parquet`
- `models/feature_scaler.pkl`
- `models/feature_columns.json`

#### Latest split strategy (current state)
**Mode:** Train on all earlier years; validate on last year.
- Train: dates `< 2022-05-24`
- Val: dates `>= 2022-05-24` (last ~365 days)
- Test: **not used in this mode**

Resulting engineered dataset:
- `data/processed/features.parquet`: 11679 rows, 120 columns
- split counts: train=9864, val=1815
- feature schema size: 58 features (`models/feature_columns.json`)

Target distribution (engineered dataset):
- mean: ~101.69
- median: ~51.73
- p99: ~838.61
- max: ~2203.55

Per-city min/max dates (after warmup drops):
- Bengaluru: 2015-04-17 → 2023-05-24 (2738 rows)
- Delhi: 2015-01-29 → 2023-05-24 (2882 rows)
- Hyderabad: 2015-05-02 → 2023-05-24 (2766 rows)
- Kolkata: 2018-07-02 → 2023-05-24 (1634 rows)
- Mumbai: 2018-06-02 → 2023-05-24 (1659 rows)

---

## 3) Training configuration + results

### 3.1 Training script (`src/03_train.py`)
Modeling choices:
- **Single global model** (not per-city)
- Model: `XGBRegressor`
- Target transform: train on `log1p(target_pm2_5)`; predictions are `expm1()` back-transformed
- Outlier handling: clip training targets at 99.5th percentile prior to log transform
- Recency weighting: linear weights from 0.6 → 1.6 over time
- Early stopping:
  - Uses explicit `val` split as `eval_set`
  - `early_stopping_rounds=75`

Baselines:
- RandomForest baseline trained on a reduced feature set **excluding** `neighbor_*` columns

Artifacts written:
- `models/best_model.pkl`
- `outputs/per_city_metrics.csv`
- `outputs/model_comparison.csv`
- `outputs/predictions.csv`
- `models/city_last7.json`

### 3.2 Note about tuning loop runtime
The time-series tuning loop can run long.
- Recommended fast run: set `AIR_TUNE=0`.
- CV fitting now supports early stopping inside folds.

---

## 4) Metrics — most recent run (train on earlier years, val = last-year)

These numbers are from running:
- `AIR_TUNE=0 python src/03_train.py`

### 4.1 Global metrics (XGBoost vs RF)
From `outputs/model_comparison.csv`:

| model | MAE | RMSE | R2 |
|------|-----:|-----:|----:|
| XGBoost (with cross-city features) | 17.39 | 57.10 | 0.940 |
| RandomForest (no cross-city features) | 76.33 | 162.36 | 0.515 |

### 4.2 Per-city metrics (report split)
From `outputs/per_city_metrics.csv`:

| city | MAE | RMSE | R2 | n |
|------|----:|-----:|----:|---:|
| Bengaluru | 2.62 | 4.87 | 0.971 | 363 |
| Delhi | 23.26 | 69.42 | 0.895 | 363 |
| Hyderabad | 8.16 | 16.83 | 0.968 | 363 |
| Kolkata | 41.08 | 103.39 | 0.922 | 363 |
| Mumbai | 11.83 | 22.08 | 0.979 | 363 |

### 4.3 Overfitting confirmation (train vs val)
`src/03_train.py` was updated to print TRAIN/VAL/TEST-REPORT blocks.

Observed:

**TRAIN**
- XGBoost: MAE 10.77, RMSE 40.19, R² 0.918

**VAL (last-year)**
- XGBoost: MAE 17.39, RMSE 57.10, R² 0.940

Interpretation:
- There is a **MAE generalization gap** (train < val), which is normal.
- R² being higher on VAL can happen when VAL has higher variance than TRAIN.
- **Important caveat:** in the current split mode there is no independent test year; VAL is also used for early stopping, so the reported VAL performance is not a fully unbiased deployment estimate.

---

## 5) Prediction output used by dashboard

`outputs/predictions.csv`
- rows: 1815
- columns: `city`, `date`, `actual_pm2_5`, `predicted_pm2_5`
- date range: 2022-05-24 → 2023-05-24

This file is used by the Streamlit **Historical** tab.

---

## 6) Explainability outputs

Explainability step executed:
- `uv run src/04_explain.py` (latest run completed successfully)

Images present in `outputs/`:
- `shap_summary.png`
- `cross_city_influence.png`
- `lag_importance.png`
- `shap_dependence.png`
- `shap_waterfall_1.png`, `shap_waterfall_2.png`, `shap_waterfall_3.png`
- `shap_waterfall_spike1.png`, `shap_waterfall_spike2.png`, `shap_waterfall_spike3.png`
- `training_curves.png`
- (legacy/older images may exist: `gnn_edge_importance.png`, `temporal_lag_importance.png`)

The Streamlit app auto-discovers and renders these in the Explainability tab.

---

## 7) Live pipeline artifacts

Live ingest outputs:
- `data/live/latest_reading.json` (most recent timestamp: Apr 16 19:25)

Live inference outputs:
- `data/live/predictions.json` (most recent timestamp: Apr 16 19:25)

Dashboard reads these under the **Live Forecast** tab.

---

## 8) Streamlit dashboard updates done during this session

File: `app/streamlit_app.py`

Changes captured:
- Added Mumbai to the city list and styling accent map.
- Historical tab improvements:
  - Metrics table (computed from `outputs/predictions.csv` if the metrics file is missing)
  - Diagnostics plots:
    - Actual vs Predicted scatter + 45° line
    - Residual histogram
    - Error by PM2.5 quantile bins chart
- Explainability tab already supports auto-discovery of `outputs/*.png`.

---

## 9) Notes / Known caveats

1. **Split strategy caveat**
   - With only train+val, the reported numbers are not a true future holdout.
   - Recommended for unbiased evaluation: keep a separate last-year **test**.

2. **City coverage imbalance**
   - Kolkata and Mumbai coverage starts in 2018, while Delhi has data from early 2015.
   - This can make “all-years” training dominated by Delhi/Bengaluru/Hyderabad.

3. **Heavy-tailed target**
   - Extreme spikes (PM2.5 > 800, max ~2203) strongly influence RMSE and can distort perceived performance. Log-target training mitigates but does not remove it.

---

## 10) Files produced / modified (high level)

### Produced artifacts
- `data/processed/features.parquet`
- `models/feature_scaler.pkl`
- `models/feature_columns.json`
- `models/best_model.pkl`
- `models/city_last7.json`
- `outputs/model_comparison.csv`
- `outputs/per_city_metrics.csv`
- `outputs/predictions.csv`
- `outputs/*.png` explainability plots
- `data/live/latest_reading.json`
- `data/live/predictions.json`

### Code updated
- `src/02_feature_engineering.py` (data source, robust schema handling, split logic, extra temporal/neighbor features, scaling skip for missing splits)
- `src/03_train.py` (early stopping, optional tuning via `AIR_TUNE`, train/val/test reporting)
- `app/streamlit_app.py` (Mumbai + new diagnostics/metrics)

---

## Appendix A) Exact metric dumps (raw)

From `outputs/model_comparison.csv`:

```
XGBoost (with cross-city features): MAE=17.39  RMSE=57.10  R2=0.940
RandomForest (no cross-city features): MAE=76.33  RMSE=162.36  R2=0.515
```

From `outputs/per_city_metrics.csv`:

```
Bengaluru: MAE=2.62  RMSE=4.87  R2=0.971  n=363
Delhi:     MAE=23.26 RMSE=69.42 R2=0.895  n=363
Hyderabad: MAE=8.16  RMSE=16.83 R2=0.968  n=363
Kolkata:   MAE=41.08 RMSE=103.39 R2=0.922 n=363
Mumbai:    MAE=11.83 RMSE=22.08 R2=0.979 n=363
```
