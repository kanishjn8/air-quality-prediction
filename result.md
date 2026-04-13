# 📊 AirMind Training Results

This document contains the detailed evaluation results of the final XGBoost architecture trained on historical PM2.5 data (2020–2023).

## 1. Overall Model Comparison

The final model uses a single XGBoost regressor trained on log-transformed targets `log1p(PM2.5)` to handle extreme right-skewness in the data distributions (especially from Kolkata and Delhi). A Random Forest is provided as a baseline ablation (trained without cross-city neighbor features).

| Model | MAE (µg/m³) | RMSE (µg/m³) | R² |
|-------|------------|-------------|-----|
| **XGBoost (with cross-city features)** | **87.38** | **176.73** | **0.411** |
| RandomForest (no cross-city features) | 87.26 | 178.19 | 0.402 |

*Note: The overall R² is artificially depressed due to extreme outlier events in Kolkata (PM2.5 reaching > 2000 µg/m³).*

---

## 2. Per-City Performance (Test Split)

Due to vast differences in geographical, meteorological, and anthropogenic factors, the model's accuracy varies significantly across the Indian subcontinent.

| City | MAE (µg/m³) | RMSE (µg/m³) | R² | Test Samples (n) |
|------|-------------|--------------|----|-----------------|
| **Bengaluru** | 15.08 | 21.35 | 0.079 | 136 |
| **Hyderabad** | 51.68 | 85.89 | 0.136 | 136 |
| **Mumbai** | 66.72 | 97.08 | 0.483 | 136 |
| **Delhi** | 99.75 | 166.79 | 0.176 | 136 |
| **Kolkata** | 203.68 | 333.30 | 0.194 | 136 |

### Analysis of City Differences:

- **Bengaluru:** Has the lowest absolute error (MAE = 15.08). The PM2.5 levels are generally low and stable, meaning the model tracks it well on average, though the R² is low simply because there is very little variance to explain.
- **Delhi & Kolkata:** Suffer from extreme seasonal variability and massive localized emission spike events (crop burning, winter inversions). Kolkata specifically has uncharacteristically extreme outliers in the raw data (surpassing 2200 µg/m³) making mean absolute error very high.
- **Mumbai:** Shown to be inherently difficult to rely on for live inference despite a mathematically decent R² during test splits. (See README for details on Mumbai's exclusion).
