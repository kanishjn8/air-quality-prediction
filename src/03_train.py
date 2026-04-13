"""Step 03 — Train XGBoost model (Plan vNext)

Input:
  - data/processed/features.parquet
  - models/feature_columns.json

Output:
  - models/best_model.pkl
  - models/city_last7.json
  - outputs/per_city_metrics.csv
  - outputs/model_comparison.csv
  - outputs/predictions.csv

Notes:
  - Uses scaled feature columns produced by Step 02 ("*_scaled").
  - Trains on log1p(target) to handle right-skewed PM2.5 distribution.
  - `city_last7.json` stores RAW (unscaled) last 7 days per city, used at inference.
  - Random Forest baseline trained WITHOUT cross-city features for ablation.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

FEATURES_PATH = "data/processed/features.parquet"
FEATURE_COLS_PATH = "models/feature_columns.json"
RAW_CSV_PATH = "data/raw/merged.csv"

MODEL_PATH = "models/best_model.pkl"
CITY_LAST7_PATH = "models/city_last7.json"

OUT_PER_CITY = "outputs/per_city_metrics.csv"
OUT_MODEL_COMP = "outputs/model_comparison.csv"
OUT_PREDICTIONS = "outputs/predictions.csv"

SEED = 42


@dataclass
class Metrics:
    mae: float
    rmse: float
    r2: float


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else 0.0
    return Metrics(mae=mae, rmse=rmse, r2=r2)


def _recency_weights(dates: pd.Series, min_w: float = 0.6, max_w: float = 1.6) -> np.ndarray:
    """Return linear recency weights so recent samples matter more."""
    dt = pd.to_datetime(dates, errors="coerce")
    min_dt, max_dt = dt.min(), dt.max()
    if pd.isna(min_dt) or pd.isna(max_dt) or min_dt == max_dt:
        return np.ones(len(dates), dtype=float)
    scaled = (dt - min_dt) / (max_dt - min_dt)
    return (min_w + scaled.astype(float).to_numpy() * (max_w - min_w)).astype(float)


def _tune_xgb_with_time_cv(
    trainval_df: pd.DataFrame, scaled_cols: list[str], seed: int
) -> dict:
    """Time-aware tuning on train+val, keeping test untouched."""
    ordered = trainval_df.sort_values("date").reset_index(drop=True)
    X = ordered[scaled_cols].astype(float).to_numpy()
    y_raw = ordered["target_pm2_5"].astype(float).to_numpy()
    dates = pd.to_datetime(ordered["date"], errors="coerce")
    y = np.log1p(y_raw)

    tscv = TimeSeriesSplit(n_splits=4)
    grid = ParameterGrid(
        {
            "max_depth": [8],
            "min_child_weight": [4],
            "subsample": [0.9],
            "colsample_bytree": [0.8],
            "learning_rate": [0.02],
            "n_estimators": [1400],
            "reg_alpha": [0.05],
            "reg_lambda": [0.5],
            "gamma": [0.1],
        }
    )

    best = {"cv_mae": float("inf"), "params": None}
    print("\nTime-series CV tuning (train+val only):")
    for i, params in enumerate(grid, start=1):
        fold_maes: list[float] = []
        for tr_idx, va_idx in tscv.split(X):
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]
            y_va_raw = y_raw[va_idx]

            # Clip only using fold-train distribution to reduce sensitivity to outlier spikes.
            y_tr_raw = np.expm1(y_tr)
            clip_hi = np.quantile(y_tr_raw, 0.995)
            y_tr = np.log1p(np.clip(y_tr_raw, 0.0, clip_hi))

            w_tr = _recency_weights(dates.iloc[tr_idx])
            mdl = XGBRegressor(
                objective="reg:squarederror",
                eval_metric="mae",
                random_state=seed,
                n_jobs=-1,
                **params,
            )
            mdl.fit(X_tr, y_tr, sample_weight=w_tr, verbose=False)
            preds = np.maximum(np.expm1(mdl.predict(X_va)), 0.0)
            fold_maes.append(float(mean_absolute_error(y_va_raw, preds)))

        avg_mae = float(np.mean(fold_maes))
        print(f"  Trial {i:02d}/{len(grid):02d}  CV MAE={avg_mae:7.2f}  params={params}")
        if avg_mae < best["cv_mae"]:
            best = {"cv_mae": avg_mae, "params": dict(params)}

    if not best["params"]:
        raise RuntimeError("XGBoost tuning failed to produce parameters")
    print(f"\nBest CV MAE: {best['cv_mae']:.2f}")
    print(f"Best params: {best['params']}")
    return best


def main() -> None:
    np.random.seed(SEED)
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"Missing {FEATURES_PATH}. Run src/02_feature_engineering.py first.")
    if not os.path.exists(FEATURE_COLS_PATH):
        raise FileNotFoundError(f"Missing {FEATURE_COLS_PATH}. Run src/02_feature_engineering.py first.")

    df = pd.read_parquet(FEATURES_PATH)
    with open(FEATURE_COLS_PATH) as f:
        feature_cols = json.load(f)

    scaled_cols = [f"{c}_scaled" for c in feature_cols]
    missing_scaled = [c for c in scaled_cols if c not in df.columns]
    if missing_scaled:
        raise ValueError(
            "features.parquet is missing expected scaled columns. "
            f"Missing: {missing_scaled[:10]}{'...' if len(missing_scaled) > 10 else ''}"
        )

    if "split" not in df.columns or "target_pm2_5" not in df.columns:
        raise ValueError("features.parquet must include 'split' and 'target_pm2_5'")

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    X_train = train_df[scaled_cols].astype(float)
    y_train_raw = train_df["target_pm2_5"].astype(float)
    X_test = test_df[scaled_cols].astype(float)
    y_test = test_df["target_pm2_5"].astype(float)
    y_train = np.log1p(y_train_raw)

    print("=" * 60)
    print("Step 03 — Train (XGBoost + RF Baseline)")
    print("=" * 60)
    print(f"Train/Val/Test: {len(train_df)}/{len(val_df)}/{len(test_df)}")
    print(f"Features: {len(scaled_cols)}")

    # ═════════════════════════════════════════════════════════════
    # XGBoost with all features (including cross-city)
    # ═════════════════════════════════════════════════════════════
    trainval_df = df[df["split"].isin(["train", "val"])].copy()
    tuned = _tune_xgb_with_time_cv(trainval_df=trainval_df, scaled_cols=scaled_cols, seed=SEED)

    X_trainval = trainval_df[scaled_cols].astype(float)
    y_trainval_raw = trainval_df["target_pm2_5"].astype(float).to_numpy()
    y_trainval_clip_hi = np.quantile(y_trainval_raw, 0.995)
    y_trainval = np.log1p(np.clip(y_trainval_raw, 0.0, y_trainval_clip_hi))
    w_trainval = _recency_weights(trainval_df["date"])

    model = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="mae",
        random_state=SEED,
        n_jobs=-1,
        **tuned["params"],
    )
    model.fit(X_trainval, y_trainval, sample_weight=w_trainval, verbose=False)

    joblib.dump(model, MODEL_PATH)
    print(f"\n✓ Saved model: {MODEL_PATH}")

    # ═════════════════════════════════════════════════════════════
    # Random Forest baseline WITHOUT cross-city features
    # ═════════════════════════════════════════════════════════════
    base_scaled_cols = [c for c in scaled_cols if "neighbor_" not in c]
    print(f"\nRF baseline features (no cross-city): {len(base_scaled_cols)}")

    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        random_state=SEED,
        n_jobs=1,
    )
    rf.fit(train_df[base_scaled_cols].astype(float), y_train)

    # ═════════════════════════════════════════════════════════════
    # Evaluate (back-transform from log-space)
    # ═════════════════════════════════════════════════════════════
    xgb_preds = np.maximum(np.expm1(model.predict(X_test)), 0)
    rf_preds = np.maximum(np.expm1(rf.predict(test_df[base_scaled_cols].astype(float))), 0)

    xgb_m = _metrics(y_test.to_numpy(), xgb_preds)
    rf_m = _metrics(y_test.to_numpy(), rf_preds)

    print(f"\n{'Model':<45s}  {'MAE':>6s}  {'RMSE':>6s}  {'R²':>6s}")
    print("-" * 70)
    print(f"{'XGBoost (with cross-city features)':<45s}  {xgb_m.mae:6.2f}  {xgb_m.rmse:6.2f}  {xgb_m.r2:6.3f}")
    print(f"{'RandomForest (no cross-city features)':<45s}  {rf_m.mae:6.2f}  {rf_m.rmse:6.2f}  {rf_m.r2:6.3f}")

    mae_improvement = (rf_m.mae - xgb_m.mae) / rf_m.mae * 100
    print(f"\nXGBoost MAE improvement over RF: {mae_improvement:.1f}%")

    # Model comparison CSV
    comp_rows = [
        {"model": "XGBoost (with cross-city features)", "MAE": round(xgb_m.mae, 2),
         "RMSE": round(xgb_m.rmse, 2), "R2": round(xgb_m.r2, 3)},
        {"model": "RandomForest (no cross-city features)", "MAE": round(rf_m.mae, 2),
         "RMSE": round(rf_m.rmse, 2), "R2": round(rf_m.r2, 3)},
    ]
    pd.DataFrame(comp_rows).to_csv(OUT_MODEL_COMP, index=False)
    print(f"\n✓ Saved: {OUT_MODEL_COMP}")

    # Per-city test metrics
    per_city = []
    test_copy = test_df.copy()
    test_copy["predicted"] = xgb_preds
    for city, grp in test_copy.groupby("city"):
        m = _metrics(grp["target_pm2_5"].to_numpy(), grp["predicted"].to_numpy())
        per_city.append({
            "city": city, "MAE": round(m.mae, 2), "RMSE": round(m.rmse, 2),
            "R2": round(m.r2, 3), "n": len(grp),
        })
        print(f"  {city:<12s}  MAE={m.mae:7.2f}  RMSE={m.rmse:7.2f}  R²={m.r2:.3f}  (n={len(grp)})")

    pd.DataFrame(per_city).to_csv(OUT_PER_CITY, index=False)
    print(f"\n✓ Saved: {OUT_PER_CITY}")

    # predictions.csv for Streamlit
    pred_df = test_copy[["city", "date", "target_pm2_5"]].copy()
    pred_df["predicted_pm2_5"] = xgb_preds
    pred_df = pred_df.rename(columns={"target_pm2_5": "actual_pm2_5"})
    pred_df["date"] = pred_df["date"].astype(str)
    pred_df = pred_df.sort_values(["city", "date"]).reset_index(drop=True)
    pred_df.to_csv(OUT_PREDICTIONS, index=False)
    print(f"✓ Saved: {OUT_PREDICTIONS} ({len(pred_df)} rows)")

    # ═════════════════════════════════════════════════════════════
    # city_last7.json — from RAW CSV
    # ═════════════════════════════════════════════════════════════
    if not os.path.exists(RAW_CSV_PATH):
        raise FileNotFoundError(f"Missing {RAW_CSV_PATH} for building city_last7.json")

    raw_df = pd.read_csv(RAW_CSV_PATH, index_col=0)
    raw_df["date"] = pd.to_datetime(raw_df["date"], errors="coerce")
    raw_df["city"] = (
        raw_df["city"].astype(str).str.strip()
        .replace({"Bangalore": "Bengaluru", "bangalore": "Bengaluru"})
        .str.title()
    )
    raw_df.loc[raw_df["pm2_5"] < 1.0, "pm2_5"] = np.nan

    RAW_COLS = [
        "aqi", "co", "no", "no2", "o3", "so2", "pm2_5", "pm10",
        "nh3", "temperature", "wind_speed", "rainfall", "pressure",
    ]

    city_last7: dict[str, list[dict]] = {}
    for city, grp in raw_df.groupby("city"):
        tail = grp.sort_values("date").tail(7)[["date"] + RAW_COLS].copy()
        tail["date"] = tail["date"].astype(str)
        city_last7[city] = tail.fillna(0).to_dict(orient="records")

    with open(CITY_LAST7_PATH, "w") as f:
        json.dump(city_last7, f, indent=2, default=str)
    print(f"✓ Saved: {CITY_LAST7_PATH} (raw last-7 rows per city)")

    # Summary
    print("\n" + "=" * 60)
    print("Training complete.")
    print(f"  Overall MAE: {xgb_m.mae:.2f} µg/m³  |  Overall R²: {xgb_m.r2:.3f}")
    print(f"  XGBoost vs RF improvement: {mae_improvement:.1f}% MAE reduction")
    print("=" * 60)


if __name__ == "__main__":
    main()
