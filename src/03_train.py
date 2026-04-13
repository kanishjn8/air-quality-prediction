"""Step 03 — Train XGBoost model (Plan vNext)

Input:
  - data/processed/features.parquet
  - models/feature_columns.json

Output:
  - models/best_model.pkl
  - models/city_last7.json
  - outputs/per_city_metrics.csv
  - outputs/model_comparison.csv

Notes:
  - Uses scaled feature columns produced by Step 02 ("*_scaled").
  - `city_last7.json` stores RAW (unscaled) last 7 days per city, used at inference.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

FEATURES_PATH = "data/processed/features.parquet"
FEATURE_COLS_PATH = "models/feature_columns.json"

MODEL_PATH = "models/best_model.pkl"
CITY_LAST7_PATH = "models/city_last7.json"

OUT_PER_CITY = "outputs/per_city_metrics.csv"
OUT_MODEL_COMP = "outputs/model_comparison.csv"


@dataclass
class Metrics:
    mae: float
    rmse: float
    r2: float


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return Metrics(mae=mae, rmse=rmse, r2=r2)


def main() -> None:
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
    y_train = train_df["target_pm2_5"].astype(float)
    X_val = val_df[scaled_cols].astype(float)
    y_val = val_df["target_pm2_5"].astype(float)
    X_test = test_df[scaled_cols].astype(float)
    y_test = test_df["target_pm2_5"].astype(float)

    print("=" * 60)
    print("Step 03 — Train (XGBoost)")
    print("=" * 60)
    print(f"Train/Val/Test: {len(train_df)}/{len(val_df)}/{len(test_df)}")

    # NOTE: The historical dataset shows noticeable distribution shift in later dates
    # (val/test). Early stopping + stronger regularization reduces overfit.
    model = XGBRegressor(
        n_estimators=5000,
        learning_rate=0.02,
        max_depth=4,
        min_child_weight=20,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=5.0,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=200,
    )

    eval_set = [(X_val, y_val)] if len(val_df) > 0 else None
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

    if getattr(model, "best_iteration", None) is not None:
        print(f"Best iteration: {model.best_iteration}")

    joblib.dump(model, MODEL_PATH)
    print(f"\n✓ Saved model: {MODEL_PATH}")

    # Evaluate
    rows = []
    for split_name, X, y, split_df in [
        ("train", X_train, y_train, train_df),
        ("val", X_val, y_val, val_df),
        ("test", X_test, y_test, test_df),
    ]:
        if len(split_df) == 0:
            continue
        pred = model.predict(X)
        m = _metrics(y.to_numpy(), pred)
        rows.append({"split": split_name, "mae": m.mae, "rmse": m.rmse, "r2": m.r2})
        print(f"{split_name:>5s}: MAE={m.mae:.2f} RMSE={m.rmse:.2f} R2={m.r2:.3f}")

    # Per-city test metrics
    per_city = []
    if len(test_df) > 0:
        test_pred = model.predict(X_test)
        test_df = test_df.copy()
        test_df["pred"] = test_pred
        for city, grp in test_df.groupby("city"):
            m = _metrics(grp["target_pm2_5"].to_numpy(), grp["pred"].to_numpy())
            per_city.append({"city": city, "split": "test", "mae": m.mae, "rmse": m.rmse, "r2": m.r2, "n": len(grp)})

    pd.DataFrame(per_city).sort_values(["split", "city"]).to_csv(OUT_PER_CITY, index=False)
    pd.DataFrame(rows).to_csv(OUT_MODEL_COMP, index=False)
    print(f"\n✓ Saved: {OUT_PER_CITY}")
    print(f"✓ Saved: {OUT_MODEL_COMP}")

    # Persist last 7 RAW days per city (for inference lag construction)
    raw_cols = [
        "date",
        "city",
        "aqi",
        "co",
        "no2",
        "o3",
        "so2",
        "pm2_5",
        "pm10",
        "temperature",
        "wind_speed",
        "rainfall",
        "pressure",
    ]
    missing_raw = [c for c in raw_cols if c not in df.columns]
    if missing_raw:
        raise ValueError(f"Cannot build city_last7.json, missing raw cols: {missing_raw}")

    city_last7: dict[str, list[dict]] = {}
    for city, grp in df.sort_values("date").groupby("city"):
        tail = grp.sort_values("date").tail(7)[raw_cols].copy()
        tail["date"] = tail["date"].astype(str)
        city_last7[city] = tail.to_dict(orient="records")

    with open(CITY_LAST7_PATH, "w") as f:
        json.dump(city_last7, f, indent=2)
    print(f"✓ Saved: {CITY_LAST7_PATH} (raw last-7 rows per city)")


if __name__ == "__main__":
    main()
