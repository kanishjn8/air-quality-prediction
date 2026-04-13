"""Step 04 — Explainability (Plan vNext)

Input:
  - models/best_model.pkl
  - models/feature_columns.json
  - data/processed/features.parquet

Output:
  - outputs/shap_summary.png
  - outputs/shap_waterfall_spike1.png
  - outputs/shap_waterfall_spike2.png
  - outputs/shap_waterfall_spike3.png
  - outputs/lag_importance.png
  - outputs/cross_city_influence.png

Notes:
  - Uses SHAP TreeExplainer for XGBoost.
  - Uses test split when available; otherwise falls back to val.
"""

from __future__ import annotations

import json
import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

FEATURES_PATH = "data/processed/features.parquet"
MODEL_PATH = "models/best_model.pkl"
FEATURE_COLS_PATH = "models/feature_columns.json"

OUT_DIR = "outputs"


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing {MODEL_PATH}. Run src/03_train.py first.")
    if not os.path.exists(FEATURE_COLS_PATH):
        raise FileNotFoundError(f"Missing {FEATURE_COLS_PATH}. Run src/02_feature_engineering.py first.")
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"Missing {FEATURES_PATH}. Run src/02_feature_engineering.py first.")

    model = joblib.load(MODEL_PATH)
    with open(FEATURE_COLS_PATH) as f:
        feature_cols = json.load(f)

    df = pd.read_parquet(FEATURES_PATH)

    candidate = df[df["split"].isin(["test", "val"])].copy() if "split" in df.columns else df.copy()
    if candidate.empty:
        candidate = df.copy()

    scaled_cols = [f"{c}_scaled" for c in feature_cols]
    X = candidate[scaled_cols].astype(float)
    y = candidate["target_pm2_5"].astype(float)

    # SHAP summary
    explainer = shap.TreeExplainer(model)
    max_samples = min(800, len(X))
    sample_idx = np.random.choice(len(X), max_samples, replace=False)
    X_explain = X.iloc[sample_idx]

    shap_values = explainer.shap_values(X_explain)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_explain, feature_names=scaled_cols, show=False)
    plt.title("SHAP Summary — XGBoost", fontsize=12, pad=12)
    plt.tight_layout()
    out_summary = os.path.join(OUT_DIR, "shap_summary.png")
    plt.savefig(out_summary, dpi=160, bbox_inches="tight")
    plt.close()

    # Waterfall for top predicted spikes
    preds = model.predict(X)
    topk = np.argsort(preds)[-3:][::-1]
    for i, ridx in enumerate(topk, start=1):
        x_row = X.iloc[[ridx]]
        sv = explainer.shap_values(x_row)
        base = float(explainer.expected_value) if np.isscalar(explainer.expected_value) else float(explainer.expected_value[0])
        exp = shap.Explanation(values=sv[0], base_values=base, data=x_row.values[0], feature_names=scaled_cols)
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(exp, show=False, max_display=16)
        city = str(candidate.iloc[ridx].get("city", "Unknown"))
        date = str(candidate.iloc[ridx].get("date", ""))
        plt.title(f"Spike #{i}: {city} {date} (pred={preds[ridx]:.1f})", fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"shap_waterfall_spike{i}.png"), dpi=160, bbox_inches="tight")
        plt.close()

    # Lag importance (pm2_5 lags)
    lag_imp: list[tuple[str, float]] = []
    lag_like = [c for c in scaled_cols if "pm2_5_lag" in c]
    if lag_like:
        imp = np.abs(shap_values).mean(axis=0)
        lag_imp = [(c, float(imp[scaled_cols.index(c)])) for c in lag_like]
        lag_imp.sort(key=lambda t: t[1], reverse=True)

    if lag_imp:
        plt.figure(figsize=(8, 4))
        y_names = [k.replace("_scaled", "") for k, _ in lag_imp]
        x_vals = [v for _, v in lag_imp]
        sns.barplot(x=x_vals, y=y_names, hue=y_names, legend=False, palette="viridis")
        plt.title("Lag feature importance (mean |SHAP|)")
        plt.xlabel("mean |SHAP|")
        plt.ylabel("feature")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "lag_importance.png"), dpi=160, bbox_inches="tight")
        plt.close()

    # Cross-city influence proxy: summarize neighbor_* SHAP
    top: list[tuple[str, float]] = []
    neighbor_like = [c for c in scaled_cols if "neighbor_" in c and c.endswith("_scaled")]
    if neighbor_like:
        imp = np.abs(shap_values).mean(axis=0)
        neigh_imp = [(c, float(imp[scaled_cols.index(c)])) for c in neighbor_like]
        neigh_imp.sort(key=lambda t: t[1], reverse=True)
        top = neigh_imp[:20]

    if top:
        plt.figure(figsize=(10, 6))
        y_names = [k.replace("_scaled", "") for k, _ in top]
        x_vals = [v for _, v in top]
        sns.barplot(x=x_vals, y=y_names, hue=y_names, legend=False, palette="magma")
        plt.title("Cross-city influence (neighbor_* mean |SHAP|)")
        plt.xlabel("mean |SHAP|")
        plt.ylabel("feature")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "cross_city_influence.png"), dpi=160, bbox_inches="tight")
        plt.close()

    print("=" * 60)
    print("Step 04 — Explainability complete")
    print("=" * 60)
    print(f"✓ {out_summary}")


if __name__ == "__main__":
    main()
