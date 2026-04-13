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
  - Model trained on log1p(target) — predictions are back-transformed for display.
  - Cross-city influence is a 5×5 heatmap (target city vs neighbor source).
  - Feature names have _scaled suffix stripped for readability.
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

CITIES = ["Delhi", "Mumbai", "Bengaluru", "Kolkata", "Hyderabad"]


def _clean_names(names: list[str]) -> list[str]:
    """Strip _scaled suffix for display."""
    return [n.replace("_scaled", "") for n in names]


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

    # Prefer test+val for explanation; fall back to all data
    candidate = df[df["split"].isin(["test", "val"])].copy() if "split" in df.columns else df.copy()
    if candidate.empty:
        candidate = df.copy()

    scaled_cols = [f"{c}_scaled" for c in feature_cols]
    display_names = _clean_names(scaled_cols)
    X = candidate[scaled_cols].astype(float)

    # ── SHAP values ──
    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    max_samples = min(800, len(X))
    np.random.seed(42)
    sample_idx = np.random.choice(len(X), max_samples, replace=False)
    X_explain = X.iloc[sample_idx]

    shap_values = explainer.shap_values(X_explain)

    # ── 1. SHAP Summary (beeswarm) ──
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_explain, feature_names=display_names,
                      show=False, max_display=25)
    plt.title("SHAP Summary — XGBoost with Cross-City Features", fontsize=12, pad=12)
    plt.tight_layout()
    out_summary = os.path.join(OUT_DIR, "shap_summary.png")
    plt.savefig(out_summary, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved {out_summary}")

    # ── 2. Waterfall plots for top 3 predicted spike events ──
    # Back-transform log predictions for ranking
    preds_log = model.predict(X)
    preds = np.expm1(preds_log)
    topk = np.argsort(preds)[-3:][::-1]

    base = float(explainer.expected_value) if np.isscalar(explainer.expected_value) else float(explainer.expected_value[0])

    for i, ridx in enumerate(topk, start=1):
        x_row = X.iloc[[ridx]]
        sv = explainer.shap_values(x_row)
        exp = shap.Explanation(
            values=sv[0], base_values=base,
            data=x_row.values[0], feature_names=display_names,
        )
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(exp, show=False, max_display=16)
        city = str(candidate.iloc[ridx].get("city", "Unknown"))
        date = str(candidate.iloc[ridx].get("date", ""))
        plt.title(f"Spike #{i}: {city} {date} (pred={preds[ridx]:.1f} µg/m³)", fontsize=11)
        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, f"shap_waterfall_spike{i}.png")
        plt.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved {out_path}")

    # ── 3. Cross-city influence heatmap (5×5 SHAP-based) ──
    influence = np.zeros((5, 5))
    for i, target_city in enumerate(CITIES):
        city_mask = (candidate["city"] == target_city).values
        if not city_mask.any():
            continue
        city_X = X[city_mask].astype(float)
        # Limit to 200 samples per city for efficiency
        if len(city_X) > 200:
            np.random.seed(42 + i)
            city_idx = np.random.choice(len(city_X), 200, replace=False)
            city_X = city_X.iloc[city_idx]
        city_shap = explainer.shap_values(city_X)

        for j, neighbor in enumerate(CITIES):
            if i == j:
                continue
            col = f"neighbor_{neighbor.lower()}_pm2_5_lag1_scaled"
            if col in scaled_cols:
                col_idx = scaled_cols.index(col)
                influence[i, j] = np.abs(city_shap[:, col_idx]).mean()

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        influence, xticklabels=CITIES, yticklabels=CITIES,
        annot=True, fmt=".3f", cmap="YlOrRd",
        linewidths=0.5, linecolor="white",
    )
    plt.title("Cross-City PM2.5 Influence (mean |SHAP|)", fontsize=12)
    plt.xlabel("Neighbor city (source of influence)")
    plt.ylabel("Target city (being predicted)")
    plt.tight_layout()
    out_influence = os.path.join(OUT_DIR, "cross_city_influence.png")
    plt.savefig(out_influence, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved {out_influence}")

    # ── 4. Lag importance bar chart ──
    lag_cols_scaled = [c for c in scaled_cols if "pm2_5_lag" in c]
    if lag_cols_scaled:
        imp = np.abs(shap_values).mean(axis=0)
        lag_imp = [(c.replace("_scaled", ""), float(imp[scaled_cols.index(c)]))
                   for c in lag_cols_scaled]
        lag_imp.sort(key=lambda t: t[1], reverse=True)

        plt.figure(figsize=(6, 4))
        plt.bar([k for k, _ in lag_imp], [v for _, v in lag_imp], color="#e07b54")
        plt.title("PM2.5 Lag Feature Importance (mean |SHAP|)", fontsize=12)
        plt.ylabel("Mean |SHAP value|")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        out_lag = os.path.join(OUT_DIR, "lag_importance.png")
        plt.savefig(out_lag, dpi=160, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved {out_lag}")

    print("\n" + "=" * 60)
    print("Step 04 — Explainability complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
