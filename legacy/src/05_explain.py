"""DEPRECATED (legacy): explainability for torch-era pipeline.

Use:
    - src/04_explain.py

Legacy source archived at:
    - legacy/src_torch/05_explain.py
"""

raise SystemExit(
        "src/05_explain.py is deprecated. Use src/04_explain.py (Plan vNext SHAP)."
)


import os
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import torch

warnings.filterwarnings("ignore")

# ────────────────────────────────────────────────────────────────
# Reproducibility
# ────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ────────────────────────────────────────────────────────────────
# Paths
# ────────────────────────────────────────────────────────────────
FEATURES_PATH = "data/processed/features.parquet"
RF_MODEL_PATH = "models/baseline_rf.pkl"
ADJ_PATH      = "data/processed/adj_matrix.npy"

os.makedirs("outputs", exist_ok=True)

print("=" * 60)
print("Step 05 — Explainability (XAI)")
print("=" * 60)

# ═══════════════════════════════════════════════════════════════
# Load data and models
# ═══════════════════════════════════════════════════════════════
df = pd.read_parquet(FEATURES_PATH)
test_df = df[df["split"] == "test"].copy()

FEATURE_COLS = [
    "city_idx",
    "aqi", "co", "no", "no2", "o3", "so2",
    "pm2_5", "pm10", "nh3",
    "temperature", "wind_speed", "rainfall", "pressure",
    "pm2_5_lag1", "pm2_5_lag2", "pm2_5_lag3", "pm2_5_lag7",
    "aqi_lag1", "aqi_lag2", "aqi_lag3", "aqi_lag7",
    "pm2_5_roll3mean", "pm2_5_roll7mean",
    "pm2_5_roll3std", "pm2_5_roll7std",
    "dayofweek", "month", "is_weekend", "quarter",
    "crop_burning_season", "monsoon_season",
]

CITY_TO_IDX = {"Delhi": 0, "Bengaluru": 1, "Kolkata": 2, "Hyderabad": 3}
IDX_TO_CITY = {v: k for k, v in CITY_TO_IDX.items()}

X_test = test_df[FEATURE_COLS].values
y_test = test_df["target_pm2_5"].values

# Load RF model
rf = joblib.load(RF_MODEL_PATH)
print(f"\n  ✓ Loaded RF model from {RF_MODEL_PATH}")
print(f"  ✓ Test set: {len(test_df)} samples")


# ═══════════════════════════════════════════════════════════════
# 5.1  SHAP for Random Forest Baseline
# ═══════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("Computing SHAP values for Random Forest …")
print("─" * 60)

explainer = shap.TreeExplainer(rf)

# Use a subsample for speed if test set is large
max_samples = min(500, len(X_test))
sample_idx = np.random.choice(len(X_test), max_samples, replace=False)
X_explain = X_test[sample_idx]

shap_values = explainer.shap_values(X_explain)

# --- SHAP Summary Plot ---
print("  Generating SHAP summary plot …")
fig = plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_explain, feature_names=FEATURE_COLS, show=False)
plt.title("SHAP Feature Importance — Random Forest Baseline", fontsize=12, pad=15)
plt.tight_layout()
plt.savefig("outputs/shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved outputs/shap_summary.png")

# --- SHAP Waterfall Plots for Top 3 PM2.5 Spikes ---
print("  Generating SHAP waterfall plots for top spike events …")
rf_preds = rf.predict(X_test)
top3_idx = np.argsort(rf_preds)[-3:]  # highest predicted PM2.5

for rank, idx in enumerate(top3_idx):
    # Find this sample in our shap subsample (or recompute)
    single_shap = explainer.shap_values(X_test[idx:idx+1])

    fig = plt.figure(figsize=(10, 6))
    explanation = shap.Explanation(
        values=single_shap[0],
        base_values=explainer.expected_value,
        data=X_test[idx],
        feature_names=FEATURE_COLS,
    )
    shap.waterfall_plot(explanation, show=False, max_display=15)
    city_name = IDX_TO_CITY.get(int(X_test[idx][0]), "Unknown")
    plt.title(f"SHAP Waterfall — Spike #{rank+1} [{city_name}] "
              f"(Pred={rf_preds[idx]:.0f} µg/m³)", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"outputs/shap_waterfall_{rank+1}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved outputs/shap_waterfall_{rank+1}.png")

# --- SHAP Dependence Plot ---
print("  Generating SHAP dependence plot …")
fig = plt.figure(figsize=(8, 5))
shap.dependence_plot(
    "pm2_5_lag1", shap_values, X_explain,
    feature_names=FEATURE_COLS, interaction_index="wind_speed",
    show=False
)
plt.title("SHAP Dependence: pm2_5_lag1 × wind_speed", fontsize=12)
plt.tight_layout()
plt.savefig("outputs/shap_dependence.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved outputs/shap_dependence.png")


# ═══════════════════════════════════════════════════════════════
# 5.2  GNN Edge Importance (from adjacency matrix)
# ═══════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("Plotting GNN edge importance …")
print("─" * 60)

adj = np.load(ADJ_PATH)
cities = [IDX_TO_CITY[i] for i in range(5)]

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(
    adj, xticklabels=cities, yticklabels=cities,
    annot=True, fmt=".2f", cmap="YlOrRd",
    linewidths=0.5, ax=ax, vmin=0, vmax=1,
    cbar_kws={"label": "Edge Weight"},
)
ax.set_title("Cross-City PM2.5 Influence Matrix", fontsize=13, pad=10)
ax.set_ylabel("Target City (influenced)")
ax.set_xlabel("Source City (influencing)")
plt.tight_layout()
plt.savefig("outputs/gnn_edge_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved outputs/gnn_edge_importance.png")


# ═══════════════════════════════════════════════════════════════
# 5.3  Temporal Lag Importance
# ═══════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("Analyzing temporal lag importance …")
print("─" * 60)

lag_cols = ["pm2_5_lag1", "pm2_5_lag2", "pm2_5_lag3", "pm2_5_lag7"]
lag_indices = [FEATURE_COLS.index(c) for c in lag_cols]

# Extract SHAP importance for lag features
lag_importance = np.abs(shap_values[:, lag_indices]).mean(axis=0)

# Split by pollution level: high (>90th percentile) vs normal
threshold = np.percentile(y_test[sample_idx], 90)
high_mask = y_test[sample_idx] > threshold
normal_mask = ~high_mask

high_importance = np.abs(shap_values[high_mask][:, lag_indices]).mean(axis=0) if high_mask.sum() > 0 else lag_importance
normal_importance = np.abs(shap_values[normal_mask][:, lag_indices]).mean(axis=0)

lag_labels = ["t-1", "t-2", "t-3", "t-7"]

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(lag_labels))
width = 0.35

bars1 = ax.bar(x - width/2, high_importance, width, label="High Pollution Days",
               color="#c0392b", alpha=0.85, edgecolor="white")
bars2 = ax.bar(x + width/2, normal_importance, width, label="Normal Days",
               color="#2980b9", alpha=0.85, edgecolor="white")

ax.set_xlabel("Lag Timestep", fontsize=11)
ax.set_ylabel("Mean |SHAP| value", fontsize=11)
ax.set_title("Temporal Lag Importance: High Pollution vs Normal Days", fontsize=13, pad=10)
ax.set_xticks(x)
ax.set_xticklabels(lag_labels)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("outputs/temporal_lag_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved outputs/temporal_lag_importance.png")


print(f"\n{'=' * 60}")
print("✓ Explainability analysis complete")
print(f"{'=' * 60}")
print("\nGenerated artefacts:")
print("  1. outputs/shap_summary.png         — Feature importance beeswarm")
print("  2. outputs/shap_waterfall_1..3.png   — Spike event explanations")
print("  3. outputs/shap_dependence.png       — pm2_5_lag1 × wind_speed")
print("  4. outputs/gnn_edge_importance.png   — Cross-city influence heatmap")
print("  5. outputs/temporal_lag_importance.png — Lag day attribution")
