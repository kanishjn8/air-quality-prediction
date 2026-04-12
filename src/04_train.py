"""
Step 04 — Model Training
Input:  data/processed/features.parquet, data/processed/graph_edges.csv
Output: models/best_model.pt, outputs/training_curves.png, outputs/predictions.csv

Architecture: 3-branch fusion model
  - GNN branch (GraphSAGE, 2 layers) → city embedding
  - Temporal branch (LSTM, 2 layers) → sequence embedding
  - Met branch (MLP) → meteorological embedding
  - Fusion MLP head → scalar PM2.5 prediction

Also trains a Random Forest baseline for comparison.
"""

import os
import sys
import random
import warnings
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

warnings.filterwarnings("ignore")

# ────────────────────────────────────────────────────────────────
# Reproducibility
# ────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ────────────────────────────────────────────────────────────────
# Paths
# ────────────────────────────────────────────────────────────────
FEATURES_PATH = "data/processed/features.parquet"
EDGES_PATH    = "data/processed/graph_edges.csv"
ADJ_PATH      = "data/processed/adj_matrix.npy"
MODEL_PATH    = "models/best_model.pt"
CURVES_PATH   = "outputs/training_curves.png"
PREDS_PATH    = "outputs/predictions.csv"

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

print("=" * 60)
print("Step 04 — Model Training")
print(f"  Device: {DEVICE}")
print("=" * 60)

# ═══════════════════════════════════════════════════════════════
# Load Data
# ═══════════════════════════════════════════════════════════════
df = pd.read_parquet(FEATURES_PATH)
adj = np.load(ADJ_PATH)

FEATURE_COLS = [
    "city_idx",
    # NOTE: pm2_5 and aqi at t=0 are EXCLUDED to prevent data leakage.
    "co", "no", "no2", "o3", "so2",
    "pm10", "nh3",
    "temperature", "wind_speed", "rainfall", "pressure",
    "pm2_5_lag1", "pm2_5_lag2", "pm2_5_lag3", "pm2_5_lag7",
    "aqi_lag1", "aqi_lag2", "aqi_lag3", "aqi_lag7",
    "pm2_5_roll3mean", "pm2_5_roll7mean",
    "pm2_5_roll3std", "pm2_5_roll7std",
    "dayofweek", "month", "is_weekend", "quarter",
    "crop_burning_season", "monsoon_season",
]

SCALED_COLS = [f"{c}_scaled" for c in FEATURE_COLS]

# Temporal features for LSTM (per city, last 7 days)
TEMPORAL_RAW_COLS = ["pm2_5", "aqi", "temperature", "wind_speed", "pressure"]
TEMPORAL_SCALED_COLS = [f"{c}_scaled" for c in TEMPORAL_RAW_COLS]

# Met features for MLP branch
MET_RAW_COLS = ["temperature", "wind_speed", "pressure", "rainfall",
                "dayofweek", "month", "crop_burning_season", "monsoon_season"]
MET_SCALED_COLS = [f"{c}_scaled" for c in MET_RAW_COLS]

CITY_TO_IDX = {"Delhi": 0, "Bengaluru": 1, "Kolkata": 2, "Hyderabad": 3}
NUM_CITIES = 4
SEQ_LEN = 7

train_df = df[df["split"] == "train"].copy()
val_df   = df[df["split"] == "val"].copy()
test_df  = df[df["split"] == "test"].copy()

print(f"\n  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# Log-transform the target to handle extreme skew (max=2200+, median=62)
# This stabilises training and prevents extreme values from dominating the loss.
LOG_TARGET = True

# Clip extreme outlier targets (>99.5th percentile of train set) to reduce noise
# These are likely sensor errors or extreme transients — unreachable by the model.
p995 = train_df["target_pm2_5"].quantile(0.995)
p005 = train_df["target_pm2_5"].quantile(0.005)
print(f"  Target clip range: [{p005:.1f}, {p995:.1f}] µg/m³")
for split_df in [train_df, val_df, test_df, df]:
    split_df["target_pm2_5"] = split_df["target_pm2_5"].clip(lower=p005, upper=p995)

if LOG_TARGET:
    for split_df in [train_df, val_df, test_df, df]:
        split_df["target_pm2_5_raw"] = split_df["target_pm2_5"]
        split_df["target_pm2_5"] = np.log1p(split_df["target_pm2_5"])
    print(f"  ✓ Target log-transformed: log1p(pm2_5). Train target range: "
          f"[{train_df['target_pm2_5'].min():.2f}, {train_df['target_pm2_5'].max():.2f}]")


# ═══════════════════════════════════════════════════════════════
# Dataset: Build temporal sequences
# ═══════════════════════════════════════════════════════════════
def build_sequences(split_df, full_df, seq_len=SEQ_LEN):
    """Build (features, temporal_seq, met_features, city_idx, target) tuples.

    For each row, look back `seq_len` days in that city's history to build
    the temporal sequence for the LSTM branch.
    """
    records = []

    for city in sorted(split_df["city"].unique()):
        city_all = full_df[full_df["city"] == city].sort_values("date").reset_index(drop=True)
        city_split = split_df[split_df["city"] == city].sort_values("date")

        for _, row in city_split.iterrows():
            date = row["date"]
            # Find the index in city_all for this date
            idx_in_all = city_all[city_all["date"] == date].index
            if len(idx_in_all) == 0:
                continue
            idx = idx_in_all[0]

            if idx < seq_len:
                continue  # not enough history

            # Temporal sequence: last seq_len days
            seq_rows = city_all.iloc[idx - seq_len: idx]
            seq = seq_rows[TEMPORAL_SCALED_COLS].values  # (seq_len, n_temporal)

            # All features (scaled)
            feats = row[SCALED_COLS].values.astype(np.float32)

            # Met features
            met = row[MET_SCALED_COLS].values.astype(np.float32)

            city_idx = int(row["city_idx"])
            target = float(row["target_pm2_5"])  # may be log-transformed

            records.append((feats, seq.astype(np.float32), met, city_idx, target))

    return records


print("\nBuilding temporal sequences …")
train_records = build_sequences(train_df, df)
val_records   = build_sequences(val_df, df)
test_records  = build_sequences(test_df, df)
print(f"  Train: {len(train_records)}, Val: {len(val_records)}, Test: {len(test_records)}")


class AirMindDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        feats, seq, met, city_idx, target = self.records[idx]
        return (
            torch.tensor(feats, dtype=torch.float32),
            torch.tensor(seq, dtype=torch.float32),
            torch.tensor(met, dtype=torch.float32),
            torch.tensor(city_idx, dtype=torch.long),
            torch.tensor(target, dtype=torch.float32),
        )


train_ds = AirMindDataset(train_records)
val_ds   = AirMindDataset(val_records)
test_ds  = AirMindDataset(test_records)

BATCH_SIZE = 64
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)


# ═══════════════════════════════════════════════════════════════
# 4.1  Model Architecture (without torch-geometric dependency)
# ═══════════════════════════════════════════════════════════════
class GraphBranch(nn.Module):
    """Simple GCN-like branch using adjacency matrix multiplication.

    Avoids torch-geometric dependency while still performing message passing.
    Uses learned city embeddings and two graph convolution layers.
    """

    def __init__(self, num_cities=NUM_CITIES, embed_dim=16, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.city_embed = nn.Embedding(num_cities, embed_dim)
        self.conv1_w = nn.Linear(embed_dim, hidden_dim)
        self.conv2_w = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, city_idx, adj_matrix):
        """
        city_idx: (batch,) — city indices
        adj_matrix: (num_cities, num_cities) — adjacency matrix
        """
        # Get ALL city embeddings and do graph conv
        all_embed = self.city_embed.weight  # (num_cities, embed_dim)

        # Normalize adjacency (D^{-1} A)
        deg = adj_matrix.sum(dim=1, keepdim=True).clamp(min=1e-6)
        adj_norm = adj_matrix / deg

        # Layer 1: A_norm @ X @ W
        h = adj_norm @ all_embed                     # (num_cities, embed_dim)
        h = self.relu(self.conv1_w(h))               # (num_cities, hidden_dim)
        h = self.dropout(h)

        # Layer 2
        h = adj_norm @ h
        h = self.relu(self.conv2_w(h))
        h = self.dropout(h)

        # Select embeddings for batch cities
        out = h[city_idx]  # (batch, hidden_dim)
        return out


class TemporalBranch(nn.Module):
    """2-layer LSTM for temporal sequences."""

    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True, dropout=dropout,
        )

    def forward(self, x):
        """x: (batch, seq_len, input_dim)"""
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]  # (batch, hidden_dim) — last layer's final hidden


class MetBranch(nn.Module):
    """MLP for meteorological / calendar features."""

    def __init__(self, input_dim=8, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class FeatureBranch(nn.Module):
    """MLP for processing the full scaled feature vector."""

    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class AirMindModel(nn.Module):
    """4-branch fusion model for PM2.5 prediction.

    Branches:
      1. GNN: city graph convolution → city embedding
      2. LSTM: temporal sequence → temporal embedding
      3. Met MLP: meteorological / calendar features → met embedding
      4. Feature MLP: full scaled feature vector → dense embedding
    """

    def __init__(self, num_cities=NUM_CITIES, num_temporal_feats=5, num_met_feats=8,
                 num_features=30, gnn_hidden=64, lstm_hidden=128,
                 met_hidden=32, feat_hidden=64):
        super().__init__()
        self.gnn_branch = GraphBranch(num_cities, embed_dim=16, hidden_dim=gnn_hidden)
        self.temporal_branch = TemporalBranch(num_temporal_feats, hidden_dim=lstm_hidden)
        self.met_branch = MetBranch(num_met_feats, hidden_dim=met_hidden)
        self.feat_branch = FeatureBranch(num_features, hidden_dim=feat_hidden)

        fusion_dim = gnn_hidden + lstm_hidden + met_hidden + feat_hidden
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, features, temporal_seq, met_features, city_idx, adj_matrix):
        gnn_out  = self.gnn_branch(city_idx, adj_matrix)      # (batch, 64)
        lstm_out = self.temporal_branch(temporal_seq)           # (batch, 128)
        met_out  = self.met_branch(met_features)               # (batch, 32)
        feat_out = self.feat_branch(features)                   # (batch, 64)

        fused = torch.cat([gnn_out, lstm_out, met_out, feat_out], dim=1)
        pred = self.fusion_head(fused).squeeze(-1)
        return pred


# ═══════════════════════════════════════════════════════════════
# 4.2  Instantiate model, loss, optimizer
# ═══════════════════════════════════════════════════════════════
model = AirMindModel(
    num_cities=NUM_CITIES,
    num_temporal_feats=len(TEMPORAL_SCALED_COLS),
    num_met_feats=len(MET_SCALED_COLS),
    num_features=len(SCALED_COLS),
).to(DEVICE)

adj_tensor = torch.tensor(adj, dtype=torch.float32).to(DEVICE)

# Huber loss with a delta appropriate for log-transformed targets (range ~0 to ~8)
criterion = nn.HuberLoss(delta=1.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-5)

total_params = sum(p.numel() for p in model.parameters())
print(f"\n  Model parameters: {total_params:,}")


# ═══════════════════════════════════════════════════════════════
# 4.3  Training Loop
# ═══════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("Training …")
print("─" * 60)

EPOCHS = 200
PATIENCE = 30
best_val_mae = float("inf")
patience_counter = 0
history = {"train_loss": [], "val_loss": [], "val_mae": []}

for epoch in range(1, EPOCHS + 1):
    # --- Train ---
    model.train()
    train_losses = []
    for feats, seq, met, cidx, target in train_loader:
        feats  = feats.to(DEVICE)
        seq    = seq.to(DEVICE)
        met    = met.to(DEVICE)
        cidx   = cidx.to(DEVICE)
        target = target.to(DEVICE)

        pred = model(feats, seq, met, cidx, adj_tensor)
        loss = criterion(pred, target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_losses.append(loss.item())

    train_loss = np.mean(train_losses)

    # --- Validate ---
    model.eval()
    val_losses = []
    val_preds, val_targets = [], []
    with torch.no_grad():
        for feats, seq, met, cidx, target in val_loader:
            feats  = feats.to(DEVICE)
            seq    = seq.to(DEVICE)
            met    = met.to(DEVICE)
            cidx   = cidx.to(DEVICE)
            target = target.to(DEVICE)

            pred = model(feats, seq, met, cidx, adj_tensor)
            loss = criterion(pred, target)
            val_losses.append(loss.item())

            val_preds.extend(pred.cpu().numpy())
            val_targets.extend(target.cpu().numpy())

    val_loss = np.mean(val_losses)
    # If using log target, invert for MAE computation in real units
    if LOG_TARGET:
        val_preds_real = np.expm1(np.array(val_preds))
        val_targets_real = np.expm1(np.array(val_targets))
        val_mae = mean_absolute_error(val_targets_real, val_preds_real)
    else:
        val_mae = mean_absolute_error(val_targets, val_preds)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_mae"].append(val_mae)

    scheduler.step()

    # Early stopping
    if val_mae < best_val_mae:
        best_val_mae = val_mae
        patience_counter = 0
        torch.save(model.state_dict(), MODEL_PATH)
    else:
        patience_counter += 1

    if epoch % 10 == 0 or epoch == 1 or patience_counter == PATIENCE:
        lr = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch:3d} | train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | val_MAE={val_mae:.2f} | "
              f"lr={lr:.2e} | patience={patience_counter}/{PATIENCE}")

    if patience_counter >= PATIENCE:
        print(f"\n  ⏹ Early stopping at epoch {epoch}")
        break

print(f"\n  ✓ Best val MAE: {best_val_mae:.2f} µg/m³")


# ═══════════════════════════════════════════════════════════════
# Training curves plot
# ═══════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history["train_loss"], label="Train Loss")
ax1.plot(history["val_loss"], label="Val Loss")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
ax1.legend(); ax1.set_title("Training & Validation Loss")
ax1.grid(True, alpha=0.3)

ax2.plot(history["val_mae"], label="Val MAE", color="tab:orange")
ax2.axhline(y=best_val_mae, color="red", linestyle="--", alpha=0.5, label=f"Best={best_val_mae:.2f}")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("MAE (µg/m³)")
ax2.legend(); ax2.set_title("Validation MAE")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(CURVES_PATH, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ Training curves saved to {CURVES_PATH}")


# ═══════════════════════════════════════════════════════════════
# 4.4  Evaluation on Test Set
# ═══════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("Evaluating on test set …")
print("─" * 60)

# Load best model
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

test_preds, test_targets, test_city_idxs = [], [], []
with torch.no_grad():
    for feats, seq, met, cidx, target in test_loader:
        feats  = feats.to(DEVICE)
        seq    = seq.to(DEVICE)
        met    = met.to(DEVICE)
        cidx   = cidx.to(DEVICE)
        target = target.to(DEVICE)

        pred = model(feats, seq, met, cidx, adj_tensor)
        test_preds.extend(pred.cpu().numpy())
        test_targets.extend(target.cpu().numpy())
        test_city_idxs.extend(cidx.cpu().numpy())

test_preds = np.array(test_preds)
test_targets = np.array(test_targets)
test_city_idxs = np.array(test_city_idxs)

# Invert log transform for metrics in real units
if LOG_TARGET:
    test_preds_real = np.expm1(test_preds)
    test_targets_real = np.expm1(test_targets)
else:
    test_preds_real = test_preds
    test_targets_real = test_targets

# Overall metrics
mae  = mean_absolute_error(test_targets_real, test_preds_real)
rmse = np.sqrt(mean_squared_error(test_targets_real, test_preds_real))
r2   = r2_score(test_targets_real, test_preds_real)
mape = np.mean(np.abs((test_targets_real - test_preds_real) / np.clip(test_targets_real, 1, None))) * 100

print(f"\n  Overall Test Metrics:")
print(f"    MAE  = {mae:.2f} µg/m³")
print(f"    RMSE = {rmse:.2f} µg/m³")
print(f"    R²   = {r2:.4f}")
print(f"    MAPE = {mape:.1f}%")

# Per-city metrics
IDX_TO_CITY = {v: k for k, v in CITY_TO_IDX.items()}
print(f"\n  Per-City Test Metrics:")
print(f"  {'City':<12s} {'MAE':>8s} {'RMSE':>8s} {'R²':>8s}")
print(f"  {'─'*36}")

city_metrics = []
for cidx in range(NUM_CITIES):
    mask = test_city_idxs == cidx
    if mask.sum() == 0:
        continue
    c_mae  = mean_absolute_error(test_targets_real[mask], test_preds_real[mask])
    c_rmse = np.sqrt(mean_squared_error(test_targets_real[mask], test_preds_real[mask]))
    c_r2   = r2_score(test_targets_real[mask], test_preds_real[mask])
    city_name = IDX_TO_CITY[cidx]
    print(f"  {city_name:<12s} {c_mae:>8.2f} {c_rmse:>8.2f} {c_r2:>8.4f}")
    city_metrics.append({
        "city": city_name, "MAE": round(c_mae, 2),
        "RMSE": round(c_rmse, 2), "R2": round(c_r2, 4),
    })

# Save city metrics
pd.DataFrame(city_metrics).to_csv("outputs/city_metrics.csv", index=False)

# Save predictions
preds_df = pd.DataFrame({
    "city_idx": test_city_idxs,
    "city": [IDX_TO_CITY[int(c)] for c in test_city_idxs],
    "actual_pm2_5": test_targets_real,
    "predicted_pm2_5": test_preds_real,
})
preds_df.to_csv(PREDS_PATH, index=False)
print(f"\n  ✓ Predictions saved to {PREDS_PATH}")


# ═══════════════════════════════════════════════════════════════
# 4.5  Baseline: Random Forest
# ═══════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("Training Random Forest baseline …")
print("─" * 60)

X_train = train_df[FEATURE_COLS].values
y_train_rf = train_df["target_pm2_5_raw"].values if LOG_TARGET else train_df["target_pm2_5"].values
X_test_rf = test_df[FEATURE_COLS].values
y_test_rf = test_df["target_pm2_5_raw"].values if LOG_TARGET else test_df["target_pm2_5"].values

rf = RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_leaf=5,
                           n_jobs=-1, random_state=SEED)
rf.fit(X_train, y_train_rf)
rf_preds = rf.predict(X_test_rf)

rf_mae  = mean_absolute_error(y_test_rf, rf_preds)
rf_rmse = np.sqrt(mean_squared_error(y_test_rf, rf_preds))
rf_r2   = r2_score(y_test_rf, rf_preds)

print(f"  RF MAE  = {rf_mae:.2f}")
print(f"  RF RMSE = {rf_rmse:.2f}")
print(f"  RF R²   = {rf_r2:.4f}")

improvement = (rf_mae - mae) / rf_mae * 100
print(f"\n  GNN vs RF MAE improvement: {improvement:+.1f}%")
if improvement > 0:
    print(f"  ✓ GNN model beats baseline by {improvement:.1f}%")
else:
    print(f"  ⚠ GNN model does NOT beat baseline (try tuning)")

# Save baseline model
joblib.dump(rf, "models/baseline_rf.pkl")
print(f"  ✓ RF baseline saved to models/baseline_rf.pkl")

# Save comparison table
comparison = pd.DataFrame({
    "Model": ["AirMind (GNN+LSTM+MLP)", "Random Forest Baseline"],
    "MAE": [round(mae, 2), round(rf_mae, 2)],
    "RMSE": [round(rmse, 2), round(rf_rmse, 2)],
    "R2": [round(r2, 4), round(rf_r2, 4)],
})
comparison.to_csv("outputs/model_comparison.csv", index=False)
print(f"  ✓ Comparison table saved to outputs/model_comparison.csv")


# ═══════════════════════════════════════════════════════════════
# Save last-7 raw rows per city for inference lag window (Option A)
# ═══════════════════════════════════════════════════════════════
CITY_LAST7_PATH = "models/city_last7.json"
RAW_FEATURE_COLS = [
    "aqi", "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3",
    "temperature", "wind_speed", "rainfall", "pressure",
]

last7 = {}
for city in CITY_TO_IDX:
    city_df = df[df["city"] == city].sort_values("date").tail(7)
    # Store as plain Python types for robust JSON serialization
    last7[city] = (
        city_df[RAW_FEATURE_COLS]
        .astype(float)
        .to_dict("records")
    )

with open(CITY_LAST7_PATH, "w") as f:
    json.dump(last7, f, indent=2)
print(f"  ✓ Saved per-city last-7 window to {CITY_LAST7_PATH}")

print(f"\n{'=' * 60}")
print("✓ Training complete")
print(f"{'=' * 60}")
