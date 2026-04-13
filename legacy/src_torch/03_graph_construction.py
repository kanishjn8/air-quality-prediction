"""Archived legacy script.

Original path:
  - src/03_graph_construction.py

This was part of the earlier (pre-Plan vNext) GNN/adjacency pipeline.
Use the Plan vNext pipeline in `src/` instead.

---

"""

# Original contents (verbatim) below

"""
Step 03 — Graph Construction
Input:  data/processed/features.parquet
Output: data/processed/graph_edges.csv, data/processed/adj_matrix.npy

Builds a static spatial graph where nodes are cities and edges encode
spatial (haversine distance) + pollutant correlation dependencies.
"""

import os
import math
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ────────────────────────────────────────────────────────────────
# Paths
# ────────────────────────────────────────────────────────────────
FEATURES_PATH = "data/processed/features.parquet"
EDGES_CSV     = "data/processed/graph_edges.csv"
ADJ_NPY       = "data/processed/adj_matrix.npy"

os.makedirs("data/processed", exist_ok=True)

print("=" * 60)
print("Step 03 — Graph Construction")
print("=" * 60)

# ═══════════════════════════════════════════════════════════════
# 3.1  City Coordinates (hardcoded)
# ═══════════════════════════════════════════════════════════════
CITY_COORDS = {
    "Delhi":     (28.6139, 77.2090),
    "Bengaluru": (12.9716, 77.5946),
    "Kolkata":   (22.5726, 88.3639),
    "Hyderabad": (17.3850, 78.4867),
}

CITY_TO_IDX = {
    "Delhi": 0, "Bengaluru": 1, "Kolkata": 2, "Hyderabad": 3,
}
IDX_TO_CITY = {v: k for k, v in CITY_TO_IDX.items()}

NUM_CITIES = len(CITY_TO_IDX)

# ═══════════════════════════════════════════════════════════════
# 3.2  Distance-based Edge Weights (Haversine)
# ═══════════════════════════════════════════════════════════════
print("\n─── Haversine Distance Matrix ───")

def haversine_km(coord1, coord2):
    """Compute great-circle distance between two (lat, lon) points in km."""
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return 6371.0 * c  # Earth radius in km


dist_matrix = np.zeros((NUM_CITIES, NUM_CITIES))
for i in range(NUM_CITIES):
    for j in range(NUM_CITIES):
        city_i = IDX_TO_CITY[i]
        city_j = IDX_TO_CITY[j]
        dist_matrix[i, j] = haversine_km(CITY_COORDS[city_i], CITY_COORDS[city_j])

# Print distance matrix
cities_list = [IDX_TO_CITY[i] for i in range(NUM_CITIES)]
dist_df = pd.DataFrame(dist_matrix, index=cities_list, columns=cities_list)
print(dist_df.round(0).to_string())

# Edge weight = exp(-distance / 500)
dist_weight_matrix = np.exp(-dist_matrix / 500.0)
print("\n  ✓ Distance weights computed (scale=500 km)")


# ═══════════════════════════════════════════════════════════════
# 3.3  Correlation-based Edge Weights (Spearman on train split)
# ═══════════════════════════════════════════════════════════════
print("\n─── Spearman Correlation Matrix (pm2_5, train split) ───")

df = pd.read_parquet(FEATURES_PATH)
train_df = df[df["split"] == "train"]

# Pivot pm2_5 by date and city
pivot = train_df.pivot_table(index="date", columns="city", values="pm2_5")
pivot = pivot.dropna()

corr_weight_matrix = np.zeros((NUM_CITIES, NUM_CITIES))
for i in range(NUM_CITIES):
    for j in range(NUM_CITIES):
        city_i = IDX_TO_CITY[i]
        city_j = IDX_TO_CITY[j]
        if city_i in pivot.columns and city_j in pivot.columns:
            corr, _ = spearmanr(pivot[city_i], pivot[city_j])
            corr_weight_matrix[i, j] = max(0.0, corr)  # no negative edges
        else:
            corr_weight_matrix[i, j] = 0.0

corr_df = pd.DataFrame(corr_weight_matrix, index=cities_list, columns=cities_list)
print(corr_df.round(3).to_string())
print("  ✓ Correlation weights computed (negative → 0)")


# ═══════════════════════════════════════════════════════════════
# 3.4  Combined Adjacency Matrix
# ═══════════════════════════════════════════════════════════════
print("\n─── Combined Adjacency Matrix (0.5 × dist + 0.5 × corr) ───")

adj = 0.5 * dist_weight_matrix + 0.5 * corr_weight_matrix
np.fill_diagonal(adj, 1.0)  # self-loops

adj_df = pd.DataFrame(adj, index=cities_list, columns=cities_list)
print(adj_df.round(3).to_string())

np.save(ADJ_NPY, adj)
print(f"\n  ✓ Adjacency matrix saved to {ADJ_NPY}")


# ═══════════════════════════════════════════════════════════════
# 3.5  Edge List for GNN (COO format)
# ═══════════════════════════════════════════════════════════════
print("\n─── Edge List (threshold > 0.1) ───")

edges = []
for i in range(NUM_CITIES):
    for j in range(NUM_CITIES):
        if adj[i, j] > 0.1:
            edges.append({"src": i, "dst": j, "weight": round(adj[i, j], 6)})

edges_df = pd.DataFrame(edges)
edges_df.to_csv(EDGES_CSV, index=False)

print(f"  ✓ {len(edges_df)} edges saved to {EDGES_CSV}")
print(f"  Edge weight range: [{edges_df['weight'].min():.4f}, {edges_df['weight'].max():.4f}]")
print(f"\n{'=' * 60}")
print("✓ Graph construction complete")
print(f"{'=' * 60}")
