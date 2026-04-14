"""
graph.py  –  Build the two FL adjacency matrices.

System A — geographic proximity
  Edge criterion : k=3 nearest neighbours by flat-Earth distance (km)
  Edge weight    : exp(−d / σ),  σ = 200 km
  Rationale      : map proximity as a proxy for shared wind conditions

System B — wind correlation
  Edge criterion : top-k=3 most correlated neighbours (Pearson r)
  Edge weight    : r  (correlation value)
  Data source    : TRAINING SET wind speed only (no leakage into val/test)
  Rationale      : shared wind regime may beat map proximity

Both graphs are made undirected (symmetric) by taking the UNION of
directed kNN edges: (i, j) is an edge if i is in j's top-k OR j is in i's
top-k.  After union the weight is still symmetric (distance and correlation
are symmetric quantities).

Saves
-----
  data/processed/adj_A.npy   shape (10, 10)
  data/processed/adj_B.npy   shape (10, 10)
  data/processed/graph_map.png   side-by-side map of both networks

Usage
-----
  python -m fmi_wind_prediction_fl.graph
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fmi_wind_prediction_fl.fetch_data import STATIONS

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

N = len(STATIONS)   # 10 stations
K = 3               # neighbours per station
SIGMA_KM = 200.0    # length-scale for System A edge weights


# ── Step 1: geographic distance matrix ────────────────────────────────────────

def distance_matrix_km(stations: list[dict]) -> np.ndarray:
    """
    Return (N × N) matrix of flat-Earth distances in km.

    Flat-Earth approximation:
      dlat_km = (lat_i - lat_j) × 111
      dlon_km = (lon_i - lon_j) × 111 × cos(mean_lat)
      d = sqrt(dlat_km² + dlon_km²)

    Accurate to ~0.3 % for the distances in our region (< 600 km).
    """
    n = len(stations)
    D = np.zeros((n, n))
    for i, si in enumerate(stations):
        for j, sj in enumerate(stations):
            if i == j:
                continue
            dlat = (si["lat"] - sj["lat"]) * 111.0
            mean_lat_rad = math.radians((si["lat"] + sj["lat"]) / 2)
            dlon = (si["lon"] - sj["lon"]) * 111.0 * math.cos(mean_lat_rad)
            D[i, j] = math.sqrt(dlat**2 + dlon**2)
    return D


# ── Step 2: correlation matrix from training data ──────────────────────────────

def correlation_matrix(stations: list[dict]) -> np.ndarray:
    """
    Return (N × N) Pearson correlation matrix computed on TRAINING SET
    wind speed only.

    Steps
    -----
    1. For each station load t_train (UTC timestamps) and X_train[:, 0]
       (the wind-speed feature, standardised — Pearson r is scale-invariant).
    2. Build a DataFrame indexed by time (one column per station).
    3. Drop rows where ANY station has missing data (inner join).
    4. Compute Pearson correlation matrix.

    No val/test data is touched — no leakage.
    """
    series = {}
    for s in stations:
        npz = np.load(PROCESSED_DIR / f"{s['slug']}.npz", allow_pickle=True)
        ws  = npz["X_train"][:, 0]          # standardised wind speed
        ts  = npz["t_train"].astype(str)     # ISO-string timestamps
        series[s["name"]] = pd.Series(ws, index=pd.to_datetime(ts))

    df = pd.concat(series, axis=1, sort=True)   # outer join: NaN where a station has no obs
    df = df.dropna()                 # keep only hours present in ALL stations

    return df.corr(method="pearson").to_numpy()


# ── Step 3: build adjacency matrix from a value matrix ─────────────────────────

def _knn_adjacency(value_matrix: np.ndarray, k: int, higher_is_better: bool) -> np.ndarray:
    """
    Build a symmetric binary mask: A[i, j] = 1 iff (i, j) is a union-kNN edge.

    higher_is_better=True  → top-k largest values (correlation)
    higher_is_better=False → top-k smallest values (distance)
    """
    n = value_matrix.shape[0]
    directed = np.zeros((n, n), dtype=bool)

    for i in range(n):
        row = value_matrix[i].copy()
        row[i] = np.inf if not higher_is_better else -np.inf   # exclude self
        if higher_is_better:
            top_k = np.argsort(row)[::-1][:k]
        else:
            top_k = np.argsort(row)[:k]
        directed[i, top_k] = True

    # Symmetrise: edge exists if directed in either direction
    symmetric = directed | directed.T
    return symmetric


def build_system_A(stations: list[dict], D: np.ndarray) -> np.ndarray:
    """
    System A adjacency matrix.

    A[i, j] = exp(−d_ij / σ)  if (i, j) is a union-kNN edge
    A[i, j] = 0                otherwise
    diagonal = 0
    """
    mask = _knn_adjacency(D, k=K, higher_is_better=False)
    W = np.exp(-D / SIGMA_KM)
    A = np.where(mask, W, 0.0)
    np.fill_diagonal(A, 0.0)
    return A


def build_system_B(stations: list[dict], C: np.ndarray) -> np.ndarray:
    """
    System B adjacency matrix.

    A[i, j] = r_ij  (Pearson correlation)  if (i, j) is a union-kNN edge
    A[i, j] = 0                             otherwise
    diagonal = 0

    Note: negative correlations would produce negative edge weights, which
    in GTVMin would push weights *apart* rather than together.  A warning is
    printed if any retained edge has r < 0.
    """
    mask = _knn_adjacency(C, k=K, higher_is_better=True)
    A = np.where(mask, C, 0.0)
    np.fill_diagonal(A, 0.0)

    neg_edges = [(i, j) for i in range(N) for j in range(i+1, N)
                 if mask[i, j] and A[i, j] < 0]
    if neg_edges:
        print(f"  ⚠  System B has {len(neg_edges)} edge(s) with negative correlation:")
        for i, j in neg_edges:
            print(f"     {stations[i]['name']} ↔ {stations[j]['name']}  r={A[i,j]:.3f}")

    return A


# ── Step 4: verification helpers ──────────────────────────────────────────────

def print_neighbor_list(A: np.ndarray, stations: list[dict], label: str) -> None:
    """Print each station's neighbours and edge weights."""
    print(f"\n  {label}  (k={K}, union-symmetric)")
    print(f"  {'Station':<28}  Neighbours (weight)")
    print("  " + "─" * 72)
    for i, s in enumerate(stations):
        nbrs = [(j, A[i, j]) for j in range(N) if A[i, j] > 0]
        nbrs.sort(key=lambda x: -x[1])
        parts = ",  ".join(
            f"{stations[j]['name'].split()[-1]} ({w:.3f})" for j, w in nbrs
        )
        print(f"  {s['name']:<28}  {parts}")


def print_matrix(A: np.ndarray, stations: list[dict], label: str) -> None:
    """Print the full 10×10 adjacency matrix."""
    names = [s["name"].split()[-1][:10] for s in stations]  # last word, max 10 chars
    col_w = 7
    header = f"  {'':<14}" + "".join(f"{n:>{col_w}}" for n in names)
    print(f"\n  {label}  (10 × 10 adjacency matrix, 0 = no edge)")
    print(header)
    print("  " + "─" * (14 + col_w * N))
    for i, s in enumerate(stations):
        row_label = s["name"].split()[-1][:14]
        row = "".join(
            f"{'—':>{col_w}}" if A[i, j] == 0 else f"{A[i, j]:>{col_w}.3f}"
            for j in range(N)
        )
        print(f"  {row_label:<14}{row}")


# ── Step 5: map visualisation ──────────────────────────────────────────────────

def save_map(A_geo: np.ndarray, A_cor: np.ndarray, stations: list[dict]) -> Path:
    """
    Save a side-by-side map of the two networks.

    Circles  : stations (orange = exposed, blue = sheltered)
    Lines    : edges, linewidth ∝ weight
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    titles = ["System A — geographic proximity", "System B — wind correlation"]
    matrices = [A_geo, A_cor]
    colors = {"exposed": "darkorange", "sheltered": "steelblue"}

    for ax, A, title in zip(axes, matrices, titles):
        # Draw edges first (so they appear under the station markers)
        for i in range(N):
            for j in range(i + 1, N):
                if A[i, j] > 0:
                    lats = [stations[i]["lat"], stations[j]["lat"]]
                    lons = [stations[i]["lon"], stations[j]["lon"]]
                    ax.plot(lons, lats, color="grey",
                            linewidth=A[i, j] * 4, alpha=0.6, zorder=1)

        # Draw stations
        for s in stations:
            c = colors[s["type"]]
            ax.scatter(s["lon"], s["lat"], color=c, s=80, zorder=3)
            ax.annotate(
                s["name"].split()[-1],          # last word (e.g. "Kalbådagrund")
                (s["lon"], s["lat"]),
                textcoords="offset points", xytext=(5, 4),
                fontsize=7, color=c,
            )

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, linestyle="--", alpha=0.4)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="darkorange",
               markersize=8, label="Exposed"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="steelblue",
               markersize=8, label="Sheltered"),
    ]
    axes[0].legend(handles=legend_elements, loc="upper left", fontsize=8)

    fig.suptitle("FL network topology — 10 FMI coastal stations (2024)", fontsize=12)
    fig.tight_layout()

    out = PROCESSED_DIR / "graph_map.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"\nBuilding adjacency matrices for {N} stations\n")
    print("─" * 60)

    # Step 1 — geographic distances
    print("Computing geographic distances … ", end="", flush=True)
    D = distance_matrix_km(STATIONS)
    print("done")

    # Step 2 — wind correlations (training set only)
    print("Computing wind correlations (training set) … ", end="", flush=True)
    C = correlation_matrix(STATIONS)
    print("done")

    # Step 3 — build adjacency matrices
    A_geo = build_system_A(STATIONS, D)
    A_cor = build_system_B(STATIONS, C)

    # Step 4 — save
    np.save(PROCESSED_DIR / "adj_A.npy", A_geo)
    np.save(PROCESSED_DIR / "adj_B.npy", A_cor)
    print(f"\nSaved  adj_A.npy  and  adj_B.npy  →  {PROCESSED_DIR}")

    # Step 5 — verification output
    print("\n\n" + "=" * 90)
    print("VERIFICATION — DISTANCE MATRIX (km, first 5 stations shown)")
    print("=" * 90)
    names5 = [s["name"].split()[-1][:12] for s in STATIONS[:5]]
    print(f"  {'':14}" + "".join(f"{n:>14}" for n in names5))
    for i in range(5):
        row_label = STATIONS[i]["name"].split()[-1][:14]
        row = "".join(f"{D[i, j]:>14.0f}" for j in range(5))
        print(f"  {row_label:<14}{row}")

    print("\n\n" + "=" * 90)
    print("VERIFICATION — PEARSON CORRELATION MATRIX (training wind speed, first 5 stations)")
    print("=" * 90)
    print(f"  {'':14}" + "".join(f"{n:>14}" for n in names5))
    for i in range(5):
        row_label = STATIONS[i]["name"].split()[-1][:14]
        row = "".join(f"{C[i, j]:>14.3f}" for j in range(5))
        print(f"  {row_label:<14}{row}")

    print("\n\n" + "=" * 90)
    print("VERIFICATION — SYSTEM A NEIGHBOUR LIST")
    print("=" * 90)
    print_neighbor_list(A_geo, STATIONS, "System A — geographic (exp(−d/200km))")

    print_matrix(A_geo, STATIONS, "System A")

    print("\n\n" + "=" * 90)
    print("VERIFICATION — SYSTEM B NEIGHBOUR LIST")
    print("=" * 90)
    print_neighbor_list(A_cor, STATIONS, "System B — wind correlation (Pearson r)")

    print_matrix(A_cor, STATIONS, "System B")

    # Step 6 — map
    out = save_map(A_geo, A_cor, STATIONS)
    print(f"\n\nMap saved → {out}")
    print("Open it to visually verify that geographic edges follow proximity and")
    print("correlation edges group stations with similar wind regimes.\n")

    # Step 7 — quick sanity numbers
    n_edges_A = int(A_geo.astype(bool).sum() // 2)
    n_edges_B = int(A_cor.astype(bool).sum() // 2)
    print(f"System A: {n_edges_A} undirected edges  (expected ≥ {K}, typically {K}–{N//2})")
    print(f"System B: {n_edges_B} undirected edges")
    print(f"\nWeight range A: [{A_geo[A_geo > 0].min():.3f}, {A_geo[A_geo > 0].max():.3f}]")
    print(f"Weight range B: [{A_cor[A_cor > 0].min():.3f}, {A_cor[A_cor > 0].max():.3f}]")
    print()


if __name__ == "__main__":
    main()
