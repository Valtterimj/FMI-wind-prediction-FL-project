"""
fl_algorithm.py  –  GTVMin federated learning update loop (pure NumPy).

Math recap
----------
Each station i has a local linear model  ŷ = wᵢᵀ x,  wᵢ ∈ R⁵.
Local MSE loss and its gradient:
    Lᵢ(wᵢ)   = (1/mᵢ) ‖Xᵢ wᵢ − yᵢ‖²
    ∇Lᵢ(wᵢ)  = (2/mᵢ) Xᵢᵀ (Xᵢ wᵢ − yᵢ)

GTVMin adds a collaboration penalty that pulls neighbouring weights together:
    min  Σᵢ Lᵢ(wᵢ)  +  α Σᵢ,ᵢ' Aᵢᵢ' ‖wᵢ − wᵢ'‖²

The gradient of the penalty term for station i is:
    2α Σᵢ'∈N(i) Aᵢᵢ' (wᵢ − wᵢ')

Rewriting with the graph Laplacian  L = D − A  (D[i,i] = Σⱼ Aᵢⱼ):
    2α  L[i,:] · W        (one row of the matrix product  2α L W)

Synchronous update rule for all N stations in one shot:
    W[t+1] = W[t] − η (G[t] + 2α L W[t])

where  G[t][i] = ∇Lᵢ(wᵢ[t])  is the stacked (N × 5) local-gradient matrix.

"Synchronous" means all rows of W[t] are read BEFORE any row is written,
so every station updates using the same snapshot of the previous weights.

Usage
-----
  python -m fmi_wind_prediction_fl.fl_algorithm   # demo: baseline vs α=0.1

Note on units
-------------
Both X and y are standardised (zero mean, unit variance) by preprocess.py.
MSE is therefore in standardised units:
  - starts at 1.0  (predicting 0 = predicting the mean of y)
  - converges toward 1 − R²  (fraction of variance not explained)
To convert to m²/s²:  MSE_raw = MSE_std × scaler_y_std²
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fmi_wind_prediction_fl.fetch_data import STATIONS

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


# ── Core math (stateless, pure NumPy) ─────────────────────────────────────────

def mse(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """MSE loss:  (1/m) ‖Xw − y‖²"""
    residual = X @ w - y          # shape (m,)
    return float(np.mean(residual ** 2))


def local_gradient(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """MSE gradient:  (2/m) Xᵀ(Xw − y)  →  shape (5,)"""
    residual = X @ w - y          # shape (m,)
    return (2.0 / len(y)) * (X.T @ residual)


def laplacian(A: np.ndarray) -> np.ndarray:
    """Graph Laplacian  L = D − A  where  D[i,i] = Σⱼ A[i,j]."""
    return np.diag(A.sum(axis=1)) - A


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(stations: list[dict] = STATIONS) -> list[dict]:
    """
    Load preprocessed NPZ files for all stations.

    Returns a list (one dict per station) with keys:
        name, slug, type,
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    data = []
    for s in stations:
        npz = np.load(PROCESSED_DIR / f"{s['slug']}.npz", allow_pickle=True)
        data.append(dict(
            name    = s["name"],
            slug    = s["slug"],
            type    = s["type"],
            X_train = npz["X_train"],
            y_train = npz["y_train"],
            X_val   = npz["X_val"],
            y_val   = npz["y_val"],
            X_test  = npz["X_test"],
            y_test  = npz["y_test"],
        ))
    return data


# ── GTVMin update loop ─────────────────────────────────────────────────────────

def run_fl(
    data:  list[dict],
    A:     np.ndarray,
    alpha: float,
    eta:   float = 0.01,
    T:     int   = 200,
) -> dict:
    """
    Run the GTVMin synchronous update loop.

    Parameters
    ----------
    data  : list of per-station dicts from load_data()
    A     : (N × N) adjacency matrix.  Pass np.zeros((N,N)) for α=0 baseline.
    alpha : collaboration strength (≥ 0)
    eta   : learning rate
    T     : number of update steps

    Returns
    -------
    dict with keys:
      'W'          : (N, 5) final weight matrix
      'train_mse'  : (T+1, N) MSE on training set  — row 0 = initial (W=0)
      'val_mse'    : (T+1, N) MSE on validation set
    """
    N = len(data)
    L = laplacian(A)              # (N, N) graph Laplacian

    W = np.zeros((N, 5))          # initialise all weights to zero

    train_mse = np.zeros((T + 1, N))
    val_mse   = np.zeros((T + 1, N))

    # ── record MSE at iteration 0 (before any update) ─────────────────────────
    for i, d in enumerate(data):
        train_mse[0, i] = mse(d["X_train"], d["y_train"], W[i])
        val_mse[0, i]   = mse(d["X_val"],   d["y_val"],   W[i])

    # ── main loop ──────────────────────────────────────────────────────────────
    for t in range(T):
        # Local gradients  G[i] = ∇Lᵢ(wᵢ)  for all stations
        G = np.zeros((N, 5))
        for i, d in enumerate(data):
            G[i] = local_gradient(d["X_train"], d["y_train"], W[i])

        # Collaboration term: 2α L W  — shape (N, 5)
        # L @ W[i] = Σⱼ L[i,j] W[j] = d_i W[i] − Σⱼ A[i,j] W[j]
        # This nudges W[i] toward the weighted average of its neighbours.
        collab = 2.0 * alpha * (L @ W)

        # Synchronous update: uses W[t] for BOTH G and collab
        W = W - eta * (G + collab)

        # Record MSE after update  (iteration t+1)
        for i, d in enumerate(data):
            train_mse[t + 1, i] = mse(d["X_train"], d["y_train"], W[i])
            val_mse[t + 1, i]   = mse(d["X_val"],   d["y_val"],   W[i])

    return {"W": W, "train_mse": train_mse, "val_mse": val_mse}


# ── Demo / verification ────────────────────────────────────────────────────────

def _check_monotone(train_mse: np.ndarray) -> tuple[bool, int]:
    """
    Check whether average training MSE decreases monotonically.

    Returns (is_monotone, first_violation_iteration).
    For a convex MSE loss with appropriate η this must hold.
    """
    avg = train_mse.mean(axis=1)         # shape (T+1,)
    violations = np.where(np.diff(avg) > 1e-10)[0]   # allow tiny float noise
    if len(violations) == 0:
        return True, -1
    return False, int(violations[0]) + 1


def main() -> None:
    print("\nLoading data … ", end="", flush=True)
    data = load_data()
    print("done")

    A_geo = np.load(PROCESSED_DIR / "adj_A.npy")
    N = len(data)

    # ── Run 1: baseline  α = 0  (no collaboration) ────────────────────────────
    print("\nRunning baseline  (α = 0) …")
    res_base = run_fl(data, A_geo, alpha=0.0, eta=0.01, T=200)

    # ── Run 2: FL with System A, α = 0.1 ──────────────────────────────────────
    print("Running System A  (α = 0.1, η = 0.01, T = 200) …")
    res_fl = run_fl(data, A_geo, alpha=0.1, eta=0.01, T=200)

    # ── Sanity check 1: monotonically decreasing training loss ────────────────
    print("\n\n" + "=" * 80)
    print("SANITY CHECK 1 — training MSE must decrease monotonically")
    print("=" * 80)
    for label, res in [("Baseline", res_base), ("System A α=0.1", res_fl)]:
        ok, first = _check_monotone(res["train_mse"])
        status = "✓ monotone" if ok else f"✗ first violation at iter {first}"
        print(f"  {label:<20}  {status}")

    # ── Progress table (every 20 iterations) ──────────────────────────────────
    print("\n\n" + "=" * 80)
    print("PROGRESS — average MSE across all stations (System A, α=0.1)")
    print("=" * 80)
    print(f"\n  {'Iter':>5}  {'Train MSE':>12}  {'Val MSE':>12}")
    print("  " + "─" * 32)
    for t in [0, 10, 20, 50, 100, 150, 200]:
        tr = res_fl["train_mse"][t].mean()
        vl = res_fl["val_mse"][t].mean()
        print(f"  {t:>5}  {tr:>12.4f}  {vl:>12.4f}")

    # ── Sanity check 2: FL vs baseline, per station ───────────────────────────
    print("\n\n" + "=" * 80)
    print("SANITY CHECK 2 — FL (α=0.1) vs baseline  (val MSE, m²/s²)")
    print("=" * 80)
    print(f"\n  {'Station':<28}  {'Type':<10}  {'Baseline':>10}  {'α=0.1':>10}  {'Δ':>8}  {'Better?':>8}")
    print("  " + "─" * 78)
    improvements = 0
    for i, d in enumerate(data):
        base_val = res_base["val_mse"][-1, i]
        fl_val   = res_fl["val_mse"][-1, i]
        delta    = fl_val - base_val
        better   = "✓" if delta < 0 else "✗"
        if delta < 0:
            improvements += 1
        print(
            f"  {d['name']:<28}  {d['type']:<10}  "
            f"{base_val:>10.4f}  {fl_val:>10.4f}  {delta:>+8.4f}  {better:>8}"
        )
    print(f"\n  FL improves on baseline for {improvements}/{N} stations at α=0.1")
    print("  (α=0.1 may not be the best α — experiments.py will sweep over α)\n")


if __name__ == "__main__":
    main()
