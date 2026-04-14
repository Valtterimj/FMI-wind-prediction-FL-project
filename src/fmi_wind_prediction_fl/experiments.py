"""
experiments.py  –  Run all FL experiments and collect results for the report.

Runs
----
  1. Baseline    α = 0  (pure local training — no collaboration)
  2. System A    α ∈ ALPHA_GRID  (geographic proximity graph)
  3. System B    α ∈ ALPHA_GRID  (wind-correlation graph)

α selection (no leakage):
  For each system, pick α* = argmin_α  mean_i  val_MSE_i(α)  at final iteration.
  Test set is touched ONLY after α* is chosen.

Sanity checks printed here (numeric):
  4. α sweep — weight diversity transitions from local to shared as α grows
  5. Comparison table — train/val/test MSE for baseline, A*, B*
  6. Exposed vs sheltered breakdown

Saves  data/results/results.npz  with everything plots.py needs:
  loss curves, per-station test MSE, α-sweep data, weight matrices, y scalers.

Usage
-----
  python -m fmi_wind_prediction_fl.experiments
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fmi_wind_prediction_fl.fetch_data import STATIONS
from fmi_wind_prediction_fl.fl_algorithm import load_data, mse, run_fl

ALPHA_GRID  = [0.001, 0.01, 0.1, 1.0, 10.0]
ETA         = 0.01
T           = 200
N           = len(STATIONS)

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
RESULTS_DIR   = Path(__file__).resolve().parents[2] / "data" / "results"


# ── Helper functions ───────────────────────────────────────────────────────────

def _y_stds(data: list[dict]) -> np.ndarray:
    """Load per-station scaler_y_std values (needed to convert MSE → m²/s²)."""
    stds = np.zeros(N)
    for i, s in enumerate(STATIONS):
        npz = np.load(PROCESSED_DIR / f"{s['slug']}.npz", allow_pickle=True)
        stds[i] = float(npz["scaler_y_std"])
    return stds


def weight_diversity(W: np.ndarray) -> float:
    """
    Average pairwise squared distance between weight vectors.

    diversity(W) = (2 / N(N-1)) × Σ_{i<j} ||w_i − w_j||²

    Sanity check 4 interpretation:
      small α → diverse weights (each station fits its local data)
      large α → weights converge (penalty pulls stations toward consensus)
    """
    total, count = 0.0, 0
    for i in range(N):
        for j in range(i + 1, N):
            total += float(np.sum((W[i] - W[j]) ** 2))
            count += 1
    return total / count


def test_mse_per_station(data: list[dict], W: np.ndarray) -> np.ndarray:
    """Compute per-station test MSE (standardised units) for weight matrix W."""
    return np.array([mse(d["X_test"], d["y_test"], W[i]) for i, d in enumerate(data)])


def run_sweep(
    data: list[dict],
    A: np.ndarray,
    label: str,
) -> dict:
    """
    Run the GTVMin loop for every α in ALPHA_GRID.

    Returns a dict keyed by α with the full run_fl() result dict
    (W, train_mse, val_mse) plus 'weight_div' (scalar diversity metric).
    """
    results = {}
    for alpha in ALPHA_GRID:
        print(f"    α = {alpha:<6}  … ", end="", flush=True)
        r = run_fl(data, A, alpha=alpha, eta=ETA, T=T)
        r["weight_div"] = weight_diversity(r["W"])
        results[alpha] = r
        val_final = r["val_mse"][-1].mean()
        print(f"avg val MSE = {val_final:.5f}   weight div = {r['weight_div']:.4f}")
    return results


def best_alpha(sweep: dict) -> float:
    """Return the α with the lowest average final validation MSE."""
    return min(sweep, key=lambda a: sweep[a]["val_mse"][-1].mean())


# ── Sanity-check print functions ───────────────────────────────────────────────

def print_alpha_sweep(sweep_A: dict, sweep_B: dict, alpha_A: float, alpha_B: float) -> None:
    """
    Sanity check 4 — α sweep table.

    Verify:
      - val MSE forms a U-shape (or monotone) — there exists an optimal α
      - weight_diversity decreases as α grows (small α ≈ local, large α ≈ shared)
      - best α marked with *
    """
    print(f"\n  {'α':>7}  {'Sys A val MSE':>14}  {'A div':>8}  {'Sys B val MSE':>14}  {'B div':>8}")
    print("  " + "─" * 58)
    for alpha in ALPHA_GRID:
        val_A  = sweep_A[alpha]["val_mse"][-1].mean()
        val_B  = sweep_B[alpha]["val_mse"][-1].mean()
        div_A  = sweep_A[alpha]["weight_div"]
        div_B  = sweep_B[alpha]["weight_div"]
        mark_A = " ←best" if alpha == alpha_A else ""
        mark_B = " ←best" if alpha == alpha_B else ""
        print(
            f"  {alpha:>7}  {val_A:>14.5f}{mark_A:<7}  {div_A:>8.4f}"
            f"  {val_B:>14.5f}{mark_B:<7}  {div_B:>8.4f}"
        )
    print(
        "\n  'div' = avg pairwise ||wᵢ − wⱼ||² across stations."
        "\n  Expect: div decreases as α grows (small α → diverse, large α → consensus)."
    )


def print_comparison_table(
    data:      list[dict],
    y_stds:    np.ndarray,
    baseline:  dict,
    best_A:    dict,
    best_B:    dict,
    alpha_A:   float,
    alpha_B:   float,
    test_base: np.ndarray,
    test_A:    np.ndarray,
    test_B:    np.ndarray,
) -> None:
    """
    Sanity check 5 — per-station train/val/test MSE for all three methods.

    MSE is shown in both standardised units and as RMSE [m/s] (back-transformed).
    """
    # Final-iteration train and val MSE
    tr_base = baseline["train_mse"][-1]
    vl_base = baseline["val_mse"][-1]
    tr_A    = best_A["train_mse"][-1]
    vl_A    = best_A["val_mse"][-1]
    tr_B    = best_B["train_mse"][-1]
    vl_B    = best_B["val_mse"][-1]

    print(f"\n  α*: System A = {alpha_A},  System B = {alpha_B}")
    print(f"  MSE in standardised units (×σ_y² → m²/s²).  RMSE column = sqrt(test MSE) × σ_y\n")
    print(
        f"  {'Station':<28} {'Type':<10}  "
        f"{'— Baseline —':^23}  {'— System A —':^23}  {'— System B —':^23}"
    )
    print(
        f"  {'':28} {'':10}  "
        f"{'train':>7} {'val':>7} {'test':>7}  "
        f"{'train':>7} {'val':>7} {'test':>7}  "
        f"{'train':>7} {'val':>7} {'test':>7}"
    )
    print("  " + "─" * 105)

    # Per-station rows
    for i, d in enumerate(data):
        sy = y_stds[i]
        print(
            f"  {d['name']:<28} {d['type']:<10}  "
            f"{tr_base[i]:>7.4f} {vl_base[i]:>7.4f} {test_base[i]:>7.4f}  "
            f"{tr_A[i]:>7.4f} {vl_A[i]:>7.4f} {test_A[i]:>7.4f}  "
            f"{tr_B[i]:>7.4f} {vl_B[i]:>7.4f} {test_B[i]:>7.4f}"
        )

    # Averages
    print("  " + "─" * 105)
    print(
        f"  {'AVERAGE':<28} {'':10}  "
        f"{tr_base.mean():>7.4f} {vl_base.mean():>7.4f} {test_base.mean():>7.4f}  "
        f"{tr_A.mean():>7.4f} {vl_A.mean():>7.4f} {test_A.mean():>7.4f}  "
        f"{tr_B.mean():>7.4f} {vl_B.mean():>7.4f} {test_B.mean():>7.4f}"
    )

    # RMSE in m/s for test set
    print(f"\n  Test RMSE [m/s]  (back-transformed via per-station σ_y):")
    print(
        f"  {'Station':<28}  {'Baseline':>10}  {'System A':>10}  {'System B':>10}  "
        f"{'ΔA−Base':>10}  {'ΔB−Base':>10}"
    )
    print("  " + "─" * 75)
    for i, d in enumerate(data):
        sy = y_stds[i]
        rmse_base = np.sqrt(test_base[i]) * sy
        rmse_A    = np.sqrt(test_A[i])    * sy
        rmse_B    = np.sqrt(test_B[i])    * sy
        print(
            f"  {d['name']:<28}  {rmse_base:>10.3f}  {rmse_A:>10.3f}  {rmse_B:>10.3f}"
            f"  {rmse_A - rmse_base:>+10.3f}  {rmse_B - rmse_base:>+10.3f}"
        )
    avg_base = np.mean([np.sqrt(test_base[i]) * y_stds[i] for i in range(N)])
    avg_A    = np.mean([np.sqrt(test_A[i])    * y_stds[i] for i in range(N)])
    avg_B    = np.mean([np.sqrt(test_B[i])    * y_stds[i] for i in range(N)])
    print("  " + "─" * 75)
    print(
        f"  {'AVERAGE':<28}  {avg_base:>10.3f}  {avg_A:>10.3f}  {avg_B:>10.3f}"
        f"  {avg_A - avg_base:>+10.3f}  {avg_B - avg_base:>+10.3f}"
    )


def print_exposed_sheltered(
    data:      list[dict],
    y_stds:    np.ndarray,
    test_base: np.ndarray,
    test_A:    np.ndarray,
    test_B:    np.ndarray,
) -> None:
    """
    Sanity check 6 — do exposed stations benefit more from FL than sheltered ones?

    Hypothesis: exposed stations share similar synoptic wind patterns; collaboration
    should help more than for sheltered stations whose wind is dominated by local
    effects (terrain, city).
    """
    for stype in ("exposed", "sheltered"):
        idx = [i for i, d in enumerate(data) if d["type"] == stype]
        rmse_base = np.mean([np.sqrt(test_base[i]) * y_stds[i] for i in idx])
        rmse_A    = np.mean([np.sqrt(test_A[i])    * y_stds[i] for i in idx])
        rmse_B    = np.mean([np.sqrt(test_B[i])    * y_stds[i] for i in idx])
        n = len(idx)
        print(
            f"  {stype.capitalize():<10} ({n} stations)  "
            f"Baseline={rmse_base:.3f}  A={rmse_A:.3f} (Δ{rmse_A - rmse_base:+.3f})  "
            f"B={rmse_B:.3f} (Δ{rmse_B - rmse_base:+.3f})  [m/s RMSE]"
        )


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data and adjacency matrices
    print("\nLoading data and adjacency matrices … ", end="", flush=True)
    data  = load_data()
    A_geo = np.load(PROCESSED_DIR / "adj_A.npy")
    A_cor = np.load(PROCESSED_DIR / "adj_B.npy")
    y_stds = _y_stds(data)
    print("done\n")

    # 2. Baseline  (α = 0 — adjacency matrix irrelevant, pass zeros)
    print("Running baseline (α = 0) …")
    A_zero   = np.zeros((N, N))
    baseline = run_fl(data, A_zero, alpha=0.0, eta=ETA, T=T)
    print(f"  avg final val MSE = {baseline['val_mse'][-1].mean():.5f}\n")

    # 3. System A α sweep
    print("System A — geographic proximity graph:")
    sweep_A = run_sweep(data, A_geo, "A")

    # 4. System B α sweep
    print("\nSystem B — wind correlation graph:")
    sweep_B = run_sweep(data, A_cor, "B")

    # 5. Select best α (validation set only — no test data touched yet)
    alpha_A = best_alpha(sweep_A)
    alpha_B = best_alpha(sweep_B)
    best_A  = sweep_A[alpha_A]
    best_B  = sweep_B[alpha_B]
    print(f"\nBest α  →  System A: {alpha_A},  System B: {alpha_B}")

    # 6. Compute test MSE once, for baseline and the two best models
    test_base = test_mse_per_station(data, baseline["W"])
    test_A    = test_mse_per_station(data, best_A["W"])
    test_B    = test_mse_per_station(data, best_B["W"])

    # 7. Print sanity checks
    print("\n\n" + "=" * 90)
    print("SANITY CHECK 4 — α sweep (val MSE + weight diversity)")
    print("=" * 90)
    print_alpha_sweep(sweep_A, sweep_B, alpha_A, alpha_B)

    print("\n\n" + "=" * 90)
    print("SANITY CHECK 5 — comparison table (train / val / test MSE)")
    print("=" * 90)
    print_comparison_table(
        data, y_stds, baseline, best_A, best_B,
        alpha_A, alpha_B, test_base, test_A, test_B,
    )

    print("\n\n" + "=" * 90)
    print("SANITY CHECK 6 — exposed vs sheltered breakdown (test RMSE [m/s])")
    print("=" * 90)
    print()
    print_exposed_sheltered(data, y_stds, test_base, test_A, test_B)
    print(
        "\n  Hypothesis: exposed stations benefit more from FL"
        " (shared synoptic wind regime)."
        "\n  Sheltered stations are influenced by local effects"
        " (buildings, terrain, city heat).\n"
    )

    # 8. Save results for plots.py
    # Build the α-sweep summary arrays
    alphas_arr    = np.array(ALPHA_GRID)
    sweep_A_val   = np.array([sweep_A[a]["val_mse"][-1].mean() for a in ALPHA_GRID])
    sweep_B_val   = np.array([sweep_B[a]["val_mse"][-1].mean() for a in ALPHA_GRID])
    sweep_A_div   = np.array([sweep_A[a]["weight_div"]          for a in ALPHA_GRID])
    sweep_B_div   = np.array([sweep_B[a]["weight_div"]          for a in ALPHA_GRID])

    out = RESULTS_DIR / "results.npz"
    np.savez(
        out,
        # α sweep data
        alphas          = alphas_arr,
        sweep_A_val     = sweep_A_val,
        sweep_B_val     = sweep_B_val,
        sweep_A_div     = sweep_A_div,
        sweep_B_div     = sweep_B_div,
        best_alpha_A    = np.float64(alpha_A),
        best_alpha_B    = np.float64(alpha_B),
        # Baseline
        baseline_train_mse = baseline["train_mse"],   # (T+1, N)
        baseline_val_mse   = baseline["val_mse"],
        baseline_W         = baseline["W"],            # (N, 5)
        baseline_test_mse  = test_base,                # (N,)
        # System A best
        sys_A_train_mse = best_A["train_mse"],
        sys_A_val_mse   = best_A["val_mse"],
        sys_A_W         = best_A["W"],
        sys_A_test_mse  = test_A,
        # System B best
        sys_B_train_mse = best_B["train_mse"],
        sys_B_val_mse   = best_B["val_mse"],
        sys_B_W         = best_B["W"],
        sys_B_test_mse  = test_B,
        # Station metadata
        station_names  = np.array([d["name"]  for d in data]),
        station_types  = np.array([d["type"]  for d in data]),
        scaler_y_stds  = y_stds,
    )
    print(f"Results saved → {out}\n")


if __name__ == "__main__":
    main()
