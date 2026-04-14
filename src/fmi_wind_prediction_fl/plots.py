"""
plots.py  –  Generate all report figures from saved experiment results.

Figures saved to  data/results/
  fig1_loss_curves.png   train + val MSE over 200 iterations
  fig2_scatter.png       predicted vs actual for 3 stations
  fig3_alpha_sweep.png   val MSE and weight diversity vs α
  fig4_rmse_bars.png     per-station test RMSE, all three methods

Usage
-----
  python -m fmi_wind_prediction_fl.plots
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from fmi_wind_prediction_fl.fetch_data import STATIONS

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
RESULTS_DIR   = Path(__file__).resolve().parents[2] / "data" / "results"

# ── Colour palette (consistent across all figures) ────────────────────────────
C_BASE  = "#555555"   # baseline — neutral grey
C_A     = "#E6821E"   # System A — orange (geographic / warm)
C_B     = "#3B7FD4"   # System B — blue   (correlation / cool)
C_EXP   = "#D64E12"   # exposed station accent
C_SHE   = "#2563A8"   # sheltered station accent

plt.rcParams.update({
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "legend.fontsize":   9,
    "figure.dpi":       130,
})


# ── Data loading ──────────────────────────────────────────────────────────────

def load_results() -> tuple[dict, list[dict]]:
    """
    Load results.npz and the per-station test data from processed NPZs.

    Returns
    -------
    r       : dict of all arrays saved by experiments.py
    sdata   : list of per-station dicts with X_test, y_test, y scalers
    """
    r = dict(np.load(RESULTS_DIR / "results.npz", allow_pickle=True))

    sdata = []
    for s in STATIONS:
        npz = np.load(PROCESSED_DIR / f"{s['slug']}.npz", allow_pickle=True)
        sdata.append(dict(
            name         = s["name"],
            slug         = s["slug"],
            type         = s["type"],
            X_test       = npz["X_test"],
            y_test       = npz["y_test"],
            y_mean       = float(npz["scaler_y_mean"]),
            y_std        = float(npz["scaler_y_std"]),
        ))
    return r, sdata


def _predictions_ms(X_test, W_row, y_mean, y_std):
    """Compute back-transformed predictions and actuals in m/s."""
    y_hat_std = X_test @ W_row
    return y_hat_std * y_std + y_mean


def _actuals_ms(y_test, y_mean, y_std):
    return y_test * y_std + y_mean


# ── Figure 1: Loss curves ─────────────────────────────────────────────────────

def fig_loss_curves(r: dict) -> None:
    """
    Train and validation MSE averaged over all 10 stations, over 200 iterations.

    Sanity checks verified here:
      1. Training MSE decreases monotonically.
      2. FL (System A* and B*) achieves lower final MSE than baseline.

    Y-axis is in standardised units:
      1.0 = predicting the mean (zero information)
      converges toward 1 − R²  as iterations increase.
    """
    iters = np.arange(201)
    alpha_A = float(r["best_alpha_A"])
    alpha_B = float(r["best_alpha_B"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

    for ax, key_suffix, title in [
        (axes[0], "train_mse", "Training MSE"),
        (axes[1], "val_mse",   "Validation MSE"),
    ]:
        base = r[f"baseline_{key_suffix}"].mean(axis=1)
        fl_A = r[f"sys_A_{key_suffix}"].mean(axis=1)
        fl_B = r[f"sys_B_{key_suffix}"].mean(axis=1)

        # Faint per-station lines for baseline to show spread
        for i in range(10):
            ax.plot(iters, r[f"baseline_{key_suffix}"][:, i],
                    color=C_BASE, alpha=0.10, linewidth=0.6)

        ax.plot(iters, base, color=C_BASE, linewidth=1.8, label="Baseline (α=0)",   zorder=3)
        ax.plot(iters, fl_A, color=C_A,    linewidth=1.8, label=f"System A (α={alpha_A})", zorder=4)
        ax.plot(iters, fl_B, color=C_B,    linewidth=1.8, label=f"System B (α={alpha_B})", linestyle="--", zorder=4)

        ax.set_xlabel("Iteration")
        ax.set_ylabel("MSE (standardised)")
        ax.set_title(title)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 200)

    fig.suptitle(
        "GTVMin convergence — average over 10 stations\n"
        "MSE = 1.0 at init (predicting mean), lower bound ≈ 1 − R²",
        fontsize=10,
    )
    fig.tight_layout()
    out = RESULTS_DIR / "fig1_loss_curves.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ── Figure 2: Scatter — predicted vs actual ───────────────────────────────────

def fig_scatter(r: dict, sdata: list[dict]) -> None:
    """
    Predicted vs actual next-hour wind speed [m/s] on the test set.

    Three stations chosen to represent different regimes:
      • Parainen Utö         — exposed, highest-wind open-sea station
      • Helsinki Kaisaniemi  — sheltered, urban reference
      • Vaasa Klemettilä     — sheltered, most improved by FL

    Each subplot shows baseline (grey) and best FL — System B (blue) — overlaid.
    Points near the diagonal y = x mean accurate prediction.

    Sanity check 3: scatter should be elongated along the diagonal with
    visible positive correlation (R ~ 0.9).
    """
    # Station indices: Utö=2, Kaisaniemi=5, Vaasa=9
    selected = [2, 5, 9]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    for ax, idx in zip(axes, selected):
        d   = sdata[idx]
        sym = float(d["y_std"])
        mea = float(d["y_mean"])

        y_actual = _actuals_ms(d["y_test"], mea, sym)

        y_base = _predictions_ms(d["X_test"], r["baseline_W"][idx], mea, sym)
        y_fl   = _predictions_ms(d["X_test"], r["sys_B_W"][idx],    mea, sym)

        lo = min(y_actual.min(), y_base.min(), y_fl.min()) - 0.5
        hi = max(y_actual.max(), y_base.max(), y_fl.max()) + 0.5

        ax.scatter(y_actual, y_base, color=C_BASE, s=3, alpha=0.25,
                   label="Baseline", rasterized=True)
        ax.scatter(y_actual, y_fl,   color=C_B,    s=3, alpha=0.30,
                   label="System B", rasterized=True)
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0,
                linestyle="--", label="y = x", zorder=5)

        # Pearson r for FL predictions
        r_fl = float(np.corrcoef(y_actual, y_fl)[0, 1])

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel("Actual  [m/s]")
        ax.set_ylabel("Predicted  [m/s]")
        short_name = d["name"].split()[-1]   # last word, e.g. "Utö"
        ax.set_title(f"{d['name']}\n({d['type']}, r = {r_fl:.3f})")
        ax.legend(loc="upper left", markerscale=3)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Predicted vs actual next-hour wind speed — test set (Nov–Dec 2024)\n"
        "Diagonal = perfect prediction.  r = Pearson correlation.",
        fontsize=10,
    )
    fig.tight_layout()
    out = RESULTS_DIR / "fig2_scatter.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {out.name}")


# ── Figure 3: α sweep ─────────────────────────────────────────────────────────

def fig_alpha_sweep(r: dict) -> None:
    """
    Left:  average validation MSE at final iteration vs α, for both systems.
    Right: weight diversity vs α.

    Sanity check 4:
      - val MSE should form a curve with a minimum at α* (may be monotone if
        α* lies at the grid boundary, as observed here).
      - weight_diversity must decrease monotonically — confirms small α gives
        diverse local models and large α gives shared consensus.
    """
    alphas   = r["alphas"]
    val_A    = r["sweep_A_val"]
    val_B    = r["sweep_B_val"]
    div_A    = r["sweep_A_div"]
    div_B    = r["sweep_B_div"]
    alpha_A  = float(r["best_alpha_A"])
    alpha_B  = float(r["best_alpha_B"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # ── Left: validation MSE ──────────────────────────────────────────────────
    ax = axes[0]
    ax.semilogx(alphas, val_A, "o-", color=C_A, linewidth=1.8, markersize=6,
                label="System A (geographic)")
    ax.semilogx(alphas, val_B, "s--", color=C_B, linewidth=1.8, markersize=6,
                label="System B (correlation)")
    # Mark selected α
    ax.axvline(alpha_A, color=C_A, linestyle=":", alpha=0.7, linewidth=1.2)
    ax.axvline(alpha_B, color=C_B, linestyle=":", alpha=0.7, linewidth=1.2)
    ax.set_xlabel("α  (log scale)")
    ax.set_ylabel("Avg val MSE (standardised)")
    ax.set_title("Validation MSE vs α")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    # ── Right: weight diversity ───────────────────────────────────────────────
    ax = axes[1]
    ax.semilogx(alphas, div_A, "o-", color=C_A, linewidth=1.8, markersize=6,
                label="System A")
    ax.semilogx(alphas, div_B, "s--", color=C_B, linewidth=1.8, markersize=6,
                label="System B")
    ax.set_xlabel("α  (log scale)")
    ax.set_ylabel(r"Avg pairwise $\|w_i - w_j\|^2$")
    ax.set_title("Weight diversity vs α")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    ax.annotate("diverse\n(local)", xy=(alphas[0], div_A[0]),
                xytext=(alphas[0]*1.5, div_A[0]*0.85), fontsize=8, color=C_BASE)
    ax.annotate("shared\n(consensus)", xy=(alphas[-1], div_A[-1]),
                xytext=(alphas[-2], div_A[-1] + (div_A[0]-div_A[-1])*0.08),
                fontsize=8, color=C_BASE)

    fig.suptitle(
        f"α grid search over {{0.001, 0.01, 0.1, 1, 10}}  "
        f"(selected: A={alpha_A}, B={alpha_B})\n"
        "Weight diversity confirms: large α → consensus model",
        fontsize=10,
    )
    fig.tight_layout()
    out = RESULTS_DIR / "fig3_alpha_sweep.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ── Figure 4: RMSE bar chart ──────────────────────────────────────────────────

def fig_rmse_bars(r: dict) -> None:
    """
    Per-station test RMSE [m/s] for baseline, System A*, System B*.
    Stations are grouped by type (exposed first, sheltered second).
    A dividing line and shaded backgrounds mark the two groups.

    Sanity check 5: FL bars should generally be ≤ baseline bar.
    Sanity check 6: compare average height of exposed vs sheltered groups.
    """
    y_stds     = r["scaler_y_stds"]
    test_base  = r["baseline_test_mse"]
    test_A     = r["sys_A_test_mse"]
    test_B     = r["sys_B_test_mse"]
    names      = [str(n) for n in r["station_names"]]
    types      = [str(t) for t in r["station_types"]]

    # Convert standardised MSE → RMSE in m/s
    rmse_base = np.sqrt(test_base) * y_stds
    rmse_A    = np.sqrt(test_A)    * y_stds
    rmse_B    = np.sqrt(test_B)    * y_stds

    # Order: exposed first, then sheltered
    order_exp = [i for i, t in enumerate(types) if t == "exposed"]
    order_she = [i for i, t in enumerate(types) if t == "sheltered"]
    order     = order_exp + order_she
    n_exp     = len(order_exp)
    n_she     = len(order_she)

    short_names = [names[i].split()[-1] for i in order]   # last word of name

    x     = np.arange(len(order))
    width = 0.25

    fig, ax = plt.subplots(figsize=(13, 5))

    bars_base = ax.bar(x - width, rmse_base[order], width, color=C_BASE,
                       alpha=0.85, label="Baseline (α=0)")
    bars_A    = ax.bar(x,         rmse_A[order],    width, color=C_A,
                       alpha=0.85, label=f"System A (α={float(r['best_alpha_A'])})")
    bars_B    = ax.bar(x + width, rmse_B[order],    width, color=C_B,
                       alpha=0.85, label=f"System B (α={float(r['best_alpha_B'])})")

    # Shaded backgrounds for exposed / sheltered groups
    ax.axvspan(-0.5,           n_exp - 0.5,          alpha=0.06, color=C_EXP, zorder=0)
    ax.axvspan(n_exp - 0.5,    n_exp + n_she - 0.5,  alpha=0.06, color=C_SHE, zorder=0)
    ax.axvline(n_exp - 0.5, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    # Group labels
    ax.text(n_exp / 2 - 0.5,         ax.get_ylim()[1] * 0.98,
            "Exposed", ha="center", va="top", fontsize=9, color=C_EXP, fontweight="bold")
    ax.text(n_exp + n_she / 2 - 0.5, ax.get_ylim()[1] * 0.98,
            "Sheltered", ha="center", va="top", fontsize=9, color=C_SHE, fontweight="bold")

    # Average lines per group
    for start, end, vals, color in [
        (0,     n_exp,           rmse_base[order[:n_exp]],    C_BASE),
        (n_exp, n_exp + n_she,   rmse_base[order[n_exp:]],    C_BASE),
    ]:
        avg = vals.mean()
        ax.hlines(avg, start - 0.5, end - 0.5, colors=color,
                  linestyles=":", linewidth=1.2, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=30, ha="right")
    ax.set_ylabel("Test RMSE  [m/s]")
    ax.set_xlabel("Station")
    ax.set_title(
        "Per-station test RMSE — baseline vs FL systems\n"
        "(Nov–Dec 2024 test set, back-transformed to m/s)",
        fontsize=11,
    )
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_xlim(-0.5, len(order) - 0.5)

    fig.tight_layout()
    out = RESULTS_DIR / "fig4_rmse_bars.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    print("\nLoading results … ", end="", flush=True)
    r, sdata = load_results()
    print("done\n")

    print("Generating figures:")
    fig_loss_curves(r)
    fig_scatter(r, sdata)
    fig_alpha_sweep(r)
    fig_rmse_bars(r)

    print(f"\nAll figures saved to {RESULTS_DIR}")
    print("\nWhat each figure verifies:")
    print("  fig1_loss_curves  — sanity checks 1 (monotone) and 2 (FL > baseline)")
    print("  fig2_scatter      — sanity check 3 (predictions near diagonal)")
    print("  fig3_alpha_sweep  — sanity check 4 (α controls local↔shared tradeoff)")
    print("  fig4_rmse_bars    — sanity checks 5 (comparison table) and 6 (exposed vs sheltered)")


if __name__ == "__main__":
    main()
