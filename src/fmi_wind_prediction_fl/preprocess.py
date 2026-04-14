"""
preprocess.py  –  Build features, label, split, and standardise for each station.

Pipeline per station
--------------------
  raw CSV  →  encode wind direction  →  create next-hour label  →
  drop bad rows  →  chronological 70 / 15 / 15 split  →
  fit scaler on train  →  standardise X  →  save .npz

Feature vector  x ∈ R⁵  at hour t:
  [ ws_t,  sin(wd_t),  cos(wd_t),  pres_t,  temp_t ]

  Why sin/cos for wind direction?
  Wind direction is circular: 359° and 1° differ by 2°, not 358°.
  Encoding on the unit circle (sin, cos) fixes this discontinuity.

Label  y  at hour t:
  ws_{t+1}  (next-hour wind speed, standardised — see below)

  A row is only kept if the *next* timestamp is exactly 1 hour later.
  Rows that precede a data gap get a nonsense label and are dropped.

Standardisation:
  X ← (X − μ_X) / σ_X   where μ_X, σ_X are computed from the training set only.
  y ← (y − μ_y) / σ_y   same principle — train-set mean and std of the label.

  Why standardise y even though PLANNING.md says "standardize features"?
  The model is  ŷ = wᵀx  (no intercept).  If x has zero mean but y has a
  large non-zero mean (e.g. 7 m/s for exposed stations), the model is forced
  to use its weights to "explain" the mean rather than the variation, and the
  minimum achievable MSE is  E[y]² + Var(y)(1−R²)  instead of  Var(y)(1−R²).
  For an exposed station, E[y]² ≈ 49 m²/s² — larger than any signal the
  model can explain — so the model would perform worse than a constant
  predictor.  Standardising y centres it at zero, making the no-intercept
  model well-posed and the MSE interpretable: it starts at 1.0 and converges
  toward 1−R² (coefficient of determination).

  Back-transformation for plots and final reporting:
    ŷ_raw [m/s] = ŷ_std × scaler_y_std  +  scaler_y_mean

Saved per station  →  data/processed/<slug>.npz
  X_train  (n_train, 5)   y_train  (n_train,)   ← y is standardised
  X_val    (n_val,   5)   y_val    (n_val,)
  X_test   (n_test,  5)   y_test   (n_test,)
  scaler_mean   (5,)   scaler_std   (5,)         ← for X
  scaler_y_mean (scalar)  scaler_y_std (scalar)  ← for y (back-transform)
  t_train  (n_train,)      t_val  (n_val,)   t_test  (n_test,)
    ↑ ISO-string timestamps — used by plots.py for the x-axis

Usage
-----
  python -m fmi_wind_prediction_fl.preprocess
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fmi_wind_prediction_fl.fetch_data import STATIONS

# ── Paths & constants ─────────────────────────────────────────────────────────
RAW_DIR       = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

FEATURE_NAMES = ["ws", "sin_wd", "cos_wd", "pres", "temp"]  # order matters

TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
# test fraction is the remainder: 1 − 0.70 − 0.15 = 0.15


# ── Pipeline steps ─────────────────────────────────────────────────────────────

def _build_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Turn the raw CSV into a clean (features + label) DataFrame.

    Steps
    -----
    1. Parse timestamps and sort by time.
    2. Encode wind direction as sin/cos (degrees → radians first).
    3. Create label y = next-row wind speed.
    4. Drop rows where the next timestamp is not exactly +1 hour
       (those would give a wrong "next-hour" label).
    5. Drop rows with any remaining NaN in features or label.

    Returns a DataFrame with columns:
        time, ws, sin_wd, cos_wd, pres, temp, y
    """
    df = df_raw.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    # Sin/cos encoding of wind direction
    wd_rad = np.deg2rad(df["wd_deg"])
    df["sin_wd"] = np.sin(wd_rad)
    df["cos_wd"] = np.cos(wd_rad)

    # Next-hour label: wind speed one row forward
    df["y"] = df["ws"].shift(-1)

    # Flag rows where the gap to the next observation is not exactly 1 hour.
    # shift(-1) on time gives the *next* row's timestamp; difference = gap.
    df["_next_time"] = df["time"].shift(-1)
    gap_hours = (df["_next_time"] - df["time"]).dt.total_seconds() / 3600
    bad_label = gap_hours != 1.0          # True for gap rows and the last row
    df.loc[bad_label, "y"] = np.nan       # nullify bad labels
    df = df.drop(columns=["_next_time"])

    # Keep only the columns we need, then drop any remaining NaN
    keep = ["time"] + FEATURE_NAMES + ["y"]
    df = df[keep].dropna().reset_index(drop=True)

    return df


def _chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split df into (train, val, test) by row position (time is already sorted)."""
    n = len(df)
    i_val  = int(n * TRAIN_FRAC)
    i_test = int(n * (TRAIN_FRAC + VAL_FRAC))

    train = df.iloc[:i_val].reset_index(drop=True)
    val   = df.iloc[i_val:i_test].reset_index(drop=True)
    test  = df.iloc[i_test:].reset_index(drop=True)
    return train, val, test


def _fit_scaler(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean and std from X (training set). std uses ddof=0."""
    mean = X.mean(axis=0)           # shape (5,)
    std  = X.std(axis=0, ddof=0)    # shape (5,)
    return mean, std


def _apply_scaler(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Standardise X: subtract mean, divide by std."""
    return (X - mean) / std


# ── Main per-station processor ─────────────────────────────────────────────────

def process_station(station: dict) -> dict:
    """
    Full pipeline for one station.

    Returns a summary dict (for the verification table) and saves a .npz file.
    """
    slug = station["slug"]
    name = station["name"]

    # 1. Load raw CSV
    raw_path = RAW_DIR / f"{slug}.csv"
    df_raw = pd.read_csv(raw_path)

    raw_rows = len(df_raw)

    # 2. Build features + label, drop bad rows
    df = _build_dataframe(df_raw)
    clean_rows = len(df)
    dropped = raw_rows - clean_rows

    # 3. Chronological split
    train, val, test = _chronological_split(df)

    # 4. Extract numpy arrays
    X_train = train[FEATURE_NAMES].to_numpy(dtype=np.float64)
    y_train = train["y"].to_numpy(dtype=np.float64)
    X_val   = val[FEATURE_NAMES].to_numpy(dtype=np.float64)
    y_val   = val["y"].to_numpy(dtype=np.float64)
    X_test  = test[FEATURE_NAMES].to_numpy(dtype=np.float64)
    y_test  = test["y"].to_numpy(dtype=np.float64)

    t_train = train["time"].astype(str).to_numpy()
    t_val   = val["time"].astype(str).to_numpy()
    t_test  = test["time"].astype(str).to_numpy()

    # 5. Fit scalers on training data only, then standardise all splits
    #    5a. Feature scaler (X)
    scaler_mean, scaler_std = _fit_scaler(X_train)
    X_train = _apply_scaler(X_train, scaler_mean, scaler_std)
    X_val   = _apply_scaler(X_val,   scaler_mean, scaler_std)
    X_test  = _apply_scaler(X_test,  scaler_mean, scaler_std)

    #    5b. Label scaler (y) — fit on y_train only
    scaler_y_mean = float(y_train.mean())
    scaler_y_std  = float(y_train.std(ddof=0))
    y_train = (y_train - scaler_y_mean) / scaler_y_std
    y_val   = (y_val   - scaler_y_mean) / scaler_y_std
    y_test  = (y_test  - scaler_y_mean) / scaler_y_std

    # 6. Save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"{slug}.npz"
    np.savez(
        out_path,
        X_train=X_train, y_train=y_train,
        X_val=X_val,     y_val=y_val,
        X_test=X_test,   y_test=y_test,
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
        scaler_y_mean=np.float64(scaler_y_mean),
        scaler_y_std=np.float64(scaler_y_std),
        t_train=t_train, t_val=t_val, t_test=t_test,
    )

    # Post-standardisation check on X_train (should be ~0 mean, ~1 std)
    post_mean = X_train.mean(axis=0)
    post_std  = X_train.std(axis=0, ddof=0)

    return dict(
        name        = name,
        type        = station["type"],
        raw_rows    = raw_rows,
        dropped     = dropped,
        n_train     = len(y_train),
        n_val       = len(y_val),
        n_test      = len(y_test),
        train_range = f"{t_train[0][:10]} → {t_train[-1][:10]}",
        val_range   = f"{t_val[0][:10]}   → {t_val[-1][:10]}",
        test_range  = f"{t_test[0][:10]}  → {t_test[-1][:10]}",
        scaler_mean = scaler_mean,
        scaler_std  = scaler_std,
        post_mean   = post_mean,
        post_std    = post_std,
    )


# ── Verification helpers ──────────────────────────────────────────────────────

def _print_split_table(summaries: list[dict]) -> None:
    print(
        f"\n{'Station':<28} {'Type':<10}  "
        f"{'raw':>5} {'drop':>4}  "
        f"{'train':>5} {'val':>5} {'test':>5}  "
        f"{'Train range':<22}  {'Val range':<22}  {'Test range'}"
    )
    print("-" * 130)
    for s in summaries:
        print(
            f"{s['name']:<28} {s['type']:<10}  "
            f"{s['raw_rows']:>5} {s['dropped']:>4}  "
            f"{s['n_train']:>5} {s['n_val']:>5} {s['n_test']:>5}  "
            f"{s['train_range']:<22}  {s['val_range']:<22}  {s['test_range']}"
        )


def _print_scaler_table(summaries: list[dict]) -> None:
    """
    For each station print the raw training-set statistics (= scaler values)
    and the post-standardisation mean/std of X_train (should be ≈ 0 and ≈ 1).
    """
    fw = 8  # field width for numbers
    header = f"  {'Feature':<9}" + "".join(f"  {'μ_raw':>{fw}} {'σ_raw':>{fw}}  {'μ_std':>{fw}} {'σ_std':>{fw}}")
    print(f"\n{'Station':<28}  Feature stats (raw train mean/std → post-standardisation mean/std)")
    print(f"{'':28}  {'':<9}  {'─── raw ───':>{2*fw+2}}    {'─ post-std ─':>{2*fw+2}}")

    for s in summaries:
        print(f"\n  {s['name']}")
        for j, fname in enumerate(FEATURE_NAMES):
            print(
                f"    {fname:<9}  "
                f"μ_raw={s['scaler_mean'][j]:>{fw}.3f}  σ_raw={s['scaler_std'][j]:>{fw}.3f}  "
                f"μ_std={s['post_mean'][j]:>{fw}.2e}  σ_std={s['post_std'][j]:>{fw}.4f}"
            )


def _print_spot_check(summaries: list[dict]) -> None:
    """
    Load the first station's NPZ and show 3 consecutive rows to verify
    that the feature→label relationship is correct.

    What to look for: x[0] (ws at t) and y (ws at t+1) should be
    close in value for calm conditions, or show a jump during storms.
    """
    station = STATIONS[0]   # Porvoo Kalbådagrund — first in registry
    npz = np.load(PROCESSED_DIR / f"{station['slug']}.npz", allow_pickle=True)
    X = npz["X_train"]
    y = npz["y_train"]
    t = npz["t_train"]

    sy_mean = float(npz["scaler_y_mean"])
    sy_std  = float(npz["scaler_y_std"])

    print(f"\n  Spot check — {station['name']} (first 5 training rows, both X and y standardised)")
    print(f"  y scaler: mean={sy_mean:.3f} m/s  std={sy_std:.3f} m/s")
    print(f"  Back-transform: y_raw = y_std × {sy_std:.3f} + {sy_mean:.3f}")
    print()
    print(f"  {'time':<22}  {'ws':>6} {'sin_wd':>7} {'cos_wd':>7} {'pres':>7} {'temp':>7}  │  {'y (std)':>8}  {'y (m/s)':>8}")
    print(f"  {'':<22}  {'(std)':>6} {'(std)':>7} {'(std)':>7} {'(std)':>7} {'(std)':>7}  │  {'':>8}  {'back-tf':>8}")
    print("  " + "─" * 90)
    for k in range(5):
        y_raw = y[k] * sy_std + sy_mean
        print(
            f"  {t[k]:<22}  "
            f"{X[k,0]:>6.2f} {X[k,1]:>7.3f} {X[k,2]:>7.3f} {X[k,3]:>7.3f} {X[k,4]:>7.3f}  │  "
            f"{y[k]:>8.3f}  {y_raw:>8.2f}"
        )
    print(
        f"\n  Verify: ws (std) at row k and y (std) at row k should move together"
        f"\n  (same wind regime — if ws is above average now, next hour tends to be too)."
        f"\n  The back-transformed y (m/s) should look like plausible wind speeds."
    )


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"\nPreprocessing {len(STATIONS)} stations  →  {PROCESSED_DIR}\n")
    print("─" * 60)

    summaries = []
    for station in STATIONS:
        print(f"  {station['name']} … ", end="", flush=True)
        s = process_station(station)
        summaries.append(s)
        print(f"train={s['n_train']}  val={s['n_val']}  test={s['n_test']}")

    # ── Verification output ───────────────────────────────────────────────────
    print("\n\n" + "=" * 130)
    print("VERIFICATION — SPLIT SIZES AND DATE RANGES")
    print("=" * 130)
    _print_split_table(summaries)
    print(
        "\n  Check: train range ≈ Jan–Sep, val ≈ Sep–Nov, test ≈ Nov–Dec  (for full-year stations)"
        "\n  'drop' = rows removed (NaN features, NaN label, or time-gap before next obs)"
    )

    print("\n\n" + "=" * 130)
    print("VERIFICATION — STANDARDISATION (μ_std should be ≈ 0, σ_std should be ≈ 1)")
    print("=" * 130)
    _print_scaler_table(summaries)

    print("\n\n" + "=" * 130)
    print("VERIFICATION — SPOT CHECK (feature→label alignment)")
    print("=" * 130)
    _print_spot_check(summaries)

    print(f"\n\nAll done. Saved {len(STATIONS)} .npz files to {PROCESSED_DIR}\n")


if __name__ == "__main__":
    main()
