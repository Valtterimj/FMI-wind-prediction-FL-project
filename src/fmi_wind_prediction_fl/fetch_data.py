"""
fetch_data.py  –  Pull hourly FMI observations for 10 stations, full year 2024.

Each station gets one CSV saved to   data/raw/<slug>.csv
Columns:  time (UTC, naive), ws [m/s], wd_deg [°], pres [hPa], temp [°C]

Usage
-----
  # Step 1 – probe: fetch one hour, print raw API structure and verify keys
  python -m fmi_wind_prediction_fl.fetch_data --probe

  # Step 2 – full fetch (≈120 API calls, a few minutes)
  python -m fmi_wind_prediction_fl.fetch_data
"""

from __future__ import annotations

import argparse
import time
from calendar import monthrange
from pathlib import Path

import pandas as pd
from fmiopendata.wfs import download_stored_query

# ── Station registry ───────────────────────────────────────────────────────────
# All FMISIDs verified against the FMI WFS API (bbox query on 2024-06-01).
# lat/lon are the values reported by the API via location_metadata.
# 'type' is the exposed/sheltered classification from PLANNING.md.
#
# NOTE: Rauma Kylmäpihlaja is an offshore island lighthouse — it may behave
# more like an "exposed" station despite being listed as "sheltered" in the plan.
# The API name is printed at runtime so you can verify each station.
STATIONS: list[dict] = [
    # Exposed — open sea / archipelago
    {"name": "Porvoo Kalbådagrund",    "slug": "kalbaadagrund", "fmisid": 101022, "lat": 59.98568, "lon": 25.59879, "type": "exposed"},
    {"name": "Hanko Russarö",           "slug": "russaro",       "fmisid": 100932, "lat": 59.77363, "lon": 22.94868, "type": "exposed"},
    {"name": "Parainen Utö",           "slug": "uto",           "fmisid": 100908, "lat": 59.77909, "lon": 21.37479, "type": "exposed"},
    {"name": "Kustavi Isokari",        "slug": "isokari",       "fmisid": 101059, "lat": 60.72220, "lon": 21.02681, "type": "exposed"},
    {"name": "Mustasaari Valassaaret", "slug": "valassaaret",   "fmisid": 101464, "lat": 63.43508, "lon": 21.06856, "type": "exposed"},
    # Sheltered — coastal towns
    {"name": "Helsinki Kaisaniemi",    "slug": "kaisaniemi",    "fmisid": 100971, "lat": 60.17523, "lon": 24.94459, "type": "sheltered"},
    {"name": "Turku Artukainen",        "slug": "artukainen",    "fmisid": 100949, "lat": 60.45439, "lon": 22.17870, "type": "sheltered"},
    {"name": "Rauma Kylmäpihlaja",     "slug": "kylmapihlaja",  "fmisid": 101061, "lat": 61.14475, "lon": 21.30273, "type": "sheltered"},
    {"name": "Pori Tahkoluoto",        "slug": "tahkoluoto",    "fmisid": 101267, "lat": 61.63042, "lon": 21.37620, "type": "sheltered"},
    {"name": "Vaasa Klemettilä",       "slug": "vaasa",         "fmisid": 101485, "lat": 63.09871, "lon": 21.63938, "type": "sheltered"},
]

# Parameter names exactly as returned by fmiopendata for the multipointcoverage query.
# Confirmed by running --probe against Helsinki Kaisaniemi.
PARAM_MAP: dict[str, str] = {
    "Wind speed":    "ws",
    "Wind direction": "wd_deg",
    "Air pressure":  "pres",
    "Air temperature": "temp",
}

YEAR = 2024
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
API_SLEEP = 0.5  # seconds between calls — polite to the FMI API


# ── Helpers ────────────────────────────────────────────────────────────────────

def _month_ranges(year: int) -> list[tuple[str, str]]:
    """Return (starttime_iso, endtime_iso) for each calendar month in year."""
    ranges = []
    for month in range(1, 13):
        _, last_day = monthrange(year, month)
        start = f"{year}-{month:02d}-01T00:00:00Z"
        end   = f"{year}-{month:02d}-{last_day:02d}T23:59:59Z"
        ranges.append((start, end))
    return ranges


def _fetch_month(fmisid: int, start: str, end: str) -> pd.DataFrame:
    """Fetch one month of hourly observations for one station.

    Returns a DataFrame with columns:
        time (datetime, UTC naive), station_api_name (str),
        ws, wd_deg, pres, temp (float or NaN).
    """
    obs = download_stored_query(
        "fmi::observations::weather::hourly::multipointcoverage",
        args=[
            f"fmisid={fmisid}",
            f"starttime={start}",
            f"endtime={end}",
        ],
    )

    rows = []
    for ts, locations in obs.data.items():
        for loc_name, params in locations.items():
            row: dict = {"time": ts, "station_api_name": loc_name}
            for api_key, col_name in PARAM_MAP.items():
                entry = params.get(api_key)
                row[col_name] = entry["value"] if isinstance(entry, dict) else float("nan")
            rows.append(row)

    if not rows:
        empty_cols = ["time", "station_api_name", "ws", "wd_deg", "pres", "temp"]
        return pd.DataFrame(columns=empty_cols)

    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"])  # naive UTC
    return df.sort_values("time").reset_index(drop=True)


# ── Main fetch logic ───────────────────────────────────────────────────────────

def fetch_station(station: dict) -> pd.DataFrame:
    """Fetch all 12 months of YEAR for one station. Prints per-month progress."""
    fmisid = station["fmisid"]
    months = _month_ranges(YEAR)
    frames: list[pd.DataFrame] = []

    for i, (start, end) in enumerate(months, 1):
        month_label = start[:7]  # "YYYY-MM"
        print(f"    [{i:2d}/12] {month_label} … ", end="", flush=True)
        try:
            df = _fetch_month(fmisid, start, end)
            frames.append(df)
            print(f"{len(df):4d} rows")
        except Exception as exc:
            print(f"ERROR — {exc}")
        time.sleep(API_SLEEP)

    if not frames:
        return pd.DataFrame()

    full = pd.concat(frames, ignore_index=True)
    # Drop duplicate timestamps that can appear at month boundaries
    full = full.drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)
    return full


def probe(fmisid: int = 100971) -> None:
    """Fetch one hour and print the raw API structure.

    Purpose: verify that PARAM_MAP keys match what the API actually returns
    before running the expensive full-year fetch.
    """
    print(f"\nProbing FMI API  (fmisid={fmisid}, Helsinki Kaisaniemi)")
    print("Time window: 2024-06-01 12:00–13:00 UTC\n")

    obs = download_stored_query(
        "fmi::observations::weather::hourly::multipointcoverage",
        args=[
            f"fmisid={fmisid}",
            "starttime=2024-06-01T12:00:00Z",
            "endtime=2024-06-01T13:00:00Z",
        ],
    )

    if not obs.data:
        print("  No data returned — check fmisid or network.")
        return

    for ts, locations in obs.data.items():
        print(f"Timestamp: {ts}  (timestamps are UTC-naive)")
        for loc_name, params in locations.items():
            print(f"Station name from API: {repr(loc_name)}")
            print(f"location_metadata:     {obs.location_metadata.get(loc_name)}\n")
            print("All available parameters:")
            for k, v in params.items():
                used = " ← in PARAM_MAP" if k in PARAM_MAP else ""
                print(f"  {repr(k):<40s}  {v}{used}")
        break  # only first timestamp needed for the probe


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch FMI hourly wind data for 2024")
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Fetch one hour for Helsinki Kaisaniemi, print raw API structure, then exit.",
    )
    args = parser.parse_args()

    if args.probe:
        probe()
        return

    # ── Full year fetch ────────────────────────────────────────────────────────
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nFetching {YEAR} hourly observations — {len(STATIONS)} stations")
    print(f"Output: {RAW_DIR}\n")
    print("─" * 70)

    summary_rows = []

    for station in STATIONS:
        name   = station["name"]
        slug   = station["slug"]
        stype  = station["type"]
        fmisid = station["fmisid"]

        print(f"\n{name}  (fmisid={fmisid}, {stype})")

        df = fetch_station(station)

        if df.empty:
            print("  ⚠  No data returned — fmisid may be wrong or station has no 2024 data.")
            summary_rows.append(dict(
                station=name, type=stype, fmisid=fmisid,
                api_name="—", rows=0, date_range="—",
                ws_miss="—", wd_miss="—", pres_miss="—", temp_miss="—",
            ))
            continue

        api_name = df["station_api_name"].iloc[0]

        # Save CSV — drop helper column, keep ws/wd_deg/pres/temp + time
        out_path = RAW_DIR / f"{slug}.csv"
        df.drop(columns=["station_api_name"]).to_csv(out_path, index=False)
        print(f"  Saved → {out_path.name}  ({len(df)} rows)")

        n = len(df)
        summary_rows.append(dict(
            station    = name,
            type       = stype,
            fmisid     = fmisid,
            api_name   = api_name,
            rows       = n,
            date_range = f"{str(df['time'].iloc[0])[:10]} → {str(df['time'].iloc[-1])[:10]}",
            ws_miss    = f"{df['ws'].isna().mean() * 100:.1f}%",
            wd_miss    = f"{df['wd_deg'].isna().mean() * 100:.1f}%",
            pres_miss  = f"{df['pres'].isna().mean() * 100:.1f}%",
            temp_miss  = f"{df['temp'].isna().mean() * 100:.1f}%",
        ))

    # ── Verification table ─────────────────────────────────────────────────────
    print("\n\n" + "=" * 100)
    print("VERIFICATION TABLE")
    print("=" * 100)
    print(
        f"\n{'Station':<28} {'Type':<10} {'FMISID':>7}  "
        f"{'API name (verify!)':<28}  {'Rows':>5}  {'Date range':<23}  "
        f"{'ws':>5} {'wd':>5} {'p':>5} {'T':>5}"
    )
    print("-" * 100)

    for r in summary_rows:
        print(
            f"{r['station']:<28} {r['type']:<10} {r['fmisid']:>7}  "
            f"{r['api_name']:<28}  {r['rows']:>5}  {r['date_range']:<23}  "
            f"{r['ws_miss']:>5} {r['wd_miss']:>5} {r['pres_miss']:>5} {r['temp_miss']:>5}"
        )

    print("\nColumns ws/wd/p/T = missing-value rate for wind speed / wind direction / pressure / temperature")
    print("Expected ~8784 rows per station (2024 is a leap year: 366 × 24 = 8784)")
    print("\nTo verify:")
    print("  1. 'API name' should match the expected station — flag any surprises")
    print("  2. Rows close to 8784 means good data coverage")
    print("  3. Missing rate should be low (< 5%) for stations we rely on\n")


if __name__ == "__main__":
    main()
