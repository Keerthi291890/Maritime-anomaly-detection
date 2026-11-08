"""
Checks H3 resolution 6 against real Stage 1 data (AIS + SAR) before
committing to it in stage2_normalize.py.

Loads the raw JSON chunk/page files from an existing Stage 1 pull
folder, computes H3 cell indices, and reports the distinct cell count
and per-cell row distribution for both AIS presence and SAR presence.
"""

import json
from pathlib import Path

import h3
import pandas as pd

# <-- set this to your actual Stage 1 pull folder
PULL_DIR = Path("data/raw/pull_2026-10-03T08-56-50Z")

H3_RESOLUTION = 6


def load_json_files(paths: list[Path]) -> pd.DataFrame:
    """Load and concatenate one or more raw JSON files into a DataFrame."""
    frames = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
        frames.append(pd.DataFrame(records))
    return pd.concat(frames, ignore_index=True)


def check_h3_distribution(df: pd.DataFrame, name: str, resolution: int = H3_RESOLUTION) -> None:
    """Compute H3 cells for a DataFrame with lat/lon columns and report stats."""
    print(f"\n=== {name.upper()} — H3 resolution {resolution} ===")

    if "lat" not in df.columns or "lon" not in df.columns:
        print(f"  SKIPPED — no lat/lon columns found. Columns present: {df.columns.tolist()}")
        return

    before = len(df)
    df = df.dropna(subset=["lat", "lon"])
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped} rows with missing lat/lon")

    df["h3_cell"] = df.apply(
        lambda r: h3.latlng_to_cell(r["lat"], r["lon"], resolution), axis=1
    )

    n_cells = df["h3_cell"].nunique()
    counts = df["h3_cell"].value_counts()

    print(f"  Total rows: {len(df)}")
    print(f"  Distinct H3 cells: {n_cells}")
    print(f"  Rows per cell — describe():")
    print(counts.describe())
    print(f"  Rows per cell — top 5 busiest cells:")
    print(counts.head())


def main() -> None:
    if not PULL_DIR.exists():
        raise SystemExit(f"Pull directory not found: {PULL_DIR} — check the path")

    manifest = json.loads((PULL_DIR / "manifest.json").read_text())

    # --- AIS presence: 3 chunk files ---
    ais_files = [
        PULL_DIR / c["file"]
        for c in manifest["results"]["ais_presence"]["chunks"]
        if c["status"] == "success"
    ]
    ais_df = load_json_files(ais_files)
    check_h3_distribution(ais_df, "ais_presence")

    # --- SAR presence: 3 chunk files ---
    sar_files = [
        PULL_DIR / c["file"]
        for c in manifest["results"]["sar_presence"]["chunks"]
        if c["status"] == "success"
    ]
    sar_df = load_json_files(sar_files)
    check_h3_distribution(sar_df, "sar_presence")


if __name__ == "__main__":
    main()