
"""
Stage 2 — Normalize raw GFW JSON into partitioned Parquet.

AIS/SAR:
    Coordinates: top-level lat/lon
    Date: date

Events:
    Coordinates: position.lat / position.lon
    Date: start

Output:
    data/processed/<dataset>/
        region=<region>/
            year=YYYY/
                month=MM/
                    part-<pull>-<source>.parquet
"""

import argparse
import json
import re
from pathlib import Path

import h3
import pandas as pd


H3_RESOLUTION = 6

RAW_DIR = Path("../data/raw")
OUTPUT_DIR = Path("../data/processed_v2")

FOURWINGS_DATASETS = [
    "ais_presence",
    "sar_presence",
]

EVENT_DATASETS = [
    "gap_events",
    "encounter_events",
    "loitering_events",
    "fishing_events",
]


def safe_name(value):
    """Convert a string into a filesystem-safe identifier."""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(value))


def find_latest_pull_dir(region_name):
    """Find latest pull matching the region name."""

    matches = sorted(
        RAW_DIR.glob(f"pull_{region_name}_*")
    )

    if not matches:
        raise SystemExit(
            f"No pull found for region: {region_name}. "
            "Use --pull-dir to specify the raw directory."
        )

    return matches[-1]


def load_manifest(pull_dir):
    manifest_path = pull_dir / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}"
        )

    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_position(df):
    """Extract event coordinates from nested position dictionaries."""

    if "position" not in df.columns:
        df["lat"] = None
        df["lon"] = None
        return df

    def get_coordinate(position, keys):
        if not isinstance(position, dict):
            return None

        for key in keys:
            if position.get(key) is not None:
                return position[key]

        return None

    df["lat"] = df["position"].apply(
        lambda p: get_coordinate(p, ["lat", "latitude"])
    )

    df["lon"] = df["position"].apply(
        lambda p: get_coordinate(
            p, ["lon", "lng", "longitude"]
        )
    )

    return df


def add_encounter_fields(df):
    """Preserve the second vessel in encounter records."""

    if "encounter" not in df.columns:
        return df

    def nested_value(record, key):
        if not isinstance(record, dict):
            return None

        vessel = record.get("vessel")

        if isinstance(vessel, dict):
            return vessel.get(key)

        return None

    df["other_vessel_id"] = df["encounter"].apply(
        lambda x: nested_value(x, "id")
    )

    df["other_vessel_name"] = df["encounter"].apply(
        lambda x: nested_value(x, "name")
    )

    df["other_vessel_ssvid"] = df["encounter"].apply(
        lambda x: nested_value(x, "ssvid")
    )

    return df
def standardize_nested_columns(df):
    """
    Convert nested dictionaries/lists into JSON strings.

    This prevents PyArrow from inferring different nested
    schemas for different monthly Parquet partitions.
    """

    df = df.copy()

    for column in df.columns:

        if not pd.api.types.is_object_dtype(df[column]):
            continue

        non_null = df[column].dropna()

        # Detect columns containing nested structures.
        has_nested = any(
            isinstance(value, (dict, list))
            for value in non_null
        )

        if has_nested:

            def to_json(value):
                if value is None or value is pd.NA:
                    return None

                if isinstance(value, float) and pd.isna(value):
                    return None

                if isinstance(value, (dict, list)):
                    return json.dumps(
                        value,
                        ensure_ascii=False,
                        default=str,
                    )

                return value

            df[column] = df[column].apply(to_json)

            # Force a consistent string representation.
            df[column] = df[column].astype("string")

        elif non_null.empty:
            # An entirely null object column must also
            # have a consistent type.
            df[column] = df[column].astype("string")

        else:
            df[column] = df[column].infer_objects()

    return df


def validate_and_prepare(
    df,
    dataset_name,
    region_name,
    pull_id,
    source_file,
):
    """Clean coordinates, dates and add metadata."""

    if df.empty:
        return df

    df = df.copy()

    # Event coordinate extraction
    if dataset_name in EVENT_DATASETS:
        df = extract_position(df)
        # Encounter enrichment
        df = add_encounter_fields(df)

    # Ensure coordinate columns exist
    for column in ["lat", "lon"]:
        if column not in df.columns:
            df[column] = None

        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    # Select the correct date field
    date_col = (
        "date"
        if dataset_name in FOURWINGS_DATASETS
        else "start"
    )

    if date_col not in df.columns:
        print(f"    Missing date field: {date_col}")
        return pd.DataFrame()

    df["_dt"] = pd.to_datetime(
        df[date_col],
        errors="coerce",
        utc=True,
    )

    before = len(df)

    df = df.dropna(
        subset=["lat", "lon", "_dt"]
    ).copy()

    # Geographic coordinate validation
    df = df[
        df["lat"].between(-90, 90)
        & df["lon"].between(-180, 180)
    ].copy()

    dropped = before - len(df)

    if dropped:
        print(
            f"    Dropped {dropped} invalid rows"
        )

    if df.empty:
        return df

    # H3 spatial indexing
    df["h3_cell"] = [
        h3.latlng_to_cell(
            lat,
            lon,
            H3_RESOLUTION,
        )
        for lat, lon in zip(df["lat"], df["lon"])
    ]

    # Encounter enrichment
    if dataset_name == "encounter_events":
        df = add_encounter_fields(df)

    # Provenance metadata
    df["region"] = region_name
    df["pull_id"] = pull_id
    df["source_file"] = source_file

    df["year"] = df["_dt"].dt.year.astype("int16")
    df["month"] = df["_dt"].dt.month.astype("int8")

    return df.drop(columns=["_dt"])


def write_partitioned(
    df,
    dataset_name,
    region_name,
    pull_id,
    source_file,
):
    """Write separate Parquet files for each month."""

    if df.empty:
        return 0

    pull_tag = safe_name(pull_id)
    file_tag = safe_name(
        Path(source_file).stem
    )

    written = 0

    for (year, month), group in df.groupby(
        ["year", "month"],
        sort=True,
    ):

        out_dir = (
            OUTPUT_DIR
            / dataset_name
            / f"region={region_name}"
            / f"year={year}"
            / f"month={month:02d}"
        )

        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        out_path = (
            out_dir
            / f"part-{pull_tag}-{file_tag}.parquet"
        )

        group = group.sort_values(
            "h3_cell"
        )

        group.to_parquet(
            out_path,
            index=False,
        )

        written += len(group)

        print(
            f"    Wrote {out_path}: "
            f"{len(group):,} rows"
        )

    return written


def process_source_file(
    path,
    dataset_name,
    region_name,
    pull_id,
):
    """Read, normalize and write one source file."""

    if not path.exists():
        raise FileNotFoundError(
            f"Source file not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise ValueError(
            f"Expected a JSON list in {path}"
        )

    df = pd.DataFrame(records)

    print(
        f"\n    Source: {path.name}"
        f"\n    Raw rows: {len(df):,}"
    )

    df = validate_and_prepare(
        df,
        dataset_name,
        region_name,
        pull_id,
        path.name,
    )

    written = write_partitioned(
        df,
        dataset_name,
        region_name,
        pull_id,
        path.name,
    )

    print(
        f"    Normalized rows written: {written:,}"
    )

    return len(records), written


def process_fourwings_dataset(
    dataset_name,
    region_name,
    manifest,
    pull_dir,
    pull_id,
):
    print(f"\n========== {dataset_name.upper()} ==========")

    result = manifest.get("results", {}).get(
        dataset_name, {}
    )

    if result.get("status") != "success":
        print("    Dataset status is not success — skipping")
        return

    chunks = result.get("chunks", [])

    raw_total = 0
    written_total = 0

    for chunk in chunks:

        if chunk.get("status") != "success":
            print("    Skipping unsuccessful chunk")
            continue

        path = pull_dir / chunk["file"]

        raw, written = process_source_file(
            path,
            dataset_name,
            region_name,
            pull_id,
        )

        raw_total += raw
        written_total += written

    print("\n    DATASET SUMMARY")
    print(f"    Raw records: {raw_total:,}")
    print(f"    Written records: {written_total:,}")
    print(f"    Dropped records: {raw_total - written_total:,}")


def process_event_dataset(
    dataset_name,
    region_name,
    manifest,
    pull_dir,
    pull_id,
):
    print(f"\n========== {dataset_name.upper()} ==========")

    result = manifest.get("results", {}).get(
        dataset_name, {}
    )

    if result.get("status") != "success":
        print("    Dataset status is not success — skipping")
        return

    pages = result.get("pages", [])

    raw_total = 0
    written_total = 0

    for page in pages:

        path = pull_dir / page["file"]

        raw, written = process_source_file(
            path,
            dataset_name,
            region_name,
            pull_id,
        )

        raw_total += raw
        written_total += written

    print("\n    DATASET SUMMARY")
    print(f"    Raw records: {raw_total:,}")
    print(f"    Written records: {written_total:,}")
    print(f"    Dropped records: {raw_total - written_total:,}")


def main():

    parser = argparse.ArgumentParser(
        description="Stage 2 GFW normalization"
    )

    parser.add_argument(
        "--region-name",
        required=False,
    )

    parser.add_argument(
        "--pull-dir",
        default=None,
    )

    args = parser.parse_args()

    if args.pull_dir:
        pull_dir = Path(args.pull_dir)
    elif args.region_name:
        pull_dir = find_latest_pull_dir(
            args.region_name
        )
    else:
        parser.error(
            "Provide --region-name or --pull-dir"
        )

    manifest = load_manifest(pull_dir)

    region_name = manifest.get("region_name")

    if not region_name:
        if not args.region_name:
            raise ValueError(
                "Manifest has no region_name. "
                "Provide --region-name explicitly."
            )
        region_name = args.region_name

    if args.region_name and args.region_name != region_name:
        raise ValueError(
            f"Region mismatch: argument={args.region_name}, "
            f"manifest={region_name}"
        )

    pull_id = pull_dir.name

    print("===================================")
    print("STAGE 2 NORMALIZATION")
    print("===================================")
    print("Region:", region_name)
    print("Pull ID:", pull_id)
    print("Pull directory:", pull_dir)
    print("H3 resolution:", H3_RESOLUTION)

    for dataset in FOURWINGS_DATASETS:
        process_fourwings_dataset(
            dataset,
            region_name,
            manifest,
            pull_dir,
            pull_id,
        )

    for dataset in EVENT_DATASETS:
        process_event_dataset(
            dataset,
            region_name,
            manifest,
            pull_dir,
            pull_id,
        )

    print("\n===================================")
    print("STAGE 2 COMPLETE")
    print("===================================")


if __name__ == "__main__":
    main()
