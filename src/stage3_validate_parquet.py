
from pathlib import Path
import argparse
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc


# Project root is the parent of src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed_v2"

# Expected counts from the four raw manifests / normalization logs.
EXPECTED = {
    "argentina": {
        "ais_presence": 1101669,
        "sar_presence": 4089,
        "gap_events": 363,
        "encounter_events": 90,
        "loitering_events": 24574,
        "fishing_events": 65780,
    },
    "ecuador_galapagos": {
        "ais_presence": 465977,
        "sar_presence": 804,
        "gap_events": 1337,
        "encounter_events": 172,
        "loitering_events": 6544,
        "fishing_events": 4487,
    },
    "ghana": {
        "ais_presence": 323192,
        "sar_presence": 2336,
        "gap_events": 141,
        "encounter_events": 12,
        "loitering_events": 9840,
        "fishing_events": 8251,
    },
    "senegal": {
        "ais_presence": 451896,
        "sar_presence": 4697,
        "gap_events": 63,
        "encounter_events": 30,
        "loitering_events": 6607,
        "fishing_events": 15132,
    },
}

IMPORTANT_COLUMNS = {
    "ais_presence": ["vessel_id", "lat", "lon"],
    "sar_presence": ["vessel_id", "lat", "lon"],
}

def count_true(mask):
    result = pc.sum(pc.cast(mask, pa.int64())).as_py()
    return result or 0


def validate_dataset(region, dataset, expected_rows):
    folder = (
        PROCESSED_DIR
        / dataset
        / f"region={region}"
    )

    files = sorted(folder.rglob("*.parquet"))

    result = {
        "region": region,
        "dataset": dataset,
        "files": len(files),
        "rows": 0,
        "expected": expected_rows,
        "read_errors": 0,
        "missing_columns": [],
        "null_counts": {},
        "invalid_lat": 0,
        "invalid_lon": 0,
        "lat_min": None,
        "lat_max": None,
        "lon_min": None,
        "lon_max": None,
        "schema_consistent": True,
    }

    if not files:
        result["read_errors"] = 1
        print(f"  ERROR: No Parquet files found in {folder}")
        return result

    reference_schema = None
    required = IMPORTANT_COLUMNS.get(dataset, [])

    for file in files:
        try:
            parquet_file = pq.ParquetFile(file)
            schema = parquet_file.schema_arrow

            if reference_schema is None:
                reference_schema = schema
            elif not schema.equals(
                reference_schema, check_metadata=False
            ):
                result["schema_consistent"] = False

            available = set(schema.names)

            for column in required:
                if column not in available:
                    if column not in result["missing_columns"]:
                        result["missing_columns"].append(column)

            # Read every row group in batches to verify actual readability.
            for batch in parquet_file.iter_batches(batch_size=65536):
                result["rows"] += batch.num_rows

                for column in required:
                    if column not in batch.schema.names:
                        continue

                    array = batch.column(
                        batch.schema.get_field_index(column)
                    )

                    result["null_counts"][column] = (
                        result["null_counts"].get(column, 0)
                        + array.null_count
                    )

                if "lat" in batch.schema.names:
                    lat = batch.column(
                        batch.schema.get_field_index("lat")
                    )
                    result["invalid_lat"] += count_true(
                        pc.or_(
                            pc.less(lat, -90),
                            pc.greater(lat, 90)
                        )
                    )
                    if lat.null_count < len(lat):
                        low = pc.min(lat).as_py()
                        high = pc.max(lat).as_py()
                        if low is not None:
                            result["lat_min"] = (
                                low if result["lat_min"] is None
                                else min(result["lat_min"], low)
                            )
                        if high is not None:
                            result["lat_max"] = (
                                high if result["lat_max"] is None
                                else max(result["lat_max"], high)
                            )

                if "lon" in batch.schema.names:
                    lon = batch.column(
                        batch.schema.get_field_index("lon")
                    )
                    result["invalid_lon"] += count_true(
                        pc.or_(
                            pc.less(lon, -180),
                            pc.greater(lon, 180)
                        )
                    )
                    if lon.null_count < len(lon):
                        low = pc.min(lon).as_py()
                        high = pc.max(lon).as_py()
                        if low is not None:
                            result["lon_min"] = (
                                low if result["lon_min"] is None
                                else min(result["lon_min"], low)
                            )
                        if high is not None:
                            result["lon_max"] = (
                                high if result["lon_max"] is None
                                else max(result["lon_max"], high)
                            )

        except Exception as exc:
            result["read_errors"] += 1
            print(f"  READ ERROR: {file}")
            print(f"    {exc}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--region",
        default="all",
        choices=["all"] + list(EXPECTED.keys())
    )
    args = parser.parse_args()

    regions = (
        list(EXPECTED.keys())
        if args.region == "all"
        else [args.region]
    )

    all_results = []

    print("=" * 65)
    print("STAGE 3: INDEPENDENT PARQUET VALIDATION")
    print("=" * 65)
    print(f"Processed directory: {PROCESSED_DIR}")

    for region in regions:
        print(f"\n{'=' * 20} {region.upper()} {'=' * 20}")

        for dataset, expected in EXPECTED[region].items():
            print(f"\nDataset: {dataset}")

            result = validate_dataset(region, dataset, expected)
            all_results.append(result)

            print(f"  Parquet files:       {result['files']}")
            print(f"  Expected rows:       {result['expected']:,}")
            print(f"  Readable rows:       {result['rows']:,}")
            print(f"  Read errors:         {result['read_errors']}")
            print(
                f"  Count matches:       "
                f"{result['rows'] == result['expected']}"
            )
            print(
                f"  Schema consistent:   "
                f"{result['schema_consistent']}"
            )

            if result["missing_columns"]:
                print(
                    "  Missing columns:     "
                    + ", ".join(result["missing_columns"])
                )

            if result["null_counts"]:
                print("  Null counts:")
                for column, count in result["null_counts"].items():
                    print(f"    {column}: {count:,}")

            if dataset in ("ais_presence", "sar_presence"):
                print(
                    f"  Invalid latitude:    "
                    f"{result['invalid_lat']}"
                )
                print(
                    f"  Invalid longitude:   "
                    f"{result['invalid_lon']}"
                )
                print(
                    f"  Latitude range:      "
                    f"{result['lat_min']} to {result['lat_max']}"
                )
                print(
                    f"  Longitude range:     "
                    f"{result['lon_min']} to {result['lon_max']}"
                )

    print("\n" + "=" * 65)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 65)

    total_expected = sum(r["expected"] for r in all_results)
    total_read = sum(r["rows"] for r in all_results)
    total_files = sum(r["files"] for r in all_results)
    total_errors = sum(r["read_errors"] for r in all_results)

    failed = []

    for r in all_results:
        problems = []
        if r["rows"] != r["expected"]:
            problems.append("row count mismatch")
        if r["read_errors"] > 0 or r["files"] == 0:
            problems.append("read error or missing files")
        if r["missing_columns"]:
            problems.append("missing required columns")
        if not r["schema_consistent"]:
            problems.append("schema inconsistency")
        if r["invalid_lat"] or r["invalid_lon"]:
            problems.append("invalid coordinates")

        if problems:
            failed.append(
                f"{r['region']}/{r['dataset']}: "
                + ", ".join(problems)
            )

    print(f"Regions checked:      {len(regions)}")
    print(f"Datasets checked:     {len(all_results)}")
    print(f"Parquet files found:  {total_files}")
    print(f"Expected rows:        {total_expected:,}")
    print(f"Readable rows:        {total_read:,}")
    print(f"Read errors:          {total_errors}")

    if failed:
        print("\nVALIDATION ISSUES:")
        for problem in failed:
            print(f"  - {problem}")
        print("\nRESULT: REVIEW REQUIRED")
    else:
        print("\nRESULT: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
