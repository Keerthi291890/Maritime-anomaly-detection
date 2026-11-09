
from pathlib import Path
import pyarrow.parquet as pq

BASE = Path("../data/processed_v2")

DATASETS = [
    "ais_presence",
    "sar_presence",
    "fishing_events",
    "loitering_events",
    "gap_events",
    "encounter_events",
]

REGIONS = [
    "argentina",
    "ecuador_galapagos",
    "ghana",
    "senegal",
]

print("=" * 75)
print("STAGE 4: MATCHING READINESS - SCHEMA AUDIT")
print("=" * 75)

for dataset in DATASETS:
    print(f"\n{'=' * 75}")
    print(f"DATASET: {dataset}")
    print("=" * 75)

    for region in REGIONS:
        folder = BASE / dataset / f"region={region}"
        files = sorted(folder.glob("year=*/month=*/*.parquet"))

        print(f"\n  REGION: {region.upper()}")

        if not files:
            print("    ERROR: No Parquet files found")
            continue

        try:
            schema = pq.read_schema(files[0])

            print(f"    Parquet files: {len(files)}")
            print("    Columns and data types:")

            for field in schema:
                print(f"      {field.name}: {field.type}")

        except Exception as exc:
            print(f"    ERROR: {exc}")

print("\n" + "=" * 75)
print("SCHEMA AUDIT COMPLETE - NO FILES MODIFIED")
print("=" * 75)
