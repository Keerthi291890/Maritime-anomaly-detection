
import json
import re
from pathlib import Path
from collections import defaultdict

RAW_DIR = Path("../data/raw")


def get_dataset_name(path):
    name = path.stem
    name = re.sub(r"_\d{4}-\d{2}-\d{2}.*$", "", name)
    name = re.sub(r"_page\d+$", "", name)
    return name


def inspect_records(path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        return None

    return data


def main():
    files = sorted(RAW_DIR.rglob("*.json"))

    # Group all chunks and pages by dataset name.
    groups = defaultdict(list)

    for path in files:
        if path.name == "manifest.json":
            continue
        groups[get_dataset_name(path)].append(path)

    print("\n========== DATASET SUMMARY ==========")

    for dataset, paths in sorted(groups.items()):
        total_records = 0
        date_values = []
        vessel_ids = set()
        nonempty_vessel_ids = 0
        coordinate_count = 0
        sample = None

        print(f"\nDATASET: {dataset}")
        print("Files:", len(paths))

        for path in paths:
            try:
                records = inspect_records(path)

                if records is None:
                    print("  Non-list JSON:", path.name)
                    continue

                total_records += len(records)

                if sample is None and records:
                    sample = records[0]

                for record in records:
                    date_value = record.get("date") or record.get("start")
                    if date_value:
                        date_values.append(date_value)

                    lat = record.get("lat")
                    lon = record.get("lon")

                    if isinstance(record.get("position"), dict):
                        lat = record["position"].get("lat")
                        lon = record["position"].get("lon")

                    if lat is not None and lon is not None:
                        coordinate_count += 1

                    vessel = record.get("vessel")
                    vessel_id = record.get("vessel_id")

                    if isinstance(vessel, dict):
                        vessel_id = vessel.get("id")

                    if vessel_id:
                        nonempty_vessel_ids += 1
                        vessel_ids.add(vessel_id)

                print(f"  {path.name}: {len(records):,} records")

            except Exception as e:
                print(f"  ERROR in {path.name}: {e}")

        print("TOTAL RECORDS:", f"{total_records:,}")
        print("Records with coordinates:", f"{coordinate_count:,}")
        print("Records with nonempty vessel ID:", f"{nonempty_vessel_ids:,}")
        print("Unique vessel IDs:", f"{len(vessel_ids):,}")

        if date_values:
            print("Earliest date/start:", min(date_values))
            print("Latest date/start:", max(date_values))

        if sample:
            print("Sample record keys:", list(sample.keys()))

            if dataset in ("ais_presence", "sar_presence"):
                print("Sample date:", sample.get("date"))
                print("Sample entry:", sample.get("entry_timestamp"))
                print("Sample exit:", sample.get("exit_timestamp"))
                print("Sample lat/lon:", sample.get("lat"), sample.get("lon"))

    print("\n========== AUDIT COMPLETE ==========")


if __name__ == "__main__":
    main()
