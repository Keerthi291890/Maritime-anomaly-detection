
import json
from pathlib import Path
from collections import defaultdict

RAW_DIR = Path("../data/raw")

for pull_dir in sorted(RAW_DIR.iterdir()):

    if not pull_dir.is_dir():
        continue

    ais_ids = set()
    ais_id_dates = set()

    sar_records = 0
    sar_with_id = 0
    sar_without_id = 0

    same_region_id = 0
    same_day_id = 0

    for file in pull_dir.glob("ais_presence*.json"):
        with open(file, "r", encoding="utf-8") as f:
            records = json.load(f)

        for record in records:
            vessel_id = record.get("vessel_id")
            date = record.get("date")

            if vessel_id:
                ais_ids.add(vessel_id)

                if date:
                    ais_id_dates.add((vessel_id, date))

    for file in pull_dir.glob("sar_presence*.json"):
        with open(file, "r", encoding="utf-8") as f:
            records = json.load(f)

        for record in records:
            sar_records += 1

            vessel_id = record.get("vessel_id")
            date = record.get("date")

            if not vessel_id:
                sar_without_id += 1
                continue

            sar_with_id += 1

            if vessel_id in ais_ids:
                same_region_id += 1

            if (vessel_id, date) in ais_id_dates:
                same_day_id += 1

    print("\n===================================")
    print("PULL:", pull_dir.name)
    print("===================================")

    print("Total SAR records:", sar_records)
    print("SAR records with ID:", sar_with_id)
    print("SAR records without ID:", sar_without_id)

    print("SAR records whose ID exists in regional AIS:",
          same_region_id)

    print("SAR records whose ID exists in AIS on same date:",
          same_day_id)

    if sar_with_id:
        print(
            "Same-region ID match rate:",
            round(same_region_id / sar_with_id * 100, 2),
            "%"
        )

        print(
            "Same-day ID match rate:",
            round(same_day_id / sar_with_id * 100, 2),
            "%"
        )

print("\nVALIDATION COMPLETE")
