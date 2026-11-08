
import json
from pathlib import Path
from collections import Counter

RAW_DIR = Path("../data/raw")

ais_ids = set()
sar_ids = set()

ais_records = 0
sar_records = 0

regional_counts = Counter()

for pull_dir in RAW_DIR.iterdir():
    if not pull_dir.is_dir():
        continue

    region = pull_dir.name

    for file in pull_dir.glob("*.json"):
        if file.name == "manifest.json":
            continue

        name = file.name

        if name.startswith("ais_presence"):
            dataset = "ais_presence"
        elif name.startswith("sar_presence"):
            dataset = "sar_presence"
        else:
            continue

        with open(file, "r", encoding="utf-8") as f:
            records = json.load(f)

        regional_counts[(region, dataset)] += len(records)

        for record in records:
            vessel_id = record.get("vessel_id")

            if not vessel_id:
                continue

            if dataset == "ais_presence":
                ais_records += 1
                ais_ids.add(vessel_id)

            elif dataset == "sar_presence":
                sar_records += 1
                sar_ids.add(vessel_id)

print("\n========== ID MATCHING ==========")

matched_ids = ais_ids & sar_ids
sar_only_ids = sar_ids - ais_ids

print("Unique AIS vessel IDs:", len(ais_ids))
print("Unique SAR vessel IDs:", len(sar_ids))
print("Matching unique IDs:", len(matched_ids))
print("SAR IDs not found in AIS:", len(sar_only_ids))

if sar_ids:
    print(
        "Percentage of SAR IDs found in AIS:",
        round(len(matched_ids) / len(sar_ids) * 100, 2),
        "%"
    )

print("\n========== REGIONAL COUNTS ==========")

for (region, dataset), count in sorted(regional_counts.items()):
    print(f"{region} | {dataset}: {count:,}")

print("\n========== AUDIT COMPLETE ==========")
