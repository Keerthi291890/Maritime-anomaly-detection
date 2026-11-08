import json
from pathlib import Path

# Original raw AIS JSON file
file_path = Path(
    "../data/raw/pull_ecuador_galapagos_2026-10-07T10-57-51Z/"
    "ais_presence_2025-01-01_2025-05-02.json"
)

target_vessel = "29048f8e0-0cf1-e750-1db8-5165932b197a"
target_date = "2025-01-17"

print("Searching raw JSON file...")

with file_path.open("r", encoding="utf-8") as f:
    data = json.load(f)

# Handle either a list or a dictionary containing records
if isinstance(data, list):
    records = data
elif isinstance(data, dict):
    records = data.get("entries", data.get("data", []))
else:
    records = []

matches = []

for record in records:
    if (
        str(record.get("vessel_id")) == target_vessel
        and str(record.get("date", "")).startswith(target_date)
    ):
        matches.append(record)

print(f"\nMatching records found: {len(matches)}")

for i, record in enumerate(matches, 1):
    print(f"\n--- Record {i} ---")
    print(json.dumps(record, indent=4))