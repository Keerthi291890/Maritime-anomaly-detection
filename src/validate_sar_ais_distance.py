
import json
import math
from pathlib import Path
from collections import defaultdict

RAW_DIR = Path("../data/raw")

def haversine_km(lat1, lon1, lat2, lon2):
    radius = 6371.0

    lat1, lon1, lat2, lon2 = map(
        math.radians, [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    return 2 * radius * math.asin(math.sqrt(a))


thresholds = [1, 5, 10, 25]

total_sar_with_id = 0
same_day_matches = 0
distance_values = []

for pull_dir in sorted(RAW_DIR.iterdir()):

    if not pull_dir.is_dir():
        continue

    # Map (vessel ID, date) to all AIS coordinates.
    ais_locations = defaultdict(list)

    for file in pull_dir.glob("ais_presence*.json"):
        with open(file, "r", encoding="utf-8") as f:
            records = json.load(f)

        for record in records:
            vessel_id = record.get("vessel_id")
            date = record.get("date")
            lat = record.get("lat")
            lon = record.get("lon")

            if vessel_id and date and lat is not None and lon is not None:
                ais_locations[(vessel_id, date)].append(
                    (float(lat), float(lon))
                )

    for file in pull_dir.glob("sar_presence*.json"):
        with open(file, "r", encoding="utf-8") as f:
            records = json.load(f)

        for record in records:
            vessel_id = record.get("vessel_id")
            date = record.get("date")
            sar_lat = record.get("lat")
            sar_lon = record.get("lon")

            if not vessel_id:
                continue

            total_sar_with_id += 1

            if sar_lat is None or sar_lon is None:
                continue

            candidates = ais_locations.get((vessel_id, date), [])

            if not candidates:
                continue

            same_day_matches += 1

            distances = [
                haversine_km(
                    float(sar_lat),
                    float(sar_lon),
                    ais_lat,
                    ais_lon
                )
                for ais_lat, ais_lon in candidates
            ]

            distance_values.append(min(distances))


print("\n========== SPATIAL VALIDATION ==========")

print("SAR records with ID:", total_sar_with_id)
print("Same-day ID matches:", same_day_matches)
print("Distances calculated:", len(distance_values))

if distance_values:
    distance_values.sort()

    print(
        "Median nearest distance (km):",
        round(distance_values[len(distance_values) // 2], 3)
    )

    for threshold in thresholds:
        count = sum(d <= threshold for d in distance_values)

        print(
            f"Within {threshold} km:",
            count,
            f"({count / len(distance_values) * 100:.2f}%)"
        )

print("\nSPATIAL VALIDATION COMPLETE")

print("\n========== DISTANCE DISTRIBUTION ==========")

if distance_values:
    buckets = [
        ("Exactly 0 km", lambda d: d == 0),
        ("Greater than 0 to 1 km", lambda d: 0 < d <= 1),
        ("Greater than 1 to 5 km", lambda d: 1 < d <= 5),
        ("Greater than 5 to 10 km", lambda d: 5 < d <= 10),
        ("Greater than 10 to 25 km", lambda d: 10 < d <= 25),
        ("Greater than 25 km", lambda d: d > 25),
    ]

    for label, condition in buckets:
        count = sum(condition(d) for d in distance_values)

        print(
            f"{label}: {count:,} "
            f"({count / len(distance_values) * 100:.2f}%)"
        )

    print("\n========== DISTANCE STATISTICS ==========")

    print("Minimum:", round(min(distance_values), 6), "km")
    print("Maximum:", round(max(distance_values), 6), "km")
    print(
        "Mean:",
        round(sum(distance_values) / len(distance_values), 6),
        "km"
    )

print("\nSPATIAL DISTRIBUTION AUDIT COMPLETE")

