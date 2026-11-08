from pathlib import Path
import pandas as pd

BASE_DIR = Path("../data/processed/ais_presence")

files = list(BASE_DIR.rglob("*.parquet"))

df = pd.concat(
    [pd.read_parquet(f) for f in files],
    ignore_index=True
)

key = ["vessel_id", "date", "lat", "lon"]

duplicates = df[
    df.duplicated(subset=key, keep=False)
].copy()

print("\n========== HOURS DUPLICATE ANALYSIS ==========")

# Group by vessel, date and location
for group_key, group in duplicates.groupby(key, dropna=False):

    if group["hours"].nunique(dropna=False) > 1:
        print(
            group[["vessel_id", "date", "lat", "lon",
                   "hours", "source_file"]].to_string(index=False)
        )
        print("-" * 60)

# Summarize differing hours patterns
conflicts = (
    duplicates.groupby(key, dropna=False)["hours"]
    .agg(
        record_count="size",
        distinct_hours="nunique",
        hours_values=lambda x: sorted(
            str(v) for v in x.dropna().unique()
        )
    )
    .reset_index()
)

conflicts = conflicts[conflicts["distinct_hours"] > 1]

print("\n========== SUMMARY ==========")
print("Groups with differing hours:", len(conflicts))

print("\nHours value combinations:")
print(
    conflicts["hours_values"]
    .astype(str)
    .value_counts()
    .to_string()
)

print("\nRecord count distribution:")
print(
    conflicts["record_count"]
    .value_counts()
    .sort_index()
    .to_string()
)