from pathlib import Path
import pandas as pd

BASE_DIR = Path("../data/processed/ais_presence")

files = list(BASE_DIR.rglob("*.parquet"))
df = pd.concat(
    [pd.read_parquet(file) for file in files],
    ignore_index=True
)

key = ["vessel_id", "date", "lat", "lon"]

# 1. Count duplicate groups
duplicate_mask = df.duplicated(subset=key, keep=False)
duplicates = df[duplicate_mask].copy()

print("\n========== DUPLICATE GROUP ANALYSIS ==========")

print("Total records:", len(df))
print("Rows in duplicate groups:", len(duplicates))
print(
    "Duplicate groups:",
    duplicates.groupby(key, dropna=False).ngroups
)

# 2. Check whether complete records are identical,
# excluding provenance and partition metadata.
ignore = ["source_file", "pull_id", "year", "month"]

comparison_columns = [
    c for c in df.columns if c not in ignore
]

full_duplicates = df.duplicated(
    subset=comparison_columns,
    keep=False
)

print("\nRows duplicated across all data columns:")
print(int(full_duplicates.sum()))

# 3. Check which source files contain duplicate records
print("\nDuplicate rows by source file:")

print(
    duplicates.groupby("source_file")
    .size()
    .sort_values(ascending=False)
    .to_string()
)

# 4. Check dates with the most duplicate records
print("\nDuplicate rows by date:")

print(
    duplicates.groupby("date")
    .size()
    .sort_values(ascending=False)
    .head(15)
    .to_string()
)

# 5. Show complete examples
print("\nSample duplicate records:")

sample_key = (
    duplicates.groupby(key, dropna=False)
    .size()
    .sort_values(ascending=False)
    .index[0]
)

if not isinstance(sample_key, tuple):
    sample_key = (sample_key,)

mask = pd.Series(True, index=df.index)

for column, value in zip(key, sample_key):
    mask &= df[column].eq(value)

print(
    df.loc[mask].T.to_string()
)