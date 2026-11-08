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

print("\n========== DUPLICATE SOURCE CHECK ==========")

source_counts = (
    duplicates.groupby(key, dropna=False)["source_file"]
    .nunique()
)

print("Groups spanning multiple source files:",
      int((source_counts > 1).sum()))

print("Groups contained in one source file:",
      int((source_counts == 1).sum()))


print("\n========== DIFFERING FIELD ANALYSIS ==========")

different_groups = []

for group_key, group in duplicates.groupby(key, dropna=False):

    varying_columns = [
        col for col in df.columns
        if group[col].nunique(dropna=False) > 1
    ]

    if varying_columns:
        different_groups.append(
            (group_key, varying_columns, group)
        )

print("Groups with differing fields:", len(different_groups))

for i, (group_key, varying_columns, group) in enumerate(
    different_groups[:10], start=1
):
    print(f"\n--- Example {i} ---")
    print("Matching key:", group_key)
    print("Differing columns:", varying_columns)

    print(
        group[["source_file"] + varying_columns]
        .to_string(index=False)
    )

print("\n========== CHECK COMPLETE ==========")