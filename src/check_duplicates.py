from pathlib import Path
import pandas as pd

BASE_DIR = Path("../data/processed")

for dataset in ["ais_presence", "sar_presence"]:
    files = list((BASE_DIR / dataset).rglob("*.parquet"))

    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True)

    print(f"\n========== {dataset.upper()} ==========")
    print("Total records:", len(df))
    print("Columns:", list(df.columns))

    key_columns = ["vessel_id", "date", "lat", "lon"]

    missing = [c for c in key_columns if c not in df.columns]

    if missing:
        print("Cannot check candidate duplicates.")
        print("Missing columns:", missing)
        continue

    duplicates = df.duplicated(
        subset=key_columns,
        keep=False
    )

    print("Potential duplicate rows:", int(duplicates.sum()))
    print("Unique candidate keys:", df[key_columns].drop_duplicates().shape[0])

    if duplicates.any():
        print("\nSample potential duplicates:")
        print(
            df.loc[duplicates, key_columns]
            .sort_values(key_columns)
            .head(10)
            .to_string(index=False)
        )