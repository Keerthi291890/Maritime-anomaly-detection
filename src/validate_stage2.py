from pathlib import Path
import pandas as pd

BASE_DIR = Path("../data/processed")

datasets = [
    "ais_presence",
    "sar_presence",
    "gap_events",
    "encounter_events",
    "loitering_events",
    "fishing_events",
]

print("\n========== STAGE 2 VALIDATION ==========\n")

for dataset in datasets:
    dataset_dir = BASE_DIR / dataset
    files = list(dataset_dir.rglob("*.parquet"))

    total_rows = 0
    failed_files = []

    for file in files:
        try:
            df = pd.read_parquet(file)
            total_rows += len(df)
        except Exception as e:
            failed_files.append((file, str(e)))

    print(f"\nDataset: {dataset}")
    print(f"Parquet files: {len(files)}")
    print(f"Readable records: {total_rows}")
    print(f"Failed files: {len(failed_files)}")

    for file, error in failed_files:
        print(f"ERROR: {file}")
        print(error)

print("\n========== VALIDATION COMPLETE ==========")