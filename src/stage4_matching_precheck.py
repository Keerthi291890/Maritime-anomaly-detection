
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

BASE = Path("../data/processed_v2")

REGIONS = [
    "argentina",
    "ecuador_galapagos",
    "ghana",
    "senegal",
]

print("=" * 75)
print("STAGE 4: MATCHING PRECHECK")
print("=" * 75)

for dataset in ["ais_presence", "sar_presence"]:
    print(f"\n{'=' * 75}")
    print(f"DATASET: {dataset}")
    print("=" * 75)

    for region in REGIONS:
        folder = BASE / dataset / f"region={region}"
        files = sorted(folder.glob("year=*/month=*/*.parquet"))

        total_rows = 0
        blank_ids = 0
        missing_ids = 0
        invalid_dates = 0
        missing_coordinates = 0
        date_examples = []
        source_files = set()

        for file in files:
            table = pq.read_table(
                file,
                columns=[
                    "date", "vessel_id", "lat", "lon",
                    "source_file"
                ]
            )
            df = table.to_pandas()

            total_rows += len(df)

            ids = df["vessel_id"].astype("string")
            missing_ids += int(ids.isna().sum())
            blank_ids += int(
                ids.fillna("").str.strip().eq("").sum()
            )

            parsed_dates = pd.to_datetime(
                df["date"], errors="coerce", utc=True
            )
            invalid_dates += int(parsed_dates.isna().sum())

            missing_coordinates += int(
                (df["lat"].isna() | df["lon"].isna()).sum()
            )

            if len(date_examples) < 5:
                date_examples.extend(
                    df["date"].dropna().astype(str).head(
                        5 - len(date_examples)
                    ).tolist()
                )

            source_files.update(
                df["source_file"].dropna().astype(str).unique()
            )

        print(f"\nREGION: {region.upper()}")
        print(f"Parquet files: {len(files)}")
        print(f"Total rows: {total_rows:,}")
        print(f"Missing vessel IDs (null): {missing_ids:,}")
        print(f"Blank vessel IDs: {blank_ids:,}")
        print(f"Invalid dates: {invalid_dates:,}")
        print(f"Missing coordinates: {missing_coordinates:,}")
        print(f"Distinct source files: {len(source_files)}")
        print(f"Date examples: {date_examples}")

print("\n" + "=" * 75)
print("PRECHECK COMPLETE - NO FILES MODIFIED")
print("=" * 75)
