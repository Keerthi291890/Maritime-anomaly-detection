
from pathlib import Path
from datetime import date
import argparse
import csv
import struct

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REGIONS = [
    "argentina",
    "ecuador_galapagos",
    "ghana",
    "senegal",
]

DATASETS = ["ais_presence", "sar_presence"]

# Binary record layout, exactly 36 bytes:
# uint64 record_id
# int32  day_number (Python date ordinal)
# int64  vessel_code (0 means missing)
# float64 latitude
# float64 longitude

RECORD_DTYPE = np.dtype([
    ("record_id", "<u8"),
    ("day", "<i4"),
    ("vessel_code", "<i8"),
    ("lat", "<f8"),
    ("lon", "<f8"),
])

METADATA_SCHEMA = pa.schema([
    ("record_id", pa.uint64()),
    ("date", pa.string()),
    ("vessel_id", pa.string()),
    ("lat", pa.float64()),
    ("lon", pa.float64()),
    ("region", pa.string()),
    ("h3_cell", pa.string()),
    ("pull_id", pa.string()),
    ("source_file", pa.string()),
])


def clean_string(value):
    if value is None:
        return None

    value = str(value).strip()

    if not value or value.lower() in {
        "nan", "none", "null", "<na>"
    }:
        return None

    return value


def prepare_dataset(dataset, region, input_root, output_root,
                    vessel_codes, next_record_id):

    source_dir = input_root / dataset / f"region={region}"

    files = sorted(source_dir.rglob("*.parquet"))

    if not files:
        raise FileNotFoundError(
            f"No Parquet files found in {source_dir}"
        )

    output_dir = output_root / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    binary_path = output_dir / f"{region}.bin"
    metadata_path = output_dir / f"{region}_metadata.parquet"

    record_count = 0
    metadata_writer = None

    columns = [
        "date",
        "vessel_id",
        "lat",
        "lon",
        "region",
        "h3_cell",
        "pull_id",
        "source_file",
    ]

    try:
        with binary_path.open("wb") as binary_file:

            for parquet_file in files:
                parquet = pq.ParquetFile(parquet_file)

                missing = set(columns) - set(parquet.schema_arrow.names)

                if missing:
                    raise ValueError(
                        f"{parquet_file}: missing columns {sorted(missing)}"
                    )

                for batch in parquet.iter_batches(
                    batch_size=50000,
                    columns=columns
                ):
                    values = {
                        name: batch.column(name).to_pylist()
                        for name in columns
                    }

                    size = batch.num_rows
                    records = np.zeros(size, dtype=RECORD_DTYPE)

                    meta = {
                        "record_id": [],
                        "date": [],
                        "vessel_id": [],
                        "lat": [],
                        "lon": [],
                        "region": [],
                        "h3_cell": [],
                        "pull_id": [],
                        "source_file": [],
                    }

                    for i in range(size):
                        date_value = clean_string(values["date"][i])

                        if date_value is None:
                            raise ValueError("Missing observation date")

                        date_text = date_value[:10]
                        day_number = date.fromisoformat(
                            date_text
                        ).toordinal()

                        vessel_id = clean_string(
                            values["vessel_id"][i]
                        )

                        if vessel_id is None:
                            vessel_code = 0
                        else:
                            if vessel_id not in vessel_codes:
                                vessel_codes[vessel_id] = (
                                    len(vessel_codes) + 1
                                )

                            vessel_code = vessel_codes[vessel_id]

                        lat = float(values["lat"][i])
                        lon = float(values["lon"][i])

                        if not (
                            np.isfinite(lat)
                            and np.isfinite(lon)
                            and -90 <= lat <= 90
                            and -180 <= lon <= 180
                        ):
                            raise ValueError(
                                f"Invalid coordinates: {lat}, {lon}"
                            )

                        record_id = next_record_id + i

                        records[i] = (
                            record_id,
                            day_number,
                            vessel_code,
                            lat,
                            lon,
                        )

                        meta["record_id"].append(record_id)
                        meta["date"].append(date_text)
                        meta["vessel_id"].append(vessel_id)
                        meta["lat"].append(lat)
                        meta["lon"].append(lon)
                        meta["region"].append(
                            clean_string(values["region"][i]) or region
                        )
                        meta["h3_cell"].append(
                            clean_string(values["h3_cell"][i])
                        )
                        meta["pull_id"].append(
                            clean_string(values["pull_id"][i])
                        )
                        meta["source_file"].append(
                            clean_string(values["source_file"][i])
                        )

                    records.tofile(binary_file)

                    metadata_table = pa.Table.from_pydict(
                        meta,
                        schema=METADATA_SCHEMA,
                    )

                    if metadata_writer is None:
                        metadata_writer = pq.ParquetWriter(
                            metadata_path,
                            METADATA_SCHEMA,
                            compression="snappy",
                        )

                    metadata_writer.write_table(metadata_table)

                    record_count += size

                    next_record_id += size

    finally:
        if metadata_writer is not None:
            metadata_writer.close()

    expected_bytes = record_count * RECORD_DTYPE.itemsize

    actual_bytes = binary_path.stat().st_size

    if actual_bytes != expected_bytes:
        raise RuntimeError(
            f"Binary size mismatch: expected {expected_bytes}, "
            f"found {actual_bytes}"
        )

    print(
        f"  {region}: {record_count:,} records | "
        f"{actual_bytes / (1024 * 1024):.2f} MB"
    )

    return next_record_id, record_count


def main():

    parser = argparse.ArgumentParser(
        description="Prepare AIS and SAR binary inputs for HPC matching"
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed_v2",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "stage4_input",
    )

    args = parser.parse_args()

    input_root = args.input_dir.resolve()
    output_root = args.output_dir.resolve()

    if not input_root.exists():
        raise FileNotFoundError(input_root)

    output_root.mkdir(parents=True, exist_ok=True)

    vessel_codes = {}

    totals = {
        "ais_presence": 0,
        "sar_presence": 0,
    }

    print("=" * 65)
    print("STAGE 4: HPC INPUT PREPARATION")
    print("=" * 65)

    for dataset in DATASETS:

        print(f"\nDATASET: {dataset}")

        next_record_id = 0

        for region in REGIONS:

            next_record_id, count = prepare_dataset(
                dataset,
                region,
                input_root,
                output_root,
                vessel_codes,
                next_record_id,
            )

            totals[dataset] += count

    dictionary_path = output_root / "vessel_dictionary.csv"

    with dictionary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)
        writer.writerow(["vessel_code", "vessel_id"])

        for vessel_id, code in sorted(
            vessel_codes.items(),
            key=lambda item: item[1],
        ):
            writer.writerow([code, vessel_id])

    print("\n" + "=" * 65)
    print("PREPARATION SUMMARY")
    print("=" * 65)

    print(f"AIS records: {totals['ais_presence']:,}")
    print(f"SAR records: {totals['sar_presence']:,}")
    print(f"Unique vessel IDs: {len(vessel_codes):,}")

    print(f"\nOutput directory: {output_root}")
    print("Binary record size:", RECORD_DTYPE.itemsize, "bytes")
    print("Vessel dictionary:", dictionary_path.name)

    print("\nRESULT: INPUT PREPARATION COMPLETED")


if __name__ == "__main__":
    main()
