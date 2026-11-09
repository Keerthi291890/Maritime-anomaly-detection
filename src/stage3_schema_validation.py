from pathlib import Path
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed_v2"

REGIONS = [
    "argentina",
    "ecuador_galapagos",
    "ghana",
    "senegal",
]

DATASETS = [
    "ais_presence",
    "sar_presence",
    "gap_events",
    "encounter_events",
    "loitering_events",
    "fishing_events",
]


def inspect_schema(region, dataset):
    folder = (
        PROCESSED_DIR
        / dataset
        / f"region={region}"
    )

    files = sorted(folder.rglob("*.parquet"))

    if not files:
        return

    schemas = []

    for file in files:
        try:
            schema = pq.read_schema(file)
            schemas.append((file, schema.remove_metadata()))
        except Exception as exc:
            print(f"Could not read schema: {file}")
            print(f"  Error: {exc}")

    if not schemas:
        return

    reference_file, reference_schema = schemas[0]

    inconsistent = []

    for file, schema in schemas[1:]:
        if not schema.equals(reference_schema):
            inconsistent.append((file, schema))

    if not inconsistent:
        return

    print("\n" + "=" * 75)
    print(f"REGION: {region}")
    print(f"DATASET: {dataset}")
    print(f"REFERENCE FILE: {reference_file.name}")
    print(f"INCONSISTENT FILES: {len(inconsistent)}")
    print("=" * 75)

    reference_fields = {
        field.name: str(field.type)
        for field in reference_schema
    }

    for file, schema in inconsistent:
        current_fields = {
            field.name: str(field.type)
            for field in schema
        }

        print(f"\nFile: {file}")

        missing = set(reference_fields) - set(current_fields)
        extra = set(current_fields) - set(reference_fields)

        if missing:
            print("  Missing columns:")
            for name in sorted(missing):
                print(f"    {name}")

        if extra:
            print("  Extra columns:")
            for name in sorted(extra):
                print(f"    {name}")

        common = set(reference_fields) & set(current_fields)

        type_differences = [
            name for name in sorted(common)
            if reference_fields[name] != current_fields[name]
        ]

        if type_differences:
            print("  Data type differences:")
            for name in type_differences:
                print(
                    f"    {name}: "
                    f"{reference_fields[name]} -> "
                    f"{current_fields[name]}"
                )

        reference_order = reference_schema.names
        current_order = schema.names

        if (
            not missing
            and not extra
            and not type_differences
            and reference_order != current_order
        ):
            print("  Column ordering differs.")

        if (
            not missing
            and not extra
            and not type_differences
            and reference_order == current_order
        ):
            print(
                "  No column/type/order difference found. "
                "Check other schema metadata."
            )


def main():
    print("=" * 75)
    print("STAGE 3: PARQUET SCHEMA DIAGNOSTICS")
    print("=" * 75)

    for region in REGIONS:
        for dataset in DATASETS:
            inspect_schema(region, dataset)

    print("\n" + "=" * 75)
    print("SCHEMA DIAGNOSTICS COMPLETE")
    print("No files were modified.")
    print("=" * 75)


if __name__ == "__main__":
    main()