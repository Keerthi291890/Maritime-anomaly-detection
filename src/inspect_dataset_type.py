
import json
import re
from pathlib import Path

RAW_DIR = Path("../data/raw")
SAMPLES_PER_FILE = 2


def dataset_type(path):
    # Remove the date range from names such as:
    # ais_presence_2025-01-01_2025-05-02.json
    name = path.stem
    name = re.sub(r"_\d{4}-\d{2}-\d{2}.*$", "", name)
    return name


def show_value(value, limit=180):
    text = repr(value)
    return text if len(text) <= limit else text[:limit] + "..."


def inspect_file(path):
    print("\n" + "=" * 85)
    print("FILE:", path)
    print("=" * 85)

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        print("Top-level type:", type(data).__name__)

        if isinstance(data, list):
            print("Total records:", len(data))
            records = data[:SAMPLES_PER_FILE]

        elif isinstance(data, dict):
            print("Top-level keys:", list(data.keys()))
            records = []

            # Look for a list of records inside the object.
            for key, value in data.items():
                if isinstance(value, list) and value:
                    if isinstance(value[0], dict):
                        print("Record list found under:", key)
                        print("Total records:", len(value))
                        records = value[:SAMPLES_PER_FILE]
                        break
        else:
            print("Unexpected JSON structure.")
            return

        for i, record in enumerate(records, 1):
            print(f"\nSAMPLE RECORD {i}:")
            if isinstance(record, dict):
                for key, value in record.items():
                    print(f"  {key}: {show_value(value)}")
            else:
                print(" ", show_value(record))

    except Exception as e:
        print("ERROR:", e)


def main():
    if not RAW_DIR.exists():
        print("Raw directory not found:", RAW_DIR.resolve())
        return

    files = [
        p for p in RAW_DIR.rglob("*.json")
        if p.is_file()
    ]

    # Select one representative file for every dataset type.
    selected = {}

    for path in sorted(files):
        kind = dataset_type(path)
        selected.setdefault(kind, path)

    print("Total JSON files:", len(files))
    print("Distinct dataset types:", len(selected))
    print("\nSelected dataset files:")

    for kind, path in selected.items():
        print(f"  {kind}: {path}")

    for path in selected.values():
        inspect_file(path)


if __name__ == "__main__":
    main()
