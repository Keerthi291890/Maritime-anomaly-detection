import json
from pathlib import Path
from collections import Counter

RAW_DIR = Path("../data/raw")
MAX_FILES_PER_DATASET = 3
SAMPLES_PER_FILE = 2


def describe_value(value):
    """Describe a value without printing large arrays or objects."""
    if isinstance(value, dict):
        return f"dict with keys: {list(value.keys())[:15]}"
    elif isinstance(value, list):
        if value:
            return f"list ({len(value)} items), first item type: {type(value[0]).__name__}"
        return "empty list"
    elif value is None:
        return "None"
    else:
        return type(value).__name__


def find_record_lists(obj, path="root"):
    """Find lists containing dictionary records inside nested JSON."""
    found = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            found.extend(find_record_lists(value, f"{path}.{key}"))

    elif isinstance(obj, list):
        if obj and isinstance(obj[0], dict):
            found.append((path, obj))
        else:
            for i, value in enumerate(obj[:3]):
                found.extend(find_record_lists(value, f"{path}[{i}]"))

    return found


def inspect_json_file(file_path):
    print("\n" + "=" * 80)
    print(f"FILE: {file_path}")
    print("=" * 80)

    try:
        if file_path.suffix.lower() == ".jsonl":
            records = []
            with file_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
                    if len(records) >= SAMPLES_PER_FILE:
                        break

            print("Format: JSON Lines")
            for i, record in enumerate(records, 1):
                print(f"\nRecord {i}:")
                if isinstance(record, dict):
                    for key, value in record.items():
                        print(f"  {key}: {describe_value(value)}")
                else:
                    print(f"  Type: {type(record).__name__}")
            return

        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"Top-level type: {type(data).__name__}")

        if isinstance(data, dict):
            print("Top-level keys:")
            for key, value in data.items():
                print(f"  {key}: {describe_value(value)}")

        candidates = find_record_lists(data)

        if not candidates:
            print("\nNo list of dictionary records found.")
            return

        for path, records in candidates[:3]:
            print(f"\nPossible records at: {path}")
            print(f"Number of records in this list: {len(records)}")

            for i, record in enumerate(records[:SAMPLES_PER_FILE], 1):
                print(f"\nSample record {i}:")
                for key, value in record.items():
                    print(f"  {key}: {describe_value(value)}")

    except Exception as e:
        print(f"Could not inspect file: {e}")


def main():
    if not RAW_DIR.exists():
        print(f"Raw data folder not found: {RAW_DIR.resolve()}")
        return

    files = sorted(
        p for p in RAW_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in {".json", ".jsonl"}
    )

    if not files:
        print(f"No JSON or JSONL files found under {RAW_DIR.resolve()}")
        return

    print(f"Raw data directory: {RAW_DIR.resolve()}")
    print(f"Found {len(files)} JSON/JSONL files.")

    # Inspect a limited number of files from each parent folder.
    grouped = {}
    for file_path in files:
        grouped.setdefault(file_path.parent, []).append(file_path)

    for folder, folder_files in grouped.items():
        print("\n\n" + "#" * 80)
        print(f"FOLDER: {folder}")
        print(f"Files found: {len(folder_files)}")
        print("#" * 80)

        for file_path in folder_files[:MAX_FILES_PER_DATASET]:
            inspect_json_file(file_path)


if __name__ == "__main__":
    main()