
from pathlib import Path
from collections import defaultdict
import pyarrow.parquet as pq

SAR_DIR = Path("../data/processed_v2/sar_presence")

# Values that may represent missing IDs despite not being actual nulls.
MISSING_MARKERS = {"", "nan", "none", "null", "<na>"}

stats = defaultdict(lambda: {
    "files": 0,
    "rows": 0,
    "null": 0,
    "blank": 0,
    "sentinel": 0,
    "valid": 0,
    "unique_ids": set(),
    "errors": 0,
})

files = sorted(
    SAR_DIR.glob("region=*/year=*/month=*/*.parquet")
)

if not files:
    raise SystemExit(f"No Parquet files found under {SAR_DIR.resolve()}")

for file in files:
    region = next(
        part.split("=", 1)[1]
        for part in file.parts
        if part.startswith("region=")
    )

    result = stats[region]

    try:
        table = pq.read_table(file, columns=["vessel_id"])
        values = table.column("vessel_id").to_pylist()

        result["files"] += 1
        result["rows"] += len(values)

        for value in values:
            if value is None:
                result["null"] += 1
                continue

            text = str(value).strip()
            normalized = text.lower()

            if normalized == "":
                result["blank"] += 1
            elif normalized in MISSING_MARKERS:
                result["sentinel"] += 1
            else:
                result["valid"] += 1
                result["unique_ids"].add(text)

    except Exception as exc:
        result["errors"] += 1
        print(f"ERROR reading {file}: {exc}")

print("=" * 65)
print("SAR VESSEL ID READINESS AUDIT")
print("=" * 65)

totals = {
    "files": 0,
    "rows": 0,
    "null": 0,
    "blank": 0,
    "sentinel": 0,
    "valid": 0,
    "errors": 0,
}

for region in sorted(stats):
    s = stats[region]

    print(f"\nREGION: {region.upper()}")
    print(f"  Parquet files:       {s['files']:,}")
    print(f"  Total rows:          {s['rows']:,}")
    print(f"  Actual null IDs:     {s['null']:,}")
    print(f"  Blank IDs:           {s['blank']:,}")
    print(f"  Missing markers:     {s['sentinel']:,}")
    print(f"  Valid IDs:           {s['valid']:,}")
    print(f"  Unique valid IDs:    {len(s['unique_ids']):,}")
    print(f"  Read errors:         {s['errors']:,}")

    for key in totals:
        totals[key] += s[key]

print("\n" + "=" * 65)
print("OVERALL TOTAL")
print("=" * 65)

for key, value in totals.items():
    print(f"{key.replace('_', ' ').title():22}: {value:,}")

print("\nAudit complete. No files were modified.")
