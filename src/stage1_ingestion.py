"""
Stage 1 — Raw ingestion with a 1-year time window, parameterized by region.

Responsibilities:
- Pull raw GFW data for a given EEZ region (passed via CLI args).
- Chunk AIS/SAR 4Wings requests into smaller date ranges.
- Paginate Events requests with offset, so no dataset silently caps
  at the page size.
- Store raw results as JSON.
- Create a fresh timestamped pull directory per region on every run.
- Maintain manifest.json incrementally so partial failures are visible.

Run:
    python stage1_ingestion.py --region-id 8466 --region-name argentina
    python stage1_ingestion.py --region-id 8371 --region-name senegal
    python stage1_ingestion.py --region-id 8400 --region-name ghana
    python stage1_ingestion.py --region-id 8403 --region-name ecuador_galapagos
"""

import argparse
import asyncio
import datetime as dt
import json
import os
from pathlib import Path

import pandas as pd
import gfwapiclient as gfw
from dotenv import load_dotenv


# ---------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Stage 1 GFW ingestion for one EEZ region")
parser.add_argument("--region-id", required=True, help="public-eez-areas id, e.g. 8466")
parser.add_argument("--region-name", required=True, help="short name for folder/log naming, e.g. argentina")
parser.add_argument("--start-date", default="2025-01-01", help="ISO date, default 2025-01-01")
parser.add_argument("--end-date", default="2026-01-01", help="ISO date, default 2026-01-01")
args = parser.parse_args()


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

load_dotenv()
token = os.getenv("GFW_API_ACCESS_TOKEN")

REGION = {"dataset": "public-eez-areas", "id": args.region_id}
REGION_NAME = args.region_name

START_DATE = args.start_date
END_DATE = args.end_date

N_CHUNKS = 3
EVENT_PAGE_SIZE = 10000
CLIENT_TIMEOUT = 180.0
CLIENT_CONNECT_TIMEOUT = 10.0

PULL_TIMESTAMP = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")

SAMPLE_DIR = Path(f"data/raw/pull_{REGION_NAME}_{PULL_TIMESTAMP}")
SAMPLE_DIR.mkdir(parents=True, exist_ok=False)

MANIFEST_PATH = SAMPLE_DIR / "manifest.json"


# ---------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------

def write_manifest(manifest: dict) -> None:
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, default=str),
        encoding="utf-8",
    )


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


# ---------------------------------------------------------------------
# Date chunking
# ---------------------------------------------------------------------

def make_date_chunks(start_date: str, end_date: str, n_chunks: int) -> list[tuple[str, str]]:
    start = dt.date.fromisoformat(start_date)
    end = dt.date.fromisoformat(end_date)

    if start >= end:
        raise ValueError("start_date must be earlier than end_date")
    if n_chunks <= 0:
        raise ValueError("n_chunks must be greater than 0")

    total_days = (end - start).days
    if n_chunks > total_days:
        raise ValueError("n_chunks cannot be greater than the number of days in the requested range")

    chunk_days = total_days // n_chunks
    chunks = []
    chunk_start = start

    for i in range(n_chunks):
        chunk_end = end if i == n_chunks - 1 else chunk_start + dt.timedelta(days=chunk_days)
        chunks.append((chunk_start.isoformat(), chunk_end.isoformat()))
        chunk_start = chunk_end

    return chunks


# ---------------------------------------------------------------------
# 4Wings chunked pull
# ---------------------------------------------------------------------

async def pull_fourwings_chunked(
    client, dataset: str, region: dict, start_date: str, end_date: str,
    n_chunks: int, dataset_name: str, manifest: dict,
) -> pd.DataFrame:
    chunks = make_date_chunks(start_date, end_date, n_chunks)
    frames = []
    dataset_manifest = manifest["results"][dataset_name]

    for index, (chunk_start, chunk_end) in enumerate(chunks, start=1):
        chunk_id = f"chunk_{index}"
        chunk_record = {
            "chunk_id": chunk_id,
            "start_date": chunk_start,
            "end_date": chunk_end,
            "status": "running",
            "pulled_at_utc": utc_now(),
            "row_count": None,
            "file": None,
            "error": None,
        }
        dataset_manifest["chunks"].append(chunk_record)
        write_manifest(manifest)

        print(f"    {chunk_id}: {chunk_start} -> {chunk_end}")

        try:
            result = await client.fourwings.create_report(
                spatial_resolution="LOW",
                temporal_resolution="DAILY",
                start_date=chunk_start,
                end_date=chunk_end,
                datasets=[dataset],
                region=region,
            )
            df = result.df()

            chunk_filename = f"{dataset_name}_{chunk_start}_{chunk_end}.json"
            chunk_path = SAMPLE_DIR / chunk_filename
            df.to_json(chunk_path, orient="records", date_format="iso")

            frames.append(df)
            chunk_record["status"] = "success"
            chunk_record["row_count"] = int(len(df))
            chunk_record["file"] = chunk_filename
            print(f"      -> SUCCESS: {len(df)} rows")

        except Exception as e:
            chunk_record["status"] = "failed"
            chunk_record["error"] = {"type": type(e).__name__, "message": str(e)}
            print(f"      -> FAILED: {type(e).__name__} - {e}")

        finally:
            write_manifest(manifest)

    successful_chunks = [c for c in dataset_manifest["chunks"] if c["status"] == "success"]
    failed_chunks = [c for c in dataset_manifest["chunks"] if c["status"] == "failed"]

    if failed_chunks and successful_chunks:
        dataset_manifest["status"] = "partial_failed"
    elif failed_chunks:
        dataset_manifest["status"] = "failed"
    else:
        dataset_manifest["status"] = "success"

    dataset_manifest["row_count"] = sum(c["row_count"] or 0 for c in successful_chunks)
    write_manifest(manifest)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------
# Event pull — paginated with offset
# ---------------------------------------------------------------------

async def pull_events(
    client, dataset: str, dataset_name: str, region: dict,
    start_date: str, end_date: str, manifest: dict, page_size: int = EVENT_PAGE_SIZE,
) -> pd.DataFrame:
    dataset_manifest = manifest["results"][dataset_name]
    dataset_manifest["status"] = "running"
    dataset_manifest["pulled_at_utc"] = utc_now()
    write_manifest(manifest)

    print(f"\n=== {dataset_name.upper()} ===")

    frames = []
    offset = 0
    page_num = 1

    try:
        while True:
            result = await client.events.get_all_events(
                datasets=[dataset],
                start_date=start_date,
                end_date=end_date,
                region=region,
                limit=page_size,
                offset=offset,
            )
            df = result.df()
            print(f"    page_{page_num} (offset {offset}): {len(df)} rows")

            page_filename = f"{dataset_name}_page{page_num}.json"
            page_path = SAMPLE_DIR / page_filename
            df.to_json(page_path, orient="records", date_format="iso")

            dataset_manifest["pages"].append({
                "page": page_num,
                "offset": offset,
                "row_count": int(len(df)),
                "file": page_filename,
            })
            write_manifest(manifest)

            frames.append(df)

            if len(df) < page_size:
                break

            offset += page_size
            page_num += 1

        total_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        dataset_manifest["status"] = "success"
        dataset_manifest["row_count"] = int(len(total_df))
        dataset_manifest["error"] = None

        print(f"  -> SUCCESS: {len(total_df)} total rows across {page_num} page(s)")
        write_manifest(manifest)
        return total_df

    except Exception as e:
        dataset_manifest["status"] = "failed"
        dataset_manifest["error"] = {"type": type(e).__name__, "message": str(e)}
        print(f"  -> FAILED: {type(e).__name__} - {e}")
        write_manifest(manifest)
        return pd.DataFrame()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

async def main() -> None:
    if not token:
        raise SystemExit("GFW_API_ACCESS_TOKEN not set — check your .env")

    client = gfw.Client(
        access_token=token,
        timeout=CLIENT_TIMEOUT,
        connect_timeout=CLIENT_CONNECT_TIMEOUT,
    )

    manifest = {
        "stage": 1,
        "description": "Raw GFW data ingestion",
        "region_name": REGION_NAME,
        "region": REGION,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "pulled_at_utc": utc_now(),

        "dataset_versions": {
            "ais_presence": "public-global-presence:latest",
            "sar_presence": "public-global-sar-presence:latest",
            "gap_events": "public-global-gaps-events:latest",
            "encounter_events": "public-global-encounters-events:latest",
            "loitering_events": "public-global-loitering-events:latest",
            "fishing_events": "public-global-fishing-events:latest",
        },

        "results": {
            "ais_presence": {"status": "pending", "chunks": [], "row_count": 0},
            "sar_presence": {"status": "pending", "chunks": [], "row_count": 0},
            "gap_events": {"status": "pending", "pages": [], "row_count": None, "error": None},
            "encounter_events": {"status": "pending", "pages": [], "row_count": None, "error": None},
            "loitering_events": {"status": "pending", "pages": [], "row_count": None, "error": None},
            "fishing_events": {"status": "pending", "pages": [], "row_count": None, "error": None},
        },
    }

    write_manifest(manifest)
    print(f"Region: {REGION_NAME} (id {args.region_id})")
    print(f"Raw pull directory: {SAMPLE_DIR}")

    print("\n=== AIS PRESENCE ===")
    await pull_fourwings_chunked(
        client=client, dataset="public-global-presence:latest", region=REGION,
        start_date=START_DATE, end_date=END_DATE, n_chunks=N_CHUNKS,
        dataset_name="ais_presence", manifest=manifest,
    )

    print("\n=== SAR PRESENCE ===")
    await pull_fourwings_chunked(
        client=client, dataset="public-global-sar-presence:latest", region=REGION,
        start_date=START_DATE, end_date=END_DATE, n_chunks=N_CHUNKS,
        dataset_name="sar_presence", manifest=manifest,
    )

    event_datasets = [
        ("gap_events", "public-global-gaps-events:latest"),
        ("encounter_events", "public-global-encounters-events:latest"),
        ("loitering_events", "public-global-loitering-events:latest"),
        ("fishing_events", "public-global-fishing-events:latest"),
    ]

    for name, dataset in event_datasets:
        await pull_events(
            client=client, dataset=dataset, dataset_name=name, region=REGION,
            start_date=START_DATE, end_date=END_DATE, manifest=manifest,
        )

    statuses = [d["status"] for d in manifest["results"].values()]
    if all(s == "success" for s in statuses):
        manifest["overall_status"] = "success"
    elif all(s == "failed" for s in statuses):
        manifest["overall_status"] = "failed"
    else:
        manifest["overall_status"] = "partial_failed"

    manifest["completed_at_utc"] = utc_now()
    write_manifest(manifest)

    print(
        f"\n=== DONE ({REGION_NAME}) ===\n"
        f"Status: {manifest['overall_status']}\n"
        f"Manifest: {MANIFEST_PATH}"
    )


if __name__ == "__main__":
    asyncio.run(main())