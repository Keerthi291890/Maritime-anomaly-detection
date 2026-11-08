"""
Final patch — retries ONLY fishing_events, with a longer client
timeout and smaller page size, into the existing pull folder.
"""

import asyncio
import json
import os
from pathlib import Path

import gfwapiclient as gfw
from dotenv import load_dotenv

import stage1_ingestion
from stage1_ingestion import pull_events, utc_now

load_dotenv()
token = os.getenv("GFW_API_ACCESS_TOKEN")

EXISTING_PULL_DIR = Path("data/raw/pull_2026-10-03T08-56-50Z")  # confirm this matches your real folder
MANIFEST_PATH = EXISTING_PULL_DIR / "manifest.json"

REGION = {"dataset": "public-eez-areas", "id": "8466"}
START_DATE = "2025-01-01"
END_DATE = "2026-01-01"


async def main():
    manifest = json.loads(MANIFEST_PATH.read_text())

    # longer timeout, specifically for this heavy dataset
    client = gfw.Client(access_token=token, timeout=180.0, connect_timeout=10.0)

    stage1_ingestion.SAMPLE_DIR = EXISTING_PULL_DIR
    stage1_ingestion.MANIFEST_PATH = MANIFEST_PATH

    await pull_events(
        client=client,
        dataset="public-global-fishing-events:latest",
        dataset_name="fishing_events",
        region=REGION,
        start_date=START_DATE,
        end_date=END_DATE,
        manifest=manifest,
        page_size=10000,  # smaller pages to reduce per-request time
    )

    statuses = [d["status"] for d in manifest["results"].values()]
    manifest["overall_status"] = (
        "success" if all(s == "success" for s in statuses) else "partial_failed"
    )
    manifest["resumed_at_utc"] = utc_now()
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, default=str))

    print(f"\nFinal patch complete. Overall status: {manifest['overall_status']}")


if __name__ == "__main__":
    asyncio.run(main())