from pathlib import Path
import json
import pandas as pd

ROOT = Path("data")
INPUT = ROOT / "stage4_input"
MATCH_DIR = ROOT / "stage4_output"
EVENT_DIR = ROOT / "processed_v2"
OUTPUT = ROOT / "stage5_results"

REGIONS = [
    "argentina",
    "ecuador_galapagos",
    "ghana",
    "senegal",
]

EVENT_DATASETS = [
    "fishing_events",
    "loitering_events",
    "gap_events",
    "encounter_events",
]


def parse_json(value):
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        result = json.loads(value)
        return result if isinstance(result, dict) else {}
    except (ValueError, TypeError):
        return {}


def load_metadata(dataset):
    frames = []

    for region in REGIONS:
        path = INPUT / dataset / f"{region}_metadata.parquet"

        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_parquet(path)
        df["record_id"] = df["record_id"].astype("string")
        frames.append(df)

    result = pd.concat(frames, ignore_index=True)

    if result["record_id"].duplicated().any():
        raise ValueError(f"Duplicate record IDs in {dataset} metadata")

    return result


def load_matches():
    files = sorted(MATCH_DIR.glob("matches_*.csv"))

    if not files:
        raise FileNotFoundError("No Stage 4 matching CSV files found")

    frames = []

    for file in files:
        df = pd.read_csv(
            file,
            dtype={
                "sar_record_id": "string",
                "ais_record_id": "string",
            },
        )
        frames.append(df)

    result = pd.concat(frames, ignore_index=True)

    if result.duplicated(
        ["sar_record_id", "ais_record_id"]
    ).any():
        raise ValueError("Duplicate SAR-AIS pairs found")

    return result


def load_events():
    all_events = []

    for dataset in EVENT_DATASETS:
        print(f"Loading {dataset}...")

        files = sorted(
            (EVENT_DIR / dataset).rglob("*.parquet")
        )

        if not files:
            raise FileNotFoundError(
                f"No Parquet files found for {dataset}"
            )

        for file in files:
            df = pd.read_parquet(file)

            for _, row in df.iterrows():
                vessel = parse_json(row.get("vessel"))

                primary_id = vessel.get("id")

                if not primary_id:
                    continue

                start = pd.to_datetime(
                    row.get("start"), utc=True, errors="coerce"
                )
                end = pd.to_datetime(
                    row.get("end"), utc=True, errors="coerce"
                )

                if pd.isna(start):
                    continue

                # Events without an end date are treated as
                # single-date events.
                if pd.isna(end):
                    end = start

                region = row.get("region")

                if pd.isna(region):
                    continue

                common = {
                    "event_dataset": dataset,
                    "event_id": row.get("id"),
                    "event_type": row.get("type"),
                    "region": str(region).lower(),
                    "event_start": start.isoformat(),
                    "event_end": end.isoformat(),
                    "event_start_date": start.strftime("%Y-%m-%d"),
                    "event_end_date": end.strftime("%Y-%m-%d"),
                    "event_lat": row.get("lat"),
                    "event_lon": row.get("lon"),
                    "event_source_file": row.get("source_file"),
                }

                # Primary vessel association.
                all_events.append({
                    **common,
                    "vessel_id": str(primary_id),
                    "participant_role": "primary",
                    "participant_ssvid": vessel.get("ssvid"),
                })

                # Encounter events also have a second vessel.
                if dataset == "encounter_events":
                    other_id = row.get("other_vessel_id")

                    if pd.notna(other_id) and str(other_id).strip():
                        all_events.append({
                            **common,
                            "vessel_id": str(other_id),
                            "participant_role": "secondary",
                            "participant_ssvid": row.get(
                                "other_vessel_ssvid"
                            ),
                        })

    events = pd.DataFrame(all_events)

    if events.empty:
        raise ValueError("No usable event records found")

    events["vessel_id"] = events["vessel_id"].astype("string")

    return events


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STAGE 5: RESULT INTEGRATION")
    print("=" * 70)

    # 1. Load matching results.
    matches = load_matches()
    print(f"\nMatching rows: {len(matches):,}")

    # 2. Load metadata.
    sar = load_metadata("sar_presence")
    ais = load_metadata("ais_presence")

    print(f"SAR metadata rows: {len(sar):,}")
    print(f"AIS metadata rows: {len(ais):,}")

    # 3. Rename metadata columns to avoid collisions.
    sar = sar.rename(columns={
        column: f"sar_{column}"
        for column in sar.columns
        if column != "record_id"
    })

    ais = ais.rename(columns={
        column: f"ais_{column}"
        for column in ais.columns
        if column != "record_id"
    })

    # 4. Join SAR metadata. Keep every matching output row.
    result = matches.merge(
        sar,
        left_on="sar_record_id",
        right_on="record_id",
        how="left",
        validate="many_to_one",
    ).drop(columns=["record_id"])

    missing_sar = result["sar_date"].isna().sum()

    if missing_sar:
        raise ValueError(
            f"{missing_sar} matching rows lack SAR metadata"
        )

    result["sar_date"] = pd.to_datetime(
        result["sar_date"]
    ).dt.strftime("%Y-%m-%d")

    # 5. Join AIS metadata.
    result = result.merge(
        ais,
        left_on="ais_record_id",
        right_on="record_id",
        how="left",
        validate="many_to_one",
    ).drop(columns=["record_id"])

    missing_ais = (
        result["match_status"].eq("matched")
        & result["ais_date"].isna()
    ).sum()

    if missing_ais:
        raise ValueError(
            f"{missing_ais} matched rows lack AIS metadata"
        )

    result["ais_date"] = pd.to_datetime(
        result["ais_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    # Check that matching and metadata regions agree.
    # Check regions only for matched records, using normalized strings.
    matched_region_check = result.loc[
        result["match_status"].eq("matched"),
        ["sar_region", "ais_region"]
    ].copy()

    matched_region_check["sar_region"] = (
        matched_region_check["sar_region"]
        .astype("string").str.strip().str.lower()
    )
    matched_region_check["ais_region"] = (
        matched_region_check["ais_region"]
        .astype("string").str.strip().str.lower()
    )

    missing_region = (
        matched_region_check["sar_region"].isna()
        | matched_region_check["ais_region"].isna()
    ).sum()

    if missing_region:
        raise ValueError(
            f"{missing_region} matched rows have missing region metadata"
        )

    region_mismatch = (
        matched_region_check["sar_region"]
        != matched_region_check["ais_region"]
    ).sum()

    if region_mismatch:
        raise ValueError(
            f"{region_mismatch} cross-region matches detected"
        )

    print("PASS: All matched records belong to the same region")

    # 6. Save consolidated matching results.
    matches_path = OUTPUT / "sar_ais_matches.parquet"
    result.to_parquet(matches_path, index=False)

    print(f"\nConsolidated matches saved: {matches_path}")
    print(f"Consolidated rows: {len(result):,}")

    # 7. Load and normalize event records.
    events = load_events()

    print(f"\nEvent participant records: {len(events):,}")

    # 8. Match events to AIS candidate vessels.
    matched = result[
        (result["match_status"] == "matched")
        & result["ais_vessel_id"].notna()
    ].copy()

    links = []

    for dataset in EVENT_DATASETS:
        subset = events[
            events["event_dataset"] == dataset
        ].copy()

        if subset.empty:
            continue

        print(f"Associating {dataset}...")

        candidates = matched.merge(
            subset,
            left_on=["sar_region", "ais_vessel_id"],
            right_on=["region", "vessel_id"],
            how="inner",
        )

        # Inclusive UTC calendar-date overlap.
        candidates = candidates[
            (candidates["sar_date"] >= candidates["event_start_date"])
            & (candidates["sar_date"] <= candidates["event_end_date"])
        ]

        if not candidates.empty:
            links.append(candidates)

    if links:
        event_links = pd.concat(links, ignore_index=True)

        event_links = event_links[[
            "sar_record_id",
            "ais_record_id",
            "sar_region",
            "sar_date",
            "ais_vessel_id",
            "event_dataset",
            "event_id",
            "event_type",
            "participant_role",
            "participant_ssvid",
            "event_start",
            "event_end",
            "event_lat",
            "event_lon",
            "event_source_file",
        ]]

        event_links = event_links.drop_duplicates()
    else:
        event_links = pd.DataFrame(columns=[
            "sar_record_id",
            "ais_record_id",
            "sar_region",
            "sar_date",
            "ais_vessel_id",
            "event_dataset",
            "event_id",
            "event_type",
            "participant_role",
            "participant_ssvid",
            "event_start",
            "event_end",
            "event_lat",
            "event_lon",
            "event_source_file",
        ])

    # 9. Save event links separately to avoid multiplying
    # the consolidated matching rows.
    links_path = OUTPUT / "match_event_links.parquet"
    event_links.to_parquet(links_path, index=False)

    print(f"\nEvent links saved: {links_path}")
    print(f"Event link rows: {len(event_links):,}")

    # 10. Generate summary.
    summary = []

    for dataset in EVENT_DATASETS:
        subset = event_links[
            event_links["event_dataset"] == dataset
        ]

        summary.append({
            "event_dataset": dataset,
            "event_links": len(subset),
            "unique_sar_records": (
                subset["sar_record_id"].nunique()
            ),
            "unique_ais_records": (
                subset["ais_record_id"].nunique()
            ),
            "unique_events": subset["event_id"].nunique(),
        })

    summary_df = pd.DataFrame(summary)

    summary_path = OUTPUT / "event_enrichment_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 70)
    print("STAGE 5 SUMMARY")
    print("=" * 70)

    print(f"Matching rows: {len(result):,}")
    print(
        "Unique SAR records:",
        result["sar_record_id"].nunique()
    )
    print(
        "Matched rows:",
        (result["match_status"] == "matched").sum()
    )
    print(
        "Unmatched rows:",
        (result["match_status"] == "unmatched").sum()
    )
    print(f"Event links: {len(event_links):,}")

    print("\nEvent enrichment:")
    print(summary_df.to_string(index=False))

    print("\nOutput directory:", OUTPUT)
    print("STAGE 5 INTEGRATION COMPLETED")


if __name__ == "__main__":
    main()
