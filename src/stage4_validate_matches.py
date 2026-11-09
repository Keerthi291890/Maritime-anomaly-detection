from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "stage4_input"
OUTPUT = ROOT / "data" / "stage4_output"
REPORT = ROOT / "data" / "stage4_validation_report.txt"

REGIONS = [
    "argentina",
    "ecuador_galapagos",
    "ghana",
    "senegal",
]

EARTH_RADIUS_KM = 6371.0088
MAX_DISTANCE_KM = 25.0
DISTANCE_TOLERANCE_KM = 0.00002


def normalize_id(series):
    """Match the clean_string() rules used by stage4_prepare_inputs.py."""
    result = series.fillna("").astype(str).str.strip()
    missing_tokens = {"nan", "none", "null", "<na>"}
    return result.mask(result.str.lower().isin(missing_tokens), "")


def normalize_dates(series):
    return pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m-%d")


def haversine_km(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin(dlon / 2.0) ** 2
    )
    a = np.clip(a, 0.0, 1.0)
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def load_metadata(region, dataset):
    path = INPUT / dataset / f"{region}_metadata.parquet"
    df = pd.read_parquet(path)

    required = {"record_id", "date", "vessel_id", "lat", "lon", "region"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")

    df = df.copy()
    df["record_id"] = df["record_id"].astype("int64")
    df["vessel_id_norm"] = normalize_id(df["vessel_id"])
    df["date_norm"] = normalize_dates(df["date"])

    if df["date_norm"].isna().any():
        raise ValueError(f"Invalid dates in {path}")

    return df


def load_outputs():
    files = sorted(OUTPUT.glob("matches_*.csv"))
    if not files:
        raise FileNotFoundError(f"No match CSV files found in {OUTPUT}")

    frames = []
    required = {
        "sar_record_id", "ais_record_id",
        "distance_km", "match_method", "match_status",
    }

    for path in files:
        df = pd.read_csv(path)
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path.name}: missing columns {sorted(missing)}")

        df["source_output_file"] = path.name
        df["sar_record_id"] = pd.to_numeric(
            df["sar_record_id"], errors="raise"
        ).astype("int64")
        df["ais_record_id"] = pd.to_numeric(
            df["ais_record_id"], errors="coerce"
        )
        df["distance_km"] = pd.to_numeric(
            df["distance_km"], errors="coerce"
        )
        df["match_status"] = (
            df["match_status"].astype(str).str.strip().str.lower()
        )
        df["match_method"] = (
            df["match_method"].astype(str).str.strip().str.lower()
        )
        frames.append(df)

    results = pd.concat(frames, ignore_index=True)

    def get_region(filename):
        for region in REGIONS:
            if f"matches_{region}_" in filename:
                return region
        raise ValueError(f"Unknown region in filename: {filename}")

    results["region"] = results["source_output_file"].map(get_region)
    return results


def main():
    print("=" * 72)
    print("STAGE 4: INDEPENDENT MATCH VALIDATION")
    print("=" * 72)

    results = load_outputs()
    print(f"\nOutput rows loaded: {len(results):,}")
    print(f"Output files: {results['source_output_file'].nunique()}")

    all_ais = {}
    all_sar = {}
    expected_sar_keys = set()
    expected_unmatched = set()
    expected_spatial_pairs = set()
    sar_lookup = {}
    expected_total_ais = 0
    expected_total_sar = 0
    id_sar_count = 0
    spatial_sar_count = 0
    expected_id_matches = 0

    for region in REGIONS:
        print(f"\nLoading {region} metadata...")
        ais = load_metadata(region, "ais_presence")
        sar = load_metadata(region, "sar_presence")
        all_ais[region] = ais
        all_sar[region] = sar

        expected_total_ais += len(ais)
        expected_total_sar += len(sar)
        print(f"  AIS: {len(ais):,}")
        print(f"  SAR: {len(sar):,}")

        if ais["record_id"].duplicated().any():
            raise ValueError(f"Duplicate AIS record IDs in {region}")
        if sar["record_id"].duplicated().any():
            raise ValueError(f"Duplicate SAR record IDs in {region}")

        ais_by_date = {
            date: group.reset_index(drop=True)
            for date, group in ais.groupby("date_norm", sort=False)
        }

        for row in sar.itertuples(index=False):
            sar_id = int(row.record_id)
            sar_key = (region, sar_id)
            if sar_key in expected_sar_keys:
                raise ValueError(f"Duplicate SAR record: {sar_key}")
            expected_sar_keys.add(sar_key)
            sar_lookup[sar_key] = row

            day_ais = ais_by_date.get(row.date_norm)
            if day_ais is None or day_ais.empty:
                expected_unmatched.add(sar_key)
                continue

            if row.vessel_id_norm:
                id_sar_count += 1
                candidates = day_ais[
                    day_ais["vessel_id_norm"] == row.vessel_id_norm
                ]
                if candidates.empty:
                    expected_unmatched.add(sar_key)
                    continue

                distances = haversine_km(
                    float(row.lat),
                    float(row.lon),
                    candidates["lat"].to_numpy(dtype=float),
                    candidates["lon"].to_numpy(dtype=float),
                )
                min_distance = float(np.min(distances))
                expected_id_matches += 1

                # The C implementation emits one nearest candidate.
                # Do not require every tied/near-tied minimum candidate.
                # The actual output is checked against min_distance below.
                continue

            spatial_sar_count += 1
            distances = haversine_km(
                float(row.lat),
                float(row.lon),
                day_ais["lat"].to_numpy(dtype=float),
                day_ais["lon"].to_numpy(dtype=float),
            )
            valid_indices = np.where(distances <= MAX_DISTANCE_KM)[0]

            if len(valid_indices) == 0:
                expected_unmatched.add(sar_key)
                continue

            for candidate_index in valid_indices:
                candidate = day_ais.iloc[int(candidate_index)]
                expected_spatial_pairs.add((
                    region,
                    sar_id,
                    int(candidate["record_id"]),
                ))

    # Basic structural checks.
    errors = []

    def check(condition, message):
        if not condition:
            errors.append(message)
            print(f"FAIL: {message}")
        else:
            print(f"PASS: {message}")

    print("\n" + "=" * 72)
    print("INDEPENDENT EXPECTED RESULTS")
    print("=" * 72)
    print(f"Total AIS records:             {expected_total_ais:,}")
    print(f"Total SAR records:             {expected_total_sar:,}")
    print(f"SAR records with usable ID:    {id_sar_count:,}")
    print(f"SAR records without usable ID: {spatial_sar_count:,}")
    print(f"ID-based SAR records with AIS candidate: {expected_id_matches:,}")
    print(f"Expected unmatched SAR:        {len(expected_unmatched):,}")
    print(f"Expected spatial candidate pairs: {len(expected_spatial_pairs):,}")

    matched = results[results["match_status"] == "matched"].copy()
    unmatched = results[results["match_status"] == "unmatched"].copy()

    check(
        results["sar_record_id"].notna().all(),
        "All output rows contain SAR record IDs",
    )
    check(
        set(results["match_status"].unique()) <= {"matched", "unmatched"},
        "Only valid match statuses are present",
    )
    check(
        results.loc[results["match_status"] == "matched", "ais_record_id"].notna().all(),
        "Every matched row contains an AIS record ID",
    )
    check(
        results.loc[results["match_status"] == "matched", "distance_km"].notna().all(),
        "Every matched row contains a distance",
    )

    output_sar_keys = set(
        zip(results["region"], results["sar_record_id"].astype(int))
    )
    check(
        output_sar_keys == expected_sar_keys,
        "Every SAR record is represented "
        f"(missing: {len(expected_sar_keys - output_sar_keys)}, "
        f"unexpected: {len(output_sar_keys - expected_sar_keys)})",
    )

    status_counts = results.groupby(
        ["region", "sar_record_id"]
    )["match_status"].nunique()
    check(
        (status_counts == 1).all(),
        "No SAR record has mixed matched/unmatched statuses",
    )

    unmatched_counts = unmatched.groupby(
        ["region", "sar_record_id"]
    ).size()
    check(
        (unmatched_counts == 1).all(),
        "Every unmatched SAR record has exactly one output row",
    )

    matched_pairs = matched[
        ["region", "sar_record_id", "ais_record_id"]
    ].copy()
    duplicate_pairs = matched_pairs.duplicated().sum()
    check(
        duplicate_pairs == 0,
        f"No duplicate SAR-AIS pairs (duplicates: {duplicate_pairs})",
    )

    # Validate each SAR record according to its matching rule.
    actual_matched_by_sar = {
        key: group
        for key, group in matched.groupby(["region", "sar_record_id"], sort=False)
    }
    actual_unmatched_keys = set(
        zip(unmatched["region"], unmatched["sar_record_id"].astype(int))
    )

    id_rule_errors = []
    spatial_actual_pairs = set()
    spatial_rule_errors = []
    distance_errors = []
    method_errors = []
    per_sar_count_errors = []

    for region in REGIONS:
        ais = all_ais[region]
        sar = all_sar[region]
        ais_by_date = {
            date: group.reset_index(drop=True)
            for date, group in ais.groupby("date_norm", sort=False)
        }

        for row in sar.itertuples(index=False):
            sar_id = int(row.record_id)
            sar_key = (region, sar_id)
            actual_rows = actual_matched_by_sar.get(sar_key)
            actual_count = 0 if actual_rows is None else len(actual_rows)
            day_ais = ais_by_date.get(row.date_norm)

            if day_ais is None or day_ais.empty:
                if sar_key not in actual_unmatched_keys or actual_count:
                    per_sar_count_errors.append(sar_key)
                continue

            if row.vessel_id_norm:
                candidates = day_ais[
                    day_ais["vessel_id_norm"] == row.vessel_id_norm
                ]

                if candidates.empty:
                    if sar_key not in actual_unmatched_keys or actual_count:
                        id_rule_errors.append(sar_key)
                    continue

                if actual_count != 1:
                    per_sar_count_errors.append((sar_key, "ID path", actual_count))
                    continue

                actual = actual_rows.iloc[0]
                ais_id = int(actual["ais_record_id"])
                candidate_rows = candidates[
                    candidates["record_id"] == ais_id
                ]
                if candidate_rows.empty:
                    id_rule_errors.append(sar_key)
                    continue

                actual_distance = float(actual["distance_km"])
                correct_distances = haversine_km(
                    float(row.lat),
                    float(row.lon),
                    candidate_rows["lat"].to_numpy(dtype=float),
                    candidate_rows["lon"].to_numpy(dtype=float),
                )
                correct_distance = float(correct_distances[0])

                all_candidate_distances = haversine_km(
                    float(row.lat),
                    float(row.lon),
                    candidates["lat"].to_numpy(dtype=float),
                    candidates["lon"].to_numpy(dtype=float),
                )
                minimum_distance = float(np.min(all_candidate_distances))

                if abs(correct_distance - minimum_distance) > DISTANCE_TOLERANCE_KM:
                    id_rule_errors.append(sar_key)

                if abs(actual_distance - correct_distance) > DISTANCE_TOLERANCE_KM:
                    distance_errors.append(sar_key)

                if str(actual["match_method"]).strip().lower() != "same_vessel_id":
                    method_errors.append(sar_key)

            else:
                distances = haversine_km(
                    float(row.lat),
                    float(row.lon),
                    day_ais["lat"].to_numpy(dtype=float),
                    day_ais["lon"].to_numpy(dtype=float),
                )
                valid_indices = np.where(distances <= MAX_DISTANCE_KM)[0]

                if len(valid_indices) == 0:
                    if sar_key not in actual_unmatched_keys or actual_count:
                        spatial_rule_errors.append(sar_key)
                    continue

                if actual_count == 0:
                    spatial_rule_errors.append(sar_key)
                    continue

                expected_for_sar = set()
                for candidate_index in valid_indices:
                    candidate = day_ais.iloc[int(candidate_index)]
                    pair = (region, sar_id, int(candidate["record_id"]))
                    expected_for_sar.add(pair)

                actual_for_sar = set()
                for actual in actual_rows.itertuples(index=False):
                    ais_id = int(actual.ais_record_id)
                    pair = (region, sar_id, ais_id)
                    actual_for_sar.add(pair)
                    if ais_id not in set(
                        day_ais.iloc[valid_indices]["record_id"].astype(int)
                    ):
                        spatial_rule_errors.append(sar_key)
                        continue

                    candidate = day_ais[
                        day_ais["record_id"] == ais_id
                    ].iloc[0]
                    correct_distance = float(haversine_km(
                        float(row.lat),
                        float(row.lon),
                        np.array([float(candidate["lat"])]),
                        np.array([float(candidate["lon"])]),
                    )[0])
                    if abs(float(actual.distance_km) - correct_distance) > DISTANCE_TOLERANCE_KM:
                        distance_errors.append(sar_key)
                    if str(actual.match_method).strip().lower() != "spatiotemporal":
                        method_errors.append(sar_key)

                spatial_actual_pairs.update(actual_for_sar)
                if actual_for_sar != expected_for_sar:
                    spatial_rule_errors.append(sar_key)

    check(
        len(id_rule_errors) == 0,
        "ID-based matches use the same vessel ID/date and select a nearest AIS record "
        f"(errors: {len(id_rule_errors)})",
    )
    check(
        len(per_sar_count_errors) == 0,
        "ID-based SAR records emit exactly one nearest match; unmatched SAR emit one row "
        f"(errors: {len(per_sar_count_errors)})",
    )
    check(
        len(spatial_rule_errors) == 0,
        "Missing-ID SAR records emit all and only same-day AIS candidates within 25 km "
        f"(errors: {len(spatial_rule_errors)})",
    )
    check(
        len(method_errors) == 0,
        "Matching methods agree with the matching rules "
        f"(errors: {len(method_errors)})",
    )
    check(
        len(distance_errors) == 0,
        "Distances agree with independent Haversine calculations "
        f"(errors: {len(distance_errors)})",
    )

    check(
        actual_unmatched_keys == expected_unmatched,
        "Unmatched SAR records are correct "
        f"(missing: {len(expected_unmatched - actual_unmatched_keys)}, "
        f"unexpected: {len(actual_unmatched_keys - expected_unmatched)})",
    )

    # Regional summary.
    print("\n" + "=" * 72)
    print("REGIONAL OUTPUT SUMMARY")
    print("=" * 72)
    for region in REGIONS:
        region_rows = results[results["region"] == region]
        region_matched = matched[matched["region"] == region]
        region_unmatched = unmatched[unmatched["region"] == region]
        print(
            f"{region:20s} | rows={len(region_rows):7,} | "
            f"matched={len(region_matched):7,} | "
            f"unmatched={len(region_unmatched):5,}"
        )

    print("\n" + "=" * 72)
    status = "ALL CHECKS PASSED" if not errors else "VALIDATION FAILED"
    print(status)
    for error in errors:
        print(" -", error)

    report_lines = [
        "STAGE 4 INDEPENDENT MATCH VALIDATION",
        "=" * 60,
        f"Total AIS: {expected_total_ais}",
        f"Total SAR: {expected_total_sar}",
        f"Output rows: {len(results)}",
        f"Matched output rows: {len(matched)}",
        f"Expected unmatched SAR: {len(expected_unmatched)}",
        f"SAR with usable ID: {id_sar_count}",
        f"SAR without usable ID: {spatial_sar_count}",
        "",
        "VALIDATION STATUS: " + status,
        "",
        "Errors:",
        *errors,
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nReport saved to: {REPORT}")

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
