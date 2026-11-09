# Maritime-anomaly-detection
# Scalable Multimodal Maritime Anomaly Detection
## Data Preparation, Hybrid HPC Processing and Dataset Handover

### 1. Project Overview

This project implements a scalable data-processing pipeline for integrating maritime vessel observations from multiple sources using High-Performance Computing (HPC) techniques.

The pipeline combines:

- Automatic Identification System (AIS) vessel-presence data.
- Synthetic Aperture Radar (SAR) vessel detections.
- Fishing activity events.
- Loitering events.
- AIS gap events.
- Encounter events.

The primary computational task is matching SAR detections with AIS observations using spatial and temporal information.

The implementation uses a hybrid parallel programming approach combining:

- **MPI (Message Passing Interface):** distributes regional processing across multiple processes.
- **OpenMP:** parallelizes matching operations within each MPI process.

The final output is a structured, integrated dataset intended for downstream maritime behavioral anomaly detection.

**Project scope:** This repository covers data auditing, normalization, preparation, HPC-based SAR–AIS matching, result integration, and validation. Machine-learning model development is a separate downstream responsibility.

---

### 2. Study Regions

The pipeline processes four maritime regions:

| Region | Description |
|---|---|
| Argentina | Argentina maritime region |
| Ecuador/Galápagos | Ecuador and Galápagos maritime region |
| Ghana | Ghana maritime region |
| Senegal | Senegal maritime region |

The processing is organized by region to enable independent parallel execution and prevent cross-region matching.

### 3. Source Datasets

The project uses six datasets obtained from Global Fishing Watch.

| Dataset | Records |
|---|---:|
| AIS presence | 2,342,734 |
| SAR presence | 11,926 |
| Fishing events | 93,650 |
| Loitering events | 47,565 |
| Gap events | 1,904 |
| Encounter events | 304 |
| **Total** | **2,498,083** |

The original source data are retained separately from the normalized and processed datasets.

### 4. Project Directory Structure

The main project directory is organized as follows:

```text
Maritime-anomaly-detection/
│
├── data/
│   ├── raw/
│   │
│   ├── processed/
│   │
│   ├── processed_v2/
│   │   ├── ais_presence/
│   │   ├── sar_presence/
│   │   ├── fishing_events/
│   │   ├── loitering_events/
│   │   ├── gap_events/
│   │   └── encounter_events/
│   │
│   ├── stage4_input/
│   │   ├── ais_presence/
│   │   ├── sar_presence/
│   │   └── vessel_dictionary.csv
│   │
│   ├── stage4_output/
│   │
│   ├── stage5_results/
│   │   ├── sar_ais_matches.parquet
│   │   ├── match_event_links.parquet
│   │   └── event_enrichment_summary.csv
│   │
│   └── stage4_validation_report.txt
│
├── src/
│   ├── stage2_normalization.py
│   ├── stage4_prepare_inputs.py
│   ├── stage4_hybrid_match.c
│   ├── stage4_validate_matches.py
│   └── stage5_integrate_results.py
│
└── README.md
```

Additional audit and validation scripts may also be present in the `src` directory.

The original raw and processed datasets are preserved and should not be overwritten.

---

### 5. Processing Pipeline

The project consists of five main stages.

#### Stage 1: Raw Data Auditing

The original datasets were inspected to establish:

- Record counts.
- Dataset schemas.
- Unique vessel identifiers.
- Geographic coverage.
- Missing values.
- Data consistency across regions.

The audit covered 2,498,083 records.

#### Stage 2: Data Normalization

Source JSON data were converted into structured Parquet datasets.

Processing included:

- Extraction of relevant vessel attributes.
- Coordinate and date normalization.
- Extraction of event locations.
- Handling of nested JSON attributes.
- Region-based and monthly partitioning.
- Preservation of source-file metadata.

The normalized datasets are stored in:

`data/processed_v2/`

#### Stage 3: Independent Dataset Validation

The normalized datasets were independently validated.

| Validation item | Result |
|---|---:|
| Regions | 4 |
| Datasets | 6 |
| Parquet files | 314 |
| Expected records | 2,498,083 |
| Readable records | 2,498,083 |
| Read errors | 0 |

All expected records were successfully recovered from the normalized Parquet datasets.

The validation also checked schemas, required fields, vessel identifiers, and coordinate validity.

**Result: All Stage 3 validation checks passed.**

#### Stage 4: Hybrid MPI + OpenMP SAR–AIS Matching

This stage performs the main computational operation of the project.

**Input preparation**

AIS and SAR records were converted into compact binary records for efficient processing.

Each binary record occupies 36 bytes and contains:

| Field | Data type |
|---|---|
| Record ID | uint64 |
| Day | int32 |
| Vessel code | int64 |
| Latitude | float64 |
| Longitude | float64 |

The prepared files are stored in:

`data/stage4_input/`

Metadata are maintained separately in Parquet files, allowing the C program to operate on compact binary records and Python to perform subsequent metadata integration.

**Parallel processing architecture**

The implementation uses four MPI processes and two OpenMP threads per process in the completed execution.

| MPI rank | Assigned region |
|---|---|
| Rank 0 | Argentina |
| Rank 1 | Ecuador/Galápagos |
| Rank 2 | Ghana |
| Rank 3 | Senegal |

Each MPI process independently processes its assigned region.

OpenMP parallelizes SAR record processing within each MPI process.

**Matching methodology**

Two matching methods are implemented.

1. Same-vessel-ID matching:
   - Searches for AIS records with the same internal vessel identifier.
   - Requires the same region and calendar date.
   - Selects the nearest AIS observation.
   - Does not impose a distance cutoff on this identifier-based matching path.

2. Spatial-temporal matching:
   - Used when the SAR vessel identifier is missing.
   - Searches AIS observations from the same region and date.
   - Calculates geographic distance using the Haversine formula.
   - Retains all candidates within 25 km.
   - Allows multiple AIS candidates for a single SAR detection.

When no candidate is available, the SAR detection is retained as unmatched.

The 25 km threshold is a provisional candidate-search threshold, not a general vessel-position accuracy threshold.

**Important interpretation:** A same-vessel-ID association is not proof of positional identity. A spatial-temporal association is a candidate vessel association, not confirmed vessel identity.

**Execution configuration**

The completed implementation was compiled using GCC and Open MPI in WSL2 Ubuntu.

Compilation:

```bash
mpicc -O3 -fopenmp \
src/stage4_hybrid_match.c \
-lm -o src/stage4_hybrid_match
```

Execution:

```bash
OMP_NUM_THREADS=2 OMP_DYNAMIC=FALSE \
mpirun --oversubscribe -np 4 \
./src/stage4_hybrid_match \
./data/stage4_input \
./data/stage4_output
```

The output directory must exist before execution.

**Completed Stage 4 results**

| Region | AIS records | SAR records | Output rows |
|---|---:|---:|---:|
| Argentina | 1,101,669 | 4,089 | 32,529 |
| Ecuador/Galápagos | 465,977 | 804 | 8,772 |
| Ghana | 323,192 | 2,336 | 28,807 |
| Senegal | 451,896 | 4,697 | 62,629 |
| **Total** | **2,342,734** | **11,926** | **132,737** |

The matching computation completed successfully with four MPI ranks and two OpenMP threads per rank.

**Independent Stage 4 validation**

The matching results were checked against the source metadata and independently recalculated geographic distances.

Validation covered:

- SAR record references.
- AIS record references.
- Duplicate SAR–AIS pairs.
- Same-region and same-date constraints.
- Nearest-candidate selection for identifier-based matches.
- All qualifying candidates within 25 km for missing-ID matches.
- Haversine distance calculations.
- Correct unmatched-record handling.

All Stage 4 validation checks passed.

#### Stage 5: Result Integration and Event Enrichment

Stage 5 combines the HPC matching output with SAR and AIS metadata.

The matching results are enriched with:

- SAR vessel identifiers.
- AIS vessel identifiers.
- Coordinates.
- Dates.
- Regions.
- H3 spatial cells.
- Pull identifiers.
- Source-file information.

The consolidated matching dataset contains 132,737 rows.

**Event enrichment**

The four contextual event datasets are associated separately with the AIS candidate vessel.

The association uses the internal Global Fishing Watch vessel identifier, extracted from the event's `vessel.id`.

For encounter events, both primary and secondary participant identifiers are considered.

The SAR observation date must fall within the event's start and end dates, inclusive.

Event records are kept in a separate association table to prevent event enrichment from multiplying the consolidated matching rows.

Events provide contextual information only. They do not determine whether a SAR–AIS match exists.

---

### 6. Final Output Datasets

All final outputs are stored in:

`data/stage5_results/`

#### 6.1 sar_ais_matches.parquet

This is the primary integrated dataset.

**Records:** 132,737

**Columns:**

| Column | Description |
|---|---|
| sar_record_id | SAR record identifier |
| ais_record_id | AIS record identifier; zero is used as the unmatched sentinel |
| distance_km | Distance between SAR and AIS coordinates |
| match_method | Matching method |
| match_status | Matched or unmatched status |
| sar_date | SAR observation date |
| sar_vessel_id | SAR internal vessel identifier |
| sar_lat | SAR latitude |
| sar_lon | SAR longitude |
| sar_region | SAR region |
| sar_h3_cell | SAR H3 spatial cell |
| sar_pull_id | SAR data pull identifier |
| sar_source_file | SAR source file |
| ais_date | AIS observation date |
| ais_vessel_id | AIS internal vessel identifier |
| ais_lat | AIS latitude |
| ais_lon | AIS longitude |
| ais_region | AIS region |
| ais_h3_cell | AIS H3 spatial cell |
| ais_pull_id | AIS data pull identifier |
| ais_source_file | AIS source file |

**Matching method values:**

- `same_vessel_id`
- `spatiotemporal`

**Matching status values:**

- `matched`
- `unmatched`

For unmatched records, the AIS record ID uses zero as a sentinel. The corresponding AIS fields must not be interpreted as an actual matched AIS observation.

#### 6.2 match_event_links.parquet

This dataset stores associations between matched SAR–AIS pairs and contextual maritime events.

**Records:** 119,630

| Column | Description |
|---|---|
| sar_record_id | SAR record identifier |
| ais_record_id | Associated AIS record identifier |
| sar_region | SAR region |
| sar_date | SAR observation date |
| ais_vessel_id | Internal AIS vessel identifier |
| event_dataset | Source event dataset |
| event_id | Event record identifier |
| event_type | Event category |
| participant_role | Primary or secondary participant |
| participant_ssvid | Participant SSVID/MMSI |
| event_start | Event start timestamp |
| event_end | Event end timestamp |
| event_lat | Event latitude |
| event_lon | Event longitude |
| event_source_file | Event source file |

This table should be joined to the primary dataset using the SAR and AIS record identifiers.

An event association represents contextual overlap, not a confirmed anomaly.

#### 6.3 event_enrichment_summary.csv

This file provides aggregated event association counts.

| Event dataset | Event links | Unique SAR records | Unique AIS records | Unique events |
|---|---:|---:|---:|---:|
| Fishing | 97,308 | 3,933 | 15,067 | 7,561 |
| Loitering | 21,799 | 2,638 | 6,595 | 3,173 |
| Gap | 139 | 61 | 94 | 34 |
| Encounter | 384 | 22 | 56 | 16 |
| **Total** | **119,630** | — | — | — |

Unique counts are calculated separately for each event category and should not be summed to infer globally unique vessels or detections.

---

### 7. Final Independent Validation

The integrated datasets underwent a separate validation after Stage 5 completed.

| Validation check | Result |
|---|---|
| Matching row count | PASS |
| Unique SAR record count | PASS |
| Matched record count | PASS |
| Unmatched record count | PASS |
| Event link count | PASS |
| Matched AIS metadata completeness | PASS |
| No cross-region matches | PASS |
| No duplicate SAR–AIS pairs | PASS |
| Event links reference valid matched pairs | PASS |
| Event dates within event intervals | PASS |
| All four event datasets represented | PASS |

**Final validation: 11 out of 11 checks passed.**

Final counts:

| Metric | Result |
|---|---:|
| Consolidated matching rows | 132,737 |
| Unique SAR records | 11,926 |
| Matched rows | 132,015 |
| Unmatched records | 722 |
| Event links | 119,630 |
| Unique event-linked SAR–AIS pairs | 45,996 |

All unmatched records were retained in the primary integrated dataset.

The independent validation found no invalid event references or event date interval violations.

---

### 8. Known Limitations and Interpretation

The following points should be considered when using the datasets.

1. **Vessel identity:** Same-vessel-ID matching is an identifier-based association and does not independently prove that the two observations represent the same physical vessel position.

2. **Spatial matching:** Missing-ID spatial matches are candidate associations within a provisional 25 km threshold. Multiple candidates may exist for one SAR detection.

3. **Unmatched records:** Unmatched SAR records are retained. Their AIS record ID uses zero as a sentinel and must not be interpreted as an actual AIS record.

4. **SAR identifiers:** Some SAR records contain blank vessel identifiers. These records were not discarded and were processed through the spatial-temporal matching method.

5. **Event associations:** Fishing, loitering, gap, and encounter events provide contextual information. They are not ground-truth anomaly or illegal-fishing labels.

6. **Encounter records:** Encounter event records and participant roles are retained as supplied by the source. Reciprocal records should not automatically be counted as unique physical encounters.

7. **Temporal matching:** Matching uses the same calendar date within each region. Event enrichment uses inclusive UTC calendar-date intervals.

8. **Data coverage:** Results reflect the available source datasets and their temporal and geographic coverage. They should not be interpreted as exhaustive observations of maritime activity.

9. **Anomaly interpretation:** The presence of an event or a SAR–AIS association does not establish illegal fishing or any other unlawful activity.

---

### 9. Handover Instructions for Downstream Model Development

The final integrated datasets are intended for use by the downstream anomaly-detection development team.

Recommended starting points:

- Use `sar_ais_matches.parquet` as the primary integrated observation table.
- Use `match_status` to distinguish matched and unmatched records.
- Use `match_method` to distinguish identifier-based and spatial-temporal associations.
- Use `distance_km` as a matching-distance attribute, with care when interpreting identifier-based matches.
- Join `match_event_links.parquet` when contextual event information is needed.
- Use `event_enrichment_summary.csv` for event-level descriptive summaries.

The downstream team should independently decide how to handle unmatched observations, multiple candidate associations, event duplication, missing vessel identifiers, and other data-quality considerations.

No anomaly labels or verified illegal-fishing labels are supplied by this data-preparation pipeline.

---

### 10. Software and Technical Environment

The completed pipeline was developed and executed using:

| Component | Environment |
|---|---|
| Operating system | Windows 11 with WSL2 Ubuntu |
| Python | Python virtual environment |
| pandas | 2.3.3 |
| PyArrow | 25.0.1 |
| NumPy | 2.5.3 |
| C compiler | GCC 13.3.0 |
| MPI implementation | Open MPI 4.1.6 |
| Parallel programming | MPI + OpenMP |
| Storage format | Apache Parquet and binary records |

The project uses Python for data processing and integration, and C with MPI and OpenMP for the parallel matching computation.

---

### 11. Reproducibility and Data Preservation

The completed outputs are the validated handover artifacts.

For reproducibility:

- Preserve the original source datasets.
- Preserve the normalized `processed_v2` datasets.
- Preserve the Stage 4 binary inputs and metadata.
- Preserve the Stage 4 matching outputs and validation report.
- Preserve the Stage 5 integrated Parquet datasets and event summary.
- Do not overwrite validated outputs without first retaining a backup.

The matching executable can be rebuilt using the compilation command documented in Stage 4. Full pipeline regeneration requires the original source datasets and the corresponding processing scripts.

---

### 12. Final Project Status

| Project stage | Status |
|---|---|
| Raw data auditing | Completed |
| Data normalization | Completed |
| Independent Parquet validation | Passed |
| Stage 4 input preparation | Completed |
| Hybrid MPI + OpenMP implementation | Completed |
| SAR–AIS matching execution | Completed |
| Stage 4 independent validation | Passed |
| Stage 5 result integration | Completed |
| Event enrichment | Completed |
| Stage 5 independent validation | Passed |
| Final dataset handover | Ready |

**Final outcome**

A validated multimodal maritime dataset has been prepared by integrating AIS vessel observations, SAR detections, and contextual maritime events across four study regions.

The core SAR–AIS matching operation was executed using a hybrid MPI + OpenMP implementation. The resulting datasets passed all independent validation checks and are ready for downstream behavioral anomaly-detection research.

**End of README.**