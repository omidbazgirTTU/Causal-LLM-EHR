"""Build the initial MIMIC-IV ICU cohort required before any LLM runs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import duckdb

DEFAULT_MIMIC_ROOT = Path("physionet.org/files/mimiciv/3.1")
DEFAULT_OUTPUT_DIR = Path("derived_data/cohort")
DEFAULT_REPORT_DIR = Path("reports/cohort")
DEFAULT_TARGET_SIZE = 1000

REQUIRED_MIMIC_FILES = {
    "patients": "hosp/patients.csv.gz",
    "admissions": "hosp/admissions.csv.gz",
    "prescriptions": "hosp/prescriptions.csv.gz",
    "labevents": "hosp/labevents.csv.gz",
    "d_labitems": "hosp/d_labitems.csv.gz",
    "microbiologyevents": "hosp/microbiologyevents.csv.gz",
    "icustays": "icu/icustays.csv.gz",
}

# This is intentionally broad because the cohort step is a loose candidate
# filter, not the final exposure definition.
ANTIBIOTIC_DRUG_REGEX = (
    r"(amoxicillin|ampicillin|penicillin|piperacillin|tazobactam|nafcillin|oxacillin|dicloxacillin|"
    r"cef[a-z]+|carbapenem|meropenem|ertapenem|imipenem|aztreonam|vancomycin|daptomycin|linezolid|"
    r"clindamycin|metronidazole|doxycycline|minocycline|tigecycline|tetracycline|azithromycin|"
    r"erythromycin|clarithromycin|ciprofloxacin|levofloxacin|moxifloxacin|gentamicin|tobramycin|"
    r"amikacin|fosfomycin|trimethoprim|sulfamethoxazole|bactrim)"
)
TOPICAL_DRUG_REGEX = r"(ophth|otic|topical|vaginal)"
NON_SYSTEMIC_ROUTE_REGEX = r"(^|[^a-z])(od|os|ou|ad|as|au)([^a-z]|$)|ophth|otic|topical|vaginal"


@dataclass(frozen=True)
class CohortBuildConfig:
    """Configuration for the initial ICU cohort build."""

    mimic_root: Path
    output_dir: Path
    report_dir: Path
    target_size: int
    duckdb_path: Path


@dataclass(frozen=True)
class CohortBuildSummary:
    """Compact summary emitted after a cohort build run."""

    generated_at_utc: str
    mimic_root: str
    duckdb_path: str
    output_dir: str
    report_path: str
    note_dataset_status: str
    note_dataset_roots: list[str]
    total_icu_stays: int
    adult_first_icu_stays: int
    antibiotic_signal_stays: int
    lactate_signal_stays: int
    blood_culture_signal_stays: int
    candidate_cohort_size: int
    sampled_cohort_size: int
    target_size: int


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the Phase 1 cohort builder."""

    parser = argparse.ArgumentParser(
        description="Build the initial adult first-ICU candidate cohort from local MIMIC-IV files."
    )
    parser.add_argument(
        "--mimic-root",
        type=Path,
        default=DEFAULT_MIMIC_ROOT,
        help="Path to the local MIMIC-IV v3.1 root directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for cohort build outputs.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Directory for markdown phase reports.",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=DEFAULT_TARGET_SIZE,
        help="Maximum number of stays to include in the reproducible sampled cohort.",
    )
    parser.add_argument(
        "--duckdb-path",
        type=Path,
        default=None,
        help="Optional explicit DuckDB file path. Defaults to <output-dir>/phase1_cohort.duckdb.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> CohortBuildConfig:
    """Resolve CLI arguments into a normalized cohort-build configuration."""

    if args.target_size <= 0:
        raise ValueError("--target-size must be a positive integer.")

    output_dir = args.output_dir.resolve()
    report_dir = args.report_dir.resolve()
    duckdb_path = args.duckdb_path.resolve() if args.duckdb_path else output_dir / "phase1_cohort.duckdb"
    return CohortBuildConfig(
        mimic_root=args.mimic_root.resolve(),
        output_dir=output_dir,
        report_dir=report_dir,
        target_size=args.target_size,
        duckdb_path=duckdb_path,
    )


def sql_string(value: str) -> str:
    """Quote a Python string for safe interpolation into DuckDB SQL."""

    return "'" + value.replace("'", "''") + "'"


def ensure_required_inputs(config: CohortBuildConfig) -> dict[str, Path]:
    """Validate that the required MIMIC-IV files are present locally."""

    resolved_paths: dict[str, Path] = {}
    missing_paths: list[Path] = []

    for name, relative_path in REQUIRED_MIMIC_FILES.items():
        absolute_path = config.mimic_root / relative_path
        resolved_paths[name] = absolute_path
        if not absolute_path.exists():
            missing_paths.append(absolute_path)

    if missing_paths:
        missing_text = "\n".join(f"- {path}" for path in missing_paths)
        raise FileNotFoundError(f"Missing required MIMIC-IV inputs:\n{missing_text}")

    return resolved_paths


def discover_note_roots(files_root: Path) -> list[Path]:
    """Find local note dataset directories if they exist alongside the MIMIC files."""

    note_roots: list[Path] = []
    for path in files_root.rglob("*"):
        if path.is_dir() and "note" in path.name.lower():
            note_roots.append(path)
    return sorted(set(note_roots))


def initialize_source_views(connection: duckdb.DuckDBPyConnection, required_paths: dict[str, Path]) -> None:
    """Expose the raw compressed CSV tables as queryable DuckDB views."""

    for name, path in required_paths.items():
        connection.execute(
            f"""
            CREATE OR REPLACE VIEW src_{name} AS
            SELECT *
            FROM read_csv_auto({sql_string(str(path))}, header = TRUE, all_varchar = TRUE);
            """
        )


def build_phase1_tables(connection: duckdb.DuckDBPyConnection) -> None:
    """Materialize the core cohort tables needed for the first implementation phase."""

    connection.execute(
        """
        CREATE OR REPLACE TABLE adult_first_icu AS
        WITH ranked_icu AS (
            SELECT
                TRY_CAST(icu.subject_id AS BIGINT) AS subject_id,
                TRY_CAST(icu.hadm_id AS BIGINT) AS hadm_id,
                TRY_CAST(icu.stay_id AS BIGINT) AS stay_id,
                TRY_CAST(pat.anchor_age AS INTEGER) AS anchor_age,
                TRY_CAST(icu.intime AS TIMESTAMP) AS icu_intime,
                TRY_CAST(icu.outtime AS TIMESTAMP) AS icu_outtime,
                TRY_CAST(adm.admittime AS TIMESTAMP) AS admittime,
                TRY_CAST(adm.dischtime AS TIMESTAMP) AS dischtime,
                ROW_NUMBER() OVER (
                    PARTITION BY TRY_CAST(icu.subject_id AS BIGINT)
                    ORDER BY TRY_CAST(icu.intime AS TIMESTAMP), TRY_CAST(icu.stay_id AS BIGINT)
                ) AS icu_rank
            FROM src_icustays AS icu
            INNER JOIN src_admissions AS adm
                ON icu.subject_id = adm.subject_id
               AND icu.hadm_id = adm.hadm_id
            INNER JOIN src_patients AS pat
                ON icu.subject_id = pat.subject_id
            WHERE TRY_CAST(pat.anchor_age AS INTEGER) >= 18
        )
        SELECT
            subject_id,
            hadm_id,
            stay_id,
            anchor_age,
            admittime,
            dischtime,
            icu_intime,
            icu_outtime,
            icu_intime - INTERVAL 24 HOUR AS study_window_start,
            icu_intime + INTERVAL 48 HOUR AS study_window_end
        FROM ranked_icu
        WHERE icu_rank = 1;
        """
    )

    connection.execute(
        """
        CREATE OR REPLACE TABLE lactate_itemids AS
        SELECT DISTINCT TRY_CAST(itemid AS BIGINT) AS itemid
        FROM src_d_labitems
        WHERE LOWER(COALESCE(label, '')) = 'lactate';
        """
    )

    connection.execute(
        f"""
        CREATE OR REPLACE TABLE cohort_signal_antibiotics AS
        SELECT
            cohort.stay_id,
            MIN(COALESCE(TRY_CAST(rx.starttime AS TIMESTAMP), TRY_CAST(rx.stoptime AS TIMESTAMP))) AS first_antibiotic_time,
            COUNT(*) AS antibiotic_event_count
        FROM adult_first_icu AS cohort
        INNER JOIN src_prescriptions AS rx
            ON TRY_CAST(rx.subject_id AS BIGINT) = cohort.subject_id
           AND TRY_CAST(rx.hadm_id AS BIGINT) = cohort.hadm_id
        WHERE COALESCE(TRY_CAST(rx.starttime AS TIMESTAMP), TRY_CAST(rx.stoptime AS TIMESTAMP))
              BETWEEN cohort.study_window_start AND cohort.study_window_end
          AND regexp_matches(LOWER(COALESCE(rx.drug, '')), {sql_string(ANTIBIOTIC_DRUG_REGEX)})
          AND NOT regexp_matches(LOWER(COALESCE(rx.drug, '')), {sql_string(TOPICAL_DRUG_REGEX)})
          AND NOT regexp_matches(LOWER(COALESCE(rx.route, '')), {sql_string(NON_SYSTEMIC_ROUTE_REGEX)})
        GROUP BY cohort.stay_id;
        """
    )

    connection.execute(
        """
        CREATE OR REPLACE TABLE cohort_signal_lactate AS
        SELECT
            cohort.stay_id,
            MIN(TRY_CAST(labs.charttime AS TIMESTAMP)) AS first_lactate_time,
            COUNT(*) AS lactate_event_count
        FROM adult_first_icu AS cohort
        INNER JOIN src_labevents AS labs
            ON TRY_CAST(labs.subject_id AS BIGINT) = cohort.subject_id
           AND TRY_CAST(labs.hadm_id AS BIGINT) = cohort.hadm_id
        INNER JOIN lactate_itemids AS lactate
            ON TRY_CAST(labs.itemid AS BIGINT) = lactate.itemid
        WHERE TRY_CAST(labs.charttime AS TIMESTAMP)
              BETWEEN cohort.study_window_start AND cohort.study_window_end
        GROUP BY cohort.stay_id;
        """
    )

    connection.execute(
        """
        CREATE OR REPLACE TABLE cohort_signal_blood_culture AS
        SELECT
            cohort.stay_id,
            MIN(COALESCE(TRY_CAST(micro.charttime AS TIMESTAMP), TRY_CAST(micro.chartdate AS TIMESTAMP))) AS first_blood_culture_time,
            COUNT(*) AS blood_culture_event_count
        FROM adult_first_icu AS cohort
        INNER JOIN src_microbiologyevents AS micro
            ON TRY_CAST(micro.subject_id AS BIGINT) = cohort.subject_id
           AND TRY_CAST(micro.hadm_id AS BIGINT) = cohort.hadm_id
        WHERE COALESCE(TRY_CAST(micro.charttime AS TIMESTAMP), TRY_CAST(micro.chartdate AS TIMESTAMP))
              BETWEEN cohort.study_window_start AND cohort.study_window_end
          AND (
              LOWER(COALESCE(micro.spec_type_desc, '')) LIKE '%blood culture%'
              OR LOWER(COALESCE(micro.test_name, '')) LIKE '%blood culture%'
          )
        GROUP BY cohort.stay_id;
        """
    )

    connection.execute(
        """
        CREATE OR REPLACE TABLE candidate_sepsis_cohort AS
        SELECT
            cohort.subject_id,
            cohort.hadm_id,
            cohort.stay_id,
            cohort.anchor_age,
            cohort.admittime,
            cohort.dischtime,
            cohort.icu_intime,
            cohort.icu_outtime,
            cohort.study_window_start,
            cohort.study_window_end,
            antibiotic.first_antibiotic_time,
            antibiotic.antibiotic_event_count,
            lactate.first_lactate_time,
            lactate.lactate_event_count,
            blood.first_blood_culture_time,
            blood.blood_culture_event_count,
            antibiotic.stay_id IS NOT NULL AS has_antibiotic_exposure,
            lactate.stay_id IS NOT NULL AS has_lactate_measurement,
            blood.stay_id IS NOT NULL AS has_blood_culture,
            CAST(antibiotic.stay_id IS NOT NULL AS INTEGER)
                + CAST(lactate.stay_id IS NOT NULL AS INTEGER)
                + CAST(blood.stay_id IS NOT NULL AS INTEGER) AS signal_count
        FROM adult_first_icu AS cohort
        LEFT JOIN cohort_signal_antibiotics AS antibiotic
            ON cohort.stay_id = antibiotic.stay_id
        LEFT JOIN cohort_signal_lactate AS lactate
            ON cohort.stay_id = lactate.stay_id
        LEFT JOIN cohort_signal_blood_culture AS blood
            ON cohort.stay_id = blood.stay_id
        WHERE antibiotic.stay_id IS NOT NULL
           OR lactate.stay_id IS NOT NULL
           OR blood.stay_id IS NOT NULL;
        """
    )


def materialize_sampled_cohort(connection: duckdb.DuckDBPyConnection, target_size: int) -> None:
    """Create a reproducible sampled subset for the initial annotation-scale cohort."""

    connection.execute(
        f"""
        CREATE OR REPLACE TABLE sampled_candidate_cohort AS
        SELECT *
        FROM candidate_sepsis_cohort
        ORDER BY md5(
            CAST(subject_id AS VARCHAR) || '-' ||
            CAST(hadm_id AS VARCHAR) || '-' ||
            CAST(stay_id AS VARCHAR)
        )
        LIMIT {target_size};
        """
    )


def collect_summary(
    connection: duckdb.DuckDBPyConnection,
    config: CohortBuildConfig,
    note_roots: list[Path],
    report_path: Path,
) -> CohortBuildSummary:
    """Compute the run summary that is emitted to JSON and markdown."""

    note_status = "missing" if not note_roots else "detected_but_not_yet_integrated"

    total_icu_stays = connection.execute("SELECT COUNT(*) FROM src_icustays").fetchone()[0]
    adult_first_icu_stays = connection.execute("SELECT COUNT(*) FROM adult_first_icu").fetchone()[0]
    antibiotic_signal_stays = connection.execute("SELECT COUNT(*) FROM cohort_signal_antibiotics").fetchone()[0]
    lactate_signal_stays = connection.execute("SELECT COUNT(*) FROM cohort_signal_lactate").fetchone()[0]
    blood_culture_signal_stays = connection.execute("SELECT COUNT(*) FROM cohort_signal_blood_culture").fetchone()[0]
    candidate_cohort_size = connection.execute("SELECT COUNT(*) FROM candidate_sepsis_cohort").fetchone()[0]
    sampled_cohort_size = connection.execute("SELECT COUNT(*) FROM sampled_candidate_cohort").fetchone()[0]

    return CohortBuildSummary(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        mimic_root=str(config.mimic_root),
        duckdb_path=str(config.duckdb_path),
        output_dir=str(config.output_dir),
        report_path=str(report_path),
        note_dataset_status=note_status,
        note_dataset_roots=[str(path) for path in note_roots],
        total_icu_stays=total_icu_stays,
        adult_first_icu_stays=adult_first_icu_stays,
        antibiotic_signal_stays=antibiotic_signal_stays,
        lactate_signal_stays=lactate_signal_stays,
        blood_culture_signal_stays=blood_culture_signal_stays,
        candidate_cohort_size=candidate_cohort_size,
        sampled_cohort_size=sampled_cohort_size,
        target_size=config.target_size,
    )


def write_outputs(
    connection: duckdb.DuckDBPyConnection,
    config: CohortBuildConfig,
    summary: CohortBuildSummary,
    report_path: Path,
) -> Path:
    """Persist cohort outputs and generate a markdown report for the build."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.report_dir.mkdir(parents=True, exist_ok=True)

    full_parquet = config.output_dir / "candidate_sepsis_cohort.parquet"
    sampled_parquet = config.output_dir / "sampled_candidate_cohort.parquet"
    summary_json = config.output_dir / "phase1_cohort_summary.json"
    connection.execute(
        f"COPY candidate_sepsis_cohort TO {sql_string(str(full_parquet))} (FORMAT PARQUET, COMPRESSION ZSTD);"
    )
    connection.execute(
        f"COPY sampled_candidate_cohort TO {sql_string(str(sampled_parquet))} (FORMAT PARQUET, COMPRESSION ZSTD);"
    )

    with open(summary_json, "w", encoding="utf-8") as file:
        json.dump(asdict(summary), file, indent=2)

    note_status_line = (
        "- note dataset not found locally; note filtering remains blocked"
        if summary.note_dataset_status == "missing"
        else "- note dataset detected locally, but note filtering is not implemented yet"
    )
    report_text = "\n".join(
        [
            "# Cohort Build Report",
            f"Date: {summary.generated_at_utc}",
            "",
            "## Inputs",
            f"- mimic_root: `{summary.mimic_root}`",
            f"- duckdb_path: `{summary.duckdb_path}`",
            f"- target_size: {summary.target_size}",
            "",
            "## Key Metrics",
            f"- total_icustays: {summary.total_icu_stays}",
            f"- adult_first_icu_stays: {summary.adult_first_icu_stays}",
            f"- antibiotic_signal_stays: {summary.antibiotic_signal_stays}",
            f"- lactate_signal_stays: {summary.lactate_signal_stays}",
            f"- blood_culture_signal_stays: {summary.blood_culture_signal_stays}",
            f"- candidate_cohort_size: {summary.candidate_cohort_size}",
            f"- sampled_cohort_size: {summary.sampled_cohort_size}",
            "",
            "## Sanity Checks",
            f"- required_mimic_tables_present: yes",
            f"- sampled_size_within_target: {'yes' if summary.sampled_cohort_size <= summary.target_size else 'no'}",
            f"- note_dataset_status: {summary.note_dataset_status}",
            note_status_line,
            "",
            "## Outputs",
            f"- `{full_parquet}`",
            f"- `{sampled_parquet}`",
            f"- `{summary_json}`",
        ]
    )
    with open(report_path, "w", encoding="utf-8") as file:
        file.write(report_text + "\n")

    return report_path


def run_build(config: CohortBuildConfig) -> CohortBuildSummary:
    """Execute the initial cohort build and return the final summary."""

    required_paths = ensure_required_inputs(config)
    note_roots = discover_note_roots(config.mimic_root.parent.parent)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.report_dir.mkdir(parents=True, exist_ok=True)

    connection = duckdb.connect(str(config.duckdb_path))
    try:
        connection.execute("PRAGMA threads=4;")
        initialize_source_views(connection, required_paths)
        build_phase1_tables(connection)
        materialize_sampled_cohort(connection, target_size=config.target_size)

        report_path = config.report_dir / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_cohort_report.md"
        summary = collect_summary(connection, config, note_roots, report_path)
        write_outputs(connection, config, summary, report_path)
        return summary
    finally:
        connection.close()


def main() -> int:
    """CLI entrypoint for the initial cohort builder."""

    config = build_config(parse_args())
    summary = run_build(config)

    print("Initial cohort build completed.")
    print(f"Candidate cohort size: {summary.candidate_cohort_size}")
    print(f"Sampled cohort size: {summary.sampled_cohort_size}")
    print(f"Note dataset status: {summary.note_dataset_status}")
    print(f"DuckDB output: {summary.duckdb_path}")
    print(f"Report: {summary.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
