"""Audit local note-source availability and link usable notes to the cohort."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import duckdb

DEFAULT_FILES_ROOT = Path("physionet.org/files")
DEFAULT_COHORT_DUCKDB_PATH = Path("derived_data/cohort/phase1_cohort.duckdb")
DEFAULT_OUTPUT_DIR = Path("derived_data/notes")
DEFAULT_REPORT_DIR = Path("reports/notes")

EXPECTED_NOTE_FILES = {
    "discharge": "discharge.csv.gz",
    "discharge_detail": "discharge_detail.csv.gz",
    "radiology": "radiology.csv.gz",
    "radiology_detail": "radiology_detail.csv.gz",
}

REQUIRED_RADIOLOGY_COLUMNS = {"subject_id", "hadm_id", "charttime", "text"}
REQUIRED_DISCHARGE_COLUMNS = {"subject_id", "hadm_id", "charttime", "text"}


@dataclass(frozen=True)
class NoteAuditConfig:
    """Configuration for the note-source audit pipeline."""

    files_root: Path
    cohort_duckdb_path: Path
    output_dir: Path
    report_dir: Path


@dataclass(frozen=True)
class NoteFileInfo:
    """Metadata about one discovered note-domain file."""

    file_key: str
    path: str
    columns: list[str]
    row_linkage_mode: str


@dataclass(frozen=True)
class NoteAuditSummary:
    """Summary artifact describing available note sources and cohort linkage."""

    generated_at_utc: str
    files_root: str
    cohort_duckdb_path: str
    note_dataset_status: str
    available_note_domains: list[str]
    discovered_note_files: list[dict[str, object]]
    radiology_candidate_note_count: int
    radiology_sampled_note_count: int
    radiology_sampled_stay_count: int
    discharge_candidate_note_count: int
    discharge_sampled_note_count: int
    discharge_sampled_stay_count: int
    recommendation: str
    report_path: str


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the note-source audit."""

    parser = argparse.ArgumentParser(
        description="Audit local note-source availability and link usable notes to the existing cohort."
    )
    parser.add_argument(
        "--files-root",
        type=Path,
        default=DEFAULT_FILES_ROOT,
        help="Path to the local PhysioNet files root.",
    )
    parser.add_argument(
        "--cohort-duckdb-path",
        type=Path,
        default=DEFAULT_COHORT_DUCKDB_PATH,
        help="Path to the Phase 1 cohort DuckDB database.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for derived note-audit outputs.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Directory for note-audit markdown reports.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> NoteAuditConfig:
    """Resolve CLI arguments into a normalized note-audit configuration."""

    return NoteAuditConfig(
        files_root=args.files_root.resolve(),
        cohort_duckdb_path=args.cohort_duckdb_path.resolve(),
        output_dir=args.output_dir.resolve(),
        report_dir=args.report_dir.resolve(),
    )


def sql_string(value: str) -> str:
    """Quote a Python string for safe interpolation into DuckDB SQL."""

    return "'" + value.replace("'", "''") + "'"


def ensure_cohort_database(config: NoteAuditConfig) -> None:
    """Validate that the Phase 1 cohort DuckDB artifact already exists."""

    if not config.cohort_duckdb_path.exists():
        raise FileNotFoundError(
            f"Missing cohort database: {config.cohort_duckdb_path}\n"
            "Run `python build_initial_cohort.py` first."
        )


def discover_note_files(files_root: Path) -> dict[str, Path]:
    """Discover MIMIC-IV-Note-style CSV files anywhere under the PhysioNet root."""

    discovered: dict[str, Path] = {}
    for path in files_root.rglob("*.csv.gz"):
        for file_key, filename in EXPECTED_NOTE_FILES.items():
            if path.name == filename and file_key not in discovered:
                discovered[file_key] = path
    return discovered


def read_gzip_header(path: Path) -> list[str]:
    """Read the header row from a gzipped CSV file without loading the full table."""

    with gzip.open(path, "rt", newline="") as file:
        reader = csv.reader(file)
        return next(reader)


def build_file_info(file_key: str, path: Path) -> NoteFileInfo:
    """Collect minimal metadata about a discovered note-domain file."""

    columns = read_gzip_header(path)
    linkage_mode = "detail_table" if file_key.endswith("_detail") else "note_table"
    return NoteFileInfo(
        file_key=file_key,
        path=str(path),
        columns=columns,
        row_linkage_mode=linkage_mode,
    )


def initialize_note_views(
    connection: duckdb.DuckDBPyConnection,
    note_files: dict[str, Path],
) -> dict[str, NoteFileInfo]:
    """Expose discovered note files as DuckDB views when their schemas are usable."""

    file_info = {file_key: build_file_info(file_key, path) for file_key, path in note_files.items()}

    if "radiology" in note_files:
        radiology_columns = set(file_info["radiology"].columns)
        if REQUIRED_RADIOLOGY_COLUMNS.issubset(radiology_columns):
            connection.execute(
                f"""
                CREATE OR REPLACE VIEW src_note_radiology AS
                SELECT *
                FROM read_csv_auto({sql_string(str(note_files["radiology"]))}, header = TRUE, all_varchar = TRUE);
                """
            )

    if "discharge" in note_files:
        discharge_columns = set(file_info["discharge"].columns)
        if REQUIRED_DISCHARGE_COLUMNS.issubset(discharge_columns):
            connection.execute(
                f"""
                CREATE OR REPLACE VIEW src_note_discharge AS
                SELECT *
                FROM read_csv_auto({sql_string(str(note_files["discharge"]))}, header = TRUE, all_varchar = TRUE);
                """
            )

    return file_info


def materialize_note_tables(connection: duckdb.DuckDBPyConnection, file_info: dict[str, NoteFileInfo]) -> None:
    """Create linked note tables for currently usable note domains."""

    if "radiology" in file_info and REQUIRED_RADIOLOGY_COLUMNS.issubset(set(file_info["radiology"].columns)):
        connection.execute(
            """
            CREATE OR REPLACE TABLE radiology_notes_candidate_window AS
            SELECT
                cohort.subject_id,
                cohort.hadm_id,
                cohort.stay_id,
                radiology.note_id,
                TRY_CAST(radiology.charttime AS TIMESTAMP) AS note_charttime,
                TRY_CAST(radiology.storetime AS TIMESTAMP) AS note_storetime,
                radiology.text
            FROM candidate_sepsis_cohort AS cohort
            INNER JOIN src_note_radiology AS radiology
                ON TRY_CAST(radiology.subject_id AS BIGINT) = cohort.subject_id
               AND TRY_CAST(radiology.hadm_id AS BIGINT) = cohort.hadm_id
            WHERE TRY_CAST(radiology.charttime AS TIMESTAMP)
                  BETWEEN cohort.study_window_start AND cohort.study_window_end;
            """
        )
        connection.execute(
            """
            CREATE OR REPLACE TABLE radiology_notes_sampled_window AS
            SELECT
                sampled.*,
                notes.note_id,
                notes.note_charttime,
                notes.note_storetime,
                notes.text
            FROM sampled_candidate_cohort AS sampled
            INNER JOIN radiology_notes_candidate_window AS notes
                ON sampled.stay_id = notes.stay_id;
            """
        )

    if "discharge" in file_info and REQUIRED_DISCHARGE_COLUMNS.issubset(set(file_info["discharge"].columns)):
        # Discharge summaries are linked for auditing only. They are excluded
        # from the main extraction workflow because they summarize the hospital
        # course and therefore conflict with the pre-decision note rule.
        connection.execute(
            """
            CREATE OR REPLACE TABLE discharge_notes_candidate_linked AS
            SELECT
                cohort.subject_id,
                cohort.hadm_id,
                cohort.stay_id,
                discharge.note_id,
                TRY_CAST(discharge.charttime AS TIMESTAMP) AS note_charttime,
                TRY_CAST(discharge.storetime AS TIMESTAMP) AS note_storetime,
                discharge.text
            FROM candidate_sepsis_cohort AS cohort
            INNER JOIN src_note_discharge AS discharge
                ON TRY_CAST(discharge.subject_id AS BIGINT) = cohort.subject_id
               AND TRY_CAST(discharge.hadm_id AS BIGINT) = cohort.hadm_id;
            """
        )
        connection.execute(
            """
            CREATE OR REPLACE TABLE discharge_notes_sampled_linked AS
            SELECT
                sampled.*,
                notes.note_id,
                notes.note_charttime,
                notes.note_storetime,
                notes.text
            FROM sampled_candidate_cohort AS sampled
            INNER JOIN discharge_notes_candidate_linked AS notes
                ON sampled.stay_id = notes.stay_id;
            """
        )


def count_if_exists(connection: duckdb.DuckDBPyConnection, table_name: str) -> int:
    """Return a table row count if the table exists, otherwise zero."""

    exists = connection.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE lower(table_name) = lower(?);
        """,
        [table_name],
    ).fetchone()[0]
    if not exists:
        return 0
    return connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]


def distinct_stays_if_exists(connection: duckdb.DuckDBPyConnection, table_name: str) -> int:
    """Return the number of distinct stays in a table if it exists, otherwise zero."""

    exists = connection.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE lower(table_name) = lower(?);
        """,
        [table_name],
    ).fetchone()[0]
    if not exists:
        return 0
    return connection.execute(f"SELECT COUNT(DISTINCT stay_id) FROM {table_name}").fetchone()[0]


def determine_recommendation(note_files: dict[str, Path]) -> str:
    """Produce a human-readable next-step recommendation from the note audit."""

    if not note_files:
        return (
            "Download MIMIC-IV-Note or provide another local note source before "
            "implementing note extraction. The current cohort step is complete, "
            "but note filtering remains blocked."
        )

    if "radiology" in note_files and "discharge" in note_files:
        return (
            "Radiology and discharge notes are available for audit, but the open "
            "MIMIC-IV-Note module still does not provide physician or nursing "
            "notes. Use radiology only as a secondary note source unless the "
            "study design is revised."
        )

    return (
        "A partial note source is available locally. Validate whether it meets "
        "the study's pre-decision note requirements before using it for extraction."
    )


def collect_summary(
    connection: duckdb.DuckDBPyConnection,
    config: NoteAuditConfig,
    note_files: dict[str, Path],
    file_info: dict[str, NoteFileInfo],
    report_path: Path,
) -> NoteAuditSummary:
    """Assemble the note-audit summary for JSON and markdown export."""

    note_status = "missing" if not note_files else "available_but_design_review_required"
    return NoteAuditSummary(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        files_root=str(config.files_root),
        cohort_duckdb_path=str(config.cohort_duckdb_path),
        note_dataset_status=note_status,
        available_note_domains=sorted(note_files.keys()),
        discovered_note_files=[asdict(info) for info in file_info.values()],
        radiology_candidate_note_count=count_if_exists(connection, "radiology_notes_candidate_window"),
        radiology_sampled_note_count=count_if_exists(connection, "radiology_notes_sampled_window"),
        radiology_sampled_stay_count=distinct_stays_if_exists(connection, "radiology_notes_sampled_window"),
        discharge_candidate_note_count=count_if_exists(connection, "discharge_notes_candidate_linked"),
        discharge_sampled_note_count=count_if_exists(connection, "discharge_notes_sampled_linked"),
        discharge_sampled_stay_count=distinct_stays_if_exists(connection, "discharge_notes_sampled_linked"),
        recommendation=determine_recommendation(note_files),
        report_path=str(report_path),
    )


def export_parquet_if_exists(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
    output_path: Path,
) -> None:
    """Export a DuckDB table to Parquet only when the table exists."""

    exists = connection.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE lower(table_name) = lower(?);
        """,
        [table_name],
    ).fetchone()[0]
    if not exists:
        return
    connection.execute(
        f"COPY {table_name} TO {sql_string(str(output_path))} (FORMAT PARQUET, COMPRESSION ZSTD);"
    )


def write_outputs(
    connection: duckdb.DuckDBPyConnection,
    config: NoteAuditConfig,
    summary: NoteAuditSummary,
    report_path: Path,
) -> None:
    """Persist the note-audit summary and any usable linked-note extracts."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.report_dir.mkdir(parents=True, exist_ok=True)

    export_parquet_if_exists(
        connection,
        "radiology_notes_sampled_window",
        config.output_dir / "radiology_notes_sampled_window.parquet",
    )
    export_parquet_if_exists(
        connection,
        "discharge_notes_sampled_linked",
        config.output_dir / "discharge_notes_sampled_linked.parquet",
    )

    summary_json_path = config.output_dir / "note_source_audit_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as file:
        json.dump(asdict(summary), file, indent=2)

    note_domains = ", ".join(summary.available_note_domains) if summary.available_note_domains else "none"
    report_lines = [
        "# Note Source Audit Report",
        f"Date: {summary.generated_at_utc}",
        "",
        "## Inputs",
        f"- files_root: `{summary.files_root}`",
        f"- cohort_duckdb_path: `{summary.cohort_duckdb_path}`",
        "",
        "## Availability",
        f"- note_dataset_status: {summary.note_dataset_status}",
        f"- available_note_domains: {note_domains}",
        "",
        "## Cohort Linkage",
        f"- radiology_candidate_note_count: {summary.radiology_candidate_note_count}",
        f"- radiology_sampled_note_count: {summary.radiology_sampled_note_count}",
        f"- radiology_sampled_stay_count: {summary.radiology_sampled_stay_count}",
        f"- discharge_candidate_note_count: {summary.discharge_candidate_note_count}",
        f"- discharge_sampled_note_count: {summary.discharge_sampled_note_count}",
        f"- discharge_sampled_stay_count: {summary.discharge_sampled_stay_count}",
        "",
        "## Recommendation",
        f"- {summary.recommendation}",
        "",
        "## Outputs",
        f"- `{summary_json_path}`",
        f"- `{report_path}`",
    ]
    with open(report_path, "w", encoding="utf-8") as file:
        file.write("\n".join(report_lines) + "\n")


def run_audit(config: NoteAuditConfig) -> NoteAuditSummary:
    """Execute the note-source audit against the existing cohort database."""

    ensure_cohort_database(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.report_dir.mkdir(parents=True, exist_ok=True)

    note_files = discover_note_files(config.files_root)
    report_path = config.report_dir / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_note_audit_report.md"

    connection = duckdb.connect(str(config.cohort_duckdb_path))
    try:
        file_info = initialize_note_views(connection, note_files)
        materialize_note_tables(connection, file_info)
        summary = collect_summary(connection, config, note_files, file_info, report_path)
        write_outputs(connection, config, summary, report_path)
        return summary
    finally:
        connection.close()


def main() -> int:
    """CLI entrypoint for the note-source audit."""

    config = build_config(parse_args())
    summary = run_audit(config)

    print("Note source audit completed.")
    print(f"Note dataset status: {summary.note_dataset_status}")
    print(f"Available note domains: {', '.join(summary.available_note_domains) or 'none'}")
    print(f"Recommendation: {summary.recommendation}")
    print(f"Report: {summary.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
