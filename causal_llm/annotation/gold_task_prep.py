"""Prepare grouped note bundles for human gold-label annotation."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from itertools import groupby

import duckdb

DEFAULT_NOTES_PARQUET = Path("derived_data/notes/radiology_notes_sampled_window.parquet")
DEFAULT_OUTPUT_DIR = Path("derived_data/annotations")
DEFAULT_REPORT_DIR = Path("reports/annotations")
DEFAULT_MAX_NOTES_PER_STAY = 20
DEFAULT_MAX_CHARS_PER_STAY = 40000

ANNOTATION_TEMPLATE = {
    "suspected_sepsis": None,
    "suspicion_time": None,
    "infection_source": None,
    "evidence_span": None,
    "confidence": None,
}

REQUIRED_NOTE_COLUMNS = {"stay_id", "subject_id", "hadm_id", "note_charttime", "text"}


@dataclass(frozen=True)
class GoldTaskPrepConfig:
    """Configuration for grouped gold-label task preparation."""

    notes_parquet: Path
    output_dir: Path
    report_dir: Path
    max_notes_per_stay: int
    max_chars_per_stay: int


@dataclass(frozen=True)
class GoldTaskPrepSummary:
    """Summary emitted after generating grouped annotation tasks."""

    generated_at_utc: str
    notes_parquet: str
    output_dir: str
    report_path: str
    stay_count: int
    total_note_count: int
    task_count: int
    max_notes_per_stay: int
    max_chars_per_stay: int
    jsonl_path: str
    csv_index_path: str


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the gold-label task generator."""

    parser = argparse.ArgumentParser(
        description="Group linked notes into human-readable annotation tasks for gold-label creation."
    )
    parser.add_argument(
        "--notes-parquet",
        type=Path,
        default=DEFAULT_NOTES_PARQUET,
        help="Parquet file containing stay-linked notes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated annotation task artifacts.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Directory for annotation-prep markdown reports.",
    )
    parser.add_argument(
        "--max-notes-per-stay",
        type=int,
        default=DEFAULT_MAX_NOTES_PER_STAY,
        help="Maximum number of notes retained per stay.",
    )
    parser.add_argument(
        "--max-chars-per-stay",
        type=int,
        default=DEFAULT_MAX_CHARS_PER_STAY,
        help="Maximum number of note text characters retained per stay.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> GoldTaskPrepConfig:
    """Resolve CLI arguments into a normalized task-preparation config."""

    if args.max_notes_per_stay <= 0:
        raise ValueError("--max-notes-per-stay must be positive.")
    if args.max_chars_per_stay <= 0:
        raise ValueError("--max-chars-per-stay must be positive.")

    return GoldTaskPrepConfig(
        notes_parquet=args.notes_parquet.resolve(),
        output_dir=args.output_dir.resolve(),
        report_dir=args.report_dir.resolve(),
        max_notes_per_stay=args.max_notes_per_stay,
        max_chars_per_stay=args.max_chars_per_stay,
    )


def sql_string(value: str) -> str:
    """Quote a Python string for safe interpolation into DuckDB SQL."""

    return "'" + value.replace("'", "''") + "'"


def ensure_inputs(config: GoldTaskPrepConfig) -> None:
    """Validate that the linked-note parquet artifact already exists."""

    if not config.notes_parquet.exists():
        raise FileNotFoundError(
            f"Missing linked-note parquet: {config.notes_parquet}\n"
            "Run `python audit_note_sources.py` after note data is available, or "
            "provide an explicit `--notes-parquet` path."
        )


def ensure_required_columns(connection: duckdb.DuckDBPyConnection, parquet_path: Path) -> None:
    """Validate the note parquet schema before task generation."""

    columns = {
        row[0].lower()
        for row in connection.execute(
            f"DESCRIBE SELECT * FROM read_parquet({sql_string(str(parquet_path))})"
        ).fetchall()
    }
    missing = sorted(REQUIRED_NOTE_COLUMNS - columns)
    if missing:
        raise ValueError(
            "Linked-note parquet is missing required columns: " + ", ".join(missing)
        )


def initialize_note_table(connection: duckdb.DuckDBPyConnection, config: GoldTaskPrepConfig) -> None:
    """Load the linked-note parquet into DuckDB and rank notes within each stay."""

    connection.execute(
        f"""
        CREATE OR REPLACE VIEW src_linked_notes AS
        SELECT *
        FROM read_parquet({sql_string(str(config.notes_parquet))});
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TABLE ranked_linked_notes AS
        WITH base AS (
            SELECT
                TRY_CAST(subject_id AS BIGINT) AS subject_id,
                TRY_CAST(hadm_id AS BIGINT) AS hadm_id,
                TRY_CAST(stay_id AS BIGINT) AS stay_id,
                CAST(note_id AS VARCHAR) AS note_id,
                TRY_CAST(note_charttime AS TIMESTAMP) AS note_charttime,
                CAST(text AS VARCHAR) AS text,
                LENGTH(CAST(text AS VARCHAR)) AS note_char_count,
                ROW_NUMBER() OVER (
                    PARTITION BY TRY_CAST(stay_id AS BIGINT)
                    ORDER BY TRY_CAST(note_charttime AS TIMESTAMP), CAST(note_id AS VARCHAR)
                ) AS note_rank,
                SUM(LENGTH(CAST(text AS VARCHAR))) OVER (
                    PARTITION BY TRY_CAST(stay_id AS BIGINT)
                    ORDER BY TRY_CAST(note_charttime AS TIMESTAMP), CAST(note_id AS VARCHAR)
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS running_char_count
            FROM src_linked_notes
            WHERE stay_id IS NOT NULL
              AND text IS NOT NULL
        )
        SELECT *
        FROM base
        WHERE note_rank <= {config.max_notes_per_stay}
          AND running_char_count <= {config.max_chars_per_stay};
        """
    )


def build_task_rows(connection: duckdb.DuckDBPyConnection) -> list[dict[str, object]]:
    """Fetch grouped stay-level note bundles for JSONL task generation."""

    rows = connection.execute(
        """
        SELECT
            stay_id,
            subject_id,
            hadm_id,
            note_id,
            note_charttime,
            text
        FROM ranked_linked_notes
        ORDER BY stay_id, note_charttime, note_id;
        """
    ).fetchall()

    task_rows: list[dict[str, object]] = []
    for (stay_id, subject_id, hadm_id), group_rows in groupby(
        rows,
        key=lambda row: (row[0], row[1], row[2]),
    ):
        grouped_rows = list(group_rows)
        notes = [
            {
                "note_id": note_id,
                "charttime": str(note_charttime) if note_charttime else None,
                "text": text,
            }
            for _, _, _, note_id, note_charttime, text in grouped_rows
        ]
        task_rows.append(
            {
                "task_id": f"stay-{stay_id}",
                "subject_id": subject_id,
                "hadm_id": hadm_id,
                "stay_id": stay_id,
                "note_count": len(notes),
                "first_note_time": notes[0]["charttime"] if notes else None,
                "last_note_time": notes[-1]["charttime"] if notes else None,
                "notes": notes,
                "annotation_template": ANNOTATION_TEMPLATE,
                "annotation_status": "pending",
            }
        )
    return task_rows


def write_outputs(
    task_rows: list[dict[str, object]],
    config: GoldTaskPrepConfig,
) -> GoldTaskPrepSummary:
    """Persist grouped annotation tasks as JSONL, CSV, and markdown report."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.report_dir.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now(timezone.utc)
    jsonl_path = config.output_dir / "gold_annotation_tasks.jsonl"
    csv_index_path = config.output_dir / "gold_annotation_index.csv"
    report_path = config.report_dir / f"{generated_at.strftime('%Y%m%dT%H%M%SZ')}_annotation_prep_report.md"

    with open(jsonl_path, "w", encoding="utf-8") as file:
        for row in task_rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(csv_index_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "task_id",
                "subject_id",
                "hadm_id",
                "stay_id",
                "note_count",
                "first_note_time",
                "last_note_time",
                "annotation_status",
                "suspected_sepsis",
                "suspicion_time",
                "infection_source",
                "evidence_span",
                "confidence",
                "reviewer_id",
                "reviewed_at",
            ],
        )
        writer.writeheader()
        for row in task_rows:
            writer.writerow(
                {
                    "task_id": row["task_id"],
                    "subject_id": row["subject_id"],
                    "hadm_id": row["hadm_id"],
                    "stay_id": row["stay_id"],
                    "note_count": row["note_count"],
                    "first_note_time": row["first_note_time"],
                    "last_note_time": row["last_note_time"],
                    "annotation_status": row["annotation_status"],
                    "suspected_sepsis": "",
                    "suspicion_time": "",
                    "infection_source": "",
                    "evidence_span": "",
                    "confidence": "",
                    "reviewer_id": "",
                    "reviewed_at": "",
                }
            )

    total_note_count = sum(int(row["note_count"]) for row in task_rows)
    summary = GoldTaskPrepSummary(
        generated_at_utc=generated_at.isoformat(),
        notes_parquet=str(config.notes_parquet),
        output_dir=str(config.output_dir),
        report_path=str(report_path),
        stay_count=len(task_rows),
        total_note_count=total_note_count,
        task_count=len(task_rows),
        max_notes_per_stay=config.max_notes_per_stay,
        max_chars_per_stay=config.max_chars_per_stay,
        jsonl_path=str(jsonl_path),
        csv_index_path=str(csv_index_path),
    )

    report_lines = [
        "# Annotation Preparation Report",
        f"Date: {summary.generated_at_utc}",
        "",
        "## Inputs",
        f"- notes_parquet: `{summary.notes_parquet}`",
        f"- max_notes_per_stay: {summary.max_notes_per_stay}",
        f"- max_chars_per_stay: {summary.max_chars_per_stay}",
        "",
        "## Outputs",
        f"- task_count: {summary.task_count}",
        f"- stay_count: {summary.stay_count}",
        f"- total_note_count: {summary.total_note_count}",
        f"- `{summary.jsonl_path}`",
        f"- `{summary.csv_index_path}`",
    ]
    with open(report_path, "w", encoding="utf-8") as file:
        file.write("\n".join(report_lines) + "\n")

    summary_json_path = config.output_dir / "gold_annotation_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as file:
        json.dump(asdict(summary), file, indent=2)

    return summary


def run_task_prep(config: GoldTaskPrepConfig) -> GoldTaskPrepSummary:
    """Execute grouped annotation task preparation from linked notes."""

    ensure_inputs(config)
    connection = duckdb.connect()
    try:
        ensure_required_columns(connection, config.notes_parquet)
        initialize_note_table(connection, config)
        task_rows = build_task_rows(connection)
        return write_outputs(task_rows, config)
    finally:
        connection.close()


def main() -> int:
    """CLI entrypoint for gold-label task preparation."""

    config = build_config(parse_args())
    summary = run_task_prep(config)

    print("Gold annotation task preparation completed.")
    print(f"Task count: {summary.task_count}")
    print(f"Total note count: {summary.total_note_count}")
    print(f"JSONL output: {summary.jsonl_path}")
    print(f"CSV index output: {summary.csv_index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
