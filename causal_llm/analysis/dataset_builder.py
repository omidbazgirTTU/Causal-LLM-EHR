"""Build matched downstream analysis datasets for gold, X_rule, and all model panels."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

from causal_llm.evaluation.extraction_eval import (
    build_model_prediction_rows,
    load_extraction_manifest,
    normalize_optional_text,
    parse_boolean,
    parse_rule_based_rows,
    parse_timestamp,
)

DEFAULT_MIMIC_ROOT = Path("physionet.org/files/mimiciv/3.1")
DEFAULT_COHORT_DUCKDB_PATH = Path("derived_data/cohort/phase1_cohort.duckdb")
DEFAULT_COHORT_TABLE = "sampled_candidate_cohort"
DEFAULT_GOLD_LABELS_CSV = Path("derived_data/annotations/finalized/gold_labels.csv")
DEFAULT_RULE_BASED_CSV = Path("derived_data/rule_based/rule_labels.csv")
DEFAULT_EXTRACTION_MANIFEST = Path("derived_data/extractions/extraction_run_manifest.json")
DEFAULT_OUTPUT_DIR = Path("derived_data/analysis")
DEFAULT_REPORT_DIR = Path("reports/analysis")
DEFAULT_EARLY_TREATMENT_WINDOW_HOURS = 3.0

REQUIRED_MIMIC_FILES = {
    "patients": "hosp/patients.csv.gz",
    "admissions": "hosp/admissions.csv.gz",
}
REQUIRED_GOLD_COLUMNS = {
    "task_id",
    "subject_id",
    "hadm_id",
    "stay_id",
    "suspected_sepsis",
    "suspicion_time",
    "infection_source",
    "evidence_span",
}
REQUIRED_COHORT_COLUMNS = {
    "subject_id",
    "hadm_id",
    "stay_id",
    "anchor_age",
    "admittime",
    "dischtime",
    "icu_intime",
    "icu_outtime",
    "study_window_start",
    "study_window_end",
    "first_antibiotic_time",
    "antibiotic_event_count",
    "first_lactate_time",
    "lactate_event_count",
    "first_blood_culture_time",
    "blood_culture_event_count",
    "has_antibiotic_exposure",
    "has_lactate_measurement",
    "has_blood_culture",
    "signal_count",
}
VALID_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
MODEL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")
DATASET_FIELDNAMES = [
    "task_id",
    "subject_id",
    "hadm_id",
    "stay_id",
    "predictor_name",
    "predictor_type",
    "predictor_id",
    "suspected_sepsis",
    "suspicion_time",
    "infection_source",
    "evidence_span",
    "predictor_confidence",
    "analysis_eligible",
    "analysis_exclusion_reason",
    "treatment_timing_category",
    "early_antibiotic_within_window",
    "treatment_delay_hours",
    "treatment_before_time_zero",
    "death_time",
    "death_delay_days",
    "death_before_time_zero",
    "mortality_28d",
    "followup_duration_days",
    "hours_from_icu_to_time_zero",
    "admittime",
    "dischtime",
    "icu_intime",
    "icu_outtime",
    "study_window_start",
    "study_window_end",
    "first_antibiotic_time",
    "antibiotic_event_count",
    "first_lactate_time",
    "lactate_event_count",
    "first_blood_culture_time",
    "blood_culture_event_count",
    "anchor_age",
    "gender",
    "admission_type",
    "insurance",
    "marital_status",
    "race",
    "has_antibiotic_exposure",
    "has_lactate_measurement",
    "has_blood_culture",
    "signal_count",
]


@dataclass(frozen=True)
class AnalysisDatasetConfig:
    """Configuration for one analysis-dataset construction run."""

    mimic_root: Path
    cohort_duckdb_path: Path
    cohort_table: str
    gold_labels_csv: Path
    rule_based_csv: Path
    extraction_manifest: Path
    output_dir: Path
    report_dir: Path
    early_treatment_window_hours: float


@dataclass(frozen=True)
class PredictorSpec:
    """One predictor dataset to materialize into a matched analysis table."""

    predictor_name: str
    predictor_type: str
    predictor_id: str
    rows_by_task_id: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class AnalysisDatasetSummary:
    """Compact summary of one predictor-specific analysis dataset."""

    predictor_name: str
    predictor_type: str
    predictor_id: str
    row_count: int
    suspected_sepsis_count: int
    analysis_eligible_count: int
    missing_time_zero_count: int
    treatment_before_time_zero_count: int
    death_before_time_zero_count: int
    early_antibiotic_count: int
    delayed_or_no_antibiotic_count: int
    mortality_28d_count: int
    csv_path: str


@dataclass(frozen=True)
class AnalysisDatasetRunSummary:
    """Summary artifact emitted after analysis-dataset construction completes."""

    generated_at_utc: str
    mimic_root: str
    cohort_duckdb_path: str
    cohort_table: str
    gold_labels_csv: str
    rule_based_csv: str
    extraction_manifest: str
    output_dir: str
    report_path: str
    early_treatment_window_hours: float
    predictor_summaries: list[dict[str, object]]
    summary_csv_path: str
    manifest_path: str


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for analysis-dataset construction."""

    parser = argparse.ArgumentParser(
        description=(
            "Build matched downstream analysis datasets for gold, X_rule, and "
            "all extracted model outputs."
        )
    )
    parser.add_argument(
        "--mimic-root",
        type=Path,
        default=DEFAULT_MIMIC_ROOT,
        help="Path to the local MIMIC-IV v3.1 root directory.",
    )
    parser.add_argument(
        "--cohort-duckdb-path",
        type=Path,
        default=DEFAULT_COHORT_DUCKDB_PATH,
        help="Path to the Phase 1 cohort DuckDB artifact.",
    )
    parser.add_argument(
        "--cohort-table",
        default=DEFAULT_COHORT_TABLE,
        help="Cohort table to join against the gold and extraction artifacts.",
    )
    parser.add_argument(
        "--gold-labels-csv",
        type=Path,
        default=DEFAULT_GOLD_LABELS_CSV,
        help="Finalized gold labels CSV produced by `python finalize_gold_annotations.py`.",
    )
    parser.add_argument(
        "--rule-based-csv",
        type=Path,
        default=DEFAULT_RULE_BASED_CSV,
        help="Rule-based baseline CSV used as the X_rule comparator.",
    )
    parser.add_argument(
        "--extraction-manifest",
        type=Path,
        default=DEFAULT_EXTRACTION_MANIFEST,
        help="Manifest produced by `python run_llm_extraction.py`.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for predictor-specific analysis datasets.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Directory for analysis-dataset markdown reports.",
    )
    parser.add_argument(
        "--early-treatment-window-hours",
        type=float,
        default=DEFAULT_EARLY_TREATMENT_WINDOW_HOURS,
        help="Window used to label antibiotics as early after time zero.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> AnalysisDatasetConfig:
    """Resolve CLI arguments into a normalized analysis-dataset config."""

    cohort_table = args.cohort_table.strip()
    if not cohort_table:
        raise ValueError("--cohort-table must be non-empty.")
    if not VALID_IDENTIFIER_PATTERN.match(cohort_table):
        raise ValueError("--cohort-table may contain only letters, numbers, and underscores.")
    if args.early_treatment_window_hours <= 0:
        raise ValueError("--early-treatment-window-hours must be positive.")

    return AnalysisDatasetConfig(
        mimic_root=args.mimic_root.resolve(),
        cohort_duckdb_path=args.cohort_duckdb_path.resolve(),
        cohort_table=cohort_table,
        gold_labels_csv=args.gold_labels_csv.resolve(),
        rule_based_csv=args.rule_based_csv.resolve(),
        extraction_manifest=args.extraction_manifest.resolve(),
        output_dir=args.output_dir.resolve(),
        report_dir=args.report_dir.resolve(),
        early_treatment_window_hours=float(args.early_treatment_window_hours),
    )


def sql_string(value: str) -> str:
    """Quote a Python string for safe interpolation into DuckDB SQL."""

    return "'" + value.replace("'", "''") + "'"


def serialize_timestamp(value: Any) -> str | None:
    """Serialize timestamp-like values into a stable second-level string."""

    normalized = parse_timestamp(value)
    return normalized


def parse_datetime(value: Any) -> datetime | None:
    """Parse a timestamp-like value into a naive UTC-normalized datetime."""

    normalized = parse_timestamp(value)
    if normalized is None:
        return None
    return datetime.fromisoformat(normalized)


def round_optional(value: float | None) -> float | None:
    """Round floating-point metrics when a value is present."""

    if value is None:
        return None
    return round(value, 6)


def describe_columns(connection: duckdb.DuckDBPyConnection, table_name: str) -> set[str]:
    """Return lowercase column names for a DuckDB table or view."""

    return {
        row[0].lower()
        for row in connection.execute(f"DESCRIBE SELECT * FROM {table_name}").fetchall()
    }


def text_column_expression(alias: str, column_name: str | None) -> str:
    """Build a trimmed nullable SQL expression for a text column."""

    if column_name is None:
        return "CAST(NULL AS VARCHAR)"
    return f"NULLIF(TRIM(COALESCE({alias}.{column_name}, '')), '')"


def ensure_inputs(config: AnalysisDatasetConfig) -> dict[str, Path]:
    """Validate the required cohort, measurement, and raw MIMIC inputs."""

    if not config.cohort_duckdb_path.exists():
        raise FileNotFoundError(
            f"Missing cohort database: {config.cohort_duckdb_path}\n"
            "Run `python build_initial_cohort.py` first."
        )
    if not config.gold_labels_csv.exists():
        raise FileNotFoundError(
            f"Missing gold labels CSV: {config.gold_labels_csv}\n"
            "Run `python finalize_gold_annotations.py` first."
        )
    if not config.rule_based_csv.exists():
        raise FileNotFoundError(
            f"Missing rule-based CSV: {config.rule_based_csv}\n"
            "Run `python build_rule_based_baseline.py` first."
        )
    if not config.extraction_manifest.exists():
        raise FileNotFoundError(
            f"Missing extraction manifest: {config.extraction_manifest}\n"
            "Run `python run_llm_extraction.py` first."
        )

    resolved_paths: dict[str, Path] = {}
    missing_paths: list[Path] = []
    for name, relative_path in REQUIRED_MIMIC_FILES.items():
        absolute_path = config.mimic_root / relative_path
        resolved_paths[name] = absolute_path
        if not absolute_path.exists():
            missing_paths.append(absolute_path)

    if missing_paths:
        missing_text = "\n".join(f"- {path}" for path in missing_paths)
        raise FileNotFoundError(f"Missing required structured MIMIC inputs:\n{missing_text}")

    return resolved_paths


def parse_int_field(row: dict[str, Any], field_name: str, *, source_path: Path) -> int:
    """Parse one required integer field from a CSV row."""

    value = normalize_optional_text(row.get(field_name))
    if value is None:
        raise ValueError(f"Missing required field `{field_name}` in {source_path}.")
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid integer value for `{field_name}` in {source_path}: {value!r}"
        ) from exc


def load_gold_rows(path: Path) -> dict[str, dict[str, Any]]:
    """Load finalized gold rows with the identity fields needed downstream."""

    rows_by_task_id: dict[str, dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        missing_columns = sorted(REQUIRED_GOLD_COLUMNS - set(reader.fieldnames or []))
        if missing_columns:
            raise ValueError(
                f"Gold labels CSV is missing required columns: {', '.join(missing_columns)}"
            )
        for row in reader:
            task_id = normalize_optional_text(row.get("task_id"))
            if task_id is None:
                raise ValueError(f"Gold labels CSV contains a blank task_id: {path}")
            if task_id in rows_by_task_id:
                raise ValueError(f"Duplicate task_id detected in gold labels CSV: {task_id}")
            rows_by_task_id[task_id] = {
                "task_id": task_id,
                "subject_id": parse_int_field(row, "subject_id", source_path=path),
                "hadm_id": parse_int_field(row, "hadm_id", source_path=path),
                "stay_id": parse_int_field(row, "stay_id", source_path=path),
                "suspected_sepsis": parse_boolean(row["suspected_sepsis"]),
                "suspicion_time": parse_timestamp(row.get("suspicion_time")),
                "infection_source": normalize_optional_text(row.get("infection_source")),
                "evidence_span": normalize_optional_text(row.get("evidence_span")),
                "confidence": normalize_optional_text(row.get("confidence")),
            }

    if not rows_by_task_id:
        raise ValueError(f"Gold labels CSV contains no rows: {path}")
    return rows_by_task_id


def build_gold_predictor_rows(gold_rows_by_task_id: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Adapt gold rows into the canonical predictor-row shape used downstream."""

    predictor_rows: dict[str, dict[str, Any]] = {}
    for task_id, row in gold_rows_by_task_id.items():
        predictor_rows[task_id] = {
            "task_id": task_id,
            "prediction_status": "ok",
            "normalized_prediction": {
                "suspected_sepsis": row["suspected_sepsis"],
                "suspicion_time": row["suspicion_time"],
                "infection_source": row["infection_source"],
                "evidence_span": row["evidence_span"],
                "confidence": row["confidence"],
            },
        }
    return predictor_rows


def load_predictors(
    config: AnalysisDatasetConfig,
    gold_rows_by_task_id: dict[str, dict[str, Any]],
) -> list[PredictorSpec]:
    """Load all predictor datasets that must follow the same causal pipeline."""

    predictors: list[PredictorSpec] = [
        PredictorSpec(
            predictor_name="gold",
            predictor_type="gold",
            predictor_id="gold_labels",
            rows_by_task_id=build_gold_predictor_rows(gold_rows_by_task_id),
        )
    ]

    rule_rows_by_task_id = parse_rule_based_rows(config.rule_based_csv, gold_rows_by_task_id)
    predictors.append(
        PredictorSpec(
            predictor_name="x_rule",
            predictor_type="rule_based",
            predictor_id="rule_based_baseline",
            rows_by_task_id=rule_rows_by_task_id,
        )
    )

    extraction_manifest = load_extraction_manifest(config.extraction_manifest)
    for predictor_name, predictor_id, rows_by_task_id in build_model_prediction_rows(
        extraction_manifest,
        gold_rows_by_task_id,
    ):
        predictors.append(
            PredictorSpec(
                predictor_name=predictor_name,
                predictor_type="model",
                predictor_id=predictor_id,
                rows_by_task_id=rows_by_task_id,
            )
        )

    return predictors


def ensure_required_cohort_columns(connection: duckdb.DuckDBPyConnection) -> None:
    """Validate that the selected cohort table exposes the expected Phase 1 schema."""

    columns = describe_columns(connection, "selected_cohort")
    missing = sorted(REQUIRED_COHORT_COLUMNS - columns)
    if missing:
        raise ValueError(
            "Selected cohort table is missing required columns: " + ", ".join(missing)
        )


def load_cohort_rows_by_stay_id(
    config: AnalysisDatasetConfig,
    required_paths: dict[str, Path],
    expected_stay_ids: set[int],
) -> dict[int, dict[str, Any]]:
    """Load the cohort, covariates, and mortality fields for the gold-task stay set."""

    connection = duckdb.connect(database=":memory:")
    connection.execute(
        f"ATTACH {sql_string(str(config.cohort_duckdb_path))} AS cohort_db (READ_ONLY);"
    )
    connection.execute(
        f"CREATE OR REPLACE VIEW selected_cohort AS SELECT * FROM cohort_db.{config.cohort_table};"
    )
    connection.execute(
        f"""
        CREATE OR REPLACE VIEW src_patients AS
        SELECT *
        FROM read_csv_auto(
            {sql_string(str(required_paths['patients']))},
            header = TRUE,
            all_varchar = TRUE
        );
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE VIEW src_admissions AS
        SELECT *
        FROM read_csv_auto(
            {sql_string(str(required_paths['admissions']))},
            header = TRUE,
            all_varchar = TRUE
        );
        """
    )

    ensure_required_cohort_columns(connection)

    patient_columns = describe_columns(connection, "src_patients")
    admission_columns = describe_columns(connection, "src_admissions")
    gender_column = "gender" if "gender" in patient_columns else None
    dod_column = "dod" if "dod" in patient_columns else None
    admission_type_column = "admission_type" if "admission_type" in admission_columns else None
    insurance_column = "insurance" if "insurance" in admission_columns else None
    marital_status_column = "marital_status" if "marital_status" in admission_columns else None
    race_column = "race" if "race" in admission_columns else ("ethnicity" if "ethnicity" in admission_columns else None)
    deathtime_column = "deathtime" if "deathtime" in admission_columns else None

    death_time_expression = "CAST(NULL AS TIMESTAMP)"
    if dod_column is not None and deathtime_column is not None:
        death_time_expression = (
            "CASE "
            f"WHEN TRY_CAST(adm.{deathtime_column} AS TIMESTAMP) IS NOT NULL "
            f"AND TRY_CAST(patient.{dod_column} AS TIMESTAMP) IS NOT NULL "
            f"THEN LEAST(TRY_CAST(adm.{deathtime_column} AS TIMESTAMP), TRY_CAST(patient.{dod_column} AS TIMESTAMP)) "
            f"ELSE COALESCE(TRY_CAST(adm.{deathtime_column} AS TIMESTAMP), TRY_CAST(patient.{dod_column} AS TIMESTAMP)) "
            "END"
        )
    elif deathtime_column is not None:
        death_time_expression = f"TRY_CAST(adm.{deathtime_column} AS TIMESTAMP)"
    elif dod_column is not None:
        death_time_expression = f"TRY_CAST(patient.{dod_column} AS TIMESTAMP)"

    cursor = connection.execute(
        f"""
        SELECT
            TRY_CAST(cohort.subject_id AS BIGINT) AS subject_id,
            TRY_CAST(cohort.hadm_id AS BIGINT) AS hadm_id,
            TRY_CAST(cohort.stay_id AS BIGINT) AS stay_id,
            TRY_CAST(cohort.anchor_age AS INTEGER) AS anchor_age,
            TRY_CAST(cohort.admittime AS TIMESTAMP) AS admittime,
            TRY_CAST(cohort.dischtime AS TIMESTAMP) AS dischtime,
            TRY_CAST(cohort.icu_intime AS TIMESTAMP) AS icu_intime,
            TRY_CAST(cohort.icu_outtime AS TIMESTAMP) AS icu_outtime,
            TRY_CAST(cohort.study_window_start AS TIMESTAMP) AS study_window_start,
            TRY_CAST(cohort.study_window_end AS TIMESTAMP) AS study_window_end,
            TRY_CAST(cohort.first_antibiotic_time AS TIMESTAMP) AS first_antibiotic_time,
            TRY_CAST(cohort.antibiotic_event_count AS BIGINT) AS antibiotic_event_count,
            TRY_CAST(cohort.first_lactate_time AS TIMESTAMP) AS first_lactate_time,
            TRY_CAST(cohort.lactate_event_count AS BIGINT) AS lactate_event_count,
            TRY_CAST(cohort.first_blood_culture_time AS TIMESTAMP) AS first_blood_culture_time,
            TRY_CAST(cohort.blood_culture_event_count AS BIGINT) AS blood_culture_event_count,
            CAST(cohort.has_antibiotic_exposure AS BOOLEAN) AS has_antibiotic_exposure,
            CAST(cohort.has_lactate_measurement AS BOOLEAN) AS has_lactate_measurement,
            CAST(cohort.has_blood_culture AS BOOLEAN) AS has_blood_culture,
            TRY_CAST(cohort.signal_count AS INTEGER) AS signal_count,
            {text_column_expression('patient', gender_column)} AS gender,
            {text_column_expression('adm', admission_type_column)} AS admission_type,
            {text_column_expression('adm', insurance_column)} AS insurance,
            {text_column_expression('adm', marital_status_column)} AS marital_status,
            {text_column_expression('adm', race_column)} AS race,
            {death_time_expression} AS death_time
        FROM selected_cohort AS cohort
        INNER JOIN src_patients AS patient
            ON TRY_CAST(cohort.subject_id AS BIGINT) = TRY_CAST(patient.subject_id AS BIGINT)
        INNER JOIN src_admissions AS adm
            ON TRY_CAST(cohort.subject_id AS BIGINT) = TRY_CAST(adm.subject_id AS BIGINT)
           AND TRY_CAST(cohort.hadm_id AS BIGINT) = TRY_CAST(adm.hadm_id AS BIGINT);
        """
    )
    columns = [column[0] for column in cursor.description]

    rows_by_stay_id: dict[int, dict[str, Any]] = {}
    for raw_row in cursor.fetchall():
        row = dict(zip(columns, raw_row))
        stay_id = row["stay_id"]
        if stay_id is None or stay_id not in expected_stay_ids:
            continue
        if stay_id in rows_by_stay_id:
            raise ValueError(f"Duplicate stay_id detected in selected cohort: {stay_id}")
        rows_by_stay_id[stay_id] = row

    missing_stay_ids = sorted(expected_stay_ids - set(rows_by_stay_id))
    if missing_stay_ids:
        preview = ", ".join(str(value) for value in missing_stay_ids[:5])
        suffix = "..." if len(missing_stay_ids) > 5 else ""
        raise ValueError(
            "Selected cohort does not cover the full gold-label stay set. "
            f"Missing {len(missing_stay_ids)} stay_id(s): {preview}{suffix}"
        )

    return rows_by_stay_id


def validate_prediction_identity(
    predictor: PredictorSpec,
    *,
    task_id: str,
    predictor_row: dict[str, Any],
    gold_row: dict[str, Any],
) -> None:
    """Ensure model outputs did not drift onto a different stay or admission."""

    for field_name in ("subject_id", "hadm_id", "stay_id"):
        if field_name not in predictor_row:
            continue
        predictor_value = predictor_row[field_name]
        if predictor_value is None:
            continue
        if int(predictor_value) != int(gold_row[field_name]):
            raise ValueError(
                f"{predictor.predictor_name} identity mismatch for task_id={task_id}: "
                f"{field_name}={predictor_value!r} does not match gold {gold_row[field_name]!r}."
            )


def require_valid_prediction(
    predictor: PredictorSpec,
    *,
    task_id: str,
    predictor_row: dict[str, Any],
) -> dict[str, Any]:
    """Return the normalized prediction payload or fail on invalid artifacts."""

    status = normalize_optional_text(predictor_row.get("prediction_status")) or "unknown"
    normalized_prediction = predictor_row.get("normalized_prediction")
    if status != "ok" or not isinstance(normalized_prediction, dict):
        raise ValueError(
            f"{predictor.predictor_name} has an invalid prediction for task_id={task_id}: "
            f"status={status!r}."
        )
    return normalized_prediction


def derive_treatment_fields(
    *,
    time_zero: datetime | None,
    first_antibiotic_time: datetime | None,
    early_window_hours: float,
) -> dict[str, Any]:
    """Derive treatment timing fields relative to the predictor-specific time zero."""

    if time_zero is None:
        return {
            "treatment_timing_category": "no_time_zero",
            "early_antibiotic_within_window": None,
            "treatment_delay_hours": None,
            "treatment_before_time_zero": False,
        }
    if first_antibiotic_time is None:
        return {
            "treatment_timing_category": "no_antibiotic",
            "early_antibiotic_within_window": False,
            "treatment_delay_hours": None,
            "treatment_before_time_zero": False,
        }

    treatment_delay_hours = (
        first_antibiotic_time - time_zero
    ).total_seconds() / 3600.0
    if treatment_delay_hours < 0:
        return {
            "treatment_timing_category": "before_time_zero",
            "early_antibiotic_within_window": None,
            "treatment_delay_hours": round_optional(treatment_delay_hours),
            "treatment_before_time_zero": True,
        }
    if treatment_delay_hours <= early_window_hours:
        return {
            "treatment_timing_category": "within_window",
            "early_antibiotic_within_window": True,
            "treatment_delay_hours": round_optional(treatment_delay_hours),
            "treatment_before_time_zero": False,
        }
    return {
        "treatment_timing_category": "after_window",
        "early_antibiotic_within_window": False,
        "treatment_delay_hours": round_optional(treatment_delay_hours),
        "treatment_before_time_zero": False,
    }


def derive_outcome_fields(
    *,
    time_zero: datetime | None,
    death_time: datetime | None,
) -> dict[str, Any]:
    """Derive 28-day mortality fields relative to predictor-specific time zero."""

    if time_zero is None:
        return {
            "death_time": serialize_timestamp(death_time),
            "death_delay_days": None,
            "death_before_time_zero": False,
            "mortality_28d": None,
            "followup_duration_days": None,
        }
    if death_time is None:
        return {
            "death_time": None,
            "death_delay_days": None,
            "death_before_time_zero": False,
            "mortality_28d": False,
            "followup_duration_days": 28.0,
        }

    death_delay_days = (death_time - time_zero).total_seconds() / 86400.0
    if death_delay_days < 0:
        return {
            "death_time": serialize_timestamp(death_time),
            "death_delay_days": round_optional(death_delay_days),
            "death_before_time_zero": True,
            "mortality_28d": None,
            "followup_duration_days": None,
        }
    if death_delay_days <= 28.0:
        rounded_delay = round_optional(death_delay_days)
        return {
            "death_time": serialize_timestamp(death_time),
            "death_delay_days": rounded_delay,
            "death_before_time_zero": False,
            "mortality_28d": True,
            "followup_duration_days": rounded_delay,
        }
    return {
        "death_time": serialize_timestamp(death_time),
        "death_delay_days": round_optional(death_delay_days),
        "death_before_time_zero": False,
        "mortality_28d": False,
        "followup_duration_days": 28.0,
    }


def ensure_time_zero_within_study_window(
    *,
    predictor_name: str,
    task_id: str,
    time_zero: datetime | None,
    study_window_start: datetime | None,
    study_window_end: datetime | None,
) -> None:
    """Fail if the predictor emits a time zero outside the audited study window."""

    if time_zero is None:
        return
    if study_window_start is None or study_window_end is None:
        raise ValueError(
            f"Selected cohort is missing study-window bounds for task_id={task_id}."
        )
    if time_zero < study_window_start or time_zero > study_window_end:
        raise ValueError(
            f"{predictor_name} emitted suspicion_time outside the study window for "
            f"task_id={task_id}: time_zero={time_zero.isoformat(sep=' ', timespec='seconds')} "
            f"window=[{study_window_start.isoformat(sep=' ', timespec='seconds')}, "
            f"{study_window_end.isoformat(sep=' ', timespec='seconds')}]."
        )


def build_dataset_rows(
    predictor: PredictorSpec,
    *,
    gold_rows_by_task_id: dict[str, dict[str, Any]],
    cohort_rows_by_stay_id: dict[int, dict[str, Any]],
    early_treatment_window_hours: float,
) -> list[dict[str, Any]]:
    """Build one predictor-specific downstream analysis dataset."""

    dataset_rows: list[dict[str, Any]] = []
    for task_id, gold_row in gold_rows_by_task_id.items():
        predictor_row = predictor.rows_by_task_id.get(task_id)
        if predictor_row is None:
            raise ValueError(f"{predictor.predictor_name} is missing task_id={task_id}.")
        validate_prediction_identity(
            predictor,
            task_id=task_id,
            predictor_row=predictor_row,
            gold_row=gold_row,
        )
        normalized_prediction = require_valid_prediction(
            predictor,
            task_id=task_id,
            predictor_row=predictor_row,
        )

        cohort_row = cohort_rows_by_stay_id.get(gold_row["stay_id"])
        if cohort_row is None:
            raise ValueError(f"Selected cohort is missing stay_id={gold_row['stay_id']}.")
        if cohort_row["subject_id"] != gold_row["subject_id"] or cohort_row["hadm_id"] != gold_row["hadm_id"]:
            raise ValueError(
                f"Cohort identity mismatch for task_id={task_id}: "
                f"cohort subject_id/hadm_id={cohort_row['subject_id']}/{cohort_row['hadm_id']} "
                f"gold subject_id/hadm_id={gold_row['subject_id']}/{gold_row['hadm_id']}."
            )

        suspected_sepsis = bool(normalized_prediction["suspected_sepsis"])
        suspicion_time = parse_datetime(normalized_prediction.get("suspicion_time"))
        if suspected_sepsis and suspicion_time is None:
            raise ValueError(
                f"{predictor.predictor_name} predicted suspected_sepsis=true but no "
                f"suspicion_time for task_id={task_id}."
            )
        if not suspected_sepsis:
            suspicion_time = None

        ensure_time_zero_within_study_window(
            predictor_name=predictor.predictor_name,
            task_id=task_id,
            time_zero=suspicion_time,
            study_window_start=cohort_row["study_window_start"],
            study_window_end=cohort_row["study_window_end"],
        )

        treatment_fields = derive_treatment_fields(
            time_zero=suspicion_time,
            first_antibiotic_time=cohort_row["first_antibiotic_time"],
            early_window_hours=early_treatment_window_hours,
        )
        outcome_fields = derive_outcome_fields(
            time_zero=suspicion_time,
            death_time=cohort_row["death_time"],
        )

        exclusion_reasons: list[str] = []
        if suspicion_time is None:
            exclusion_reasons.append("missing_time_zero")
        if treatment_fields["treatment_before_time_zero"]:
            exclusion_reasons.append("treatment_before_time_zero")
        if outcome_fields["death_before_time_zero"]:
            exclusion_reasons.append("death_before_time_zero")

        hours_from_icu_to_time_zero = None
        if suspicion_time is not None and cohort_row["icu_intime"] is not None:
            hours_from_icu_to_time_zero = round_optional(
                (suspicion_time - cohort_row["icu_intime"]).total_seconds() / 3600.0
            )

        dataset_rows.append(
            {
                "task_id": task_id,
                "subject_id": gold_row["subject_id"],
                "hadm_id": gold_row["hadm_id"],
                "stay_id": gold_row["stay_id"],
                "predictor_name": predictor.predictor_name,
                "predictor_type": predictor.predictor_type,
                "predictor_id": predictor.predictor_id,
                "suspected_sepsis": suspected_sepsis,
                "suspicion_time": serialize_timestamp(suspicion_time),
                "infection_source": normalize_optional_text(normalized_prediction.get("infection_source")),
                "evidence_span": normalize_optional_text(normalized_prediction.get("evidence_span")),
                "predictor_confidence": normalized_prediction.get("confidence"),
                "analysis_eligible": not exclusion_reasons,
                "analysis_exclusion_reason": "|".join(exclusion_reasons) or None,
                "treatment_timing_category": treatment_fields["treatment_timing_category"],
                "early_antibiotic_within_window": treatment_fields["early_antibiotic_within_window"],
                "treatment_delay_hours": treatment_fields["treatment_delay_hours"],
                "treatment_before_time_zero": treatment_fields["treatment_before_time_zero"],
                "death_time": outcome_fields["death_time"],
                "death_delay_days": outcome_fields["death_delay_days"],
                "death_before_time_zero": outcome_fields["death_before_time_zero"],
                "mortality_28d": outcome_fields["mortality_28d"],
                "followup_duration_days": outcome_fields["followup_duration_days"],
                "hours_from_icu_to_time_zero": hours_from_icu_to_time_zero,
                "admittime": serialize_timestamp(cohort_row["admittime"]),
                "dischtime": serialize_timestamp(cohort_row["dischtime"]),
                "icu_intime": serialize_timestamp(cohort_row["icu_intime"]),
                "icu_outtime": serialize_timestamp(cohort_row["icu_outtime"]),
                "study_window_start": serialize_timestamp(cohort_row["study_window_start"]),
                "study_window_end": serialize_timestamp(cohort_row["study_window_end"]),
                "first_antibiotic_time": serialize_timestamp(cohort_row["first_antibiotic_time"]),
                "antibiotic_event_count": cohort_row["antibiotic_event_count"],
                "first_lactate_time": serialize_timestamp(cohort_row["first_lactate_time"]),
                "lactate_event_count": cohort_row["lactate_event_count"],
                "first_blood_culture_time": serialize_timestamp(cohort_row["first_blood_culture_time"]),
                "blood_culture_event_count": cohort_row["blood_culture_event_count"],
                "anchor_age": cohort_row["anchor_age"],
                "gender": cohort_row["gender"],
                "admission_type": cohort_row["admission_type"],
                "insurance": cohort_row["insurance"],
                "marital_status": cohort_row["marital_status"],
                "race": cohort_row["race"],
                "has_antibiotic_exposure": cohort_row["has_antibiotic_exposure"],
                "has_lactate_measurement": cohort_row["has_lactate_measurement"],
                "has_blood_culture": cohort_row["has_blood_culture"],
                "signal_count": cohort_row["signal_count"],
            }
        )

    return dataset_rows


def summarize_dataset(
    predictor: PredictorSpec,
    *,
    dataset_rows: list[dict[str, Any]],
    csv_path: Path,
) -> AnalysisDatasetSummary:
    """Compute the per-predictor dataset summary used in the report and manifest."""

    return AnalysisDatasetSummary(
        predictor_name=predictor.predictor_name,
        predictor_type=predictor.predictor_type,
        predictor_id=predictor.predictor_id,
        row_count=len(dataset_rows),
        suspected_sepsis_count=sum(bool(row["suspected_sepsis"]) for row in dataset_rows),
        analysis_eligible_count=sum(bool(row["analysis_eligible"]) for row in dataset_rows),
        missing_time_zero_count=sum(row["analysis_exclusion_reason"] == "missing_time_zero" for row in dataset_rows),
        treatment_before_time_zero_count=sum(bool(row["treatment_before_time_zero"]) for row in dataset_rows),
        death_before_time_zero_count=sum(bool(row["death_before_time_zero"]) for row in dataset_rows),
        early_antibiotic_count=sum(row["early_antibiotic_within_window"] is True for row in dataset_rows),
        delayed_or_no_antibiotic_count=sum(row["early_antibiotic_within_window"] is False for row in dataset_rows),
        mortality_28d_count=sum(row["mortality_28d"] is True for row in dataset_rows),
        csv_path=str(csv_path),
    )


def write_dataset_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write one predictor-specific analysis dataset to CSV."""

    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=DATASET_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_outputs(
    *,
    config: AnalysisDatasetConfig,
    predictor_outputs: list[tuple[PredictorSpec, list[dict[str, Any]], AnalysisDatasetSummary]],
) -> AnalysisDatasetRunSummary:
    """Persist predictor-specific datasets, the run manifest, and the markdown report."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.report_dir.mkdir(parents=True, exist_ok=True)

    summary_csv_path = config.output_dir / "analysis_dataset_summary.csv"
    manifest_path = config.output_dir / "analysis_dataset_manifest.json"
    report_path = config.report_dir / (
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_analysis_dataset_report.md"
    )

    with open(summary_csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "predictor_name",
                "predictor_type",
                "predictor_id",
                "row_count",
                "suspected_sepsis_count",
                "analysis_eligible_count",
                "missing_time_zero_count",
                "treatment_before_time_zero_count",
                "death_before_time_zero_count",
                "early_antibiotic_count",
                "delayed_or_no_antibiotic_count",
                "mortality_28d_count",
                "csv_path",
            ],
        )
        writer.writeheader()
        for _, _, summary in predictor_outputs:
            writer.writerow(asdict(summary))

    run_summary = AnalysisDatasetRunSummary(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        mimic_root=str(config.mimic_root),
        cohort_duckdb_path=str(config.cohort_duckdb_path),
        cohort_table=config.cohort_table,
        gold_labels_csv=str(config.gold_labels_csv),
        rule_based_csv=str(config.rule_based_csv),
        extraction_manifest=str(config.extraction_manifest),
        output_dir=str(config.output_dir),
        report_path=str(report_path),
        early_treatment_window_hours=config.early_treatment_window_hours,
        predictor_summaries=[asdict(summary) for _, _, summary in predictor_outputs],
        summary_csv_path=str(summary_csv_path),
        manifest_path=str(manifest_path),
    )
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(asdict(run_summary), file, indent=2)

    report_lines = [
        "# Analysis Dataset Report",
        f"Date: {run_summary.generated_at_utc}",
        "",
        "## Inputs",
        f"- cohort_duckdb_path: `{config.cohort_duckdb_path}`",
        f"- cohort_table: `{config.cohort_table}`",
        f"- gold_labels_csv: `{config.gold_labels_csv}`",
        f"- rule_based_csv: `{config.rule_based_csv}`",
        f"- extraction_manifest: `{config.extraction_manifest}`",
        f"- early_treatment_window_hours: {config.early_treatment_window_hours}",
        "",
        "## Predictor Datasets",
    ]
    for _, _, summary in predictor_outputs:
        report_lines.append(
            "- "
            + ", ".join(
                [
                    f"predictor={summary.predictor_name}",
                    f"row_count={summary.row_count}",
                    f"analysis_eligible_count={summary.analysis_eligible_count}",
                    f"early_antibiotic_count={summary.early_antibiotic_count}",
                    f"mortality_28d_count={summary.mortality_28d_count}",
                    f"treatment_before_time_zero_count={summary.treatment_before_time_zero_count}",
                    f"death_before_time_zero_count={summary.death_before_time_zero_count}",
                ]
            )
        )
    report_lines.extend(
        [
            "",
            "## Outputs",
            f"- `{summary_csv_path}`",
            f"- `{manifest_path}`",
            f"- `{report_path}`",
        ]
    )
    with open(report_path, "w", encoding="utf-8") as file:
        file.write("\n".join(report_lines) + "\n")

    return run_summary


def build_analysis_datasets(config: AnalysisDatasetConfig) -> AnalysisDatasetRunSummary:
    """Build all predictor-specific downstream analysis datasets for causal estimation."""

    required_paths = ensure_inputs(config)
    gold_rows_by_task_id = load_gold_rows(config.gold_labels_csv)
    predictors = load_predictors(config, gold_rows_by_task_id)
    expected_stay_ids = {row["stay_id"] for row in gold_rows_by_task_id.values()}
    cohort_rows_by_stay_id = load_cohort_rows_by_stay_id(
        config,
        required_paths,
        expected_stay_ids,
    )

    predictor_outputs: list[tuple[PredictorSpec, list[dict[str, Any]], AnalysisDatasetSummary]] = []
    for predictor in predictors:
        if not MODEL_NAME_PATTERN.match(predictor.predictor_name):
            raise ValueError(
                f"Predictor name {predictor.predictor_name!r} is not safe for output filenames."
            )
        dataset_rows = build_dataset_rows(
            predictor,
            gold_rows_by_task_id=gold_rows_by_task_id,
            cohort_rows_by_stay_id=cohort_rows_by_stay_id,
            early_treatment_window_hours=config.early_treatment_window_hours,
        )
        csv_path = config.output_dir / f"analysis_dataset_{predictor.predictor_name}.csv"
        config.output_dir.mkdir(parents=True, exist_ok=True)
        write_dataset_csv(csv_path, dataset_rows)
        summary = summarize_dataset(
            predictor,
            dataset_rows=dataset_rows,
            csv_path=csv_path,
        )
        predictor_outputs.append((predictor, dataset_rows, summary))

    return write_outputs(
        config=config,
        predictor_outputs=predictor_outputs,
    )


def main() -> int:
    """CLI entrypoint for analysis-dataset construction."""

    args = parse_args()
    config = build_config(args)
    summary = build_analysis_datasets(config)
    print("Analysis dataset construction completed.")
    print(f"Predictor count: {len(summary.predictor_summaries)}")
    print(f"Manifest: {summary.manifest_path}")
    print(f"Report: {summary.report_path}")
    return 0
