"""Run the fixed propensity-weighted survival analysis across all predictor datasets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.exceptions import ConvergenceWarning
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning as SklearnConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from causal_llm.evaluation.extraction_eval import normalize_optional_text, parse_boolean

DEFAULT_ANALYSIS_MANIFEST = Path("derived_data/analysis/analysis_dataset_manifest.json")
DEFAULT_PERTURBATION_MANIFEST = Path(
    "derived_data/perturbations/time_shift_perturbation_manifest.json"
)
DEFAULT_MIMIC_ROOT = Path("physionet.org/files/mimiciv/3.1")
DEFAULT_OUTPUT_DIR = Path("derived_data/causal_analysis")
DEFAULT_REPORT_DIR = Path("reports/causal_analysis")
DEFAULT_MAX_ALLOWED_WEIGHT = 100.0
DEFAULT_SEVERE_OVERLAP_MIN_WIDTH = 0.10
PRIMARY_TRUNCATION_QUANTILES = (0.01, 0.99)
SENSITIVITY_TRUNCATION_QUANTILES = (0.05, 0.95)
EPSILON = 1e-6

REQUIRED_MIMIC_FILES = {
    "admissions": "hosp/admissions.csv.gz",
    "diagnoses_icd": "hosp/diagnoses_icd.csv.gz",
    "transfers": "hosp/transfers.csv.gz",
    "labevents": "hosp/labevents.csv.gz",
    "d_labitems": "hosp/d_labitems.csv.gz",
    "chartevents": "icu/chartevents.csv.gz",
    "d_items": "icu/d_items.csv.gz",
    "inputevents": "icu/inputevents.csv.gz",
    "procedureevents": "icu/procedureevents.csv.gz",
}
REQUIRED_DATASET_COLUMNS = {
    "task_id",
    "subject_id",
    "hadm_id",
    "stay_id",
    "predictor_name",
    "predictor_type",
    "predictor_id",
    "suspected_sepsis",
    "suspicion_time",
    "analysis_eligible",
    "analysis_exclusion_reason",
    "early_antibiotic_within_window",
    "mortality_28d",
    "followup_duration_days",
    "admittime",
    "study_window_start",
    "study_window_end",
    "anchor_age",
    "gender",
    "admission_type",
}
REQUIRED_COVARIATE_COLUMNS = {
    "age_years",
    "sex",
    "admission_type",
    "pre_t0_careunit",
    "calendar_year",
    "calendar_month",
    "prior_admission_count",
    "prior_diagnosis_root_count",
    "pre_t0_heart_rate",
    "pre_t0_sbp",
    "pre_t0_temperature_c",
    "pre_t0_wbc",
    "pre_t0_lactate",
    "pre_t0_creatinine",
    "pre_t0_mechanical_ventilation",
    "pre_t0_vasopressor",
}
DETAIL_FIELDNAMES = [
    "task_id",
    "subject_id",
    "hadm_id",
    "stay_id",
    "dataset_name",
    "dataset_type",
    "dataset_id",
    "analysis_scope",
    "treatment",
    "event",
    "followup_duration_days",
    "propensity_score",
    "stabilized_weight_raw",
    "stabilized_weight_primary",
    "stabilized_weight_sensitivity",
    "overlap_weight",
    "age_years",
    "sex",
    "admission_type",
    "pre_t0_careunit",
    "calendar_year",
    "calendar_month",
    "prior_admission_count",
    "prior_diagnosis_root_count",
    "pre_t0_heart_rate",
    "pre_t0_sbp",
    "pre_t0_temperature_c",
    "pre_t0_wbc",
    "pre_t0_lactate",
    "pre_t0_creatinine",
    "pre_t0_mechanical_ventilation",
    "pre_t0_vasopressor",
]
SUMMARY_FIELDNAMES = [
    "dataset_name",
    "dataset_type",
    "dataset_id",
    "analysis_scope",
    "source_predictor_name",
    "source_predictor_type",
    "source_predictor_id",
    "synthetic_shift_hours",
    "input_row_count",
    "analysis_row_count",
    "excluded_row_count",
    "treated_count",
    "control_count",
    "event_count",
    "primary_hr",
    "primary_ci_lower",
    "primary_ci_upper",
    "sensitivity_hr",
    "sensitivity_ci_lower",
    "sensitivity_ci_upper",
    "overlap_hr",
    "overlap_ci_lower",
    "overlap_ci_upper",
    "propensity_min",
    "propensity_max",
    "treated_ps_min",
    "treated_ps_max",
    "control_ps_min",
    "control_ps_max",
    "overlap_lower",
    "overlap_upper",
    "overlap_width",
    "severe_overlap_flag",
    "primary_truncated_rate",
    "sensitivity_truncated_rate",
    "primary_truncation_lower",
    "primary_truncation_upper",
    "sensitivity_truncation_lower",
    "sensitivity_truncation_upper",
    "primary_weight_min",
    "primary_weight_p01",
    "primary_weight_p05",
    "primary_weight_p25",
    "primary_weight_p50",
    "primary_weight_p75",
    "primary_weight_p95",
    "primary_weight_p99",
    "primary_weight_max",
    "primary_weight_mean",
    "primary_weight_std",
    "convergence_status",
    "detail_csv_path",
    "survival_curve_csv_path",
]
HR_ITEMIDS = ("220045",)
SBP_ITEMIDS = ("220050", "220179", "225309", "224167", "227243")
TEMP_C_ITEMIDS = ("223762", "226329")
TEMP_F_ITEMIDS = ("223761",)
WBC_ITEMIDS = ("51516", "52407")
LACTATE_ITEMIDS = ("50813", "52442", "53154")
CREATININE_ITEMIDS = ("50912", "52546")
VASOPRESSOR_ITEMIDS = ("221906", "221289", "221749", "222315", "221653", "221662")
VENTILATION_CHARTEVENT_ITEMIDS = ("225792",)
VENTILATION_PROCEDURE_LABELS = (
    "Intubation",
    "Invasive Ventilation",
    "Percutaneous Tracheostomy",
)
MODEL_DATASET_TYPE = "model"
ALLOWED_SYNTHETIC_SCHEMA_EXTRAS = {
    "synthetic_source_predictor_name",
    "synthetic_source_predictor_type",
    "synthetic_source_predictor_id",
    "synthetic_shift_hours",
    "original_suspicion_time",
    "original_analysis_eligible",
    "original_analysis_exclusion_reason",
}


@dataclass(frozen=True)
class CausalAnalysisConfig:
    """Configuration for one causal-analysis run."""

    analysis_manifest: Path
    perturbation_manifest: Path | None
    mimic_root: Path
    output_dir: Path
    report_dir: Path
    max_allowed_weight: float
    severe_overlap_min_width: float


@dataclass(frozen=True)
class DatasetSpec:
    """One dataset file that should pass through the fixed causal pipeline."""

    dataset_name: str
    dataset_type: str
    dataset_id: str
    csv_path: Path
    analysis_scope: str
    source_predictor_name: str | None = None
    source_predictor_type: str | None = None
    source_predictor_id: str | None = None
    synthetic_shift_hours: float | None = None


@dataclass(frozen=True)
class PipelineResult:
    """Compact summary row for one fitted causal analysis."""

    dataset_name: str
    dataset_type: str
    dataset_id: str
    analysis_scope: str
    source_predictor_name: str | None
    source_predictor_type: str | None
    source_predictor_id: str | None
    synthetic_shift_hours: float | None
    input_row_count: int
    analysis_row_count: int
    excluded_row_count: int
    treated_count: int
    control_count: int
    event_count: int
    primary_hr: float | None
    primary_ci_lower: float | None
    primary_ci_upper: float | None
    sensitivity_hr: float | None
    sensitivity_ci_lower: float | None
    sensitivity_ci_upper: float | None
    overlap_hr: float | None
    overlap_ci_lower: float | None
    overlap_ci_upper: float | None
    propensity_min: float | None
    propensity_max: float | None
    treated_ps_min: float | None
    treated_ps_max: float | None
    control_ps_min: float | None
    control_ps_max: float | None
    overlap_lower: float | None
    overlap_upper: float | None
    overlap_width: float | None
    severe_overlap_flag: bool
    primary_truncated_rate: float | None
    sensitivity_truncated_rate: float | None
    primary_truncation_lower: float | None
    primary_truncation_upper: float | None
    sensitivity_truncation_lower: float | None
    sensitivity_truncation_upper: float | None
    primary_weight_min: float | None
    primary_weight_p01: float | None
    primary_weight_p05: float | None
    primary_weight_p25: float | None
    primary_weight_p50: float | None
    primary_weight_p75: float | None
    primary_weight_p95: float | None
    primary_weight_p99: float | None
    primary_weight_max: float | None
    primary_weight_mean: float | None
    primary_weight_std: float | None
    convergence_status: str
    detail_csv_path: str
    survival_curve_csv_path: str


@dataclass(frozen=True)
class CausalAnalysisRunSummary:
    """Summary artifact emitted after the full causal pipeline completes."""

    generated_at_utc: str
    analysis_manifest: str
    perturbation_manifest: str | None
    mimic_root: str
    output_dir: str
    report_path: str
    summary_csv_path: str
    summary_json_path: str
    robustness_json_path: str
    dataset_results: list[dict[str, object]]
    robustness_summary: dict[str, object]


@dataclass
class PreprocessorSpec:
    """Reference preprocessing contract shared across all fitted datasets."""

    continuous_columns: list[str]
    categorical_columns: list[str]
    binary_columns: list[str]
    transformer: ColumnTransformer


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the pre-specified causal analysis pipeline."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the fixed L2-IPTW Cox analysis across gold, x_rule, model, and "
            "optional synthetic perturbation datasets."
        )
    )
    parser.add_argument(
        "--analysis-manifest",
        type=Path,
        default=DEFAULT_ANALYSIS_MANIFEST,
        help="Manifest produced by `python build_analysis_datasets.py`.",
    )
    parser.add_argument(
        "--perturbation-manifest",
        type=Path,
        default=DEFAULT_PERTURBATION_MANIFEST,
        help=(
            "Optional manifest produced by `python generate_time_perturbations.py`. "
            "If the file does not exist it is skipped."
        ),
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
        help="Directory for causal-analysis CSV and JSON artifacts.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Directory for causal-analysis markdown reports.",
    )
    parser.add_argument(
        "--max-allowed-weight",
        type=float,
        default=DEFAULT_MAX_ALLOWED_WEIGHT,
        help="Fail the primary analysis if a truncated primary weight exceeds this value.",
    )
    parser.add_argument(
        "--severe-overlap-min-width",
        type=float,
        default=DEFAULT_SEVERE_OVERLAP_MIN_WIDTH,
        help="Flag severe lack of overlap when the propensity-score overlap width falls below this threshold.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> CausalAnalysisConfig:
    """Resolve CLI arguments into a normalized causal-analysis configuration."""

    if args.max_allowed_weight <= 0:
        raise ValueError("--max-allowed-weight must be positive.")
    if args.severe_overlap_min_width <= 0:
        raise ValueError("--severe-overlap-min-width must be positive.")

    perturbation_manifest = args.perturbation_manifest.resolve()
    if not perturbation_manifest.exists():
        perturbation_manifest = None

    return CausalAnalysisConfig(
        analysis_manifest=args.analysis_manifest.resolve(),
        perturbation_manifest=perturbation_manifest,
        mimic_root=args.mimic_root.resolve(),
        output_dir=args.output_dir.resolve(),
        report_dir=args.report_dir.resolve(),
        max_allowed_weight=float(args.max_allowed_weight),
        severe_overlap_min_width=float(args.severe_overlap_min_width),
    )


def sql_string(value: str) -> str:
    """Quote a Python string for safe interpolation into DuckDB SQL."""

    return "'" + value.replace("'", "''") + "'"


def sql_string_list(values: tuple[str, ...]) -> str:
    """Render a tuple of string literals as a SQL IN-list."""

    return ", ".join(sql_string(value) for value in values)


def round_optional(value: float | None) -> float | None:
    """Round floating-point outputs when a value is present."""

    if value is None:
        return None
    return round(float(value), 6)


def parse_optional_bool(value: Any) -> bool | None:
    """Parse optional boolean fields from CSV artifacts."""

    if value is None or pd.isna(value):
        return None
    normalized = normalize_optional_text(value)
    if normalized is None:
        return None
    return parse_boolean(normalized)


def ensure_inputs(config: CausalAnalysisConfig) -> dict[str, Path]:
    """Validate that the required analysis manifest and raw MIMIC files exist."""

    if not config.analysis_manifest.exists():
        raise FileNotFoundError(
            f"Missing analysis manifest: {config.analysis_manifest}\n"
            "Run `python build_analysis_datasets.py` first."
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


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON document from disk."""

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_dataset_specs(config: CausalAnalysisConfig) -> tuple[list[DatasetSpec], list[DatasetSpec]]:
    """Load the primary and optional synthetic dataset specs from the manifest files."""

    analysis_manifest = load_json(config.analysis_manifest)
    predictor_summaries = analysis_manifest.get("predictor_summaries")
    if not isinstance(predictor_summaries, list) or not predictor_summaries:
        raise ValueError("Analysis manifest is missing a valid `predictor_summaries` list.")

    primary_specs: list[DatasetSpec] = []
    for summary in predictor_summaries:
        primary_specs.append(
            DatasetSpec(
                dataset_name=str(summary["predictor_name"]),
                dataset_type=str(summary["predictor_type"]),
                dataset_id=str(summary["predictor_id"]),
                csv_path=Path(str(summary["csv_path"])).resolve(),
                analysis_scope="full_cohort",
            )
        )

    synthetic_specs: list[DatasetSpec] = []
    if config.perturbation_manifest is not None:
        perturbation_manifest = load_json(config.perturbation_manifest)
        perturbation_summaries = perturbation_manifest.get("perturbation_summaries")
        if not isinstance(perturbation_summaries, list):
            raise ValueError(
                "Perturbation manifest is missing a valid `perturbation_summaries` list."
            )
        for summary in perturbation_summaries:
            synthetic_specs.append(
                DatasetSpec(
                    dataset_name=Path(str(summary["csv_path"])).stem,
                    dataset_type="synthetic",
                    dataset_id=(
                        f"{summary['source_predictor_name']}|shift={summary['shift_hours']}"
                    ),
                    csv_path=Path(str(summary["csv_path"])).resolve(),
                    analysis_scope="full_cohort",
                    source_predictor_name=str(summary["source_predictor_name"]),
                    source_predictor_type=str(summary["source_predictor_type"]),
                    source_predictor_id=str(summary["source_predictor_id"]),
                    synthetic_shift_hours=float(summary["shift_hours"]),
                )
            )

    return primary_specs, synthetic_specs


def load_dataset_frame(dataset_spec: DatasetSpec) -> pd.DataFrame:
    """Load one analysis dataset CSV and validate the minimum schema."""

    frame = pd.read_csv(dataset_spec.csv_path)
    missing_columns = sorted(REQUIRED_DATASET_COLUMNS - set(frame.columns))
    if missing_columns:
        raise ValueError(
            f"Dataset {dataset_spec.dataset_name} is missing required columns: "
            + ", ".join(missing_columns)
        )
    if frame.empty:
        raise ValueError(f"Dataset {dataset_spec.dataset_name} contains no rows.")

    frame = frame.copy()
    frame["task_id"] = frame["task_id"].astype(str)
    for field_name in ("subject_id", "hadm_id", "stay_id"):
        frame[field_name] = frame[field_name].astype(int)
    frame["analysis_eligible"] = frame["analysis_eligible"].map(parse_boolean)
    frame["treatment"] = frame["early_antibiotic_within_window"].map(parse_optional_bool)
    frame["event"] = frame["mortality_28d"].map(parse_optional_bool)
    frame["followup_duration_days"] = pd.to_numeric(frame["followup_duration_days"], errors="coerce")
    frame["time_zero"] = pd.to_datetime(frame["suspicion_time"], errors="coerce")
    frame["admittime"] = pd.to_datetime(frame["admittime"], errors="coerce")
    frame["study_window_start"] = pd.to_datetime(frame["study_window_start"], errors="coerce")
    frame["study_window_end"] = pd.to_datetime(frame["study_window_end"], errors="coerce")
    frame["anchor_age"] = pd.to_numeric(frame["anchor_age"], errors="coerce")
    frame["gender"] = frame["gender"].astype(str)
    frame["admission_type"] = frame["admission_type"].astype(str)
    return frame


def validate_primary_dataset_alignment(dataset_frames: dict[str, pd.DataFrame]) -> None:
    """Fail when the primary datasets do not cover the same shared task rows."""

    dataset_names = list(dataset_frames)
    reference_name = dataset_names[0]
    reference_frame = dataset_frames[reference_name].sort_values("task_id")
    reference_index = reference_frame.set_index("task_id")

    for dataset_name in dataset_names[1:]:
        frame = dataset_frames[dataset_name].sort_values("task_id")
        task_ids = set(frame["task_id"])
        reference_task_ids = set(reference_frame["task_id"])
        if task_ids != reference_task_ids:
            missing = sorted(reference_task_ids - task_ids)
            extra = sorted(task_ids - reference_task_ids)
            raise ValueError(
                f"Dataset mismatch across runs for {dataset_name}: "
                f"missing={len(missing)} extra={len(extra)}."
            )
        merged = (
            frame.set_index("task_id")[["subject_id", "hadm_id", "stay_id"]]
            .join(
                reference_index[["subject_id", "hadm_id", "stay_id"]],
                how="inner",
                lsuffix="_left",
                rsuffix="_right",
            )
        )
        mismatched = merged[
            (merged["subject_id_left"] != merged["subject_id_right"])
            | (merged["hadm_id_left"] != merged["hadm_id_right"])
            | (merged["stay_id_left"] != merged["stay_id_right"])
        ]
        if not mismatched.empty:
            raise ValueError(
                f"Dataset identity mismatch across runs for {dataset_name}: "
                f"{len(mismatched)} task_id rows differ from {reference_name}."
            )


def validate_dataset_schema(
    dataset_name: str,
    dataset_columns: list[str],
    *,
    allowed_extra_columns: set[str] | None = None,
    reference_name: str,
    reference_columns: list[str],
) -> None:
    """Enforce the fixed cross-dataset schema contract before fitting models."""

    allowed_extra_columns = allowed_extra_columns or set()
    effective_dataset_columns = [
        column for column in dataset_columns if column not in allowed_extra_columns
    ]
    if effective_dataset_columns == reference_columns:
        return

    reference_column_set = set(reference_columns)
    effective_dataset_column_set = set(effective_dataset_columns)
    raw_extra_columns = [
        column for column in dataset_columns if column not in reference_column_set
    ]
    disallowed_extra_columns = [
        column for column in raw_extra_columns if column not in allowed_extra_columns
    ]
    missing_columns = [
        column for column in reference_columns if column not in effective_dataset_column_set
    ]
    raise ValueError(
        f"Dataset schema mismatch across runs for {dataset_name} relative to {reference_name}: "
        f"missing={missing_columns or 'none'}, extra={disallowed_extra_columns or 'none'}, "
        f"column_order_matches={effective_dataset_columns == reference_columns}."
    )


def validate_primary_dataset_schema_alignment(dataset_frames: dict[str, pd.DataFrame]) -> None:
    """Fail when primary datasets do not share an identical column contract."""

    dataset_names = list(dataset_frames)
    reference_name = dataset_names[0]
    reference_columns = dataset_frames[reference_name].columns.tolist()
    for dataset_name in dataset_names[1:]:
        validate_dataset_schema(
            dataset_name,
            dataset_frames[dataset_name].columns.tolist(),
            reference_name=reference_name,
            reference_columns=reference_columns,
        )


def initialize_connection(
    required_paths: dict[str, Path],
    all_keys_frame: pd.DataFrame,
) -> duckdb.DuckDBPyConnection:
    """Create a DuckDB connection with filtered source tables for the shared cohort."""

    connection = duckdb.connect(database=":memory:")
    connection.register("analysis_cohort_keys_df", all_keys_frame)
    connection.execute(
        """
        CREATE OR REPLACE TABLE analysis_cohort_keys AS
        SELECT DISTINCT
            CAST(subject_id AS BIGINT) AS subject_id,
            CAST(hadm_id AS BIGINT) AS hadm_id,
            CAST(stay_id AS BIGINT) AS stay_id
        FROM analysis_cohort_keys_df;
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
    connection.execute(
        f"""
        CREATE OR REPLACE VIEW src_diagnoses_icd AS
        SELECT *
        FROM read_csv_auto(
            {sql_string(str(required_paths['diagnoses_icd']))},
            header = TRUE,
            all_varchar = TRUE
        );
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE VIEW src_transfers AS
        SELECT *
        FROM read_csv_auto(
            {sql_string(str(required_paths['transfers']))},
            header = TRUE,
            all_varchar = TRUE
        );
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE VIEW src_labevents AS
        SELECT *
        FROM read_csv_auto(
            {sql_string(str(required_paths['labevents']))},
            header = TRUE,
            all_varchar = TRUE
        );
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE VIEW src_chartevents AS
        SELECT *
        FROM read_csv_auto(
            {sql_string(str(required_paths['chartevents']))},
            header = TRUE,
            all_varchar = TRUE
        );
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE VIEW src_inputevents AS
        SELECT *
        FROM read_csv_auto(
            {sql_string(str(required_paths['inputevents']))},
            header = TRUE,
            all_varchar = TRUE
        );
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE VIEW src_procedureevents AS
        SELECT *
        FROM read_csv_auto(
            {sql_string(str(required_paths['procedureevents']))},
            header = TRUE,
            all_varchar = TRUE
        );
        """
    )

    connection.execute(
        """
        CREATE OR REPLACE TABLE filtered_admissions AS
        SELECT
            TRY_CAST(adm.subject_id AS BIGINT) AS subject_id,
            TRY_CAST(adm.hadm_id AS BIGINT) AS hadm_id,
            TRY_CAST(adm.admittime AS TIMESTAMP) AS admittime
        FROM src_admissions AS adm
        INNER JOIN (
            SELECT DISTINCT subject_id
            FROM analysis_cohort_keys
        ) AS keys
            ON TRY_CAST(adm.subject_id AS BIGINT) = keys.subject_id;
        """
    )
    connection.execute(
        """
        CREATE OR REPLACE TABLE filtered_diagnoses_icd AS
        SELECT
            TRY_CAST(dx.subject_id AS BIGINT) AS subject_id,
            TRY_CAST(dx.hadm_id AS BIGINT) AS hadm_id,
            UPPER(REPLACE(COALESCE(dx.icd_code, ''), '.', '')) AS normalized_icd_code
        FROM src_diagnoses_icd AS dx
        INNER JOIN (
            SELECT DISTINCT subject_id
            FROM analysis_cohort_keys
        ) AS keys
            ON TRY_CAST(dx.subject_id AS BIGINT) = keys.subject_id;
        """
    )
    connection.execute(
        """
        CREATE OR REPLACE TABLE filtered_transfers AS
        SELECT
            TRY_CAST(transfer.subject_id AS BIGINT) AS subject_id,
            TRY_CAST(transfer.hadm_id AS BIGINT) AS hadm_id,
            TRY_CAST(transfer.transfer_id AS BIGINT) AS transfer_id,
            NULLIF(TRIM(COALESCE(transfer.careunit, '')), '') AS careunit,
            TRY_CAST(transfer.intime AS TIMESTAMP) AS intime,
            TRY_CAST(transfer.outtime AS TIMESTAMP) AS outtime
        FROM src_transfers AS transfer
        INNER JOIN (
            SELECT DISTINCT subject_id, hadm_id
            FROM analysis_cohort_keys
        ) AS keys
            ON TRY_CAST(transfer.subject_id AS BIGINT) = keys.subject_id
           AND TRY_CAST(transfer.hadm_id AS BIGINT) = keys.hadm_id;
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TABLE filtered_chartevents AS
        SELECT
            TRY_CAST(event.subject_id AS BIGINT) AS subject_id,
            TRY_CAST(event.hadm_id AS BIGINT) AS hadm_id,
            TRY_CAST(event.stay_id AS BIGINT) AS stay_id,
            TRY_CAST(event.charttime AS TIMESTAMP) AS charttime,
            TRY_CAST(event.storetime AS TIMESTAMP) AS storetime,
            event.itemid AS itemid,
            TRY_CAST(event.valuenum AS DOUBLE) AS valuenum,
            NULLIF(TRIM(COALESCE(event.value, '')), '') AS value
        FROM src_chartevents AS event
        INNER JOIN (
            SELECT DISTINCT stay_id
            FROM analysis_cohort_keys
        ) AS keys
            ON TRY_CAST(event.stay_id AS BIGINT) = keys.stay_id
        WHERE itemid IN ({sql_string_list(HR_ITEMIDS + SBP_ITEMIDS + TEMP_C_ITEMIDS + TEMP_F_ITEMIDS + VENTILATION_CHARTEVENT_ITEMIDS)})
          AND (
              TRY_CAST(event.valuenum AS DOUBLE) IS NOT NULL
              OR event.itemid IN ('225792')
          );
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TABLE filtered_labevents AS
        SELECT
            TRY_CAST(lab.subject_id AS BIGINT) AS subject_id,
            TRY_CAST(lab.hadm_id AS BIGINT) AS hadm_id,
            TRY_CAST(lab.charttime AS TIMESTAMP) AS charttime,
            TRY_CAST(lab.storetime AS TIMESTAMP) AS storetime,
            lab.itemid AS itemid,
            TRY_CAST(lab.valuenum AS DOUBLE) AS valuenum
        FROM src_labevents AS lab
        INNER JOIN (
            SELECT DISTINCT subject_id, hadm_id
            FROM analysis_cohort_keys
        ) AS keys
            ON TRY_CAST(lab.subject_id AS BIGINT) = keys.subject_id
           AND TRY_CAST(lab.hadm_id AS BIGINT) = keys.hadm_id
        WHERE itemid IN ({sql_string_list(WBC_ITEMIDS + LACTATE_ITEMIDS + CREATININE_ITEMIDS)})
          AND TRY_CAST(lab.valuenum AS DOUBLE) IS NOT NULL;
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TABLE filtered_inputevents AS
        SELECT
            TRY_CAST(input_event.subject_id AS BIGINT) AS subject_id,
            TRY_CAST(input_event.hadm_id AS BIGINT) AS hadm_id,
            TRY_CAST(input_event.stay_id AS BIGINT) AS stay_id,
            TRY_CAST(input_event.starttime AS TIMESTAMP) AS starttime,
            TRY_CAST(input_event.endtime AS TIMESTAMP) AS endtime,
            input_event.itemid AS itemid
        FROM src_inputevents AS input_event
        INNER JOIN (
            SELECT DISTINCT stay_id
            FROM analysis_cohort_keys
        ) AS keys
            ON TRY_CAST(input_event.stay_id AS BIGINT) = keys.stay_id
        WHERE itemid IN ({sql_string_list(VASOPRESSOR_ITEMIDS)});
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TABLE filtered_procedureevents AS
        SELECT
            TRY_CAST(p.subject_id AS BIGINT) AS subject_id,
            TRY_CAST(p.hadm_id AS BIGINT) AS hadm_id,
            TRY_CAST(p.stay_id AS BIGINT) AS stay_id,
            TRY_CAST(p.starttime AS TIMESTAMP) AS starttime,
            TRY_CAST(p.endtime AS TIMESTAMP) AS endtime,
            NULLIF(TRIM(COALESCE(d.label, '')), '') AS label
        FROM src_procedureevents AS p
        INNER JOIN (
            SELECT DISTINCT stay_id
            FROM analysis_cohort_keys
        ) AS keys
            ON TRY_CAST(p.stay_id AS BIGINT) = keys.stay_id
        LEFT JOIN read_csv_auto(
            {sql_string(str(required_paths['d_items']))},
            header = TRUE,
            all_varchar = TRUE
        ) AS d
            ON p.itemid = d.itemid
        WHERE NULLIF(TRIM(COALESCE(d.label, '')), '') IN ({sql_string_list(VENTILATION_PROCEDURE_LABELS)});
        """
    )

    return connection


def register_dataset_rows(connection: duckdb.DuckDBPyConnection, frame: pd.DataFrame) -> None:
    """Register one dataset frame as a DuckDB temp table for covariate derivation."""

    minimal_frame = frame[
        [
            "task_id",
            "subject_id",
            "hadm_id",
            "stay_id",
            "time_zero",
            "admittime",
            "study_window_start",
            "study_window_end",
            "anchor_age",
            "gender",
            "admission_type",
        ]
    ].copy()
    connection.register("dataset_rows_df", minimal_frame)
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE dataset_rows AS
        SELECT
            CAST(task_id AS VARCHAR) AS task_id,
            CAST(subject_id AS BIGINT) AS subject_id,
            CAST(hadm_id AS BIGINT) AS hadm_id,
            CAST(stay_id AS BIGINT) AS stay_id,
            CAST(time_zero AS TIMESTAMP) AS time_zero,
            CAST(admittime AS TIMESTAMP) AS admittime,
            CAST(study_window_start AS TIMESTAMP) AS study_window_start,
            CAST(study_window_end AS TIMESTAMP) AS study_window_end,
            CAST(anchor_age AS DOUBLE) AS anchor_age,
            CAST(gender AS VARCHAR) AS gender,
            CAST(admission_type AS VARCHAR) AS admission_type
        FROM dataset_rows_df;
        """
    )


def derive_covariates(
    connection: duckdb.DuckDBPyConnection,
    dataset_spec: DatasetSpec,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Derive the fixed pre-t0 covariate set for one dataset."""

    register_dataset_rows(connection, frame)
    covariate_frame = connection.execute(
        """
        WITH careunit AS (
            SELECT
                task_id,
                careunit AS pre_t0_careunit
            FROM (
                SELECT
                    row.task_id,
                    transfer.careunit,
                    ROW_NUMBER() OVER (
                        PARTITION BY row.task_id
                        ORDER BY transfer.intime DESC, transfer.transfer_id DESC
                    ) AS row_rank
                FROM dataset_rows AS row
                INNER JOIN filtered_transfers AS transfer
                    ON transfer.subject_id = row.subject_id
                   AND transfer.hadm_id = row.hadm_id
                WHERE transfer.intime < row.time_zero
                  AND transfer.careunit IS NOT NULL
            )
            WHERE row_rank = 1
        ),
        prior_admissions AS (
            SELECT
                row.task_id,
                COUNT(DISTINCT prior_adm.hadm_id) AS prior_admission_count
            FROM dataset_rows AS row
            LEFT JOIN filtered_admissions AS prior_adm
                ON prior_adm.subject_id = row.subject_id
               AND prior_adm.admittime < row.admittime
            GROUP BY row.task_id
        ),
        prior_dx_burden AS (
            SELECT
                row.task_id,
                COUNT(
                    DISTINCT NULLIF(
                        regexp_extract(dx.normalized_icd_code, '^[A-Z]?[0-9]{2,3}'),
                        ''
                    )
                ) AS prior_diagnosis_root_count
            FROM dataset_rows AS row
            LEFT JOIN filtered_admissions AS prior_adm
                ON prior_adm.subject_id = row.subject_id
               AND prior_adm.admittime < row.admittime
            LEFT JOIN filtered_diagnoses_icd AS dx
                ON dx.subject_id = row.subject_id
               AND dx.hadm_id = prior_adm.hadm_id
            GROUP BY row.task_id
        ),
        latest_heart_rate AS (
            SELECT
                task_id,
                valuenum AS pre_t0_heart_rate
            FROM (
                SELECT
                    row.task_id,
                    event.valuenum,
                    ROW_NUMBER() OVER (
                        PARTITION BY row.task_id
                        ORDER BY event.charttime DESC, event.storetime DESC NULLS LAST
                    ) AS row_rank
                FROM dataset_rows AS row
                INNER JOIN filtered_chartevents AS event
                    ON event.stay_id = row.stay_id
                WHERE event.itemid IN ('220045')
                  AND event.charttime >= row.study_window_start
                  AND event.charttime < row.time_zero
                  AND event.valuenum BETWEEN 20 AND 250
            )
            WHERE row_rank = 1
        ),
        latest_sbp AS (
            SELECT
                task_id,
                valuenum AS pre_t0_sbp
            FROM (
                SELECT
                    row.task_id,
                    event.valuenum,
                    ROW_NUMBER() OVER (
                        PARTITION BY row.task_id
                        ORDER BY event.charttime DESC, event.storetime DESC NULLS LAST
                    ) AS row_rank
                FROM dataset_rows AS row
                INNER JOIN filtered_chartevents AS event
                    ON event.stay_id = row.stay_id
                WHERE event.itemid IN ('220050', '220179', '225309', '224167', '227243')
                  AND event.charttime >= row.study_window_start
                  AND event.charttime < row.time_zero
                  AND event.valuenum BETWEEN 30 AND 300
            )
            WHERE row_rank = 1
        ),
        latest_temperature AS (
            SELECT
                task_id,
                temperature_c AS pre_t0_temperature_c
            FROM (
                SELECT
                    row.task_id,
                    CASE
                        WHEN event.itemid = '223761'
                            THEN (event.valuenum - 32.0) * (5.0 / 9.0)
                        ELSE event.valuenum
                    END AS temperature_c,
                    ROW_NUMBER() OVER (
                        PARTITION BY row.task_id
                        ORDER BY event.charttime DESC, event.storetime DESC NULLS LAST
                    ) AS row_rank
                FROM dataset_rows AS row
                INNER JOIN filtered_chartevents AS event
                    ON event.stay_id = row.stay_id
                WHERE event.itemid IN ('223762', '226329', '223761')
                  AND event.charttime >= row.study_window_start
                  AND event.charttime < row.time_zero
                  AND CASE
                        WHEN event.itemid = '223761'
                            THEN (event.valuenum - 32.0) * (5.0 / 9.0)
                        ELSE event.valuenum
                      END BETWEEN 25 AND 45
            )
            WHERE row_rank = 1
        ),
        latest_wbc AS (
            SELECT
                task_id,
                valuenum AS pre_t0_wbc
            FROM (
                SELECT
                    row.task_id,
                    lab.valuenum,
                    ROW_NUMBER() OVER (
                        PARTITION BY row.task_id
                        ORDER BY lab.charttime DESC, lab.storetime DESC NULLS LAST
                    ) AS row_rank
                FROM dataset_rows AS row
                INNER JOIN filtered_labevents AS lab
                    ON lab.subject_id = row.subject_id
                   AND lab.hadm_id = row.hadm_id
                WHERE lab.itemid IN ('51516', '52407')
                  AND lab.charttime >= row.study_window_start
                  AND lab.charttime < row.time_zero
                  AND lab.valuenum BETWEEN 0 AND 500
            )
            WHERE row_rank = 1
        ),
        latest_lactate AS (
            SELECT
                task_id,
                valuenum AS pre_t0_lactate
            FROM (
                SELECT
                    row.task_id,
                    lab.valuenum,
                    ROW_NUMBER() OVER (
                        PARTITION BY row.task_id
                        ORDER BY lab.charttime DESC, lab.storetime DESC NULLS LAST
                    ) AS row_rank
                FROM dataset_rows AS row
                INNER JOIN filtered_labevents AS lab
                    ON lab.subject_id = row.subject_id
                   AND lab.hadm_id = row.hadm_id
                WHERE lab.itemid IN ('50813', '52442', '53154')
                  AND lab.charttime >= row.study_window_start
                  AND lab.charttime < row.time_zero
                  AND lab.valuenum BETWEEN 0 AND 50
            )
            WHERE row_rank = 1
        ),
        latest_creatinine AS (
            SELECT
                task_id,
                valuenum AS pre_t0_creatinine
            FROM (
                SELECT
                    row.task_id,
                    lab.valuenum,
                    ROW_NUMBER() OVER (
                        PARTITION BY row.task_id
                        ORDER BY lab.charttime DESC, lab.storetime DESC NULLS LAST
                    ) AS row_rank
                FROM dataset_rows AS row
                INNER JOIN filtered_labevents AS lab
                    ON lab.subject_id = row.subject_id
                   AND lab.hadm_id = row.hadm_id
                WHERE lab.itemid IN ('50912', '52546')
                  AND lab.charttime >= row.study_window_start
                  AND lab.charttime < row.time_zero
                  AND lab.valuenum BETWEEN 0 AND 30
            )
            WHERE row_rank = 1
        ),
        ventilation_flag AS (
            SELECT
                row.task_id,
                (
                    MAX(
                        CASE
                            WHEN chart_event.stay_id IS NOT NULL THEN 1
                            ELSE 0
                        END
                    ) = 1
                    OR MAX(
                        CASE
                            WHEN proc_event.stay_id IS NOT NULL THEN 1
                            ELSE 0
                        END
                    ) = 1
                ) AS pre_t0_mechanical_ventilation
            FROM dataset_rows AS row
            LEFT JOIN filtered_chartevents AS chart_event
                ON chart_event.stay_id = row.stay_id
               AND chart_event.itemid IN ('225792')
               AND chart_event.charttime >= row.study_window_start
               AND chart_event.charttime < row.time_zero
            LEFT JOIN filtered_procedureevents AS proc_event
                ON proc_event.stay_id = row.stay_id
               AND proc_event.starttime >= row.study_window_start
               AND proc_event.starttime < row.time_zero
            GROUP BY row.task_id
        ),
        vasopressor_flag AS (
            SELECT
                row.task_id,
                MAX(
                    CASE
                        WHEN input_event.stay_id IS NOT NULL THEN 1
                        ELSE 0
                    END
                ) = 1 AS pre_t0_vasopressor
            FROM dataset_rows AS row
            LEFT JOIN filtered_inputevents AS input_event
                ON input_event.stay_id = row.stay_id
               AND input_event.starttime >= row.study_window_start
               AND input_event.starttime < row.time_zero
            GROUP BY row.task_id
        )
        SELECT
            row.task_id,
            row.anchor_age AS age_years,
            NULLIF(TRIM(COALESCE(row.gender, '')), '') AS sex,
            NULLIF(TRIM(COALESCE(row.admission_type, '')), '') AS admission_type,
            careunit.pre_t0_careunit,
            EXTRACT(YEAR FROM row.time_zero) AS calendar_year,
            EXTRACT(MONTH FROM row.time_zero) AS calendar_month,
            COALESCE(prior_admissions.prior_admission_count, 0) AS prior_admission_count,
            COALESCE(prior_dx_burden.prior_diagnosis_root_count, 0) AS prior_diagnosis_root_count,
            latest_heart_rate.pre_t0_heart_rate,
            latest_sbp.pre_t0_sbp,
            latest_temperature.pre_t0_temperature_c,
            latest_wbc.pre_t0_wbc,
            latest_lactate.pre_t0_lactate,
            latest_creatinine.pre_t0_creatinine,
            COALESCE(ventilation_flag.pre_t0_mechanical_ventilation, FALSE) AS pre_t0_mechanical_ventilation,
            COALESCE(vasopressor_flag.pre_t0_vasopressor, FALSE) AS pre_t0_vasopressor
        FROM dataset_rows AS row
        LEFT JOIN careunit
            ON row.task_id = careunit.task_id
        LEFT JOIN prior_admissions
            ON row.task_id = prior_admissions.task_id
        LEFT JOIN prior_dx_burden
            ON row.task_id = prior_dx_burden.task_id
        LEFT JOIN latest_heart_rate
            ON row.task_id = latest_heart_rate.task_id
        LEFT JOIN latest_sbp
            ON row.task_id = latest_sbp.task_id
        LEFT JOIN latest_temperature
            ON row.task_id = latest_temperature.task_id
        LEFT JOIN latest_wbc
            ON row.task_id = latest_wbc.task_id
        LEFT JOIN latest_lactate
            ON row.task_id = latest_lactate.task_id
        LEFT JOIN latest_creatinine
            ON row.task_id = latest_creatinine.task_id
        LEFT JOIN ventilation_flag
            ON row.task_id = ventilation_flag.task_id
        LEFT JOIN vasopressor_flag
            ON row.task_id = vasopressor_flag.task_id
        ORDER BY row.task_id;
        """
    ).df()

    overlapping_columns = [
        column
        for column in covariate_frame.columns
        if column != "task_id" and column in frame.columns
    ]
    merged = frame.drop(columns=overlapping_columns).merge(
        covariate_frame,
        on="task_id",
        how="left",
        validate="one_to_one",
    )
    eligible_rows = merged["analysis_eligible"] == True  # noqa: E712
    missing_covariates = {
        column: int(merged.loc[eligible_rows, column].isna().sum())
        for column in REQUIRED_COVARIATE_COLUMNS
    }
    missing_covariates = {
        column: count for column, count in missing_covariates.items() if count > 0
    }
    if missing_covariates:
        missing_text = ", ".join(
            f"{column}={count}" for column, count in sorted(missing_covariates.items())
        )
        raise ValueError(
            f"Missing covariates for analysis-eligible rows in {dataset_spec.dataset_name}: "
            f"{missing_text}."
        )

    return merged


def build_preprocessor(
    reference_frame: pd.DataFrame,
    union_frame: pd.DataFrame,
) -> PreprocessorSpec:
    """Fit one shared preprocessing contract on the reference dataset."""

    continuous_columns = [
        "age_years",
        "calendar_year",
        "calendar_month",
        "prior_admission_count",
        "prior_diagnosis_root_count",
        "pre_t0_heart_rate",
        "pre_t0_sbp",
        "pre_t0_temperature_c",
        "pre_t0_wbc",
        "pre_t0_lactate",
        "pre_t0_creatinine",
    ]
    categorical_columns = [
        "sex",
        "admission_type",
        "pre_t0_careunit",
    ]
    binary_columns = [
        "pre_t0_mechanical_ventilation",
        "pre_t0_vasopressor",
    ]

    encoder_categories: list[list[str]] = []
    for column in categorical_columns:
        encoder_categories.append(
            sorted(union_frame[column].dropna().astype(str).unique().tolist())
        )

    transformer = ColumnTransformer(
        transformers=[
            ("continuous", StandardScaler(), continuous_columns),
            (
                "categorical",
                OneHotEncoder(
                    categories=encoder_categories,
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categorical_columns,
            ),
            ("binary", "passthrough", binary_columns),
        ],
        remainder="drop",
    )
    transformer.fit(reference_frame[continuous_columns + categorical_columns + binary_columns])
    return PreprocessorSpec(
        continuous_columns=continuous_columns,
        categorical_columns=categorical_columns,
        binary_columns=binary_columns,
        transformer=transformer,
    )


def transform_covariates(
    frame: pd.DataFrame,
    preprocessor: PreprocessorSpec,
) -> np.ndarray:
    """Apply the shared preprocessing contract to one dataset frame."""

    covariate_columns = (
        preprocessor.continuous_columns
        + preprocessor.categorical_columns
        + preprocessor.binary_columns
    )
    return preprocessor.transformer.transform(frame[covariate_columns])


def truncate_weights(
    weights: np.ndarray,
    quantiles: tuple[float, float],
) -> tuple[np.ndarray, float, float, float]:
    """Truncate weights at fixed quantiles and report the truncation rate."""

    lower_bound = float(np.quantile(weights, quantiles[0]))
    upper_bound = float(np.quantile(weights, quantiles[1]))
    truncated = np.clip(weights, lower_bound, upper_bound)
    truncated_rate = float(np.mean(truncated != weights))
    return truncated, truncated_rate, lower_bound, upper_bound


def build_weight_summary(weights: np.ndarray) -> dict[str, float]:
    """Return a fixed summary of one weight distribution."""

    return {
        "min": float(np.min(weights)),
        "p01": float(np.quantile(weights, 0.01)),
        "p05": float(np.quantile(weights, 0.05)),
        "p25": float(np.quantile(weights, 0.25)),
        "p50": float(np.quantile(weights, 0.50)),
        "p75": float(np.quantile(weights, 0.75)),
        "p95": float(np.quantile(weights, 0.95)),
        "p99": float(np.quantile(weights, 0.99)),
        "max": float(np.max(weights)),
        "mean": float(np.mean(weights)),
        "std": float(np.std(weights)),
    }


def compute_overlap_summary(
    propensity_scores: np.ndarray,
    treatment: np.ndarray,
    severe_overlap_min_width: float,
) -> dict[str, Any]:
    """Compute the propensity-score overlap diagnostics required by the contract."""

    treated_scores = propensity_scores[treatment == 1]
    control_scores = propensity_scores[treatment == 0]
    treated_min = float(np.min(treated_scores))
    treated_max = float(np.max(treated_scores))
    control_min = float(np.min(control_scores))
    control_max = float(np.max(control_scores))
    raw_overlap_lower = max(treated_min, control_min)
    raw_overlap_upper = min(treated_max, control_max)
    overlap_width = max(0.0, raw_overlap_upper - raw_overlap_lower)
    if raw_overlap_upper >= raw_overlap_lower:
        overlap_lower: float | None = raw_overlap_lower
        overlap_upper: float | None = raw_overlap_upper
    else:
        overlap_lower = None
        overlap_upper = None
    return {
        "propensity_min": float(np.min(propensity_scores)),
        "propensity_max": float(np.max(propensity_scores)),
        "treated_ps_min": treated_min,
        "treated_ps_max": treated_max,
        "control_ps_min": control_min,
        "control_ps_max": control_max,
        "overlap_lower": overlap_lower,
        "overlap_upper": overlap_upper,
        "overlap_width": overlap_width,
        "severe_overlap_flag": overlap_width < severe_overlap_min_width,
    }


def fit_propensity_model(
    design_matrix: np.ndarray,
    treatment: np.ndarray,
) -> LogisticRegression:
    """Fit the fixed L2-regularized propensity model."""

    model = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=2000,
        fit_intercept=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", SklearnConvergenceWarning)
        model.fit(design_matrix, treatment)
    return model


def fit_weighted_cox(
    frame: pd.DataFrame,
    *,
    weight_column: str,
) -> tuple[float, float, float]:
    """Fit the weighted Cox model and return HR plus 95% confidence interval."""

    cox_frame = frame[["followup_duration_days", "event", "treatment", weight_column]].copy()
    cox_frame["event"] = cox_frame["event"].astype(int)
    cox_frame["treatment"] = cox_frame["treatment"].astype(int)
    cox_frame = cox_frame.rename(columns={weight_column: "analysis_weight"})

    cox_model = CoxPHFitter()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        cox_model.fit(
            cox_frame,
            duration_col="followup_duration_days",
            event_col="event",
            weights_col="analysis_weight",
            formula="treatment",
            robust=True,
        )

    coefficient = float(cox_model.params_["treatment"])
    confidence_interval = cox_model.confidence_intervals_.loc["treatment"]
    return (
        round_optional(math.exp(coefficient)),
        round_optional(math.exp(float(confidence_interval.iloc[0]))),
        round_optional(math.exp(float(confidence_interval.iloc[1]))),
    )


def build_survival_curves(frame: pd.DataFrame, weight_column: str) -> pd.DataFrame:
    """Build weighted Kaplan-Meier curves for the treated and control groups."""

    curve_frames: list[pd.DataFrame] = []
    for treatment_value, treatment_label in ((0, "control"), (1, "treated")):
        group = frame[frame["treatment"] == treatment_value]
        kmf = KaplanMeierFitter()
        kmf.fit(
            durations=group["followup_duration_days"],
            event_observed=group["event"].astype(int),
            weights=group[weight_column],
            label=treatment_label,
        )
        curve_frame = kmf.survival_function_.reset_index()
        curve_frame.columns = ["timeline_days", "survival_probability"]
        curve_frame["treatment_group"] = treatment_label
        curve_frames.append(curve_frame)
    return pd.concat(curve_frames, ignore_index=True)


def analyze_one_dataset(
    dataset_spec: DatasetSpec,
    frame: pd.DataFrame,
    *,
    preprocessor: PreprocessorSpec,
    config: CausalAnalysisConfig,
    output_dir: Path,
    fail_on_error: bool,
) -> tuple[PipelineResult, pd.DataFrame]:
    """Run the fixed causal pipeline for one prepared dataset frame."""

    input_row_count = len(frame)
    analysis_frame = frame[frame["analysis_eligible"] == True].copy()  # noqa: E712
    excluded_row_count = input_row_count - len(analysis_frame)

    try:
        if analysis_frame.empty:
            raise ValueError("No analysis-eligible rows remain after applying the explicit eligibility flag.")
        if analysis_frame["treatment"].isna().any():
            raise ValueError("Treatment labels are missing for analysis-eligible rows.")
        if analysis_frame["event"].isna().any():
            raise ValueError("Outcome labels are missing for analysis-eligible rows.")
        if analysis_frame["followup_duration_days"].isna().any():
            raise ValueError("Follow-up durations are missing for analysis-eligible rows.")
        if (analysis_frame["followup_duration_days"] <= 0).any():
            raise ValueError("Follow-up durations must be strictly positive for analysis-eligible rows.")

        treatment = analysis_frame["treatment"].astype(int).to_numpy()
        if treatment.min() == treatment.max():
            raise ValueError("Treatment has no variation in the analysis-eligible cohort.")
        event = analysis_frame["event"].astype(int).to_numpy()
        if event.min() == event.max():
            raise ValueError("Outcome has no variation in the analysis-eligible cohort.")

        design_matrix = transform_covariates(analysis_frame, preprocessor)
        propensity_model = fit_propensity_model(design_matrix, treatment)
        propensity_scores = propensity_model.predict_proba(design_matrix)[:, 1]
        propensity_scores = np.clip(propensity_scores, EPSILON, 1 - EPSILON)

        marginal_treated_probability = float(np.mean(treatment))
        stabilized_weight_raw = np.where(
            treatment == 1,
            marginal_treated_probability / propensity_scores,
            (1 - marginal_treated_probability) / (1 - propensity_scores),
        )
        (
            stabilized_weight_primary,
            primary_truncated_rate,
            primary_truncation_lower,
            primary_truncation_upper,
        ) = truncate_weights(
            stabilized_weight_raw,
            PRIMARY_TRUNCATION_QUANTILES,
        )
        (
            stabilized_weight_sensitivity,
            sensitivity_truncated_rate,
            sensitivity_truncation_lower,
            sensitivity_truncation_upper,
        ) = truncate_weights(
            stabilized_weight_raw,
            SENSITIVITY_TRUNCATION_QUANTILES,
        )
        overlap_weight = np.where(treatment == 1, 1 - propensity_scores, propensity_scores)

        if float(np.max(stabilized_weight_primary)) > config.max_allowed_weight:
            raise ValueError(
                "Extreme primary weights remain after truncation: "
                f"max={np.max(stabilized_weight_primary):.6f}."
            )

        overlap_summary = compute_overlap_summary(
            propensity_scores,
            treatment,
            config.severe_overlap_min_width,
        )
        weight_summary = build_weight_summary(stabilized_weight_primary)

        analysis_frame = analysis_frame.copy()
        analysis_frame["treatment"] = treatment
        analysis_frame["event"] = event
        analysis_frame["propensity_score"] = propensity_scores
        analysis_frame["stabilized_weight_raw"] = stabilized_weight_raw
        analysis_frame["stabilized_weight_primary"] = stabilized_weight_primary
        analysis_frame["stabilized_weight_sensitivity"] = stabilized_weight_sensitivity
        analysis_frame["overlap_weight"] = overlap_weight

        primary_hr, primary_ci_lower, primary_ci_upper = fit_weighted_cox(
            analysis_frame,
            weight_column="stabilized_weight_primary",
        )
        sensitivity_hr, sensitivity_ci_lower, sensitivity_ci_upper = fit_weighted_cox(
            analysis_frame,
            weight_column="stabilized_weight_sensitivity",
        )
        overlap_hr, overlap_ci_lower, overlap_ci_upper = fit_weighted_cox(
            analysis_frame,
            weight_column="overlap_weight",
        )
        survival_curves = build_survival_curves(
            analysis_frame,
            "stabilized_weight_primary",
        )

        dataset_slug = f"{dataset_spec.dataset_name}_{dataset_spec.analysis_scope}"
        detail_csv_path = output_dir / f"causal_detail_{dataset_slug}.csv"
        survival_curve_csv_path = output_dir / f"survival_curve_{dataset_slug}.csv"
        analysis_frame = analysis_frame.copy()
        analysis_frame["dataset_name"] = dataset_spec.dataset_name
        analysis_frame["dataset_type"] = dataset_spec.dataset_type
        analysis_frame["dataset_id"] = dataset_spec.dataset_id
        analysis_frame["analysis_scope"] = dataset_spec.analysis_scope
        analysis_frame.to_csv(detail_csv_path, index=False, columns=DETAIL_FIELDNAMES)
        survival_curves.to_csv(survival_curve_csv_path, index=False)

        result = PipelineResult(
            dataset_name=dataset_spec.dataset_name,
            dataset_type=dataset_spec.dataset_type,
            dataset_id=dataset_spec.dataset_id,
            analysis_scope=dataset_spec.analysis_scope,
            source_predictor_name=dataset_spec.source_predictor_name,
            source_predictor_type=dataset_spec.source_predictor_type,
            source_predictor_id=dataset_spec.source_predictor_id,
            synthetic_shift_hours=dataset_spec.synthetic_shift_hours,
            input_row_count=input_row_count,
            analysis_row_count=len(analysis_frame),
            excluded_row_count=excluded_row_count,
            treated_count=int(np.sum(treatment == 1)),
            control_count=int(np.sum(treatment == 0)),
            event_count=int(np.sum(event == 1)),
            primary_hr=primary_hr,
            primary_ci_lower=primary_ci_lower,
            primary_ci_upper=primary_ci_upper,
            sensitivity_hr=sensitivity_hr,
            sensitivity_ci_lower=sensitivity_ci_lower,
            sensitivity_ci_upper=sensitivity_ci_upper,
            overlap_hr=overlap_hr,
            overlap_ci_lower=overlap_ci_lower,
            overlap_ci_upper=overlap_ci_upper,
            propensity_min=round_optional(overlap_summary["propensity_min"]),
            propensity_max=round_optional(overlap_summary["propensity_max"]),
            treated_ps_min=round_optional(overlap_summary["treated_ps_min"]),
            treated_ps_max=round_optional(overlap_summary["treated_ps_max"]),
            control_ps_min=round_optional(overlap_summary["control_ps_min"]),
            control_ps_max=round_optional(overlap_summary["control_ps_max"]),
            overlap_lower=round_optional(overlap_summary["overlap_lower"]),
            overlap_upper=round_optional(overlap_summary["overlap_upper"]),
            overlap_width=round_optional(overlap_summary["overlap_width"]),
            severe_overlap_flag=bool(overlap_summary["severe_overlap_flag"]),
            primary_truncated_rate=round_optional(primary_truncated_rate),
            sensitivity_truncated_rate=round_optional(sensitivity_truncated_rate),
            primary_truncation_lower=round_optional(primary_truncation_lower),
            primary_truncation_upper=round_optional(primary_truncation_upper),
            sensitivity_truncation_lower=round_optional(sensitivity_truncation_lower),
            sensitivity_truncation_upper=round_optional(sensitivity_truncation_upper),
            primary_weight_min=round_optional(weight_summary["min"]),
            primary_weight_p01=round_optional(weight_summary["p01"]),
            primary_weight_p05=round_optional(weight_summary["p05"]),
            primary_weight_p25=round_optional(weight_summary["p25"]),
            primary_weight_p50=round_optional(weight_summary["p50"]),
            primary_weight_p75=round_optional(weight_summary["p75"]),
            primary_weight_p95=round_optional(weight_summary["p95"]),
            primary_weight_p99=round_optional(weight_summary["p99"]),
            primary_weight_max=round_optional(weight_summary["max"]),
            primary_weight_mean=round_optional(weight_summary["mean"]),
            primary_weight_std=round_optional(weight_summary["std"]),
            convergence_status="ok",
            detail_csv_path=str(detail_csv_path),
            survival_curve_csv_path=str(survival_curve_csv_path),
        )
        return result, analysis_frame
    except Exception as exc:
        if fail_on_error:
            raise

        empty_detail_path = output_dir / (
            f"causal_detail_{dataset_spec.dataset_name}_{dataset_spec.analysis_scope}.csv"
        )
        empty_survival_path = output_dir / (
            f"survival_curve_{dataset_spec.dataset_name}_{dataset_spec.analysis_scope}.csv"
        )
        empty_frame = pd.DataFrame(columns=DETAIL_FIELDNAMES)
        empty_frame.to_csv(empty_detail_path, index=False)
        pd.DataFrame(columns=["timeline_days", "survival_probability", "treatment_group"]).to_csv(
            empty_survival_path,
            index=False,
        )
        failure_result = PipelineResult(
            dataset_name=dataset_spec.dataset_name,
            dataset_type=dataset_spec.dataset_type,
            dataset_id=dataset_spec.dataset_id,
            analysis_scope=dataset_spec.analysis_scope,
            source_predictor_name=dataset_spec.source_predictor_name,
            source_predictor_type=dataset_spec.source_predictor_type,
            source_predictor_id=dataset_spec.source_predictor_id,
            synthetic_shift_hours=dataset_spec.synthetic_shift_hours,
            input_row_count=input_row_count,
            analysis_row_count=0,
            excluded_row_count=input_row_count,
            treated_count=0,
            control_count=0,
            event_count=0,
            primary_hr=None,
            primary_ci_lower=None,
            primary_ci_upper=None,
            sensitivity_hr=None,
            sensitivity_ci_lower=None,
            sensitivity_ci_upper=None,
            overlap_hr=None,
            overlap_ci_lower=None,
            overlap_ci_upper=None,
            propensity_min=None,
            propensity_max=None,
            treated_ps_min=None,
            treated_ps_max=None,
            control_ps_min=None,
            control_ps_max=None,
            overlap_lower=None,
            overlap_upper=None,
            overlap_width=None,
            severe_overlap_flag=False,
            primary_truncated_rate=None,
            sensitivity_truncated_rate=None,
            primary_truncation_lower=None,
            primary_truncation_upper=None,
            sensitivity_truncation_lower=None,
            sensitivity_truncation_upper=None,
            primary_weight_min=None,
            primary_weight_p01=None,
            primary_weight_p05=None,
            primary_weight_p25=None,
            primary_weight_p50=None,
            primary_weight_p75=None,
            primary_weight_p95=None,
            primary_weight_p99=None,
            primary_weight_max=None,
            primary_weight_mean=None,
            primary_weight_std=None,
            convergence_status=f"failed:{type(exc).__name__}:{exc}",
            detail_csv_path=str(empty_detail_path),
            survival_curve_csv_path=str(empty_survival_path),
        )
        return failure_result, empty_frame


def build_disagreement_task_ids(model_frames: dict[str, pd.DataFrame]) -> set[str]:
    """Identify task rows where the extracted models disagree materially."""

    shared_task_ids = set.intersection(*(set(frame["task_id"]) for frame in model_frames.values()))
    disagreement_task_ids: set[str] = set()

    ordered_names = sorted(model_frames)
    for task_id in shared_task_ids:
        task_rows = [model_frames[name].set_index("task_id").loc[task_id] for name in ordered_names]
        suspected_values = {bool(row["suspected_sepsis"]) for row in task_rows}
        if len(suspected_values) > 1:
            disagreement_task_ids.add(task_id)
            continue

        infection_sources = {
            normalize_optional_text(row["infection_source"]) or ""
            for row in task_rows
        }
        if len(infection_sources) > 1:
            disagreement_task_ids.add(task_id)
            continue

        suspicion_times = []
        for row in task_rows:
            time_value = pd.to_datetime(row["suspicion_time"], errors="coerce")
            if pd.isna(time_value):
                suspicion_times.append(None)
            else:
                suspicion_times.append(time_value.to_pydatetime())
        if any(value is None for value in suspicion_times) and not all(value is None for value in suspicion_times):
            disagreement_task_ids.add(task_id)
            continue
        if all(value is not None for value in suspicion_times):
            min_time = min(suspicion_times)
            max_time = max(suspicion_times)
            if (max_time - min_time).total_seconds() > 3600:
                disagreement_task_ids.add(task_id)

    return disagreement_task_ids


def select_reference_frame(primary_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Choose the gold dataset when present, else fall back to the first primary dataset."""

    if "gold" in primary_frames:
        return primary_frames["gold"]
    return primary_frames[sorted(primary_frames)[0]]


def compute_robustness_summary(results: list[PipelineResult]) -> dict[str, object]:
    """Compute the cross-dataset robustness metrics required by the contract."""

    primary_full = [
        result
        for result in results
        if result.analysis_scope == "full_cohort" and result.primary_hr is not None
    ]
    result_by_name = {result.dataset_name: result for result in primary_full}
    gold_result = result_by_name.get("gold")

    model_results = [
        result
        for result in primary_full
        if result.dataset_type == MODEL_DATASET_TYPE
    ]
    model_hrs = [result.primary_hr for result in model_results if result.primary_hr is not None]
    bias_by_dataset: dict[str, float] = {}
    if gold_result is not None and gold_result.primary_hr is not None:
        for result in model_results:
            if result.primary_hr is not None:
                bias_by_dataset[result.dataset_name] = round_optional(
                    result.primary_hr - gold_result.primary_hr
                )
        x_rule_result = result_by_name.get("x_rule")
        if x_rule_result is not None and x_rule_result.primary_hr is not None:
            bias_by_dataset["x_rule"] = round_optional(
                x_rule_result.primary_hr - gold_result.primary_hr
            )

    synthetic_results = [
        result
        for result in primary_full
        if result.dataset_type == "synthetic" and result.synthetic_shift_hours is not None
    ]
    synthetic_trend = [
        {
            "dataset_name": result.dataset_name,
            "source_predictor_name": result.source_predictor_name,
            "shift_hours": result.synthetic_shift_hours,
            "primary_hr": result.primary_hr,
            "gold_hr_difference": (
                round_optional(result.primary_hr - gold_result.primary_hr)
                if gold_result is not None and gold_result.primary_hr is not None and result.primary_hr is not None
                else None
            ),
        }
        for result in sorted(
            synthetic_results,
            key=lambda row: (row.source_predictor_name or "", row.synthetic_shift_hours or 0.0),
        )
    ]

    return {
        "bias_by_dataset": bias_by_dataset,
        "stability": (
            round_optional(max(model_hrs) - min(model_hrs))
            if len(model_hrs) >= 2
            else None
        ),
        "variance_across_models": (
            round_optional(float(np.var(model_hrs)))
            if len(model_hrs) >= 2
            else None
        ),
        "synthetic_trend": synthetic_trend,
    }


def ensure_successful_results(results: list[PipelineResult]) -> None:
    """Stop the run when any dataset-specific analysis violates the fixed contract."""

    failures = [
        f"{result.dataset_name}:{result.analysis_scope}:{result.convergence_status}"
        for result in results
        if result.convergence_status != "ok"
    ]
    if failures:
        failure_text = "\n".join(f"- {failure}" for failure in failures)
        raise RuntimeError(
            "One or more dataset analyses failed under the fixed causal contract:\n"
            f"{failure_text}"
        )


def write_outputs(
    *,
    config: CausalAnalysisConfig,
    results: list[PipelineResult],
    robustness_summary: dict[str, object],
) -> CausalAnalysisRunSummary:
    """Persist the causal-analysis summaries, robustness metrics, and report."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.report_dir.mkdir(parents=True, exist_ok=True)

    summary_csv_path = config.output_dir / "causal_analysis_summary.csv"
    summary_json_path = config.output_dir / "causal_analysis_summary.json"
    robustness_json_path = config.output_dir / "causal_analysis_robustness.json"
    report_path = config.report_dir / (
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_causal_analysis_report.md"
    )

    with open(summary_csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))

    summary_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_results": [asdict(result) for result in results],
    }
    with open(summary_json_path, "w", encoding="utf-8") as file:
        json.dump(summary_payload, file, indent=2)
    with open(robustness_json_path, "w", encoding="utf-8") as file:
        json.dump(robustness_summary, file, indent=2)

    report_lines = [
        "# Causal Analysis Report",
        f"Date: {summary_payload['generated_at_utc']}",
        "",
        "## Inputs",
        f"- analysis_manifest: `{config.analysis_manifest}`",
        (
            f"- perturbation_manifest: `{config.perturbation_manifest}`"
            if config.perturbation_manifest is not None
            else "- perturbation_manifest: none"
        ),
        f"- mimic_root: `{config.mimic_root}`",
        f"- primary_weight_truncation: {PRIMARY_TRUNCATION_QUANTILES[0]} / {PRIMARY_TRUNCATION_QUANTILES[1]}",
        f"- sensitivity_weight_truncation: {SENSITIVITY_TRUNCATION_QUANTILES[0]} / {SENSITIVITY_TRUNCATION_QUANTILES[1]}",
        "",
        "## Dataset Results",
    ]
    for result in results:
        report_lines.extend(
            [
                f"### {result.dataset_name} ({result.analysis_scope})",
                f"- dataset_type: {result.dataset_type}",
                f"- dataset_id: `{result.dataset_id}`",
                f"- n_patients: {result.analysis_row_count}",
                f"- n_events: {result.event_count}",
                f"- treated: {result.treated_count}",
                f"- control: {result.control_count}",
                f"- primary_hr_ci: {result.primary_hr} ({result.primary_ci_lower}, {result.primary_ci_upper})",
                f"- sensitivity_hr_ci: {result.sensitivity_hr} ({result.sensitivity_ci_lower}, {result.sensitivity_ci_upper})",
                f"- overlap_hr_ci: {result.overlap_hr} ({result.overlap_ci_lower}, {result.overlap_ci_upper})",
                f"- propensity_range: ({result.propensity_min}, {result.propensity_max})",
                (
                    "- overlap_region: "
                    f"({result.overlap_lower}, {result.overlap_upper}), "
                    f"width={result.overlap_width}, severe={result.severe_overlap_flag}"
                ),
                (
                    "- primary_truncation: "
                    f"bounds=({result.primary_truncation_lower}, {result.primary_truncation_upper}), "
                    f"truncated_rate={result.primary_truncated_rate}"
                ),
                (
                    "- sensitivity_truncation: "
                    f"bounds=({result.sensitivity_truncation_lower}, {result.sensitivity_truncation_upper}), "
                    f"truncated_rate={result.sensitivity_truncated_rate}"
                ),
                (
                    "- primary_weight_summary: "
                    f"min={result.primary_weight_min}, "
                    f"p01={result.primary_weight_p01}, "
                    f"p05={result.primary_weight_p05}, "
                    f"p25={result.primary_weight_p25}, "
                    f"p50={result.primary_weight_p50}, "
                    f"p75={result.primary_weight_p75}, "
                    f"p95={result.primary_weight_p95}, "
                    f"p99={result.primary_weight_p99}, "
                    f"max={result.primary_weight_max}, "
                    f"mean={result.primary_weight_mean}, "
                    f"std={result.primary_weight_std}"
                ),
                f"- convergence_status: {result.convergence_status}",
                "",
            ]
        )

    report_lines.extend(
        [
            "",
            "## Robustness",
            f"- bias_by_dataset: {json.dumps(robustness_summary.get('bias_by_dataset', {}), sort_keys=True)}",
            f"- stability: {robustness_summary.get('stability')}",
            f"- variance_across_models: {robustness_summary.get('variance_across_models')}",
        ]
    )
    synthetic_trend = robustness_summary.get("synthetic_trend", [])
    if synthetic_trend:
        report_lines.append("- synthetic_trend:")
        for row in synthetic_trend:
            report_lines.append(
                "  "
                + ", ".join(
                    [
                        f"dataset={row['dataset_name']}",
                        f"source_predictor={row['source_predictor_name']}",
                        f"shift_hours={row['shift_hours']}",
                        f"primary_hr={row['primary_hr']}",
                        f"gold_hr_difference={row['gold_hr_difference']}",
                    ]
                )
            )

    report_lines.extend(
        [
            "",
            "## Outputs",
            f"- `{summary_csv_path}`",
            f"- `{summary_json_path}`",
            f"- `{robustness_json_path}`",
            f"- `{report_path}`",
        ]
    )
    with open(report_path, "w", encoding="utf-8") as file:
        file.write("\n".join(report_lines) + "\n")

    return CausalAnalysisRunSummary(
        generated_at_utc=summary_payload["generated_at_utc"],
        analysis_manifest=str(config.analysis_manifest),
        perturbation_manifest=(
            str(config.perturbation_manifest)
            if config.perturbation_manifest is not None
            else None
        ),
        mimic_root=str(config.mimic_root),
        output_dir=str(config.output_dir),
        report_path=str(report_path),
        summary_csv_path=str(summary_csv_path),
        summary_json_path=str(summary_json_path),
        robustness_json_path=str(robustness_json_path),
        dataset_results=[asdict(result) for result in results],
        robustness_summary=robustness_summary,
    )


def run_causal_analysis(config: CausalAnalysisConfig) -> CausalAnalysisRunSummary:
    """Execute the fixed causal-analysis contract across all configured datasets."""

    required_paths = ensure_inputs(config)
    primary_specs, synthetic_specs = load_dataset_specs(config)
    primary_frames = {
        spec.dataset_name: load_dataset_frame(spec)
        for spec in primary_specs
    }
    validate_primary_dataset_schema_alignment(primary_frames)
    validate_primary_dataset_alignment(primary_frames)
    reference_primary_name = primary_specs[0].dataset_name
    reference_primary_columns = primary_frames[reference_primary_name].columns.tolist()

    all_keys_frame = pd.concat(
        [
            frame[["subject_id", "hadm_id", "stay_id"]]
            for frame in primary_frames.values()
        ],
        ignore_index=True,
    ).drop_duplicates()
    connection = initialize_connection(required_paths, all_keys_frame)

    processed_primary_frames: dict[str, pd.DataFrame] = {}
    for spec in primary_specs:
        processed_primary_frames[spec.dataset_name] = derive_covariates(
            connection,
            spec,
            primary_frames[spec.dataset_name],
        )

    reference_frame = select_reference_frame(processed_primary_frames)
    reference_analysis_frame = reference_frame[reference_frame["analysis_eligible"] == True].copy()  # noqa: E712
    union_analysis_frame = pd.concat(
        [
            frame[frame["analysis_eligible"] == True].copy()  # noqa: E712
            for frame in processed_primary_frames.values()
        ],
        ignore_index=True,
    )
    preprocessor = build_preprocessor(
        reference_analysis_frame,
        union_analysis_frame,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[PipelineResult] = []
    analyzed_frames: dict[tuple[str, str], pd.DataFrame] = {}
    for spec in primary_specs:
        result, analyzed_frame = analyze_one_dataset(
            spec,
            processed_primary_frames[spec.dataset_name],
            preprocessor=preprocessor,
            config=config,
            output_dir=config.output_dir,
            fail_on_error=True,
        )
        results.append(result)
        analyzed_frames[(spec.dataset_name, spec.analysis_scope)] = analyzed_frame

    model_processed_frames = {
        name: frame
        for name, frame in processed_primary_frames.items()
        if str(frame["predictor_type"].iloc[0]) == MODEL_DATASET_TYPE
    }
    disagreement_task_ids = (
        build_disagreement_task_ids(model_processed_frames)
        if len(model_processed_frames) >= 2
        else set()
    )
    if disagreement_task_ids:
        for spec in primary_specs:
            disagreement_spec = DatasetSpec(
                dataset_name=spec.dataset_name,
                dataset_type=spec.dataset_type,
                dataset_id=spec.dataset_id,
                csv_path=spec.csv_path,
                analysis_scope="disagreement_subset",
                source_predictor_name=spec.source_predictor_name,
                source_predictor_type=spec.source_predictor_type,
                source_predictor_id=spec.source_predictor_id,
                synthetic_shift_hours=spec.synthetic_shift_hours,
            )
            disagreement_frame = processed_primary_frames[spec.dataset_name]
            disagreement_frame = disagreement_frame[
                disagreement_frame["task_id"].isin(disagreement_task_ids)
            ].copy()
            result, analyzed_frame = analyze_one_dataset(
                disagreement_spec,
                disagreement_frame,
                preprocessor=preprocessor,
                config=config,
                output_dir=config.output_dir,
                fail_on_error=False,
            )
            results.append(result)
            analyzed_frames[(disagreement_spec.dataset_name, disagreement_spec.analysis_scope)] = analyzed_frame

    for synthetic_spec in synthetic_specs:
        synthetic_frame = load_dataset_frame(synthetic_spec)
        validate_dataset_schema(
            synthetic_spec.dataset_name,
            synthetic_frame.columns.tolist(),
            allowed_extra_columns=ALLOWED_SYNTHETIC_SCHEMA_EXTRAS,
            reference_name=reference_primary_name,
            reference_columns=reference_primary_columns,
        )
        processed_synthetic_frame = derive_covariates(
            connection,
            synthetic_spec,
            synthetic_frame,
        )
        result, analyzed_frame = analyze_one_dataset(
            synthetic_spec,
            processed_synthetic_frame,
            preprocessor=preprocessor,
            config=config,
            output_dir=config.output_dir,
            fail_on_error=False,
        )
        results.append(result)
        analyzed_frames[(synthetic_spec.dataset_name, synthetic_spec.analysis_scope)] = analyzed_frame

    ensure_successful_results(results)
    robustness_summary = compute_robustness_summary(results)
    return write_outputs(
        config=config,
        results=results,
        robustness_summary=robustness_summary,
    )


def main() -> int:
    """CLI entrypoint for the fixed causal analysis pipeline."""

    args = parse_args()
    config = build_config(args)
    summary = run_causal_analysis(config)
    print("Causal analysis completed.")
    print(f"Dataset result count: {len(summary.dataset_results)}")
    print(f"Summary CSV: {summary.summary_csv_path}")
    print(f"Report: {summary.report_path}")
    return 0
