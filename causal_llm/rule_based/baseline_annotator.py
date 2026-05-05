"""Build a deterministic v2 rule-based baseline measurement from MIMIC-IV data."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

DEFAULT_MIMIC_ROOT = Path("physionet.org/files/mimiciv/3.1")
DEFAULT_COHORT_DUCKDB_PATH = Path("derived_data/cohort/phase1_cohort.duckdb")
DEFAULT_COHORT_TABLE = "sampled_candidate_cohort"
DEFAULT_OUTPUT_DIR = Path("derived_data/rule_based")
DEFAULT_REPORT_DIR = Path("reports/rule_based")
DEFAULT_HIGH_CONFIDENCE_WINDOW_HOURS = 6

REQUIRED_MIMIC_FILES = {
    "emar": "hosp/emar.csv.gz",
    "microbiologyevents": "hosp/microbiologyevents.csv.gz",
    "diagnoses_icd": "hosp/diagnoses_icd.csv.gz",
    "d_icd_diagnoses": "hosp/d_icd_diagnoses.csv.gz",
    "labevents": "hosp/labevents.csv.gz",
    "d_labitems": "hosp/d_labitems.csv.gz",
    "chartevents": "icu/chartevents.csv.gz",
    "d_items": "icu/d_items.csv.gz",
}
REQUIRED_COHORT_COLUMNS = {"subject_id", "hadm_id", "stay_id", "study_window_start", "study_window_end"}
REQUIRED_NOTE_COLUMNS = {"stay_id", "note_charttime", "text"}
VALID_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

ANTIBIOTIC_ADMIN_EVENT_REGEX = r"(administered|given|started)"
NOTE_SUSPICION_PATTERNS: tuple[tuple[str, str], ...] = (
    ("sepsis", r"\bsepsis\b"),
    ("suspected infection", r"suspect(?:ed|ing)? infection"),
    ("possible pneumonia", r"possible pneumonia"),
    ("rule out infection", r"(rule out|r/o) infection"),
    ("empiric antibiotics", r"empiric antibiotics?"),
    ("concern for source", r"concern for source"),
    ("concern for infection", r"concern for infection"),
)

# Source mapping stays explicit on purpose. Broad lexical coverage would make
# the baseline drift toward a heuristic classifier rather than a transparent
# deterministic reference measurement.
PULMONARY_CONTEXT_REGEX = (
    r"(pneumonia|empyema|lung abscess|pulmonary source|respiratory source|"
    r"sputum|bronch|bal|bronchoalveolar|pleural|tracheal)"
)
URINARY_CONTEXT_REGEX = r"(urinary tract infection|\buti\b|\bcystitis\b|pyelonephritis|urine|urosepsis)"
ABDOMINAL_CONTEXT_REGEX = (
    r"(peritonitis|cholangitis|cholecystitis|appendicitis|diverticulitis|"
    r"intra-abdominal infection|hepatic abscess|liver abscess|biliary infection|"
    r"peritoneal abscess|bile)"
)
SKIN_SOFT_TISSUE_CONTEXT_REGEX = (
    r"(cellulitis|wound infection|infected ulcer|necrotizing fasciitis|"
    r"soft tissue infection|cutaneous abscess|skin abscess)"
)
CNS_CONTEXT_REGEX = r"(meningitis|encephalitis|ventriculitis|brain abscess|csf|spinal fluid)"
ENDOCARDIAL_CONTEXT_REGEX = r"(endocarditis)"
BONE_JOINT_CONTEXT_REGEX = r"(osteomyelitis|septic arthritis|prosthetic joint infection)"
LINE_DEVICE_CONTEXT_REGEX = (
    r"(line infection|catheter infection|port infection|device infection|"
    r"central line-associated bloodstream infection|clabsi)"
)
PULMONARY_DIAGNOSIS_REGEX = r"(pneumonia|empyema|lung abscess|pulmonary infection|respiratory infection)"
URINARY_DIAGNOSIS_REGEX = r"(urinary tract infection|\buti\b|\bcystitis\b|pyelonephritis|urosepsis)"
ABDOMINAL_DIAGNOSIS_REGEX = (
    r"(peritonitis|cholangitis|cholecystitis|appendicitis|diverticulitis|"
    r"intra-abdominal infection|hepatic abscess|liver abscess|biliary infection|"
    r"peritoneal abscess)"
)
SKIN_SOFT_TISSUE_DIAGNOSIS_REGEX = (
    r"(cellulitis|wound infection|infected ulcer|necrotizing fasciitis|"
    r"soft tissue infection|cutaneous abscess|skin abscess)"
)
CNS_DIAGNOSIS_REGEX = r"(meningitis|encephalitis|ventriculitis|brain abscess|central nervous system infection)"
ENDOCARDIAL_DIAGNOSIS_REGEX = r"(endocarditis)"
BONE_JOINT_DIAGNOSIS_REGEX = r"(osteomyelitis|septic arthritis|prosthetic joint infection|bone infection|joint infection)"
LINE_DEVICE_DIAGNOSIS_REGEX = (
    r"(line infection|catheter infection|port infection|device infection|"
    r"central line-associated bloodstream infection|clabsi)"
)
BLOODSTREAM_DIAGNOSIS_REGEX = r"(bacteremia|septicemia|bloodstream infection)"


@dataclass(frozen=True)
class RuleBasedConfig:
    """Configuration for deterministic v2 baseline generation."""

    mimic_root: Path
    cohort_duckdb_path: Path
    cohort_table: str
    notes_parquet: Path | None
    output_dir: Path
    report_dir: Path
    high_confidence_window_hours: int
    include_vitals: bool


@dataclass(frozen=True)
class RuleBasedSummary:
    """Summary emitted after the deterministic v2 baseline run completes."""

    generated_at_utc: str
    mimic_root: str
    cohort_duckdb_path: str
    cohort_table: str
    notes_parquet: str | None
    output_dir: str
    report_path: str
    cohort_row_count: int
    labeled_row_count: int
    suspicion_time_rule_count: int
    null_rule_count: int
    tier1_note_signal_count: int
    tier2_antibiotic_admin_count: int
    tier2_blood_culture_order_count: int
    tier3_lactate_count: int
    tier3_abnormal_wbc_count: int
    tier3_fever_count: int
    tier3_hypotension_count: int
    tier3_tachycardia_count: int
    infection_source_rule_count: int
    suspicion_time_source_counts: dict[str, int]
    confidence_counts: dict[str, int]
    infection_source_counts: dict[str, int]
    rule_labels_csv_path: str
    rule_labels_json_path: str
    execution_log_path: str


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the deterministic baseline builder."""

    parser = argparse.ArgumentParser(
        description="Build a deterministic v2 rule-based baseline from MIMIC-IV data."
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
        help="Cohort table to label. Defaults to the sampled cohort for initial-scale runs.",
    )
    parser.add_argument(
        "--notes-parquet",
        type=Path,
        default=None,
        help="Optional stay-linked note parquet with stay_id, note_charttime, and text columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for deterministic baseline outputs.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Directory for markdown baseline reports.",
    )
    parser.add_argument(
        "--high-confidence-window-hours",
        type=int,
        default=DEFAULT_HIGH_CONFIDENCE_WINDOW_HOURS,
        help="Tier 1 and Tier 2 agreement window for assigning high confidence.",
    )
    parser.add_argument(
        "--include-vitals",
        action="store_true",
        help="Enable Tier 3 vital-sign support from chartevents. Disabled by default for faster reruns.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> RuleBasedConfig:
    """Resolve CLI arguments into a normalized deterministic-baseline config."""

    cohort_table = args.cohort_table.strip()
    if not cohort_table:
        raise ValueError("--cohort-table must be non-empty.")
    if not VALID_IDENTIFIER_PATTERN.match(cohort_table):
        raise ValueError("--cohort-table may contain only letters, numbers, and underscores.")
    if args.high_confidence_window_hours <= 0:
        raise ValueError("--high-confidence-window-hours must be positive.")

    return RuleBasedConfig(
        mimic_root=args.mimic_root.resolve(),
        cohort_duckdb_path=args.cohort_duckdb_path.resolve(),
        cohort_table=cohort_table,
        notes_parquet=args.notes_parquet.resolve() if args.notes_parquet else None,
        output_dir=args.output_dir.resolve(),
        report_dir=args.report_dir.resolve(),
        high_confidence_window_hours=args.high_confidence_window_hours,
        include_vitals=args.include_vitals,
    )


def sql_string(value: str) -> str:
    """Quote a Python string for safe interpolation into DuckDB SQL."""

    return "'" + value.replace("'", "''") + "'"


def ensure_inputs(config: RuleBasedConfig) -> dict[str, Path]:
    """Validate that the required cohort and structured MIMIC inputs exist."""

    if not config.cohort_duckdb_path.exists():
        raise FileNotFoundError(
            f"Missing cohort database: {config.cohort_duckdb_path}\n"
            "Run `python build_initial_cohort.py` first."
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

    if config.notes_parquet and not config.notes_parquet.exists():
        raise FileNotFoundError(f"Missing notes parquet: {config.notes_parquet}")

    return resolved_paths


def build_context_source_case_sql(text_expression: str, *, include_bloodstream: bool) -> str:
    """Build a CASE expression for explicit source mapping."""

    lines = [
        f"WHEN regexp_matches({text_expression}, {sql_string(PULMONARY_CONTEXT_REGEX)}) THEN 'pulmonary'",
        f"WHEN regexp_matches({text_expression}, {sql_string(URINARY_CONTEXT_REGEX)}) THEN 'urinary'",
        f"WHEN regexp_matches({text_expression}, {sql_string(ABDOMINAL_CONTEXT_REGEX)}) THEN 'abdominal'",
        f"WHEN regexp_matches({text_expression}, {sql_string(SKIN_SOFT_TISSUE_CONTEXT_REGEX)}) THEN 'skin_soft_tissue'",
        f"WHEN regexp_matches({text_expression}, {sql_string(CNS_CONTEXT_REGEX)}) THEN 'cns'",
        f"WHEN regexp_matches({text_expression}, {sql_string(ENDOCARDIAL_CONTEXT_REGEX)}) THEN 'endocardial'",
        f"WHEN regexp_matches({text_expression}, {sql_string(BONE_JOINT_CONTEXT_REGEX)}) THEN 'bone_joint'",
        f"WHEN regexp_matches({text_expression}, {sql_string(LINE_DEVICE_CONTEXT_REGEX)}) THEN 'line_device'",
    ]
    if include_bloodstream:
        lines.append(
            f"WHEN regexp_matches({text_expression}, {sql_string(BLOODSTREAM_DIAGNOSIS_REGEX)}) THEN 'bloodstream'"
        )
    return "CASE\n    " + "\n    ".join(lines) + "\n    ELSE NULL\nEND"


def build_diagnosis_source_case_sql(text_expression: str) -> str:
    """Build a diagnosis-title-only CASE expression for conservative source fallback."""

    lines = [
        f"WHEN regexp_matches({text_expression}, {sql_string(PULMONARY_DIAGNOSIS_REGEX)}) THEN 'pulmonary'",
        f"WHEN regexp_matches({text_expression}, {sql_string(URINARY_DIAGNOSIS_REGEX)}) THEN 'urinary'",
        f"WHEN regexp_matches({text_expression}, {sql_string(ABDOMINAL_DIAGNOSIS_REGEX)}) THEN 'abdominal'",
        f"WHEN regexp_matches({text_expression}, {sql_string(SKIN_SOFT_TISSUE_DIAGNOSIS_REGEX)}) THEN 'skin_soft_tissue'",
        f"WHEN regexp_matches({text_expression}, {sql_string(CNS_DIAGNOSIS_REGEX)}) THEN 'cns'",
        f"WHEN regexp_matches({text_expression}, {sql_string(ENDOCARDIAL_DIAGNOSIS_REGEX)}) THEN 'endocardial'",
        f"WHEN regexp_matches({text_expression}, {sql_string(BONE_JOINT_DIAGNOSIS_REGEX)}) THEN 'bone_joint'",
        f"WHEN regexp_matches({text_expression}, {sql_string(LINE_DEVICE_DIAGNOSIS_REGEX)}) THEN 'line_device'",
        f"WHEN regexp_matches({text_expression}, {sql_string(BLOODSTREAM_DIAGNOSIS_REGEX)}) THEN 'bloodstream'",
    ]
    return "CASE\n    " + "\n    ".join(lines) + "\n    ELSE NULL\nEND"


def build_note_phrase_case_sql(text_expression: str) -> str:
    """Build the CASE expression that records which note phrase triggered Tier 1."""

    lines = [
        f"WHEN regexp_matches({text_expression}, {sql_string(pattern)}) THEN {sql_string(label)}"
        for label, pattern in NOTE_SUSPICION_PATTERNS
    ]
    return "CASE\n    " + "\n    ".join(lines) + "\n    ELSE NULL\nEND"


def initialize_source_views(
    connection: duckdb.DuckDBPyConnection,
    config: RuleBasedConfig,
    required_paths: dict[str, Path],
) -> None:
    """Attach the cohort database and expose the required source tables as views."""

    connection.execute(
        f"ATTACH {sql_string(str(config.cohort_duckdb_path))} AS cohort_db (READ_ONLY);"
    )
    connection.execute(
        f"CREATE OR REPLACE VIEW selected_cohort AS SELECT * FROM cohort_db.{config.cohort_table};"
    )

    for name, path in required_paths.items():
        connection.execute(
            f"""
            CREATE OR REPLACE VIEW src_rule_{name} AS
            SELECT *
            FROM read_csv_auto({sql_string(str(path))}, header = TRUE, all_varchar = TRUE);
            """
        )


def ensure_required_cohort_columns(connection: duckdb.DuckDBPyConnection) -> None:
    """Validate that the selected cohort table exposes the required columns."""

    columns = {
        row[0].lower()
        for row in connection.execute("DESCRIBE SELECT * FROM selected_cohort").fetchall()
    }
    missing = sorted(REQUIRED_COHORT_COLUMNS - columns)
    if missing:
        raise ValueError(
            "Selected cohort table is missing required columns: " + ", ".join(missing)
        )


def initialize_note_signal_table(
    connection: duckdb.DuckDBPyConnection,
    config: RuleBasedConfig,
) -> None:
    """Create the Tier 1 note-evidence table or an empty placeholder when notes are unavailable."""

    if config.notes_parquet is None:
        connection.execute(
            """
            CREATE OR REPLACE TABLE rule_note_evidence_signal AS
            SELECT
                CAST(NULL AS BIGINT) AS subject_id,
                CAST(NULL AS BIGINT) AS hadm_id,
                CAST(NULL AS BIGINT) AS stay_id,
                CAST(NULL AS VARCHAR) AS first_note_evidence_id,
                CAST(NULL AS TIMESTAMP) AS first_note_evidence_time,
                CAST(NULL AS VARCHAR) AS first_note_match_phrase,
                CAST(NULL AS VARCHAR) AS first_note_source,
                CAST(NULL AS BIGINT) AS note_evidence_count
            WHERE FALSE;
            """
        )
        return

    columns = {
        row[0].lower()
        for row in connection.execute(
            f"DESCRIBE SELECT * FROM read_parquet({sql_string(str(config.notes_parquet))})"
        ).fetchall()
    }
    missing = sorted(REQUIRED_NOTE_COLUMNS - columns)
    if missing:
        raise ValueError(
            "Notes parquet is missing required columns for Tier 1 note evidence: "
            + ", ".join(missing)
        )

    note_id_expression = "CAST(note_id AS VARCHAR)" if "note_id" in columns else "CAST(NULL AS VARCHAR)"
    note_source_case = build_context_source_case_sql(
        "LOWER(COALESCE(text, ''))",
        include_bloodstream=False,
    )
    note_phrase_case = build_note_phrase_case_sql("LOWER(COALESCE(text, ''))")
    note_regex = "|".join(f"(?:{pattern})" for _, pattern in NOTE_SUSPICION_PATTERNS)

    connection.execute(
        f"""
        CREATE OR REPLACE TABLE rule_note_evidence_signal AS
        WITH note_candidates AS (
            SELECT
                cohort.subject_id,
                cohort.hadm_id,
                cohort.stay_id,
                {note_id_expression} AS note_id,
                TRY_CAST(note_charttime AS TIMESTAMP) AS note_time,
                {note_phrase_case} AS note_match_phrase,
                {note_source_case} AS note_source_candidate,
                ROW_NUMBER() OVER (
                    PARTITION BY cohort.stay_id
                    ORDER BY TRY_CAST(note_charttime AS TIMESTAMP), {note_id_expression}
                ) AS note_rank
            FROM selected_cohort AS cohort
            INNER JOIN read_parquet({sql_string(str(config.notes_parquet))}) AS notes
                ON TRY_CAST(notes.stay_id AS BIGINT) = cohort.stay_id
            WHERE TRY_CAST(note_charttime AS TIMESTAMP)
                  BETWEEN cohort.study_window_start AND cohort.study_window_end
              AND regexp_matches(LOWER(COALESCE(text, '')), {sql_string(note_regex)})
        )
        SELECT
            subject_id,
            hadm_id,
            stay_id,
            MAX(CASE WHEN note_rank = 1 THEN note_id END) AS first_note_evidence_id,
            MIN(note_time) AS first_note_evidence_time,
            MAX(CASE WHEN note_rank = 1 THEN note_match_phrase END) AS first_note_match_phrase,
            MAX(CASE WHEN note_rank = 1 THEN note_source_candidate END) AS first_note_source,
            COUNT(*) AS note_evidence_count
        FROM note_candidates
        GROUP BY subject_id, hadm_id, stay_id;
        """
    )


def build_rule_tables(connection: duckdb.DuckDBPyConnection, *, include_vitals: bool) -> None:
    """Materialize raw rule-signal tables and the combined deterministic input table."""

    culture_context_case = build_context_source_case_sql(
        "LOWER(COALESCE(micro.spec_type_desc, '') || ' ' || COALESCE(micro.test_name, ''))",
        include_bloodstream=False,
    )
    diagnosis_source_case = build_diagnosis_source_case_sql("LOWER(COALESCE(dict.long_title, ''))")

    connection.execute(
        f"""
        CREATE OR REPLACE TABLE rule_antibiotic_admin_signal AS
        SELECT
            cohort.subject_id,
            cohort.hadm_id,
            cohort.stay_id,
            MIN(TRY_CAST(emar.charttime AS TIMESTAMP)) AS first_antibiotic_admin_time,
            COUNT(*) AS antibiotic_admin_event_count
        FROM selected_cohort AS cohort
        INNER JOIN src_rule_emar AS emar
            ON TRY_CAST(emar.subject_id AS BIGINT) = cohort.subject_id
           AND TRY_CAST(emar.hadm_id AS BIGINT) = cohort.hadm_id
        WHERE TRY_CAST(emar.charttime AS TIMESTAMP)
              BETWEEN cohort.study_window_start AND cohort.study_window_end
          AND regexp_matches(LOWER(COALESCE(emar.event_txt, '')), {sql_string(ANTIBIOTIC_ADMIN_EVENT_REGEX)})
          AND regexp_matches(
                LOWER(COALESCE(emar.medication, '')),
                {sql_string(r"(amoxicillin|ampicillin|penicillin|piperacillin|tazobactam|nafcillin|oxacillin|dicloxacillin|cef[a-z]+|carbapenem|meropenem|ertapenem|imipenem|aztreonam|vancomycin|daptomycin|linezolid|clindamycin|metronidazole|doxycycline|minocycline|tigecycline|tetracycline|azithromycin|erythromycin|clarithromycin|ciprofloxacin|levofloxacin|moxifloxacin|gentamicin|tobramycin|amikacin|fosfomycin|trimethoprim|sulfamethoxazole|bactrim)")}
              )
        GROUP BY cohort.subject_id, cohort.hadm_id, cohort.stay_id;
        """
    )

    connection.execute(
        """
        CREATE OR REPLACE TABLE rule_blood_culture_order_signal AS
        SELECT
            cohort.subject_id,
            cohort.hadm_id,
            cohort.stay_id,
            MIN(
                COALESCE(
                    TRY_CAST(micro.charttime AS TIMESTAMP),
                    TRY_CAST(micro.chartdate AS TIMESTAMP)
                )
            ) AS first_blood_culture_order_time,
            COUNT(*) AS blood_culture_order_event_count
        FROM selected_cohort AS cohort
        INNER JOIN src_rule_microbiologyevents AS micro
            ON TRY_CAST(micro.subject_id AS BIGINT) = cohort.subject_id
           AND TRY_CAST(micro.hadm_id AS BIGINT) = cohort.hadm_id
        WHERE COALESCE(
                TRY_CAST(micro.charttime AS TIMESTAMP),
                TRY_CAST(micro.chartdate AS TIMESTAMP)
              ) BETWEEN cohort.study_window_start AND cohort.study_window_end
          AND (
              LOWER(COALESCE(micro.spec_type_desc, '')) LIKE '%blood culture%'
              OR LOWER(COALESCE(micro.test_name, '')) LIKE '%blood culture%'
          )
        GROUP BY cohort.subject_id, cohort.hadm_id, cohort.stay_id;
        """
    )

    connection.execute(
        f"""
        CREATE OR REPLACE TABLE rule_culture_context_signal AS
        WITH culture_candidates AS (
            SELECT
                cohort.subject_id,
                cohort.hadm_id,
                cohort.stay_id,
                COALESCE(
                    TRY_CAST(micro.charttime AS TIMESTAMP),
                    TRY_CAST(micro.chartdate AS TIMESTAMP)
                ) AS culture_context_time,
                CAST(micro.spec_type_desc AS VARCHAR) AS spec_type_desc,
                CAST(micro.test_name AS VARCHAR) AS test_name,
                {culture_context_case} AS culture_source_candidate,
            FROM selected_cohort AS cohort
            INNER JOIN src_rule_microbiologyevents AS micro
                ON TRY_CAST(micro.subject_id AS BIGINT) = cohort.subject_id
               AND TRY_CAST(micro.hadm_id AS BIGINT) = cohort.hadm_id
            WHERE COALESCE(
                    TRY_CAST(micro.charttime AS TIMESTAMP),
                    TRY_CAST(micro.chartdate AS TIMESTAMP)
                  ) BETWEEN cohort.study_window_start AND cohort.study_window_end
        ),
        filtered AS (
            SELECT *
            FROM culture_candidates
            WHERE culture_source_candidate IS NOT NULL
        ),
        ranked AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY stay_id
                    ORDER BY culture_context_time, spec_type_desc, test_name
                ) AS culture_rank
            FROM filtered
        )
        SELECT
            subject_id,
            hadm_id,
            stay_id,
            MIN(culture_context_time) AS first_culture_context_time,
            MAX(CASE WHEN culture_rank = 1 THEN culture_source_candidate END) AS first_culture_context_source,
            MAX(CASE WHEN culture_rank = 1 THEN spec_type_desc END) AS first_culture_context_specimen,
            MAX(CASE WHEN culture_rank = 1 THEN test_name END) AS first_culture_context_test,
            COUNT(*) AS culture_context_event_count
        FROM ranked
        GROUP BY subject_id, hadm_id, stay_id;
        """
    )

    connection.execute(
        """
        CREATE OR REPLACE TABLE lactate_itemids AS
        SELECT DISTINCT TRY_CAST(itemid AS BIGINT) AS itemid
        FROM src_rule_d_labitems
        WHERE LOWER(COALESCE(label, '')) = 'lactate';
        """
    )

    connection.execute(
        """
        CREATE OR REPLACE TABLE rule_lactate_support_signal AS
        SELECT
            cohort.subject_id,
            cohort.hadm_id,
            cohort.stay_id,
            MIN(TRY_CAST(labs.charttime AS TIMESTAMP)) AS first_lactate_time,
            COUNT(*) AS lactate_event_count
        FROM selected_cohort AS cohort
        INNER JOIN src_rule_labevents AS labs
            ON TRY_CAST(labs.subject_id AS BIGINT) = cohort.subject_id
           AND TRY_CAST(labs.hadm_id AS BIGINT) = cohort.hadm_id
        INNER JOIN lactate_itemids AS item
            ON TRY_CAST(labs.itemid AS BIGINT) = item.itemid
        WHERE TRY_CAST(labs.charttime AS TIMESTAMP)
              BETWEEN cohort.study_window_start AND cohort.study_window_end
        GROUP BY cohort.subject_id, cohort.hadm_id, cohort.stay_id;
        """
    )

    connection.execute(
        f"""
        CREATE OR REPLACE TABLE wbc_itemids AS
        SELECT DISTINCT TRY_CAST(itemid AS BIGINT) AS itemid
        FROM src_rule_d_labitems
        WHERE regexp_matches(LOWER(COALESCE(label, '')), {sql_string(r"(^wbc$|white blood cells?)")});
        """
    )

    connection.execute(
        """
        CREATE OR REPLACE TABLE rule_abnormal_wbc_signal AS
        SELECT
            cohort.subject_id,
            cohort.hadm_id,
            cohort.stay_id,
            MIN(TRY_CAST(labs.charttime AS TIMESTAMP)) AS first_abnormal_wbc_time,
            COUNT(*) AS abnormal_wbc_event_count
        FROM selected_cohort AS cohort
        INNER JOIN src_rule_labevents AS labs
            ON TRY_CAST(labs.subject_id AS BIGINT) = cohort.subject_id
           AND TRY_CAST(labs.hadm_id AS BIGINT) = cohort.hadm_id
        INNER JOIN wbc_itemids AS item
            ON TRY_CAST(labs.itemid AS BIGINT) = item.itemid
        WHERE TRY_CAST(labs.charttime AS TIMESTAMP)
              BETWEEN cohort.study_window_start AND cohort.study_window_end
          AND TRY_CAST(labs.valuenum AS DOUBLE) IS NOT NULL
          AND (
              TRY_CAST(labs.valuenum AS DOUBLE) < 4.0
              OR TRY_CAST(labs.valuenum AS DOUBLE) > 12.0
          )
        GROUP BY cohort.subject_id, cohort.hadm_id, cohort.stay_id;
        """
    )

    if include_vitals:
        connection.execute(
            """
            CREATE OR REPLACE TABLE rule_chartevent_support_event AS
            WITH support_itemids AS (
                SELECT
                    TRY_CAST(itemid AS BIGINT) AS itemid,
                    LOWER(COALESCE(label, '')) AS label
                FROM src_rule_d_items
                WHERE
                    LOWER(COALESCE(label, '')) = 'heart rate'
                    OR LOWER(COALESCE(label, '')) LIKE '%temperature c%'
                    OR LOWER(COALESCE(label, '')) LIKE '%temperature f%'
                    OR regexp_matches(
                        LOWER(COALESCE(label, '')),
                        '(arterial blood pressure mean|non invasive blood pressure mean|abp mean|nbp mean|map|arterial blood pressure systolic|non invasive blood pressure systolic|abp systolic|nbp systolic)'
                    )
            ),
            candidate_events AS (
                SELECT
                    cohort.subject_id,
                    cohort.hadm_id,
                    cohort.stay_id,
                    TRY_CAST(chart.charttime AS TIMESTAMP) AS support_time,
                    CASE
                        WHEN item.label LIKE '%temperature c%' AND TRY_CAST(chart.valuenum AS DOUBLE) >= 38.0 THEN 'fever'
                        WHEN item.label LIKE '%temperature f%' AND TRY_CAST(chart.valuenum AS DOUBLE) >= 100.4 THEN 'fever'
                        WHEN regexp_matches(
                                item.label,
                                '(arterial blood pressure mean|non invasive blood pressure mean|abp mean|nbp mean|map)'
                             )
                             AND TRY_CAST(chart.valuenum AS DOUBLE) < 65.0 THEN 'hypotension'
                        WHEN regexp_matches(
                                item.label,
                                '(arterial blood pressure systolic|non invasive blood pressure systolic|abp systolic|nbp systolic)'
                             )
                             AND TRY_CAST(chart.valuenum AS DOUBLE) < 90.0 THEN 'hypotension'
                        WHEN item.label = 'heart rate' AND TRY_CAST(chart.valuenum AS DOUBLE) > 100.0 THEN 'tachycardia'
                        ELSE NULL
                    END AS support_type
                FROM selected_cohort AS cohort
                INNER JOIN src_rule_chartevents AS chart
                    ON TRY_CAST(chart.stay_id AS BIGINT) = cohort.stay_id
                INNER JOIN support_itemids AS item
                    ON TRY_CAST(chart.itemid AS BIGINT) = item.itemid
                WHERE TRY_CAST(chart.charttime AS TIMESTAMP)
                      BETWEEN cohort.study_window_start AND cohort.study_window_end
                  AND TRY_CAST(chart.valuenum AS DOUBLE) IS NOT NULL
            )
            SELECT *
            FROM candidate_events
            WHERE support_type IS NOT NULL;
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE rule_fever_signal AS
            SELECT
                subject_id,
                hadm_id,
                stay_id,
                MIN(support_time) AS first_fever_time,
                COUNT(*) AS fever_event_count
            FROM rule_chartevent_support_event
            WHERE support_type = 'fever'
            GROUP BY subject_id, hadm_id, stay_id;
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE rule_hypotension_signal AS
            SELECT
                subject_id,
                hadm_id,
                stay_id,
                MIN(support_time) AS first_hypotension_time,
                COUNT(*) AS hypotension_event_count
            FROM rule_chartevent_support_event
            WHERE support_type = 'hypotension'
            GROUP BY subject_id, hadm_id, stay_id;
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE rule_tachycardia_signal AS
            SELECT
                subject_id,
                hadm_id,
                stay_id,
                MIN(support_time) AS first_tachycardia_time,
                COUNT(*) AS tachycardia_event_count
            FROM rule_chartevent_support_event
            WHERE support_type = 'tachycardia'
            GROUP BY subject_id, hadm_id, stay_id;
            """
        )
    else:
        connection.execute(
            """
            CREATE OR REPLACE TABLE rule_fever_signal AS
            SELECT
                CAST(NULL AS BIGINT) AS subject_id,
                CAST(NULL AS BIGINT) AS hadm_id,
                CAST(NULL AS BIGINT) AS stay_id,
                CAST(NULL AS TIMESTAMP) AS first_fever_time,
                CAST(NULL AS BIGINT) AS fever_event_count
            WHERE FALSE;
            """
        )
        connection.execute(
            """
            CREATE OR REPLACE TABLE rule_hypotension_signal AS
            SELECT
                CAST(NULL AS BIGINT) AS subject_id,
                CAST(NULL AS BIGINT) AS hadm_id,
                CAST(NULL AS BIGINT) AS stay_id,
                CAST(NULL AS TIMESTAMP) AS first_hypotension_time,
                CAST(NULL AS BIGINT) AS hypotension_event_count
            WHERE FALSE;
            """
        )
        connection.execute(
            """
            CREATE OR REPLACE TABLE rule_tachycardia_signal AS
            SELECT
                CAST(NULL AS BIGINT) AS subject_id,
                CAST(NULL AS BIGINT) AS hadm_id,
                CAST(NULL AS BIGINT) AS stay_id,
                CAST(NULL AS TIMESTAMP) AS first_tachycardia_time,
                CAST(NULL AS BIGINT) AS tachycardia_event_count
            WHERE FALSE;
            """
        )

    connection.execute(
        f"""
        CREATE OR REPLACE TABLE rule_diagnosis_source_signal AS
        WITH diagnosis_candidates AS (
            SELECT
                cohort.subject_id,
                cohort.hadm_id,
                cohort.stay_id,
                TRY_CAST(dx.seq_num AS INTEGER) AS seq_num,
                CAST(dict.long_title AS VARCHAR) AS diagnosis_title,
                {diagnosis_source_case} AS diagnosis_source_candidate
            FROM selected_cohort AS cohort
            INNER JOIN src_rule_diagnoses_icd AS dx
                ON TRY_CAST(dx.subject_id AS BIGINT) = cohort.subject_id
               AND TRY_CAST(dx.hadm_id AS BIGINT) = cohort.hadm_id
            INNER JOIN src_rule_d_icd_diagnoses AS dict
                ON dx.icd_code = dict.icd_code
               AND dx.icd_version = dict.icd_version
        ),
        filtered AS (
            SELECT *
            FROM diagnosis_candidates
            WHERE diagnosis_source_candidate IS NOT NULL
        )
        SELECT
            *,
            ROW_NUMBER() OVER (
                PARTITION BY stay_id
                ORDER BY seq_num, diagnosis_title
            ) AS diagnosis_rank
        FROM filtered;
        """
    )

    connection.execute(
        """
        CREATE OR REPLACE TABLE rule_diagnosis_source_summary AS
        SELECT
            subject_id,
            hadm_id,
            stay_id,
            COUNT(*) AS diagnosis_match_count,
            MAX(CASE WHEN diagnosis_rank = 1 THEN diagnosis_source_candidate END) AS first_diagnosis_source,
            MAX(CASE WHEN diagnosis_rank = 1 THEN diagnosis_title END) AS first_diagnosis_title
        FROM rule_diagnosis_source_signal
        GROUP BY subject_id, hadm_id, stay_id;
        """
    )

    connection.execute(
        """
        CREATE OR REPLACE TABLE rule_based_signal_rows AS
        SELECT
            cohort.subject_id,
            cohort.subject_id AS patient_id,
            cohort.hadm_id,
            cohort.stay_id,
            cohort.stay_id AS icustay_id,
            note.first_note_evidence_id,
            note.first_note_evidence_time,
            note.first_note_match_phrase,
            note.first_note_source,
            note.note_evidence_count,
            admin.first_antibiotic_admin_time,
            admin.antibiotic_admin_event_count,
            blood.first_blood_culture_order_time,
            blood.blood_culture_order_event_count,
            culture.first_culture_context_time,
            culture.first_culture_context_source,
            culture.first_culture_context_specimen,
            culture.first_culture_context_test,
            culture.culture_context_event_count,
            lactate.first_lactate_time,
            lactate.lactate_event_count,
            wbc.first_abnormal_wbc_time,
            wbc.abnormal_wbc_event_count,
            fever.first_fever_time,
            fever.fever_event_count,
            hypotension.first_hypotension_time,
            hypotension.hypotension_event_count,
            tachy.first_tachycardia_time,
            tachy.tachycardia_event_count,
            diagnosis.first_diagnosis_source,
            diagnosis.first_diagnosis_title,
            diagnosis.diagnosis_match_count
        FROM selected_cohort AS cohort
        LEFT JOIN rule_note_evidence_signal AS note
            ON cohort.stay_id = note.stay_id
        LEFT JOIN rule_antibiotic_admin_signal AS admin
            ON cohort.stay_id = admin.stay_id
        LEFT JOIN rule_blood_culture_order_signal AS blood
            ON cohort.stay_id = blood.stay_id
        LEFT JOIN rule_culture_context_signal AS culture
            ON cohort.stay_id = culture.stay_id
        LEFT JOIN rule_lactate_support_signal AS lactate
            ON cohort.stay_id = lactate.stay_id
        LEFT JOIN rule_abnormal_wbc_signal AS wbc
            ON cohort.stay_id = wbc.stay_id
        LEFT JOIN rule_fever_signal AS fever
            ON cohort.stay_id = fever.stay_id
        LEFT JOIN rule_hypotension_signal AS hypotension
            ON cohort.stay_id = hypotension.stay_id
        LEFT JOIN rule_tachycardia_signal AS tachy
            ON cohort.stay_id = tachy.stay_id
        LEFT JOIN rule_diagnosis_source_summary AS diagnosis
            ON cohort.stay_id = diagnosis.stay_id;
        """
    )


def fetch_rows(connection: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    """Fetch the combined rule-signal rows as dictionaries."""

    result = connection.execute("SELECT * FROM rule_based_signal_rows ORDER BY stay_id;")
    columns = [column[0] for column in result.description]
    return [dict(zip(columns, row)) for row in result.fetchall()]


def serialize_value(value: Any) -> Any:
    """Convert DuckDB-returned Python values into JSON-safe output values."""

    if isinstance(value, datetime):
        return value.isoformat(sep=" ", timespec="seconds")
    return value


def choose_earliest_tier2_signal(row: dict[str, Any]) -> tuple[str | None, datetime | None]:
    """Choose the earliest Tier 2 action when Tier 1 note evidence is absent."""

    tier2_candidates = [
        ("tier2_antibiotic_administration", row["first_antibiotic_admin_time"]),
        ("tier2_blood_culture_order", row["first_blood_culture_order_time"]),
    ]
    available = [(name, time_value) for name, time_value in tier2_candidates if time_value is not None]
    if not available:
        return None, None
    return min(available, key=lambda item: (item[1], item[0]))


def build_rule_trigger_list(row: dict[str, Any]) -> list[str]:
    """Build the ordered tier-aware trigger list for one rule row."""

    triggers: list[str] = []
    if row["first_note_evidence_time"] is not None:
        triggers.append("tier1_note_evidence")
    if row["first_antibiotic_admin_time"] is not None:
        triggers.append("tier2_antibiotic_administration")
    if row["first_blood_culture_order_time"] is not None:
        triggers.append("tier2_blood_culture_order")
    if row["first_lactate_time"] is not None:
        triggers.append("tier3_lactate")
    if row["first_abnormal_wbc_time"] is not None:
        triggers.append("tier3_abnormal_wbc")
    if row["first_fever_time"] is not None:
        triggers.append("tier3_fever")
    if row["first_hypotension_time"] is not None:
        triggers.append("tier3_hypotension")
    if row["first_tachycardia_time"] is not None:
        triggers.append("tier3_tachycardia")
    return triggers


def build_evidence_source_list(row: dict[str, Any]) -> list[str]:
    """Build the evidence-source provenance list for one deterministic row."""

    evidence_sources: list[str] = []
    if row["first_note_evidence_time"] is not None:
        evidence_sources.extend(["notes:note_charttime", "notes:text"])
        if row["first_note_evidence_id"] is not None:
            evidence_sources.append("notes:note_id")
    if row["first_antibiotic_admin_time"] is not None:
        evidence_sources.extend(["emar:charttime", "emar:medication", "emar:event_txt"])
    if row["first_blood_culture_order_time"] is not None:
        evidence_sources.extend(
            [
                "microbiologyevents:charttime",
                "microbiologyevents:chartdate",
                "microbiologyevents:spec_type_desc",
                "microbiologyevents:test_name",
            ]
        )
    if row["first_lactate_time"] is not None:
        evidence_sources.extend(["labevents:charttime", "labevents:itemid", "d_labitems:label"])
    if row["first_abnormal_wbc_time"] is not None:
        evidence_sources.extend(["labevents:charttime", "labevents:valuenum", "d_labitems:label"])
    if row["first_fever_time"] is not None:
        evidence_sources.extend(["chartevents:charttime", "chartevents:valuenum", "d_items:label"])
    if row["first_hypotension_time"] is not None:
        evidence_sources.extend(["chartevents:charttime", "chartevents:valuenum", "d_items:label"])
    if row["first_tachycardia_time"] is not None:
        evidence_sources.extend(["chartevents:charttime", "chartevents:valuenum", "d_items:label"])
    if row["first_diagnosis_source"] is not None:
        evidence_sources.extend(["diagnoses_icd:seq_num", "d_icd_diagnoses:long_title"])

    # Deduplicate while preserving stable output order.
    seen: set[str] = set()
    deduped: list[str] = []
    for source in evidence_sources:
        if source in seen:
            continue
        seen.add(source)
        deduped.append(source)
    return deduped


def has_tier3_support(row: dict[str, Any]) -> bool:
    """Return whether any Tier 3 support signal fired for the stay."""

    return any(
        row[field] is not None
        for field in (
            "first_lactate_time",
            "first_abnormal_wbc_time",
            "first_fever_time",
            "first_hypotension_time",
            "first_tachycardia_time",
        )
    )


def determine_confidence(
    row: dict[str, Any],
    *,
    suspicion_time_rule_source: str | None,
    suspicion_time_rule: datetime | None,
    high_confidence_window_hours: int,
) -> str:
    """Assign categorical confidence according to the v2 rule definition."""

    if suspicion_time_rule_source == "tier1_note_evidence":
        tier2_times = [
            time_value
            for time_value in (
                row["first_antibiotic_admin_time"],
                row["first_blood_culture_order_time"],
            )
            if time_value is not None
        ]
        if suspicion_time_rule is not None and tier2_times:
            smallest_gap_seconds = min(
                abs((time_value - suspicion_time_rule).total_seconds())
                for time_value in tier2_times
            )
            if smallest_gap_seconds <= high_confidence_window_hours * 3600:
                return "high"
        return "medium"

    if suspicion_time_rule_source is not None:
        return "medium"

    if has_tier3_support(row):
        return "low"
    return "low"


def determine_rule_no_fire_reason(row: dict[str, Any], *, suspicion_time_rule: datetime | None) -> str | None:
    """Explain why no rule timestamp was emitted."""

    if suspicion_time_rule is not None:
        return None
    if has_tier3_support(row):
        return "only_tier3_supporting_signals"
    return "no_tier1_or_tier2_signal"


def build_output_rows(
    signal_rows: list[dict[str, Any]],
    *,
    high_confidence_window_hours: int,
) -> list[dict[str, Any]]:
    """Convert raw signal rows into JSON/CSV-ready deterministic v2 output records."""

    output_rows: list[dict[str, Any]] = []
    for row in signal_rows:
        note_time = row["first_note_evidence_time"]
        tier2_source, tier2_time = choose_earliest_tier2_signal(row)

        if note_time is not None:
            suspicion_time_rule = note_time
            suspicion_time_rule_source = "tier1_note_evidence"
        else:
            suspicion_time_rule = tier2_time
            suspicion_time_rule_source = tier2_source

        if suspicion_time_rule is None:
            infection_source_rule = None
            infection_source_rule_source = None
        else:
            infection_source_rule = (
                row["first_note_source"]
                or row["first_culture_context_source"]
                or row["first_diagnosis_source"]
            )
            if row["first_note_source"] is not None:
                infection_source_rule_source = "note_text"
            elif row["first_culture_context_source"] is not None:
                infection_source_rule_source = "culture_context"
            elif row["first_diagnosis_source"] is not None:
                infection_source_rule_source = "diagnosis_fallback"
            else:
                infection_source_rule_source = None

        confidence = determine_confidence(
            row,
            suspicion_time_rule_source=suspicion_time_rule_source,
            suspicion_time_rule=suspicion_time_rule,
            high_confidence_window_hours=high_confidence_window_hours,
        )
        rule_trigger = build_rule_trigger_list(row)
        evidence_source = build_evidence_source_list(row)
        rule_no_fire_reason = determine_rule_no_fire_reason(
            row,
            suspicion_time_rule=suspicion_time_rule,
        )

        serialized_row = {key: serialize_value(value) for key, value in row.items()}
        output_rows.append(
            {
                "patient_id": serialized_row["patient_id"],
                "subject_id": serialized_row["subject_id"],
                "hadm_id": serialized_row["hadm_id"],
                "stay_id": serialized_row["stay_id"],
                "icustay_id": serialized_row["icustay_id"],
                "suspicion_time_rule": serialize_value(suspicion_time_rule),
                "infection_source_rule": infection_source_rule,
                "rule_confidence": confidence,
                "rule_trigger": rule_trigger,
                "evidence_source": evidence_source,
                "suspicion_time_rule_source": suspicion_time_rule_source,
                "infection_source_rule_source": infection_source_rule_source,
                "first_note_evidence_id": serialized_row["first_note_evidence_id"],
                "first_note_evidence_time": serialized_row["first_note_evidence_time"],
                "first_note_match_phrase": serialized_row["first_note_match_phrase"],
                "first_note_source": serialized_row["first_note_source"],
                "first_antibiotic_admin_time": serialized_row["first_antibiotic_admin_time"],
                "first_blood_culture_order_time": serialized_row["first_blood_culture_order_time"],
                "first_culture_context_time": serialized_row["first_culture_context_time"],
                "first_culture_context_source": serialized_row["first_culture_context_source"],
                "first_culture_context_specimen": serialized_row["first_culture_context_specimen"],
                "first_culture_context_test": serialized_row["first_culture_context_test"],
                "first_lactate_time": serialized_row["first_lactate_time"],
                "first_abnormal_wbc_time": serialized_row["first_abnormal_wbc_time"],
                "first_fever_time": serialized_row["first_fever_time"],
                "first_hypotension_time": serialized_row["first_hypotension_time"],
                "first_tachycardia_time": serialized_row["first_tachycardia_time"],
                "first_diagnosis_source": serialized_row["first_diagnosis_source"],
                "first_diagnosis_title": serialized_row["first_diagnosis_title"],
                "rule_no_fire_reason": rule_no_fire_reason,
            }
        )
    return output_rows


def validate_output_rows(output_rows: list[dict[str, Any]], cohort_row_count: int) -> None:
    """Run core validation checks for the deterministic v2 output rows."""

    if len(output_rows) != cohort_row_count:
        raise ValueError(
            f"Output row count {len(output_rows)} does not match cohort row count {cohort_row_count}."
        )

    for row in output_rows:
        suspicion_time_rule = row["suspicion_time_rule"]
        if suspicion_time_rule is None:
            if row["rule_confidence"] != "low":
                raise ValueError("Rows without suspicion_time_rule must have low confidence.")
            if row["infection_source_rule"] is not None:
                raise ValueError(
                    f"Rows without suspicion_time_rule must not emit infection_source_rule: stay_id={row['stay_id']}."
                )
            if row["infection_source_rule_source"] is not None:
                raise ValueError(
                    "Rows without suspicion_time_rule must not emit infection_source_rule_source: "
                    f"stay_id={row['stay_id']}."
                )
            continue

        valid_times = {
            row["first_note_evidence_time"],
            row["first_antibiotic_admin_time"],
            row["first_blood_culture_order_time"],
        }
        if suspicion_time_rule not in valid_times:
            raise ValueError(
                f"Invalid suspicion_time_rule for stay_id={row['stay_id']}: not traceable to Tier 1 or Tier 2 timestamps."
            )
        if row["suspicion_time_rule_source"] is None:
            raise ValueError(
                f"Missing suspicion_time_rule_source for stay_id={row['stay_id']}."
            )
        if row["rule_confidence"] not in {"high", "medium"}:
            raise ValueError(
                f"Rows with suspicion_time_rule must have high or medium confidence: stay_id={row['stay_id']}."
            )


def write_outputs(
    output_rows: list[dict[str, Any]],
    config: RuleBasedConfig,
) -> RuleBasedSummary:
    """Persist deterministic baseline outputs, execution log, and markdown report."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.report_dir.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now(timezone.utc)
    generated_at_utc = generated_at.isoformat()
    rule_labels_csv_path = config.output_dir / "rule_labels.csv"
    rule_labels_json_path = config.output_dir / "rule_labels.json"
    execution_log_path = config.output_dir / "rule_execution_log.json"
    report_path = config.report_dir / (
        f"{generated_at.strftime('%Y%m%dT%H%M%SZ')}_rule_based_report.md"
    )

    with open(rule_labels_json_path, "w", encoding="utf-8") as file:
        json.dump(output_rows, file, indent=2)

    with open(rule_labels_csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "patient_id",
                "subject_id",
                "hadm_id",
                "stay_id",
                "icustay_id",
                "suspicion_time_rule",
                "infection_source_rule",
                "rule_confidence",
                "rule_trigger",
                "evidence_source",
                "suspicion_time_rule_source",
                "infection_source_rule_source",
                "first_note_evidence_id",
                "first_note_evidence_time",
                "first_note_match_phrase",
                "first_note_source",
                "first_antibiotic_admin_time",
                "first_blood_culture_order_time",
                "first_culture_context_time",
                "first_culture_context_source",
                "first_culture_context_specimen",
                "first_culture_context_test",
                "first_lactate_time",
                "first_abnormal_wbc_time",
                "first_fever_time",
                "first_hypotension_time",
                "first_tachycardia_time",
                "first_diagnosis_source",
                "first_diagnosis_title",
                "rule_no_fire_reason",
            ],
        )
        writer.writeheader()
        for row in output_rows:
            writer.writerow(
                {
                    **row,
                    "rule_trigger": "|".join(row["rule_trigger"]),
                    "evidence_source": "|".join(row["evidence_source"]),
                }
            )

    suspicion_time_source_counts = Counter(
        row["suspicion_time_rule_source"]
        for row in output_rows
        if row["suspicion_time_rule_source"] is not None
    )
    confidence_counts = Counter(row["rule_confidence"] for row in output_rows)
    infection_source_counts = Counter(
        row["infection_source_rule"]
        for row in output_rows
        if row["infection_source_rule"] is not None
    )

    summary = RuleBasedSummary(
        generated_at_utc=generated_at_utc,
        mimic_root=str(config.mimic_root),
        cohort_duckdb_path=str(config.cohort_duckdb_path),
        cohort_table=config.cohort_table,
        notes_parquet=str(config.notes_parquet) if config.notes_parquet else None,
        output_dir=str(config.output_dir),
        report_path=str(report_path),
        cohort_row_count=len(output_rows),
        labeled_row_count=len(output_rows),
        suspicion_time_rule_count=sum(row["suspicion_time_rule"] is not None for row in output_rows),
        null_rule_count=sum(row["suspicion_time_rule"] is None for row in output_rows),
        tier1_note_signal_count=sum("tier1_note_evidence" in row["rule_trigger"] for row in output_rows),
        tier2_antibiotic_admin_count=sum("tier2_antibiotic_administration" in row["rule_trigger"] for row in output_rows),
        tier2_blood_culture_order_count=sum("tier2_blood_culture_order" in row["rule_trigger"] for row in output_rows),
        tier3_lactate_count=sum("tier3_lactate" in row["rule_trigger"] for row in output_rows),
        tier3_abnormal_wbc_count=sum("tier3_abnormal_wbc" in row["rule_trigger"] for row in output_rows),
        tier3_fever_count=sum("tier3_fever" in row["rule_trigger"] for row in output_rows),
        tier3_hypotension_count=sum("tier3_hypotension" in row["rule_trigger"] for row in output_rows),
        tier3_tachycardia_count=sum("tier3_tachycardia" in row["rule_trigger"] for row in output_rows),
        infection_source_rule_count=sum(row["infection_source_rule"] is not None for row in output_rows),
        suspicion_time_source_counts=dict(sorted(suspicion_time_source_counts.items())),
        confidence_counts=dict(sorted(confidence_counts.items())),
        infection_source_counts=dict(sorted(infection_source_counts.items())),
        rule_labels_csv_path=str(rule_labels_csv_path),
        rule_labels_json_path=str(rule_labels_json_path),
        execution_log_path=str(execution_log_path),
    )

    execution_log = {
        "generated_at_utc": summary.generated_at_utc,
        "cohort_table": summary.cohort_table,
        "notes_parquet": summary.notes_parquet,
        "cohort_row_count": summary.cohort_row_count,
        "suspicion_time_rule_count": summary.suspicion_time_rule_count,
        "null_rule_count": summary.null_rule_count,
        "suspicion_time_source_counts": summary.suspicion_time_source_counts,
        "confidence_counts": summary.confidence_counts,
        "infection_source_counts": summary.infection_source_counts,
        "high_confidence_window_hours": config.high_confidence_window_hours,
        "limitations": [
            "Tier 1 note evidence is used only when an explicit linked-note parquet is supplied",
            "diagnoses_icd does not provide event timestamps in MIMIC-IV, so diagnosis titles are used only as a conservative infection-source fallback after a Tier 1 or Tier 2 rule fires",
            "infection-directed imaging orders are not yet encoded in the deterministic baseline",
            "Tier 3 support always uses lactate and abnormal WBC; fever, hypotension, and tachycardia are optional via --include-vitals",
            "X_rule is a deterministic baseline measurement, not clinical ground truth",
        ],
    }
    with open(execution_log_path, "w", encoding="utf-8") as file:
        json.dump(execution_log, file, indent=2)

    note_status = "enabled" if config.notes_parquet else "disabled"
    vital_status = "enabled" if config.include_vitals else "disabled"
    report_lines = [
        "# Rule-Based Baseline Report",
        f"Date: {summary.generated_at_utc}",
        "",
        "## Inputs",
        f"- mimic_root: `{summary.mimic_root}`",
        f"- cohort_duckdb_path: `{summary.cohort_duckdb_path}`",
        f"- cohort_table: `{summary.cohort_table}`",
        f"- notes_parquet: `{summary.notes_parquet}`" if summary.notes_parquet else "- notes_parquet: none",
        f"- high_confidence_window_hours: {config.high_confidence_window_hours}",
        "",
        "## Counts",
        f"- cohort_row_count: {summary.cohort_row_count}",
        f"- suspicion_time_rule_count: {summary.suspicion_time_rule_count}",
        f"- null_rule_count: {summary.null_rule_count}",
        f"- tier1_note_signal_count: {summary.tier1_note_signal_count}",
        f"- tier2_antibiotic_admin_count: {summary.tier2_antibiotic_admin_count}",
        f"- tier2_blood_culture_order_count: {summary.tier2_blood_culture_order_count}",
        f"- tier3_lactate_count: {summary.tier3_lactate_count}",
        f"- tier3_abnormal_wbc_count: {summary.tier3_abnormal_wbc_count}",
        f"- tier3_fever_count: {summary.tier3_fever_count}",
        f"- tier3_hypotension_count: {summary.tier3_hypotension_count}",
        f"- tier3_tachycardia_count: {summary.tier3_tachycardia_count}",
        f"- infection_source_rule_count: {summary.infection_source_rule_count}",
        "",
        "## Design Notes",
        f"- Tier 1 note evidence is {note_status}.",
        f"- Tier 3 vital-sign support is {vital_status}.",
        "- If Tier 1 note evidence exists, it overrides Tier 2 for suspicion_time_rule.",
        "- If Tier 1 is absent, suspicion_time_rule falls back to the earliest Tier 2 action.",
        "- Tier 3 signals support confidence but never define suspicion_time_rule.",
        "- Diagnosis titles can support infection_source_rule only when the title is unambiguous and a Tier 1 or Tier 2 rule already fired.",
        "- X_rule is a deterministic baseline, not a gold label.",
        "",
        "## Outputs",
        f"- `{summary.rule_labels_csv_path}`",
        f"- `{summary.rule_labels_json_path}`",
        f"- `{summary.execution_log_path}`",
    ]
    with open(report_path, "w", encoding="utf-8") as file:
        file.write("\n".join(report_lines) + "\n")

    summary_json_path = config.output_dir / "rule_based_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as file:
        json.dump(asdict(summary), file, indent=2)

    return summary


def run_rule_based_baseline(config: RuleBasedConfig) -> RuleBasedSummary:
    """Execute deterministic v2 baseline generation from MIMIC-IV tables."""

    required_paths = ensure_inputs(config)
    connection = duckdb.connect()
    try:
        connection.execute("PRAGMA threads=4;")
        initialize_source_views(connection, config, required_paths)
        ensure_required_cohort_columns(connection)
        initialize_note_signal_table(connection, config)
        build_rule_tables(connection, include_vitals=config.include_vitals)
        signal_rows = fetch_rows(connection)
        output_rows = build_output_rows(
            signal_rows,
            high_confidence_window_hours=config.high_confidence_window_hours,
        )
        validate_output_rows(output_rows, cohort_row_count=len(signal_rows))
        return write_outputs(output_rows, config)
    finally:
        connection.close()


def main() -> int:
    """CLI entrypoint for the deterministic v2 baseline builder."""

    config = build_config(parse_args())
    summary = run_rule_based_baseline(config)

    print("Rule-based baseline build completed.")
    print(f"Cohort table: {summary.cohort_table}")
    print(f"Notes parquet: {summary.notes_parquet or 'none'}")
    print(f"Vital support: {'enabled' if config.include_vitals else 'disabled'}")
    print(f"Suspicion time labels: {summary.suspicion_time_rule_count}")
    print(f"Null rule labels: {summary.null_rule_count}")
    print(f"CSV output: {summary.rule_labels_csv_path}")
    print(f"Report: {summary.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
