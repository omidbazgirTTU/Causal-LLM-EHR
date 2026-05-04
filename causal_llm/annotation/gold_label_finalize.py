"""Validate reviewer labels and export finalized gold annotations."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

DEFAULT_TASKS_JSONL = Path("derived_data/annotations/gold_annotation_tasks.jsonl")
DEFAULT_OUTPUT_DIR = Path("derived_data/annotations/finalized")
DEFAULT_REPORT_DIR = Path("reports/annotations")

REQUIRED_TASK_FIELDS = {
    "task_id",
    "subject_id",
    "hadm_id",
    "stay_id",
    "note_count",
    "first_note_time",
    "last_note_time",
    "notes",
}
TRUTHY_VALUES = {"1", "true", "t", "yes", "y"}
FALSY_VALUES = {"0", "false", "f", "no", "n"}
PENDING_STATUSES = {"", "pending", "in_progress"}
SKIPPED_STATUSES = {"skipped", "excluded"}
REVIEWER_COMPLETED_STATUSES = {"completed"}


@dataclass(frozen=True)
class GoldLabelFinalizationConfig:
    """Configuration for gold-label finalization."""

    tasks_jsonl: Path
    annotation_csv_paths: tuple[Path, ...]
    adjudication_csv_paths: tuple[Path, ...]
    assignment_manifest_path: Path | None
    output_dir: Path
    report_dir: Path
    allow_unresolved: bool


@dataclass(frozen=True)
class NormalizedAnnotation:
    """A validated annotation row ready for consensus or adjudication logic."""

    task_id: str
    reviewer_id: str
    annotation_status: str
    suspected_sepsis: bool
    suspicion_time: str | None
    infection_source: str | None
    evidence_span: str | None
    confidence: float
    reviewed_at: str | None
    source_path: str
    source_row_number: int
    source_kind: str


@dataclass(frozen=True)
class AnnotationLoadSummary:
    """Counts produced while parsing reviewer or adjudication CSVs."""

    completed_count: int
    pending_count: int
    skipped_count: int


@dataclass(frozen=True)
class GoldLabelFinalizationSummary:
    """Summary emitted after final gold-label export."""

    generated_at_utc: str
    tasks_jsonl: str
    output_dir: str
    report_path: str
    task_count: int
    completed_annotation_count: int
    completed_adjudication_count: int
    pending_annotation_row_count: int
    skipped_annotation_row_count: int
    finalized_task_count: int
    blocking_task_count: int
    single_review_task_count: int
    reviewer_agreement_task_count: int
    reviewer_core_agreement_task_count: int
    adjudicated_task_count: int
    needs_adjudication_task_count: int
    awaiting_assigned_review_task_count: int
    missing_annotation_task_count: int
    double_review_task_count: int
    double_review_exact_agreement_rate: float | None
    double_review_core_agreement_rate: float | None
    missing_assigned_review_count: int
    gold_labels_csv_path: str
    gold_label_tasks_jsonl_path: str
    resolution_index_path: str
    needs_adjudication_csv_path: str
    needs_adjudication_tasks_jsonl_path: str


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for gold-label finalization."""

    parser = argparse.ArgumentParser(
        description="Validate reviewer outputs and export finalized gold labels."
    )
    parser.add_argument(
        "--tasks-jsonl",
        type=Path,
        default=DEFAULT_TASKS_JSONL,
        help="Task JSONL produced by `python prepare_gold_annotations.py`.",
    )
    parser.add_argument(
        "--annotation-csv",
        action="append",
        dest="annotation_csv_paths",
        type=Path,
        required=True,
        help="Completed reviewer annotation CSV. Repeat for multiple reviewers.",
    )
    parser.add_argument(
        "--adjudication-csv",
        action="append",
        dest="adjudication_csv_paths",
        type=Path,
        default=[],
        help="Completed adjudication CSV generated from the needs-adjudication template.",
    )
    parser.add_argument(
        "--assignment-manifest",
        type=Path,
        default=None,
        help="Optional assignment manifest created by `python assign_gold_annotations.py`.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for finalized gold-label artifacts.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Directory for finalization markdown reports.",
    )
    parser.add_argument(
        "--allow-unresolved",
        action="store_true",
        help="Write partial outputs without failing when reviews or adjudication are still incomplete.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> GoldLabelFinalizationConfig:
    """Resolve CLI arguments into a normalized gold-label finalization config."""

    annotation_csv_paths = tuple(path.resolve() for path in args.annotation_csv_paths)
    adjudication_csv_paths = tuple(path.resolve() for path in args.adjudication_csv_paths)
    assignment_manifest_path = args.assignment_manifest.resolve() if args.assignment_manifest else None

    return GoldLabelFinalizationConfig(
        tasks_jsonl=args.tasks_jsonl.resolve(),
        annotation_csv_paths=annotation_csv_paths,
        adjudication_csv_paths=adjudication_csv_paths,
        assignment_manifest_path=assignment_manifest_path,
        output_dir=args.output_dir.resolve(),
        report_dir=args.report_dir.resolve(),
        allow_unresolved=args.allow_unresolved,
    )


def ensure_inputs(config: GoldLabelFinalizationConfig) -> None:
    """Validate that required inputs exist before finalization starts."""

    if not config.tasks_jsonl.exists():
        raise FileNotFoundError(
            f"Missing task JSONL: {config.tasks_jsonl}\n"
            "Run `python prepare_gold_annotations.py` first."
        )
    for path in config.annotation_csv_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing annotation CSV: {path}")
    for path in config.adjudication_csv_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing adjudication CSV: {path}")
    if config.assignment_manifest_path and not config.assignment_manifest_path.exists():
        raise FileNotFoundError(f"Missing assignment manifest: {config.assignment_manifest_path}")


def load_task_rows(tasks_jsonl: Path) -> list[dict[str, object]]:
    """Load task rows from JSONL while validating the required schema."""

    task_rows: list[dict[str, object]] = []
    seen_task_ids: set[str] = set()
    with open(tasks_jsonl, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            missing = sorted(REQUIRED_TASK_FIELDS - set(row))
            if missing:
                raise ValueError(
                    f"Task JSONL row {line_number} is missing required fields: {', '.join(missing)}"
                )
            task_id = str(row["task_id"])
            if task_id in seen_task_ids:
                raise ValueError(f"Duplicate task_id detected in task JSONL: {task_id}")
            seen_task_ids.add(task_id)
            task_rows.append(row)

    if not task_rows:
        raise ValueError(f"Task JSONL contains no tasks: {tasks_jsonl}")

    return task_rows


def load_assignment_manifest(path: Path | None) -> dict[str, set[str]]:
    """Load expected reviewer assignments per task when a manifest is available."""

    if path is None:
        return {}

    with open(path, "r", encoding="utf-8") as file:
        manifest = json.load(file)

    assignments = manifest.get("assignments")
    if not isinstance(assignments, list):
        raise ValueError("Assignment manifest is missing a valid `assignments` list.")

    expected_reviewers_by_task: dict[str, set[str]] = defaultdict(set)
    for assignment in assignments:
        task_id = str(assignment.get("task_id", "")).strip()
        reviewer_id = str(assignment.get("reviewer_id", "")).strip()
        if not task_id or not reviewer_id:
            raise ValueError("Assignment manifest contains a row missing task_id or reviewer_id.")
        expected_reviewers_by_task[task_id].add(reviewer_id)

    return expected_reviewers_by_task


def normalize_optional_text(value: object) -> str | None:
    """Trim optional text fields while preserving meaningful internal content."""

    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def normalize_signature_text(value: str | None) -> str | None:
    """Normalize label text for agreement comparisons."""

    if value is None:
        return None
    return " ".join(value.casefold().split())


def normalize_evidence_span(value: str | None) -> str | None:
    """Normalize an evidence span for stable comparison without changing meaning."""

    if value is None:
        return None
    return "\n".join(line.strip() for line in value.strip().splitlines()) or None


def parse_boolean(value: object, *, field_name: str) -> bool:
    """Parse a human-entered boolean field from common CSV representations."""

    if isinstance(value, bool):
        return value

    normalized = str(value).strip().casefold()
    if normalized in TRUTHY_VALUES:
        return True
    if normalized in FALSY_VALUES:
        return False
    raise ValueError(
        f"Could not parse {field_name!r} value {value!r}. Use yes/no, true/false, or 1/0."
    )


def parse_timestamp(value: object, *, field_name: str) -> str | None:
    """Parse and normalize timestamps used in annotation labels."""

    normalized = normalize_optional_text(value)
    if normalized is None:
        return None

    candidate = normalized.replace("Z", "+00:00")
    timestamp_formats = (
        None,
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
    )
    parsed_datetime = None
    for fmt in timestamp_formats:
        try:
            if fmt is None:
                parsed_datetime = datetime.fromisoformat(candidate)
            else:
                parsed_datetime = datetime.strptime(candidate, fmt)
            break
        except ValueError:
            continue

    if parsed_datetime is None:
        raise ValueError(
            f"Could not parse {field_name!r} value {value!r}. Use ISO-like timestamp formatting."
        )

    if parsed_datetime.tzinfo is not None:
        parsed_datetime = parsed_datetime.astimezone(timezone.utc)
    return parsed_datetime.isoformat(sep=" ", timespec="seconds")


def parse_confidence(value: object) -> float:
    """Parse a confidence score and enforce the expected [0, 1] range."""

    normalized = normalize_optional_text(value)
    if normalized is None:
        raise ValueError("Missing confidence value. Use a numeric score between 0 and 1.")

    try:
        confidence = float(normalized)
    except ValueError as exc:
        raise ValueError(f"Could not parse confidence value {value!r}.") from exc

    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"Confidence value {confidence} is outside the allowed [0, 1] range.")
    return confidence


def validate_annotation_fields(
    *,
    suspected_sepsis: bool,
    suspicion_time: str | None,
    infection_source: str | None,
    evidence_span: str | None,
) -> None:
    """Enforce the structured annotation contract used by the study."""

    if suspected_sepsis:
        if suspicion_time is None:
            raise ValueError("suspected_sepsis=yes requires a suspicion_time.")
        if infection_source is None:
            raise ValueError("suspected_sepsis=yes requires an infection_source.")
        if evidence_span is None:
            raise ValueError("suspected_sepsis=yes requires an evidence_span.")
        return

    if suspicion_time is not None:
        raise ValueError("suspected_sepsis=no must leave suspicion_time blank.")
    if infection_source is not None:
        raise ValueError("suspected_sepsis=no must leave infection_source blank.")


def normalize_reviewer_annotation_row(
    row: dict[str, str],
    *,
    source_path: Path,
    row_number: int,
    valid_task_ids: set[str],
) -> NormalizedAnnotation | None:
    """Normalize a reviewer annotation CSV row into a validated label record."""

    task_id = str(row.get("task_id", "")).strip()
    if not task_id:
        raise ValueError(f"{source_path} row {row_number} is missing task_id.")
    if task_id not in valid_task_ids:
        raise ValueError(f"{source_path} row {row_number} references unknown task_id {task_id!r}.")

    reviewer_id = str(row.get("reviewer_id", "")).strip()
    if not reviewer_id:
        raise ValueError(f"{source_path} row {row_number} is missing reviewer_id.")

    status = str(row.get("annotation_status", "")).strip().casefold()
    suspected_sepsis_value = normalize_optional_text(row.get("suspected_sepsis"))
    suspicion_time_value = normalize_optional_text(row.get("suspicion_time"))
    infection_source_value = normalize_optional_text(row.get("infection_source"))
    evidence_span_value = normalize_optional_text(row.get("evidence_span"))
    confidence_value = normalize_optional_text(row.get("confidence"))
    has_label_content = any(
        value is not None
        for value in (
            suspected_sepsis_value,
            suspicion_time_value,
            infection_source_value,
            evidence_span_value,
            confidence_value,
        )
    )

    if status in PENDING_STATUSES and not has_label_content:
        return None
    if status in SKIPPED_STATUSES and not has_label_content:
        return None
    if status in PENDING_STATUSES and has_label_content:
        status = "completed"
    if status not in REVIEWER_COMPLETED_STATUSES:
        raise ValueError(
            f"{source_path} row {row_number} has unsupported annotation_status {status!r}."
        )

    suspected_sepsis = parse_boolean(
        row.get("suspected_sepsis"),
        field_name="suspected_sepsis",
    )
    suspicion_time = parse_timestamp(
        row.get("suspicion_time"),
        field_name="suspicion_time",
    )
    infection_source = normalize_optional_text(row.get("infection_source"))
    evidence_span = normalize_evidence_span(normalize_optional_text(row.get("evidence_span")))
    confidence = parse_confidence(row.get("confidence"))
    validate_annotation_fields(
        suspected_sepsis=suspected_sepsis,
        suspicion_time=suspicion_time,
        infection_source=infection_source,
        evidence_span=evidence_span,
    )

    return NormalizedAnnotation(
        task_id=task_id,
        reviewer_id=reviewer_id,
        annotation_status=status,
        suspected_sepsis=suspected_sepsis,
        suspicion_time=suspicion_time,
        infection_source=infection_source,
        evidence_span=evidence_span,
        confidence=confidence,
        reviewed_at=parse_timestamp(row.get("reviewed_at"), field_name="reviewed_at"),
        source_path=str(source_path),
        source_row_number=row_number,
        source_kind="annotation",
    )


def normalize_adjudication_row(
    row: dict[str, str],
    *,
    source_path: Path,
    row_number: int,
    valid_task_ids: set[str],
) -> NormalizedAnnotation | None:
    """Normalize an adjudication CSV row into a validated final label record."""

    task_id = str(row.get("task_id", "")).strip()
    if not task_id:
        raise ValueError(f"{source_path} row {row_number} is missing task_id.")
    if task_id not in valid_task_ids:
        raise ValueError(f"{source_path} row {row_number} references unknown task_id {task_id!r}.")

    adjudicator_id = str(row.get("adjudicator_id", row.get("reviewer_id", ""))).strip()
    if not adjudicator_id:
        raise ValueError(f"{source_path} row {row_number} is missing adjudicator_id.")

    suspected_sepsis_value = normalize_optional_text(row.get("final_suspected_sepsis"))
    suspicion_time_value = normalize_optional_text(row.get("final_suspicion_time"))
    infection_source_value = normalize_optional_text(row.get("final_infection_source"))
    evidence_span_value = normalize_optional_text(row.get("final_evidence_span"))
    confidence_value = normalize_optional_text(row.get("final_confidence"))
    has_label_content = any(
        value is not None
        for value in (
            suspected_sepsis_value,
            suspicion_time_value,
            infection_source_value,
            evidence_span_value,
            confidence_value,
        )
    )
    if not has_label_content:
        return None

    suspected_sepsis = parse_boolean(
        row.get("final_suspected_sepsis"),
        field_name="final_suspected_sepsis",
    )
    suspicion_time = parse_timestamp(
        row.get("final_suspicion_time"),
        field_name="final_suspicion_time",
    )
    infection_source = normalize_optional_text(row.get("final_infection_source"))
    evidence_span = normalize_evidence_span(normalize_optional_text(row.get("final_evidence_span")))
    confidence = parse_confidence(row.get("final_confidence"))
    validate_annotation_fields(
        suspected_sepsis=suspected_sepsis,
        suspicion_time=suspicion_time,
        infection_source=infection_source,
        evidence_span=evidence_span,
    )

    return NormalizedAnnotation(
        task_id=task_id,
        reviewer_id=adjudicator_id,
        annotation_status="adjudicated",
        suspected_sepsis=suspected_sepsis,
        suspicion_time=suspicion_time,
        infection_source=infection_source,
        evidence_span=evidence_span,
        confidence=confidence,
        reviewed_at=parse_timestamp(
            row.get("adjudicated_at"),
            field_name="adjudicated_at",
        ),
        source_path=str(source_path),
        source_row_number=row_number,
        source_kind="adjudication",
    )


def load_reviewer_annotations(
    paths: tuple[Path, ...],
    valid_task_ids: set[str],
) -> tuple[list[NormalizedAnnotation], AnnotationLoadSummary]:
    """Load and validate completed reviewer annotations across multiple CSVs."""

    annotations: list[NormalizedAnnotation] = []
    pending_count = 0
    skipped_count = 0
    seen_reviewer_pairs: set[tuple[str, str]] = set()

    for path in paths:
        with open(path, "r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None:
                raise ValueError(f"Annotation CSV has no header row: {path}")
            for row_number, row in enumerate(reader, start=2):
                task_id = str(row.get("task_id", "")).strip()
                reviewer_id = str(row.get("reviewer_id", "")).strip()
                status = str(row.get("annotation_status", "")).strip().casefold()

                annotation = normalize_reviewer_annotation_row(
                    row,
                    source_path=path,
                    row_number=row_number,
                    valid_task_ids=valid_task_ids,
                )
                if annotation is None:
                    if status in SKIPPED_STATUSES:
                        skipped_count += 1
                    else:
                        pending_count += 1
                    continue

                reviewer_pair = (annotation.task_id, annotation.reviewer_id)
                if reviewer_pair in seen_reviewer_pairs:
                    raise ValueError(
                        f"Duplicate completed annotation detected for task_id={task_id!r}, "
                        f"reviewer_id={reviewer_id!r}."
                    )
                seen_reviewer_pairs.add(reviewer_pair)
                annotations.append(annotation)

    summary = AnnotationLoadSummary(
        completed_count=len(annotations),
        pending_count=pending_count,
        skipped_count=skipped_count,
    )
    return annotations, summary


def load_adjudications(
    paths: tuple[Path, ...],
    valid_task_ids: set[str],
) -> tuple[dict[str, NormalizedAnnotation], AnnotationLoadSummary]:
    """Load adjudication CSVs and ensure only one final adjudication exists per task."""

    adjudications: dict[str, NormalizedAnnotation] = {}
    pending_count = 0

    for path in paths:
        with open(path, "r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None:
                raise ValueError(f"Adjudication CSV has no header row: {path}")
            for row_number, row in enumerate(reader, start=2):
                annotation = normalize_adjudication_row(
                    row,
                    source_path=path,
                    row_number=row_number,
                    valid_task_ids=valid_task_ids,
                )
                if annotation is None:
                    pending_count += 1
                    continue
                if annotation.task_id in adjudications:
                    raise ValueError(
                        f"Duplicate adjudication detected for task_id={annotation.task_id!r}."
                    )
                adjudications[annotation.task_id] = annotation

    summary = AnnotationLoadSummary(
        completed_count=len(adjudications),
        pending_count=pending_count,
        skipped_count=0,
    )
    return adjudications, summary


def core_signature(annotation: NormalizedAnnotation) -> tuple[bool, str | None, str | None]:
    """Build the structured-label signature used to detect blocking disagreements."""

    return (
        annotation.suspected_sepsis,
        annotation.suspicion_time,
        normalize_signature_text(annotation.infection_source),
    )


def full_signature(
    annotation: NormalizedAnnotation,
) -> tuple[bool, str | None, str | None, str | None]:
    """Build the full-label signature used for exact agreement metrics."""

    return core_signature(annotation) + (normalize_signature_text(annotation.evidence_span),)


def latest_timestamp(values: list[str | None]) -> str | None:
    """Select the latest non-null timestamp string from a small in-memory list."""

    normalized_values = [value for value in values if value is not None]
    if not normalized_values:
        return None
    return max(normalized_values)


def build_final_annotation(
    annotations: list[NormalizedAnnotation],
    *,
    resolution_status: str,
) -> dict[str, object]:
    """Convert one or more reviewer annotations into the finalized label payload."""

    if len(annotations) == 1:
        annotation = annotations[0]
        return {
            "review_resolution": resolution_status,
            "suspected_sepsis": annotation.suspected_sepsis,
            "suspicion_time": annotation.suspicion_time,
            "infection_source": annotation.infection_source,
            "evidence_span": annotation.evidence_span,
            "confidence": annotation.confidence,
            "final_reviewer_id": annotation.reviewer_id,
            "reviewed_at": annotation.reviewed_at,
            "source_annotation_count": 1,
            "supporting_reviewer_ids": [annotation.reviewer_id],
        }

    unique_evidence_spans: list[str] = []
    seen_evidence_spans: set[str] = set()
    for annotation in annotations:
        if annotation.evidence_span is None:
            continue
        normalized_span = normalize_signature_text(annotation.evidence_span)
        if normalized_span in seen_evidence_spans:
            continue
        seen_evidence_spans.add(normalized_span)
        unique_evidence_spans.append(annotation.evidence_span)

    reference = annotations[0]
    return {
        "review_resolution": resolution_status,
        "suspected_sepsis": reference.suspected_sepsis,
        "suspicion_time": reference.suspicion_time,
        "infection_source": reference.infection_source,
        "evidence_span": " || ".join(unique_evidence_spans) if unique_evidence_spans else None,
        "confidence": round(mean(annotation.confidence for annotation in annotations), 6),
        "final_reviewer_id": "|".join(annotation.reviewer_id for annotation in annotations),
        "reviewed_at": latest_timestamp([annotation.reviewed_at for annotation in annotations]),
        "source_annotation_count": len(annotations),
        "supporting_reviewer_ids": [annotation.reviewer_id for annotation in annotations],
    }


def build_resolution_row(
    task_row: dict[str, object],
    *,
    resolution_status: str,
    completed_annotations: list[NormalizedAnnotation],
    expected_reviewer_ids: set[str],
    missing_expected_reviewers: set[str],
) -> dict[str, object]:
    """Create a compact per-task resolution record for audit and progress tracking."""

    reviewer_ids = [annotation.reviewer_id for annotation in completed_annotations]
    return {
        "task_id": task_row["task_id"],
        "subject_id": task_row["subject_id"],
        "hadm_id": task_row["hadm_id"],
        "stay_id": task_row["stay_id"],
        "note_count": task_row["note_count"],
        "resolution_status": resolution_status,
        "completed_annotation_count": len(completed_annotations),
        "reviewer_ids": "|".join(reviewer_ids),
        "expected_reviewer_ids": "|".join(sorted(expected_reviewer_ids)),
        "missing_expected_reviewers": "|".join(sorted(missing_expected_reviewers)),
    }


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a list of dictionaries as newline-delimited JSON."""

    with open(path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_adjudication_rows(
    conflict_entries: list[tuple[dict[str, object], list[NormalizedAnnotation]]]
) -> tuple[list[dict[str, object]], int]:
    """Create a human-readable adjudication table from conflicting reviewer labels."""

    max_annotation_count = max((len(annotations) for _, annotations in conflict_entries), default=0)
    adjudication_rows: list[dict[str, object]] = []

    for task_row, annotations in conflict_entries:
        row: dict[str, object] = {
            "task_id": task_row["task_id"],
            "subject_id": task_row["subject_id"],
            "hadm_id": task_row["hadm_id"],
            "stay_id": task_row["stay_id"],
            "note_count": task_row["note_count"],
            "first_note_time": task_row["first_note_time"],
            "last_note_time": task_row["last_note_time"],
        }
        for annotation_index in range(max_annotation_count):
            prefix = f"reviewer_{annotation_index + 1}"
            if annotation_index < len(annotations):
                annotation = annotations[annotation_index]
                row.update(
                    {
                        f"{prefix}_id": annotation.reviewer_id,
                        f"{prefix}_suspected_sepsis": annotation.suspected_sepsis,
                        f"{prefix}_suspicion_time": annotation.suspicion_time or "",
                        f"{prefix}_infection_source": annotation.infection_source or "",
                        f"{prefix}_evidence_span": annotation.evidence_span or "",
                        f"{prefix}_confidence": annotation.confidence,
                        f"{prefix}_reviewed_at": annotation.reviewed_at or "",
                    }
                )
            else:
                row.update(
                    {
                        f"{prefix}_id": "",
                        f"{prefix}_suspected_sepsis": "",
                        f"{prefix}_suspicion_time": "",
                        f"{prefix}_infection_source": "",
                        f"{prefix}_evidence_span": "",
                        f"{prefix}_confidence": "",
                        f"{prefix}_reviewed_at": "",
                    }
                )

        row.update(
            {
                "final_suspected_sepsis": "",
                "final_suspicion_time": "",
                "final_infection_source": "",
                "final_evidence_span": "",
                "final_confidence": "",
                "adjudicator_id": "",
                "adjudicated_at": "",
                "adjudication_notes": "",
            }
        )
        adjudication_rows.append(row)

    return adjudication_rows, max_annotation_count


def write_outputs(
    *,
    task_rows: list[dict[str, object]],
    reviewer_annotations: list[NormalizedAnnotation],
    adjudications: dict[str, NormalizedAnnotation],
    reviewer_load_summary: AnnotationLoadSummary,
    adjudication_load_summary: AnnotationLoadSummary,
    expected_reviewers_by_task: dict[str, set[str]],
    config: GoldLabelFinalizationConfig,
) -> GoldLabelFinalizationSummary:
    """Resolve reviewer labels, persist outputs, and generate a markdown report."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.report_dir.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now(timezone.utc)
    annotations_by_task: dict[str, list[NormalizedAnnotation]] = defaultdict(list)
    observed_reviewers_by_task: dict[str, set[str]] = defaultdict(set)
    for annotation in reviewer_annotations:
        annotations_by_task[annotation.task_id].append(annotation)
        observed_reviewers_by_task[annotation.task_id].add(annotation.reviewer_id)

    final_gold_rows: list[dict[str, object]] = []
    gold_task_rows: list[dict[str, object]] = []
    resolution_rows: list[dict[str, object]] = []
    conflict_entries: list[tuple[dict[str, object], list[NormalizedAnnotation]]] = []
    conflict_task_rows: list[dict[str, object]] = []

    single_review_task_count = 0
    reviewer_agreement_task_count = 0
    reviewer_core_agreement_task_count = 0
    adjudicated_task_count = 0
    needs_adjudication_task_count = 0
    awaiting_assigned_review_task_count = 0
    missing_annotation_task_count = 0
    double_review_task_count = 0
    double_review_exact_agreement_count = 0
    double_review_core_agreement_count = 0
    missing_assigned_review_count = 0

    for task_row in task_rows:
        task_id = str(task_row["task_id"])
        completed_annotations = sorted(
            annotations_by_task.get(task_id, []),
            key=lambda annotation: (annotation.reviewer_id, annotation.source_row_number),
        )
        expected_reviewer_ids = expected_reviewers_by_task.get(task_id, set())
        observed_reviewer_ids = observed_reviewers_by_task.get(task_id, set())
        missing_expected_reviewers = expected_reviewer_ids - observed_reviewer_ids

        if task_id in adjudications:
            missing_expected_reviewers = set()
            final_annotation = build_final_annotation(
                [adjudications[task_id]],
                resolution_status="adjudicated",
            )
            final_annotation["source_annotation_count"] = len(completed_annotations) + 1
            final_annotation["supporting_reviewer_ids"] = [
                *(annotation.reviewer_id for annotation in completed_annotations),
                adjudications[task_id].reviewer_id,
            ]
            adjudicated_task_count += 1
            resolution_status = "adjudicated"
        elif expected_reviewer_ids and missing_expected_reviewers:
            missing_assigned_review_count += len(missing_expected_reviewers)
            resolution_status = "awaiting_assigned_review"
            final_annotation = None
            awaiting_assigned_review_task_count += 1
        elif not completed_annotations:
            resolution_status = "missing_annotation"
            final_annotation = None
            missing_annotation_task_count += 1
        elif len(completed_annotations) == 1:
            resolution_status = "single_review"
            final_annotation = build_final_annotation(
                completed_annotations,
                resolution_status=resolution_status,
            )
            single_review_task_count += 1
        else:
            double_review_task_count += 1
            exact_signatures = {full_signature(annotation) for annotation in completed_annotations}
            core_signatures = {core_signature(annotation) for annotation in completed_annotations}
            if len(core_signatures) == 1:
                double_review_core_agreement_count += 1
                if len(exact_signatures) == 1:
                    resolution_status = "reviewer_agreement"
                    reviewer_agreement_task_count += 1
                    double_review_exact_agreement_count += 1
                else:
                    resolution_status = "reviewer_core_agreement"
                    reviewer_core_agreement_task_count += 1
                final_annotation = build_final_annotation(
                    completed_annotations,
                    resolution_status=resolution_status,
                )
            else:
                resolution_status = "needs_adjudication"
                final_annotation = None
                needs_adjudication_task_count += 1

        resolution_rows.append(
            build_resolution_row(
                task_row,
                resolution_status=resolution_status,
                completed_annotations=completed_annotations,
                expected_reviewer_ids=expected_reviewer_ids,
                missing_expected_reviewers=missing_expected_reviewers,
            )
        )

        if final_annotation is None:
            if resolution_status == "needs_adjudication":
                conflict_entries.append((task_row, completed_annotations))
                conflict_task_rows.append(
                    {
                        **task_row,
                        "reviewer_annotations": [asdict(annotation) for annotation in completed_annotations],
                    }
                )
            continue

        final_gold_row = {
            "task_id": task_row["task_id"],
            "subject_id": task_row["subject_id"],
            "hadm_id": task_row["hadm_id"],
            "stay_id": task_row["stay_id"],
            "note_count": task_row["note_count"],
            "first_note_time": task_row["first_note_time"],
            "last_note_time": task_row["last_note_time"],
            **final_annotation,
        }
        final_gold_rows.append(final_gold_row)
        gold_task_rows.append(
            {
                **task_row,
                "gold_annotation": final_gold_row,
            }
        )

    gold_labels_csv_path = config.output_dir / "gold_labels.csv"
    gold_label_tasks_jsonl_path = config.output_dir / "gold_label_tasks.jsonl"
    resolution_index_path = config.output_dir / "gold_resolution_index.csv"
    needs_adjudication_csv_path = config.output_dir / "needs_adjudication.csv"
    needs_adjudication_tasks_jsonl_path = config.output_dir / "needs_adjudication_tasks.jsonl"
    report_path = config.report_dir / (
        f"{generated_at.strftime('%Y%m%dT%H%M%SZ')}_gold_label_finalization_report.md"
    )

    with open(gold_labels_csv_path, "w", encoding="utf-8", newline="") as file:
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
                "review_resolution",
                "suspected_sepsis",
                "suspicion_time",
                "infection_source",
                "evidence_span",
                "confidence",
                "final_reviewer_id",
                "reviewed_at",
                "source_annotation_count",
                "supporting_reviewer_ids",
            ],
        )
        writer.writeheader()
        for row in final_gold_rows:
            writer.writerow(
                {
                    **row,
                    "supporting_reviewer_ids": "|".join(row["supporting_reviewer_ids"]),
                }
            )

    write_jsonl(gold_label_tasks_jsonl_path, gold_task_rows)

    with open(resolution_index_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "task_id",
                "subject_id",
                "hadm_id",
                "stay_id",
                "note_count",
                "resolution_status",
                "completed_annotation_count",
                "reviewer_ids",
                "expected_reviewer_ids",
                "missing_expected_reviewers",
            ],
        )
        writer.writeheader()
        for row in resolution_rows:
            writer.writerow(row)

    adjudication_rows, max_annotation_count = build_adjudication_rows(conflict_entries)
    adjudication_fieldnames = [
        "task_id",
        "subject_id",
        "hadm_id",
        "stay_id",
        "note_count",
        "first_note_time",
        "last_note_time",
    ]
    for annotation_index in range(max_annotation_count):
        prefix = f"reviewer_{annotation_index + 1}"
        adjudication_fieldnames.extend(
            [
                f"{prefix}_id",
                f"{prefix}_suspected_sepsis",
                f"{prefix}_suspicion_time",
                f"{prefix}_infection_source",
                f"{prefix}_evidence_span",
                f"{prefix}_confidence",
                f"{prefix}_reviewed_at",
            ]
        )
    adjudication_fieldnames.extend(
        [
            "final_suspected_sepsis",
            "final_suspicion_time",
            "final_infection_source",
            "final_evidence_span",
            "final_confidence",
            "adjudicator_id",
            "adjudicated_at",
            "adjudication_notes",
        ]
    )

    with open(needs_adjudication_csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=adjudication_fieldnames)
        writer.writeheader()
        for row in adjudication_rows:
            writer.writerow(row)

    write_jsonl(needs_adjudication_tasks_jsonl_path, conflict_task_rows)

    double_review_exact_agreement_rate = None
    double_review_core_agreement_rate = None
    if double_review_task_count:
        double_review_exact_agreement_rate = round(
            double_review_exact_agreement_count / double_review_task_count,
            6,
        )
        double_review_core_agreement_rate = round(
            double_review_core_agreement_count / double_review_task_count,
            6,
        )

    blocking_task_count = (
        needs_adjudication_task_count
        + awaiting_assigned_review_task_count
        + missing_annotation_task_count
    )
    summary = GoldLabelFinalizationSummary(
        generated_at_utc=generated_at.isoformat(),
        tasks_jsonl=str(config.tasks_jsonl),
        output_dir=str(config.output_dir),
        report_path=str(report_path),
        task_count=len(task_rows),
        completed_annotation_count=reviewer_load_summary.completed_count,
        completed_adjudication_count=adjudication_load_summary.completed_count,
        pending_annotation_row_count=reviewer_load_summary.pending_count,
        skipped_annotation_row_count=reviewer_load_summary.skipped_count,
        finalized_task_count=len(final_gold_rows),
        blocking_task_count=blocking_task_count,
        single_review_task_count=single_review_task_count,
        reviewer_agreement_task_count=reviewer_agreement_task_count,
        reviewer_core_agreement_task_count=reviewer_core_agreement_task_count,
        adjudicated_task_count=adjudicated_task_count,
        needs_adjudication_task_count=needs_adjudication_task_count,
        awaiting_assigned_review_task_count=awaiting_assigned_review_task_count,
        missing_annotation_task_count=missing_annotation_task_count,
        double_review_task_count=double_review_task_count,
        double_review_exact_agreement_rate=double_review_exact_agreement_rate,
        double_review_core_agreement_rate=double_review_core_agreement_rate,
        missing_assigned_review_count=missing_assigned_review_count,
        gold_labels_csv_path=str(gold_labels_csv_path),
        gold_label_tasks_jsonl_path=str(gold_label_tasks_jsonl_path),
        resolution_index_path=str(resolution_index_path),
        needs_adjudication_csv_path=str(needs_adjudication_csv_path),
        needs_adjudication_tasks_jsonl_path=str(needs_adjudication_tasks_jsonl_path),
    )

    report_lines = [
        "# Gold Label Finalization Report",
        f"Date: {summary.generated_at_utc}",
        "",
        "## Inputs",
        f"- tasks_jsonl: `{summary.tasks_jsonl}`",
        f"- annotation_csv_count: {len(config.annotation_csv_paths)}",
        f"- adjudication_csv_count: {len(config.adjudication_csv_paths)}",
        f"- assignment_manifest: `{config.assignment_manifest_path}`" if config.assignment_manifest_path else "- assignment_manifest: none",
        "",
        "## Review Counts",
        f"- completed_annotation_count: {summary.completed_annotation_count}",
        f"- completed_adjudication_count: {summary.completed_adjudication_count}",
        f"- pending_annotation_row_count: {summary.pending_annotation_row_count}",
        f"- skipped_annotation_row_count: {summary.skipped_annotation_row_count}",
        "",
        "## Resolution Counts",
        f"- finalized_task_count: {summary.finalized_task_count}",
        f"- blocking_task_count: {summary.blocking_task_count}",
        f"- single_review_task_count: {summary.single_review_task_count}",
        f"- reviewer_agreement_task_count: {summary.reviewer_agreement_task_count}",
        f"- reviewer_core_agreement_task_count: {summary.reviewer_core_agreement_task_count}",
        f"- adjudicated_task_count: {summary.adjudicated_task_count}",
        f"- needs_adjudication_task_count: {summary.needs_adjudication_task_count}",
        f"- awaiting_assigned_review_task_count: {summary.awaiting_assigned_review_task_count}",
        f"- missing_annotation_task_count: {summary.missing_annotation_task_count}",
        f"- missing_assigned_review_count: {summary.missing_assigned_review_count}",
        "",
        "## Agreement",
        f"- double_review_task_count: {summary.double_review_task_count}",
        f"- double_review_exact_agreement_rate: {summary.double_review_exact_agreement_rate}",
        f"- double_review_core_agreement_rate: {summary.double_review_core_agreement_rate}",
        "",
        "## Outputs",
        f"- gold_labels_csv: `{summary.gold_labels_csv_path}`",
        f"- gold_label_tasks_jsonl: `{summary.gold_label_tasks_jsonl_path}`",
        f"- resolution_index: `{summary.resolution_index_path}`",
        f"- needs_adjudication_csv: `{summary.needs_adjudication_csv_path}`",
        f"- needs_adjudication_tasks_jsonl: `{summary.needs_adjudication_tasks_jsonl_path}`",
    ]
    with open(report_path, "w", encoding="utf-8") as file:
        file.write("\n".join(report_lines) + "\n")

    summary_json_path = config.output_dir / "gold_label_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as file:
        json.dump(asdict(summary), file, indent=2)

    return summary


def run_gold_label_finalization(config: GoldLabelFinalizationConfig) -> GoldLabelFinalizationSummary:
    """Execute reviewer-label validation and gold-label export."""

    ensure_inputs(config)
    task_rows = load_task_rows(config.tasks_jsonl)
    valid_task_ids = {str(task_row["task_id"]) for task_row in task_rows}
    expected_reviewers_by_task = load_assignment_manifest(config.assignment_manifest_path)
    reviewer_annotations, reviewer_load_summary = load_reviewer_annotations(
        config.annotation_csv_paths,
        valid_task_ids,
    )
    adjudications, adjudication_load_summary = load_adjudications(
        config.adjudication_csv_paths,
        valid_task_ids,
    )
    return write_outputs(
        task_rows=task_rows,
        reviewer_annotations=reviewer_annotations,
        adjudications=adjudications,
        reviewer_load_summary=reviewer_load_summary,
        adjudication_load_summary=adjudication_load_summary,
        expected_reviewers_by_task=expected_reviewers_by_task,
        config=config,
    )


def main() -> int:
    """CLI entrypoint for gold-label finalization."""

    config = build_config(parse_args())
    summary = run_gold_label_finalization(config)

    print("Gold label finalization completed.")
    print(f"Finalized task count: {summary.finalized_task_count}")
    print(f"Blocking task count: {summary.blocking_task_count}")
    print(f"Gold labels CSV: {summary.gold_labels_csv_path}")
    print(f"Needs adjudication CSV: {summary.needs_adjudication_csv_path}")

    if summary.blocking_task_count > 0 and not config.allow_unresolved:
        print(
            "Blocking issues remain. Complete assigned reviews or adjudication, then rerun."
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
