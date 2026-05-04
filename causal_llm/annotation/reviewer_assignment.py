"""Create balanced reviewer assignments for gold-label annotation tasks."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_TASKS_JSONL = Path("derived_data/annotations/gold_annotation_tasks.jsonl")
DEFAULT_OUTPUT_DIR = Path("derived_data/annotations/assignments")
DEFAULT_REPORT_DIR = Path("reports/annotations")
DEFAULT_DOUBLE_REVIEW_FRACTION = 0.2
DEFAULT_RANDOM_SEED = 20260504

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


@dataclass(frozen=True)
class ReviewerAssignmentConfig:
    """Configuration for reviewer assignment generation."""

    tasks_jsonl: Path
    output_dir: Path
    report_dir: Path
    reviewers: tuple[str, ...]
    double_review_fraction: float
    random_seed: int


@dataclass(frozen=True)
class AssignmentRecord:
    """One reviewer-to-task assignment."""

    task_id: str
    reviewer_id: str
    assignment_role: str


@dataclass(frozen=True)
class ReviewerAssignmentSummary:
    """Summary emitted after reviewer packets are created."""

    generated_at_utc: str
    tasks_jsonl: str
    output_dir: str
    report_path: str
    reviewer_count: int
    task_count: int
    assignment_count: int
    overlap_task_count: int
    double_review_fraction: float
    random_seed: int
    reviewer_assignment_counts: dict[str, int]
    assignment_manifest_path: str
    assignment_index_path: str


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for reviewer assignment generation."""

    parser = argparse.ArgumentParser(
        description="Create balanced reviewer assignments for prepared gold-label tasks."
    )
    parser.add_argument(
        "--tasks-jsonl",
        type=Path,
        default=DEFAULT_TASKS_JSONL,
        help="JSONL file produced by `python prepare_gold_annotations.py`.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for reviewer-specific packets and assignment manifests.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Directory for reviewer-assignment markdown reports.",
    )
    parser.add_argument(
        "--reviewers",
        nargs="+",
        required=True,
        help="Reviewer identifiers used to generate balanced assignments.",
    )
    parser.add_argument(
        "--double-review-fraction",
        type=float,
        default=DEFAULT_DOUBLE_REVIEW_FRACTION,
        help="Fraction of tasks that should receive a second reviewer.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed for deterministic assignment generation.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ReviewerAssignmentConfig:
    """Resolve CLI arguments into a normalized reviewer-assignment config."""

    normalized_reviewers = tuple(reviewer.strip() for reviewer in args.reviewers if reviewer.strip())
    if not normalized_reviewers:
        raise ValueError("At least one non-empty reviewer identifier is required.")
    if len(set(normalized_reviewers)) != len(normalized_reviewers):
        raise ValueError("Reviewer identifiers must be unique.")
    if not 0.0 <= args.double_review_fraction <= 1.0:
        raise ValueError("--double-review-fraction must be between 0.0 and 1.0.")
    if args.double_review_fraction > 0.0 and len(normalized_reviewers) < 2:
        raise ValueError("At least two reviewers are required when overlap is requested.")

    return ReviewerAssignmentConfig(
        tasks_jsonl=args.tasks_jsonl.resolve(),
        output_dir=args.output_dir.resolve(),
        report_dir=args.report_dir.resolve(),
        reviewers=normalized_reviewers,
        double_review_fraction=args.double_review_fraction,
        random_seed=args.random_seed,
    )


def ensure_inputs(config: ReviewerAssignmentConfig) -> None:
    """Validate that prepared annotation tasks already exist."""

    if not config.tasks_jsonl.exists():
        raise FileNotFoundError(
            f"Missing task JSONL: {config.tasks_jsonl}\n"
            "Run `python prepare_gold_annotations.py` first."
        )


def load_task_rows(tasks_jsonl: Path) -> list[dict[str, object]]:
    """Load and validate task rows from the task-preparation JSONL artifact."""

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


def calculate_overlap_task_count(task_count: int, fraction: float) -> int:
    """Convert an overlap fraction into a concrete task count."""

    if task_count == 0 or fraction <= 0.0:
        return 0

    overlap_task_count = int(round(task_count * fraction))
    if overlap_task_count == 0:
        return 1
    return min(task_count, overlap_task_count)


def build_assignments(
    task_rows: list[dict[str, object]],
    config: ReviewerAssignmentConfig,
) -> list[AssignmentRecord]:
    """Generate balanced primary and overlapping secondary reviewer assignments."""

    shuffled_tasks = list(task_rows)
    rng = random.Random(config.random_seed)
    rng.shuffle(shuffled_tasks)

    assignments: list[AssignmentRecord] = []
    reviewer_to_index = {
        reviewer_id: reviewer_index
        for reviewer_index, reviewer_id in enumerate(config.reviewers)
    }
    primary_reviewer_by_task: dict[str, str] = {}

    for task_index, task_row in enumerate(shuffled_tasks):
        reviewer_id = config.reviewers[task_index % len(config.reviewers)]
        task_id = str(task_row["task_id"])
        primary_reviewer_by_task[task_id] = reviewer_id
        assignments.append(
            AssignmentRecord(
                task_id=task_id,
                reviewer_id=reviewer_id,
                assignment_role="primary",
            )
        )

    overlap_task_count = calculate_overlap_task_count(
        task_count=len(shuffled_tasks),
        fraction=config.double_review_fraction,
    )
    if overlap_task_count == 0:
        return assignments

    reviewer_count = len(config.reviewers)
    for overlap_index, task_row in enumerate(shuffled_tasks[:overlap_task_count]):
        task_id = str(task_row["task_id"])
        primary_reviewer_id = primary_reviewer_by_task[task_id]
        primary_index = reviewer_to_index[primary_reviewer_id]
        offset = 1 + (overlap_index % (reviewer_count - 1))
        secondary_reviewer_id = config.reviewers[(primary_index + offset) % reviewer_count]
        assignments.append(
            AssignmentRecord(
                task_id=task_id,
                reviewer_id=secondary_reviewer_id,
                assignment_role="secondary",
            )
        )

    return assignments


def sanitize_filename_component(value: str) -> str:
    """Convert a reviewer identifier into a filesystem-safe filename token."""

    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return sanitized or "reviewer"


def write_reviewer_packets(
    task_rows: list[dict[str, object]],
    assignments: list[AssignmentRecord],
    config: ReviewerAssignmentConfig,
    generated_at_utc: str,
) -> dict[str, int]:
    """Write reviewer-specific JSONL task packets and blank annotation CSVs."""

    task_by_id = {str(task_row["task_id"]): task_row for task_row in task_rows}
    assignments_by_reviewer: dict[str, list[AssignmentRecord]] = defaultdict(list)
    for assignment in assignments:
        assignments_by_reviewer[assignment.reviewer_id].append(assignment)

    reviewer_assignment_counts: dict[str, int] = {}
    for reviewer_id in config.reviewers:
        reviewer_assignments = sorted(
            assignments_by_reviewer.get(reviewer_id, []),
            key=lambda item: (item.assignment_role != "primary", item.task_id),
        )
        reviewer_assignment_counts[reviewer_id] = len(reviewer_assignments)

        safe_reviewer_id = sanitize_filename_component(reviewer_id)
        reviewer_jsonl_path = config.output_dir / f"{safe_reviewer_id}_tasks.jsonl"
        reviewer_csv_path = config.output_dir / f"{safe_reviewer_id}_annotations.csv"

        with open(reviewer_jsonl_path, "w", encoding="utf-8") as file:
            for assignment in reviewer_assignments:
                task_row = dict(task_by_id[assignment.task_id])
                task_row["reviewer_id"] = reviewer_id
                task_row["assignment_role"] = assignment.assignment_role
                task_row["assignment_generated_at_utc"] = generated_at_utc
                file.write(json.dumps(task_row, ensure_ascii=False) + "\n")

        with open(reviewer_csv_path, "w", encoding="utf-8", newline="") as file:
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
                    "reviewer_id",
                    "assignment_role",
                    "annotation_status",
                    "suspected_sepsis",
                    "suspicion_time",
                    "infection_source",
                    "evidence_span",
                    "confidence",
                    "reviewed_at",
                ],
            )
            writer.writeheader()
            for assignment in reviewer_assignments:
                task_row = task_by_id[assignment.task_id]
                writer.writerow(
                    {
                        "task_id": task_row["task_id"],
                        "subject_id": task_row["subject_id"],
                        "hadm_id": task_row["hadm_id"],
                        "stay_id": task_row["stay_id"],
                        "note_count": task_row["note_count"],
                        "first_note_time": task_row["first_note_time"],
                        "last_note_time": task_row["last_note_time"],
                        "reviewer_id": reviewer_id,
                        "assignment_role": assignment.assignment_role,
                        "annotation_status": "pending",
                        "suspected_sepsis": "",
                        "suspicion_time": "",
                        "infection_source": "",
                        "evidence_span": "",
                        "confidence": "",
                        "reviewed_at": "",
                    }
                )

    return reviewer_assignment_counts


def write_outputs(
    task_rows: list[dict[str, object]],
    assignments: list[AssignmentRecord],
    config: ReviewerAssignmentConfig,
) -> ReviewerAssignmentSummary:
    """Persist reviewer packets, manifest files, and a markdown report."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.report_dir.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now(timezone.utc)
    generated_at_utc = generated_at.isoformat()
    overlap_task_count = len({assignment.task_id for assignment in assignments if assignment.assignment_role == "secondary"})

    reviewer_assignment_counts = write_reviewer_packets(
        task_rows=task_rows,
        assignments=assignments,
        config=config,
        generated_at_utc=generated_at_utc,
    )

    assignment_manifest_path = config.output_dir / "assignment_manifest.json"
    assignment_index_path = config.output_dir / "assignment_index.csv"
    report_path = config.report_dir / (
        f"{generated_at.strftime('%Y%m%dT%H%M%SZ')}_reviewer_assignment_report.md"
    )

    manifest = {
        "generated_at_utc": generated_at_utc,
        "tasks_jsonl": str(config.tasks_jsonl),
        "reviewers": list(config.reviewers),
        "task_count": len(task_rows),
        "assignment_count": len(assignments),
        "overlap_task_count": overlap_task_count,
        "double_review_fraction": config.double_review_fraction,
        "random_seed": config.random_seed,
        "reviewer_assignment_counts": reviewer_assignment_counts,
        "assignments": [asdict(assignment) for assignment in assignments],
    }
    with open(assignment_manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    task_by_id = {str(task_row["task_id"]): task_row for task_row in task_rows}
    with open(assignment_index_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "task_id",
                "subject_id",
                "hadm_id",
                "stay_id",
                "note_count",
                "reviewer_id",
                "assignment_role",
            ],
        )
        writer.writeheader()
        for assignment in sorted(assignments, key=lambda item: (item.task_id, item.assignment_role, item.reviewer_id)):
            task_row = task_by_id[assignment.task_id]
            writer.writerow(
                {
                    "task_id": assignment.task_id,
                    "subject_id": task_row["subject_id"],
                    "hadm_id": task_row["hadm_id"],
                    "stay_id": task_row["stay_id"],
                    "note_count": task_row["note_count"],
                    "reviewer_id": assignment.reviewer_id,
                    "assignment_role": assignment.assignment_role,
                }
            )

    summary = ReviewerAssignmentSummary(
        generated_at_utc=generated_at_utc,
        tasks_jsonl=str(config.tasks_jsonl),
        output_dir=str(config.output_dir),
        report_path=str(report_path),
        reviewer_count=len(config.reviewers),
        task_count=len(task_rows),
        assignment_count=len(assignments),
        overlap_task_count=overlap_task_count,
        double_review_fraction=config.double_review_fraction,
        random_seed=config.random_seed,
        reviewer_assignment_counts=reviewer_assignment_counts,
        assignment_manifest_path=str(assignment_manifest_path),
        assignment_index_path=str(assignment_index_path),
    )

    report_lines = [
        "# Reviewer Assignment Report",
        f"Date: {summary.generated_at_utc}",
        "",
        "## Inputs",
        f"- tasks_jsonl: `{summary.tasks_jsonl}`",
        f"- reviewers: {', '.join(config.reviewers)}",
        f"- double_review_fraction: {summary.double_review_fraction}",
        f"- random_seed: {summary.random_seed}",
        "",
        "## Outputs",
        f"- task_count: {summary.task_count}",
        f"- assignment_count: {summary.assignment_count}",
        f"- overlap_task_count: {summary.overlap_task_count}",
        f"- assignment_manifest: `{summary.assignment_manifest_path}`",
        f"- assignment_index: `{summary.assignment_index_path}`",
        "",
        "## Reviewer Counts",
    ]
    for reviewer_id, assignment_count in summary.reviewer_assignment_counts.items():
        report_lines.append(f"- {reviewer_id}: {assignment_count}")

    with open(report_path, "w", encoding="utf-8") as file:
        file.write("\n".join(report_lines) + "\n")

    summary_json_path = config.output_dir / "reviewer_assignment_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as file:
        json.dump(asdict(summary), file, indent=2)

    return summary


def run_reviewer_assignment(config: ReviewerAssignmentConfig) -> ReviewerAssignmentSummary:
    """Execute reviewer-assignment generation from prepared task bundles."""

    ensure_inputs(config)
    task_rows = load_task_rows(config.tasks_jsonl)
    assignments = build_assignments(task_rows, config)
    return write_outputs(task_rows, assignments, config)


def main() -> int:
    """CLI entrypoint for reviewer assignment generation."""

    config = build_config(parse_args())
    summary = run_reviewer_assignment(config)

    print("Reviewer assignment generation completed.")
    print(f"Task count: {summary.task_count}")
    print(f"Assignment count: {summary.assignment_count}")
    print(f"Overlap task count: {summary.overlap_task_count}")
    print(f"Assignment manifest: {summary.assignment_manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
