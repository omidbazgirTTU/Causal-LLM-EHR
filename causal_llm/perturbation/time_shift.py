"""Generate synthetic time-shift perturbation datasets from matched analysis tables."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from causal_llm.analysis.dataset_builder import (
    DATASET_FIELDNAMES,
    derive_outcome_fields,
    derive_treatment_fields,
    parse_datetime,
    round_optional,
    serialize_timestamp,
)
from causal_llm.evaluation.extraction_eval import normalize_optional_text, parse_boolean

DEFAULT_ANALYSIS_MANIFEST = Path("derived_data/analysis/analysis_dataset_manifest.json")
DEFAULT_OUTPUT_DIR = Path("derived_data/perturbations")
DEFAULT_REPORT_DIR = Path("reports/perturbations")
DEFAULT_PREDICTORS = ("gold",)
DEFAULT_SHIFT_HOURS = (-3.0, -2.0, -1.0, 1.0, 2.0, 3.0)
PERTURBATION_FIELDNAMES = [
    "synthetic_source_predictor_name",
    "synthetic_source_predictor_type",
    "synthetic_source_predictor_id",
    "synthetic_shift_hours",
    "original_suspicion_time",
    "original_analysis_eligible",
    "original_analysis_exclusion_reason",
] + DATASET_FIELDNAMES


@dataclass(frozen=True)
class TimePerturbationConfig:
    """Configuration for synthetic time-shift perturbation generation."""

    analysis_manifest: Path
    output_dir: Path
    report_dir: Path
    predictors: tuple[str, ...]
    shift_hours: tuple[float, ...]


@dataclass(frozen=True)
class PerturbationSummary:
    """Compact summary of one generated synthetic perturbation dataset."""

    source_predictor_name: str
    source_predictor_type: str
    source_predictor_id: str
    shift_hours: float
    row_count: int
    analysis_eligible_count: int
    missing_time_zero_count: int
    shifted_outside_window_count: int
    treatment_before_time_zero_count: int
    death_before_time_zero_count: int
    early_antibiotic_count: int
    mortality_28d_count: int
    csv_path: str


@dataclass(frozen=True)
class TimePerturbationRunSummary:
    """Summary artifact emitted after synthetic perturbation generation."""

    generated_at_utc: str
    analysis_manifest: str
    output_dir: str
    report_path: str
    perturbation_summaries: list[dict[str, object]]
    summary_csv_path: str
    manifest_path: str


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for synthetic time-shift generation."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic perturbation datasets by shifting suspicion_time "
            "while keeping the underlying cohort rows fixed."
        )
    )
    parser.add_argument(
        "--analysis-manifest",
        type=Path,
        default=DEFAULT_ANALYSIS_MANIFEST,
        help="Manifest produced by `python build_analysis_datasets.py`.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for synthetic perturbation CSVs and manifests.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Directory for perturbation markdown reports.",
    )
    parser.add_argument(
        "--predictor",
        dest="predictors",
        action="append",
        default=None,
        help=(
            "Predictor name from the analysis manifest to perturb. Repeat to perturb "
            "multiple predictors. Defaults to gold only."
        ),
    )
    parser.add_argument(
        "--shift-hours",
        dest="shift_hours",
        type=float,
        action="append",
        default=None,
        help=(
            "One signed hour shift to apply. Repeat for multiple shifts. Defaults "
            "to -3, -2, -1, 1, 2, and 3."
        ),
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TimePerturbationConfig:
    """Resolve CLI arguments into a normalized perturbation config."""

    predictors = tuple(
        normalize_optional_text(value)
        for value in (args.predictors or DEFAULT_PREDICTORS)
    )
    predictors = tuple(value for value in predictors if value is not None)
    if not predictors:
        raise ValueError("At least one predictor must be selected for perturbation.")

    shift_hours = tuple(args.shift_hours or DEFAULT_SHIFT_HOURS)
    if not shift_hours:
        raise ValueError("At least one non-zero shift must be selected.")
    if any(shift_value == 0 for shift_value in shift_hours):
        raise ValueError("Zero-hour perturbations are not allowed.")

    return TimePerturbationConfig(
        analysis_manifest=args.analysis_manifest.resolve(),
        output_dir=args.output_dir.resolve(),
        report_dir=args.report_dir.resolve(),
        predictors=predictors,
        shift_hours=shift_hours,
    )


def ensure_inputs(config: TimePerturbationConfig) -> None:
    """Validate that the analysis manifest exists before perturbation generation."""

    if not config.analysis_manifest.exists():
        raise FileNotFoundError(
            f"Missing analysis manifest: {config.analysis_manifest}\n"
            "Run `python build_analysis_datasets.py` first."
        )


def load_analysis_manifest(path: Path) -> dict[str, Any]:
    """Load the analysis-dataset manifest from JSON."""

    with open(path, "r", encoding="utf-8") as file:
        manifest = json.load(file)
    predictor_summaries = manifest.get("predictor_summaries")
    if not isinstance(predictor_summaries, list) or not predictor_summaries:
        raise ValueError("Analysis manifest is missing a valid `predictor_summaries` list.")
    return manifest


def load_dataset_rows(path: Path) -> list[dict[str, Any]]:
    """Load one predictor-specific analysis dataset from CSV."""

    with open(path, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Analysis dataset CSV contains no rows: {path}")
    return rows


def parse_optional_boolean(value: Any) -> bool | None:
    """Parse optional boolean fields stored in CSV artifacts."""

    normalized = normalize_optional_text(value)
    if normalized is None:
        return None
    return parse_boolean(normalized)


def shift_label(shift_hours: float) -> str:
    """Convert a signed shift value into a stable filename label."""

    formatted = str(int(shift_hours)) if float(shift_hours).is_integer() else str(shift_hours).replace(".", "p")
    if shift_hours < 0:
        return f"minus{formatted.lstrip('-')}h"
    return f"plus{formatted}h"


def apply_time_shift(
    row: dict[str, Any],
    *,
    shift_hours: float,
    early_treatment_window_hours: float,
) -> dict[str, Any]:
    """Shift one analysis row's time zero and recompute downstream fields."""

    original_suspicion_time = parse_datetime(row.get("suspicion_time"))
    shifted_suspicion_time = (
        original_suspicion_time + timedelta(hours=shift_hours)
        if original_suspicion_time is not None
        else None
    )
    study_window_start = parse_datetime(row.get("study_window_start"))
    study_window_end = parse_datetime(row.get("study_window_end"))
    first_antibiotic_time = parse_datetime(row.get("first_antibiotic_time"))
    death_time = parse_datetime(row.get("death_time"))
    icu_intime = parse_datetime(row.get("icu_intime"))

    shifted_outside_window = False
    if shifted_suspicion_time is not None and study_window_start is not None and study_window_end is not None:
        shifted_outside_window = (
            shifted_suspicion_time < study_window_start
            or shifted_suspicion_time > study_window_end
        )

    treatment_fields = derive_treatment_fields(
        time_zero=shifted_suspicion_time,
        first_antibiotic_time=first_antibiotic_time,
        early_window_hours=early_treatment_window_hours,
    )
    outcome_fields = derive_outcome_fields(
        time_zero=shifted_suspicion_time,
        death_time=death_time,
    )

    exclusion_reasons: list[str] = []
    if shifted_suspicion_time is None:
        exclusion_reasons.append("missing_time_zero")
    if shifted_outside_window:
        exclusion_reasons.append("shifted_time_zero_outside_study_window")
    if treatment_fields["treatment_before_time_zero"]:
        exclusion_reasons.append("treatment_before_time_zero")
    if outcome_fields["death_before_time_zero"]:
        exclusion_reasons.append("death_before_time_zero")

    hours_from_icu_to_time_zero = None
    if shifted_suspicion_time is not None and icu_intime is not None:
        hours_from_icu_to_time_zero = round_optional(
            (shifted_suspicion_time - icu_intime).total_seconds() / 3600.0
        )

    return {
        "synthetic_source_predictor_name": row["predictor_name"],
        "synthetic_source_predictor_type": row["predictor_type"],
        "synthetic_source_predictor_id": row["predictor_id"],
        "synthetic_shift_hours": shift_hours,
        "original_suspicion_time": serialize_timestamp(original_suspicion_time),
        "original_analysis_eligible": parse_optional_boolean(row.get("analysis_eligible")),
        "original_analysis_exclusion_reason": normalize_optional_text(row.get("analysis_exclusion_reason")),
        **row,
        "predictor_name": f"synthetic_{row['predictor_name']}_{shift_label(shift_hours)}",
        "predictor_type": "synthetic",
        "predictor_id": f"{row['predictor_name']}|shift={shift_hours}",
        "suspicion_time": serialize_timestamp(shifted_suspicion_time),
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
    }


def summarize_rows(
    *,
    source_predictor_name: str,
    source_predictor_type: str,
    source_predictor_id: str,
    shift_hours: float,
    rows: list[dict[str, Any]],
    csv_path: Path,
) -> PerturbationSummary:
    """Compute one perturbation summary row for the manifest and report."""

    return PerturbationSummary(
        source_predictor_name=source_predictor_name,
        source_predictor_type=source_predictor_type,
        source_predictor_id=source_predictor_id,
        shift_hours=shift_hours,
        row_count=len(rows),
        analysis_eligible_count=sum(parse_optional_boolean(row.get("analysis_eligible")) is True for row in rows),
        missing_time_zero_count=sum(
            normalize_optional_text(row.get("analysis_exclusion_reason")) == "missing_time_zero"
            for row in rows
        ),
        shifted_outside_window_count=sum(
            "shifted_time_zero_outside_study_window"
            in (normalize_optional_text(row.get("analysis_exclusion_reason")) or "")
            for row in rows
        ),
        treatment_before_time_zero_count=sum(
            parse_optional_boolean(row.get("treatment_before_time_zero")) is True
            for row in rows
        ),
        death_before_time_zero_count=sum(
            parse_optional_boolean(row.get("death_before_time_zero")) is True
            for row in rows
        ),
        early_antibiotic_count=sum(
            parse_optional_boolean(row.get("early_antibiotic_within_window")) is True
            for row in rows
        ),
        mortality_28d_count=sum(
            parse_optional_boolean(row.get("mortality_28d")) is True
            for row in rows
        ),
        csv_path=str(csv_path),
    )


def write_dataset_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write one synthetic perturbation dataset to CSV."""

    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=PERTURBATION_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_outputs(
    *,
    config: TimePerturbationConfig,
    perturbation_summaries: list[PerturbationSummary],
) -> TimePerturbationRunSummary:
    """Persist perturbation summaries, manifest JSON, and markdown report."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.report_dir.mkdir(parents=True, exist_ok=True)

    summary_csv_path = config.output_dir / "time_shift_perturbation_summary.csv"
    manifest_path = config.output_dir / "time_shift_perturbation_manifest.json"
    report_path = config.report_dir / (
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_time_shift_perturbation_report.md"
    )

    with open(summary_csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "source_predictor_name",
                "source_predictor_type",
                "source_predictor_id",
                "shift_hours",
                "row_count",
                "analysis_eligible_count",
                "missing_time_zero_count",
                "shifted_outside_window_count",
                "treatment_before_time_zero_count",
                "death_before_time_zero_count",
                "early_antibiotic_count",
                "mortality_28d_count",
                "csv_path",
            ],
        )
        writer.writeheader()
        for summary in perturbation_summaries:
            writer.writerow(asdict(summary))

    run_summary = TimePerturbationRunSummary(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        analysis_manifest=str(config.analysis_manifest),
        output_dir=str(config.output_dir),
        report_path=str(report_path),
        perturbation_summaries=[asdict(summary) for summary in perturbation_summaries],
        summary_csv_path=str(summary_csv_path),
        manifest_path=str(manifest_path),
    )
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(asdict(run_summary), file, indent=2)

    report_lines = [
        "# Time-Shift Perturbation Report",
        f"Date: {run_summary.generated_at_utc}",
        "",
        "## Inputs",
        f"- analysis_manifest: `{config.analysis_manifest}`",
        f"- predictors: {', '.join(config.predictors)}",
        f"- shift_hours: {', '.join(str(value) for value in config.shift_hours)}",
        "",
        "## Generated Datasets",
    ]
    for summary in perturbation_summaries:
        report_lines.append(
            "- "
            + ", ".join(
                [
                    f"source_predictor={summary.source_predictor_name}",
                    f"shift_hours={summary.shift_hours}",
                    f"analysis_eligible_count={summary.analysis_eligible_count}",
                    f"shifted_outside_window_count={summary.shifted_outside_window_count}",
                    f"treatment_before_time_zero_count={summary.treatment_before_time_zero_count}",
                    f"mortality_28d_count={summary.mortality_28d_count}",
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


def generate_time_shifts(config: TimePerturbationConfig) -> TimePerturbationRunSummary:
    """Generate synthetic time-shift perturbation datasets from analysis tables."""

    ensure_inputs(config)
    analysis_manifest = load_analysis_manifest(config.analysis_manifest)
    early_treatment_window_hours = float(analysis_manifest["early_treatment_window_hours"])

    predictor_summaries = {
        str(summary["predictor_name"]): summary
        for summary in analysis_manifest["predictor_summaries"]
    }
    missing_predictors = sorted(set(config.predictors) - set(predictor_summaries))
    if missing_predictors:
        raise ValueError(
            "Requested predictor(s) not found in the analysis manifest: "
            + ", ".join(missing_predictors)
        )

    perturbation_summaries: list[PerturbationSummary] = []
    config.output_dir.mkdir(parents=True, exist_ok=True)

    for predictor_name in config.predictors:
        predictor_summary = predictor_summaries[predictor_name]
        source_rows = load_dataset_rows(Path(str(predictor_summary["csv_path"])).resolve())

        for shift_hours in config.shift_hours:
            shifted_rows = [
                apply_time_shift(
                    row,
                    shift_hours=shift_hours,
                    early_treatment_window_hours=early_treatment_window_hours,
                )
                for row in source_rows
            ]
            csv_path = config.output_dir / (
                f"synthetic_{predictor_name}_{shift_label(shift_hours)}.csv"
            )
            write_dataset_csv(csv_path, shifted_rows)
            perturbation_summaries.append(
                summarize_rows(
                    source_predictor_name=predictor_summary["predictor_name"],
                    source_predictor_type=predictor_summary["predictor_type"],
                    source_predictor_id=predictor_summary["predictor_id"],
                    shift_hours=shift_hours,
                    rows=shifted_rows,
                    csv_path=csv_path,
                )
            )

    return write_outputs(
        config=config,
        perturbation_summaries=perturbation_summaries,
    )


def main() -> int:
    """CLI entrypoint for synthetic time-shift perturbation generation."""

    args = parse_args()
    config = build_config(args)
    summary = generate_time_shifts(config)
    print("Time-shift perturbation generation completed.")
    print(f"Dataset count: {len(summary.perturbation_summaries)}")
    print(f"Manifest: {summary.manifest_path}")
    print(f"Report: {summary.report_path}")
    return 0
