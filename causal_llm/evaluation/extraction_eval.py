"""Evaluate model extraction outputs and the rule-based baseline against gold labels."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

DEFAULT_GOLD_LABELS_CSV = Path("derived_data/annotations/finalized/gold_labels.csv")
DEFAULT_EXTRACTION_MANIFEST = Path("derived_data/extractions/extraction_run_manifest.json")
DEFAULT_RULE_BASED_CSV = Path("derived_data/rule_based/rule_labels.csv")
DEFAULT_OUTPUT_DIR = Path("derived_data/evaluation")
DEFAULT_REPORT_DIR = Path("reports/evaluation")
TRUTHY_VALUES = {"1", "true", "t", "yes", "y"}
FALSY_VALUES = {"0", "false", "f", "no", "n"}


@dataclass(frozen=True)
class ExtractionEvaluationConfig:
    """Configuration for extraction-vs-gold evaluation."""

    gold_labels_csv: Path
    extraction_manifest: Path
    rule_based_csv: Path | None
    output_dir: Path
    report_dir: Path


@dataclass(frozen=True)
class PredictorEvaluationSummary:
    """Compact metric bundle for one predictor evaluated against gold labels."""

    predictor_name: str
    predictor_type: str
    predictor_id: str
    evaluated_task_count: int
    valid_prediction_count: int
    valid_prediction_rate: float
    true_positive_count: int
    false_positive_count: int
    false_negative_count: int
    true_negative_count: int
    precision: float | None
    recall: float | None
    f1_score: float | None
    time_eval_count: int
    mean_time_error_hours: float | None
    median_time_error_hours: float | None
    within_1h_rate: float | None
    within_3h_rate: float | None
    source_eval_count: int
    infection_source_agreement_rate: float | None
    exact_label_match_rate: float | None


@dataclass(frozen=True)
class ExtractionEvaluationSummary:
    """Summary artifact emitted after extraction evaluation completes."""

    generated_at_utc: str
    gold_labels_csv: str
    extraction_manifest: str
    output_dir: str
    report_path: str
    predictor_summaries: list[dict[str, object]]
    pairwise_disagreement: list[dict[str, object]]
    summary_csv_path: str
    summary_json_path: str


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for extraction evaluation."""

    parser = argparse.ArgumentParser(
        description="Evaluate extracted model labels and X_rule against finalized gold labels."
    )
    parser.add_argument(
        "--gold-labels-csv",
        type=Path,
        default=DEFAULT_GOLD_LABELS_CSV,
        help="Finalized gold labels CSV produced by `python finalize_gold_annotations.py`.",
    )
    parser.add_argument(
        "--extraction-manifest",
        type=Path,
        default=DEFAULT_EXTRACTION_MANIFEST,
        help="Manifest produced by `python run_llm_extraction.py`.",
    )
    parser.add_argument(
        "--rule-based-csv",
        type=Path,
        default=DEFAULT_RULE_BASED_CSV,
        help="Optional rule-based baseline CSV used as the X_rule comparator.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for evaluation artifacts.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Directory for evaluation markdown reports.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ExtractionEvaluationConfig:
    """Resolve CLI arguments into a normalized evaluation config."""

    return ExtractionEvaluationConfig(
        gold_labels_csv=args.gold_labels_csv.resolve(),
        extraction_manifest=args.extraction_manifest.resolve(),
        rule_based_csv=args.rule_based_csv.resolve() if args.rule_based_csv else None,
        output_dir=args.output_dir.resolve(),
        report_dir=args.report_dir.resolve(),
    )


def ensure_inputs(config: ExtractionEvaluationConfig) -> None:
    """Validate that the required gold and extraction artifacts exist."""

    if not config.gold_labels_csv.exists():
        raise FileNotFoundError(
            f"Missing gold labels CSV: {config.gold_labels_csv}\n"
            "Run `python finalize_gold_annotations.py` first."
        )
    if not config.extraction_manifest.exists():
        raise FileNotFoundError(
            f"Missing extraction manifest: {config.extraction_manifest}\n"
            "Run `python run_llm_extraction.py` first."
        )
    if config.rule_based_csv and not config.rule_based_csv.exists():
        raise FileNotFoundError(f"Missing rule-based CSV: {config.rule_based_csv}")


def parse_boolean(value: Any) -> bool:
    """Parse a boolean field stored in CSV or JSON output artifacts."""

    if isinstance(value, bool):
        return value
    normalized = str(value).strip().casefold()
    if normalized in TRUTHY_VALUES:
        return True
    if normalized in FALSY_VALUES:
        return False
    raise ValueError(f"Could not parse boolean value {value!r}.")


def normalize_optional_text(value: Any) -> str | None:
    """Normalize optional string fields."""

    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def normalize_signature_text(value: str | None) -> str | None:
    """Normalize label text for stable comparisons."""

    if value is None:
        return None
    return " ".join(value.casefold().split())


def parse_timestamp(value: Any) -> str | None:
    """Normalize timestamps into a stable string representation when possible."""

    normalized = normalize_optional_text(value)
    if normalized is None:
        return None

    candidate = normalized.replace("Z", "+00:00")
    formats = (
        None,
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
    )
    parsed_datetime = None
    for fmt in formats:
        try:
            if fmt is None:
                parsed_datetime = datetime.fromisoformat(candidate)
            else:
                parsed_datetime = datetime.strptime(candidate, fmt)
            break
        except ValueError:
            continue
    if parsed_datetime is None:
        return None
    if parsed_datetime.tzinfo is not None:
        parsed_datetime = parsed_datetime.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed_datetime.isoformat(sep=" ", timespec="seconds")


def parse_gold_labels(path: Path) -> dict[str, dict[str, Any]]:
    """Load finalized gold labels keyed by task_id."""

    rows_by_task_id: dict[str, dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            task_id = str(row["task_id"]).strip()
            if task_id in rows_by_task_id:
                raise ValueError(f"Duplicate task_id detected in gold labels CSV: {task_id}")
            rows_by_task_id[task_id] = {
                "task_id": task_id,
                "stay_id": str(row["stay_id"]).strip(),
                "suspected_sepsis": parse_boolean(row["suspected_sepsis"]),
                "suspicion_time": parse_timestamp(row.get("suspicion_time")),
                "infection_source": normalize_optional_text(row.get("infection_source")),
                "evidence_span": normalize_optional_text(row.get("evidence_span")),
            }
    if not rows_by_task_id:
        raise ValueError(f"Gold labels CSV contains no rows: {path}")
    return rows_by_task_id


def load_extraction_manifest(path: Path) -> dict[str, Any]:
    """Load the extraction run manifest from JSON."""

    with open(path, "r", encoding="utf-8") as file:
        manifest = json.load(file)
    model_outputs = manifest.get("model_outputs")
    if not isinstance(model_outputs, list) or not model_outputs:
        raise ValueError("Extraction manifest is missing a valid `model_outputs` list.")
    return manifest


def parse_model_outputs(path: Path) -> dict[str, dict[str, Any]]:
    """Load model extraction outputs keyed by task_id."""

    with open(path, "r", encoding="utf-8") as file:
        rows = json.load(file)
    if not isinstance(rows, list):
        raise ValueError(f"Model extraction output must be a JSON list: {path}")

    rows_by_task_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = str(row["task_id"]).strip()
        if task_id in rows_by_task_id:
            raise ValueError(f"Duplicate task_id detected in model output {path}: {task_id}")
        extraction_status = normalize_optional_text(row.get("extraction_status")) or "unknown"
        normalized_extraction = row.get("normalized_extraction")
        if extraction_status == "ok" and not isinstance(normalized_extraction, dict):
            raise ValueError(
                f"Model output row for task {task_id} is marked ok but is missing "
                f"`normalized_extraction`: {path}"
            )
        rows_by_task_id[task_id] = {
            **row,
            "prediction_status": extraction_status,
            "normalized_prediction": normalized_extraction if extraction_status == "ok" else None,
        }
    return rows_by_task_id


def parse_rule_based_rows(path: Path, gold_rows_by_task_id: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Load the X_rule baseline rows keyed by task_id via shared stay_id."""

    task_id_by_stay_id = {
        row["stay_id"]: task_id
        for task_id, row in gold_rows_by_task_id.items()
    }
    rows_by_task_id: dict[str, dict[str, Any]] = {}
    overlapping_stay_ids: set[str] = set()
    with open(path, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            stay_id = str(row["stay_id"]).strip()
            task_id = task_id_by_stay_id.get(stay_id)
            if task_id is None:
                continue
            if stay_id in overlapping_stay_ids:
                raise ValueError(
                    f"Duplicate overlapping stay_id detected in rule-based CSV: {stay_id}"
                )
            overlapping_stay_ids.add(stay_id)
            rows_by_task_id[task_id] = {
                "task_id": task_id,
                "predictor_name": "x_rule",
                "prediction_status": "ok",
                "normalized_prediction": {
                    "suspected_sepsis": normalize_optional_text(row.get("suspicion_time_rule")) is not None,
                    "suspicion_time": parse_timestamp(row.get("suspicion_time_rule")),
                    "infection_source": normalize_optional_text(row.get("infection_source_rule")),
                    "evidence_span": None,
                    "confidence": None,
                },
            }
    missing_task_ids = sorted(set(gold_rows_by_task_id) - set(rows_by_task_id))
    if missing_task_ids:
        preview = ", ".join(missing_task_ids[:5])
        suffix = "..." if len(missing_task_ids) > 5 else ""
        raise ValueError(
            "Rule-based CSV does not cover the gold-label task set. "
            f"Missing {len(missing_task_ids)} task_id(s): {preview}{suffix}"
        )
    return rows_by_task_id


def build_model_prediction_rows(
    manifest: dict[str, Any],
    gold_rows_by_task_id: dict[str, dict[str, Any]],
) -> list[tuple[str, str, dict[str, dict[str, Any]]]]:
    """Load each model output listed in the extraction manifest."""

    predictors: list[tuple[str, str, dict[str, dict[str, Any]]]] = []
    expected_task_ids = set(gold_rows_by_task_id)

    for model_summary in manifest["model_outputs"]:
        model_slot = str(model_summary["model_slot"])
        model_id = str(model_summary["model_id"])
        json_path = Path(str(model_summary["json_path"])).resolve()
        rows_by_task_id = parse_model_outputs(json_path)
        missing_task_ids = sorted(expected_task_ids - set(rows_by_task_id))
        extra_task_ids = sorted(set(rows_by_task_id) - expected_task_ids)
        if missing_task_ids or extra_task_ids:
            raise ValueError(
                f"Model output task mismatch for {model_slot}. "
                f"Missing={len(missing_task_ids)} Extra={len(extra_task_ids)}"
            )
        predictors.append((model_slot, model_id, rows_by_task_id))

    return predictors


def compute_classification_counts(
    gold_rows_by_task_id: dict[str, dict[str, Any]],
    predictor_rows_by_task_id: dict[str, dict[str, Any]],
) -> tuple[int, int, int, int, int]:
    """Compute confusion counts and valid-prediction coverage."""

    true_positive_count = 0
    false_positive_count = 0
    false_negative_count = 0
    true_negative_count = 0
    valid_prediction_count = 0

    for task_id, gold_row in gold_rows_by_task_id.items():
        predictor_row = predictor_rows_by_task_id.get(task_id)
        if predictor_row is None:
            continue
        normalized_prediction = predictor_row.get("normalized_prediction")
        if not isinstance(normalized_prediction, dict):
            continue
        valid_prediction_count += 1
        predicted_positive = bool(normalized_prediction["suspected_sepsis"])
        gold_positive = bool(gold_row["suspected_sepsis"])

        if predicted_positive and gold_positive:
            true_positive_count += 1
        elif predicted_positive and not gold_positive:
            false_positive_count += 1
        elif not predicted_positive and gold_positive:
            false_negative_count += 1
        else:
            true_negative_count += 1

    return (
        true_positive_count,
        false_positive_count,
        false_negative_count,
        true_negative_count,
        valid_prediction_count,
    )


def safe_divide(numerator: int, denominator: int) -> float | None:
    """Return a rounded ratio, or None when the denominator is zero."""

    if denominator == 0:
        return None
    return round(numerator / denominator, 6)


def compute_time_metrics(
    gold_rows_by_task_id: dict[str, dict[str, Any]],
    predictor_rows_by_task_id: dict[str, dict[str, Any]],
) -> tuple[int, float | None, float | None, float | None, float | None]:
    """Compute time-error metrics on true-positive rows with parseable timestamps."""

    absolute_errors_hours: list[float] = []
    within_1h_count = 0
    within_3h_count = 0

    for task_id, gold_row in gold_rows_by_task_id.items():
        predictor_row = predictor_rows_by_task_id.get(task_id)
        if predictor_row is None:
            continue
        normalized_prediction = predictor_row.get("normalized_prediction")
        if not isinstance(normalized_prediction, dict):
            continue
        if not gold_row["suspected_sepsis"] or not normalized_prediction["suspected_sepsis"]:
            continue
        gold_timestamp = parse_timestamp(gold_row["suspicion_time"])
        predicted_timestamp = parse_timestamp(normalized_prediction.get("suspicion_time"))
        if gold_timestamp is None or predicted_timestamp is None:
            continue

        gold_datetime = datetime.fromisoformat(gold_timestamp)
        predicted_datetime = datetime.fromisoformat(predicted_timestamp)
        absolute_error_hours = abs((predicted_datetime - gold_datetime).total_seconds()) / 3600.0
        absolute_errors_hours.append(absolute_error_hours)
        if absolute_error_hours <= 1.0:
            within_1h_count += 1
        if absolute_error_hours <= 3.0:
            within_3h_count += 1

    if not absolute_errors_hours:
        return 0, None, None, None, None

    return (
        len(absolute_errors_hours),
        round(mean(absolute_errors_hours), 6),
        round(median(absolute_errors_hours), 6),
        round(within_1h_count / len(absolute_errors_hours), 6),
        round(within_3h_count / len(absolute_errors_hours), 6),
    )


def compute_source_metrics(
    gold_rows_by_task_id: dict[str, dict[str, Any]],
    predictor_rows_by_task_id: dict[str, dict[str, Any]],
) -> tuple[int, float | None]:
    """Compute infection-source agreement on positive rows with non-null sources."""

    compared_count = 0
    matched_count = 0

    for task_id, gold_row in gold_rows_by_task_id.items():
        predictor_row = predictor_rows_by_task_id.get(task_id)
        if predictor_row is None:
            continue
        normalized_prediction = predictor_row.get("normalized_prediction")
        if not isinstance(normalized_prediction, dict):
            continue
        if not gold_row["suspected_sepsis"] or not normalized_prediction["suspected_sepsis"]:
            continue
        gold_source = normalize_signature_text(gold_row["infection_source"])
        predicted_source = normalize_signature_text(normalized_prediction.get("infection_source"))
        if gold_source is None or predicted_source is None:
            continue
        compared_count += 1
        if gold_source == predicted_source:
            matched_count += 1

    return compared_count, safe_divide(matched_count, compared_count)


def compute_exact_match_rate(
    gold_rows_by_task_id: dict[str, dict[str, Any]],
    predictor_rows_by_task_id: dict[str, dict[str, Any]],
) -> float | None:
    """Compute exact label agreement across classification, time, and source."""

    compared_count = 0
    matched_count = 0
    for task_id, gold_row in gold_rows_by_task_id.items():
        predictor_row = predictor_rows_by_task_id.get(task_id)
        if predictor_row is None:
            continue
        normalized_prediction = predictor_row.get("normalized_prediction")
        if not isinstance(normalized_prediction, dict):
            continue
        compared_count += 1
        if (
            bool(gold_row["suspected_sepsis"]) == bool(normalized_prediction["suspected_sepsis"])
            and parse_timestamp(gold_row["suspicion_time"]) == parse_timestamp(normalized_prediction.get("suspicion_time"))
            and normalize_signature_text(gold_row["infection_source"])
            == normalize_signature_text(normalized_prediction.get("infection_source"))
        ):
            matched_count += 1
    return safe_divide(matched_count, compared_count)


def evaluate_predictor(
    *,
    predictor_name: str,
    predictor_type: str,
    predictor_id: str,
    gold_rows_by_task_id: dict[str, dict[str, Any]],
    predictor_rows_by_task_id: dict[str, dict[str, Any]],
) -> PredictorEvaluationSummary:
    """Evaluate one predictor against finalized gold labels."""

    (
        true_positive_count,
        false_positive_count,
        false_negative_count,
        true_negative_count,
        valid_prediction_count,
    ) = compute_classification_counts(gold_rows_by_task_id, predictor_rows_by_task_id)
    time_eval_count, mean_time_error_hours, median_time_error_hours, within_1h_rate, within_3h_rate = (
        compute_time_metrics(gold_rows_by_task_id, predictor_rows_by_task_id)
    )
    source_eval_count, infection_source_agreement_rate = compute_source_metrics(
        gold_rows_by_task_id,
        predictor_rows_by_task_id,
    )

    precision = safe_divide(true_positive_count, true_positive_count + false_positive_count)
    recall = safe_divide(true_positive_count, true_positive_count + false_negative_count)
    f1_score = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1_score = round((2 * precision * recall) / (precision + recall), 6)

    return PredictorEvaluationSummary(
        predictor_name=predictor_name,
        predictor_type=predictor_type,
        predictor_id=predictor_id,
        evaluated_task_count=len(gold_rows_by_task_id),
        valid_prediction_count=valid_prediction_count,
        valid_prediction_rate=round(valid_prediction_count / len(gold_rows_by_task_id), 6),
        true_positive_count=true_positive_count,
        false_positive_count=false_positive_count,
        false_negative_count=false_negative_count,
        true_negative_count=true_negative_count,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        time_eval_count=time_eval_count,
        mean_time_error_hours=mean_time_error_hours,
        median_time_error_hours=median_time_error_hours,
        within_1h_rate=within_1h_rate,
        within_3h_rate=within_3h_rate,
        source_eval_count=source_eval_count,
        infection_source_agreement_rate=infection_source_agreement_rate,
        exact_label_match_rate=compute_exact_match_rate(gold_rows_by_task_id, predictor_rows_by_task_id),
    )


def build_pairwise_disagreement(
    predictors: list[tuple[str, str, dict[str, dict[str, Any]]]],
) -> list[dict[str, object]]:
    """Build pairwise disagreement summaries across all loaded predictors."""

    disagreement_rows: list[dict[str, object]] = []
    for left_index in range(len(predictors)):
        left_name, left_id, left_rows = predictors[left_index]
        for right_index in range(left_index + 1, len(predictors)):
            right_name, right_id, right_rows = predictors[right_index]
            shared_task_ids = sorted(set(left_rows) & set(right_rows))
            classification_disagreement_count = 0
            source_disagreement_count = 0
            time_disagreement_over_1h_count = 0

            for task_id in shared_task_ids:
                left_prediction = left_rows[task_id].get("normalized_prediction")
                right_prediction = right_rows[task_id].get("normalized_prediction")
                if not isinstance(left_prediction, dict) or not isinstance(right_prediction, dict):
                    continue
                if bool(left_prediction["suspected_sepsis"]) != bool(right_prediction["suspected_sepsis"]):
                    classification_disagreement_count += 1

                left_source = normalize_signature_text(left_prediction.get("infection_source"))
                right_source = normalize_signature_text(right_prediction.get("infection_source"))
                if left_source is not None and right_source is not None and left_source != right_source:
                    source_disagreement_count += 1

                left_time = parse_timestamp(left_prediction.get("suspicion_time"))
                right_time = parse_timestamp(right_prediction.get("suspicion_time"))
                if left_time is not None and right_time is not None:
                    delta_hours = abs(
                        (datetime.fromisoformat(left_time) - datetime.fromisoformat(right_time)).total_seconds()
                    ) / 3600.0
                    if delta_hours > 1.0:
                        time_disagreement_over_1h_count += 1

            disagreement_rows.append(
                {
                    "left_predictor": left_name,
                    "left_predictor_id": left_id,
                    "right_predictor": right_name,
                    "right_predictor_id": right_id,
                    "shared_task_count": len(shared_task_ids),
                    "classification_disagreement_count": classification_disagreement_count,
                    "classification_disagreement_rate": safe_divide(
                        classification_disagreement_count,
                        len(shared_task_ids),
                    ),
                    "source_disagreement_count": source_disagreement_count,
                    "time_disagreement_over_1h_count": time_disagreement_over_1h_count,
                }
            )

    return disagreement_rows


def write_outputs(
    *,
    predictor_summaries: list[PredictorEvaluationSummary],
    pairwise_disagreement: list[dict[str, object]],
    config: ExtractionEvaluationConfig,
) -> ExtractionEvaluationSummary:
    """Persist the evaluation summaries as CSV, JSON, and markdown."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.report_dir.mkdir(parents=True, exist_ok=True)

    generated_at_utc = datetime.now(timezone.utc).isoformat()
    summary_csv_path = config.output_dir / "extraction_evaluation_summary.csv"
    summary_json_path = config.output_dir / "extraction_evaluation_summary.json"
    report_path = config.report_dir / (
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_extraction_evaluation_report.md"
    )

    with open(summary_csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "predictor_name",
                "predictor_type",
                "predictor_id",
                "evaluated_task_count",
                "valid_prediction_count",
                "valid_prediction_rate",
                "true_positive_count",
                "false_positive_count",
                "false_negative_count",
                "true_negative_count",
                "precision",
                "recall",
                "f1_score",
                "time_eval_count",
                "mean_time_error_hours",
                "median_time_error_hours",
                "within_1h_rate",
                "within_3h_rate",
                "source_eval_count",
                "infection_source_agreement_rate",
                "exact_label_match_rate",
            ],
        )
        writer.writeheader()
        for summary in predictor_summaries:
            writer.writerow(asdict(summary))

    payload = {
        "generated_at_utc": generated_at_utc,
        "predictor_summaries": [asdict(summary) for summary in predictor_summaries],
        "pairwise_disagreement": pairwise_disagreement,
    }
    with open(summary_json_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    report_lines = [
        "# Extraction Evaluation Report",
        f"Date: {generated_at_utc}",
        "",
        "## Inputs",
        f"- gold_labels_csv: `{config.gold_labels_csv}`",
        f"- extraction_manifest: `{config.extraction_manifest}`",
        f"- rule_based_csv: `{config.rule_based_csv}`" if config.rule_based_csv else "- rule_based_csv: none",
        "",
        "## Predictor Metrics",
    ]
    for summary in predictor_summaries:
        report_lines.append(
            "- "
            + ", ".join(
                [
                    f"predictor={summary.predictor_name}",
                    f"precision={summary.precision}",
                    f"recall={summary.recall}",
                    f"f1={summary.f1_score}",
                    f"valid_prediction_rate={summary.valid_prediction_rate}",
                    f"time_eval_count={summary.time_eval_count}",
                    f"source_agreement_rate={summary.infection_source_agreement_rate}",
                ]
            )
        )
    report_lines.extend(["", "## Pairwise Disagreement"])
    for row in pairwise_disagreement:
        report_lines.append(
            "- "
            + ", ".join(
                [
                    f"{row['left_predictor']} vs {row['right_predictor']}",
                    f"classification_disagreement_rate={row['classification_disagreement_rate']}",
                    f"source_disagreement_count={row['source_disagreement_count']}",
                    f"time_disagreement_over_1h_count={row['time_disagreement_over_1h_count']}",
                ]
            )
        )
    report_lines.extend(
        [
            "",
            "## Outputs",
            f"- `{summary_csv_path}`",
            f"- `{summary_json_path}`",
            f"- `{report_path}`",
        ]
    )
    with open(report_path, "w", encoding="utf-8") as file:
        file.write("\n".join(report_lines) + "\n")

    return ExtractionEvaluationSummary(
        generated_at_utc=generated_at_utc,
        gold_labels_csv=str(config.gold_labels_csv),
        extraction_manifest=str(config.extraction_manifest),
        output_dir=str(config.output_dir),
        report_path=str(report_path),
        predictor_summaries=[asdict(summary) for summary in predictor_summaries],
        pairwise_disagreement=pairwise_disagreement,
        summary_csv_path=str(summary_csv_path),
        summary_json_path=str(summary_json_path),
    )


def run_evaluation(config: ExtractionEvaluationConfig) -> ExtractionEvaluationSummary:
    """Evaluate all extracted model outputs and the optional X_rule baseline."""

    ensure_inputs(config)
    gold_rows_by_task_id = parse_gold_labels(config.gold_labels_csv)
    manifest = load_extraction_manifest(config.extraction_manifest)
    model_predictors = build_model_prediction_rows(manifest, gold_rows_by_task_id)

    predictor_rows: list[tuple[str, str, str, dict[str, dict[str, Any]]]] = []
    for predictor_name, predictor_id, rows_by_task_id in model_predictors:
        predictor_rows.append(("model", predictor_name, predictor_id, rows_by_task_id))

    if config.rule_based_csv is not None:
        predictor_rows.append(
            (
                "rule_based",
                "x_rule",
                "rule_based_baseline",
                parse_rule_based_rows(config.rule_based_csv, gold_rows_by_task_id),
            )
        )

    predictor_summaries = [
        evaluate_predictor(
            predictor_name=predictor_name,
            predictor_type=predictor_type,
            predictor_id=predictor_id,
            gold_rows_by_task_id=gold_rows_by_task_id,
            predictor_rows_by_task_id=rows_by_task_id,
        )
        for predictor_type, predictor_name, predictor_id, rows_by_task_id in predictor_rows
    ]
    pairwise_disagreement = build_pairwise_disagreement(
        [
            (predictor_name, predictor_id, rows_by_task_id)
            for _, predictor_name, predictor_id, rows_by_task_id in predictor_rows
        ]
    )
    return write_outputs(
        predictor_summaries=predictor_summaries,
        pairwise_disagreement=pairwise_disagreement,
        config=config,
    )


def main() -> int:
    """CLI entrypoint for extraction evaluation."""

    config = build_config(parse_args())
    summary = run_evaluation(config)

    print("Extraction evaluation completed.")
    print(f"Predictor count: {len(summary.predictor_summaries)}")
    print(f"Summary CSV: {summary.summary_csv_path}")
    print(f"Report: {summary.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
