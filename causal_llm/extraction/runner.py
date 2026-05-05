"""Run the same extraction prompt across a three-model OCI panel."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

from openai import BadRequestError

from causal_llm.model_selection import load_model_panel
from causal_llm.oci_support.utils.oci_openai import OciOpenAI

DEFAULT_TASKS_JSONL = Path("derived_data/annotations/finalized/gold_label_tasks.jsonl")
DEFAULT_OUTPUT_DIR = Path("derived_data/extractions")
DEFAULT_REPORT_DIR = Path("reports/extractions")
DEFAULT_MODEL_PANEL = "paper"
DEFAULT_PROMPT_VERSION = "extraction_v1"
DEFAULT_MAX_COMPLETION_TOKENS = 600
DEFAULT_TIMEOUT_SECONDS = 120.0

REQUIRED_TASK_FIELDS = {
    "task_id",
    "subject_id",
    "hadm_id",
    "stay_id",
    "notes",
}
TRUTHY_VALUES = {"1", "true", "t", "yes", "y"}
FALSY_VALUES = {"0", "false", "f", "no", "n"}

SYSTEM_PROMPT = (
    "You are a clinical information extraction model for a measurement-error study. "
    "Use only the provided notes. Do not infer from outcomes or information that is "
    "not stated in the note bundle. Return exactly one JSON object and nothing else."
)

USER_PROMPT_TEMPLATE = """Extract the following structured variables from the provided note bundle.

Return exactly one JSON object with this schema:
{{
  "suspected_sepsis": true or false,
  "suspicion_time": "YYYY-MM-DD HH:MM:SS" or null,
  "infection_source": "string" or null,
  "evidence_span": "short verbatim evidence from the notes" or null,
  "confidence": 0.0 to 1.0
}}

Rules:
- use only the notes shown below
- do not use information after the provided time window
- if suspected_sepsis is false, set suspicion_time, infection_source, and evidence_span to null
- if suspected_sepsis is true, suspicion_time must be the earliest explicit or strongly implied time in the notes
- evidence_span must be copied verbatim from the notes and must support the label
- do not add keys
- do not wrap the JSON in markdown fences

Task metadata:
- task_id: {task_id}
- subject_id: {subject_id}
- hadm_id: {hadm_id}
- stay_id: {stay_id}
- note_count: {note_count}

Notes:
{rendered_notes}
"""


@dataclass(frozen=True)
class ExtractionRunConfig:
    """Configuration for one three-model extraction run."""

    tasks_jsonl: Path
    output_dir: Path
    report_dir: Path
    model_panel: str
    max_completion_tokens: int
    timeout_seconds: float
    limit: int | None


@dataclass(frozen=True)
class ExtractionRunSummary:
    """Summary artifact emitted after an extraction run completes."""

    generated_at_utc: str
    tasks_jsonl: str
    output_dir: str
    report_path: str
    run_manifest_path: str
    model_panel: str
    prompt_version: str
    task_count: int
    completed_task_count: int
    model_outputs: list[dict[str, object]]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the extraction runner."""

    parser = argparse.ArgumentParser(
        description="Run the same extraction prompt across the selected OCI model panel."
    )
    parser.add_argument(
        "--tasks-jsonl",
        type=Path,
        default=DEFAULT_TASKS_JSONL,
        help="Finalized gold-label task JSONL used as the shared note bundle input.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for extraction artifacts.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Directory for extraction markdown reports.",
    )
    parser.add_argument(
        "--model-panel",
        choices=("paper", "dev"),
        default=DEFAULT_MODEL_PANEL,
        help="Model panel loaded from `.agents/MODEL_SELECTION.md`.",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=DEFAULT_MAX_COMPLETION_TOKENS,
        help="Completion token budget for each extraction call.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of tasks to run, for development validation.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ExtractionRunConfig:
    """Resolve CLI arguments into a normalized extraction-run config."""

    if args.max_completion_tokens <= 0:
        raise ValueError("--max-completion-tokens must be positive.")
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive.")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive when provided.")

    return ExtractionRunConfig(
        tasks_jsonl=args.tasks_jsonl.resolve(),
        output_dir=args.output_dir.resolve(),
        report_dir=args.report_dir.resolve(),
        model_panel=args.model_panel,
        max_completion_tokens=args.max_completion_tokens,
        timeout_seconds=args.timeout_seconds,
        limit=args.limit,
    )


def ensure_inputs(config: ExtractionRunConfig) -> None:
    """Validate that the shared task JSONL exists before model execution."""

    if not config.tasks_jsonl.exists():
        raise FileNotFoundError(
            f"Missing task JSONL: {config.tasks_jsonl}\n"
            "Run `python finalize_gold_annotations.py` first."
        )


def load_task_rows(tasks_jsonl: Path, *, limit: int | None) -> list[dict[str, Any]]:
    """Load task rows and validate the minimum note-bundle schema."""

    task_rows: list[dict[str, Any]] = []
    seen_task_ids: set[str] = set()
    with open(tasks_jsonl, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped_line = line.strip()
            if not stripped_line:
                continue
            row = json.loads(stripped_line)
            missing_fields = sorted(REQUIRED_TASK_FIELDS - set(row))
            if missing_fields:
                raise ValueError(
                    f"Task JSONL row {line_number} is missing required fields: "
                    + ", ".join(missing_fields)
                )
            task_id = str(row["task_id"])
            if task_id in seen_task_ids:
                raise ValueError(f"Duplicate task_id detected in task JSONL: {task_id}")
            seen_task_ids.add(task_id)
            notes = row.get("notes")
            if not isinstance(notes, list) or not notes:
                raise ValueError(f"Task {task_id} must contain a non-empty `notes` list.")
            task_rows.append(row)
            if limit is not None and len(task_rows) >= limit:
                break

    if not task_rows:
        raise ValueError(f"No tasks were loaded from {tasks_jsonl}.")

    return task_rows


def render_note_bundle(notes: list[dict[str, Any]]) -> str:
    """Render a stable note bundle string for the shared extraction prompt."""

    rendered_blocks: list[str] = []
    for note_index, note in enumerate(notes, start=1):
        note_id = note.get("note_id") or f"note-{note_index}"
        charttime = note.get("charttime") or "unknown"
        text = str(note.get("text") or "").strip()
        rendered_blocks.append(
            "\n".join(
                [
                    f"[Note {note_index}]",
                    f"note_id: {note_id}",
                    f"charttime: {charttime}",
                    "text:",
                    text,
                ]
            )
        )
    return "\n\n".join(rendered_blocks)


def build_messages(task_row: dict[str, Any]) -> list[dict[str, str]]:
    """Build the shared system and user messages for one extraction task."""

    rendered_notes = render_note_bundle(task_row["notes"])
    user_prompt = USER_PROMPT_TEMPLATE.format(
        task_id=task_row["task_id"],
        subject_id=task_row["subject_id"],
        hadm_id=task_row["hadm_id"],
        stay_id=task_row["stay_id"],
        note_count=len(task_row["notes"]),
        rendered_notes=rendered_notes,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_client(timeout_seconds: float) -> OciOpenAI:
    """Build the synchronous OCI-backed OpenAI-compatible client."""

    from causal_llm.oci_support.config.openai_client_config import load_config

    config = load_config()
    profile = config.get("profile")
    region = config.get("region")
    compartment_id = config.get("compartment_id")
    stage = config.get("oci_stage") or "ppe"
    if not profile or not region or not compartment_id:
        raise ValueError("PROFILE, REGION, and COMPARTMENT_ID must be set in .env.")
    return OciOpenAI(
        profile=profile,
        region=region,
        compartment_id=compartment_id,
        stage=stage,
        timeout=timeout_seconds,
        max_retries=0,
    )


def build_request_variants(
    model_id: str,
    messages: list[dict[str, str]],
    max_completion_tokens: int,
) -> list[dict[str, Any]]:
    """Handle provider differences for completion token parameters."""

    base_request: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "temperature": 0,
    }

    max_completion_variant = dict(base_request)
    max_completion_variant["max_completion_tokens"] = max_completion_tokens
    if model_id.startswith("openai."):
        max_completion_variant["reasoning_effort"] = "low"

    max_tokens_variant = dict(base_request)
    max_tokens_variant["max_tokens"] = max_completion_tokens

    if model_id.startswith("openai."):
        return [max_completion_variant, max_tokens_variant]
    return [max_tokens_variant, max_completion_variant]


def extract_text(response: Any) -> str:
    """Extract assistant text across heterogeneous OCI response shapes."""

    message = response.choices[0].message
    content = getattr(message, "content", None)

    if isinstance(content, str) and content.strip():
        return content.strip()

    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
            else:
                text = getattr(item, "text", None)
            if isinstance(text, str) and text.strip():
                text_parts.append(text.strip())
        if text_parts:
            return " ".join(text_parts)

    return ""


def extract_first_json_object(text: str) -> dict[str, Any]:
    """Extract the first JSON object from a model response string."""

    decoder = json.JSONDecoder()
    for character_index, character in enumerate(text):
        if character != "{":
            continue
        try:
            parsed_value, _ = decoder.raw_decode(text[character_index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed_value, dict):
            return parsed_value
    raise ValueError("No JSON object could be parsed from the model response.")


def parse_boolean(value: Any) -> bool:
    """Parse a model-produced boolean field from common JSON representations."""

    if isinstance(value, bool):
        return value
    normalized = str(value).strip().casefold()
    if normalized in TRUTHY_VALUES:
        return True
    if normalized in FALSY_VALUES:
        return False
    raise ValueError(f"Could not parse suspected_sepsis value {value!r}.")


def normalize_optional_text(value: Any) -> str | None:
    """Normalize optional string-like fields from the model response."""

    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def normalize_timestamp(value: Any) -> str | None:
    """Normalize a model timestamp into a stable ISO-like string."""

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
        raise ValueError(f"Could not parse suspicion_time value {value!r}.")

    if parsed_datetime.tzinfo is not None:
        parsed_datetime = parsed_datetime.astimezone(timezone.utc)
    return parsed_datetime.isoformat(sep=" ", timespec="seconds")


def normalize_confidence(value: Any) -> float:
    """Normalize a model confidence score into the required [0, 1] range."""

    try:
        confidence = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Could not parse confidence value {value!r}.") from exc

    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"Confidence value {confidence} is outside the required [0, 1] range.")
    return round(confidence, 6)


def normalize_extraction_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize the model JSON payload into the study schema."""

    missing_keys = sorted(
        {
            "suspected_sepsis",
            "suspicion_time",
            "infection_source",
            "evidence_span",
            "confidence",
        }
        - set(payload)
    )
    if missing_keys:
        raise ValueError("Model JSON is missing required keys: " + ", ".join(missing_keys))

    suspected_sepsis = parse_boolean(payload["suspected_sepsis"])
    suspicion_time = normalize_timestamp(payload["suspicion_time"])
    infection_source = normalize_optional_text(payload["infection_source"])
    evidence_span = normalize_optional_text(payload["evidence_span"])
    confidence = normalize_confidence(payload["confidence"])

    if suspected_sepsis:
        if suspicion_time is None:
            raise ValueError("suspected_sepsis=true requires a suspicion_time.")
        if infection_source is None:
            raise ValueError("suspected_sepsis=true requires an infection_source.")
        if evidence_span is None:
            raise ValueError("suspected_sepsis=true requires an evidence_span.")
    else:
        suspicion_time = None
        infection_source = None
        evidence_span = None

    return {
        "suspected_sepsis": suspected_sepsis,
        "suspicion_time": suspicion_time,
        "infection_source": infection_source,
        "evidence_span": evidence_span,
        "confidence": confidence,
    }


def run_one_task(
    client: OciOpenAI,
    *,
    model_slot: str,
    model_id: str,
    task_row: dict[str, Any],
    max_completion_tokens: int,
) -> dict[str, Any]:
    """Run one extraction request against one model and normalize the result."""

    messages = build_messages(task_row)
    last_error: Exception | None = None
    raw_response_text = ""
    latency_seconds = 0.0
    usage_total_tokens: int | None = None

    for request_kwargs in build_request_variants(model_id, messages, max_completion_tokens):
        start = perf_counter()
        try:
            response = client.chat.completions.create(**request_kwargs)
            latency_seconds = perf_counter() - start
            raw_response_text = extract_text(response)
            usage = getattr(response, "usage", None)
            usage_total_tokens = getattr(usage, "total_tokens", None)
            parsed_payload = extract_first_json_object(raw_response_text)
            normalized_payload = normalize_extraction_payload(parsed_payload)
            return {
                "task_id": task_row["task_id"],
                "subject_id": task_row["subject_id"],
                "hadm_id": task_row["hadm_id"],
                "stay_id": task_row["stay_id"],
                "model_slot": model_slot,
                "model_id": model_id,
                "prompt_version": DEFAULT_PROMPT_VERSION,
                "extraction_status": "ok",
                "error_detail": None,
                "latency_seconds": round(latency_seconds, 6),
                "usage_total_tokens": usage_total_tokens,
                "raw_response_text": raw_response_text,
                "normalized_extraction": normalized_payload,
            }
        except BadRequestError as exc:
            latency_seconds = perf_counter() - start
            last_error = exc
            message = str(exc)
            if "Unsupported parameter" in message or "not supported with this model" in message:
                continue
            break
        except ValueError as exc:
            latency_seconds = perf_counter() - start
            last_error = exc
            break
        except Exception as exc:  # pragma: no cover - integration path
            latency_seconds = perf_counter() - start
            last_error = exc
            break

    error_detail = str(last_error) if last_error is not None else "Unknown extraction failure."
    status = "request_error"
    if "No JSON object" in error_detail:
        status = "parse_error"
    if "requires" in error_detail or "missing required keys" in error_detail:
        status = "schema_error"
    return {
        "task_id": task_row["task_id"],
        "subject_id": task_row["subject_id"],
        "hadm_id": task_row["hadm_id"],
        "stay_id": task_row["stay_id"],
        "model_slot": model_slot,
        "model_id": model_id,
        "prompt_version": DEFAULT_PROMPT_VERSION,
        "extraction_status": status,
        "error_detail": error_detail,
        "latency_seconds": round(latency_seconds, 6),
        "usage_total_tokens": usage_total_tokens,
        "raw_response_text": raw_response_text,
        "normalized_extraction": None,
    }


def write_model_outputs(
    *,
    model_slot_name: str,
    model_id: str,
    results: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    """Persist one model's extraction results as JSON and CSV artifacts."""

    json_path = output_dir / f"extracted_{model_slot_name}.json"
    csv_path = output_dir / f"extracted_{model_slot_name}.csv"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    with open(csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "task_id",
                "subject_id",
                "hadm_id",
                "stay_id",
                "model_slot",
                "model_id",
                "prompt_version",
                "extraction_status",
                "error_detail",
                "latency_seconds",
                "usage_total_tokens",
                "suspected_sepsis",
                "suspicion_time",
                "infection_source",
                "evidence_span",
                "confidence",
            ],
        )
        writer.writeheader()
        for row in results:
            normalized_extraction = row["normalized_extraction"] or {}
            writer.writerow(
                {
                    **{key: row[key] for key in writer.fieldnames if key in row},
                    "suspected_sepsis": normalized_extraction.get("suspected_sepsis"),
                    "suspicion_time": normalized_extraction.get("suspicion_time"),
                    "infection_source": normalized_extraction.get("infection_source"),
                    "evidence_span": normalized_extraction.get("evidence_span"),
                    "confidence": normalized_extraction.get("confidence"),
                }
            )

    ok_results = [row for row in results if row["extraction_status"] == "ok"]
    average_latency = round(mean(row["latency_seconds"] for row in results), 6) if results else None
    return {
        "model_slot": model_slot_name,
        "model_id": model_id,
        "result_count": len(results),
        "ok_count": len(ok_results),
        "error_count": len(results) - len(ok_results),
        "average_latency_seconds": average_latency,
        "json_path": str(json_path),
        "csv_path": str(csv_path),
    }


def write_run_outputs(
    *,
    task_rows: list[dict[str, Any]],
    model_summaries: list[dict[str, Any]],
    config: ExtractionRunConfig,
    report_path: Path,
    manifest_path: Path,
) -> ExtractionRunSummary:
    """Write the extraction run manifest and markdown report."""

    generated_at_utc = datetime.now(timezone.utc).isoformat()
    summary = ExtractionRunSummary(
        generated_at_utc=generated_at_utc,
        tasks_jsonl=str(config.tasks_jsonl),
        output_dir=str(config.output_dir),
        report_path=str(report_path),
        run_manifest_path=str(manifest_path),
        model_panel=config.model_panel,
        prompt_version=DEFAULT_PROMPT_VERSION,
        task_count=len(task_rows),
        completed_task_count=len(task_rows),
        model_outputs=model_summaries,
    )

    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(asdict(summary), file, indent=2)

    report_lines = [
        "# LLM Extraction Report",
        f"Date: {summary.generated_at_utc}",
        "",
        "## Inputs",
        f"- tasks_jsonl: `{summary.tasks_jsonl}`",
        f"- model_panel: {summary.model_panel}",
        f"- prompt_version: {summary.prompt_version}",
        f"- task_count: {summary.task_count}",
        "",
        "## Model Outputs",
    ]
    for model_summary in summary.model_outputs:
        report_lines.append(
            "- "
            + ", ".join(
                [
                    f"slot={model_summary['model_slot']}",
                    f"model_id={model_summary['model_id']}",
                    f"ok_count={model_summary['ok_count']}",
                    f"error_count={model_summary['error_count']}",
                    f"average_latency_seconds={model_summary['average_latency_seconds']}",
                ]
            )
        )
    report_lines.extend(
        [
            "",
            "## Outputs",
            f"- `{summary.run_manifest_path}`",
            f"- `{summary.report_path}`",
        ]
    )
    with open(report_path, "w", encoding="utf-8") as file:
        file.write("\n".join(report_lines) + "\n")

    return summary


def run_extraction(config: ExtractionRunConfig) -> ExtractionRunSummary:
    """Execute the shared extraction prompt across the selected model panel."""

    ensure_inputs(config)
    task_rows = load_task_rows(config.tasks_jsonl, limit=config.limit)
    model_slots = load_model_panel(config.model_panel)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.report_dir.mkdir(parents=True, exist_ok=True)
    client = build_client(timeout_seconds=config.timeout_seconds)
    try:
        model_summaries: list[dict[str, Any]] = []
        for slot_index, model_slot in enumerate(model_slots, start=1):
            slot_name = f"model_{slot_index}"
            results = [
                run_one_task(
                    client,
                    model_slot=slot_name,
                    model_id=model_slot.model_id,
                    task_row=task_row,
                    max_completion_tokens=config.max_completion_tokens,
                )
                for task_row in task_rows
            ]
            model_summaries.append(
                write_model_outputs(
                    model_slot_name=slot_name,
                    model_id=model_slot.model_id,
                    results=results,
                    output_dir=config.output_dir,
                )
            )
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()

    report_path = config.report_dir / (
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_llm_extraction_report.md"
    )
    manifest_path = config.output_dir / "extraction_run_manifest.json"
    return write_run_outputs(
        task_rows=task_rows,
        model_summaries=model_summaries,
        config=config,
        report_path=report_path,
        manifest_path=manifest_path,
    )


def main() -> int:
    """CLI entrypoint for the three-model extraction runner."""

    config = build_config(parse_args())
    summary = run_extraction(config)

    print("LLM extraction completed.")
    print(f"Model panel: {summary.model_panel}")
    print(f"Task count: {summary.task_count}")
    print(f"Manifest: {summary.run_manifest_path}")
    print(f"Report: {summary.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
