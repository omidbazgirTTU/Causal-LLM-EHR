"""Load the repo's model panels from `.agents/MODEL_SELECTION.md`."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL_SELECTION_PATH = Path(".agents/MODEL_SELECTION.md")
MODEL_LINE_REGEX = re.compile(r"^\s*-\s+`(?P<label>[^`]+)`:\s+`(?P<model_id>[^`]+)`\s*$")


@dataclass(frozen=True)
class ModelSlot:
    """One ordered model slot from the repository model-selection file."""

    slot_name: str
    model_id: str


def _read_model_selection_lines(path: Path) -> list[str]:
    """Read the model-selection markdown file as plain text lines."""

    resolved_path = path.resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Missing model-selection file: {resolved_path}\n"
            "The repository requires `.agents/MODEL_SELECTION.md`."
        )
    with open(resolved_path, "r", encoding="utf-8") as file:
        return file.readlines()


def load_model_panel(panel_name: str, path: Path = DEFAULT_MODEL_SELECTION_PATH) -> list[ModelSlot]:
    """Load an ordered model panel from `.agents/MODEL_SELECTION.md`."""

    normalized_panel_name = panel_name.strip().casefold()
    if normalized_panel_name not in {"paper", "dev"}:
        raise ValueError(f"Unsupported model panel {panel_name!r}. Use 'paper' or 'dev'.")

    lines = _read_model_selection_lines(path)
    current_section: str | None = None
    slots: list[ModelSlot] = []

    for raw_line in lines:
        stripped_line = raw_line.strip()
        if stripped_line == "## Paper Panel":
            current_section = "paper"
            continue
        if stripped_line == "## Development Smoke-Test Panel":
            current_section = "dev"
            continue
        if stripped_line.startswith("## "):
            current_section = None
            continue
        if current_section != normalized_panel_name:
            continue

        match = MODEL_LINE_REGEX.match(raw_line)
        if match is None:
            continue
        slots.append(
            ModelSlot(
                slot_name=match.group("label"),
                model_id=match.group("model_id"),
            )
        )

    if not slots:
        raise ValueError(
            f"No models were found for panel {panel_name!r} in {path.resolve()}."
        )

    return slots
