"""
Extraction of evaluation instances from PoetaV2 benchmark tasks.

Each task is parsed into a list of EvalInstance objects containing the
text fields relevant for contamination detection.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class EvalInstance:
    """A single evaluation instance extracted from a PoetaV2 task."""

    task_name: str
    instance_id: str
    split: str  # e.g. "test", "validation"
    question: str = ""
    context: str = ""
    answer: str = ""
    choices: list[str] = field(default_factory=list)

    @property
    def all_text_fields(self) -> list[str]:
        """Return all non-empty text fields for this instance."""
        fields = [self.question, self.context, self.answer] + self.choices
        return [f for f in fields if f.strip()]


def extract_from_jsonl(path: Path, task_name: str, min_length: int = 20) -> list[EvalInstance]:
    """Extract evaluation instances from a JSONL file.

    Parameters
    ----------
    path : Path
        Path to the JSONL file.
    task_name : str
        Name of the PoetaV2 task.
    min_length : int
        Minimum character length for a text field to be included.

    Returns
    -------
    list[EvalInstance]
        Extracted instances.
    """
    instances: list[EvalInstance] = []

    with open(path, encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            if not line.strip():
                continue
            row = json.loads(line)
            inst = EvalInstance(
                task_name=task_name,
                instance_id=row.get("id", str(idx)),
                split=row.get("split", "test"),
                question=row.get("question", row.get("input", "")),
                context=row.get("context", row.get("passage", "")),
                answer=row.get("answer", row.get("target", row.get("output", ""))),
                choices=row.get("choices", row.get("options", [])),
            )

            # Filter out instances with no text long enough
            if any(len(f) >= min_length for f in inst.all_text_fields):
                instances.append(inst)

    logger.info("Extracted %d instances from %s (%s)", len(instances), task_name, path.name)
    return instances


def extract_all_tasks(
    poetav2_dir: Path,
    tasks: list[str] | None = None,
    min_length: int = 20,
) -> list[EvalInstance]:
    """Extract instances from all (or selected) PoetaV2 tasks.

    Parameters
    ----------
    poetav2_dir : Path
        Root directory containing PoetaV2 task folders/files.
    tasks : list[str] or None
        If provided, only extract these task names. Otherwise, extract all.
    min_length : int
        Minimum character length for text fields.

    Returns
    -------
    list[EvalInstance]
        All extracted instances across tasks.
    """
    all_instances: list[EvalInstance] = []

    for path in sorted(poetav2_dir.rglob("*.jsonl")):
        task_name = path.stem
        if tasks and task_name not in tasks:
            continue
        all_instances.extend(extract_from_jsonl(path, task_name, min_length))

    logger.info("Total: %d instances from %d files", len(all_instances), len({i.task_name for i in all_instances}))
    return all_instances
