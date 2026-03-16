"""Helpers for comparing semantic overlap runs across embedding models."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SemanticRun:
    """Loaded semantic overlap run artifacts."""

    run_dir: Path
    metadata: dict
    instances: list[dict[str, str]]
    summary: list[dict[str, str]]

    @property
    def model_name(self) -> str:
        return str(self.metadata.get("model_name", "unknown"))

    @property
    def threshold(self) -> float:
        return float(self.metadata.get("threshold", 0.0))

    @property
    def total_instances(self) -> int:
        return int(self.metadata.get("total_instances", len(self.instances)))

    @property
    def overlapping_instances(self) -> int:
        return int(self.metadata.get("overlapping_instances", 0))

    @property
    def overlap_rate(self) -> float:
        total = self.total_instances
        return self.overlapping_instances / total if total else 0.0


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def load_semantic_run(run_dir: str | Path) -> SemanticRun:
    """Load one semantic run from a result directory."""
    run_path = Path(run_dir)
    metadata_path = run_path / "tables" / "semantic_overlap_run_metadata.json"
    instances_path = run_path / "tables" / "semantic_overlap_instances.csv"
    summary_path = run_path / "tables" / "semantic_overlap_by_task.csv"

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    instances = _read_csv(instances_path)
    summary = _read_csv(summary_path)
    return SemanticRun(run_dir=run_path, metadata=metadata, instances=instances, summary=summary)


def _overlap_keys(run: SemanticRun) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for row in run.instances:
        if row.get("is_overlap", "").lower() == "true":
            keys.add((row["task"], row["instance_id"]))
    return keys


def _summary_lookup(run: SemanticRun) -> dict[str, dict[str, str]]:
    return {row["task"]: row for row in run.summary}


def build_markdown_report(
    baseline: SemanticRun,
    candidate: SemanticRun,
    *,
    top_n_tasks: int = 10,
) -> str:
    """Build a concise markdown comparison between two semantic runs."""
    baseline_keys = _overlap_keys(baseline)
    candidate_keys = _overlap_keys(candidate)

    common = baseline_keys & candidate_keys
    baseline_only = baseline_keys - candidate_keys
    candidate_only = candidate_keys - baseline_keys

    baseline_summary = _summary_lookup(baseline)
    candidate_summary = _summary_lookup(candidate)
    task_names = sorted(set(baseline_summary) | set(candidate_summary))

    changed_tasks = []
    for task in task_names:
        base_rate = float(baseline_summary.get(task, {}).get("overlap_rate", 0.0))
        cand_rate = float(candidate_summary.get(task, {}).get("overlap_rate", 0.0))
        diff = cand_rate - base_rate
        if diff != 0:
            changed_tasks.append((task, base_rate, cand_rate, diff))

    changed_tasks.sort(key=lambda item: abs(item[3]), reverse=True)

    lines = [
        "# Semantic Run Comparison",
        "",
        "## Overall",
        "",
        "| Run | Model | Threshold | Overlaps | Total | Rate (%) |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
        (
            f"| Baseline | `{baseline.model_name}` | {baseline.threshold:.2f} | "
            f"{baseline.overlapping_instances} | {baseline.total_instances} | {baseline.overlap_rate * 100:.2f} |"
        ),
        (
            f"| Candidate | `{candidate.model_name}` | {candidate.threshold:.2f} | "
            f"{candidate.overlapping_instances} | {candidate.total_instances} | {candidate.overlap_rate * 100:.2f} |"
        ),
        "",
        "## Agreement",
        "",
        f"- Overlaps found by both runs: {len(common)}",
        f"- Overlaps found only by baseline: {len(baseline_only)}",
        f"- Overlaps found only by candidate: {len(candidate_only)}",
        "",
    ]

    if changed_tasks:
        lines.extend([
            "## Tasks With Largest Rate Changes",
            "",
            "| Task | Baseline Rate (%) | Candidate Rate (%) | Delta (pp) |",
            "| --- | ---: | ---: | ---: |",
        ])
        for task, base_rate, cand_rate, diff in changed_tasks[:top_n_tasks]:
            lines.append(
                f"| `{task}` | {base_rate * 100:.2f} | {cand_rate * 100:.2f} | {diff * 100:.2f} |"
            )
        lines.append("")
    else:
        lines.extend([
            "## Tasks With Largest Rate Changes",
            "",
            "No per-task rate changes were detected between the two runs.",
            "",
        ])

    return "\n".join(lines)
