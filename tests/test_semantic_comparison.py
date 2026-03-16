"""Unit tests for semantic run comparison helpers."""

import csv
import json
from pathlib import Path

from contamination.semantic_comparison import build_markdown_report, load_semantic_run


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_load_semantic_run_and_build_report(tmp_path: Path):
    baseline_dir = tmp_path / "baseline" / "tables"
    candidate_dir = tmp_path / "candidate" / "tables"
    baseline_dir.mkdir(parents=True)
    candidate_dir.mkdir(parents=True)

    (baseline_dir / "semantic_overlap_run_metadata.json").write_text(
        json.dumps({
            "model_name": "BAAI/bge-m3",
            "threshold": 0.85,
            "total_instances": 3,
            "overlapping_instances": 1,
        }),
        encoding="utf-8",
    )
    (candidate_dir / "semantic_overlap_run_metadata.json").write_text(
        json.dumps({
            "model_name": "intfloat/multilingual-e5-large-instruct",
            "threshold": 0.85,
            "total_instances": 3,
            "overlapping_instances": 2,
        }),
        encoding="utf-8",
    )

    _write_csv(
        baseline_dir / "semantic_overlap_instances.csv",
        [
            {"task": "task_a", "instance_id": "1", "is_overlap": "True"},
            {"task": "task_a", "instance_id": "2", "is_overlap": "False"},
            {"task": "task_b", "instance_id": "3", "is_overlap": "False"},
        ],
        ["task", "instance_id", "is_overlap"],
    )
    _write_csv(
        candidate_dir / "semantic_overlap_instances.csv",
        [
            {"task": "task_a", "instance_id": "1", "is_overlap": "True"},
            {"task": "task_a", "instance_id": "2", "is_overlap": "True"},
            {"task": "task_b", "instance_id": "3", "is_overlap": "False"},
        ],
        ["task", "instance_id", "is_overlap"],
    )

    _write_csv(
        baseline_dir / "semantic_overlap_by_task.csv",
        [
            {"task": "task_a", "overlap_rate": "0.5"},
            {"task": "task_b", "overlap_rate": "0.0"},
        ],
        ["task", "overlap_rate"],
    )
    _write_csv(
        candidate_dir / "semantic_overlap_by_task.csv",
        [
            {"task": "task_a", "overlap_rate": "1.0"},
            {"task": "task_b", "overlap_rate": "0.0"},
        ],
        ["task", "overlap_rate"],
    )

    baseline = load_semantic_run(tmp_path / "baseline")
    candidate = load_semantic_run(tmp_path / "candidate")
    report = build_markdown_report(baseline, candidate)

    assert baseline.model_name == "BAAI/bge-m3"
    assert candidate.overlapping_instances == 2
    assert "Overlaps found by both runs: 1" in report
    assert "Overlaps found only by candidate: 1" in report
    assert "`task_a`" in report
