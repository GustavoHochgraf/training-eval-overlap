#!/usr/bin/env python3
"""Compare two semantic overlap runs produced by run_semantic_search.py."""

from __future__ import annotations

import argparse
from pathlib import Path

from contamination.semantic_comparison import build_markdown_report, load_semantic_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two semantic overlap runs")
    parser.add_argument("--baseline", required=True, help="Baseline result directory")
    parser.add_argument("--candidate", required=True, help="Candidate result directory")
    parser.add_argument("--output", default=None, help="Optional path to save the markdown comparison")
    args = parser.parse_args()

    baseline = load_semantic_run(args.baseline)
    candidate = load_semantic_run(args.candidate)
    report = build_markdown_report(baseline, candidate)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")

    print(report)


if __name__ == "__main__":
    main()
