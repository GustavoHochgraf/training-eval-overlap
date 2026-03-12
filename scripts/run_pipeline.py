#!/usr/bin/env python3
"""
Main entry point: run the full contamination detection pipeline.

Usage:
    python scripts/run_pipeline.py --config configs/local.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

from contamination.extraction import extract_all_tasks
from contamination.indexing import NgramIndex
from contamination.matching import check_exact, check_near, MatchResult
from contamination.normalization import normalize
from contamination.reporting import to_csv, to_latex
from contamination.scoring import aggregate_results

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def iter_carolina_documents(carolina_dir: Path):
    """Yield (doc_id, raw_text) pairs from the Carolina corpus.

    Supports JSONL and Parquet formats. Adjust field names as needed
    for the actual Carolina data layout.
    """
    for jsonl_path in sorted(carolina_dir.rglob("*.jsonl")):
        with open(jsonl_path, encoding="utf-8") as fh:
            for idx, line in enumerate(fh):
                row = json.loads(line)
                doc_id = row.get("id", f"{jsonl_path.stem}_{idx}")
                text = row.get("text", row.get("content", ""))
                if text:
                    yield doc_id, text

    # Parquet support
    try:
        import pyarrow.parquet as pq

        for pq_path in sorted(carolina_dir.rglob("*.parquet")):
            table = pq.read_table(pq_path, columns=["id", "text"])
            for row in table.to_pylist():
                yield str(row.get("id", "")), row.get("text", "")
    except ImportError:
        logger.warning("pyarrow not installed — skipping Parquet files")


def main():
    parser = argparse.ArgumentParser(description="Carolina × PoetaV2 contamination analysis")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    norm_cfg = cfg.get("normalization", {})
    idx_cfg = cfg.get("indexing", {})
    match_cfg = cfg.get("matching", {})
    report_cfg = cfg.get("reporting", {})

    carolina_dir = Path(paths["carolina_dir"])
    poetav2_dir = Path(paths["poetav2_dir"])
    output_dir = Path(paths.get("output_dir", "results"))

    # --- Step 1: Extract evaluation instances ---
    logger.info("=== Step 1: Extracting PoetaV2 evaluation instances ===")
    ext_cfg = cfg.get("extraction", {})
    instances = extract_all_tasks(
        poetav2_dir,
        tasks=ext_cfg.get("tasks"),
        min_length=ext_cfg.get("min_length", 20),
    )
    logger.info("Extracted %d instances across %d tasks", len(instances), len({i.task_name for i in instances}))

    # --- Step 2: Build n-gram index over Carolina ---
    logger.info("=== Step 2: Building n-gram index over Carolina corpus ===")
    ngram_size = idx_cfg.get("ngram_sizes", [8])[0]
    index = NgramIndex(n=ngram_size)

    norm_kwargs = {
        "unicode_form": norm_cfg.get("unicode", "NFKC"),
        "lowercase": norm_cfg.get("lowercase", True),
        "strip_whitespace": norm_cfg.get("strip_whitespace", True),
        "strip_punctuation": norm_cfg.get("strip_punctuation", False),
        "strip_accents": norm_cfg.get("strip_accents", False),
    }

    def normalized_docs():
        for doc_id, text in iter_carolina_documents(carolina_dir):
            yield doc_id, normalize(text, **norm_kwargs)

    index.build_from_texts(normalized_docs())

    # --- Step 3: Match evaluation instances against index ---
    logger.info("=== Step 3: Running contamination matching ===")
    all_results: list[MatchResult] = []

    for inst in instances:
        if match_cfg.get("exact", True):
            all_results.extend(check_exact(inst, index, normalize_kwargs=norm_kwargs))

        near_cfg = match_cfg.get("near", {})
        if near_cfg.get("enabled", True):
            all_results.extend(
                check_near(inst, index, threshold=near_cfg.get("threshold", 0.70), normalize_kwargs=norm_kwargs)
            )

    logger.info("Found %d total match results", len(all_results))

    # --- Step 4: Aggregate & score ---
    logger.info("=== Step 4: Aggregating results ===")
    total_per_task = {}
    for inst in instances:
        total_per_task[inst.task_name] = total_per_task.get(inst.task_name, 0) + 1

    reports = aggregate_results(all_results, total_per_task)

    # --- Step 5: Generate reports ---
    logger.info("=== Step 5: Generating reports ===")
    to_csv(reports, output_dir / "tables" / "contamination_results.csv")
    to_latex(reports, output_dir / "tables" / "contamination_results.tex")

    # Summary
    total_n = sum(r.total_instances for r in reports)
    total_contaminated = sum(r.exact_count + r.near_count + r.partial_count for r in reports)
    logger.info(
        "=== DONE === Overall: %d / %d instances contaminated (%.1f%%)",
        total_contaminated,
        total_n,
        100 * total_contaminated / total_n if total_n else 0,
    )


if __name__ == "__main__":
    main()
