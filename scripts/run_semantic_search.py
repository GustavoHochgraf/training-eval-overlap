#!/usr/bin/env python3
"""
Semantic overlap detection: embed PoetaV2 tasks and Carolina corpus,
then find which evaluation instances have near-duplicates in training data.

Usage:
    python scripts/run_semantic_search.py --config configs/local.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml
from rich.console import Console
from rich.table import Table

from contamination.embeddings import EmbeddingIndex
from contamination.extraction import EvalInstance, extract_all_tasks
from contamination.normalization import normalize

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
console = Console()


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def iter_carolina_texts(carolina_dir: Path, max_docs: int | None = None):
    """Yield (doc_id, text) from Carolina corpus files."""
    count = 0
    for jsonl_path in sorted(carolina_dir.rglob("*.jsonl")):
        with open(jsonl_path, encoding="utf-8") as fh:
            for idx, line in enumerate(fh):
                if not line.strip():
                    continue
                row = json.loads(line)
                doc_id = row.get("id", f"{jsonl_path.stem}_{idx}")
                text = row.get("text", row.get("content", ""))
                if text and len(text) >= 50:
                    yield doc_id, text
                    count += 1
                    if max_docs and count >= max_docs:
                        return

    try:
        import pyarrow.parquet as pq

        for pq_path in sorted(carolina_dir.rglob("*.parquet")):
            table = pq.read_table(pq_path)
            text_col = "text" if "text" in table.column_names else "content"
            id_col = "id" if "id" in table.column_names else None

            for i, row in enumerate(table.to_pylist()):
                text = row.get(text_col, "")
                doc_id = str(row[id_col]) if id_col else f"{pq_path.stem}_{i}"
                if text and len(text) >= 50:
                    yield doc_id, text
                    count += 1
                    if max_docs and count >= max_docs:
                        return
    except ImportError:
        logger.warning("pyarrow not installed — skipping Parquet files")


def instance_to_query(inst: EvalInstance) -> str:
    """Combine instance fields into a single query string for embedding."""
    parts = []
    if inst.question:
        parts.append(inst.question)
    if inst.context:
        parts.append(inst.context)
    if inst.answer:
        parts.append(inst.answer)
    return " ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Semantic overlap: Carolina × PoetaV2")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--top-k", type=int, default=5, help="Number of nearest neighbors per query")
    parser.add_argument("--threshold", type=float, default=0.85, help="Cosine similarity threshold for 'overlap'")
    parser.add_argument("--max-carolina-docs", type=int, default=None, help="Limit Carolina docs (for testing)")
    parser.add_argument("--model", type=str, default=None, help="Override embedding model name")
    parser.add_argument("--index-cache", type=str, default=None, help="Path to cache/load FAISS index")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    carolina_dir = Path(paths["carolina_dir"])
    poetav2_dir = Path(paths["poetav2_dir"])
    output_dir = Path(paths.get("output_dir", "results"))

    embed_cfg = cfg.get("embeddings", {})
    model_name = args.model or embed_cfg.get("model", "BAAI/bge-m3")
    top_k = args.top_k or embed_cfg.get("top_k", 5)
    threshold = args.threshold or embed_cfg.get("threshold", 0.85)

    # --- Step 1: Extract PoetaV2 instances ---
    logger.info("=== Step 1: Extracting PoetaV2 evaluation instances ===")
    ext_cfg = cfg.get("extraction", {})
    instances = extract_all_tasks(
        poetav2_dir,
        tasks=ext_cfg.get("tasks"),
        min_length=ext_cfg.get("min_length", 20),
    )
    logger.info("Extracted %d instances from %d tasks", len(instances), len({i.task_name for i in instances}))

    # --- Step 2: Build or load embedding index over Carolina ---
    logger.info("=== Step 2: Building embedding index over Carolina corpus ===")
    emb_index = EmbeddingIndex(model_name=model_name, batch_size=embed_cfg.get("batch_size", 64))

    index_cache = Path(args.index_cache) if args.index_cache else None
    if index_cache and (index_cache / "index.faiss").exists():
        logger.info("Loading cached index from %s", index_cache)
        emb_index.load(index_cache)
    else:
        carolina_docs = list(iter_carolina_texts(carolina_dir, max_docs=args.max_carolina_docs))
        logger.info("Loaded %d Carolina documents", len(carolina_docs))
        emb_index.build_index(carolina_docs)
        if index_cache:
            emb_index.save(index_cache)

    # --- Step 3: Search for each evaluation instance ---
    logger.info("=== Step 3: Searching for semantic overlap ===")
    queries = [instance_to_query(inst) for inst in instances]
    all_results = emb_index.search(queries, top_k=top_k)

    # --- Step 4: Classify and aggregate ---
    logger.info("=== Step 4: Classifying overlap ===")
    rows = []
    task_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "overlapping": 0})

    for inst, results in zip(instances, all_results):
        best_sim = results[0].similarity if results else 0.0
        is_overlap = best_sim >= threshold

        task_stats[inst.task_name]["total"] += 1
        if is_overlap:
            task_stats[inst.task_name]["overlapping"] += 1

        rows.append({
            "task": inst.task_name,
            "instance_id": inst.instance_id,
            "query_preview": instance_to_query(inst)[:200],
            "top1_similarity": best_sim,
            "top1_doc_id": results[0].doc_id if results else "",
            "top1_doc_preview": results[0].doc_text[:200] if results else "",
            "is_overlap": is_overlap,
            "top_k_similarities": [r.similarity for r in results],
        })

    # --- Step 5: Save results ---
    logger.info("=== Step 5: Saving results ===")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "tables" / "semantic_overlap_instances.csv", index=False)
    df.to_parquet(output_dir / "tables" / "semantic_overlap_instances.parquet", index=False)

    # Per-task summary
    summary_rows = []
    for task, stats in sorted(task_stats.items()):
        total = stats["total"]
        overlapping = stats["overlapping"]
        rate = overlapping / total if total else 0.0
        summary_rows.append({
            "task": task,
            "total_instances": total,
            "overlapping_instances": overlapping,
            "overlap_rate": rate,
        })

    df_summary = pd.DataFrame(summary_rows).sort_values("overlap_rate", ascending=False)
    df_summary.to_csv(output_dir / "tables" / "semantic_overlap_by_task.csv", index=False)

    # --- Display results ---
    console.print()
    table = Table(title=f"Semantic Overlap Summary (threshold={threshold})")
    table.add_column("Task", style="cyan")
    table.add_column("Total", justify="right")
    table.add_column("Overlapping", justify="right", style="red")
    table.add_column("Rate (%)", justify="right", style="bold")

    for _, row in df_summary.iterrows():
        table.add_row(
            row["task"],
            str(row["total_instances"]),
            str(row["overlapping_instances"]),
            f"{row['overlap_rate'] * 100:.1f}",
        )

    # Overall
    total_all = sum(s["total"] for s in task_stats.values())
    overlap_all = sum(s["overlapping"] for s in task_stats.values())
    table.add_section()
    table.add_row(
        "OVERALL",
        str(total_all),
        str(overlap_all),
        f"{overlap_all / total_all * 100:.1f}" if total_all else "0.0",
        style="bold",
    )

    console.print(table)
    logger.info("Results saved to %s", output_dir / "tables")


if __name__ == "__main__":
    main()
