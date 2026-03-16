from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

THRESHOLDS = [0.80, 0.82, 0.85, 0.88, 0.90]


def _load_run(slug: str) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    run_dir = RUNS_DIR / slug / "tables"
    metadata = pd.read_json(run_dir / "semantic_overlap_run_metadata.json", typ="series")
    summary = pd.read_csv(run_dir / "semantic_overlap_by_task.csv")
    instances = pd.read_csv(run_dir / "semantic_overlap_instances.csv")
    return metadata, summary, instances


def _overlap_mask(frame: pd.DataFrame) -> pd.Series:
    return frame["is_overlap"].astype(str).str.lower().isin({"true", "1"})


def build_run_summary() -> pd.DataFrame:
    snapshot_summary = pd.read_csv(TABLES_DIR / "overlap_summary_snapshot.csv")
    snapshot_tasks = snapshot_summary[snapshot_summary["task"] != "OVERALL"].copy()
    bge_meta, bge_summary, bge_instances = _load_run("bge_m3")
    e5_meta, e5_summary, e5_instances = _load_run("multilingual_e5_large_instruct")

    rows = [
        {
            "run_slug": "snapshot_bge_small_en",
            "run_label": "Snapshot exploratorio",
            "model_name": "BAAI/bge-small-en-v1.5",
            "motivation": "Baseline rapido inicial para verificar viabilidade do pipeline.",
            "carolina_documents": 993,
            "tasks_loaded": int(len(snapshot_tasks)),
            "total_instances": 11409,
            "threshold": 0.85,
            "caps_truncation": "MAX_CAROLINA_DOCS=1000; MAX_INSTANCES_PER_TASK=500; sem truncation explicita",
            "overlaps": 15,
            "overlap_rate_pct": 15 / 11409 * 100,
            "tasks_nonzero": int((snapshot_tasks["rate_pct"] > 0).sum()),
            "mean_top1_similarity": 0.763,
            "max_similarity": 0.8696,
        },
        {
            "run_slug": str(bge_meta["model_slug"]),
            "run_label": str(bge_meta["model_label"]),
            "model_name": str(bge_meta["model_name"]),
            "motivation": "Rerun multilingue principal com encoder retrieval-oriented.",
            "carolina_documents": int(bge_meta["carolina_documents"]),
            "tasks_loaded": int(len(bge_summary)),
            "total_instances": int(bge_meta["total_instances"]),
            "threshold": float(bge_meta["threshold"]),
            "caps_truncation": (
                f"MAX_CAROLINA_DOCS={int(bge_meta['max_carolina_docs'])}; "
                f"MAX_INSTANCES_PER_TASK={int(bge_meta['max_instances_per_task'])}; "
                f"max_seq_length={int(bge_meta['model_max_seq_length'])}; "
                f"max_document_chars={int(bge_meta['max_document_chars'])}"
            ),
            "overlaps": int(bge_meta["overlapping_instances"]),
            "overlap_rate_pct": int(bge_meta["overlapping_instances"]) / int(bge_meta["total_instances"]) * 100,
            "tasks_nonzero": int((bge_summary["overlap_rate_pct"] > 0).sum()),
            "mean_top1_similarity": float(bge_instances["top1_similarity"].mean()),
            "max_similarity": float(bge_instances["top1_similarity"].max()),
        },
        {
            "run_slug": str(e5_meta["model_slug"]),
            "run_label": str(e5_meta["model_label"]),
            "model_name": str(e5_meta["model_name"]),
            "motivation": "Rerun multilingue alternativo com query instruction explicita.",
            "carolina_documents": int(e5_meta["carolina_documents"]),
            "tasks_loaded": int(len(e5_summary)),
            "total_instances": int(e5_meta["total_instances"]),
            "threshold": float(e5_meta["threshold"]),
            "caps_truncation": (
                f"MAX_CAROLINA_DOCS={int(e5_meta['max_carolina_docs'])}; "
                f"MAX_INSTANCES_PER_TASK={int(e5_meta['max_instances_per_task'])}; "
                f"max_seq_length={int(e5_meta['model_max_seq_length'])}; "
                f"max_document_chars={int(e5_meta['max_document_chars'])}"
            ),
            "overlaps": int(e5_meta["overlapping_instances"]),
            "overlap_rate_pct": int(e5_meta["overlapping_instances"]) / int(e5_meta["total_instances"]) * 100,
            "tasks_nonzero": int((e5_summary["overlap_rate_pct"] > 0).sum()),
            "mean_top1_similarity": float(e5_instances["top1_similarity"].mean()),
            "max_similarity": float(e5_instances["top1_similarity"].max()),
        },
    ]

    frame = pd.DataFrame(rows)
    frame.to_csv(TABLES_DIR / "study_run_summary.csv", index=False)
    return frame


def build_threshold_sensitivity() -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []

    for slug in ["bge_m3", "multilingual_e5_large_instruct"]:
        metadata, _, instances = _load_run(slug)
        similarities = instances["top1_similarity"]
        total = len(similarities)

        for threshold in THRESHOLDS:
            overlaps = int((similarities >= threshold).sum())
            rows.append(
                {
                    "run_slug": slug,
                    "model_name": str(metadata["model_name"]),
                    "threshold": threshold,
                    "overlaps": overlaps,
                    "total_instances": total,
                    "overlap_rate_pct": overlaps / total * 100,
                }
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(TABLES_DIR / "threshold_sensitivity_summary.csv", index=False)
    return frame


def build_hub_documents() -> pd.DataFrame:
    _, _, instances = _load_run("multilingual_e5_large_instruct")
    overlaps = instances[_overlap_mask(instances)].copy()

    grouped = (
        overlaps.groupby("top1_doc_id")
        .agg(
            overlap_hits=("top1_doc_id", "size"),
            distinct_tasks=("task", "nunique"),
            mean_similarity=("top1_similarity", "mean"),
            example_preview=("top1_doc_text", "first"),
        )
        .reset_index()
        .sort_values(["overlap_hits", "mean_similarity"], ascending=[False, False])
        .head(15)
    )
    grouped["example_preview"] = grouped["example_preview"].str.replace(r"\s+", " ", regex=True).str.slice(0, 220)
    grouped.to_csv(TABLES_DIR / "e5_hub_documents.csv", index=False)
    return grouped


def build_task_delta_summary() -> pd.DataFrame:
    source = pd.read_csv(RUNS_DIR / "comparison" / "model_comparison_task_deltas.csv")
    summary = source.head(15).copy()
    summary.to_csv(TABLES_DIR / "top_task_deltas_summary.csv", index=False)
    return summary


def build_overall_plot(summary: pd.DataFrame) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    labels = ["Snapshot", "BGE-M3", "multilingual-e5"]
    values = summary["overlap_rate_pct"].tolist()
    colors = ["#5B8FF9", "#61DDAA", "#F08C6C"]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=1.0)
    ax.set_ylabel("Overlap rate (%)")
    ax.set_title("Taxa de overlap por rodada")
    ax.set_ylim(0, max(values) * 1.15 + 0.5)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4)

    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2, f"{value:.2f}%", ha="center", va="bottom")

    fig.tight_layout()
    output_path = FIGURES_DIR / "overall_overlap_rate_comparison.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_study_metadata(summary: pd.DataFrame) -> None:
    payload = {
        "runs": json.loads(summary.to_json(orient="records")),
        "thresholds_compared": THRESHOLDS,
        "notes": [
            "Snapshot inicial usa metadados consolidados do notebook exploratorio previamente versionado.",
            "Rodadas locais multilingues usam truncation explicita e max_seq_length reduzido para viabilizar execucao reproduzivel em GPU local.",
        ],
    }
    (TABLES_DIR / "study_report_metadata.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    summary = build_run_summary()
    build_threshold_sensitivity()
    build_hub_documents()
    build_task_delta_summary()
    build_overall_plot(summary)
    build_study_metadata(summary)


if __name__ == "__main__":
    main()
