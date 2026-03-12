"""
Generate publication-ready tables and figures from contamination results.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from .scoring import TaskContaminationReport, bootstrap_ci

logger = logging.getLogger(__name__)


def to_csv(reports: list[TaskContaminationReport], output_path: Path) -> None:
    """Write contamination results to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "task",
            "total",
            "exact",
            "near",
            "partial",
            "clean",
            "exact_rate",
            "near_rate",
            "any_rate",
        ])
        for r in reports:
            writer.writerow([
                r.task_name,
                r.total_instances,
                r.exact_count,
                r.near_count,
                r.partial_count,
                r.clean_count,
                f"{r.exact_rate:.4f}",
                f"{r.near_rate:.4f}",
                f"{r.any_contamination_rate:.4f}",
            ])

    logger.info("CSV written to %s", output_path)


def to_latex(
    reports: list[TaskContaminationReport],
    output_path: Path,
    *,
    n_bootstrap: int = 10_000,
    ci: float = 0.95,
) -> None:
    """Write contamination results as a LaTeX table with confidence intervals."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Data contamination rates: Carolina corpus $\times$ PoetaV2 benchmarks.}",
        r"\label{tab:contamination}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Task & $N$ & Exact (\%) & Near (\%) & Any (\%) & 95\% CI \\",
        r"\midrule",
    ]

    for r in reports:
        _, lo, hi = bootstrap_ci(
            r.exact_count + r.near_count + r.partial_count,
            r.total_instances,
            n_bootstrap=n_bootstrap,
            ci=ci,
        )
        lines.append(
            f"{r.task_name} & {r.total_instances} & "
            f"{r.exact_rate * 100:.1f} & {r.near_rate * 100:.1f} & "
            f"{r.any_contamination_rate * 100:.1f} & "
            f"[{lo * 100:.1f}, {hi * 100:.1f}] \\\\"
        )

    # Overall row
    total_n = sum(r.total_instances for r in reports)
    total_exact = sum(r.exact_count for r in reports)
    total_near = sum(r.near_count for r in reports)
    total_partial = sum(r.partial_count for r in reports)
    total_any = total_exact + total_near + total_partial

    if total_n:
        _, lo, hi = bootstrap_ci(total_any, total_n, n_bootstrap=n_bootstrap, ci=ci)
        lines.append(r"\midrule")
        lines.append(
            f"\\textbf{{Overall}} & {total_n} & "
            f"{total_exact / total_n * 100:.1f} & {total_near / total_n * 100:.1f} & "
            f"{total_any / total_n * 100:.1f} & "
            f"[{lo * 100:.1f}, {hi * 100:.1f}] \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("LaTeX table written to %s", output_path)
