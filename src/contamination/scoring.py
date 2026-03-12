"""
Aggregate contamination scoring and statistical analysis.

Computes per-task and overall contamination rates with bootstrap
confidence intervals.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .matching import ContaminationLevel, MatchResult

logger = logging.getLogger(__name__)


@dataclass
class TaskContaminationReport:
    """Contamination summary for a single PoetaV2 task."""

    task_name: str
    total_instances: int
    exact_count: int
    near_count: int
    partial_count: int
    clean_count: int

    @property
    def exact_rate(self) -> float:
        return self.exact_count / self.total_instances if self.total_instances else 0.0

    @property
    def near_rate(self) -> float:
        return self.near_count / self.total_instances if self.total_instances else 0.0

    @property
    def any_contamination_rate(self) -> float:
        return (self.exact_count + self.near_count + self.partial_count) / self.total_instances if self.total_instances else 0.0


def aggregate_results(
    results: list[MatchResult],
    total_per_task: dict[str, int],
) -> list[TaskContaminationReport]:
    """Aggregate per-instance match results into per-task reports.

    Parameters
    ----------
    results : list[MatchResult]
        All match results across instances and tasks.
    total_per_task : dict[str, int]
        Total number of instances per task (including clean ones).

    Returns
    -------
    list[TaskContaminationReport]
        One report per task, sorted by contamination rate (descending).
    """
    # Deduplicate: keep the highest contamination level per instance
    best_per_instance: dict[tuple[str, str], MatchResult] = {}
    level_rank = {
        ContaminationLevel.EXACT: 3,
        ContaminationLevel.NEAR: 2,
        ContaminationLevel.PARTIAL: 1,
        ContaminationLevel.NONE: 0,
    }

    for r in results:
        key = (r.task_name, r.instance_id)
        if key not in best_per_instance or level_rank[r.level] > level_rank[best_per_instance[key].level]:
            best_per_instance[key] = r

    # Count per task
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for (task, _), r in best_per_instance.items():
        counts[task][r.level.value] += 1

    reports = []
    for task, total in sorted(total_per_task.items()):
        c = counts.get(task, {})
        exact = c.get("exact", 0)
        near = c.get("near", 0)
        partial = c.get("partial", 0)
        clean = total - exact - near - partial

        reports.append(
            TaskContaminationReport(
                task_name=task,
                total_instances=total,
                exact_count=exact,
                near_count=near,
                partial_count=partial,
                clean_count=clean,
            )
        )

    reports.sort(key=lambda r: r.any_contamination_rate, reverse=True)
    return reports


def bootstrap_ci(
    contaminated: int,
    total: int,
    n_bootstrap: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for a contamination rate.

    Parameters
    ----------
    contaminated : int
        Number of contaminated instances.
    total : int
        Total number of instances.
    n_bootstrap : int
        Number of bootstrap iterations.
    ci : float
        Confidence level (e.g. 0.95 for 95% CI).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[float, float, float]
        (point_estimate, ci_lower, ci_upper)
    """
    if total == 0:
        return 0.0, 0.0, 0.0

    rng = np.random.default_rng(seed)
    labels = np.array([1] * contaminated + [0] * (total - contaminated))
    boot_rates = np.array([rng.choice(labels, size=total, replace=True).mean() for _ in range(n_bootstrap)])

    alpha = (1 - ci) / 2
    return (
        contaminated / total,
        float(np.percentile(boot_rates, 100 * alpha)),
        float(np.percentile(boot_rates, 100 * (1 - alpha))),
    )
