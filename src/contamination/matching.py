"""
Contamination matching strategies.

Implements three complementary detection methods:
1. Exact substring match (after normalization)
2. N-gram overlap (Jaccard-based)
3. BM25 retrieval scoring
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

from .extraction import EvalInstance
from .indexing import NgramIndex
from .normalization import normalize

logger = logging.getLogger(__name__)


class ContaminationLevel(str, Enum):
    """Contamination severity levels following Oren et al. (2024)."""

    NONE = "none"
    PARTIAL = "partial"
    NEAR = "near"
    EXACT = "exact"


@dataclass
class MatchResult:
    """Result of contamination detection for a single evaluation instance."""

    instance_id: str
    task_name: str
    level: ContaminationLevel
    confidence: float  # 0.0 – 1.0
    matched_doc_ids: list[str]
    overlap_score: float  # n-gram Jaccard or BM25 score
    matched_field: str  # which field triggered the match (question, context, etc.)


def check_exact(
    instance: EvalInstance,
    index: NgramIndex,
    *,
    normalize_kwargs: dict | None = None,
) -> list[MatchResult]:
    """Check for exact contamination via n-gram index.

    An instance is *exactly contaminated* if all its n-grams appear
    in at least one training document.
    """
    norm_kw = normalize_kwargs or {}
    results: list[MatchResult] = []

    for field_name, text in [
        ("question", instance.question),
        ("context", instance.context),
        ("answer", instance.answer),
    ]:
        if not text.strip():
            continue

        normed = normalize(text, **norm_kw)
        hits = index.query_exact(normed)

        if hits:
            results.append(
                MatchResult(
                    instance_id=instance.instance_id,
                    task_name=instance.task_name,
                    level=ContaminationLevel.EXACT,
                    confidence=1.0,
                    matched_doc_ids=list(hits)[:10],
                    overlap_score=1.0,
                    matched_field=field_name,
                )
            )

    return results


def check_near(
    instance: EvalInstance,
    index: NgramIndex,
    *,
    threshold: float = 0.70,
    normalize_kwargs: dict | None = None,
) -> list[MatchResult]:
    """Check for near contamination via n-gram overlap.

    An instance is *near contaminated* if its n-gram Jaccard overlap
    with any training document exceeds *threshold*.
    """
    norm_kw = normalize_kwargs or {}
    results: list[MatchResult] = []

    for field_name, text in [
        ("question", instance.question),
        ("context", instance.context),
        ("answer", instance.answer),
    ]:
        if not text.strip():
            continue

        normed = normalize(text, **norm_kw)
        overlaps = index.query_overlap(normed)

        for doc_id, score in overlaps.items():
            if score >= threshold:
                results.append(
                    MatchResult(
                        instance_id=instance.instance_id,
                        task_name=instance.task_name,
                        level=ContaminationLevel.NEAR,
                        confidence=score,
                        matched_doc_ids=[doc_id],
                        overlap_score=score,
                        matched_field=field_name,
                    )
                )

    return results
