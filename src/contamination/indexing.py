"""
N-gram inverted index and MinHash LSH index over the Carolina corpus.

Provides efficient lookup structures for both exact substring matching
and approximate near-duplicate detection.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .normalization import ngrams, normalize

logger = logging.getLogger(__name__)


class NgramIndex:
    """Inverted index mapping n-grams to (document_id, position) pairs.

    Parameters
    ----------
    n : int
        N-gram size (in tokens).
    """

    def __init__(self, n: int = 8) -> None:
        self.n = n
        self.index: dict[str, list[tuple[str, int]]] = defaultdict(list)
        self._doc_count = 0

    def add_document(self, doc_id: str, text: str) -> None:
        """Index a single document.

        Parameters
        ----------
        doc_id : str
            Unique document identifier.
        text : str
            Already-normalized text content.
        """
        grams = ngrams(text, self.n)
        for pos, gram in enumerate(grams):
            self.index[gram].append((doc_id, pos))
        self._doc_count += 1

    def build_from_texts(self, texts: Iterable[tuple[str, str]]) -> None:
        """Bulk-index an iterable of (doc_id, normalized_text) pairs."""
        for doc_id, text in texts:
            self.add_document(doc_id, text)
            if self._doc_count % 50_000 == 0:
                logger.info("Indexed %d documents (%d unique n-grams)", self._doc_count, len(self.index))

        logger.info(
            "Index complete: %d documents, %d unique %d-grams",
            self._doc_count,
            len(self.index),
            self.n,
        )

    def query_exact(self, text: str) -> set[str]:
        """Return document IDs containing *all* n-grams of the query text (exact match candidate)."""
        grams = ngrams(text, self.n)
        if not grams:
            return set()

        # Start with docs matching the first n-gram, then intersect
        candidates = {doc_id for doc_id, _ in self.index.get(grams[0], [])}
        for gram in grams[1:]:
            candidates &= {doc_id for doc_id, _ in self.index.get(gram, [])}
            if not candidates:
                return set()

        return candidates

    def query_overlap(self, text: str) -> dict[str, float]:
        """Return document IDs with their n-gram Jaccard overlap ratio.

        Returns
        -------
        dict[str, float]
            Mapping from doc_id to overlap ratio (|intersection| / |query_ngrams|).
        """
        grams = ngrams(text, self.n)
        if not grams:
            return {}

        query_set = set(grams)
        hit_counts: dict[str, int] = defaultdict(int)

        for gram in query_set:
            for doc_id, _ in self.index.get(gram, []):
                hit_counts[doc_id] += 1

        total = len(query_set)
        return {doc_id: count / total for doc_id, count in hit_counts.items()}

    @property
    def num_documents(self) -> int:
        return self._doc_count

    @property
    def num_ngrams(self) -> int:
        return len(self.index)
