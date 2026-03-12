"""
Embedding-based semantic overlap detection.

Uses a multilingual embedding model (default: BAAI/bge-m3) to encode
both evaluation instances and training corpus passages, then performs
nearest-neighbor retrieval via FAISS to identify semantic overlap.
"""

from __future__ import annotations

import logging
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-m3"


class EmbeddingIndex:
    """FAISS-backed embedding index for semantic similarity search.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID for the sentence-transformer.
    batch_size : int
        Encoding batch size.
    device : str or None
        Torch device ("cuda", "cpu", or None for auto-detect).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 64,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size

        logger.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = self.model.get_sentence_embedding_dimension()

        self.index: faiss.IndexFlatIP | None = None
        self.doc_ids: list[str] = []
        self.doc_texts: list[str] = []

    def build_index(self, documents: list[tuple[str, str]]) -> None:
        """Build a FAISS index from a list of (doc_id, text) pairs.

        Parameters
        ----------
        documents : list[tuple[str, str]]
            Each element is (doc_id, text_content).
        """
        self.doc_ids = [doc_id for doc_id, _ in documents]
        self.doc_texts = [text for _, text in documents]

        logger.info("Encoding %d documents with %s ...", len(self.doc_texts), self.model_name)
        embeddings = self.model.encode(
            self.doc_texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,  # cosine similarity via inner product
        )
        embeddings = np.array(embeddings, dtype=np.float32)

        # Inner-product index (equivalent to cosine similarity with normalized vectors)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)

        logger.info("FAISS index built: %d vectors, dim=%d", self.index.ntotal, self.dimension)

    def search(
        self,
        queries: list[str],
        top_k: int = 5,
    ) -> list[list[SearchResult]]:
        """Search the index for nearest neighbors of each query.

        Parameters
        ----------
        queries : list[str]
            Query texts to search for.
        top_k : int
            Number of nearest neighbors to return per query.

        Returns
        -------
        list[list[SearchResult]]
            For each query, a list of top-k results sorted by similarity.
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        logger.info("Encoding %d queries ...", len(queries))
        query_embeddings = self.model.encode(
            queries,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        query_embeddings = np.array(query_embeddings, dtype=np.float32)

        scores, indices = self.index.search(query_embeddings, top_k)

        results = []
        for q_idx in range(len(queries)):
            q_results = []
            for rank in range(top_k):
                doc_idx = indices[q_idx][rank]
                if doc_idx == -1:
                    continue
                q_results.append(
                    SearchResult(
                        doc_id=self.doc_ids[doc_idx],
                        doc_text=self.doc_texts[doc_idx],
                        similarity=float(scores[q_idx][rank]),
                        rank=rank + 1,
                    )
                )
            results.append(q_results)

        return results

    def save(self, path: Path) -> None:
        """Persist FAISS index and metadata to disk."""
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        np.save(path / "doc_ids.npy", np.array(self.doc_ids, dtype=object))
        np.save(path / "doc_texts.npy", np.array(self.doc_texts, dtype=object))
        logger.info("Index saved to %s", path)

    def load(self, path: Path) -> None:
        """Load a previously saved FAISS index."""
        self.index = faiss.read_index(str(path / "index.faiss"))
        self.doc_ids = list(np.load(path / "doc_ids.npy", allow_pickle=True))
        self.doc_texts = list(np.load(path / "doc_texts.npy", allow_pickle=True))
        logger.info("Index loaded from %s: %d vectors", path, self.index.ntotal)


class SearchResult:
    """A single search result from the embedding index."""

    __slots__ = ("doc_id", "doc_text", "similarity", "rank")

    def __init__(self, doc_id: str, doc_text: str, similarity: float, rank: int) -> None:
        self.doc_id = doc_id
        self.doc_text = doc_text
        self.similarity = similarity
        self.rank = rank

    def __repr__(self) -> str:
        return f"SearchResult(doc_id={self.doc_id!r}, sim={self.similarity:.4f}, rank={self.rank})"
