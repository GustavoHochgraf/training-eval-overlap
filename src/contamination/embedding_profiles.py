"""Helpers for model-specific embedding text formatting."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_E5_QUERY_INSTRUCTION = (
    "Given a search query, retrieve relevant passages from the training corpus "
    "that may overlap with the query"
)


@dataclass(frozen=True)
class EmbeddingTextConfig:
    """Resolved text-formatting settings for an embedding model."""

    query_instruction: str | None = None
    query_prefix: str | None = None
    document_prefix: str | None = None


def default_query_instruction_for_model(model_name: str) -> str | None:
    """Return a sensible default query instruction for known model families."""
    normalized = model_name.lower()
    if "e5" in normalized and "instruct" in normalized:
        return DEFAULT_E5_QUERY_INSTRUCTION
    return None


def resolve_text_config(
    model_name: str,
    *,
    query_instruction: str | None = None,
    query_prefix: str | None = None,
    document_prefix: str | None = None,
) -> EmbeddingTextConfig:
    """Resolve final text-formatting settings for a model.

    If a query instruction is set explicitly or implied by the model family,
    it takes precedence over a raw query prefix.
    """
    resolved_instruction = query_instruction or default_query_instruction_for_model(model_name)
    resolved_query_prefix = None if resolved_instruction else query_prefix

    return EmbeddingTextConfig(
        query_instruction=resolved_instruction,
        query_prefix=resolved_query_prefix,
        document_prefix=document_prefix,
    )


def format_query_text(
    text: str,
    *,
    query_instruction: str | None = None,
    query_prefix: str | None = None,
) -> str:
    """Format a query string for embedding."""
    if query_instruction:
        return f"Instruct: {query_instruction}\nQuery: {text}"
    if query_prefix:
        return f"{query_prefix}{text}"
    return text


def format_document_text(text: str, *, document_prefix: str | None = None) -> str:
    """Format a document/passsage string for embedding."""
    if document_prefix:
        return f"{document_prefix}{text}"
    return text
