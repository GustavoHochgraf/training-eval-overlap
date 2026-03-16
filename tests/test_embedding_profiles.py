"""Unit tests for model-specific embedding text formatting."""

from contamination.embedding_profiles import (
    DEFAULT_E5_QUERY_INSTRUCTION,
    default_query_instruction_for_model,
    format_document_text,
    format_query_text,
    resolve_text_config,
)


class TestEmbeddingProfiles:
    def test_bge_m3_has_no_default_instruction(self):
        assert default_query_instruction_for_model("BAAI/bge-m3") is None

    def test_e5_instruct_gets_default_instruction(self):
        assert (
            default_query_instruction_for_model("intfloat/multilingual-e5-large-instruct")
            == DEFAULT_E5_QUERY_INSTRUCTION
        )

    def test_query_instruction_takes_precedence_over_prefix(self):
        config = resolve_text_config(
            "intfloat/multilingual-e5-large-instruct",
            query_prefix="query: ",
        )
        assert config.query_instruction == DEFAULT_E5_QUERY_INSTRUCTION
        assert config.query_prefix is None

    def test_format_query_with_instruction(self):
        formatted = format_query_text("sobreposicao treino avaliacao", query_instruction="Retrieve overlaps")
        assert formatted == "Instruct: Retrieve overlaps\nQuery: sobreposicao treino avaliacao"

    def test_format_query_with_prefix(self):
        formatted = format_query_text("texto", query_prefix="query: ")
        assert formatted == "query: texto"

    def test_format_document_with_prefix(self):
        formatted = format_document_text("documento", document_prefix="passage: ")
        assert formatted == "passage: documento"
