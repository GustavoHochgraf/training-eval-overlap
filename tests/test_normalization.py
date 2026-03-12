"""Unit tests for text normalization."""

from contamination.normalization import ngrams, normalize


class TestNormalize:
    def test_basic_lowercase(self):
        assert normalize("Hello World") == "hello world"

    def test_unicode_nfkc(self):
        # NFKC normalizes ﬁ ligature to "fi"
        assert normalize("ﬁnance") == "finance"

    def test_whitespace_collapse(self):
        assert normalize("  hello   world  ") == "hello world"

    def test_strip_punctuation(self):
        result = normalize("Olá, mundo!", strip_punctuation=True)
        assert result == "olá mundo"

    def test_preserve_accents_by_default(self):
        result = normalize("É possível não?")
        assert "é" in result
        assert "ã" in result

    def test_strip_accents(self):
        result = normalize("São Paulo", strip_accents=True)
        assert result == "sao paulo"

    def test_empty_string(self):
        assert normalize("") == ""


class TestNgrams:
    def test_basic(self):
        grams = ngrams("a b c d e", 3)
        assert grams == ["a b c", "b c d", "c d e"]

    def test_short_text(self):
        assert ngrams("a b", 3) == []

    def test_exact_length(self):
        assert ngrams("a b c", 3) == ["a b c"]
