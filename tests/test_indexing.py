"""Unit tests for n-gram indexing."""

from contamination.indexing import NgramIndex


class TestNgramIndex:
    def test_exact_match(self):
        idx = NgramIndex(n=3)
        idx.add_document("doc1", "the quick brown fox jumps over the lazy dog")
        hits = idx.query_exact("quick brown fox jumps over")
        assert "doc1" in hits

    def test_no_match(self):
        idx = NgramIndex(n=3)
        idx.add_document("doc1", "the quick brown fox")
        hits = idx.query_exact("completely different text here now")
        assert len(hits) == 0

    def test_overlap_score(self):
        idx = NgramIndex(n=2)
        idx.add_document("doc1", "a b c d e f g h")
        overlap = idx.query_overlap("a b c d x y z w")
        # 3 out of 7 bigrams match: "a b", "b c", "c d"
        assert "doc1" in overlap
        assert 0.3 < overlap["doc1"] < 0.5

    def test_multiple_documents(self):
        idx = NgramIndex(n=2)
        idx.add_document("doc1", "a b c d")
        idx.add_document("doc2", "c d e f")
        overlap = idx.query_overlap("a b c d e f")
        assert "doc1" in overlap
        assert "doc2" in overlap
