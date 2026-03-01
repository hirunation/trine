"""
Comprehensive test suite for pytrine.

Tests cover encoding, comparison, index operations, routing, canonicalization,
batch encoding, lens presets, and pack/unpack round-trips.

Requires libtrine.so to be compiled:
    cd hteb/python && make lib
"""

import os
import tempfile

import pytest

from pytrine import (
    Canon,
    Embedding,
    Lens,
    Result,
    TrineEncoder,
    TrineIndex,
    TrineRouter,
)


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def encoder():
    """Shared encoder instance."""
    return TrineEncoder()


@pytest.fixture
def sample_texts():
    """A set of sample texts for testing."""
    return [
        "The quick brown fox jumps over the lazy dog",
        "The quick brown fox leaps over the lazy dog",
        "Quantum computing leverages superposition and entanglement",
        "Hello, world!",
        "hello, world!",
        "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit",
        "",
    ]


# ── Encoding Tests ───────────────────────────────────────────────────────

class TestEncode:
    """Tests for TrineEncoder.encode."""

    def test_encode_returns_embedding(self, encoder):
        """encode() should return an Embedding object."""
        emb = encoder.encode("hello")
        assert isinstance(emb, Embedding)
        assert len(emb) == 240

    def test_encode_deterministic(self, encoder):
        """Same input should always produce the same embedding."""
        text = "The quick brown fox"
        emb1 = encoder.encode(text)
        emb2 = encoder.encode(text)
        assert emb1 == emb2

    def test_encode_case_insensitive(self, encoder):
        """Shingle encoder is case-insensitive."""
        emb_lower = encoder.encode("hello world")
        emb_upper = encoder.encode("HELLO WORLD")
        emb_mixed = encoder.encode("Hello World")
        assert emb_lower == emb_upper
        assert emb_lower == emb_mixed

    def test_encode_nonempty_produces_nonzero(self, encoder):
        """Non-empty text should produce at least some non-zero trits."""
        emb = encoder.encode("test")
        assert emb.fill_ratio > 0.0

    def test_encode_empty_text(self, encoder):
        """Empty text should produce an all-zero embedding."""
        emb = encoder.encode("")
        assert emb.fill_ratio == 0.0

    def test_encode_long_text(self, encoder):
        """Long texts should encode without error."""
        text = "word " * 1000
        emb = encoder.encode(text)
        assert isinstance(emb, Embedding)
        assert emb.fill_ratio > 0.0

    def test_encode_unicode_text(self, encoder):
        """Non-ASCII characters should be handled (masked to 7-bit)."""
        emb = encoder.encode("caf\u00e9 na\u00efve")
        assert isinstance(emb, Embedding)


class TestBatchEncode:
    """Tests for TrineEncoder.encode_batch."""

    def test_batch_encode_empty(self, encoder):
        """Empty list should return empty list."""
        result = encoder.encode_batch([])
        assert result == []

    def test_batch_encode_matches_single(self, encoder):
        """Batch encoding should produce identical results to single encoding."""
        texts = ["hello", "world", "test"]
        batch_results = encoder.encode_batch(texts)
        single_results = [encoder.encode(t) for t in texts]

        assert len(batch_results) == len(single_results)
        for batch_emb, single_emb in zip(batch_results, single_results):
            assert batch_emb == single_emb

    def test_batch_encode_count(self, encoder):
        """Batch should return one embedding per input text."""
        texts = ["a", "b", "c", "d", "e"]
        results = encoder.encode_batch(texts)
        assert len(results) == 5


# ── Similarity Tests ─────────────────────────────────────────────────────

class TestSimilarity:
    """Tests for Embedding.similarity."""

    def test_similarity_identity(self, encoder):
        """Self-similarity should be 1.0 (or very close)."""
        emb = encoder.encode("The quick brown fox")
        sim = emb.similarity(emb)
        assert sim >= 0.99, f"Self-similarity = {sim}, expected ~1.0"

    def test_similarity_similar_texts(self, encoder):
        """Similar texts should have high similarity."""
        a = encoder.encode("The quick brown fox jumps over the lazy dog")
        b = encoder.encode("The quick brown fox leaps over the lazy dog")
        sim = a.similarity(b, lens=Lens.DEDUP)
        assert sim > 0.5, f"Similar text similarity = {sim}, expected > 0.5"

    def test_similarity_dissimilar(self, encoder):
        """Unrelated texts should have lower similarity than similar texts."""
        a = encoder.encode("The quick brown fox jumps over the lazy dog")
        b = encoder.encode("Quantum computing leverages superposition and entanglement")
        c = encoder.encode("The quick brown fox leaps over the lazy dog")
        sim_dissimilar = a.similarity(b, lens=Lens.DEDUP)
        sim_similar = a.similarity(c, lens=Lens.DEDUP)
        assert sim_dissimilar < sim_similar, (
            f"Dissimilar ({sim_dissimilar:.4f}) should be < similar ({sim_similar:.4f})"
        )

    def test_similarity_symmetric(self, encoder):
        """similarity(a, b) should equal similarity(b, a)."""
        a = encoder.encode("hello world")
        b = encoder.encode("world hello")
        sim_ab = a.similarity(b)
        sim_ba = b.similarity(a)
        assert abs(sim_ab - sim_ba) < 1e-6, f"Asymmetric: {sim_ab} vs {sim_ba}"

    def test_similarity_with_different_lenses(self, encoder):
        """Different lenses should produce different similarity values."""
        a = encoder.encode("The quick brown fox")
        b = encoder.encode("The quick brown cat")
        sim_uniform = a.similarity(b, lens=Lens.UNIFORM)
        sim_edit = a.similarity(b, lens=Lens.EDIT)
        sim_vocab = a.similarity(b, lens=Lens.VOCAB)
        # At least two of the three should differ
        values = {round(sim_uniform, 4), round(sim_edit, 4), round(sim_vocab, 4)}
        assert len(values) >= 2, "Expected different lenses to produce different values"


# ── Pack/Unpack Tests ────────────────────────────────────────────────────

class TestPacking:
    """Tests for Embedding pack/unpack."""

    def test_pack_unpack_roundtrip(self, encoder):
        """Pack then unpack should recover the original embedding."""
        emb = encoder.encode("Hello, world!")
        packed = emb.packed
        assert len(packed) == 48

        restored = Embedding.from_packed(packed)
        assert emb == restored

    def test_packed_size(self, encoder):
        """Packed form should always be exactly 48 bytes."""
        for text in ["a", "hello world", "x" * 100]:
            emb = encoder.encode(text)
            assert len(emb.packed) == 48


# ── Fill Ratio Tests ─────────────────────────────────────────────────────

class TestFillRatio:
    """Tests for Embedding.fill_ratio."""

    def test_fill_ratio_range(self, encoder):
        """Fill ratio should be in [0.0, 1.0]."""
        for text in ["", "a", "hello world", "x" * 1000]:
            emb = encoder.encode(text)
            fr = emb.fill_ratio
            assert 0.0 <= fr <= 1.0, f"fill_ratio={fr} out of range for {text!r}"

    def test_fill_ratio_empty_is_zero(self, encoder):
        """Empty text should have fill_ratio = 0.0."""
        emb = encoder.encode("")
        assert emb.fill_ratio == 0.0

    def test_fill_ratio_long_text_is_high(self, encoder):
        """Long text should produce a high fill ratio."""
        emb = encoder.encode("The quick brown fox jumps over the lazy dog")
        assert emb.fill_ratio > 0.3


# ── Index Tests ──────────────────────────────────────────────────────────

class TestIndex:
    """Tests for TrineIndex."""

    def test_index_create(self):
        """Index should be creatable with default params."""
        with TrineIndex() as idx:
            assert len(idx) == 0

    def test_index_add_query(self):
        """Add then query should find duplicates."""
        with TrineIndex(threshold=0.95, lens=Lens.DEDUP, calibrate=False) as idx:
            idx.add("The quick brown fox jumps over the lazy dog", tag="fox")
            idx.add("Quantum computing uses superposition", tag="quantum")

            result = idx.query("The quick brown fox jumps over the lazy dog")
            assert result.is_duplicate
            assert result.tag == "fox"

            result2 = idx.query("xyzzy plugh abracadabra")
            assert not result2.is_duplicate

    def test_index_len(self):
        """len() should track the number of entries."""
        with TrineIndex() as idx:
            assert len(idx) == 0
            idx.add("first")
            assert len(idx) == 1
            idx.add("second")
            assert len(idx) == 2

    def test_index_tag(self):
        """Tags should be retrievable by index."""
        with TrineIndex() as idx:
            idx.add("hello", tag="tag_hello")
            idx.add("world", tag="tag_world")
            assert idx.tag(0) == "tag_hello"
            assert idx.tag(1) == "tag_world"

    def test_index_no_tag(self):
        """Entries without tags should return None."""
        with TrineIndex() as idx:
            idx.add("hello")
            assert idx.tag(0) is None

    def test_index_contains(self):
        """__contains__ should detect duplicates."""
        with TrineIndex(threshold=0.95, calibrate=False) as idx:
            idx.add("The quick brown fox")
            assert "The quick brown fox" in idx
            assert "xyzzy plugh abracadabra" not in idx

    def test_index_save_load(self):
        """Save and load should preserve all entries."""
        with tempfile.NamedTemporaryFile(suffix=".trine", delete=False) as f:
            path = f.name

        try:
            # Build and save
            with TrineIndex(threshold=0.60, lens=Lens.DEDUP) as idx:
                idx.add("Hello, world!", tag="doc1")
                idx.add("Goodbye, world!", tag="doc2")
                idx.add("Quantum computing", tag="doc3")
                idx.save(path)
                original_count = len(idx)

            # Load and verify
            loaded = TrineIndex.load(path)
            try:
                assert len(loaded) == original_count
                assert loaded.tag(0) == "doc1"
                assert loaded.tag(1) == "doc2"
                assert loaded.tag(2) == "doc3"

                # Queries should still work
                result = loaded.query("Hello world")
                assert result.matched_index >= 0
            finally:
                loaded.close()
        finally:
            os.unlink(path)

    def test_index_context_manager(self):
        """Context manager should close properly."""
        idx = TrineIndex()
        idx.add("test")
        idx.close()
        with pytest.raises(RuntimeError, match="closed|freed"):
            idx.add("another")

    def test_index_repr(self):
        """repr should show entry count."""
        with TrineIndex() as idx:
            assert "n=0" in repr(idx)
            idx.add("test")
            assert "n=1" in repr(idx)


# ── Router Tests ─────────────────────────────────────────────────────────

class TestRouter:
    """Tests for TrineRouter."""

    def test_router_create(self):
        """Router should be creatable with default params."""
        with TrineRouter() as rt:
            assert len(rt) == 0

    def test_router_add_query(self):
        """Add then query should find duplicates via routing."""
        with TrineRouter(threshold=0.95, lens=Lens.DEDUP, calibrate=False) as rt:
            rt.add("The quick brown fox jumps over the lazy dog", tag="fox")
            rt.add("Quantum computing uses superposition", tag="quantum")

            result = rt.query("The quick brown fox jumps over the lazy dog")
            assert result.is_duplicate
            assert result.tag == "fox"
            # Result should have stats
            assert hasattr(result, "stats")
            assert result.stats.total_entries == 2

    def test_router_len(self):
        """len() should track entries."""
        with TrineRouter() as rt:
            assert len(rt) == 0
            rt.add("first")
            assert len(rt) == 1

    def test_router_recall_modes(self):
        """Recall modes should be settable."""
        with TrineRouter() as rt:
            rt.set_recall(TrineRouter.FAST)
            assert rt.get_recall() == TrineRouter.FAST
            rt.set_recall(TrineRouter.BALANCED)
            assert rt.get_recall() == TrineRouter.BALANCED
            rt.set_recall(TrineRouter.STRICT)
            assert rt.get_recall() == TrineRouter.STRICT

    def test_router_invalid_recall(self):
        """Invalid recall mode should raise ValueError."""
        with TrineRouter() as rt:
            with pytest.raises(ValueError):
                rt.set_recall(999)

    def test_router_save_load(self):
        """Save and load should preserve all entries."""
        with tempfile.NamedTemporaryFile(suffix=".trrt", delete=False) as f:
            path = f.name

        try:
            with TrineRouter(threshold=0.60) as rt:
                rt.add("Hello, world!", tag="r1")
                rt.add("Goodbye, world!", tag="r2")
                rt.save(path)
                original_count = len(rt)

            loaded = TrineRouter.load(path)
            try:
                assert len(loaded) == original_count
                assert loaded.tag(0) == "r1"
                assert loaded.tag(1) == "r2"
            finally:
                loaded.close()
        finally:
            os.unlink(path)

    def test_router_context_manager(self):
        """Context manager should close properly."""
        rt = TrineRouter()
        rt.add("test")
        rt.close()
        with pytest.raises(RuntimeError, match="closed|freed"):
            rt.add("another")


# ── Canon Tests ──────────────────────────────────────────────────────────

class TestCanon:
    """Tests for Canon canonicalization."""

    def test_canon_none(self):
        """NONE preset should pass through unchanged."""
        text = "  Hello,   World!  "
        result = Canon.apply(text, Canon.NONE)
        assert result == text

    def test_canon_general(self):
        """GENERAL preset should normalize whitespace."""
        text = "  Hello,   World!  "
        result = Canon.apply(text, Canon.GENERAL)
        assert "  " not in result  # no double spaces
        assert result.strip() == result  # no leading/trailing whitespace

    def test_canon_support(self):
        """SUPPORT preset should strip timestamps and UUIDs."""
        text = "Error at 2024-01-15 12:30:45 UUID=550e8400-e29b-41d4-a716-446655440000 code 42"
        result = Canon.apply(text, Canon.SUPPORT)
        assert "2024-01-15" not in result
        assert "550e8400" not in result

    def test_canon_code(self):
        """CODE preset should normalize identifiers."""
        text = "  myVariableName  some_snake_case  "
        result = Canon.apply(text, Canon.CODE)
        assert isinstance(result, str)

    def test_canon_policy(self):
        """POLICY preset should bucket numbers."""
        text = "Error code 12345 on port 8080"
        result = Canon.apply(text, Canon.POLICY)
        assert "12345" not in result

    def test_canon_presets_names(self):
        """All presets should have names."""
        assert Canon.name(Canon.NONE) == "NONE"
        assert Canon.name(Canon.SUPPORT) == "SUPPORT"
        assert Canon.name(Canon.CODE) == "CODE"
        assert Canon.name(Canon.POLICY) == "POLICY"
        assert Canon.name(Canon.GENERAL) == "GENERAL"

    def test_encode_with_canon(self, encoder):
        """Encoding with canonicalization should succeed."""
        emb = encoder.encode("  Hello,   World!  ", canon=Canon.GENERAL)
        assert isinstance(emb, Embedding)
        assert emb.fill_ratio > 0.0


# ── Lens Tests ───────────────────────────────────────────────────────────

class TestLens:
    """Tests for Lens presets and custom lenses."""

    def test_lens_presets_exist(self):
        """All predefined lens presets should be accessible."""
        presets = [
            Lens.UNIFORM, Lens.DEDUP, Lens.EDIT, Lens.VOCAB,
            Lens.CODE, Lens.LEGAL, Lens.MEDICAL, Lens.SUPPORT, Lens.POLICY,
        ]
        for p in presets:
            assert isinstance(p, Lens)
            assert len(p.weights) == 4

    def test_custom_lens(self):
        """Custom lens should store the provided weights."""
        custom = Lens(edit=0.9, morph=0.8, phrase=0.7, vocab=0.6)
        assert custom.weights == (0.9, 0.8, 0.7, 0.6)

    def test_lens_equality(self):
        """Lenses with same weights should be equal."""
        a = Lens(edit=1.0, morph=0.5, phrase=0.3, vocab=0.2)
        b = Lens(edit=1.0, morph=0.5, phrase=0.3, vocab=0.2)
        assert a == b

    def test_lens_repr(self):
        """Repr should include weight values."""
        lens = Lens.DEDUP
        r = repr(lens)
        assert "Lens(" in r
        assert "0.50" in r

    def test_lens_presets_values(self):
        """Verify specific preset weight values match C header."""
        assert Lens.UNIFORM.weights == (1.0, 1.0, 1.0, 1.0)
        assert Lens.DEDUP.weights == (0.5, 0.5, 0.7, 1.0)
        assert Lens.EDIT.weights == (1.0, 0.3, 0.1, 0.0)
        assert Lens.VOCAB.weights == (0.0, 0.2, 0.3, 1.0)


# ── Embedding Object Tests ──────────────────────────────────────────────

class TestEmbedding:
    """Tests for the Embedding class itself."""

    def test_embedding_from_bytes(self):
        """Embedding can be created from bytes."""
        data = bytes([1] * 240)
        emb = Embedding(data)
        assert len(emb) == 240

    def test_embedding_from_list(self):
        """Embedding can be created from a list."""
        data = [0] * 240
        emb = Embedding(data)
        assert emb.fill_ratio == 0.0

    def test_embedding_wrong_size(self):
        """Wrong-sized data should raise ValueError."""
        with pytest.raises(ValueError):
            Embedding(bytes([0] * 100))

    def test_embedding_repr(self):
        """Repr should include fill count."""
        emb = Embedding(bytes([0] * 240))
        assert "fill=0/240" in repr(emb)

    def test_embedding_equality(self):
        """Two embeddings with same data should be equal."""
        data = bytes([1, 2, 0] * 80)
        a = Embedding(data)
        b = Embedding(data)
        assert a == b

    def test_embedding_trits_property(self):
        """trits property should return a copy of the data."""
        data = bytes([1, 2, 0] * 80)
        emb = Embedding(data)
        trits = emb.trits
        assert len(trits) == 240
        # Modifying the returned trits should not affect the embedding
        trits[0] = 99
        assert emb.trits[0] != 99


# ── Result Tests ─────────────────────────────────────────────────────────

class TestResult:
    """Tests for the Result class."""

    def test_result_bool_duplicate(self):
        """Duplicate result should be truthy."""
        r = Result(is_duplicate=True, similarity=0.85, calibrated=0.88,
                   matched_index=3, tag="test")
        assert r
        assert r.is_duplicate

    def test_result_bool_unique(self):
        """Non-duplicate result should be falsy."""
        r = Result(is_duplicate=False, similarity=0.30, calibrated=0.32,
                   matched_index=-1)
        assert not r

    def test_result_repr(self):
        """Repr should include key fields."""
        r = Result(is_duplicate=True, similarity=0.9, calibrated=0.92,
                   matched_index=0, tag="doc1")
        s = repr(r)
        assert "DUPLICATE" in s
        assert "doc1" in s
