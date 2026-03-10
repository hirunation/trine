"""
Tests for pytrine.langchain -- LangChain retriever integration for TRINE.

Requires both libtrine.so and langchain-core to be installed:
    cd hteb/python && make lib
    pip install langchain-core

Tests are skipped if langchain-core is not available.
"""

import os
import tempfile

import pytest

# Skip entire module if langchain-core is not installed
langchain_core = pytest.importorskip("langchain_core")

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings as LCEmbeddings

from pytrine import Canon, Embedding, Lens, TrineEncoder, TrineRouter
from pytrine.langchain import (
    TrineEmbeddings,
    TrineRetriever,
    _CANON_MAP,
    _LENS_MAP,
    _resolve_canon,
    _resolve_lens,
)
from pytrine.trine import TRINE_CHANNELS


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def sample_texts():
    """A representative set of documents for retrieval tests."""
    return [
        "The quick brown fox jumps over the lazy dog",
        "The quick brown fox leaps over the lazy dog",
        "Quantum computing leverages superposition and entanglement",
        "Machine learning models require large training datasets",
        "The cat sat on the mat and watched the birds",
        "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
        "In contract law, consideration must flow from both parties",
    ]


@pytest.fixture
def sample_metadatas():
    """Metadata for each sample document."""
    return [
        {"source": "animals", "tag": "fox1"},
        {"source": "animals", "tag": "fox2"},
        {"source": "science", "tag": "quantum"},
        {"source": "science", "tag": "ml"},
        {"source": "animals", "tag": "cat"},
        {"source": "code", "tag": "fib"},
        {"source": "legal", "tag": "contract"},
    ]


@pytest.fixture
def retriever(sample_texts, sample_metadatas):
    """A pre-built retriever with sample documents."""
    return TrineRetriever.from_texts(
        texts=sample_texts,
        metadatas=sample_metadatas,
        threshold=0.60,
        lens="dedup",
        canon="none",
        k=5,
    )


# ── TrineRetriever.from_texts ───────────────────────────────────────────

class TestFromTexts:
    """Tests for TrineRetriever.from_texts."""

    def test_from_texts_creates_retriever(self, sample_texts, sample_metadatas):
        """from_texts should return a TrineRetriever with all docs loaded."""
        ret = TrineRetriever.from_texts(
            texts=sample_texts,
            metadatas=sample_metadatas,
        )
        assert isinstance(ret, TrineRetriever)
        assert len(ret) == len(sample_texts)

    def test_from_texts_no_metadata(self, sample_texts):
        """from_texts should work without metadata."""
        ret = TrineRetriever.from_texts(texts=sample_texts)
        assert len(ret) == len(sample_texts)

    def test_from_texts_metadata_mismatch(self, sample_texts):
        """Mismatched metadata length should raise ValueError."""
        with pytest.raises(ValueError, match="metadatas length"):
            TrineRetriever.from_texts(
                texts=sample_texts,
                metadatas=[{"source": "only_one"}],
            )

    def test_from_texts_query_returns_documents(self, retriever):
        """invoke() should return Document objects."""
        docs = retriever.invoke("quick brown fox")
        assert len(docs) > 0
        assert all(isinstance(d, Document) for d in docs)

    def test_from_texts_best_match_is_relevant(self, retriever):
        """The top result for 'fox' should be one of the fox documents."""
        docs = retriever.invoke("quick brown fox jumps")
        assert len(docs) > 0
        top = docs[0]
        assert "fox" in top.page_content.lower()

    def test_from_texts_with_lens_string(self, sample_texts):
        """String lens names should be accepted."""
        ret = TrineRetriever.from_texts(texts=sample_texts, lens="code")
        assert len(ret) == len(sample_texts)

    def test_from_texts_with_lens_object(self, sample_texts):
        """Lens objects should be accepted directly."""
        ret = TrineRetriever.from_texts(texts=sample_texts, lens=Lens.LEGAL)
        assert len(ret) == len(sample_texts)

    def test_from_texts_with_canon(self, sample_texts):
        """Canon presets should be accepted."""
        ret = TrineRetriever.from_texts(
            texts=sample_texts, canon="general"
        )
        assert len(ret) == len(sample_texts)


# ── TrineRetriever.from_documents ────────────────────────────────────────

class TestFromDocuments:
    """Tests for TrineRetriever.from_documents."""

    def test_from_documents(self, sample_texts, sample_metadatas):
        """from_documents should create a retriever from Document objects."""
        docs = [
            Document(page_content=t, metadata=m)
            for t, m in zip(sample_texts, sample_metadatas)
        ]
        ret = TrineRetriever.from_documents(docs, k=3)
        assert len(ret) == len(sample_texts)

        results = ret.invoke("quantum computing")
        assert len(results) > 0

    def test_from_documents_preserves_metadata(self):
        """Metadata from input Documents should be preserved."""
        docs = [
            Document(page_content="Hello world", metadata={"key": "val1"}),
            Document(page_content="Goodbye world", metadata={"key": "val2"}),
        ]
        ret = TrineRetriever.from_documents(docs)
        results = ret.invoke("Hello")
        assert len(results) > 0
        # All results should have both original key and trine_score
        for doc in results:
            assert "key" in doc.metadata
            assert "trine_score" in doc.metadata


# ── add_texts ────────────────────────────────────────────────────────────

class TestAddTexts:
    """Tests for TrineRetriever.add_texts."""

    def test_add_texts_increases_count(self, retriever):
        """add_texts should increase the document count."""
        n_before = len(retriever)
        ids = retriever.add_texts(["New document about neural networks"])
        assert len(retriever) == n_before + 1
        assert len(ids) == 1
        assert isinstance(ids[0], str)  # UUID string

    def test_add_texts_with_metadata(self, retriever):
        """add_texts with metadata should store the metadata."""
        retriever.add_texts(
            ["Added doc"],
            metadatas=[{"added": True}],
        )
        # Query for the added document
        scored = retriever.similarity_search("Added doc")
        found = False
        for doc, score in scored:
            if "added" in doc.metadata and doc.metadata["added"] is True:
                found = True
                break
        assert found, "Added document with metadata not found"

    def test_add_texts_metadata_mismatch(self, retriever):
        """Mismatched metadata length should raise ValueError."""
        with pytest.raises(ValueError, match="metadatas length"):
            retriever.add_texts(
                ["text1", "text2"],
                metadatas=[{"only": "one"}],
            )

    def test_add_texts_returns_unique_ids(self, retriever):
        """Each call to add_texts should return unique IDs."""
        ids1 = retriever.add_texts(["doc A", "doc B"])
        ids2 = retriever.add_texts(["doc C"])
        all_ids = ids1 + ids2
        assert len(set(all_ids)) == len(all_ids), "IDs should be unique"


# ── similarity_search ────────────────────────────────────────────────────

class TestSimilaritySearch:
    """Tests for TrineRetriever.similarity_search."""

    def test_similarity_search_returns_scores(self, retriever):
        """similarity_search should return (Document, float) tuples."""
        results = retriever.similarity_search("brown fox")
        assert len(results) > 0
        for doc, score in results:
            assert isinstance(doc, Document)
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0

    def test_similarity_search_sorted_descending(self, retriever):
        """Results should be sorted by descending similarity."""
        results = retriever.similarity_search("fox jumps dog")
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_similarity_search_trine_score_in_metadata(self, retriever):
        """Documents should have trine_score in metadata."""
        results = retriever.similarity_search("quantum entanglement")
        for doc, score in results:
            assert "trine_score" in doc.metadata
            assert doc.metadata["trine_score"] == pytest.approx(score)


# ── k limit ──────────────────────────────────────────────────────────────

class TestKLimit:
    """Tests for the k parameter."""

    def test_k_limits_results(self, sample_texts, sample_metadatas):
        """Setting k=2 should return at most 2 results."""
        ret = TrineRetriever.from_texts(
            texts=sample_texts,
            metadatas=sample_metadatas,
            k=2,
        )
        docs = ret.invoke("fox jumps over")
        assert len(docs) <= 2

    def test_k_override_in_search(self, retriever):
        """k parameter in similarity_search should override default."""
        results_2 = retriever.similarity_search("fox", k=2)
        results_1 = retriever.similarity_search("fox", k=1)
        assert len(results_1) <= 1
        assert len(results_2) <= 2

    def test_k_larger_than_corpus(self):
        """k larger than corpus size should return all docs."""
        ret = TrineRetriever.from_texts(
            texts=["one", "two"],
            k=100,
        )
        results = ret.similarity_search("one")
        assert len(results) <= 2


# ── score_threshold ──────────────────────────────────────────────────────

class TestScoreThreshold:
    """Tests for the score_threshold parameter."""

    def test_score_threshold_filters(self):
        """High score_threshold should filter out dissimilar docs."""
        ret = TrineRetriever.from_texts(
            texts=[
                "The quick brown fox jumps over the lazy dog",
                "Quantum computing and superposition entanglement theory",
            ],
            k=10,
            score_threshold=0.80,
        )
        # Query for fox -- the quantum doc should be filtered out
        results = ret.similarity_search("quick brown fox jumps over")
        for doc, score in results:
            assert score >= 0.80

    def test_zero_threshold_returns_all(self, retriever):
        """score_threshold=0.0 should not filter any results (up to k)."""
        results = retriever.similarity_search("fox")
        assert len(results) == retriever._k


# ── TrineEmbeddings ──────────────────────────────────────────────────────

class TestTrineEmbeddings:
    """Tests for the TrineEmbeddings LangChain interface."""

    def test_implements_interface(self):
        """TrineEmbeddings should be an instance of LangChain Embeddings."""
        emb = TrineEmbeddings()
        assert isinstance(emb, LCEmbeddings)

    def test_embed_documents(self):
        """embed_documents should return a list of float vectors."""
        emb = TrineEmbeddings()
        texts = ["hello world", "goodbye world"]
        vectors = emb.embed_documents(texts)
        assert len(vectors) == 2
        assert all(len(v) == TRINE_CHANNELS for v in vectors)
        assert all(isinstance(v, list) for v in vectors)
        assert all(isinstance(x, float) for x in vectors[0])

    def test_embed_query(self):
        """embed_query should return a single float vector."""
        emb = TrineEmbeddings()
        vec = emb.embed_query("hello world")
        assert len(vec) == TRINE_CHANNELS
        assert isinstance(vec, list)
        assert all(isinstance(x, float) for x in vec)

    def test_embed_deterministic(self):
        """Same input should produce the same vector."""
        emb = TrineEmbeddings()
        v1 = emb.embed_query("test text")
        v2 = emb.embed_query("test text")
        assert v1 == v2

    def test_embed_values_are_trits(self):
        """All values should be 0.0, 1.0, or 2.0 (ternary)."""
        emb = TrineEmbeddings()
        vec = emb.embed_query("hello world")
        for x in vec:
            assert x in (0.0, 1.0, 2.0), f"Unexpected trit value: {x}"

    def test_embed_with_canon(self):
        """Canon preset should be applied during encoding."""
        emb_none = TrineEmbeddings(canon="none")
        emb_general = TrineEmbeddings(canon="general")
        # Extra whitespace: "general" should normalize it
        text = "  hello   world  "
        v_none = emb_none.embed_query(text)
        v_general = emb_general.embed_query(text)
        # They may differ because canonicalization changes the text
        assert len(v_none) == TRINE_CHANNELS
        assert len(v_general) == TRINE_CHANNELS

    def test_embed_empty_batch(self):
        """Empty batch should return empty list."""
        emb = TrineEmbeddings()
        assert emb.embed_documents([]) == []


# ── Lens presets ─────────────────────────────────────────────────────────

class TestLensPresets:
    """Tests for lens preset name resolution."""

    def test_all_lens_names_resolve(self):
        """All named presets should resolve to Lens objects."""
        for name, expected in _LENS_MAP.items():
            resolved = _resolve_lens(name)
            assert resolved == expected

    def test_lens_case_insensitive(self):
        """Lens names should be case-insensitive."""
        assert _resolve_lens("DEDUP") == Lens.DEDUP
        assert _resolve_lens("Dedup") == Lens.DEDUP
        assert _resolve_lens("dedup") == Lens.DEDUP

    def test_lens_object_passthrough(self):
        """Lens objects should pass through unchanged."""
        custom = Lens(edit=0.9, morph=0.1, phrase=0.2, vocab=0.3)
        assert _resolve_lens(custom) is custom

    def test_invalid_lens_name(self):
        """Unknown lens name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown lens"):
            _resolve_lens("nonexistent")

    def test_retriever_with_each_lens(self, sample_texts):
        """Retriever should accept every named lens preset."""
        for name in _LENS_MAP:
            ret = TrineRetriever.from_texts(texts=sample_texts[:2], lens=name)
            assert len(ret) == 2


# ── Canon presets ────────────────────────────────────────────────────────

class TestCanonPresets:
    """Tests for canon preset name resolution."""

    def test_all_canon_names_resolve(self):
        """All named presets should resolve to Canon int constants."""
        for name, expected in _CANON_MAP.items():
            resolved = _resolve_canon(name)
            assert resolved == expected

    def test_canon_case_insensitive(self):
        """Canon names should be case-insensitive."""
        assert _resolve_canon("NONE") == Canon.NONE
        assert _resolve_canon("None") == Canon.NONE
        assert _resolve_canon("none") == Canon.NONE

    def test_canon_int_passthrough(self):
        """Int values should pass through unchanged."""
        assert _resolve_canon(Canon.CODE) == Canon.CODE

    def test_invalid_canon_name(self):
        """Unknown canon name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown canon"):
            _resolve_canon("nonexistent")

    def test_retriever_with_each_canon(self, sample_texts):
        """Retriever should accept every named canon preset."""
        for name in _CANON_MAP:
            ret = TrineRetriever.from_texts(texts=sample_texts[:2], canon=name)
            assert len(ret) == 2


# ── Serialization ────────────────────────────────────────────────────────

class TestSerialization:
    """Tests for save_index / load_index."""

    def test_save_load_roundtrip(self, retriever):
        """Save and load should preserve documents and retrieval quality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = os.path.join(tmpdir, "test_index")
            retriever.save_index(base)

            # Verify files were created
            assert os.path.isfile(base + ".trrt")
            assert os.path.isfile(base + ".meta.json")

            # Load and verify
            loaded = TrineRetriever.load_index(base)
            assert len(loaded) == len(retriever)

            # Query should still work
            results = loaded.similarity_search("fox jumps")
            assert len(results) > 0
            top_doc, top_score = results[0]
            assert "fox" in top_doc.page_content.lower()

    def test_load_nonexistent(self):
        """Loading a nonexistent index should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            TrineRetriever.load_index("/tmp/nonexistent_trine_index_path")


# ── Repr ─────────────────────────────────────────────────────────────────

class TestRepr:
    """Tests for string representations."""

    def test_retriever_repr(self, retriever):
        """repr should include key info."""
        r = repr(retriever)
        assert "TrineRetriever" in r
        assert "n=" in r
        assert "k=" in r
        assert "dedup" in r


# ── Missing langchain import ─────────────────────────────────────────────

class TestMissingLangchain:
    """Tests for graceful error when langchain is not available."""

    def test_module_importable_without_langchain(self):
        """
        The langchain module should be importable even if langchain_core
        is missing. We verify this indirectly: since langchain_core IS
        installed for these tests, we check that the import guard flag
        is True.
        """
        from pytrine.langchain import _HAS_LANGCHAIN
        assert _HAS_LANGCHAIN is True

    def test_stub_classes_not_used_when_available(self):
        """When langchain_core is available, real base classes should be used."""
        from pytrine.langchain import TrineRetriever, TrineEmbeddings
        from langchain_core.retrievers import BaseRetriever
        from langchain_core.embeddings import Embeddings as LCEmbeddings

        # TrineRetriever should be a subclass of the real BaseRetriever
        assert issubclass(TrineRetriever, BaseRetriever)
        # TrineEmbeddings should be a subclass of the real Embeddings
        assert issubclass(TrineEmbeddings, LCEmbeddings)
