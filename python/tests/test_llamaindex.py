"""
Tests for pytrine.llamaindex — LlamaIndex retriever and embedding integration.

Requires both pytrine (with libtrine.so) and llama_index to be installed.
Tests are skipped automatically if llama_index is not available.
"""

import sys
import types
from unittest import mock

import pytest

# Skip the entire module if llama_index is not installed
llama_index_core = pytest.importorskip("llama_index.core")

from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from pytrine.llamaindex import (
    TrineEmbedding,
    TrineRetriever,
    _CANON_MAP,
    _HAS_LLAMAINDEX,
    _LENS_MAP,
)


# ── Fixtures ────────────────────────────────────────────────────────────

SAMPLE_TEXTS = [
    "The quick brown fox jumps over the lazy dog",
    "Machine learning models require large datasets",
    "Quantum computing leverages superposition and entanglement",
    "The quick brown fox leaps over the lazy dog",
    "Python is a popular programming language",
    "Artificial intelligence is transforming industries",
    "The lazy dog sleeps under the brown fox",
]


@pytest.fixture
def sample_nodes():
    """Create TextNode instances from sample texts."""
    return [TextNode(text=t) for t in SAMPLE_TEXTS]


@pytest.fixture
def retriever(sample_nodes):
    """A TrineRetriever pre-loaded with sample nodes."""
    return TrineRetriever(
        nodes=sample_nodes,
        threshold=0.60,
        lens="dedup",
        similarity_top_k=5,
        score_threshold=0.0,
    )


# ── Retriever Tests ─────────────────────────────────────────────────────

class TestTrineRetriever:
    """Tests for TrineRetriever."""

    def test_from_nodes(self, retriever, sample_nodes):
        """Retriever should index all nodes and return results."""
        results = retriever.retrieve("quick brown fox")
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, NodeWithScore) for r in results)
        # The top result should be one of the fox-related documents
        top = results[0]
        assert top.score > 0.0
        assert "fox" in top.node.get_content().lower() or "dog" in top.node.get_content().lower()

    def test_from_nodes_scores_descending(self, retriever):
        """Results should be sorted by score in descending order."""
        results = retriever.retrieve("quick brown fox")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True), (
            f"Scores not descending: {scores}"
        )

    def test_from_texts(self):
        """from_texts classmethod should create a working retriever."""
        retriever = TrineRetriever.from_texts(
            SAMPLE_TEXTS,
            similarity_top_k=3,
            lens="dedup",
        )
        results = retriever.retrieve("programming language")
        assert isinstance(results, list)
        assert len(results) <= 3
        assert len(results) > 0

    def test_add_nodes(self, retriever):
        """add_nodes should expand the searchable corpus."""
        initial_count = len(retriever._nodes)
        new_nodes = [
            TextNode(text="Rust is a systems programming language"),
            TextNode(text="Go was designed at Google"),
        ]
        retriever.add_nodes(new_nodes)
        assert len(retriever._nodes) == initial_count + 2
        assert len(retriever._embeddings) == initial_count + 2

        # New documents should now be retrievable
        results = retriever.retrieve("Rust programming")
        texts = [r.node.get_content() for r in results]
        assert any("Rust" in t for t in texts)

    def test_top_k_limit(self, sample_nodes):
        """Results should respect the similarity_top_k limit."""
        retriever = TrineRetriever(
            nodes=sample_nodes,
            similarity_top_k=2,
        )
        results = retriever.retrieve("quick brown fox")
        assert len(results) <= 2

    def test_top_k_one(self, sample_nodes):
        """top_k=1 should return at most 1 result."""
        retriever = TrineRetriever(
            nodes=sample_nodes,
            similarity_top_k=1,
        )
        results = retriever.retrieve("fox")
        assert len(results) <= 1

    def test_score_threshold(self, sample_nodes):
        """Results below score_threshold should be filtered out."""
        # Use a very high threshold that filters most results
        retriever = TrineRetriever(
            nodes=sample_nodes,
            similarity_top_k=10,
            score_threshold=0.99,
        )
        results = retriever.retrieve("completely unrelated xyzzy plugh")
        # With a 0.99 threshold, very few (or no) results should pass
        for r in results:
            assert r.score >= 0.99

    def test_score_threshold_zero_returns_all(self, sample_nodes):
        """score_threshold=0.0 should not filter any results."""
        retriever = TrineRetriever(
            nodes=sample_nodes,
            similarity_top_k=100,
            score_threshold=0.0,
        )
        results = retriever.retrieve("test query")
        # Should return all nodes (up to top_k)
        assert len(results) == len(sample_nodes)

    def test_empty_retriever(self):
        """Retriever with no nodes should return empty results."""
        retriever = TrineRetriever(similarity_top_k=5)
        results = retriever.retrieve("anything")
        assert results == []

    def test_retrieve_with_query_bundle(self, retriever):
        """_retrieve should work directly with a QueryBundle."""
        bundle = QueryBundle(query_str="quantum computing")
        results = retriever._retrieve(bundle)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_different_lenses(self, sample_nodes):
        """Different lens settings should produce valid results."""
        for lens_name in ["uniform", "dedup", "edit", "vocab", "code", "legal"]:
            retriever = TrineRetriever(
                nodes=sample_nodes,
                lens=lens_name,
                similarity_top_k=3,
            )
            results = retriever.retrieve("quick brown fox")
            assert len(results) > 0, f"No results with lens={lens_name}"

    def test_different_canons(self, sample_nodes):
        """Different canon settings should produce valid results."""
        for canon_name in ["none", "general", "support", "code", "policy"]:
            retriever = TrineRetriever(
                nodes=sample_nodes,
                canon=canon_name,
                similarity_top_k=3,
            )
            results = retriever.retrieve("quick brown fox")
            assert isinstance(results, list)

    def test_invalid_lens_raises(self):
        """Invalid lens name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown lens"):
            TrineRetriever(lens="nonexistent")

    def test_invalid_canon_raises(self):
        """Invalid canon name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown canon"):
            TrineRetriever(canon="nonexistent")

    def test_node_identity_preserved(self, sample_nodes):
        """Retrieved nodes should be the exact same objects as indexed."""
        retriever = TrineRetriever(
            nodes=sample_nodes,
            similarity_top_k=len(sample_nodes),
            score_threshold=0.0,
        )
        results = retriever.retrieve("fox")
        for r in results:
            assert r.node in sample_nodes


# ── Embedding Tests ─────────────────────────────────────────────────────

class TestTrineEmbedding:
    """Tests for TrineEmbedding (LlamaIndex BaseEmbedding)."""

    def test_text_embedding(self):
        """_get_text_embedding should return a 240-float list."""
        model = TrineEmbedding(lens="dedup", canon="none")
        vector = model._get_text_embedding("Hello, world!")
        assert isinstance(vector, list)
        assert len(vector) == 240
        assert all(isinstance(v, float) for v in vector)

    def test_query_embedding(self):
        """_get_query_embedding should return a 240-float list."""
        model = TrineEmbedding()
        vector = model._get_query_embedding("test query")
        assert isinstance(vector, list)
        assert len(vector) == 240

    def test_text_embeddings_batch(self):
        """_get_text_embeddings should return one vector per text."""
        model = TrineEmbedding()
        texts = ["hello", "world", "test"]
        vectors = model._get_text_embeddings(texts)
        assert isinstance(vectors, list)
        assert len(vectors) == 3
        for v in vectors:
            assert len(v) == 240

    def test_batch_matches_single(self):
        """Batch encoding should match individual encoding."""
        model = TrineEmbedding()
        texts = ["alpha", "beta", "gamma"]
        batch = model._get_text_embeddings(texts)
        singles = [model._get_text_embedding(t) for t in texts]
        for b, s in zip(batch, singles):
            assert b == s

    def test_deterministic(self):
        """Same text should always produce the same vector."""
        model = TrineEmbedding()
        v1 = model._get_text_embedding("deterministic test")
        v2 = model._get_text_embedding("deterministic test")
        assert v1 == v2

    def test_empty_text(self):
        """Empty text should produce an all-zero vector."""
        model = TrineEmbedding()
        vector = model._get_text_embedding("")
        assert all(v == 0.0 for v in vector)

    def test_trit_values_in_range(self):
        """All values should be valid trit values (0.0, 1.0, or 2.0)."""
        model = TrineEmbedding()
        vector = model._get_text_embedding("The quick brown fox")
        for v in vector:
            assert v in (0.0, 1.0, 2.0), f"Unexpected trit value: {v}"

    def test_async_query_embedding(self):
        """Async stub should return the same result as sync."""
        import asyncio

        model = TrineEmbedding()
        sync_result = model._get_query_embedding("async test")
        async_result = asyncio.get_event_loop().run_until_complete(
            model._aget_query_embedding("async test")
        )
        assert sync_result == async_result

    def test_different_lenses_accepted(self):
        """All lens names should be accepted."""
        for lens_name in _LENS_MAP:
            model = TrineEmbedding(lens=lens_name)
            vector = model._get_text_embedding("test")
            assert len(vector) == 240

    def test_different_canons_accepted(self):
        """All canon names should be accepted."""
        for canon_name in _CANON_MAP:
            model = TrineEmbedding(canon=canon_name)
            vector = model._get_text_embedding("test")
            assert len(vector) == 240


# ── Missing LlamaIndex Test ────────────────────────────────────────────

class TestMissingLlamaIndex:
    """Test graceful behavior when llama_index is not installed."""

    def test_missing_llamaindex_retriever(self):
        """
        TrineRetriever should raise ImportError with a helpful message
        when llama_index is not installed.
        """
        # Temporarily patch _HAS_LLAMAINDEX to False
        import pytrine.llamaindex as mod

        original = mod._HAS_LLAMAINDEX
        try:
            mod._HAS_LLAMAINDEX = False
            with pytest.raises(ImportError, match="llama-index-core"):
                TrineRetriever()
        finally:
            mod._HAS_LLAMAINDEX = original

    def test_missing_llamaindex_embedding(self):
        """
        TrineEmbedding should raise ImportError with a helpful message
        when llama_index is not installed.
        """
        import pytrine.llamaindex as mod

        original = mod._HAS_LLAMAINDEX
        try:
            mod._HAS_LLAMAINDEX = False
            with pytest.raises(ImportError, match="llama-index-core"):
                TrineEmbedding()
        finally:
            mod._HAS_LLAMAINDEX = original
