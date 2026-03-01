"""
pytrine.llamaindex — LlamaIndex retriever integration for TRINE.

Provides TrineRetriever (BaseRetriever) and TrineEmbedding (BaseEmbedding)
for integrating TRINE's fast ternary embeddings with LlamaIndex pipelines.

Usage:
    from pytrine.llamaindex import TrineRetriever
    from llama_index.core.schema import TextNode

    nodes = [TextNode(text="doc 1"), TextNode(text="doc 2")]
    retriever = TrineRetriever(nodes=nodes, threshold=0.60, lens="dedup")
    results = retriever.retrieve("query text")

Embedding interface:
    from pytrine.llamaindex import TrineEmbedding

    embed_model = TrineEmbedding(lens="dedup", canon="none")
    vector = embed_model.get_text_embedding("some text")
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from pytrine.trine import Canon, Embedding, Lens, TrineEncoder, TRINE_CHANNELS

# ── Import guard ────────────────────────────────────────────────────────
# LlamaIndex is an optional dependency.  If it is not installed we define
# lightweight stubs so that *importing* this module never fails — but
# actually instantiating the classes will raise a clear error.

_LLAMAINDEX_INSTALL_MSG = (
    "LlamaIndex is required for pytrine.llamaindex integration.\n"
    "Install it with:  pip install llama-index-core"
)

try:
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.schema import (
        BaseNode,
        NodeWithScore,
        QueryBundle,
        TextNode,
    )
    from llama_index.core.embeddings import BaseEmbedding as _LIBaseEmbedding

    _HAS_LLAMAINDEX = True
except ImportError:
    _HAS_LLAMAINDEX = False

    # Minimal stubs so the module can still be imported for introspection
    # or testing the "missing dependency" path.
    class BaseRetriever:  # type: ignore[no-redef]
        """Stub — llama_index is not installed."""

        def __init_subclass__(cls, **kwargs: Any) -> None:
            super().__init_subclass__(**kwargs)

    class _LIBaseEmbedding:  # type: ignore[no-redef]
        """Stub — llama_index is not installed."""

        def __init_subclass__(cls, **kwargs: Any) -> None:
            super().__init_subclass__(**kwargs)

    class BaseNode:  # type: ignore[no-redef]
        pass

    class NodeWithScore:  # type: ignore[no-redef]
        pass

    class QueryBundle:  # type: ignore[no-redef]
        pass

    class TextNode:  # type: ignore[no-redef]
        pass


def _require_llamaindex() -> None:
    """Raise ImportError with a helpful message if LlamaIndex is missing."""
    if not _HAS_LLAMAINDEX:
        raise ImportError(_LLAMAINDEX_INSTALL_MSG)


# ── Name-to-object maps ────────────────────────────────────────────────

_LENS_MAP = {
    "uniform": Lens.UNIFORM,
    "dedup": Lens.DEDUP,
    "edit": Lens.EDIT,
    "vocab": Lens.VOCAB,
    "code": Lens.CODE,
    "legal": Lens.LEGAL,
    "medical": Lens.MEDICAL,
    "support": Lens.SUPPORT,
    "policy": Lens.POLICY,
}

_CANON_MAP = {
    "none": Canon.NONE,
    "support": Canon.SUPPORT,
    "code": Canon.CODE,
    "policy": Canon.POLICY,
    "general": Canon.GENERAL,
}


def _resolve_lens(name: str) -> Lens:
    """Resolve a lens name string to a Lens object."""
    key = name.lower()
    if key not in _LENS_MAP:
        raise ValueError(
            f"Unknown lens {name!r}. "
            f"Available: {', '.join(sorted(_LENS_MAP))}"
        )
    return _LENS_MAP[key]


def _resolve_canon(name: str) -> int:
    """Resolve a canon name string to a Canon preset integer."""
    key = name.lower()
    if key not in _CANON_MAP:
        raise ValueError(
            f"Unknown canon {name!r}. "
            f"Available: {', '.join(sorted(_CANON_MAP))}"
        )
    return _CANON_MAP[key]


def _embedding_to_floats(emb: Embedding) -> List[float]:
    """Convert a TRINE Embedding (240 trits) to a list of floats."""
    trits = emb.trits
    return [float(t) for t in trits]


# ── TrineRetriever ──────────────────────────────────────────────────────

class TrineRetriever(BaseRetriever):
    """
    LlamaIndex retriever backed by TRINE ternary embeddings.

    Performs brute-force comparison against all stored node embeddings,
    which is practical because TRINE achieves ~5.8M compares/sec.

    Parameters
    ----------
    nodes : list[BaseNode], optional
        Initial nodes to index.
    threshold : float
        Similarity threshold (unused for retrieval ranking but stored
        for compatibility).  Default 0.60.
    lens : str
        Lens name for comparison.  Default ``"dedup"``.
    canon : str
        Canonicalization preset name.  Default ``"none"``.
    similarity_top_k : int
        Maximum number of results to return.  Default 5.
    score_threshold : float
        Minimum similarity score for a result to be included.  Default 0.0.
    """

    def __init__(
        self,
        nodes: Optional[List[BaseNode]] = None,
        threshold: float = 0.60,
        lens: str = "dedup",
        canon: str = "none",
        similarity_top_k: int = 5,
        score_threshold: float = 0.0,
        **kwargs: Any,
    ) -> None:
        _require_llamaindex()
        super().__init__(**kwargs)

        self._encoder = TrineEncoder()
        self._lens = _resolve_lens(lens)
        self._canon = _resolve_canon(canon)
        self._threshold = float(threshold)
        self._similarity_top_k = int(similarity_top_k)
        self._score_threshold = float(score_threshold)

        self._nodes: List[BaseNode] = []
        self._embeddings: List[Embedding] = []

        if nodes:
            self.add_nodes(nodes)

    # ── Public API ──────────────────────────────────────────────────

    def add_nodes(self, nodes: Sequence[BaseNode]) -> None:
        """
        Add nodes to the retriever's internal store.

        Each node's ``.get_content()`` is encoded into a TRINE embedding.

        Parameters
        ----------
        nodes : sequence of BaseNode
            Nodes to add.
        """
        texts = [node.get_content() for node in nodes]
        new_embeddings = self._encoder.encode_batch(texts, canon=self._canon)
        self._nodes.extend(nodes)
        self._embeddings.extend(new_embeddings)

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        similarity_top_k: int = 5,
        **kwargs: Any,
    ) -> "TrineRetriever":
        """
        Create a TrineRetriever from a list of plain text strings.

        Each text is wrapped in a ``TextNode`` automatically.

        Parameters
        ----------
        texts : list of str
            Documents to index.
        similarity_top_k : int
            Maximum results per query.
        **kwargs
            Additional keyword arguments forwarded to the constructor
            (e.g. ``lens``, ``canon``, ``threshold``, ``score_threshold``).

        Returns
        -------
        TrineRetriever
        """
        _require_llamaindex()
        nodes = [TextNode(text=t) for t in texts]
        return cls(nodes=nodes, similarity_top_k=similarity_top_k, **kwargs)

    # ── BaseRetriever protocol ──────────────────────────────────────

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Retrieve the most similar nodes for a query.

        This is the core method required by LlamaIndex's BaseRetriever.

        Steps:
            1. Encode the query text with TRINE.
            2. Compare against all stored embeddings using the configured lens.
            3. Sort by descending similarity.
            4. Filter by score_threshold.
            5. Return top-k as NodeWithScore objects.

        Parameters
        ----------
        query_bundle : QueryBundle
            The query (uses ``query_bundle.query_str``).

        Returns
        -------
        list[NodeWithScore]
            Ranked results with similarity scores.
        """
        if not self._nodes:
            return []

        query_emb = self._encoder.encode(query_bundle.query_str, canon=self._canon)

        # Score all stored embeddings
        scored: List[tuple] = []
        for idx, stored_emb in enumerate(self._embeddings):
            sim = query_emb.similarity(stored_emb, lens=self._lens)
            if sim >= self._score_threshold:
                scored.append((idx, sim))

        # Sort descending by similarity
        scored.sort(key=lambda x: x[1], reverse=True)

        # Take top-k
        top = scored[: self._similarity_top_k]

        # Build NodeWithScore results
        results = []
        for idx, sim in top:
            results.append(NodeWithScore(node=self._nodes[idx], score=sim))

        return results


# ── TrineEmbedding ──────────────────────────────────────────────────────

class TrineEmbedding(_LIBaseEmbedding):
    """
    LlamaIndex embedding model backed by TRINE.

    Converts text to 240-dimensional float vectors (one float per trit).
    This allows TRINE embeddings to be used with any LlamaIndex component
    that expects a standard embedding model.

    Parameters
    ----------
    lens : str
        Lens name (used for metadata; encoding itself is lens-independent).
        Default ``"dedup"``.
    canon : str
        Canonicalization preset name.  Default ``"none"``.
    """

    # Pydantic model config for LlamaIndex BaseEmbedding
    model_name: str = "trine"

    def __init__(
        self,
        lens: str = "dedup",
        canon: str = "none",
        **kwargs: Any,
    ) -> None:
        _require_llamaindex()
        super().__init__(**kwargs)
        self._encoder = TrineEncoder()
        self._lens = _resolve_lens(lens)
        self._canon = _resolve_canon(canon)

    def _get_text_embedding(self, text: str) -> List[float]:
        """
        Encode a single document text into a float embedding vector.

        Parameters
        ----------
        text : str
            The document text.

        Returns
        -------
        list[float]
            240-dimensional float vector.
        """
        emb = self._encoder.encode(text, canon=self._canon)
        return _embedding_to_floats(emb)

    def _get_query_embedding(self, query: str) -> List[float]:
        """
        Encode a query string into a float embedding vector.

        Parameters
        ----------
        query : str
            The query text.

        Returns
        -------
        list[float]
            240-dimensional float vector.
        """
        emb = self._encoder.encode(query, canon=self._canon)
        return _embedding_to_floats(emb)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Encode multiple texts into float embedding vectors.

        Uses TRINE's batch encoding API for efficiency.

        Parameters
        ----------
        texts : list of str
            Document texts.

        Returns
        -------
        list[list[float]]
            One 240-dimensional vector per text.
        """
        embeddings = self._encoder.encode_batch(texts, canon=self._canon)
        return [_embedding_to_floats(emb) for emb in embeddings]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """
        Async stub for query embedding.

        TRINE encoding is CPU-bound and completes in microseconds,
        so this simply delegates to the synchronous implementation.

        Parameters
        ----------
        query : str
            The query text.

        Returns
        -------
        list[float]
            240-dimensional float vector.
        """
        return self._get_query_embedding(query)


__all__ = [
    "TrineRetriever",
    "TrineEmbedding",
]
