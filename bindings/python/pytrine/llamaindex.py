"""
pytrine.llamaindex — LlamaIndex retriever integration for TRINE.

Provides TrineRetriever (BaseRetriever) and TrineEmbedding (BaseEmbedding)
for integrating TRINE's fast ternary embeddings with LlamaIndex pipelines.

Stage-2 Support
---------------
Both TrineEmbedding and TrineRetriever accept an optional ``stage2_model``
parameter.  When provided (as a file path or a ``Stage2Model`` instance),
embeddings are computed through the Stage-2 semantic projection pipeline
instead of raw Stage-1 shingle encoding.  TrineRetriever also supports
``blend_alpha`` for weighted S1+S2 similarity scoring.

Block-Diagonal Models
~~~~~~~~~~~~~~~~~~~~~
Set ``block_diagonal=True`` to use a block-diagonal projection model,
which operates on 4 independent 60x60 chain-local blocks instead of
the full 240x240 matrix.  When ``stage2_model`` is provided, the model
is loaded and its projection mode is set to block-diagonal.  When
``stage2_model`` is omitted, a random block-diagonal model is created.

Adaptive Alpha
~~~~~~~~~~~~~~
Provide ``adaptive_alpha`` as a list of 10 floats to enable per-S1-bucket
alpha selection for blending.  Each float corresponds to an S1 similarity
bucket: [0.0-0.1), [0.1-0.2), ..., [0.9-1.0].

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

    # Stage-2 embeddings:
    embed_s2 = TrineEmbedding(stage2_model="trained.trine2", depth=0)
    vector_s2 = embed_s2.get_text_embedding("some text")

    # Block-diagonal Stage-2 embeddings with adaptive alpha:
    embed_bd = TrineEmbedding(
        stage2_model="trained_bd.trine2",
        block_diagonal=True,
        adaptive_alpha=[0.5] * 10,
    )
    vector_bd = embed_bd.get_text_embedding("some text")
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

from pytrine.trine import Canon, Embedding, Lens, TrineEncoder, TRINE_CHANNELS
from pytrine.stage2 import (
    Stage2Encoder,
    Stage2Model,
    s2_compare,
    PROJ_BLOCK_DIAG,
)

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


def _build_s2_encoder(
    stage2_model: Optional[Any],
    depth: int,
    block_diagonal: bool,
    adaptive_alpha: Optional[List[float]],
) -> Optional[Stage2Encoder]:
    """
    Build a Stage2Encoder with optional block-diagonal and adaptive alpha.

    Parameters
    ----------
    stage2_model : str, Stage2Model, or None
        Model path, instance, or None for no Stage-2.
        When None and ``block_diagonal`` is True, a random block-diagonal
        model is created automatically.
    depth : int
        Cascade depth for Stage-2 encoding.
    block_diagonal : bool
        If True, set the model's projection mode to block-diagonal.
        When ``stage2_model`` is None, a random block-diagonal model
        is created instead.
    adaptive_alpha : list of float or None
        10-element list of per-S1-bucket alpha values for adaptive
        blending.  Pass None to skip.

    Returns
    -------
    Stage2Encoder or None
        The configured encoder, or None when Stage-2 is not requested.
    """
    if stage2_model is None and not block_diagonal:
        return None

    # When block_diagonal is requested without an explicit model,
    # create a random block-diagonal model for experimentation.
    if stage2_model is None and block_diagonal:
        model = Stage2Model.random(cells=512, seed=42)
        model.set_projection_mode(PROJ_BLOCK_DIAG)
        s2_enc = Stage2Encoder(model=model, depth=depth)
    else:
        s2_enc = Stage2Encoder(model=stage2_model, depth=depth)
        if block_diagonal:
            s2_enc._model.set_projection_mode(PROJ_BLOCK_DIAG)

    # Apply adaptive alpha if provided
    if adaptive_alpha is not None:
        s2_enc._model.set_adaptive_alpha(adaptive_alpha)

    return s2_enc


# ── TrineRetriever ──────────────────────────────────────────────────────

class TrineRetriever(BaseRetriever):
    """
    LlamaIndex retriever backed by TRINE ternary embeddings.

    Performs brute-force comparison against all stored node embeddings,
    which is practical because TRINE achieves ~5.8M compares/sec.

    When ``stage2_model`` is provided, similarity_search uses blended
    scoring: ``blend_alpha * S1 + (1 - blend_alpha) * S2``.

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
    stage2_model : str or Stage2Model, optional
        A path to a ``.trine2`` file or a ``Stage2Model`` instance.
        When provided, retrieval uses blended S1+S2 scoring.
    blend_alpha : float
        Blending weight (``alpha * S1 + (1 - alpha) * S2``).  Default 0.65.
    block_diagonal : bool
        If True, use block-diagonal projection mode (4 independent
        60x60 chain-local blocks instead of full 240x240 matrix).
        When ``stage2_model`` is also provided, its projection mode
        is set to block-diagonal.  When ``stage2_model`` is None,
        a random block-diagonal model is created.  Default False.
    adaptive_alpha : list of float, optional
        10-element list of per-S1-bucket alpha values for adaptive
        blending.  Each element corresponds to an S1 similarity
        bucket: [0.0-0.1), [0.1-0.2), ..., [0.9-1.0].
        Default is None (disabled).
    """

    def __init__(
        self,
        nodes: Optional[List[BaseNode]] = None,
        threshold: float = 0.60,
        lens: str = "dedup",
        canon: str = "none",
        similarity_top_k: int = 5,
        score_threshold: float = 0.0,
        stage2_model: Optional[Any] = None,
        blend_alpha: float = 0.65,
        block_diagonal: bool = False,
        adaptive_alpha: Optional[List[float]] = None,
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

        # Stage-2 support (with optional block-diagonal and adaptive alpha)
        self._blend_alpha = float(blend_alpha)
        self._s2_encoder = _build_s2_encoder(
            stage2_model, 0, block_diagonal, adaptive_alpha,
        )

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
        stage2_model: Optional[Any] = None,
        blend_alpha: float = 0.65,
        block_diagonal: bool = False,
        adaptive_alpha: Optional[List[float]] = None,
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
        stage2_model : str or Stage2Model, optional
            Stage-2 model for blended scoring.
        blend_alpha : float
            Blending weight (default 0.65).
        block_diagonal : bool
            If True, use block-diagonal projection mode for Stage-2.
            Default False.
        adaptive_alpha : list of float, optional
            10-element per-S1-bucket alpha list for adaptive blending.
            Default None.
        **kwargs
            Additional keyword arguments forwarded to the constructor
            (e.g. ``lens``, ``canon``, ``threshold``, ``score_threshold``).

        Returns
        -------
        TrineRetriever
        """
        _require_llamaindex()
        nodes = [TextNode(text=t) for t in texts]
        return cls(
            nodes=nodes,
            similarity_top_k=similarity_top_k,
            stage2_model=stage2_model,
            blend_alpha=blend_alpha,
            block_diagonal=block_diagonal,
            adaptive_alpha=adaptive_alpha,
            **kwargs,
        )

    # ── BaseRetriever protocol ──────────────────────────────────────

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Retrieve the most similar nodes for a query.

        This is the core method required by LlamaIndex's BaseRetriever.

        Steps:
            1. Encode the query text with TRINE.
            2. Compare against all stored embeddings using the configured lens.
               When a Stage-2 model is configured, uses blended scoring:
               ``blend_alpha * S1 + (1 - blend_alpha) * S2``.
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

        # Stage-2 query embedding (if configured)
        query_s2_emb = None
        if self._s2_encoder is not None:
            query_s2_emb = self._s2_encoder.encode(query_bundle.query_str)

        # Score all stored embeddings
        scored: List[Tuple[int, float]] = []
        for idx, stored_emb in enumerate(self._embeddings):
            s1_sim = query_emb.similarity(stored_emb, lens=self._lens)

            if self._s2_encoder is not None and query_s2_emb is not None:
                # Compute Stage-2 embedding for stored node and blend
                stored_s2_emb = self._s2_encoder.encode(
                    self._nodes[idx].get_content()
                )
                s2_sim = s2_compare(query_s2_emb, stored_s2_emb)
                sim = (
                    self._blend_alpha * s1_sim
                    + (1.0 - self._blend_alpha) * s2_sim
                )
            else:
                sim = s1_sim

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

    When ``stage2_model`` is provided, embeddings are computed through
    the Stage-2 semantic projection pipeline instead of raw Stage-1
    shingle encoding.

    Parameters
    ----------
    lens : str
        Lens name (used for metadata; encoding itself is lens-independent).
        Default ``"dedup"``.
    canon : str
        Canonicalization preset name.  Default ``"none"``.
    stage2_model : str or Stage2Model, optional
        A path to a ``.trine2`` file or a ``Stage2Model`` instance.
        When provided, Stage-2 encoding is used.  Default is None (Stage-1).
    depth : int
        Cascade depth for Stage-2 encoding (0 = projection only).
        Only used when ``stage2_model`` is provided.  Default 0.
    block_diagonal : bool
        If True, use block-diagonal projection mode (4 independent
        60x60 chain-local blocks instead of full 240x240 matrix).
        When ``stage2_model`` is also provided, its projection mode
        is set to block-diagonal.  When ``stage2_model`` is None,
        a random block-diagonal model is created.  Default False.
    adaptive_alpha : list of float, optional
        10-element list of per-S1-bucket alpha values for adaptive
        blending.  Each element corresponds to an S1 similarity
        bucket: [0.0-0.1), [0.1-0.2), ..., [0.9-1.0].
        Default is None (disabled).
    """

    # Pydantic model config for LlamaIndex BaseEmbedding
    model_name: str = "trine"

    def __init__(
        self,
        lens: str = "dedup",
        canon: str = "none",
        stage2_model: Optional[Any] = None,
        depth: int = 0,
        block_diagonal: bool = False,
        adaptive_alpha: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> None:
        _require_llamaindex()
        super().__init__(**kwargs)
        self._encoder = TrineEncoder()
        self._lens = _resolve_lens(lens)
        self._canon = _resolve_canon(canon)

        # Stage-2 support (with optional block-diagonal and adaptive alpha)
        self._s2_encoder = _build_s2_encoder(
            stage2_model, depth, block_diagonal, adaptive_alpha,
        )

    def _get_text_embedding(self, text: str) -> List[float]:
        """
        Encode a single document text into a float embedding vector.

        When a Stage-2 model is configured, uses Stage-2 encoding.

        Parameters
        ----------
        text : str
            The document text.

        Returns
        -------
        list[float]
            240-dimensional float vector.
        """
        if self._s2_encoder is not None:
            emb = self._s2_encoder.encode(text)
        else:
            emb = self._encoder.encode(text, canon=self._canon)
        return _embedding_to_floats(emb)

    def _get_query_embedding(self, query: str) -> List[float]:
        """
        Encode a query string into a float embedding vector.

        When a Stage-2 model is configured, uses Stage-2 encoding.

        Parameters
        ----------
        query : str
            The query text.

        Returns
        -------
        list[float]
            240-dimensional float vector.
        """
        if self._s2_encoder is not None:
            emb = self._s2_encoder.encode(query)
        else:
            emb = self._encoder.encode(query, canon=self._canon)
        return _embedding_to_floats(emb)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Encode multiple texts into float embedding vectors.

        When a Stage-2 model is configured, uses Stage-2 encoding.
        Otherwise uses TRINE's batch encoding API for efficiency.

        Parameters
        ----------
        texts : list of str
            Document texts.

        Returns
        -------
        list[list[float]]
            One 240-dimensional vector per text.
        """
        if self._s2_encoder is not None:
            embeddings = self._s2_encoder.encode_batch(texts)
        else:
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
