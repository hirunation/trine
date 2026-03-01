"""
pytrine.langchain -- LangChain retriever integration for TRINE.

Provides TrineRetriever (BaseRetriever) and TrineEmbeddings (Embeddings)
that use TRINE's fast ternary embedding for document retrieval and
embedding generation.

TRINE's brute-force comparison runs at ~5.8M compares/sec, making it
practical for corpora up to tens of thousands of documents without
any approximate indexing overhead.

Usage::

    from pytrine.langchain import TrineRetriever

    retriever = TrineRetriever.from_texts(
        texts=["doc 1", "doc 2", "doc 3"],
        metadatas=[{"source": "a"}, {"source": "b"}, {"source": "c"}],
        threshold=0.60,
        lens="dedup",
        canon="none",
    )
    docs = retriever.invoke("query text")

    # Or with the embeddings interface:
    from pytrine.langchain import TrineEmbeddings

    embeddings = TrineEmbeddings(lens="code", canon="none")
    vectors = embeddings.embed_documents(["text1", "text2"])
    query_vec = embeddings.embed_query("query text")

Requires:
    pip install langchain-core
"""

import json
import os
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pytrine import (
    Canon,
    Embedding,
    Lens,
    TrineEncoder,
    TrineRouter,
)
from pytrine.trine import TRINE_CHANNELS

# ── LangChain import guard ──────────────────────────────────────────────

try:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings as LCEmbeddings
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.callbacks.manager import (
        CallbackManagerForRetrieverRun,
    )

    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False

    _LANGCHAIN_IMPORT_MSG = (
        "langchain-core is required for pytrine.langchain.\n"
        "Install it with:  pip install langchain-core\n"
        "Or:               pip install pytrine[langchain]"
    )

    # Stubs so the module can be imported for inspection without langchain,
    # but instantiation raises a clear error.  We override __new__ so the
    # ImportError fires before Python checks __init__ argument counts.
    class _LangChainStub:
        """Stub base that raises ImportError on instantiation."""

        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(**kwargs)

        def __new__(cls, *args, **kwargs):
            raise ImportError(_LANGCHAIN_IMPORT_MSG)

    BaseRetriever = _LangChainStub  # type: ignore[misc,assignment]
    LCEmbeddings = _LangChainStub  # type: ignore[misc,assignment]
    Document = None  # type: ignore[misc,assignment]
    CallbackManagerForRetrieverRun = None  # type: ignore[misc,assignment]


# ── Name-to-object maps ─────────────────────────────────────────────────

_LENS_MAP: Dict[str, Lens] = {
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

_CANON_MAP: Dict[str, int] = {
    "none": Canon.NONE,
    "support": Canon.SUPPORT,
    "code": Canon.CODE,
    "policy": Canon.POLICY,
    "general": Canon.GENERAL,
}


def _resolve_lens(lens: Any) -> Lens:
    """Resolve a lens name (str) or Lens object to a Lens instance."""
    if isinstance(lens, Lens):
        return lens
    if isinstance(lens, str):
        key = lens.lower()
        if key not in _LENS_MAP:
            raise ValueError(
                f"Unknown lens name: {lens!r}. "
                f"Available: {', '.join(sorted(_LENS_MAP))}"
            )
        return _LENS_MAP[key]
    raise TypeError(f"Expected str or Lens, got {type(lens)}")


def _resolve_canon(canon: Any) -> int:
    """Resolve a canon name (str) or int to a Canon preset int."""
    if isinstance(canon, int):
        return canon
    if isinstance(canon, str):
        key = canon.lower()
        if key not in _CANON_MAP:
            raise ValueError(
                f"Unknown canon name: {canon!r}. "
                f"Available: {', '.join(sorted(_CANON_MAP))}"
            )
        return _CANON_MAP[key]
    raise TypeError(f"Expected str or int, got {type(canon)}")


# ── TrineEmbeddings ─────────────────────────────────────────────────────

class TrineEmbeddings(LCEmbeddings):
    """
    LangChain Embeddings interface backed by TRINE ternary encoding.

    Produces 240-dimensional vectors where each dimension is a trit
    (0, 1, or 2). These are returned as ``list[float]`` per the
    LangChain Embeddings contract.

    Parameters
    ----------
    lens : str or Lens
        Lens preset name or Lens instance (default ``"dedup"``).
    canon : str or int
        Canonicalization preset name or Canon int (default ``"none"``).
    """

    def __init__(self, lens: str = "dedup", canon: str = "none", **kwargs: Any):
        # Do NOT call super().__init__() for the stub path
        if not _HAS_LANGCHAIN:
            raise ImportError(
                "langchain-core is required for pytrine.langchain.\n"
                "Install it with:  pip install langchain-core\n"
                "Or:               pip install pytrine[langchain]"
            )
        self._lens = _resolve_lens(lens)
        self._canon = _resolve_canon(canon)
        self._encoder = TrineEncoder()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Encode a list of document texts into float vectors.

        Parameters
        ----------
        texts : list of str
            Document texts to embed.

        Returns
        -------
        list of list of float
            One 240-element vector per text, values in {0.0, 1.0, 2.0}.
        """
        embeddings = self._encoder.encode_batch(texts, canon=self._canon)
        return [list(float(t) for t in emb.trits) for emb in embeddings]

    def embed_query(self, text: str) -> List[float]:
        """
        Encode a single query text into a float vector.

        Parameters
        ----------
        text : str
            Query text.

        Returns
        -------
        list of float
            240-element vector, values in {0.0, 1.0, 2.0}.
        """
        emb = self._encoder.encode(text, canon=self._canon)
        return [float(t) for t in emb.trits]


# ── TrineRetriever ──────────────────────────────────────────────────────

class TrineRetriever(BaseRetriever):
    """
    LangChain retriever backed by TRINE ternary embeddings.

    Uses brute-force lens-weighted comparison against all stored
    embeddings for top-k retrieval. TRINE's ~5.8M compares/sec
    makes this practical for corpora up to tens of thousands of
    documents.

    The underlying TrineRouter is maintained alongside the embedding
    store for dedup checks and optional routing-accelerated queries.

    Parameters
    ----------
    router : TrineRouter
        The underlying TRINE router (for dedup and tag storage).
    documents : list of Document
        Stored LangChain Document objects, parallel to embeddings.
    embeddings : list of Embedding
        Stored TRINE embeddings, parallel to documents.
    threshold : float
        Similarity threshold for the router's dedup detection.
    lens : str or Lens
        Lens preset name or Lens instance for similarity computation.
    canon : str or int
        Canonicalization preset for encoding.
    k : int
        Maximum number of results to return.
    score_threshold : float
        Minimum similarity score to include in results.

    Notes
    -----
    Mutations (``add_texts``) are NOT thread-safe. Concurrent reads
    (``invoke`` / ``similarity_search``) on a stable index are safe
    because the underlying comparisons are stateless.

    Examples
    --------
    >>> retriever = TrineRetriever.from_texts(
    ...     texts=["The cat sat on the mat", "The dog ran in the park"],
    ...     metadatas=[{"source": "cats"}, {"source": "dogs"}],
    ...     threshold=0.60,
    ...     lens="dedup",
    ... )
    >>> docs = retriever.invoke("cat on a mat")
    >>> docs[0].metadata["source"]
    'cats'
    """

    # ---- Pydantic v2 config for arbitrary types ---
    class Config:
        arbitrary_types_allowed = True

    # ---- Instance state (set in __init__, not as class-level Fields) ----

    def __init__(
        self,
        router: TrineRouter,
        documents: List,  # List[Document]
        embeddings: List[Embedding],
        threshold: float = 0.60,
        lens: Any = "dedup",
        canon: Any = "none",
        k: int = 5,
        score_threshold: float = 0.0,
        **kwargs: Any,
    ):
        if not _HAS_LANGCHAIN:
            raise ImportError(
                "langchain-core is required for pytrine.langchain.\n"
                "Install it with:  pip install langchain-core\n"
                "Or:               pip install pytrine[langchain]"
            )
        super().__init__(**kwargs)
        self._router = router
        self._documents = list(documents)
        self._embeddings = list(embeddings)
        self._threshold = float(threshold)
        self._lens = _resolve_lens(lens)
        self._canon = _resolve_canon(canon)
        self._k = int(k)
        self._score_threshold = float(score_threshold)
        self._encoder = TrineEncoder()

    # ---- BaseRetriever protocol ----

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Any = None,
    ) -> List:
        """
        Retrieve documents relevant to the query.

        Encodes the query with TRINE, compares against all stored
        embeddings using the configured lens, and returns the top-k
        documents above the score threshold.

        Parameters
        ----------
        query : str
            Query text.
        run_manager : CallbackManagerForRetrieverRun, optional
            LangChain callback manager (unused).

        Returns
        -------
        list of Document
            Up to ``k`` documents, each with ``metadata["trine_score"]``
            set to the similarity score.
        """
        scored = self.similarity_search(query)
        return [doc for doc, _score in scored]

    # ---- Public API ----

    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> List[Tuple[Any, float]]:
        """
        Search for documents similar to the query, returning scores.

        Parameters
        ----------
        query : str
            Query text.
        k : int, optional
            Override the default ``k`` for this call.

        Returns
        -------
        list of (Document, float)
            Tuples of (document, similarity_score), sorted by descending
            similarity, filtered by ``score_threshold``, limited to ``k``.
        """
        if k is None:
            k = self._k

        query_emb = self._encoder.encode(query, canon=self._canon)

        # Brute-force comparison against all stored embeddings
        scored: List[Tuple[int, float]] = []
        for i, stored_emb in enumerate(self._embeddings):
            sim = query_emb.similarity(stored_emb, lens=self._lens)
            if sim >= self._score_threshold:
                scored.append((i, sim))

        # Sort by descending similarity, take top-k
        scored.sort(key=lambda x: x[1], reverse=True)
        scored = scored[:k]

        # Build result documents with score metadata
        results = []
        for idx, sim in scored:
            doc = self._documents[idx]
            # Copy document to avoid mutating the stored one
            result_doc = Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "trine_score": sim},
            )
            results.append((result_doc, sim))

        return results

    def add_texts(
        self,
        texts: Sequence[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """
        Add texts to the retriever.

        Parameters
        ----------
        texts : sequence of str
            Texts to add.
        metadatas : list of dict, optional
            Per-document metadata. Length must match ``texts`` if given.

        Returns
        -------
        list of str
            IDs assigned to the added documents (UUIDs).
        """
        texts = list(texts)
        if metadatas is None:
            metadatas = [{} for _ in texts]
        elif len(metadatas) != len(texts):
            raise ValueError(
                f"metadatas length ({len(metadatas)}) must match "
                f"texts length ({len(texts)})"
            )

        ids = []
        for text, meta in zip(texts, metadatas):
            doc_id = str(uuid.uuid4())
            emb = self._encoder.encode(text, canon=self._canon)
            tag = meta.get("tag", doc_id)

            self._router.add_embedding(emb, tag=str(tag))

            doc = Document(
                page_content=text,
                metadata={**meta, "trine_id": doc_id},
            )
            self._documents.append(doc)
            self._embeddings.append(emb)
            ids.append(doc_id)

        return ids

    # ---- Serialization ----

    def save_index(self, path: str) -> None:
        """
        Save the retriever state to disk.

        Creates two files:
        - ``{path}.trrt`` -- the TRINE router binary index
        - ``{path}.meta.json`` -- document metadata and page content

        Parameters
        ----------
        path : str
            Base path (without extension).
        """
        router_path = path + ".trrt"
        meta_path = path + ".meta.json"

        self._router.save(router_path)

        # Serialize documents and embeddings
        meta = {
            "threshold": self._threshold,
            "lens": self._lens_name(),
            "canon": self._canon_name(),
            "k": self._k,
            "score_threshold": self._score_threshold,
            "documents": [
                {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                }
                for doc in self._documents
            ],
            "embeddings_packed": [
                list(emb.packed) for emb in self._embeddings
            ],
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_index(cls, path: str) -> "TrineRetriever":
        """
        Load a retriever from disk.

        Parameters
        ----------
        path : str
            Base path (same as passed to ``save_index``).

        Returns
        -------
        TrineRetriever
            The restored retriever.

        Raises
        ------
        FileNotFoundError
            If the index files do not exist.
        ImportError
            If langchain-core is not installed.
        """
        if not _HAS_LANGCHAIN:
            raise ImportError(
                "langchain-core is required for pytrine.langchain.\n"
                "Install it with:  pip install langchain-core\n"
                "Or:               pip install pytrine[langchain]"
            )

        router_path = path + ".trrt"
        meta_path = path + ".meta.json"

        if not os.path.isfile(router_path):
            raise FileNotFoundError(f"Router file not found: {router_path}")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        router = TrineRouter.load(router_path)

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        documents = [
            Document(
                page_content=d["page_content"],
                metadata=d["metadata"],
            )
            for d in meta["documents"]
        ]

        embeddings = [
            Embedding.from_packed(bytes(packed))
            for packed in meta["embeddings_packed"]
        ]

        return cls(
            router=router,
            documents=documents,
            embeddings=embeddings,
            threshold=meta.get("threshold", 0.60),
            lens=meta.get("lens", "dedup"),
            canon=meta.get("canon", "none"),
            k=meta.get("k", 5),
            score_threshold=meta.get("score_threshold", 0.0),
        )

    # ---- Class constructors ----

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        threshold: float = 0.60,
        lens: Any = "dedup",
        canon: Any = "none",
        k: int = 5,
        score_threshold: float = 0.0,
        **kwargs: Any,
    ) -> "TrineRetriever":
        """
        Create a retriever from a list of texts.

        Parameters
        ----------
        texts : list of str
            Document texts.
        metadatas : list of dict, optional
            Per-document metadata.
        threshold : float
            Similarity threshold for dedup detection.
        lens : str or Lens
            Lens preset name or Lens instance.
        canon : str or int
            Canonicalization preset.
        k : int
            Max results to return.
        score_threshold : float
            Minimum similarity to include in results.

        Returns
        -------
        TrineRetriever
        """
        if not _HAS_LANGCHAIN:
            raise ImportError(
                "langchain-core is required for pytrine.langchain.\n"
                "Install it with:  pip install langchain-core\n"
                "Or:               pip install pytrine[langchain]"
            )

        resolved_lens = _resolve_lens(lens)
        resolved_canon = _resolve_canon(canon)
        encoder = TrineEncoder()

        router = TrineRouter(
            threshold=threshold,
            lens=resolved_lens,
            calibrate=True,
        )

        if metadatas is None:
            metadatas = [{} for _ in texts]
        elif len(metadatas) != len(texts):
            raise ValueError(
                f"metadatas length ({len(metadatas)}) must match "
                f"texts length ({len(texts)})"
            )

        documents = []
        embeddings = []
        for i, (text, meta) in enumerate(zip(texts, metadatas)):
            doc_id = str(uuid.uuid4())
            emb = encoder.encode(text, canon=resolved_canon)
            tag = meta.get("tag", doc_id)

            router.add_embedding(emb, tag=str(tag))

            doc = Document(
                page_content=text,
                metadata={**meta, "trine_id": doc_id},
            )
            documents.append(doc)
            embeddings.append(emb)

        return cls(
            router=router,
            documents=documents,
            embeddings=embeddings,
            threshold=threshold,
            lens=lens,
            canon=canon,
            k=k,
            score_threshold=score_threshold,
            **kwargs,
        )

    @classmethod
    def from_documents(
        cls,
        documents: List,  # List[Document]
        threshold: float = 0.60,
        lens: Any = "dedup",
        canon: Any = "none",
        k: int = 5,
        score_threshold: float = 0.0,
        **kwargs: Any,
    ) -> "TrineRetriever":
        """
        Create a retriever from a list of LangChain Document objects.

        Parameters
        ----------
        documents : list of Document
            LangChain documents with ``page_content`` and ``metadata``.
        threshold : float
            Similarity threshold.
        lens : str or Lens
            Lens preset.
        canon : str or int
            Canonicalization preset.
        k : int
            Max results.
        score_threshold : float
            Minimum similarity.

        Returns
        -------
        TrineRetriever
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return cls.from_texts(
            texts=texts,
            metadatas=metadatas,
            threshold=threshold,
            lens=lens,
            canon=canon,
            k=k,
            score_threshold=score_threshold,
            **kwargs,
        )

    # ---- Internal helpers ----

    def _lens_name(self) -> str:
        """Return the string name for the current lens."""
        for name, preset in _LENS_MAP.items():
            if preset == self._lens:
                return name
        # Custom lens -- serialize weights
        w = self._lens.weights
        return f"custom({w[0]},{w[1]},{w[2]},{w[3]})"

    def _canon_name(self) -> str:
        """Return the string name for the current canon."""
        for name, preset in _CANON_MAP.items():
            if preset == self._canon:
                return name
        return str(self._canon)

    def __repr__(self) -> str:
        n = len(self._documents)
        return (
            f"TrineRetriever(n={n}, k={self._k}, "
            f"lens={self._lens_name()!r}, "
            f"threshold={self._threshold:.2f})"
        )

    def __len__(self) -> int:
        return len(self._documents)


# ── Module-level exports ─────────────────────────────────────────────────

__all__ = [
    "TrineRetriever",
    "TrineEmbeddings",
]
