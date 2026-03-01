"""
pytrine — Python bindings for the TRINE ternary embedding library.

TRINE (Ternary Resonance Interference Network Embedding) is a fast,
deterministic text embedding system that maps text into 240-dimensional
ternary vectors using multi-scale n-gram shingling.

Classes:
    TrineEncoder   Stateless text-to-embedding encoder (~4M embeddings/sec)
    TrineIndex     Linear-scan dedup index (up to ~5K entries)
    TrineRouter    Band-LSH routed index (10K+ entries, sub-linear scaling)
    Embedding      240-trit embedding with comparison and packing
    Lens           4-weight comparison lens (predefined + custom)
    Canon          Canonicalization presets (NONE, SUPPORT, CODE, POLICY, GENERAL)
    Result         Query result (is_duplicate, similarity, tag, etc.)
    RouteStats     Routing statistics from router queries

Quick Start:
    >>> from pytrine import TrineEncoder, Lens
    >>> enc = TrineEncoder()
    >>> a = enc.encode("The quick brown fox")
    >>> b = enc.encode("The quick brown fox jumps")
    >>> a.similarity(b, lens=Lens.DEDUP)
    0.82
"""

__version__ = "1.0.1"

from pytrine.trine import (
    Canon,
    Embedding,
    Lens,
    Result,
    RouteStats,
    TrineEncoder,
    TrineIndex,
    TrineRouter,
)

__all__ = [
    "__version__",
    "Canon",
    "Embedding",
    "Lens",
    "Result",
    "RouteStats",
    "TrineEncoder",
    "TrineIndex",
    "TrineRouter",
]
