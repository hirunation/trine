"""
pytrine — Python bindings for the TRINE ternary embedding library.

TRINE (Ternary Resonance Interference Network Embedding) is a fast,
deterministic text embedding system that maps text into 240-dimensional
ternary vectors using multi-scale n-gram shingling.

Stage-1 Classes:
    TrineEncoder   Stateless text-to-embedding encoder (~4M embeddings/sec)
    TrineIndex     Linear-scan dedup index (up to ~5K entries)
    TrineRouter    Band-LSH routed index (10K+ entries, sub-linear scaling)
    Embedding      240-trit embedding with comparison and packing
    Lens           4-weight comparison lens (predefined + custom)
    Canon          Canonicalization presets (NONE, SUPPORT, CODE, POLICY, GENERAL)
    Result         Query result (is_duplicate, similarity, tag, etc.)
    RouteStats     Routing statistics from router queries

Stage-2 Classes:
    Stage2Model    Low-level Stage-2 model wrapper (identity, random, or trained)
    Stage2Encoder  High-level Stage-2 encode/compare/blend API
    HebbianTrainer Hebbian training harness for learning projections

Quick Start:
    >>> from pytrine import TrineEncoder, Lens
    >>> enc = TrineEncoder()
    >>> a = enc.encode("The quick brown fox")
    >>> b = enc.encode("The quick brown fox jumps")
    >>> a.similarity(b, lens=Lens.DEDUP)
    0.82

Stage-2 Quick Start:
    >>> from pytrine import Stage2Model, Stage2Encoder
    >>> model = Stage2Model.load("trained.trine2")
    >>> enc = Stage2Encoder(model, depth=0)
    >>> sim = enc.similarity("Hello", "Hi there")
"""

__version__ = "1.0.2"

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

from pytrine.stage2 import (
    Stage2Model,
    Stage2Encoder,
    HebbianTrainer,
)

__all__ = [
    "__version__",
    # Stage-1
    "Canon",
    "Embedding",
    "Lens",
    "Result",
    "RouteStats",
    "TrineEncoder",
    "TrineIndex",
    "TrineRouter",
    # Stage-2
    "Stage2Model",
    "Stage2Encoder",
    "HebbianTrainer",
]
