"""
pytrine.stage2 — Stage-2 semantic embedding layer for TRINE.

Stage-2 adds a learned ternary projection + cascade mixing network
on top of Stage-1 surface fingerprints.  The projection is trained
via Hebbian accumulation on text pairs, then frozen to ternary weights.

Classes:
    Stage2Model      Low-level wrapper around trine_s2_model_t*
    Stage2Encoder    High-level encode/compare API with blending
    HebbianTrainer   Hebbian training harness for learning projections

Constants:
    PROJ_SIGN        Sign-based projection mode (full 240x240 matmul)
    PROJ_DIAGONAL    Diagonal gating mode (per-channel keep/flip/zero)
    PROJ_BLOCK_DIAG  Block-diagonal mode (4 independent 60x60 chain-local)
"""

import ctypes
from collections import namedtuple

from pytrine._binding import (
    get_lib,
    TrineS1Lens,
    TrineS2Info,
    TrineS2SaveConfig,
    TrineHebbianConfig,
    TrineHebbianMetrics,
    Channels240,
)
from pytrine.trine import Embedding, Lens, TRINE_CHANNELS


# ── Constants ────────────────────────────────────────────────────────────

PROJ_SIGN = 0           # Sign-based projection (full matmul)
PROJ_DIAGONAL = 1       # Diagonal gating (per-channel)
PROJ_SPARSE = 2         # Sparse cross-channel (top-K per row)
PROJ_BLOCK_DIAG = 3     # Block-diagonal (4 independent 60x60 chain-local)


# ── Named tuples for introspection ──────────────────────────────────────

S2Info = namedtuple("S2Info", [
    "projection_k",
    "projection_dims",
    "cascade_cells",
    "max_depth",
    "is_identity",
])

S2Metrics = namedtuple("S2Metrics", [
    "pairs_observed",
    "max_abs_counter",
    "n_positive_weights",
    "n_negative_weights",
    "n_zero_weights",
    "weight_density",
    "effective_threshold",
])


# ── Stage2Model ─────────────────────────────────────────────────────────

class Stage2Model:
    """
    Low-level wrapper around a Stage-2 model (trine_s2_model_t*).

    Use the class methods to create or load a model:
        Stage2Model.identity()   -- pass-through (Stage-1 unchanged)
        Stage2Model.random(...)  -- random projection + cascade
        Stage2Model.load(path)   -- load from .trine2 file

    The model is immutable after construction and thread-safe for encoding.

    Examples
    --------
    >>> model = Stage2Model.identity()
    >>> emb = model.encode("Hello, world!")
    >>> model.close()
    """

    __slots__ = ("_ptr", "_lib")

    def __init__(self, _ptr):
        """
        Private constructor. Use identity(), random(), or load() instead.

        Parameters
        ----------
        _ptr : ctypes.c_void_p
            Pointer to a trine_s2_model_t.
        """
        if not _ptr:
            raise RuntimeError("Cannot create Stage2Model from NULL pointer")
        self._lib = get_lib()
        self._ptr = _ptr

    @classmethod
    def identity(cls):
        """
        Create an identity model (pass-through).

        The identity model returns Stage-1 encodings unchanged.
        This is the backward-compatibility contract.

        Returns
        -------
        Stage2Model
        """
        lib = get_lib()
        ptr = lib.trine_s2_create_identity()
        if not ptr:
            raise MemoryError("Failed to create identity Stage-2 model")
        return cls(ptr)

    @classmethod
    def random(cls, cells=512, seed=42):
        """
        Create a model with random projection and cascade topology.

        Parameters
        ----------
        cells : int
            Number of cascade mixing cells (default 512).
        seed : int
            PRNG seed for deterministic initialization.

        Returns
        -------
        Stage2Model
        """
        lib = get_lib()
        ptr = lib.trine_s2_create_random(cells, seed)
        if not ptr:
            raise MemoryError("Failed to create random Stage-2 model")
        return cls(ptr)

    @classmethod
    def load(cls, path):
        """
        Load a trained model from a .trine2 file.

        Parameters
        ----------
        path : str
            Path to the .trine2 file.

        Returns
        -------
        Stage2Model

        Raises
        ------
        IOError
            If the file cannot be loaded or is invalid.
        """
        lib = get_lib()
        ptr = lib.trine_s2_load(path.encode("utf-8"))
        if not ptr:
            raise IOError(f"Failed to load Stage-2 model from {path}")
        return cls(ptr)

    @classmethod
    def create_block_diagonal(cls, weights, K=3, n_cells=512, topo_seed=42):
        """
        Create a model with block-diagonal projection weights.

        Block-diagonal projection uses 4 independent 60x60 per-chain
        projections instead of a single 240x240 matrix, reducing
        parameter count while preserving chain-local structure.

        Parameters
        ----------
        weights : bytes or list of int
            Flat array of K * 4 * 60 * 60 ternary values (0/1/2).
            Layout: weights[k][chain][row][col] in row-major order.
        K : int
            Number of projection copies (default 3).
        n_cells : int
            Number of cascade mixing cells (default 512).
        topo_seed : int
            PRNG seed for deterministic cascade topology (default 42).

        Returns
        -------
        Stage2Model

        Raises
        ------
        ValueError
            If weights has wrong length (expected K * 4 * 60 * 60).
        MemoryError
            If allocation fails.
        """
        expected_len = K * 4 * 60 * 60
        if len(weights) != expected_len:
            raise ValueError(
                f"Expected {expected_len} weight bytes (K={K} * 4 * 60 * 60), "
                f"got {len(weights)}"
            )
        lib = get_lib()
        arr = (ctypes.c_uint8 * expected_len)(*weights)
        ptr = lib.trine_s2_create_block_diagonal(arr, K, n_cells, topo_seed)
        if not ptr:
            raise MemoryError("Failed to create block-diagonal Stage-2 model")
        return cls(ptr)

    def save(self, path, similarity_threshold=0.0, density=0.0, topo_seed=0):
        """
        Save the model to a .trine2 file.

        Parameters
        ----------
        path : str
            Output file path.
        similarity_threshold : float
            Training similarity threshold to embed in the header.
        density : float
            Freeze target density to embed in the header.
        topo_seed : int
            Topology PRNG seed to embed in the header.

        Raises
        ------
        IOError
            If the save fails.
        """
        self._check_alive()
        config = TrineS2SaveConfig()
        config.similarity_threshold = similarity_threshold
        config.density = density
        config.topo_seed = topo_seed
        rc = self._lib.trine_s2_save(self._ptr, path.encode("utf-8"), ctypes.byref(config))
        if rc != 0:
            raise IOError(f"Failed to save Stage-2 model to {path}")

    def encode(self, text, depth=0):
        """
        Encode text through the full Stage-2 pipeline.

        Pipeline: Stage-1 shingle encode -> projection -> cascade.

        Parameters
        ----------
        text : str
            Input text.
        depth : int
            Number of cascade ticks (0 = projection only).

        Returns
        -------
        Embedding
            The 240-trit Stage-2 embedding.
        """
        self._check_alive()
        text_bytes = text.encode("utf-8")
        out = Channels240()
        rc = self._lib.trine_s2_encode(self._ptr, text_bytes, len(text_bytes), depth, out)
        if rc != 0:
            raise RuntimeError("trine_s2_encode failed")
        return Embedding(out)

    def encode_from_trits(self, embedding, depth=0):
        """
        Apply Stage-2 projection + cascade to a pre-computed Stage-1 embedding.

        Parameters
        ----------
        embedding : Embedding
            A Stage-1 embedding (240 trits).
        depth : int
            Number of cascade ticks (0 = projection only).

        Returns
        -------
        Embedding
            The 240-trit Stage-2 embedding.
        """
        self._check_alive()
        if not isinstance(embedding, Embedding):
            raise TypeError(f"Expected Embedding, got {type(embedding)}")
        inp = embedding._to_ctypes_array()
        out = Channels240()
        rc = self._lib.trine_s2_encode_from_trits(self._ptr, inp, depth, out)
        if rc != 0:
            raise RuntimeError("trine_s2_encode_from_trits failed")
        return Embedding(out)

    def set_projection_mode(self, mode):
        """
        Set the projection mode.

        Parameters
        ----------
        mode : int
            PROJ_SIGN (0), PROJ_DIAGONAL (1), or PROJ_SPARSE (2).
        """
        self._check_alive()
        self._lib.trine_s2_set_projection_mode(self._ptr, mode)

    def get_projection_mode(self):
        """
        Get the current projection mode.

        Returns
        -------
        int
            PROJ_SIGN (0), PROJ_DIAGONAL (1), or PROJ_SPARSE (2).
        """
        self._check_alive()
        return self._lib.trine_s2_get_projection_mode(self._ptr)

    def set_stacked_depth(self, enable):
        """
        Enable or disable stacked depth.

        When enabled, each depth tick re-applies the learned projection
        instead of running the random cascade network.

        Parameters
        ----------
        enable : bool
            True to enable stacked depth, False to disable.
        """
        self._check_alive()
        self._lib.trine_s2_set_stacked_depth(self._ptr, 1 if enable else 0)

    def get_stacked_depth(self):
        """
        Check if stacked depth is enabled.

        Returns
        -------
        bool
            True if stacked depth is enabled.
        """
        self._check_alive()
        return bool(self._lib.trine_s2_get_stacked_depth(self._ptr))

    def compare_gated(self, emb_a, emb_b):
        """
        Gate-aware comparison using only channels with active diagonal gates.

        Channels where the majority of K=3 diagonal gates are zero
        (uninformative) are excluded from the cosine similarity.

        Parameters
        ----------
        emb_a : Embedding
            First Stage-2 embedding.
        emb_b : Embedding
            Second Stage-2 embedding.

        Returns
        -------
        float
            Similarity in [-1.0, 1.0], or 0.0 if no active channels.
        """
        self._check_alive()
        if not isinstance(emb_a, Embedding) or not isinstance(emb_b, Embedding):
            raise TypeError("Both arguments must be Embedding instances")
        a_arr = emb_a._to_ctypes_array()
        b_arr = emb_b._to_ctypes_array()
        return self._lib.trine_s2_compare_gated(self._ptr, a_arr, b_arr)

    def set_adaptive_alpha(self, buckets):
        """
        Set per-S1-bucket alpha values for adaptive blending.

        When set, compare_adaptive_blend() uses the bucket lookup
        to select alpha based on the S1 similarity range.

        Parameters
        ----------
        buckets : list of float or None
            10-element list of alpha values, one per S1 similarity
            bucket: [0.0-0.1), [0.1-0.2), ..., [0.9-1.0].
            Pass None to disable adaptive blending.
        """
        self._check_alive()
        if buckets is None:
            self._lib.trine_s2_set_adaptive_alpha(self._ptr, None)
        else:
            if len(buckets) != 10:
                raise ValueError(f"Expected 10 alpha buckets, got {len(buckets)}")
            arr = (ctypes.c_float * 10)(*buckets)
            self._lib.trine_s2_set_adaptive_alpha(self._ptr, arr)

    def compare_adaptive_blend(self, s1_a, s1_b, s2_a, s2_b):
        """
        Adaptive blend comparison: alpha selected by S1 similarity bucket.

        Computes S1 similarity, looks up alpha from the model's adaptive
        buckets, then blends: alpha * s1_sim + (1 - alpha) * s2_sim.

        Requires set_adaptive_alpha() to have been called first.

        Parameters
        ----------
        s1_a : Embedding
            First Stage-1 embedding.
        s1_b : Embedding
            Second Stage-1 embedding.
        s2_a : Embedding
            First Stage-2 embedding.
        s2_b : Embedding
            Second Stage-2 embedding.

        Returns
        -------
        float
            Blended similarity, or 0.0 if adaptive alpha is not set.
        """
        self._check_alive()
        for name, emb in [("s1_a", s1_a), ("s1_b", s1_b),
                           ("s2_a", s2_a), ("s2_b", s2_b)]:
            if not isinstance(emb, Embedding):
                raise TypeError(f"{name} must be an Embedding, got {type(emb)}")
        return self._lib.trine_s2_compare_adaptive_blend(
            self._ptr,
            s1_a._to_ctypes_array(), s1_b._to_ctypes_array(),
            s2_a._to_ctypes_array(), s2_b._to_ctypes_array(),
        )

    def get_block_projection(self):
        """
        Get the block-diagonal projection weights.

        Only valid for models in block-diagonal projection mode
        (PROJ_BLOCK_DIAG = 3).

        Returns
        -------
        list of int or None
            Flat list of K * 4 * 60 * 60 ternary values, or None
            if the model is not in block-diagonal mode.
        """
        self._check_alive()
        ptr = self._lib.trine_s2_get_block_projection(self._ptr)
        if not ptr:
            return None
        i = self.info
        total = i.projection_k * 4 * 60 * 60
        return [ptr[j] for j in range(total)]

    @property
    def info(self):
        """
        Introspect model parameters.

        Returns
        -------
        S2Info
            Named tuple with projection_k, projection_dims, cascade_cells,
            max_depth, and is_identity.
        """
        self._check_alive()
        c_info = TrineS2Info()
        rc = self._lib.trine_s2_info(self._ptr, ctypes.byref(c_info))
        if rc != 0:
            raise RuntimeError("trine_s2_info failed")
        return S2Info(
            projection_k=c_info.projection_k,
            projection_dims=c_info.projection_dims,
            cascade_cells=c_info.cascade_cells,
            max_depth=c_info.max_depth,
            is_identity=bool(c_info.is_identity),
        )

    def _check_alive(self):
        """Raise if the model has been freed."""
        if self._ptr is None:
            raise RuntimeError("Stage2Model has been closed/freed")

    def close(self):
        """Free the underlying C model."""
        if self._ptr is not None:
            self._lib.trine_s2_free(self._ptr)
            self._ptr = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self):
        if self._ptr is None:
            return "Stage2Model(closed)"
        try:
            i = self.info
            if i.is_identity:
                return "Stage2Model(identity)"
            return (
                f"Stage2Model(k={i.projection_k}, dim={i.projection_dims}, "
                f"cells={i.cascade_cells})"
            )
        except Exception:
            return "Stage2Model(?)"


# ── Module-level comparison function ────────────────────────────────────

def s2_compare(a, b, lens=None):
    """
    Compare two Stage-2 embeddings using the Stage-1 lens system.

    This is a free function (does not require a model) because
    trine_s2_compare operates directly on trit arrays.

    Parameters
    ----------
    a : Embedding
        First embedding (240 trits).
    b : Embedding
        Second embedding (240 trits).
    lens : Lens, optional
        Comparison lens. None means uniform weights.

    Returns
    -------
    float
        Similarity in [0.0, 1.0], or -1.0 on error.
    """
    if not isinstance(a, Embedding) or not isinstance(b, Embedding):
        raise TypeError("Both arguments must be Embedding instances")

    lib = get_lib()
    a_arr = a._to_ctypes_array()
    b_arr = b._to_ctypes_array()

    if lens is not None:
        c_lens = lens._to_ctypes()
        lens_ptr = ctypes.cast(ctypes.byref(c_lens), ctypes.c_void_p)
    else:
        lens_ptr = None

    return lib.trine_s2_compare(a_arr, b_arr, lens_ptr)


def s2_validate(path):
    """
    Validate a .trine2 file without loading it.

    Parameters
    ----------
    path : str
        Path to the .trine2 file.

    Returns
    -------
    bool
        True if valid, False otherwise.
    """
    lib = get_lib()
    rc = lib.trine_s2_validate(path.encode("utf-8"))
    return rc == 0


# ── Stage2Encoder ───────────────────────────────────────────────────────

class Stage2Encoder:
    """
    High-level Stage-2 encoding API with blending support.

    Wraps a Stage2Model and provides convenient encode/compare/blend
    methods.  If no model is provided, uses an identity model (which
    produces Stage-1 encodings unchanged).

    Parameters
    ----------
    model : Stage2Model or str, optional
        A Stage2Model instance, a path to a .trine2 file, or None
        for an identity model.
    depth : int
        Default cascade depth for encoding (0 = projection only).

    Examples
    --------
    >>> enc = Stage2Encoder()
    >>> emb = enc.encode("Hello, world!")
    >>> sim = enc.similarity("Hello", "Hi there")
    >>> blended = enc.blend("Hello", "Hi there", alpha=0.65)
    """

    def __init__(self, model=None, depth=0):
        self._depth = depth
        self._owns_model = False

        if model is None:
            self._model = Stage2Model.identity()
            self._owns_model = True
        elif isinstance(model, str):
            self._model = Stage2Model.load(model)
            self._owns_model = True
        elif isinstance(model, Stage2Model):
            self._model = model
            self._owns_model = False
        else:
            raise TypeError(f"Expected Stage2Model, str, or None, got {type(model)}")

    def encode(self, text):
        """
        Encode a single text through the Stage-2 pipeline.

        Parameters
        ----------
        text : str
            Input text.

        Returns
        -------
        Embedding
            The 240-trit Stage-2 embedding.
        """
        return self._model.encode(text, depth=self._depth)

    def encode_batch(self, texts):
        """
        Encode multiple texts through the Stage-2 pipeline.

        Parameters
        ----------
        texts : list of str
            Input texts.

        Returns
        -------
        list of Embedding
            One Stage-2 embedding per input text.
        """
        return [self._model.encode(t, depth=self._depth) for t in texts]

    def similarity(self, text_a, text_b, lens=None):
        """
        Compute Stage-2 similarity between two texts.

        Parameters
        ----------
        text_a : str
            First text.
        text_b : str
            Second text.
        lens : Lens, optional
            Comparison lens (None = uniform).

        Returns
        -------
        float
            Similarity in [0.0, 1.0].
        """
        emb_a = self.encode(text_a)
        emb_b = self.encode(text_b)
        return s2_compare(emb_a, emb_b, lens=lens)

    def blend(self, text_a, text_b, alpha=0.65, lens=None):
        """
        Compute blended similarity: alpha * S1 + (1 - alpha) * S2.

        This is the primary evaluation metric for Stage-2. The default
        alpha=0.65 was determined by grid search on the validation set.

        Parameters
        ----------
        text_a : str
            First text.
        text_b : str
            Second text.
        alpha : float
            Blending weight (0.0 = pure S2, 1.0 = pure S1).
        lens : Lens, optional
            Comparison lens (None = uniform).

        Returns
        -------
        float
            Blended similarity score.
        """
        from pytrine.trine import TrineEncoder

        # Stage-1 similarity
        s1_enc = TrineEncoder()
        s1_a = s1_enc.encode(text_a)
        s1_b = s1_enc.encode(text_b)
        s1_sim = s1_a.similarity(s1_b, lens=lens)

        # Stage-2 similarity
        s2_a = self.encode(text_a)
        s2_b = self.encode(text_b)
        s2_sim = s2_compare(s2_a, s2_b, lens=lens)

        return alpha * s1_sim + (1.0 - alpha) * s2_sim

    def close(self):
        """Free the underlying model if this encoder owns it."""
        if self._owns_model and self._model is not None:
            self._model.close()
            self._model = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self):
        if self._model is None:
            return "Stage2Encoder(closed)"
        return f"Stage2Encoder(depth={self._depth}, model={self._model!r})"


# ── HebbianTrainer ──────────────────────────────────────────────────────

class HebbianTrainer:
    """
    Hebbian training harness for learning Stage-2 projections.

    The training signal comes from Stage-1 cosine similarity: pairs
    with s1 > threshold are "similar" (positive Hebbian update) and
    pairs with s1 <= threshold are "dissimilar" (negative update).

    After training, call freeze() to get a Stage2Model.

    Parameters
    ----------
    config : dict, optional
        Configuration overrides. Supported keys:
            similarity_threshold (float): s1 > thresh = similar. Default 0.5.
            freeze_threshold (int): Quantization threshold. 0 = auto.
            freeze_target_density (float): Target density for auto-T. Default 0.33.
            cascade_cells (int): Cascade mixing cells. Default 512.
            cascade_depth (int): Default inference depth. Default 4.
            projection_mode (int): 0=sign, 1=diagonal, 2=sparse. Default 0.
            weighted_mode (int): 0=binary sign, 1=weighted magnitude. Default 0.
            pos_scale (float): Positive magnitude scale. Default 10.0.
            neg_scale (float): Negative magnitude scale. Default 3.0.
            sparse_k (int): Sparse top-K per row (0=disabled). Default 0.
            block_diagonal (int): 1=block-diagonal mode. Default 0.
            rng_seed (int): RNG seed for downsampling. Default 0.
    block_diagonal : bool
        If True, enable block-diagonal projection mode (4 independent
        60x60 per-chain projections). Default False.

    Examples
    --------
    >>> trainer = HebbianTrainer({"similarity_threshold": 0.90, "projection_mode": 1})
    >>> trainer.train_file("data/train.jsonl", epochs=1)
    >>> model = trainer.freeze()
    >>> model.save("trained.trine2")
    >>> trainer.close()

    Block-diagonal training:

    >>> trainer = HebbianTrainer.create_block_diagonal(similarity_threshold=0.90)
    >>> trainer.train_file("data/train.jsonl", epochs=1)
    >>> model = trainer.freeze()
    """

    __slots__ = ("_ptr", "_lib")

    def __init__(self, config=None, block_diagonal=False):
        self._lib = get_lib()

        c_config = TrineHebbianConfig()
        # Set defaults matching TRINE_HEBBIAN_CONFIG_DEFAULT
        c_config.similarity_threshold = 0.5
        c_config.freeze_threshold = 0
        c_config.freeze_target_density = 0.33
        c_config.cascade_cells = 512
        c_config.cascade_depth = 4
        c_config.projection_mode = 0
        c_config.weighted_mode = 0
        c_config.pos_scale = 10.0
        c_config.neg_scale = 3.0
        c_config.n_source_weights = 0
        c_config.sparse_k = 0
        c_config.block_diagonal = 1 if block_diagonal else 0
        c_config.rng_seed = 0

        if config is not None:
            for key in ("similarity_threshold", "freeze_threshold",
                        "freeze_target_density", "cascade_cells",
                        "cascade_depth", "projection_mode",
                        "weighted_mode", "pos_scale", "neg_scale",
                        "sparse_k", "block_diagonal", "rng_seed"):
                if key in config:
                    setattr(c_config, key, config[key])

        self._ptr = self._lib.trine_hebbian_create(ctypes.byref(c_config))
        if not self._ptr:
            raise MemoryError("Failed to create Hebbian training state")

    @classmethod
    def create_block_diagonal(cls, similarity_threshold=0.5,
                               freeze_target_density=0.33,
                               cascade_cells=512, cascade_depth=4,
                               weighted_mode=0, pos_scale=10.0,
                               neg_scale=3.0):
        """
        Factory method for a block-diagonal Hebbian trainer.

        Creates a trainer configured for block-diagonal projection
        (4 independent 60x60 per-chain projections).

        Parameters
        ----------
        similarity_threshold : float
            S1 > threshold = similar pair. Default 0.5.
        freeze_target_density : float
            Target density for auto-threshold. Default 0.33.
        cascade_cells : int
            Number of cascade mixing cells. Default 512.
        cascade_depth : int
            Default inference depth. Default 4.
        weighted_mode : int
            0=binary sign, 1=weighted magnitude. Default 0.
        pos_scale : float
            Positive magnitude scale. Default 10.0.
        neg_scale : float
            Negative magnitude scale. Default 3.0.

        Returns
        -------
        HebbianTrainer
        """
        config = {
            "similarity_threshold": similarity_threshold,
            "freeze_target_density": freeze_target_density,
            "cascade_cells": cascade_cells,
            "cascade_depth": cascade_depth,
            "weighted_mode": weighted_mode,
            "pos_scale": pos_scale,
            "neg_scale": neg_scale,
        }
        return cls(config=config, block_diagonal=True)

    def observe(self, emb_a, emb_b, similarity):
        """
        Observe a training pair with pre-computed embeddings.

        Parameters
        ----------
        emb_a : Embedding
            First Stage-1 embedding (240 trits).
        emb_b : Embedding
            Second Stage-1 embedding (240 trits).
        similarity : float
            Stage-1 cosine similarity between the pair.
        """
        self._check_alive()
        if not isinstance(emb_a, Embedding) or not isinstance(emb_b, Embedding):
            raise TypeError("Both embeddings must be Embedding instances")
        a_arr = emb_a._to_ctypes_array()
        b_arr = emb_b._to_ctypes_array()
        self._lib.trine_hebbian_observe(self._ptr, a_arr, b_arr, similarity)

    def observe_text(self, text_a, text_b):
        """
        Observe a training pair from raw text.

        The C function handles Stage-1 encoding and similarity computation.

        Parameters
        ----------
        text_a : str
            First text.
        text_b : str
            Second text.
        """
        self._check_alive()
        a_bytes = text_a.encode("utf-8")
        b_bytes = text_b.encode("utf-8")
        self._lib.trine_hebbian_observe_text(
            self._ptr, a_bytes, len(a_bytes), b_bytes, len(b_bytes)
        )

    def train_file(self, path, epochs=1):
        """
        Train from a JSONL file of text pairs.

        Each line should have text_a, text_b, score, and label fields.

        Parameters
        ----------
        path : str
            Path to the JSONL training file.
        epochs : int
            Number of passes over the file (default 1).

        Returns
        -------
        int
            Number of pairs processed, or -1 on error.
        """
        self._check_alive()
        result = self._lib.trine_hebbian_train_file(
            self._ptr, path.encode("utf-8"), epochs
        )
        if result < 0:
            raise RuntimeError(f"trine_hebbian_train_file failed (path={path})")
        return result

    def freeze(self):
        """
        Freeze current accumulators into a Stage-2 model.

        Quantizes the accumulated Hebbian weights into ternary values
        and constructs a Stage2Model ready for inference.

        Returns
        -------
        Stage2Model
            The frozen model. Caller owns it.

        Raises
        ------
        RuntimeError
            If freeze fails (e.g., no pairs observed).
        """
        self._check_alive()
        ptr = self._lib.trine_hebbian_freeze(self._ptr)
        if not ptr:
            raise RuntimeError("trine_hebbian_freeze failed")
        return Stage2Model(ptr)

    @property
    def metrics(self):
        """
        Get current training metrics.

        Returns
        -------
        S2Metrics
            Named tuple with pairs_observed, max_abs_counter,
            n_positive_weights, n_negative_weights, n_zero_weights,
            weight_density, and effective_threshold.
        """
        self._check_alive()
        c_metrics = TrineHebbianMetrics()
        rc = self._lib.trine_hebbian_metrics(self._ptr, ctypes.byref(c_metrics))
        if rc != 0:
            raise RuntimeError("trine_hebbian_metrics failed")
        return S2Metrics(
            pairs_observed=c_metrics.pairs_observed,
            max_abs_counter=c_metrics.max_abs_counter,
            n_positive_weights=c_metrics.n_positive_weights,
            n_negative_weights=c_metrics.n_negative_weights,
            n_zero_weights=c_metrics.n_zero_weights,
            weight_density=c_metrics.weight_density,
            effective_threshold=c_metrics.effective_threshold,
        )

    def reset(self):
        """
        Reset accumulators (start fresh, keep config).
        """
        self._check_alive()
        self._lib.trine_hebbian_reset(self._ptr)

    def _check_alive(self):
        """Raise if the trainer has been freed."""
        if self._ptr is None:
            raise RuntimeError("HebbianTrainer has been closed/freed")

    def close(self):
        """Free the underlying C training state."""
        if self._ptr is not None:
            self._lib.trine_hebbian_free(self._ptr)
            self._ptr = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self):
        if self._ptr is None:
            return "HebbianTrainer(closed)"
        try:
            m = self.metrics
            return f"HebbianTrainer(pairs={m.pairs_observed}, density={m.weight_density:.3f})"
        except Exception:
            return "HebbianTrainer(?)"
