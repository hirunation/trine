"""
pytrine.trine — High-level Pythonic API for the TRINE embedding library.

Classes:
    Embedding   — 240-trit ternary embedding with comparison and packing
    Lens        — Weighted comparison lens (4 chain weights)
    Canon       — Canonicalization presets
    TrineEncoder — Text-to-embedding encoder
    TrineIndex  — Linear-scan dedup index
    TrineRouter — Band-LSH routed index for large corpora
    Result      — Query result from index/router operations
    RouteStats  — Routing statistics from router queries
"""

import ctypes
import os

from pytrine._binding import (
    get_lib,
    TrineS1Lens,
    TrineS1Config,
    TrineS1Result,
    TrineRouteStats,
)

# Optional numpy support
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


# ── Constants ────────────────────────────────────────────────────────────

TRINE_CHANNELS = 240
TRINE_PACKED_SIZE = 48
TRINE_CHAINS = 4
TRINE_CHAIN_WIDTH = 60


# ── Lens ─────────────────────────────────────────────────────────────────

class Lens:
    """
    Weighted comparison lens for TRINE embeddings.

    Each lens assigns weights to the 4 encoding chains:
        [0] Character unigrams + bigrams (edit-level)
        [1] Character trigrams (morpheme-level)
        [2] Character 5-grams (phrase-level)
        [3] Word unigrams (vocabulary-level)

    Predefined presets are available as class attributes.

    Parameters
    ----------
    edit : float
        Weight for chain 0 (character unigrams/bigrams).
    morph : float
        Weight for chain 1 (character trigrams).
    phrase : float
        Weight for chain 2 (character 5-grams).
    vocab : float
        Weight for chain 3 (word unigrams).

    Examples
    --------
    >>> custom = Lens(edit=1.0, morph=0.5, phrase=0.3, vocab=0.2)
    >>> Lens.DEDUP.weights
    (0.5, 0.5, 0.7, 1.0)
    """

    # Predefined lens presets (matching C header defines)
    UNIFORM = None   # Initialized below after class definition
    DEDUP = None
    EDIT = None
    VOCAB = None
    CODE = None
    LEGAL = None
    MEDICAL = None
    SUPPORT = None
    POLICY = None

    def __init__(self, edit=1.0, morph=1.0, phrase=1.0, vocab=1.0):
        self._weights = (float(edit), float(morph), float(phrase), float(vocab))

    @property
    def weights(self):
        """Return the 4 chain weights as a tuple."""
        return self._weights

    def _to_ctypes(self):
        """Convert to a ctypes TrineS1Lens structure."""
        lens = TrineS1Lens()
        for i, w in enumerate(self._weights):
            lens.weights[i] = w
        return lens

    def __repr__(self):
        return (
            f"Lens(edit={self._weights[0]:.2f}, morph={self._weights[1]:.2f}, "
            f"phrase={self._weights[2]:.2f}, vocab={self._weights[3]:.2f})"
        )

    def __eq__(self, other):
        if not isinstance(other, Lens):
            return NotImplemented
        return self._weights == other._weights

    def __hash__(self):
        return hash(self._weights)


# Initialize lens presets
Lens.UNIFORM = Lens(edit=1.0, morph=1.0, phrase=1.0, vocab=1.0)
Lens.DEDUP = Lens(edit=0.5, morph=0.5, phrase=0.7, vocab=1.0)
Lens.EDIT = Lens(edit=1.0, morph=0.3, phrase=0.1, vocab=0.0)
Lens.VOCAB = Lens(edit=0.0, morph=0.2, phrase=0.3, vocab=1.0)
Lens.CODE = Lens(edit=1.0, morph=0.8, phrase=0.4, vocab=0.2)
Lens.LEGAL = Lens(edit=0.2, morph=0.4, phrase=1.0, vocab=0.8)
Lens.MEDICAL = Lens(edit=0.3, morph=1.0, phrase=0.6, vocab=0.5)
Lens.SUPPORT = Lens(edit=0.2, morph=0.4, phrase=0.7, vocab=1.0)
Lens.POLICY = Lens(edit=0.1, morph=0.3, phrase=1.0, vocab=0.8)


# ── Canon ────────────────────────────────────────────────────────────────

class Canon:
    """
    Text canonicalization presets.

    Presets apply deterministic transforms before encoding to improve
    near-duplicate detection on real-world corpora.

    Attributes
    ----------
    NONE : int
        No transforms (passthrough).
    SUPPORT : int
        Whitespace + timestamps + UUIDs + numbers.
    CODE : int
        Whitespace + identifier normalization.
    POLICY : int
        Whitespace + number bucketing.
    GENERAL : int
        Whitespace normalization only.
    """

    NONE = 0
    SUPPORT = 1
    CODE = 2
    POLICY = 3
    GENERAL = 4

    _NAMES = {0: "NONE", 1: "SUPPORT", 2: "CODE", 3: "POLICY", 4: "GENERAL"}

    @classmethod
    def apply(cls, text, preset=GENERAL):
        """
        Apply a canonicalization preset to text.

        Parameters
        ----------
        text : str
            Input text.
        preset : int
            One of Canon.NONE, Canon.SUPPORT, Canon.CODE, Canon.POLICY, Canon.GENERAL.

        Returns
        -------
        str
            Canonicalized text.

        Raises
        ------
        ValueError
            If preset is invalid or buffer is too small.
        """
        lib = get_lib()
        text_bytes = text.encode("utf-8")
        text_len = len(text_bytes)
        # Identifier normalization (CODE preset) can expand camelCase into
        # space-separated words, so allocate extra room. The C function
        # returns -1 if out_cap is too small.
        out_cap = text_len * 2 + 1
        out_buf = ctypes.create_string_buffer(out_cap)
        out_len = ctypes.c_size_t(0)

        rc = lib.trine_canon_apply(
            text_bytes, text_len, preset,
            out_buf, out_cap, ctypes.byref(out_len)
        )
        if rc != 0:
            raise ValueError(f"trine_canon_apply failed (preset={preset}, rc={rc})")

        return out_buf.value[:out_len.value].decode("utf-8", errors="replace")

    @classmethod
    def name(cls, preset):
        """Return human-readable name for a preset."""
        return cls._NAMES.get(preset, "UNKNOWN")


# ── Embedding ────────────────────────────────────────────────────────────

class Embedding:
    """
    A 240-trit TRINE embedding.

    Wraps the raw 240-byte trit array produced by the encoder. Provides
    comparison, packing, and introspection methods.

    Parameters
    ----------
    data : bytes, bytearray, list, or numpy array
        240 trit values (each 0, 1, or 2).
    """

    __slots__ = ("_data",)

    def __init__(self, data):
        if _HAS_NUMPY and isinstance(data, np.ndarray):
            if data.shape != (TRINE_CHANNELS,):
                raise ValueError(f"Expected shape ({TRINE_CHANNELS},), got {data.shape}")
            self._data = data.astype(np.uint8)
        elif isinstance(data, (bytes, bytearray)):
            if len(data) != TRINE_CHANNELS:
                raise ValueError(f"Expected {TRINE_CHANNELS} bytes, got {len(data)}")
            if _HAS_NUMPY:
                self._data = np.frombuffer(data, dtype=np.uint8).copy()
            else:
                self._data = bytearray(data)
        elif isinstance(data, (list, tuple)):
            if len(data) != TRINE_CHANNELS:
                raise ValueError(f"Expected {TRINE_CHANNELS} values, got {len(data)}")
            if _HAS_NUMPY:
                self._data = np.array(data, dtype=np.uint8)
            else:
                self._data = bytearray(data)
        elif isinstance(data, ctypes.Array):
            raw = bytes(data)
            if _HAS_NUMPY:
                self._data = np.frombuffer(raw, dtype=np.uint8).copy()
            else:
                self._data = bytearray(raw)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

    @property
    def trits(self):
        """
        Raw trit values as bytes (or numpy array if numpy is available).

        Returns
        -------
        numpy.ndarray or bytearray
            240 trit values, each in {0, 1, 2}.
        """
        if _HAS_NUMPY:
            return self._data.copy()
        return bytearray(self._data)

    @property
    def packed(self):
        """
        48-byte packed representation (5 trits per byte).

        Returns
        -------
        bytes
            48 packed bytes.
        """
        lib = get_lib()
        trits_arr = (ctypes.c_uint8 * TRINE_CHANNELS)(*self._data)
        packed_arr = (ctypes.c_uint8 * TRINE_PACKED_SIZE)()
        rc = lib.trine_s1_pack(trits_arr, packed_arr)
        if rc != 0:
            raise RuntimeError("trine_s1_pack failed")
        return bytes(packed_arr)

    @classmethod
    def from_packed(cls, packed_bytes):
        """
        Create an Embedding from 48-byte packed representation.

        Parameters
        ----------
        packed_bytes : bytes
            48 bytes of packed trits.

        Returns
        -------
        Embedding
        """
        if len(packed_bytes) != TRINE_PACKED_SIZE:
            raise ValueError(f"Expected {TRINE_PACKED_SIZE} bytes, got {len(packed_bytes)}")
        lib = get_lib()
        packed_arr = (ctypes.c_uint8 * TRINE_PACKED_SIZE)(*packed_bytes)
        trits_arr = (ctypes.c_uint8 * TRINE_CHANNELS)()
        rc = lib.trine_s1_unpack(packed_arr, trits_arr)
        if rc != 0:
            raise RuntimeError("trine_s1_unpack failed")
        return cls(trits_arr)

    @property
    def fill_ratio(self):
        """
        Fraction of non-zero channels (sparsity measure).

        Returns
        -------
        float
            Value in [0.0, 1.0].
        """
        lib = get_lib()
        arr = (ctypes.c_uint8 * TRINE_CHANNELS)(*self._data)
        return lib.trine_s1_fill_ratio(arr)

    def similarity(self, other, lens=None):
        """
        Compute lens-weighted cosine similarity to another embedding.

        Parameters
        ----------
        other : Embedding
            The embedding to compare against.
        lens : Lens, optional
            Comparison lens. Defaults to Lens.UNIFORM.

        Returns
        -------
        float
            Similarity in [0.0, 1.0], or -1.0 on error.
        """
        if not isinstance(other, Embedding):
            raise TypeError(f"Expected Embedding, got {type(other)}")
        if lens is None:
            lens = Lens.UNIFORM

        lib = get_lib()
        a_arr = (ctypes.c_uint8 * TRINE_CHANNELS)(*self._data)
        b_arr = (ctypes.c_uint8 * TRINE_CHANNELS)(*other._data)
        c_lens = lens._to_ctypes()
        return lib.trine_s1_compare(a_arr, b_arr, ctypes.byref(c_lens))

    def _to_ctypes_array(self):
        """Return a ctypes uint8 array suitable for passing to C functions."""
        return (ctypes.c_uint8 * TRINE_CHANNELS)(*self._data)

    def __eq__(self, other):
        if not isinstance(other, Embedding):
            return NotImplemented
        if _HAS_NUMPY:
            return bool(np.array_equal(self._data, other._data))
        return self._data == other._data

    def __repr__(self):
        nz = sum(1 for v in self._data if v != 0)
        return f"Embedding(fill={nz}/{TRINE_CHANNELS})"

    def __len__(self):
        return TRINE_CHANNELS


# ── Result ───────────────────────────────────────────────────────────────

class Result:
    """
    Result of an index or router query.

    Attributes
    ----------
    is_duplicate : bool
        True if similarity exceeds the configured threshold.
    similarity : float
        Raw lens-weighted cosine similarity.
    calibrated : float
        Length-calibrated similarity score.
    matched_index : int
        Index of best match in the index, or -1 if no match.
    tag : str or None
        Tag of the matched entry, if available.
    stats : RouteStats or None
        Routing statistics (only set for router queries).
    """

    def __init__(self, is_duplicate, similarity, calibrated, matched_index, tag=None):
        self.is_duplicate = bool(is_duplicate)
        self.similarity = float(similarity)
        self.calibrated = float(calibrated)
        self.matched_index = int(matched_index)
        self.tag = tag

    @classmethod
    def _from_ctypes(cls, c_result, tag=None):
        """Create a Result from a TrineS1Result ctypes structure."""
        return cls(
            is_duplicate=c_result.is_duplicate,
            similarity=c_result.similarity,
            calibrated=c_result.calibrated,
            matched_index=c_result.matched_index,
            tag=tag,
        )

    def __repr__(self):
        dup = "DUPLICATE" if self.is_duplicate else "unique"
        tag_str = f", tag={self.tag!r}" if self.tag else ""
        return (
            f"Result({dup}, sim={self.similarity:.4f}, "
            f"cal={self.calibrated:.4f}, idx={self.matched_index}{tag_str})"
        )

    def __bool__(self):
        """A Result is truthy if it represents a duplicate."""
        return self.is_duplicate


# ── RouteStats ───────────────────────────────────────────────────────────

class RouteStats:
    """
    Routing statistics from a router query.

    Attributes
    ----------
    candidates_checked : int
        Number of full comparisons performed.
    total_entries : int
        Total entries in the index.
    candidate_ratio : float
        Fraction of entries checked (candidates_checked / total_entries).
    speedup : float
        Routing speedup factor (total_entries / candidates_checked).
    recall_mode : str or None
        Name of the active recall preset.
    """

    __slots__ = ("candidates_checked", "total_entries", "candidate_ratio",
                 "speedup", "recall_mode")

    def __init__(self, candidates_checked, total_entries, candidate_ratio,
                 speedup, recall_mode):
        self.candidates_checked = int(candidates_checked)
        self.total_entries = int(total_entries)
        self.candidate_ratio = float(candidate_ratio)
        self.speedup = float(speedup)
        self.recall_mode = recall_mode

    @classmethod
    def _from_ctypes(cls, c_stats):
        """Create RouteStats from a TrineRouteStats ctypes structure."""
        recall = None
        if c_stats.recall_mode:
            recall = c_stats.recall_mode.decode("utf-8", errors="replace")
        return cls(
            candidates_checked=c_stats.candidates_checked,
            total_entries=c_stats.total_entries,
            candidate_ratio=c_stats.candidate_ratio,
            speedup=c_stats.speedup,
            recall_mode=recall,
        )

    def __repr__(self):
        return (
            f"RouteStats(checked={self.candidates_checked}/{self.total_entries}, "
            f"speedup={self.speedup:.1f}x, mode={self.recall_mode})"
        )


# ── TrineEncoder ─────────────────────────────────────────────────────────

class TrineEncoder:
    """
    Text-to-embedding encoder using TRINE shingle encoding.

    Thread-safe: encoding is stateless.

    Examples
    --------
    >>> enc = TrineEncoder()
    >>> emb = enc.encode("Hello, world!")
    >>> emb.fill_ratio
    0.85
    """

    def encode(self, text, canon=Canon.NONE):
        """
        Encode a single text string into a TRINE embedding.

        Parameters
        ----------
        text : str
            Input text (any length).
        canon : int, optional
            Canonicalization preset to apply before encoding.
            Default is Canon.NONE.

        Returns
        -------
        Embedding
            The 240-trit embedding.
        """
        lib = get_lib()

        # Canonicalize if requested
        if canon != Canon.NONE:
            text = Canon.apply(text, canon)

        text_bytes = text.encode("utf-8")
        channels = (ctypes.c_uint8 * TRINE_CHANNELS)()
        rc = lib.trine_encode_shingle(text_bytes, len(text_bytes), channels)
        if rc != 0:
            raise MemoryError("trine_encode_shingle: allocation failed")
        return Embedding(channels)

    def encode_batch(self, texts, canon=Canon.NONE):
        """
        Encode multiple texts into embeddings.

        Uses the C batch API for efficiency (single FFI call).

        Parameters
        ----------
        texts : list of str
            Input texts.
        canon : int, optional
            Canonicalization preset. Default is Canon.NONE.

        Returns
        -------
        list of Embedding
            One embedding per input text.
        """
        if not texts:
            return []

        lib = get_lib()
        count = len(texts)

        # Canonicalize if requested
        if canon != Canon.NONE:
            texts = [Canon.apply(t, canon) for t in texts]

        # Encode to bytes
        byte_texts = [t.encode("utf-8") for t in texts]

        # Build C arrays
        c_texts = (ctypes.c_char_p * count)(*byte_texts)
        c_lens = (ctypes.c_size_t * count)(*[len(b) for b in byte_texts])
        out_buf = (ctypes.c_uint8 * (count * TRINE_CHANNELS))()

        rc = lib.trine_s1_encode_batch(c_texts, c_lens, count, out_buf)
        if rc != 0:
            raise RuntimeError(f"trine_s1_encode_batch failed (rc={rc})")

        # Split output into individual embeddings
        result = []
        for i in range(count):
            offset = i * TRINE_CHANNELS
            chunk = bytes(out_buf[offset:offset + TRINE_CHANNELS])
            result.append(Embedding(chunk))
        return result


# ── TrineIndex ───────────────────────────────────────────────────────────

class TrineIndex:
    """
    In-memory linear-scan dedup index.

    Suitable for up to a few thousand entries. For larger corpora,
    use TrineRouter instead.

    Parameters
    ----------
    threshold : float
        Cosine similarity threshold for duplicate detection (0.0-1.0).
    lens : Lens
        Comparison lens.
    calibrate : bool
        Whether to apply length-aware score calibration.

    Examples
    --------
    >>> idx = TrineIndex(threshold=0.60, lens=Lens.DEDUP)
    >>> idx.add("Hello, world!", tag="doc1")
    0
    >>> result = idx.query("Hello world")
    >>> result.is_duplicate
    True
    """

    def __init__(self, threshold=0.60, lens=None, calibrate=True):
        if lens is None:
            lens = Lens.DEDUP
        self._lib = get_lib()
        self._encoder = TrineEncoder()
        self._threshold = threshold
        self._lens = lens
        self._calibrate = calibrate

        config = self._make_config()
        self._ptr = self._lib.trine_s1_index_create(ctypes.byref(config))
        if not self._ptr:
            raise MemoryError("Failed to create TRINE index")

    def _make_config(self):
        """Build a TrineS1Config ctypes structure."""
        config = TrineS1Config()
        config.threshold = self._threshold
        c_lens = self._lens._to_ctypes()
        config.lens = c_lens
        config.calibrate_length = 1 if self._calibrate else 0
        return config

    def add(self, text, tag=None, canon=Canon.NONE):
        """
        Add a text to the index.

        Parameters
        ----------
        text : str
            Text to add.
        tag : str, optional
            Metadata tag for identification.
        canon : int, optional
            Canonicalization preset.

        Returns
        -------
        int
            Assigned index (0-based), or -1 on failure.
        """
        self._check_alive()
        emb = self._encoder.encode(text, canon=canon)
        arr = emb._to_ctypes_array()
        c_tag = tag.encode("utf-8") if tag else None
        return self._lib.trine_s1_index_add(self._ptr, arr, c_tag)

    def add_embedding(self, embedding, tag=None):
        """
        Add a pre-computed embedding to the index.

        Parameters
        ----------
        embedding : Embedding
            Pre-computed embedding.
        tag : str, optional
            Metadata tag.

        Returns
        -------
        int
            Assigned index (0-based), or -1 on failure.
        """
        self._check_alive()
        arr = embedding._to_ctypes_array()
        c_tag = tag.encode("utf-8") if tag else None
        return self._lib.trine_s1_index_add(self._ptr, arr, c_tag)

    def query(self, text, canon=Canon.NONE):
        """
        Query the index for the best match to a text.

        Parameters
        ----------
        text : str
            Query text.
        canon : int, optional
            Canonicalization preset.

        Returns
        -------
        Result
            Query result with similarity, duplicate flag, and matched index.
        """
        self._check_alive()
        emb = self._encoder.encode(text, canon=canon)
        return self.query_embedding(emb)

    def query_embedding(self, embedding):
        """
        Query the index with a pre-computed embedding.

        Parameters
        ----------
        embedding : Embedding
            Query embedding.

        Returns
        -------
        Result
        """
        self._check_alive()
        arr = embedding._to_ctypes_array()
        c_result = self._lib.trine_s1_index_query(self._ptr, arr)

        tag = None
        if c_result.matched_index >= 0:
            raw_tag = self._lib.trine_s1_index_tag(self._ptr, c_result.matched_index)
            if raw_tag:
                tag = raw_tag.decode("utf-8", errors="replace")

        return Result._from_ctypes(c_result, tag=tag)

    def tag(self, index):
        """
        Get the tag for an entry by index.

        Parameters
        ----------
        index : int
            Entry index (0-based).

        Returns
        -------
        str or None
        """
        self._check_alive()
        raw = self._lib.trine_s1_index_tag(self._ptr, index)
        if raw:
            return raw.decode("utf-8", errors="replace")
        return None

    def save(self, path):
        """
        Save the index to a binary file.

        Parameters
        ----------
        path : str
            Output file path.

        Raises
        ------
        IOError
            If the save fails.
        """
        self._check_alive()
        rc = self._lib.trine_s1_index_save(self._ptr, path.encode("utf-8"))
        if rc != 0:
            raise IOError(f"Failed to save index to {path}")

    @classmethod
    def load(cls, path):
        """
        Load an index from a binary file.

        Parameters
        ----------
        path : str
            Path to the index file.

        Returns
        -------
        TrineIndex
            The loaded index.

        Raises
        ------
        IOError
            If the load fails.
        """
        lib = get_lib()
        ptr = lib.trine_s1_index_load(path.encode("utf-8"))
        if not ptr:
            raise IOError(f"Failed to load index from {path}")

        # Create instance without calling __init__
        obj = object.__new__(cls)
        obj._lib = lib
        obj._encoder = TrineEncoder()
        obj._ptr = ptr
        # These are unknown from a loaded file; use defaults
        obj._threshold = 0.60
        obj._lens = Lens.DEDUP
        obj._calibrate = True
        return obj

    def _check_alive(self):
        """Raise if the index has been freed."""
        if self._ptr is None:
            raise RuntimeError("TrineIndex has been closed/freed")

    def close(self):
        """Free the underlying C index."""
        if self._ptr is not None:
            self._lib.trine_s1_index_free(self._ptr)
            self._ptr = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __len__(self):
        """Return the number of entries in the index."""
        self._check_alive()
        return self._lib.trine_s1_index_count(self._ptr)

    def __contains__(self, text):
        """Check if text has a duplicate in the index."""
        result = self.query(text)
        return result.is_duplicate

    def __repr__(self):
        if self._ptr is None:
            return "TrineIndex(closed)"
        n = len(self)
        return f"TrineIndex(n={n}, threshold={self._threshold:.2f})"


# ── TrineRouter ──────────────────────────────────────────────────────────

class TrineRouter:
    """
    Band-LSH routed index for large corpora.

    Provides the same API as TrineIndex but uses locality-sensitive hashing
    to reduce per-query comparisons by 10-50x.

    Parameters
    ----------
    threshold : float
        Cosine similarity threshold for duplicate detection.
    lens : Lens
        Comparison lens.
    calibrate : bool
        Whether to apply length-aware score calibration.

    Examples
    --------
    >>> router = TrineRouter(threshold=0.60, lens=Lens.DEDUP)
    >>> router.add("Hello, world!", tag="doc1")
    0
    >>> result = router.query("Hello world")
    >>> result.is_duplicate
    True
    >>> result.stats.speedup
    1.0
    """

    # Recall mode constants
    FAST = 0
    BALANCED = 1
    STRICT = 2

    def __init__(self, threshold=0.60, lens=None, calibrate=True):
        if lens is None:
            lens = Lens.DEDUP
        self._lib = get_lib()
        self._encoder = TrineEncoder()
        self._threshold = threshold
        self._lens = lens
        self._calibrate = calibrate

        config = self._make_config()
        self._ptr = self._lib.trine_route_create(ctypes.byref(config))
        if not self._ptr:
            raise MemoryError("Failed to create TRINE router")

    def _make_config(self):
        """Build a TrineS1Config ctypes structure."""
        config = TrineS1Config()
        config.threshold = self._threshold
        config.lens = self._lens._to_ctypes()
        config.calibrate_length = 1 if self._calibrate else 0
        return config

    def add(self, text, tag=None, canon=Canon.NONE):
        """
        Add a text to the routed index.

        Parameters
        ----------
        text : str
            Text to add.
        tag : str, optional
            Metadata tag.
        canon : int, optional
            Canonicalization preset.

        Returns
        -------
        int
            Assigned index (0-based), or -1 on failure.
        """
        self._check_alive()
        emb = self._encoder.encode(text, canon=canon)
        arr = emb._to_ctypes_array()
        c_tag = tag.encode("utf-8") if tag else None
        return self._lib.trine_route_add(self._ptr, arr, c_tag)

    def add_embedding(self, embedding, tag=None):
        """
        Add a pre-computed embedding to the router.

        Parameters
        ----------
        embedding : Embedding
            Pre-computed embedding.
        tag : str, optional
            Metadata tag.

        Returns
        -------
        int
            Assigned index, or -1 on failure.
        """
        self._check_alive()
        arr = embedding._to_ctypes_array()
        c_tag = tag.encode("utf-8") if tag else None
        return self._lib.trine_route_add(self._ptr, arr, c_tag)

    def query(self, text, canon=Canon.NONE):
        """
        Query the routed index for the best match.

        Parameters
        ----------
        text : str
            Query text.
        canon : int, optional
            Canonicalization preset.

        Returns
        -------
        Result
            Query result. The result has an additional `.stats` attribute
            with routing statistics.
        """
        self._check_alive()
        emb = self._encoder.encode(text, canon=canon)
        return self.query_embedding(emb)

    def query_embedding(self, embedding):
        """
        Query with a pre-computed embedding.

        Parameters
        ----------
        embedding : Embedding
            Query embedding.

        Returns
        -------
        Result
            Query result with `.stats` attribute.
        """
        self._check_alive()
        arr = embedding._to_ctypes_array()
        c_stats = TrineRouteStats()
        c_result = self._lib.trine_route_query(self._ptr, arr, ctypes.byref(c_stats))

        tag = None
        if c_result.matched_index >= 0:
            raw_tag = self._lib.trine_route_tag(self._ptr, c_result.matched_index)
            if raw_tag:
                tag = raw_tag.decode("utf-8", errors="replace")

        result = Result._from_ctypes(c_result, tag=tag)
        result.stats = RouteStats._from_ctypes(c_stats)
        return result

    def tag(self, index):
        """Get the tag for an entry by index."""
        self._check_alive()
        raw = self._lib.trine_route_tag(self._ptr, index)
        if raw:
            return raw.decode("utf-8", errors="replace")
        return None

    def set_recall(self, mode):
        """
        Set the recall mode.

        Parameters
        ----------
        mode : int
            One of TrineRouter.FAST, TrineRouter.BALANCED, TrineRouter.STRICT.

        Raises
        ------
        ValueError
            If mode is invalid.
        """
        self._check_alive()
        rc = self._lib.trine_route_set_recall(self._ptr, mode)
        if rc != 0:
            raise ValueError(f"Invalid recall mode: {mode}")

    def get_recall(self):
        """
        Get the current recall mode.

        Returns
        -------
        int
            Current recall mode constant.
        """
        self._check_alive()
        return self._lib.trine_route_get_recall(self._ptr)

    def global_stats(self):
        """
        Get global statistics for the index.

        Returns
        -------
        RouteStats
        """
        self._check_alive()
        c_stats = TrineRouteStats()
        self._lib.trine_route_global_stats(self._ptr, ctypes.byref(c_stats))
        return RouteStats._from_ctypes(c_stats)

    def save(self, path):
        """
        Save the routed index to a binary file.

        Parameters
        ----------
        path : str
            Output file path.

        Raises
        ------
        IOError
            If the save fails.
        """
        self._check_alive()
        rc = self._lib.trine_route_save(self._ptr, path.encode("utf-8"))
        if rc != 0:
            raise IOError(f"Failed to save router to {path}")

    @classmethod
    def load(cls, path):
        """
        Load a routed index from a binary file.

        Parameters
        ----------
        path : str
            Path to the index file.

        Returns
        -------
        TrineRouter
            The loaded router.

        Raises
        ------
        IOError
            If the load fails.
        """
        lib = get_lib()
        ptr = lib.trine_route_load(path.encode("utf-8"))
        if not ptr:
            raise IOError(f"Failed to load router from {path}")

        obj = object.__new__(cls)
        obj._lib = lib
        obj._encoder = TrineEncoder()
        obj._ptr = ptr
        obj._threshold = 0.60
        obj._lens = Lens.DEDUP
        obj._calibrate = True
        return obj

    def _check_alive(self):
        """Raise if the router has been freed."""
        if self._ptr is None:
            raise RuntimeError("TrineRouter has been closed/freed")

    def close(self):
        """Free the underlying C routed index."""
        if self._ptr is not None:
            self._lib.trine_route_free(self._ptr)
            self._ptr = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __len__(self):
        """Return the number of entries."""
        self._check_alive()
        return self._lib.trine_route_count(self._ptr)

    def __contains__(self, text):
        """Check if text has a duplicate in the index."""
        result = self.query(text)
        return result.is_duplicate

    def __repr__(self):
        if self._ptr is None:
            return "TrineRouter(closed)"
        n = len(self)
        return f"TrineRouter(n={n}, threshold={self._threshold:.2f})"
