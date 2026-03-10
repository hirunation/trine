"""
Low-level ctypes binding to libtrine.

Locates and loads the shared library, defines C structure wrappers,
and sets argtypes/restype for every exported C function.

Usage:
    from pytrine._binding import get_lib
    lib = get_lib()
    lib.trine_encode_shingle(text, len, channels)
"""

import ctypes
import ctypes.util
import os
import sys

# ── C Structure Definitions ──────────────────────────────────────────────

class TrineS1Lens(ctypes.Structure):
    """Mirrors trine_s1_lens_t: 4 float weights, one per chain."""
    _fields_ = [
        ("weights", ctypes.c_float * 4),
    ]


class TrineS1Config(ctypes.Structure):
    """Mirrors trine_s1_config_t: threshold, lens, calibrate_length."""
    _fields_ = [
        ("threshold", ctypes.c_float),
        ("lens", TrineS1Lens),
        ("calibrate_length", ctypes.c_int),
    ]


class TrineS1Result(ctypes.Structure):
    """Mirrors trine_s1_result_t: dedup check result."""
    _fields_ = [
        ("is_duplicate", ctypes.c_int),
        ("similarity", ctypes.c_float),
        ("calibrated", ctypes.c_float),
        ("matched_index", ctypes.c_int),
    ]


class TrineEncodeInfo(ctypes.Structure):
    """Mirrors trine_encode_info_t: encoding metadata."""
    _fields_ = [
        ("char_count", ctypes.c_int),
        ("is_truncated", ctypes.c_int),
        ("overflow_hash", ctypes.c_uint32),
    ]


class TrineRouteStats(ctypes.Structure):
    """Mirrors trine_route_stats_t: routing query statistics."""
    _fields_ = [
        ("candidates_checked", ctypes.c_int),
        ("total_entries", ctypes.c_int),
        ("candidate_ratio", ctypes.c_float),
        ("speedup", ctypes.c_float),
        ("recall_mode", ctypes.c_char_p),
    ]


# ── Stage-2 Structure Definitions ────────────────────────────────────────

class TrineS2Info(ctypes.Structure):
    """Mirrors trine_s2_info_t: Stage-2 model introspection."""
    _fields_ = [
        ("projection_k", ctypes.c_uint32),
        ("projection_dims", ctypes.c_uint32),
        ("cascade_cells", ctypes.c_uint32),
        ("max_depth", ctypes.c_uint32),
        ("is_identity", ctypes.c_int),
    ]


class TrineSourceWeight(ctypes.Structure):
    """Mirrors trine_source_weight_t: per-source training weight."""
    _fields_ = [
        ("name", ctypes.c_char * 16),
        ("weight", ctypes.c_float),
    ]


class TrineHebbianConfig(ctypes.Structure):
    """Mirrors trine_hebbian_config_t: Hebbian training configuration."""
    _fields_ = [
        ("similarity_threshold", ctypes.c_float),
        ("freeze_threshold", ctypes.c_int32),
        ("freeze_target_density", ctypes.c_float),
        ("cascade_cells", ctypes.c_uint32),
        ("cascade_depth", ctypes.c_uint32),
        ("projection_mode", ctypes.c_int),
        # Phase A1: Weighted Hebbian
        ("weighted_mode", ctypes.c_int),
        ("pos_scale", ctypes.c_float),
        ("neg_scale", ctypes.c_float),
        # Phase A2: Dataset rebalancing
        ("source_weights", TrineSourceWeight * 8),
        ("n_source_weights", ctypes.c_int),
        # Phase D1: Sparse projection
        ("sparse_k", ctypes.c_uint32),
        # v1.0.3: Block-diagonal projection
        ("block_diagonal", ctypes.c_int),
        # RNG seed for deterministic downsampling
        ("rng_seed", ctypes.c_uint64),
    ]


class TrineHebbianMetrics(ctypes.Structure):
    """Mirrors trine_hebbian_metrics_t: Hebbian training metrics."""
    _fields_ = [
        ("pairs_observed", ctypes.c_int64),
        ("max_abs_counter", ctypes.c_int32),
        ("n_positive_weights", ctypes.c_uint32),
        ("n_negative_weights", ctypes.c_uint32),
        ("n_zero_weights", ctypes.c_uint32),
        ("weight_density", ctypes.c_float),
        ("effective_threshold", ctypes.c_int32),
    ]


class TrineS2SaveConfig(ctypes.Structure):
    """Mirrors trine_s2_save_config_t: persistence configuration."""
    _fields_ = [
        ("similarity_threshold", ctypes.c_float),
        ("density", ctypes.c_float),
        ("topo_seed", ctypes.c_uint64),
    ]


# ── Type Aliases ─────────────────────────────────────────────────────────

# Opaque pointer types
TrineS1IndexPtr = ctypes.c_void_p
TrineRoutePtr = ctypes.c_void_p
TrineS2ModelPtr = ctypes.c_void_p
TrineHebbianPtr = ctypes.c_void_p

# Array types
Channels240 = ctypes.c_uint8 * 240
Packed48 = ctypes.c_uint8 * 48


# ── Library Loading ──────────────────────────────────────────────────────

_lib_cache = None


def _find_lib():
    """Search for libtrine shared library in multiple locations."""
    if sys.platform == "darwin":
        names = ["libtrine.dylib"]
    elif sys.platform == "win32":
        names = ["trine.dll", "libtrine.dll"]
    else:
        names = ["libtrine.so"]

    # Search paths in priority order
    search_dirs = []

    # 1. Same directory as this Python file (package install)
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    search_dirs.append(pkg_dir)

    # 2. TRINE_LIB_DIR environment variable
    env_dir = os.environ.get("TRINE_LIB_DIR")
    if env_dir:
        search_dirs.append(env_dir)

    # 3. Build directory relative to package (bindings/python/pytrine -> project/build)
    project_build = os.path.normpath(os.path.join(pkg_dir, "..", "..", "..", "build"))
    search_dirs.append(project_build)

    # 4. Project root directory (in case .so was built there)
    project_dir = os.path.normpath(os.path.join(pkg_dir, "..", "..", ".."))
    search_dirs.append(project_dir)

    # 5. Current working directory
    search_dirs.append(os.getcwd())

    for d in search_dirs:
        for name in names:
            path = os.path.join(d, name)
            if os.path.isfile(path):
                return path

    # 6. System library search (LD_LIBRARY_PATH, etc.)
    for name in names:
        found = ctypes.util.find_library(name.replace("lib", "").split(".")[0])
        if found:
            return found

    return None


def get_lib():
    """
    Load and return the TRINE shared library with all function signatures set.

    The library is loaded once and cached for subsequent calls.

    Returns
    -------
    ctypes.CDLL
        The loaded library with argtypes and restype configured.

    Raises
    ------
    OSError
        If the shared library cannot be found or loaded.
    """
    global _lib_cache
    if _lib_cache is not None:
        return _lib_cache

    path = _find_lib()
    if path is None:
        raise OSError(
            "Cannot find libtrine shared library.\n"
            "Build it with: make lib  (from bindings/python/)\n"
            "Or set TRINE_LIB_DIR to the directory containing the library.\n"
            "Or run: python -m pytrine.build_lib"
        )

    lib = ctypes.CDLL(path)

    # ── Encode API ───────────────────────────────────────────────────

    # void trine_encode(const char *text, size_t len, uint8_t channels[240])
    lib.trine_encode.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint8)]
    lib.trine_encode.restype = None

    # int trine_decode(const uint8_t channels[240], char *text, size_t max_len)
    lib.trine_decode.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_char_p, ctypes.c_size_t]
    lib.trine_decode.restype = ctypes.c_int

    # void trine_encode_info(const char *text, size_t len, trine_encode_info_t *info)
    lib.trine_encode_info.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(TrineEncodeInfo)]
    lib.trine_encode_info.restype = None

    # int trine_encode_shingle(const char *text, size_t len, uint8_t channels[240])
    lib.trine_encode_shingle.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint8)]
    lib.trine_encode_shingle.restype = ctypes.c_int

    # ── Stage-1 Encode API ───────────────────────────────────────────

    # int trine_s1_encode(const char *text, size_t len, uint8_t out[240])
    lib.trine_s1_encode.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint8)]
    lib.trine_s1_encode.restype = ctypes.c_int

    # int trine_s1_encode_batch(const char **texts, const size_t *lens, int count, uint8_t *out)
    lib.trine_s1_encode_batch.argtypes = [
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8),
    ]
    lib.trine_s1_encode_batch.restype = ctypes.c_int

    # ── Stage-1 Compare API ──────────────────────────────────────────

    # float trine_s1_compare(const uint8_t a[240], const uint8_t b[240], const trine_s1_lens_t *lens)
    lib.trine_s1_compare.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(TrineS1Lens),
    ]
    lib.trine_s1_compare.restype = ctypes.c_float

    # trine_s1_result_t trine_s1_check(...)
    lib.trine_s1_check.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(TrineS1Config),
    ]
    lib.trine_s1_check.restype = TrineS1Result

    # int trine_s1_compare_batch(...)
    lib.trine_s1_compare_batch.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int,
        ctypes.POINTER(TrineS1Config),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.trine_s1_compare_batch.restype = ctypes.c_int

    # ── Stage-1 Index API ────────────────────────────────────────────

    # trine_s1_index_t *trine_s1_index_create(const trine_s1_config_t *config)
    lib.trine_s1_index_create.argtypes = [ctypes.POINTER(TrineS1Config)]
    lib.trine_s1_index_create.restype = TrineS1IndexPtr

    # int trine_s1_index_add(trine_s1_index_t *idx, const uint8_t emb[240], const char *tag)
    lib.trine_s1_index_add.argtypes = [TrineS1IndexPtr, ctypes.POINTER(ctypes.c_uint8), ctypes.c_char_p]
    lib.trine_s1_index_add.restype = ctypes.c_int

    # trine_s1_result_t trine_s1_index_query(const trine_s1_index_t *idx, const uint8_t candidate[240])
    lib.trine_s1_index_query.argtypes = [TrineS1IndexPtr, ctypes.POINTER(ctypes.c_uint8)]
    lib.trine_s1_index_query.restype = TrineS1Result

    # int trine_s1_index_count(const trine_s1_index_t *idx)
    lib.trine_s1_index_count.argtypes = [TrineS1IndexPtr]
    lib.trine_s1_index_count.restype = ctypes.c_int

    # const char *trine_s1_index_tag(const trine_s1_index_t *idx, int index)
    lib.trine_s1_index_tag.argtypes = [TrineS1IndexPtr, ctypes.c_int]
    lib.trine_s1_index_tag.restype = ctypes.c_char_p

    # void trine_s1_index_free(trine_s1_index_t *idx)
    lib.trine_s1_index_free.argtypes = [TrineS1IndexPtr]
    lib.trine_s1_index_free.restype = None

    # int trine_s1_index_save(const trine_s1_index_t *idx, const char *path)
    lib.trine_s1_index_save.argtypes = [TrineS1IndexPtr, ctypes.c_char_p]
    lib.trine_s1_index_save.restype = ctypes.c_int

    # trine_s1_index_t *trine_s1_index_load(const char *path)
    lib.trine_s1_index_load.argtypes = [ctypes.c_char_p]
    lib.trine_s1_index_load.restype = TrineS1IndexPtr

    # ── Stage-1 Pack/Unpack API ──────────────────────────────────────

    # int trine_s1_pack(const uint8_t trits[240], uint8_t packed[48])
    lib.trine_s1_pack.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8)]
    lib.trine_s1_pack.restype = ctypes.c_int

    # int trine_s1_unpack(const uint8_t packed[48], uint8_t trits[240])
    lib.trine_s1_unpack.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8)]
    lib.trine_s1_unpack.restype = ctypes.c_int

    # float trine_s1_compare_packed(const uint8_t a[48], const uint8_t b[48], const trine_s1_lens_t *lens)
    lib.trine_s1_compare_packed.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(TrineS1Lens),
    ]
    lib.trine_s1_compare_packed.restype = ctypes.c_float

    # ── Stage-1 Calibration API ──────────────────────────────────────

    # float trine_s1_fill_ratio(const uint8_t emb[240])
    lib.trine_s1_fill_ratio.argtypes = [ctypes.POINTER(ctypes.c_uint8)]
    lib.trine_s1_fill_ratio.restype = ctypes.c_float

    # float trine_s1_calibrate(float raw_cosine, float fill_a, float fill_b)
    lib.trine_s1_calibrate.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float]
    lib.trine_s1_calibrate.restype = ctypes.c_float

    # ── Route API ────────────────────────────────────────────────────

    # trine_route_t *trine_route_create(const trine_s1_config_t *config)
    lib.trine_route_create.argtypes = [ctypes.POINTER(TrineS1Config)]
    lib.trine_route_create.restype = TrineRoutePtr

    # int trine_route_add(trine_route_t *rt, const uint8_t emb[240], const char *tag)
    lib.trine_route_add.argtypes = [TrineRoutePtr, ctypes.POINTER(ctypes.c_uint8), ctypes.c_char_p]
    lib.trine_route_add.restype = ctypes.c_int

    # trine_s1_result_t trine_route_query(const trine_route_t *rt, const uint8_t candidate[240], trine_route_stats_t *stats)
    lib.trine_route_query.argtypes = [TrineRoutePtr, ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(TrineRouteStats)]
    lib.trine_route_query.restype = TrineS1Result

    # int trine_route_count(const trine_route_t *rt)
    lib.trine_route_count.argtypes = [TrineRoutePtr]
    lib.trine_route_count.restype = ctypes.c_int

    # const char *trine_route_tag(const trine_route_t *rt, int index)
    lib.trine_route_tag.argtypes = [TrineRoutePtr, ctypes.c_int]
    lib.trine_route_tag.restype = ctypes.c_char_p

    # const uint8_t *trine_route_embedding(const trine_route_t *rt, int index)
    lib.trine_route_embedding.argtypes = [TrineRoutePtr, ctypes.c_int]
    lib.trine_route_embedding.restype = ctypes.POINTER(ctypes.c_uint8)

    # void trine_route_free(trine_route_t *rt)
    lib.trine_route_free.argtypes = [TrineRoutePtr]
    lib.trine_route_free.restype = None

    # int trine_route_save(const trine_route_t *rt, const char *path)
    lib.trine_route_save.argtypes = [TrineRoutePtr, ctypes.c_char_p]
    lib.trine_route_save.restype = ctypes.c_int

    # trine_route_t *trine_route_load(const char *path)
    lib.trine_route_load.argtypes = [ctypes.c_char_p]
    lib.trine_route_load.restype = TrineRoutePtr

    # void trine_route_global_stats(const trine_route_t *rt, trine_route_stats_t *stats)
    lib.trine_route_global_stats.argtypes = [TrineRoutePtr, ctypes.POINTER(TrineRouteStats)]
    lib.trine_route_global_stats.restype = None

    # int trine_route_set_recall(trine_route_t *rt, int mode)
    lib.trine_route_set_recall.argtypes = [TrineRoutePtr, ctypes.c_int]
    lib.trine_route_set_recall.restype = ctypes.c_int

    # int trine_route_get_recall(const trine_route_t *rt)
    lib.trine_route_get_recall.argtypes = [TrineRoutePtr]
    lib.trine_route_get_recall.restype = ctypes.c_int

    # int trine_route_bucket_sizes(const trine_route_t *rt, int band, int *sizes)
    lib.trine_route_bucket_sizes.argtypes = [TrineRoutePtr, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    lib.trine_route_bucket_sizes.restype = ctypes.c_int

    # ── Canon API ────────────────────────────────────────────────────

    # int trine_canon_apply(const char *text, size_t len, int preset, char *out, size_t out_cap, size_t *out_len)
    lib.trine_canon_apply.argtypes = [
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.trine_canon_apply.restype = ctypes.c_int

    # const char *trine_canon_preset_name(int preset)
    lib.trine_canon_preset_name.argtypes = [ctypes.c_int]
    lib.trine_canon_preset_name.restype = ctypes.c_char_p

    # ── Stage-2 Inference API ────────────────────────────────────────

    # trine_s2_model_t *trine_s2_create_identity(void)
    lib.trine_s2_create_identity.argtypes = []
    lib.trine_s2_create_identity.restype = TrineS2ModelPtr

    # trine_s2_model_t *trine_s2_create_random(uint32_t n_cells, uint64_t seed)
    lib.trine_s2_create_random.argtypes = [ctypes.c_uint32, ctypes.c_uint64]
    lib.trine_s2_create_random.restype = TrineS2ModelPtr

    # trine_s2_model_t *trine_s2_create_from_parts(const void *proj, uint32_t n_cells, uint64_t topo_seed)
    lib.trine_s2_create_from_parts.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint64]
    lib.trine_s2_create_from_parts.restype = TrineS2ModelPtr

    # trine_s2_model_t *trine_s2_create_block_diagonal(const uint8_t *block_weights, int K, uint32_t n_cells, uint64_t topo_seed)
    lib.trine_s2_create_block_diagonal.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int,
        ctypes.c_uint32,
        ctypes.c_uint64,
    ]
    lib.trine_s2_create_block_diagonal.restype = TrineS2ModelPtr

    # void trine_s2_free(trine_s2_model_t *model)
    lib.trine_s2_free.argtypes = [TrineS2ModelPtr]
    lib.trine_s2_free.restype = None

    # int trine_s2_encode(const trine_s2_model_t *model, const char *text, size_t len, uint32_t depth, uint8_t out[240])
    lib.trine_s2_encode.argtypes = [
        TrineS2ModelPtr,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint8),
    ]
    lib.trine_s2_encode.restype = ctypes.c_int

    # int trine_s2_encode_from_trits(const trine_s2_model_t *model, const uint8_t stage1[240], uint32_t depth, uint8_t out[240])
    lib.trine_s2_encode_from_trits.argtypes = [
        TrineS2ModelPtr,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint8),
    ]
    lib.trine_s2_encode_from_trits.restype = ctypes.c_int

    # int trine_s2_encode_depths(const trine_s2_model_t *model, const char *text, size_t len, uint32_t max_depth, uint8_t *out, size_t out_size)
    lib.trine_s2_encode_depths.argtypes = [
        TrineS2ModelPtr,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
    ]
    lib.trine_s2_encode_depths.restype = ctypes.c_int

    # float trine_s2_compare(const uint8_t a[240], const uint8_t b[240], const void *lens)
    lib.trine_s2_compare.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_void_p,
    ]
    lib.trine_s2_compare.restype = ctypes.c_float

    # int trine_s2_info(const trine_s2_model_t *model, trine_s2_info_t *info)
    lib.trine_s2_info.argtypes = [TrineS2ModelPtr, ctypes.POINTER(TrineS2Info)]
    lib.trine_s2_info.restype = ctypes.c_int

    # void trine_s2_set_projection_mode(trine_s2_model_t *model, int mode)
    lib.trine_s2_set_projection_mode.argtypes = [TrineS2ModelPtr, ctypes.c_int]
    lib.trine_s2_set_projection_mode.restype = None

    # int trine_s2_get_projection_mode(const trine_s2_model_t *model)
    lib.trine_s2_get_projection_mode.argtypes = [TrineS2ModelPtr]
    lib.trine_s2_get_projection_mode.restype = ctypes.c_int

    # int trine_s2_is_identity(const trine_s2_model_t *model)
    lib.trine_s2_is_identity.argtypes = [TrineS2ModelPtr]
    lib.trine_s2_is_identity.restype = ctypes.c_int

    # ── Stage-2 Gate-Aware Comparison (Phase B1) ────────────────────

    # float trine_s2_compare_gated(const trine_s2_model_t *model, const uint8_t a[240], const uint8_t b[240])
    lib.trine_s2_compare_gated.argtypes = [
        TrineS2ModelPtr,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
    ]
    lib.trine_s2_compare_gated.restype = ctypes.c_float

    # ── Stage-2 Per-Chain Blend Comparison (Phase B2) ─────────────

    # float trine_s2_compare_chain_blend(s1_a, s1_b, s2_a, s2_b, alpha[4])
    lib.trine_s2_compare_chain_blend.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.trine_s2_compare_chain_blend.restype = ctypes.c_float

    # ── Stage-2 Stacked Depth ────────────────────────────────────

    # void trine_s2_set_stacked_depth(trine_s2_model_t *model, int enable)
    lib.trine_s2_set_stacked_depth.argtypes = [TrineS2ModelPtr, ctypes.c_int]
    lib.trine_s2_set_stacked_depth.restype = None

    # int trine_s2_get_stacked_depth(const trine_s2_model_t *model)
    lib.trine_s2_get_stacked_depth.argtypes = [TrineS2ModelPtr]
    lib.trine_s2_get_stacked_depth.restype = ctypes.c_int

    # ── Stage-2 Additional Introspection ─────────────────────────

    # uint32_t trine_s2_get_cascade_cells(const trine_s2_model_t *model)
    lib.trine_s2_get_cascade_cells.argtypes = [TrineS2ModelPtr]
    lib.trine_s2_get_cascade_cells.restype = ctypes.c_uint32

    # uint32_t trine_s2_get_default_depth(const trine_s2_model_t *model)
    lib.trine_s2_get_default_depth.argtypes = [TrineS2ModelPtr]
    lib.trine_s2_get_default_depth.restype = ctypes.c_uint32

    # const void *trine_s2_get_projection(const trine_s2_model_t *model)
    lib.trine_s2_get_projection.argtypes = [TrineS2ModelPtr]
    lib.trine_s2_get_projection.restype = ctypes.c_void_p

    # const uint8_t *trine_s2_get_block_projection(const trine_s2_model_t *model)
    lib.trine_s2_get_block_projection.argtypes = [TrineS2ModelPtr]
    lib.trine_s2_get_block_projection.restype = ctypes.POINTER(ctypes.c_uint8)

    # ── Stage-2 Adaptive Blend ─────────────────────────────────────

    # void trine_s2_set_adaptive_alpha(trine_s2_model_t *model, const float buckets[10])
    lib.trine_s2_set_adaptive_alpha.argtypes = [TrineS2ModelPtr, ctypes.POINTER(ctypes.c_float)]
    lib.trine_s2_set_adaptive_alpha.restype = None

    # float trine_s2_compare_adaptive_blend(model, s1_a, s1_b, s2_a, s2_b)
    lib.trine_s2_compare_adaptive_blend.argtypes = [
        TrineS2ModelPtr,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
    ]
    lib.trine_s2_compare_adaptive_blend.restype = ctypes.c_float

    # ── Stage-2 Persistence API ──────────────────────────────────────

    # int trine_s2_save(const trine_s2_model_t *model, const char *path, const trine_s2_save_config_t *config)
    lib.trine_s2_save.argtypes = [TrineS2ModelPtr, ctypes.c_char_p, ctypes.POINTER(TrineS2SaveConfig)]
    lib.trine_s2_save.restype = ctypes.c_int

    # trine_s2_model_t *trine_s2_load(const char *path)
    lib.trine_s2_load.argtypes = [ctypes.c_char_p]
    lib.trine_s2_load.restype = TrineS2ModelPtr

    # int trine_s2_validate(const char *path)
    lib.trine_s2_validate.argtypes = [ctypes.c_char_p]
    lib.trine_s2_validate.restype = ctypes.c_int

    # ── Stage-2 Hebbian Training API ─────────────────────────────────

    # trine_hebbian_state_t *trine_hebbian_create(const trine_hebbian_config_t *config)
    lib.trine_hebbian_create.argtypes = [ctypes.POINTER(TrineHebbianConfig)]
    lib.trine_hebbian_create.restype = TrineHebbianPtr

    # void trine_hebbian_free(trine_hebbian_state_t *state)
    lib.trine_hebbian_free.argtypes = [TrineHebbianPtr]
    lib.trine_hebbian_free.restype = None

    # void trine_hebbian_observe(state, a[240], b[240], float similarity)
    lib.trine_hebbian_observe.argtypes = [
        TrineHebbianPtr,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_float,
    ]
    lib.trine_hebbian_observe.restype = None

    # void trine_hebbian_observe_text(state, text_a, len_a, text_b, len_b)
    lib.trine_hebbian_observe_text.argtypes = [
        TrineHebbianPtr,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.trine_hebbian_observe_text.restype = None

    # int64_t trine_hebbian_train_file(state, path, uint32_t epochs)
    lib.trine_hebbian_train_file.argtypes = [TrineHebbianPtr, ctypes.c_char_p, ctypes.c_uint32]
    lib.trine_hebbian_train_file.restype = ctypes.c_int64

    # trine_s2_model_t *trine_hebbian_freeze(const trine_hebbian_state_t *state)
    lib.trine_hebbian_freeze.argtypes = [TrineHebbianPtr]
    lib.trine_hebbian_freeze.restype = TrineS2ModelPtr

    # int trine_hebbian_metrics(const trine_hebbian_state_t *state, trine_hebbian_metrics_t *out)
    lib.trine_hebbian_metrics.argtypes = [TrineHebbianPtr, ctypes.POINTER(TrineHebbianMetrics)]
    lib.trine_hebbian_metrics.restype = ctypes.c_int

    # void trine_hebbian_reset(trine_hebbian_state_t *state)
    lib.trine_hebbian_reset.argtypes = [TrineHebbianPtr]
    lib.trine_hebbian_reset.restype = None

    # trine_hebbian_config_t trine_hebbian_get_config(const trine_hebbian_state_t *state)
    lib.trine_hebbian_get_config.argtypes = [TrineHebbianPtr]
    lib.trine_hebbian_get_config.restype = TrineHebbianConfig

    # ── Stage-2 Hebbian Threshold Schedule ──────────────────────────

    # void trine_hebbian_set_threshold(trine_hebbian_state_t *state, float threshold)
    lib.trine_hebbian_set_threshold.argtypes = [TrineHebbianPtr, ctypes.c_float]
    lib.trine_hebbian_set_threshold.restype = None

    _lib_cache = lib
    return lib
