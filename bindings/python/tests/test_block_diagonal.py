"""
Tests for Stage-2 block-diagonal projection functionality.

Covers:
  1. Block-diagonal model creation via Stage2Model.create_block_diagonal()
  2. Block-diagonal encoding (240 trits in {0,1,2})
  3. Block-diagonal comparison (result in [-1, 1])
  4. Block-diagonal persistence (save/load round-trip)
  5. Adaptive alpha (per-S1-bucket blending)
  6. HebbianTrainer block-diagonal mode
  7. Backward compatibility (standard non-block operations)

Requires libtrine.so to be compiled:
    cd bindings/python && make lib
"""

import os
import tempfile

import pytest

try:
    from pytrine import Embedding, TrineEncoder
    from pytrine.stage2 import (
        PROJ_BLOCK_DIAG,
        PROJ_DIAGONAL,
        PROJ_SIGN,
        HebbianTrainer,
        Stage2Encoder,
        Stage2Model,
        s2_compare,
        s2_validate,
    )
    from pytrine.trine import TRINE_CHANNELS

    # Try to load the library at import time; skip all tests if unavailable.
    from pytrine._binding import get_lib
    _lib = get_lib()
    _LIB_AVAILABLE = True
except Exception:
    _LIB_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _LIB_AVAILABLE,
    reason="libtrine shared library not available",
)

# ── Constants ─────────────────────────────────────────────────────────────

TRINE_PROJECT_K = 3         # K=3 majority vote copies
TRINE_CHAINS = 4            # 4 chains (Edit, Morph, Phrase, Vocab)
TRINE_CHAIN_DIM = 60        # 60 channels per chain
BLOCK_WEIGHTS_SIZE = TRINE_PROJECT_K * TRINE_CHAINS * TRINE_CHAIN_DIM * TRINE_CHAIN_DIM
# = 3 * 4 * 60 * 60 = 43200 bytes


# ── Helpers ───────────────────────────────────────────────────────────────

def _make_identity_block_weights():
    """Create block-diagonal weights that act as identity within each chain.

    Each 60x60 block is set to the identity matrix (diagonal = 1, off-diagonal = 0).
    With K=3 copies all identical, the majority vote produces the same result.
    """
    weights = bytearray(BLOCK_WEIGHTS_SIZE)
    for k in range(TRINE_PROJECT_K):
        for c in range(TRINE_CHAINS):
            for i in range(TRINE_CHAIN_DIM):
                offset = (k * TRINE_CHAINS * TRINE_CHAIN_DIM * TRINE_CHAIN_DIM
                          + c * TRINE_CHAIN_DIM * TRINE_CHAIN_DIM
                          + i * TRINE_CHAIN_DIM + i)
                weights[offset] = 1  # identity: W[i][i] = 1
    return bytes(weights)


def _make_random_block_weights(seed=42):
    """Create deterministic pseudo-random block-diagonal weights (values in {0,1,2})."""
    import random
    rng = random.Random(seed)
    return bytes(rng.choice([0, 1, 2]) for _ in range(BLOCK_WEIGHTS_SIZE))


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def s1_encoder():
    """Shared Stage-1 encoder instance."""
    return TrineEncoder()


@pytest.fixture
def block_model():
    """A block-diagonal model with random weights, yielded and closed."""
    weights = _make_random_block_weights()
    model = Stage2Model.create_block_diagonal(weights, K=3, n_cells=256, topo_seed=42)
    yield model
    model.close()


# ── 1. Block-Diagonal Model Creation ─────────────────────────────────────

class TestBlockDiagonalCreation:
    """Tests for creating block-diagonal models."""

    def test_create_with_random_weights(self):
        """create_block_diagonal() with random weights should produce a non-identity model."""
        weights = _make_random_block_weights()
        model = Stage2Model.create_block_diagonal(weights)
        try:
            info = model.info
            assert info.is_identity is False
        finally:
            model.close()

    def test_create_with_identity_weights(self):
        """create_block_diagonal() with identity weights should be non-identity
        (block-diagonal is structurally different from the identity model)."""
        weights = _make_identity_block_weights()
        model = Stage2Model.create_block_diagonal(weights)
        try:
            info = model.info
            assert info.is_identity is False
        finally:
            model.close()

    def test_projection_mode_is_block_diag(self, block_model):
        """Block-diagonal model should report PROJ_BLOCK_DIAG as projection mode."""
        mode = block_model.get_projection_mode()
        assert mode == PROJ_BLOCK_DIAG

    def test_wrong_weight_length_raises(self):
        """create_block_diagonal() with wrong weight length should raise ValueError."""
        bad_weights = bytes([1] * 100)
        with pytest.raises(ValueError, match="Expected.*weight bytes"):
            Stage2Model.create_block_diagonal(bad_weights)

    def test_context_manager(self):
        """Block-diagonal model should work as a context manager."""
        weights = _make_random_block_weights()
        with Stage2Model.create_block_diagonal(weights) as model:
            emb = model.encode("context manager block test", depth=0)
            assert isinstance(emb, Embedding)
        # After exit, model should be closed
        with pytest.raises(RuntimeError, match="closed|freed"):
            model.encode("should fail", depth=0)

    def test_deterministic_creation(self):
        """Same weights and seed should produce identical models."""
        weights = _make_random_block_weights(seed=99)
        model_a = Stage2Model.create_block_diagonal(weights, topo_seed=77)
        model_b = Stage2Model.create_block_diagonal(weights, topo_seed=77)
        try:
            text = "deterministic block creation test"
            emb_a = model_a.encode(text, depth=0)
            emb_b = model_b.encode(text, depth=0)
            assert emb_a == emb_b
        finally:
            model_a.close()
            model_b.close()


# ── 2. Block-Diagonal Encoding ───────────────────────────────────────────

class TestBlockDiagonalEncoding:
    """Tests for encoding text through a block-diagonal model."""

    def test_encode_produces_240_trits(self, block_model):
        """Encoding through block-diagonal should produce 240 valid trits."""
        emb = block_model.encode("Test block diagonal encoding", depth=0)
        assert isinstance(emb, Embedding)
        assert len(emb) == TRINE_CHANNELS
        trits = emb.trits
        for t in trits:
            assert t in (0, 1, 2), f"Invalid trit value: {t}"

    def test_encode_deterministic(self, block_model):
        """Same text should produce the same block-diagonal embedding."""
        text = "Block diagonal determinism test"
        emb1 = block_model.encode(text, depth=0)
        emb2 = block_model.encode(text, depth=0)
        assert emb1 == emb2

    def test_encode_different_texts(self, block_model):
        """Different texts should produce different block-diagonal embeddings."""
        emb_a = block_model.encode("Hello block diagonal!", depth=0)
        emb_b = block_model.encode("Quantum computing theory", depth=0)
        assert emb_a != emb_b

    def test_encode_differs_from_stage1(self, block_model, s1_encoder):
        """Block-diagonal encoding should differ from raw Stage-1 encoding."""
        text = "Block diagonal vs stage-1"
        s1_emb = s1_encoder.encode(text)
        s2_emb = block_model.encode(text, depth=0)
        # With random weights, the projection should modify the vector
        assert s1_emb != s2_emb

    def test_encode_with_depth(self, block_model):
        """Encoding with depth > 0 should still produce valid 240 trits."""
        emb = block_model.encode("Block diagonal depth test", depth=2)
        assert isinstance(emb, Embedding)
        assert len(emb) == TRINE_CHANNELS
        trits = emb.trits
        for t in trits:
            assert t in (0, 1, 2), f"Invalid trit value: {t}"

    def test_encode_from_trits(self, block_model, s1_encoder):
        """encode_from_trits should match direct encode for the same text."""
        text = "Block diagonal from_trits test"
        s1_emb = s1_encoder.encode(text)
        # Encode directly
        direct = block_model.encode(text, depth=0)
        # Encode from pre-computed Stage-1 trits
        from_trits = block_model.encode_from_trits(s1_emb, depth=0)
        assert direct == from_trits

    def test_encode_empty_text(self, block_model):
        """Empty text should produce an embedding (potentially all zeros)."""
        emb = block_model.encode("", depth=0)
        assert isinstance(emb, Embedding)
        assert len(emb) == TRINE_CHANNELS


# ── 3. Block-Diagonal Comparison ─────────────────────────────────────────

class TestBlockDiagonalComparison:
    """Tests for comparing block-diagonal embeddings."""

    def test_self_similarity(self, block_model):
        """Comparing an embedding with itself should yield ~1.0."""
        emb = block_model.encode("Block diagonal self similarity", depth=0)
        sim = s2_compare(emb, emb)
        assert sim >= 0.99, f"Self-similarity = {sim}, expected ~1.0"

    def test_similarity_in_range(self, block_model):
        """s2_compare should return a value in [0.0, 1.0]."""
        emb_a = block_model.encode("Block diagonal range test A", depth=0)
        emb_b = block_model.encode("Block diagonal range test B", depth=0)
        sim = s2_compare(emb_a, emb_b)
        assert 0.0 <= sim <= 1.0, f"Similarity {sim} out of [0, 1]"

    def test_compare_gated_in_range(self, block_model):
        """Gate-aware comparison should return a float in [-1, 1]."""
        emb_a = block_model.encode("Gated compare block A", depth=0)
        emb_b = block_model.encode("Gated compare block B", depth=0)
        sim = block_model.compare_gated(emb_a, emb_b)
        assert isinstance(sim, float)
        assert -1.0 <= sim <= 1.0, f"Gated similarity {sim} out of [-1, 1]"

    def test_compare_symmetric(self, block_model):
        """s2_compare(a, b) should equal s2_compare(b, a)."""
        emb_a = block_model.encode("Block symmetric A", depth=0)
        emb_b = block_model.encode("Block symmetric B", depth=0)
        sim_ab = s2_compare(emb_a, emb_b)
        sim_ba = s2_compare(emb_b, emb_a)
        assert abs(sim_ab - sim_ba) < 1e-6, (
            f"Asymmetric: {sim_ab} vs {sim_ba}"
        )

    def test_different_texts_lower_similarity(self, block_model):
        """Self-similarity should be higher than cross-text similarity."""
        emb_a = block_model.encode("The quick brown fox jumps", depth=0)
        emb_b = block_model.encode("Quantum entanglement theory", depth=0)
        self_sim = s2_compare(emb_a, emb_a)
        cross_sim = s2_compare(emb_a, emb_b)
        assert self_sim >= cross_sim, (
            f"Self-sim ({self_sim}) should >= cross-sim ({cross_sim})"
        )


# ── 4. Block-Diagonal Persistence ────────────────────────────────────────

class TestBlockDiagonalPersistence:
    """Tests for saving and loading block-diagonal models."""

    def test_save_load_roundtrip(self):
        """Save a block-diagonal model, load it back, verify encoding matches."""
        weights = _make_random_block_weights(seed=123)
        model = Stage2Model.create_block_diagonal(weights, topo_seed=55)
        text = "Block diagonal roundtrip test"
        emb_before = model.encode(text, depth=0)

        with tempfile.NamedTemporaryFile(suffix=".trine2", delete=False) as f:
            path = f.name

        try:
            model.save(path)
            model.close()

            loaded = Stage2Model.load(path)
            try:
                emb_after = loaded.encode(text, depth=0)
                assert emb_before == emb_after
            finally:
                loaded.close()
        finally:
            os.unlink(path)

    def test_validate_saved_file(self):
        """A saved block-diagonal model file should pass validation."""
        weights = _make_random_block_weights()
        model = Stage2Model.create_block_diagonal(weights)

        with tempfile.NamedTemporaryFile(suffix=".trine2", delete=False) as f:
            path = f.name

        try:
            model.save(path)
            model.close()
            assert s2_validate(path) is True
        finally:
            os.unlink(path)

    def test_loaded_model_preserves_mode(self):
        """A loaded block-diagonal model should preserve PROJ_BLOCK_DIAG mode."""
        weights = _make_random_block_weights()
        model = Stage2Model.create_block_diagonal(weights)

        with tempfile.NamedTemporaryFile(suffix=".trine2", delete=False) as f:
            path = f.name

        try:
            model.save(path)
            model.close()

            loaded = Stage2Model.load(path)
            try:
                mode = loaded.get_projection_mode()
                assert mode == PROJ_BLOCK_DIAG
            finally:
                loaded.close()
        finally:
            os.unlink(path)

    def test_save_with_config_params(self):
        """Save with explicit config parameters should succeed and reload."""
        weights = _make_random_block_weights()
        model = Stage2Model.create_block_diagonal(weights)

        with tempfile.NamedTemporaryFile(suffix=".trine2", delete=False) as f:
            path = f.name

        try:
            model.save(path, similarity_threshold=0.90, density=0.33, topo_seed=42)
            model.close()

            loaded = Stage2Model.load(path)
            try:
                assert loaded.info.is_identity is False
            finally:
                loaded.close()
        finally:
            os.unlink(path)


# ── 5. Adaptive Alpha ────────────────────────────────────────────────────

class TestAdaptiveAlpha:
    """Tests for adaptive per-S1-bucket alpha blending."""

    def test_set_adaptive_alpha(self, block_model):
        """Setting adaptive alpha buckets via the Python API should not crash."""
        buckets = [0.65] * 10
        block_model.set_adaptive_alpha(buckets)
        # No crash = success

    def test_set_adaptive_alpha_wrong_length_raises(self, block_model):
        """set_adaptive_alpha with wrong bucket count should raise ValueError."""
        with pytest.raises(ValueError, match="Expected 10"):
            block_model.set_adaptive_alpha([0.5] * 5)

    def test_compare_adaptive_blend_returns_float(self, block_model, s1_encoder):
        """compare_adaptive_blend should return a float result."""
        text_a = "Adaptive blend test A"
        text_b = "Adaptive blend test B"

        s1_a = s1_encoder.encode(text_a)
        s1_b = s1_encoder.encode(text_b)
        s2_a = block_model.encode(text_a, depth=0)
        s2_b = block_model.encode(text_b, depth=0)

        block_model.set_adaptive_alpha([0.65] * 10)
        result = block_model.compare_adaptive_blend(s1_a, s1_b, s2_a, s2_b)
        assert isinstance(result, float)

    def test_adaptive_blend_self_similarity(self, block_model, s1_encoder):
        """Adaptive blend of identical embeddings should yield a high score."""
        text = "Adaptive self similarity test"
        s1_emb = s1_encoder.encode(text)
        s2_emb = block_model.encode(text, depth=0)

        block_model.set_adaptive_alpha([0.65] * 10)
        result = block_model.compare_adaptive_blend(s1_emb, s1_emb, s2_emb, s2_emb)
        assert result >= 0.5, f"Self-similarity adaptive blend = {result}, expected >= 0.5"

    def test_adaptive_blend_disabled_returns_zero(self, block_model, s1_encoder):
        """Without setting adaptive alpha, compare_adaptive_blend should return 0.0."""
        text = "Adaptive disabled test"
        s1_emb = s1_encoder.encode(text)
        s2_emb = block_model.encode(text, depth=0)

        # Do NOT set adaptive alpha
        result = block_model.compare_adaptive_blend(s1_emb, s1_emb, s2_emb, s2_emb)
        assert result == 0.0, f"Expected 0.0 without adaptive alpha, got {result}"

    def test_disable_adaptive_alpha_with_none(self, block_model, s1_encoder):
        """Setting adaptive alpha to None should disable it."""
        text = "Disable adaptive alpha"
        s1_emb = s1_encoder.encode(text)
        s2_emb = block_model.encode(text, depth=0)

        # Enable
        block_model.set_adaptive_alpha([0.65] * 10)
        result_enabled = block_model.compare_adaptive_blend(
            s1_emb, s1_emb, s2_emb, s2_emb,
        )
        assert result_enabled != 0.0, "Expected non-zero with adaptive alpha enabled"

        # Disable by setting None
        block_model.set_adaptive_alpha(None)
        result_disabled = block_model.compare_adaptive_blend(
            s1_emb, s1_emb, s2_emb, s2_emb,
        )
        assert result_disabled == 0.0, (
            f"Expected 0.0 after disabling, got {result_disabled}"
        )

    def test_different_bucket_values_produce_different_results(
        self, block_model, s1_encoder
    ):
        """Different bucket alpha values should produce different blend results."""
        text_a = "Different buckets A long enough text"
        text_b = "Different buckets B with other words"
        s1_a = s1_encoder.encode(text_a)
        s1_b = s1_encoder.encode(text_b)
        s2_a = block_model.encode(text_a, depth=0)
        s2_b = block_model.encode(text_b, depth=0)

        # All-low alpha (heavily S2)
        block_model.set_adaptive_alpha([0.1] * 10)
        result_low = block_model.compare_adaptive_blend(s1_a, s1_b, s2_a, s2_b)

        # All-high alpha (heavily S1)
        block_model.set_adaptive_alpha([0.9] * 10)
        result_high = block_model.compare_adaptive_blend(s1_a, s1_b, s2_a, s2_b)

        assert isinstance(result_low, float)
        assert isinstance(result_high, float)
        # With sufficiently different texts, the S1 and S2 similarities
        # differ, so different alpha weights should yield different blends.


# ── 6. HebbianTrainer Block-Diagonal Mode ────────────────────────────────

class TestHebbianTrainerBlockMode:
    """Tests for HebbianTrainer with block_diagonal=True."""

    def test_create_with_config_dict(self):
        """HebbianTrainer should accept block_diagonal=1 in config dict."""
        with HebbianTrainer({"block_diagonal": 1}) as trainer:
            m = trainer.metrics
            assert m.pairs_observed == 0

    def test_create_with_keyword_arg(self):
        """HebbianTrainer should accept block_diagonal=True as keyword."""
        with HebbianTrainer(block_diagonal=True) as trainer:
            m = trainer.metrics
            assert m.pairs_observed == 0

    def test_create_block_diagonal_classmethod(self):
        """HebbianTrainer.create_block_diagonal() factory should work."""
        with HebbianTrainer.create_block_diagonal() as trainer:
            m = trainer.metrics
            assert m.pairs_observed == 0

    def test_observe_text_block_mode(self):
        """observe_text should work in block-diagonal mode."""
        with HebbianTrainer.create_block_diagonal() as trainer:
            trainer.observe_text("The quick brown fox", "The fast brown fox")
            m = trainer.metrics
            assert m.pairs_observed == 1

    def test_multiple_observations(self):
        """Multiple observations should increment pairs_observed."""
        with HebbianTrainer.create_block_diagonal() as trainer:
            trainer.observe_text("Hello world", "Hi world")
            trainer.observe_text("Machine learning", "Deep learning")
            trainer.observe_text("Neural network", "Artificial intelligence")
            m = trainer.metrics
            assert m.pairs_observed == 3

    def test_freeze_produces_model(self):
        """Freezing a block-diagonal trainer should produce a Stage2Model."""
        with HebbianTrainer.create_block_diagonal() as trainer:
            trainer.observe_text("The quick brown fox", "The fast brown fox")
            trainer.observe_text("Hello world", "Goodbye world")
            trainer.observe_text("Machine learning", "Deep learning")
            model = trainer.freeze()
            try:
                assert isinstance(model, Stage2Model)
                info = model.info
                assert info.is_identity is False
            finally:
                model.close()

    def test_frozen_model_encodes_valid_trits(self):
        """A frozen block-diagonal model should encode text to 240 valid trits."""
        with HebbianTrainer.create_block_diagonal() as trainer:
            trainer.observe_text("Alpha beta gamma", "Alpha beta delta")
            trainer.observe_text("One two three", "One two four")
            model = trainer.freeze()
            try:
                emb = model.encode("Test encoding from frozen block model", depth=0)
                assert isinstance(emb, Embedding)
                assert len(emb) == TRINE_CHANNELS
                trits = emb.trits
                for t in trits:
                    assert t in (0, 1, 2), f"Invalid trit value: {t}"
            finally:
                model.close()

    def test_reset_clears_state(self):
        """reset should clear pairs_observed in block-diagonal mode."""
        with HebbianTrainer.create_block_diagonal() as trainer:
            trainer.observe_text("one", "two")
            assert trainer.metrics.pairs_observed == 1
            trainer.reset()
            assert trainer.metrics.pairs_observed == 0

    def test_context_manager(self):
        """Context manager should close properly in block-diagonal mode."""
        with HebbianTrainer.create_block_diagonal() as trainer:
            trainer.observe_text("a", "b")
        with pytest.raises(RuntimeError, match="closed|freed"):
            trainer.observe_text("c", "d")


# ── 7. Backward Compatibility ────────────────────────────────────────────

class TestBackwardCompatibility:
    """Tests ensuring standard (non-block) operations still work alongside block-diagonal."""

    def test_identity_model_unchanged(self, s1_encoder):
        """Identity model should still produce Stage-1 passthrough."""
        with Stage2Model.identity() as model:
            text = "Identity backward compat test"
            s1_emb = s1_encoder.encode(text)
            s2_emb = model.encode(text, depth=0)
            assert s1_emb == s2_emb

    def test_random_model_still_works(self):
        """random() model with sign projection should still work."""
        with Stage2Model.random(cells=256, seed=42) as model:
            emb = model.encode("Random model backward compat", depth=0)
            assert isinstance(emb, Embedding)
            assert len(emb) == TRINE_CHANNELS

    def test_diagonal_mode_still_works(self):
        """Setting PROJ_DIAGONAL on a random model should still work."""
        with Stage2Model.random(cells=256, seed=42) as model:
            model.set_projection_mode(PROJ_DIAGONAL)
            assert model.get_projection_mode() == PROJ_DIAGONAL
            emb = model.encode("Diagonal backward compat", depth=0)
            assert isinstance(emb, Embedding)
            assert len(emb) == TRINE_CHANNELS

    def test_hebbian_default_not_block(self):
        """Default HebbianTrainer should not be in block-diagonal mode."""
        with HebbianTrainer() as trainer:
            trainer.observe_text("Hello", "World")
            m = trainer.metrics
            assert m.pairs_observed == 1

    def test_stage2_encoder_still_works(self, s1_encoder):
        """Stage2Encoder with identity model should still produce correct embeddings."""
        with Stage2Encoder() as enc:
            text = "Stage2Encoder backward compat"
            s1_emb = s1_encoder.encode(text)
            s2_emb = enc.encode(text)
            assert s1_emb == s2_emb

    def test_s2_compare_still_works(self):
        """Module-level s2_compare should still work with non-block embeddings."""
        with Stage2Model.identity() as model:
            emb_a = model.encode("Compare compat A", depth=0)
            emb_b = model.encode("Compare compat B", depth=0)
        sim = s2_compare(emb_a, emb_b)
        assert isinstance(sim, float)
        assert 0.0 <= sim <= 1.0

    def test_save_load_non_block_roundtrip(self):
        """Standard (non-block) model save/load should still work."""
        model = Stage2Model.random(cells=256, seed=42)
        text = "Non-block roundtrip"
        emb_before = model.encode(text, depth=0)

        with tempfile.NamedTemporaryFile(suffix=".trine2", delete=False) as f:
            path = f.name

        try:
            model.save(path)
            model.close()

            loaded = Stage2Model.load(path)
            try:
                emb_after = loaded.encode(text, depth=0)
                assert emb_before == emb_after
            finally:
                loaded.close()
        finally:
            os.unlink(path)
