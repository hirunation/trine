"""
Comprehensive test suite for pytrine Stage-2 (semantic projection layer).

Tests cover Stage2Model, Stage2Encoder, HebbianTrainer, and module-level
comparison/validation functions.

Requires libtrine.so to be compiled:
    cd bindings/python && make lib
"""

import os
import tempfile

import pytest

from pytrine import Embedding, Lens, TrineEncoder
from pytrine.stage2 import (
    PROJ_DIAGONAL,
    PROJ_SIGN,
    HebbianTrainer,
    S2Info,
    S2Metrics,
    Stage2Encoder,
    Stage2Model,
    s2_compare,
    s2_validate,
)
from pytrine.trine import TRINE_CHANNELS


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def s1_encoder():
    """Shared Stage-1 encoder instance."""
    return TrineEncoder()


@pytest.fixture
def sample_texts():
    """A set of sample texts for testing."""
    return [
        "The quick brown fox jumps over the lazy dog",
        "The quick brown fox leaps over the lazy dog",
        "Quantum computing leverages superposition and entanglement",
        "Hello, world!",
        "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
    ]


# ── TestStage2Model ──────────────────────────────────────────────────────

class TestStage2Model:
    """Tests for Stage2Model: creation, encoding, save/load, introspection."""

    def test_identity_create(self):
        """identity() should create a model with is_identity=True."""
        model = Stage2Model.identity()
        try:
            info = model.info
            assert info.is_identity is True
        finally:
            model.close()

    def test_random_create(self):
        """random() should create a model with is_identity=False."""
        model = Stage2Model.random(cells=256, seed=42)
        try:
            info = model.info
            assert info.is_identity is False
            assert info.cascade_cells > 0
        finally:
            model.close()

    def test_identity_passthrough(self, s1_encoder):
        """Identity model encode should match Stage-1 encode."""
        model = Stage2Model.identity()
        try:
            text = "The quick brown fox"
            s1_emb = s1_encoder.encode(text)
            s2_emb = model.encode(text, depth=0)
            assert s1_emb == s2_emb
        finally:
            model.close()

    def test_encode_deterministic(self):
        """Same text should produce the same embedding twice."""
        model = Stage2Model.random(cells=256, seed=42)
        try:
            text = "Hello, world!"
            emb1 = model.encode(text, depth=0)
            emb2 = model.encode(text, depth=0)
            assert emb1 == emb2
        finally:
            model.close()

    def test_encode_produces_240_trits(self):
        """Encode output should be an Embedding with 240 valid trits (0, 1, 2)."""
        model = Stage2Model.random(cells=256, seed=42)
        try:
            emb = model.encode("Test text for encoding", depth=0)
            assert isinstance(emb, Embedding)
            assert len(emb) == TRINE_CHANNELS
            trits = emb.trits
            for t in trits:
                assert t in (0, 1, 2), f"Invalid trit value: {t}"
        finally:
            model.close()

    def test_encode_different_texts(self):
        """Different texts should produce different embeddings."""
        model = Stage2Model.random(cells=256, seed=42)
        try:
            emb_a = model.encode("Hello, world!", depth=0)
            emb_b = model.encode("Quantum computing theory", depth=0)
            assert emb_a != emb_b
        finally:
            model.close()

    def test_encode_from_trits(self, s1_encoder):
        """encode_from_trits should match encode for the same text."""
        model = Stage2Model.identity()
        try:
            text = "Testing encode_from_trits"
            s1_emb = s1_encoder.encode(text)
            direct = model.encode(text, depth=0)
            from_trits = model.encode_from_trits(s1_emb, depth=0)
            assert direct == from_trits
        finally:
            model.close()

    def test_compare_identity(self):
        """Comparing the same text's embedding with itself should yield ~1.0."""
        model = Stage2Model.identity()
        try:
            emb = model.encode("The quick brown fox", depth=0)
            sim = s2_compare(emb, emb)
            assert sim >= 0.99, f"Self-similarity = {sim}, expected ~1.0"
        finally:
            model.close()

    def test_compare_different(self):
        """Different texts should have lower similarity than self-similarity."""
        model = Stage2Model.identity()
        try:
            emb_a = model.encode("The quick brown fox", depth=0)
            emb_b = model.encode("Quantum computing theory", depth=0)
            sim = s2_compare(emb_a, emb_b)
            assert sim < 0.99, f"Dissimilar text similarity = {sim}, expected < 0.99"
        finally:
            model.close()

    def test_save_load_roundtrip(self):
        """Save to file, load back, verify encoding produces same result."""
        model = Stage2Model.random(cells=256, seed=42)
        text = "Roundtrip test text"
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

    def test_load_nonexistent(self):
        """Loading a nonexistent file should raise IOError."""
        with pytest.raises(IOError):
            Stage2Model.load("/tmp/nonexistent_trine2_model_file.trine2")

    def test_projection_mode(self):
        """set/get projection mode should roundtrip."""
        model = Stage2Model.random(cells=256, seed=42)
        try:
            model.set_projection_mode(PROJ_DIAGONAL)
            assert model.get_projection_mode() == PROJ_DIAGONAL
            model.set_projection_mode(PROJ_SIGN)
            assert model.get_projection_mode() == PROJ_SIGN
        finally:
            model.close()

    def test_context_manager(self):
        """with statement should work and close the model."""
        with Stage2Model.identity() as model:
            emb = model.encode("context manager test", depth=0)
            assert isinstance(emb, Embedding)
        # After exiting, model should be closed
        with pytest.raises(RuntimeError, match="closed|freed"):
            model.encode("should fail", depth=0)

    def test_info_fields(self):
        """info should return an S2Info namedtuple with expected fields."""
        with Stage2Model.random(cells=512, seed=99) as model:
            info = model.info
            assert isinstance(info, S2Info)
            assert hasattr(info, "projection_k")
            assert hasattr(info, "projection_dims")
            assert hasattr(info, "cascade_cells")
            assert hasattr(info, "max_depth")
            assert hasattr(info, "is_identity")
            assert info.projection_dims == TRINE_CHANNELS
            assert info.is_identity is False

    def test_repr_identity(self):
        """repr of identity model should include 'identity'."""
        with Stage2Model.identity() as model:
            r = repr(model)
            assert "identity" in r.lower()

    def test_repr_closed(self):
        """repr of closed model should indicate closed state."""
        model = Stage2Model.identity()
        model.close()
        r = repr(model)
        assert "closed" in r.lower()


# ── TestStage2Encoder ────────────────────────────────────────────────────

class TestStage2Encoder:
    """Tests for Stage2Encoder: high-level encode/compare/blend API."""

    def test_encode(self):
        """encode should produce an Embedding."""
        with Stage2Encoder() as enc:
            emb = enc.encode("Hello, world!")
            assert isinstance(emb, Embedding)
            assert len(emb) == TRINE_CHANNELS

    def test_encode_batch(self):
        """encode_batch should produce a list matching single encodes."""
        with Stage2Encoder() as enc:
            texts = ["Hello", "World", "Test"]
            batch = enc.encode_batch(texts)
            assert isinstance(batch, list)
            assert len(batch) == 3
            for i, text in enumerate(texts):
                single = enc.encode(text)
                assert batch[i] == single

    def test_from_path(self):
        """Construct Stage2Encoder from a .trine2 file path."""
        # Create a model, save it, then load via encoder
        model = Stage2Model.random(cells=256, seed=42)
        with tempfile.NamedTemporaryFile(suffix=".trine2", delete=False) as f:
            path = f.name
        try:
            model.save(path)
            model.close()

            with Stage2Encoder(model=path, depth=0) as enc:
                emb = enc.encode("path test")
                assert isinstance(emb, Embedding)
        finally:
            os.unlink(path)

    def test_from_model(self):
        """Construct Stage2Encoder from a Stage2Model object."""
        model = Stage2Model.identity()
        enc = Stage2Encoder(model=model, depth=0)
        emb = enc.encode("model test")
        assert isinstance(emb, Embedding)
        # Encoder does NOT own the model, so we close model separately
        enc.close()
        # Model should still be alive since encoder doesn't own it
        emb2 = model.encode("still alive", depth=0)
        assert isinstance(emb2, Embedding)
        model.close()

    def test_identity_default(self):
        """No args should use identity model (same as Stage-1)."""
        s1_enc = TrineEncoder()
        with Stage2Encoder() as enc:
            text = "Identity default test"
            s1_emb = s1_enc.encode(text)
            s2_emb = enc.encode(text)
            assert s1_emb == s2_emb

    def test_similarity(self):
        """similarity should return a float in [0, 1]."""
        with Stage2Encoder() as enc:
            sim = enc.similarity("Hello world", "Hello world")
            assert isinstance(sim, float)
            assert 0.0 <= sim <= 1.0
            assert sim >= 0.99, f"Self-text similarity = {sim}"

    def test_blend(self):
        """blend should return a weighted combination of S1 and S2."""
        with Stage2Encoder() as enc:
            blended = enc.blend("Hello", "Hello", alpha=0.65)
            assert isinstance(blended, float)
            assert 0.0 <= blended <= 1.0
            # Same text should have very high blended similarity
            assert blended >= 0.99, f"Same-text blend = {blended}"

    def test_context_manager(self):
        """with statement should work and close the encoder."""
        with Stage2Encoder() as enc:
            emb = enc.encode("context test")
            assert isinstance(emb, Embedding)
        # After close, internal model should be freed
        assert enc._model is None


# ── TestHebbianTrainer ───────────────────────────────────────────────────

class TestHebbianTrainer:
    """Tests for HebbianTrainer: Hebbian learning harness."""

    def test_create_default(self):
        """Create with default config should succeed."""
        with HebbianTrainer() as trainer:
            m = trainer.metrics
            assert m.pairs_observed == 0

    def test_observe(self, s1_encoder):
        """observe() should increment pairs_observed."""
        with HebbianTrainer() as trainer:
            emb_a = s1_encoder.encode("Hello world")
            emb_b = s1_encoder.encode("Hi there")
            sim = emb_a.similarity(emb_b)
            trainer.observe(emb_a, emb_b, sim)
            m = trainer.metrics
            assert m.pairs_observed == 1

    def test_observe_text(self):
        """observe_text should work with raw strings."""
        with HebbianTrainer() as trainer:
            trainer.observe_text("Hello world", "Hi there")
            m = trainer.metrics
            assert m.pairs_observed == 1

    def test_freeze(self):
        """freeze should produce a Stage2Model after observing pairs."""
        with HebbianTrainer() as trainer:
            # Observe a few pairs to accumulate some signal
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

    def test_metrics(self):
        """metrics should return an S2Metrics with expected fields."""
        with HebbianTrainer() as trainer:
            m = trainer.metrics
            assert isinstance(m, S2Metrics)
            assert hasattr(m, "pairs_observed")
            assert hasattr(m, "max_abs_counter")
            assert hasattr(m, "n_positive_weights")
            assert hasattr(m, "n_negative_weights")
            assert hasattr(m, "n_zero_weights")
            assert hasattr(m, "weight_density")
            assert hasattr(m, "effective_threshold")

    def test_reset(self):
        """reset should clear pairs_observed back to 0."""
        with HebbianTrainer() as trainer:
            trainer.observe_text("one", "two")
            assert trainer.metrics.pairs_observed == 1
            trainer.reset()
            assert trainer.metrics.pairs_observed == 0

    def test_train_file(self):
        """train_file should process pairs from a JSONL file."""
        val_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "data", "splits", "val.jsonl"
        )
        val_path = os.path.normpath(val_path)
        if not os.path.isfile(val_path):
            pytest.skip("val.jsonl not found at " + val_path)

        with HebbianTrainer({"similarity_threshold": 0.90, "projection_mode": 1}) as trainer:
            n_pairs = trainer.train_file(val_path, epochs=1)
            assert n_pairs > 0
            m = trainer.metrics
            assert m.pairs_observed > 0

    def test_context_manager(self):
        """with statement should work and close the trainer."""
        with HebbianTrainer() as trainer:
            trainer.observe_text("a", "b")
        with pytest.raises(RuntimeError, match="closed|freed"):
            trainer.observe_text("c", "d")


# ── TestS2Compare ────────────────────────────────────────────────────────

class TestS2Compare:
    """Tests for module-level s2_compare and s2_validate."""

    def test_compare_identical(self):
        """s2_compare with same embedding should return ~1.0."""
        with Stage2Model.identity() as model:
            emb = model.encode("identical test", depth=0)
        sim = s2_compare(emb, emb)
        assert sim >= 0.99, f"Self-compare = {sim}"

    def test_compare_symmetric(self):
        """s2_compare(a, b) should equal s2_compare(b, a)."""
        with Stage2Model.identity() as model:
            emb_a = model.encode("symmetric test A", depth=0)
            emb_b = model.encode("symmetric test B", depth=0)
        sim_ab = s2_compare(emb_a, emb_b)
        sim_ba = s2_compare(emb_b, emb_a)
        assert abs(sim_ab - sim_ba) < 1e-6, f"Asymmetric: {sim_ab} vs {sim_ba}"

    def test_validate(self):
        """s2_validate on a valid .trine2 file should return True."""
        model = Stage2Model.random(cells=256, seed=42)
        with tempfile.NamedTemporaryFile(suffix=".trine2", delete=False) as f:
            path = f.name
        try:
            model.save(path)
            model.close()
            assert s2_validate(path) is True
        finally:
            os.unlink(path)

    def test_validate_nonexistent(self):
        """s2_validate on a nonexistent file should return False."""
        assert s2_validate("/tmp/nonexistent_trine2_file_for_validate.trine2") is False

    def test_validate_invalid_file(self):
        """s2_validate on an invalid file should return False."""
        with tempfile.NamedTemporaryFile(suffix=".trine2", delete=False) as f:
            f.write(b"not a valid trine2 file contents")
            path = f.name
        try:
            assert s2_validate(path) is False
        finally:
            os.unlink(path)
