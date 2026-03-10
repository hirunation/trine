# ── TRINE Embedding Library ──────────────────────────────────────────
# Standalone build.  No OICOS dependencies.
# Usage:  make            — library + all tools
#         make test        — build + run test harness (Stage-1 + Stage-2)
#         make test_s1     — build + run Stage-1 tests only (163 tests)
#         make test_s2     — build + run Stage-2 tests only
#         make test_s2_p3  — Stage-2 Phase 3 tests (projection/cascade/pipeline)
#         make test_s2_p4  — Stage-2 Phase 4 tests (hebbian/freeze/self-deepen)
#         make bench       — build + run benchmarks (--quick)
#         make clean       — remove build artifacts (preserves .snap/.trine)
# ─────────────────────────────────────────────────────────────────────

CC       = cc
CFLAGS   = -O2 -Wall -Wextra -Werror
LDFLAGS  = -lm
AR       = ar

BUILDDIR = build

# ── Source directories ──────────────────────────────────────────────
# -I flags let source files keep flat #include "trine_*.h" style.

INCLUDES = -Isrc/encode -Isrc/compare -Isrc/index -Isrc/canon \
           -Isrc/algebra -Isrc/model \
           -Isrc/stage2/projection -Isrc/stage2/cascade -Isrc/stage2/inference \
           -Isrc/stage2/hebbian -Isrc/stage2/persist

# ── Library source files ────────────────────────────────────────────

LIB_OBJS = $(BUILDDIR)/trine_encode.o \
           $(BUILDDIR)/trine_stage1.o  \
           $(BUILDDIR)/trine_route.o   \
           $(BUILDDIR)/trine_canon.o   \
           $(BUILDDIR)/trine_csidf.o   \
           $(BUILDDIR)/trine_field.o   \
           $(BUILDDIR)/trine_simd.o    \
           $(BUILDDIR)/trine_batch_compare.o

# ── Stage-2 library objects ─────────────────────────────────────────

S2_OBJS = $(BUILDDIR)/trine_project.o \
          $(BUILDDIR)/trine_majority.o \
          $(BUILDDIR)/trine_learned_cascade.o \
          $(BUILDDIR)/trine_topology_gen.o \
          $(BUILDDIR)/trine_stage2.o

# ── Hebbian (Phase 4) library objects ───────────────────────────────

HEB_OBJS = $(BUILDDIR)/trine_accumulator.o \
           $(BUILDDIR)/trine_freeze.o \
           $(BUILDDIR)/trine_hebbian.o \
           $(BUILDDIR)/trine_self_deepen.o \
           $(BUILDDIR)/trine_jsonl.o

# ── Persistence (Phase 7) library objects ─────────────────────────

PERSIST_OBJS = $(BUILDDIR)/trine_s2_persist.o \
               $(BUILDDIR)/trine_accumulator_persist.o \
               $(BUILDDIR)/trine_pack.o

LIB      = $(BUILDDIR)/libtrine.a

# ── Tool binaries ────────────────────────────────────────────────────

TOOLS = $(BUILDDIR)/trine_embed    \
        $(BUILDDIR)/trine_dedup    \
        $(BUILDDIR)/trine_test_sim \
        $(BUILDDIR)/trine_bench    \
        $(BUILDDIR)/trine_recall   \
        $(BUILDDIR)/trine_corpus_bench \
        $(BUILDDIR)/trine_train

# ── Stage-2 test binaries (Phase 3) ─────────────────────────────────

S2_P3_TESTS = $(BUILDDIR)/test_projection    \
              $(BUILDDIR)/test_majority      \
              $(BUILDDIR)/test_cascade       \
              $(BUILDDIR)/test_full_pipeline

# ── Stage-2 test binaries (Phase 4) ─────────────────────────────────

S2_P4_TESTS = $(BUILDDIR)/test_hebbian       \
              $(BUILDDIR)/test_freeze        \
              $(BUILDDIR)/test_self_deepen   \
              $(BUILDDIR)/test_phase_ab      \
              $(BUILDDIR)/test_sparse

# ── Stage-2 test binaries (Phase 7) ─────────────────────────────────

S2_P7_TESTS = $(BUILDDIR)/test_persistence

S2_BLOCK_TESTS = $(BUILDDIR)/test_block_diagonal \
                 $(BUILDDIR)/test_block_training \
                 $(BUILDDIR)/test_block_persist \
                 $(BUILDDIR)/test_adaptive_alpha \
                 $(BUILDDIR)/test_depth_ensemble

# ── Golden tests ──────────────────────────────────────────────────────

GOLDEN_TESTS = $(BUILDDIR)/test_golden

# ── Integration tests ─────────────────────────────────────────────────

INTEGRATION_TESTS = $(BUILDDIR)/test_backward_compat \
                    $(BUILDDIR)/test_training_pipeline

# ── Benchmark binaries ────────────────────────────────────────────────

BENCH_BINS = $(BUILDDIR)/bench_throughput \
             $(BUILDDIR)/bench_projection \
             $(BUILDDIR)/gen_synthetic

S2_TESTS = $(S2_P3_TESTS) $(S2_P4_TESTS) $(S2_P7_TESTS) $(S2_BLOCK_TESTS)

# ── Phony targets ────────────────────────────────────────────────────

.PHONY: all libtrine.a trine_embed trine_dedup trine_test trine_bench \
        trine_recall trine_corpus_bench trine_train \
        test test_s1 test_s2 test_s2_p3 test_s2_p4 test_s2_p7 test_s2_block \
        test_golden test_integration bench bench_v103 clean

all: $(LIB) $(TOOLS)

libtrine.a: $(LIB)

trine_embed:  $(BUILDDIR)/trine_embed
trine_dedup:  $(BUILDDIR)/trine_dedup
trine_test:   $(BUILDDIR)/trine_test_sim
trine_bench:  $(BUILDDIR)/trine_bench
trine_recall: $(BUILDDIR)/trine_recall
trine_corpus_bench: $(BUILDDIR)/trine_corpus_bench
trine_train:  $(BUILDDIR)/trine_train

test_s1: $(BUILDDIR)/trine_test_sim
	./$(BUILDDIR)/trine_test_sim

test_s2_p3: $(S2_P3_TESTS)
	@echo "=== Running Stage-2 Phase 3 Tests ==="
	./$(BUILDDIR)/test_projection && \
	./$(BUILDDIR)/test_majority && \
	./$(BUILDDIR)/test_cascade && \
	./$(BUILDDIR)/test_full_pipeline

test_s2_p4: $(S2_P4_TESTS)
	@echo "=== Running Stage-2 Phase 4 Tests ==="
	./$(BUILDDIR)/test_hebbian && \
	./$(BUILDDIR)/test_freeze && \
	./$(BUILDDIR)/test_self_deepen && \
	./$(BUILDDIR)/test_phase_ab && \
	./$(BUILDDIR)/test_sparse

test_s2_p7: $(S2_P7_TESTS)
	@echo "=== Running Stage-2 Phase 7 Tests ==="
	./$(BUILDDIR)/test_persistence

test_s2_block: $(S2_BLOCK_TESTS)
	@echo "=== Running Stage-2 Block-Diagonal Tests ==="
	./$(BUILDDIR)/test_block_diagonal && \
	./$(BUILDDIR)/test_block_training && \
	./$(BUILDDIR)/test_block_persist && \
	./$(BUILDDIR)/test_adaptive_alpha && \
	./$(BUILDDIR)/test_depth_ensemble

test_s2: test_s2_p3 test_s2_p4 test_s2_p7 test_s2_block

test_golden: $(GOLDEN_TESTS)
	@echo "=== Running Golden Tests ==="
	./$(BUILDDIR)/test_golden

test_integration: $(INTEGRATION_TESTS)
	@echo "=== Running Integration Tests ==="
	./$(BUILDDIR)/test_backward_compat && \
	./$(BUILDDIR)/test_training_pipeline

test: test_s1 test_s2 test_golden test_integration

bench: $(BUILDDIR)/trine_bench
	./$(BUILDDIR)/trine_bench --quick

bench_v103: $(BENCH_BINS)
	@echo "=== Running v1.0.3 Benchmarks ==="
	./$(BUILDDIR)/bench_throughput && \
	./$(BUILDDIR)/bench_projection

# ── Build directory ──────────────────────────────────────────────────

$(BUILDDIR):
	@mkdir -p $(BUILDDIR)

# ── Static library ───────────────────────────────────────────────────

$(BUILDDIR)/trine_encode.o: src/encode/trine_encode.c src/encode/trine_encode.h \
                            src/encode/trine_idf.h | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ src/encode/trine_encode.c

$(BUILDDIR)/trine_stage1.o: src/compare/trine_stage1.c src/compare/trine_stage1.h \
                            src/encode/trine_encode.h | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ src/compare/trine_stage1.c

$(BUILDDIR)/trine_route.o: src/index/trine_route.c src/index/trine_route.h \
                           src/compare/trine_stage1.h src/compare/trine_csidf.h \
                           src/index/trine_field.h | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ src/index/trine_route.c

$(BUILDDIR)/trine_canon.o: src/canon/trine_canon.c src/canon/trine_canon.h | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ src/canon/trine_canon.c

$(BUILDDIR)/trine_csidf.o: src/compare/trine_csidf.c src/compare/trine_csidf.h | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ src/compare/trine_csidf.c

$(BUILDDIR)/trine_field.o: src/index/trine_field.c src/index/trine_field.h \
                           src/encode/trine_encode.h | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ src/index/trine_field.c

$(BUILDDIR)/trine_simd.o: src/compare/trine_simd.c src/compare/trine_simd.h | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -msse2 -c -o $@ src/compare/trine_simd.c

$(BUILDDIR)/trine_batch_compare.o: src/compare/trine_batch_compare.c \
                                    src/compare/trine_batch_compare.h \
                                    src/compare/trine_stage1.h | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ src/compare/trine_batch_compare.c

# ── Stage-2 library objects ──────────────────────────────────────────

$(BUILDDIR)/trine_project.o: src/stage2/projection/trine_project.c \
                              src/stage2/projection/trine_project.h | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ src/stage2/projection/trine_project.c

$(BUILDDIR)/trine_majority.o: src/stage2/projection/trine_majority.c \
                               src/stage2/projection/trine_project.h | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ src/stage2/projection/trine_majority.c

$(BUILDDIR)/trine_learned_cascade.o: src/stage2/cascade/trine_learned_cascade.c \
                                      src/stage2/cascade/trine_learned_cascade.h \
                                      src/algebra/trine_algebra.h | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ src/stage2/cascade/trine_learned_cascade.c

$(BUILDDIR)/trine_topology_gen.o: src/stage2/cascade/trine_topology_gen.c \
                                   src/stage2/cascade/trine_learned_cascade.h | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ src/stage2/cascade/trine_topology_gen.c

$(BUILDDIR)/trine_stage2.o: src/stage2/inference/trine_stage2.c \
                             src/stage2/inference/trine_stage2.h \
                             src/stage2/projection/trine_project.h \
                             src/stage2/cascade/trine_learned_cascade.h \
                             src/encode/trine_encode.h \
                             src/compare/trine_stage1.h | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ src/stage2/inference/trine_stage2.c

# ── Hebbian (Phase 4) library objects ────────────────────────────────

$(BUILDDIR)/trine_accumulator.o: src/stage2/hebbian/trine_accumulator.c \
                                  src/stage2/hebbian/trine_accumulator.h | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ src/stage2/hebbian/trine_accumulator.c

$(BUILDDIR)/trine_freeze.o: src/stage2/hebbian/trine_freeze.c \
                             src/stage2/hebbian/trine_freeze.h \
                             src/stage2/hebbian/trine_accumulator.h | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ src/stage2/hebbian/trine_freeze.c

$(BUILDDIR)/trine_hebbian.o: src/stage2/hebbian/trine_hebbian.c \
                              src/stage2/hebbian/trine_hebbian.h \
                              src/stage2/hebbian/trine_accumulator.h \
                              src/stage2/hebbian/trine_freeze.h \
                              src/stage2/inference/trine_stage2.h \
                              src/encode/trine_encode.h \
                              src/compare/trine_stage1.h | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ src/stage2/hebbian/trine_hebbian.c

$(BUILDDIR)/trine_self_deepen.o: src/stage2/hebbian/trine_self_deepen.c \
                                  src/stage2/hebbian/trine_hebbian.h \
                                  src/stage2/inference/trine_stage2.h | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ src/stage2/hebbian/trine_self_deepen.c

$(BUILDDIR)/trine_jsonl.o: src/stage2/hebbian/trine_jsonl.c \
                            src/stage2/hebbian/trine_jsonl.h | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ src/stage2/hebbian/trine_jsonl.c

$(BUILDDIR)/trine_s2_persist.o: src/stage2/persist/trine_s2_persist.c \
                                 src/stage2/persist/trine_s2_persist.h \
                                 src/stage2/inference/trine_stage2.h \
                                 src/stage2/projection/trine_project.h | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ src/stage2/persist/trine_s2_persist.c

$(BUILDDIR)/trine_accumulator_persist.o: src/stage2/persist/trine_accumulator_persist.c \
                                          src/stage2/persist/trine_accumulator_persist.h \
                                          src/stage2/hebbian/trine_accumulator.h \
                                          src/stage2/projection/trine_project.h | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ src/stage2/persist/trine_accumulator_persist.c

$(BUILDDIR)/trine_pack.o: src/stage2/persist/trine_pack.c \
                           src/stage2/persist/trine_pack.h | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ src/stage2/persist/trine_pack.c

$(LIB): $(LIB_OBJS) $(S2_OBJS) $(HEB_OBJS) $(PERSIST_OBJS)
	$(AR) rcs $@ $(LIB_OBJS) $(S2_OBJS) $(HEB_OBJS) $(PERSIST_OBJS)

# ── Tool binaries ────────────────────────────────────────────────────

$(BUILDDIR)/trine_embed: src/tools/trine_embed.c src/model/trine.c \
                         src/algebra/trine_format.c $(LIB) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -Isrc/tools -o $@ src/tools/trine_embed.c src/model/trine.c src/algebra/trine_format.c $(LIB) $(LDFLAGS)

$(BUILDDIR)/trine_dedup: src/tools/trine_dedup.c $(LIB) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ src/tools/trine_dedup.c $(LIB) $(LDFLAGS)

$(BUILDDIR)/trine_test_sim: src/tools/trine_test_sim.c $(LIB) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ src/tools/trine_test_sim.c $(LIB) $(LDFLAGS)

$(BUILDDIR)/trine_bench: src/tools/trine_bench.c $(LIB) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ src/tools/trine_bench.c $(LIB) $(LDFLAGS)

$(BUILDDIR)/trine_recall: src/tools/trine_recall.c $(LIB) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ src/tools/trine_recall.c $(LIB) $(LDFLAGS)

$(BUILDDIR)/trine_corpus_bench: bench/legacy/trine_corpus_bench.c $(LIB) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ bench/legacy/trine_corpus_bench.c $(LIB) $(LDFLAGS)

$(BUILDDIR)/trine_train: src/tools/trine_train.c $(LIB) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ src/tools/trine_train.c $(LIB) $(LDFLAGS)

# ── Stage-2 test binaries (Phase 3) ─────────────────────────────────

$(BUILDDIR)/test_projection: tests/stage2/test_projection.c $(LIB) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ tests/stage2/test_projection.c $(LIB) $(LDFLAGS)

$(BUILDDIR)/test_majority: tests/stage2/test_majority.c $(LIB) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ tests/stage2/test_majority.c $(LIB) $(LDFLAGS)

$(BUILDDIR)/test_cascade: tests/stage2/test_cascade.c $(LIB) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ tests/stage2/test_cascade.c $(LIB) $(LDFLAGS)

$(BUILDDIR)/test_full_pipeline: tests/stage2/test_full_pipeline.c $(LIB) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ tests/stage2/test_full_pipeline.c $(LIB) $(LDFLAGS)

# ── Stage-2 test binaries (Phase 4) ─────────────────────────────────

$(BUILDDIR)/test_hebbian: tests/stage2/test_hebbian.c $(LIB) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ tests/stage2/test_hebbian.c $(LIB) $(LDFLAGS)

$(BUILDDIR)/test_freeze: tests/stage2/test_freeze.c $(LIB) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ tests/stage2/test_freeze.c $(LIB) $(LDFLAGS)

$(BUILDDIR)/test_self_deepen: tests/stage2/test_self_deepen.c $(LIB) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ tests/stage2/test_self_deepen.c $(LIB) $(LDFLAGS)

$(BUILDDIR)/test_phase_ab: tests/stage2/test_phase_ab.c $(LIB) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ tests/stage2/test_phase_ab.c $(LIB) $(LDFLAGS)

$(BUILDDIR)/test_sparse: tests/stage2/test_sparse.c $(LIB) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ tests/stage2/test_sparse.c $(LIB) $(LDFLAGS)

# ── Stage-2 test binaries (Phase 7) ─────────────────────────────────

$(BUILDDIR)/test_persistence: tests/stage2/test_persistence.c $(LIB) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ tests/stage2/test_persistence.c $(LIB) $(LDFLAGS)

# ── Stage-2 test binaries (Block-Diagonal) ────────────────────────────

$(BUILDDIR)/test_block_diagonal: tests/stage2/test_block_diagonal.c $(LIB) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ tests/stage2/test_block_diagonal.c $(LIB) $(LDFLAGS)

$(BUILDDIR)/test_block_training: tests/stage2/test_block_training.c $(LIB) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ tests/stage2/test_block_training.c $(LIB) $(LDFLAGS)

$(BUILDDIR)/test_block_persist: tests/stage2/test_block_persist.c $(LIB) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ tests/stage2/test_block_persist.c $(LIB) $(LDFLAGS)

$(BUILDDIR)/test_adaptive_alpha: tests/stage2/test_adaptive_alpha.c $(LIB) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ tests/stage2/test_adaptive_alpha.c $(LIB) $(LDFLAGS)

$(BUILDDIR)/test_depth_ensemble: tests/stage2/test_depth_ensemble.c $(LIB) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ tests/stage2/test_depth_ensemble.c $(LIB) $(LDFLAGS)

# ── Golden tests ──────────────────────────────────────────────────────

$(BUILDDIR)/test_golden: tests/golden/test_golden.c $(LIB) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ tests/golden/test_golden.c $(LIB) $(LDFLAGS)

# ── Integration tests ─────────────────────────────────────────────────

$(BUILDDIR)/test_backward_compat: tests/integration/test_backward_compat.c $(LIB) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ tests/integration/test_backward_compat.c $(LIB) $(LDFLAGS)

$(BUILDDIR)/test_training_pipeline: tests/integration/test_training_pipeline.c $(LIB) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ tests/integration/test_training_pipeline.c $(LIB) $(LDFLAGS)

# ── Benchmark binaries ────────────────────────────────────────────────

$(BUILDDIR)/bench_throughput: bench/harness/bench_throughput.c $(LIB) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ bench/harness/bench_throughput.c $(LIB) $(LDFLAGS)

$(BUILDDIR)/bench_projection: bench/harness/bench_projection.c $(LIB) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ bench/harness/bench_projection.c $(LIB) $(LDFLAGS)

$(BUILDDIR)/gen_synthetic: bench/harness/gen_synthetic.c | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ bench/harness/gen_synthetic.c

# ── Clean (preserves .snap and .trine files) ─────────────────────────

clean:
	rm -f $(BUILDDIR)/*.o $(BUILDDIR)/*.a
	rm -f $(BUILDDIR)/trine_embed $(BUILDDIR)/trine_dedup
	rm -f $(BUILDDIR)/trine_test_sim $(BUILDDIR)/trine_bench
	rm -f $(BUILDDIR)/trine_recall $(BUILDDIR)/trine_pack
	rm -f $(BUILDDIR)/trine_corpus_bench $(BUILDDIR)/trine_train
	rm -f $(BUILDDIR)/test_projection $(BUILDDIR)/test_majority
	rm -f $(BUILDDIR)/test_cascade $(BUILDDIR)/test_full_pipeline
	rm -f $(BUILDDIR)/test_hebbian $(BUILDDIR)/test_freeze
	rm -f $(BUILDDIR)/test_self_deepen
	rm -f $(BUILDDIR)/test_phase_ab
	rm -f $(BUILDDIR)/test_sparse
	rm -f $(BUILDDIR)/test_persistence
	rm -f $(BUILDDIR)/test_block_diagonal $(BUILDDIR)/test_block_training
	rm -f $(BUILDDIR)/test_block_persist $(BUILDDIR)/test_adaptive_alpha
	rm -f $(BUILDDIR)/test_depth_ensemble
	rm -f $(BUILDDIR)/test_golden
	rm -f $(BUILDDIR)/test_backward_compat $(BUILDDIR)/test_training_pipeline
	rm -f $(BUILDDIR)/bench_throughput $(BUILDDIR)/bench_projection
	rm -f $(BUILDDIR)/gen_synthetic
