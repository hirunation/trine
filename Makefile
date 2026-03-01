# ── TRINE Embedding Library ──────────────────────────────────────────
# Standalone build.  No OICOS dependencies.
# Usage:  make            — library + all tools
#         make test        — build + run test harness
#         make bench       — build + run benchmarks (--quick)
#         make clean       — remove build artifacts (preserves .snap/.trine)
# ─────────────────────────────────────────────────────────────────────

CC       = cc
CFLAGS   = -O2 -Wall -Wextra -Werror
LDFLAGS  = -lm
AR       = ar

BUILDDIR = build

# ── Library objects ──────────────────────────────────────────────────

LIB_OBJS = $(BUILDDIR)/trine_encode.o \
           $(BUILDDIR)/trine_stage1.o  \
           $(BUILDDIR)/trine_route.o   \
           $(BUILDDIR)/trine_canon.o   \
           $(BUILDDIR)/trine_csidf.o   \
           $(BUILDDIR)/trine_field.o

LIB      = $(BUILDDIR)/libtrine.a

# ── Tool binaries ────────────────────────────────────────────────────

TOOLS = $(BUILDDIR)/trine_embed    \
        $(BUILDDIR)/trine_dedup    \
        $(BUILDDIR)/trine_test_sim \
        $(BUILDDIR)/trine_bench    \
        $(BUILDDIR)/trine_recall   \
        $(BUILDDIR)/trine_corpus_bench

# ── Phony targets ────────────────────────────────────────────────────

.PHONY: all libtrine.a trine_embed trine_dedup trine_test trine_bench \
        trine_recall trine_corpus_bench test bench clean

all: $(LIB) $(TOOLS)

libtrine.a: $(LIB)

trine_embed:  $(BUILDDIR)/trine_embed
trine_dedup:  $(BUILDDIR)/trine_dedup
trine_test:   $(BUILDDIR)/trine_test_sim
trine_bench:  $(BUILDDIR)/trine_bench
trine_recall: $(BUILDDIR)/trine_recall
trine_corpus_bench: $(BUILDDIR)/trine_corpus_bench

test: $(BUILDDIR)/trine_test_sim
	./$(BUILDDIR)/trine_test_sim

bench: $(BUILDDIR)/trine_bench
	./$(BUILDDIR)/trine_bench --quick

# ── Build directory ──────────────────────────────────────────────────

$(BUILDDIR):
	@mkdir -p $(BUILDDIR)

# ── Static library ───────────────────────────────────────────────────

$(BUILDDIR)/trine_encode.o: trine_encode.c trine_encode.h trine_idf.h | $(BUILDDIR)
	$(CC) $(CFLAGS) -c -o $@ trine_encode.c

$(BUILDDIR)/trine_stage1.o: trine_stage1.c trine_stage1.h trine_encode.h | $(BUILDDIR)
	$(CC) $(CFLAGS) -c -o $@ trine_stage1.c

$(BUILDDIR)/trine_route.o: trine_route.c trine_route.h trine_stage1.h \
                           trine_csidf.h trine_field.h | $(BUILDDIR)
	$(CC) $(CFLAGS) -c -o $@ trine_route.c

$(BUILDDIR)/trine_canon.o: trine_canon.c trine_canon.h | $(BUILDDIR)
	$(CC) $(CFLAGS) -c -o $@ trine_canon.c

$(BUILDDIR)/trine_csidf.o: trine_csidf.c trine_csidf.h | $(BUILDDIR)
	$(CC) $(CFLAGS) -c -o $@ trine_csidf.c

$(BUILDDIR)/trine_field.o: trine_field.c trine_field.h trine_encode.h | $(BUILDDIR)
	$(CC) $(CFLAGS) -c -o $@ trine_field.c

$(LIB): $(LIB_OBJS)
	$(AR) rcs $@ $(LIB_OBJS)

# ── Tool binaries ────────────────────────────────────────────────────

$(BUILDDIR)/trine_embed: trine_embed.c trine.c trine_format.c \
                         trine_encode.c trine.h trine_format.h \
                         trine_algebra.h trine_encode.h trine_idf.h | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ trine_embed.c trine.c trine_format.c trine_encode.c $(LDFLAGS)

$(BUILDDIR)/trine_dedup: trine_dedup.c trine_encode.c trine_stage1.c \
                         trine_route.c trine_csidf.c trine_field.c \
                         trine_encode.h trine_idf.h \
                         trine_stage1.h trine_route.h \
                         trine_csidf.h trine_field.h | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ trine_dedup.c trine_encode.c trine_stage1.c trine_route.c trine_csidf.c trine_field.c $(LDFLAGS)

$(BUILDDIR)/trine_test_sim: trine_test_sim.c trine_encode.c trine_stage1.c \
                            trine_canon.c trine_route.c trine_csidf.c trine_field.c \
                            trine_encode.h trine_idf.h \
                            trine_stage1.h trine_canon.h \
                            trine_route.h trine_csidf.h trine_field.h | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ trine_test_sim.c trine_encode.c trine_stage1.c trine_canon.c trine_route.c trine_csidf.c trine_field.c $(LDFLAGS)

$(BUILDDIR)/trine_bench: trine_bench.c trine_encode.c trine_stage1.c \
                         trine_route.c trine_csidf.c trine_field.c \
                         trine_encode.h trine_idf.h \
                         trine_stage1.h trine_route.h \
                         trine_csidf.h trine_field.h | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ trine_bench.c trine_encode.c trine_stage1.c trine_route.c trine_csidf.c trine_field.c $(LDFLAGS)

$(BUILDDIR)/trine_recall: trine_recall.c trine_encode.c trine_stage1.c \
                          trine_route.c trine_csidf.c trine_field.c \
                          trine_encode.h trine_stage1.h \
                          trine_route.h trine_csidf.h trine_field.h | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ trine_recall.c trine_encode.c trine_stage1.c trine_route.c trine_csidf.c trine_field.c $(LDFLAGS)

$(BUILDDIR)/trine_corpus_bench: benchmarks/trine_corpus_bench.c trine_encode.c \
                                trine_stage1.c trine_route.c trine_canon.c \
                                trine_csidf.c trine_field.c \
                                trine_encode.h trine_stage1.h trine_route.h \
                                trine_canon.h trine_csidf.h trine_field.h | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ benchmarks/trine_corpus_bench.c trine_encode.c trine_stage1.c trine_route.c trine_canon.c trine_csidf.c trine_field.c $(LDFLAGS)

# ── Clean (preserves .snap and .trine files) ─────────────────────────

clean:
	rm -f $(BUILDDIR)/*.o $(BUILDDIR)/*.a
	rm -f $(BUILDDIR)/trine_embed $(BUILDDIR)/trine_dedup
	rm -f $(BUILDDIR)/trine_test_sim $(BUILDDIR)/trine_bench
	rm -f $(BUILDDIR)/trine_recall $(BUILDDIR)/trine_pack
	rm -f $(BUILDDIR)/trine_corpus_bench
