# ── TrineDB — TRINE Embedding REST Server ────────────────────────────
# Build:   make
# Clean:   make clean
# ─────────────────────────────────────────────────────────────────────

CC       = cc
CFLAGS   = -O2 -Wall -Wextra -Werror
LDFLAGS  = -lm

TRINE_ROOT = ../..
INCLUDES = -I$(TRINE_ROOT)/src/encode  -I$(TRINE_ROOT)/src/compare \
           -I$(TRINE_ROOT)/src/index   -I$(TRINE_ROOT)/src/canon   \
           -I$(TRINE_ROOT)/src/algebra -I$(TRINE_ROOT)/src/model   \
           -I$(TRINE_ROOT)/src/stage2/projection \
           -I$(TRINE_ROOT)/src/stage2/cascade    \
           -I$(TRINE_ROOT)/src/stage2/inference  \
           -I$(TRINE_ROOT)/src/stage2/persist

TRINE_SRCS = $(TRINE_ROOT)/src/encode/trine_encode.c  \
             $(TRINE_ROOT)/src/compare/trine_stage1.c  \
             $(TRINE_ROOT)/src/index/trine_route.c     \
             $(TRINE_ROOT)/src/canon/trine_canon.c     \
             $(TRINE_ROOT)/src/compare/trine_csidf.c   \
             $(TRINE_ROOT)/src/index/trine_field.c     \
             $(TRINE_ROOT)/src/stage2/projection/trine_project.c   \
             $(TRINE_ROOT)/src/stage2/projection/trine_majority.c  \
             $(TRINE_ROOT)/src/stage2/cascade/trine_learned_cascade.c \
             $(TRINE_ROOT)/src/stage2/cascade/trine_topology_gen.c \
             $(TRINE_ROOT)/src/stage2/inference/trine_stage2.c     \
             $(TRINE_ROOT)/src/stage2/persist/trine_s2_persist.c   \
             $(TRINE_ROOT)/src/stage2/persist/trine_pack.c

.PHONY: all clean

all: trinedb

trinedb: trinedb.c $(TRINE_SRCS)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ trinedb.c $(TRINE_SRCS) $(LDFLAGS)

clean:
	rm -f trinedb
