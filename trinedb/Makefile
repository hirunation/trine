# ── TrineDB — TRINE Embedding REST Server ────────────────────────────
# Build:   make
# Clean:   make clean
# ─────────────────────────────────────────────────────────────────────

CC       = cc
CFLAGS   = -O2 -Wall -Wextra -Werror
LDFLAGS  = -lm

TRINE_DIR = ..
TRINE_SRCS = $(TRINE_DIR)/trine_encode.c $(TRINE_DIR)/trine_stage1.c \
             $(TRINE_DIR)/trine_route.c $(TRINE_DIR)/trine_canon.c \
             $(TRINE_DIR)/trine_csidf.c $(TRINE_DIR)/trine_field.c

.PHONY: all clean

all: trinedb

trinedb: trinedb.c $(TRINE_SRCS)
	$(CC) $(CFLAGS) -I$(TRINE_DIR) -o $@ trinedb.c $(TRINE_SRCS) $(LDFLAGS)

clean:
	rm -f trinedb
