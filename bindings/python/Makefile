# ── pytrine build ────────────────────────────────────────────────────────
# Usage:  make lib    — compile libtrine.so from C sources
#         make test   — run pytest suite
#         make wheel  — build distributable wheel
#         make clean  — remove build artifacts
# ────────────────────────────────────────────────────────────────────────

CC       = cc
CFLAGS   = -O2 -Wall -fPIC
TRINE_ROOT = ../..

# Include paths for flat #include "trine_*.h" resolution
INCLUDES = -I$(TRINE_ROOT)/src/encode  -I$(TRINE_ROOT)/src/compare \
           -I$(TRINE_ROOT)/src/index   -I$(TRINE_ROOT)/src/canon   \
           -I$(TRINE_ROOT)/src/algebra -I$(TRINE_ROOT)/src/model   \
           -I$(TRINE_ROOT)/src/stage2/projection \
           -I$(TRINE_ROOT)/src/stage2/cascade    \
           -I$(TRINE_ROOT)/src/stage2/inference  \
           -I$(TRINE_ROOT)/src/stage2/hebbian    \
           -I$(TRINE_ROOT)/src/stage2/persist

# C sources (relative to TRINE_ROOT)
C_SRCS = $(TRINE_ROOT)/src/encode/trine_encode.c  \
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
         $(TRINE_ROOT)/src/stage2/hebbian/trine_hebbian.c      \
         $(TRINE_ROOT)/src/stage2/hebbian/trine_accumulator.c  \
         $(TRINE_ROOT)/src/stage2/hebbian/trine_freeze.c       \
         $(TRINE_ROOT)/src/stage2/hebbian/trine_self_deepen.c  \
         $(TRINE_ROOT)/src/stage2/persist/trine_s2_persist.c

UNAME := $(shell uname -s)
ifeq ($(UNAME),Darwin)
  LIB_NAME = libtrine.dylib
  SHARED_FLAGS = -shared -dynamiclib
else ifeq ($(OS),Windows_NT)
  LIB_NAME = trine.dll
  SHARED_FLAGS = -shared
else
  LIB_NAME = libtrine.so
  SHARED_FLAGS = -shared
endif

LIB_OUT = pytrine/$(LIB_NAME)

.PHONY: all lib test wheel clean

all: lib

lib: $(LIB_OUT)

$(LIB_OUT): $(C_SRCS)
	@mkdir -p pytrine
	$(CC) $(CFLAGS) $(SHARED_FLAGS) $(INCLUDES) -o $@ $(C_SRCS) -lm
	@echo "[pytrine] Built $@ ($$(stat -c%s $@ 2>/dev/null || stat -f%z $@) bytes)"

test: lib
	python3 -m pytest tests/ -v

wheel: lib
	pip wheel . --no-build-isolation -w dist/

clean:
	rm -f pytrine/libtrine.so pytrine/libtrine.dylib pytrine/trine.dll
	rm -rf build/ dist/ *.egg-info pytrine/__pycache__ tests/__pycache__
	rm -rf pytrine/*.pyc tests/*.pyc
