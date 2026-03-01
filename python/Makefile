# ── pytrine build ────────────────────────────────────────────────────────
# Usage:  make lib    — compile libtrine.so from C sources
#         make test   — run pytest suite
#         make wheel  — build distributable wheel
#         make clean  — remove build artifacts
# ────────────────────────────────────────────────────────────────────────

CC       = cc
CFLAGS   = -O2 -Wall -fPIC
HTEB_DIR = ..

# C sources (relative to HTEB_DIR)
C_SRCS = $(HTEB_DIR)/trine_encode.c \
         $(HTEB_DIR)/trine_stage1.c \
         $(HTEB_DIR)/trine_route.c  \
         $(HTEB_DIR)/trine_canon.c  \
         $(HTEB_DIR)/trine_csidf.c  \
         $(HTEB_DIR)/trine_field.c

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
	$(CC) $(CFLAGS) $(SHARED_FLAGS) -I$(HTEB_DIR) -o $@ $(C_SRCS) -lm
	@echo "[pytrine] Built $@ ($$(stat -c%s $@ 2>/dev/null || stat -f%z $@) bytes)"

test: lib
	python3 -m pytest tests/ -v

wheel: lib
	pip wheel . --no-build-isolation -w dist/

clean:
	rm -f pytrine/libtrine.so pytrine/libtrine.dylib pytrine/trine.dll
	rm -rf build/ dist/ *.egg-info pytrine/__pycache__ tests/__pycache__
	rm -rf pytrine/*.pyc tests/*.pyc
