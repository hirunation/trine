# Contributing

Changes are proposed via pull request against `main`. Every PR must include a clear description of what changed and why, updated or new tests that exercise the change, and a clean build under `make clean && make all` with zero warnings (`-Werror`). PRs that reduce test coverage, introduce external dependencies, or break deterministic output will not be merged.

The technical standard is simple: the system must be at least as correct, at least as fast, and no larger after your change than before it. Benchmark regressions require justification. New public API surface requires documentation, a test, and a reason that existing API cannot serve the need.

## Build

```bash
make                   # library + all tools (Stage-1 + Stage-2)
make test              # all tests (S1 + S2 + golden + integration)
make test_s1           # Stage-1 tests only
make test_s2           # Stage-2 tests (phases 3, 4, 7, block-diagonal)
make test_s2_block     # block-diagonal projection tests only
make test_golden       # golden determinism tests
make test_integration  # backward-compat + training pipeline tests
make bench             # throughput benchmarks (--quick)
make bench_v103        # v1.0.3 benchmarks (throughput + projection)
make clean             # remove artifacts (preserves .snap/.trine files)
```

Requires only a C99 compiler and `make`. No external dependencies.

## Test Conventions

Every test file follows the same pattern:

```c
static int g_passed = 0;
static int g_failed = 0;
static int g_total  = 0;

static void check(const char *name, int cond)
{
    g_total++;
    if (cond) {
        g_passed++;
    } else {
        g_failed++;
        printf("  FAIL  category: %s\n", name);
    }
}
```

- `main()` prints a category header (`=== Category Tests ===`), calls test functions, prints a summary line (`Category: %d passed, %d failed, %d total`), and returns `g_failed`.
- Each `check()` call is one assertion. Prefix the name with the subsystem being tested.
- Tests must be deterministic. Use fixed seeds for random projections.
- Test files live in `tests/stage2/`, `tests/golden/`, or `tests/integration/`.

## Code Style

- C99. No C++ or GNU extensions required.
- Compiled with `-O2 -Wall -Wextra -Werror`. Zero warnings.
- Flat includes: `#include "trine_project.h"`, not `#include "stage2/projection/trine_project.h"`. The Makefile `-I` flags resolve paths.
- No external dependencies. The only link flag is `-lm`.
- Header guards: `#ifndef TRINE_FOO_H` / `#define TRINE_FOO_H` / `#endif`.
- Functions prefixed `trine_`. Types suffixed `_t`. Constants prefixed `TRINE_`.

## How to Add a New Projection Mode

1. Declare the function in `src/stage2/projection/trine_project.h`:
   ```c
   void trine_project_mymode(const uint8_t W[TRINE_PROJECT_DIM][TRINE_PROJECT_DIM],
                              const uint8_t x[TRINE_PROJECT_DIM],
                              uint8_t y[TRINE_PROJECT_DIM]);
   ```
2. Implement in `src/stage2/projection/trine_project.c` (or a new `.c` in the same directory). All outputs must be in {0,1,2}.
3. If it uses K=3 majority vote, add a `trine_project_majority_mymode()` variant.
4. Add tests in `tests/stage2/test_projection.c` or a new test file. At minimum: Z3 closure, identity pass-through, determinism.
5. Wire into the cascade in `src/stage2/inference/trine_stage2.c` if it should be selectable at runtime.
6. Add a Makefile rule if you created a new `.c` file or test binary.

## How to Add a New Test

1. Create `tests/<category>/test_foo.c` with the `g_passed`/`g_failed`/`g_total` + `check()` pattern.
2. Add a build rule in the Makefile following existing patterns:
   ```makefile
   $(BUILDDIR)/test_foo: tests/category/test_foo.c $(LIB) | $(BUILDDIR)
   	$(CC) $(CFLAGS) $(INCLUDES) -o $@ tests/category/test_foo.c $(LIB) $(LDFLAGS)
   ```
3. Add the binary to the appropriate test list variable (`S2_BLOCK_TESTS`, `GOLDEN_TESTS`, `INTEGRATION_TESTS`, etc.).
4. Add a `rm -f` line to the `clean` target.
5. Verify: `make clean && make test` must pass with zero warnings.

## How to Update Bindings

Bindings wrap the C core via FFI. The C API is the source of truth.

**Python** (`bindings/python/pytrine/`):
- `_binding.py` defines ctypes function signatures matching the C headers.
- `stage2.py` wraps Stage-2 (Stage2Model, Stage2Encoder, HebbianTrainer).
- Run `python -m pytest bindings/python/tests/` after changes.

**Rust** (`bindings/rust/src/`):
- `ffi.rs` declares `extern "C"` bindings matching the C headers.
- `stage2.rs` wraps Stage-2 types.
- Run `cargo test` from `bindings/rust/` after changes.

When the C API changes: update the FFI declarations first, then the wrapper types, then the tests.

## Release Checklist

1. `make clean && make all` -- zero warnings.
2. `make test` -- all C tests pass (S1 + S2 + golden + integration).
3. `make bench` -- no throughput regressions vs. previous release.
4. Run Python tests: `cd bindings/python && python -m pytest`.
5. Run Rust tests: `cd bindings/rust && cargo test`.
6. Verify golden tests match expected snapshots.
7. Update version string in headers, CLAUDE.md, and README if present.
8. Tag the release: `git tag v1.0.X`.
