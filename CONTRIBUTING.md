# Contributing

Changes are proposed via pull request against `main`. Every PR must include a clear description of what changed and why, updated or new tests that exercise the change, and a clean build under `make clean && make all` with zero warnings (`-Werror`). PRs that reduce test coverage, introduce external dependencies, or break deterministic output will not be merged.

The technical standard is simple: the system must be at least as correct, at least as fast, and no larger after your change than before it. Benchmark regressions require justification. New public API surface requires documentation, a test, and a reason that existing API cannot serve the need. Refactors that do not change behavior are welcome when they reduce complexity.

Review exists to preserve coherence, not to gatekeep. If your change strengthens the architecture — tightens a bound, removes a branch, eliminates a dependency — it will be recognized and integrated. Evolution that concentrates the system is valued above evolution that expands it.
