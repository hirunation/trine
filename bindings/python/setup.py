"""
pytrine setup script.

Compiles the TRINE C library into a shared object (libtrine.so / libtrine.dylib)
and installs it alongside the Python package.
"""

import os
import sys
import subprocess
import shutil
from setuptools import setup
from setuptools.command.build_py import build_py


# Paths relative to this setup.py
HERE = os.path.dirname(os.path.abspath(__file__))
TRINE_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))  # project root
PKG_DIR = os.path.join(HERE, "pytrine")

# Source subdirectories under src/
SRC_DIRS = [
    "encode", "compare", "index", "canon", "algebra", "model",
    "stage2/projection", "stage2/cascade", "stage2/inference",
    "stage2/hebbian", "stage2/persist",
]

# C source files (relative to TRINE_ROOT/src/<subdir>/)
C_SOURCES = [
    "src/encode/trine_encode.c",
    "src/compare/trine_stage1.c",
    "src/index/trine_route.c",
    "src/canon/trine_canon.c",
    "src/compare/trine_csidf.c",
    "src/index/trine_field.c",
    "src/stage2/projection/trine_project.c",
    "src/stage2/projection/trine_majority.c",
    "src/stage2/cascade/trine_learned_cascade.c",
    "src/stage2/cascade/trine_topology_gen.c",
    "src/stage2/inference/trine_stage2.c",
    "src/stage2/hebbian/trine_hebbian.c",
    "src/stage2/hebbian/trine_accumulator.c",
    "src/stage2/hebbian/trine_freeze.c",
    "src/stage2/hebbian/trine_self_deepen.c",
    "src/stage2/persist/trine_s2_persist.c",
]

# Headers needed for compilation (just for dependency tracking)
C_HEADERS = [
    "src/encode/trine_encode.h",
    "src/encode/trine_idf.h",
    "src/compare/trine_stage1.h",
    "src/index/trine_route.h",
    "src/canon/trine_canon.h",
    "src/compare/trine_csidf.h",
    "src/index/trine_field.h",
    "src/stage2/projection/trine_project.h",
    "src/stage2/cascade/trine_learned_cascade.h",
    "src/stage2/inference/trine_stage2.h",
    "src/stage2/hebbian/trine_hebbian.h",
    "src/stage2/hebbian/trine_accumulator.h",
    "src/stage2/hebbian/trine_freeze.h",
    "src/stage2/persist/trine_s2_persist.h",
]


def _lib_name():
    """Return platform-appropriate shared library name."""
    if sys.platform == "darwin":
        return "libtrine.dylib"
    elif sys.platform == "win32":
        return "trine.dll"
    else:
        return "libtrine.so"


def _compile_shared_lib(output_dir):
    """Compile TRINE C sources into a shared library."""
    lib_name = _lib_name()
    output_path = os.path.join(output_dir, lib_name)

    # Check if all source files exist
    src_paths = []
    for src in C_SOURCES:
        full = os.path.join(TRINE_ROOT, src)
        if not os.path.isfile(full):
            raise FileNotFoundError(
                f"TRINE C source not found: {full}\n"
                f"Ensure you are building from the bindings/python/ directory."
            )
        src_paths.append(full)

    # Determine compiler
    cc = os.environ.get("CC", "cc")

    # Include flags for all src/ subdirectories
    include_flags = []
    for d in SRC_DIRS:
        include_flags.extend(["-I", os.path.join(TRINE_ROOT, "src", d)])

    # Build command
    flags = ["-O2", "-Wall", "-fPIC", "-shared"]
    if sys.platform == "darwin":
        flags.append("-dynamiclib")

    cmd = [cc] + flags + include_flags + ["-o", output_path] + src_paths + ["-lm"]

    print(f"[pytrine] Compiling {lib_name}:")
    print(f"  {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[pytrine] Compilation FAILED (exit {result.returncode}):")
        print(result.stderr)
        raise RuntimeError(f"Failed to compile {lib_name}")

    if result.stderr:
        print(f"[pytrine] Compiler warnings:\n{result.stderr}")

    print(f"[pytrine] Built {output_path}")
    return output_path


class BuildWithLib(build_py):
    """Custom build_py that compiles the C shared library first."""

    def run(self):
        # Compile into the package source directory so it ships with the wheel
        os.makedirs(PKG_DIR, exist_ok=True)
        _compile_shared_lib(PKG_DIR)
        super().run()


setup(
    cmdclass={"build_py": BuildWithLib},
    package_data={"pytrine": ["libtrine.so", "libtrine.dylib", "trine.dll"]},
)
