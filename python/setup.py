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
HTEB_DIR = os.path.dirname(HERE)  # parent hteb/ directory with C sources
PKG_DIR = os.path.join(HERE, "pytrine")

# C source files (relative to HTEB_DIR)
C_SOURCES = [
    "trine_encode.c",
    "trine_stage1.c",
    "trine_route.c",
    "trine_canon.c",
]

# Headers needed for compilation (just for dependency tracking)
C_HEADERS = [
    "trine_encode.h",
    "trine_idf.h",
    "trine_stage1.h",
    "trine_route.h",
    "trine_canon.h",
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
        full = os.path.join(HTEB_DIR, src)
        if not os.path.isfile(full):
            raise FileNotFoundError(
                f"TRINE C source not found: {full}\n"
                f"Ensure you are building from the hteb/python/ directory."
            )
        src_paths.append(full)

    # Determine compiler
    cc = os.environ.get("CC", "cc")

    # Build command
    if sys.platform == "darwin":
        cmd = [
            cc, "-O2", "-Wall", "-fPIC", "-shared",
            "-dynamiclib",
            "-I", HTEB_DIR,
            "-o", output_path,
        ] + src_paths + ["-lm"]
    elif sys.platform == "win32":
        cmd = [
            cc, "-O2", "-Wall", "-shared",
            "-I", HTEB_DIR,
            "-o", output_path,
        ] + src_paths + ["-lm"]
    else:
        cmd = [
            cc, "-O2", "-Wall", "-fPIC", "-shared",
            "-I", HTEB_DIR,
            "-o", output_path,
        ] + src_paths + ["-lm"]

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
