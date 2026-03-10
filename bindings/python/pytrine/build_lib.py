"""
Helper to compile libtrine.so from C sources.

Can be invoked standalone:
    python -m pytrine.build_lib [--source-dir PATH] [--output-dir PATH]

Or used programmatically:
    from pytrine.build_lib import build_shared_lib
    build_shared_lib("/path/to/trine-root", "/path/to/output")
"""

import os
import sys
import subprocess


# Source subdirectories under src/
SRC_DIRS = ["encode", "compare", "index", "canon", "algebra", "model"]

# C source files (relative to project root)
C_SOURCES = [
    "src/encode/trine_encode.c",
    "src/compare/trine_stage1.c",
    "src/index/trine_route.c",
    "src/canon/trine_canon.c",
    "src/compare/trine_csidf.c",
    "src/index/trine_field.c",
]


def lib_name():
    """Return the platform-appropriate shared library name."""
    if sys.platform == "darwin":
        return "libtrine.dylib"
    elif sys.platform == "win32":
        return "trine.dll"
    else:
        return "libtrine.so"


def build_shared_lib(source_dir, output_dir):
    """
    Compile C sources into a shared library.

    Parameters
    ----------
    source_dir : str
        Path to the TRINE project root containing the src/ directory.
    output_dir : str
        Path where the compiled shared library will be placed.

    Returns
    -------
    str
        Full path to the compiled shared library.

    Raises
    ------
    FileNotFoundError
        If any required C source file is missing.
    RuntimeError
        If compilation fails.
    """
    os.makedirs(output_dir, exist_ok=True)
    name = lib_name()
    output_path = os.path.join(output_dir, name)

    # Resolve source file paths
    src_paths = []
    for src in C_SOURCES:
        full = os.path.join(source_dir, src)
        if not os.path.isfile(full):
            raise FileNotFoundError(
                f"TRINE C source not found: {full}\n"
                f"Expected source files in: {source_dir}/src/"
            )
        src_paths.append(full)

    cc = os.environ.get("CC", "cc")

    # Platform-specific flags
    flags = ["-O2", "-Wall", "-fPIC", "-shared"]
    if sys.platform == "darwin":
        flags.append("-dynamiclib")

    # Include flags for all src/ subdirectories
    include_flags = []
    for d in SRC_DIRS:
        include_flags.extend(["-I", os.path.join(source_dir, "src", d)])

    cmd = [cc] + flags + include_flags + ["-o", output_path] + src_paths + ["-lm"]

    print(f"[pytrine.build_lib] Compiling {name}:")
    print(f"  {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[pytrine.build_lib] Compilation FAILED (exit {result.returncode}):")
        print(result.stderr)
        raise RuntimeError(f"Failed to compile {name}")

    if result.stderr:
        print(f"[pytrine.build_lib] Compiler warnings:\n{result.stderr}")

    size = os.path.getsize(output_path)
    print(f"[pytrine.build_lib] Built {output_path} ({size:,} bytes)")
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compile TRINE shared library")
    parser.add_argument(
        "--source-dir",
        default=os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")),
        help="Path to TRINE project root (default: ../../../ relative to this file)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Output directory for the shared library (default: this package dir)",
    )
    args = parser.parse_args()
    build_shared_lib(args.source_dir, args.output_dir)
