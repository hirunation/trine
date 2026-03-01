"""
Helper to compile libtrine.so from C sources.

Can be invoked standalone:
    python -m pytrine.build_lib [--source-dir PATH] [--output-dir PATH]

Or used programmatically:
    from pytrine.build_lib import build_shared_lib
    build_shared_lib("/path/to/hteb", "/path/to/output")
"""

import os
import sys
import subprocess


# C source files required
C_SOURCES = [
    "trine_encode.c",
    "trine_stage1.c",
    "trine_route.c",
    "trine_canon.c",
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
        Path to the hteb/ directory containing the C source files.
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
                f"Expected source files in: {source_dir}"
            )
        src_paths.append(full)

    cc = os.environ.get("CC", "cc")

    # Platform-specific flags
    flags = ["-O2", "-Wall", "-fPIC", "-shared"]
    if sys.platform == "darwin":
        flags.append("-dynamiclib")

    cmd = [cc] + flags + ["-I", source_dir, "-o", output_path] + src_paths + ["-lm"]

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
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."),
        help="Path to hteb/ directory with C sources (default: ../../ relative to this file)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Output directory for the shared library (default: this package dir)",
    )
    args = parser.parse_args()
    build_shared_lib(args.source_dir, args.output_dir)
