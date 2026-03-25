"""Build the CPU expert dylib from source."""

from __future__ import annotations

import os
import subprocess
import sys


def build_cpu_expert_dylib() -> str:
    """Compile libkandiga_cpu_expert.dylib from the Objective-C source.

    Returns the path to the built dylib.
    """
    metal_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metal")
    source = os.path.join(metal_dir, "kandiga_cpu_expert.m")
    header = os.path.join(metal_dir, "kandiga_cpu_expert.h")
    dylib = os.path.join(metal_dir, "libkandiga_cpu_expert.dylib")

    if not os.path.exists(source):
        raise FileNotFoundError(f"Source file not found: {source}")

    # Build command
    cmd = [
        "clang",
        "-shared",
        "-o", dylib,
        source,
        "-fobjc-arc",
        "-framework", "Foundation",
        "-O2",
        "-march=native",
    ]

    print(f"  Building: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Compilation failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

    if not os.path.exists(dylib):
        raise RuntimeError(f"Build succeeded but dylib not found at {dylib}")

    return dylib
