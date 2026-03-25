"""One-time setup: download model, split experts, pack binary format, build dylib."""

from __future__ import annotations

import os
import sys
import time

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

console = Console()

EXPERT_CACHE_DIR = os.path.expanduser("~/.kandiga/experts")


def run_setup(model_path: str = "mlx-community/Qwen3.5-35B-A3B-4bit"):
    """Run the full setup pipeline."""
    console.print()
    console.print("[bold cyan]Kandiga Setup[/]")
    console.print()

    model_name = model_path.split("/")[-1]
    model_cache_dir = os.path.join(EXPERT_CACHE_DIR, model_name)
    packed_dir = os.path.join(model_cache_dir, "packed")

    # -----------------------------------------------------------------------
    # Step 1: Download model
    # -----------------------------------------------------------------------
    console.print("[bold]Step 1/4:[/] Downloading model from HuggingFace...")
    console.print(f"  [dim]{model_path}[/]")

    try:
        from huggingface_hub import snapshot_download
        model_dir = snapshot_download(model_path)
        console.print(f"  [green]\u2713[/] Model ready at {model_dir}")
    except Exception as e:
        console.print(f"  [red]\u2717[/] Download failed: {e}")
        console.print("  [dim]Make sure you have huggingface_hub installed and internet access.[/]")
        sys.exit(1)

    console.print()

    # -----------------------------------------------------------------------
    # Step 2: Split expert weights
    # -----------------------------------------------------------------------
    console.print("[bold]Step 2/4:[/] Splitting expert weights (one-time, ~45s)...")
    console.print(f"  [dim]40 layers x 256 experts = 10,240 files[/]")

    if os.path.isdir(os.path.join(model_cache_dir, "layer_00")):
        console.print(f"  [yellow]\u2713[/] Already split, skipping.")
    else:
        try:
            from kandiga._split_experts import split_experts
            split_experts(model_dir, model_cache_dir)
            console.print(f"  [green]\u2713[/] Experts split to {model_cache_dir}")
        except Exception as e:
            console.print(f"  [red]\u2717[/] Split failed: {e}")
            sys.exit(1)

    console.print()

    # -----------------------------------------------------------------------
    # Step 3: Pack binary format
    # -----------------------------------------------------------------------
    console.print("[bold]Step 3/4:[/] Packing binary format (one-time, ~60s)...")
    console.print(f"  [dim]40 layer files, ~441MB each[/]")

    if os.path.isdir(packed_dir) and any(f.endswith(".bin") for f in os.listdir(packed_dir)):
        console.print(f"  [yellow]\u2713[/] Already packed, skipping.")
    else:
        try:
            from kandiga._pack_experts import pack_experts
            pack_experts(model_cache_dir, packed_dir)
            console.print(f"  [green]\u2713[/] Binary format ready at {packed_dir}")
        except Exception as e:
            console.print(f"  [red]\u2717[/] Pack failed: {e}")
            sys.exit(1)

    console.print()

    # -----------------------------------------------------------------------
    # Step 4: Build CPU expert dylib
    # -----------------------------------------------------------------------
    console.print("[bold]Step 4/4:[/] Building CPU expert library...")

    try:
        from kandiga._build import build_cpu_expert_dylib
        dylib_path = build_cpu_expert_dylib()
        console.print(f"  [green]\u2713[/] Library built at {dylib_path}")
    except Exception as e:
        console.print(f"  [red]\u2717[/] Build failed: {e}")
        console.print("  [dim]Make sure Xcode command line tools are installed.[/]")
        sys.exit(1)

    console.print()

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    # Calculate total size
    total_bytes = 0
    if os.path.isdir(packed_dir):
        for f in os.listdir(packed_dir):
            fp = os.path.join(packed_dir, f)
            if os.path.isfile(fp):
                total_bytes += os.path.getsize(fp)

    console.print("[bold green]Setup complete![/]")
    console.print()
    console.print(f"  Expert files: {packed_dir}")
    if total_bytes > 0:
        console.print(f"  Disk usage:   {total_bytes / 1e9:.1f}GB (packed binary)")
    console.print()
    console.print("  Run [bold cyan]kandiga chat[/] to start chatting.")
    console.print("  Run [bold cyan]kandiga chat --fast[/] for ~2x speed.")
    console.print()
