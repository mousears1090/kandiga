"""Benchmarking suite for Kandiga inference engine."""

from __future__ import annotations

import time

from rich.console import Console
from rich.table import Table

console = Console()

PROMPTS = [
    ("Short", "What is 2+2?"),
    ("Medium", "Explain how a transformer neural network works in 3 sentences."),
    ("Long", "Write a detailed comparison of Python and Rust for systems programming. "
     "Cover performance, safety, ecosystem, and learning curve."),
]


def _bench_one(engine, prompt: str, max_tokens: int = 256) -> dict:
    """Benchmark a single prompt. Returns timing stats."""
    # Warmup: ensure model is loaded
    if not engine.is_ready:
        engine.load()

    # Time to first token
    t_start = time.time()
    tokens = []
    t_first = None

    for token in engine.generate(prompt, max_tokens=max_tokens, stream=True):
        if t_first is None:
            t_first = time.time()
        tokens.append(token)

    t_end = time.time()

    total_time = t_end - t_start
    ttft = (t_first - t_start) if t_first else total_time
    gen_time = (t_end - t_first) if t_first else 0
    num_tokens = len(tokens)
    tps = num_tokens / gen_time if gen_time > 0 else 0

    return {
        "num_tokens": num_tokens,
        "total_time": total_time,
        "ttft": ttft,
        "gen_time": gen_time,
        "tps": tps,
    }


def run_bench():
    """Run inference benchmarks and display results."""
    console.print()
    console.print("[bold cyan]Kandiga Benchmark[/]")
    console.print()

    from kandiga.engine import KandigaEngine

    # Benchmark both modes
    for mode_name, fast in [("Quality (K=8)", False), ("Fast (K=4)", True)]:
        console.print(f"[bold]{mode_name}[/]")
        console.print("[dim]Loading model...[/]")

        engine = KandigaEngine(fast_mode=fast, log_memory=False)
        engine.load()

        stats = engine.stats
        console.print(
            f"[dim]RSS: {stats['rss_mb']:.0f}MB | "
            f"GPU: {stats['gpu_active_mb']:.0f}MB[/]"
        )
        console.print()

        table = Table(show_header=True, header_style="bold")
        table.add_column("Prompt", width=12)
        table.add_column("Tokens", justify="right")
        table.add_column("TTFT", justify="right")
        table.add_column("Gen Time", justify="right")
        table.add_column("tok/s", justify="right", style="cyan")

        for label, prompt in PROMPTS:
            console.print(f"  Running: {label}...", end=" ")
            result = _bench_one(engine, prompt, max_tokens=256)
            console.print(f"[green]done[/]")

            table.add_row(
                label,
                str(result["num_tokens"]),
                f"{result['ttft']:.2f}s",
                f"{result['gen_time']:.1f}s",
                f"{result['tps']:.1f}",
            )

        console.print()
        console.print(table)
        console.print()

        # Clean up
        del engine

    console.print("[bold green]Benchmark complete.[/]")
    console.print()
