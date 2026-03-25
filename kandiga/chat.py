"""Interactive chat interface using Rich."""

from __future__ import annotations

import time
from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

console = Console()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

_LOGO = r"""[bold cyan]
 ██╗  ██╗ █████╗ ███╗   ██╗██████╗ ██╗ ██████╗  █████╗
 ██║ ██╔╝██╔══██╗████╗  ██║██╔══██╗██║██╔════╝ ██╔══██╗
 █████╔╝ ███████║██╔██╗ ██║██║  ██║██║██║  ███╗███████║
 ██╔═██╗ ██╔══██║██║╚██╗██║██║  ██║██║██║   ██║██╔══██║
 ██║  ██╗██║  ██║██║ ╚████║██████╔╝██║╚██████╔╝██║  ██║
 ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═════╝ ╚═╝ ╚═════╝ ╚═╝  ╚═╝[/]
[dim]  35B intelligence. 1.5GB memory.[/]"""


def _print_header(fast: bool = False, model: str = ""):
    """Print the startup banner."""
    mode_text = "[yellow]Fast mode (K=4)[/]" if fast else "[green]Quality mode (K=8)[/]"
    model_short = model.split("/")[-1] if model else "Qwen3.5-35B-A3B"

    console.print(_LOGO)
    console.print()
    console.print(f"  [dim]Model:[/] {model_short}")
    console.print(f"  [dim]Mode:[/]  {mode_text}")
    console.print()
    console.print(
        "  [dim]Commands:[/] /quit  /fast  /stats  /clear"
    )
    console.print(
        "  [dim]         Ctrl+C to interrupt generation[/]"
    )
    console.print()


# ---------------------------------------------------------------------------
# Interactive chat
# ---------------------------------------------------------------------------

def run_chat(
    fast: bool = False,
    tools: bool = False,
    model: Optional[str] = None,
    max_tokens: int = 2048,
):
    """Interactive chat loop with streaming output."""
    from kandiga.engine import KandigaEngine

    model = model or KandigaEngine.DEFAULT_MODEL
    _print_header(fast=fast, model=model)

    engine = KandigaEngine(model_path=model, fast_mode=fast)

    console.print("[dim]Loading model...[/]")
    t0 = time.time()
    engine.load()
    load_time = time.time() - t0
    stats = engine.stats
    console.print(
        f"[green]Ready.[/] [dim]({load_time:.1f}s, "
        f"RSS {stats['rss_mb']:.0f}MB, "
        f"GPU {stats['gpu_active_mb']:.0f}MB)[/]\n"
    )

    history = []

    while True:
        try:
            user_input = console.input("[bold cyan]\u203a[/] ")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Bye.[/]")
            break

        stripped = user_input.strip()
        if not stripped:
            continue

        # Commands
        if stripped == "/quit" or stripped == "/exit":
            console.print("[dim]Bye.[/]")
            break

        if stripped == "/fast":
            fast = not fast
            engine.fast_mode = fast
            label = "[yellow]Fast mode ON[/] (K=4)" if fast else "[green]Quality mode ON[/] (K=8)"
            console.print(f"  {label}")
            continue

        if stripped == "/stats":
            s = engine.stats
            console.print(
                f"  [dim]RSS: {s['rss_mb']:.0f}MB | "
                f"GPU: {s['gpu_active_mb']:.0f}MB | "
                f"Mode: {s['mode']}[/]"
            )
            continue

        if stripped == "/clear":
            history.clear()
            console.clear()
            _print_header(fast=fast, model=model)
            console.print("[dim]History cleared.[/]\n")
            continue

        if stripped.startswith("/"):
            console.print(f"  [dim]Unknown command: {stripped}[/]")
            continue

        # Generate response with streaming
        console.print()
        response_text = ""
        token_count = 0
        start_time = time.time()

        try:
            for token in engine.generate(
                user_input, max_tokens=max_tokens, stream=True
            ):
                response_text += token
                token_count += 1
                console.print(token, end="", highlight=False)
        except KeyboardInterrupt:
            console.print("\n[dim](interrupted)[/]")

        elapsed = time.time() - start_time
        if elapsed > 0 and token_count > 0:
            tps = token_count / elapsed
            console.print(
                f"\n\n[dim]{token_count} tokens \u00b7 {elapsed:.1f}s \u00b7 {tps:.1f} tok/s[/]\n"
            )
        else:
            console.print("\n")


# ---------------------------------------------------------------------------
# One-shot mode
# ---------------------------------------------------------------------------

def one_shot(prompt: str, fast: bool = False, model: Optional[str] = None):
    """Single prompt, print response, exit."""
    from kandiga.engine import KandigaEngine

    engine = KandigaEngine(model_path=model, fast_mode=fast)
    engine.load()

    for token in engine.generate(prompt, stream=True):
        print(token, end="", flush=True)
    print()
