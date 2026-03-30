"""Interactive agent chat with Rich — tools, skills, memory, macOS."""

from __future__ import annotations

import time
from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

console = Console()

_LOGO = r"""[bold cyan]
 ██╗  ██╗ █████╗ ███╗   ██╗██████╗ ██╗ ██████╗  █████╗
 ██║ ██╔╝██╔══██╗████╗  ██║██╔══██╗██║██╔════╝ ██╔══██╗
 █████╔╝ ███████║██╔██╗ ██║██║  ██║██║██║  ███╗███████║
 ██╔═██╗ ██╔══██║██║╚██╗██║██║  ██║██║██║   ██║██╔══██║
 ██║  ██╗██║  ██║██║ ╚████║██████╔╝██║╚██████╔╝██║  ██║
 ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═════╝ ╚═╝ ╚═════╝ ╚═╝  ╚═╝[/]
[dim]  Agent mode — tools, skills, memory, persistent sessions[/]"""


def run_agent_chat(fast: bool = False, model: Optional[str] = None):
    """Interactive agent chat with tools and persistent KV cache."""
    from kandiga.engine import KandigaEngine
    from kandiga.agents.pipeline import AgentPipeline
    from kandiga.agents.tools import default_tools
    from kandiga.agents.macos import register_macos_tools
    from kandiga.agents.memory import Memory
    from kandiga.agents.skills import SkillEngine

    console.print(_LOGO)
    console.print()

    # Load engine
    model = model or KandigaEngine.DEFAULT_MODEL
    model_short = model.split("/")[-1]
    mode = "[yellow]Fast (K=4)[/]" if fast else "[green]Quality (K=8)[/]"
    console.print(f"  [dim]Model:[/] {model_short}")
    console.print(f"  [dim]Mode:[/]  {mode}")
    console.print()

    console.print("[dim]Loading model...[/]")
    t0 = time.time()
    engine = KandigaEngine(model_path=model, fast_mode=fast)
    engine.load()
    load_s = time.time() - t0
    stats = engine.stats
    console.print(
        f"[green]Ready.[/] [dim]({load_s:.1f}s, "
        f"RSS {stats['rss_mb']:.0f}MB, GPU {stats['gpu_active_mb']:.0f}MB)[/]"
    )

    # Set up tools
    registry = default_tools()
    macos_count = register_macos_tools(registry)
    console.print(f"[dim]Tools: {len(registry.tool_names)} ({macos_count} macOS native)[/]")

    # Memory
    memory = Memory()
    mem_ctx = memory.build_context("")
    if mem_ctx:
        console.print(f"[dim]Memory loaded ({memory.stats['memory_bytes']} bytes)[/]")

    # Skills
    skills = SkillEngine()
    skill_count = skills.load_all()
    if skill_count > 0:
        console.print(f"[dim]Skills: {skill_count}[/]")

    console.print()
    console.print("  [dim]Commands:[/] /quit /save /load /skills /memory /tools /stats /clear")
    console.print("  [dim]         Persistent KV cache active — follow-ups are instant[/]")
    console.print()

    # Pipeline with stage callbacks
    def on_stage(stage, detail):
        console.print(f"  [dim][{stage}][/] [dim]{detail}[/]")

    pipeline = AgentPipeline(engine, registry=registry, on_stage=on_stage)

    # Start persistent session
    pipeline.start_session()
    turn = 0

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
        if stripped in ("/quit", "/exit", "/q"):
            pipeline.end_session()
            console.print("[dim]Session ended. Bye.[/]")
            break

        if stripped == "/save":
            import os
            path = os.path.expanduser("~/.kandiga/sessions/last_session.npz")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            try:
                pipeline.save_session(path)
                console.print(f"  [green]Session saved[/] [dim]({path})[/]")
            except Exception as e:
                console.print(f"  [red]Save failed: {e}[/]")
            continue

        if stripped == "/load":
            import os
            path = os.path.expanduser("~/.kandiga/sessions/last_session.npz")
            try:
                pipeline.load_session(path)
                console.print(f"  [green]Session loaded[/] [dim]({path})[/]")
            except Exception as e:
                console.print(f"  [red]Load failed: {e}[/]")
            continue

        if stripped == "/skills":
            all_skills = skills.list_all()
            if all_skills:
                for s in all_skills:
                    tags = f" [{', '.join(s.tags)}]" if s.tags else ""
                    console.print(f"  [cyan]{s.name}[/]: {s.description}{tags}")
            else:
                console.print("  [dim]No skills loaded. Create one with: /skill create <name>[/]")
            continue

        if stripped == "/memory":
            text = memory.read_memory()
            if text:
                console.print(Panel(text[-1000:], title="Memory", border_style="dim"))
            else:
                console.print("  [dim]No memories saved yet.[/]")
            continue

        if stripped == "/tools":
            console.print(f"  [dim]{registry.describe_tools()}[/]")
            continue

        if stripped == "/stats":
            s = engine.stats
            console.print(
                f"  [dim]RSS: {s['rss_mb']:.0f}MB | GPU: {s['gpu_active_mb']:.0f}MB | "
                f"Mode: {s['mode']} | Turn: {turn}[/]"
            )
            console.print(f"  [dim]Memory: {memory.stats}[/]")
            continue

        if stripped == "/clear":
            pipeline.end_session()
            pipeline.start_session()
            turn = 0
            console.clear()
            console.print(_LOGO)
            console.print("\n  [dim]Session reset. Fresh start.[/]\n")
            continue

        if stripped.startswith("/remember "):
            content = stripped[10:].strip()
            memory.add_memory(content, category="user")
            console.print(f"  [green]Remembered.[/]")
            continue

        if stripped.startswith("/"):
            console.print(f"  [dim]Unknown command: {stripped}[/]")
            continue

        # Build context with memory
        context = ""
        mem_ctx = memory.build_context(stripped, max_chars=1000)
        if mem_ctx:
            context = f"Agent memory:\n{mem_ctx}"

        # Check for matching skills
        matched_skills = skills.match(stripped)
        if matched_skills:
            skill = matched_skills[0]
            context += f"\n\nActive skill [{skill.name}]:\n{skill.instructions}"
            console.print(f"  [dim]Using skill: {skill.name}[/]")

        # Run pipeline
        console.print()
        t0 = time.time()
        try:
            result = pipeline.run(stripped, context=context)
        except KeyboardInterrupt:
            console.print("\n[dim](interrupted)[/]")
            continue

        elapsed = time.time() - t0
        turn += 1

        # Display response
        try:
            md = Markdown(result.content)
            console.print(md)
        except Exception:
            console.print(result.content)

        # Status line
        parts = [
            f"Turn {turn}",
            f"{elapsed:.1f}s",
            f"route={result.route}",
            f"conf={result.confidence:.2f}",
        ]
        if result.tool_results:
            tools_ok = sum(1 for tr in result.tool_results if tr.success)
            tools_total = len(result.tool_results)
            parts.append(f"tools={tools_ok}/{tools_total}")
        if result.verified:
            parts.append("[green]verified[/]")

        console.print(f"\n[dim]{' · '.join(parts)}[/]\n")

        # Log to daily memory
        memory.log_daily(f"Q: {stripped[:100]}\nA: {result.content[:200]}\nRoute: {result.route} Conf: {result.confidence:.2f}")
