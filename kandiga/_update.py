"""Check for updates from PyPI and manage upgrades."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Optional

from kandiga import __version__

CACHE_DIR = os.path.expanduser("~/.kandiga")
CACHE_FILE = os.path.join(CACHE_DIR, "update_check.json")
CHECK_INTERVAL = 86400  # 24 hours


def _parse_version(v: str) -> tuple:
    """Parse '0.1.0' -> (0, 1, 0) for comparison."""
    try:
        return tuple(int(x) for x in v.strip().split("."))
    except (ValueError, AttributeError):
        return (0, 0, 0)


def _fetch_latest() -> Optional[str]:
    """Fetch latest version from PyPI. Returns None on failure."""
    try:
        import urllib.request
        req = urllib.request.Request(
            "https://pypi.org/pypi/kandiga/json",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
            return data["info"]["version"]
    except Exception:
        return None


def check_for_update(quiet: bool = False) -> Optional[str]:
    """Check if a newer version is available. Returns new version or None.

    Caches result for 24h to avoid hitting PyPI on every run.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Check cache first
    try:
        with open(CACHE_FILE) as f:
            cache = json.load(f)
        if time.time() - cache.get("checked_at", 0) < CHECK_INTERVAL:
            latest = cache.get("latest")
            if latest and _parse_version(latest) > _parse_version(__version__):
                return latest
            return None
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        pass

    # Fetch from PyPI
    latest = _fetch_latest()
    if latest is None:
        return None

    # Save cache
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump({"latest": latest, "checked_at": time.time()}, f)
    except OSError:
        pass

    if _parse_version(latest) > _parse_version(__version__):
        return latest
    return None


def print_update_notice(latest: str):
    """Print a subtle update notice."""
    from rich.console import Console
    console = Console(stderr=True)
    console.print(
        f"\n  [dim]Update available:[/] [yellow]{__version__}[/] → [green]{latest}[/]"
        f"  [dim]Run[/] [bold]kandiga update[/] [dim]to upgrade[/]\n"
    )


def run_update():
    """Upgrade kandiga via pip."""
    from rich.console import Console
    console = Console()

    console.print(f"\n[bold cyan]Kandiga Update[/]")
    console.print(f"  [dim]Current version:[/] {__version__}")

    latest = _fetch_latest()
    if latest is None:
        console.print("  [red]Could not reach PyPI.[/]")
        return

    if _parse_version(latest) <= _parse_version(__version__):
        console.print(f"  [green]Already up to date.[/]\n")
        return

    console.print(f"  [dim]Latest version:[/]  {latest}")
    console.print(f"\n  Upgrading...\n")

    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", "kandiga"],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        console.print(f"  [green]✓[/] Updated to {latest}")
        console.print(f"  [dim]Restart kandiga to use the new version.[/]\n")
        # Clear cache
        try:
            os.remove(CACHE_FILE)
        except OSError:
            pass
    else:
        console.print(f"  [red]✗[/] Update failed:")
        console.print(f"  [dim]{result.stderr.strip()}[/]\n")


def run_changelog():
    """Show changelog / release notes."""
    from rich.console import Console
    console = Console()

    console.print(f"\n[bold cyan]Kandiga Changelog[/]\n")

    # Fetch all versions from PyPI
    try:
        import urllib.request
        req = urllib.request.Request(
            "https://pypi.org/pypi/kandiga/json",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
    except Exception:
        console.print("  [red]Could not reach PyPI.[/]\n")
        return

    current = data["info"]["version"]
    releases = sorted(data["releases"].keys(), key=_parse_version, reverse=True)

    for ver in releases[:10]:  # Show last 10 versions
        files = data["releases"][ver]
        if not files:
            continue
        upload_date = files[0].get("upload_time_iso_8601", "")[:10]
        marker = " [cyan](installed)[/]" if ver == __version__ else ""
        latest_marker = " [green](latest)[/]" if ver == current else ""
        console.print(f"  [bold]{ver}[/]{marker}{latest_marker}  [dim]{upload_date}[/]")

    console.print(f"\n  [dim]Full history: https://pypi.org/project/kandiga/#history[/]\n")
