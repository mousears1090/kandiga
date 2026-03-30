"""Memory system — persistent, OpenClaw-compatible format.

Three layers:
1. KV cache (Kandiga native) — model literally remembers, instant
2. MEMORY.md — curated long-term memory (survives model reloads)
3. Daily notes — memory/YYYY-MM-DD.md (session logs)

Memory directory: ~/.kandiga/memory/
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Dict, List, Optional


MEMORY_DIR = os.path.expanduser("~/.kandiga/memory")
MEMORY_FILE = os.path.join(MEMORY_DIR, "MEMORY.md")


class Memory:
    """Persistent memory system."""

    def __init__(self, memory_dir: str = MEMORY_DIR):
        self.memory_dir = memory_dir
        self._memory_file = os.path.join(memory_dir, "MEMORY.md")
        os.makedirs(memory_dir, exist_ok=True)

    # --- Long-term memory (MEMORY.md) ---

    def read_memory(self) -> str:
        """Read the full MEMORY.md file."""
        if os.path.isfile(self._memory_file):
            with open(self._memory_file, "r") as f:
                return f.read()
        return ""

    def add_memory(self, content: str, category: str = "general") -> None:
        """Append a memory entry to MEMORY.md."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        entry = f"\n## [{category}] {timestamp}\n{content}\n"

        with open(self._memory_file, "a") as f:
            f.write(entry)

    def search_memory(self, query: str) -> List[str]:
        """Search MEMORY.md for relevant entries."""
        text = self.read_memory()
        if not text:
            return []

        # Split into sections
        sections = re.split(r"\n## ", text)
        q_words = set(query.lower().split())
        matches = []
        for section in sections:
            section_words = set(section.lower().split())
            overlap = q_words & section_words
            if len(overlap) >= 2 or any(w in section.lower() for w in q_words if len(w) > 3):
                matches.append(section.strip())

        return matches[:5]

    def clear_memory(self) -> None:
        """Clear all long-term memory."""
        if os.path.isfile(self._memory_file):
            os.remove(self._memory_file)

    # --- Daily notes ---

    def today_file(self) -> str:
        """Path to today's daily note."""
        date_str = datetime.now().strftime("%Y-%m-%d")
        return os.path.join(self.memory_dir, f"{date_str}.md")

    def log_daily(self, content: str) -> None:
        """Append to today's daily note."""
        path = self.today_file()
        timestamp = datetime.now().strftime("%H:%M")
        entry = f"\n### {timestamp}\n{content}\n"
        with open(path, "a") as f:
            f.write(entry)

    def read_daily(self, date: Optional[str] = None) -> str:
        """Read a daily note. Default: today."""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        path = os.path.join(self.memory_dir, f"{date}.md")
        if os.path.isfile(path):
            with open(path, "r") as f:
                return f.read()
        return ""

    def list_daily_notes(self) -> List[str]:
        """List all daily note files."""
        if not os.path.isdir(self.memory_dir):
            return []
        notes = [f for f in os.listdir(self.memory_dir)
                 if re.match(r"\d{4}-\d{2}-\d{2}\.md$", f)]
        return sorted(notes, reverse=True)

    # --- Context builder ---

    def build_context(self, query: str, max_chars: int = 2000) -> str:
        """Build memory context for a query.

        Searches MEMORY.md for relevant entries and includes recent daily notes.
        Returns a string suitable for injection into the agent prompt.
        """
        parts = []

        # Search long-term memory
        matches = self.search_memory(query)
        if matches:
            parts.append("Relevant memories:")
            for m in matches[:3]:
                parts.append(f"  {m[:300]}")

        # Recent daily note (today)
        daily = self.read_daily()
        if daily:
            parts.append(f"\nToday's notes:\n{daily[-500:]}")

        text = "\n".join(parts)
        return text[:max_chars] if text else ""

    # --- Stats ---

    @property
    def stats(self) -> Dict[str, int]:
        memory_size = 0
        if os.path.isfile(self._memory_file):
            memory_size = os.path.getsize(self._memory_file)
        daily_count = len(self.list_daily_notes())
        return {
            "memory_bytes": memory_size,
            "daily_notes": daily_count,
        }
