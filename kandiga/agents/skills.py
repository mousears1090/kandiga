"""Skill engine — OpenClaw-compatible SKILL.md format.

Skills are directories containing a SKILL.md file with YAML frontmatter
and markdown instructions. Compatible with ClawHub's 13K+ skills.

Skill format:
    ---
    name: my-skill
    description: What this skill does
    version: 1.0.0
    author: Your Name
    tags: [category1, category2]
    ---

    Instructions for the agent in markdown...

Built-in skills live in kandiga/skills/.
User skills live in ~/.kandiga/skills/.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


SKILLS_DIR = os.path.expanduser("~/.kandiga/skills")
BUILTIN_SKILLS_DIR = os.path.join(os.path.dirname(__file__), "skills")


@dataclass
class Skill:
    """A loaded skill."""
    name: str
    description: str
    version: str = "1.0.0"
    author: str = ""
    tags: List[str] = field(default_factory=list)
    instructions: str = ""
    path: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_prompt(self) -> str:
        """Convert skill to a system prompt addition."""
        return f"[Skill: {self.name}]\n{self.instructions}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "author": self.author,
            "tags": self.tags,
            "path": self.path,
        }


def parse_skill_md(content: str, path: str = "") -> Optional[Skill]:
    """Parse a SKILL.md file into a Skill object."""
    # Extract YAML frontmatter
    fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", content, re.DOTALL)
    if not fm_match:
        return None

    frontmatter_text = fm_match.group(1)
    instructions = fm_match.group(2).strip()

    # Parse YAML manually (no dependency)
    meta = _parse_yaml_simple(frontmatter_text)
    name = meta.get("name", "")
    if not name:
        return None

    tags = meta.get("tags", [])
    if isinstance(tags, str):
        # Handle "tags: [a, b, c]" format
        tags = [t.strip().strip('"').strip("'") for t in tags.strip("[]").split(",") if t.strip()]

    return Skill(
        name=name,
        description=meta.get("description", ""),
        version=meta.get("version", "1.0.0"),
        author=meta.get("author", ""),
        tags=tags,
        instructions=instructions,
        path=path,
        metadata=meta,
    )


def _parse_yaml_simple(text: str) -> Dict[str, Any]:
    """Minimal YAML parser for frontmatter (no dependency)."""
    result: Dict[str, Any] = {}
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r'^(\w[\w.-]*)\s*:\s*(.*)$', line)
        if m:
            key = m.group(1)
            val = m.group(2).strip()
            # Strip quotes
            if (val.startswith('"') and val.endswith('"')) or \
               (val.startswith("'") and val.endswith("'")):
                val = val[1:-1]
            # Handle arrays
            if val.startswith("[") and val.endswith("]"):
                val = [v.strip().strip('"').strip("'") for v in val[1:-1].split(",") if v.strip()]
            # Handle booleans
            elif val.lower() == "true":
                val = True
            elif val.lower() == "false":
                val = False
            result[key] = val
    return result


class SkillEngine:
    """Loads, manages, and matches skills."""

    def __init__(self, extra_dirs: Optional[List[str]] = None):
        self._skills: Dict[str, Skill] = {}
        self._dirs = [BUILTIN_SKILLS_DIR, SKILLS_DIR]
        if extra_dirs:
            self._dirs.extend(extra_dirs)

    def load_all(self) -> int:
        """Load all skills from all directories. Returns count."""
        count = 0
        for d in self._dirs:
            if not os.path.isdir(d):
                continue
            for entry in os.listdir(d):
                skill_dir = os.path.join(d, entry)
                if os.path.isdir(skill_dir):
                    skill_md = os.path.join(skill_dir, "SKILL.md")
                    if os.path.isfile(skill_md):
                        self._load_file(skill_md)
                        count += 1
                elif entry.endswith(".md") and entry != "README.md":
                    self._load_file(os.path.join(d, entry))
                    count += 1
        return count

    def _load_file(self, path: str) -> Optional[Skill]:
        try:
            with open(path, "r") as f:
                content = f.read()
            skill = parse_skill_md(content, path=path)
            if skill:
                self._skills[skill.name] = skill
                return skill
        except Exception:
            pass
        return None

    def get(self, name: str) -> Optional[Skill]:
        return self._skills.get(name)

    def match(self, query: str) -> List[Skill]:
        """Find skills relevant to a query."""
        q = query.lower()
        matches = []
        for skill in self._skills.values():
            score = 0
            if skill.name.lower() in q:
                score += 10
            for tag in skill.tags:
                if tag.lower() in q:
                    score += 5
            desc_words = set(skill.description.lower().split())
            query_words = set(q.split())
            overlap = desc_words & query_words
            score += len(overlap) * 2
            if score > 0:
                matches.append((score, skill))
        matches.sort(key=lambda x: -x[0])
        return [s for _, s in matches[:3]]

    def list_all(self) -> List[Skill]:
        return list(self._skills.values())

    def create_skill(self, name: str, description: str, instructions: str,
                     tags: Optional[List[str]] = None, author: str = "") -> str:
        """Create a new skill and save it to disk. Returns the path."""
        os.makedirs(SKILLS_DIR, exist_ok=True)
        skill_dir = os.path.join(SKILLS_DIR, name)
        os.makedirs(skill_dir, exist_ok=True)

        tags_str = ", ".join(tags) if tags else ""
        content = f"""---
name: {name}
description: {description}
version: 1.0.0
author: {author}
tags: [{tags_str}]
---

{instructions}
"""
        path = os.path.join(skill_dir, "SKILL.md")
        with open(path, "w") as f:
            f.write(content)

        skill = parse_skill_md(content, path=path)
        if skill:
            self._skills[name] = skill
        return path
