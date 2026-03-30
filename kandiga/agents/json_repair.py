"""Robust JSON extraction from model output — never crashes."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Set


def parse_json(text: str, defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Extract valid JSON from model output with multi-level fallback."""
    if not text or not text.strip():
        return dict(defaults)

    # Strategy 1: Direct parse
    try:
        result = json.loads(text.strip())
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, TypeError):
        pass

    # Strategy 2: Find first complete balanced JSON object
    for m in re.finditer(r"\{", text):
        candidate = _extract_balanced(text, m.start())
        if candidate:
            try:
                result = json.loads(candidate)
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                continue

    # Strategy 3: Repair truncated JSON
    repaired = _repair_truncated(text)
    if repaired:
        try:
            result = json.loads(repaired)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    # Strategy 4: Aggressive trim
    trimmed = _trim_to_last_value(text)
    if trimmed:
        try:
            result = json.loads(trimmed)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    return dict(defaults)


def validate_tool_calls(parsed: Dict[str, Any], valid_tools: Set[str]) -> List[Dict[str, Any]]:
    """Extract and validate tool_calls from parsed JSON."""
    raw_calls = parsed.get("tool_calls", [])
    if not isinstance(raw_calls, list):
        return []
    valid = []
    for call in raw_calls:
        if not isinstance(call, dict):
            continue
        tool = call.get("tool", "")
        if not tool or not isinstance(tool, str):
            continue
        if valid_tools and tool not in valid_tools:
            continue
        args = call.get("args", {})
        if not isinstance(args, dict):
            args = {}
        valid.append({"tool": tool, "args": args})
    return valid


def validate_plan(parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract and validate plan steps."""
    raw_plan = parsed.get("plan", [])
    if not isinstance(raw_plan, list):
        return []
    valid = []
    for item in raw_plan:
        if not isinstance(item, dict):
            continue
        action = item.get("action", "")
        if not action:
            continue
        deps = item.get("depends_on", [])
        if not isinstance(deps, list):
            deps = []
        deps = [int(d) for d in deps if isinstance(d, (int, float))]
        valid.append({
            "step": int(item.get("step", len(valid) + 1)),
            "action": action,
            "description": str(item.get("description", "")),
            "depends_on": deps,
        })
    return valid


def extract_write_file(raw: str) -> Optional[Dict[str, Any]]:
    """Extract write_file from truncated output."""
    path_match = re.search(r'"tool"\s*:\s*"write_file".*?"path"\s*:\s*"([^"]+)"', raw, re.DOTALL)
    if not path_match:
        return None
    path = path_match.group(1)
    content_match = re.search(r'"content"\s*:\s*"((?:[^"\\]|\\.)*)', raw, re.DOTALL)
    content = content_match.group(1) if content_match else ""
    try:
        content = content.encode().decode("unicode_escape")
    except (UnicodeDecodeError, ValueError):
        pass
    return {"tool": "write_file", "args": {"path": path, "content": content}}


def _extract_balanced(text: str, start: int) -> Optional[str]:
    if start >= len(text) or text[start] != '{':
        return None
    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _repair_truncated(text: str) -> Optional[str]:
    start = text.find("{")
    if start < 0:
        return None
    fragment = text[start:]
    in_string = False
    escape_next = False
    stack: list = []
    for ch in fragment:
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            if in_string:
                escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in ("{", "["):
            stack.append(ch)
        elif ch == "}":
            if stack and stack[-1] == "{":
                stack.pop()
        elif ch == "]":
            if stack and stack[-1] == "[":
                stack.pop()
    if not stack:
        return fragment
    repaired = fragment
    if in_string:
        repaired += '"'
    for opener in reversed(stack):
        repaired += "}" if opener == "{" else "]"
    return repaired


def _trim_to_last_value(text: str) -> Optional[str]:
    start = text.find("{")
    if start < 0:
        return None
    fragment = text[start:]
    quote_positions = []
    in_escape = False
    for i, ch in enumerate(fragment):
        if in_escape:
            in_escape = False
            continue
        if ch == "\\":
            in_escape = True
            continue
        if ch == '"':
            quote_positions.append(i)
    for qi in range(len(quote_positions) - 1, 0, -2):
        pos = quote_positions[qi]
        candidate = fragment[:pos + 1]
        stack: list = []
        in_str = False
        esc = False
        for ch in candidate:
            if esc:
                esc = False
                continue
            if ch == "\\":
                if in_str:
                    esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch in ("{", "["):
                stack.append(ch)
            elif ch == "}":
                if stack and stack[-1] == "{":
                    stack.pop()
            elif ch == "]":
                if stack and stack[-1] == "[":
                    stack.pop()
        for opener in reversed(stack):
            candidate += "}" if opener == "{" else "]"
        try:
            result = json.loads(candidate)
            if isinstance(result, dict):
                return candidate
        except json.JSONDecodeError:
            continue
    return None
