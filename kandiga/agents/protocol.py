"""Typed protocol — no raw strings between pipeline stages."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ToolCall:
    """A request to execute a tool."""
    tool: str
    args: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"tool": self.tool, "args": dict(self.args)}


@dataclass
class ToolResult:
    """Result of executing a tool."""
    tool: str
    args: Dict[str, Any]
    output: Any
    success: bool
    error: Optional[str] = None
    duration_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "tool": self.tool,
            "args": self.args,
            "output": str(self.output) if self.output is not None else None,
            "success": self.success,
            "duration_ms": self.duration_ms,
        }
        if self.error:
            d["error"] = self.error
        return d


@dataclass
class AgentResult:
    """Final output of the agent pipeline."""
    content: str
    confidence: float = 1.0
    verified: bool = False
    route: str = "direct"  # direct | tool | agentic
    tool_results: List[ToolResult] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)
    plan_summary: str = ""
    duration_ms: int = 0
    tokens_generated: int = 0
    timestamp: float = field(default_factory=time.time)

    @property
    def all_tools_succeeded(self) -> bool:
        return all(tr.success for tr in self.tool_results)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "confidence": self.confidence,
            "verified": self.verified,
            "route": self.route,
            "tool_results": [tr.to_dict() for tr in self.tool_results],
            "flags": self.flags,
            "plan_summary": self.plan_summary,
            "duration_ms": self.duration_ms,
            "tokens_generated": self.tokens_generated,
            "timestamp": self.timestamp,
        }
