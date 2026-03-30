"""Kandiga agent layer — tools, skills, memory, macOS integrations.

Usage:
    from kandiga.engine import KandigaEngine
    from kandiga.agents import AgentPipeline, DualEngine

    # Dual-model (fast): 4B worker + 35B brain
    engine = DualEngine(fast_mode=True)
    engine.load()
    pipeline = AgentPipeline(engine)
    result = pipeline.run("read config.yaml and fix the bug")

    # Single-model: 35B only
    engine = KandigaEngine()
    engine.load()
    pipeline = AgentPipeline(engine)
    result = pipeline.run("explain recursion")
"""

from kandiga.agents.protocol import AgentResult, ToolCall, ToolResult
from kandiga.agents.tools import ToolRegistry, default_tools
from kandiga.agents.pipeline import AgentPipeline
from kandiga.agents.skills import SkillEngine, Skill
from kandiga.agents.memory import Memory

__all__ = [
    "AgentResult", "ToolCall", "ToolResult",
    "ToolRegistry", "default_tools",
    "AgentPipeline",
    "SkillEngine", "Skill",
    "Memory",
]

try:
    from kandiga.agents.dual_engine import DualEngine
    __all__.append("DualEngine")
except ImportError:
    pass
