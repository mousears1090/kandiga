"""Agent pipeline — dual-model or single-model, tools, verification, skills.

Dual-model mode (default):
  4B handles: tool call JSON, classification, content generation (~34 tok/s)
  35B handles: planning, final synthesis, complex reasoning (~3-6 tok/s)
  Result: tool queries ~15s instead of ~76s

Single-model mode (fallback):
  35B handles everything (slower but still works)

Pipeline:
  1. Classify query via keywords (instant, no model call)
  2. If tools: 4B generates tool JSON → execute → 35B synthesizes
  3. If multi-step: 35B plans → 4B executes each step → 35B synthesizes
  4. Verify result (deterministic cross-check)
  5. Return typed AgentResult
"""

from __future__ import annotations

import os
import re
import time
from typing import Any, Callable, Dict, List, Optional, Set

from kandiga.agents.json_repair import (
    extract_write_file,
    parse_json,
    validate_plan,
    validate_tool_calls,
)
from kandiga.agents.protocol import AgentResult, ToolCall, ToolResult
from kandiga.agents.tools import ToolRegistry, default_tools


# ---------------------------------------------------------------------------
# Keyword detection (instant, no model call)
# ---------------------------------------------------------------------------

_ACTION_WORDS: Set[str] = {
    "create", "write", "save", "make", "read", "open", "delete",
    "remove", "list", "find", "search", "run", "execute", "grep",
    "mkdir", "copy", "move", "rename", "replace", "change", "add",
    "edit", "modify", "update", "rewrite", "fix", "correct", "improve",
    "schedule", "remind", "note", "email", "send", "check", "show",
    "notify", "reveal", "speak", "say", "look", "google", "browse",
}

_FILE_EXTENSIONS = (
    ".txt", ".py", ".md", ".json", ".csv", ".html", ".js", ".ts",
    ".yaml", ".toml", ".sh", ".xml", ".sql", ".log", ".docx", ".pdf",
)

_MULTI_STEP_SIGNALS = (" then ", " and then ", " after that ", ", then ")


def _needs_tools(query: str) -> bool:
    words = set(query.lower().split())
    q = query.lower()
    if _ACTION_WORDS & words:
        targets = {"file", "files", "folder", "directory", "dir", "code",
                    "content", "calendar", "reminder", "note", "email", "meeting",
                    "notification", "finder", "contact", "contacts", "event",
                    "web", "internet", "online", "benchmark", "performance"}
        if targets & words:
            return True
    if any(ext in q for ext in _FILE_EXTENSIONS):
        return True
    if "/" in q and any(w in words for w in ("read", "write", "save", "create", "list", "open")):
        return True
    # Direct triggers — these always need tools
    direct_triggers = [
        "search the web", "web search", "google", "look up",
        "send notification", "send me a notification", "notify me",
        "create reminder", "add reminder", "remind me",
        "create event", "schedule", "add to calendar",
        "create note", "add note", "search notes",
        "open in finder", "reveal in finder",
        "what's on my calendar", "what is on my calendar", "my reminders", "my contacts",
        "say aloud", "say hello", "speak", "read aloud", "aloud",
        "system info", "hardware info",
    ]
    if any(t in q for t in direct_triggers):
        return True
    return False


def _needs_multi_step(query: str) -> bool:
    q = query.lower()
    if any(c in q for c in _MULTI_STEP_SIGNALS):
        return True
    write_w = ("write ", "create ", "save ", "generate ")
    file_w = ("file", ".txt", ".py", ".md", ".json", ".csv", ".html")
    if any(w in q for w in write_w) and any(f in q for f in file_w):
        return True
    edit_w = ("replace ", "change ", "edit ", "modify ", "fix ", "rewrite ")
    if any(w in q for w in edit_w) and any(f in q for f in file_w):
        return True
    verbs = ["read", "write", "list", "find", "search", "run", "create", "delete",
             "save", "replace", "edit", "fix", "add", "remove"]
    return sum(1 for v in verbs if v in q) >= 2


# ---------------------------------------------------------------------------
# Verification (deterministic, no model call)
# ---------------------------------------------------------------------------

def _verify(response: str, context: str, tool_results: List[ToolResult]) -> tuple:
    resp = response.lower()
    ctx = context.lower()
    combined = resp + " " + ctx
    flags = []
    confidence = 0.85

    any_failed = any(not tr.success for tr in tool_results)
    any_wrote = any(tr.success and tr.tool == "write_file" for tr in tool_results)
    any_read = any(tr.success and tr.tool == "read_file" for tr in tool_results)

    error_patterns = ["error:", "file not found", "not a directory", "permission denied"]
    has_error = any(p in ctx for p in error_patterns) or any_failed

    refusal_phrases = ["i cannot", "i can't", "i'm unable", "i am unable",
                       "i don't have access", "beyond my capabilities"]
    is_refusal = any(p in resp for p in refusal_phrases)

    claims_saved = any(p in resp for p in ["has been saved", "saved the file",
                                            "saved successfully", "written to"])
    has_write_confirm = any_wrote or ("wrote" in combined and "bytes" in combined)
    false_claim = claims_saved and not has_write_confirm and not has_error

    if has_error:
        confidence = min(confidence, 0.3)
        flags.append("Tool error detected")
        for tr in tool_results:
            if not tr.success:
                flags.append(f"{tr.tool}: {tr.error}")

    if is_refusal:
        confidence = min(confidence, 0.25)
        flags.append("Response refuses to help")

    if false_claim:
        confidence = min(confidence, 0.4)
        flags.append("Claims saved but no write confirmation")

    if (any_wrote or any_read) and not has_error and not is_refusal and not false_claim:
        confidence = max(confidence, 0.92)

    if not tool_results and not has_error and not is_refusal and not false_claim:
        confidence = 0.85

    return confidence, confidence >= 0.7, flags


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

TOOL_PROMPT = (
    "You have these tools:\n{tool_descriptions}\n\n"
    "Respond with ONLY valid JSON:\n"
    '{{"tool_calls": [{{"tool": "TOOL_NAME", "args": {{"key": "value"}}}}], "reasoning": "brief"}}\n\n'
    "Rules:\n"
    "- Use write_file to create files, NOT run_shell\n"
    "- Use read_file to read files, NOT run_shell with cat\n"
    "- Use list_dir to list dirs, NOT run_shell with ls\n"
    "- Output ONLY valid JSON, nothing else"
)

PLAN_PROMPT = (
    "You are a task planner with these tools:\n{tool_descriptions}\n\n"
    "Respond with ONLY valid JSON:\n"
    '{{"plan": [{{"step": 1, "action": "tool_name", "description": "what", "depends_on": []}}]}}\n\n'
    "Rules:\n"
    "- action must be an exact tool name\n"
    "- For edit operations: plan read_file then write_file\n"
    "- Use full absolute paths"
)

SYNTH_PROMPT = (
    "You are an AI assistant. The context contains tool execution results from the user's system. "
    "Report what was done or found directly and confidently. "
    "Never fabricate beyond what the context provides."
)

StageCallback = Callable[[str, str], None]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class AgentPipeline:
    """Agent layer on Kandiga. Supports dual-model (4B+35B) or single-model.

    Dual-model usage:
        from kandiga.agents.dual_engine import DualEngine
        from kandiga.agents import AgentPipeline

        engine = DualEngine(fast_mode=True)
        engine.load()
        pipeline = AgentPipeline(engine)
        result = pipeline.run("read config.yaml and fix the bug")

    Single-model usage:
        from kandiga.engine import KandigaEngine
        from kandiga.agents import AgentPipeline

        engine = KandigaEngine()
        engine.load()
        pipeline = AgentPipeline(engine)
        result = pipeline.run("explain recursion")
    """

    def __init__(
        self,
        engine,
        registry: Optional[ToolRegistry] = None,
        working_dirs: Optional[List[str]] = None,
        max_steps: int = 6,
        on_stage: Optional[StageCallback] = None,
    ):
        self.engine = engine
        self.registry = registry or default_tools()
        self.working_dirs = working_dirs or []
        self.max_steps = max_steps
        self.on_stage = on_stage or (lambda *_: None)

        # Detect if we have a DualEngine
        self._dual = hasattr(engine, 'generate_fast') and hasattr(engine, 'generate_brain')

        # Session state for multi-turn (persistent KV cache)
        self._session_active = False
        self._turn_count = 0

    # --- Session management (persistent KV cache) ---

    def start_session(self) -> None:
        """Start a persistent session. Follow-up turns only process new tokens.

        This is the key advantage over OpenClaw/Hermes: turn 50 takes
        the same time as turn 1 because previous turns are in the KV cache.
        """
        brain = self.engine if not self._dual else self.engine.brain
        if hasattr(brain, 'start_session'):
            brain.start_session()
        self._session_active = True
        self._turn_count = 0
        self.on_stage("session", "started (persistent KV cache active)")

    def end_session(self) -> None:
        """End session, free KV cache memory."""
        brain = self.engine if not self._dual else self.engine.brain
        if hasattr(brain, 'end_session'):
            brain.end_session()
        self._session_active = False
        self._turn_count = 0
        self.on_stage("session", "ended")

    def save_session(self, path: str) -> None:
        """Save session to disk — survive restarts."""
        brain = self.engine if not self._dual else self.engine.brain
        if hasattr(brain, 'save_session'):
            brain.save_session(path)
            self.on_stage("session", f"saved to {path}")

    def load_session(self, path: str) -> None:
        """Load session from disk — instant resume."""
        brain = self.engine if not self._dual else self.engine.brain
        if hasattr(brain, 'load_session'):
            brain.load_session(path)
            self._session_active = True
            self.on_stage("session", f"loaded from {path}")

    def run(self, query: str, context: str = "") -> AgentResult:
        """Run the full agent pipeline.

        In session mode, the 35B brain uses persistent KV cache.
        Turn N only processes the new query — all prior turns are cached.
        TTFT stays constant regardless of conversation length.
        """
        t_start = time.time()

        full_ctx = context
        if self.working_dirs:
            dirs = "\n".join(f"  - {d}" for d in self.working_dirs)
            full_ctx = f"Accessible directories:\n{dirs}\n\n{full_ctx}" if full_ctx else f"Accessible directories:\n{dirs}"

        needs_t = _needs_tools(query)
        multi = _needs_multi_step(query) if needs_t else False

        if multi:
            self.on_stage("route", "agentic")
            result = self._run_agentic(query, full_ctx)
        elif needs_t:
            self.on_stage("route", "tool")
            result = self._run_tool(query, full_ctx)
        else:
            self.on_stage("route", "direct")
            result = self._run_direct(query, full_ctx)

        result.duration_ms = int((time.time() - t_start) * 1000)
        return result

    # --- Generation helpers ---

    def _gen_fast(self, system: str, user: str, max_tokens: int = 1200) -> str:
        """Fast generation — 4B in dual mode, engine in single mode."""
        if self._dual:
            return self.engine.generate_fast(system, user, max_tokens=max_tokens)
        return self._gen_single(system, user, max_tokens)

    def _gen_brain(self, system: str, user: str, max_tokens: int = 2048) -> str:
        """Brain generation — 35B in dual mode, engine in single mode.

        If a session is active, uses session_generate for persistent KV cache.
        Only new tokens are processed — previous turns are already cached.
        """
        # Session mode: use persistent KV cache (constant-time follow-ups)
        if self._session_active:
            brain = self.engine if not self._dual else self.engine.brain
            if hasattr(brain, 'session_generate'):
                # Combine system+user for the session message
                message = f"[System: {system}]\n\n{user}" if self._turn_count == 0 else user
                tokens = []
                for token in brain.session_generate(message, max_tokens=max_tokens):
                    tokens.append(token)
                self._turn_count += 1
                return "".join(tokens)

        # Non-session: standard generation
        if self._dual:
            return self.engine.generate_brain(system, user, max_tokens=max_tokens)
        return self._gen_single(system, user, max_tokens)

    def _gen_single(self, system: str, user: str, max_tokens: int = 2048) -> str:
        """Generate with single KandigaEngine."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            formatted = self.engine._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            formatted = self.engine._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        return self.engine.generate(formatted, max_tokens=max_tokens, temp=0.0, stream=False)

    # --- Direct (no tools) ---

    def _run_direct(self, query: str, context: str) -> AgentResult:
        self.on_stage("brain", "Generating response...")
        user = f"Context:\n{context}\n\nQuery: {query}" if context else query
        content = self._gen_brain(SYNTH_PROMPT, user)
        conf, verified, flags = _verify(content, context, [])
        self.on_stage("done", f"confidence={conf:.2f}")
        return AgentResult(content=content, confidence=conf, verified=verified,
                           route="direct", flags=flags)

    # --- Tool call ---

    def _run_tool(self, query: str, context: str) -> AgentResult:
        # 4B generates tool call JSON (fast)
        self.on_stage("fast", "Parsing tool calls...")
        tool_system = TOOL_PROMPT.format(tool_descriptions=self.registry.describe_tools())
        user = f"Context:\n{context}\n\nTask: {query}" if context else query
        raw = self._gen_fast(tool_system, user, max_tokens=1200)

        defaults = {"tool_calls": [], "reasoning": ""}
        parsed = parse_json(raw, defaults)
        calls = validate_tool_calls(parsed, self.registry.tool_names)

        if not calls and "write_file" in raw:
            extracted = extract_write_file(raw)
            if extracted:
                calls = [extracted]

        if not calls:
            calls = self._infer_tool(query)

        calls = self._fix_shell(calls)

        # Execute tools
        self.on_stage("tools", f"Executing {len(calls)} tool(s)...")
        tool_results = []
        for c in calls:
            tc = ToolCall(tool=c["tool"], args=c.get("args", {}))
            tr = self.registry.execute(tc)
            tool_results.append(tr)

        tool_output = "\n\n".join(
            f"[{tr.tool}] {'OK' if tr.success else 'FAILED'}\n{str(tr.output) if tr.output else tr.error or ''}"
            for tr in tool_results
        )

        # 35B synthesizes with tool results (smart)
        self.on_stage("brain", "Synthesizing response...")
        synth_ctx = f"{context}\n\nTool results:\n{tool_output}" if context else f"Tool results:\n{tool_output}"
        content = self._gen_brain(SYNTH_PROMPT, f"Context:\n{synth_ctx}\n\nQuery: {query}")

        conf, verified, flags = _verify(content, synth_ctx, tool_results)
        self.on_stage("done", f"confidence={conf:.2f}")
        return AgentResult(content=content, confidence=conf, verified=verified,
                           route="tool", tool_results=tool_results, flags=flags)

    # --- Agentic (multi-step) ---

    def _run_agentic(self, query: str, context: str) -> AgentResult:
        # 35B plans (needs reasoning)
        self.on_stage("brain", "Planning...")
        plan_system = PLAN_PROMPT.format(tool_descriptions=self.registry.describe_tools())
        user = f"Context:\n{context}\n\nTask: {query}" if context else query
        raw = self._gen_brain(plan_system, user, max_tokens=512)
        parsed = parse_json(raw, {"plan": []})
        steps = validate_plan(parsed)

        if not steps:
            return self._run_tool(query, context)

        plan_summary = "; ".join(f"{s['action']}: {s['description']}" for s in steps)
        self.on_stage("plan", f"{len(steps)} steps")

        # 4B executes each step (fast)
        all_results: List[ToolResult] = []
        accumulated = ""
        consecutive_errors = 0

        for i, step in enumerate(steps[:self.max_steps]):
            if consecutive_errors >= 2:
                break

            self.on_stage("fast", f"Step {i+1}/{len(steps)}: {step['description'][:60]}")
            step_ctx = f"{context}\n\nPrior results:\n{accumulated}" if accumulated else context

            if step["action"] == "write_file":
                # 4B generates content (fast)
                content = self._gen_fast(
                    SYNTH_PROMPT,
                    f"Context:\n{step_ctx}\n\nGenerate content for: {step['description']}\nOriginal request: {query}",
                    max_tokens=2048,
                )
                path_m = re.search(r"(?:to|at|in)\s+(/[^\s,]+|\S+\.\w+)", step["description"])
                path = path_m.group(1) if path_m else "output.txt"
                tc = ToolCall(tool="write_file", args={"path": path, "content": content})
                tr = self.registry.execute(tc)
            else:
                # 4B generates tool call JSON (fast)
                tool_raw = self._gen_fast(
                    TOOL_PROMPT.format(tool_descriptions=self.registry.describe_tools()),
                    f"Context:\n{step_ctx}\n\nTask: {step['description']}",
                    max_tokens=1200,
                )
                step_parsed = parse_json(tool_raw, {"tool_calls": []})
                step_calls = validate_tool_calls(step_parsed, self.registry.tool_names)
                if not step_calls:
                    step_calls = [{"tool": step["action"], "args": {}}]
                tc = ToolCall(tool=step_calls[0]["tool"], args=step_calls[0].get("args", {}))
                tr = self.registry.execute(tc)

            all_results.append(tr)
            output = str(tr.output) if tr.output else (tr.error or "")
            accumulated += f"\nStep {i+1} [{tr.tool}]: {output[:500]}"

            if not tr.success:
                consecutive_errors += 1
            else:
                consecutive_errors = 0

        # 35B synthesizes everything (smart)
        self.on_stage("brain", "Synthesizing...")
        synth_ctx = f"{context}\n\nExecution results:\n{accumulated}" if context else f"Execution results:\n{accumulated}"
        content = self._gen_brain(SYNTH_PROMPT, f"Context:\n{synth_ctx}\n\nQuery: {query}")

        conf, verified, flags = _verify(content, synth_ctx, all_results)
        if consecutive_errors >= 2:
            flags.append("Stopped: too many consecutive errors")

        self.on_stage("done", f"confidence={conf:.2f}")
        return AgentResult(content=content, confidence=conf, verified=verified,
                           route="agentic", tool_results=all_results, flags=flags,
                           plan_summary=plan_summary)

    # --- Helpers ---

    def _infer_tool(self, query: str) -> List[Dict[str, Any]]:
        q = query.lower()
        if any(w in q for w in ("find", "search")) and any(e in q for e in (".py", ".txt", ".md", "*.")):
            return [{"tool": "search_files", "args": {"pattern": "*.txt", "path": "."}}]
        if any(w in q for w in ("list", "directory", "ls")):
            return [{"tool": "list_dir", "args": {"path": "."}}]
        if "read" in q and "file" in q:
            return [{"tool": "read_file", "args": {"path": "."}}]
        return []

    def _fix_shell(self, calls: List[Dict]) -> List[Dict]:
        fixed = []
        for c in calls:
            if c.get("tool") != "run_shell":
                fixed.append(c)
                continue
            cmd = c.get("args", {}).get("command", "").lower()
            if cmd.startswith("find ") or "find " in cmd:
                m = re.search(r"-name\s+[\"']?\*?\.?(\w+)[\"']?", cmd)
                pattern = f"*.{m.group(1)}" if m else "*.txt"
                fixed.append({"tool": "search_files", "args": {"pattern": pattern, "path": "."}})
            elif cmd.startswith("ls "):
                fixed.append({"tool": "list_dir", "args": {"path": "."}})
            elif cmd.startswith("cat "):
                m = re.search(r"cat\s+(\S+)", cmd)
                fixed.append({"tool": "read_file", "args": {"path": m.group(1) if m else "."}})
            else:
                fixed.append(c)
        return fixed
