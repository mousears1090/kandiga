"""Agent HTTP server — web UI + API endpoints."""

import json
import os
import time
import uuid
from typing import Optional

from pydantic import BaseModel


class QueryBody(BaseModel):
    query: str
    context: str = ""


class MemoryBody(BaseModel):
    content: str
    category: str = "user"


_engine = None
_pipeline = None
_memory = None
_skills = None


def create_agent_app(fast: bool = False, model: Optional[str] = None):
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

    app = FastAPI(title="Kandiga Agent", version="0.1.0")

    @app.on_event("startup")
    async def startup():
        global _engine, _pipeline, _memory, _skills

        from kandiga.engine import KandigaEngine
        from kandiga.agents.pipeline import AgentPipeline
        from kandiga.agents.tools import default_tools
        from kandiga.agents.macos import register_macos_tools
        from kandiga.agents.memory import Memory
        from kandiga.agents.skills import SkillEngine

        model_path = model or KandigaEngine.DEFAULT_MODEL
        _engine = KandigaEngine(model_path=model_path, fast_mode=fast, log_memory=True)
        _engine.load()

        registry = default_tools()
        register_macos_tools(registry)

        _pipeline = AgentPipeline(_engine, registry=registry)
        _pipeline.start_session()

        _memory = Memory()
        _skills = SkillEngine()
        _skills.load_all()

    @app.on_event("shutdown")
    async def shutdown():
        if _pipeline:
            _pipeline.end_session()

    @app.get("/", response_class=HTMLResponse)
    async def index():
        html_path = os.path.join(os.path.dirname(__file__), "..", "static", "agent.html")
        if os.path.isfile(html_path):
            with open(html_path) as f:
                return f.read()
        return "<h1>Kandiga Agent</h1><p>UI not found</p>"

    @app.get("/api/status")
    async def status():
        return {
            "status": "ok" if _engine and _engine.is_ready else "loading",
            "model": _engine.model_path if _engine else None,
            "fast_mode": _engine.fast_mode if _engine else False,
            "tools": len(_pipeline.registry.tool_names) if _pipeline else 0,
            "skills": len(_skills.list_all()) if _skills else 0,
            "memory": _memory.stats if _memory else {},
            "session_active": _pipeline._session_active if _pipeline else False,
        }

    @app.post("/api/query")
    async def do_query(data: QueryBody):
        q = data.query
        context = data.context
        if _memory:
            mem_ctx = _memory.build_context(q, max_chars=1000)
            if mem_ctx:
                context = f"Memory:\n{mem_ctx}\n\n{context}" if context else f"Memory:\n{mem_ctx}"

        result = _pipeline.run(q, context=context)

        if _memory:
            _memory.log_daily(f"Q: {q[:100]}\nRoute: {result.route} Conf: {result.confidence:.2f}")

        return JSONResponse(result.to_dict())

    @app.post("/api/stream")
    async def do_stream(data: QueryBody):
        q = data.query

        async def generate():
            context = data.context
            if _memory:
                mem_ctx = _memory.build_context(q, max_chars=1000)
                if mem_ctx:
                    context = f"Memory:\n{mem_ctx}\n\n{context}" if context else f"Memory:\n{mem_ctx}"

            stages = []
            def on_stage(stage, detail):
                stages.append({"stage": stage, "detail": detail})

            old_cb = _pipeline.on_stage
            _pipeline.on_stage = on_stage
            result = _pipeline.run(q, context=context)
            _pipeline.on_stage = old_cb

            for s in stages:
                yield f"data: {json.dumps({'type': 'stage', **s})}\n\n"
            yield f"data: {json.dumps({'type': 'result', **result.to_dict()})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    @app.post("/api/session/start")
    async def session_start():
        if _pipeline:
            _pipeline.start_session()
        return {"status": "started"}

    @app.post("/api/session/end")
    async def session_end():
        if _pipeline:
            _pipeline.end_session()
        return {"status": "ended"}

    @app.get("/api/tools")
    async def list_tools():
        if not _pipeline:
            return {"tools": []}
        return {"tools": sorted(_pipeline.registry.tool_names)}

    @app.get("/api/skills")
    async def list_skills():
        if not _skills:
            return {"skills": []}
        return {"skills": [s.to_dict() for s in _skills.list_all()]}

    @app.get("/api/memory")
    async def read_memory():
        if not _memory:
            return {"memory": "", "stats": {}}
        return {"memory": _memory.read_memory(), "stats": _memory.stats}

    @app.post("/api/memory")
    async def add_memory(data: MemoryBody):
        if not data.content:
            return JSONResponse({"error": "content required"}, status_code=400)
        if _memory:
            _memory.add_memory(data.content, category=data.category)
        return {"status": "saved"}

    return app


def run_agent_serve(port: int = 3000, fast: bool = False, model: Optional[str] = None):
    import uvicorn
    app = create_agent_app(fast=fast, model=model)
    print(f"Kandiga Agent — http://localhost:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
