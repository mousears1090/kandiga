"""OpenAI-compatible HTTP API server for Kandiga."""

from __future__ import annotations

import json
import time
import uuid
from typing import Optional

engine = None


def _create_app():
    """Create the FastAPI app. Deferred to avoid import cost at CLI parse time."""
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse, JSONResponse

    app = FastAPI(
        title="Kandiga",
        description="35B AI in 1.5GB RAM \u2014 OpenAI-compatible API",
        version="0.1.0",
    )

    @app.get("/")
    async def root():
        return {
            "name": "kandiga",
            "version": "0.1.0",
            "ready": engine.is_ready if engine else False,
        }

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": engine.model_path if engine else "unknown",
                    "object": "model",
                    "ready": engine.is_ready if engine else False,
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        messages = body.get("messages", [])
        stream = body.get("stream", False)
        max_tokens = body.get("max_tokens", 2048)
        temperature = body.get("temperature", 0.0)

        prompt = messages[-1]["content"] if messages else ""
        request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        if stream:
            async def generate_stream():
                for token in engine.generate(
                    prompt, max_tokens=max_tokens, temp=temperature, stream=True
                ):
                    chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": engine.model_path,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": token},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

                # Final chunk
                final = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": engine.model_path,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ],
                }
                yield f"data: {json.dumps(final)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                generate_stream(), media_type="text/event-stream"
            )
        else:
            # Non-streaming: collect full response
            response_text = ""
            for token in engine.generate(
                prompt, max_tokens=max_tokens, temp=temperature, stream=True
            ):
                response_text += token

            return {
                "id": request_id,
                "object": "chat.completion",
                "created": created,
                "model": engine.model_path,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": response_text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": -1,
                    "completion_tokens": -1,
                    "total_tokens": -1,
                },
            }

    @app.get("/health")
    async def health():
        if engine and engine.is_ready:
            return {"status": "ok"}
        return JSONResponse({"status": "loading"}, status_code=503)

    return app


def run_serve(
    port: int = 8340,
    fast: bool = False,
    model: Optional[str] = None,
):
    """Start the HTTP API server."""
    from rich.console import Console
    console = Console()

    console.print()
    console.print("[bold cyan]Kandiga Server[/]")
    console.print()

    from kandiga.engine import KandigaEngine
    global engine
    engine = KandigaEngine(model_path=model, fast_mode=fast)

    console.print("[dim]Loading model...[/]")
    t0 = time.time()
    engine.load()
    elapsed = time.time() - t0
    stats = engine.stats
    console.print(
        f"[green]Ready.[/] [dim]({elapsed:.1f}s, "
        f"RSS {stats['rss_mb']:.0f}MB, "
        f"Mode: {stats['mode']})[/]"
    )
    console.print()
    console.print(f"  [bold]Listening:[/] http://0.0.0.0:{port}")
    console.print(f"  [bold]API:[/]       http://localhost:{port}/v1/chat/completions")
    console.print(f"  [bold]Models:[/]    http://localhost:{port}/v1/models")
    console.print()

    app = _create_app()

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
