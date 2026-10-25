"""FastAPI application for Distributed Attention.

The server owns no model weights. Capture runs in a subprocess, attention
workers are spawned per run, and the dashboard only sees artifacts and events.
"""
from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel, Field

from distributed_llm.config import (
    ALLOWED_HOSTS,
    APP_VERSION,
    LIMITS,
    SESSION_SECRET,
)
from distributed_llm.manager import (
    RunBusyError,
    RunConfig,
    RunManager,
    RunManagerError,
    RunNotFoundError,
)

from .security import (
    SESSION_COOKIE,
    BodyLimitMiddleware,
    HostGuardMiddleware,
    LoginLimiter,
    OriginGuardMiddleware,
    SecurityHeadersMiddleware,
    SessionSigner,
    api_key_required,
    check_api_key,
    check_bearer,
)

BASE_DIR = Path(__file__).resolve().parent.parent
templates_dir = BASE_DIR / "app" / "templates"
static_dir = BASE_DIR / "app" / "static"

env = Environment(loader=FileSystemLoader(str(templates_dir)), autoescape=True)
signer = SessionSigner(SESSION_SECRET, LIMITS.session_ttl_s)
login_limiter = LoginLimiter()
API_BODY_LIMIT = min(LIMITS.max_request_bytes, 1024 * 1024)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.manager = RunManager()
    try:
        yield
    finally:
        await asyncio.to_thread(app.state.manager.shutdown, 20.0)


app = FastAPI(
    title="Distributed Attention",
    description="Distributed LLM inference across edge devices.",
    version=APP_VERSION,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)
app.add_middleware(BodyLimitMiddleware, max_bytes=API_BODY_LIMIT)
app.add_middleware(OriginGuardMiddleware)
app.add_middleware(HostGuardMiddleware, allowed_hosts=ALLOWED_HOSTS)
app.add_middleware(SecurityHeadersMiddleware)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


class RunRequest(BaseModel):
    prompt: str = Field(min_length=1)
    workers: int = Field(default=4, ge=LIMITS.min_workers, le=LIMITS.max_workers)
    max_tokens: int = Field(
        default=512, ge=16, le=LIMITS.max_context_tokens
    )
    overlap_pct: float = Field(default=0.0, ge=0.0, le=LIMITS.max_overlap_pct)
    strategy: str = Field(default="pseudorandom", pattern="^(pseudorandom|contiguous)$")
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    error_tol: float = Field(default=1e-3, gt=0.0, le=1.0)
    safe_gamma: float = Field(default=LIMITS.safe_gamma, ge=0.0, le=2.0)
    generate: bool = True
    max_new_tokens: int = Field(
        default=LIMITS.default_max_new_tokens,
        ge=1,
        le=LIMITS.max_generation_tokens,
    )
    layers: Optional[List[int]] = None
    kv_heads: Optional[List[int]] = None

    model_config = {"extra": "forbid"}


class LoginRequest(BaseModel):
    # Empty is legal because keyless local mode issues a session cookie to any
    # loopback client. When LAB_API_KEY is set, create_session still compares
    # this value and an empty key is rejected.
    api_key: str = Field(default="", max_length=512)

    model_config = {"extra": "forbid"}


def manager_for(request: Request) -> RunManager:
    manager = getattr(request.app.state, "manager", None)
    if manager is None:
        raise HTTPException(status_code=503, detail="run manager is not available")
    return manager


def auth_state(request: Request) -> tuple[bool, bool]:
    cookie_ok = signer.verify(request.cookies.get(SESSION_COOKIE))
    bearer_ok = check_bearer(request.headers.get("authorization"))
    return cookie_ok, bearer_ok


def require_auth(request: Request) -> None:
    cookie_ok, bearer_ok = auth_state(request)
    if not cookie_ok and not bearer_ok:
        raise HTTPException(status_code=401, detail="authentication required")


def _client_key(request: Request) -> str:
    return request.client.host if request.client else "unknown"


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok", "app_version": APP_VERSION}


@app.get("/readyz")
async def readyz(request: Request) -> dict:
    manager = manager_for(request)
    return manager.readiness()


@app.get("/api/public-config")
async def public_config() -> dict:
    return {
        "app_version": APP_VERSION,
        "auth_required": api_key_required(),
        "limits": {
            "min_workers": LIMITS.min_workers,
            "max_workers": LIMITS.max_workers,
            "max_context_tokens": LIMITS.max_context_tokens,
            "default_context_tokens": LIMITS.default_context_tokens,
            "max_overlap_pct": LIMITS.max_overlap_pct,
            "max_prompt_chars": LIMITS.max_prompt_chars,
            "session_ttl_s": LIMITS.session_ttl_s,
            "max_generation_tokens": LIMITS.max_generation_tokens,
            "default_max_new_tokens": LIMITS.default_max_new_tokens,
            "max_generation_positions": LIMITS.max_generation_positions,
            "generation_timeout_s": LIMITS.generation_timeout_s,
            "generation_step_timeout_s": LIMITS.generation_step_timeout_s,
        },
    }


def _set_session(response: Response) -> None:
    response.set_cookie(
        SESSION_COOKIE,
        signer.issue(),
        max_age=signer.ttl_s,
        httponly=True,
        samesite="strict",
        secure=False,
        path="/",
    )


@app.post("/api/session")
async def create_session(payload: LoginRequest, request: Request, response: Response):
    client = _client_key(request)
    if login_limiter.blocked(client):
        raise HTTPException(status_code=429, detail="too many failed attempts")
    if api_key_required() and not check_api_key(payload.api_key):
        login_limiter.failure(client)
        raise HTTPException(status_code=401, detail="invalid API key")
    login_limiter.success(client)
    _set_session(response)
    return {"ok": True, "auth_required": api_key_required()}


@app.post("/api/logout")
async def logout(response: Response) -> dict:
    response.delete_cookie(SESSION_COOKIE, path="/")
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    template = env.get_template("index.html")
    response = HTMLResponse(
        template.render(
            app_version=APP_VERSION,
            auth_required=api_key_required(),
            max_workers=LIMITS.max_workers,
            min_workers=LIMITS.min_workers,
        )
    )
    cookie_ok, bearer_ok = auth_state(request)
    if (not api_key_required() or cookie_ok or bearer_ok) and not cookie_ok:
        _set_session(response)
    return response


api = APIRouter(prefix="/api", dependencies=[Depends(require_auth)])


@api.get("/status")
async def status(request: Request) -> dict:
    manager = manager_for(request)
    record = manager.current()
    return {
        "app_version": APP_VERSION,
        "auth_required": api_key_required(),
        "metrics": manager.metrics(),
        "current_run": record.status_payload() if record else None,
    }


@api.get("/metrics")
async def metrics(request: Request) -> dict:
    return manager_for(request).metrics()


@api.post("/runs", status_code=202)
async def create_run(payload: RunRequest, request: Request) -> dict:
    manager = manager_for(request)
    try:
        config = RunConfig(
            prompt=payload.prompt,
            workers=payload.workers,
            max_tokens=payload.max_tokens,
            overlap_pct=payload.overlap_pct,
            strategy=payload.strategy,
            seed=payload.seed,
            error_tol=payload.error_tol,
            safe_gamma=payload.safe_gamma,
            generate=payload.generate,
            max_new_tokens=payload.max_new_tokens,
            layers=tuple(payload.layers) if payload.layers else None,
            kv_heads=tuple(payload.kv_heads) if payload.kv_heads else None,
        )
        record = manager.start(config)
    except RunBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RunManagerError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"run": record.status_payload()}


@api.get("/runs/current")
async def current_run(request: Request) -> dict:
    record = manager_for(request).current()
    return {"run": record.status_payload() if record else None}


@api.get("/runs/{run_id}")
async def get_run(run_id: str, request: Request) -> dict:
    try:
        record = manager_for(request).get(run_id)
    except RunNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"run": record.status_payload()}


@api.post("/runs/{run_id}/cancel")
async def cancel_run(run_id: str, request: Request) -> dict:
    try:
        cancelled = manager_for(request).cancel(run_id)
    except RunNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"cancelled": cancelled}


@api.get("/runs/{run_id}/events")
async def run_events(run_id: str, request: Request, after: int = 0) -> StreamingResponse:
    try:
        record = manager_for(request).get(run_id)
    except RunNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if after < 0 or after > 2**31:
        raise HTTPException(status_code=422, detail="after must be non-negative")

    async def stream():
        seq = after
        last_ping = time.monotonic()
        while True:
            if await request.is_disconnected():
                break
            events = record.stream.since(seq)
            if events:
                for event in events:
                    seq = int(event["seq"])
                    payload = json.dumps(
                        event, ensure_ascii=False, separators=(",", ":")
                    )
                    yield f"id: {seq}\ndata: {payload}\n\n"
                last_ping = time.monotonic()
            elif record.stream.closed:
                yield "event: end\ndata: {}\n\n"
                break
            if time.monotonic() - last_ping >= 15.0:
                yield ": ping\n\n"
                last_ping = time.monotonic()
            await asyncio.sleep(0.2)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
        },
    )


@api.get("/runs/{run_id}/result")
async def run_result(run_id: str, request: Request):
    try:
        record = manager_for(request).get(run_id)
    except RunNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    summary_path = record.run_dir / "summary.json"
    if not summary_path.is_file():
        raise HTTPException(
            status_code=409, detail="run has not produced a summary yet"
        )
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail="summary is unreadable") from exc
    return JSONResponse(payload)


@api.get("/runs/{run_id}/artifacts")
async def run_artifacts(run_id: str, request: Request) -> dict:
    manager = manager_for(request)
    try:
        items = manager.list_artifacts(run_id)
    except RunNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"artifacts": items}


@api.get("/runs/{run_id}/artifacts/{name}")
async def download_artifact(run_id: str, name: str, request: Request):
    manager = manager_for(request)
    try:
        path = manager.artifact_path(run_id, name)
    except RunNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    media = "application/octet-stream"
    if name.endswith(".json"):
        media = "application/json"
    elif name.endswith(".jsonl"):
        media = "application/x-ndjson"
    return FileResponse(path, media_type=media, filename=name)


app.include_router(api)
