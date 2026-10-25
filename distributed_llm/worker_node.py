"""Isolated attention worker.

One process holds one token shard for one benchmark session. It never sees the
full prompt, the model weights, or the reference output. All data endpoints
require the per-session bearer token, and a worker refuses shards from any other
session.

Run directly:
    LAB_WORKER_TOKEN=... LAB_SESSION_ID=... LAB_WORKER_ID=0 \
        python -m distributed_llm.worker_node 43101
"""
from __future__ import annotations

import hmac
import io
import json
import math
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from .config import APP_VERSION, LIMITS, Limits
from .logging_utils import process_rss_mb
from .protocol import (
    block_bounds_batch,
    causal_local_stats_batch,
    local_stats_batch,
)

MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_GENERATION_BYTES = 32 * 1024 * 1024



class ShardRequest(BaseModel):
    session_id: str = Field(min_length=8, max_length=128)
    layer: int = Field(ge=0, le=1024)
    kv_head: int = Field(ge=0, le=256)
    q: List[List[float]]
    multiplicities: Optional[Dict[str, int]] = None
    subset: str = "assigned"


@dataclass
class Shard:
    session_id: str
    worker_id: int
    tokens: np.ndarray
    primary: np.ndarray
    layers: Tuple[int, ...]
    kv_heads: Tuple[int, ...]
    q_per_kv: int
    head_dim: int
    keys: Dict[Tuple[int, int], np.ndarray] = field(default_factory=dict)
    values: Dict[Tuple[int, int], np.ndarray] = field(default_factory=dict)
    centroids: Dict[Tuple[int, int], np.ndarray] = field(default_factory=dict)
    radii: Dict[Tuple[int, int], float] = field(default_factory=dict)
    created_at: float = 0.0
    last_access: float = 0.0
    bytes_received: int = 0

    @property
    def token_count(self) -> int:
        return int(self.tokens.shape[0])


@dataclass
class GenerationShard:
    num_workers: int
    q_per_kv: int
    head_dim: int
    layers: Tuple[int, ...]
    kv_heads: Tuple[int, ...]
    max_positions: int
    positions: Dict[Tuple[int, int], np.ndarray] = field(default_factory=dict)
    keys: Dict[Tuple[int, int], np.ndarray] = field(default_factory=dict)
    values: Dict[Tuple[int, int], np.ndarray] = field(default_factory=dict)
    created_at: float = 0.0
    last_access: float = 0.0
    bytes_received: int = 0

    @property
    def units(self) -> List[Tuple[int, int]]:
        return [(layer, kv) for layer in self.layers for kv in self.kv_heads]

    @property
    def position_count(self) -> int:
        return max((len(p) for p in self.positions.values()), default=0)


class WorkerState:
    def __init__(self, token: str, session_id: str, worker_id: int, limits: Limits):
        self.token = token
        self.session_id = session_id
        self.worker_id = worker_id
        self.limits = limits
        self.lock = threading.RLock()
        self.shard: Optional[Shard] = None
        self.generation: Optional[GenerationShard] = None
        self.started_at = time.monotonic()
        self.counters = {
            "uploads": 0,
            "bounds_calls": 0,
            "stats_calls": 0,
            "gen_resets": 0,
            "gen_appends": 0,
            "gen_attend_calls": 0,
            "gen_tokens": 0,
            "rejected": 0,
            "bytes_received": 0,
        }


    def authorized(self, authorization: Optional[str]) -> bool:
        if not authorization or not authorization.startswith("Bearer "):
            return False
        presented = authorization[len("Bearer ") :].strip()
        return hmac.compare_digest(presented, self.token)

    def require_session(self, session_id: str) -> None:
        if not hmac.compare_digest(session_id, self.session_id):
            self.counters["rejected"] += 1
            raise HTTPException(status_code=403, detail="session mismatch")

    def live_shard(self) -> Shard:
        shard = self.shard
        if shard is None:
            raise HTTPException(status_code=409, detail="no shard loaded")
        if time.monotonic() - shard.last_access >= self.limits.session_ttl_s:
            self.shard = None
            self.counters["rejected"] += 1
            raise HTTPException(status_code=410, detail="session expired")
        shard.last_access = time.monotonic()
        return shard

    def live_generation(self) -> GenerationShard:
        generation = self.generation
        if generation is None:
            raise HTTPException(status_code=409, detail="no generation session initialized")
        if time.monotonic() - generation.last_access >= self.limits.session_ttl_s:
            self.generation = None
            self.counters["rejected"] += 1
            raise HTTPException(status_code=410, detail="generation session expired")
        generation.last_access = time.monotonic()
        return generation

    def clear(self) -> None:
        self.shard = None
        self.generation = None



def _require_auth(request: Request, state: WorkerState) -> None:
    if not state.authorized(request.headers.get("authorization")):
        state.counters["rejected"] += 1
        raise HTTPException(status_code=401, detail="invalid or missing bearer token")


async def _read_body(request: Request, limit: int) -> bytes:
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            if int(declared) > limit:
                raise HTTPException(status_code=413, detail="request body too large")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid content-length") from exc
    chunks: List[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > limit:
            raise HTTPException(status_code=413, detail="request body too large")
        chunks.append(chunk)
    return b"".join(chunks)


def _parse_int_csv(raw: str, name: str, upper: int) -> Tuple[int, ...]:
    try:
        values = tuple(sorted({int(part) for part in raw.split(",") if part.strip() != ""}))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"invalid {name}") from exc
    if not values or values[0] < 0 or values[-1] > upper:
        raise HTTPException(status_code=400, detail=f"{name} out of range")
    return values


def _validate_query(q: Sequence[Sequence[float]], shard: Shard, limits: Limits) -> np.ndarray:
    try:
        matrix = np.asarray(q, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="q must be a numeric matrix") from exc
    if matrix.ndim != 2 or matrix.shape != (shard.q_per_kv, shard.head_dim):
        raise HTTPException(
            status_code=422,
            detail=f"q must have shape ({shard.q_per_kv}, {shard.head_dim})",
        )
    if matrix.shape[0] > limits.max_query_heads:
        raise HTTPException(status_code=422, detail="too many query heads")
    if not np.isfinite(matrix).all():
        raise HTTPException(status_code=422, detail="q contains non-finite values")
    return matrix


def create_app(
    token: str,
    session_id: str,
    worker_id: int,
    limits: Limits = LIMITS,
    ready_file: Optional[str] = None,
    port: int = 0,
) -> FastAPI:
    state = WorkerState(token, session_id, worker_id, limits)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if ready_file:
            Path(ready_file).write_text(
                json.dumps({"worker_id": worker_id, "port": port, "pid": os.getpid()}),
                encoding="utf-8",
            )
        try:
            yield
        finally:
            if ready_file:
                try:
                    Path(ready_file).unlink()
                except FileNotFoundError:
                    pass

    app = FastAPI(
        title="Tessera Worker",
        version=APP_VERSION,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def reject_oversized(request: Request, call_next):
        if request.method == "POST":
            if request.url.path == "/v1/shard":
                limit = limits.max_request_bytes
            elif request.url.path.startswith("/v1/gen/"):
                limit = MAX_GENERATION_BYTES
            else:
                limit = MAX_JSON_BYTES
            declared = request.headers.get("content-length")
            if declared is not None:
                try:
                    too_big = int(declared) > limit
                except ValueError:
                    return JSONResponse({"detail": "invalid content-length"}, status_code=400)
                if too_big:
                    state.counters["rejected"] += 1
                    return JSONResponse({"detail": "request body too large"}, status_code=413)
        return await call_next(request)

    @app.get("/health")
    def health():
        return {"status": "alive", "service": "attention-worker", "version": APP_VERSION}

    @app.get("/v1/status")
    def status(request: Request):
        _require_auth(request, state)
        with state.lock:
            shard = state.shard
            generation = state.generation
            return {
                "worker_id": state.worker_id,
                "session_id": state.session_id,
                "version": APP_VERSION,
                "uptime_s": round(time.monotonic() - state.started_at, 2),
                "rss_mb": round(process_rss_mb(), 1),
                "loaded": shard is not None,
                "token_count": shard.token_count if shard else 0,
                "primary_count": int(shard.primary.sum()) if shard else 0,
                "layers": list(shard.layers) if shard else [],
                "kv_heads": list(shard.kv_heads) if shard else [],
                "head_dim": shard.head_dim if shard else 0,
                "generation_loaded": generation is not None,
                "generation_positions": generation.position_count if generation else 0,
                "generation_units": len(generation.units) if generation else 0,
                "generation_head_dim": generation.head_dim if generation else 0,
                "bytes_received": state.counters["bytes_received"],
                "counters": dict(state.counters),
            }

    @app.post("/v1/shard")
    async def upload_shard(
        request: Request,
        worker_id_param: int = Query(alias="worker_id", ge=0, le=4096),
        session_id_param: str = Query(alias="session_id", min_length=8, max_length=128),
        layers_raw: str = Query(alias="layers"),
        kv_heads_raw: str = Query(alias="kv_heads"),
        q_per_kv: int = Query(ge=1, le=64),
    ):
        _require_auth(request, state)
        state.require_session(session_id_param)
        if worker_id_param != state.worker_id:
            raise HTTPException(status_code=403, detail="worker id mismatch")
        layers = _parse_int_csv(layers_raw, "layers", 1024)
        kv_heads = _parse_int_csv(kv_heads_raw, "kv_heads", 256)

        body = await _read_body(request, limits.max_request_bytes)
        if len(body) == 0:
            raise HTTPException(status_code=400, detail="empty shard body")

        try:
            archive = np.load(io.BytesIO(body), allow_pickle=False)
            with archive as npz:
                tokens = np.asarray(npz["tokens"], dtype=np.int64)
                primary = np.asarray(npz["primary"], dtype=bool)
                arrays: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
                for layer in layers:
                    for kv_head in kv_heads:
                        k = np.asarray(npz[f"k_{layer}_{kv_head}"], dtype=np.float64)
                        v = np.asarray(npz[f"v_{layer}_{kv_head}"], dtype=np.float64)
                        arrays[(layer, kv_head)] = (k, v)
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=f"shard is missing array {exc}") from exc
        except (OSError, ValueError, EOFError) as exc:
            raise HTTPException(status_code=400, detail="shard is not a valid npz archive") from exc

        n = int(tokens.shape[0])
        if n < 1 or n > limits.max_context_tokens:
            raise HTTPException(status_code=422, detail="shard token count out of range")
        if primary.shape != (n,):
            raise HTTPException(status_code=422, detail="shard metadata length mismatch")
        if np.any(tokens < 0) or np.any(np.diff(tokens) <= 0):
            raise HTTPException(status_code=422, detail="token positions must be sorted and unique")
        if not np.any(primary):
            raise HTTPException(status_code=422, detail="empty primary set")

        head_dim = 0
        centroids: Dict[Tuple[int, int], np.ndarray] = {}
        radii: Dict[Tuple[int, int], float] = {}
        checked_keys: Dict[Tuple[int, int], np.ndarray] = {}
        checked_values: Dict[Tuple[int, int], np.ndarray] = {}
        for key, (k, v) in arrays.items():
            if k.ndim != 2 or v.ndim != 2 or k.shape != v.shape or k.shape[0] != n:
                raise HTTPException(status_code=422, detail="K/V shape mismatch")
            if head_dim == 0:
                head_dim = int(k.shape[1])
            if k.shape[1] != head_dim or head_dim < 8 or head_dim > 4096:
                raise HTTPException(status_code=422, detail="inconsistent head dimension")
            if not np.isfinite(k).all() or not np.isfinite(v).all():
                raise HTTPException(status_code=422, detail="K/V contains non-finite values")
            centroid = k.mean(axis=0)
            diffs = k - centroid
            radius = float(np.sqrt(np.max(np.sum(diffs * diffs, axis=1))))
            centroids[key] = centroid
            radii[key] = radius
            checked_keys[key] = k
            checked_values[key] = v

        with state.lock:
            existing = state.shard
            if existing is not None and existing.worker_id != state.worker_id:
                raise HTTPException(status_code=409, detail="worker already owns another shard")
            now = time.monotonic()
            state.shard = Shard(
                session_id=state.session_id,
                worker_id=state.worker_id,
                tokens=tokens.copy(),
                primary=primary.copy(),
                layers=layers,
                kv_heads=kv_heads,
                q_per_kv=int(q_per_kv),
                head_dim=head_dim,
                keys=checked_keys,
                values=checked_values,
                centroids=centroids,
                radii=radii,
                created_at=now,
                last_access=now,
                bytes_received=len(body),
            )
            state.counters["uploads"] += 1
            state.counters["bytes_received"] += len(body)
        return {
            "status": "loaded",
            "worker_id": state.worker_id,
            "token_count": n,
            "primary_count": int(primary.sum()),
            "layers": list(layers),
            "kv_heads": list(kv_heads),
            "head_dim": head_dim,
            "bytes": len(body),
        }

    @app.post("/v1/bounds")
    def compute_bounds(request: Request, payload: ShardRequest):
        _require_auth(request, state)
        state.require_session(payload.session_id)
        started = time.perf_counter()
        with state.lock:
            shard = state.live_shard()
            key = (payload.layer, payload.kv_head)
            if key not in shard.keys:
                raise HTTPException(status_code=404, detail="layer and kv head not in this shard")
            q = _validate_query(payload.q, shard, limits)
            bounds = block_bounds_batch(
                q, shard.centroids[key], shard.radii[key], 1.0 / math.sqrt(shard.head_dim)
            )
            state.counters["bounds_calls"] += 1
        return {
            "worker_id": state.worker_id,
            "bounds": bounds.tolist(),
            "token_count": shard.token_count,
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }

    @app.post("/v1/stats")
    def compute_stats(request: Request, payload: ShardRequest):
        _require_auth(request, state)
        state.require_session(payload.session_id)
        if payload.subset not in {"assigned", "primary"}:
            raise HTTPException(status_code=422, detail="subset must be assigned or primary")
        multiplicities = payload.multiplicities
        if multiplicities is not None and len(multiplicities) > limits.max_multiplicity_entries:
            raise HTTPException(status_code=422, detail="too many multiplicity entries")
        started = time.perf_counter()
        with state.lock:
            shard = state.live_shard()
            key = (payload.layer, payload.kv_head)
            if key not in shard.keys:
                raise HTTPException(status_code=404, detail="layer and kv head not in this shard")
            q = _validate_query(payload.q, shard, limits)
            if payload.subset == "primary":
                selector = shard.primary
                effective_multiplicities = None
            else:
                selector = np.ones(shard.token_count, dtype=bool)
                effective_multiplicities = multiplicities
            stats = local_stats_batch(
                q,
                shard.keys[key][selector],
                shard.values[key][selector],
                shard.tokens[selector],
                effective_multiplicities,
                1.0 / math.sqrt(shard.head_dim),
            )
            state.counters["stats_calls"] += 1
        return {
            "worker_id": state.worker_id,
            "layer": payload.layer,
            "kv_head": payload.kv_head,
            "lmax": np.asarray(stats["lmax"]).tolist(),
            "sumexp": np.asarray(stats["sumexp"]).tolist(),
            "local_out": np.asarray(stats["local_out"]).tolist(),
            "token_count": stats["token_count"],
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }

    @app.post("/v1/gen/reset")
    def reset_generation(
        request: Request,
        worker_id_param: int = Query(alias="worker_id", ge=0, le=4096),
        session_id_param: str = Query(alias="session_id", min_length=8, max_length=128),
        num_workers: int = Query(ge=1, le=4096),
        layers_raw: str = Query(alias="layers"),
        kv_heads_raw: str = Query(alias="kv_heads"),
        q_per_kv: int = Query(ge=1, le=64),
        head_dim: int = Query(ge=8, le=4096),
        max_positions: int = Query(ge=1, le=32768),
    ):
        _require_auth(request, state)
        state.require_session(session_id_param)
        if worker_id_param != state.worker_id:
            raise HTTPException(status_code=403, detail="worker id mismatch")
        if state.worker_id >= num_workers:
            raise HTTPException(status_code=422, detail="worker id exceeds the worker count")
        if max_positions > limits.max_generation_positions:
            raise HTTPException(
                status_code=422, detail="max_positions exceeds the generation limit"
            )
        layers = _parse_int_csv(layers_raw, "layers", 1024)
        kv_heads = _parse_int_csv(kv_heads_raw, "kv_heads", 256)
        with state.lock:
            now = time.monotonic()
            generation = GenerationShard(
                num_workers=num_workers,
                q_per_kv=q_per_kv,
                head_dim=head_dim,
                layers=layers,
                kv_heads=kv_heads,
                max_positions=max_positions,
                created_at=now,
                last_access=now,
            )
            for layer in layers:
                for kv_head in kv_heads:
                    unit = (layer, kv_head)
                    generation.positions[unit] = np.zeros(0, dtype=np.int64)
                    generation.keys[unit] = np.zeros(
                        (0, head_dim), dtype=np.float32
                    )
                    generation.values[unit] = np.zeros(
                        (0, head_dim), dtype=np.float32
                    )
            state.generation = generation
            state.counters["gen_resets"] += 1
        return {
            "status": "initialized",
            "worker_id": state.worker_id,
            "num_workers": num_workers,
            "layers": list(layers),
            "kv_heads": list(kv_heads),
            "q_per_kv": q_per_kv,
            "head_dim": head_dim,
            "max_positions": max_positions,
        }

    @app.post("/v1/gen/append")
    async def append_generation(
        request: Request,
        worker_id_param: int = Query(alias="worker_id", ge=0, le=4096),
        session_id_param: str = Query(alias="session_id", min_length=8, max_length=128),
    ):
        _require_auth(request, state)
        state.require_session(session_id_param)
        if worker_id_param != state.worker_id:
            raise HTTPException(status_code=403, detail="worker id mismatch")
        body = await _read_body(request, MAX_GENERATION_BYTES)
        if len(body) == 0:
            raise HTTPException(status_code=400, detail="empty append body")

        with state.lock:
            generation = state.live_generation()
            units = set(generation.units)
            try:
                archive = np.load(io.BytesIO(body), allow_pickle=False)
                with archive as npz:
                    names = set(npz.files)
                    if "positions" not in names:
                        raise HTTPException(
                            status_code=422, detail="append archive has no positions"
                        )
                    raw_positions = np.asarray(npz["positions"])
                    if raw_positions.ndim != 1 or raw_positions.dtype.kind not in "iu":
                        raise HTTPException(
                            status_code=422, detail="positions must be a 1-D integer array"
                        )
                    positions = raw_positions.astype(np.int64, copy=True)

                    array_units: Dict[str, Dict[Tuple[int, int], np.ndarray]] = {
                        "k": {},
                        "v": {},
                    }
                    for name in names - {"positions"}:
                        parts = name.split("_")
                        if len(parts) != 3 or parts[0] not in array_units:
                            raise HTTPException(
                                status_code=422,
                                detail="append archive has unexpected arrays",
                            )
                        try:
                            unit = (int(parts[1]), int(parts[2]))
                        except ValueError as exc:
                            raise HTTPException(
                                status_code=422,
                                detail="append archive has invalid unit names",
                            ) from exc
                        if unit not in units:
                            raise HTTPException(
                                status_code=404,
                                detail="append references a layer or kv head outside this session",
                            )
                        array_units[parts[0]][unit] = np.asarray(
                            npz[name], dtype=np.float32
                        )
                    if not array_units["k"] or array_units["k"].keys() != array_units["v"].keys():
                        raise HTTPException(
                            status_code=422,
                            detail="append must provide matching K/V unit blocks",
                        )
                    blocks = {
                        unit: (array_units["k"][unit], array_units["v"][unit])
                        for unit in array_units["k"]
                    }
                    for k, v in blocks.values():
                        if (
                            k.ndim != 2
                            or v.ndim != 2
                            or k.shape != v.shape
                            or k.shape[0] != positions.shape[0]
                            or k.shape[1] != generation.head_dim
                        ):
                            raise HTTPException(
                                status_code=422, detail="K/V shape mismatch"
                            )
                        if not np.isfinite(k).all() or not np.isfinite(v).all():
                            raise HTTPException(
                                status_code=422, detail="K/V contains non-finite values"
                            )
            except HTTPException:
                raise
            except (KeyError, ValueError, EOFError, OSError) as exc:
                raise HTTPException(
                    status_code=422, detail="append body is not a valid npz archive"
                ) from exc

            count = int(positions.shape[0])
            if count < 1:
                raise HTTPException(status_code=422, detail="append contains no positions")
            if np.any(np.diff(positions) <= 0):
                raise HTTPException(
                    status_code=422, detail="append positions must be sorted and unique"
                )
            if positions[0] < 0 or positions[-1] >= generation.max_positions:
                raise HTTPException(status_code=422, detail="position out of range")
            if np.any(positions % generation.num_workers != state.worker_id):
                raise HTTPException(
                    status_code=403, detail="worker received positions it does not own"
                )

            overlap_by_unit = {}
            for unit in blocks:
                existing = generation.positions[unit]
                overlap = np.isin(positions, existing)
                overlap_by_unit[unit] = overlap
                if overlap.any() and not overlap.all():
                    raise HTTPException(
                        status_code=409,
                        detail="append mixes new and already stored positions",
                    )
            if any(overlap.all() for overlap in overlap_by_unit.values()):
                if not all(overlap.all() for overlap in overlap_by_unit.values()):
                    raise HTTPException(
                        status_code=409,
                        detail="units disagree on already stored positions",
                    )
                for unit, (k, v) in blocks.items():
                    existing = generation.positions[unit]
                    rows = np.searchsorted(existing, positions)
                    if not (
                        np.array_equal(generation.keys[unit][rows], k)
                        and np.array_equal(generation.values[unit][rows], v)
                    ):
                        raise HTTPException(
                            status_code=409,
                            detail="append conflicts with stored K/V",
                        )
                state.counters["gen_appends"] += 1
                state.counters["bytes_received"] += len(body)
                return {
                    "status": "duplicate",
                    "worker_id": state.worker_id,
                    "positions": count,
                    "stored_positions": generation.position_count,
                }

            for unit in blocks:
                if len(generation.positions[unit]) + count > generation.max_positions:
                    raise HTTPException(status_code=422, detail="generation cache is full")
            for unit, (k, v) in blocks.items():
                existing = generation.positions[unit]
                combined = np.concatenate([existing, positions])
                order = np.argsort(combined, kind="stable")
                generation.keys[unit] = np.concatenate(
                    [generation.keys[unit], k], axis=0
                )[order]
                generation.values[unit] = np.concatenate(
                    [generation.values[unit], v], axis=0
                )[order]
                generation.positions[unit] = combined[order]
            generation.bytes_received += len(body)
            generation.last_access = time.monotonic()
            state.counters["gen_appends"] += 1
            state.counters["gen_tokens"] += count
            state.counters["bytes_received"] += len(body)
            stored = generation.position_count
        return {
            "status": "appended",
            "worker_id": state.worker_id,
            "positions": count,
            "stored_positions": stored,
            "bytes": len(body),
        }

    @app.post("/v1/gen/attend")
    async def attend_generation(
        request: Request,
        worker_id_param: int = Query(alias="worker_id", ge=0, le=4096),
        session_id_param: str = Query(alias="session_id", min_length=8, max_length=128),
        layer: int = Query(ge=0, le=1024),
        kv_head: int = Query(ge=0, le=256),
    ):
        _require_auth(request, state)
        state.require_session(session_id_param)
        if worker_id_param != state.worker_id:
            raise HTTPException(status_code=403, detail="worker id mismatch")
        body = await _read_body(request, MAX_GENERATION_BYTES)
        if len(body) == 0:
            raise HTTPException(status_code=400, detail="empty attention body")

        with state.lock:
            generation = state.live_generation()
            unit = (layer, kv_head)
            if unit not in generation.keys:
                raise HTTPException(
                    status_code=404, detail="layer and kv head not in this generation session"
                )
            try:
                archive = np.load(io.BytesIO(body), allow_pickle=False)
                with archive as npz:
                    if set(npz.files) != {"positions", "q"}:
                        raise HTTPException(
                            status_code=422,
                            detail="attention archive must contain positions and q",
                        )
                    raw_positions = np.asarray(npz["positions"])
                    if raw_positions.ndim != 1 or raw_positions.dtype.kind not in "iu":
                        raise HTTPException(
                            status_code=422, detail="positions must be a 1-D integer array"
                        )
                    positions = raw_positions.astype(np.int64, copy=True)
                    q = np.asarray(npz["q"], dtype=np.float32)
            except HTTPException:
                raise
            except (KeyError, ValueError, EOFError, OSError) as exc:
                raise HTTPException(
                    status_code=422, detail="attention body is not a valid npz archive"
                ) from exc

            count = int(positions.shape[0])
            if count < 1:
                raise HTTPException(status_code=422, detail="attention contains no queries")
            if np.any(np.diff(positions) <= 0):
                raise HTTPException(
                    status_code=422, detail="query positions must be sorted and unique"
                )
            if positions[0] < 0 or positions[-1] >= generation.max_positions:
                raise HTTPException(status_code=422, detail="query position out of range")
            if q.shape != (count, generation.q_per_kv, generation.head_dim):
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"q must have shape ({count}, {generation.q_per_kv}, "
                        f"{generation.head_dim})"
                    ),
                )
            if not np.isfinite(q).all():
                raise HTTPException(status_code=422, detail="q contains non-finite values")

            started = time.perf_counter()
            stats = causal_local_stats_batch(
                q,
                generation.keys[unit],
                generation.values[unit],
                generation.positions[unit],
                positions,
                1.0 / math.sqrt(generation.head_dim),
            )
            state.counters["gen_attend_calls"] += 1
            elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
            owned_keys = len(generation.positions[unit])

        buffer = io.BytesIO()
        np.savez(
            buffer,
            positions=positions,
            lmax=np.asarray(stats["lmax"], dtype=np.float64),
            sumexp=np.asarray(stats["sumexp"], dtype=np.float64),
            out=np.asarray(stats["local_out"], dtype=np.float64),
            visible_keys=np.asarray(stats["visible_keys"], dtype=np.int64),
            owned_keys=np.int64(owned_keys),
            elapsed_ms=np.float64(elapsed_ms),
        )
        return Response(content=buffer.getvalue(), media_type="application/x-npz")

    @app.post("/v1/clear")
    def clear(request: Request):
        _require_auth(request, state)
        with state.lock:
            state.clear()
        return {"status": "cleared", "worker_id": state.worker_id}

    return app


def _load_environment() -> Tuple[str, str, int, int]:
    token = os.environ.get("LAB_WORKER_TOKEN", "")
    session_id = os.environ.get("LAB_SESSION_ID", "")
    worker_id = int(os.environ.get("LAB_WORKER_ID", "0"))
    port = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("LAB_WORKER_PORT", "8001"))
    if len(token) < 16:
        raise SystemExit("LAB_WORKER_TOKEN must contain at least 16 characters")
    if len(session_id) < 8:
        raise SystemExit("LAB_SESSION_ID must contain at least 8 characters")
    if not 1 <= port <= 65535:
        raise SystemExit("worker port is out of range")
    return token, session_id, worker_id, port


def main() -> None:
    token, session_id, worker_id, port = _load_environment()
    ready_file = os.environ.get("LAB_READY_FILE") or None
    application = create_app(token, session_id, worker_id, ready_file=ready_file, port=port)
    uvicorn.run(
        application,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        access_log=False,
        timeout_keep_alive=10,
        limit_concurrency=64,
    )


if __name__ == "__main__":
    main()
