"""Lifecycle and HTTP client for local attention workers.

Every run gets fresh worker processes, fresh ports, a fresh bearer token, and a
fresh session id. Workers are spawned on loopback only and are always torn down,
including when startup or a request fails.
"""
from __future__ import annotations

import json
import os
import secrets
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import httpx
import numpy as np

from .config import BASE_DIR, LIMITS, Limits
from .logging_utils import available_ram_gb

MIN_TOKEN_LENGTH = 24
ESTIMATED_WORKER_RSS_MB = 72.0
SPAWN_HEADROOM_GB = 0.25
# Appended KV arrays are grown per unit, so admission keeps a transient margin.
KV_GROWTH_FACTOR = 1.25


class ClusterError(RuntimeError):
    pass


class WorkerCallError(RuntimeError):
    def __init__(self, worker_id: int, path: str, detail: str):
        super().__init__(f"worker {worker_id} {path} failed: {detail}")
        self.worker_id = worker_id
        self.path = path
        self.detail = detail


@dataclass
class WorkerHandle:
    worker_id: int
    port: int
    process: subprocess.Popen
    log_path: Path
    ready_path: Path
    base_url: str


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _tail(path: Path, lines: int = 20) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(content[-lines:])


class WorkerCluster:
    def __init__(
        self,
        num_workers: int,
        run_dir: Path,
        event_cb: Optional[Callable[[dict], None]] = None,
        limits: Limits = LIMITS,
    ):
        if not limits.min_workers <= num_workers <= limits.max_workers:
            raise ClusterError(
                f"worker count must be between {limits.min_workers} and {limits.max_workers}"
            )
        self.num_workers = num_workers
        self.run_dir = Path(run_dir)
        self.worker_dir = self.run_dir / "workers"
        self.worker_dir.mkdir(parents=True, exist_ok=True)
        self.event_cb = event_cb
        self.limits = limits
        self.session_id = secrets.token_hex(16)
        self.token = secrets.token_urlsafe(32)
        if len(self.token) < MIN_TOKEN_LENGTH:
            raise ClusterError("generated worker token is too short")
        self.workers: List[WorkerHandle] = []
        self._client: Optional[httpx.Client] = None
        self._executor: Optional[ThreadPoolExecutor] = None
        self.started = False
        self.metrics = {
            "requests": 0,
            "retries": 0,
            "failures": 0,
            "bytes_sent": 0,
            "bytes_received": 0,
            "worker_wall_ms": 0.0,
            "workers_spawned": 0,
        }

    def _emit(self, message: str, level: str = "info", **extra) -> None:
        if self.event_cb is None:
            return
        payload = {"type": "comm", "msg": message, "level": level}
        payload.update(extra)
        try:
            self.event_cb(payload)
        except Exception:
            pass

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}

    def preflight(self) -> float:
        needed_gb = SPAWN_HEADROOM_GB + self.num_workers * ESTIMATED_WORKER_RSS_MB / 1024.0
        free = available_ram_gb()
        if 0 <= free < needed_gb:
            raise ClusterError(
                f"starting {self.num_workers} workers needs about {needed_gb:.2f} GB "
                f"of free RAM, only {free:.2f} GB is available. Lower the worker count."
            )
        return free

    def preflight_generation(
        self,
        max_positions: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        process_headroom_gb: float,
    ) -> dict:
        """Admission check for the KV caches plus the decoder process.

        Call this after start(): the worker processes already exist, so the
        free-RAM figure already reflects their resident memory. Only the future
        KV growth and the decoder's measured peak need extra headroom.

        Rank-3 KV arrays are float32, stored twice (keys and values) for every
        layer and KV head, and each position lives on exactly one worker because
        ownership is strided, so the cluster total is independent of the worker
        count. A transient factor covers the array growth during appends.
        """
        if not self.started:
            raise ClusterError("generation preflight requires a started cluster")
        kv_total_bytes = (
            int(max_positions)
            * int(num_layers)
            * int(num_kv_heads)
            * int(head_dim)
            * 4
            * 2
        )
        kv_total_gb = kv_total_bytes * KV_GROWTH_FACTOR / 1024.0**3
        decoder_gb = max(0.0, float(process_headroom_gb))
        needed_gb = kv_total_gb + decoder_gb
        free = available_ram_gb()
        if 0 <= free < needed_gb:
            raise ClusterError(
                f"distributed generation needs about {needed_gb:.2f} GB of free RAM "
                f"(KV {kv_total_gb:.2f} GB across {self.num_workers} workers, "
                f"decoder {decoder_gb:.2f} GB), only {free:.2f} GB is available. "
                f"Lower the generation token budget or the worker count."
            )
        workers = max(1, int(self.num_workers))
        return {
            "free_ram_gb": round(free, 3),
            "needed_ram_gb": round(needed_gb, 3),
            "kv_total_mb": round(kv_total_bytes / 1e6, 3),
            "kv_bytes_per_worker": round(kv_total_bytes / workers, 1),
            "decoder_headroom_gb": round(decoder_gb, 3),
        }

    def connection_info(self) -> dict:
        if not self.started:
            raise ClusterError("cluster is not started")
        return {
            "session_id": self.session_id,
            "token": self.token,
            "workers": [
                {"worker_id": handle.worker_id, "base_url": handle.base_url}
                for handle in self.workers
            ],
        }

    def start(self) -> None:
        if self.started:
            raise ClusterError("cluster is already started")
        free = self.preflight()
        self._emit(
            f"cluster: spawning {self.num_workers} isolated workers "
            f"(session {self.session_id[:8]}..., {free:.2f} GB free)",
            event="start",
        )
        self._client = httpx.Client(
            timeout=httpx.Timeout(self.limits.worker_request_timeout_s, connect=5.0),
            limits=httpx.Limits(max_connections=self.num_workers * 2, max_keepalive_connections=self.num_workers),
        )
        try:
            for worker_id in range(self.num_workers):
                self.workers.append(self._spawn(worker_id))
            self._wait_ready()
            self.started = True
            self._executor = ThreadPoolExecutor(
                max_workers=self.num_workers, thread_name_prefix="worker-http"
            )
        except Exception:
            self.shutdown()
            raise

    def _spawn(self, worker_id: int) -> WorkerHandle:
        port = _free_port()
        log_path = self.worker_dir / f"worker_{worker_id}.log"
        ready_path = self.worker_dir / f"worker_{worker_id}.ready"
        for path in (ready_path,):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        env = os.environ.copy()
        env.update(
            {
                "LAB_WORKER_TOKEN": self.token,
                "LAB_SESSION_ID": self.session_id,
                "LAB_WORKER_ID": str(worker_id),
                "LAB_WORKER_PORT": str(port),
                "LAB_READY_FILE": str(ready_path),
                "PYTHONUNBUFFERED": "1",
            }
        )
        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        log_file = open(log_path, "ab", buffering=0)
        try:
            process = subprocess.Popen(
                [sys.executable, "-m", "distributed_llm.worker_node", str(port)],
                cwd=str(BASE_DIR),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=log_file,
                creationflags=creationflags,
            )
        finally:
            log_file.close()
        self.metrics["workers_spawned"] += 1
        self._emit(
            f"worker-{worker_id}: process {process.pid} on 127.0.0.1:{port}",
            event="spawn",
            worker_id=worker_id,
            port=port,
            pid=process.pid,
        )
        return WorkerHandle(
            worker_id=worker_id,
            port=port,
            process=process,
            log_path=log_path,
            ready_path=ready_path,
            base_url=f"http://127.0.0.1:{port}",
        )

    def _wait_ready(self) -> None:
        assert self._client is not None
        deadline = time.monotonic() + self.limits.worker_start_timeout_s
        pending = list(self.workers)
        while pending:
            still: List[WorkerHandle] = []
            for handle in pending:
                if handle.process.poll() is not None:
                    detail = _tail(handle.log_path)
                    raise ClusterError(
                        f"worker {handle.worker_id} exited during startup "
                        f"(code {handle.process.returncode}). Log:\n{detail}"
                    )
                if not handle.ready_path.exists():
                    still.append(handle)
                    continue
                try:
                    response = self._client.get(f"{handle.base_url}/health", timeout=1.0)
                except httpx.HTTPError:
                    still.append(handle)
                    continue
                if response.status_code != 200:
                    still.append(handle)
                    continue
                status = self._client.get(
                    f"{handle.base_url}/v1/status",
                    headers=self._headers(),
                    timeout=2.0,
                )
                if status.status_code != 200:
                    still.append(handle)
                    continue
                body = status.json()
                if body.get("worker_id") != handle.worker_id:
                    raise ClusterError(f"worker {handle.worker_id} reported the wrong identity")
                self._emit(
                    f"worker-{handle.worker_id}: ready on port {handle.port}",
                    event="ready",
                    worker_id=handle.worker_id,
                    port=handle.port,
                )
            if not still:
                return
            if time.monotonic() > deadline:
                names = ", ".join(str(h.worker_id) for h in still)
                raise ClusterError(
                    f"workers {names} did not become ready within "
                    f"{self.limits.worker_start_timeout_s:.0f}s"
                )
            pending = still
            time.sleep(0.05)

    # -- HTTP primitives -------------------------------------------------
    def _request(
        self,
        worker_id: int,
        method: str,
        path: str,
        *,
        json_body: Optional[dict] = None,
        content: Optional[bytes] = None,
        params: Optional[dict] = None,
        label: Optional[str] = None,
    ) -> dict:
        if not self.started:
            raise ClusterError("cluster is not started")
        assert self._client is not None
        handle = self.workers[worker_id]
        url = handle.base_url + path
        attempts = 3
        last_error = ""
        for attempt in range(attempts):
            started = time.perf_counter()
            try:
                response = self._client.request(
                    method,
                    url,
                    json=json_body,
                    content=content,
                    params=params,
                    headers=self._headers(),
                )
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                sent = len(content) if content is not None else 0
                self.metrics["requests"] += 1
                self.metrics["bytes_sent"] += sent
                self.metrics["bytes_received"] += len(response.content)
                self.metrics["worker_wall_ms"] += elapsed_ms
                if response.status_code < 400:
                    self._emit(
                        f"{label or method} worker-{worker_id} {path} "
                        f"{response.status_code} in {elapsed_ms:.1f}ms",
                        event="call",
                        worker_id=worker_id,
                        path=path,
                        status=response.status_code,
                        ms=round(elapsed_ms, 3),
                    )
                    try:
                        return response.json()
                    except ValueError as exc:
                        raise WorkerCallError(
                            worker_id, path, "response was not valid JSON"
                        ) from exc
                if response.status_code < 500:
                    self.metrics["failures"] += 1
                    detail = response.text[:500]
                    raise WorkerCallError(
                        worker_id, path, f"HTTP {response.status_code}: {detail}"
                    )
                last_error = f"HTTP {response.status_code}: {response.text[:300]}"
            self.metrics["retries"] += 1
            if attempt + 1 < attempts:
                self._emit(
                    f"retry worker-{worker_id} {path}: {last_error}",
                    level="warn",
                    event="retry",
                    worker_id=worker_id,
                    path=path,
                )
                time.sleep(0.15 * (attempt + 1))
        self.metrics["failures"] += 1
        raise WorkerCallError(worker_id, path, last_error)

    def _parallel(self, jobs: Sequence[Tuple[int, Callable[[int], dict]]]) -> List[dict]:
        assert self._executor is not None
        futures = [self._executor.submit(job, worker_id) for worker_id, job in jobs]
        results = []
        for future in futures:
            results.append(future.result())
        return results

    # -- high level ------------------------------------------------------
    def upload_shards(
        self,
        shards: Sequence[bytes],
        layers: Sequence[int],
        kv_heads: Sequence[int],
        q_per_kv: int,
    ) -> List[dict]:
        if len(shards) != self.num_workers:
            raise ClusterError("one shard per worker is required")
        params_common = {
            "session_id": self.session_id,
            "layers": ",".join(str(x) for x in layers),
            "kv_heads": ",".join(str(x) for x in kv_heads),
            "q_per_kv": int(q_per_kv),
        }

        def job(worker_id: int) -> dict:
            params = dict(params_common)
            params["worker_id"] = worker_id
            return self._request(
                worker_id,
                "POST",
                "/v1/shard",
                content=shards[worker_id],
                params=params,
                label="upload",
            )

        return self._parallel([(worker_id, job) for worker_id in range(self.num_workers)])

    def bounds(self, layer: int, kv_head: int, q: np.ndarray) -> np.ndarray:
        payload = {
            "session_id": self.session_id,
            "layer": int(layer),
            "kv_head": int(kv_head),
            "q": np.asarray(q, dtype=np.float64).tolist(),
        }

        def job(worker_id: int) -> dict:
            return self._request(worker_id, "POST", "/v1/bounds", json_body=payload, label="bounds")

        replies = self._parallel([(worker_id, job) for worker_id in range(self.num_workers)])
        return np.asarray([reply["bounds"] for reply in replies], dtype=np.float64)

    def stats(
        self,
        layer: int,
        kv_head: int,
        q: np.ndarray,
        multiplicities: Optional[Dict[str, int]],
        subset: str,
        worker_ids: Sequence[int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        payload = {
            "session_id": self.session_id,
            "layer": int(layer),
            "kv_head": int(kv_head),
            "q": np.asarray(q, dtype=np.float64).tolist(),
            "multiplicities": multiplicities,
            "subset": subset,
        }

        def job(worker_id: int) -> dict:
            return self._request(worker_id, "POST", "/v1/stats", json_body=payload, label="stats")

        replies = self._parallel([(worker_id, job) for worker_id in worker_ids])
        lmax = np.asarray([reply["lmax"] for reply in replies], dtype=np.float64)
        sumexp = np.asarray([reply["sumexp"] for reply in replies], dtype=np.float64)
        local_out = np.asarray([reply["local_out"] for reply in replies], dtype=np.float64)
        meta = [
            {"worker_id": worker_id, "token_count": reply["token_count"], "elapsed_ms": reply["elapsed_ms"]}
            for worker_id, reply in zip(worker_ids, replies)
        ]
        return lmax, sumexp, local_out, meta

    def clear(self) -> None:
        if not self.started:
            return
        for handle in self.workers:
            try:
                assert self._client is not None
                self._client.post(
                    f"{handle.base_url}/v1/clear", headers=self._headers(), timeout=2.0
                )
            except httpx.HTTPError:
                pass

    def shutdown(self) -> None:
        self.started = False
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
        for handle in self.workers:
            if handle.process.poll() is None:
                try:
                    handle.process.terminate()
                except OSError:
                    pass
        deadline = time.monotonic() + 5.0
        for handle in self.workers:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                handle.process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                try:
                    handle.process.kill()
                    handle.process.wait(timeout=2.0)
                except (OSError, subprocess.TimeoutExpired):
                    pass
        if self._client is not None:
            self._client.close()
            self._client = None
        for handle in self.workers:
            try:
                handle.ready_path.unlink()
            except FileNotFoundError:
                pass
        self.workers.clear()

    def summary(self) -> dict:
        return {
            "workers": self.num_workers,
            "session_id": self.session_id,
            "spawned": self.metrics["workers_spawned"],
            "requests": self.metrics["requests"],
            "retries": self.metrics["retries"],
            "failures": self.metrics["failures"],
            "bytes_sent": self.metrics["bytes_sent"],
            "bytes_received": self.metrics["bytes_received"],
            "worker_wall_ms": round(self.metrics["worker_wall_ms"], 3),
        }
