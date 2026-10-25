"""Run orchestration for real-attention benchmarks.

A run executes in three isolated stages:

1. Capture runs in a subprocess so PyTorch and model memory never enter the web
   process or the attention workers.
2. The benchmark starts a fresh worker cluster with a per-run bearer token and
   replays the captured tensors through the certified pruning protocol.
3. Generation, when requested, starts a second fresh worker cluster and decodes
   with the exact distributed attention path. The decoder process streams one
   layer at a time and never owns a KV cache.

The manager enforces one active run, bounded inputs, disk headroom, per-stage
timeouts, and cancellation. Prompts are written to disk only for the duration of
the capture and generation subprocesses unless LAB_STORE_PROMPTS is enabled.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .benchmark import (
    BenchmarkCancelled,
    BenchmarkConfig,
    CaptureStore,
    run_benchmark,
)
from .cluster import WorkerCluster
from .config import (
    APP_VERSION,
    BASE_DIR,
    LIMITS,
    MODEL_ID,
    MODEL_REVISION,
    RUNS_DIR,
    STORE_PROMPTS,
    Limits,
)
from .decoder import GENERATION_METHOD, GENERATION_PROVENANCE
from .logging_utils import (
    JsonlLogger,
    available_ram_gb,
    new_run_id,
    process_rss_mb,
    utc_now_iso,
)

ATTENTION_PROVENANCE = (
    "Q/K/V captured from the pinned Qwen2.5-Coder-1.5B snapshot and replayed "
    "through isolated HTTP workers running the certified pruning protocol."
)
RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")

ARTIFACT_NAMES = (
    "summary.json",
    "manifest.json",
    "benchmark.json",
    "cluster.json",
    "capture.json",
    "capture.npz",
    "request.json",
    "generation.json",
    "events.jsonl",
)


class RunManagerError(RuntimeError):
    pass


class RunBusyError(RunManagerError):
    pass


class RunNotFoundError(RunManagerError):
    pass


class CaptureFailure(RuntimeError):
    pass


class GenerationFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class RunConfig:
    prompt: str
    workers: int = 4
    max_tokens: int = 512
    overlap_pct: float = 0.0
    strategy: str = "pseudorandom"
    seed: int = 42
    error_tol: float = 1e-3
    safe_gamma: float = LIMITS.safe_gamma
    generate: bool = True
    max_new_tokens: int = LIMITS.default_max_new_tokens
    layers: Optional[Tuple[int, ...]] = None
    kv_heads: Optional[Tuple[int, ...]] = None

    def validate(self, limits: Limits = LIMITS) -> None:
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("prompt must not be empty")
        if len(self.prompt) > limits.max_prompt_chars:
            raise ValueError(f"prompt exceeds {limits.max_prompt_chars} characters")
        if not limits.min_workers <= self.workers <= limits.max_workers:
            raise ValueError(
                f"workers must be between {limits.min_workers} and {limits.max_workers}"
            )
        if not 16 <= self.max_tokens <= limits.max_context_tokens:
            raise ValueError(
                f"max_tokens must be between 16 and {limits.max_context_tokens}"
            )
        if not 0.0 <= self.overlap_pct <= limits.max_overlap_pct:
            raise ValueError(f"overlap_pct must be between 0 and {limits.max_overlap_pct}")
        if self.strategy not in {"pseudorandom", "contiguous"}:
            raise ValueError("strategy must be 'pseudorandom' or 'contiguous'")
        if not 0 <= self.seed <= 2**31 - 1:
            raise ValueError("seed must be between 0 and 2^31-1")
        if not 1e-9 <= self.error_tol <= 1.0:
            raise ValueError("error_tol must be between 1e-9 and 1")
        if not 0.0 <= self.safe_gamma <= 2.0:
            raise ValueError("safe_gamma must be between 0 and 2")
        if not 1 <= self.max_new_tokens <= limits.max_generation_tokens:
            raise ValueError(
                f"max_new_tokens must be between 1 and {limits.max_generation_tokens}"
            )
        for name, values, upper in (
            ("layers", self.layers, 256),
            ("kv_heads", self.kv_heads, 64),
        ):
            if values is None:
                continue
            if not values or any(not isinstance(v, int) for v in values):
                raise ValueError(f"{name} must be a non-empty list of integers")
            if any(v < 0 or v >= upper for v in values):
                raise ValueError(f"{name} values must be between 0 and {upper - 1}")
        if self.layers is not None and len(set(self.layers)) != len(self.layers):
            raise ValueError("layers must not contain duplicates")
        if self.kv_heads is not None and len(set(self.kv_heads)) != len(self.kv_heads):
            raise ValueError("kv_heads must not contain duplicates")

    def prompt_sha256(self) -> str:
        return hashlib.sha256(self.prompt.encode("utf-8")).hexdigest()

    def public_dict(self) -> dict:
        return {
            "workers": self.workers,
            "max_tokens": self.max_tokens,
            "overlap_pct": self.overlap_pct,
            "strategy": self.strategy,
            "seed": self.seed,
            "error_tol": self.error_tol,
            "safe_gamma": self.safe_gamma,
            "generate": self.generate,
            "max_new_tokens": self.max_new_tokens,
            "layers": list(self.layers) if self.layers else None,
            "kv_heads": list(self.kv_heads) if self.kv_heads else None,
        }


class EventStream:
    """Thread safe, bounded event history for one run.

    The stream keeps sequence numbers so an SSE client can reconnect with
    ``after=<last seen>`` and receive every event that is still buffered.
    """

    def __init__(self, max_events: int = 20_000):
        self.max_events = max_events
        self._events: List[dict] = []
        self._lock = threading.Lock()
        self._seq = 0
        self._closed = False

    def publish(self, event: dict) -> dict:
        with self._lock:
            self._seq += 1
            record = {"seq": self._seq, "ts": utc_now_iso(), **event}
            self._events.append(record)
            if len(self._events) > self.max_events:
                del self._events[: len(self._events) - self.max_events]
            return record

    def since(self, after: int) -> List[dict]:
        with self._lock:
            if not self._events or after >= self._seq:
                return []
            first = self._events[0]["seq"]
            start = max(0, after + 1 - first)
            return list(self._events[start:])

    def close(self) -> None:
        with self._lock:
            self._closed = True

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    @property
    def last_seq(self) -> int:
        with self._lock:
            return self._seq


@dataclass
class RunRecord:
    run_id: str
    config: RunConfig
    run_dir: Path
    stream: EventStream
    logger: JsonlLogger
    started_utc: str
    started_monotonic: float
    cancel: threading.Event
    status: str = "running"
    finished_utc: Optional[str] = None
    error: Optional[str] = None
    summary: Optional[dict] = None
    thread: Optional[threading.Thread] = None

    def status_payload(self) -> dict:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "started_utc": self.started_utc,
            "finished_utc": self.finished_utc,
            "error": self.error,
            "config": self.config.public_dict(),
            "prompt_sha256": self.config.prompt_sha256(),
            "last_seq": self.stream.last_seq,
        }


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    os.replace(temporary, path)


def _hash_file(path: Path) -> Tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    return digest.hexdigest(), size


def _tail(path: Path, max_bytes: int = 4000) -> str:
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size > max_bytes:
                handle.seek(size - max_bytes)
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


class RunManager:
    """Single-active-run orchestrator shared by the web application."""

    def __init__(self, runs_dir: Path = RUNS_DIR, limits: Limits = LIMITS):
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.limits = limits
        self._lock = threading.Lock()
        self._runs: Dict[str, RunRecord] = {}
        self._active: Optional[RunRecord] = None
        self._started_at = time.time()
        self._shutting_down = threading.Event()
        self._counters = {"started": 0, "completed": 0, "failed": 0, "cancelled": 0}

    # -- public API ------------------------------------------------------
    def start(self, config: RunConfig) -> RunRecord:
        config.validate(self.limits)
        if self._shutting_down.is_set():
            raise RunManagerError("server is shutting down")
        with self._lock:
            if (
                self._active is not None
                and self._active.thread is not None
                and self._active.thread.is_alive()
            ):
                raise RunBusyError(
                    f"run {self._active.run_id} is active; one run at a time"
                )
            run_id = new_run_id("run")
            run_dir = self.runs_dir / run_id
            while run_dir.exists():
                run_id = new_run_id("run")
                run_dir = self.runs_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=False)
            if os.name == "posix":
                try:
                    os.chmod(run_dir, 0o700)
                except OSError:
                    pass
            stream = EventStream()
            logger = JsonlLogger(run_dir / "events.jsonl", run_id=run_id)
            record = RunRecord(
                run_id=run_id,
                config=config,
                run_dir=run_dir,
                stream=stream,
                logger=logger,
                started_utc=utc_now_iso(),
                started_monotonic=time.monotonic(),
                cancel=threading.Event(),
            )
            self._active = record
            self._runs[run_id] = record
            self._counters["started"] += 1
            self._trim_runs_locked()
        self._publish(
            record,
            {
                "type": "run",
                "run_id": run_id,
                "config": config.public_dict(),
                "prompt_sha256": config.prompt_sha256(),
                "prompt_chars": len(config.prompt),
            },
        )
        thread = threading.Thread(
            target=self._execute, args=(record,), name=f"run-{run_id}", daemon=True
        )
        record.thread = thread
        try:
            thread.start()
        except Exception as exc:  # noqa: BLE001 - thread start is the last failure point
            record.status = "failed"
            record.error = str(exc)
            record.stream.publish(
                {"type": "error", "error_type": type(exc).__name__, "message": str(exc)}
            )
            record.stream.close()
            raise RunManagerError(f"could not start the run thread: {exc}") from exc
        return record

    def get(self, run_id: str) -> RunRecord:
        if not RUN_ID_RE.match(run_id or ""):
            raise RunNotFoundError("unknown run id")
        with self._lock:
            record = self._runs.get(run_id)
        if record is None:
            raise RunNotFoundError("unknown run id")
        return record

    def current(self) -> Optional[RunRecord]:
        with self._lock:
            return self._active

    def cancel(self, run_id: str) -> bool:
        record = self.get(run_id)
        if record.status != "running":
            return False
        record.cancel.set()
        record.logger.log("cancel_requested", level="warn")
        self._publish(record, {"type": "info", "msg": "cancellation requested"})
        return True

    def shutdown(self, timeout: float = 15.0) -> None:
        self._shutting_down.set()
        with self._lock:
            record = self._active
        if record is None:
            return
        record.cancel.set()
        if record.thread is not None:
            record.thread.join(timeout=timeout)

    def artifact_path(self, run_id: str, name: str) -> Path:
        record = self.get(run_id)
        if name not in ARTIFACT_NAMES:
            raise RunNotFoundError("unknown artifact")
        path = record.run_dir / name
        if not path.is_file():
            raise RunNotFoundError("artifact is not available")
        resolved = path.resolve()
        if record.run_dir.resolve() not in resolved.parents:
            raise RunNotFoundError("invalid artifact path")
        return path

    def list_artifacts(self, run_id: str) -> List[dict]:
        record = self.get(run_id)
        items = []
        for name in ARTIFACT_NAMES:
            path = record.run_dir / name
            if path.is_file():
                items.append({"name": name, "bytes": path.stat().st_size})
        return items

    def metrics(self) -> dict:
        with self._lock:
            active = self._active
            counters = dict(self._counters)
        active_payload = None
        if active is not None:
            active_payload = {
                "run_id": active.run_id,
                "status": active.status,
                "started_utc": active.started_utc,
                "elapsed_s": round(time.monotonic() - active.started_monotonic, 3),
                "workers": active.config.workers,
                "context_tokens": active.config.max_tokens,
            }
        return {
            "app_version": APP_VERSION,
            "uptime_s": round(time.time() - self._started_at, 3),
            "runs": counters,
            "active_run": active_payload,
            "process_rss_mb": process_rss_mb(),
            "available_ram_gb": available_ram_gb(),
            "free_disk_gb": self._free_disk_gb(),
            "limits": {
                "min_workers": self.limits.min_workers,
                "max_workers": self.limits.max_workers,
                "max_context_tokens": self.limits.max_context_tokens,
                "max_overlap_pct": self.limits.max_overlap_pct,
                "max_prompt_chars": self.limits.max_prompt_chars,
                "max_generation_tokens": self.limits.max_generation_tokens,
                "default_max_new_tokens": self.limits.default_max_new_tokens,
                "max_generation_positions": self.limits.max_generation_positions,
                "max_concurrent_runs": self.limits.max_concurrent_runs,
            },
        }

    def readiness(self) -> dict:
        with self._lock:
            active = self._active
            busy = bool(active and active.thread and active.thread.is_alive())
        free = self._free_disk_gb()
        disk_ok = free < 0.0 or free >= self.limits.min_free_disk_gb
        shutting_down = self._shutting_down.is_set()
        return {
            "ready": disk_ok and not shutting_down,
            "shutting_down": shutting_down,
            "busy": busy,
            "active_run_id": active.run_id if busy and active else None,
            "free_disk_gb": free,
            "minimum_free_disk_gb": self.limits.min_free_disk_gb,
            "available_ram_gb": available_ram_gb(),
            "app_version": APP_VERSION,
        }

    # -- internals -------------------------------------------------------
    def _trim_runs_locked(self) -> None:
        if len(self._runs) <= 32:
            return
        active_id = self._active.run_id if self._active else None
        for run_id in list(self._runs):
            if len(self._runs) <= 32:
                break
            if run_id == active_id:
                continue
            record = self._runs[run_id]
            if record.thread is not None and record.thread.is_alive():
                continue
            del self._runs[run_id]

    def _publish(self, record: RunRecord, event: dict) -> None:
        payload = {"run_id": record.run_id, **event}
        record.stream.publish(payload)

    def _free_disk_gb(self) -> float:
        try:
            return round(shutil.disk_usage(str(self.runs_dir)).free / 1e9, 3)
        except OSError:
            return -1.0

    def _require_disk(self, minimum_gb: float) -> None:
        free = self._free_disk_gb()
        if 0.0 <= free < minimum_gb:
            raise RunManagerError(
                f"only {free:.2f} GB free on the artifacts volume, "
                f"at least {minimum_gb:.2f} GB is required"
            )

    def _execute(self, record: RunRecord) -> None:
        started_wall = time.monotonic()
        store: Optional[CaptureStore] = None
        status = "completed"
        try:
            self._require_disk(self.limits.min_free_disk_gb)
            self._write_request(record)
            self._publish(
                record,
                {
                    "type": "info",
                    "msg": (
                        f"run accepted: K={record.config.workers}, "
                        f"N<={record.config.max_tokens}, "
                        f"overlap={record.config.overlap_pct}%, "
                        f"eps={record.config.error_tol:g}"
                    ),
                },
            )
            capture = self._run_capture(record)
            store = self._open_capture(record, capture)
            payload = self._run_benchmark(record, store)
            self._publish(
                record,
                {
                    "type": "stage",
                    "stage": "benchmark_done",
                    "aggregate": payload["aggregate"],
                    "timings": payload["timings"],
                },
            )
            generation = self._run_generation(record, payload, capture)
            if record.cancel.is_set():
                raise BenchmarkCancelled("run cancelled before completion")
            summary = self._write_summary(
                record, payload, generation, started_wall, status="completed"
            )
            record.summary = summary
            self._publish(
                record,
                {
                    "type": "done",
                    "status": "completed",
                    "aggregate": payload["aggregate"],
                    "timings": summary["timings"],
                    "artifacts": self.list_artifacts(record.run_id),
                },
            )
            record.logger.log("run_complete", status="completed", duration_s=summary["duration_s"])
            with self._lock:
                self._counters["completed"] += 1
        except BenchmarkCancelled:
            status = "cancelled"
            record.error = "run cancelled by operator"
            self._publish(
                record, {"type": "info", "msg": "run cancelled before completion"}
            )
            record.logger.log("run_cancelled", level="warn")
            self._write_failure_summary(record, started_wall, "cancelled", record.error)
            self._publish(record, {"type": "done", "status": "cancelled"})
            with self._lock:
                self._counters["cancelled"] += 1
        except Exception as exc:  # noqa: BLE001 - every failure is reported
            status = "failed"
            record.error = str(exc)
            record.logger.log(
                "run_failed",
                level="error",
                error_type=type(exc).__name__,
                error=str(exc),
                traceback=traceback.format_exc(limit=12),
            )
            self._publish(
                record,
                {
                    "type": "error",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
            )
            self._write_failure_summary(
                record, started_wall, "failed", str(exc), type(exc).__name__
            )
            self._publish(record, {"type": "done", "status": "failed"})
            with self._lock:
                self._counters["failed"] += 1
        finally:
            record.status = status
            record.finished_utc = utc_now_iso()
            record.stream.close()
            if store is not None:
                store.close()
            if not STORE_PROMPTS:
                try:
                    (record.run_dir / "prompt.txt").unlink()
                except FileNotFoundError:
                    pass

    # -- request and capture --------------------------------------------
    def _write_request(self, record: RunRecord) -> None:
        config = record.config
        request = {
            "run_id": record.run_id,
            "app_version": APP_VERSION,
            "created_utc": record.started_utc,
            "config": config.public_dict(),
            "prompt": {
                "sha256": config.prompt_sha256(),
                "chars": len(config.prompt),
                "stored_on_disk": STORE_PROMPTS,
            },
            "model": {"model_id": MODEL_ID, "revision": MODEL_REVISION},
        }
        _write_json_atomic(record.run_dir / "request.json", request)
        if STORE_PROMPTS:
            prompt_path = record.run_dir / "prompt.txt"
            prompt_path.write_text(config.prompt, encoding="utf-8")
            if os.name == "posix":
                try:
                    os.chmod(prompt_path, 0o600)
                except OSError:
                    pass
        record.logger.log("run_start", config=config.public_dict())

    def _capture_argv(self, record: RunRecord) -> List[str]:
        run_dir = record.run_dir
        argv = [
            sys.executable,
            "-m",
            "distributed_llm.capture_runner",
            "--prompt-file",
            str(run_dir / "prompt.txt"),
            "--max-tokens",
            str(record.config.max_tokens),
            "--out",
            str(run_dir / "capture.npz"),
            "--meta",
            str(run_dir / "capture.json"),
            "--model-id",
            MODEL_ID,
            "--revision",
            MODEL_REVISION,
        ]
        if record.config.layers:
            argv.extend(["--layers", ",".join(str(x) for x in record.config.layers)])
        if record.config.kv_heads:
            argv.extend(["--kv-heads", ",".join(str(x) for x in record.config.kv_heads)])
        return argv

    def _run_capture(self, record: RunRecord) -> dict:
        prompt_path = record.run_dir / "prompt.txt"
        if not prompt_path.is_file():
            prompt_path.write_text(record.config.prompt, encoding="utf-8")
        self._publish(
            record,
            {
                "type": "stage",
                "stage": "capture",
                "phase": "starting",
                "model_id": MODEL_ID,
                "revision": MODEL_REVISION,
                "max_tokens": record.config.max_tokens,
            },
        )
        record.logger.log(
            "capture_start", model_id=MODEL_ID, revision=MODEL_REVISION,
            max_tokens=record.config.max_tokens,
        )
        final = self._stream_subprocess(
            record,
            self._capture_argv(record),
            record.run_dir / "capture.stderr.log",
            self.limits.capture_timeout_s,
            stage_label="capture",
        )
        if final.get("stage") != "done":
            raise CaptureFailure("capture process did not report completion")
        self._publish(
            record,
            {
                "type": "stage",
                "stage": "capture",
                "phase": "done",
                "token_count": final.get("token_count"),
                "layers": final.get("layers"),
                "kv_heads": final.get("kv_heads"),
                "timing_s": final.get("timing_s"),
            },
        )
        record.logger.log(
            "capture_done",
            token_count=final.get("token_count"),
            layers=final.get("layers"),
            kv_heads=final.get("kv_heads"),
            timing_s=final.get("timing_s"),
        )
        if not STORE_PROMPTS:
            try:
                prompt_path.unlink()
            except FileNotFoundError:
                pass
        return final

    def _stream_subprocess(
        self,
        record: RunRecord,
        argv: Sequence[str],
        stderr_path: Path,
        timeout_s: float,
        stage_label: str,
        extra_env: Optional[Dict[str, str]] = None,
    ) -> dict:
        env = os.environ.copy()
        env.update(
            {
                "PYTHONUNBUFFERED": "1",
                "PYTHONIOENCODING": "utf-8",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "USE_TF": "0",
                "USE_FLAX": "0",
                "TRANSFORMERS_NO_TF": "1",
                "TOKENIZERS_PARALLELISM": "false",
            }
        )
        if extra_env:
            env.update(extra_env)
        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        stderr_handle = open(stderr_path, "wb")
        try:
            process = subprocess.Popen(
                list(argv),
                cwd=str(BASE_DIR),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=stderr_handle,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creationflags,
            )
        except OSError as exc:
            stderr_handle.close()
            raise CaptureFailure(f"could not start the {stage_label} process: {exc}") from exc

        lines: "queue.Queue[Optional[str]]" = queue.Queue()

        def reader() -> None:
            try:
                assert process.stdout is not None
                for line in process.stdout:
                    lines.put(line)
            finally:
                lines.put(None)

        threading.Thread(target=reader, name=f"{stage_label}-reader", daemon=True).start()
        final: Optional[dict] = None
        timed_out = False
        deadline = time.monotonic() + timeout_s
        try:
            while True:
                if record.cancel.is_set():
                    self._terminate(process)
                    raise BenchmarkCancelled("run cancelled during capture")
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    timed_out = True
                    self._terminate(process)
                    break
                try:
                    line = lines.get(timeout=min(0.5, max(0.01, remaining)))
                except queue.Empty:
                    continue
                if line is None:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    record.logger.log(
                        "capture_output", level="warn", line=line[:500]
                    )
                    continue
                if payload.get("stage") == stage_label:
                    self._publish(record, {"type": "stage", **payload})
                elif payload.get("stage") in {"done", "error"}:
                    final = payload
                    if payload.get("stage") == "error":
                        break
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._terminate(process)
        finally:
            stderr_handle.close()
        if timed_out:
            raise CaptureFailure(
                f"{stage_label} exceeded the {timeout_s:.0f}s timeout"
            )
        if final is not None and final.get("stage") == "error":
            raise CaptureFailure(str(final.get("message", "capture failed")))
        if process.returncode != 0:
            detail = _tail(stderr_path)
            raise CaptureFailure(
                f"{stage_label} exited with code {process.returncode}"
                + (f": {detail}" if detail else "")
            )
        if final is None:
            raise CaptureFailure(f"{stage_label} produced no completion record")
        return final

    @staticmethod
    def _terminate(process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return
        try:
            process.terminate()
            process.wait(timeout=5.0)
        except (OSError, subprocess.TimeoutExpired):
            try:
                process.kill()
                process.wait(timeout=5.0)
            except (OSError, subprocess.TimeoutExpired):
                pass

    def _open_capture(self, record: RunRecord, final: dict) -> CaptureStore:
        npz_path = record.run_dir / "capture.npz"
        meta_path = record.run_dir / "capture.json"
        if not npz_path.is_file() or not meta_path.is_file():
            raise CaptureFailure("capture artifacts are missing")
        size_mb = npz_path.stat().st_size / 1e6
        if size_mb > self.limits.max_capture_artifact_mb:
            raise CaptureFailure(
                f"capture artifact is {size_mb:.1f} MB, above the "
                f"{self.limits.max_capture_artifact_mb:.0f} MB limit"
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("truncated"):
            raise CaptureFailure(
                "capture was truncated; the benchmark requires the real final prompt token"
            )
        if meta.get("prompt_sha256") != record.config.prompt_sha256():
            raise CaptureFailure("capture prompt hash does not match the request")
        token_count = int(meta.get("token_count", 0))
        if token_count < record.config.workers:
            raise CaptureFailure(
                f"prompt has {token_count} tokens, fewer than the "
                f"{record.config.workers} workers"
            )
        store = CaptureStore(npz_path, meta_path)
        if record.config.layers:
            missing = set(record.config.layers) - set(store.layers)
            if missing:
                raise CaptureFailure(f"capture is missing layers {sorted(missing)}")
            store.layers = sorted(set(record.config.layers))
        if record.config.kv_heads:
            missing = set(record.config.kv_heads) - set(store.kv_heads)
            if missing:
                raise CaptureFailure(f"capture is missing KV heads {sorted(missing)}")
            store.kv_heads = sorted(set(record.config.kv_heads))
        self._publish(
            record,
            {
                "type": "info",
                "msg": (
                    f"capture verified: {token_count} tokens, "
                    f"{len(store.layers)} layers, {len(store.kv_heads)} KV heads, "
                    f"{size_mb:.1f} MB"
                ),
            },
        )
        self._require_disk(self.limits.min_free_disk_gb)
        return store

    # -- benchmark -------------------------------------------------------
    def _run_benchmark(self, record: RunRecord, store: CaptureStore) -> dict:
        config = BenchmarkConfig(
            workers=record.config.workers,
            overlap_pct=record.config.overlap_pct,
            strategy=record.config.strategy,
            seed=record.config.seed,
            error_tol=record.config.error_tol,
            safe_gamma=record.config.safe_gamma,
        )
        cluster_dir = record.run_dir / "cluster"

        def on_event(event: dict) -> None:
            payload = {"type": event.get("type", "info"), **event}
            if event.get("type") == "comm":
                payload.setdefault("level", "info")
            self._publish(record, payload)

        cluster = WorkerCluster(
            num_workers=config.workers,
            run_dir=cluster_dir,
            event_cb=on_event,
            limits=self.limits,
        )
        self._publish(
            record,
            {
                "type": "stage",
                "stage": "cluster",
                "phase": "spawning",
                "workers": config.workers,
            },
        )
        try:
            cluster.start()
            self._publish(
                record,
                {
                    "type": "stage",
                    "stage": "cluster",
                    "phase": "ready",
                    "workers": config.workers,
                    "session": cluster.session_id[:8],
                    "free_ram_gb": cluster.preflight(),
                },
            )
            payload = run_benchmark(
                store,
                cluster,
                config,
                limits=self.limits,
                logger=record.logger,
                event_cb=on_event,
                run_id=record.run_id,
                cancel_event=record.cancel,
            )
        finally:
            cluster.shutdown()
        payload["app_version"] = APP_VERSION
        payload["created_utc"] = utc_now_iso()
        payload["runner"] = {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "implementation": "distributed_llm.manager",
        }
        payload["config"]["prompt_sha256"] = record.config.prompt_sha256()
        _write_json_atomic(record.run_dir / "benchmark.json", payload)
        _write_json_atomic(record.run_dir / "cluster.json", payload.get("cluster", {}))
        return payload

    # -- generation ------------------------------------------------------
    def _generation_timeout_s(self, max_new_tokens: int) -> float:
        # Flat timeouts either kill long decodes or leave short ones hanging for
        # hours, so the watchdog budget grows with the requested token count.
        return self.limits.generation_timeout_s + max_new_tokens * (
            self.limits.generation_step_timeout_s
        )

    def _generation_argv(self, record: RunRecord, expected_prompt_tokens: int) -> List[str]:
        return [
            sys.executable,
            "-m",
            "distributed_llm.generation_runner",
            "--prompt-file",
            str(record.run_dir / "prompt.txt"),
            "--out",
            str(record.run_dir / "generation.json"),
            "--max-new-tokens",
            str(record.config.max_new_tokens),
            "--max-generation-positions",
            str(self.limits.max_generation_positions),
            "--expected-prompt-tokens",
            str(expected_prompt_tokens),
            "--model-id",
            MODEL_ID,
            "--revision",
            MODEL_REVISION,
        ]

    def _load_generation_artifact(
        self,
        run_dir: Path,
        expected_prompt_sha256: str,
        expected_prompt_tokens: int,
        max_new_tokens: int,
    ) -> dict:
        path = run_dir / "generation.json"
        if not path.is_file():
            raise GenerationFailure("generation.json is missing")
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise GenerationFailure("generation.json is not valid JSON") from exc
        if artifact.get("status") != "ok":
            raise GenerationFailure(
                f"generation reported {artifact.get('error_type', 'error')}: "
                f"{artifact.get('error', 'unknown error')}"
            )
        if artifact.get("method") != GENERATION_METHOD:
            raise GenerationFailure(
                "generation artifact does not come from the distributed decoder"
            )
        prompt = artifact.get("prompt") or {}
        if prompt.get("sha256") != expected_prompt_sha256:
            raise GenerationFailure("generation prompt hash does not match the request")
        if int(prompt.get("token_count") or 0) != expected_prompt_tokens:
            raise GenerationFailure(
                "generation tokenized the prompt differently from the capture"
            )
        output = artifact.get("output") or {}
        generated = int(output.get("token_count") or 0)
        if not 1 <= generated <= max_new_tokens:
            raise GenerationFailure("generation produced an invalid token count")
        return artifact

    def _run_generation(
        self, record: RunRecord, payload: dict, capture: dict
    ) -> Optional[dict]:
        if not record.config.generate:
            return None
        if record.cancel.is_set():
            raise BenchmarkCancelled("run cancelled before generation")

        geometry = payload.get("geometry") or {}
        token_count = int(capture.get("token_count") or payload.get("token_count") or 0)
        max_new_tokens = int(record.config.max_new_tokens)
        max_positions = token_count + max_new_tokens
        if token_count < 1:
            raise GenerationFailure("capture reported no prompt tokens")
        if max_positions > self.limits.max_generation_positions:
            raise GenerationFailure(
                f"prompt has {token_count} tokens; decoding {max_new_tokens} more "
                f"exceeds the {self.limits.max_generation_positions}-token "
                "generation window"
            )

        self._publish(
            record,
            {
                "type": "stage",
                "stage": "generation",
                "phase": "starting",
                "method": GENERATION_METHOD,
                "prompt_tokens": token_count,
                "max_new_tokens": max_new_tokens,
                "provenance": GENERATION_PROVENANCE,
            },
        )
        record.logger.log(
            "generation_start",
            method=GENERATION_METHOD,
            prompt_tokens=token_count,
            max_new_tokens=max_new_tokens,
            workers=record.config.workers,
        )

        prompt_path = record.run_dir / "prompt.txt"
        prompt_path.write_text(record.config.prompt, encoding="utf-8")
        if os.name == "posix":
            try:
                os.chmod(prompt_path, 0o600)
            except OSError:
                pass

        def on_event(event: dict) -> None:
            event_payload = {"type": event.get("type", "info"), **event}
            if event.get("type") == "comm":
                event_payload.setdefault("level", "info")
            self._publish(record, event_payload)

        cluster = WorkerCluster(
            num_workers=record.config.workers,
            run_dir=record.run_dir / "generation_cluster",
            event_cb=on_event,
            limits=self.limits,
        )
        self._publish(
            record,
            {
                "type": "stage",
                "stage": "generation_cluster",
                "phase": "spawning",
                "workers": record.config.workers,
            },
        )
        try:
            cluster.start()
            admission = cluster.preflight_generation(
                max_positions=max_positions,
                num_layers=int(geometry.get("num_layers") or 0),
                num_kv_heads=int(geometry.get("num_key_value_heads") or 0),
                head_dim=int(geometry.get("head_dim") or 0),
                process_headroom_gb=self.limits.generation_decoder_headroom_gb,
            )
            self._publish(
                record,
                {
                    "type": "stage",
                    "stage": "generation_cluster",
                    "phase": "ready",
                    "workers": record.config.workers,
                    "session": cluster.session_id[:8],
                    "free_ram_gb": admission["free_ram_gb"],
                    "needed_ram_gb": admission["needed_ram_gb"],
                },
            )
            final = self._stream_subprocess(
                record,
                self._generation_argv(record, token_count),
                record.run_dir / "generation.stderr.log",
                self._generation_timeout_s(max_new_tokens),
                stage_label="generation",
                extra_env={"LAB_GEN_CONNECTION": json.dumps(cluster.connection_info())},
            )
            if final.get("stage") != "done":
                raise GenerationFailure(
                    "generation process did not report completion"
                )
            artifact = self._load_generation_artifact(
                record.run_dir,
                record.config.prompt_sha256(),
                token_count,
                max_new_tokens,
            )
            output = artifact["output"]
            timings = artifact.get("timings") or {}
            self._publish(
                record,
                {
                    "type": "generation",
                    "status": "ok",
                    "method": GENERATION_METHOD,
                    "text": str(output.get("text", ""))[:8000],
                    "token_count": output.get("token_count"),
                    "stop_reason": output.get("stop_reason"),
                    "timing_s": timings.get("total_s"),
                    "prefill_s": timings.get("prefill_s"),
                    "decode_s": timings.get("decode_s"),
                    "workers": len(artifact.get("workers") or []),
                    "provenance": GENERATION_PROVENANCE,
                },
            )
            record.logger.log(
                "generation_ok",
                method=GENERATION_METHOD,
                tokens=output.get("token_count"),
                stop_reason=output.get("stop_reason"),
                timing_s=timings.get("total_s"),
                peak_rss_mb=(artifact.get("memory") or {}).get("peak_rss_mb"),
            )
            return artifact
        except BenchmarkCancelled:
            raise
        except Exception as exc:  # noqa: BLE001 - any failure invalidates the run
            message = str(exc)
            try:
                failure = json.loads(
                    (record.run_dir / "generation.json").read_text(encoding="utf-8")
                )
            except (OSError, ValueError):
                failure = {}
            if failure.get("status") == "error" and failure.get("error"):
                message = f"{message}: {failure['error']}"
            self._publish(
                record,
                {
                    "type": "generation",
                    "status": "error",
                    "message": message,
                    "method": GENERATION_METHOD,
                    "provenance": GENERATION_PROVENANCE,
                },
            )
            record.logger.log(
                "generation_error",
                level="error",
                error_type=type(exc).__name__,
                error=message,
            )
            raise GenerationFailure(message) from exc
        finally:
            cluster.shutdown()
            if not STORE_PROMPTS:
                try:
                    prompt_path.unlink()
                except FileNotFoundError:
                    pass

    # -- summaries and artifacts ----------------------------------------
    def _dashboard(self, payload: dict) -> dict:
        units = []
        for layer in payload.get("layers", []):
            for row in layer.get("kv_heads", []):
                bounds = np.asarray(row["block_bounds"], dtype=np.float64)
                units.append(
                    {
                        "layer": row["layer"],
                        "kv_head": row["kv_head"],
                        "block_bounds": [
                            float(bounds[w].max(initial=float("-inf")))
                            for w in range(bounds.shape[0])
                        ],
                        "retained_workers": row["retained_workers"],
                        "retained_fraction": row["retained_fraction"],
                        "retention_curve": row["retention_curve"],
                        "max_certified_error": max(row["certified_error"]),
                        "max_relative_error": max(row["relative_error"]),
                        "v_max": row["v_max"],
                    }
                )
        worst = payload.get("worst_case") or {}
        worst_trimmed = None
        if worst:
            worst_trimmed = {
                "layer": worst.get("layer"),
                "kv_head": worst.get("kv_head"),
                "q_head": worst.get("q_head"),
                "relative_error": worst.get("relative_error"),
                "certified_error": worst.get("certified_error"),
                "full_output": list(worst.get("full_output", []))[:32],
                "distributed_output": list(worst.get("distributed_output", []))[:32],
            }
        partition = payload.get("partition", {})
        return {
            "units": units,
            "worst_case": worst_trimmed,
            "workers": partition.get("workers"),
            "positions_per_worker": partition.get("positions_per_worker"),
            "primary_per_worker": partition.get("primary_per_worker"),
            "max_copies": partition.get("max_copies"),
        }

    def _write_summary(
        self,
        record: RunRecord,
        payload: dict,
        generation: Optional[dict],
        started_wall: float,
        status: str,
    ) -> dict:
        timings = dict(payload.get("timings", {}))
        timings["total_s"] = round(time.monotonic() - started_wall, 3)
        summary = {
            "run_id": record.run_id,
            "status": status,
            "app_version": APP_VERSION,
            "started_utc": record.started_utc,
            "finished_utc": utc_now_iso(),
            "duration_s": timings["total_s"],
            "request": record.config.public_dict(),
            "prompt": {
                "sha256": record.config.prompt_sha256(),
                "chars": len(record.config.prompt),
                "stored_on_disk": STORE_PROMPTS,
            },
            "model": {
                "model_id": payload.get("model", {}).get("model_id"),
                "revision": payload.get("model", {}).get("revision"),
                "model_path": payload.get("model", {}).get("model_path"),
            },
            "geometry": payload.get("geometry"),
            "token_count": payload.get("token_count"),
            "capture": {
                "token_count": payload.get("token_count"),
                "original_token_count": payload.get("original_token_count"),
                "truncated": payload.get("truncated"),
                "tokenizer_class": payload.get("tokenizer_class"),
                "layers": len(payload.get("layers", [])),
                "timing_s": payload.get("timings", {}).get("capture_s"),
            },
            "partition": payload.get("partition"),
            "aggregate": payload.get("aggregate"),
            "timings": timings,
            "cluster": payload.get("cluster"),
            "provenance": {
                "attention": ATTENTION_PROVENANCE,
                "generation": GENERATION_PROVENANCE if generation else "disabled",
            },
            "dashboard": self._dashboard(payload),
            "artifacts": [
                name
                for name in ARTIFACT_NAMES
                if name in {"manifest.json"} or (record.run_dir / name).is_file()
            ],
        }
        if generation:
            output = generation.get("output") or {}
            attention = generation.get("attention") or {}
            summary["generation"] = {
                "status": generation.get("status"),
                "computed_by": generation.get("computed_by"),
                "method": generation.get("method"),
                "model": (generation.get("model") or {}).get("model_id"),
                "prompt_token_count": (generation.get("prompt") or {}).get("token_count"),
                "generated_token_count": output.get("token_count"),
                "stop_reason": output.get("stop_reason"),
                "text": output.get("text"),
                "timings": generation.get("timings"),
                "decoder": generation.get("decoder"),
                "attention": {
                    key: attention.get(key)
                    for key in (
                        "mode",
                        "append_calls",
                        "attend_calls",
                        "key_vectors_uploaded",
                    )
                },
                "worker_status": generation.get("workers"),
                "memory": generation.get("memory"),
                "determinism": generation.get("determinism"),
            }
        else:
            summary["generation"] = {"status": "disabled"}
        summary_path = record.run_dir / "summary.json"
        _write_json_atomic(summary_path, summary)
        self._write_manifest(record)
        return summary

    def _write_failure_summary(
        self,
        record: RunRecord,
        started_wall: float,
        status: str,
        message: str,
        error_type: str = "Cancelled",
    ) -> None:
        summary = {
            "run_id": record.run_id,
            "status": status,
            "app_version": APP_VERSION,
            "started_utc": record.started_utc,
            "finished_utc": utc_now_iso(),
            "duration_s": round(time.monotonic() - started_wall, 3),
            "request": record.config.public_dict(),
            "prompt": {
                "sha256": record.config.prompt_sha256(),
                "chars": len(record.config.prompt),
                "stored_on_disk": STORE_PROMPTS,
            },
            "error": {"type": error_type, "message": message},
            "artifacts": [
                name
                for name in ARTIFACT_NAMES
                if name in {"manifest.json"} or (record.run_dir / name).is_file()
            ],
        }
        if record.config.generate:
            summary["generation"] = {"status": "not_run", "message": message}
            try:
                artifact = json.loads(
                    (record.run_dir / "generation.json").read_text(encoding="utf-8")
                )
            except (OSError, ValueError):
                artifact = {}
            if artifact.get("status") == "error":
                summary["generation"] = {
                    "status": "error",
                    "computed_by": artifact.get("computed_by"),
                    "method": artifact.get("method"),
                    "message": artifact.get("error") or message,
                }
        else:
            summary["generation"] = {"status": "disabled"}
        _write_json_atomic(record.run_dir / "summary.json", summary)
        try:
            self._write_manifest(record)
        except OSError:
            pass
        record.summary = summary

    def _write_manifest(self, record: RunRecord) -> dict:
        artifacts = []
        for path in sorted(record.run_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.name == "manifest.json" or path.suffix == ".tmp":
                continue
            digest, size = _hash_file(path)
            artifacts.append(
                {
                    "name": path.relative_to(record.run_dir).as_posix(),
                    "sha256": digest,
                    "bytes": size,
                }
            )
        manifest = {
            "run_id": record.run_id,
            "app_version": APP_VERSION,
            "created_utc": utc_now_iso(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "model": {"model_id": MODEL_ID, "revision": MODEL_REVISION},
            "artifacts": artifacts,
        }
        _write_json_atomic(record.run_dir / "manifest.json", manifest)
        return manifest
