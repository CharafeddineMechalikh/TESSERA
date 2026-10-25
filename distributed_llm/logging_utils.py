"""Minimal structured JSON logging with no third-party dependency."""
from __future__ import annotations

import json
import os
import secrets
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def new_run_id(prefix: str = "run") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{prefix}-{secrets.token_hex(4)}"


class JsonlLogger:
    def __init__(
        self,
        path: Optional[Path] = None,
        run_id: Optional[str] = None,
        echo: bool = False,
    ) -> None:
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.echo = echo
        self._lock = threading.Lock()

    def log(self, event: str, level: str = "info", **fields: Any) -> dict:
        record = {
            "ts": utc_now_iso(),
            "level": level,
            "event": event,
            "run_id": self.run_id,
            **fields,
        }
        line = json.dumps(record, ensure_ascii=False, default=str, sort_keys=True)
        with self._lock:
            if self.path:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
            if self.echo:
                print(line, file=sys.stderr, flush=True)
        return record


def tail(path: Path, max_bytes: int = 8192) -> str:
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size > max_bytes:
                handle.seek(size - max_bytes)
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def process_rss_mb() -> float:
    try:
        import psutil

        return round(psutil.Process(os.getpid()).memory_info().rss / 1e6, 1)
    except Exception:
        return -1.0


def available_ram_gb() -> float:
    try:
        import psutil

        return round(psutil.virtual_memory().available / 1e9, 3)
    except Exception:
        return -1.0


def monotonic_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 3)
