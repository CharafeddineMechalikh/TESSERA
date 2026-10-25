"""CLI entry point for a real-attention benchmark run.

Loads a capture produced by capture_runner, starts an isolated worker cluster,
runs the exact distributed protocol with certified pruning, writes artifacts,
and emits one JSON event per line on stdout for a supervising process.
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import os
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .benchmark import BenchmarkConfig, CaptureStore, run_benchmark
from .cluster import WorkerCluster
from .config import APP_VERSION, LIMITS
from .logging_utils import JsonlLogger, new_run_id


def _emit(event: dict) -> None:
    record = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), **event}
    print(json.dumps(record, ensure_ascii=False, default=str), flush=True)


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    faulthandler.enable()
    parser = argparse.ArgumentParser(description="Run the distributed attention benchmark")
    parser.add_argument("--capture", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overlap", type=float, default=0.0)
    parser.add_argument("--strategy", default="pseudorandom")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=LIMITS.safe_gamma)
    parser.add_argument("--layers", default="")
    parser.add_argument("--kv-heads", default="")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = new_run_id("bench")
    logger = JsonlLogger(out_dir / "events.jsonl", run_id=run_id)
    store: CaptureStore | None = None
    cluster: WorkerCluster | None = None

    def on_event(event: dict[str, Any]) -> None:
        logger.log("progress", detail=event)
        if args.verbose or event.get("type") != "comm":
            _emit(event)

    try:
        store = CaptureStore(Path(args.capture), Path(args.meta))
        if args.layers:
            store.layers = [int(x) for x in args.layers.split(",")]
        if args.kv_heads:
            store.kv_heads = [int(x) for x in args.kv_heads.split(",")]
        config = BenchmarkConfig(
            workers=args.workers,
            overlap_pct=args.overlap,
            strategy=args.strategy,
            seed=args.seed,
            error_tol=args.eps,
            safe_gamma=args.gamma,
        )
        config.validate(LIMITS)
        cluster = WorkerCluster(
            num_workers=args.workers, run_dir=out_dir, event_cb=on_event, limits=LIMITS
        )
        _emit(
            {
                "stage": "start",
                "run_id": run_id,
                "workers": args.workers,
                "overlap_pct": args.overlap,
                "error_tol": args.eps,
                "token_count": len(store.tokens),
                "layers": store.layers,
                "kv_heads": store.kv_heads,
            }
        )
        cluster.start()
        _emit(
            {
                "stage": "cluster_ready",
                "workers": args.workers,
                "session": cluster.session_id,
                "free_ram_gb": cluster.preflight(),
            }
        )
        payload = run_benchmark(
            store,
            cluster,
            config,
            limits=LIMITS,
            logger=logger,
            event_cb=on_event,
            run_id=run_id,
        )
        payload["app_version"] = APP_VERSION
        payload["created_utc"] = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        payload["runner"] = {
            "argv": sys.argv[1:] if argv is None else argv,
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
        }
        benchmark_path = out_dir / "benchmark.json"
        _write_json_atomic(benchmark_path, payload)
        _write_json_atomic(out_dir / "cluster.json", payload.get("cluster", {}))
        _emit(
            {
                "stage": "done",
                "run_id": run_id,
                "benchmark_path": str(benchmark_path),
                "aggregate": payload["aggregate"],
                "timings": payload["timings"],
            }
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - the supervisor needs the failure
        logger.log(
            "error",
            level="error",
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(limit=8),
        )
        _emit(
            {
                "stage": "error",
                "run_id": run_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        return 1
    finally:
        if cluster is not None:
            try:
                cluster.shutdown()
            except Exception:
                pass
        if store is not None:
            store.close()


if __name__ == "__main__":
    raise SystemExit(main())
