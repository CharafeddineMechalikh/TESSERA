"""Run the real benchmark over a grid of planted-needle scenarios.

Every combination gets a fresh worker cluster, the same capture schema, the same
certificate, and both controls. The result is a screening table, not a model
result: it says which protocol settings recover a positive saving when the data
is separable by construction, and which settings still retain everything.

Writes ``sweep.json`` for the full record and ``sweep.md`` for reading.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .benchmark import BenchmarkConfig, CaptureStore, run_benchmark
from .cluster import WorkerCluster
from .config import APP_VERSION, LIMITS, OUTPUT_DIR, Limits
from .needle import NeedleScenario, write_capture

SWEEP_KIND = "planted-needle-sweep"
WARNING = (
    "Synthetic planted-needle data. These captures test the pruning mechanism, "
    "not attention on language, and their savings do not transfer to real prompts."
)


@dataclass(frozen=True)
class SweepSpec:
    out_dir: Path
    tokens: int = 64
    layers: Tuple[int, ...] = (0, 1)
    kv_heads: Tuple[int, ...] = (0, 1)
    q_per_kv: int = 2
    head_dim: int = 8
    needles: Tuple[int, ...] = (0, 1)
    margins: Tuple[float, ...] = (2.5, 6.0)
    workers: Tuple[int, ...] = (2, 4)
    overlaps: Tuple[float, ...] = (0.0, 15.0)
    error_tols: Tuple[float, ...] = (1e-2, 1e-3)
    strategies: Tuple[str, ...] = ("pseudorandom",)
    background_scale: float = 0.05
    value_scale: float = 0.05
    seed: int = 11
    max_runs: int = 64

    def validate(self, limits: Limits = LIMITS) -> None:
        if not self.needles:
            raise ValueError("needles must contain at least one value")
        if not self.margins or not self.workers or not self.overlaps or not self.error_tols:
            raise ValueError("workers, overlaps, error_tols, and margins must not be empty")
        if self.tokens < max(max(self.workers), 2):
            raise ValueError("tokens must be at least the largest worker count and 2")
        if min(self.needles) < 0 or max(self.needles) > self.tokens:
            raise ValueError("needles must be between 0 and tokens")
        for workers in self.workers:
            if not limits.min_workers <= workers <= limits.max_workers:
                raise ValueError(
                    f"workers must be between {limits.min_workers} and {limits.max_workers}"
                )
        for overlap in self.overlaps:
            if not 0.0 <= overlap <= limits.max_overlap_pct:
                raise ValueError(f"overlap must be between 0 and {limits.max_overlap_pct}")
        for tol in self.error_tols:
            if not 1e-9 <= tol <= 1.0:
                raise ValueError("error_tols must be between 1e-9 and 1")
        for strategy in self.strategies:
            if strategy not in {"pseudorandom", "contiguous"}:
                raise ValueError("strategy must be 'pseudorandom' or 'contiguous'")
        if self.max_runs < 1:
            raise ValueError("max_runs must be positive")
        planned = len(plan_runs(self))
        if planned > self.max_runs:
            raise ValueError(
                f"the grid plans {planned} runs, above max_runs={self.max_runs}. "
                "Narrow a list or raise --max-runs."
            )


def plan_runs(spec: SweepSpec) -> List[Tuple[NeedleScenario, BenchmarkConfig]]:
    """Expand the grid without touching the filesystem or spawning workers."""
    scenarios: List[NeedleScenario] = []
    for needles in spec.needles:
        margins = (0.0,) if needles == 0 else spec.margins
        for margin in margins:
            scenarios.append(
                NeedleScenario(
                    n_tokens=spec.tokens,
                    layers=spec.layers,
                    kv_heads=spec.kv_heads,
                    q_per_kv=spec.q_per_kv,
                    head_dim=spec.head_dim,
                    needle_count=needles,
                    needle_margin=margin,
                    background_scale=spec.background_scale,
                    value_scale=spec.value_scale,
                    seed=spec.seed,
                )
            )
    runs: List[Tuple[NeedleScenario, BenchmarkConfig]] = []
    for scenario in scenarios:
        for workers in spec.workers:
            for overlap in spec.overlaps:
                for tol in spec.error_tols:
                    for strategy in spec.strategies:
                        runs.append(
                            (
                                scenario,
                                BenchmarkConfig(
                                    workers=workers,
                                    overlap_pct=overlap,
                                    strategy=strategy,
                                    seed=spec.seed,
                                    error_tol=tol,
                                ),
                            )
                        )
    return runs


def _scenario_slug(scenario: NeedleScenario) -> str:
    return f"needles{scenario.needle_count}-margin{scenario.needle_margin:g}"


def _config_slug(config: BenchmarkConfig) -> str:
    return (
        f"w{config.workers}-overlap{config.overlap_pct:g}"
        f"-eps{config.error_tol:g}-{config.strategy}"
    )


def _event(
    event_cb: Optional[Callable[[dict], None]], message: str, **extra
) -> None:
    if event_cb is None:
        return
    payload = {"type": "sweep", "msg": message}
    payload.update(extra)
    event_cb(payload)


def _summarise_run(
    payload: dict, scenario: NeedleScenario, wall_s: float, payload_path: Path
) -> dict:
    aggregate = payload["aggregate"]
    per_unit = [
        {
            "layer": row["layer"],
            "kv_head": row["kv_head"],
            "retained_workers": row["retained_worker_count"],
            "retained_fraction": row["retained_fraction"],
            "max_certified_error": max(row["certified_error"]),
            "max_relative_error": max(row["relative_error"]),
        }
        for layer in payload["layers"]
        for row in layer["kv_heads"]
    ]
    return {
        "scenario": scenario.as_dict(),
        "config": payload["config"],
        "aggregate": aggregate,
        "per_unit": per_unit,
        "pruned": bool(aggregate["mean_retained_workers"] < payload["config"]["workers"] - 1e-9),
        "positive_online_savings": bool(aggregate["net_flop_savings_pct"] > 0.0),
        "positive_charged_savings": bool(
            aggregate["net_flop_savings_with_preprocessing_pct"] > 0.0
        ),
        "wall_s": wall_s,
        "payload_path": str(payload_path),
        "error": None,
    }


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _fmt(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}g}"


def _markdown(summary: dict) -> str:
    runs = summary["runs"]
    lines = [
        "# Planted-needle sweep",
        "",
        f"> {WARNING}",
        "",
        "Each row is one capture scenario and one protocol configuration, run through",
        "a fresh worker cluster with the real certificate and both controls. The",
        "needle-free scenario (`needles=0`) is the control: it has the same shapes and",
        "value scale but no deliberately separable token.",
        "",
        "| needles | margin | workers | overlap | eps | strategy | retained | tokens | certified | rel. error | online % | charged % | overlap ctrl | pruned | wall s |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for run in runs:
        scenario = run["scenario"]
        config = run["config"]
        aggregate = run.get("aggregate")
        if run["error"] is not None or aggregate is None:
            margin = "n/a" if scenario["needle_count"] == 0 else f"{scenario['needle_margin']:g}"
            retained = tokens = certified = relative = online = charged = overlap_ctrl = "error"
            pruned = "no"
        else:
            margin = "n/a" if scenario["needle_count"] == 0 else f"{scenario['needle_margin']:g}"
            retained = f"{aggregate['mean_retained_workers']:.2f}/{config['workers']}"
            tokens = f"{100.0 * aggregate['mean_retained_fraction']:.1f}%"
            certified = _fmt(aggregate["max_certified_error"])
            relative = _fmt(aggregate["max_relative_error"])
            online = f"{aggregate['net_flop_savings_pct']:.2f}"
            charged = f"{aggregate['net_flop_savings_with_preprocessing_pct']:.2f}"
            overlap_ctrl = _fmt(aggregate["max_overlap_control_rel_error"])
            pruned = "yes" if run["pruned"] else "no"
        lines.append(
            "| {needles} | {margin} | {workers} | {overlap:g} | {eps:g} | {strategy} | "
            "{retained} | {tokens} | {certified} | {relative} | {online} | {charged} | "
            "{overlap_ctrl} | {pruned} | {wall:.2f} |".format(
                needles=scenario["needle_count"],
                margin=margin,
                workers=config["workers"],
                overlap=config["overlap_pct"],
                eps=config["error_tol"],
                strategy=config["strategy"],
                retained=retained,
                tokens=tokens,
                certified=certified,
                relative=relative,
                online=online,
                charged=charged,
                overlap_ctrl=overlap_ctrl,
                pruned=pruned,
                wall=run["wall_s"],
            )
        )
    lines.extend(["", "## Best pruned combination", ""])
    best = summary.get("best")
    if best is None:
        lines.append(
            "No combination both pruned and produced a positive saving. "
            "The certificate never accepted a smaller retained set at these tolerances."
        )
    else:
        lines.append(
            "Charged once (the conservative single-run figure): `needles={needles}`, "
            "`margin={margin}`, `workers={workers}`, `overlap={overlap:g}`, "
            "`eps={eps:g}`, `{strategy}`: charged `{charged:.2f}%`, online "
            "`{online:.2f}%`, retained `{retained:.2f}/{workers}`, certified "
            "`{certified}`, relative error `{relative}`.".format(
                needles=best["scenario"]["needle_count"],
                margin=(
                    "n/a"
                    if best["scenario"]["needle_count"] == 0
                    else f"{best['scenario']['needle_margin']:g}"
                ),
                workers=best["config"]["workers"],
                overlap=best["config"]["overlap_pct"],
                eps=best["config"]["error_tol"],
                strategy=best["config"]["strategy"],
                charged=best["aggregate"]["net_flop_savings_with_preprocessing_pct"],
                online=best["aggregate"]["net_flop_savings_pct"],
                retained=best["aggregate"]["mean_retained_workers"],
                certified=_fmt(best["aggregate"]["max_certified_error"]),
                relative=_fmt(best["aggregate"]["max_relative_error"]),
            )
        )
        cached = summary.get("best_online")
        if cached is not None and cached is not best:
            lines.append("")
            lines.append(
                "Cached bounds (the preprocessing term excluded): `needles={needles}`, "
                "`margin={margin}`, `workers={workers}`, `overlap={overlap:g}`, "
                "`eps={eps:g}`, `{strategy}`: online `{online:.2f}%`, charged "
                "`{charged:.2f}%`, retained `{retained:.2f}/{workers}`.".format(
                    needles=cached["scenario"]["needle_count"],
                    margin=(
                        "n/a"
                        if cached["scenario"]["needle_count"] == 0
                        else f"{cached['scenario']['needle_margin']:g}"
                    ),
                    workers=cached["config"]["workers"],
                    overlap=cached["config"]["overlap_pct"],
                    eps=cached["config"]["error_tol"],
                    strategy=cached["config"]["strategy"],
                    online=cached["aggregate"]["net_flop_savings_pct"],
                    charged=cached["aggregate"]["net_flop_savings_with_preprocessing_pct"],
                    retained=cached["aggregate"]["mean_retained_workers"],
                )
            )
    control = summary.get("control", {})
    lines.extend(
        [
            "",
            "The needle-free control retained every worker in "
            f"`{control.get('unpruned', 0)}/{control.get('runs', 0)}` runs.",
            "",
            "## How to read it",
            "",
            "- The control should retain every worker. If it prunes, a setting is",
            "  accepting an error larger than the data justifies, or the capture is not",
            "  the control you think it is.",
            "- A larger margin makes the needle more separable and the certificate accepts",
            "  fewer workers. That is the generator property quoted from the prototype,",
            "  demonstrated rather than assumed.",
            "- Overlap changes the retry protocol and adds selection work on duplicated",
            "  rows. Compare `online %` at equal margin before comparing worker counts.",
            "- The charged column includes one bounds pass at `R = 1`. A positive online",
            "  figure with a negative charged figure only works if the bounds are cached.",
            "- No row here is evidence about a real prompt. The same configuration can",
            "  prune on planted data and retain every worker on captured Qwen2 tensors,",
            "  which is the measured result in `docs/02-protocol-math.md`.",
            "",
        ]
    )
    return "\n".join(lines)


def run_sweep(
    spec: SweepSpec,
    limits: Limits = LIMITS,
    event_cb: Optional[Callable[[dict], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> dict:
    spec.validate(limits)
    out_dir = Path(spec.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    planned = plan_runs(spec)
    _event(event_cb, f"sweep starting: {len(planned)} runs", planned=len(planned))

    started = time.perf_counter()
    runs: List[dict] = []
    scenarios: Dict[str, NeedleScenario] = {}
    current_slug: Optional[str] = None
    store: Optional[CaptureStore] = None
    try:
        for index, (scenario, config) in enumerate(planned, start=1):
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("sweep cancelled by operator")
            scenario_slug = _scenario_slug(scenario)
            if scenario_slug != current_slug:
                if store is not None:
                    store.close()
                capture_dir = out_dir / "scenarios" / scenario_slug
                artifacts = write_capture(scenario, capture_dir)
                store = CaptureStore(Path(artifacts["capture"]), Path(artifacts["meta"]))
                current_slug = scenario_slug
                scenarios[scenario_slug] = scenario
            assert store is not None

            config_slug = _config_slug(config)
            run_dir = out_dir / "runs" / f"{scenario_slug}__{config_slug}"
            run_dir.mkdir(parents=True, exist_ok=True)
            _event(
                event_cb,
                f"run {index}/{len(planned)}: {scenario_slug} {config_slug}",
                position=index,
                total=len(planned),
                scenario=scenario_slug,
                config=config_slug,
            )
            cluster = WorkerCluster(
                num_workers=config.workers, run_dir=run_dir, limits=limits
            )
            run_started = time.perf_counter()
            try:
                cluster.start()
                payload = run_benchmark(
                    store, cluster, config, limits=limits, run_id=f"{scenario_slug}-{config_slug}"
                )
                payload["app_version"] = APP_VERSION
                payload["created_utc"] = datetime.now(timezone.utc).isoformat(
                    timespec="milliseconds"
                )
                payload["cluster"] = cluster.summary()
                payload_path = run_dir / "benchmark.json"
                _write_json(payload_path, payload)
                row = _summarise_run(
                    payload, scenario, time.perf_counter() - run_started, payload_path
                )
            except Exception as exc:  # noqa: BLE001 - one bad cell must not stop the grid
                row = {
                    "scenario": scenario.as_dict(),
                    "config": asdict(config),
                    "aggregate": None,
                    "per_unit": [],
                    "pruned": False,
                    "positive_online_savings": False,
                    "positive_charged_savings": False,
                    "wall_s": time.perf_counter() - run_started,
                    "payload_path": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            finally:
                try:
                    cluster.shutdown()
                except Exception:
                    pass
            runs.append(row)
            _event(
                event_cb,
                f"run {index}/{len(planned)} done: {row['error'] or 'ok'}",
                position=index,
                total=len(planned),
                error=row["error"],
            )
    finally:
        if store is not None:
            store.close()

    eligible = []
    for run in runs:
        if run["error"] is not None or run["aggregate"] is None:
            continue
        aggregate = run["aggregate"]
        if (
            run["pruned"]
            and aggregate["tolerance_satisfied"]
            and aggregate["net_flop_savings_pct"] > 0.0
            and aggregate["max_disjoint_control_rel_error"] < 1e-8
            and (
                aggregate["max_overlap_control_rel_error"] is None
                or aggregate["max_overlap_control_rel_error"] < 1e-8
            )
        ):
            eligible.append(run)

    def _rank(run: dict, key: str) -> tuple:
        return (run["aggregate"][key], -run["config"]["workers"])

    best_online = (
        max(eligible, key=lambda run: _rank(run, "net_flop_savings_pct"))
        if eligible
        else None
    )
    best_charged = (
        max(eligible, key=lambda run: _rank(run, "net_flop_savings_with_preprocessing_pct"))
        if eligible
        else None
    )
    control_runs = [
        run
        for run in runs
        if run["scenario"]["needle_count"] == 0 and run["error"] is None
    ]

    summary = {
        "kind": SWEEP_KIND,
        "synthetic": True,
        "warning": WARNING,
        "app_version": APP_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "spec": {
            **asdict(spec),
            "out_dir": str(out_dir),
            "layers": list(spec.layers),
            "kv_heads": list(spec.kv_heads),
            "needles": list(spec.needles),
            "margins": list(spec.margins),
            "workers": list(spec.workers),
            "overlaps": list(spec.overlaps),
            "error_tols": list(spec.error_tols),
            "strategies": list(spec.strategies),
        },
        "planned_runs": len(planned),
        "completed_runs": len(runs),
        "elapsed_s": time.perf_counter() - started,
        "runs": runs,
        "best": best_charged if best_charged is not None else best_online,
        "best_online": best_online,
        "best_charged": best_charged,
        "control": {
            "runs": len(control_runs),
            "unpruned": sum(1 for run in control_runs if not run["pruned"]),
        },
    }
    _write_json(out_dir / "sweep.json", summary)
    (out_dir / "sweep.md").write_text(_markdown(summary), encoding="utf-8")
    _event(
        event_cb,
        f"sweep done: {len(runs)} runs in {summary['elapsed_s']:.1f}s",
        best=summary["best"]["config"] if summary["best"] else None,
    )
    return summary


def _int_tuple(value: str) -> Tuple[int, ...]:
    return tuple(int(part) for part in value.split(",") if part.strip())


def _float_tuple(value: str) -> Tuple[float, ...]:
    return tuple(float(part) for part in value.split(",") if part.strip())


def _str_tuple(value: str) -> Tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sweep planted-needle pruning combinations")
    parser.add_argument(
        "--out-dir",
        default=str(OUTPUT_DIR / "sweeps" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")),
    )
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--layers", default="0,1")
    parser.add_argument("--kv-heads", default="0,1")
    parser.add_argument("--q-per-kv", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=8)
    parser.add_argument("--needles", default="0,1")
    parser.add_argument("--margins", default="2.5,6")
    parser.add_argument("--workers", default="2,4")
    parser.add_argument("--overlap", default="0,15")
    parser.add_argument("--eps", default="1e-2,1e-3")
    parser.add_argument("--strategy", default="pseudorandom")
    parser.add_argument("--background-scale", type=float, default=0.05)
    parser.add_argument("--value-scale", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--max-runs", type=int, default=64)
    args = parser.parse_args(argv)
    spec = SweepSpec(
        out_dir=Path(args.out_dir),
        tokens=args.tokens,
        layers=_int_tuple(args.layers),
        kv_heads=_int_tuple(args.kv_heads),
        q_per_kv=args.q_per_kv,
        head_dim=args.head_dim,
        needles=_int_tuple(args.needles),
        margins=_float_tuple(args.margins),
        workers=_int_tuple(args.workers),
        overlaps=_float_tuple(args.overlap),
        error_tols=_float_tuple(args.eps),
        strategies=_str_tuple(args.strategy),
        background_scale=args.background_scale,
        value_scale=args.value_scale,
        seed=args.seed,
        max_runs=args.max_runs,
    )
    summary = run_sweep(spec)
    print(json.dumps({"sweep": str(Path(spec.out_dir).resolve()), "runs": len(summary["runs"])}, indent=2))
    return 0 if all(run["error"] is None for run in summary["runs"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
