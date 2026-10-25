"""Real-attention benchmark over captured Qwen2 tensors.

For every layer and KV head the coordinator sends block bounds and local softmax
statistics to K isolated workers, retains the smallest number of whole blocks
that satisfies the certified error tolerance, and compares the merged worker
output with the dense reference. Workers never receive the reference output.
"""
from __future__ import annotations

import io
import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .capture import Geometry
from .cluster import WorkerCluster
from .config import LIMITS, Limits
from .logging_utils import JsonlLogger
from .protocol import (
    block_bounds_batch,
    certified_error_bound,
    merge_worker_stats,
    omitted_mass_bound,
    partition_tokens,
    primary_worker,
    reference_attention,
    retained_multiplicities,
)


class BenchmarkError(RuntimeError):
    pass


class BenchmarkCancelled(BenchmarkError):
    pass


def _check_cancel(cancel_event: Optional[threading.Event]) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise BenchmarkCancelled("run cancelled by operator")


@dataclass(frozen=True)
class BenchmarkConfig:
    workers: int
    overlap_pct: float = 0.0
    strategy: str = "pseudorandom"
    seed: int = 42
    error_tol: float = 1e-3
    safe_gamma: float = LIMITS.safe_gamma

    def validate(self, limits: Limits) -> None:
        if not limits.min_workers <= self.workers <= limits.max_workers:
            raise ValueError(f"workers must be between {limits.min_workers} and {limits.max_workers}")
        if not 0.0 <= self.overlap_pct <= limits.max_overlap_pct:
            raise ValueError(f"overlap_pct must be between 0 and {limits.max_overlap_pct}")
        if self.strategy not in {"pseudorandom", "contiguous"}:
            raise ValueError("strategy must be 'pseudorandom' or 'contiguous'")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not 1e-9 <= self.error_tol <= 1.0:
            raise ValueError("error_tol must be between 1e-9 and 1")


class CaptureStore:
    """Lazy reader for the capture npz. Only one layer head pair is resident."""

    def __init__(self, npz_path: Path, meta_path: Path):
        self.npz_path = Path(npz_path)
        self.meta_path = Path(meta_path)
        if not self.npz_path.exists() or not self.meta_path.exists():
            raise BenchmarkError("capture artifacts are missing")
        self.meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        geometry = dict(self.meta["geometry"])
        geometry.pop("q_per_kv", None)
        self.geometry = Geometry(**geometry)
        self.tokens = np.asarray(self.meta["tokens"], dtype=np.int64)
        self.layers = [int(x) for x in self.meta["layers"]]
        self.kv_heads = [int(x) for x in self.meta["kv_heads"]]
        self._npz = np.load(self.npz_path, allow_pickle=False)

    def keys_values(self, layer: int, kv_head: int) -> Tuple[np.ndarray, np.ndarray]:
        key = f"k_{layer}_{kv_head}"
        if key not in self._npz.files:
            raise BenchmarkError(f"capture is missing {key}")
        k = np.asarray(self._npz[key], dtype=np.float64)
        v = np.asarray(self._npz[f"v_{layer}_{kv_head}"], dtype=np.float64)
        return k, v

    def query(self, layer: int, kv_head: int) -> np.ndarray:
        key = f"q_{layer}_{kv_head}"
        if key not in self._npz.files:
            raise BenchmarkError(f"capture is missing {key}")
        return np.asarray(self._npz[key], dtype=np.float64)

    def close(self) -> None:
        self._npz.close()


def _build_shard(
    store: CaptureStore,
    positions: np.ndarray,
    primary: np.ndarray,
    limits: Limits,
    head_dim: int,
) -> bytes:
    count = int(positions.shape[0])
    estimated = count * len(store.layers) * len(store.kv_heads) * head_dim * 8 * 2
    if estimated > limits.max_shard_bytes:
        raise BenchmarkError(
            f"a worker shard would need about {estimated / 1e6:.1f} MB, above the "
            f"{limits.max_shard_bytes / 1e6:.0f} MB limit. Lower the context length "
            f"or overlap, or increase the worker count."
        )
    arrays: Dict[str, np.ndarray] = {
        "tokens": np.asarray(positions, dtype=np.int64),
        "primary": np.asarray(primary, dtype=bool),
    }
    for layer in store.layers:
        for kv_head in store.kv_heads:
            keys, values = store.keys_values(layer, kv_head)
            arrays[f"k_{layer}_{kv_head}"] = np.ascontiguousarray(keys[positions])
            arrays[f"v_{layer}_{kv_head}"] = np.ascontiguousarray(values[positions])
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


def _retained_positions(assignments: Sequence[np.ndarray], retained: Sequence[int]) -> np.ndarray:
    if not retained:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate([assignments[w] for w in retained]))


def _omitted_upper_bounds(
    assignments: Sequence[np.ndarray],
    retained_positions: np.ndarray,
    pruned_workers: Sequence[int],
    bounds: np.ndarray,
    head: int,
) -> Dict[int, float]:
    retained_set = set(int(x) for x in retained_positions)
    uppers: Dict[int, float] = {}
    for worker in pruned_workers:
        value = float(bounds[worker, head])
        for position in assignments[worker]:
            position = int(position)
            if position in retained_set:
                continue
            current = uppers.get(position)
            if current is None or value > current:
                uppers[position] = value
    return uppers


def _dense_reference(
    q: np.ndarray, keys: np.ndarray, values: np.ndarray, scale: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    probs = np.empty((q.shape[0], keys.shape[0]), dtype=np.float64)
    outputs = np.empty_like(q, dtype=np.float64)
    for head in range(q.shape[0]):
        head_probs, head_out = reference_attention(q[head], keys, values, scale)
        probs[head] = head_probs
        outputs[head] = head_out
    return probs, outputs, np.linalg.norm(values, axis=1)


def prepare_worker_shards(
    store: CaptureStore,
    assignments: Sequence[np.ndarray],
    primary: np.ndarray,
    limits: Limits,
    event_cb: Optional[Callable[[dict], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> List[bytes]:
    shards: List[bytes] = []
    for worker, positions in enumerate(assignments):
        _check_cancel(cancel_event)
        started = time.perf_counter()
        shard = _build_shard(
            store,
            positions,
            primary[positions] == worker,
            limits,
            store.geometry.head_dim,
        )
        shards.append(shard)
        if event_cb:
            event_cb(
                {
                    "type": "comm",
                    "msg": f"shard-{worker}: {len(positions)} positions, {len(shard) / 1e6:.2f} MB",
                    "event": "shard_built",
                    "worker_id": worker,
                }
            )
    return shards


def run_benchmark(
    store: CaptureStore,
    cluster: WorkerCluster,
    config: BenchmarkConfig,
    limits: Limits = LIMITS,
    logger: Optional[JsonlLogger] = None,
    event_cb: Optional[Callable[[dict], None]] = None,
    run_id: str = "run",
    cancel_event: Optional[threading.Event] = None,
) -> dict:
    config.validate(limits)
    _check_cancel(cancel_event)
    n_tokens = len(store.tokens)
    if n_tokens < config.workers:
        raise BenchmarkError("token count must be at least the worker count")
    geometry = store.geometry
    heads = geometry.q_per_kv
    head_dim = geometry.head_dim
    scale = 1.0 / math.sqrt(head_dim)

    started = time.perf_counter()
    assignments, counts = partition_tokens(
        n_tokens, config.workers, config.strategy, config.overlap_pct, config.seed
    )
    primary = primary_worker(n_tokens, config.workers, config.strategy, config.seed)
    copies_per_token = counts.astype(np.int64)
    assigned_rows = int(copies_per_token.sum())
    extra_copies = assigned_rows - n_tokens
    copied_token_fraction = float(np.mean(copies_per_token > 1))
    # Each worker computes a centroid and radius per unit while loading its
    # shard, so the bounds pass costs about 5 * rows * d per unit. Rows include
    # secondary copies, and the term is reported separately because the
    # dashboard can charge it once, amortise it, or exclude it.
    preprocessing_flops_per_unit = 5 * assigned_rows * head_dim

    if event_cb:
        event_cb(
            {
                "type": "stage",
                "stage": "partition",
                "workers": config.workers,
                "tokens": n_tokens,
                "mean_copies": float(copies_per_token.mean()),
                "primary_per_worker": [int((primary == w).sum()) for w in range(config.workers)],
            }
        )

    shards = prepare_worker_shards(
        store, assignments, primary, limits, event_cb, cancel_event=cancel_event
    )
    upload_started = time.perf_counter()
    cluster.upload_shards(shards, store.layers, store.kv_heads, heads)
    upload_s = time.perf_counter() - upload_started
    del shards
    _check_cancel(cancel_event)

    layers_payload: List[dict] = []
    worst_case: Optional[dict] = None
    worst_rel = -1.0
    max_relative_error = 0.0
    max_certified_error = 0.0
    max_actual_error = 0.0
    max_control_error = 0.0
    total_baseline_flops = 0
    total_distributed_flops = 0
    retained_fractions: List[float] = []
    retained_worker_counts: List[int] = []
    peak_masses: List[float] = []
    entropies: List[float] = []
    tolerance_ok = True
    stage_ms: Dict[str, float] = {
        "bounds": 0.0,
        "prune": 0.0,
        "final": 0.0,
        "control": 0.0,
        "overlap_control": 0.0,
    }
    total_preprocessing_flops = 0
    total_overlap_stats_flops = 0
    max_overlap_control_error: Optional[float] = None
    # With unique ownership every worker's local statistics are final, so the
    # coordinator can query workers one at a time in bound order and reuse the
    # accumulated statistics. With overlap the correction for a position can
    # change when a second copy becomes retained, so the retained set must be
    # re-queried; that work is counted explicitly below.
    incremental = config.overlap_pct == 0.0

    total_units = len(store.layers) * len(store.kv_heads)
    unit = 0
    for layer_idx, layer in enumerate(store.layers):
        layer_rows = []
        for kv_head in store.kv_heads:
            _check_cancel(cancel_event)
            unit += 1
            q = store.query(layer, kv_head)
            keys, values = store.keys_values(layer, kv_head)
            if q.shape != (heads, head_dim) or keys.shape[1] != head_dim:
                raise BenchmarkError("captured tensor shapes do not match the geometry")
            if not (np.isfinite(q).all() and np.isfinite(keys).all() and np.isfinite(values).all()):
                raise BenchmarkError("captured tensors contain non-finite values")

            probs, full_out, value_norms = _dense_reference(q, keys, values, scale)
            v_max = float(value_norms.max())
            peak_masses.append(float(probs.max(axis=1).mean() * 100.0))
            entropies.append(
                float(np.mean(-np.sum(probs * np.log(np.maximum(probs, 1e-300)), axis=1)) / math.log(2.0))
            )

            bounds_started = time.perf_counter()
            bounds = cluster.bounds(layer, kv_head, q)
            stage_ms["bounds"] += (time.perf_counter() - bounds_started) * 1000.0
            if bounds.shape != (config.workers, heads):
                raise BenchmarkError("worker returned bounds with the wrong shape")
            effective = bounds.max(axis=1)
            order = np.lexsort((np.arange(config.workers), -effective))

            selection = None
            curve: List[dict] = []
            accumulated_lmax: List[np.ndarray] = []
            accumulated_sumexp: List[np.ndarray] = []
            accumulated_outs: List[np.ndarray] = []
            selection_stats_occurrences = 0
            selection_duplicate_occurrences = 0
            for m in range(1, config.workers + 1):
                _check_cancel(cancel_event)
                stage_started = time.perf_counter()
                retained = [int(w) for w in order[:m]]
                if incremental:
                    newly_queried = [retained[-1]]
                    lmax, sumexp, outs, _meta = cluster.stats(
                        layer, kv_head, q, None, "assigned", newly_queried
                    )
                    accumulated_lmax.append(lmax[0])
                    accumulated_sumexp.append(sumexp[0])
                    accumulated_outs.append(outs[0])
                    lmax = np.stack(accumulated_lmax)
                    sumexp = np.stack(accumulated_sumexp)
                    outs = np.stack(accumulated_outs)
                    selection_stats_occurrences += len(assignments[retained[-1]])
                else:
                    multiplicities = retained_multiplicities(assignments, retained)
                    lmax, sumexp, outs, _meta = cluster.stats(
                        layer, kv_head, q, multiplicities, "assigned", retained
                    )
                    selection_stats_occurrences += sum(len(assignments[w]) for w in retained)
                dist_out, retained_totals, retained_max = merge_worker_stats(lmax, sumexp, outs)
                stage_ms["prune"] += (time.perf_counter() - stage_started) * 1000.0

                held = _retained_positions(assignments, retained)
                if not incremental:
                    queried_rows = sum(len(assignments[w]) for w in retained)
                    selection_duplicate_occurrences += queried_rows - int(len(held))
                omitted = np.zeros(heads, dtype=np.float64)
                certified = np.zeros(heads, dtype=np.float64)
                if m < config.workers:
                    pruned = [w for w in range(config.workers) if w not in set(retained)]
                    for head in range(heads):
                        uppers = _omitted_upper_bounds(
                            assignments, held, pruned, bounds, head
                        )
                        omitted[head] = omitted_mass_bound(
                            float(retained_totals[head]),
                            float(retained_max[head]),
                            uppers,
                            config.safe_gamma,
                        )
                        certified[head] = certified_error_bound(omitted[head], v_max)
                curve_relative = np.linalg.norm(dist_out - full_out, axis=1) / np.maximum(
                    np.linalg.norm(full_out, axis=1), 1e-30
                )
                curve.append(
                    {
                        "retained_workers": int(m),
                        "retained_unique_tokens": int(len(held)),
                        "retained_fraction": float(len(held) / n_tokens),
                        "max_certified_error": float(certified.max(initial=0.0)),
                        "max_relative_error": float(curve_relative.max(initial=0.0)),
                        "tolerance_satisfied": bool(
                            float(certified.max(initial=0.0)) <= config.error_tol
                        ),
                    }
                )
                if float(certified.max(initial=0.0)) <= config.error_tol:
                    selection = {
                        "retained": retained,
                        "dist_out": dist_out,
                        "omitted": omitted,
                        "certified": certified,
                        "m": m,
                    }
                    break

            if selection is None:
                raise BenchmarkError("no retained set satisfied the tolerance, including all workers")
            retained = selection["retained"]
            dist_out = selection["dist_out"]
            omitted = selection["omitted"]
            certified = selection["certified"]
            held = _retained_positions(assignments, retained)
            retained_occurrences = int(sum(len(assignments[w]) for w in retained))

            actual = np.linalg.norm(dist_out - full_out, axis=1)
            reference_norms = np.linalg.norm(full_out, axis=1)
            relative = actual / np.maximum(reference_norms, 1e-30)
            allowed = certified + 1e-8 * (1.0 + reference_norms)
            if np.any(actual > allowed):
                raise BenchmarkError(
                    f"soundness check failed at layer {layer} kv {kv_head}: "
                    f"actual={actual.tolist()} certified={certified.tolist()}"
                )
            unit_tolerance = bool(np.all(certified <= config.error_tol))
            tolerance_ok = tolerance_ok and unit_tolerance

            control_started = time.perf_counter()
            _check_cancel(cancel_event)
            c_lmax, c_sumexp, c_out, _ = cluster.stats(
                layer,
                kv_head,
                q,
                None,
                "primary",
                list(range(config.workers)),
            )
            control_out, _, _ = merge_worker_stats(c_lmax, c_sumexp, c_out)
            control_error = float(
                np.max(np.linalg.norm(control_out - full_out, axis=1) / np.maximum(reference_norms, 1e-30))
            )
            stage_ms["control"] += (time.perf_counter() - control_started) * 1000.0
            if control_error > 1e-8:
                raise BenchmarkError(
                    f"disjoint control does not match dense attention at layer {layer} "
                    f"kv {kv_head}: relative error {control_error:.3e}"
                )

            # The disjoint control never sends a duplicated position, so it
            # cannot validate the -log(c) overlap correction. This control asks
            # every worker for every copy and merges them with the same
            # multiplicities the retained path uses.
            overlap_control_error: Optional[float] = None
            if config.overlap_pct > 0.0:
                overlap_started = time.perf_counter()
                _check_cancel(cancel_event)
                every_worker = list(range(config.workers))
                overlap_multiplicities = retained_multiplicities(assignments, every_worker)
                o_lmax, o_sumexp, o_out, _ = cluster.stats(
                    layer, kv_head, q, overlap_multiplicities, "assigned", every_worker
                )
                overlap_out, _, _ = merge_worker_stats(o_lmax, o_sumexp, o_out)
                overlap_control_error = float(
                    np.max(
                        np.linalg.norm(overlap_out - full_out, axis=1)
                        / np.maximum(reference_norms, 1e-30)
                    )
                )
                stage_ms["overlap_control"] += (time.perf_counter() - overlap_started) * 1000.0
                if overlap_control_error > 1e-8:
                    raise BenchmarkError(
                        f"overlap control does not match dense attention at layer {layer} "
                        f"kv {kv_head}: relative error {overlap_control_error:.3e}"
                    )
                max_overlap_control_error = (
                    overlap_control_error
                    if max_overlap_control_error is None
                    else max(max_overlap_control_error, overlap_control_error)
                )

            baseline_flops = heads * n_tokens * (4 * head_dim + 3)
            bounds_flops = config.workers * heads * (2 * head_dim + 2) + heads * (2 * head_dim + 1)
            stats_flops = heads * selection_stats_occurrences * (4 * head_dim + 3)
            merge_flops = len(retained) * heads * (2 * head_dim + 1)
            distributed_flops = bounds_flops + stats_flops + merge_flops
            savings = 100.0 * (1.0 - distributed_flops / baseline_flops)
            preprocessing_flops = preprocessing_flops_per_unit
            overlap_stats_flops = (
                heads * selection_duplicate_occurrences * (4 * head_dim + 3)
            )

            total_baseline_flops += baseline_flops
            total_distributed_flops += distributed_flops
            total_preprocessing_flops += preprocessing_flops
            total_overlap_stats_flops += overlap_stats_flops
            max_relative_error = max(max_relative_error, float(relative.max()))
            max_certified_error = max(max_certified_error, float(certified.max()))
            max_actual_error = max(max_actual_error, float(actual.max()))
            max_control_error = max(max_control_error, control_error)
            retained_fractions.append(len(held) / n_tokens)
            retained_worker_counts.append(len(retained))

            row = {
                "layer": int(layer),
                "kv_head": int(kv_head),
                "q_heads": geometry.q_heads_for_kv(kv_head),
                "v_max": v_max,
                "scale": scale,
                "retained_workers": retained,
                "retained_worker_count": len(retained),
                "retained_unique_tokens": int(len(held)),
                "retained_token_occurrences": retained_occurrences,
                "selection_stats_occurrences": int(selection_stats_occurrences),
                "selection_unique_occurrences": int(
                    selection_stats_occurrences - selection_duplicate_occurrences
                ),
                "total_tokens": n_tokens,
                "retained_fraction": len(held) / n_tokens,
                "block_bounds": bounds.tolist(),
                "omitted_mass": omitted.tolist(),
                "certified_error": certified.tolist(),
                "actual_l2_error": actual.tolist(),
                "relative_error": relative.tolist(),
                "tolerance_satisfied": unit_tolerance,
                "retention_curve": curve,
                "bounds_flops": int(bounds_flops),
                "stats_flops": int(stats_flops),
                "merge_flops": int(merge_flops),
                "baseline_flops": int(baseline_flops),
                "distributed_flops": int(distributed_flops),
                "preprocessing_flops": int(preprocessing_flops),
                "overlap_stats_flops": int(overlap_stats_flops),
                "flop_savings_pct": savings,
                "disjoint_control_rel_error": control_error,
                "overlap_control_rel_error": overlap_control_error,
                "peak_attention_mass_pct": float(probs.max(axis=1).mean() * 100.0),
                "attention_entropy_bits": float(
                    np.mean(-np.sum(probs * np.log(np.maximum(probs, 1e-300)), axis=1)) / math.log(2.0)
                ),
            }
            layer_rows.append(row)

            if float(relative.max()) > worst_rel:
                head = int(np.argmax(relative))
                worst_rel = float(relative.max())
                worst_case = {
                    "layer": int(layer),
                    "kv_head": int(kv_head),
                    "q_head": int(geometry.q_heads_for_kv(kv_head)[head]),
                    "head_index": head,
                    "relative_error": float(relative[head]),
                    "certified_error": float(certified[head]),
                    "full_output": full_out[head].tolist(),
                    "distributed_output": dist_out[head].tolist(),
                }

            if logger:
                logger.log(
                    "layer_kv",
                    layer=int(layer),
                    kv_head=int(kv_head),
                    retained=retained,
                    retained_unique=int(len(held)),
                    max_certified=float(certified.max()),
                    max_relative=float(relative.max()),
                    control=control_error,
                    savings_pct=round(savings, 4),
                    v_max=v_max,
                )
            if event_cb:
                event_cb(
                    {
                        "type": "stage",
                        "stage": "benchmark",
                        "layer": int(layer),
                        "kv_head": int(kv_head),
                        "position": unit,
                        "total": total_units,
                        "retained": len(retained),
                        "workers": config.workers,
                        "certified": float(certified.max()),
                        "relative_error": float(relative.max()),
                        "savings_pct": float(savings),
                        "tolerance_satisfied": unit_tolerance,
                    }
                )
        layers_payload.append(
            {
                "layer": int(layer),
                "kv_heads": layer_rows,
                "max_certified_error": max(row["certified_error"] for row in layer_rows),
                "max_relative_error": max(max(row["relative_error"]) for row in layer_rows),
                "mean_flop_savings_pct": float(np.mean([row["flop_savings_pct"] for row in layer_rows])),
            }
        )

    benchmark_s = time.perf_counter() - started
    net_savings = 100.0 * (1.0 - total_distributed_flops / total_baseline_flops)
    total_with_preprocessing = total_distributed_flops + total_preprocessing_flops
    net_savings_with_preprocessing = 100.0 * (
        1.0 - total_with_preprocessing / total_baseline_flops
    )
    online_headroom = total_baseline_flops - total_distributed_flops
    # Smallest number of runs sharing one bounds pass for the preprocessing
    # charge to stop deciding the sign. None means no finite reuse can help:
    # the online path already costs at least as much as dense attention.
    break_even_reuse = (
        int(math.floor(total_preprocessing_flops / online_headroom)) + 1
        if online_headroom > 0 and total_preprocessing_flops > 0
        else None
    )
    aggregate = {
        "max_relative_error": max_relative_error,
        "max_actual_l2_error": max_actual_error,
        "max_certified_error": max_certified_error,
        "tolerance_satisfied": tolerance_ok,
        "max_disjoint_control_rel_error": max_control_error,
        "max_overlap_control_rel_error": max_overlap_control_error,
        "mean_flop_savings_pct": float(
            np.mean([row["flop_savings_pct"] for layer in layers_payload for row in layer["kv_heads"]])
        ),
        "net_flop_savings_pct": net_savings,
        "net_flop_savings_with_preprocessing_pct": net_savings_with_preprocessing,
        "baseline_flops": int(total_baseline_flops),
        "distributed_flops": int(total_distributed_flops),
        "preprocessing_flops": int(total_preprocessing_flops),
        "preprocessing_share_pct": 100.0 * total_preprocessing_flops / total_baseline_flops,
        "preprocessing_break_even_runs": break_even_reuse,
        "total_flops_with_preprocessing": int(total_with_preprocessing),
        "overlap_stats_extra_flops": int(total_overlap_stats_flops),
        "overlap_stats_extra_pct": (
            100.0 * total_overlap_stats_flops / total_distributed_flops
            if total_distributed_flops > 0
            else 0.0
        ),
        "assigned_rows": int(assigned_rows),
        "extra_copies": int(extra_copies),
        "copied_token_fraction": copied_token_fraction,
        "mean_retained_fraction": float(np.mean(retained_fractions)),
        "mean_retained_workers": float(np.mean(retained_worker_counts)),
        "mean_copies_per_token": float(copies_per_token.mean()),
        "peak_attention_mass_pct": float(np.mean(peak_masses)),
        "attention_entropy_bits": float(np.mean(entropies)),
    }
    cluster_summary = cluster.summary()
    payload = {
        "run_id": run_id,
        "config": {
            "workers": config.workers,
            "overlap_pct": config.overlap_pct,
            "strategy": config.strategy,
            "seed": config.seed,
            "error_tol": config.error_tol,
            "safe_gamma": config.safe_gamma,
            "selection_protocol": "incremental" if incremental else "requery",
            "flop_model": (
                "baseline H*N*(4d+3); bounds K*H*(2d+2)+H*(2d+1) for the query "
                "norms; stats occurrences*H*(4d+3); merge retained_workers*H*(2d+1). "
                "Selection work actually performed is included in stats for the "
                "requery protocol, and rows duplicated by overlap are reported "
                "separately as overlap_stats_flops. Bounds preprocessing is "
                "5*assigned_rows*d per unit, reported separately so it can be "
                "charged once, amortised over reused bounds, or excluded from the "
                "online figure."
            ),
            "accounting": {
                "preprocessing_flops_per_unit": "5 * assigned_rows * head_dim",
                "assigned_rows": int(assigned_rows),
                "extra_copies": int(extra_copies),
                "bounds_preprocessing_in_net_flop_savings": False,
            },
        },
        "geometry": geometry.as_dict(),
        "token_count": n_tokens,
        "prompt_sha256": store.meta.get("prompt_sha256"),
        "tokenizer_class": store.meta.get("tokenizer_class"),
        "truncated": store.meta.get("truncated"),
        "original_token_count": store.meta.get("original_token_count"),
        "partition": {
            "strategy": config.strategy,
            "seed": config.seed,
            "overlap_pct": config.overlap_pct,
            "workers": config.workers,
            "primary_per_worker": [int((primary == w).sum()) for w in range(config.workers)],
            "positions_per_worker": [int(len(a)) for a in assignments],
            "max_copies": int(copies_per_token.max()),
        },
        "aggregate": aggregate,
        "worst_case": worst_case,
        "layers": layers_payload,
        "timings": {
            "upload_s": upload_s,
            "benchmark_s": benchmark_s,
            "capture_s": store.meta.get("capture_timing_s"),
            "stage_ms": {key: round(value, 3) for key, value in stage_ms.items()},
        },
        "cluster": cluster_summary,
        "model": {
            "model_id": store.meta.get("model_id"),
            "revision": store.meta.get("revision"),
            "model_path": store.meta.get("model_path"),
        },
    }
    return payload
