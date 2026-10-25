"""Pure numerical core for partitioned attention and certified pruning.

All functions operate on float64 NumPy arrays. They contain no HTTP or model code
so the equivalence and bound properties can be tested directly.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

_UINT32 = 0xFFFFFFFF
_EXP_LIMIT = 700.0


def fmix32(key: int, seed: int = 0x977F23A7) -> int:
    """MurmurHash3 32-bit finaliser, bit-identical to the JS and prior Python versions."""
    h = (int(key) ^ int(seed)) & _UINT32
    h = ((h ^ (h >> 16)) * 0x85EBCA6B) & _UINT32
    h = ((h ^ (h >> 13)) * 0xC2B2AE35) & _UINT32
    return (h ^ (h >> 16)) & _UINT32


def _validate_partition(n_tokens: int, workers: int, overlap_pct: float, strategy: str) -> None:
    if n_tokens < 1:
        raise ValueError("n_tokens must be positive")
    if workers < 1:
        raise ValueError("workers must be positive")
    if not 0.0 <= overlap_pct <= 100.0:
        raise ValueError("overlap_pct must be in [0, 100]")
    if strategy not in {"pseudorandom", "contiguous"}:
        raise ValueError("strategy must be 'pseudorandom' or 'contiguous'")
    if strategy == "contiguous" and n_tokens < workers:
        raise ValueError("contiguous partitioning requires n_tokens >= workers")


def primary_worker(n_tokens: int, workers: int, strategy: str, seed: int) -> np.ndarray:
    """Owning worker for every token position, independent of overlap."""
    _validate_partition(n_tokens, workers, 0.0, strategy)
    if strategy == "pseudorandom":
        return np.asarray(
            [fmix32(token, seed) % workers for token in range(n_tokens)], dtype=np.int32
        )
    chunk = max(1, n_tokens // workers)
    return np.asarray(
        [min(token // chunk, workers - 1) for token in range(n_tokens)], dtype=np.int32
    )


def partition_tokens(
    n_tokens: int,
    workers: int,
    strategy: str,
    overlap_pct: float,
    seed: int,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Assign each token to its primary worker, plus optional secondary copies.

    Multiplicities are per position, never per vocabulary id: a prompt may repeat
    the same token at different positions and those are distinct softmax entries.
    """
    _validate_partition(n_tokens, workers, overlap_pct, strategy)
    primary = primary_worker(n_tokens, workers, strategy, seed)
    assignments: List[List[int]] = [[] for _ in range(workers)]
    counts = np.zeros(n_tokens, dtype=np.int32)
    rng = np.random.default_rng(int(seed) + 1)

    if strategy == "pseudorandom":
        for token in range(n_tokens):
            owner = int(primary[token])
            assignments[owner].append(token)
            counts[token] += 1
            if overlap_pct > 0.0 and rng.random() < overlap_pct / 100.0:
                second = fmix32(token, seed + 0x1337) % workers
                if second == owner:
                    second = (owner + 1) % workers
                assignments[second].append(token)
                counts[token] += 1
    else:
        chunk = max(1, n_tokens // workers)
        margin = int(chunk * overlap_pct / 100.0)
        for worker in range(workers):
            start = worker * chunk
            end = n_tokens if worker == workers - 1 else (worker + 1) * chunk
            left = max(0, start - margin // 2)
            right = min(n_tokens, end + (margin + 1) // 2)
            for token in range(left, right):
                assignments[worker].append(token)
                counts[token] += 1

    if not np.all(counts >= 1):
        raise AssertionError("partition left tokens uncovered")
    if not np.array_equal(primary_worker(n_tokens, workers, strategy, seed), primary):
        raise AssertionError("primary assignment is not deterministic")
    return [np.asarray(sorted(a), dtype=np.int64) for a in assignments], counts


def retained_multiplicities(
    assignments: Sequence[np.ndarray], retained: Sequence[int]
) -> Dict[str, int]:
    """Copies per position across the retained workers, for correction."""
    retained_set = set(int(x) for x in retained)
    counts: Dict[str, int] = {}
    for worker in retained_set:
        for position in assignments[worker]:
            key = str(int(position))
            counts[key] = counts.get(key, 0) + 1
    return {key: value for key, value in counts.items() if value > 1}


def block_bound(
    q: np.ndarray,
    keys: np.ndarray,
    scale: float,
    centroid: Optional[np.ndarray] = None,
    radius: Optional[float] = None,
) -> float:
    """Cauchy-Schwarz upper bound on q.k/sqrt(d) for every key in a block.

    Q must not be normalised: the bound and the scores both use the raw query.
    """
    if keys.shape[0] == 0:
        return float("-inf")
    if centroid is None:
        centroid = keys.mean(axis=0)
    if radius is None:
        diffs = keys - centroid
        radius = float(np.sqrt(np.max(np.sum(diffs * diffs, axis=1))))
    q_norm = float(np.linalg.norm(q))
    return float((np.dot(q, centroid) + q_norm * radius) * scale)


def token_log_correction(
    positions: np.ndarray, multiplicities: Optional[Dict[str, int]]
) -> np.ndarray:
    """log(c_j) for every shard row; zero when a position has one retained copy."""
    correction = np.zeros(len(positions), dtype=np.float64)
    if multiplicities:
        for row, position in enumerate(positions):
            count = int(multiplicities.get(str(int(position)), 1))
            if count > 1:
                correction[row] = math.log(count)
    return correction


def local_scores(
    q: np.ndarray,
    keys: np.ndarray,
    positions: np.ndarray,
    multiplicities: Optional[Dict[str, int]],
    scale: float,
) -> np.ndarray:
    """Raw attention scores with the multiplicity correction applied."""
    scores = (keys @ q) * scale
    return scores - token_log_correction(positions, multiplicities)


def block_bounds_batch(
    q: np.ndarray,
    centroid: np.ndarray,
    radius: float,
    scale: float,
) -> np.ndarray:
    """Per-query-head Cauchy-Schwarz bounds for one worker block.

    q is (q_per_kv, head_dim) and must be unnormalised.
    """
    q = np.asarray(q, dtype=np.float64)
    q_norm = np.linalg.norm(q, axis=1)
    return (q @ np.asarray(centroid, dtype=np.float64) + q_norm * float(radius)) * float(scale)


def local_stats_batch(
    q: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    positions: np.ndarray,
    multiplicities: Optional[Dict[str, int]],
    scale: float,
) -> Dict[str, object]:
    """Local softmax statistics and output for every query head in a KV group."""
    q = np.asarray(q, dtype=np.float64)
    if keys.shape[0] == 0:
        heads = q.shape[0]
        return {
            "lmax": np.full(heads, -np.inf),
            "sumexp": np.zeros(heads),
            "local_out": np.zeros((heads, values.shape[1])),
            "token_count": 0,
        }
    scores = (q @ keys.T) * float(scale)
    scores -= token_log_correction(positions, multiplicities)[None, :]
    lmax = scores.max(axis=1)
    exp_scores = np.exp(scores - lmax[:, None])
    sumexp = exp_scores.sum(axis=1)
    local_out = (exp_scores @ values) / sumexp[:, None]
    return {
        "lmax": lmax,
        "sumexp": sumexp,
        "local_out": local_out,
        "token_count": int(keys.shape[0]),
    }


def causal_local_stats_batch(
    q: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    key_positions: np.ndarray,
    query_positions: np.ndarray,
    scale: float,
) -> Dict[str, np.ndarray]:
    """Local causal softmax statistics for a batch of query positions.

    Every query attends only to stored keys at positions <= its own position.
    Queries with no visible keys return lmax=-inf, sumexp=0 and a zero output so
    the cross-worker merge can ignore them without special cases.
    """
    q = np.asarray(q, dtype=np.float64)
    keys = np.asarray(keys, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    key_positions = np.asarray(key_positions, dtype=np.int64)
    query_positions = np.asarray(query_positions, dtype=np.int64)
    if q.ndim != 3:
        raise ValueError("q must have shape (queries, q_heads, head_dim)")
    if keys.ndim != 2 or values.shape != keys.shape:
        raise ValueError("keys and values must be matching matrices")
    if q.shape[2] != keys.shape[1]:
        raise ValueError("query and key head dimensions differ")
    if len(key_positions) != keys.shape[0] or len(query_positions) != q.shape[0]:
        raise ValueError("position arrays do not match the tensor rows")
    if len(key_positions) > 0 and np.any(np.diff(key_positions) <= 0):
        raise ValueError("key positions must be sorted and unique")
    if len(query_positions) > 0 and np.any(np.diff(query_positions) <= 0):
        raise ValueError("query positions must be sorted and unique")

    heads = q.shape[1]
    head_dim = q.shape[2]
    if len(query_positions) == 0 or len(key_positions) == 0:
        return {
            "lmax": np.full((len(query_positions), heads), -np.inf),
            "sumexp": np.zeros((len(query_positions), heads)),
            "local_out": np.zeros((len(query_positions), heads, head_dim)),
            "visible_keys": np.zeros(len(query_positions), dtype=np.int64),
        }

    visible = key_positions[None, :] <= query_positions[:, None]
    scores = np.einsum("phd,kd->phk", q, keys) * float(scale)
    scores = np.where(visible[:, None, :], scores, -np.inf)
    lmax = scores.max(axis=2)
    finite = np.isfinite(lmax)
    safe_lmax = np.where(finite, lmax, 0.0)
    exp_scores = np.exp(scores - safe_lmax[:, :, None])
    exp_scores = np.where(visible[:, None, :], exp_scores, 0.0)
    sumexp = exp_scores.sum(axis=2)
    total = np.where(sumexp > 0.0, sumexp, 1.0)
    local_out = np.einsum("phk,kd->phd", exp_scores, values) / total[:, :, None]
    local_out = np.where((sumexp > 0.0)[:, :, None], local_out, 0.0)
    return {
        "lmax": lmax,
        "sumexp": sumexp,
        "local_out": local_out,
        "visible_keys": visible.sum(axis=1),
    }


def merge_causal_stats(
    lmaxes: np.ndarray,
    sumexps: np.ndarray,
    outputs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Merge causal worker statistics across workers for every query position.

    lmaxes and sumexps are (workers, queries, heads); outputs are
    (workers, queries, heads, head_dim). Workers that own no visible key report
    lmax=-inf and contribute nothing.
    """
    lmaxes = np.asarray(lmaxes, dtype=np.float64)
    sumexps = np.asarray(sumexps, dtype=np.float64)
    outputs = np.asarray(outputs, dtype=np.float64)
    if lmaxes.ndim != 3 or lmaxes.shape != sumexps.shape:
        raise ValueError("worker statistics must have shape (workers, queries, heads)")
    if outputs.ndim != 4 or outputs.shape[:3] != lmaxes.shape:
        raise ValueError("worker outputs do not match the statistic shapes")
    if lmaxes.shape[0] == 0:
        raise ValueError("cannot merge an empty worker set")

    global_max = lmaxes.max(axis=0)
    safe_max = np.where(np.isfinite(global_max), global_max, 0.0)
    weights = np.where(
        np.isfinite(lmaxes), sumexps * np.exp(lmaxes - safe_max[None, :, :]), 0.0
    )
    total = weights.sum(axis=0)
    safe_total = np.where(total > 0.0, total, 1.0)
    merged = np.einsum("wph,wphd->phd", weights, outputs) / safe_total[:, :, None]
    merged = np.where((total > 0.0)[:, :, None], merged, 0.0)
    return merged, total


def merge_worker_stats(
    lmaxes: np.ndarray,
    sumexps: np.ndarray,
    outputs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge (workers, heads) statistics into outputs, retained totals, and maxima."""
    lmaxes = np.asarray(lmaxes, dtype=np.float64)
    sumexps = np.asarray(sumexps, dtype=np.float64)
    outputs = np.asarray(outputs, dtype=np.float64)
    if lmaxes.shape != sumexps.shape or outputs.shape[:2] != lmaxes.shape:
        raise ValueError("inconsistent worker statistic shapes")
    if lmaxes.size == 0:
        raise ValueError("cannot merge an empty worker set")
    global_max = lmaxes.max(axis=0)
    weights = sumexps * np.exp(lmaxes - global_max[None, :])
    total = weights.sum(axis=0)
    safe_total = np.where(total > 0.0, total, 1.0)
    merged = np.einsum("wh,whd->hd", weights, outputs) / safe_total[:, None]
    return np.where(total[:, None] > 0.0, merged, 0.0), total, global_max


def merge_stats_batch(
    lmaxes: np.ndarray,
    sumexps: np.ndarray,
    outputs: np.ndarray,
) -> np.ndarray:
    """Merge (workers, heads) statistics into (heads, head_dim) outputs."""
    merged, _, _ = merge_worker_stats(lmaxes, sumexps, outputs)
    return merged


def local_stats(
    q: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    token_ids: np.ndarray,
    multiplicities: Optional[Dict[str, int]],
    scale: float,
    need_output: bool = True,
) -> Dict[str, object]:
    if keys.shape[0] == 0:
        return {
            "lmax": float("-inf"),
            "sumexp": 0.0,
            "local_out": [0.0] * int(values.shape[1]) if need_output else None,
            "token_count": 0,
        }
    scores = local_scores(q, keys, token_ids, multiplicities, scale)
    lmax = float(np.max(scores))
    exp_scores = np.exp(scores - lmax)
    sumexp = float(np.sum(exp_scores))
    result: Dict[str, object] = {
        "lmax": lmax,
        "sumexp": sumexp,
        "token_count": int(keys.shape[0]),
    }
    if need_output:
        weights = exp_scores / sumexp
        result["local_out"] = (weights @ values).astype(np.float64).tolist()
    return result


def merge_stats(
    lmaxes: Sequence[float],
    sumexps: Sequence[float],
    outputs: Optional[Sequence[Sequence[float]]] = None,
) -> Tuple[float, float, Optional[np.ndarray]]:
    """Stable log-sum-exp merge of per-worker denominators and outputs."""
    if len(lmaxes) != len(sumexps):
        raise ValueError("lmaxes and sumexps must have equal length")
    if not lmaxes:
        raise ValueError("cannot merge an empty worker set")
    global_max = max(float(x) for x in lmaxes)
    weights = np.asarray(
        [float(s) * math.exp(float(m) - global_max) for m, s in zip(lmaxes, sumexps)],
        dtype=np.float64,
    )
    total = float(weights.sum())
    if total <= 0.0:
        if outputs is None:
            return global_max, 0.0, None
        return global_max, 0.0, np.zeros(len(outputs[0]), dtype=np.float64)
    if outputs is None:
        return global_max, total, None
    merged = sum(
        (weights[i] / total) * np.asarray(outputs[i], dtype=np.float64)
        for i in range(len(outputs))
    )
    return global_max, total, np.asarray(merged, dtype=np.float64)


def safe_exp(x: float) -> float:
    if x == float("-inf"):
        return 0.0
    if x > _EXP_LIMIT:
        return float("inf")
    return math.exp(x)


def omitted_mass_bound(
    z_retained: float,
    max_retained_score: float,
    pruned_unique_ubs: Dict[int, float],
    safe_gamma: float,
) -> float:
    """Upper bound on omitted softmax mass over tokens with zero retained copies.

    Retained copies of a token are corrected by log(number of retained copies),
    so those copies reconstruct the full token mass. Only tokens with no retained
    copy contribute omitted mass.
    """
    if not pruned_unique_ubs:
        return 0.0
    z_pruned = 0.0
    for token in sorted(pruned_unique_ubs):
        z_pruned += safe_exp(float(pruned_unique_ubs[token]) - max_retained_score)
    z_pruned *= safe_gamma
    denominator = z_retained + z_pruned
    if not math.isfinite(denominator) or denominator <= 0.0:
        return 1.0 if z_pruned > 0.0 else 0.0
    return min(1.0, z_pruned / denominator)


def certified_error_bound(omitted_mass: float, max_value_norm: float) -> float:
    return 2.0 * float(max_value_norm) * float(omitted_mass)


def reference_attention(
    q: np.ndarray, keys: np.ndarray, values: np.ndarray, scale: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Dense causal-free attention for the final query, matching the model."""
    scores = (keys @ q) * scale
    scores = scores - np.max(scores)
    probs = np.exp(scores)
    probs /= np.sum(probs)
    return probs, (probs @ values).astype(np.float64)
