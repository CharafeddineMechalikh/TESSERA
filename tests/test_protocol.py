"""Properties the benchmark relies on: coverage, soundness, and exact merges."""
from __future__ import annotations

import math

import numpy as np
import pytest

from distributed_llm.protocol import (
    block_bound,
    block_bounds_batch,
    causal_local_stats_batch,
    certified_error_bound,
    fmix32,
    local_stats_batch,
    merge_causal_stats,
    merge_stats_batch,
    omitted_mass_bound,
    partition_tokens,
    primary_worker,
    reference_attention,
    retained_multiplicities,
    token_log_correction,
)


def test_fmix32_is_stable():
    assert fmix32(0) == 2436065507
    assert fmix32(1) == 859193535
    assert all(0 <= fmix32(x) < 2**32 for x in range(1000))


@pytest.mark.parametrize("strategy", ["pseudorandom", "contiguous"])
@pytest.mark.parametrize("overlap", [0.0, 10.0, 35.0])
def test_partition_covers_every_token_once_as_primary(strategy, overlap):
    n, workers, seed = 137, 5, 42
    assignments, counts = partition_tokens(n, workers, strategy, overlap, seed)
    owner = primary_worker(n, workers, strategy, seed)
    assert np.all(counts >= 1)
    for worker, positions in enumerate(assignments):
        assert np.all(np.diff(positions) > 0)
        assert np.all(owner[positions] >= 0)
    primary_rows = []
    for worker, positions in enumerate(assignments):
        primary_rows.extend(int(p) for p in positions if int(owner[p]) == worker)
    assert sorted(primary_rows) == list(range(n))
    if strategy == "contiguous":
        assert len(assignments) == workers


def test_retained_multiplicities_counts_positions():
    assignments = [np.array([0, 1, 4]), np.array([1, 2, 3])]
    counts = retained_multiplicities(assignments, [0, 1])
    assert counts == {"1": 2}
    assert retained_multiplicities(assignments, [0]) == {}
    assert retained_multiplicities(assignments, [1]) == {}


def test_token_correction_uses_position_not_vocabulary_id():
    positions = np.array([3, 7])
    multiplicities = {"3": 2, "7": 1}
    correction = token_log_correction(positions, multiplicities)
    assert correction[0] == pytest.approx(math.log(2.0))
    assert correction[1] == 0.0


def test_block_bound_is_sound():
    rng = np.random.default_rng(0)
    for _ in range(50):
        keys = rng.normal(size=(37, 16)) * rng.uniform(0.1, 10.0)
        q = rng.normal(size=16) * rng.uniform(0.1, 10.0)
        scale = 1.0 / math.sqrt(16)
        centroid = keys.mean(axis=0)
        diffs = keys - centroid
        radius = float(np.sqrt(np.max(np.sum(diffs * diffs, axis=1))))
        bound = block_bound(q, keys, scale)
        assert bound + 1e-12 >= float(np.max((keys @ q) * scale))
        batch = block_bounds_batch(q[None, :], centroid, radius, scale)[0]
        assert batch + 1e-12 >= float(np.max((keys @ q) * scale))


def test_merge_matches_dense_attention_with_duplicates():
    rng = np.random.default_rng(1)
    n, dim, heads = 41, 16, 3
    keys = rng.normal(size=(n, dim))
    values = rng.normal(size=(n, dim))
    q = rng.normal(size=(heads, dim))
    scale = 1.0 / math.sqrt(dim)

    assignments = [np.arange(0, n, 3), np.arange(1, n, 3), np.arange(2, n, 3)]
    assignments[1] = np.union1d(assignments[1], [4, 10])
    multiplicities = retained_multiplicities(assignments, [0, 1, 2])

    lmaxes, sumexps, outs = [], [], []
    for positions in assignments:
        stats = local_stats_batch(
            q, keys[positions], values[positions], positions, multiplicities, scale
        )
        lmaxes.append(stats["lmax"])
        sumexps.append(stats["sumexp"])
        outs.append(stats["local_out"])
    merged = merge_stats_batch(np.asarray(lmaxes), np.asarray(sumexps), np.asarray(outs))

    for head in range(heads):
        _, expected = reference_attention(q[head], keys, values, scale)
        assert np.abs(merged[head] - expected).max() < 1e-13


def test_merge_is_stable_with_large_offsets():
    lmaxes = np.array([[1000.0, -800.0], [1000.5, -800.0]])
    sumexps = np.array([[1.0, 0.0], [2.0, 0.0]])
    outs = np.array([[[1.0, 2.0], [0.0, 0.0]], [[3.0, 4.0], [0.0, 0.0]]])
    merged = merge_stats_batch(lmaxes, sumexps, outs)
    w0 = math.exp(0.0) * 1.0
    w1 = math.exp(0.5) * 2.0
    expected = (w0 * np.array([1.0, 2.0]) + w1 * np.array([3.0, 4.0])) / (w0 + w1)
    assert np.abs(merged[0] - expected).max() < 1e-15
    assert np.all(merged[1] == 0.0)


def test_omitted_mass_and_certified_error_are_upper_bounds():
    rng = np.random.default_rng(2)
    n, dim = 64, 16
    keys = rng.normal(size=(n, dim))
    values = rng.normal(size=(n, dim))
    q = rng.normal(size=dim)
    scale = 1.0 / math.sqrt(dim)

    retained = np.array([0, 2, 5, 6, 9, 12, 13, 20, 21, 30, 31, 40, 41, 50, 51, 60, 61])
    pruned = np.setdiff1d(np.arange(n), retained)
    scores = (keys @ q) * scale

    full_probs, full_out = reference_attention(q, keys, values, scale)
    retained_scores = scores[retained]
    retained_probs = np.exp(retained_scores - retained_scores.max())
    retained_probs /= retained_probs.sum()
    retained_out = retained_probs @ values[retained]
    actual_error = float(np.linalg.norm(full_out - retained_out))
    true_omitted_mass = float(full_probs[pruned].sum())

    retained_max = float(retained_scores.max())
    z_retained = float(np.exp(retained_scores - retained_max).sum())
    block_ub = block_bound(q, keys[pruned], scale)
    z_pruned = sum(math.exp(block_ub - retained_max) for _ in pruned)
    p_bound = omitted_mass_bound(z_retained, retained_max, {int(p): block_ub for p in pruned}, 1.05)
    assert p_bound + 1e-12 >= true_omitted_mass
    v_max = float(np.max(np.linalg.norm(values, axis=1)))
    certified = certified_error_bound(p_bound, v_max)
    assert certified + 1e-12 >= actual_error


def _dense_causal(q, keys, values, key_positions, query_positions, scale):
    outputs = np.zeros((len(query_positions), q.shape[1], q.shape[2]))
    for row, position in enumerate(query_positions):
        visible = np.flatnonzero(np.asarray(key_positions) <= position)
        if len(visible) == 0:
            continue
        scores = (keys[visible] @ q[row].T) * scale
        scores -= scores.max(axis=0, keepdims=True)
        probs = np.exp(scores)
        probs /= probs.sum(axis=0, keepdims=True)
        outputs[row] = probs.T @ values[visible]
    return outputs


@pytest.mark.parametrize("n_queries", [1, 7, 23])
def test_causal_worker_merge_matches_dense_attention(n_queries):
    rng = np.random.default_rng(7)
    n, dim, heads = 29, 12, 4
    keys = rng.normal(size=(n, dim))
    values = rng.normal(size=(n, dim))
    query_positions = np.arange(n - n_queries, n)
    q = rng.normal(size=(len(query_positions), heads, dim))
    scale = 1.0 / math.sqrt(dim)
    key_positions = np.arange(n)

    workers = 3
    lmaxes, sumexps, outs, visible_total = [], [], [], np.zeros(len(query_positions), dtype=np.int64)
    for worker in range(workers):
        owned = key_positions[key_positions % workers == worker]
        stats = causal_local_stats_batch(
            q, keys[owned], values[owned], owned, query_positions, scale
        )
        lmaxes.append(stats["lmax"])
        sumexps.append(stats["sumexp"])
        outs.append(stats["local_out"])
        visible_total += stats["visible_keys"]

    merged, totals = merge_causal_stats(
        np.asarray(lmaxes), np.asarray(sumexps), np.asarray(outs)
    )
    expected = _dense_causal(q, keys, values, key_positions, query_positions, scale)
    assert np.abs(merged - expected).max() < 1e-13
    assert np.all(totals > 0.0)
    assert np.all(visible_total == query_positions + 1)


def test_causal_worker_without_visible_keys_is_ignored():
    rng = np.random.default_rng(11)
    dim = 8
    q = rng.normal(size=(1, 2, dim))
    keys = rng.normal(size=(3, dim))
    values = rng.normal(size=(3, dim))
    stats = causal_local_stats_batch(
        q, keys, values, np.array([5, 6, 7]), np.array([2]), 1.0 / math.sqrt(dim)
    )
    assert np.all(stats["lmax"] == -np.inf)
    assert np.all(stats["sumexp"] == 0.0)
    assert np.all(stats["local_out"] == 0.0)
    assert np.all(stats["visible_keys"] == 0)

    good = causal_local_stats_batch(
        q, keys, values, np.array([0, 1, 2]), np.array([2]), 1.0 / math.sqrt(dim)
    )
    merged, totals = merge_causal_stats(
        np.stack([stats["lmax"], good["lmax"]]),
        np.stack([stats["sumexp"], good["sumexp"]]),
        np.stack([stats["local_out"], good["local_out"]]),
    )
    assert np.allclose(merged, good["local_out"])
    assert np.all(totals > 0.0)

