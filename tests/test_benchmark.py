"""End-to-end benchmark orchestration on a synthetic capture.

The capture mimics the real artifact schema but uses small random tensors, so
the test exercises partitioning, shard upload, bounds, incremental selection,
the overlap requery path, and the certificate without the 1.5B model.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from distributed_llm.benchmark import BenchmarkConfig, CaptureStore, run_benchmark
from distributed_llm.cluster import WorkerCluster

pytestmark = pytest.mark.integration

N_TOKENS = 64


@pytest.fixture(autouse=True)
def _skip_ram_admission(monkeypatch):
    """Keep integration tests deterministic while pytest itself holds torch memory."""
    monkeypatch.setattr("distributed_llm.cluster.SPAWN_HEADROOM_GB", 0.0)
    monkeypatch.setattr("distributed_llm.cluster.ESTIMATED_WORKER_RSS_MB", 0.0)
NUM_LAYERS = 2
NUM_KV = 2
Q_PER_KV = 2
HEAD_DIM = 8


def _make_capture(tmp_path, seed=7):
    rng = np.random.default_rng(seed)
    geometry = {
        "model_id": "synthetic/tiny",
        "revision": "test",
        "num_layers": NUM_LAYERS,
        "num_attention_heads": NUM_KV * Q_PER_KV,
        "num_key_value_heads": NUM_KV,
        "head_dim": HEAD_DIM,
        "hidden_size": NUM_KV * Q_PER_KV * HEAD_DIM,
        "rope_theta": 1_000_000.0,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 32768,
        "vocab_size": 1000,
    }
    layers = list(range(NUM_LAYERS))
    kv_heads = list(range(NUM_KV))
    arrays = {"tokens": np.arange(N_TOKENS, dtype=np.int64)}
    for layer in layers:
        for kv in kv_heads:
            keys = rng.normal(size=(N_TOKENS, HEAD_DIM))
            values = rng.normal(size=(N_TOKENS, HEAD_DIM)) * 0.1
            query = rng.normal(size=(Q_PER_KV, HEAD_DIM))
            # One strongly preferred token per head so the certificate has
            # something to separate even on random data.
            keys[layer * 3 + kv] *= 6.0
            arrays[f"k_{layer}_{kv}"] = keys
            arrays[f"v_{layer}_{kv}"] = values
            arrays[f"q_{layer}_{kv}"] = query
    npz_path = tmp_path / "capture.npz"
    with open(npz_path, "wb") as handle:
        np.savez(handle, **arrays)
    meta = {
        "geometry": geometry,
        "tokens": list(range(N_TOKENS)),
        "layers": layers,
        "kv_heads": kv_heads,
        "prompt_sha256": "synthetic",
        "tokenizer_class": "Synthetic",
        "truncated": False,
        "original_token_count": N_TOKENS,
        "capture_timing_s": 0.0,
        "model_id": geometry["model_id"],
        "revision": geometry["revision"],
        "model_path": "synthetic",
    }
    meta_path = tmp_path / "capture.json"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return CaptureStore(npz_path, meta_path)


def _run(tmp_path, config, threshold):
    store = _make_capture(tmp_path)
    cluster = WorkerCluster(num_workers=config.workers, run_dir=tmp_path)
    cluster.start()
    try:
        payload = run_benchmark(store, cluster, config, run_id="test")
    finally:
        cluster.shutdown()
        store.close()
    for layer in payload["layers"]:
        for row in layer["kv_heads"]:
            actual = np.asarray(row["actual_l2_error"])
            certified = np.asarray(row["certified_error"])
            assert np.all(actual <= certified + 1e-6)
            assert row["disjoint_control_rel_error"] < 1e-8
            assert row["tolerance_satisfied"]
            assert len(row["retention_curve"]) >= 1
            assert row["retention_curve"][-1]["retained_workers"] == row["retained_worker_count"]
            assert len(row["retention_curve"]) <= config.workers
    assert payload["aggregate"]["max_certified_error"] <= threshold
    assert payload["config"]["selection_protocol"] in {"incremental", "requery"}
    return payload


def test_benchmark_exact_when_tolerance_is_strict(tmp_path):
    config = BenchmarkConfig(workers=2, overlap_pct=0.0, error_tol=1e-9)
    payload = _run(tmp_path, config, 1e-9)
    assert payload["aggregate"]["mean_retained_workers"] == 2.0
    assert payload["config"]["selection_protocol"] == "incremental"
    assert payload["aggregate"]["max_relative_error"] < 1e-10
    assert payload["aggregate"]["max_certified_error"] == 0.0
    assert payload["aggregate"]["max_disjoint_control_rel_error"] < 1e-10
    assert payload["timings"]["upload_s"] > 0.0


def test_benchmark_selects_one_worker_at_loose_tolerance(tmp_path):
    config = BenchmarkConfig(workers=2, overlap_pct=0.0, error_tol=1.0)
    payload = _run(tmp_path, config, 1.0)
    assert payload["aggregate"]["mean_retained_workers"] == 1.0
    assert payload["aggregate"]["mean_retained_fraction"] < 1.0
    curve = payload["layers"][0]["kv_heads"][0]["retention_curve"]
    assert curve[0]["retained_workers"] == 1
    assert curve[0]["tolerance_satisfied"]


def test_benchmark_overlap_requery_is_exact_when_all_retained(tmp_path):
    config = BenchmarkConfig(workers=2, overlap_pct=25.0, error_tol=1e-9)
    payload = _run(tmp_path, config, 1e-9)
    assert payload["config"]["selection_protocol"] == "requery"
    assert payload["aggregate"]["mean_retained_workers"] == 2.0
    assert payload["aggregate"]["mean_copies_per_token"] > 1.0
    assert payload["aggregate"]["max_relative_error"] < 1e-10
