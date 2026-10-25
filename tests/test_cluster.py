"""Real process lifecycle: spawn, upload, query, tear down."""
from __future__ import annotations

import io
import socket

import numpy as np
import pytest

from distributed_llm.cluster import ClusterError, WorkerCluster
from distributed_llm.protocol import merge_stats_batch, reference_attention

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _skip_ram_admission(monkeypatch):
    """Keep integration tests deterministic while pytest itself holds torch memory."""
    monkeypatch.setattr("distributed_llm.cluster.SPAWN_HEADROOM_GB", 0.0)
    monkeypatch.setattr("distributed_llm.cluster.ESTIMATED_WORKER_RSS_MB", 0.0)


def _shard_bytes(tokens, keys, values, layer=0, kv=0):
    buffer = io.BytesIO()
    np.savez(
        buffer,
        tokens=np.asarray(tokens, dtype=np.int64),
        primary=np.ones(len(tokens), dtype=bool),
        **{
            f"k_{layer}_{kv}": keys,
            f"v_{layer}_{kv}": values,
        },
    )
    return buffer.getvalue()


def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def test_cluster_lifecycle_and_distributed_merge(tmp_path):
    cluster = WorkerCluster(num_workers=2, run_dir=tmp_path)
    cluster.start()
    ports = [handle.port for handle in cluster.workers]
    try:
        assert len(set(ports)) == 2
        for port in ports:
            assert _port_open(port)

        rng = np.random.default_rng(3)
        n, dim, q_per_kv = 16, 8, 2
        keys = rng.normal(size=(n, dim))
        values = rng.normal(size=(n, dim))
        positions = {0: np.arange(0, n, 2), 1: np.arange(1, n, 2)}
        shards = [
            _shard_bytes(positions[w], keys[positions[w]], values[positions[w]])
            for w in range(2)
        ]
        replies = cluster.upload_shards(shards, layers=[0], kv_heads=[0], q_per_kv=q_per_kv)
        assert [reply["token_count"] for reply in replies] == [8, 8]

        q = rng.normal(size=(q_per_kv, dim))
        bounds = cluster.bounds(0, 0, q)
        dense = keys @ q.T / np.sqrt(dim)
        assert np.all(bounds >= dense.max(axis=0) - 1e-12)

        lmax, sumexp, outs, meta = cluster.stats(0, 0, q, {}, "assigned", [0, 1])
        assert lmax.shape == (2, q_per_kv)
        assert len(meta) == 2
        merged = merge_stats_batch(lmax, sumexp, outs)
        for head in range(q_per_kv):
            _, expected = reference_attention(q[head], keys, values, 1.0 / np.sqrt(dim))
            assert np.abs(merged[head] - expected).max() < 1e-12

        summary = cluster.summary()
        assert summary["requests"] >= 5
        assert summary["failures"] == 0
    finally:
        cluster.shutdown()

    assert cluster.workers == []
    for port in ports:
        assert not _port_open(port)


def test_generation_admission_charges_the_strided_kv_total_once(tmp_path, monkeypatch):
    # Ownership is strided, so the whole cluster holds one copy of every position.
    # Charging num_workers * kv_total refused runs that actually fit.
    from distributed_llm.cluster import KV_GROWTH_FACTOR

    max_positions = 1000
    layers, kv_heads, head_dim = 28, 2, 128
    expected_kv_mb = max_positions * layers * kv_heads * head_dim * 4 * 2 / 1e6
    kv_gb = expected_kv_mb * 1e6 * KV_GROWTH_FACTOR / 1024.0**3
    needed_gb = 0.7 + kv_gb

    # The host's free RAM is an input to the check, not the behaviour under
    # test, and this machine frequently sits below the 0.77 GB it asks for.
    monkeypatch.setattr("distributed_llm.cluster.available_ram_gb", lambda: 8.0)

    cluster = WorkerCluster(num_workers=2, run_dir=tmp_path)
    cluster.start()
    try:
        admission = cluster.preflight_generation(
            max_positions=max_positions,
            num_layers=layers,
            num_kv_heads=kv_heads,
            head_dim=head_dim,
            process_headroom_gb=0.7,
        )
        assert admission["free_ram_gb"] == pytest.approx(8.0, abs=1e-3)
        assert admission["kv_total_mb"] == pytest.approx(expected_kv_mb, rel=1e-6)
        assert admission["kv_bytes_per_worker"] == pytest.approx(
            expected_kv_mb * 1e6 / 2, rel=1e-6
        )
        assert admission["needed_ram_gb"] == pytest.approx(needed_gb, rel=1e-3)

        short = round(needed_gb - 0.01, 2)
        monkeypatch.setattr("distributed_llm.cluster.available_ram_gb", lambda: short)
        with pytest.raises(ClusterError) as excinfo:
            cluster.preflight_generation(
                max_positions=max_positions,
                num_layers=layers,
                num_kv_heads=kv_heads,
                head_dim=head_dim,
                process_headroom_gb=0.7,
            )
        message = str(excinfo.value)
        assert f"KV {kv_gb:.2f} GB across 2 workers, decoder 0.70 GB" in message
        assert f"only {short:.2f} GB is available" in message
    finally:
        cluster.shutdown()
