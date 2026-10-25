"""Worker contract: authentication, isolation, bounds, and merged statistics."""
from __future__ import annotations

import io
from dataclasses import replace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from distributed_llm.config import LIMITS
from distributed_llm.protocol import merge_stats_batch, reference_attention
from distributed_llm.worker_node import create_app

TOKEN = "test-token-0123456789"
SESSION = "session-abcdef12"


def _shard_bytes(
    tokens,
    primary,
    keys,
    values,
    layers,
    kv_heads,
):
    payload = {
        "tokens": np.asarray(tokens, dtype=np.int64),
        "primary": np.asarray(primary, dtype=bool),
    }
    for layer in layers:
        for kv in kv_heads:
            payload[f"k_{layer}_{kv}"] = keys[(layer, kv)]
            payload[f"v_{layer}_{kv}"] = values[(layer, kv)]
    buffer = io.BytesIO()
    np.savez(buffer, **payload)
    return buffer.getvalue()


def _make_shard(rng, positions, dim=8, layers=(0, 1), kv_heads=(0,), q_per_kv=2):
    positions = np.asarray(positions, dtype=np.int64)
    keys = {
        (layer, kv): rng.normal(size=(len(positions), dim))
        for layer in layers
        for kv in kv_heads
    }
    values = {
        (layer, kv): rng.normal(size=(len(positions), dim))
        for layer in layers
        for kv in kv_heads
    }
    return positions, keys, values


@pytest.fixture()
def client():
    app = create_app(TOKEN, SESSION, 0)
    return TestClient(app)


def _auth():
    return {"Authorization": f"Bearer {TOKEN}"}


def _upload(
    client,
    positions,
    keys,
    values,
    layers=(0, 1),
    kv_heads=(0,),
    primary=None,
    q_per_kv=2,
    worker_id=0,
):
    if primary is None:
        primary = np.ones(len(positions), dtype=bool)
    body = _shard_bytes(positions, primary, keys, values, layers, kv_heads)
    return client.post(
        "/v1/shard",
        params={
            "worker_id": worker_id,
            "session_id": SESSION,
            "layers": ",".join(map(str, layers)),
            "kv_heads": ",".join(map(str, kv_heads)),
            "q_per_kv": q_per_kv,
        },
        content=body,
        headers=_auth(),
    )


def test_health_is_public_status_is_authenticated(client):
    assert client.get("/health").status_code == 200
    assert client.get("/v1/status").status_code == 401
    response = client.get("/v1/status", headers=_auth())
    assert response.status_code == 200
    assert response.json()["worker_id"] == 0


def test_upload_and_bounds(client):
    rng = np.random.default_rng(0)
    positions, keys, values = _make_shard(rng, range(12))
    response = _upload(client, positions, keys, values)
    assert response.status_code == 200, response.text
    assert response.json()["token_count"] == 12

    q = rng.normal(size=(2, 8))
    response = client.post(
        "/v1/bounds",
        json={"session_id": SESSION, "layer": 0, "kv_head": 0, "q": q.tolist()},
        headers=_auth(),
    )
    assert response.status_code == 200, response.text
    bounds = np.asarray(response.json()["bounds"])
    dense = keys[(0, 0)] @ q.T / np.sqrt(8.0)
    assert np.all(bounds >= dense.max(axis=0) - 1e-12)


def test_wrong_session_rejected(client):
    rng = np.random.default_rng(0)
    positions, keys, values = _make_shard(rng, range(4))
    body = _shard_bytes(positions, np.ones(4, bool), keys, values, (0,), (0,))
    response = client.post(
        "/v1/shard",
        params={
            "worker_id": 0,
            "session_id": "different-session",
            "layers": "0",
            "kv_heads": "0",
            "q_per_kv": 2,
        },
        content=body,
        headers=_auth(),
    )
    assert response.status_code == 403


def test_invalid_query_shape_rejected(client):
    rng = np.random.default_rng(0)
    positions, keys, values = _make_shard(rng, range(6))
    assert _upload(client, positions, keys, values).status_code == 200
    response = client.post(
        "/v1/bounds",
        json={"session_id": SESSION, "layer": 0, "kv_head": 0, "q": [[1.0, 2.0]]},
        headers=_auth(),
    )
    assert response.status_code == 422


def test_oversized_body_rejected():
    app = create_app(TOKEN, SESSION, 0, replace(LIMITS, max_request_bytes=64))
    client = TestClient(app)
    response = client.post(
        "/v1/shard",
        params={
            "worker_id": 0,
            "session_id": SESSION,
            "layers": "0",
            "kv_heads": "0",
            "q_per_kv": 2,
        },
        content=b"x" * 128,
        headers=_auth(),
    )
    assert response.status_code == 413


def test_expired_session_rejected():
    app = create_app(TOKEN, SESSION, 0, replace(LIMITS, session_ttl_s=0.0))
    client = TestClient(app)
    rng = np.random.default_rng(0)
    positions, keys, values = _make_shard(rng, range(4), layers=(0,), kv_heads=(0,))
    assert _upload(client, positions, keys, values, layers=(0,), kv_heads=(0,)).status_code == 200
    response = client.post(
        "/v1/bounds",
        json={"session_id": SESSION, "layer": 0, "kv_head": 0, "q": np.zeros((2, 8)).tolist()},
        headers=_auth(),
    )
    assert response.status_code == 410


def test_distributed_stats_match_dense_attention():
    rng = np.random.default_rng(7)
    n, dim, q_per_kv = 20, 8, 2
    all_positions = np.arange(n)
    keys = {(0, 0): rng.normal(size=(n, dim))}
    values = {(0, 0): rng.normal(size=(n, dim))}

    assignment = {w: np.arange(w, n, 2) for w in range(2)}
    overlap_position = 3
    assignment[0] = np.union1d(assignment[0], [overlap_position])
    multiplicities = {str(int(overlap_position)): 2}
    q = rng.normal(size=(q_per_kv, dim))

    lmaxes, sumexps, outs = [], [], []
    for worker_id in range(2):
        app = create_app(TOKEN, SESSION, worker_id)
        client = TestClient(app)
        positions = assignment[worker_id]
        primary = positions % 2 == worker_id
        response = _upload(
            client,
            positions,
            {(0, 0): keys[(0, 0)][positions]},
            {(0, 0): values[(0, 0)][positions]},
            layers=(0,),
            kv_heads=(0,),
            primary=primary,
            q_per_kv=q_per_kv,
            worker_id=worker_id,
        )
        assert response.status_code == 200, response.text
        response = client.post(
            "/v1/stats",
            json={
                "session_id": SESSION,
                "layer": 0,
                "kv_head": 0,
                "q": q.tolist(),
                "multiplicities": multiplicities,
                "subset": "assigned",
            },
            headers=_auth(),
        )
        assert response.status_code == 200, response.text
        body = response.json()
        lmaxes.append(body["lmax"])
        sumexps.append(body["sumexp"])
        outs.append(body["local_out"])

    merged = merge_stats_batch(
        np.asarray(lmaxes), np.asarray(sumexps), np.asarray(outs)
    )
    scale = 1.0 / np.sqrt(dim)
    for head in range(q_per_kv):
        _, expected = reference_attention(q[head], keys[(0, 0)], values[(0, 0)], scale)
        assert np.abs(merged[head] - expected).max() < 1e-12


def test_primary_subset_ignores_overlap(client):
    rng = np.random.default_rng(11)
    positions = np.array([0, 1, 2, 3])
    keys = {(0, 0): rng.normal(size=(4, 8))}
    values = {(0, 0): rng.normal(size=(4, 8))}
    primary = np.array([True, True, False, False])
    assert _upload(
        client, positions, keys, values, layers=(0,), kv_heads=(0,), primary=primary
    ).status_code == 200

    q = rng.normal(size=(2, 8))
    response = client.post(
        "/v1/stats",
        json={
            "session_id": SESSION,
            "layer": 0,
            "kv_head": 0,
            "q": q.tolist(),
            "subset": "primary",
            "multiplicities": {"0": 5},
        },
        headers=_auth(),
    )
    assert response.status_code == 200
    assert response.json()["token_count"] == 2


def test_bad_npz_rejected(client):
    response = client.post(
        "/v1/shard",
        params={
            "worker_id": 0,
            "session_id": SESSION,
            "layers": "0",
            "kv_heads": "0",
            "q_per_kv": 2,
        },
        content=b"not an npz",
        headers=_auth(),
    )
    assert response.status_code == 400


def _gen_reset(client, worker_id=0, num_workers=2, layers=(0,), kv_heads=(0,), q_per_kv=2,
               head_dim=8, max_positions=16):
    return client.post(
        "/v1/gen/reset",
        params={
            "worker_id": worker_id,
            "session_id": SESSION,
            "num_workers": num_workers,
            "layers": ",".join(map(str, layers)),
            "kv_heads": ",".join(map(str, kv_heads)),
            "q_per_kv": q_per_kv,
            "head_dim": head_dim,
            "max_positions": max_positions,
        },
        headers=_auth(),
    )


def _gen_append(client, positions, keys, values, worker_id=0, layers=(0,), kv_heads=(0,)):
    payload = {"positions": np.asarray(positions, dtype=np.int64)}
    for layer in layers:
        for kv in kv_heads:
            payload[f"k_{layer}_{kv}"] = np.asarray(keys[(layer, kv)], dtype=np.float32)
            payload[f"v_{layer}_{kv}"] = np.asarray(values[(layer, kv)], dtype=np.float32)
    buffer = io.BytesIO()
    np.savez(buffer, **payload)
    return client.post(
        "/v1/gen/append",
        params={"worker_id": worker_id, "session_id": SESSION},
        content=buffer.getvalue(),
        headers=_auth(),
    )


def _gen_attend(client, positions, q, layer=0, kv_head=0, worker_id=0):
    buffer = io.BytesIO()
    np.savez(
        buffer,
        positions=np.asarray(positions, dtype=np.int64),
        q=np.asarray(q, dtype=np.float32),
    )
    return client.post(
        "/v1/gen/attend",
        params={
            "worker_id": worker_id,
            "session_id": SESSION,
            "layer": layer,
            "kv_head": kv_head,
        },
        content=buffer.getvalue(),
        headers=_auth(),
    )


def test_generation_endpoints_require_auth_and_init(client):
    anonymous = client.post(
        "/v1/gen/reset",
        params={
            "worker_id": 0,
            "session_id": SESSION,
            "num_workers": 2,
            "layers": "0",
            "kv_heads": "0",
            "q_per_kv": 2,
            "head_dim": 8,
            "max_positions": 16,
        },
    )
    assert anonymous.status_code == 401
    rng = np.random.default_rng(3)
    q = rng.normal(size=(4, 2, 8)).astype(np.float32)
    buffer = io.BytesIO()
    np.savez(buffer, positions=np.arange(4), q=q)
    response = client.post(
        "/v1/gen/attend",
        params={"worker_id": 0, "session_id": SESSION, "layer": 0, "kv_head": 0},
        content=buffer.getvalue(),
        headers=_auth(),
    )
    assert response.status_code == 409


def test_generation_causal_attend_matches_dense(client):
    rng = np.random.default_rng(5)
    n, dim, q_per_kv = 12, 8, 2
    positions = np.arange(0, n, 2)
    keys = {(0, 0): rng.normal(size=(len(positions), dim))}
    values = {(0, 0): rng.normal(size=(len(positions), dim))}
    assert _gen_reset(client, worker_id=0, num_workers=2).status_code == 200
    response = _gen_append(client, positions, keys, values)
    assert response.status_code == 200, response.text
    assert response.json()["stored_positions"] == len(positions)

    query_positions = np.arange(n)
    q = rng.normal(size=(n, q_per_kv, dim)).astype(np.float32)
    response = _gen_attend(client, query_positions, q)
    assert response.status_code == 200, response.text
    with np.load(io.BytesIO(response.content), allow_pickle=False) as npz:
        lmax = npz["lmax"]
        sumexp = npz["sumexp"]
        out = npz["out"]
        owned = int(npz["owned_keys"])
        visible = npz["visible_keys"]
    assert owned == len(positions)
    expected_visible = np.array([len(positions[positions <= p]) for p in query_positions])
    assert np.array_equal(visible, expected_visible)
    assert np.all(np.isfinite(lmax[expected_visible > 0]))
    if np.any(expected_visible == 0):
        assert np.all(lmax[expected_visible == 0] == -np.inf)
    assert np.all(sumexp[expected_visible == 0] == 0.0)

    from distributed_llm.protocol import causal_local_stats_batch

    direct = causal_local_stats_batch(
        q, keys[(0, 0)], values[(0, 0)], positions, query_positions, 1.0 / np.sqrt(dim)
    )
    assert np.allclose(out, direct["local_out"])
    assert np.allclose(sumexp, direct["sumexp"])


def test_generation_append_is_idempotent_and_rejects_conflicts(client):
    rng = np.random.default_rng(9)
    positions = np.array([0, 2])
    keys = {(0, 0): rng.normal(size=(2, 8))}
    values = {(0, 0): rng.normal(size=(2, 8))}
    assert _gen_reset(client, worker_id=0).status_code == 200
    assert _gen_append(client, positions, keys, values).status_code == 200
    duplicate = _gen_append(client, positions, keys, values)
    assert duplicate.status_code == 200
    assert duplicate.json()["status"] == "duplicate"

    keys[(0, 0)] = keys[(0, 0)] + 1.0
    conflict = _gen_append(client, positions, keys, values)
    assert conflict.status_code == 409

    mixed = _gen_append(client, [0, 4], {k: np.vstack([v[:1], v[:1]]) for k, v in keys.items()},
                        {k: np.vstack([v[:1], v[:1]]) for k, v in values.items()})
    assert mixed.status_code == 409


def test_generation_rejects_foreign_positions_and_bad_shapes(client):
    rng = np.random.default_rng(13)
    assert _gen_reset(client, worker_id=0, num_workers=2).status_code == 200
    keys = {(0, 0): rng.normal(size=(2, 8))}
    values = {(0, 0): rng.normal(size=(2, 8))}
    assert _gen_append(client, [1, 3], keys, values).status_code == 403
    assert _gen_append(client, [0, 2], keys, values).status_code == 200

    q = rng.normal(size=(3, 2, 8)).astype(np.float32)
    assert _gen_attend(client, [0, 1, 2], q).status_code == 200
    assert _gen_attend(client, [0, 1], q).status_code == 422
    bad = q.copy()
    bad[0, 0, 0] = np.nan
    assert _gen_attend(client, [0, 1, 2], bad).status_code == 422
    assert _gen_attend(client, [2, 1, 0], q).status_code == 422
