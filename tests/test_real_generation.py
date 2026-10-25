"""Real Qwen2.5-Coder-1.5B distributed generation through the worker cluster.

Skipped when the pinned snapshot is not cached or the machine cannot host the
cluster and decoder. The tiny-model suite proves token-for-token equivalence
with Hugging Face; this test proves the same paths carry real weights.
"""
from __future__ import annotations

import pytest

from distributed_llm.capture import ModelUnavailableError, load_model_metadata
from distributed_llm.cluster import ClusterError, WorkerCluster
from distributed_llm.config import Limits
from distributed_llm.generation_runner import generate

pytestmark = [pytest.mark.real_model, pytest.mark.integration]

REAL_LIMITS = Limits(min_available_ram_gb=0.0)


@pytest.fixture(scope="module")
def real_model():
    try:
        geometry, root = load_model_metadata()
    except ModelUnavailableError as exc:
        pytest.skip(str(exc))
    return geometry, root


def test_real_distributed_generation(real_model, tmp_path):
    geometry, root = real_model
    cluster = WorkerCluster(2, tmp_path / "real-generation", limits=REAL_LIMITS)
    try:
        cluster.start()
    except ClusterError as exc:
        cluster.shutdown()
        pytest.skip(f"not enough free RAM for a worker cluster: {exc}")

    try:
        result = generate(
            "def add(a, b):",
            max_new_tokens=2,
            connection=cluster.connection_info(),
            max_generation_positions=64,
            model_path=str(root),
            limits=REAL_LIMITS,
        )
    finally:
        cluster.shutdown()

    assert result["status"] == "ok"
    assert result["method"] == "distributed_exact_causal_decode"
    assert "standard forward pass" not in result["computed_by"]
    assert result["model"]["geometry"]["num_layers"] == geometry.num_layers

    prompt_tokens = result["prompt"]["token_count"]
    output = result["output"]
    assert prompt_tokens > 0
    assert 1 <= output["token_count"] <= 2
    assert all(0 <= token < geometry.vocab_size for token in output["token_ids"])

    forward_passes = 1 + max(0, output["token_count"] - 1)
    assert result["decoder"]["layers_forwarded"] == geometry.num_layers * forward_passes
    assert result["attention"]["append_calls"] == geometry.num_layers * forward_passes
    assert result["attention"]["attend_calls"] == (
        geometry.num_layers
        * geometry.num_key_value_heads
        * forward_passes
    )
    assert result["transport"]["failures"] == 0
    assert result["timings"]["prefill_s"] > 0.0
    assert result["memory"]["peak_rss_mb"] > 0.0

    workers = result["workers"]
    assert len(workers) == 2
    for worker in workers:
        assert worker["generation_loaded"] is True
        assert worker["gen_appends"] > 0
        assert worker["gen_attend_calls"] > 0
    assert sum(worker["generation_positions"] for worker in workers) == (
        prompt_tokens + max(0, output["token_count"] - 1)
    )
