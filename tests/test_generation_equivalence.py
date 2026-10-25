"""Distributed greedy decoding must match Hugging Face Qwen2 exactly.

These tests start a real worker cluster over loopback HTTP, stream one layer at
a time, and compare token-by-token greedy output with Qwen2ForCausalLM. No
network transport, ownership rule, or merge step is stubbed.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from distributed_llm.capture import load_model_metadata
from distributed_llm.cluster import ClusterError, WorkerCluster
from distributed_llm.config import Limits
from distributed_llm.decoder import (
    DistributedAttentionBackend,
    GenerationError,
    StreamingQwen2Decoder,
)
from distributed_llm.generation_runner import generate
from distributed_llm.gen_client import GenerationClient

TINY = dict(
    vocab_size=128,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=3,
    num_attention_heads=8,
    num_key_value_heads=2,
    max_position_embeddings=64,
    rope_theta=10000.0,
    rms_norm_eps=1e-6,
    tie_word_embeddings=False,
)
TINY_LIMITS = Limits(min_available_ram_gb=0.0, max_generation_positions=64)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def tiny_generation_model(tmp_path_factory):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(20241024)
    config = Qwen2Config(**TINY)
    model = Qwen2ForCausalLM(config)
    model.eval()
    path = tmp_path_factory.mktemp("tiny-qwen2-generation")
    model.save_pretrained(path, safe_serialization=True)
    config.save_pretrained(path)

    vocabulary = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[EOS]": 2,
        "hello": 3,
        "world": 4,
        "the": 5,
        "quick": 6,
        "brown": 7,
        "fox": 8,
        "jumps": 9,
        "over": 10,
        "lazy": 11,
        "dog": 12,
    }
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
    )
    tokenizer.save_pretrained(path)
    return path, model, tokenizer


def _start_cluster(cluster: WorkerCluster) -> None:
    try:
        cluster.start()
    except ClusterError as exc:
        cluster.shutdown()
        pytest.skip(f"not enough free RAM for a worker cluster: {exc}")


def _hf_greedy(model, prompt_tokens, max_new_tokens, eos_token_id):
    produced = []
    sequence = list(prompt_tokens)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(
                torch.tensor([sequence], dtype=torch.long), use_cache=False
            ).logits[0, -1]
            token = int(torch.argmax(logits))
            produced.append(token)
            sequence.append(token)
            if eos_token_id is not None and token == int(eos_token_id):
                break
    return produced


@pytest.mark.parametrize("workers", [2, 3])
def test_distributed_decoder_matches_hugging_face(tiny_generation_model, tmp_path, workers):
    path, model, _ = tiny_generation_model
    prompt_tokens = [3, 4, 5, 6, 7, 8, 3, 4, 9, 10]
    max_new = 3

    geometry, root = load_model_metadata(model_path=str(path))
    cluster = WorkerCluster(workers, tmp_path / f"decoder-{workers}", limits=TINY_LIMITS)
    _start_cluster(cluster)
    client = GenerationClient(cluster.connection_info(), timeout_s=60.0)
    try:
        attention = DistributedAttentionBackend(client, geometry, TINY_LIMITS)
        decoder = StreamingQwen2Decoder(str(root), geometry, attention, TINY_LIMITS)
        attention.reset(len(prompt_tokens) + max_new)

        hidden = decoder.prefill(prompt_tokens)
        produced = []
        margins = []
        for step in range(max_new):
            normed = decoder.final_norm(hidden[-1].reshape(1, -1))
            logits = decoder.logits(normed[0])
            produced.append(logits["token"])
            margins.append(logits["margin"])
            if step < max_new - 1:
                hidden = decoder.decode_step(
                    logits["token"], len(prompt_tokens) + step
                )
            assert np.isfinite(logits["top1_logit"])
        summary = attention.summary()
    finally:
        client.close()
        cluster.shutdown()

    expected = _hf_greedy(model, prompt_tokens, max_new, eos_token_id=None)
    assert produced == expected, f"greedy tokens diverged: {produced} vs {expected}"
    assert all(isinstance(margin, float) and margin >= 0.0 for margin in margins)

    steps_forwarded = 1 + (max_new - 1)
    assert summary["append_calls"] == geometry.num_layers * steps_forwarded
    assert summary["attend_calls"] == (
        geometry.num_layers * geometry.num_key_value_heads * steps_forwarded
    )
    assert summary["key_vectors_uploaded"] == (
        2
        * geometry.num_layers
        * geometry.num_key_value_heads
        * (len(prompt_tokens) + max_new - 1)
    )
    assert summary["transport"]["requests"] > 0
    assert summary["transport"]["failures"] == 0


def test_generation_runner_produces_distributed_tokens(tiny_generation_model, tmp_path):
    path, model, tokenizer = tiny_generation_model
    prompt = "hello world the quick brown fox"
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    max_new = 3

    cluster = WorkerCluster(2, tmp_path / "runner", limits=TINY_LIMITS)
    _start_cluster(cluster)
    try:
        connection = cluster.connection_info()
        result = generate(
            prompt,
            max_new_tokens=max_new,
            connection=connection,
            max_generation_positions=48,
            model_path=str(path),
            limits=TINY_LIMITS,
            expected_prompt_tokens=len(prompt_tokens),
        )
    finally:
        cluster.shutdown()

    assert result["status"] == "ok"
    assert result["method"] == "distributed_exact_causal_decode"
    assert result["computed_by"].startswith("Exact causal decoding")
    assert "standard forward pass" not in result["computed_by"]
    assert result["prompt"]["token_count"] == len(prompt_tokens)
    assert result["attention"]["mode"] == "distributed_exact_causal"
    assert result["attention"]["append_calls"] == result["decoder"]["layers_forwarded"]
    assert result["transport"]["failures"] == 0
    assert len(result["workers"]) == 2
    for worker in result["workers"]:
        assert worker["generation_loaded"] is True
        assert worker["generation_units"] == (
            result["model"]["geometry"]["num_layers"]
            * result["model"]["geometry"]["num_key_value_heads"]
        )
        assert worker["gen_appends"] > 0
        assert worker["gen_attend_calls"] > 0
        assert worker["generation_positions"] > 0
    assert sum(worker["generation_positions"] for worker in result["workers"]) == (
        len(prompt_tokens) + max(0, result["output"]["token_count"] - 1)
    )

    expected = _hf_greedy(model, prompt_tokens, max_new, tokenizer.eos_token_id)
    assert result["output"]["token_ids"] == expected
    assert result["output"]["token_count"] == len(expected)
    assert result["timings"]["prefill_s"] > 0.0
    assert result["memory"]["process_rss_mb"] > 0.0


def test_generation_runner_rejects_capture_token_mismatch(tiny_generation_model):
    path, _, tokenizer = tiny_generation_model
    prompt = "hello world"
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    # The prompt/capture agreement check runs before any connection is opened.
    with pytest.raises(GenerationError, match="refusing to decode"):
        generate(
            prompt,
            max_new_tokens=1,
            connection={},
            max_generation_positions=48,
            model_path=str(path),
            limits=TINY_LIMITS,
            expected_prompt_tokens=len(prompt_tokens) + 1,
        )
