"""Checks against the real Qwen2.5-Coder-1.5B weights in the local cache.

These tests skip when the pinned snapshot is not present. The decoder-layer test
is the ground truth: our fp32 layer implementation must reproduce the Hugging
Face layer exactly, including the captured Q/K/V arrays.
"""
from __future__ import annotations

import gc

import numpy as np
import pytest
import torch

from distributed_llm.capture import (
    ModelUnavailableError,
    build_rope,
    capture_attention,
    causal_mask,
    load_layer_weights,
    load_model_metadata,
    open_weights,
    qwen2_layer_forward,
)

pytestmark = pytest.mark.real_model

PROMPT = "Distributed attention splits the key and value cache across workers."


@pytest.fixture(scope="module")
def real():
    try:
        geometry, root = load_model_metadata()
    except ModelUnavailableError as exc:
        pytest.skip(str(exc))
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(str(root), local_files_only=True)
    return geometry, root, config


def _embedding_rows(root, tokens):
    from safetensors import safe_open

    with safe_open(str(root / "model.safetensors"), framework="pt", device="cpu") as handle:
        rows = handle.get_slice("model.embed_tokens.weight")
        cache = {}
        out = []
        for token in tokens:
            if token not in cache:
                cache[token] = rows[token : token + 1].to(torch.float32)
            out.append(cache[token][0])
    return torch.stack(out).unsqueeze(0)


def test_real_layer0_matches_hugging_face(real):
    geometry, root, config = real
    from safetensors import safe_open
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

    from distributed_llm.capture import tokenize_prompt

    tokens = tokenize_prompt(PROMPT, max_tokens=32).tokens
    embeds = _embedding_rows(root, tokens)

    layer = Qwen2DecoderLayer(config, 0).eval()
    with safe_open(str(root / "model.safetensors"), framework="pt", device="cpu") as handle:
        prefix = "model.layers.0."
        state = {
            key[len(prefix) :]: handle.get_tensor(key).to(torch.float32)
            for key in handle.keys()
            if key.startswith(prefix)
        }
    layer.load_state_dict(state, strict=True)
    del state
    gc.collect()

    position_ids = torch.arange(len(tokens)).unsqueeze(0)
    mask4d = causal_mask(len(tokens), torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        ref_hidden, ref_attn = layer(
            embeds,
            attention_mask=mask4d,
            position_ids=position_ids,
            output_attentions=True,
            use_cache=False,
        )[:2]

    cos, sin = build_rope(len(tokens), geometry.head_dim, geometry.rope_theta, "cpu", torch.float32)
    with open_weights(root / "model.safetensors") as handle:
        weights = load_layer_weights(handle, 0, torch)
    ours_hidden, captured = qwen2_layer_forward(
        embeds[0].clone(),
        weights,
        geometry,
        cos,
        sin,
        causal_mask(len(tokens), torch.float32),
        capture_heads=list(range(geometry.num_key_value_heads)),
        capture_last_attention=True,
    )

    hidden_diff = float((ours_hidden - ref_hidden[0]).abs().max())
    attn_diff = float((captured["last_attention"] - ref_attn[0, :, -1, :]).abs().max())
    assert hidden_diff < 1e-4, f"real layer hidden mismatch {hidden_diff:.3e}"
    assert attn_diff < 1e-5, f"real layer attention mismatch {attn_diff:.3e}"

    ref_last = ref_attn[0, :, -1, :].numpy()
    for kv_head in range(geometry.num_key_value_heads):
        q = captured[kv_head]["q_last"].numpy()
        k = captured[kv_head]["keys"].numpy()
        scores = (q @ k.T) / np.sqrt(geometry.head_dim)
        scores -= scores.max(axis=1, keepdims=True)
        probs = np.exp(scores)
        probs /= probs.sum(axis=1, keepdims=True)
        q_heads = geometry.q_heads_for_kv(kv_head)
        # Independent numpy recomputation from captured fp32 tensors; Hugging
        # Face applies its own fused softmax, so allow float32 round-off.
        assert np.abs(probs - ref_last[q_heads]).max() < 5e-5


def test_real_full_capture_shapes_and_finiteness(real):
    geometry, root, _ = real
    from distributed_llm.capture import tokenize_prompt
    from distributed_llm.config import LIMITS
    from distributed_llm.logging_utils import available_ram_gb

    free = available_ram_gb()
    if 0 <= free < LIMITS.min_available_ram_gb:
        pytest.skip(f"only {free:.2f} GB free; capture needs {LIMITS.min_available_ram_gb:.2f} GB")

    tokenized = tokenize_prompt(PROMPT, max_tokens=32)
    result = capture_attention(
        tokenized.tokens,
        model_path=str(root),
        capture_last_attention=True,
    )
    n = len(tokenized.tokens)
    assert result.geometry.num_layers == 28
    assert result.geometry.num_key_value_heads == 2
    assert result.geometry.q_per_kv == 6
    assert len(result.heads) == 28 * 2
    assert len(result.last_attention) == 28

    for (layer, kv), head in result.heads.items():
        assert head.q_last.shape == (6, 128), (layer, kv)
        assert head.keys.shape == (n, 128)
        assert head.values.shape == (n, 128)
        assert np.isfinite(head.q_last).all()
        assert np.isfinite(head.keys).all()
        assert np.isfinite(head.values).all()

    for layer, probs in result.last_attention.items():
        assert probs.shape == (12, n), layer
        assert np.abs(probs.sum(axis=1) - 1.0).max() < 1e-6

    head = result.heads[(27, 0)]
    scores = (head.q_last @ head.keys.T) / np.sqrt(geometry.head_dim)
    scores -= scores.max(axis=1, keepdims=True)
    probs = np.exp(scores)
    probs /= probs.sum(axis=1, keepdims=True)
    assert np.abs(probs - result.last_attention[27][:6]).max() < 1e-5
