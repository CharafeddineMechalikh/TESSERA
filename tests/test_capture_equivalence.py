"""Capture must reproduce Hugging Face Qwen2 attention exactly.

Tiny random model: compares hidden states, last-query attention probabilities,
and pre-o_proj attention outputs for every layer and KV head.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from distributed_llm.capture import (
    capture_attention,
    load_layer_weights,
    load_model_metadata,
    open_weights,
    qwen2_layer_forward,
)
from distributed_llm.config import Limits

# The tiny model needs a few hundred MB at most; skip the host admission check
# so the suite does not depend on how much RAM other test modules are holding.
TINY_LIMITS = Limits(min_available_ram_gb=0.0)

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


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(1234)
    config = Qwen2Config(**TINY)
    model = Qwen2ForCausalLM(config)
    model.eval()
    path = tmp_path_factory.mktemp("tiny-qwen2")
    model.save_pretrained(path, safe_serialization=True)
    config.save_pretrained(path)
    return path, model


def _hf_forward(model, tokens):
    with torch.no_grad():
        outputs = model(
            torch.tensor([tokens], dtype=torch.long),
            output_attentions=True,
            output_hidden_states=True,
            use_cache=False,
        )
    return outputs


def test_capture_matches_hugging_face(tiny_model):
    path, model = tiny_model
    tokens = [1, 5, 9, 17, 33, 65, 7, 3, 42, 11, 90, 4]

    result = capture_attention(
        tokens,
        model_path=str(path),
        capture_last_attention=True,
        capture_hidden_states=True,
        limits=TINY_LIMITS,
    )
    outputs = _hf_forward(model, tokens)

    assert result.geometry.num_layers == TINY["num_hidden_layers"]
    assert result.geometry.q_per_kv == TINY["num_attention_heads"] // TINY["num_key_value_heads"]
    assert set(result.heads) == {
        (layer, kv) for layer in range(TINY["num_hidden_layers"]) for kv in range(TINY["num_key_value_heads"])
    }

    from distributed_llm.capture import rms_norm

    worst_prob = 0.0
    worst_hidden = 0.0
    final_norm = model.model.norm.weight.detach()
    for layer in range(TINY["num_hidden_layers"]):
        ours_hidden = torch.tensor(result.hidden_states[layer], dtype=torch.float32)
        if layer == TINY["num_hidden_layers"] - 1:
            # HF appends the post-final-norm tensor as the last hidden state.
            ours_hidden = rms_norm(ours_hidden, final_norm, TINY["rms_norm_eps"])
        ref_hidden = outputs.hidden_states[layer + 1].detach().numpy()[0]
        worst_hidden = max(worst_hidden, float(np.abs(ours_hidden.numpy() - ref_hidden).max()))

        ours_attn = result.last_attention[layer]
        ref_attn = outputs.attentions[layer].detach().numpy()[0, :, -1, :]
        worst_prob = max(worst_prob, float(np.abs(ours_attn - ref_attn).max()))

        for kv in range(TINY["num_key_value_heads"]):
            head = result.heads[(layer, kv)]
            q = head.q_last
            k = head.keys
            scores = (q @ k.T) / np.sqrt(result.geometry.head_dim)
            scores -= scores.max(axis=1, keepdims=True)
            probs = np.exp(scores)
            probs /= probs.sum(axis=1, keepdims=True)
            ours_qk = probs
            ref_qk = ref_attn[head.q_heads]
            worst_prob = max(worst_prob, float(np.abs(ours_qk - ref_qk).max()))

    assert worst_hidden < 2e-5, f"hidden-state mismatch {worst_hidden:.3e}"
    assert worst_prob < 2e-6, f"attention mismatch {worst_prob:.3e}"

    # Captured V must reproduce the attention output that HF feeds to o_proj.
    captured = {}
    hooks = []

    def make_hook(layer_idx):
        def hook(_module, inputs):
            captured[layer_idx] = inputs[0].detach()

        return hook

    for layer_idx in range(TINY["num_hidden_layers"]):
        hooks.append(
            model.model.layers[layer_idx].self_attn.o_proj.register_forward_pre_hook(make_hook(layer_idx))
        )
    try:
        _hf_forward(model, tokens)
    finally:
        for hook in hooks:
            hook.remove()

    worst_out = 0.0
    head_dim = result.geometry.head_dim
    for layer in range(TINY["num_hidden_layers"]):
        ref_last = captured[layer].numpy()[0, -1]
        for kv in range(TINY["num_key_value_heads"]):
            head = result.heads[(layer, kv)]
            q = head.q_last
            scores = (q @ head.keys.T) / np.sqrt(head_dim)
            scores -= scores.max(axis=1, keepdims=True)
            probs = np.exp(scores)
            probs /= probs.sum(axis=1, keepdims=True)
            for offset, q_head in enumerate(head.q_heads):
                start = q_head * head_dim
                ref_head = ref_last[start : start + head_dim]
                worst_out = max(worst_out, float(np.abs(probs[offset] @ head.values - ref_head).max()))
    assert worst_out < 3e-6, f"pre-o_proj output mismatch {worst_out:.3e}"


def test_decoder_layer_matches_hugging_face(tiny_model):
    """Our whole layer function, not just attention, matches Qwen2DecoderLayer."""
    path, model = tiny_model
    tokens = [1, 5, 9, 17, 33, 65, 7, 3, 42]

    geometry, root = load_model_metadata(model_path=str(path))
    from distributed_llm.capture import build_rope, causal_mask

    with torch.no_grad():
        embeds = model.model.embed_tokens(torch.tensor([tokens])).detach()
        cos, sin = build_rope(len(tokens), geometry.head_dim, geometry.rope_theta, "cpu", torch.float32)
        mask = causal_mask(len(tokens), torch.float32)

        for layer_idx in range(geometry.num_layers):
            ref_layer = model.model.layers[layer_idx]
            ref_out = ref_layer(
                embeds,
                attention_mask=mask.unsqueeze(0).unsqueeze(0),
                position_ids=torch.arange(len(tokens)).unsqueeze(0),
                position_embeddings=(cos.unsqueeze(0), sin.unsqueeze(0)),
            )[0]

            with open_weights(root / "model.safetensors") as handle:
                weights = load_layer_weights(handle, layer_idx, torch)
            ours, _ = qwen2_layer_forward(
                embeds[0].clone(), weights, geometry, cos, sin, mask
            )
            diff = float((ours - ref_out[0]).abs().max())
            assert diff < 3e-5, f"layer {layer_idx} mismatch {diff:.3e}"

            embeds = ref_out
