"""Exact causal decoder whose attention runs on the isolated worker cluster.

The decoder owns tokenization, embeddings, projections, MLP blocks, and the LM
head. It never keeps a local copy of the KV cache: every K/V vector is uploaded
to the worker that owns its token position, and every attention query is sent to
all workers. The coordinator merges worker statistics with a stable float64
log-sum-exp, so the result includes every visible position exactly.

Weights are streamed one layer at a time: only the current layer and the MLP
activations are materialised in fp32, because the fp32 model does not fit in
this machine's RAM. The bf16 weights file stays mapped for the decoder lifetime,
so the process RSS reports it, but those pages are file-backed and reclaimable
rather than anonymous memory. Generation is therefore slower than a
resident-weight runtime, and the artifacts report the measured cost rather than
implying a speedup.
"""
from __future__ import annotations

import math
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from safetensors import safe_open

from .capture import Geometry, load_layer_weights, rms_norm
from .config import LIMITS, Limits
from .logging_utils import process_rss_mb
from .protocol import (
    causal_local_stats_batch,
    merge_causal_stats,
)


class GenerationError(RuntimeError):
    pass


GENERATION_PROVENANCE = (
    "Exact causal decoding through the isolated attention workers. Every K/V "
    "vector is uploaded to the worker that owns its token position, every "
    "attention query is evaluated on all workers with causal masking, and the "
    "coordinator merges float64 log-sum-exp statistics. No monolithic model "
    "forward pass is involved."
)
GENERATION_METHOD = "distributed_exact_causal_decode"


def _torch():
    import torch

    return torch


def build_rope_positions(positions, head_dim: int, base: float, dtype):
    """RoPE cosine/sine rows for arbitrary absolute token positions."""
    torch = _torch()
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim)
    )
    freqs = torch.outer(positions.to(torch.float32), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


def _rotate_half(x):
    torch = _torch()
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope_positions(x, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (x * cos) + (_rotate_half(x) * sin)


class LocalAttentionBackend:
    """Reference backend that keeps K/V in this process.

    Used by tests to compare the decoder against Hugging Face without HTTP. It
    uses the same numerical kernel as the workers, so it is not a stand-in for
    the distributed transport.
    """

    def __init__(self, geometry: Geometry):
        self.geometry = geometry
        self.layers = tuple(range(geometry.num_layers))
        self.kv_heads = tuple(range(geometry.num_key_value_heads))
        self.keys: Dict[Tuple[int, int], np.ndarray] = {}
        self.values: Dict[Tuple[int, int], np.ndarray] = {}
        self.positions: Dict[Tuple[int, int], np.ndarray] = {}
        self.max_positions = 0

    def reset(self, max_positions: int) -> None:
        self.keys = {}
        self.values = {}
        self.positions = {}
        self.max_positions = int(max_positions)

    def _store(self, layer, positions, k_by_kv, v_by_kv):
        positions = np.asarray(positions, dtype=np.int64)
        for kv in k_by_kv:
            unit = (layer, kv)
            block_k = np.asarray(k_by_kv[kv], dtype=np.float64)
            block_v = np.asarray(v_by_kv[kv], dtype=np.float64)
            existing = self.positions.get(unit)
            if existing is None:
                self.positions[unit] = positions.copy()
                self.keys[unit] = block_k.copy()
                self.values[unit] = block_v.copy()
                continue
            combined = np.concatenate([existing, positions])
            order = np.argsort(combined, kind="stable")
            self.keys[unit] = np.concatenate([self.keys[unit], block_k], axis=0)[order]
            self.values[unit] = np.concatenate([self.values[unit], block_v], axis=0)[order]
            self.positions[unit] = combined[order]

    def _attend(self, layer, query_positions, q_by_kv):
        outputs = np.zeros(
            (len(query_positions), self.geometry.num_attention_heads, self.geometry.head_dim),
            dtype=np.float64,
        )
        scale = 1.0 / math.sqrt(self.geometry.head_dim)
        for kv, q in q_by_kv.items():
            unit = (layer, kv)
            stats = causal_local_stats_batch(
                q,
                self.keys[unit],
                self.values[unit],
                self.positions[unit],
                query_positions,
                scale,
            )
            start = kv * self.geometry.q_per_kv
            outputs[:, start : start + self.geometry.q_per_kv, :] = stats["local_out"]
        return outputs

    def prefill_layer(self, layer, positions, q_by_kv, k_by_kv, v_by_kv):
        self._store(layer, positions, k_by_kv, v_by_kv)
        return self._attend(layer, np.asarray(positions, dtype=np.int64), q_by_kv)

    def decode_layer(self, layer, position, q_by_kv, k_by_kv, v_by_kv):
        positions = np.asarray([position], dtype=np.int64)
        self._store(layer, positions, k_by_kv, v_by_kv)
        return self._attend(layer, positions, q_by_kv)

    def summary(self) -> dict:
        return {"mode": "local_reference", "layers": len(self.layers)}

    def close(self) -> None:
        self.keys = {}
        self.values = {}


class DistributedAttentionBackend:
    """Attention backend that sends every query and KV block to the workers."""

    def __init__(self, client, geometry: Geometry, limits: Limits = LIMITS):
        self.client = client
        self.geometry = geometry
        self.limits = limits
        self.layers = tuple(range(geometry.num_layers))
        self.kv_heads = tuple(range(geometry.num_key_value_heads))
        self.max_positions = 0
        self.counters = {
            "append_calls": 0,
            "attend_calls": 0,
            "query_rows": 0,
            "key_vectors_uploaded": 0,
            "visible_key_count": 0,
        }

    def reset(self, max_positions: int) -> None:
        self.max_positions = int(max_positions)
        self.client.reset(
            self.layers,
            self.kv_heads,
            self.geometry.q_per_kv,
            self.geometry.head_dim,
            self.max_positions,
        )

    def prefill_layer(self, layer, positions, q_by_kv, k_by_kv, v_by_kv):
        positions = np.asarray(positions, dtype=np.int64)
        self.client.append(
            positions,
            {(layer, kv): k_by_kv[kv] for kv in self.kv_heads},
            {(layer, kv): v_by_kv[kv] for kv in self.kv_heads},
        )
        self.counters["append_calls"] += 1
        self.counters["key_vectors_uploaded"] += len(positions) * len(self.kv_heads) * 2
        return self._attend(layer, positions, q_by_kv)

    def decode_layer(self, layer, position, q_by_kv, k_by_kv, v_by_kv):
        positions = np.asarray([position], dtype=np.int64)
        self.client.append(
            positions,
            {(layer, kv): k_by_kv[kv] for kv in self.kv_heads},
            {(layer, kv): v_by_kv[kv] for kv in self.kv_heads},
        )
        self.counters["append_calls"] += 1
        self.counters["key_vectors_uploaded"] += len(self.kv_heads) * 2
        return self._attend(layer, positions, q_by_kv)

    def _attend(self, layer: int, query_positions: np.ndarray, q_by_kv) -> np.ndarray:
        outputs = np.zeros(
            (len(query_positions), self.geometry.num_attention_heads, self.geometry.head_dim),
            dtype=np.float64,
        )
        max_position = int(query_positions.max())
        for kv, q in q_by_kv.items():
            replies = self.client.attend(layer, kv, query_positions, q)
            lmaxes = np.stack([reply["lmax"] for reply in replies])
            sumexps = np.stack([reply["sumexp"] for reply in replies])
            outs = np.stack([reply["out"] for reply in replies])
            merged, totals = merge_causal_stats(lmaxes, sumexps, outs)
            if np.any(totals <= 0.0):
                raise GenerationError(
                    f"attention merge found no visible keys at layer {layer} kv {kv}"
                )
            if not np.isfinite(merged).all():
                raise GenerationError(
                    f"attention merge produced non-finite values at layer {layer} kv {kv}"
                )
            visible = np.sum(
                np.stack([reply["visible_keys"] for reply in replies]), axis=0
            )
            expected_visible = query_positions + 1
            if not np.array_equal(visible, expected_visible):
                raise GenerationError(
                    f"visible key accounting mismatch at layer {layer} kv {kv}: "
                    f"workers reported {visible.tolist()[:8]}, expected "
                    f"{expected_visible.tolist()[:8]}"
                )
            for reply in replies:
                expected_owned = _owned_positions(max_position, self.client.num_workers, reply["worker_id"])
                if int(reply["owned_keys"]) != expected_owned:
                    raise GenerationError(
                        f"worker {reply['worker_id']} owns {reply['owned_keys']} positions "
                        f"but should own {expected_owned} up to position {max_position}"
                    )
            self.counters["visible_key_count"] += int(visible.sum())
            start = kv * self.geometry.q_per_kv
            outputs[:, start : start + self.geometry.q_per_kv, :] = merged
        self.counters["attend_calls"] += len(q_by_kv)
        self.counters["query_rows"] += len(query_positions)
        return outputs

    def summary(self) -> dict:
        payload = {"mode": "distributed_exact_causal"}
        payload.update(self.counters)
        payload["transport"] = self.client.summary()
        payload["kv_ownership"] = f"position % {self.client.num_workers} == worker_id"
        return payload

    def close(self) -> None:
        self.client.close()


def _owned_positions(max_position: int, num_workers: int, worker_id: int) -> int:
    """Number of positions in [0, max_position] assigned to a strided worker."""
    if max_position < 0 or worker_id >= num_workers:
        return 0
    return (max_position - worker_id) // num_workers + 1


class StreamingQwen2Decoder:
    """Layer-streamed Qwen2 forward pass with a pluggable attention backend."""

    def __init__(
        self,
        model_path: str,
        geometry: Geometry,
        attention,
        limits: Limits = LIMITS,
        logger=None,
    ):
        self.model_path = Path(model_path)
        self.weights_path = self.model_path / "model.safetensors"
        if not self.weights_path.is_file():
            raise GenerationError(f"model.safetensors is missing under {self.model_path}")
        self.geometry = geometry
        self.attention = attention
        self.limits = limits
        self.logger = logger
        self.torch = _torch()
        self.torch.set_grad_enabled(False)
        threads = min(8, __import__("os").cpu_count() or 1)
        self.torch.set_num_threads(threads)
        try:
            self.torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        self.timings = {
            "weight_load_s": 0.0,
            "attention_s": 0.0,
            "mlp_s": 0.0,
            "embedding_s": 0.0,
            "logits_s": 0.0,
            "layers_forwarded": 0,
        }
        # One mapping for the whole decoder lifetime. Opening a fresh handle per
        # layer unmapped and remapped the multi-gigabyte weights file hundreds of
        # times per decode, which faulted natively under memory pressure.
        self._stack = ExitStack()
        self._handle = None
        try:
            self._handle = self._stack.enter_context(
                safe_open(str(self.weights_path), framework="pt", device="cpu")
            )
            self._available = set(self._handle.keys())
            if "model.norm.weight" not in self._available:
                raise GenerationError("model.norm.weight is missing")
            self.norm_weight = self._handle.get_tensor("model.norm.weight").to(
                self.torch.float32
            )
            self.lm_head_key = (
                "lm_head.weight"
                if "lm_head.weight" in self._available
                else "model.embed_tokens.weight"
            )
            if self.lm_head_key not in self._available:
                raise GenerationError("no LM head or tied embedding matrix found")
            shape = self._handle.get_slice(self.lm_head_key).get_shape()
            if int(shape[1]) != geometry.hidden_size:
                raise GenerationError("LM head hidden dimension does not match the model")
            self.vocab_size = int(shape[0])
            if self.vocab_size < geometry.vocab_size:
                raise GenerationError("LM head is smaller than the configured vocabulary")
        except BaseException:
            self.close()
            raise

    @contextmanager
    def _weights(self):
        if self._handle is None:
            raise GenerationError("decoder is closed")
        yield self._handle

    def close(self) -> None:
        handle, self._handle = self._handle, None
        self._stack.close()
        del handle

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False

    def embed_tokens(self, tokens: Sequence[int]) -> object:
        started = time.perf_counter()
        torch = self.torch
        hidden = torch.empty(
            (len(tokens), self.geometry.hidden_size), dtype=torch.float32
        )
        cache: Dict[int, object] = {}
        with self._weights() as handle:
            embedding_slice = handle.get_slice("model.embed_tokens.weight")
            for row, token in enumerate(tokens):
                token = int(token)
                if token < 0 or token >= self.geometry.vocab_size:
                    raise GenerationError(f"token id {token} is outside the vocabulary")
                cached = cache.get(token)
                if cached is None:
                    cached = embedding_slice[token : token + 1].to(torch.float32)
                    cache[token] = cached
                hidden[row] = cached[0]
        self.timings["embedding_s"] += time.perf_counter() - started
        return hidden

    def _forward_layer(
        self,
        hidden,
        layer_idx: int,
        positions: np.ndarray,
        decode: bool,
    ):
        torch = self.torch
        geometry = self.geometry
        load_started = time.perf_counter()
        with self._weights() as handle:
            weights = load_layer_weights(handle, layer_idx, torch)
        self.timings["weight_load_s"] += time.perf_counter() - load_started

        n_tokens = hidden.shape[0]
        linear = torch.nn.functional.linear
        normed = rms_norm(hidden, weights["input_norm"], geometry.rms_norm_eps)
        q = linear(normed, weights["q_w"], weights["q_b"])
        k = linear(normed, weights["k_w"], weights["k_b"])
        v = linear(normed, weights["v_w"], weights["v_b"])
        q = q.view(n_tokens, geometry.num_attention_heads, geometry.head_dim)
        k = k.view(n_tokens, geometry.num_key_value_heads, geometry.head_dim)
        v = v.view(n_tokens, geometry.num_key_value_heads, geometry.head_dim)

        cos, sin = build_rope_positions(
            torch.as_tensor(positions, dtype=torch.int64),
            geometry.head_dim,
            geometry.rope_theta,
            torch.float32,
        )
        q = apply_rope_positions(q, cos, sin)
        k = apply_rope_positions(k, cos, sin)
        q_np = q.numpy().astype(np.float32, copy=False)
        k_np = k.numpy().astype(np.float32, copy=False)
        v_np = v.numpy().astype(np.float32, copy=False)
        q_by_kv = {
            kv: np.ascontiguousarray(
                q_np[:, kv * geometry.q_per_kv : (kv + 1) * geometry.q_per_kv, :]
            )
            for kv in range(geometry.num_key_value_heads)
        }
        k_by_kv = {kv: np.ascontiguousarray(k_np[:, kv, :]) for kv in range(geometry.num_key_value_heads)}
        v_by_kv = {kv: np.ascontiguousarray(v_np[:, kv, :]) for kv in range(geometry.num_key_value_heads)}
        del q, k, v

        attention_started = time.perf_counter()
        if decode:
            attention_out = self.attention.decode_layer(
                layer_idx, int(positions[0]), q_by_kv, k_by_kv, v_by_kv
            )
        else:
            attention_out = self.attention.prefill_layer(
                layer_idx, positions, q_by_kv, k_by_kv, v_by_kv
            )
        self.timings["attention_s"] += time.perf_counter() - attention_started
        if attention_out.shape != (
            n_tokens,
            geometry.num_attention_heads,
            geometry.head_dim,
        ):
            raise GenerationError(
                f"attention backend returned shape {attention_out.shape} at layer {layer_idx}"
            )

        attn = torch.from_numpy(np.ascontiguousarray(attention_out)).to(torch.float32)
        attn = attn.reshape(n_tokens, geometry.hidden_size)
        hidden = hidden + linear(attn, weights["o_w"], weights["o_b"])

        mlp_started = time.perf_counter()
        normed2 = rms_norm(hidden, weights["post_norm"], geometry.rms_norm_eps)
        gate = linear(normed2, weights["gate_w"], None)
        up = linear(normed2, weights["up_w"], None)
        hidden = hidden + linear(torch.nn.functional.silu(gate) * up, weights["down_w"], None)
        self.timings["mlp_s"] += time.perf_counter() - mlp_started
        self.timings["layers_forwarded"] += 1
        del weights, normed, normed2, gate, up, attn, attention_out
        return hidden

    def prefill(self, tokens: Sequence[int], progress: Optional[Callable[[dict], None]] = None):
        if not tokens:
            raise GenerationError("prompt token sequence is empty")
        n_tokens = len(tokens)
        if n_tokens > self.geometry.max_position_embeddings:
            raise GenerationError("prompt exceeds the model position limit")
        hidden = self.embed_tokens(tokens)
        positions = np.arange(n_tokens, dtype=np.int64)
        for layer_idx in range(self.geometry.num_layers):
            hidden = self._forward_layer(hidden, layer_idx, positions, decode=False)
            if progress:
                progress(
                    {
                        "phase": "prefill",
                        "layer": layer_idx,
                        "layers": self.geometry.num_layers,
                        "token": n_tokens,
                        "rss_mb": round(process_rss_mb(), 1),
                    }
                )
            if self.logger:
                self.logger.log(
                    "generation_prefill_layer",
                    layer=layer_idx,
                    tokens=n_tokens,
                    rss_mb=process_rss_mb(),
                )
        return hidden

    def decode_step(
        self,
        token: int,
        position: int,
        progress: Optional[Callable[[dict], None]] = None,
    ):
        hidden = self.embed_tokens([token])
        positions = np.asarray([position], dtype=np.int64)
        for layer_idx in range(self.geometry.num_layers):
            hidden = self._forward_layer(hidden, layer_idx, positions, decode=True)
            if progress:
                progress(
                    {
                        "phase": "decode",
                        "layer": layer_idx,
                        "layers": self.geometry.num_layers,
                        "position": position,
                        "rss_mb": round(process_rss_mb(), 1),
                    }
                )
            if self.logger:
                self.logger.log(
                    "generation_decode_layer",
                    layer=layer_idx,
                    position=position,
                    rss_mb=process_rss_mb(),
                )
        return hidden

    def logits(self, hidden_row) -> dict:
        """Greedy logits via the tied LM head, read in vocabulary chunks."""
        started = time.perf_counter()
        torch = self.torch
        hidden = hidden_row.reshape(1, -1)
        chunk_size = min(8192, max(2048, math.ceil(self.vocab_size / 32)))
        best_value = float("-inf")
        best_index = -1
        second_value = float("-inf")
        with self._weights() as handle:
            head = handle.get_slice(self.lm_head_key)
            for start in range(0, self.vocab_size, chunk_size):
                stop = min(self.vocab_size, start + chunk_size)
                weight = head[start:stop].to(torch.float32)
                values = torch.nn.functional.linear(hidden, weight)[0]
                chunk_index = int(torch.argmax(values))
                chunk_value = float(values[chunk_index])
                chunk_second = float("-inf")
                if stop - start > 1:
                    masked = values.clone()
                    masked[chunk_index] = float("-inf")
                    chunk_second = float(masked.max())
                if chunk_value > best_value:
                    second_value = max(second_value, best_value, chunk_second)
                    best_value = chunk_value
                    best_index = start + chunk_index
                else:
                    second_value = max(second_value, chunk_value, chunk_second)
                del weight, values
                if stop - start > 1:
                    del masked
        self.timings["logits_s"] += time.perf_counter() - started
        if best_index < 0:
            raise GenerationError("LM head produced no logits")
        return {
            "token": int(best_index),
            "top1_logit": round(best_value, 6),
            "top2_logit": round(second_value, 6) if second_value > float("-inf") else None,
            "margin": round(best_value - second_value, 6)
            if second_value > float("-inf")
            else None,
        }

    def final_norm(self, hidden):
        return rms_norm(hidden, self.norm_weight, self.geometry.rms_norm_eps)
