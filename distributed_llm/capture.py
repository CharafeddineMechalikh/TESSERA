"""Exact Qwen2 attention capture with layer-streamed fp32 weights.

The full 1.5B model in fp32 does not fit this machine's RAM. We therefore run the
same forward computation Hugging Face uses, but load one decoder layer from the
safetensors file at a time and free it after use. The implementation is tested
against Hugging Face on a tiny Qwen2 model and against one real decoder layer.
"""
from __future__ import annotations

import gc
import hashlib
import math
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .config import LIMITS, MODEL_ID, MODEL_REVISION, Limits
from .logging_utils import JsonlLogger, available_ram_gb, process_rss_mb

try:
    from huggingface_hub import snapshot_download
except ImportError:  # pragma: no cover
    snapshot_download = None


class ModelUnavailableError(RuntimeError):
    pass


class InsufficientMemoryError(RuntimeError):
    pass


class CaptureError(RuntimeError):
    pass


@dataclass(frozen=True)
class Geometry:
    model_id: str
    revision: str
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    hidden_size: int
    rope_theta: float
    rms_norm_eps: float
    max_position_embeddings: int
    vocab_size: int

    @property
    def q_per_kv(self) -> int:
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise CaptureError("attention heads are not divisible by KV heads")
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def num_cases(self) -> int:
        return self.num_layers * self.num_key_value_heads * self.q_per_kv

    def kv_for_q_head(self, q_head: int) -> int:
        return q_head // self.q_per_kv

    def q_heads_for_kv(self, kv_head: int) -> List[int]:
        start = kv_head * self.q_per_kv
        return list(range(start, start + self.q_per_kv))

    def as_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "hidden_size": self.hidden_size,
            "rope_theta": self.rope_theta,
            "rms_norm_eps": self.rms_norm_eps,
            "max_position_embeddings": self.max_position_embeddings,
            "vocab_size": self.vocab_size,
            "q_per_kv": self.q_per_kv,
        }


@dataclass
class CapturedHead:
    layer: int
    kv_head: int
    q_heads: List[int]
    q_last: np.ndarray  # (q_per_kv, head_dim), float64
    keys: np.ndarray  # (N, head_dim), float64
    values: np.ndarray  # (N, head_dim), float64


@dataclass
class CaptureResult:
    tokens: List[int]
    geometry: Geometry
    heads: Dict[Tuple[int, int], CapturedHead]
    timing_s: float
    layers: List[int]
    kv_heads: List[int]
    peak_rss_mb: float
    available_ram_start_gb: float
    model_path: str
    last_attention: Dict[int, np.ndarray] = field(default_factory=dict)
    hidden_states: Dict[int, np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True)
class TokenizedPrompt:
    tokens: List[int]
    original_token_count: int
    truncated: bool
    tokenizer_class: str
    prompt_sha256: str


def _torch():
    import torch

    return torch


def _resolve_snapshot(model_id: str, revision: str) -> Tuple[Path, str]:
    if snapshot_download is None:
        raise ModelUnavailableError("huggingface_hub is not installed")
    try:
        path = snapshot_download(
            repo_id=model_id,
            revision=revision,
            local_files_only=True,
            allow_patterns=["config.json", "model.safetensors", "tokenizer*", "*.json", "*.txt"],
        )
    except Exception as exc:
        raise ModelUnavailableError(
            f"Model {model_id}@{revision} is not available in the local Hugging Face cache. "
            f"Run: huggingface-cli download {model_id} --revision {revision}"
        ) from exc
    snapshot = Path(path)
    if not (snapshot / "config.json").exists() or not (snapshot / "model.safetensors").exists():
        raise ModelUnavailableError(
            f"Snapshot {snapshot} is incomplete; config.json and model.safetensors are required."
        )
    resolved = snapshot.name
    return snapshot, resolved


def geometry_from_config(config, model_id: str, revision: str) -> Geometry:
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        head_dim = config.hidden_size // config.num_attention_heads
    return Geometry(
        model_id=model_id,
        revision=revision,
        num_layers=int(config.num_hidden_layers),
        num_attention_heads=int(config.num_attention_heads),
        num_key_value_heads=int(config.num_key_value_heads),
        head_dim=int(head_dim),
        hidden_size=int(config.hidden_size),
        rope_theta=float(getattr(config, "rope_theta", 10000.0)),
        rms_norm_eps=float(config.rms_norm_eps),
        max_position_embeddings=int(config.max_position_embeddings),
        vocab_size=int(config.vocab_size),
    )


def load_model_metadata(
    model_id: str = MODEL_ID,
    revision: str = MODEL_REVISION,
    model_path: Optional[str] = None,
) -> Tuple[Geometry, Path]:
    from transformers import AutoConfig

    if model_path:
        root = Path(model_path)
        resolved = "local:" + hashlib.sha256(str(root).encode()).hexdigest()[:12]
    else:
        root, resolved = _resolve_snapshot(model_id, revision)
    try:
        config = AutoConfig.from_pretrained(str(root), local_files_only=True, trust_remote_code=False)
    except Exception as exc:
        raise ModelUnavailableError(f"Cannot load model config from {root}: {exc}") from exc
    return geometry_from_config(config, model_id, resolved), root


def tokenize_prompt(
    prompt: str,
    max_tokens: int,
    model_id: str = MODEL_ID,
    revision: str = MODEL_REVISION,
    model_path: Optional[str] = None,
) -> TokenizedPrompt:
    from transformers import AutoTokenizer

    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path or model_id,
            revision=revision if not model_path else None,
            local_files_only=True,
            trust_remote_code=False,
        )
    except Exception as exc:
        raise ModelUnavailableError(
            f"Tokenizer for {model_id}@{revision} is unavailable locally: {exc}"
        ) from exc
    tokens = tokenizer.encode(prompt, add_special_tokens=True)
    original = len(tokens)
    truncated = original > max_tokens
    tokens = tokens[:max_tokens]
    return TokenizedPrompt(
        tokens=[int(t) for t in tokens],
        original_token_count=original,
        truncated=truncated,
        tokenizer_class=type(tokenizer).__name__,
        prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
    )


@contextmanager
def open_weights(path: Path):
    from safetensors import safe_open

    handle = safe_open(str(path), framework="pt", device="cpu")
    try:
        yield handle
    finally:
        del handle
        gc.collect()


def load_layer_weights(handle, layer_idx: int, torch) -> Dict[str, object]:
    prefix = f"model.layers.{layer_idx}"
    names = {
        "input_norm": ("input_layernorm.weight", True),
        "post_norm": ("post_attention_layernorm.weight", True),
        "q_w": ("self_attn.q_proj.weight", True),
        "q_b": ("self_attn.q_proj.bias", True),
        "k_w": ("self_attn.k_proj.weight", True),
        "k_b": ("self_attn.k_proj.bias", True),
        "v_w": ("self_attn.v_proj.weight", True),
        "v_b": ("self_attn.v_proj.bias", True),
        "o_w": ("self_attn.o_proj.weight", True),
        "o_b": ("self_attn.o_proj.bias", False),
        "gate_w": ("mlp.gate_proj.weight", True),
        "up_w": ("mlp.up_proj.weight", True),
        "down_w": ("mlp.down_proj.weight", True),
    }
    available = set(handle.keys())
    weights: Dict[str, object] = {}
    for short, (suffix, required) in names.items():
        key = f"{prefix}.{suffix}"
        if key in available:
            weights[short] = handle.get_tensor(key).to(torch.float32)
        elif required:
            raise CaptureError(f"missing tensor {key}")
        else:
            weights[short] = None
    return weights


def rms_norm(x, weight, eps: float):
    xf = x.to(__import__("torch").float32)
    return (xf * __import__("torch").rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)) * weight


def build_rope(seq_len: int, head_dim: int, base: float, device, dtype):
    torch = _torch()
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, head_dim, 2, dtype=torch.int64, device=device).float()
            / head_dim
        )
    )
    positions = torch.arange(seq_len, device=device, dtype=torch.int64).float()
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


def rotate_half(x):
    half = x.shape[-1] // 2
    return __import__("torch").cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(x, cos, sin):
    cos = cos.unsqueeze(0)
    sin = sin.unsqueeze(0)
    return (x * cos) + (rotate_half(x) * sin)


def causal_mask(seq_len: int, dtype):
    torch = _torch()
    mask = torch.full(
        (seq_len, seq_len),
        torch.finfo(dtype).min,
        dtype=dtype,
        device="cpu",
    )
    return torch.triu(mask, diagonal=1)


def qwen2_layer_forward(
    hidden: object,
    weights: Dict[str, object],
    geometry: Geometry,
    cos,
    sin,
    mask,
    capture_heads: Optional[Sequence[int]] = None,
    capture_last_attention: bool = False,
):
    """Run one decoder layer. Shapes follow Hugging Face Qwen2 exactly."""
    torch = _torch()
    n_tokens = hidden.shape[0]
    q_per_kv = geometry.q_per_kv

    linear = torch.nn.functional.linear
    normed = rms_norm(hidden, weights["input_norm"], geometry.rms_norm_eps)
    q = linear(normed, weights["q_w"], weights["q_b"])
    k = linear(normed, weights["k_w"], weights["k_b"])
    v = linear(normed, weights["v_w"], weights["v_b"])

    q = q.view(n_tokens, geometry.num_attention_heads, geometry.head_dim).transpose(0, 1)
    k = k.view(n_tokens, geometry.num_key_value_heads, geometry.head_dim).transpose(0, 1)
    v = v.view(n_tokens, geometry.num_key_value_heads, geometry.head_dim).transpose(0, 1)

    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)

    captured = {}
    if capture_heads is not None:
        for kv_head in capture_heads:
            start = kv_head * q_per_kv
            stop = start + q_per_kv
            captured[kv_head] = {
                "q_last": q[start:stop, -1, :].clone(),
                "keys": k[kv_head].clone(),
                "values": v[kv_head].clone(),
            }

    k_rep = k.repeat_interleave(q_per_kv, dim=0)
    v_rep = v.repeat_interleave(q_per_kv, dim=0)
    scores = torch.matmul(q, k_rep.transpose(-2, -1)) / math.sqrt(geometry.head_dim)
    scores = scores + mask
    probs = torch.softmax(scores, dim=-1, dtype=torch.float32)
    if capture_last_attention:
        captured["last_attention"] = probs[:, -1, :].clone()

    attn = torch.matmul(probs, v_rep)
    attn = attn.transpose(0, 1).contiguous().view(n_tokens, geometry.hidden_size)
    hidden = hidden + linear(attn, weights["o_w"], weights["o_b"])

    normed2 = rms_norm(hidden, weights["post_norm"], geometry.rms_norm_eps)
    gate = linear(normed2, weights["gate_w"], None)
    up = linear(normed2, weights["up_w"], None)
    hidden = hidden + linear(torch.nn.functional.silu(gate) * up, weights["down_w"], None)
    return hidden, captured


def _to_numpy(tensor) -> np.ndarray:
    return tensor.detach().to("cpu", __import__("torch").float64).numpy().copy()


def capture_attention(
    prompt_tokens: Sequence[int],
    layers: Optional[Sequence[int]] = None,
    kv_heads: Optional[Sequence[int]] = None,
    limits: Limits = LIMITS,
    model_id: str = MODEL_ID,
    revision: str = MODEL_REVISION,
    model_path: Optional[str] = None,
    capture_last_attention: bool = False,
    capture_hidden_states: bool = False,
    progress: Optional[Callable[[dict], None]] = None,
    logger: Optional[JsonlLogger] = None,
) -> CaptureResult:
    tokens = [int(t) for t in prompt_tokens]
    if not tokens:
        raise CaptureError("token sequence is empty")
    if len(tokens) > limits.max_context_tokens:
        raise CaptureError(
            f"token sequence has {len(tokens)} tokens, limit is {limits.max_context_tokens}"
        )

    ram_start = available_ram_gb()
    if 0 <= ram_start < limits.min_available_ram_gb:
        raise InsufficientMemoryError(
            f"Only {ram_start:.2f} GB RAM is available; at least "
            f"{limits.min_available_ram_gb:.2f} GB is required for capture."
        )

    geometry, root = load_model_metadata(model_id, revision, model_path)
    if len(tokens) > geometry.max_position_embeddings:
        raise CaptureError("token sequence exceeds the model's position embedding limit")

    selected_layers = sorted(set(int(x) for x in (layers if layers is not None else range(geometry.num_layers))))
    selected_kv = sorted(set(int(x) for x in (kv_heads if kv_heads is not None else range(geometry.num_key_value_heads))))
    if not selected_layers or selected_layers[0] < 0 or selected_layers[-1] >= geometry.num_layers:
        raise CaptureError("layer selection is out of range")
    if not selected_kv or selected_kv[0] < 0 or selected_kv[-1] >= geometry.num_key_value_heads:
        raise CaptureError("KV head selection is out of range")

    torch = _torch()
    torch.set_grad_enabled(False)
    threads = min(8, os.cpu_count() or 1)
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    ram_after_import = available_ram_gb()
    if 0 <= ram_after_import < limits.min_available_ram_gb:
        raise InsufficientMemoryError(
            f"Only {ram_after_import:.2f} GB RAM remains after loading PyTorch; "
            f"close applications or raise LAB_MIN_AVAILABLE_RAM_GB carefully."
        )

    weights_path = root / "model.safetensors"
    start = time.perf_counter()
    heads: Dict[Tuple[int, int], CapturedHead] = {}
    last_attention: Dict[int, np.ndarray] = {}
    hidden_states: Dict[int, np.ndarray] = {}

    try:
        with torch.inference_mode():
            with open_weights(weights_path) as handle:
                embedding_slice = handle.get_slice("model.embed_tokens.weight")
                row_cache: Dict[int, object] = {}
                hidden = torch.empty((len(tokens), geometry.hidden_size), dtype=torch.float32)
                for row, token in enumerate(tokens):
                    cached = row_cache.get(token)
                    if cached is None:
                        if token < 0 or token >= geometry.vocab_size:
                            raise CaptureError(f"token id {token} is outside the vocabulary")
                        cached = embedding_slice[token : token + 1].to(torch.float32)
                        row_cache[token] = cached
                    hidden[row] = cached[0]
                del row_cache

            cos, sin = build_rope(
                len(tokens), geometry.head_dim, geometry.rope_theta, "cpu", torch.float32
            )
            mask = causal_mask(len(tokens), torch.float32)

            for position, layer_idx in enumerate(selected_layers):
                # Reopen per layer so the memory-mapped file pages for previous
                # layers can be released; a persistent handle keeps every layer resident.
                with open_weights(weights_path) as layer_handle:
                    weights = load_layer_weights(layer_handle, layer_idx, torch)
                hidden, captured = qwen2_layer_forward(
                    hidden,
                    weights,
                    geometry,
                    cos,
                    sin,
                    mask,
                    capture_heads=selected_kv,
                    capture_last_attention=capture_last_attention,
                )
                for kv_head in selected_kv:
                    block = captured[kv_head]
                    heads[(layer_idx, kv_head)] = CapturedHead(
                        layer=layer_idx,
                        kv_head=kv_head,
                        q_heads=geometry.q_heads_for_kv(kv_head),
                        q_last=_to_numpy(block["q_last"]),
                        keys=_to_numpy(block["keys"]),
                        values=_to_numpy(block["values"]),
                    )
                if capture_last_attention:
                    last_attention[layer_idx] = _to_numpy(captured["last_attention"])
                if capture_hidden_states:
                    hidden_states[layer_idx] = _to_numpy(hidden)

                del weights, captured
                gc.collect()
                if progress:
                    progress(
                        {
                            "stage": "capture",
                            "layer": layer_idx,
                            "position": position + 1,
                            "selected_layers": len(selected_layers),
                            "rss_mb": process_rss_mb(),
                        }
                    )
                if logger:
                    logger.log(
                        "capture_layer",
                        layer=layer_idx,
                        position=position + 1,
                        total=len(selected_layers),
                        rss_mb=process_rss_mb(),
                    )
    except (MemoryError, RuntimeError) as exc:
        message = str(exc)
        if "out of memory" in message.lower() or isinstance(exc, MemoryError):
            raise InsufficientMemoryError(
                "Capture ran out of memory. Close other applications or lower "
                "LAB_MAX_CONTEXT_TOKENS before retrying."
            ) from exc
        raise

    elapsed = round(time.perf_counter() - start, 2)
    return CaptureResult(
        tokens=tokens,
        geometry=geometry,
        heads=heads,
        timing_s=elapsed,
        layers=selected_layers,
        kv_heads=selected_kv,
        peak_rss_mb=process_rss_mb(),
        available_ram_start_gb=ram_start,
        model_path=str(root),
        last_attention=last_attention,
        hidden_states=hidden_states,
    )
