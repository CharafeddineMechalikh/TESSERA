"""Planted-needle captures for stress-testing certified pruning.

These captures are synthetic. They write the same npz and metadata schema as
capture_runner, so the benchmark, the worker cluster, the certificate, and both
controls run unchanged. A needle is a key placed parallel to the query, so its
score is exactly ``log(N) + margin``. Every other key is small isotropic noise.
That makes block separability a known property of the generator, which is the
point: this exercises the pruning mechanism, not attention on language.

A needle-free capture from the same generator is the control. It has the same
shapes, value scale, and block geometry, but no token is deliberately far above
the rest, so a certificate that accepts it is not just detecting the plant.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

SCENARIO_KIND = "planted-needle"
MODEL_ID = "synthetic/planted-needle"
REVISION = "v1"


@dataclass(frozen=True)
class NeedleScenario:
    n_tokens: int = 64
    layers: Tuple[int, ...] = (0, 1)
    kv_heads: Tuple[int, ...] = (0, 1)
    q_per_kv: int = 2
    head_dim: int = 8
    needle_count: int = 1
    needle_margin: float = 6.0
    background_scale: float = 0.05
    value_scale: float = 0.05
    seed: int = 11

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.n_tokens < 2:
            raise ValueError("n_tokens must be at least 2")
        if not self.layers or not self.kv_heads:
            raise ValueError("layers and kv_heads must not be empty")
        if min(self.layers) < 0 or min(self.kv_heads) < 0:
            raise ValueError("layer and kv_head indices must be non-negative")
        if len(set(self.layers)) != len(self.layers):
            raise ValueError("layers must be unique")
        if len(set(self.kv_heads)) != len(self.kv_heads):
            raise ValueError("kv_heads must be unique")
        if self.q_per_kv < 1 or self.head_dim < 1:
            raise ValueError("q_per_kv and head_dim must be at least 1")
        if not 0 <= self.needle_count <= self.n_tokens:
            raise ValueError("needle_count must be between 0 and n_tokens")
        if not math.isfinite(self.needle_margin):
            raise ValueError("needle_margin must be finite")
        if not math.isfinite(self.background_scale) or self.background_scale < 0.0:
            raise ValueError("background_scale must be finite and non-negative")
        if not math.isfinite(self.value_scale) or self.value_scale <= 0.0:
            raise ValueError("value_scale must be finite and positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")

    @property
    def target_score(self) -> float:
        return math.log(self.n_tokens) + self.needle_margin

    def needle_positions(self) -> np.ndarray:
        if self.needle_count == 0:
            return np.empty(0, dtype=np.int64)
        offsets = (np.arange(self.needle_count, dtype=np.float64) + 0.5) * (
            self.n_tokens / self.needle_count
        )
        return np.floor(offsets).astype(np.int64)

    def expected_needle_mass(self) -> float:
        """Expected softmax mass on the needles, with background weight 1.

        Background scores are zero-mean noise with scale ``background_scale``, so
        their expected weight is close to 1. The estimate is analytic and is used
        only to state what the generator intends; the test compares it with the
        dense attention computed from the actual arrays.
        """
        if self.needle_count == 0:
            return 0.0
        needle_weight = self.n_tokens * math.exp(self.needle_margin)
        total = self.needle_count * needle_weight
        return total / (total + (self.n_tokens - self.needle_count))

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["layers"] = list(self.layers)
        payload["kv_heads"] = list(self.kv_heads)
        payload["kind"] = SCENARIO_KIND
        payload["synthetic"] = True
        payload["target_score"] = self.target_score
        payload["expected_needle_mass"] = self.expected_needle_mass()
        payload["needle_positions"] = self.needle_positions().tolist()
        return payload


def build_capture(scenario: NeedleScenario) -> Tuple[Dict[str, np.ndarray], dict]:
    scenario.validate()
    rng = np.random.default_rng(scenario.seed)
    positions = scenario.needle_positions()
    target = scenario.target_score
    arrays: Dict[str, np.ndarray] = {
        "tokens": np.arange(scenario.n_tokens, dtype=np.int64),
    }
    for layer in scenario.layers:
        for kv_head in scenario.kv_heads:
            direction = rng.normal(size=scenario.head_dim)
            norm = float(np.linalg.norm(direction))
            if norm == 0.0:
                direction = np.zeros(scenario.head_dim, dtype=np.float64)
                direction[0] = 1.0
                norm = 1.0
            direction = direction / norm
            query = np.tile(direction * math.sqrt(scenario.head_dim), (scenario.q_per_kv, 1))
            keys = rng.normal(0.0, scenario.background_scale, size=(scenario.n_tokens, scenario.head_dim))
            values = rng.normal(0.0, scenario.value_scale, size=(scenario.n_tokens, scenario.head_dim))
            keys[positions] = target * direction
            arrays[f"q_{layer}_{kv_head}"] = query
            arrays[f"k_{layer}_{kv_head}"] = keys
            arrays[f"v_{layer}_{kv_head}"] = values

    geometry = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "num_layers": len(scenario.layers),
        "num_attention_heads": len(scenario.kv_heads) * scenario.q_per_kv,
        "num_key_value_heads": len(scenario.kv_heads),
        "head_dim": scenario.head_dim,
        "hidden_size": len(scenario.kv_heads) * scenario.q_per_kv * scenario.head_dim,
        "rope_theta": 1_000_000.0,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": scenario.n_tokens,
        "vocab_size": scenario.n_tokens,
    }
    scenario_payload = scenario.as_dict()
    metadata = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "model_path": "synthetic",
        "geometry": geometry,
        "tokens": list(range(scenario.n_tokens)),
        "token_count": scenario.n_tokens,
        "original_token_count": scenario.n_tokens,
        "truncated": False,
        "prompt_sha256": hashlib.sha256(
            json.dumps(scenario_payload, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "tokenizer_class": "SyntheticPlantedNeedle",
        "layers": list(scenario.layers),
        "kv_heads": list(scenario.kv_heads),
        "capture_timing_s": 0.0,
        "capture_peak_rss_mb": 0.0,
        "scenario": scenario_payload,
    }
    return arrays, metadata


def write_capture(scenario: NeedleScenario, out_dir: Path) -> dict:
    arrays, metadata = build_capture(scenario)
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "capture.npz"
    meta_path = out_dir / "capture.json"
    with open(npz_path, "wb") as handle:
        np.savez(handle, **arrays)
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "capture": str(npz_path),
        "meta": str(meta_path),
        "scenario": metadata["scenario"],
    }


def _int_tuple(value: str) -> Tuple[int, ...]:
    return tuple(int(part) for part in value.split(",") if part.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a synthetic planted-needle capture")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--layers", default="0,1")
    parser.add_argument("--kv-heads", default="0,1")
    parser.add_argument("--q-per-kv", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=8)
    parser.add_argument("--needles", type=int, default=1)
    parser.add_argument("--margin", type=float, default=6.0)
    parser.add_argument("--background-scale", type=float, default=0.05)
    parser.add_argument("--value-scale", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args(argv)
    scenario = NeedleScenario(
        n_tokens=args.tokens,
        layers=_int_tuple(args.layers),
        kv_heads=_int_tuple(args.kv_heads),
        q_per_kv=args.q_per_kv,
        head_dim=args.head_dim,
        needle_count=args.needles,
        needle_margin=args.margin,
        background_scale=args.background_scale,
        value_scale=args.value_scale,
        seed=args.seed,
    )
    print(json.dumps(write_capture(scenario, Path(args.out_dir)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
