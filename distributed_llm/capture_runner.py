"""Subprocess entry point for exact attention capture.

Running capture in its own process keeps PyTorch out of the web server and worker
processes, so model memory is returned to the OS before the cluster starts.

Protocol: JSON lines on stdout. The final line is either
{"stage": "done", ...} or {"stage": "error", ...}.
"""
from __future__ import annotations

import argparse
import datetime as dt
import faulthandler
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

from .capture import (
    CaptureError,
    InsufficientMemoryError,
    ModelUnavailableError,
    capture_attention,
    tokenize_prompt,
)
from .config import APP_VERSION, LIMITS, MODEL_ID, MODEL_REVISION


def _parse_layers(raw: str) -> Optional[List[int]]:
    if raw.strip().lower() in {"", "all"}:
        return None
    return [int(part) for part in raw.split(",") if part.strip()]


def _emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _write_atomic(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    writer(temporary)
    os.replace(temporary, path)


def _tokenize(args: argparse.Namespace) -> dict:
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8")
        if len(prompt) > LIMITS.max_prompt_chars:
            raise ValueError(
                f"prompt has {len(prompt)} characters, limit is {LIMITS.max_prompt_chars}"
            )
        if not 1 <= args.max_tokens <= LIMITS.max_context_tokens:
            raise ValueError(
                f"max tokens must be between 1 and {LIMITS.max_context_tokens}"
            )
        try:
            tokenized = tokenize_prompt(
                prompt,
                args.max_tokens,
                model_id=args.model_id,
                revision=args.revision,
                model_path=args.model_path,
            )
        except ModelUnavailableError:
            raise
        return {
            "tokens": tokenized.tokens,
            "original_token_count": tokenized.original_token_count,
            "truncated": tokenized.truncated,
            "tokenizer_class": tokenized.tokenizer_class,
            "prompt_sha256": tokenized.prompt_sha256,
        }
    token_data = json.loads(Path(args.tokens_json).read_text(encoding="utf-8"))
    tokens = [int(t) for t in token_data["tokens"]]
    if not tokens:
        raise ValueError("token sequence is empty")
    if len(tokens) > LIMITS.max_context_tokens:
        raise ValueError(
            f"token sequence has {len(tokens)} tokens, limit is {LIMITS.max_context_tokens}"
        )
    return token_data


def run(args: argparse.Namespace) -> int:
    try:
        token_data = _tokenize(args)
    except ModelUnavailableError as exc:
        _emit({"stage": "error", "error_type": "model_unavailable", "message": str(exc)})
        return 3
    except (ValueError, OSError) as exc:
        _emit({"stage": "error", "error_type": "invalid_input", "message": str(exc)})
        return 2
    if token_data.get("truncated") and not args.allow_truncation:
        original = int(token_data.get("original_token_count", 0))
        _emit(
            {
                "stage": "error",
                "error_type": "prompt_too_long",
                "message": (
                    f"prompt tokenizes to {original} tokens, above the configured "
                    f"limit of {args.max_tokens}. Shorten the prompt or raise the "
                    f"context token limit. Truncation is disabled so the captured "
                    f"query is always the real final prompt token."
                ),
            }
        )
        return 2
    tokens = [int(t) for t in token_data["tokens"]]
    layers = _parse_layers(args.layers)
    kv_heads = _parse_layers(args.kv_heads)
    if args.tokens_out:
        _write_atomic(
            Path(args.tokens_out),
            lambda temporary: temporary.write_text(
                json.dumps(
                    {
                        "tokens": tokens,
                        "original_token_count": int(
                            token_data.get("original_token_count", len(tokens))
                        ),
                        "truncated": bool(token_data.get("truncated", False)),
                        "tokenizer_class": token_data.get("tokenizer_class"),
                        "prompt_sha256": token_data.get("prompt_sha256"),
                    }
                ),
                encoding="utf-8",
            ),
        )

    def progress(event: dict) -> None:
        _emit({"stage": "capture", **event})

    try:
        result = capture_attention(
            tokens,
            layers=layers,
            kv_heads=kv_heads,
            model_id=args.model_id,
            revision=args.revision,
            model_path=args.model_path,
            capture_last_attention=False,
            capture_hidden_states=False,
            progress=progress,
        )
    except InsufficientMemoryError as exc:
        _emit({"stage": "error", "error_type": "insufficient_memory", "message": str(exc)})
        return 4
    except ModelUnavailableError as exc:
        _emit({"stage": "error", "error_type": "model_unavailable", "message": str(exc)})
        return 3
    except (CaptureError, ValueError) as exc:
        _emit({"stage": "error", "error_type": "capture_failed", "message": str(exc)})
        return 5

    arrays = {"tokens": np.asarray(tokens, dtype=np.int64)}
    for (layer, kv_head), head in sorted(result.heads.items()):
        arrays[f"q_{layer}_{kv_head}"] = np.asarray(head.q_last, dtype=np.float64)
        arrays[f"k_{layer}_{kv_head}"] = np.asarray(head.keys, dtype=np.float64)
        arrays[f"v_{layer}_{kv_head}"] = np.asarray(head.values, dtype=np.float64)

    output_path = Path(args.out)

    def write_npz(temporary: Path) -> None:
        with open(temporary, "wb") as handle:
            np.savez(handle, **arrays)

    _write_atomic(output_path, write_npz)

    metadata = {
        "app_version": APP_VERSION,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model_id": args.model_id,
        "revision": args.revision,
        "model_path": result.model_path,
        "geometry": result.geometry.as_dict(),
        "tokens": tokens,
        "token_count": len(tokens),
        "original_token_count": int(token_data.get("original_token_count", len(tokens))),
        "truncated": bool(token_data.get("truncated", False)),
        "tokenizer_class": token_data.get("tokenizer_class"),
        "prompt_sha256": token_data.get("prompt_sha256"),
        "layers": result.layers,
        "kv_heads": result.kv_heads,
        "capture_timing_s": result.timing_s,
        "capture_peak_rss_mb": result.peak_rss_mb,
        "available_ram_start_gb": result.available_ram_start_gb,
        "arrays": sorted(arrays),
    }
    meta_path = Path(args.meta)
    _write_atomic(
        meta_path,
        lambda temporary: temporary.write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        ),
    )
    _emit(
        {
            "stage": "done",
            "capture_path": str(output_path),
            "meta_path": str(meta_path),
            "timing_s": result.timing_s,
            "token_count": len(tokens),
            "layers": len(result.layers),
            "kv_heads": len(result.kv_heads),
        }
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture real Qwen2 attention tensors")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--tokens-json")
    source.add_argument("--prompt-file")
    parser.add_argument("--max-tokens", type=int, default=LIMITS.default_context_tokens)
    parser.add_argument("--tokens-out", default=None)
    parser.add_argument(
        "--allow-truncation",
        action="store_true",
        help="keep the leading tokens when the prompt exceeds --max-tokens",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--kv-heads", default="all")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--model-path", default=None)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    faulthandler.enable()
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
