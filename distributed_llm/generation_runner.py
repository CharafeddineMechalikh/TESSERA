"""Isolated distributed generation runner.

This process owns tokenization and the decoder math but never owns a KV cache.
Attention is executed by the worker cluster described in LAB_GEN_CONNECTION.
The manager starts and stops the workers, streams this process's JSON progress,
and enforces the overall timeout and cancellation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .capture import load_model_metadata
from .config import LIMITS, MODEL_ID, MODEL_REVISION, Limits
from .decoder import (
    GENERATION_METHOD,
    GENERATION_PROVENANCE,
    DistributedAttentionBackend,
    GenerationError,
    StreamingQwen2Decoder,
)
from .gen_client import GenerationClient, GenerationClientError
from .logging_utils import (
    JsonlLogger,
    available_ram_gb,
    process_rss_mb,
    utc_now_iso,
)


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _emit(payload: dict) -> None:
    print(json.dumps(payload, allow_nan=False), flush=True)


def _tokenize(prompt: str, model_id: str, revision: str, model_path: Optional[str]):
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path or model_id,
            revision=revision if not model_path else None,
            local_files_only=True,
            trust_remote_code=False,
        )
    except Exception as exc:
        raise GenerationError(
            f"Tokenizer for {model_id}@{revision} is unavailable locally: {exc}"
        ) from exc
    tokens = [int(t) for t in tokenizer.encode(prompt, add_special_tokens=True)]
    return tokenizer, tokens


def generate(
    prompt: str,
    *,
    max_new_tokens: int,
    connection: dict,
    max_generation_positions: int = LIMITS.max_generation_positions,
    model_id: str = MODEL_ID,
    revision: str = MODEL_REVISION,
    model_path: Optional[str] = None,
    limits: Limits = LIMITS,
    expected_prompt_tokens: Optional[int] = None,
    logger: Optional[JsonlLogger] = None,
    progress: Optional[Callable[[dict], None]] = None,
) -> dict:
    started_wall = time.perf_counter()
    if not isinstance(prompt, str) or not prompt.strip():
        raise GenerationError("prompt must not be empty")
    if not 1 <= int(max_new_tokens) <= limits.max_generation_tokens:
        raise GenerationError(
            f"max_new_tokens must be between 1 and {limits.max_generation_tokens}"
        )
    if not 1 <= int(max_generation_positions) <= limits.max_generation_positions:
        raise GenerationError("max_generation_positions is outside the configured limit")

    ram_start = available_ram_gb()
    if 0 <= ram_start < limits.min_available_ram_gb:
        raise GenerationError(
            f"only {ram_start:.2f} GB RAM is available; at least "
            f"{limits.min_available_ram_gb:.2f} GB is required for generation"
        )

    tokenize_started = time.perf_counter()
    tokenizer, tokens = _tokenize(prompt, model_id, revision, model_path)
    tokenize_s = time.perf_counter() - tokenize_started
    if not tokens:
        raise GenerationError("tokenizer produced an empty prompt")
    if expected_prompt_tokens is not None and len(tokens) != int(expected_prompt_tokens):
        raise GenerationError(
            f"generation tokenized the prompt into {len(tokens)} tokens but capture "
            f"reported {expected_prompt_tokens}; refusing to decode a different context"
        )
    if len(tokens) > int(max_generation_positions) - int(max_new_tokens):
        raise GenerationError(
            f"prompt has {len(tokens)} tokens; with max_new_tokens={max_new_tokens} it "
            f"exceeds the {max_generation_positions}-token generation window"
        )

    geometry, root = load_model_metadata(model_id, revision, model_path)
    if len(tokens) > geometry.max_position_embeddings:
        raise GenerationError("prompt exceeds the model position limit")

    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    token_sha256 = hashlib.sha256(
        np.asarray(tokens, dtype=np.int64).tobytes()
    ).hexdigest()
    max_positions = len(tokens) + int(max_new_tokens)

    client = GenerationClient(connection, limits.worker_request_timeout_s)
    peak_rss_mb = process_rss_mb()

    def note_progress(event: dict) -> None:
        nonlocal peak_rss_mb
        peak_rss_mb = max(peak_rss_mb, process_rss_mb())
        if progress:
            progress(event)

    try:
        attention = DistributedAttentionBackend(client, geometry, limits)
        decoder = StreamingQwen2Decoder(
            str(root), geometry, attention, limits, logger=logger
        )
        if progress:
            progress(
                {
                    "phase": "prefill_start",
                    "prompt_tokens": len(tokens),
                    "max_positions": max_positions,
                    "workers": client.num_workers,
                }
            )
        attention.reset(max_positions)
        prefill_started = time.perf_counter()
        hidden = decoder.prefill(tokens, note_progress)
        prefill_s = time.perf_counter() - prefill_started
        hidden_last = hidden[-1]
        del hidden

        eos_token_id = tokenizer.eos_token_id
        output_ids = []
        steps = []
        stop_reason = "max_new_tokens"
        decode_s = 0.0
        for step in range(int(max_new_tokens)):
            step_started = time.perf_counter()
            if progress:
                progress(
                    {
                        "phase": "sampling",
                        "step": step + 1,
                        "of": int(max_new_tokens),
                        "position": len(tokens) + step,
                        "rss_mb": round(process_rss_mb(), 1),
                    }
                )
            normed = decoder.final_norm(hidden_last.reshape(1, -1))
            logits = decoder.logits(normed[0])
            peak_rss_mb = max(peak_rss_mb, process_rss_mb())
            token_id = int(logits["token"])
            output_ids.append(token_id)
            record = {
                "index": step,
                "position": len(tokens) + step,
                "token_id": token_id,
                "top1_logit": logits["top1_logit"],
                "top2_logit": logits["top2_logit"],
                "margin": logits["margin"],
            }
            is_eos = eos_token_id is not None and token_id == int(eos_token_id)
            record["elapsed_s"] = round(time.perf_counter() - step_started, 3)
            steps.append(record)
            if is_eos:
                stop_reason = "eos"
                break
            if step == int(max_new_tokens) - 1:
                break
            decode_started = time.perf_counter()
            hidden_next = decoder.decode_step(
                token_id, len(tokens) + step, note_progress
            )
            decode_s += time.perf_counter() - decode_started
            hidden_last = hidden_next[0]
            del hidden_next

        text = tokenizer.decode(output_ids, skip_special_tokens=True)
        worker_status = client.status()
        transport = client.summary()
        attention_summary = attention.summary()
        available_end = available_ram_gb()
    finally:
        client.close()

    total_s = time.perf_counter() - started_wall
    result = {
        "status": "ok",
        "computed_by": GENERATION_PROVENANCE,
        "method": GENERATION_METHOD,
        "created_utc": utc_now_iso(),
        "model": {
            "model_id": model_id,
            "revision": revision,
            "resolved": geometry.model_id,
            "geometry": geometry.as_dict(),
        },
        "prompt": {
            "sha256": prompt_sha256,
            "chars": len(prompt),
            "token_count": len(tokens),
            "token_ids_sha256": token_sha256,
            "tokenizer_class": type(tokenizer).__name__,
        },
        "context": {
            "max_generation_positions": int(max_generation_positions),
            "max_positions": max_positions,
            "max_new_tokens": int(max_new_tokens),
        },
        "output": {
            "token_ids": output_ids,
            "text": text,
            "token_count": len(output_ids),
            "stop_reason": stop_reason,
            "steps": steps,
        },
        "timings": {
            "total_s": round(total_s, 3),
            "tokenize_s": round(tokenize_s, 3),
            "prefill_s": round(prefill_s, 3),
            "decode_s": round(decode_s, 3),
            "mean_decode_s": round(decode_s / max(1, len(steps) - 1), 3),
        },
        "decoder": {
            key: round(value, 3) if isinstance(value, float) else value
            for key, value in decoder.timings.items()
        },
        "attention": attention_summary,
        "transport": transport,
        "workers": worker_status,
        "memory": {
            "available_ram_start_gb": round(ram_start, 3),
            "available_ram_end_gb": round(available_end, 3),
            "process_rss_mb": round(process_rss_mb(), 1),
            "peak_rss_mb": round(peak_rss_mb, 1),
        },
        "determinism": {
            "decoding": "greedy_argmax",
            "temperature": 0.0,
            "sampling": False,
        },
    }
    return result


def _failure(exc: Exception, started: float) -> dict:
    return {
        "status": "error",
        "computed_by": GENERATION_PROVENANCE,
        "method": GENERATION_METHOD,
        "created_utc": utc_now_iso(),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "timing_s": round(time.perf_counter() - started, 3),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="distributed generation runner")
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--max-generation-positions",
        type=int,
        default=LIMITS.max_generation_positions,
    )
    parser.add_argument("--expected-prompt-tokens", type=int, default=None)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--model-path", default=None)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    out_path = Path(args.out)
    try:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8")
        raw = os.environ.get("LAB_GEN_CONNECTION", "")
        if not raw:
            raise GenerationError("LAB_GEN_CONNECTION is not set")
        try:
            connection = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise GenerationError("LAB_GEN_CONNECTION is not valid JSON") from exc
        result = generate(
            prompt,
            max_new_tokens=args.max_new_tokens,
            connection=connection,
            max_generation_positions=args.max_generation_positions,
            model_id=args.model_id,
            revision=args.revision,
            model_path=args.model_path,
            expected_prompt_tokens=args.expected_prompt_tokens,
            progress=lambda event: _emit({"stage": "generation", **event}),
        )
        _write_json_atomic(out_path, result)
        _emit(
            {
                "stage": "done",
                "status": "ok",
                "token_count": result["output"]["token_count"],
                "stop_reason": result["output"]["stop_reason"],
                "timing_s": result["timings"]["total_s"],
                "text": result["output"]["text"][:8000],
                "prefill_s": result["timings"]["prefill_s"],
                "decode_s": result["timings"]["decode_s"],
            }
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - the manager needs a typed failure
        failure = _failure(exc, started)
        try:
            _write_json_atomic(out_path, failure)
        except OSError:
            pass
        traceback.print_exc(limit=8, file=sys.stderr)
        _emit(
            {
                "stage": "error",
                "message": str(exc),
                "error_type": type(exc).__name__,
            }
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
