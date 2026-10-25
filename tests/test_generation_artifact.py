"""Manager-side validation of the distributed generation artifact."""
from __future__ import annotations

import hashlib
import json
import threading
import time

import pytest

from distributed_llm.decoder import GENERATION_METHOD
from distributed_llm.logging_utils import JsonlLogger, utc_now_iso
from distributed_llm.manager import (
    EventStream,
    GenerationFailure,
    RunConfig,
    RunManager,
    RunRecord,
)

PROMPT = "hello world"
PROMPT_SHA = hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()
TOKENS = 6


def _artifact(**overrides):
    payload = {
        "status": "ok",
        "method": GENERATION_METHOD,
        "prompt": {"sha256": PROMPT_SHA, "token_count": TOKENS},
        "output": {"token_count": 2, "text": " ok", "stop_reason": "max_new_tokens"},
    }
    payload.update(overrides)
    return payload


def _write(tmp_path, payload):
    (tmp_path / "generation.json").write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture()
def manager(tmp_path):
    return RunManager(runs_dir=tmp_path / "runs")


def test_valid_generation_artifact_is_accepted(manager, tmp_path):
    _write(tmp_path, _artifact())
    artifact = manager._load_generation_artifact(tmp_path, PROMPT_SHA, TOKENS, 4)
    assert artifact["output"]["token_count"] == 2


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"method": "ollama_forward"}, "distributed decoder"),
        ({"status": "error", "error_type": "GenerationError", "error": "boom"}, "boom"),
        ({"prompt": {"sha256": "0" * 64, "token_count": TOKENS}}, "hash"),
        ({"prompt": {"sha256": PROMPT_SHA, "token_count": TOKENS + 1}}, "tokenized"),
        ({"output": {"token_count": 0, "text": ""}}, "token count"),
        ({"output": {"token_count": 9, "text": ""}}, "token count"),
    ],
)
def test_invalid_generation_artifacts_are_rejected(manager, tmp_path, overrides, message):
    _write(tmp_path, _artifact(**overrides))
    with pytest.raises(GenerationFailure, match=message):
        manager._load_generation_artifact(tmp_path, PROMPT_SHA, TOKENS, 4)


def test_missing_or_malformed_artifact_is_rejected(manager, tmp_path):
    with pytest.raises(GenerationFailure, match="missing"):
        manager._load_generation_artifact(tmp_path, PROMPT_SHA, TOKENS, 4)
    (tmp_path / "generation.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(GenerationFailure, match="valid JSON"):
        manager._load_generation_artifact(tmp_path, PROMPT_SHA, TOKENS, 4)


def _record(tmp_path, generate: bool) -> RunRecord:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    return RunRecord(
        run_id="run-abcdefgh",
        config=RunConfig(prompt=PROMPT, workers=2, max_tokens=64, generate=generate),
        run_dir=run_dir,
        stream=EventStream(),
        logger=JsonlLogger(run_dir / "events.jsonl", run_id="run-abcdefgh"),
        started_utc=utc_now_iso(),
        started_monotonic=time.monotonic(),
        cancel=threading.Event(),
    )


def test_failure_summary_distinguishes_not_run_from_failed(manager, tmp_path):
    record = _record(tmp_path, generate=True)
    manager._write_failure_summary(
        record, time.monotonic(), "failed", "benchmark exploded", "RunManagerError"
    )
    summary = json.loads((record.run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["generation"]["status"] == "not_run"
    assert "benchmark exploded" in summary["generation"]["message"]

    _write(
        record.run_dir,
        {
            "status": "error",
            "computed_by": "exact decoder",
            "method": GENERATION_METHOD,
            "error": "worker refused the append",
        },
    )
    manager._write_failure_summary(
        record, time.monotonic(), "failed", "generation failed", "GenerationFailure"
    )
    summary = json.loads((record.run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["generation"]["status"] == "error"
    assert summary["generation"]["message"] == "worker refused the append"


def test_failure_summary_reports_disabled_generation(manager, tmp_path):
    record = _record(tmp_path, generate=False)
    manager._write_failure_summary(
        record, time.monotonic(), "failed", "capture failed", "CaptureFailure"
    )
    summary = json.loads((record.run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["generation"] == {"status": "disabled"}
