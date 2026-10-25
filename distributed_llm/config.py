"""Central configuration. Every limit is explicit and environment-overridable."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(os.getenv("LAB_OUTPUT_DIR", str(BASE_DIR / "output"))).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = os.getenv("LAB_MODEL_ID", "Qwen/Qwen2.5-Coder-1.5B")
MODEL_REVISION = os.getenv(
    "LAB_MODEL_REVISION", "df3ce67c0e24480f20468b6ef2894622d69eb73b"
)
# Pinned snapshot config.json: max_position_embeddings.
MAX_MODEL_POSITIONS = 32768

RUNS_DIR = OUTPUT_DIR / "runs"

# Empty means "single-user local mode": the server issues signed session
# cookies to loopback clients and no API key is required.
API_KEY = os.getenv("LAB_API_KEY", "")
SESSION_SECRET = os.getenv("LAB_SESSION_SECRET", "")

APP_VERSION = "11.4.0"


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be a float") from exc


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


STORE_PROMPTS = _bool("LAB_STORE_PROMPTS", False)
ALLOWED_HOSTS = tuple(
    host.strip()
    for host in os.getenv(
        "LAB_ALLOWED_HOSTS", "127.0.0.1,localhost,::1"
    ).split(",")
    if host.strip()
)


@dataclass(frozen=True)
class Limits:
    min_workers: int = 2
    max_workers: int = 8
    default_workers: int = 4
    max_overlap_pct: float = 50.0
    max_context_tokens: int = 1024
    default_context_tokens: int = 512
    max_prompt_chars: int = 12_000
    max_query_heads: int = 512
    max_sessions: int = 8
    session_ttl_s: int = 900
    max_shard_bytes: int = 64 * 1024 * 1024
    max_request_bytes: int = 80 * 1024 * 1024
    max_multiplicity_entries: int = 8192
    max_concurrent_runs: int = 1
    worker_start_timeout_s: float = 60.0
    worker_health_timeout_s: float = 2.0
    worker_request_timeout_s: float = 180.0
    capture_timeout_s: float = 900.0
    min_free_disk_gb: float = 1.0
    min_available_ram_gb: float = 0.45
    safe_gamma: float = 1.05
    max_generation_tokens: int = 4096
    default_max_new_tokens: int = 1000
    max_generation_positions: int = 4096
    generation_timeout_s: float = 1800.0
    generation_step_timeout_s: float = 30.0
    generation_decoder_headroom_gb: float = 0.7
    max_capture_artifact_mb: float = 192.0


LIMITS = Limits(
    min_workers=_int("LAB_MIN_WORKERS", 2),
    max_workers=_int("LAB_MAX_WORKERS", 8),
    default_workers=_int("LAB_DEFAULT_WORKERS", 4),
    max_overlap_pct=_float("LAB_MAX_OVERLAP_PCT", 50.0),
    max_context_tokens=_int("LAB_MAX_CONTEXT_TOKENS", 1024),
    default_context_tokens=_int("LAB_DEFAULT_CONTEXT_TOKENS", 512),
    max_prompt_chars=_int("LAB_MAX_PROMPT_CHARS", 12_000),
    max_query_heads=_int("LAB_MAX_QUERY_HEADS", 512),
    max_sessions=_int("LAB_MAX_SESSIONS", 8),
    session_ttl_s=_int("LAB_SESSION_TTL_S", 900),
    max_shard_bytes=_int("LAB_MAX_SHARD_BYTES", 64 * 1024 * 1024),
    max_request_bytes=_int("LAB_MAX_REQUEST_BYTES", 80 * 1024 * 1024),
    max_multiplicity_entries=_int("LAB_MAX_MULTIPLICITY_ENTRIES", 8192),
    max_concurrent_runs=_int("LAB_MAX_CONCURRENT_RUNS", 1),
    worker_start_timeout_s=_float("LAB_WORKER_START_TIMEOUT_S", 60.0),
    worker_health_timeout_s=_float("LAB_WORKER_HEALTH_TIMEOUT_S", 2.0),
    worker_request_timeout_s=_float("LAB_WORKER_REQUEST_TIMEOUT_S", 180.0),
    capture_timeout_s=_float("LAB_CAPTURE_TIMEOUT_S", 900.0),
    min_free_disk_gb=_float("LAB_MIN_FREE_DISK_GB", 1.0),
    min_available_ram_gb=_float("LAB_MIN_AVAILABLE_RAM_GB", 0.45),
    safe_gamma=_float("LAB_SAFE_GAMMA", 1.05),
    max_generation_tokens=_int("LAB_MAX_GENERATION_TOKENS", 4096),
    default_max_new_tokens=_int("LAB_DEFAULT_MAX_NEW_TOKENS", 1000),
    max_generation_positions=_int("LAB_MAX_GENERATION_POSITIONS", 4096),
    generation_timeout_s=_float("LAB_GENERATION_TIMEOUT_S", 1800.0),
    generation_step_timeout_s=_float("LAB_GENERATION_STEP_TIMEOUT_S", 30.0),
    generation_decoder_headroom_gb=_float("LAB_GENERATION_DECODER_HEADROOM_GB", 0.7),
    max_capture_artifact_mb=_float("LAB_MAX_CAPTURE_ARTIFACT_MB", 192.0),
)


def ensure_worker_limits_consistent(limits: Limits) -> None:
    if limits.min_workers < 1 or limits.max_workers < limits.min_workers:
        raise ValueError("invalid worker bounds")
    if limits.max_context_tokens < 16:
        raise ValueError("LAB_MAX_CONTEXT_TOKENS must be at least 16")
    if not 0.0 <= limits.safe_gamma <= 2.0:
        raise ValueError("LAB_SAFE_GAMMA must be in [0, 2]")
    if limits.capture_timeout_s <= 0.0:
        raise ValueError("LAB_CAPTURE_TIMEOUT_S must be positive")
    if limits.min_free_disk_gb < 0.0:
        raise ValueError("LAB_MIN_FREE_DISK_GB must be non-negative")
    if not 1 <= limits.max_generation_tokens <= MAX_MODEL_POSITIONS:
        raise ValueError(
            f"LAB_MAX_GENERATION_TOKENS must be between 1 and {MAX_MODEL_POSITIONS}"
        )
    if not 16 <= limits.max_generation_positions <= MAX_MODEL_POSITIONS:
        raise ValueError(
            f"LAB_MAX_GENERATION_POSITIONS must be between 16 and {MAX_MODEL_POSITIONS}"
        )
    if not 1 <= limits.default_max_new_tokens <= limits.max_generation_tokens:
        raise ValueError(
            "LAB_DEFAULT_MAX_NEW_TOKENS must be between 1 and LAB_MAX_GENERATION_TOKENS"
        )
    if limits.default_max_new_tokens >= limits.max_generation_positions:
        raise ValueError(
            "LAB_DEFAULT_MAX_NEW_TOKENS must leave room for the prompt within "
            "LAB_MAX_GENERATION_POSITIONS"
        )
    if limits.generation_timeout_s <= 0.0:
        raise ValueError("LAB_GENERATION_TIMEOUT_S must be positive")
    if limits.generation_step_timeout_s <= 0.0:
        raise ValueError("LAB_GENERATION_STEP_TIMEOUT_S must be positive")
    if limits.generation_decoder_headroom_gb < 0.0:
        raise ValueError("LAB_GENERATION_DECODER_HEADROOM_GB must be non-negative")
    if limits.max_capture_artifact_mb <= 0.0:
        raise ValueError("LAB_MAX_CAPTURE_ARTIFACT_MB must be positive")


ensure_worker_limits_consistent(LIMITS)
