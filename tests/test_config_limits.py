"""Environment-limit validation, including the long-generation budget."""
from __future__ import annotations

import pytest

from distributed_llm.config import Limits, ensure_worker_limits_consistent


def test_default_limits_allow_a_1000_token_decode():
    limits = Limits()
    ensure_worker_limits_consistent(limits)
    assert limits.default_max_new_tokens == 1000
    assert limits.max_generation_tokens >= limits.default_max_new_tokens
    assert limits.max_generation_positions > limits.default_max_new_tokens


def test_generation_default_above_the_position_window_is_rejected():
    limits = Limits(default_max_new_tokens=4096, max_generation_positions=4096)
    with pytest.raises(ValueError, match="LAB_MAX_GENERATION_POSITIONS"):
        ensure_worker_limits_consistent(limits)


def test_generation_default_above_the_request_ceiling_is_rejected():
    limits = Limits(max_generation_tokens=10, default_max_new_tokens=11)
    with pytest.raises(ValueError, match="LAB_DEFAULT_MAX_NEW_TOKENS"):
        ensure_worker_limits_consistent(limits)


def test_absurd_generation_ceiling_is_rejected():
    limits = Limits(max_generation_tokens=100_000)
    with pytest.raises(ValueError, match="LAB_MAX_GENERATION_TOKENS"):
        ensure_worker_limits_consistent(limits)


def test_position_window_beyond_the_model_is_rejected():
    limits = Limits(max_generation_positions=100_000)
    with pytest.raises(ValueError, match="LAB_MAX_GENERATION_POSITIONS"):
        ensure_worker_limits_consistent(limits)


def test_non_positive_step_timeout_is_rejected():
    limits = Limits(generation_step_timeout_s=0.0)
    with pytest.raises(ValueError, match="LAB_GENERATION_STEP_TIMEOUT_S"):
        ensure_worker_limits_consistent(limits)


def test_capture_context_limit_is_untouched_by_generation_defaults():
    limits = Limits(default_max_new_tokens=1000)
    ensure_worker_limits_consistent(limits)
    assert limits.max_context_tokens == 1024
