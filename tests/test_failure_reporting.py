"""Failure reporting for subprocess stages that die without an exception."""
from __future__ import annotations

from distributed_llm.manager import describe_exit_code


def test_windows_access_violation_is_named():
    assert describe_exit_code(3221225477) == " (0xC0000005: access violation)"


def test_unknown_windows_status_keeps_the_hex_code():
    assert describe_exit_code(0xC0000006) == " (0xC0000006)"


def test_posix_signal_number_is_reported():
    assert describe_exit_code(-11) == " (signal 11)"


def test_ordinary_and_clean_exits_add_nothing():
    assert describe_exit_code(0) == ""
    assert describe_exit_code(1) == ""
    assert describe_exit_code(2) == ""
