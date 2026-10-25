from __future__ import annotations

import threading

import pytest
from fastapi.testclient import TestClient

from app import main as app_module
from app import security
from app.security import LoginLimiter, SessionSigner
from distributed_llm.manager import (
    EventStream,
    RunBusyError,
    RunConfig,
    RunManager,
)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "login_limiter", LoginLimiter())
    monkeypatch.setattr(security, "API_KEY", "")
    manager = RunManager(runs_dir=tmp_path / "runs")
    with TestClient(app_module.app, base_url="http://127.0.0.1:8000") as test_client:
        app_module.app.state.manager = manager
        yield test_client
    manager.shutdown(timeout=2.0)


def test_session_signer_roundtrip_and_expiry():
    signer = SessionSigner("secret", ttl_s=60)
    token = signer.issue(now=1_000)
    assert signer.verify(token, now=1_010)
    assert not signer.verify(token, now=1_061)
    assert not signer.verify(token[:-1] + ("0" if token[-1] != "0" else "1"), now=1_010)
    assert not signer.verify("not-a-token", now=1_010)


def test_login_limiter_blocks_after_failures():
    limiter = LoginLimiter(max_failures=3, window_s=60.0, lockout_s=60.0)
    assert not limiter.blocked("client")
    for _ in range(2):
        limiter.failure("client")
    assert not limiter.blocked("client")
    limiter.failure("client")
    assert limiter.blocked("client")
    limiter.success("client")
    assert not limiter.blocked("client")


def test_index_sets_cookie_and_security_headers(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "dal_session" in response.cookies
    assert response.headers["x-frame-options"] == "DENY"
    assert "default-src 'self'" in response.headers["content-security-policy"]
    assert response.headers["cache-control"] == "no-store"
    assert "cdn" not in response.text


def test_api_requires_session(client):
    fresh = TestClient(app_module.app, base_url="http://127.0.0.1:8000")
    app_module.app.state.manager = client.app.state.manager
    response = fresh.get("/api/status")
    assert response.status_code == 401
    reply = fresh.get("/")
    assert reply.status_code == 200
    assert fresh.get("/api/status").status_code == 200


def test_cross_origin_post_is_rejected(client):
    client.get("/")
    response = client.post(
        "/api/runs",
        json={"prompt": "hello"},
        headers={"Origin": "http://evil.example"},
    )
    assert response.status_code == 403


def test_same_origin_post_is_accepted(client):
    client.get("/")
    response = client.post(
        "/api/session",
        json={"api_key": "unused-in-local-mode"},
        headers={"Origin": "http://127.0.0.1:8000", "Sec-Fetch-Site": "same-origin"},
    )
    assert response.status_code == 200


def test_run_validation_rejects_unknown_fields_and_bad_workers(client):
    client.get("/")
    assert client.post("/api/runs", json={"prompt": "x", "workers": 99}).status_code == 422
    assert client.post("/api/runs", json={"prompt": ""}).status_code == 422
    assert (
        client.post(
            "/api/runs", json={"prompt": "x", "unexpected": True}
        ).status_code
        == 422
    )


def test_public_config_exposes_generation_limits(client):
    client.get("/")
    config = client.get("/api/public-config").json()
    assert "default_model" not in config
    limits = config["limits"]
    assert 1 <= limits["default_max_new_tokens"] <= limits["max_generation_tokens"]
    assert limits["default_max_new_tokens"] == 1000
    assert limits["default_max_new_tokens"] < limits["max_generation_positions"]


def test_run_request_validates_generation_budget(client):
    client.get("/")
    bad_requests = (
        {"prompt": "x", "max_new_tokens": 0},
        {"prompt": "x", "max_new_tokens": 10_000},
        {"prompt": "x", "ollama_model": "legacy-model"},
    )
    for payload in bad_requests:
        assert client.post("/api/runs", json=payload).status_code == 422


def test_generation_budget_defaults_to_1000_without_being_specified():
    # Validation only; no run is started. The default must be usable on the
    # generation path even though the capture context stays at 1024.
    from app.main import RunRequest

    from distributed_llm.config import LIMITS

    assert RunRequest(prompt="x").max_new_tokens == 1000
    assert RunRequest(prompt="x", max_new_tokens=1234).max_new_tokens == 1234
    RunConfig(prompt="x", max_new_tokens=4096).validate()
    with pytest.raises(ValueError, match="max_new_tokens"):
        RunConfig(prompt="x", max_new_tokens=4097).validate()
    assert LIMITS.max_context_tokens == 1024


def test_keyless_mode_reissues_session_without_a_secret(client):
    # The dashboard refreshes an expired cookie on its own, so an empty key must
    # be accepted when no LAB_API_KEY is configured.
    assert security.API_KEY == ""
    fresh = TestClient(app_module.app, base_url="http://127.0.0.1:8000")
    app_module.app.state.manager = client.app.state.manager
    login = fresh.post("/api/session", json={"api_key": ""})
    assert login.status_code == 200
    assert login.json() == {"ok": True, "auth_required": False}
    assert fresh.get("/api/status").status_code == 200


def test_api_key_mode_still_rejects_an_empty_key(client, monkeypatch):
    monkeypatch.setattr(security, "API_KEY", "s3cret-key")
    app_module.login_limiter = LoginLimiter()
    fresh = TestClient(app_module.app, base_url="http://127.0.0.1:8000")
    app_module.app.state.manager = client.app.state.manager
    assert fresh.post("/api/session", json={"api_key": ""}).status_code == 401
    assert fresh.get("/api/status").status_code == 401


def test_api_key_mode_requires_exchange(client, monkeypatch):
    monkeypatch.setattr(security, "API_KEY", "s3cret-key")
    app_module.login_limiter = LoginLimiter()
    fresh = TestClient(app_module.app, base_url="http://127.0.0.1:8000")
    app_module.app.state.manager = client.app.state.manager
    response = fresh.get("/")
    assert response.status_code == 200
    assert "dal_session" not in response.cookies
    assert fresh.get("/api/status").status_code == 401
    assert (
        fresh.post("/api/session", json={"api_key": "wrong"}).status_code == 401
    )
    login = fresh.post("/api/session", json={"api_key": "s3cret-key"})
    assert login.status_code == 200
    assert fresh.get("/api/status").status_code == 200
    bearer = fresh.get(
        "/api/status", headers={"Authorization": "Bearer s3cret-key"}
    )
    assert bearer.status_code == 200


def test_host_guard_rejects_unknown_host(client):
    response = client.get("/healthz", headers={"host": "evil.example"})
    assert response.status_code == 400


class _BlockingManager(RunManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.entered = threading.Event()
        self.release = threading.Event()

    def _execute(self, record) -> None:
        self.entered.set()
        self.release.wait(10.0)
        record.status = "completed"
        record.stream.publish({"type": "info", "msg": "done"})
        record.stream.close()


def test_manager_allows_only_one_active_run(tmp_path):
    manager = _BlockingManager(runs_dir=tmp_path / "runs")
    config = RunConfig(prompt="hello", workers=2, max_tokens=64)
    record = manager.start(config)
    assert manager.entered.wait(2.0)
    with pytest.raises(RunBusyError):
        manager.start(config)
    assert manager.cancel(record.run_id)
    manager.release.set()
    manager.shutdown(timeout=5.0)
    assert record.status == "completed"


def test_manager_rejects_invalid_config(tmp_path):
    manager = RunManager(runs_dir=tmp_path / "runs")
    with pytest.raises(ValueError):
        manager.start(RunConfig(prompt="", workers=2))
    with pytest.raises(ValueError):
        manager.start(RunConfig(prompt="hello", workers=1))
    with pytest.raises(ValueError):
        manager.start(RunConfig(prompt="hello", max_tokens=8))
    assert manager.current() is None


def test_event_stream_replay_and_bound():
    stream = EventStream(max_events=3)
    for index in range(5):
        stream.publish({"type": "info", "msg": str(index)})
    assert [event["msg"] for event in stream.since(0)] == ["2", "3", "4"]
    assert [event["msg"] for event in stream.since(3)] == ["3", "4"]
    assert [event["msg"] for event in stream.since(5)] == []
    stream.close()
    assert stream.closed
