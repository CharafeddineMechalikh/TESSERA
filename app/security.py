"""Authentication and request guards for the local lab server."""
from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
import time
from typing import Optional
from urllib.parse import urlsplit

from starlette.responses import JSONResponse

from distributed_llm.config import API_KEY

SESSION_COOKIE = "dal_session"


class SessionSigner:
    """HMAC signed, expiring session cookies."""

    def __init__(self, secret: str, ttl_s: int):
        material = (secret or secrets.token_hex(32)).encode("utf-8")
        self._secret = material
        self.ttl_s = int(ttl_s)

    def issue(self, now: Optional[int] = None) -> str:
        timestamp = int(time.time() if now is None else now)
        nonce = secrets.token_urlsafe(16)
        payload = f"{timestamp}.{nonce}"
        signature = hmac.new(
            self._secret, payload.encode("ascii"), hashlib.sha256
        ).hexdigest()
        return f"{payload}.{signature}"

    def verify(self, token: Optional[str], now: Optional[int] = None) -> bool:
        if not token:
            return False
        parts = token.split(".")
        if len(parts) != 3:
            return False
        timestamp, nonce, signature = parts
        if not timestamp.isdigit() or not nonce:
            return False
        issued = int(timestamp)
        current = int(time.time() if now is None else now)
        if issued > current + 60 or current - issued > self.ttl_s:
            return False
        expected = hmac.new(
            self._secret, f"{timestamp}.{nonce}".encode("ascii"), hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(expected, signature)


def api_key_required() -> bool:
    return bool(API_KEY)


def check_bearer(header: Optional[str]) -> bool:
    if not API_KEY or not header:
        return False
    scheme, _, value = header.partition(" ")
    if scheme.lower() != "bearer" or not value:
        return False
    return hmac.compare_digest(value.strip(), API_KEY)


def check_api_key(value: str) -> bool:
    if not API_KEY or not isinstance(value, str):
        return False
    return hmac.compare_digest(value, API_KEY)


class LoginLimiter:
    """Small in-memory brute force brake for the key exchange endpoint."""

    def __init__(self, max_failures: int = 8, window_s: float = 300.0, lockout_s: float = 300.0):
        self.max_failures = max_failures
        self.window_s = window_s
        self.lockout_s = lockout_s
        self._lock = threading.Lock()
        self._state: dict[str, list[float]] = {}
        self._blocked_until: dict[str, float] = {}

    def blocked(self, client: str) -> bool:
        with self._lock:
            until = self._blocked_until.get(client, 0.0)
            if until <= time.monotonic():
                self._blocked_until.pop(client, None)
                return False
            return True

    def failure(self, client: str) -> None:
        now = time.monotonic()
        with self._lock:
            attempts = [t for t in self._state.get(client, []) if now - t <= self.window_s]
            attempts.append(now)
            if len(attempts) >= self.max_failures:
                self._blocked_until[client] = now + self.lockout_s
                attempts = []
            self._state[client] = attempts

    def success(self, client: str) -> None:
        with self._lock:
            self._state.pop(client, None)
            self._blocked_until.pop(client, None)


def _host_name(value: str) -> str:
    try:
        return (urlsplit("//" + value).hostname or "").lower()
    except ValueError:
        return ""


def _origin_name(value: str) -> str:
    try:
        return (urlsplit(value).hostname or "").lower()
    except ValueError:
        return ""


class HostGuardMiddleware:
    """Reject requests whose Host header is not in the configured allowlist."""

    def __init__(self, app, allowed_hosts):
        self.app = app
        self.allowed = {host.lower() for host in allowed_hosts}

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
        name = _host_name(headers.get("host", ""))
        if name not in self.allowed:
            response = JSONResponse(
                {"detail": "invalid Host header"}, status_code=400
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


class OriginGuardMiddleware:
    """Block cross-site unsafe requests; SameSite cookies remain the second line."""

    UNSAFE = {"POST", "PUT", "PATCH", "DELETE"}

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"].upper() not in self.UNSAFE:
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
        host = headers.get("host", "")
        origin = headers.get("origin")
        if origin:
            if _origin_name(origin) != _host_name(host) or urlsplit(origin).scheme not in {
                "http",
                "https",
            }:
                response = JSONResponse(
                    {"detail": "cross-origin request rejected"}, status_code=403
                )
                await response(scope, receive, send)
                return
        fetch_site = headers.get("sec-fetch-site")
        if fetch_site and fetch_site not in {"same-origin", "none"}:
            response = JSONResponse(
                {"detail": "cross-site request rejected"}, status_code=403
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


class BodyLimitMiddleware:
    """Reject oversized request bodies before they reach a handler."""

    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = int(max_bytes)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
        declared = headers.get("content-length")
        if declared is not None:
            try:
                if int(declared) > self.max_bytes:
                    await self._reject(send)
                    return
            except ValueError:
                await self._reject(send)
                return
        received = 0
        response_started = False

        class _TooLarge(Exception):
            pass

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _TooLarge()
            return message

        async def watched_send(message):
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, watched_send)
        except _TooLarge:
            if not response_started:
                await self._reject(send)

    async def _reject(self, send) -> None:
        response = JSONResponse(
            {"detail": f"request body exceeds {self.max_bytes} bytes"},
            status_code=413,
        )
        await response({"type": "http"}, _empty_receive, send)


async def _empty_receive():
    return {"type": "http.disconnect"}


class SecurityHeadersMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def watched_send(message):
            if message["type"] == "http.response.start":
                raw = list(message.get("headers", []))
                present = {key.lower() for key, _ in raw}
                additions = [
                    (b"x-content-type-options", b"nosniff"),
                    (b"x-frame-options", b"DENY"),
                    (b"referrer-policy", b"no-referrer"),
                    (
                        b"permissions-policy",
                        b"camera=(), microphone=(), geolocation=(), usb=()",
                    ),
                    (
                        b"content-security-policy",
                        b"default-src 'self'; script-src 'self'; "
                        b"style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
                        b"connect-src 'self'; font-src 'self'; object-src 'none'; "
                        b"base-uri 'none'; frame-ancestors 'none'; form-action 'self'",
                    ),
                    (b"cache-control", b"no-store"),
                ]
                for key, value in additions:
                    if key not in present:
                        raw.append((key, value))
                message = {**message, "headers": raw}
            await send(message)

        await self.app(scope, receive, watched_send)
