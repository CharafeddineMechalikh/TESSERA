"""Binary HTTP client for the distributed generation KV caches.

This client runs inside the isolated decoder process. It never sees a bearer
token in an artifact or log, sends only loopback requests, and uses NPZ bodies
so K/V and attention payloads are never expanded into JSON text.
"""
from __future__ import annotations

import io
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Sequence, Tuple
from urllib.parse import urlsplit

import httpx
import numpy as np

MIN_TOKEN_LENGTH = 24
ATTEMPTS = 3
ALLOWED_HOSTS = {"127.0.0.1", "localhost", "::1"}


class GenerationClientError(RuntimeError):
    pass


class GenerationClient:
    def __init__(self, connection: dict, timeout_s: float):
        if not isinstance(connection, dict):
            raise GenerationClientError("generation connection is not an object")
        session_id = str(connection.get("session_id", ""))
        token = str(connection.get("token", ""))
        workers = connection.get("workers")
        if len(session_id) < 8:
            raise GenerationClientError("generation session id is invalid")
        if len(token) < MIN_TOKEN_LENGTH:
            raise GenerationClientError("generation bearer token is invalid")
        if not isinstance(workers, list) or not workers:
            raise GenerationClientError("generation worker list is empty")
        parsed = []
        seen = set()
        for entry in workers:
            try:
                worker_id = int(entry["worker_id"])
                base_url = str(entry["base_url"]).rstrip("/")
            except (KeyError, TypeError, ValueError) as exc:
                raise GenerationClientError("generation worker entry is invalid") from exc
            split = urlsplit(base_url)
            if split.scheme != "http" or (split.hostname or "").lower() not in ALLOWED_HOSTS:
                raise GenerationClientError("generation workers must be loopback HTTP services")
            if worker_id < 0 or worker_id in seen:
                raise GenerationClientError("generation worker ids are invalid")
            seen.add(worker_id)
            parsed.append((worker_id, base_url))
        self.session_id = session_id
        self.token = token
        self.worker_ids = [worker_id for worker_id, _ in parsed]
        self._urls = {worker_id: url for worker_id, url in parsed}
        self.num_workers = len(parsed)
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout_s, connect=5.0),
            limits=httpx.Limits(
                max_connections=self.num_workers * 2,
                max_keepalive_connections=self.num_workers,
            ),
        )
        self._executor = ThreadPoolExecutor(
            max_workers=self.num_workers, thread_name_prefix="gen-http"
        )
        self.metrics = {
            "requests": 0,
            "retries": 0,
            "failures": 0,
            "bytes_sent": 0,
            "bytes_received": 0,
            "worker_wall_ms": 0.0,
            "workers": {
                worker_id: {
                    "calls": 0,
                    "resets": 0,
                    "appends": 0,
                    "attends": 0,
                    "positions": 0,
                    "bytes_sent": 0,
                    "bytes_received": 0,
                    "wall_ms": 0.0,
                }
                for worker_id in self.worker_ids
            },
        }

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}

    def _record(self, worker_id: int, path: str, elapsed_ms: float, sent: int, received: int) -> None:
        self.metrics["requests"] += 1
        self.metrics["bytes_sent"] += sent
        self.metrics["bytes_received"] += received
        self.metrics["worker_wall_ms"] += elapsed_ms
        entry = self.metrics["workers"][worker_id]
        entry["calls"] += 1
        entry["bytes_sent"] += sent
        entry["bytes_received"] += received
        entry["wall_ms"] += elapsed_ms
        if path.endswith("/reset"):
            entry["resets"] += 1
        elif path.endswith("/append"):
            entry["appends"] += 1
        elif path.endswith("/attend"):
            entry["attends"] += 1

    def _request(
        self,
        worker_id: int,
        path: str,
        *,
        params: dict,
        content: bytes,
        binary: bool,
    ):
        url = self._urls[worker_id] + path
        last_error = ""
        for attempt in range(ATTEMPTS):
            started = time.perf_counter()
            try:
                response = self._client.post(
                    url, params=params, content=content, headers=self._headers()
                )
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                self._record(worker_id, path, elapsed_ms, len(content), len(response.content))
                if response.status_code < 400:
                    if binary:
                        return response.content
                    try:
                        return response.json()
                    except ValueError as exc:
                        raise GenerationClientError(
                            f"worker {worker_id} {path} returned invalid JSON"
                        ) from exc
                if response.status_code < 500:
                    detail = response.text[:400]
                    raise GenerationClientError(
                        f"worker {worker_id} {path} rejected the request "
                        f"(HTTP {response.status_code}): {detail}"
                    )
                last_error = f"HTTP {response.status_code}: {response.text[:300]}"
            self.metrics["retries"] += 1
            if attempt + 1 < ATTEMPTS:
                time.sleep(0.15 * (attempt + 1))
        self.metrics["failures"] += 1
        raise GenerationClientError(f"worker {worker_id} {path} failed: {last_error}")

    def _parallel(self, job):
        futures = [self._executor.submit(job, worker_id) for worker_id in self.worker_ids]
        return [future.result() for future in futures]

    def reset(
        self,
        layers: Sequence[int],
        kv_heads: Sequence[int],
        q_per_kv: int,
        head_dim: int,
        max_positions: int,
    ) -> None:
        params_common = {
            "session_id": self.session_id,
            "num_workers": self.num_workers,
            "layers": ",".join(str(x) for x in layers),
            "kv_heads": ",".join(str(x) for x in kv_heads),
            "q_per_kv": int(q_per_kv),
            "head_dim": int(head_dim),
            "max_positions": int(max_positions),
        }

        def job(worker_id: int) -> dict:
            params = dict(params_common)
            params["worker_id"] = worker_id
            return self._request(
                worker_id, "/v1/gen/reset", params=params, content=b"", binary=False
            )

        self._parallel(job)

    def append(
        self,
        positions: np.ndarray,
        keys: Dict[Tuple[int, int], np.ndarray],
        values: Dict[Tuple[int, int], np.ndarray],
    ) -> None:
        positions = np.asarray(positions, dtype=np.int64)
        if positions.ndim != 1 or len(positions) == 0:
            raise GenerationClientError("append positions must be a non-empty vector")
        if keys.keys() != values.keys() or not keys:
            raise GenerationClientError("K/V unit sets do not match")

        def job(worker_id: int) -> dict:
            owned = positions % self.num_workers == worker_id
            selected = positions[owned]
            if len(selected) == 0:
                return {"status": "empty"}
            payload = {"positions": selected}
            for unit in keys:
                layer, kv_head = unit
                payload[f"k_{layer}_{kv_head}"] = np.asarray(keys[unit][owned], dtype=np.float32)
                payload[f"v_{layer}_{kv_head}"] = np.asarray(values[unit][owned], dtype=np.float32)
            buffer = io.BytesIO()
            np.savez(buffer, **payload)
            reply = self._request(
                worker_id,
                "/v1/gen/append",
                params={"worker_id": worker_id, "session_id": self.session_id},
                content=buffer.getvalue(),
                binary=False,
            )
            self.metrics["workers"][worker_id]["positions"] += int(len(selected))
            return reply

        self._parallel(job)

    def attend(
        self,
        layer: int,
        kv_head: int,
        positions: np.ndarray,
        q: np.ndarray,
    ) -> List[dict]:
        positions = np.asarray(positions, dtype=np.int64)
        q = np.asarray(q, dtype=np.float32)
        if positions.ndim != 1 or len(positions) == 0:
            raise GenerationClientError("query positions must be a non-empty vector")
        if q.ndim != 3 or q.shape[0] != len(positions):
            raise GenerationClientError("query tensor does not match the query positions")
        buffer = io.BytesIO()
        np.savez(buffer, positions=positions, q=q)
        body = buffer.getvalue()

        def job(worker_id: int) -> dict:
            raw = self._request(
                worker_id,
                "/v1/gen/attend",
                params={
                    "worker_id": worker_id,
                    "session_id": self.session_id,
                    "layer": int(layer),
                    "kv_head": int(kv_head),
                },
                content=body,
                binary=True,
            )
            return self._parse_attend(raw, worker_id, layer, kv_head, positions, q)

        return self._parallel(job)

    @staticmethod
    def _parse_attend(
        raw: bytes,
        worker_id: int,
        layer: int,
        kv_head: int,
        positions: np.ndarray,
        q: np.ndarray,
    ) -> dict:
        required = {"positions", "lmax", "sumexp", "out", "visible_keys", "owned_keys", "elapsed_ms"}
        try:
            archive = np.load(io.BytesIO(raw), allow_pickle=False)
            with archive as npz:
                if set(npz.files) != required:
                    raise GenerationClientError(
                        f"worker {worker_id} attend response has unexpected arrays"
                    )
                returned = np.asarray(npz["positions"], dtype=np.int64)
                lmax = np.asarray(npz["lmax"], dtype=np.float64)
                sumexp = np.asarray(npz["sumexp"], dtype=np.float64)
                out = np.asarray(npz["out"], dtype=np.float64)
                visible = np.asarray(npz["visible_keys"], dtype=np.int64)
                owned = int(npz["owned_keys"])
                elapsed_ms = float(npz["elapsed_ms"])
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as exc:
            raise GenerationClientError(
                f"worker {worker_id} attend response is not a valid npz archive"
            ) from exc
        queries, heads, head_dim = q.shape
        if not np.array_equal(returned, positions):
            raise GenerationClientError(f"worker {worker_id} returned the wrong positions")
        if lmax.shape != (queries, heads) or sumexp.shape != (queries, heads):
            raise GenerationClientError(f"worker {worker_id} returned the wrong statistic shape")
        if out.shape != (queries, heads, head_dim) or visible.shape != (queries,):
            raise GenerationClientError(f"worker {worker_id} returned the wrong output shape")
        if not np.isfinite(out).all() or np.any(sumexp < 0.0):
            raise GenerationClientError(f"worker {worker_id} returned invalid statistics")
        return {
            "worker_id": worker_id,
            "layer": layer,
            "kv_head": kv_head,
            "lmax": lmax,
            "sumexp": sumexp,
            "out": out,
            "visible_keys": visible,
            "owned_keys": owned,
            "elapsed_ms": elapsed_ms,
        }

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._client.close()

    def status(self) -> List[dict]:
        """Final worker state, without tokens or session identifiers."""
        results = []
        for worker_id in self.worker_ids:
            entry = {"worker_id": worker_id}
            try:
                response = self._client.get(
                    self._urls[worker_id] + "/v1/status",
                    headers=self._headers(),
                    timeout=5.0,
                )
                if response.status_code != 200:
                    entry["error"] = f"HTTP {response.status_code}"
                else:
                    body = response.json()
                    for key in (
                        "rss_mb",
                        "generation_loaded",
                        "generation_positions",
                        "generation_units",
                        "generation_head_dim",
                        "uptime_s",
                    ):
                        entry[key] = body.get(key)
                    counters = body.get("counters") or {}
                    for key in (
                        "gen_resets",
                        "gen_appends",
                        "gen_attend_calls",
                        "gen_tokens",
                    ):
                        entry[key] = counters.get(key)
            except (httpx.HTTPError, ValueError) as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"
            results.append(entry)
        return results

    def summary(self) -> dict:
        return {
            "requests": self.metrics["requests"],
            "retries": self.metrics["retries"],
            "failures": self.metrics["failures"],
            "bytes_sent": self.metrics["bytes_sent"],
            "bytes_received": self.metrics["bytes_received"],
            "worker_wall_ms": round(self.metrics["worker_wall_ms"], 3),
            "workers": {
                str(worker_id): dict(entry)
                for worker_id, entry in sorted(self.metrics["workers"].items())
            },
        }
