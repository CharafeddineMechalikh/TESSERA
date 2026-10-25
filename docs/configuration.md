# Configuration

All limits live in `distributed_llm/config.py`. Every one is environment
overridable, parsed at import time, and then cross-checked by
`ensure_worker_limits_consistent`. The server refuses to start on an invalid
combination instead of running with a limit it cannot honor.

Precedence: the environment value wins over the code default; the code default is
what the dashboard and the API schemas advertise; `GET /api/public-config` reports
the effective values so the two can never silently diverge.

## Model and workspace

| Variable | Default | Meaning |
|---|---|---|
| `LAB_MODEL_ID` | `Qwen/Qwen2.5-Coder-1.5B` | Snapshot to capture from and decode with |
| `LAB_MODEL_REVISION` | `df3ce67c0e24480f20468b6ef2894622d69eb73b` | Pinned revision; the lab uses `local_files_only=True` and never downloads implicitly |
| `LAB_OUTPUT_DIR` | `./output` | Root for run directories, created at import |
| `LAB_STORE_PROMPTS` | `false` | When false, `prompt.txt` is deleted as soon as capture and generation are done; the summary keeps only a SHA-256 |
| `LAB_ALLOWED_HOSTS` | `127.0.0.1,localhost,::1` | HTTP `Host` allowlist, comma separated |

Model geometry is read from the pinned `config.json`, not from an environment
variable: 28 layers, 12 query heads, 2 KV heads, `q_per_kv` 6, `head_dim` 128,
hidden 1536, and `max_position_embeddings` 32768. That last value is hard-coded as
`MAX_MODEL_POSITIONS` and is the upper bound two of the generation limits are
checked against.

## Authentication

| Variable | Default | Meaning |
|---|---|---|
| `LAB_API_KEY` | empty | Empty selects keyless local mode: no key is demanded and sessions are issued to loopback clients. Any non-empty value requires the key (or the bearer form) on every `/api` route |
| `LAB_SESSION_SECRET` | empty | HMAC-SHA256 key for the `dal_session` cookie. Empty means a random secret per process, so restarting invalidates every session |

Set both when exposing the dashboard to anything other than the local machine.
Neither value is ever logged, echoed into an artifact, or included in an event.

## Worker bounds and timeouts

| Variable | Default | Meaning |
|---|---|---|
| `LAB_MIN_WORKERS` | `2` | Smallest cluster the API accepts |
| `LAB_MAX_WORKERS` | `8` | Largest cluster; sized for this machine's RAM, see [operations.md](operations.md) |
| `LAB_DEFAULT_WORKERS` | `4` | Declared default, currently not consumed by any code path |
| `LAB_WORKER_START_TIMEOUT_S` | `60.0` | How long `cluster.py` waits for ready files plus `/health` |
| `LAB_WORKER_HEALTH_TIMEOUT_S` | `2.0` | Declared, currently not consumed; readiness probes use inline timeouts |
| `LAB_WORKER_REQUEST_TIMEOUT_S` | `180.0` | Per-request timeout on the benchmark HTTP client, 5 s connect |
| `LAB_SESSION_TTL_S` | `900` | Idle window for worker shard and generation cache state; expiry answers 410. Also the cookie lifetime |
| `LAB_MAX_SESSIONS` | `8` | Declared, currently not consumed; workers are single-session by construction |
| `LAB_MAX_CONCURRENT_RUNS` | `1` | One active run; a second `POST /api/runs` is 409 |

## Capture and benchmark budgets

| Variable | Default | Meaning |
|---|---|---|
| `LAB_MAX_CONTEXT_TOKENS` | `1024` | Ceiling on the captured attention context (`max_tokens` in a run request). Sized for this machine's RAM, not for the model |
| `LAB_DEFAULT_CONTEXT_TOKENS` | `512` | Dashboard and API default |
| `LAB_MAX_PROMPT_CHARS` | `12000` | Prompt length guard before tokenization |
| `LAB_MAX_QUERY_HEADS` | `512` | Worker-side cap on query rows per request |
| `LAB_MAX_MULTIPLICITY_ENTRIES` | `8192` | Cap on the overlap correction map in `/v1/stats` |
| `LAB_MAX_REQUEST_BYTES` | `80 MiB` | Worker body cap for `/v1/shard`, also the JSON API cap through `min(LAB_MAX_REQUEST_BYTES, 1 MiB)` |
| `LAB_MAX_SHARD_BYTES` | `64 MiB` | Coordinator-side estimate of a shard before it is sent; exceeded means the run fails with a message telling you to lower the context length or raise the worker count |
| `LAB_CAPTURE_TIMEOUT_S` | `900.0` | Watchdog for the capture subprocess |
| `LAB_MIN_AVAILABLE_RAM_GB` | `0.45` | Capture preflight floor |
| `LAB_MIN_FREE_DISK_GB` | `1.0` | Disk floor, checked at run start and again before the benchmark |
| `LAB_MAX_CAPTURE_ARTIFACT_MB` | `192.0` | Refuse a `capture.npz` bigger than this; a full 28-layer capture of a 1024-token prompt is roughly at this scale |
| `LAB_SAFE_GAMMA` | `1.05` | Safety factor multiplying the pruned softmax mass in the certificate |
| `LAB_MAX_OVERLAP_PCT` | `50.0` | Ceiling on secondary copies during partitioning |

## Generation budgets

Four limits interact, so read them together.

| Variable | Default | Meaning |
|---|---|---|
| `LAB_MAX_GENERATION_TOKENS` | `4096` | Ceiling on `max_new_tokens` in a run request |
| `LAB_DEFAULT_MAX_NEW_TOKENS` | `1000` | What the dashboard and the API use when the field is omitted |
| `LAB_MAX_GENERATION_POSITIONS` | `4096` | Ceiling on prompt tokens plus generated tokens, i.e. the decode cache window each worker preallocates against |
| `LAB_GENERATION_TIMEOUT_S` | `1800.0` | Fixed part of the generation watchdog: cluster spawn, RAM admission, prefill |
| `LAB_GENERATION_STEP_TIMEOUT_S` | `30.0` | Per-token part of the watchdog budget |
| `LAB_GENERATION_DECODER_HEADROOM_GB` | `0.7` | Decoder peak resident memory charged during RAM admission |

The decode window is deliberately separate from `LAB_MAX_CONTEXT_TOKENS`. The
capture budget is about holding a dense attention probe in RAM on this machine;
the generation window is about the strided KV cache, which is far cheaper per
token. Conflating them made a 1000-token decode impossible behind a 1024-token
capture cap.

The watchdog budget is

```
LAB_GENERATION_TIMEOUT_S + max_new_tokens * LAB_GENERATION_STEP_TIMEOUT_S
```

so a 1000-token request gets 1800 + 30000 s and a 10-token request gets 1800 + 300 s.
A flat timeout would either kill long decodes or leave a wedged short one running
for half an hour.

Worker-side caps that are constants, not environment variables, because they are
transport facts rather than policy: generation bodies max out at 32 MiB
(`MAX_GENERATION_BYTES`), JSON bodies at 4 MiB (`MAX_JSON_BYTES`), and the client
retries three times (`ATTEMPTS`).

## Validator rules

`ensure_worker_limits_consistent` raises `ValueError` at import time on any of:

| Rule | Message fragment |
|---|---|
| `min_workers >= 1` and `max_workers >= min_workers` | `invalid worker bounds` |
| `max_context_tokens >= 16` | `LAB_MAX_CONTEXT_TOKENS must be at least 16` |
| `0 <= safe_gamma <= 2` | `LAB_SAFE_GAMMA must be in [0, 2]` |
| `capture_timeout_s > 0` | `LAB_CAPTURE_TIMEOUT_S must be positive` |
| `min_free_disk_gb >= 0` | `LAB_MIN_FREE_DISK_GB must be non-negative` |
| `1 <= max_generation_tokens <= 32768` | `LAB_MAX_GENERATION_TOKENS must be between 1 and 32768` |
| `16 <= max_generation_positions <= 32768` | `LAB_MAX_GENERATION_POSITIONS must be between 16 and 32768` |
| `1 <= default_max_new_tokens <= max_generation_tokens` | `LAB_DEFAULT_MAX_NEW_TOKENS must be between 1 and LAB_MAX_GENERATION_TOKENS` |
| `default_max_new_tokens < max_generation_positions` | `... must leave room for the prompt within LAB_MAX_GENERATION_POSITIONS` |
| `generation_timeout_s > 0` | `LAB_GENERATION_TIMEOUT_S must be positive` |
| `generation_step_timeout_s > 0` | `LAB_GENERATION_STEP_TIMEOUT_S must be positive` |
| `generation_decoder_headroom_gb >= 0` | `LAB_GENERATION_DECODER_HEADROOM_GB must be non-negative` |
| `max_capture_artifact_mb > 0` | `LAB_MAX_CAPTURE_ARTIFACT_MB must be positive` |

The last two default rules exist because a default that cannot fit inside the
window, or leaves no room for the prompt, is a configuration that would fail every
run at the first request. `tests/test_config_limits.py` pins all of them.

Non-integer or non-numeric values are rejected during parsing with
`<NAME> must be an integer` or `must be a float`; a boolean variable that is not
one of `1/true/yes/on` or `0/false/no/off` raises `<NAME> must be a boolean`.

## Worked examples

Raise the decode window to 8192 positions while keeping the 1000-token default:

```bash
LAB_MAX_GENERATION_POSITIONS=8192 LAB_MAX_GENERATION_TOKENS=8192 \
LAB_DEFAULT_MAX_NEW_TOKENS=2000 ./launch.sh fast
```

`LAB_DEFAULT_MAX_NEW_TOKENS` must stay strictly below the window, and the KV
charge grows linearly: 2021 positions across 28 layers, 2 KV heads, 128 dims,
float32, keys and values is 57,344 bytes per position, about 116 MB cluster-wide
and about 0.13 GB with the transient factor, plus the decoder headroom. The
worker count does not change that total, because each position lives on exactly
one worker. See the admission formula in [operations.md](operations.md).

Disable generation for attention-only probes:

```bash
LAB_DEFAULT_MAX_NEW_TOKENS=1 ./launch.sh fast
```

The toggle is per run (`"generate": false`), which is usually what you want; the
environment form only changes the default token budget.

Run with key authentication:

```bash
LAB_API_KEY="$(python -c 'import secrets;print(secrets.token_urlsafe(32))')" \
LAB_SESSION_SECRET="$(python -c 'import secrets;print(secrets.token_hex(32))')" \
./launch.sh fast
```

Keep both secrets out of the repository and out of shell history where possible;
the key is compared with `hmac.compare_digest` and never logged.

## Internal constants worth knowing

| Constant | Value | Where |
|---|---|---|
| `MIN_TOKEN_LENGTH` | 24 | bearer token floor accepted by `gen_client.py`; the worker boot floor is 16 |
| `ESTIMATED_WORKER_RSS_MB` | 72.0 | per-worker RAM charged at cluster spawn |
| `SPAWN_HEADROOM_GB` | 0.25 | extra charged at cluster spawn |
| `KV_GROWTH_FACTOR` | 1.25 | transient slack while append concatenations grow |
| `MAX_MODEL_POSITIONS` | 32768 | pinned `max_position_embeddings` |
| `_EXP_LIMIT` | 700.0 | overflow guard in `safe_exp` |
| `fmix32` seed | `0x977F23A7` | MurmurHash3 finalizer used for pseudorandom ownership |
| `ATTEMPTS` | 3 | client retries per worker request |

## Known unenforced knobs

`LAB_DEFAULT_WORKERS`, `LAB_WORKER_HEALTH_TIMEOUT_S`, and `LAB_MAX_SESSIONS` are
parsed into `Limits` but no code path reads them: the dashboard hard-codes 4
workers, the readiness probes pass inline timeouts, and each worker is bound to a
single session at boot, which makes a session count limit redundant. They are
listed here rather than hidden because a documented knob that does nothing is
easier to fix or delete than an undocumented one that appears to work.
