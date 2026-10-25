# Changelog

All functional changes are recorded here. The lab version is `APP_VERSION` in
`distributed_llm/config.py`.

## 11.2.0

Keyless local mode no longer asks for an API key that does not exist.

- `LoginRequest.api_key` accepts an empty string, and the dashboard login card
  keeps the key field and its help text hidden until the server reports
  `auth_required: true`. Previously the overlay always rendered a `required`
  input, so a session expiry in keyless mode was unsatisfiable: the browser
  demanded a key, and `POST /api/session` rejected an empty one with HTTP 422.
- An expired session now heals itself: on a 401 the client silently re-opens one
  through `/api/session` and resumes, with a guard so a failed bootstrap cannot
  recurse. The key form only appears when key mode is actually configured.
- `GET /` seeds the mode in the page (`data-auth-required`), so the first render
  knows whether a key is expected without waiting for a failed request.
- Key mode is unchanged in substance: with `LAB_API_KEY` set, `create_session`
  still compares the submitted key and an empty key is rejected with 401, so this
  fix does not weaken authentication.
- Tests: `test_keyless_mode_reissues_session_without_a_secret`,
  `test_api_key_mode_still_rejects_an_empty_key`, and the public-config and
  security-header cases in `tests/test_app.py`.

## 11.1.0

Generation defaults to a 1000-token budget, and the limits that blocked it were
separated from the capture-path limits.

- `default_max_new_tokens` is now `1000` (was `8`) and `max_generation_tokens`
  is `4096` (was `32`), so a run that does not specify the field decodes until
  end-of-sequence or the budget instead of stopping after a few tokens.
- Added `LAB_MAX_GENERATION_POSITIONS` (`4096`): the ceiling on prompt plus
  generated tokens for the decode path. It used to borrow
  `LAB_MAX_CONTEXT_TOKENS`, which is a capture-memory budget, so long decodes
  were refused by a limit that had nothing to do with them. The capture context
  stays at 1024. Renamed the runner parameter and CLI flag accordingly.
- Relaxed the worker-side `max_positions` check onto the same generation limit;
  it previously refused any session longer than the capture context.
- The generation watchdog budget now scales with the request:
  `LAB_GENERATION_TIMEOUT_S + max_new_tokens * LAB_GENERATION_STEP_TIMEOUT_S`
  (30 s per token against a measured 8.4 s), so a 1000-token decode is not killed
  at 214 tokens and a short decode is not left to hang.
- Fixed generation RAM admission: ownership is strided, so the cluster holds one
  copy of every position, but the estimate charged `num_workers * kv_total`. At 8
  workers and a 1000-token budget that demanded 1.16 GB where 0.77 GB sufficed.
  It now charges the cluster-wide total with a documented 1.25 transient factor.
- `/api/public-config` and `metrics()` expose the new limits; the dashboard input
  and its note state the real per-token cost and that the default can run hours
  unless the sequence ends or Cancel is pressed.
- Version 11.1.0. New tests cover the environment-limit rules
  (`tests/test_config_limits.py`), the corrected admission arithmetic against a
  real worker cluster, and the 1000-token default through `RunRequest` and
  `RunConfig`.

## 11.0.0

Generated text now comes from the distributed attention path instead of a
monolithic Ollama forward pass.

- Added a layer-streamed Qwen2 decoder that never holds the full fp32 model or a
  KV cache. Weights stream from `model.safetensors` one layer at a time, and the
  LM head is evaluated in vocabulary chunks.
- Added `/v1/gen/reset`, `/v1/gen/append`, and `/v1/gen/attend` to the workers.
  Each `(layer, KV head)` unit owns its positions, keys, and values. KV ownership
  is strided by token position.
- Added an exact causal statistics kernel and a float64 log-sum-exp merge. Every
  attend call verifies that each query sees exactly `position + 1` keys and that
  worker-owned key counts match the ownership rule.
- Added a binary NPZ generation client with bounded retries, idempotent appends,
  strict shape/finiteness validation, and sanitized status reporting.
- The manager now starts a second worker cluster for generation, enforces a
  decoder-memory admission check and a generation timeout, fails the run when
  requested generation fails, and deletes the prompt unless configured to keep
  it.
- Replaced the Ollama controls and provenance text with a distributed generation
  toggle and a new-token budget. Removed all Ollama configuration and launch
  checks.
- `generation.json` now records token ids, text, per-step margins, decoder and
  transport timings, worker KV counts and RSS, peak memory, and determinism.
- Added tiny-model tests that compare distributed greedy decoding against
  `Qwen2ForCausalLM` through a real two- and three-worker cluster, plus a
  real-model generation test that skips when the snapshot or RAM is unavailable.
- Added a pre-decode context check on the captured prompt plus new tokens, and
  capped new tokens at 32 by default.

## 10.0.0

Production rewrite of the lab around measurement on real model internals.

- Attention capture now runs in a separate process and streams one decoder layer
  at a time from the pinned Qwen2.5-Coder-1.5B snapshot, so the web server and
  workers never load PyTorch.
- The benchmark replays captured Q/K/V through isolated HTTP workers, keeps a
  dense reference on the coordinator, checks soundness of every certificate, and
  runs a disjoint control that must match dense attention to floating point.
- Selection now runs in one pass when overlap is zero and records a retention
  curve for every layer and KV head.
- FLOP accounting reports bounds, selection statistics, and merge work separately
  from the dense baseline.
- Added per-run artifacts: request, capture, benchmark, cluster, summary,
  generation, event log, worker logs, and a manifest with SHA-256 hashes.
- Added bounded inputs, capture and request timeouts, disk headroom checks,
  cancellation, one active run, and a per-run worker token.
- Added signed session cookies, optional API key authentication, origin and host
  guards, body size limits, security headers, and an artifact whitelist.
- Replaced the CDN dashboard with a self-contained UI that reports the real
  schema and labels Ollama generation as a standard forward pass.
- Removed the synthetic manager that generated Q/K/V from token ids and the old
  SSE endpoint that reported simulated numbers.
- Added tests for the protocol, worker API, cluster lifecycle, real capture,
  benchmark integration, and application security, plus CI and pinned
  dependencies.

## 9.0.0

- Reworked capture to match Hugging Face Qwen2 layers exactly and established the
  measured negative result for certified pruning on dense attention.
