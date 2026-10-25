# Tests and CI

```bash
python -m pytest                     # everything, 78 tests collected here
python -m pytest -m "not integration" -q   # skip cluster-spawning tests
python -m pytest -m "not real_model" -q    # skip the pinned-snapshot tests
python -m pytest tests/test_protocol.py -q
```

78 tests collect on this machine: 66 with no marker, 3 marked `real_model`, 10
marked `integration` (one test carries both markers, so the marked sets overlap).
`pytest.ini` sets `testpaths = tests`, declares both markers, and ignores
`DeprecationWarning`.

## What each file protects

| File | Test functions | Guarantees |
|---|---|---|
| `test_protocol.py` | 10 | partition coverage, deterministic primary ownership, multiplicity counted per position not per vocabulary id, bound soundness, merge exactness with duplicates, stability under large offsets, omitted-mass and certified-error upper bounds, causal worker merge against dense attention, empty-visible workers ignored |
| `test_worker_api.py` | 13 | the whole HTTP contract: public health versus authenticated status, upload and bounds, session isolation, query shape rejection, oversize rejection, TTL expiry, distributed stats against dense attention, `primary` subset ignoring overlap, malformed NPZ, generation auth and init, causal attend against dense, append idempotency and conflict rejection, foreign positions and bad shapes |
| `test_app.py` | 17 | signer round trip and expiry, login lockout, security headers and cookie on `/`, `/api` auth, cross-origin POST rejection, same-origin acceptance, request validation, public-config limits, the 1000-token default, keyless re-issue versus key mode rejection, host guard, one active run, invalid config, event-stream replay and its bound |
| `test_config_limits.py` | 7 | the validator rules in [configuration.md](configuration.md), including that the capture context limit is untouched by the generation defaults |
| `test_generation_artifact.py` | 5 | manager-side validation of `generation.json`, parametrized rejections, missing or malformed artifact, and the `not_run` versus `failed` distinction in failure summaries |
| `test_benchmark.py` | 3 | full orchestration on a synthetic capture: exact at strict tolerance, single-worker selection at loose tolerance, and overlap re-query exactness at full retention |
| `test_cluster.py` | 2 | real process lifecycle: spawn workers, upload shards, query, merge, tear down; plus generation admission charging the strided KV total once |
| `test_capture_equivalence.py` | 2 | the capture path reproduces Hugging Face `Qwen2Attention` and `Qwen2DecoderLayer` on a tiny random Qwen2 built in the test |
| `test_generation_equivalence.py` | 3 | distributed greedy decoding matches `Qwen2ForCausalLM` exactly at 2 and 3 workers, the runner emits real distributed tokens, and a capture token mismatch is rejected |
| `test_real_model.py` | 2 | layer 0 of the pinned snapshot matches Hugging Face, and a full capture has the right shapes and stays finite |
| `test_real_generation.py` | 1 | the pinned model produces tokens through a real worker cluster end to end |

## Design rules this suite follows

Synthetic models are built at runtime from a tiny Qwen2 config, so equivalence
tests need no download and run in CI. They are still real `Qwen2ForCausalLM`
instances with real attention layers; there are no stubs standing in for the model
or the dispatch path.

The three heavy guarantees are all cross-checks against an independent
implementation rather than self-consistency:

| Check | Independent oracle |
|---|---|
| distributed merge | dense attention on the same captured arrays |
| capture and decoder math | Hugging Face `Qwen2` modules |
| generated tokens | `Qwen2ForCausalLM.generate` greedy, token for token |

`real_model` tests skip when the pinned snapshot is not in the local Hugging Face
cache, which is why CI is green without a 3.1 GB download. `integration` tests
spawn loopback worker processes and cost about 72 MB of RAM each, so on a busy
machine run them when nothing else is decoding.

## CI

`.github/workflows/ci.yml`, on push and pull request:

| Step | Command |
|---|---|
| Runtime | Python 3.12 on ubuntu-latest, pip cache, 45 minute timeout |
| Dependencies | `python -m pip install -r requirements.txt`, all pins exact |
| Compile | `python -m compileall -q app distributed_llm` |
| Tests | `python -m pytest -q` with `LAB_OUTPUT_DIR` pointed at the runner temp dir |
| On failure | uploads that output directory for 7 days |

CI does not download the model, so the 3 `real_model` tests skip there and the
suite runs the synthetic and HTTP coverage. That is a deliberate trade: the
snapshot equivalence is verified on the machine that owns the snapshot, and the
commit that changes capture or decode math must be run locally before it is
pushed. `test_generation_admission_charges_the_strided_kv_total_once` takes about
30 s on its own because it really spawns two worker processes.

## Adding a test

- Numerical property: put it in `test_protocol.py`, float64 arrays only, no I/O.
- Worker behavior: `test_worker_api.py`, using the app factory with an explicit
  token and session so isolation is part of the assertion.
- HTTP or lifecycle contract: mark `integration` if it spawns processes.
- Anything that needs the pinned weights: mark `real_model` and let it skip.
- A new limit: extend `test_config_limits.py` with the rejection you intend, not
  only the acceptance. Rejected defaults that would break every run are the point.
