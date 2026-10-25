# Contributing to Tessera

Tessera is a distributed inference framework whose numbers come from real
tensors, real worker processes, and real runs. Contributions are welcome under
the [MIT license](LICENSE).

## Before you start

Read [docs/00-index.md](docs/00-index.md), then the document behind the part you
want to change:

| Area | Read first |
|---|---|
| Partitioning, bounds, certificate, merge | [docs/02-protocol-math.md](docs/02-protocol-math.md) |
| Worker endpoints and payloads | [docs/03-worker-protocol.md](docs/03-worker-protocol.md) |
| Coordinator API and artifacts | [docs/04-coordinator-api.md](docs/04-coordinator-api.md) |
| Configuration and limits | [docs/05-configuration.md](docs/05-configuration.md) |
| Running a long decode | [docs/06-operations.md](docs/06-operations.md) |
| The test suite | [docs/07-testing.md](docs/07-testing.md) |

Check [docs/08-roadmap.md](docs/08-roadmap.md) for what is in scope. Open an
issue before a large change so the design can be discussed before code exists.

## Set up

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux or macOS
source .venv/bin/activate

pip install -r requirements.txt
python -m pytest -m "not real_model"
```

Most of the suite needs no model download. The `real_model` tests use the pinned
snapshot and skip when it is not cached:

```bash
hf download Qwen/Qwen2.5-Coder-1.5B \
    --revision df3ce67c0e24480f20468b6ef2894622d69eb73b
```

Run everything before opening a pull request:

```bash
python -m pytest
```

## What a change has to preserve

- **Exactness.** Distributed attention must match dense attention on the same
  captured arrays, and distributed decoding must match `Qwen2ForCausalLM`
  token for token. A change that loosens either comparison needs a measurement
  that justifies it.
- **Honest measurement.** Report what the run produced, including overhead,
  full retention, and negative results. Never relax a certificate or a test
  threshold to make a result look better.
- **Isolation.** Each run owns its worker processes, session token, output
  directory, and artifacts. A change must not share worker state between runs
  or expose one run's files to another.
- **Fail closed.** Unknown fields, out-of-range values, a mismatched token
  count, or a missing artifact must be rejected with a typed error, not
  ignored. A requested stage that fails fails the run.
- **Bounded resources.** Every new transport call needs a timeout, every upload
  a size cap, and every run path a clean teardown.

## Tests

- Add a test that fails before the fix and passes after it.
- Numerical changes need an independent oracle: dense attention for the
  protocol, `Qwen2ForCausalLM` for decoding. Do not compare an implementation
  against itself.
- The tiny Qwen2 models built in the test suite are real models with real
  attention layers. Do not replace them with stubs or record-and-replay fakes.
- Mark tests that spawn clusters as `integration`, and tests that need the
  pinned snapshot as `real_model`. CI runs neither snapshot download nor GPU
  code.
- Keep the suite fast enough for CI. A long decode or a large capture belongs
  behind a marker and a clear skip condition.

## Documentation

- Behavior changes update the numbered document that describes the behavior,
  plus `CHANGELOG.md` for functional work. Purely cosmetic or documentation
  changes do not get a changelog entry.
- `CHANGELOG.md` records what changed and why it matters, not a file-by-file
  diff.
- Diagrams are Mermaid so GitHub renders them inline.
- Prose avoids em dashes.

## Style

- Python 3.12. Standard library first; justify a new dependency in the pull
  request, including its license.
- Keep functions small enough to test directly, and prefer explicit values over
  hidden defaults.
- Comments explain why something is done, not what the code already says.
- No secrets in code, tests, logs, or artifacts. Configuration comes from the
  environment variables documented in
  [docs/05-configuration.md](docs/05-configuration.md).

## Security

Report vulnerabilities privately through the Security tab, using "Report a
vulnerability". Do not open a public issue with exploit details. Include the
impact, the affected endpoint or command, and a minimal reproduction. Do not
attach credentials, real prompts, or artifacts that contain personal data.

## License

By contributing, you agree that your contribution is licensed under the MIT
license in [LICENSE](LICENSE).
