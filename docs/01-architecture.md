---
description: "Tessera process model: the web coordinator, capture and generation subprocesses, isolated worker clusters, and the run lifecycle."
---

# Tessera architecture

## Process model

Tessera is one web process, up to two model-bearing subprocesses, and two
clusters of worker processes. Nothing in the web process ever imports PyTorch.

| Process | Lives | Holds | Talks to |
|---|---|---|---|
| `app.main:app` (uvicorn, 127.0.0.1) | whole session | no weights, no tensors | browser (HTTP + SSE), subprocesses, workers |
| `distributed_llm.capture_runner` | capture stage | one decoder layer at a time, the tokenizer | disk (writes `capture.npz`), the manager over stdout |
| `distributed_llm.generation_runner` | generation stage | the layer-streamed Qwen2 decoder, no KV cache | its worker cluster over loopback HTTP, the manager over stdout |
| benchmark workers (`worker_node`) | benchmark stage | prompt shards as float64 K/V plus centroid and radius | the coordinator over loopback HTTP |
| generation workers (`worker_node`) | generation stage | per-unit generation cache as float32 K/V | the decoder over loopback HTTP |

The two clusters are separate. They get different session ids, different bearer
tokens, different ports, and different run subdirectories (`cluster/` and
`generation_cluster/`), so a leaked token from one never opens the other.

```mermaid
flowchart LR
    B["Browser dashboard<br/>app/static/js/visualizer.js"]
    subgraph S["Server process (no torch)"]
        M["FastAPI coordinator<br/>app/main.py"]
        R["RunManager<br/>distributed_llm/manager.py"]
        K["Run thread"]
        BM["benchmark.py<br/>partition, retention, merge"]
        M --> R --> K --> BM
    end
    subgraph C["Capture subprocess"]
        CR["capture_runner.py + capture.py<br/>layer-streamed Q/K/V"]
    end
    subgraph W1["Benchmark worker cluster, K processes"]
        W1A["worker 0..K-1<br/>float64 shards"]
    end
    subgraph G["Generation subprocess"]
        D["generation_runner.py + decoder.py<br/>layer-streamed Qwen2 decoder"]
        GC["gen_client.py"]
        D --> GC
    end
    subgraph W2["Generation worker cluster, K processes"]
        W2A["worker 0..K-1<br/>float32 KV cache"]
    end
    A["Run artifacts<br/>output/runs/&lt;run_id&gt;/"]

    B -- "HTTP + SSE" --> M
    K -- "subprocess, JSON lines on stdout" --> CR
    CR -- "capture.npz, capture.json" --> A
    BM -- "NPZ shards, JSON stats" --> W1A
    W1A -- "bounds and local statistics" --> BM
    K -- "subprocess, LAB_GEN_CONNECTION" --> D
    GC -- "NPZ append and attend" --> W2A
    W2A -- "NPZ statistics" --> GC
    BM --> A
    D -- "generation.json" --> A
```

## Run lifecycle

One run at a time. `RunManager.start()` rejects a second request with HTTP 409
before any process is spawned, and the whole pipeline below executes in a single
worker thread so the server keeps answering the browser while a run is live.

```mermaid
sequenceDiagram
    autonumber
    participant Br as Browser
    participant Sv as FastAPI + RunManager
    participant Cp as capture subprocess
    participant Bk as benchmark (server thread)
    participant Wb as benchmark workers
    participant Gn as generation subprocess
    participant Wg as generation workers

    Br->>Sv: POST /api/runs
    Sv-->>Br: 202 with run status
    Br->>Sv: GET /api/runs/{id}/events (SSE)
    Sv->>Cp: spawn, stream stdout events
    Cp->>Cp: tokenize, load model config, stream 28 layers
    Cp-->>Sv: one JSON line per progress event
    Cp->>Cp: write capture.npz and capture.json
    Sv->>Sv: open and verify CaptureStore, size cap
    Sv->>Bk: run_benchmark(store, cluster, config)
    Bk->>Wb: spawn cluster, POST /v1/shard per worker
    loop every unit and every retention size
        Bk->>Wb: POST /v1/bounds
        Bk->>Bk: greedy retention from the bounds
        Bk->>Wb: POST /v1/stats (retained workers only)
        Bk->>Bk: log-sum-exp merge, dense reference, disjoint control
    end
    Bk->>Wb: shutdown cluster
    Bk-->>Sv: benchmark.json payload
    Sv->>Wg: spawn second cluster
    Sv->>Wg: preflight RAM for KV plus decoder
    Sv->>Gn: spawn with LAB_GEN_CONNECTION
    loop prefill, then every generated token
        Gn->>Wg: POST /v1/gen/append (own strided positions)
        Gn->>Wg: POST /v1/gen/attend (all workers, per unit)
        Wg-->>Gn: NPZ lmax, sumexp, out, visible_keys, owned_keys
        Gn->>Gn: merge in float64, verify invariants, argmax
        Gn-->>Sv: progress line, forwarded as SSE
    end
    Gn->>Gn: write generation.json
    Sv->>Wg: shutdown cluster
    Sv->>Sv: summary.json then manifest.json
    Sv-->>Br: done event with aggregate, timings, artifacts
```

## Run states

```mermaid
stateDiagram-v2
    [*] --> accepted
    accepted --> running: run thread started
    running --> running: capture, benchmark, generation
    running --> completed: summary.json and manifest.json written
    running --> failed: typed error, failure summary written
    running --> cancelled: cancel event set
    completed --> [*]
    failed --> [*]
    cancelled --> [*]
```

A run is never left half-reported. The failure path writes a failure summary
with the exception type and message, then a manifest, then the terminal `done`
event with `status: "failed"`. A cancelled run does the same with
`status: "cancelled"`. Generation failure is a run failure: there is no path that
reports attention numbers and quietly omits the text.

## Stage placement

| Stage | Code | Where it executes | Why |
|---|---|---|---|
| Capture | `capture_runner.py`, `capture.py` | subprocess | PyTorch must be able to leave memory when capture ends |
| Partition, upload, bounds, retention, merge | `benchmark.py` via `cluster.py` | server run thread | NumPy only, no model weights |
| Decode | `generation_runner.py`, `decoder.py`, `gen_client.py` | subprocess | holds streamed weights, still no KV cache locally |
| Attention math | `protocol.py` | wherever the caller is | pure float64 NumPy, no I/O, so it is directly testable |
| Worker kernels | `worker_node.py` | worker process | sees only its shard, its session, its token |

## Subprocess contract

Both model-bearing subprocesses speak the same line protocol on stdout: one
JSON object per line, with a `stage` key. The manager forwards a line whose
`stage` matches the label it is watching, treats `stage: "done"` as the
completion record, and turns `stage: "error"` into a typed failure carrying
`error_type` and `message`. Stderr is redirected to `capture.stderr.log` or
`generation.stderr.log` and never parsed. The same protocol is what the CLI
emits, so a failure seen in the dashboard is reproducible by hand with the same
error taxonomy.

The subprocess environment is fixed by `_stream_subprocess`: `PYTHONUNBUFFERED=1`,
`PYTHONIOENCODING=utf-8`, `TOKENIZERS_PARALLELISM=false`, and the
`USE_TF`/`USE_FLAX`/`TRANSFORMERS_NO_TF` guards that stop transformers probing
optional backends. Generation additionally receives `LAB_GEN_CONNECTION`, a
JSON blob with the session id, the bearer token, and the worker base URLs. It is
passed by environment, never written to an artifact or a log.

## Isolation boundaries

```mermaid
flowchart TB
    subgraph T["Trust domain: server"]
        S1["prompt text, weights, reference output, generated text"]
    end
    subgraph U["Trust domain: one worker process"]
        W1["its shard only, its per-unit KV only"]
    end
    S1 -- "NPZ shard, bearer token, session id" --> W1
    W1 -- "local statistics only" --> S1
```

A worker rejects a shard, an append, or an attend whose session id is not the one
it booted with (HTTP 403), and rejects any body without its bearer token (401).
Ownership is checked on the worker as well as the client: an append that contains
a position the worker does not own is refused with 403, so a coordinator bug
cannot silently cross-contaminate shards. `gen_client.py` refuses to connect to a
worker URL that is not loopback HTTP, which keeps a tampered connection blob from
turning the decoder into an exfiltration client.

The full contract is in [03-worker-protocol.md](03-worker-protocol.md).

## Module map

| Module | Responsibility |
|---|---|
| `app/main.py` | routes, session auth, static dashboard, artifact downloads |
| `app/security.py` | signed cookie session, host and origin guards, body limits, login lockout |
| `app/templates/index.html`, `app/static/js/visualizer.js` | dashboard, SSE client, charts |
| `distributed_llm/config.py` | every limit, every environment variable, the validator |
| `distributed_llm/manager.py` | run lifecycle, subprocess orchestration, artifacts, event stream |
| `distributed_llm/capture*.py` | Q/K/V capture from the pinned snapshot, one layer at a time |
| `distributed_llm/cluster.py` | benchmark worker lifecycle plus HTTP client for shards, bounds, stats |
| `distributed_llm/worker_node.py` | the worker HTTP service and its kernels |
| `distributed_llm/benchmark.py` | the certified-pruning benchmark loop and its accounting |
| `distributed_llm/protocol.py` | pure numerical core: partitioning, bounds, statistics, merge, certificates |
| `distributed_llm/decoder.py` | layer-streamed Qwen2 decoder plus distributed attention |
| `distributed_llm/gen_client.py` | binary loopback client for the generation cache |
| `distributed_llm/generation_runner.py` | greedy decode loop, artifacts, invariant reporting |
| `distributed_llm/logging_utils.py` | structured event log, RSS and RAM probes |
