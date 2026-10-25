---
description: "Entry point for the Tessera documentation set, with the reading order, current measured results, and the glossary."
---

# Tessera: documentation

Tessera is a framework for running LLM inference across edge devices. It starts
where the cost is: attention work grows quadratically with context length, so
attention is partitioned across independent workers, computed there, and merged
exactly. The remaining layers, container packaging, and orchestration are the
next stages, tracked in [08-roadmap.md](08-roadmap.md).

Today the framework captures real Q/K/V from Qwen2.5-Coder-1.5B, benchmarks a
certified pruning protocol over those tensors, and decodes tokens through a
distributed KV cache on the same worker contract. Everything here is derived from
the code in `app/` and `distributed_llm/`, and every number quoted is a
measurement from this machine, not an estimate.

## How to read the set

The file names carry the reading order. Start here, then go down the list; each
document assumes the ones above it.

| Document | Read it for |
|---|---|
| [00-index.md](00-index.md) | what the framework is, current results, glossary, and this reading order |
| [01-architecture.md](01-architecture.md) | the process model, the run lifecycle, and where each stage executes |
| [02-protocol-math.md](02-protocol-math.md) | the numerical contract: partitioning, bounds, retention, merge, invariants |
| [03-worker-protocol.md](03-worker-protocol.md) | every worker endpoint, payload field set, and status code |
| [04-coordinator-api.md](04-coordinator-api.md) | the HTTP API, auth modes, SSE events, artifacts, error taxonomy |
| [05-configuration.md](05-configuration.md) | every environment variable, validator rule, and worked example |
| [06-operations.md](06-operations.md) | measured costs, RAM admission, failure modes, artifact verification |
| [07-testing.md](07-testing.md) | what each test file protects, the markers, and what CI runs |
| [08-roadmap.md](08-roadmap.md) | what runs today, what comes next, and the open questions |

Read [01-architecture.md](01-architecture.md) before touching `worker_node.py`,
`cluster.py`, or `gen_client.py`; read [03-worker-protocol.md](03-worker-protocol.md)
before debugging a 4xx from a worker.

## Current results

Measured on the reference machine (Windows 11, Python 3.12, CPU only):

| Result | Value |
|---|---|
| Exact merge error, 4 workers, 243 tokens, 56 units | 4.26e-14 relative |
| Disjoint control error | 4.26e-14 relative |
| Certified pruning at `head_dim = 128` | never triggers; online FLOP savings -1.85 percent, worse once the bounds pass is charged |
| 1000-token distributed decode, 3 workers | completed, 7.93 s per token, decoder peak 728.6 MB |

A separate planted-needle sweep in
[02-protocol-math.md](02-protocol-math.md) tests whether the certificate responds
to a known separation. It is synthetic and excluded from this table.

## The three-sentence version

The web server loads no model. It spawns a capture subprocess, then a cluster of
isolated HTTP worker processes on loopback ports, then a benchmark that
partitions the captured attention across those workers and merges their replies
in float64, and finally a second cluster plus a layer-streamed decoder that
produces the generated text through the same distributed attention path. Every
artifact is hashed, and a run that cannot prove its own numbers fails loudly
rather than reporting a nice result.

## Glossary

| Term | Meaning |
|---|---|
| Unit | One `(layer, kv_head)` pair. Qwen2.5-Coder-1.5B has 28 layers and 2 KV heads, so 56 units |
| Shard | The subset of token positions (plus their K/V) held by one worker in the benchmark cluster |
| Generation cache | Per-unit positions, keys and values stored on a worker in the decode cluster |
| Ownership | Which worker stores a position. The benchmark uses a hash or a contiguous chunk, the decode path uses `position % K == worker_id` |
| Retention | The set of workers the coordinator asks for real softmax statistics after pruning on the bounds |
| Generation window | `LAB_MAX_GENERATION_POSITIONS`, the ceiling on prompt tokens plus generated tokens |
| Disjoint control | A merge over a different worker set that must match the dense reference to floating point |
| Overlap control | A full-ownership merge with `-log(c)` multiplicities that must match the dense reference, which validates the overlap correction itself |
| Planted needle | A synthetic key placed parallel to the query so its score is `log(N) + margin`. It tests the certificate's response to known separation, not model behavior |
