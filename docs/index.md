# Distributed Attention: documentation

Distributed Attention is a framework for running LLM inference across edge
devices. It starts where the cost is: attention work grows quadratically with
context length, so attention is partitioned across independent workers, computed
there, and merged exactly. The remaining layers, container packaging, and
orchestration are the next stages, tracked in [roadmap.md](roadmap.md).

Today the framework captures real Q/K/V from Qwen2.5-Coder-1.5B, benchmarks a
certified pruning protocol over those tensors, and decodes tokens through a
distributed KV cache on the same worker contract. Everything here is derived from
the code in `app/` and `distributed_llm/`, and every number quoted is a
measurement from this machine, not an estimate.

## How to read the set

| Document | Read it when |
|---|---|
| [roadmap.md](roadmap.md) | You want the current state, the stage plan, and the open questions |
| [architecture.md](architecture.md) | You need the process model, the run lifecycle, or where a stage executes |
| [worker-protocol.md](worker-protocol.md) | You are touching `worker_node.py`, `cluster.py`, or `gen_client.py`, or debugging a 4xx from a worker |
| [coordinator-api.md](coordinator-api.md) | You are calling the HTTP API, driving the dashboard, or adding a route or SSE event |
| [configuration.md](configuration.md) | You are changing a limit, an environment variable, or a validator rule |
| [protocol-math.md](protocol-math.md) | You need the numerical contract: partitioning, bounds, retention, merge, invariants |
| [operations.md](operations.md) | You are running long decodes, hitting RAM admission, or triaging a failed run |
| [testing.md](testing.md) | You are adding tests or wondering what CI actually covers |

## Current results

Measured on the reference machine (Windows 11, Python 3.12, CPU only):

| Result | Value |
|---|---|
| Exact merge error, 4 workers, 243 tokens, 56 units | 4.26e-14 relative |
| Disjoint control error | 4.26e-14 relative |
| Certified pruning at `head_dim = 128` | never triggers; net FLOP savings -1.85 percent |
| 1000-token distributed decode, 3 workers | completed, 7.93 s per token, decoder peak 728.6 MB |

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
