# Distributed Attention Lab: documentation

This folder is the reference for the lab that measures a distributed attention
protocol on real Q/K/V tensors captured from Qwen2.5-Coder-1.5B, and for the
distributed decode path that turns those same workers into a KV cache.

Everything here is derived from the code in `app/` and `distributed_llm/`, and
every number quoted is a measurement from this machine, not an estimate.

## How to read the set

| Document | Read it when |
|---|---|
| [architecture.md](architecture.md) | You need the process model, the run lifecycle, or where a stage executes |
| [worker-protocol.md](worker-protocol.md) | You are touching `worker_node.py`, `cluster.py`, or `gen_client.py`, or debugging a 4xx from a worker |
| [coordinator-api.md](coordinator-api.md) | You are calling the HTTP API, driving the dashboard, or adding a route or SSE event |
| [configuration.md](configuration.md) | You are changing a limit, an environment variable, or a validator rule |
| [protocol-math.md](protocol-math.md) | You need the numerical contract: partitioning, bounds, retention, merge, invariants |
| [operations.md](operations.md) | You are running long decodes, hitting RAM admission, or triaging a failed run |
| [testing.md](testing.md) | You are adding tests or wondering what CI actually covers |

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
