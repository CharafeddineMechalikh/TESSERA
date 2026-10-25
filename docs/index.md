---
description: "Tessera documentation for distributed LLM inference across edge devices, with exact partitioned attention, a distributed KV cache, and certified pruning."
---

# Tessera

**Distributed LLM inference across edge devices.** Tessera partitions transformer
attention across independent worker processes, merges the partial results exactly
in float64, and decodes tokens through a KV cache that lives on the workers. It
runs Qwen2.5-Coder-1.5B on CPU today, and the remaining layers, container
packaging, and multi-node orchestration are the next stages.

Start with the [documentation index](00-index.md), which sets the reading order.
Each document assumes the ones above it.

| Document | Covers |
|---|---|
| [Architecture](01-architecture.md) | process model, run lifecycle, isolation boundaries |
| [Numerical protocol](02-protocol-math.md) | partitioning, bounds, certificate, merge, decode invariants |
| [Worker protocol](03-worker-protocol.md) | every worker endpoint and payload |
| [Coordinator API](04-coordinator-api.md) | HTTP API, auth modes, SSE events, artifacts |
| [Configuration](05-configuration.md) | every environment variable and validator rule |
| [Operations](06-operations.md) | measured costs, RAM admission, failure modes |
| [Tests and CI](07-testing.md) | what each test file protects |
| [Roadmap](08-roadmap.md) | what runs today and what comes next |

Measured results, installation, and security notes live in the
[README](https://github.com/CharafeddineMechalikh/TESSERA#readme). The project is
released under the [MIT license](https://github.com/CharafeddineMechalikh/TESSERA/blob/main/LICENSE).
