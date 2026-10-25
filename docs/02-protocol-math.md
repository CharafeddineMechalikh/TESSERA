# Numerical protocol: exact distributed attention

The protocol that makes Tessera's distributed attention exact lives in
`distributed_llm/protocol.py`, which is pure float64
NumPy with no HTTP and no model code, precisely so the equivalence and bound
properties can be tested directly. Workers import the kernels, the coordinator
imports the merges, and both call the same functions.

Conventions: `N` prompt positions, `K` workers, `H` query heads per KV group
(`q_per_kv`, 6 for Qwen2.5-Coder-1.5B), `d` head dimension (128), and
`scale = 1/sqrt(d)`.

## Ownership and partitioning

```mermaid
flowchart LR
    P["positions 0..N-1"] --> S{"strategy"}
    S -- "pseudorandom" --> A["owner = fmix32(position, seed) % K"]
    S -- "contiguous" --> B["owner = min(position // chunk, K-1)"]
    A --> O["optional overlap: with probability overlap_pct/100 a second copy on fmix32(position, seed+0x1337) % K"]
    B --> M["optional overlap: a margin of chunk*overlap_pct/100 tokens shared with the neighbour"]
    O --> R["assignments per worker, counts per position"]
    M --> R
```

`fmix32` is the MurmurHash3 32-bit finalizer, seeded with `0x977F23A7`. It is
implemented bit-identically to the earlier JavaScript prototype, so historical
partition tables remain comparable.

Invariants asserted inside `partition_tokens` on every call:

| Check | Failure |
|---|---|
| every position has at least one copy | `AssertionError: partition left tokens uncovered` |
| `primary_worker` recomputed matches the assignment pass | `AssertionError: primary assignment is not deterministic` |
| argument ranges, `N >= 1`, `K >= 1`, `0 <= overlap <= 100`, known strategy | `ValueError` with the specific complaint |
| `contiguous` requires `N >= K` | `ValueError: contiguous partitioning requires n_tokens >= workers` |

Multiplicities are counted per position, never per vocabulary id. A prompt that
repeats the same token at two positions has two distinct softmax entries, and
collapsing them would silently change the denominator.

## Block bounds

For each unit `(layer, kv_head)` and each worker, the shard stores a centroid and
a radius at upload time:

```
centroid = mean(keys, axis=0)
radius   = max ||key - centroid||
bound(head) = (q[head] . centroid + ||q[head]|| * radius) * scale
```

That is Cauchy-Schwarz over the whole block, so it upper-bounds every score in the
block at once. It is therefore a bound on the maximum, which is what pruning needs,
and it is loose by construction: the looseness grows with `sqrt(d)`, and that is
the whole story of the measured result below.

The query must be unnormalised. Normalising it here would make the bound
incompatible with the real scores, which also use the raw query.

## Retention, certificate, and correction

```mermaid
flowchart TD
    B["bounds per worker per head"] --> Ord["sort by descending bound, ties by worker id"]
    Ord --> M["m from 1 to K"]
    M --> Ask["ask the newly retained workers, or re-ask all of them when overlap is on"]
    Ask --> Merge["merge their local statistics"]
    Merge --> Cert["certified error = 2 * v_max * omitted mass"]
    Cert --> Ok{"certified is at most error_tol?"}
    Ok -- yes --> Keep["accept this retained set, break"]
    Ok -- no --> More{"m below K?"}
    More -- yes --> M
    More -- no --> Fail["BenchmarkError: no retained set satisfied the tolerance, including all workers"]
```

The coordinator sorts workers by descending `max over heads` bound (ties broken by
worker id, so the order is deterministic) and walks `m = 1..K`:

1. Ask the newly retained workers for local statistics, or re-ask all retained
   workers when overlap is enabled. Incremental accumulation is only valid at
   `overlap_pct == 0`, because with overlap the `-log(c)` correction for a
   position changes when its second copy becomes retained.
2. Merge into a candidate output.
3. Compute, per head, the omitted mass over positions with zero retained copies:

```
z_pruned  = gamma * sum over pruned unique tokens of exp(upper_bound - max_retained_score)
omitted   = z_pruned / (z_retained + z_pruned)
certified = 2 * v_max * omitted
```

`v_max` is the largest value-vector norm in the unit, taken from the dense
reference. `gamma` is `LAB_SAFE_GAMMA`, default 1.05. `safe_exp` clamps at
exponent 700 and maps `-inf` to exactly 0.0 so an empty block cannot raise. A
non-finite denominator returns 1.0 when there is pruned mass and 0.0 when there is
none, which is the conservative direction.

4. Accept the first `m` with `certified <= error_tol`. If none qualifies, including
   `m = K`, raise `BenchmarkError: no retained set satisfied the tolerance,
   including all workers`. Full retention yields zero omitted mass, so this error
   means the certificate is unsatisfied even at complete retention, which cannot
   happen by construction and therefore signals a real defect.

### Overlap correction

A position retained `c` times contributes `exp(score - log c)`
per copy, so the `c` copies sum to exactly the unreplicated mass. `token_log_correction`
supplies the `-log c` term, `retained_multiplicities` counts the copies over the
retained set only, and the worker drops the correction entirely for the `primary`
subset, where each position appears exactly once by construction.

## Soundness check

For the selected set, the coordinator compares the merged output against the dense
reference computed on the coordinator from the same captured arrays:

```
actual   = || dist_out - full_out ||
allowed  = certified + 1e-8 * (1 + || full_out ||)
```

Any `actual > allowed` raises `BenchmarkError: soundness check failed at layer L
kv H`. This is the property the certificate claims, checked rather than assumed,
per unit and per head.

## Disjoint control

After selection, every worker is queried again with `subset: "primary"` and no
multiplicity correction, and the merged result must match the dense reference to a
relative error below `1e-8`. The control uses a different selection of rows (each
position exactly once, by owner) than the retained path (each copy of a retained
position), so a self-consistent but wrong correction cannot pass it. It is the
check that would catch a partition bug that the certificate, which is computed
from the same bounds that the bug corrupted, would happily certify.

## Overlap control

The disjoint control never sends a duplicated position, so it cannot validate the
`-log(c)` correction. A second control runs only when `overlap_pct > 0`: the
coordinator asks every worker for every assigned copy, merges with
`retained_multiplicities` over the full worker set, and compares the result with
dense attention.

```
retained     = all workers
multiplicity = count of copies per position over the retained set
error        = || merged(overlap) - full_out || / max(|| full_out ||, 1e-30)
```

The run fails with `BenchmarkError: overlap control does not match dense attention`
above `1e-8`. Unlike the disjoint control, this path exercises the correction term
on positions that really are retained twice, so a wrong multiplicity map, a
sign error in `-log c`, or a worker that silently de-duplicates its shard stops the
run instead of producing a plausible number. The per-unit result is recorded as
`overlap_control_rel_error` and its cost under `overlap_control` in the stage
timings.

Both controls are verification overhead. Their query round trips are timed and
reported, but they are not charged to `distributed_flops`, because they do not
exist in a deployment that trusts the protocol. They are what makes trusting it
defensible.

## Merge

The merge is a two-level stable log-sum-exp. With per-worker maxima `m_w`,
denominators `s_w`, and locally normalised outputs `o_w`:

```
M = max_w m_w
weight_w = s_w * exp(m_w - M)
out = sum_w weight_w * o_w / sum_w weight_w
```

Subtracting the global maximum keeps every exponent at or below zero, so no
overflow is possible. Workers with no contribution report `m_w = -inf`, `s_w = 0`,
`o_w = 0` and are excluded explicitly rather than through `exp(-inf - (-inf))`,
which would produce NaN. `merge_worker_stats` merges `(workers, heads)` statistics
for a single query, `merge_stats_batch` returns only the outputs, and
`merge_causal_stats` handles the batched decode case where the statistics are
`(workers, queries, heads)`.

## Distributed causal attention (decode)

The decode path replaces hash-based selection with strided ownership: position `p`
lives on worker `p % K`, exactly once, per unit. The worker kernel
`causal_local_stats_batch` masks by position rather than by index:

```
visible[key, query] = key_position <= query_position
scores  = einsum(q[query, head, :], keys) * scale, masked to -inf
lmax    = max over visible keys
out     = (exp(scores - lmax) @ values) / sum(exp(scores - lmax))
```

Keys, values, and positions are appended in sorted order and a query batch is
strictly increasing in position, so the visible prefix is contiguous and the mask
is a prefix condition, not a full matrix scan, in intent. Both sides validate the
sortedness on ingress; unsorted or duplicated positions are 422.

```mermaid
flowchart LR
    Q["query positions and q"] --> All["every worker computes its causal prefix"]
    All --> Merge["float64 log-sum-exp merge"]
    Merge --> V1{"sum of visible_keys == position + 1?"}
    V1 -- no --> Raise["GenerationError"]
    V1 -- yes --> V2{"owned_keys == (max_position - worker_id) // K + 1?"}
    V2 -- no --> Raise
    V2 -- yes --> Out["merged attention output for the next layer"]
```

Two invariants are verified by the decoder on every attend round trip:

| Invariant | Why it catches real bugs |
|---|---|
| `sum over workers of visible_keys[query] == query_position + 1` | any lost, duplicated, or wrongly routed key breaks the softmax denominator |
| `owned_keys == (max_position - worker_id) // K + 1` | proves strided routing stored each key up to the window on exactly one worker |
| merged denominator `> 0` and merged output finite for every query head | a query that saw no key anywhere, or a NaN anywhere in the merge, stops the decode |

A mismatch raises `GenerationError`, and a run that raises fails. There is no path
where the decode continues on a broken cache.

## FLOP accounting

`benchmark.py` counts arithmetic, not wall time, and reports it per unit:

| Term | Formula | Meaning |
|---|---|---|
| baseline | `H * N * (4d + 3)` | dense attention for the query |
| bounds | `K * H * (2d + 2) + H * (2d + 1)` | per-worker dot and norm work, plus query norms |
| stats | `H * occurrences * (4d + 3)` | real attention over retained rows, counting selection work actually performed |
| merge | `retained * H * (2d + 1)` | weighted combination |
| preprocessing | `5 * assigned_rows * d` | centroid and radius per unit at shard upload |
| overlap extra | `H * duplicate_occurrences * (4d + 3)` | the part of `stats` spent on secondary copies |

`preprocessing` is the offline bounds pass. A worker computes a centroid and a
radius for every unit while it loads its shard, so the work is proportional to the
rows it actually holds: `assigned_rows = N + extra copies`. It is reported
separately rather than folded into `stats`, because it is the one term a real
deployment can pay once.

`savings = 100 * (1 - distributed / baseline)` per unit, and
`net_flop_savings_pct` uses the totals across units. `occurrences` is the reason a
pruned run can still lose: incremental selection queries each retained worker once
in bound order, so a selection at `m = K` costs the same row count as the dense
baseline plus the bounds stage, which is a net loss. With overlap the requery
protocol re-reads duplicated rows, and `overlap_stats_extra_flops` isolates that
share. It is a decomposition of `stats`, not an additional charge.

### Reporting modes for the bounds pass

Three figures are reported, and the dashboard can show any of them:

| Mode | Formula | When it is the honest one |
|---|---|---|
| online | `distributed_flops` | the bounds are already cached from a previous run |
| charged | `distributed_flops + preprocessing_flops` | a single run pays for its own bounds |
| amortised | `distributed_flops + preprocessing_flops / R` | `R` runs share one cached bounds pass |

`net_flop_savings_with_preprocessing_pct` is the charged figure. The break-even
reuse count is

```
B = baseline_flops
D = distributed_flops
P = preprocessing_flops
R* = floor(P / (B - D)) + 1   if B > D, otherwise never
```

so `R*` is the smallest number of runs that share one bounds pass before the
charged path beats dense attention on total FLOPs. When the online path already
loses (`D >= B`), no reuse count can rescue it and the dashboard reports `never`.
The benchmark emits `preprocessing_break_even_runs: null` for that case and for
`P = 0`.

Captured centroids and radii live in the worker shards for the life of a session,
but they are not persisted across runs. There is no cross-run cache to point at, so
the amortised figure is an accounting mode, not a measured cache hit; the default
is `R = 1`.

## The measured result: pruning does not trigger

A full run on 2026-10-24, 243 context tokens, 4 workers, overlap 0, `epsilon =
1e-3`:

| Retained workers | Certified error | Actual relative error |
|---|---|---|
| 1 | 9.617 | 0.258 |
| 2 | 9.617 | 0.154 |
| 3 | 9.617 | 0.082 |
| 4 | 0.0 | 4.26e-14 |

The certificate is valid and useless at this geometry: it stays above 9 while the
true error is already under 0.1, so the protocol retains every worker and reports
`-1.85` percent net FLOP savings, meaning pure overhead. Exact merge error and
disjoint control error are both 4.26e-14, which is float64 round-off, so the
distributed path is numerically correct and the pruning idea is what fails.

The framework reports this instead of relaxing the certificate, tightening
`gamma`, or silently switching to a tolerance the data cannot meet. If you change
the bound estimator, expect that 4.26e-14 and a mean retained fraction of 1.0 to
move, and add a test that states which one you intend to move.

## The planted-needle control experiment

The prototype's contrast comes from its generator, not from a better estimator
on the same data. Its scenarios place a token whose score against the query is
`log(N) + margin`, so one token carries most of the softmax mass by construction,
and the per-block radius is small relative to that gap. The repository now ships
the same construction as a labeled diagnostic:

```bash
python -m distributed_llm.scenario_sweep --out-dir output/sweeps/needle
```

The generator plants keys parallel to the query, which makes each needle score
exactly `log(N) + margin`, and draws every other key and value as small
isotropic noise. It writes the normal capture schema, so the worker cluster,
certificate, and both controls are the ones used on real Qwen2 tensors. It is
still synthetic: no model produced these tensors.

Measured on 2026-10-25, 64 tokens, 2 layers, 2 KV heads, 2 query heads,
`head_dim = 8`, background and value scale 0.05, pseudorandom partition, seed 11:

| Scenario | Workers | `eps` | Overlap | Retained | Certified | Relative | Online % | Charged % |
|---|---|---|---|---|---|---|---|---|
| no needle | 2 | 1e-3 | 0 | 2.00/2 | 0.0 | 3.9e-16 | -3.88 | -61.03 |
| no needle | 4 | 1e-3 | 0 | 4.00/4 | 0.0 | 2.2e-16 | -7.01 | -64.15 |
| 1 needle, margin 2.5 | 2 | 1e-3 | 0 | 2.00/2 | 0.0 | 4.1e-16 | -3.88 | -61.03 |
| 1 needle, margin 2.5 | 4 | 1e-2 | 0 | 3.25/4 | 9.9e-3 | 1.7e-2 | 8.79 | -48.35 |
| 1 needle, margin 6 | 2 | 1e-3 | 0 | 1.00/2 | 8.1e-4 | 1.3e-3 | 45.31 | -11.83 |
| 1 needle, margin 6 | 4 | 1e-2 | 0 | 1.00/4 | 1.3e-3 | 2.1e-3 | 76.52 | 19.37 |
| 1 needle, margin 6 | 4 | 1e-3 | 0 | 2.00/4 | 8.6e-4 | 1.5e-3 | 48.42 | -8.73 |
| 1 needle, margin 6 | 4 | 1e-2 | 15 | 1.00/4 | 1.3e-3 | 2.0e-3 | 74.96 | 10.67 |

The needle-free control retained every worker in all 8 control cells, so the
certificate is not accepting arbitrary data. The margin changes the outcome
monotonically: at margin 2.5 and `eps = 1e-3` the certificate still retains all
workers, while at margin 6 it accepts one worker at the same tolerance. The
prototype's collapse is therefore real but conditional. It needs enough
separation for the chosen tolerance, and the charged figure shows that the
bounds pass can still erase the saving. The single-run best here is margin 6,
4 workers, overlap 0, `eps = 1e-2`: 76.52 percent online and 19.37 percent
charged, with a certified error of 1.3e-3 and an actual relative error of 2.1e-3.

That result does not move the real-prompt number above. It localizes the
disagreement: the certificate responds when the data is separable by
construction, and stays conservative when it is not. A sweep row is evidence
about the pruning mechanism, never about a prompt.
