# Scale benchmarks

How each component of `veritor` behaves as its size parameter grows, measured by `python -m benchmarks.run` and rendered by `python -m benchmarks.report` from `docs/data/benchmarks.json`.  Each sweep records the median wall time of a few repeats, the `tracemalloc` peak and the relevant sizes, and fits `y = a · x^b` in log-log space; `b` is the exponent reported under each table.  Read *Bottlenecks* at the end for what the numbers mean.

## Manifest

- commit `ae1f20f3c6d8` on `scale-benchmarks` (dirty tree)
- CPython 3.12.11 on macOS-26.5.1-arm64-arm-64bit (arm)
- 2026-09-03T01:32:24+00:00, 3 timed repeats per point (median reported), 34.9 min in total
- times are single-threaded CPython wall clock; peak memory is the `tracemalloc` peak of one extra run, above the baseline at the call

## Contents

1. [Compile: description in, lazy circuit and index out](#1-compile-description-in-lazy-circuit-and-index-out)
2. [Lazy gate lookup: `circuit[i]` latency](#2-lazy-gate-lookup-circuiti-latency)
3. [Kind table and lazy address sets of the index](#3-kind-table-and-lazy-address-sets-of-the-index)
4. [Bound, Cost, expected_work and Optimize](#4-bound-cost-expectedwork-and-optimize)
5. [Challenge sampling: binomial count and Floyd subset](#5-challenge-sampling-binomial-count-and-floyd-subset)
6. [Merkle commitments: build, open, verify](#6-merkle-commitments-build-open-verify)
7. [The protocol end to end: prover and verifier phases, message bytes](#7-the-protocol-end-to-end-prover-and-verifier-phases-message-bytes)
8. [`output_reach` vs description size](#8-outputreach-vs-description-size)
9. [Bottlenecks](#bottlenecks)
10. [Performance bugs](#performance-bugs)

## 1. Compile: description in, lazy circuit and index out

Wall time of `Compiler.compile` (parse, validate, summarize, mark check, build `Index` and `DescriptionCircuit`) with the constructor's trace time (`G(x, a)`) and the whole `Compile(G, x, a)` alongside.  The x-axis is the parameter swept; `n` is the flattened gate count, `description_bytes` what the compiler actually reads.

`compile` ran in 8.17 s.

#### `cluster_d_model`

ClusterG with 2 layers, 4 requests of 4 + 4 tokens on 2 slots; compile time should follow `description_bytes` (roughly linear in d_model: the tracer unrolls each dot product), not `n` (quadratic).

| d_model | time | peak mem | trace | `Compile` (research API) | description | definitions | n | RUs | VUs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 4.76 ms | 479 KB | 1.74 ms | 7 ms | 38.9 KB | 41 | 21,177 | 9 | 3,205 |
| 8 | 4.9 ms | 484 KB | 1.65 ms | 6.49 ms | 39.6 KB | 41 | 70,633 | 9 | 6,613 |
| 16 | 5.03 ms | 514 KB | 1.48 ms | 7.39 ms | 42.1 KB | 45 | 257,097 | 9 | 14,965 |
| 32 | 5.42 ms | 524 KB | 1.52 ms | 7.63 ms | 43.2 KB | 45 | 980,233 | 9 | 37,813 |
| 64 | 5.92 ms | 532 KB | 1.6 ms | 6.89 ms | 43.7 KB | 45 | 3,827,337 | 9 | 108,085 |
| 128 | 5.86 ms | 542 KB | 1.65 ms | 7.67 ms | 44.6 KB | 45 | 15,124,873 | 9 | 346,933 |

| d_model | advice |
|---:|---:|
| 4 | 100 B |
| 8 | 100 B |
| 16 | 100 B |
| 32 | 100 B |
| 64 | 100 B |
| 128 | 100 B |

Fitted exponents: description ∝ x^0.04 (R² 0.95, 6 pts); n ∝ x^1.90 (R² 1.00, 6 pts); time ∝ x^0.07 (R² 0.93, 6 pts); trace ∝ x^-0.01 (R² 0.09, 6 pts).

#### `cluster_requests`

ClusterG d_model=8, 1 layer, 2 slots, requests of 3 + 3 tokens; every extra wave adds root steps and schedule bytes but no new kinds.  The sweep stops at 64 requests: from 96 on, the root's declared outputs (every request's tokens) resolve to more than `CompilationLimits.max_output_runs = 256` runs and the description is rejected, a limit of the toy constructor's output layout rather than of the compiler.

| requests | time | peak mem | trace | `Compile` (research API) | description | definitions | n | RUs | VUs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 3.01 ms | 298 KB | 817 µs | 4.02 ms | 23.7 KB | 35 | 13,909 | 4 | 1,699 |
| 8 | 3.57 ms | 330 KB | 1.26 ms | 4.92 ms | 25.8 KB | 35 | 53,689 | 13 | 4,849 |
| 16 | 4.34 ms | 374 KB | 1.54 ms | 5.34 ms | 28.6 KB | 35 | 106,729 | 25 | 9,049 |
| 32 | 5.37 ms | 460 KB | 2.51 ms | 21.8 ms | 34.3 KB | 35 | 212,809 | 49 | 17,449 |
| 64 | 19.4 ms | 632 KB | 11.5 ms | 13 ms | 45.7 KB | 35 | 424,969 | 97 | 34,249 |

| requests | advice |
|---:|---:|
| 2 | 68 B |
| 8 | 164 B |
| 16 | 292 B |
| 32 | 548 B |
| 64 | 1.04 KB |

Fitted exponents: advice ∝ x^0.79 (R² 0.99, 5 pts); description ∝ x^0.18 (R² 0.86, 5 pts); n ∝ x^0.99 (R² 1.00, 5 pts); time ∝ x^0.46 (R² 0.69, 5 pts); trace ∝ x^0.68 (R² 0.78, 5 pts).

#### `cluster_slots`

ClusterG d_model=8, 1 layer, `2 * slots` requests of 3 + 3 tokens: a step kind per distinct occupant tuple, so the description grows with the number of slots.

| slots | time | peak mem | trace | `Compile` (research API) | description | definitions | n | RUs | VUs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3.01 ms | 295 KB | 779 µs | 3.79 ms | 23.6 KB | 35 | 13,909 | 7 | 1,699 |
| 2 | 3.06 ms | 308 KB | 876 µs | 4.19 ms | 24.4 KB | 35 | 27,169 | 7 | 2,749 |
| 4 | 3.99 ms | 336 KB | 1.06 ms | 4.6 ms | 26 KB | 35 | 53,689 | 7 | 4,849 |
| 8 | 4.48 ms | 391 KB | 1.33 ms | 6.51 ms | 29.3 KB | 35 | 106,729 | 7 | 9,049 |
| 16 | 5.98 ms | 499 KB | 2.16 ms | 8.71 ms | 35.9 KB | 35 | 212,809 | 7 | 17,449 |
| 32 | 10.9 ms | 718 KB | 3.38 ms | 15.3 ms | 49.2 KB | 35 | 424,969 | 7 | 34,249 |

| slots | advice |
|---:|---:|
| 1 | 68 B |
| 2 | 100 B |
| 4 | 164 B |
| 8 | 292 B |
| 16 | 548 B |
| 32 | 1.04 KB |

Fitted exponents: description ∝ x^0.20 (R² 0.88, 6 pts); n ∝ x^0.99 (R² 1.00, 6 pts); time ∝ x^0.35 (R² 0.89, 6 pts).

#### `deep_repeat`

`repeat(10, ...)` nested `depth` times over a 3-gate VU: `n = 3 * 10**depth + 1`.  The compiler never touches a gate, so time and memory follow the description (`O(depth)`), not `n`.

| n (gates) | time | peak mem | depth | description | definitions | RUs |
|---:|---:|---:|---:|---:|---:|---:|
| 31 | 217 µs | 26.5 KB | 1 | 1.54 KB | 5 | 2 |
| 301 | 242 µs | 30.2 KB | 2 | 1.82 KB | 6 | 11 |
| 30,001 | 329 µs | 38.3 KB | 4 | 2.38 KB | 8 | 1,001 |
| 3,000,001 | 399 µs | 46.6 KB | 6 | 2.94 KB | 10 | 100,001 |
| 300,000,001 | 456 µs | 54.8 KB | 8 | 3.49 KB | 12 | 10,000,001 |
| 3e+10 | 542 µs | 63.1 KB | 10 | 4.05 KB | 14 | 1e+09 |
| 3e+12 | 710 µs | 70.7 KB | 12 | 4.61 KB | 16 | 1e+11 |

Fitted exponents: description ∝ x^0.04 (R² 0.98, 7 pts); peak mem ∝ x^0.04 (R² 0.98, 7 pts); time ∝ x^0.04 (R² 0.99, 7 pts).

#### `definitions`

A root calling `D` distinct RU kinds once each (`repeat(i + 1, cell)`): compile time linear in the number of definitions, i.e. in the description.

| definitions | time | peak mem | description | n |
|---:|---:|---:|---:|---:|
| 12 | 534 µs | 63.1 KB | 4.31 KB | 109 |
| 36 | 1.47 ms | 171 KB | 13.8 KB | 1,585 |
| 132 | 6.19 ms | 595 KB | 52.1 KB | 24,769 |
| 516 | 22.9 ms | 2.3 MB | 205 KB | 393,985 |
| 2,052 | 103 ms | 9.26 MB | 821 KB | 6,294,529 |
| 8,196 | 473 ms | 37.2 MB | 3.21 MB | 100,675,585 |

Fitted exponents: description ∝ x^1.01 (R² 1.00, 6 pts); peak mem ∝ x^0.99 (R² 1.00, 6 pts); time ∝ x^1.04 (R² 1.00, 6 pts).

## 2. Lazy gate lookup: `circuit[i]` latency

Per-call latency (seconds) of `DescriptionCircuit.__getitem__` for random, sequential and strided addresses, of `Index.replay_units.owner(address)` and of `Index.replay_units.unit(k)`.  `time_s` is the random-access latency.

`lookup` ran in 1.03 s.

#### `vs_n`

Three `repeat` levels of factor `k`, `n = 3 k**3 + 1`: depth is fixed, so every latency should be flat in `n` (exponent near 0).

| n (gates) | time | sequential | strided | `owner` | `unit` | depth | k |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3,001 | 6.36 µs | 6.4 µs | 6.67 µs | 2.96 µs | 3.42 µs | 4 | 10 |
| 3,000,001 | 7.27 µs | 7.49 µs | 7.38 µs | 3.41 µs | 3.48 µs | 4 | 100 |
| 3e+09 | 7.82 µs | 7.44 µs | 6.52 µs | 3.52 µs | 3.56 µs | 4 | 1,000 |
| 3e+12 | 7.43 µs | 7.35 µs | 7.36 µs | 3.46 µs | 3.59 µs | 4 | 10,000 |
| 3e+15 | 7.51 µs | 7.71 µs | 7.44 µs | 3.63 µs | 3.73 µs | 4 | 100,000 |

Fitted exponents: `owner` ∝ x^0.01 (R² 0.72, 5 pts); sequential ∝ x^0.01 (R² 0.58, 5 pts); strided ∝ x^0.00 (R² 0.29, 5 pts); time ∝ x^0.01 (R² 0.50, 5 pts); `unit` ∝ x^0.00 (R² 0.95, 5 pts).

#### `vs_depth`

`repeat(2, ...)` nested `depth` times (`n = 3 * 2**depth + 1`, up to ~10**15): latency should be linear in depth (exponent near 1).

| depth | time | sequential | strided | `owner` | `unit` | n | depth |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 5.62 µs | 5.33 µs | 5.37 µs | 2.18 µs | 1.94 µs | 13 | 3 |
| 4 | 8.31 µs | 7.82 µs | 6.78 µs | 3.57 µs | 4.03 µs | 49 | 5 |
| 8 | 12.3 µs | 12.3 µs | 12.5 µs | 9.07 µs | 8.68 µs | 769 | 9 |
| 16 | 24.2 µs | 22.5 µs | 22.8 µs | 16.8 µs | 18.5 µs | 196,609 | 17 |
| 32 | 45.9 µs | 46.2 µs | 46.5 µs | 34 µs | 36.2 µs | 1.29e+10 | 33 |
| 48 | 69.2 µs | 70.1 µs | 69.1 µs | 58.8 µs | 57 µs | 8.44e+14 | 49 |

Fitted exponents: `owner` ∝ x^1.04 (R² 0.99, 6 pts); sequential ∝ x^0.82 (R² 0.98, 6 pts); strided ∝ x^0.84 (R² 0.98, 6 pts); time ∝ x^0.80 (R² 0.99, 6 pts); `unit` ∝ x^1.06 (R² 1.00, 6 pts).

#### `cluster`

`ClusterG` circuits of growing width: the decoder's hierarchy has a fixed depth, so lookups stay flat while `n` grows four orders of magnitude.

| n (gates) | time | sequential | strided | `owner` | `unit` | depth | case |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4,437 | 7.37 µs | 7.37 µs | 7.98 µs | 1.11 µs | 1.12 µs | 6 | d4-L1 |
| 27,169 | 8.21 µs | 8.95 µs | 8.35 µs | 1.11 µs | 1.13 µs | 6 | d8-L1 |
| 509,833 | 8.76 µs | 8.26 µs | 8.28 µs | 1.27 µs | 1.33 µs | 6 | d16-L2 |
| 1,943,561 | 8.93 µs | 7.81 µs | 7.91 µs | 1.14 µs | 1.13 µs | 6 | d32-L2 |
| 7,588,105 | 8.69 µs | 8.43 µs | 8.6 µs | 1.21 µs | 1.4 µs | 6 | d64-L2 |
| 59,794,185 | 8.58 µs | 8.61 µs | 7.86 µs | 1.24 µs | 1.25 µs | 6 | d128-L4 |

Fitted exponents: `owner` ∝ x^0.01 (R² 0.51, 6 pts); time ∝ x^0.02 (R² 0.60, 6 pts); `unit` ∝ x^0.02 (R² 0.34, 6 pts).

## 3. Kind table and lazy address sets of the index

`Index.kinds()` / `kind_table()` wall time against the description, and the per-call latency of the boundary and interior address sets and the unit lookups against the number of RUs.

`kinds` ran in 4.85 s.

#### `kind_table_vs_definitions`

`Index.kind_table()` on a root calling `D` distinct RU kinds: one row per kind, linear in the description.  `kinds_s` is `Index.kinds()` alone (the table adds the totals).

| kinds | time | peak mem | `kinds()` | definitions | n |
|---:|---:|---:|---:|---:|---:|
| 12 | 94.3 µs | 17.3 KB | 84.2 µs | 11 | 109 |
| 36 | 276 µs | 39.3 KB | 260 µs | 35 | 1,585 |
| 132 | 1.02 ms | 110 KB | 986 µs | 131 | 24,769 |
| 516 | 4.45 ms | 359 KB | 4.13 ms | 515 | 393,985 |
| 2,052 | 19.1 ms | 1.42 MB | 18 ms | 2,051 | 6,294,529 |
| 8,196 | 92 ms | 5.71 MB | 88.6 ms | 8,195 | 100,675,585 |

Fitted exponents: `kinds()` ∝ x^1.06 (R² 1.00, 6 pts); peak mem ∝ x^0.89 (R² 1.00, 6 pts); time ∝ x^1.05 (R² 1.00, 6 pts).

#### `vs_output_runs`

1000 copies of one RU whose declared `Out` resolves to `R` runs (outputs at irregular gaps); every gate is a one-gate VU, so the interior is the RU's gates minus its `Out`.  `time_s` is `kind_table()`.  `boundary.rank` and a boundary `contains` miss scan the runs (`O(R)`), `boundary.unrank` bisects (`O(log R)`); the interior's `contains`/`rank` subtract the RU outputs below the address (`O(R)`) after an `O(depth)` descent to the VU, and its `unrank` bisects the steps with the same subtraction per probe (`O(R log |R_r|)`).

| runs in Out | RU size | #Out | #interior | time | ∂ `rank` | ∂ `unrank` | ∂ `contains` (miss) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 2 | 62 | 398 µs | 3.48 µs | 1.14 µs | 3.31 µs |
| 4 | 64 | 8 | 56 | 394 µs | 3.54 µs | 1.18 µs | 3.66 µs |
| 16 | 64 | 32 | 32 | 398 µs | 3.98 µs | 1.23 µs | 4.23 µs |
| 64 | 256 | 128 | 128 | 1.75 ms | 5.55 µs | 1.21 µs | 6.55 µs |
| 128 | 512 | 256 | 256 | 3.91 ms | 7.92 µs | 1.21 µs | 9.69 µs |

| runs in Out | `interior(u)` | int. `rank` | int. `unrank` | int. `contains` |
|---:|---:|---:|---:|---:|
| 1 | 3.77 µs | 1.61 µs | 5.42 µs | 1.6 µs |
| 4 | 3.35 µs | 2.12 µs | 8.56 µs | 2 µs |
| 16 | 3.49 µs | 3.3 µs | 17.2 µs | 3.64 µs |
| 64 | 3.54 µs | 8.52 µs | 65.8 µs | 8.69 µs |
| 128 | 3.62 µs | 15.9 µs | 144 µs | 16.3 µs |

Fitted exponents: ∂ `contains` (miss) ∝ x^0.21 (R² 0.87, 5 pts); ∂ `rank` ∝ x^0.16 (R² 0.82, 5 pts); ∂ `unrank` ∝ x^0.01 (R² 0.64, 5 pts); `interior(u)` ∝ x^-0.00 (R² 0.02, 5 pts); int. `contains` ∝ x^0.48 (R² 0.94, 5 pts); int. `unrank` ∝ x^0.68 (R² 0.95, 5 pts); time ∝ x^0.47 (R² 0.74, 5 pts).

#### `address_sets_vs_units_repeat`

`repeat(U, block)` of an 8-cell RU: every lookup descends by division, so all latencies should be flat in `U`.  `time_s` is `boundary.rank`.

| replay units | n | #∂ | `boundary()` | time | ∂ `unrank` | ∂ `contains` |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 2,401 | 101 | 5.42 µs | 5.29 µs | 2.01 µs | 5.07 µs |
| 1,000 | 24,001 | 1,001 | 5.62 µs | 5.38 µs | 2.07 µs | 5.18 µs |
| 10,000 | 240,001 | 10,001 | 5.46 µs | 5.04 µs | 2.13 µs | 5.33 µs |
| 100,000 | 2,400,001 | 100,001 | 5.38 µs | 5.29 µs | 2.15 µs | 5.24 µs |
| 1,000,000 | 24,000,001 | 1,000,001 | 5.37 µs | 5.41 µs | 2.13 µs | 5.27 µs |
| 10,000,000 | 240,000,001 | 10,000,001 | 5.42 µs | 5.51 µs | 2.28 µs | 5.25 µs |

| replay units | `interior(u)` | int. `unrank` | int. `contains` | `unit` | `owner` | `verification_unit` | `verification_units` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 4.68 µs | 3.43 µs | 1.63 µs | 2.01 µs | 1.95 µs | 3.05 µs | 2.19 µs |
| 1,000 | 4.77 µs | 3.48 µs | 1.41 µs | 2.04 µs | 2 µs | 3.21 µs | 2.34 µs |
| 10,000 | 4.41 µs | 3.15 µs | 1.48 µs | 2.2 µs | 2.06 µs | 3.29 µs | 2.25 µs |
| 100,000 | 4.57 µs | 3.52 µs | 1.44 µs | 2.33 µs | 2.07 µs | 3.06 µs | 2.27 µs |
| 1,000,000 | 4.43 µs | 3.46 µs | 1.45 µs | 2.03 µs | 2.05 µs | 3.02 µs | 2.3 µs |
| 10,000,000 | 4.81 µs | 3.73 µs | 1.4 µs | 2.14 µs | 2.08 µs | 3.34 µs | 2.26 µs |

Fitted exponents: ∂ `contains` ∝ x^0.00 (R² 0.40, 6 pts); ∂ `unrank` ∝ x^0.01 (R² 0.85, 6 pts); `interior(u)` ∝ x^-0.00 (R² 0.01, 6 pts); int. `unrank` ∝ x^0.01 (R² 0.24, 6 pts); `owner` ∝ x^0.01 (R² 0.73, 6 pts); time ∝ x^0.00 (R² 0.22, 6 pts); `unit` ∝ x^0.00 (R² 0.10, 6 pts); `verification_unit` ∝ x^0.00 (R² 0.07, 6 pts).

#### `address_sets_vs_units_unrolled`

A root of `U` separate `call` steps of one RU: descent bisects the root's step list, `O(log U)`; the description itself is `O(U)` (about 120 bytes per step, so 10^5 unrolled steps exceed `max_description_bytes = 10 MB`; the sweep stops at 50,000).  `time_s` is `boundary.rank`.

| replay units | n | #∂ | `boundary()` | time | ∂ `unrank` | ∂ `contains` |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 301 | 101 | 4.92 µs | 3.16 µs | 1.1 µs | 3.26 µs |
| 1,000 | 3,001 | 1,001 | 5.46 µs | 3.63 µs | 1.27 µs | 3.52 µs |
| 10,000 | 30,001 | 10,001 | 5.67 µs | 3.55 µs | 1.28 µs | 3.73 µs |
| 50,000 | 150,001 | 50,001 | 6.17 µs | 3.63 µs | 1.37 µs | 3.82 µs |

| replay units | `interior(u)` | `unit` | `owner` | `verification_unit` | `verification_units` |
|---:|---:|---:|---:|---:|---:|
| 100 | 3.66 µs | 1.13 µs | 1.13 µs | 2.08 µs | 1.25 µs |
| 1,000 | 3.7 µs | 1.3 µs | 1.25 µs | 2.24 µs | 1.39 µs |
| 10,000 | 4.33 µs | 1.26 µs | 1.2 µs | 2.31 µs | 1.4 µs |
| 50,000 | 4.26 µs | 1.34 µs | 1.3 µs | 2.73 µs | 1.41 µs |

Fitted exponents: ∂ `contains` ∝ x^0.03 (R² 0.98, 4 pts); ∂ `unrank` ∝ x^0.03 (R² 0.87, 4 pts); `interior(u)` ∝ x^0.03 (R² 0.82, 4 pts); `owner` ∝ x^0.02 (R² 0.66, 4 pts); time ∝ x^0.02 (R² 0.62, 4 pts); `unit` ∝ x^0.02 (R² 0.69, 4 pts); `verification_unit` ∝ x^0.04 (R² 0.85, 4 pts).

## 4. Bound, Cost, expected_work and Optimize

Folds over the kind table.  `time_s` is `Bound` with the Laplace fold alone (`FRONTIER_OPTIONS`: `knapsack=False, max_buckets=2**22`); `knapsack_s` is `Bound` with the default options (knapsack on a 2048-bucket grid plus Laplace); `cost_s` and `work_s` are `Cost` and `expected_work`.  Policy `q = 1/128, s = 1/8`, `eta = 1/100`.

`analysis` ran in 3.68 min.

#### `serving_request_row`

`serving_table(shape, 'request', 'row')` from toy dimensions to the 70B-class frontier shape (`d_model = 8192`, 80 layers, 2048 requests of 512 + 512 tokens, 2.7e17 gates); the number of kinds grows with the number of distinct decode contexts.  `build_s` is the table construction.

| kinds | shape | replay kinds | n | `serving_table` | time | Laplace bits | peak mem |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 33 | toy | 2 | 8,673 | 434 µs | 6.56 ms | 192 | 116 KB |
| 101 | small | 2 | 325,958,913 | 1.81 ms | 24.8 ms | 8,192 | 137 KB |
| 661 | mid | 2 | 1.89e+13 | 13.1 ms | 205 ms | 524,288 | 253 KB |
| 2,578 | frontier-70B | 2 | 2.74e+17 | 58.8 ms | 1.02 s | 16,777,216 | 657 KB |

| kinds | knapsack | knapsack bits | buckets | `cost` | `expected_work` |
|---:|---:|---:|---:|---:|---:|
| 33 | 6.42 ms | 192 | 2,048 | 33 µs | 6.29 µs |
| 101 | 23.2 ms | 8,192 | 2,048 | 69.1 µs | 7.83 µs |
| 661 | 183 ms | 524,288 | 2,048 | 367 µs | 27.6 µs |
| 2,578 | 875 ms | 16,777,216 | 2,048 | 1.68 ms | 94 µs |

Fitted exponents: `serving_table` ∝ x^1.11 (R² 1.00, 4 pts); `cost` ∝ x^0.90 (R² 0.99, 4 pts); knapsack ∝ x^1.12 (R² 1.00, 4 pts); time ∝ x^1.15 (R² 1.00, 4 pts); `expected_work` ∝ x^0.63 (R² 0.96, 4 pts).

#### `serving_step_row`

`serving_table(shape, 'step', 'row')` from toy dimensions to the 70B-class frontier shape (`d_model = 8192`, 80 layers, 2048 requests of 512 + 512 tokens, 2.7e17 gates); the number of kinds grows with the number of distinct decode contexts.  `build_s` is the table construction.

| kinds | shape | replay kinds | n | `serving_table` | time | Laplace bits | peak mem |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 35 | toy | 4 | 8,673 | 480 µs | 9.15 ms | 192 | 119 KB |
| 116 | small | 17 | 325,958,913 | 1.81 ms | 45.6 ms | 8,192 | 160 KB |
| 788 | mid | 129 | 1.89e+13 | 14.4 ms | 391 ms | 524,288 | 601 KB |
| 3,089 | frontier-70B | 513 | 2.74e+17 | 63.2 ms | 1.78 s | 16,777,216 | 2.26 MB |

| kinds | knapsack | knapsack bits | buckets | `cost` | `expected_work` |
|---:|---:|---:|---:|---:|---:|
| 35 | 25.7 ms | 192 | 2,048 | 42.3 µs | 5.46 µs |
| 116 | 172 ms | 8,192 | 2,048 | 112 µs | 8.21 µs |
| 788 | 1.47 s | 524,288 | 2,048 | 728 µs | 29.6 µs |
| 3,089 | 7.23 s | 16,777,216 | 2,048 | 2.91 ms | 95.8 µs |

Fitted exponents: `serving_table` ∝ x^1.09 (R² 1.00, 4 pts); `cost` ∝ x^0.95 (R² 1.00, 4 pts); knapsack ∝ x^1.24 (R² 0.99, 4 pts); time ∝ x^1.17 (R² 1.00, 4 pts); `expected_work` ∝ x^0.65 (R² 0.98, 4 pts).

#### `serving_cell_gate`

`serving_table(shape, 'cell', 'gate')` from toy dimensions to the 70B-class frontier shape (`d_model = 8192`, 80 layers, 2048 requests of 512 + 512 tokens, 2.7e17 gates); the number of kinds grows with the number of distinct decode contexts.  `build_s` is the table construction.

| kinds | shape | replay kinds | n | `serving_table` | time | Laplace bits | peak mem |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 44 | toy | 17 | 8,673 | 536 µs | 19.9 ms | 192 | 34.5 KB |
| 113 | small | 45 | 325,958,913 | 2.14 ms | 55.1 ms | 8,192 | 135 KB |
| 673 | mid | 269 | 1.89e+13 | 14.6 ms | 357 ms | 185,847 | 880 KB |
| 2,590 | frontier-70B | 1,036 | 2.74e+17 | 67.5 ms | 1.4 s | 236,711 | 3.39 MB |

| kinds | knapsack | knapsack bits | buckets | `cost` | `expected_work` |
|---:|---:|---:|---:|---:|---:|
| 44 | 181 ms | 192 | 2,048 | 175 µs | 5.67 µs |
| 113 | 1.57 s | 8,192 | 2,048 | 366 µs | 5.92 µs |
| 673 | 11.6 s | 185,847 | 2,048 | 2.57 ms | 12.3 µs |
| 2,590 | — | — | — | 10.1 ms | 31.7 µs |

Fitted exponents: `serving_table` ∝ x^1.17 (R² 1.00, 4 pts); `cost` ∝ x^1.01 (R² 1.00, 4 pts); knapsack ∝ x^1.47 (R² 0.96, 3 pts); time ∝ x^1.04 (R² 1.00, 4 pts); `expected_work` ∝ x^0.43 (R² 0.93, 4 pts).

#### `synthetic_vs_replay_kinds`

A synthetic table of `K` distinct RU kinds (1000 copies each, 64 VUs of one kind per copy): the Laplace fold is one series per kind and ~130 evaluations of a `K`-term sum (linear in `K`); the knapsack forms a cost polynomial per kind and convolves `K` of them on the grid.

| replay kinds | time | peak mem | rows | replay copies | ru gates | ru positions | n | Laplace bits |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 5.94 ms | 28.9 KB | 6 | 4,000 | 256 | 63 | 1,024,000 | 64,000 |
| 16 | 21.1 ms | 69.7 KB | 18 | 16,000 | 256 | 63 | 4,096,000 | 89,408 |
| 64 | 81.6 ms | 230 KB | 66 | 64,000 | 256 | 63 | 16,384,000 | 99,695 |
| 256 | 324 ms | 874 KB | 258 | 256,000 | 256 | 63 | 65,536,000 | 109,316 |
| 1,024 | 1.31 s | 3.36 MB | 1,026 | 1,024,000 | 256 | 63 | 262,144,000 | 118,790 |

| replay kinds | knapsack | knapsack bits | buckets | `cost` | `expected_work` |
|---:|---:|---:|---:|---:|---:|
| 4 | 196 ms | 64,000 | 2,048 | 22.7 µs | 4.29 µs |
| 16 | 839 ms | 89,408 | 2,048 | 48.8 µs | 4.08 µs |
| 64 | 3.38 s | 99,695 | 2,048 | 159 µs | 4.42 µs |
| 256 | 13.6 s | 109,316 | 2,048 | 643 µs | 6.21 µs |
| 1,024 | — | — | — | 2.4 ms | 12 µs |

Fitted exponents: `cost` ∝ x^0.86 (R² 0.99, 5 pts); knapsack ∝ x^1.02 (R² 1.00, 4 pts); time ∝ x^0.97 (R² 1.00, 5 pts); `expected_work` ∝ x^0.18 (R² 0.75, 5 pts).

#### `knapsack_vs_buckets`

`Bound` with the knapsack on, on the `small` `step/row` table (116 kinds, 17 replay kinds), against the cost grid: `sparse_power` and the convolutions are `O(buckets * terms)` to `O(buckets**2)`.  `time_s` is the whole `Bound`.

| max_buckets | time | peak mem | bits | knapsack bits |
|---:|---:|---:|---:|---:|
| 128 | 39.6 ms | 577 KB | 8,192 | 69,664 |
| 512 | 49.2 ms | 4.11 MB | 8,192 | 69,664 |
| 2,048 | 172 ms | 13.4 MB | 8,192 | 69,879 |
| 8,192 | 1.53 s | 52.1 MB | 8,192 | 69,911 |
| 32,768 | 23.3 s | — | 8,192 | 69,951 |

Fitted exponents: peak mem ∝ x^1.06 (R² 0.99, 4 pts); time ∝ x^1.17 (R² 0.91, 5 pts).

#### `optimize_vs_grid`

`Optimize` (`max_bits` objective, Laplace-only bound) on the `small` `step/row` table over `PolicyGrid.uniform(steps)`, `(steps + 1)**2` points: one `Cost` and one `Bound` per point.

| grid points | time | steps |
|---:|---:|---:|
| 4 | 253 ms | 1 |
| 9 | 442 ms | 2 |
| 25 | 1.06 s | 4 |
| 81 | 3.3 s | 8 |
| 289 | 12.1 s | 16 |

Fitted exponents: time ∝ x^0.91 (R² 1.00, 5 pts).

## 5. Challenge sampling: binomial count and Floyd subset

Wall time of `bernoulli_subset(seed, stage, phase, N, p)` -- `K ~ Binomial(N, p)` by 512-bit CDF inversion, then a uniform `K`-subset from `K` HMAC-SHA256 draws with rejection -- and of the two derivations the verifier runs on a compiled circuit.

`challenge` ran in 1.14 s.

#### `vs_N_fixed_K`

`p = 64 / N`: sixty-four selections expected whatever `N`; time should be flat in `N` up to the `O(log N)` initial power (the exponent should be near 0, far below 1).

| N (candidates) | time | selected | expected | per selected | denominator bits |
|---:|---:|---:|---:|---:|---:|
| 1,000 | 182 µs | 59 | 64 | 3.09 µs | 7 |
| 10,000 | 247 µs | 57 | 64 | 4.33 µs | 10 |
| 100,000 | 225 µs | 65 | 64 | 3.46 µs | 12 |
| 1,000,000 | 208 µs | 73 | 64 | 2.85 µs | 14 |
| 10,000,000 | 255 µs | 59 | 64 | 4.33 µs | 18 |
| 100,000,000 | 200 µs | 56 | 64 | 3.57 µs | 21 |
| 1e+09 | 176 µs | 59 | 64 | 2.98 µs | 24 |
| 1e+11 | 230 µs | 60 | 64 | 3.84 µs | 31 |
| 1e+13 | 336 µs | 65 | 64 | 5.16 µs | 38 |
| 1e+15 | 325 µs | 68 | 64 | 4.78 µs | 44 |

Fitted exponents: per selected ∝ x^0.01 (R² 0.33, 10 pts); time ∝ x^0.02 (R² 0.43, 10 pts).

#### `vs_K_fixed_N`

`N = 10**6` candidates, `p = K / N`: one HMAC draw (plus rejections) per selection, so time is linear in `K` (exponent near 1) once `K` dominates the fixed cost of the binomial inversion.

| K (expected selections) | time | selected | per selected | denominator bits |
|---:|---:|---:|---:|---:|
| 1 | 21.3 µs | 1 | 21.3 µs | 20 |
| 10 | 43.7 µs | 6 | 7.28 µs | 17 |
| 100 | 292 µs | 106 | 2.76 µs | 14 |
| 1,000 | 2.86 ms | 1,035 | 2.76 µs | 10 |
| 10,000 | 31.2 ms | 10,025 | 3.11 µs | 7 |
| 100,000 | 320 ms | 100,439 | 3.18 µs | 4 |

Fitted exponents: time ∝ x^0.87 (R² 0.98, 6 pts).

#### `derive_selections_vs_units`

`derive_replay_selection` (`q = 64 / #RU`) then `derive_sample_selection` (`s = 1/2`) on a `repeat(U, block)` of 8-cell RUs; the verifier's `max_units` limit (10**6) is raised for the sweep.  `time_s` is the replay selection, `sample_s` the VU selection over the 8 * |J| candidates.

| replay units | time | `derive_sample_selection` | #J | #T | VUs |
|---:|---:|---:|---:|---:|---:|
| 1,001 | 176 µs | 989 µs | 56 | 227 | 8,001 |
| 10,001 | 297 µs | 978 µs | 59 | 212 | 80,001 |
| 100,001 | 229 µs | 1.18 ms | 67 | 277 | 800,001 |
| 1,000,001 | 247 µs | 1.39 ms | 80 | 299 | 8,000,001 |
| 10,000,001 | 328 µs | 1.43 ms | 76 | 281 | 80,000,001 |
| 100,000,001 | 310 µs | 1.2 ms | 67 | 277 | 800,000,001 |

Fitted exponents: `derive_sample_selection` ∝ x^0.03 (R² 0.56, 6 pts); time ∝ x^0.04 (R² 0.53, 6 pts).

## 6. Merkle commitments: build, open, verify

`MerkleTree(domain, values, schema)` over `L` 16-bit leaves on a range domain: build wall time and `tracemalloc` peak, then the per-call latency of `tree.open(position)` and `verify_opening`, with the opening's size in bytes.  `commit_weights_s` builds `kappa_W` for `L` weights through `commit_weights`.

`merkle` ran in 81.2 s.

#### `build_vs_leaves`

Linear in `L`: `2L - 1` domain-bound SHA-256 calls.  `values_per_s` is the build throughput, `bytes_per_leaf` the retained tree (`_values` and every level) per leaf.

| leaves | time | peak mem | values/s | hashes/s | retained/leaf | `commit_weights` | depth |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 222 µs | 22.1 KB | 451k/s | 1.15M/s | 226 B | 245 µs | 7 |
| 1,000 | 1.8 ms | 172 KB | 556k/s | 1.14M/s | 177 B | 2.06 ms | 10 |
| 10,000 | 26.8 ms | 2.57 MB | 372k/s | 1.22M/s | 270 B | 28.4 ms | 14 |
| 100,000 | 224 ms | 20.9 MB | 447k/s | 1.17M/s | 219 B | 239 ms | 17 |
| 1,000,000 | 1.89 s | 170 MB | 529k/s | 1.11M/s | 178 B | 2.05 s | 20 |
| 2,000,000 | 3.81 s | 340 MB | 525k/s | 1.1M/s | 178 B | 4.15 s | 21 |

Fitted exponents: `commit_weights` ∝ x^0.99 (R² 1.00, 6 pts); peak mem ∝ x^0.98 (R² 1.00, 6 pts); time ∝ x^0.99 (R² 1.00, 6 pts).

#### `open_verify_vs_leaves`

`time_s` is `tree.open(position)` (a rank lookup and `depth` sibling reads), `verify_s` is `verify_opening` (`depth + 1` hashes): both `O(log L)`.  `proof_bytes` is `2 + 32 * depth`.

| leaves | time | `verify_opening` | proof | depth |
|---:|---:|---:|---:|---:|
| 100 | 794 ns | 7.56 µs | 226 B | 7 |
| 1,000 | 988 ns | 10.2 µs | 322 B | 10 |
| 10,000 | 1.29 µs | 13.9 µs | 450 B | 14 |
| 100,000 | 1.44 µs | 17.2 µs | 546 B | 17 |
| 1,000,000 | 1.69 µs | 20.5 µs | 642 B | 20 |
| 2,000,000 | 2.03 µs | 21.4 µs | 674 B | 21 |

Fitted exponents: proof ∝ x^0.11 (R² 0.97, 6 pts); time ∝ x^0.09 (R² 0.98, 6 pts); `verify_opening` ∝ x^0.10 (R² 0.98, 6 pts).

## 7. The protocol end to end: prover and verifier phases, message bytes

`time_s` is the prover's total (setup, boundary commitment, replay + interior commitments, openings); `verifier_total_s` the verifier's (admission, boundary check and `J`, interior check and `T`, evidence check).  `evaluate_s` is the honest computation itself through the lazy circuit (`circuit.evaluate`, `gates_per_s`), `kappa_w_s` the one-off weight commitment.  Sizes: `boundary_count = |∂|`, `replayed_gates` recomputed (every non-source gate of the selected RUs), `interior_positions` committed (the VU outputs among them that are not RU outputs), `openings` sent, message bytes as canonical JSON.

`protocol` ran in 5.63 min.

#### `cluster_vs_n`

`ClusterG` at growing model width, `q = 1/2, s = 1/4`.  Steps are the RUs: the boundary is the KV cache and the tokens, so `|∂|` grows with `n`.

| n (gates) | case | RUs | VUs | evaluate | gates/s | `kappa_W` | time | V total | transcript |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4,437 | d4-L1 | 4 | 771 | 34.2 ms | 130k/s | 536 µs | 126 ms | 68.4 ms | 786 KB |
| 27,169 | d8-L1 | 7 | 2,749 | 222 ms | 122k/s | 1.86 ms | 670 ms | 328 ms | 4.48 MB |
| 509,833 | d16-L2 | 9 | 25,569 | 4.27 s | 119k/s | 17.7 ms | 6.88 s | 3.6 s | 58.2 MB |
| 980,233 | d32-L2-r4 | 5 | 37,813 | 8.12 s | 121k/s | 55 ms | 24.2 s | 12.4 s | 235 MB |

| n (gates) | P setup | P commit ∂ | P replay | P commit interiors | P openings | V admit | V ∂ + `J` | V interiors + `T` | V Merkle | V recompute |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4,437 | 168 µs | 1.19 ms | 67.8 ms | 10.1 ms | 47.1 ms | 2.1 ms | 348 µs | 976 µs | 16 ms | 49.1 ms |
| 27,169 | 288 µs | 5.21 ms | 395 ms | 31.9 ms | 236 ms | 2.57 ms | 769 µs | 2.93 ms | 88.7 ms | 233 ms |
| 509,833 | 738 µs | 51 ms | 4.28 s | 216 ms | 2.32 s | 6.14 ms | 2.01 ms | 14.5 ms | 1.11 s | 2.47 s |
| 980,233 | 461 µs | 45.6 ms | 16.1 s | 399 ms | 7.72 s | 5.26 ms | 1.21 ms | 39.5 ms | 4.33 s | 8 s |

| n (gates) | #∂ | #J | replayed gates | interior positions | #T | openings | ∂ msg | interiors msg | evidence msg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4,437 | 92 | 4 | 4,230 | 568 | 196 | 1,378 | 6.1 KB | 367 B | 778 KB |
| 27,169 | 344 | 5 | 23,658 | 1,808 | 466 | 6,860 | 15.2 KB | 457 B | 4.46 MB |
| 509,833 | 3,648 | 5 | 252,720 | 9,760 | 3,782 | 69,080 | 53.1 KB | 459 B | 58.2 MB |
| 980,233 | 3,616 | 5 | 963,312 | 19,168 | 9,530 | 250,773 | 26.6 KB | 460 B | 235 MB |

| n (gates) | replay / gate | commit / position | open / opening | V Merkle / opening | V recompute / opening | bytes / opening |
|---:|---:|---:|---:|---:|---:|---:|
| 4,437 | 16 µs | 17.8 µs | 34.2 µs | 11.6 µs | 35.6 µs | 578 B |
| 27,169 | 16.7 µs | 17.7 µs | 34.4 µs | 12.9 µs | 33.9 µs | 682 B |
| 509,833 | 17 µs | 22.1 µs | 33.6 µs | 16.1 µs | 35.7 µs | 883 B |
| 980,233 | 16.7 µs | 20.8 µs | 30.8 µs | 17.3 µs | 31.9 µs | 982 B |

Fitted exponents: evaluate ∝ x^1.01 (R² 1.00, 4 pts); P commit ∂ ∝ x^0.71 (R² 0.98, 4 pts); P commit interiors ∝ x^0.67 (R² 1.00, 4 pts); P openings ∝ x^0.90 (R² 0.99, 4 pts); P replay ∝ x^0.96 (R² 0.99, 4 pts); time ∝ x^0.92 (R² 0.99, 4 pts); transcript ∝ x^1.01 (R² 0.99, 4 pts); V ∂ + `J` ∝ x^0.27 (R² 0.83, 4 pts); V interiors + `T` ∝ x^0.64 (R² 0.98, 4 pts); V Merkle ∝ x^0.99 (R² 0.99, 4 pts); V recompute ∝ x^0.90 (R² 0.99, 4 pts); V total ∝ x^0.92 (R² 0.99, 4 pts).

#### `requests_vs_n`

`RequestsG` at growing model width, `q = 1/2, s = 1/4`.  Requests are the RUs: the boundary is prompts and tokens only, the interiors hold everything else.

| n (gates) | case | RUs | VUs | evaluate | gates/s | `kappa_W` | time | V total | transcript |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4,437 | d4-L1 | 3 | 771 | 34.4 ms | 129k/s | 545 µs | 2.27 ms | 4.31 ms | 27.9 KB |
| 27,169 | d8-L1 | 5 | 2,749 | 226 ms | 120k/s | 2.07 ms | 560 ms | 298 ms | 4.12 MB |
| 509,833 | d16-L2 | 9 | 25,569 | 4.2 s | 122k/s | 13.9 ms | 6.39 s | 3.19 s | 56 MB |
| 980,233 | d32-L2-r4 | 5 | 37,813 | 8.16 s | 120k/s | 56 ms | 11.8 s | 5.99 s | 114 MB |

| n (gates) | P setup | P commit ∂ | P replay | P commit interiors | P openings | V admit | V ∂ + `J` | V interiors + `T` | V Merkle | V recompute |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4,437 | 154 µs | 261 µs | 463 µs | 462 µs | 933 µs | 1.78 ms | 268 µs | 206 µs | 687 µs | 1.29 ms |
| 27,169 | 293 µs | 510 µs | 331 ms | 31 ms | 198 ms | 2.05 ms | 523 µs | 2.38 ms | 82.2 ms | 210 ms |
| 509,833 | 739 µs | 1.23 ms | 4.19 s | 194 ms | 2.01 s | 4.6 ms | 1.3 ms | 14 ms | 1.06 s | 2.11 s |
| 980,233 | 388 µs | 759 µs | 8.09 s | 206 ms | 3.54 s | 4.02 ms | 721 µs | 13.4 ms | 2.1 s | 3.87 s |

| n (gates) | #∂ | #J | replayed gates | interior positions | #T | openings | ∂ msg | interiors msg | evidence msg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4,437 | 12 | 1 | 0 | 0 | 41 | 41 | 3.74 KB | 103 B | 23.2 KB |
| 27,169 | 24 | 4 | 19,881 | 1,752 | 581 | 6,056 | 8.96 KB | 367 B | 4.11 MB |
| 509,833 | 64 | 5 | 252,720 | 11,552 | 3,724 | 66,815 | 28 KB | 459 B | 55.9 MB |
| 980,233 | 32 | 2 | 481,656 | 11,376 | 2,601 | 122,501 | 12 KB | 195 B | 114 MB |

| n (gates) | replay / gate | commit / position | open / opening | V Merkle / opening | V recompute / opening | bytes / opening |
|---:|---:|---:|---:|---:|---:|---:|
| 4,437 | — | — | 22.7 µs | 16.8 µs | 31.5 µs | 579 B |
| 27,169 | 16.7 µs | 17.7 µs | 32.6 µs | 13.6 µs | 34.6 µs | 711 B |
| 509,833 | 16.6 µs | 16.8 µs | 30 µs | 15.8 µs | 31.6 µs | 878 B |
| 980,233 | 16.8 µs | 18.1 µs | 28.9 µs | 17.1 µs | 31.6 µs | 978 B |

Fitted exponents: evaluate ∝ x^1.01 (R² 1.00, 4 pts); P commit ∂ ∝ x^0.23 (R² 0.82, 4 pts); P commit interiors ∝ x^1.06 (R² 0.88, 4 pts); P openings ∝ x^1.41 (R² 0.90, 4 pts); P replay ∝ x^1.66 (R² 0.88, 4 pts); time ∝ x^1.46 (R² 0.90, 4 pts); transcript ∝ x^1.44 (R² 0.93, 4 pts); V ∂ + `J` ∝ x^0.23 (R² 0.76, 4 pts); V interiors + `T` ∝ x^0.76 (R² 0.94, 4 pts); V Merkle ∝ x^1.38 (R² 0.93, 4 pts); V recompute ∝ x^1.37 (R² 0.91, 4 pts); V total ∝ x^1.25 (R² 0.93, 4 pts).

#### `cluster_vs_q`

`ClusterG` `d8-r32` (49 RUs), `s = 1/4`, mean over 2 session seed(s): the replay phase and the interior commitments scale with the selected RUs, `q * #RU`, so the realized counts (`selected_replay_units`, `interior_positions`) are the honest x-axis.

| q | case | n | RUs | VUs | evaluate | gates/s | `kappa_W` | time | V total | transcript |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1/32 | d8-r32 | 212,809 | 49 | 17,449 | 1.78 s | 120k/s | 2 ms | 395 ms | 184 ms | 2.51 MB |
| 1/8 | d8-r32 | 212,809 | 49 | 17,449 | 1.78 s | 120k/s | 2 ms | 762 ms | 369 ms | 5.05 MB |
| 1/2 | d8-r32 | 212,809 | 49 | 17,449 | 1.78 s | 120k/s | 2 ms | 3.04 s | 1.46 s | 20.3 MB |
| 1 | d8-r32 | 212,809 | 49 | 17,449 | 1.78 s | 120k/s | 2 ms | 5.82 s | 2.82 s | 39.1 MB |

| q | P setup | P commit ∂ | P replay | P commit interiors | P openings | V admit | V ∂ + `J` | V interiors + `T` | V Merkle | V recompute |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1/32 | 2.04 ms | 36.6 ms | 220 ms | 17.6 ms | 118 ms | 6.95 ms | 5.22 ms | 1.03 ms | 47.4 ms | 124 ms |
| 1/8 | 2.11 ms | 36.5 ms | 445 ms | 35.9 ms | 243 ms | 7.04 ms | 5.25 ms | 2.3 ms | 98.3 ms | 256 ms |
| 1/2 | 2.13 ms | 36.7 ms | 1.85 s | 148 ms | 1.01 s | 6.78 ms | 5.6 ms | 8.84 ms | 398 ms | 1.04 s |
| 1 | 2.08 ms | 36.3 ms | 3.54 s | 284 ms | 1.95 s | 6.84 ms | 5.4 ms | 17.3 ms | 769 ms | 2.02 s |

| q | #∂ | #J | replayed gates | interior positions | #T | openings | ∂ msg | interiors msg | evidence msg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1/32 | 2,752 | 3 | 13,222 | 1,008 | 264 | 3,596 | 159 KB | 281 B | 2.35 MB |
| 1/8 | 2,752 | 6 | 26,508 | 2,016 | 536 | 7,444 | 159 KB | 545 B | 4.88 MB |
| 1/2 | 2,752 | 26 | 110,307 | 8,376 | 2,237 | 30,744 | 159 KB | 2.25 KB | 20.2 MB |
| 1 | 2,752 | 49 | 212,064 | 16,128 | 4,365 | 59,220 | 159 KB | 4.23 KB | 38.9 MB |

| q | replay / gate | commit / position | open / opening | V Merkle / opening | V recompute / opening | bytes / opening |
|---:|---:|---:|---:|---:|---:|---:|
| 1/32 | 16.6 µs | 17.5 µs | 32.9 µs | 13.2 µs | 34.4 µs | 686 B |
| 1/8 | 16.8 µs | 17.8 µs | 32.7 µs | 13.2 µs | 34.4 µs | 688 B |
| 1/2 | 16.7 µs | 17.7 µs | 32.8 µs | 13 µs | 33.8 µs | 687 B |
| 1 | 16.7 µs | 17.6 µs | 32.9 µs | 13 µs | 34.1 µs | 689 B |

Fitted exponents: evaluate ∝ x^0.00 (R² 1.00, 4 pts); P commit ∂ ∝ x^-0.00 (R² 0.12, 4 pts); P commit interiors ∝ x^0.82 (R² 0.98, 4 pts); P openings ∝ x^0.83 (R² 0.98, 4 pts); P replay ∝ x^0.82 (R² 0.98, 4 pts); time ∝ x^0.79 (R² 0.97, 4 pts); transcript ∝ x^0.81 (R² 0.98, 4 pts); V ∂ + `J` ∝ x^0.02 (R² 0.56, 4 pts); V interiors + `T` ∝ x^0.82 (R² 0.99, 4 pts); V Merkle ∝ x^0.82 (R² 0.98, 4 pts); V recompute ∝ x^0.82 (R² 0.98, 4 pts); V total ∝ x^0.80 (R² 0.98, 4 pts).

#### `cluster_vs_s`

`ClusterG` `d8-r32`, `q = 1/2`, mean over 2 session seed(s): the openings and the verifier's evidence check scale with the sampled VUs, `q s * #VU`; the replay phase depends only on `J`, which is redrawn per policy (the policy is in the header digest), hence its scatter.

| s | case | n | RUs | VUs | evaluate | gates/s | `kappa_W` | time | V total | transcript |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1/64 | d8-r32 | 212,809 | 49 | 17,449 | 1.78 s | 120k/s | 2 ms | 2.47 s | 128 ms | 1.74 MB |
| 1/16 | d8-r32 | 212,809 | 49 | 17,449 | 1.78 s | 120k/s | 2 ms | 2 s | 330 ms | 4.46 MB |
| 1/4 | d8-r32 | 212,809 | 49 | 17,449 | 1.78 s | 120k/s | 2 ms | 3.03 s | 1.46 s | 20.3 MB |
| 1 | d8-r32 | 212,809 | 49 | 17,449 | 1.78 s | 120k/s | 2 ms | 5.4 s | 5.18 s | 72.7 MB |

| s | P setup | P commit ∂ | P replay | P commit interiors | P openings | V admit | V ∂ + `J` | V interiors + `T` | V Merkle | V recompute |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1/64 | 2.02 ms | 36.3 ms | 2.17 s | 174 ms | 80.8 ms | 6.94 ms | 5.34 ms | 1.13 ms | 31.2 ms | 83.3 ms |
| 1/16 | 2.09 ms | 36.1 ms | 1.61 s | 130 ms | 219 ms | 6.91 ms | 5.43 ms | 2.13 ms | 86.1 ms | 229 ms |
| 1/4 | 2.13 ms | 36.5 ms | 1.84 s | 146 ms | 1 s | 6.98 ms | 5.33 ms | 8.82 ms | 400 ms | 1.04 s |
| 1 | 2.05 ms | 36.4 ms | 1.64 s | 132 ms | 3.6 s | 6.78 ms | 5.45 ms | 2.13 ms | 1.44 s | 3.73 s |

| s | #∂ | #J | replayed gates | interior positions | #T | openings | ∂ msg | interiors msg | evidence msg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1/64 | 2,752 | 29 | 130,553 | 9,960 | 170 | 2,380 | 159 KB | 2.51 KB | 1.58 MB |
| 1/16 | 2,752 | 24 | 96,190 | 7,280 | 494 | 6,586 | 159 KB | 2.08 KB | 4.3 MB |
| 1/4 | 2,752 | 26 | 110,307 | 8,376 | 2,237 | 30,744 | 159 KB | 2.25 KB | 20.2 MB |
| 1 | 2,752 | 20.5 | 98,281 | 7,528 | 7,845 | 109,525 | 159 KB | 1.78 KB | 72.5 MB |

| s | replay / gate | commit / position | open / opening | V Merkle / opening | V recompute / opening | bytes / opening |
|---:|---:|---:|---:|---:|---:|---:|
| 1/64 | 16.6 µs | 17.4 µs | 33.9 µs | 13.1 µs | 35 µs | 694 B |
| 1/16 | 16.7 µs | 17.8 µs | 33.2 µs | 13.1 µs | 34.8 µs | 685 B |
| 1/4 | 16.7 µs | 17.4 µs | 32.6 µs | 13 µs | 34 µs | 687 B |
| 1 | 16.6 µs | 17.5 µs | 32.8 µs | 13.1 µs | 34 µs | 695 B |

Fitted exponents: evaluate ∝ x^0.00 (R² 1.00, 4 pts); P commit ∂ ∝ x^0.00 (R² 0.26, 4 pts); P commit interiors ∝ x^-0.05 (R² 0.46, 4 pts); P openings ∝ x^0.93 (R² 0.99, 4 pts); P replay ∝ x^-0.05 (R² 0.45, 4 pts); time ∝ x^0.20 (R² 0.70, 4 pts); transcript ∝ x^0.92 (R² 0.99, 4 pts); V ∂ + `J` ∝ x^0.00 (R² 0.22, 4 pts); V interiors + `T` ∝ x^0.24 (R² 0.24, 4 pts); V Merkle ∝ x^0.94 (R² 0.99, 4 pts); V recompute ∝ x^0.93 (R² 0.99, 4 pts); V total ∝ x^0.91 (R² 0.99, 4 pts).

## 8. `output_reach` vs description size

`time_s` is `output_reach(root)` on the parsed root; `transient_ports_s` the sibling pass, `kinds_s` the whole `Index.kinds()` (both passes plus the per-kind summaries), `parse_s` `parse_description`.  `root_steps` is the step count of the root definition, `definitions` the reachable kinds.

`reach` ran in 5.48 s.

#### `chain_vs_steps`

A root with `S` sequential `call` steps of one 3-gate RU, each reading the previous step's output (a decode chain).  The closure `Down(j)` is every later step, one interval `[j, S)` in the sweep of `_step_reach`: `O(S log S)` (two range additions and one extraction on the segment tree per step), against `Θ(S)` for the parse.  The bitmask closure this replaced was `Θ(S³ / w)` here (14.3 s at `S = 8192`).

| root steps | time | peak mem | description | definitions | root steps | n | `transient_ports` | `kinds()` | `parse_description` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 282 µs | 27.2 KB | 8.78 KB | 5 | 65 | 193 | 91.8 µs | 440 µs | 775 µs |
| 256 | 1.47 ms | 96.5 KB | 31.1 KB | 5 | 257 | 769 | 313 µs | 1.8 ms | 2.87 ms |
| 1,024 | 6.87 ms | 393 KB | 120 KB | 5 | 1,025 | 3,073 | 1.4 ms | 8.53 ms | 10.5 ms |
| 4,096 | 32.3 ms | 1.54 MB | 480 KB | 5 | 4,097 | 12,289 | 5.67 ms | 40.2 ms | 41.1 ms |
| 8,192 | 68.8 ms | 3.09 MB | 960 KB | 5 | 8,193 | 24,577 | 11.2 ms | 83.6 ms | 87 ms |

Fitted exponents: `kinds()` ∝ x^1.09 (R² 1.00, 5 pts); `parse_description` ∝ x^0.97 (R² 1.00, 5 pts); peak mem ∝ x^0.99 (R² 1.00, 5 pts); time ∝ x^1.13 (R² 1.00, 5 pts); `transient_ports` ∝ x^1.00 (R² 1.00, 5 pts).

#### `independent_vs_steps`

A root with `S` independent `call` steps of one RU, every step reading the input step (siblings over one broadcast step, as requests over a weights step).  `Down(j) = {j}` and `Down(input) = [0, S + 1)`, so the pass is `O(S log S)`; the bitmask closure ORed `S` masks of `S` bits here, `Θ(S² / w)`.

| root steps | time | peak mem | description | definitions | root steps | n | `transient_ports` | `kinds()` | `parse_description` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 81.2 µs | 26.6 KB | 8.73 KB | 5 | 65 | 193 | 44.8 µs | 227 µs | 825 µs |
| 256 | 364 µs | 102 KB | 30.7 KB | 5 | 257 | 769 | 163 µs | 652 µs | 2.83 ms |
| 1,024 | 1.42 ms | 406 KB | 118 KB | 5 | 1,025 | 3,073 | 720 µs | 2.68 ms | 10.2 ms |
| 4,096 | 5.58 ms | 1.6 MB | 469 KB | 5 | 4,097 | 12,289 | 2.95 ms | 10.4 ms | 40.8 ms |
| 8,192 | 11.4 ms | 3.21 MB | 937 KB | 5 | 8,193 | 24,577 | 5.74 ms | 20.5 ms | 87.1 ms |

Fitted exponents: `kinds()` ∝ x^0.94 (R² 1.00, 5 pts); `parse_description` ∝ x^0.96 (R² 1.00, 5 pts); peak mem ∝ x^0.99 (R² 1.00, 5 pts); time ∝ x^1.01 (R² 1.00, 5 pts); `transient_ports` ∝ x^1.01 (R² 1.00, 5 pts).

#### `definitions_vs_count`

A root calling `D` distinct RU definitions once each, all reading the input: linear in `D` (one `_step_reach` per definition with call steps, here only the root).

| definitions | time | peak mem | description | definitions | root steps | n | `transient_ports` | `kinds()` | `parse_description` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 178 µs | 29.1 KB | 26.6 KB | 68 | 65 | 6,241 | 116 µs | 686 µs | 3.12 ms |
| 256 | 773 µs | 114 KB | 103 KB | 260 | 257 | 98,689 | 381 µs | 2.17 ms | 11.6 ms |
| 1,024 | 3.61 ms | 414 KB | 410 KB | 1,028 | 1,025 | 1,574,401 | 1.85 ms | 9.47 ms | 45.8 ms |
| 4,096 | 14.1 ms | 1.63 MB | 1.6 MB | 4,100 | 4,097 | 25,171,969 | 7.79 ms | 46.1 ms | 208 ms |

Fitted exponents: `kinds()` ∝ x^1.02 (R² 1.00, 4 pts); `parse_description` ∝ x^1.01 (R² 1.00, 4 pts); peak mem ∝ x^0.97 (R² 1.00, 4 pts); time ∝ x^1.06 (R² 1.00, 4 pts); `transient_ports` ∝ x^1.02 (R² 1.00, 4 pts).

#### `deep_repeat_vs_depth`

`repeat` nesting `depth` levels deep with branching 4 (`n = 4^depth * 8`): one definition per level, one step each, so the pass is linear in `depth` whatever `n` is.

| depth | time | peak mem | description | definitions | root steps | n | `transient_ports` | `kinds()` | `parse_description` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 21.6 µs | 4.18 KB | 2.38 KB | 8 | 2 | 769 | 10.3 µs | 47.4 µs | 433 µs |
| 8 | 22.2 µs | 4.46 KB | 3.48 KB | 12 | 2 | 196,609 | 10 µs | 85.9 µs | 422 µs |
| 12 | 31.1 µs | 4.49 KB | 4.59 KB | 16 | 2 | 50,331,649 | 12.4 µs | 87.8 µs | 581 µs |
| 16 | 34.9 µs | 6.09 KB | 5.7 KB | 20 | 2 | 1.29e+10 | 21 µs | 128 µs | 642 µs |
| 20 | 46.5 µs | 6.48 KB | 6.81 KB | 24 | 2 | 3.3e+12 | 17.3 µs | 121 µs | 771 µs |
| 24 | 50.5 µs | 6.57 KB | 7.92 KB | 28 | 2 | 8.44e+14 | 20.8 µs | 179 µs | 1.16 ms |

Fitted exponents: `kinds()` ∝ x^0.66 (R² 0.93, 6 pts); `parse_description` ∝ x^0.50 (R² 0.77, 6 pts); peak mem ∝ x^0.28 (R² 0.80, 6 pts); time ∝ x^0.51 (R² 0.88, 6 pts); `transient_ports` ∝ x^0.45 (R² 0.75, 6 pts).

#### `cluster_vs_description`

The `ClusterG` ladder: the description grows with the layer count and the schedule (steps of the decode loop), not with the model width.

| description bytes | time | peak mem | definitions | root steps | n | `transient_ports` | `kinds()` | `parse_description` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 23.4 KB | 318 µs | 11.7 KB | 35 | 4 | 4,437 | 314 µs | 840 µs | 2.89 ms |
| 24.4 KB | 348 µs | 11.9 KB | 35 | 7 | 27,169 | 392 µs | 824 µs | 2.94 ms |
| 45.5 KB | 892 µs | 18.5 KB | 45 | 9 | 509,833 | 970 µs | 2.1 ms | 6.27 ms |
| 46.8 KB | 860 µs | 18.5 KB | 45 | 9 | 1,943,561 | 1.05 ms | 2.3 ms | 6.43 ms |
| 47.4 KB | 1.01 ms | 18.5 KB | 45 | 9 | 7,588,105 | 1.16 ms | 2.46 ms | 6.33 ms |

| description bytes | case |
|---:|---:|
| 23.4 KB | d4-L1 |
| 24.4 KB | d8-L1 |
| 45.5 KB | d16-L2 |
| 46.8 KB | d32-L2 |
| 47.4 KB | d64-L2 |

Fitted exponents: `kinds()` ∝ x^1.51 (R² 0.99, 5 pts); `parse_description` ∝ x^1.16 (R² 1.00, 5 pts); peak mem ∝ x^0.67 (R² 1.00, 5 pts); time ∝ x^1.52 (R² 0.99, 5 pts); `transient_ports` ∝ x^1.66 (R² 0.99, 5 pts).

## Bottlenecks

What limits each component asymptotically, the constant this CPython prototype measures for it, and what the constant means for a deployment.  Every number below is read from the tables above; the deployment figures use the stated conversion rates and nothing else.

### Compile

- **Description-bound, never n-bound.**  `Compile` of a `repeat` tower with `n = 3e+12` gates takes 710 µs and 70.7 KB (`4,716`-byte description, 16 definitions); time ∝ n^0.04 across the sweep.  The compiler never materializes a gate: parse, validate, and build the per-definition index frames.
- **Linear in the description.**  Over the `definitions` sweep time ∝ definitions^1.04; at 8,196 definitions (3.21 MB) compiling takes 473 ms, i.e. 7.11M/s of description, 57.8 µs per definition.
- **The cluster ladder confirms it.**  From `d_model = 4` to `128` the circuit grows ∝ d_model^1.90 (to n = 15,124,873) while compile time goes ∝ d_model^0.07: 5.86 ms at the top, of which the tracer's own `trace_s` is 1.65 ms.  Compile is not a deployment bottleneck at any model size; the description size (schedule length × layers) is what matters.

### Lazy gate lookup

- **O(depth), flat in n.**  Random `circuit[i]` costs 6.36 µs at n = 3,001 and 7.51 µs at n = 3e+15 (ratio 1.2×, exponent 0.01); sequential access 7.71 µs, strided 7.44 µs, `owner` 3.63 µs, `unit` 3.73 µs at the top.
- **Per level: 1.38 µs.**  Along the depth sweep time ∝ depth^0.80; at depth 48 a lookup is 69.2 µs.  Each level is a `_LazyAddresses` descent (an integer division to pick the copy plus a frame allocation); the Python constant is a few microseconds per level.  A gate-by-gate replay through `circuit[i]` therefore runs at ~14.5k/s gates at this depth, which is why the sessions use `replay_unit` (one frame walk per RU) rather than indexing gates one by one.

### Kind table and address sets

- **`kind_table()` is linear in the definitions**: ∝ kinds^1.05, 92 ms for 8,196 kinds (11.2 µs per kind; `kinds()` alone 88.6 ms), with n = 100,675,585 gates behind it.  It never touches n.
- **The output runs of one RU are the only per-call factor.**  With 128 runs in one RU's `Out`, the boundary barely moves (`∂.rank` 7.92 µs, ∝ runs^0.16; `∂.unrank` 1.21 µs, ∝ runs^0.01) while the interior grows with them: `interior(u)` 3.62 µs (∝ runs^-0.00), its `unrank` 144 µs (∝ runs^0.68), its `contains` 16.3 µs.  The interior (`_Interior`: the VU outputs of the RU minus its own `Out`) descends to the VU in `O(depth)` and then subtracts the RU outputs below the address run by run, and `unrank` repeats that subtraction at each bisection probe; `max_output_runs = 256` caps this at a few hundred microseconds per call, so it is bounded, not asymptotic, but a hot loop over interior positions should iterate the domain rather than `unrank` each position.
- **Flat in the RU count.**  `∂.unrank`/`contains`, `interior(u)`, `unit(k)`, `owner(a)` and `verification_unit(k)` do not move from 100 to 10,000,000 RUs under a `repeat` (exponents 0.01, -0.00, 0.00) nor across 50,000 unrolled `call` steps (`owner` 1.3 µs, exponent 0.02): the unrolled root binary-searches its step table, the `repeat` divides.  `boundary()` itself is O(#definitions) and cached.

### Analysis

- **`serving_request_row`**: the largest shape (`frontier-70B`, 2,578 rows, 2 replay kinds, n = 2.74e+17, 1.34e+14 gates and 1.08e+10 interior positions per RU) folds in 1.02 s (Laplace only; ∝ kinds^1.15), knapsack 875 ms (∝ kinds^1.12), `cost` 1.68 ms, `serving_table` itself 58.8 ms.
- **`serving_step_row`**: the largest shape (`frontier-70B`, 3,089 rows, 513 replay kinds, n = 2.74e+17, 8.38e+12 gates and 590,278,562 interior positions per RU) folds in 1.78 s (Laplace only; ∝ kinds^1.17), knapsack 7.23 s (∝ kinds^1.24), `cost` 2.91 ms, `serving_table` itself 63.2 ms.
- **`serving_cell_gate`**: the largest shape (`frontier-70B`, 2,590 rows, 1,036 replay kinds, n = 2.74e+17, 7,912 gates and 7,911 interior positions per RU) folds in 1.4 s (Laplace only; ∝ kinds^1.04), knapsack skipped at this size, `cost` 10.1 ms, `serving_table` itself 67.5 ms.
- **Linear in the kinds; the knapsack is the constant.**  On synthetic tables the Laplace fold is ∝ kinds^0.97 (1.27 ms per replay kind at 1,024 kinds) and the knapsack ∝ kinds^1.02 (13.6 s vs 324 ms at 256 kinds, 41.9×; not run at 1,024 kinds, where it would take about a minute).
- **Buckets**: `max_buckets` from 128 to 32,768 moves the knapsack ∝ buckets^1.17 in time and ∝ buckets^1.06 in memory (23.3 s at the top, 52.1 MB at 8,192 buckets); the knapsack term goes from 69,664 to 69,951 bits over that range and the reported bound stays 8,192 bits (the Laplace term is the minimum here).
- **`optimize` is one fold per grid point**: ∝ points^0.91, 41.8 ms per policy on the shape used, 12.1 s for 289 policies.  A `(q, s)` grid of 10^4 points is minutes; a coarse-to-fine search is the obvious deployment move.

### Challenge sampling

- **Sublinear in N, linear in K.**  Drawing K ≈ 64 of N candidates takes 182 µs at N = 1,000 and 325 µs at N = 1e+15 (∝ N^0.02): the binomial count is an O(log N)-bit inversion and Floyd's subset costs O(K) draws; 4.78 µs per selected element.
- Along K the cost is ∝ K^0.87 (320 ms for K ≈ 100,000), i.e. ~3.18 µs per selected element: one SHA-256 counter-mode draw plus a set insertion.  The verifier's challenge derivation is never the bottleneck: even a 10^6-RU selection is a second.
- **`derive_replay_selection` is flat in the RU count** (∝ RUs^0.04, 310 µs at 100,000,001 RUs) and `derive_sample_selection` (∝ RUs^0.03, 1.2 ms) depends only on |J| and the VUs per RU: the candidate sets are lazy `Units` domains, never lists.

### Merkle commitments

- **Build is linear: 525k/s values, 1.1M/s domain-separated SHA-256 calls** at 2,000,000 leaves (∝ leaves^0.99), retaining 178 B per leaf (∝ leaves^0.98; 340 MB at the top).  `commit_weights` runs at the same rate (4.15 s for 2,000,000 weights).  The 32-byte digests are Python `bytes` objects and every level is a list: the memory constant is ~3× the digests themselves.
- **Open and verify are O(log L)**: `open` 2.03 µs, `verify_opening` 21.4 µs, proof 674 B at 2,000,000 leaves (depth 21; exponents 0.09, 0.10).
- **Deployment.**  Committing the interior of a sampled RU costs one leaf hash per position plus the tree (2 hashes per position).  The positions are the declared outputs of the RU's verification units (not its own outputs), so the VU marks set the count: a `step` RU of the largest serving shape measured (`frontier-70B`, the 70B-class shape of `docs/frontier-report.md`) has 590,278,562 positions, a `cell` RU 7,911.  At the measured 1.1M/s one `step` interior is 17.9 min in this prototype; at 1G/s (a GPU-class SHA-256 rate, taken as 10^9/s here) it is 1.18 s, far longer than the decode step it belongs to, so a per-value hash over a whole step cannot hide behind the computation at any hashing rate; a `cell` interior is 15.8 µs.  This is the `h` per committed value that the frontier report's cost model charges.  Memory is the other constraint: at 178 B per leaf the Python tree for a `step` interior would be 98 GB; a packed tree is 64 B per leaf.

### Protocol end to end

- **Everything is linear in what the policy selects, and the constants are per position.**  On `ClusterG` up to n = 980,233 (`d32-L2-r4`, q = 1/2, s = 1/4) the prover's total is ∝ n^0.92 and the verifier's ∝ n^0.92; the honest evaluation through the lazy circuit runs at 121k/s gates (8.12 s at the top), the prover then spends 24.2 s (3.0× the evaluation) and the verifier 12.4 s (1.5×).
- **Prover constants** (mean over the ladder): replay 16.6 µs per replayed gate (`replay_unit`: one lazy frame walk and one gate evaluation for every non-source gate of a selected RU), interior commitment 19.6 µs per committed position (value encoding plus the Merkle build), openings 33.2 µs per opening.  The interior is committed at VU-output granularity -- the declared outputs of the RU's verification units that are not its own outputs -- so it holds 3% of the replayed gates on this ladder (19,168 positions for 963,312 gates at the top) and replay + commit together are ~17.1 µs per replayed gate; the prover's marginal cost is q × Σ_selected (|R| × replay + |interior| × commit), independent of s.  The constants drift along the ladder (16 µs → 16.7 µs replay, 17.8 µs → 20.8 µs commit) as the per-RU value dictionaries and the transcript strings grow: allocator and cache pressure, not an asymptotic term.
- **Verifier constants**: 14.5 µs per opening for the Merkle path (250,773 openings at the top, ~9,530 VUs) and 34.3 µs per opening to recompute the gate and compare (the gate lookup dominates); the boundary check and the two challenge derivations are 40.8 ms, i.e. noise.
- **Bytes**: 781 B per opening as canonical JSON (hex digests, one path per opening, no sibling sharing), 235 MB for the whole transcript at the top; the boundary and interior commitment messages are 26.6 KB and 460 B.  Evidence dominates and it is the one message that grows with q·s·|VU|; batching the openings of one VU into a multiproof would cut it by the shared prefix, about `depth - log2(positions per VU)` hashes per opening.
- **`RequestsG` vs `ClusterG`**: per-request RUs make the boundary the prompts and tokens only (|∂| = 32 vs 3,616 at the same n), so the boundary commitment is 759 µs vs 45.6 ms; the per-gate constants are the same (16.7 µs replay).
- **Along q** the replay and interior-commitment phases scale with the realized |J| (∝ q^0.82 on `d8-r32`, 49 RUs; sublinear because |J| is a binomial draw, mean over the seeds 3 at q = 1/32, 6 at q = 1/8, 26 at q = 1/2, 49 at q = 1, and the fixed costs show at small q); **along s** the openings and the verifier's evidence check scale ∝ s^0.93 and ∝ s^0.93 while replay stays put (∝ s^-0.05).
- **Deployment.**  The prover's marginal cost is q × Σ_selected (|R| × 16.6 µs + |interior| × 19.6 µs here), 2.1× the honest per-gate evaluation in the same interpreter at this ladder's interior density — the ratio is the meaningful number: the prover re-runs the selected RUs, so it pays the honest computation once more (at the model's own rate on real hardware), and hashes one value per VU output rather than per gate.  The hash is the part that does not shrink with a faster evaluator, and the VU marks decide how many there are: at 1G/s a `step` RU of the `frontier-70B` serving shape (590,278,562 VU-output positions) commits in 1.18 s, so a deployment either marks VUs coarse enough that the interiors it must hash are affordable (the `cell`/`request` trade-off the frontier report prices) or commits to a compressed digest of the interior.  The verifier's work is q·s·|VU| × (inputs + outputs per VU) × (48.8 µs per opening here, 20,508 openings/s): 10^6 openings a step is 48.8 s in this prototype and a few seconds compiled, and the evidence for them is 745 MB as JSON (610 MB as packed 20-deep paths).

### `output_reach`

- **Linear up to a logarithm in the steps of one definition, whatever their dependency structure.**  `output_reach` on a root with 8,192 sequential `call` steps (a decode chain, the closure `Down(j)` of every step being every later step) takes 68.8 ms and 3.09 MB (∝ steps^1.13 in time, ∝ steps^0.99 in memory) against 87 ms for the parse and 11.2 ms for `transient_ports`; `Index.kinds()` is 83.6 ms.  The closure is swept as intervals of steps over a segment tree (`_step_reach`; the comment above `_segment_bits` in `src/veritor/core/index.py`), `O((S + R) · log S)` for `S` steps and `R` recorded argument ranges on a chain or on siblings reading one step, so the 10^6-step `max_steps_per_definition` limit is seconds, like the parse.
- The same number of independent steps costs 11.4 ms (∝ steps^1.01), many distinct definitions 14.1 ms for 4,096 (∝ ^1.06), and `repeat` nesting is flat in n: the pass is linear in the description, up to a logarithm of the longest step list, everywhere.

## Performance bugs

None observed in this run.

- **Fixed: `output_reach` was Θ(S²) Python iterations on Θ(S)-bit integers for a chain of S steps** (14.3 s at S = 8,192, extrapolating to ~35 days and ~116 GB of bitmasks at the 10^6-step limit).  `_step_reach` (`src/veritor/core/index.py`) now records reads as ranges of steps and sweeps the closure `Down` as intervals over a segment tree: `chain_vs_steps` is ∝ steps^1.13 in this run, 68.8 ms at S = 8,192.  A strided argument run over more than 64 steps and a closure of more than 64 maximal intervals are recorded as hulls, which only enlarge a closure (every reach stays a downstream cut) and are exact on every definition of at most 64 steps; `tests/veritor/core/test_reach.py` checks the sweep against the bitmask closure it replaced.
