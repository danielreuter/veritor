# Honest-server frontier: the 70B serving shape

## Summary

This report prices what an honest inference server naturally achieves under the protocol, so that the verifier's maximum admissible capacity `U_max` can be calibrated to it rather than dictated. The circuit is `FRONTIER_SHAPE` from `veritor.evaluation.frontier`: a 70B-class dense decoder (`d_model = 8192`, 80 layers, 64 heads, `hidden = 4 * d_model`, vocabulary 32768, 16-bit words) serving 2048 requests of 512 prompt + 512 generated tokens, batched 32 at a time: 2.7e17 gates, an output interface of 2048 × 512 × 16 = 16,777,216 bits (16 Mbit), and an honest replay cost of 4.117e17 cost units (identical for all seven partitions, since they tile the same gates). Every admissible partition of the run into replay and verification units was swept (`request/row`, `request/gate`, `step/row`, `layer/row`, `matvec/row`, `row/gate`, `cell/gate`) over the grid `q ∈ {1/2, 1/8, 1/32, 1/128, 1/512, 1/2048, 1/8192}` × `s ∈ {1, 1/8, 1/64, 1/512}` at `eta ∈ {1/2, 1/100, 1/10^6}`: 7 × 28 × 3 = 588 points. Each point is the protocol's own `Bound` (Laplace-only fold, `BoundOptions(knapsack=False, max_buckets=1 << 22)`), `Cost` (prover's expected cost, divided by the honest replay cost) and `expected_work` (verifier's, same denominator). The sweep took 3019 s of wall time (50.3 min of bound computation).

The calibration answer, for a prover overhead budget of at most 1× (the checked run costs no more than twice the honest one) and a verifier work budget of at most 10× the honest replay cost: the smallest capacity any honest partition reaches is obtained by `cell/gate` at policy `q = 1/2, s = 1/8` (prover overhead 0.896×, verifier work 7.42×), and it is 640 bits (0.0038% of the output) at `eta = 1/2`, 4021 bits ≈ 3.9 kbit (0.024%) at `eta = 1/100`, and 11,712 bits ≈ 11.4 kbit (0.070%) at `eta = 1/10^6`. The natural request-level partition `request/row` reaches only 42.6 kbit / 283 kbit / 848 kbit (0.25% / 1.7% / 5.1%) at the same budgets, at `q = 1/8, s = 1`; `request/gate` sits in between at 8.3 kbit / 54.8 kbit / 164 kbit. The four intermediate partitions (`step`, `layer`, `matvec`, `row` replay units) certify nothing at any policy in the grid: their bound is the full 16 Mbit everywhere.

## Method

Points were produced by

~~~python
from pathlib import Path
from veritor.evaluation.frontier import FRONTIER_SHAPE, save, sweep

points = sweep(FRONTIER_SHAPE)           # DEFAULT_PARTITIONS x DEFAULT_GRID x DEFAULT_ETAS
save(points, FRONTIER_SHAPE, Path("docs/data/frontier-70b.json"))
~~~

and the raw points are in `docs/data/frontier-70b.json`; `veritor.evaluation.frontier.load(path)` returns the shape and the points, and `calibration_table`, `partition_table` and `certify` reproduce every table below. Budgets are relative to the honest replay cost: overhead `1` means the prover's expected cost equals the honest computation (so the checked run is 2× honest in total), work `10` means the verifier does ten honest runs' worth of replay.

### The partitions

| partition | replay units | widest replay-unit kind | its `out_bits` | copies |
|---|---|---|---|---|
| `request/row` | 2.05e+03 | `request` | 8 kbit | 2.05e+03 |
| `request/gate` | 2.05e+03 | `request` | 8 kbit | 2.05e+03 |
| `step/row` | 3.28e+04 | `prefill_step(32)` | 320 Gbit | 64 |
| `layer/row` | 8.7e+07 | `layer(512 positions, prefill)` | 192 Mbit | 1.64e+05 |
| `matvec/row` | 1.2e+10 | `square_block(16777216)` | 256 Mbit | 1.64e+05 |
| `row/gate` | 1.24e+13 | `square_block(16777216)` | 256 Mbit | 1.64e+05 |
| `cell/gate` | 3.47e+13 | `cell_unit(eq)` (every unit) | 16 bit | 6.87e+10 |

Only `request` units have an interface narrower than the circuit's output (a request's 512 generated tokens, 8192 bits); the `step`, `layer`, `matvec` and `row` units expose internal activations that are wider than the whole output, and `cell` units expose one word.

## Calibration tables

Each cell is the smallest certified capacity over all partitions and grid policies within both budgets (rows: verifier work; columns: prover overhead), with the partition and policy that reach it. Ties go to the cheaper prover, then the cheaper verifier.

### eta = 1/2

| verifier work \ prover overhead | 1% | 10% | 50% | 100% | 200% | 500% |
|---|---|---|---|---|---|---|
| 1% | 543 kbit (3.3%) `cell/gate` q=1/2048 s=1/8 | 543 kbit (3.3%) `cell/gate` q=1/2048 s=1/8 | 543 kbit (3.3%) `cell/gate` q=1/2048 s=1/8 | 543 kbit (3.3%) `cell/gate` q=1/2048 s=1/8 | 543 kbit (3.3%) `cell/gate` q=1/2048 s=1/8 | 543 kbit (3.3%) `cell/gate` q=1/2048 s=1/8 |
| 10% | 72.9 kbit (0.44%) `cell/gate` q=1/2048 s=1 | 72 kbit (0.44%) `cell/gate` q=1/32 s=1/64 | 72 kbit (0.44%) `cell/gate` q=1/32 s=1/64 | 72 kbit (0.44%) `cell/gate` q=1/32 s=1/64 | 72 kbit (0.44%) `cell/gate` q=1/32 s=1/64 | 72 kbit (0.44%) `cell/gate` q=1/32 s=1/64 |
| 100% | 18.9 kbit (0.12%) `cell/gate` q=1/512 s=1 | 4.89 kbit (0.03%) `cell/gate` q=1/128 s=1 | 4.89 kbit (0.03%) `cell/gate` q=1/128 s=1 | 4.86 kbit (0.03%) `cell/gate` q=1/2 s=1/64 | 4.86 kbit (0.03%) `cell/gate` q=1/2 s=1/64 | 4.86 kbit (0.03%) `cell/gate` q=1/2 s=1/64 |
| 1000% | 18.9 kbit (0.12%) `cell/gate` q=1/512 s=1 | 1.25 kbit (0.008%) `cell/gate` q=1/32 s=1 | 1.25 kbit (0.008%) `cell/gate` q=1/32 s=1 | 640 bit (0.004%) `cell/gate` q=1/2 s=1/8 | 640 bit (0.004%) `cell/gate` q=1/2 s=1/8 | 640 bit (0.004%) `cell/gate` q=1/2 s=1/8 |
| 10000% | 18.9 kbit (0.12%) `cell/gate` q=1/512 s=1 | 1.25 kbit (0.008%) `cell/gate` q=1/32 s=1 | 316 bit (0.002%) `cell/gate` q=1/8 s=1 | 316 bit (0.002%) `cell/gate` q=1/8 s=1 | 63.9 bit (0.0004%) `cell/gate` q=1/2 s=1 | 63.9 bit (0.0004%) `cell/gate` q=1/2 s=1 |

### eta = 1/100

| verifier work \ prover overhead | 1% | 10% | 50% | 100% | 200% | 500% |
|---|---|---|---|---|---|---|
| 1% | 3.33 Mbit (21%) `cell/gate` q=1/2048 s=1/8 | 3.33 Mbit (21%) `cell/gate` q=1/2048 s=1/8 | 3.33 Mbit (21%) `cell/gate` q=1/2048 s=1/8 | 3.33 Mbit (21%) `cell/gate` q=1/2048 s=1/8 | 3.33 Mbit (21%) `cell/gate` q=1/2048 s=1/8 | 3.33 Mbit (21%) `cell/gate` q=1/2048 s=1/8 |
| 10% | 459 kbit (2.8%) `cell/gate` q=1/2048 s=1 | 453 kbit (2.8%) `cell/gate` q=1/32 s=1/64 | 453 kbit (2.8%) `cell/gate` q=1/32 s=1/64 | 453 kbit (2.8%) `cell/gate` q=1/32 s=1/64 | 453 kbit (2.8%) `cell/gate` q=1/32 s=1/64 | 453 kbit (2.8%) `cell/gate` q=1/32 s=1/64 |
| 100% | 119 kbit (0.73%) `cell/gate` q=1/512 s=1 | 30.9 kbit (0.19%) `cell/gate` q=1/128 s=1 | 30.9 kbit (0.19%) `cell/gate` q=1/128 s=1 | 30.5 kbit (0.19%) `cell/gate` q=1/2 s=1/64 | 30.5 kbit (0.19%) `cell/gate` q=1/2 s=1/64 | 30.5 kbit (0.19%) `cell/gate` q=1/2 s=1/64 |
| 1000% | 119 kbit (0.73%) `cell/gate` q=1/512 s=1 | 7.92 kbit (0.05%) `cell/gate` q=1/32 s=1 | 7.92 kbit (0.05%) `cell/gate` q=1/32 s=1 | 3.93 kbit (0.02%) `cell/gate` q=1/2 s=1/8 | 3.93 kbit (0.02%) `cell/gate` q=1/2 s=1/8 | 3.93 kbit (0.02%) `cell/gate` q=1/2 s=1/8 |
| 10000% | 119 kbit (0.73%) `cell/gate` q=1/512 s=1 | 7.92 kbit (0.05%) `cell/gate` q=1/32 s=1 | 1.95 kbit (0.01%) `cell/gate` q=1/8 s=1 | 1.95 kbit (0.01%) `cell/gate` q=1/8 s=1 | 402 bit (0.002%) `cell/gate` q=1/2 s=1 | 402 bit (0.002%) `cell/gate` q=1/2 s=1 |

### eta = 1/10^6

| verifier work \ prover overhead | 1% | 10% | 50% | 100% | 200% | 500% |
|---|---|---|---|---|---|---|
| 1% | 9.63 Mbit (60%) `cell/gate` q=1/2048 s=1/8 | 9.63 Mbit (60%) `cell/gate` q=1/2048 s=1/8 | 9.63 Mbit (60%) `cell/gate` q=1/2048 s=1/8 | 9.63 Mbit (60%) `cell/gate` q=1/2048 s=1/8 | 9.63 Mbit (60%) `cell/gate` q=1/2048 s=1/8 | 9.63 Mbit (60%) `cell/gate` q=1/2048 s=1/8 |
| 10% | 1.3 Mbit (8.1%) `cell/gate` q=1/2048 s=1 | 1.29 Mbit (8%) `cell/gate` q=1/32 s=1/64 | 1.29 Mbit (8%) `cell/gate` q=1/32 s=1/64 | 1.29 Mbit (8%) `cell/gate` q=1/32 s=1/64 | 1.29 Mbit (8%) `cell/gate` q=1/32 s=1/64 | 1.29 Mbit (8%) `cell/gate` q=1/32 s=1/64 |
| 100% | 347 kbit (2.1%) `cell/gate` q=1/512 s=1 | 89.9 kbit (0.55%) `cell/gate` q=1/128 s=1 | 89.9 kbit (0.55%) `cell/gate` q=1/128 s=1 | 88.9 kbit (0.54%) `cell/gate` q=1/2 s=1/64 | 88.9 kbit (0.54%) `cell/gate` q=1/2 s=1/64 | 88.9 kbit (0.54%) `cell/gate` q=1/2 s=1/64 |
| 1000% | 347 kbit (2.1%) `cell/gate` q=1/512 s=1 | 23.1 kbit (0.14%) `cell/gate` q=1/32 s=1 | 23.1 kbit (0.14%) `cell/gate` q=1/32 s=1 | 11.4 kbit (0.07%) `cell/gate` q=1/2 s=1/8 | 11.4 kbit (0.07%) `cell/gate` q=1/2 s=1/8 | 11.4 kbit (0.07%) `cell/gate` q=1/2 s=1/8 |
| 10000% | 347 kbit (2.1%) `cell/gate` q=1/512 s=1 | 23.1 kbit (0.14%) `cell/gate` q=1/32 s=1 | 5.7 kbit (0.03%) `cell/gate` q=1/8 s=1 | 5.7 kbit (0.03%) `cell/gate` q=1/8 s=1 | 1.14 kbit (0.007%) `cell/gate` q=1/2 s=1 | 1.14 kbit (0.007%) `cell/gate` q=1/2 s=1 |

## Per-partition tables

For each partition, the smallest capacity within the verifier work budget (no prover budget), and what that point costs.

### eta = 1/2, verifier work ≤ 10

| partition | U | of output | q | s | prover overhead | verifier work |
|---|---|---|---|---|---|---|
| `request/row` | 41.6 kbit | 0.25% | 1/8 | 1 | 33% | 989% |
| `request/gate` | 8.06 kbit | 0.05% | 1/2 | 1/8 | 90% | 742% |
| `step/row` | 16 Mbit | 100% | 1/8192 | 1/512 | 0.02% | 0.002% |
| `layer/row` | 16 Mbit | 100% | 1/8192 | 1/512 | 0.02% | 0.002% |
| `matvec/row` | 16 Mbit | 100% | 1/8192 | 1/512 | 0.03% | 0.002% |
| `row/gate` | 16 Mbit | 100% | 1/8192 | 1/512 | 0.03% | 0.003% |
| `cell/gate` | 640 bit | 0.004% | 1/2 | 1/8 | 90% | 742% |

### eta = 1/2, verifier work ≤ 1

| partition | U | of output | q | s | prover overhead | verifier work |
|---|---|---|---|---|---|---|
| `request/row` | 708 kbit | 4.3% | 1/2 | 1/64 | 84% | 62% |
| `request/gate` | 8.68 kbit | 0.05% | 1/2 | 1/64 | 84% | 93% |
| `step/row` | 16 Mbit | 100% | 1/8192 | 1/512 | 0.02% | 0.002% |
| `layer/row` | 16 Mbit | 100% | 1/8192 | 1/512 | 0.02% | 0.002% |
| `matvec/row` | 16 Mbit | 100% | 1/8192 | 1/512 | 0.03% | 0.002% |
| `row/gate` | 16 Mbit | 100% | 1/8192 | 1/512 | 0.03% | 0.003% |
| `cell/gate` | 4.86 kbit | 0.03% | 1/2 | 1/64 | 84% | 93% |

### eta = 1/100, verifier work ≤ 10

| partition | U | of output | q | s | prover overhead | verifier work |
|---|---|---|---|---|---|---|
| `request/row` | 276 kbit | 1.7% | 1/8 | 1 | 33% | 989% |
| `request/gate` | 53.5 kbit | 0.33% | 1/2 | 1/8 | 90% | 742% |
| `step/row` | 16 Mbit | 100% | 1/8192 | 1/512 | 0.02% | 0.002% |
| `layer/row` | 16 Mbit | 100% | 1/8192 | 1/512 | 0.02% | 0.002% |
| `matvec/row` | 16 Mbit | 100% | 1/8192 | 1/512 | 0.03% | 0.002% |
| `row/gate` | 16 Mbit | 100% | 1/8192 | 1/512 | 0.03% | 0.003% |
| `cell/gate` | 3.93 kbit | 0.02% | 1/2 | 1/8 | 90% | 742% |

### eta = 1/100, verifier work ≤ 1

| partition | U | of output | q | s | prover overhead | verifier work |
|---|---|---|---|---|---|---|
| `request/row` | 4.59 Mbit | 29% | 1/2 | 1/64 | 84% | 62% |
| `request/gate` | 57.5 kbit | 0.35% | 1/2 | 1/64 | 84% | 93% |
| `step/row` | 16 Mbit | 100% | 1/8192 | 1/512 | 0.02% | 0.002% |
| `layer/row` | 16 Mbit | 100% | 1/8192 | 1/512 | 0.02% | 0.002% |
| `matvec/row` | 16 Mbit | 100% | 1/8192 | 1/512 | 0.03% | 0.002% |
| `row/gate` | 16 Mbit | 100% | 1/8192 | 1/512 | 0.03% | 0.003% |
| `cell/gate` | 30.5 kbit | 0.19% | 1/2 | 1/64 | 84% | 93% |

### eta = 1/10^6, verifier work ≤ 10

| partition | U | of output | q | s | prover overhead | verifier work |
|---|---|---|---|---|---|---|
| `request/row` | 828 kbit | 5.1% | 1/8 | 1 | 33% | 989% |
| `request/gate` | 160 kbit | 0.98% | 1/2 | 1/8 | 90% | 742% |
| `step/row` | 16 Mbit | 100% | 1/8192 | 1/512 | 0.02% | 0.002% |
| `layer/row` | 16 Mbit | 100% | 1/8192 | 1/512 | 0.02% | 0.002% |
| `matvec/row` | 16 Mbit | 100% | 1/8192 | 1/512 | 0.03% | 0.002% |
| `row/gate` | 16 Mbit | 100% | 1/8192 | 1/512 | 0.03% | 0.003% |
| `cell/gate` | 11.4 kbit | 0.07% | 1/2 | 1/8 | 90% | 742% |

### eta = 1/10^6, verifier work ≤ 1

| partition | U | of output | q | s | prover overhead | verifier work |
|---|---|---|---|---|---|---|
| `request/row` | 13.8 Mbit | 86% | 1/2 | 1/64 | 84% | 62% |
| `request/gate` | 172 kbit | 1.1% | 1/2 | 1/64 | 84% | 93% |
| `step/row` | 16 Mbit | 100% | 1/8192 | 1/512 | 0.02% | 0.002% |
| `layer/row` | 16 Mbit | 100% | 1/8192 | 1/512 | 0.02% | 0.002% |
| `matvec/row` | 16 Mbit | 100% | 1/8192 | 1/512 | 0.03% | 0.002% |
| `row/gate` | 16 Mbit | 100% | 1/8192 | 1/512 | 0.03% | 0.003% |
| `cell/gate` | 88.9 kbit | 0.54% | 1/2 | 1/64 | 84% | 93% |

### Best point per partition under both budgets (overhead ≤ 1, work ≤ 10)

| partition | eta | q | s | U (bits) | U | of output | prover overhead | verifier work |
|---|---|---|---|---|---|---|---|---|
| `request/row` | 1/2 | 1/8 | 1 | 42,581.2 | 41.6 kbit | 0.25% | 0.333 | 9.891 |
| `request/row` | 1/100 | 1/8 | 1 | 282,809.0 | 276 kbit | 1.7% | 0.333 | 9.891 |
| `request/row` | 1/10^6 | 1/8 | 1 | 848,260.5 | 828 kbit | 5.1% | 0.333 | 9.891 |
| `request/gate` | 1/2 | 1/2 | 1/8 | 8,254.1 | 8.06 kbit | 0.049% | 0.896 | 7.416 |
| `request/gate` | 1/100 | 1/2 | 1/8 | 54,750.2 | 53.5 kbit | 0.33% | 0.896 | 7.416 |
| `request/gate` | 1/10^6 | 1/2 | 1/8 | 164,206.2 | 160 kbit | 0.98% | 0.896 | 7.416 |
| `step/row` | all | 1/8192 | 1/512 | 16,777,216.0 | 16 Mbit | 100% | 0.000 | 0.000 |
| `layer/row` | all | 1/8192 | 1/512 | 16,777,216.0 | 16 Mbit | 100% | 0.000 | 0.000 |
| `matvec/row` | all | 1/8192 | 1/512 | 16,777,216.0 | 16 Mbit | 100% | 0.000 | 0.000 |
| `row/gate` | all | 1/8192 | 1/512 | 16,777,216.0 | 16 Mbit | 100% | 0.000 | 0.000 |
| `cell/gate` | 1/2 | 1/2 | 1/8 | 640.0 | 640 bit | 0.0038% | 0.896 | 7.416 |
| `cell/gate` | 1/100 | 1/2 | 1/8 | 4,021.3 | 3.93 kbit | 0.024% | 0.896 | 7.416 |
| `cell/gate` | 1/10^6 | 1/2 | 1/8 | 11,711.7 | 11.4 kbit | 0.070% | 0.896 | 7.416 |

For the four intermediate partitions every grid point is at the cap, so the "best" point is simply the cheapest one.

### The `cell/gate` grid at eta = 1/100

Cells are `U as % of output / prover overhead / verifier work`.

| q \ s | 1 | 1/8 | 1/64 | 1/512 |
|---|---|---|---|---|
| 1/2 | 0.0024% / 1.33 / 59.3 | 0.024% / 0.896 / 7.42 | 0.19% / 0.841 / 0.927 | 1.4% / 0.834 / 0.116 |
| 1/8 | 0.012% / 0.333 / 14.8 | 0.095% / 0.224 / 1.85 | 0.72% / 0.21 / 0.232 | 5.4% / 0.209 / 0.029 |
| 1/32 | 0.048% / 0.0834 / 3.71 | 0.37% / 0.0561 / 0.464 | 2.8% / 0.0527 / 0.0579 | 20.8% / 0.0522 / 0.00725 |
| 1/128 | 0.19% / 0.0209 / 0.927 | 1.4% / 0.0141 / 0.116 | 10.6% / 0.0132 / 0.0145 | 79.5% / 0.0131 / 0.00181 |
| 1/512 | 0.73% / 0.00529 / 0.232 | 5.4% / 0.00358 / 0.029 | 40.7% / 0.00337 / 0.00362 | 100% / 0.00334 / 0.000453 |
| 1/2048 | 2.8% / 0.00139 / 0.0579 | 20.8% / 0.000959 / 0.00724 | 100% / 0.000906 / 0.000905 | 100% / 0.000899 / 0.000113 |
| 1/8192 | 10.8% / 0.00041 / 0.0145 | 79.5% / 0.000303 / 0.00181 | 100% / 0.00029 / 0.000226 | 100% / 0.000288 / 2.83e-05 |

The same grid for `request/row` reads 0.32% / 1.7% / 7.1% / 28.7% down the `s = 1` column and is at 100% from `q = 1/512` on; `request/gate` reads 0.32% / 1.7% / 7.1% / 28.7% at `s = 1` and, unlike `request/row`, barely degrades as `s` falls (0.33%, 0.35%, 1.4% at `q = 1/2`).

## Reading the frontier

- The bound is governed by how much output a single admitted unit error can reach. With the Laplace fold, `U` is close to (number of unit errors whose joint survival stays above `eta`) × (bits reachable per error). The first factor is `Lambda / c(1) = -ln(eta) / -ln(1 - q s)`: 10.7, 71.4 and 214 errors at `q = 1/2, s = 1/8` for the three etas. The second factor is the unit's `out_bits` plus the `log2` of the number of units it could be: for `cell/gate` that is 16 + log2(3.5e13) ≈ 61 bits, and the sweep gives 60, 56 and 55 bits per error at the three etas. For `request/row` one admitted error is a whole request, 8192 bits, and wherever both are below the cap `request/row` is 130× to 150× worse than `cell/gate` at the same policy (8205 versus 64 bits at `q = 1/2, s = 1, eta = 1/2`).
- The four intermediate partitions are useless at this scale because their replay units are "super-escapes": a `prefill_step` exposes 320 Gbit of activations, a prefill `layer` 192 Mbit, a `square_block` 256 Mbit, all wider than the 16 Mbit output, so a single admitted unit error reaches every output and the bound is capped at `out_bits` for every grid policy (436 of the 588 points sit at the cap, all 336 points of these four partitions among them). The `request` unit is the only coarse unit whose interface (its 512 tokens) is narrower than the circuit's output, which is why `request/*` is informative and `step/*` is not, though `step` is finer.
- Verification-unit width matters too. `request/row` and `request/gate` share replay units but `request/row` marks the one-hot as one verification unit with 524 kbit of output and each attention head with 2 kbit, so a single erroneous row is covered by the whole request (8192 bits); `request/gate` marks 16-bit gates, so a single error within a request costs about 16 + log2(1.3e14) ≈ 63 bits. That is the gap between 276 kbit and 53.5 kbit at `eta = 1/100` under the headline budgets, and it is why `request/gate` keeps its capacity as `s` falls while `request/row` loses it.
- Monotonicities hold without exception: `bits` is nondecreasing as `eta` decreases (0 violations over 392 pairs), as `q` decreases (0 of 1764) and as `s` decreases (0 of 882); `bits ≤ out_bits` everywhere; `cell/gate ≤ request/row` at all 84 (q, s, eta) triples, and `cell/gate ≤ request/gate` at 83 of 84 (the exception is a 1-bit difference at `q = 1/2, s = 1/512, eta = 1/2`, 37,596 vs 37,597 bits, which is fold rounding). Going from `eta = 1/2` to `1/100` multiplies `U` by about 6.3, and to `1/10^6` by about 18, tracking `ln(1/eta)`.
- The frontier is one-dimensional in `q s` and bends on verifier work, not prover overhead. Verifier work is `118.7 q s` to four digits for `cell/gate` (the proofs of the sampled verification units; the replays themselves are negligible), and `U` depends on the policy almost only through the per-error survival `1 - q s`: `q = 1/2, s = 1/64` and `q = 1/128, s = 1` both give work 0.927 and 31.3 versus 31.6 kbit at `eta = 1/100`. Prover overhead is `q (1.67 + s)` plus a negligible constant and is the same for every partition at a given policy (0.896 at `q = 1/2, s = 1/8` for all seven; the finest partition adds under 0.01% for its unit boundaries), so the 1× prover budget only excludes `s = 1` at `q = 1/2` (1.33×). Every cell of every calibration table is `cell/gate`, and the columns beyond 10% overhead rarely change: the binding constraint is the verifier's budget, and it buys, at `eta = 1/100`, 3.33 Mbit at 1% work (`q s = 1/16384`), 453 kbit at 10% (`1/2048`), 30.5 kbit at 100% (`1/128`), 3.93 kbit at 1000% (`1/16`) and 402 bit at 10000% (`1/2`). Within a fixed `q s`, a higher `q` lowers `U` by about one percent and raises the prover's overhead in proportion, so the prover budget decides how the product is split.
- Recommendation for `U_max`. The verifier does not choose the partition; the server does, and it will choose the cheapest one that the verifier accepts. In this cost model marking at `cell/gate` costs the prover nothing measurable over `request/row` at the same policy, so a verifier can set `U_max` to what `cell/gate` achieves at its budget: with prover overhead ≤ 1× and verifier work ≤ 10×, `U_max ≈ 640 bits` at `eta = 1/2`, `≈ 4.0 kbit` at `eta = 1/100`, `≈ 11.7 kbit` at `eta = 1/10^6`, at policy `q = 1/2, s = 1/8`. If the verifier must also accept servers that mark only at the request level (`request/row`, the natural serving partition), `U_max` has to be 42.6 kbit / 283 kbit / 848 kbit instead, 66× to 72× looser, and such servers are Pareto-dominated by `request/gate` at 8.3 / 54.8 / 164 kbit. Any `U_max` below the cheapest compliant partition's number turns honest servers away.

## Sanity checks

- The honest replay cost `honest_cost(serving_table(FRONTIER_SHAPE, "request", "row"))` is 411,684,960,876,888,064 (4.117e17) and is identical for all seven partitions; `out_bits` is 16,777,216 for every point.
- 588 points, one per (partition, q, s, eta); every partition has its 84 points.
- All monotonicity checks passed (see above); no point exceeds `out_bits`.
- `cell/gate` is below `request/row` at every (q, s, eta).

## Timing

Bound computation per partition (`Point.seconds`, summed):

| partition | points | total s | mean s | max s |
|---|---|---|---|---|
| `request/row` | 84 | 349 | 4.2 | 7.7 |
| `request/gate` | 84 | 1090 | 13.0 | 23.4 |
| `step/row` | 84 | 454 | 5.4 | 9.3 |
| `layer/row` | 84 | 154 | 1.8 | 2.8 |
| `matvec/row` | 84 | 194 | 2.3 | 3.4 |
| `row/gate` | 84 | 616 | 7.3 | 12.5 |
| `cell/gate` | 84 | 161 | 1.9 | 2.7 |

Total: 3018 s (50.3 min) of bound time; the sweep's wall time was 3019 s. `Cost` and `expected_work` are negligible. `request/gate` is slowest because its `request` kind has 1.3e14 verification units and the per-copy power series runs to the error limit at every eta.

## Caveats

- The `KindTable` is synthetic: `veritor.evaluation.serving` writes the per-kind profile the compiler would produce for the toy decoder of `veritor.constructors.lm` at these dimensions. It has been checked against compiled toy circuits only at toy scale (`tests/veritor/evaluation/test_serving.py`); at 70B scale nothing is traced, and the structure is the toy's (square MLP, argmax head, no normalisation, no softmax).
- The bound is the Laplace-only fold (`BoundOptions(knapsack=False, max_buckets = 1 << 22)`). It is an upper bound on `U` and is looser than the knapsack fold, which was not run because at these scales a single unit error costs far less than `Lambda / 2048` and the knapsack grid would round it to zero. The Chernoff bound also does not use the strict inequality `sigma(E) > eta`, so at `q = 1/2, s = 1` it admits the single error that survives with probability exactly `1/2`.
- The cost model's weights are the ISA gate set's replay and proof costs (`add`/`sub`/`lt`/`eq`/`shr` 1, `mul` 2, sources 0 to replay and 1 to prove); prover overhead and verifier work are ratios of those unit counts to the honest replay cost, not measured time, and no per-unit commitment overhead beyond what `Cost` charges is included, which is why `cell/gate` is nearly free.
- The bound is not reach-aware: a replay unit's escape is its `out_bits`, not `min(out_bits, reach_bits)`, so the intermediate partitions are charged their full activation width even though nothing but the request's tokens reaches the output. With that refinement pending, their 100% rows should be read as "not yet bounded" rather than "unsafe".
- The grid is coarse (powers of 2 in `q` and 8 in `s`), so the optima quoted are grid optima; `certify` breaks ties towards the cheaper prover.
