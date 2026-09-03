# Overnight report (2026-09-02 → 09-03)

Second night. Everything referenced here is on `main`; commit hashes are given where a decision
landed. The previous report is `docs/overnight-report.md` (09-01 → 09-02); its §5.1 (`Bound` must
use output reach) was resolved on the morning of 09-02 (`fb3d074`) and is the starting point here.

## 1. Where the implementation stands

One hundred and twenty commits on `main` since the last report (`8c9d501..`), 49 on the first
parent, 401 files, +88 000 / −47 000 lines: 31 000 of the additions are recorded data
(`docs/data/*.json`, golden vectors, GPU captures) and 45 000 of the deletions are the retired
legacy package. `src/veritor` is 29 300 lines in 85 modules; the suite is 1 604 tests, 1 591 in
the default run of about three minutes and 13 marked `slow` (SP1 proofs, GPU-simulator
comparisons, the million-request scale test). `mypy`, `ruff check` and `ruff format --check`
are clean over `src`, `tests`, `benchmarks` and `zk`, and CI blocks on all three.

| Landed | Commit | What |
|---|---|---|
| Honest-server frontier | `1ba6fae`, `840772a` | `KindTable`, `ServingShape`, `RequestsG`; 588-point sweep for the 70B shape under the reach-aware, recompute-honest `Bound`/`Cost` |
| Reproducible sweeps, CI | `3f69054`, `d5b2000` | parallel resumable sweep with a manifest; CI on the project's Python |
| GPT-2 structure | `7638c51`, `8242ee4` | structural ML gate set (placeholder, see §2.3), `GPT2G`: 54.6 G gates / 176.8 M VUs compile in 0.1 s; 99.97 % of gates sit in 16-bit VUs |
| Stress catalogue | `c21c270` | mechanisms M1–M8, scenarios S/C/N/W/E, hypothesised compiler gaps, order of work (`docs/stress-tests.md`) |
| Benchmarks | `eac882e`, `863d01d` | eight sweep families, `docs/benchmarks.md` regenerated from `docs/data/benchmarks.json`, fast asymptotic perf tests |
| Datacenter simulation | `6578b34`, `97466b5` | schedules with attempt lengths and restarts, the sampled kind over public randomness, a workload simulator and an exfiltration adversary; observed survival within 1σ of the prediction; renamed `veritor.demo` → `veritor.simulation` |
| Literature brief | `453cad8` | `docs/notes/datacenter-realities.md`: twelve topics, 93 checked sources |
| Hardware semantics | `a5edb96` | `core/silicon.py`: Ada BF16 and FP8 `mma.sync` reproduced bit-exactly against golden vectors; fixed-order GEMM chain bit-exact on all 60 GPT-2 shapes |
| ZK proof backend | `203fe1a` | `protocol/proofs`: obligations, statements, witnesses, `ProofBackend`, cross-session `BatchPlan`, coverage check; `TransparentBackend` byte-identical to the old path; SP1 6.4.0 checker with a measured proof |
| Rate and ancestor cut | `0426d6c` | `cut_bits = min(out, reach, ancestor)`; closed-form rate ρ with a direct proof; α in `CostParameters` |
| Headline estimate | `42b9e2b` | `veritor.evaluation.global_estimate`, `docs/global-estimate.md` |
| Control-flow scenarios | `211e913` | toy MoE layer and speculative decoder, each padded (M5) and advised with an in-circuit check (M4); rows C1, C2, C3 |
| `output_reach` | `a93584f` | interval sweep over a segment tree; 10⁶-step definitions in seconds |
| Fault declarations M6, scenarios N4/N5/S7/C5/W1–W3 | `25c1afb`, `e642fc4` | `max_faults` in the header, declarations on the interior message, `DECLARED_PROGRAM` obligations; `Bound` charges the adaptive prover (§2.6); `ModelsG`, `AdaptedRequestsG`, `TruncatedRequestsG`; rows in `docs/data/stress-protocol.json` |
| Stress framework, scenarios S1–S8/C4/C6/N2/N3/E1 | `ec463bd` | `veritor.stress` (rows, measure, report renderer for `docs/stress-tests.md` §4); Schedule v3 (`resume`, `chunk`), time-major `ClusterG`, grouped `RequestsG` (same-kind requests share one `repeat`), constrained decoding, tensor parallel, `PrefixG`, `union_gate_set`; 40 rows rendered (§3.5) |
| GPT-2 Small on pinned gates, RTX 4090 capture | `e063ea6`, `cdf85b0` | `Gate.arg_widths`; the pinned gate set replaces `ml_gates` (1-, 16-, 32-bit gates: `tc_dot16` chains, fp32 elementwise with own `exp`/`tanh`/`rsqrt`); numpy reference; 4090 capture through our own kernels matched bit-exactly on CPU; 37 225 sampled VUs re-executed, 0 disagreements; `run_protocol` on 1- and 12-layer one-token slices (§3.6); proof statements take a mixed-width gate set |
| Epoch layer | `0c87c60` | `protocol/epoch.py`: commitment stream (hash chain), sealed rounds, verifier-private round seed with HMAC-derived per-run seeds, `EpochVerifier`/`EpochProver`/`run_epoch`, `EpochReport` = Σ_rounds `Bound(union, θ, η/rounds, f_max)`; `analysis.union(tables)`; `VerifierSession` takes a seedless `Claim` (§2.7, `docs/epoch.md`); 52 tests |
| VU-granularity interior | `6eca696`, `6aa78c0` | `Index.interior(r)` = ⋃ `Out(V)` over the VUs in `R_r` minus `Out(R_r)` (lazy, O(depth) `rank`/`unrank`, one linear walk to iterate); `KindSummary.interior_count` drives `Cost` and `parameters`; evidence for a sampled VU opens its inputs and outputs and the verifier recomputes the VU; proof layer, SP1 guest and wire v7 rebuilt; interior commitment on the ClusterG ladder 11.1 s → 0.4 s (§2.8, §3.3) |
| Compiler gaps closed | `91f5595` | check outputs (a root's `checks` field; `FORMAT_VERSION` 3; worth 0 bits in every cut; `CHECK_MISMATCH` at admission and on the opened boundary); exact advice bits (`G.advice_bits(x, a)` with the canonical zero-padded encoding checked; Schedule v4 bit-packed); route-independent MoE step bodies (the route enters at the call site); conditionally-absent outputs as blank check outputs (`TruncatedRequestsG` v2); the `Tracer` normalises a count-one range to a wire (§2.9) |
| Legacy code retired | `cf09060` | `src/circuit_cut_analysis` (2.2 MB, 89 lint findings) and `archive/` deleted, −45 560 lines; the exact min-cut reference is a 30-line networkx max-flow; the GPT-2 cross-check keeps the legacy DAG's recorded counts; dependencies are now `networkx` and `numpy` (`jax`, `scipy` were the legacy package's) |

Packages now: `core/` (+ `silicon.py`), `compile/`, `protocol/` (+ `proofs/`), `analysis/`
(+ `rate.py`, `faults.py`), `evaluation/` (`frontier`, `serving`, `global_estimate`),
`constructors/` (client code: `lm`, `moe`, `speculative`, `requests`, `prefix`, `cluster`,
`schedule`, `gpt2`, `tenancy`, `tracer`), `simulation/` (`datacenter`, `faults`), `stress/` (`rows`, `measure`, `report`);
`benchmarks/` and `gpu/` at the top level.

## 2. Decisions made tonight

### 2.1 The bottleneck has three cuts, and the rate has a closed form (`fb3d074`, `0426d6c`)

For an error set confined to a unit `S`, the outputs it can move are bounded by
`2^{min(out_bits(S), reach_bits(S), ancestor_bits(S))}`: the unit's declared output width, the
width of the circuit outputs downstream of it, and the width of the *interfaces of its ancestors*
(a unit deep inside a step cannot move more than the step's own outputs, whatever its reach).
`Bound` charges the minimum; every approximation still rounds toward admitting more, and the
exhaustive reference agrees on random circuits.

With the cuts in place the fold has a closed form: ρ (bits of capacity per bit of threshold) is
a function of four numbers of the kind table, `analysis/rate.py` states and proves it, and it is
within 0.4 % of the fold wherever the whole-RU channel binds (17–27 % above it at tiny `s`, where
the fold's scattered-error terms dominate). This is what makes the headline estimate (§3.1) a
formula rather than a sweep.

### 2.2 Proving cost is a parameter, and the proof layer is pluggable (`203fe1a`, `0426d6c`)

`CostParameters.proof_factor` (α, proving MACs per native MAC) enters `Cost` where the sampled
VUs are proved; `Optimize` trades it against replay and commitment. The protocol derives
*obligations* (one per sampled VU: a statement over committed positions, a witness) and hands them
to a `ProofBackend`; `TransparentBackend` reproduces the previous evidence byte for byte, so the
whole existing suite is the conformance test for the abstraction. A `BatchPlan` groups
obligations across sessions and a coverage check refuses a plan that leaves any sampled VU
unproved. The SP1 backend executes, proves and verifies a generic checker guest; one real 16-VU
proof took 18 s, 2.8 MB, verified in 0.07 s. Measured cycle counts: 116–212 per gate, 2 694 per
Merkle level, 3 419 per opened position; 85 % of a proof is Merkle authentication.

### 2.3 Gate semantics are pinned at the silicon level, not by a specification (`a5edb96`, `8242ee4`)

Three findings decided this. (i) Ada's BF16 and FP8 `mma.sync` are reproducible bit-exactly by
a small software model (24.9 M elements per dtype, 100 %). (ii) The synthetic "tensor-core dot"
contract we had written down matches no real GPU (18 % / 9 % agreement). (iii) A published
Hopper FP8 simulation is wrong on Ada (Ada sums two groups of sixteen). So the ML gate set's
semantics cannot be a paragraph in the paper; they are a reference implementation plus golden
vectors per silicon, and `ml_gates` is marked as the structural placeholder until each gate has
its pinned sequence. The GPT-2 branch (`e063ea6`) built the first such set and retired the
placeholder: `tc_dot16` chains for the matmuls and fp32 elementwise gates with our own
`exp`/`tanh`/`rsqrt` sequences (no `libdevice`); its 4090 capture agrees with the CPU model on
every one of the 11 664 980 recorded elements and every generated token (§3.6).

### 2.4 Advice for control flow: padded or advised, with the check inside the circuit (`211e913`)

An MoE route or a speculative-decoding acceptance can be *padded* (M5: run every expert, mask;
zero advice) or *advised* (M4: the route rides in `a`, the circuit recomputes the check and emits
an `ok` word the verifier requires to be 1). Both are constructor choices; the compiler and
`Bound` are unchanged. Findings: at equal θ padding wins in `U` everywhere, by exactly `|a|` plus
the `ok` words; at equal absolute prover cost advice wins once the compute it saves buys back more
`Bound` than its bits cost (E = 8, k = 1 in the toy; the crossover is `E/k` against
`k·log₂E`, so at datacenter ratios advice should win outright). A lying route yields `ok = 0`; a
forced `ok = 1` is `RELATION_REJECTED`.

### 2.5 `output_reach` is an interval sweep (`a93584f`)

Reads are recorded as ranges of steps and the closure `Down(j)` is swept last-to-first over a
segment tree. Chain of 8 192 steps: 14.2 s → 70 ms; 10⁶ steps (the `max_steps_per_definition`
limit): 11.5 s and 350 MB where the bitmask closure extrapolated to ~35 days. The only
over-approximations are hulls (a strided read spanning more than 64 steps, a closure of more than
64 maximal intervals); both only enlarge a closure, so every reach stays a downstream cut, and
the flagship shapes (decode chains, KV-cache chains, siblings over a broadcast weights step) are
exact at any length.

### 2.6 Fault declarations are priced for the adaptive prover (M6; `25c1afb`, `e642fc4`)

`VerifierParameters.max_faults` (`f_max`) is bound in the header. After the q-challenge the
prover may declare up to `f_max` VUs of the opened RUs incorrect on the interior message; a
declared VU's openings are still authenticated and every reader is checked against its value,
but its own relation is obliged under `DECLARED_PROGRAM` (zero gates) in the transparent path
and every proof backend alike. Existing transcripts are byte-identical (`max_faults = 0` and
empty declarations are omitted from the wire). The honest server's detector is `self_check`:
recompute every gate of an opened RU from the values it holds.

The brief said `Bound += f_max · u(1)` with `u(1) = W_V + log₂|S|`, the bits that name one VU
and its contents. That is the price of a declaration *fixed before the challenges*, and it is
what the q = 1 case pays. But the declaration comes after `J`, and an adaptive adversary
corrupts one VU in each of many RUs and pardons whichever `f` of them were opened: with `N_J`
corrupted VUs among the opened RUs it survives with `(1 − s)^{max(0, N_J − f)}` instead of
`(1 − s)^{N_J}`, a factor `1/(1 − s)` per declaration, worth about `1/q` extra corrupted VUs
when `s` is small. `bound(..., max_faults=f)` therefore charges the smaller of two rigorous
bounds: the fold at threshold `η(1 − s)^f` (since `σ_f(E) ≤ (1 − s)^{−f} σ_0(E)`; vacuous at
`s = 1`), and the fold at `η/(1 + |S|)^f` plus `f · u(1)` (the best declaration is one of at
most `(1 + |S|)^f` sets and a maximum is at most a sum; valid for every `s`), plus the fixed
price at `q = 1`. For the headline policy one declaration costs about `ρ · log₂(1/(1 − s)) ≈
6·10⁹` bits against `U = 1.9·10¹³`, so `f_max = 4` is 0.13 % of `U`: affordable, but three
orders above `u(1)`. A protocol that wanted the `u(1)` price would have to take declarations
before the q-challenge, which an honest server can only do for faults it detects without
replaying (ECC events, crashes, NaN checks), not silent corruption.

Measured on the small simulation (3 791 VUs, `W_V = 64`: the one-hot VU declares eight 16-bit
outputs but its ancestors' interface cuts the 128 bits to 64): `u(1) = 75.9` bits; at θ = (1, 1)
`Bound` goes 0 → 75.9 with `f_max = 1`; at θ = (1/2, 1/8) it is capped at the 480 output bits
with or without the term. The injected fault
is `RELATION_REJECTED` at `f_max = 0` once sampled, accepted with one declaration at
`f_max = 1`, and a second undeclared corruption is rejected again. The fleet arithmetic in the
N4 row: at the literature's SDC rate (2.8·10⁻⁷ per device-hour) a 16 384-GPU fleet needs
`f_max = 2` per hour and `f_max = 4` per day at a 10⁻⁶ tail.

### 2.7 The epoch is the unit of the guarantee (`0c87c60`, `docs/epoch.md`)

Your decision (η per epoch, a cap in `pp`) is implemented as *rounds*: an epoch has `rounds`
challenge times, each round is bounded at `η / rounds` and the epoch's capacity is the sum over
rounds; `rounds = 1` is the single end-of-epoch challenge the headline estimate prices, and
`rounds = N` recovers per-run challenges. Runs are admitted and their boundaries committed into
a hash chain as they happen; at round close the verifier draws a round seed, derives every run's
`q` and `s` seeds by HMAC over the seal, and each run finishes through the existing session
machinery. The union-of-runs `Bound` is sound because every boundary of the round is fixed
before any seed exists, interiors of a run are committed after its `J` and before its `T` as
today, and per-unit Bernoulli sampling makes the union's sample the product of the runs'
samples; an adversary spreading one corrupted VU over three runs of a round is accepted at
`(1 − qs)³` over 400 epochs, within the simulation's tolerance. `analysis.union` merges kind
tables (copies sum; `reach`/`ancestor` take the max, sound since the cut is a min), and
`bound(union([T] * N))` equals the headline's scaled-table method. Two things the agent settled
that you should know: **the round seed is verifier-private, not a beacon** — the `s` seeds derive
from it, and a public round seed would let the prover grind interiors against the sample (a
beacon can serve `q` only if `s` comes from a second secret); and `Bound`'s monotonicity under
adding runs, which the running `U_max` admission check relies on, held on every random mixture
tried but is not proved.

### 2.8 The interior is the VU outputs (`6eca696`)

Flag 3 of the earlier draft, now built. The interior of a replay unit is the union of its
verification units' declared outputs minus its own outputs (`Out(R) ⊆ ⋃_V Out(V)` follows from
the marks tiling `R` with VUs and `R` holding no gate steps of its own; `validate_marks` states
it and `Definition.interior_total` enforces the count). The prover commits 3 % of the gates it
replays on the ClusterG ladder (19,168 positions where 279,168 were) at 19.6 µs a position, so
interior commitment is 2.4 % of prover time and replay is the whole marginal cost. The sampled-VU
check moved with it: the verifier opens a VU's inputs and outputs and recomputes the VU's whole
program, so internal gates are checked by recomputation with no committed value to equivocate on
and every VU output has one owner; what is lost is attribution — a fault inside a VU is now
named by its output, which is what an M6 declaration names anyway. Two consequences downstream:
`union` sums `interior_count` over its roots; and the cheaper interior moves the MoE
padding-versus-advice crossover at equal absolute prover cost from E = 8 to E = 16 (`k = 1`),
since the padded server's budget at θ = (1/2, 1/8) fell from 333,957 to 228,933 and the advised
server at E = 8 can no longer afford the policy that made it win. Stress rows were re-recorded
(overheads fall, capacities unchanged).

### 2.9 Check outputs, exact advice, one body per route-dependent step (`91f5595`)

The five gaps the control-flow scenarios surfaced are closed. A root definition may now mark
declared outputs as *check outputs*, fixed to a constant of the description: the verifier
requires the claim to hold the constant at admission and the opened boundary value to equal it
(`CHECK_MISMATCH`, checked before the public-I/O comparison so a lying route or acceptance is
named as such), and `Bound`, the reach, the ancestor cut and the exact reference all price them
at zero (an output fixed to a constant contributes a factor 1 to `|Y|`; the sentence is in
`bound.py`). The MoE and speculative `ok` words and S7's absent slots are check outputs: C2b's
`U` fell 168 → 132 and C3b's 232 → 194, the `ok` words' 16 bits and their reach inflation.
Advice is charged at the bit length the constructor declares, after the compiler checks that
`a` is the canonical `ceil(bits/8)`-byte zero-padded encoding of that many bits (undeclared
advice still costs `8|a|`): C2b 24 → 20 bits, C3b and S7 24 → 18, and the schedule, which was a
20-byte magic plus 28 bytes per join, is bit-packed (Schedule v4: Elias-gamma header, fixed-width
fields, one bit for `resume`), so C1b's advice fell 1 632 → 118 bits and its capacity
2 208 → 694, N4/N5's 3 648 → 271 (capacity 4 128 → 751). The advised MoE step is one definition
whatever the route: the router's columns, the chosen experts' weights and the expert *ids*
(constant-table words) enter as the parent ranges at the call site, and `route_check` ranks the
advised experts against the rest, so the advised descriptions in the crossover sweep fell from
55–169 kB to 39–52 kB at the cost of `5k(E − 1)` gates per token (+2 %). Two things did not
change: the request RU's own digest still carries the route, because its body holds the call
sites whose ranges *are* the route (the design, not a gap), and nothing is charged zero for an
output-determined choice (flag 5 stands; S7's stream length is 18 bits of advice, and
`docs/stress-tests.md` §1 says why a presence mask is not free). Landing it on the VU-output
interior took the wire to v8 (both branches had claimed v7), a re-pinned baseline transcript and
regenerated SP1 vectors; the C2 crossover ladder runs at vocabulary 16 because advised routing
names experts by the constant table.

## 3. Measurements

### 3.1 The headline estimate (`42b9e2b`, `docs/global-estimate.md`)

A year of serving as one circuit (2.93·10¹³ requests of 1 024 tokens, 4.96·10²³ VUs on a
70B-class dense decoder), the prover spending 1 % of the honest computation on the protocol:

**U(λ = 40) = 1.90·10¹³ bits = 2.37 TB per year of observed outputs**, at α = 7.14·10⁷
(OpenVM, TC_MATMUL); 5–6 TB at the SP1 α; 0.24 TB at a 10 % budget; 23.7 TB at 0.1 %.

The optimum replays `q ≈ 1.6·10⁻⁸` of requests and proves `s ≈ 0.9 %` of their VUs; the budget
goes entirely to proofs. Consequently the estimate is insensitive to hashing cost, to
gate-versus-VU interior granularity, and to the token count: within 5 %,
`U ≈ λ (W_V + log₂ #VUs + 2) α ln 2 / budget`. The capacity is set by the proving factor and the
budget, and depends on the size of the year's computation only through `log₂ #VUs`.

### 3.2 Proving cost, α (`docs/zk-backend.md`)

OpenVM TC_MATMUL 7.1·10⁷, TC_DOT 1.5·10⁸, SP1 (TC_DOT-patched) 1.8·10⁸, all on an RTX 4090
against a native 1.8·10¹⁴ fp8 MAC/s. The toy ISA through SP1 is 2–3·10⁹, three orders worse,
entirely per-scalar Merkle authentication. Open: guest SHA over a precompile (~2.5×), verifying-key
caching, an OpenVM host binary.

### 3.3 Benchmarks and bottlenecks (`docs/benchmarks.md`)

Prover, before VU granularity: 56.8 µs per interior position, i.e. per-position hashing *was*
the prover's cost. After (§2.8): 19.6 µs per committed position over 3 % of the gates, 16.6 µs
per replayed gate, and replay is the prover's whole marginal cost. Verifier: 14.5 µs Merkle plus
34.3 µs recomputation per opening. Merkle: 1.06 M hashes/s. `Compile` and the kind table are linear in the description everywhere; the one
super-linear pass (`output_reach`) is fixed (§2.5).

### 3.4 Hardware semantics (`docs/hardware-semantics.md`)

Ada BF16 and FP8 `mma.sync`: 100 % bit-exact over 24.9 M elements per dtype. Fixed-order GEMM
chain: 60/60 GPT-2 shapes bit-exact at 1.2–3.7× cuBLAS time. Pod cost $0.63. GPT-2 Small whole
forward on the 4090 with our own kernels versus the numpy model: 11 664 980 elements, 0
mismatches, identical tokens (27-token prompt, 8 generated); 423.9 M gates, 133.3 M VUs, two
replay units. Our fixed-order chain kernels run 1.8–2.8× slower than cuBLAS on the 768-wide
shapes, 9× on the 3072→768 projection and 28–47× on the 50 k-wide vocabulary projection
(`gpu/gpt2/results/bench.json`); the vocabulary projection is where a production kernel would
have to be tuned.

### 3.5 Stress scenarios (`ec463bd`, `docs/stress-tests.md` §4)

Forty rows in `docs/data/stress*.json`, each written by the test that measured it and
rendered by `python -m veritor.stress.report` (`--check` fails when the table is stale).
What the scheduling rows say, in one line each:

- **Serving is schedule-invariant under M1.** `RequestsG` compiles the same requests to a
  digest-identical circuit under fcfs, reversed, and one-slot schedules, through preemption by
  recompute (S3) or by swap (S4), chunked prefill (S5), a prefill-to-decode pod handoff (S6),
  and retries (S8); `ClusterG` (M4, the schedule as advice) pays ~1 400–1 600 advice bits and a
  1.4× overhead against 0.89× for the same outputs.
- **Scale.** 10³–10⁶ requests over ~100–120 shapes (S2, E1): compile 54–80 ms, kind table
  20–33 ms, description ~340 KB, `Bound` 2–6 s (the honest bottleneck). Grouping same-kind
  requests into one `repeat` keeps the root's output runs from growing with the request count.
- **Constrained decoding (C4)** is 8–23 gates per token and `2..8 × vocab` per request over the
  reference; `ClusterG` cannot express per-occupant masks (a recorded gap).
- **Prefix caching (C6).** Making the shared prefix its own replay unit (`PrefixG`) saves
  `(k − 1) × 4 776` replay gates per request for 64 boundary words, at a capacity cost (uncapped
  818 vs 428 bits) and 1.38× overhead against 0.89× for recompute.
- **Fleets (N2, N3).** Tensor parallel 1/2/4 gives identical outputs and `U`, two dot kinds
  differ per degree; a mixed sm80/sm90 fleet compiles under one gate set (`union_gate_set`,
  `name@namespace` operators, shared sources), and a foreign gate name fails to compile.

The M6/W rows (N4, N5, S7, C5, W1–W3) are in §2.6 and `docs/stress-tests.md`; landing the two
branches together exposed one bug — `ModelsG`/`AdaptedRequestsG` laid requests out in `x` order
while the grouped `RequestsG` orders them by kind — fixed by taking the inner order everywhere.
Row IDs now follow the catalogue (variable-length generation is C1, sampling over public
randomness is C5).

### 3.6 GPT-2 Small on silicon (`e063ea6`, `docs/gpt2-silicon.md`)

The whole forward pass of GPT-2 Small (27-token prompt, 8 greedy tokens) run on an RTX 4090
through our own fixed-order kernels matches the pinned CPU model on **11 664 980 elements with 0
mismatches** across 363 tensors and every gate family (`tc_dot16`, `f32_add`/`sub`, `ln_*`,
`exp`, `gelu_tanh`, `f32_to_bf16`, argmax, `token_eq`), on arm64 and on the pod's x86, with HF
fp32's tokens. 37 225 VUs sampled by the framework (19 822 uniform via `derive_sample_selection`,
17 403 targeted at rare kinds; 3.5 M gates, 49 162 recorded outputs) re-execute with 0
disagreements. `run_protocol` on a one-token prefill at `q = 1`, `s = 1/500`:

| | 1 layer | 12 layers |
|---|---|---|
| gates / weights / VUs | 52.0 M / 46.5 M / 46.6 M | 135.2 M / 124.5 M / 124.8 M |
| interior positions committed | 5.46 M | 10.7 M |
| honest replay (pure Python) | 877 s | 1 699 s |
| prover / verifier | 748 s / 19 s | 1 443 s / 45 s |
| VUs sampled / openings / transcript | 93 409 / 273 099 / 470 MB | 249 669 / 896 685 / 1.60 GB |
| honest verdict; one flipped bit in a mix `f32_to_bf16` | accepted; `relation_rejected` | accepted; `relation_rejected` |

The pure-Python prover ran at ~135 µs per committed position and 160 µs per replayed gate on
the gate-granular interior; the VU-output interior (§2.8, landed after these measurements)
removes most of the committed positions. Pod cost $0.63
(50.8 min); no pod of ours is running. Landing this next to the proof layer surfaced one rule to
relax: proof statements required a gate set of one word width, which the pinned set (1-, 16- and
32-bit gates) is not; the statement's word is now the widest gate (`cdf85b0`; the transparent
backend decodes by leaf schema, a zkVM guest resolves its gate set by `(id, width)` and rejects
others). Open, per the report's §8: Hopper/Ampere and FP8 variants, fused-attention orders,
sampling instead of argmax, a `gather` gate for the embedding.

## 4. Flags for the architect

Decisions the implementation surfaced that belong in the paper.

1. **The restatement's α = 100 and U ≈ 60 Mbit are stale.** Measured α is 7·10⁷–1.8·10⁸ and the
   headline is terabytes (§3.1), which matches the abstract's order of magnitude. The restatement
   should carry the measured α and the formula in §3.1.

2. **Corollary 6.3 needs a grouping step that Theorem 6.2 does not supply.** The theorem's sum
   ranges over all error profiles, including those with `k_r > k⋆` errors in a replay unit; the
   corollary's `max` over `k ≤ k⋆` is only justified by covering every subset of `k⋆` or more
   erroneous VUs with the *replay unit's own* cover (one cover, `2^{W_R}`, cost at least
   `ε_θ(k⋆)` since `ε_θ` is increasing) and grouping profiles by `min(k_r, k⋆)`; a Chernoff step
   then gives `log₂|Y_λ| ≤ ρλ + log₂e`. The full derivation is the docstring of
   `analysis/rate.py`; the paper's ρ should carry its `log₂(k(k+1))` per-channel term and the
   `+ log₂ e`. Measured against the fold: within 0.4 % where the whole-RU channel binds, 17–27 %
   above it at tiny `s` (the fold saves about `log₂ B` bits per error when `B` errors are
   affordable, which the closed form gives up).

3. **Interior commitment is at VU-output granularity — built (§2.8).** Committing every gate
   value cost 56.8 µs per position and dominated the prover; the interior is now the VU outputs,
   a sampled VU opens its inputs and outputs and the verifier recomputes it, which is what the
   headline estimate assumed and what the proof layer verifies. The protocol section's interior
   commitment should be defined over VU outputs, and the security argument's local check should
   read "recompute the unit from its opened inputs and compare with its opened outputs".

4. **Public parameters gain `f_max`, and its price is not `f_max · u(1)`** (§2.6). Because the
   declaration follows the q-challenge, the capacity statement must charge the adaptive prover:
   `U_f ≤ min(U_0(η(1 − s)^f), U_0(η/(1 + |S|)^f) + f·u(1))`, which at serving replay rates is
   `≈ ρ f log₂(1/(1 − s))`, three orders above `u(1)` though still a small fraction of `U`. If
   the paper wants the `u(1)` price, declarations must precede the q-challenge, which limits them
   to faults detected without replay. `pp` should also be restructured: `q`, `s`, caps, `η`
   and `f_max` are the verifier's; the client proposes only θ.

5. **Output-determined advice is not free under the per-shape union bound.** A length, a
   route pattern or an acceptance count that is "readable off the outputs" still selects one of
   `#shapes` circuits, and the union over shapes adds `log₂ #shapes` — the same bits the advice
   would cost. A zero charge needs a lemma that embeds every shape in one padded circuit whose
   outputs determine the shape. Until it is proved, M3 is not implemented and S7 charges the
   stream length as advice.

6. **Outputs the verifier requires to equal a constant carry no capacity — built (§2.9).** The
   in-circuit check words of §2.4 are accepted only when they equal 1, so they contribute a
   factor 1 to `|Y|`, not `2^{width}`; the compiler's check outputs and `Bound` now price them
   at zero. The capacity theorem needs the one sentence, and the compilation section the
   `checks` field of a root definition.

7. **Gate semantics are pinned per silicon by reference implementation and golden vectors**
   (§2.3). The gate-set section should say so, and say that the "tensor-core dot" contract in the
   draft matches no shipping GPU.

8. **The bottleneck is `min(out, reach, ancestor)`** (§2.1), and the bound's headline is the
   closed form ρ. The reach theorem in the previous report's §4.4 gains the ancestor cut.

9. **Epoch versus session — built (§2.7); the vocabulary has to give.** The paper and
   `docs/stress-tests.md` use *epoch* for one protocol execution and *session* for the epochs
   between decisive rejections; the layer uses *epoch* for the year and *run* for the paper's
   epoch. Two caps do not compose across a round: `W_max` and `A` are per run (nothing caps the
   round's total verifier work; the epoch's capacity is `Σ_rounds Bound + Σ_runs |a|`, not
   `Bound + A`). `cost(union)` counts the model's weight commitment once per run.

10. **Advice charging — built (§2.9).** A constructor declares the exact bit length of its
    advice and the compiler checks the canonical encoding before charging it; the paper's `|a|`
    should be read as that bit length, with the canonical-encoding condition stated (otherwise
    the padding bits are a free channel). The schedule's per-join encoding still carries the
    output-determined length (flag 5).

## 5. Outstanding

- **Epoch layer, not built**: boundary storage and deterministic replay from a KV state, a wire
  format for `RoundChallenge`/`EpochReport`, a beacon binding for the `q` side, and rewiring the
  datacenter simulation onto the layer (the adversary test is dedicated).
- **Proof layer**: guest SHA over a precompile, vk caching, the OpenVM host binary (§3.2).
- **RunPod**: none of this work's pods is running (both 4090 pods were terminated, $0.63 each). A
  pod that is not ours, `porep-h100-b` at $3.29/h, was running on the account at 23:50 PDT with
  42 minutes of uptime; left alone.
- **Unfinished in the scenarios**: bounded expert capacity for MoE, multi-layer routing, workload
  size in the crossover sweep (the ladder stops at E = 16, where advice first wins).
- **Repository**: after the last merge the whole tree was `ruff format`ted at the default width
  (120 files, no semantic change) and CI now blocks on `ruff check`, `ruff format --check` and
  `mypy`; `docs/README.md` indexes the documents and how each is regenerated; the README
  describes the objects as they are (VU-output interior, recomputed units, check outputs, exact
  advice bits, `f_max`, the epoch layer). The tests over three seconds (the rate-versus-fold
  comparison at 13 s, the datacenter fixture at 12 s, the epoch adversary at 10 s) were left in
  the default run: they are the correctness checks of the bound, not perf tests.
