# Overnight report (2026-09-01 → 09-02)

Living document; the final section is written last. Everything referenced here is on `main` in
this repository; commit hashes are given where a decision landed.

## 1. Where the implementation stands

Thirty-two commits on `main` since last evening (`6d15b1f..HEAD`), about 70 files, +9 500 /
−1 200 lines; 806 tests, 90 s, ruff clean on `src/veritor` and `tests`. `src/veritor` is 9 900
lines in five packages:

| Package | Role | Trusted |
|---|---|---|
| `core/` | `GateSet` (modular word gates + `in`/`weight` sources; the toy ISA), `Description` (gate/call/repeat with run-typed relative arguments), `Circuit` (`FlatCircuit`, lazy `DescriptionCircuit`), `Index` (kinds, the two antichains, `inputs()/weights()/boundary()/interior()`), `Compiled`, `VerificationPolicy` | yes |
| `compile/` | `Compiler` (bytes → `Compiled` under `CompilationLimits`), `Constructor` protocol, `Compilation` = the record of one `Compile(G, x, a)` | yes |
| `protocol/` | messages (v6), canonical wire, Merkle commitments, sublinear challenges, `ProverSession`/`VerifierSession`, `verify_transcript`, `VerifierParameters` | yes |
| `analysis/` | `Bound` (knapsack + Laplace fold over kinds, integer count), `Cost`, `Optimize`, the exhaustive reference | yes |
| `constructors/` | `Tracer`, `DemoG`, `MatmulG`, the toy decoder `lm.py`, `Schedule`, `ClusterG` | **no** (client code) |

What runs end to end tonight: `Compile(G, x, a)` for `MatmulG`, `DemoG` and `ClusterG`; the
two-stage protocol at full and fractional policies with transcripts that round-trip through the
wire format and re-verify offline; `Bound`/`Cost`/`Optimize` on every compiled index; the
security suite (`tests/veritor/security/`, 117 tests) and the cluster protocol tests
(`tests/veritor/constructors/test_cluster_protocol.py`: honest runs accepted, an altered token
rejected at the boundary, a corrupted interior rejected at its relation, κ_W shared across
batch shapes).

Scale: a 1024³ matmul compiles in 18 ms from a 4 KB description (2.1·10⁹ gates); the per-kind
table costs 0.08 ms; `Bound` runs in 15–250 ms on the cluster indexes below; verifier time is
flat in the number of gates (`test_scaling.py`). The 328 641-gate cluster run (24 requests,
4 pods × 4 slots × 12 steps, 2 layers) compiles in 25 ms from a 420-byte schedule.

## 2. Decisions made tonight

### 2.1 Every gate is in a unit (b433dbd, 353b85c, 97a2c21)

The root definition has no ports. The circuit's inputs and weights are zero-arity *source
gates* (`in`, `weight`) in the public gate set, placed by descriptions like any other gate, so
they sit inside replay and verification units and the tiling check covers them. `x_k` is the
value at the `k`-th `in` gate in address order (`Index.inputs()` is an `O(depth)` rank/unrank
domain, as is `Index.weights()`). Ports survive only *inside* the hierarchy, between a
definition and its children, which is what keeps kind sharing.

Consequences accepted:

- A unit of source gates has `Out` width 0, so `Bound` gives it zero capacity without a new
  rule (a tightening is possible: a source-only kind can never hold an error, see §3).
- Outputs stay *declared* (an interface), not gates. The paper's `Out_1(6)` copy gate has no
  counterpart; the uniform statement is "every gate belongs to one replay unit and one
  verification unit; the outputs are declared positions among them".
- `∂ = In ∪ ⋃_r Out(R_r)` with `Out` excluding pinned gates, and `Int(r) = R_r \ Out(R_r) \
  pinned(R_r)`: source gates are never interior. κ_W covers the weight gates; the
  `exclude=range` carve-out of an input prefix is gone.
- A sampled `in` gate is compared with `x[rank]`; a sampled `weight` gate is accepted on its
  κ_W opening alone; the boundary phase still opens every `in` gate (Θ(|x|), inherent).
- Matmul layout: `[activations][weights][row × rows]`. The activation row is *not* inside its
  `row` unit: a row holding its `k` input gates plus its dots would put the batch's outputs on
  a 2-D grid and cost `min(rows, cols)` output runs; with pure rows they are one run. 1024³:
  4010-byte description, 9 definitions, 18 ms compile, 2.1·10⁹ gates, per-kind table 0.08 ms.
- Every source gate is by default its own verification unit (`Tracer.inputs(n)` emits one-gate
  cells), so the priced work `q s Σ_v (proof(V_v) + c_0)` gains `(|x| + |W|)(1 + c_0)`. A
  client that wants a smaller `W_max` price defines wider input units; nothing forces the
  one-gate cells.

### 2.2 `Bound` reports the log of an integer count (a3d8f0f)

`|Y_η|` is an integer, so `|Y_η| ≤ 2^b` implies `|Y_η| ≤ ⌊2^b⌋`. The fold's upward rounding
made a fully checked run bound to `~1e-14` bits and `U_max = 0` unsatisfiable; it is now
exactly `0.0`. Still an upper bound (the power is rounded up before the floor).

### 2.3 Interface resolution is bounded by what it produces (fd478dd)

Declared interfaces are runs `(start, count, stride, width)`. A strided output range over a
child's slots could cost time proportional to its element count (a 900-byte description made
the verifier's `parse` spend seconds). Slot-linear children and within-copy strides are now
one run in `O(1)`, and `CompilationLimits.max_output_runs` (256 per definition) and
`max_output_runs_total` (16 384) cap the pieces a definition may resolve to, with the bounded
generator stopping *at* the cap rather than after doing the work. The distinctness check
(declared outputs must be pairwise distinct) is `O(#runs²)` per definition under those caps.

### 2.4 `In` of a kind is its declared port count (e3850a5)

Admission pricing (`W_max`) uses the declared input interface of a verification kind, not the
set of inputs actually read; declared ≥ read, so the priced openings only grow, and it is
`O(1)` per kind. `Definition.reads` is only ever evaluated for *sampled* units, after the
work budget has admitted the run. A client can no longer make admission itself cost
`Θ(|x|)`.

### 2.5 κ_W commits the weight *vector*, per model, not per description (676f379)

Before tonight the weight domain was `I.weights()`, whose identity digest depends on the
compiled description, so κ_W would have changed with every request of a continual-batching
cluster (each request compiles a different circuit from the same model). Now the domain is the
rank space `0 .. |W|-1` bound to a fixed tag and the gate set digest: position `k` is the
`k`-th `weight` gate in address order of whichever circuit is verified. `commit_weights(gate_set,
values)` needs no circuit; a sampled weight gate at address `a` is opened at rank
`weight_rank(a)`. Headline test: one root is accepted for two batch shapes of the same model.

Security note: κ_W binds the gate set and the vector, not a model *name*. Which root a request
must run under is the verifier's choice in the `Expectation`; the verifier must have obtained
κ_W before the epoch's first request (a deployment rule, not enforced in code).

### 2.6 Advice is structural, and the verifier runs `G` (9641d18, 77a89a6)

The question was whether advice should be an opaque string that only shapes the circuit, or
should also flow into specific units as values. Decision: **structural only**, as the paper's
`Compile(G, x, a)` has it. Reasons: (i) every use of advice we could name for an inference
cluster (which pod, which slot, which step a request joins; batch size per step; context
length per slot) is a structural choice, and the schedule module encodes exactly those choices
as canonical bytes; (ii) charging `8·|a|` bits per request is the whole accounting, with no
per-unit bookkeeping and no interaction with `Bound`; (iii) value-level advice is a clean later
extension (a third source-gate class, charged by width) if a use case appears.

What landed: a constructor is any object with a `digest` and `G(x, a) -> (description bytes,
flat inputs)`. `research.Compile(G, x, a, gate_set, *, limits, max_advice_bits)` checks
`8·|a| ≤ max_advice_bits`, runs `G` (any exception, including `SystemExit`, is a
`CompileError`: the client's constructor failed, which is a rejection), compiles the
description, and returns a `Compilation(compiled, constructor digest, inputs, advice)`. The
header binds `G.digest` and `a` (protocol v6); admission rejects advice over the verifier's
`max_advice_bits`; `Capacity(compilation, θ, η) = Bound + 8·|a|`. With `U_max` enforced at
admission and `max_advice_bits = A`, every accepted request has capacity ≤ `U_max + A`.

Trust model, stated in the docstring: in this prototype the verifier executes `G` as ordinary
Python identified by a versioned digest (public code both parties hold, like the gate set); a
deployment runs it sandboxed and metered, or has the client prove `Compile(G, x, a) = (C, I)`
(paper §7). The compilation limits bound `G`'s *output*; nothing bounds its running time.

Two knobs exist for one bound (`Compile(max_advice_bits=)` pre-checks before running `G`;
`VerifierParameters.max_advice_bits` is the authoritative admission check that also runs on
the transcript path). Keep them equal; a mismatch only causes rejections, never acceptances.

### 2.7 `U_max` has no default (afa92d4)

The security pass found that `VerifierParameters()` defaulted `max_capacity` to `None` (waive)
and `make_expectation` defaulted to `VerifierParameters()`, so a verifier that never thought
about capacity admitted `θ = (0, 0)` and accepted any claimed output with nothing sampled.
`max_capacity` is now a required keyword (`None` still waives, but has to be written) and
`make_expectation` / `make_verification_expectation` require the verifier's parameters. Test
fixtures state `VerifierParameters(max_capacity=None)` explicitly; the README example states
`max_capacity=0` for its fully checked run.

### 2.8 The inference cluster (6d15b1f, 58c7a71, b910bc8, 998828d, ab86f4e)

The question was what "an inference data center with continual batching" is as a circuit, and
what role advice plays. What landed:

- **Toy ISA** `make_isa_gate_set(B)`: `add`, `sub`, `mul`, `lt`, `eq`, `shr` over `Z_{2^B}`
  beside the sources. Nothing that approximates real arithmetic; the grammar has no immediates,
  so every constant is a `weight` (the one-hot table, shift amounts, the argmax seed).
- **Toy decoder** (`constructors/lm.py`): `LMShape(vocab, d_model, heads, layers, context,
  width)`, a `Parameters` vector, a sequential reference `reference_generate`, and a tracer
  program: one-hot embedding, per layer `q/k/v` matvecs, attention over a KV cache with the
  polynomial softmax `w_j = s_j²` and a right shift, an MLP with squares, an argmax chain.
  Faithful *structure*; the numerics are a toy and produce garbage tokens on purpose.
- **Schedule as advice** (`constructors/schedule.py`): a run is `pods × slots × steps`; the
  advice is the sorted list of *joins* `(pod, step, slot, request)`, 16 bytes each, in canonical
  bytes with a magic prefix; occupancy is derived (a join holds its slot until the next join in
  that slot, the request's `max_new`, or the end of the run), so a schedule can never
  double-book and needs no leave events. `schedule_fcfs` is one client-side scheduler.
- **`ClusterG(shape, pods, slots, steps)`**: `x` is the tuple of `Request`s, `a` the schedule.
  The root calls the `weights` unit once and then one `step` per occupied `(pod, step)`. A step
  is a replay unit ("replay decode step *t* of pod *p*"); its declared outputs are the new
  KV-cache rows and the tokens the next step reads through ports, so the KV cache is exactly
  the cross-step state and the boundary. Verification units are row-sized kinds (`dot_k`,
  `onehot`, `attend_head_c`, `argmax`, residual and square cells). Two steps with the same
  occupant shapes are one kind, so the description is `O(distinct prompt lengths + distinct
  context lengths + distinct step shapes + |schedule|)`: 34.5 KB for the default fixture.
- **Batching is semantically transparent**: for every schedule the circuit's outputs equal
  `reference_generate` per request (tested over FCFS, hand-written reassignments, two pods,
  two layers). This is the property that makes the schedule *pure* advice: it changes the
  circuit, never the function.

Role of advice, settled by building it: the schedule is the only thing about the circuit that
`(G, x)` does not fix, it is structural, and it is small (100 B for 4 requests, 420 B for 24).
`Capacity = Bound + 8|a|` charges it honestly. Nothing in the cluster wanted value-level
advice.

Tracer friction met while writing it, left as is (each is a small change with churn risk):
no 2-D ranges (a row × column interface is written as a Python loop of 1-D ranges); no
primitive for concatenating results from several children into one declared output (one has to
name the runs); `Wire` vs `Wires` for single-output definitions.

## 3. Security argument

The full argument is `docs/security-argument.md` (one section per component: claim, argument
with code citations, attack test, gaps; a findings list; a verdict table). The attack tests are
`tests/veritor/security/` (117 tests, ~5 s). Summary:

| Component | Verdict |
|---|---|
| Position binding and domain separation (Merkle leaves bind domain/rank/position/schema/value; domains bind phase digest/owner/position set/count; one owner per address) | proved + tested |
| Staged commitments: `J` from `(q_seed, header, κ_∂)` with the seed released only after the boundary; `T` from `(s_seed, …, κ_r)`; both state machines fail closed | proved + tested |
| Seed freshness | operational requirement; negative test shows reuse is fatal |
| Sampling law: independent Bernoulli coins within 2⁻¹⁹⁰; acceptance rate = σ(E) | proved + tested (4σ, 2000 trials) |
| Local checks: every gate of a sampled unit against owners' values; inputs exhaustive at the boundary; weights only via κ_W; canonical encodings; exact coverage | proved + tested |
| P[accept] ≤ σ(E*) for every strategy | reduction written out; statistical evidence |
| `Bound ≥ log₂|Y_η|` under the union definition; every approximation rounds toward admitting more | proved step by step against the code; exhaustive reference on random circuits with random client-chosen marks |
| Admission: η verifier-owned and in the header; denominators capped; `U_max`/`W_max` before any commitment | proved + tested; F2 fixed |
| Compile determinism; canonical bytes rejected first; bounded compile work; marks in the digest | proved + tested |
| Tiling and refinement; cross-unit reads only through declared outputs | proved + tested (the cross-cut read is unconstructible through the compiler; tested on a forged `Compiled`) |
| Offline transcript verification recomputes every challenge | proved + tested (30 field mutations, each with its verdict code) |

Findings and what was done:

- **F1 seed reuse** (medium, operational): documented; `make_expectation` draws fresh seeds from
  the CSPRNG; a deployment should derive `HMAC(master, session_id ‖ tag)` so freshness reduces
  to session-id uniqueness.
- **F2 default `U_max = None`** (medium): fixed (§2.7).
- **F3 `_integer_count` could undercount by one ulp**: fixed (scale by `1 + 2⁻⁴⁵` before the
  floor; `log2` rounded up unless exact).
- **F4 source-only kinds counted with `l ≥ 1`** (looseness): fixed. A verification unit made only
  of source gates can never hold an error (an `in` gate must equal the header's input, checked
  for every input at the boundary; a `weight` gate's only admissible value is its κ_W leaf), so
  for any admissible `E` containing such a unit `V`, `outputs(E) = outputs(E \ V)` and
  `σ(E \ V) ≥ σ(E)`; dropping those `E` leaves the union unchanged and only removes terms.
- **F5 κ_W is not self-describing** (by design): the root binds the gate set and the vector,
  not a model name; the header binds it to the run.
- **F6** `VerificationLimits.max_nesting_depth` / `max_artifact_bytes` declared but unused.
- **F7 output over-resolution** (medium, fixed ba43d29, 3fcbebc): the within-copy branch of the
  run resolver added for the fan-out bound (fd478dd) did not clamp to the declared count, so a
  strided declaration inside one copy resolved to every slot of the copy at that stride. Found
  at 03:30 by compiling cluster shapes with mixed lifetimes in one pod: 480 of 2 325 small shapes
  were rejected by the distinctness rule; shapes where the extra gates were distinct compiled
  with an inflated `Out` (boundary gained undeclared positions, `Bound` looser, `Cost` off; the
  per-ordinal resolver that places values was right throughout, so no wrong value could be
  opened). Fixed; the compiler now checks resolved positions against `output_count`; a 400-case
  fuzzer compares the run resolver with the per-ordinal one on strided declarations over
  permuted-output children. Lesson: fuzz every run-typed summary against its per-element path.

Not achieved, and said so: no sandbox for `G` (only output limits; `SystemExit` is caught,
`os._exit` is not); κ_W provenance and timing are the operator's; retries must be bounded
outside the protocol; no privacy (values are opened in the clear); SHA-256 collision resistance
and HMAC-SHA256 as a PRF are assumed; the float rounding discipline of `Bound` has explicit
margins but no machine-checked end-to-end error budget.

## 4. Discrepancies with the paper outline, with recommendations

1. **Inputs and weights are gates inside units; the root has no ports.** The outline's circuit
   has `In` positions outside the units and `Out` copy gates. Recommend the outline adopt the
   implemented statement: *every gate, source gates included, lies in exactly one replay unit
   and one verification unit; `In` is the set of `in` gates, `Out` a set of declared positions;
   `∂ = In ∪ ⋃_r Out(R_r)`, `Int(r) = R_r \ Out(R_r) \ pinned(R_r)`.* It removes a case from
   every proof ("positions not in any unit") and the `Out` copy gates from the figures.

2. **Weights are neither inputs nor part of `G`.** They are committed once per model under
   `κ_W`, a root over the weight *vector* by rank bound to the gate set (§2.5). The outline
   folds weights into `x` (Θ(|W|) per request at the boundary) or into `G` (a `G` per model
   epoch). Recommend a sentence in §4: "model weights are a third class of source gate, opened
   against a per-epoch commitment; they are never boundary positions."

3. **`Bound` is the union over admissible error sets, not `max_E`.** The audit found the earlier
   "per-strategy" reading; the fold now certifies `|⋃_{E admissible} outputs(E)|` (§3). The
   outline's headline numbers were computed under the per-strategy definition and should be
   recomputed; expect them to grow by at most the log of the number of admissible covers.

4. **Cut width vs. output reach (the important one, found tonight, §5.1).** The outline's
   downstream-cut argument bounds the outputs reachable from an error set `S` by
   `2^{out_bits(S)}`. For a state-carrying replay unit (a decode step whose declared outputs
   are its KV-cache rows), `out_bits(S)` is the *state* width, which dwarfs the circuit's
   outputs. Measured on the default cluster fixture: unit cut widths 144–672 bits against true
   output reaches of 16–80 bits, and the circuit has 160 output bits in total. Consequence:
   for any `q < 1 − η` an adversary may corrupt one whole replay unit with probability
   `1 − q > η`, so `Bound ≥ min(out_bits(C), max_r out_bits(R_r))`, which for a cluster is
   `out_bits(C)`: the bound is vacuous at every fractional policy (table in §5.1). The fix is a
   theorem the outline should state and the fold should use: with errors confined to
   `S_1 ∪ … ∪ S_m`, `|reachable outputs| ≤ Π_j 2^{min(out_bits(S_j), reach_bits(S_j))}` where
   `reach_bits(S)` is the width of the circuit outputs downstream of `S`. Proof: fix all cuts
   but `c_j`; varying `c_j` changes outputs only inside `reach(S_j)` and through at most
   `2^{out_bits(S_j)}` cut values, so the image grows by at most `2^{min(·,·)}` per node;
   induct on `j`. For a decode step, `reach` is the tokens its pod emits from step `t` on:
   kilobits, against a cut of `10⁸–10⁹` bits for a real model. This is what makes the cluster
   application certify anything at `q < 1`.

5. **Advice is charged in bits of its canonical encoding, before `Bound`, at admission.** The
   outline charges `|a|` in the capacity statement but does not say where it is enforced; the
   implementation enforces `8|a| ≤ A` at admission and binds `a` in the header (§2.6). Recommend
   the theorem read "every accepted run has capacity `≤ U_max + A`".

6. **`η` is the verifier's and travels in the header; `θ = (q, s)` is the client's proposal.**
   The outline writes `θ = (q, s, η)`. Recommend the split; a client-chosen `η` is a soundness
   hole, not a parameter.

7. **`Bound` reports `log₂` of an integer count** (§2.2), so a fully checked run has capacity
   exactly 0 and `U_max = 0` is a satisfiable statement. Worth one sentence.

8. **Source-only verification units hold no errors** (F4). The outline treats all units alike in
   the knapsack; the tightening is free and the proof is three lines (§3).

9. **The verifier runs `G` in this prototype.** §7 of the paper (secure circuit compilation) is
   the deployment answer; the outline should say the prototype's trust model plainly (public
   versioned `G`, output-bounded, not time-bounded).

10. **Value semantics remain deferred.** Everything is modular word arithmetic; the LLM
    constructors that need fixed-point or float semantics wait on that decision.

## 5. Outstanding

### 5.1 `Bound` must use output reach, not only cut width (blocking for the cluster story)

Measured tonight (`Bound`, `Capacity = Bound + 8|a|`, `Cost` per request with `h = 1`, `c_0 = 0`):

Default fixture: 6 564 gates, 8 replay units, 1 065 verification units, 42 kinds, `|∂| = 139`,
160 output bits, advice 100 B.

| q | s | η | Bound (bits) | Capacity | Cost / request |
|---|---|---|---|---|---|
| 1 | 1 | any | 0 | 800 | 26 021 |
| 1/2 | 1 | 10⁻² … 10⁻⁶ | 160 | 960 | 13 080 |
| 1/2 | 1/2 | " | 160 | 960 | 10 596 |
| 1/4 | 1/2 | " | 160 | 960 | 5 368 |
| 1/8 | 1/4 | " | 160 | 960 | 2 443 |

Larger run: 328 641 gates, 30 replay units, 25 465 verification units, 70 kinds, `|∂| = 3 984`,
1 552 output bits, advice 420 B: Bound is 0 at `θ = (1, 1)` and exactly 1 552 (the cap) at every
fractional policy, `10⁻²` or `10⁻⁶`. The uncapped fold reports 2 080 bits on the small fixture:
the sum of all eight cut widths, because at `η = 10⁻⁶` and `q = 1/2` all eight units may be
corrupted at once (`2⁻⁸ > 10⁻⁶`).

Exact per-unit reach on the default fixture (BFS on the flat circuit):

| unit | gates | `out_bits(R_r)` | reach bits |
|---|---|---|---|
| 0 (weights) | 201 | 0 | 160 |
| 1 | 1 574 | 544 | 80 |
| 2 | 930 | 288 | 48 |
| 3 | 497 | 144 | 16 |
| 4 | 1 919 | 672 | 80 |
| 5 | 465 | 144 | 48 |
| 6 | 481 | 144 | 32 |
| 7 | 497 | 144 | 16 |

What the reach factor buys, estimated offline (the fold rerun with `out_bits := min(out_bits,
reach_bits)` per kind, reach as the max over copies from the BFS oracle; `Bound` itself is
unchanged): on a run of 64 one-token requests over 8 pods × 4 slots × 8 steps (56 600 gates,
41 replay units, 2 032 output bits) every step kind has cut width 576 or 288 bits against a
reach of 16–144 bits, and at `η = 10⁻²`:

| q | s | cut `Bound` | reach `Bound` |
|---|---|---|---|
| 1/2 | 1 | 2 032 (cap) | 869 |
| 1/2 | 1/2 | 2 032 (cap) | 1 152 |
| 1/4 | 1/2 | 2 032 (cap) | 2 032 (cap) |

At `η = 10⁻⁶` both are capped: 41 units is still too few (see (a) below). The per-unit factor
is 4–7× tighter; in a real cluster it is the ratio of KV-state width to downstream tokens,
several orders of magnitude.

Two things are true at once. (a) Eight or thirty replay units is far below the regime where
sampling certifies anything: with `Λ = ln(1/η)` and a whole-unit cost of `−ln(1 − q)`, the
adversary affords `Λ / −ln(1−q)` whole units (20 at `q = 1/2`, `η = 10⁻⁶`), so a meaningful bound
needs thousands of replay units each with an interface small against the output. That is the
paper's regime and the toy cannot reach it. (b) Independently of scale, cut width is the wrong
quantity for state-carrying units; a real decode step's KV rows are `10⁸–10⁹` bits while its
downstream tokens are kilobits. Recommendation, in order:

1. State and prove the reach theorem (§4.4) in the paper.
2. Implement `reach_bits` per kind structurally: a reverse dependency pass over the hierarchy
   at `O(|description|)` (per definition: which declared outputs depend on which step, as run
   bitsets; compose upward; take the max over copies of a kind, which is sound). Add it to the
   fold as `min(out_bits, reach_bits)` per node. Validate against the BFS oracle above on the
   cluster fixtures (`tests/veritor/analysis/` already compares the fold to exhaustive
   references).
3. Recompute the paper's headline numbers with reach and with the union definition together.

### 5.2 Also outstanding

- **Value semantics** (fixed-point vs. float, and what "correct" means for a float gate) before
  GPT-2 / Kimi-K3 / DeepSeek-V4-Pro / Inkling constructors; the ISA is the placeholder.
- **Phase 3 description features**: canonical chunking of long step lists; parametric
  descriptions (shape / advice-bound integers) if MoE routing is to be advice rather than
  padding. The cluster did not need either.
- **Sandboxing `G`**, or the client-side compilation proof of §7. The prototype bounds `G`'s
  output, not its time; `os._exit` escapes the `SystemExit` catch.
- **Seed derivation as `HMAC(master, session_id ‖ tag)`** so seed freshness reduces to
  session-id uniqueness (F1); retry budgets outside the protocol.
- **`VerificationLimits.max_nesting_depth` / `max_artifact_bytes`** are declared and unused
  (F6): wire them or delete them.
- **Two knobs for `A`** (`Compile(max_advice_bits=)` and `VerifierParameters.max_advice_bits`):
  keep equal or collapse to one (§2.6).
- **κ_W provenance**: which root a request runs under, and that the verifier held it before the
  epoch, are deployment rules; the code binds the root in the `Expectation` only.
- **Tracer ergonomics** (§2.8): 2-D ranges, a concatenation primitive, `Wire`/`Wires`.
- **Paper §7 draft** (`docs/paper/section-7-secure-circuit-compilation.md`) is committed as a
  draft and awaits the Thaler-style revision pass; the Notion outline was unreachable from the
  integration last night.
- **`W_max` price of one-gate source cells**: `(|x| + |W|)(1 + c_0)` proof cost at `q s`; a
  client defines wider input units to pay less (§2.1). Decide whether the paper's `Cost` should
  charge source gates at all.
