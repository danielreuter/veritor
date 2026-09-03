# Stress tests: what a real datacenter throws at the compiler and the protocol

Not a demo. A catalogue of things that happen in a production inference
datacenter, each turned into an executable scenario that must compile,
run through the protocol, and be priced. The end product is the table in
the last section, generated from the scenario results: what happened, how
the protocol absorbed it, and what it cost (advice bits, capacity, prover
overhead, description size). Where a scenario does not fit the compiler
today, that is the finding; the fix goes in with the scenario's test.

Terminology (paper): *epoch* = one execution of the protocol (compile,
commit, challenge, replay, challenge, evidence, verdict); *session* = the
epochs between decisive rejections; `U`, advice charging and the fault cap
are per session. RU = replay unit, VU = verification unit.

## 1. Mechanisms: the five ways a complexity can enter

Every real-world complexity changes *which circuit was run*, *which values
it holds*, or *nothing*. There are exactly five admissible ways to account
for it, and the table should name one per row.

| # | Mechanism | Charge | When it applies |
|---|---|---|---|
| M1 | **Invariance.** The circuit does not depend on it. | 0 | RU = request and kernels are batch-invariant and fixed-order: batching, arrival timing, placement, preemption-by-recompute, retries, disaggregation, chunked prefill all vanish from the circuit. Only the kernel's throughput loss is paid. |
| M2 | **Public configuration.** Fixed by the deployment, known to the verifier, part of `G` or `pp`. | 0 | Model version and weight root, tensor-parallel degree and its reduction tree, GPU architecture per pod (gate set), kernel tiling, quantization scheme, EOS id, `max_tokens`. |
| M3 | **Output-determined advice.** A structural choice that is a verifier-checkable function of `(x, y)`. | 0 | Generated length: the circuit has `len(y)` steps and the verifier checks the last token is EOS or `len(y) = max_tokens`. Anything the observed output already reveals is free advice, provided the verifier checks the rule. |
| M4 | **Charged compile-time advice `a`.** Names the circuit; charged `|a|` bits at admission (`A = max`). | `|a|` | Structure not determined by `(x, y)` and not decided in-circuit: MoE routing given as advice, speculative-decoding acceptance pattern given as advice, schedule when RU = step. |
| M5 | **In-circuit decision (padding).** The choice is computed by gates and the structure is padded to its maximum. | wasted gates | MoE top-k with capacity-padded experts, speculative acceptance with masked candidate positions, constrained-decoding masks, token sampling from published randomness. Trades advice bits for prover cost; both must be measured. |
| M6 | **Replay-time advice (fault declarations).** After the challenge, the prover names up to `f_max` VUs it does not claim correct; the verifier skips their relation. | `f_max · u(1)`, `u(1) = W_V + log2 |S|` | Silent data corruption, bit flips, a GPU that produced a wrong token. Completeness becomes exact for any honest prover with at most `f_max` undetected faults per session. |
| M7 | **Published randomness.** Sampling randomness is part of `x`. | 0 | Temperature/top-p sampling, dropout-free inference has no other randomness. The server never chooses its own randomness. |
| M8 | **Separate commitments.** Weights, adapters and model versions each under their own root, referenced publicly per request. | 0 | Multiple models, hot swaps, LoRA multi-tenancy. |

M3 is new relative to the current code: `Compilation.advice_bits` charges
`8·len(a)` unconditionally. The rule becomes: charged bits = bits of `a`
not fixed by a verifier-checked function of `(x, y, pp)`. M6 is
implemented (below): the interior message carries the declarations and
`Bound` has the `f_max` term.

### M6: fault declarations

*Why.* A server learns of a silent hardware fault only when it replays an
opened RU and the recomputed values disagree with what it streamed (or when
it re-executes and gets a different answer). Without declarations every
incorrect VU that is sampled is a rejection, so an honest server with a
realistic SDC rate is eventually rejected: Llama-3 405B saw 6 SDC events in
54 days on 16,384 GPUs, about `2.8e-7` per device-hour
(`veritor.simulation.faults.SDC_RATE_PER_DEVICE_HOUR`); a 16,384-GPU fleet
accumulates a mean of `0.0046` faults per hour and `0.11` per day. The
`f_max` a verifier admits is the smallest count whose Poisson tail is below
a target (`fault_budget`): `f_max = 2` for an hour and `4` for a day of
that fleet at tail `1e-6`. Exceeding it is one rejected session and a
retry, not a soundness event.

*Protocol.* `VerifierParameters.max_faults` (`f_max`, default 0) is bound
into the `Header` (manifest, digest and wire omit it when 0, so every
existing transcript is byte-identical). After the q-challenge the prover's
`InteriorMessage` carries `declarations`: a sorted tuple of global VU
indices, no duplicates, each inside an opened RU, each a VU with a relation
(not source gates only), at most `f_max` of them; the verifier rejects with
`FAULTS_EXCEEDED` or `FAULT_DECLARATION_INVALID` before deriving the
s-challenge, whose seed binds the declarations through the interior phase.
A sampled VU that is declared is obliged under `DECLARED_PROGRAM` -- a kind
of zero gates and ports over the same positions -- so its openings are
authenticated and its relation check is skipped, in the transparent path
and in every proof backend alike; both parties derive this from the
challenge and the declarations, never from the prover's statement. VUs
that read the declared value are checked against it as usual. The honest
server's detector is `self_check` (recompute every gate of an opened RU
from the values it holds; the VUs whose own gate disagrees are the faults)
and `honest_declare` wraps it.

*Charge.* Let `D` be the declared set, `E` the error set and `E' = E \ D`
what stays exposed. The output is a function of the correct computation
outside `E`, the values at `E'` (which must survive the sampling as before)
and the values at `D`. Enumerating `D` and its values,
`|Y_eta| <= sum_{|D| <= f} 2^kappa(D) · sum_{E' admissible} 2^kappa(E')
<= (1 + |S| 2^W_V)^f · 2^U_0`, using `kappa(D) <= sum_{v in D} kappa(v)`
(the union of downstream cuts is a downstream cut) and
`sum_{j <= f} C(|S|, j) 2^(j W_V) <= (1 + |S| 2^W_V)^f`. One declared fault
therefore costs `u(1) = log2(1 + |S| 2^W_V) = W_V + log2(|S| + 2^-W_V)`,
which is `W_V + log2 |S|` to within `2^-W_V`; `|S|` counts the VUs and
`W_V` is the widest cut over the VU kinds that have a relation.
`fault_allowance_bits(target, f_max) = f_max · u(1)` and `bound(...,
max_faults=f_max)` adds it (capped by the interface). This is value-level
advice: compile-time advice names the circuit and is charged by how many
circuits it can name; a declaration names a VU (`log2 |S|` bits) and its
contents (`W_V` bits).

*Numbers.* Small simulation (2 pods × 16 steps, 12 requests, 30 streamed
tokens, 26,314 gates): `|S| = 3,791`, `W_V = 128` (the `onehot` VU's eight
16-bit outputs), `u(1) = 139.9` bits; `Bound` at `theta = (1/2, 1/8)` is
capped at `out_bits = 480` with or without the term, and at `theta = (1,
1)` it goes from `0` to `139.9`. One declared fault is thus 29% of that
run's entire output -- the mechanism is priced for serving-scale runs.
GPT-2 Small's kind table (`GPT2G`, one request of 64 prompt + 64 new
tokens): `|S| = 161.3 M` VUs and `W_V = 32`, because the LayerNorm
statistics and the accumulator-width cells are 32-bit VU outputs, giving
`u(1) = 59.3` bits; under the 16-bit-word assumption `W_V = 16` and
`|S| = 176.8 M` the figure is `43.4` bits. Either way a fault costs about
three to four tokens' worth of capacity.

*Adaptivity (for the architect).* The charge prices a `D` fixed before the
q-challenge (or `q = 1`). The protocol lets the prover choose `D` after
seeing `J` -- an honest server cannot know its faults before it replays
the opened RUs -- and an adversary can use that: corrupt one VU in each of
many RUs and pardon whichever `f` of them were opened. With `N_J`
corrupted VUs in the opened RUs the acceptance probability is
`(1 - s)^max(0, N_J - f)` instead of `(1 - s)^N_J`, a factor `1/(1 - s)`
per declaration, worth about `1/q` extra corrupted VUs when `s` is small.
Since `sigma_f(E) <= (1 - s)^-f · sigma_0(E)`, the fold at the lowered
threshold `eta (1 - s)^f` is a rigorous upper bound on the adaptive
capacity for `s < 1` (`veritor.analysis.faults.adaptive_fault_bound`); it
coincides with `f · u(1)` in order of magnitude only when `q` is near one.
`bound()` charges the specified `f · u(1)`; whether to charge the adaptive
figure instead, or to make the declaration non-adaptive (declare before the
q-challenge, which an honest server can do only if it self-checks every RU
it ran), is a decision for the architect.

## 2. Scenarios

Columns: what happens; mechanism; what the compiler/protocol must be able
to express; the hypothesised gap (H) and its fix; the test.

### 2.1 Scheduling and timing

| ID | Scenario | Mech. | Gap / fix | Test |
|---|---|---|---|---|
| S1 | Continual batching: occupancy changes every step | M1 (RU = request) or M4 (RU = step, schedule as advice) | none; both constructors exist (`RequestsG`, `ClusterG`) | same requests under three schedules: identical outputs, identical request kinds; advice 0 vs `|schedule|` bits; `U` and overhead for both |
| S2 | Poisson arrivals, FCFS, variable prompt and generation lengths | M1 + M3 | none | 10^3–10^6 requests with random lengths: description bytes, compile time, `Out` runs; H: `max_output_runs = 256` on the root is hit at 257 distinct shapes → group by shape with a two-level `repeat` or raise the limit with an `O(#runs)` verifier cost argument |
| S3 | Preemption by recompute (evicted request re-prefilled later) | M1 | with RU = request the circuit is unchanged; with RU = step a request must join twice → `Schedule` re-join (in progress) | a request evicted at token *t* and restarted: outputs equal reference; both RU choices |
| S4 | Preemption by swap (KV retained across a gap) | M1 | RU = request: unchanged. RU = step: occupancy must allow gaps; H: `Schedule` derives contiguous occupancy → add explicit leave/resume or accept M1 only | gap of *g* steps; ports read KV declared *g* steps earlier |
| S5 | Chunked prefill (long prompt over several steps) | M1 | numerically identical with fixed-order kernels; RU = request unchanged | prompt of 4 chunks vs 1: same request kind digest and values |
| S6 | Prefill/decode disaggregation (KV moves between pods) | M1 (request) / cross-pod ports (step) | H: `ClusterG` assumes pods share nothing but weights → allow a decode step to read a prefill step on another pod | one request prefilled on pod 0, decoded on pod 1 |
| S7 | Client disconnect / timeout truncates generation | M3 | the observed length is not EOS-terminated: the verifier must accept a *truncation* code as public `x`-side data (client action) — the client, not the server, declares it | truncated at *t* with client attestation; without it, rejected |
| S8 | Retries and duplicate execution (one result discarded) | M1 | discarded work has no observed output and is not in the circuit; document the principle | duplicate run; circuit unchanged |

### 2.2 Data-dependent control flow

| ID | Scenario | Mech. | Gap / fix | Test |
|---|---|---|---|---|
| C1 | EOS-terminated generation | M3 | H: `advice_bits` charges the length; add the verifier rule (last token EOS or `max_tokens`) and charge 0 | 1000 requests: charged advice 0; a server that stops before EOS without EOS is rejected |
| C2 | MoE routing (top-k of E experts per token) | M4 vs M5 | H: no in-circuit routing; build both: advice route (`log2 C(E,k)` bits/token) and padded route (top-k gates, capacity factor, masked experts) | per-token bits vs padded gate overhead; `U` for both; crossover |
| C3 | Speculative decoding (draft proposes *k*, target accepts *j ≤ k*) | M5 or M4 | H: dependent reads (next positions depend on *j*) → masked candidate positions (tree attention with validity masks) or acceptance counts as advice (`log2(k+1)` bits/step); the draft model's logits enter the sampling relation with speculative sampling | both routes on the toy LM with a toy draft; bits vs waste |
| C4 | Constrained decoding (grammar/JSON masks, banned tokens, repetition penalty) | M5 | masks are functions of public `x` and prior tokens → in-circuit mask gates before argmax/sample | grammar mask on the toy vocabulary; outputs equal reference |
| C5 | Sampling with temperature/top-p and published randomness | M7 + M5 | `sample` kind (in progress) | randomness in `x`; server-chosen randomness impossible by construction |
| C6 | Multi-turn continuation (re-prefill vs cached prefix) | M1 / M3 | with a cached prefix the request reads another request's KV: H: cross-request reads require the prefix to be an RU whose declared outputs are its KV → `W_R` blows up for prefix RUs; alternative: recompute the prefix per request (M1, wasteful) | shared system prompt across *n* requests: `W_R`, `U`, overhead for prefix-RU vs recompute |

### 2.3 Numerics and hardware

| ID | Scenario | Mech. | Gap / fix | Test |
|---|---|---|---|---|
| N1 | Nondeterministic kernel selection / split-K / atomics | M1 by fixed-order kernels | hardware track: fixed-order GEMM vs cuBLAS throughput | (GPU) bit-exact chain at GPT-2 shapes; cost |
| N2 | Tensor parallelism changes the reduction tree | M2 | H: constructors have no TP; the circuit for degree *d* has *d* partial chains + an fp32 reduction in a fixed order | toy matvec at TP = 1, 2, 4: distinct kinds, `U` unchanged |
| N3 | Heterogeneous fleet (A100 and H100 pods) | M2 | H: one gate set per compilation → namespaced gates (`tc_dot16@sm80`, `tc_dot16@sm90`) in one `GateSet`; pod → arch is public placement | two pods, two gate sets, one circuit |
| N4 | Silent data corruption / bit flip in one dot product | M6 | H: no declaration message; add `Declare(VU ids)` after the first challenge, cap `f_max ∈ pp`, `Bound += f_max · u(1)` | inject one fault into a sampled RU's replay: declared → accept; undeclared → detected with the predicted probability |
| N5 | A GPU produces a wrong token that was streamed | M6 | one declaration for the token's VU, everything downstream replayed honestly from it | fault at token *t*; outputs stand; accept with one declaration |
| N6 | Node failure mid-request, restart on another pod | M1 / M4 | streamed tokens stand; restart is a second join (in progress) | (demo agent) |
| N7 | Dynamic activation quantization (per-block scales computed at runtime) | M5 | scales are gates (`max`, `recip`) feeding the dot inputs | toy: scale gates in the VU table; `W_V` unchanged |
| N8 | Fused attention (online softmax) reduction order | M2 | the fused kernel's order is the circuit's order; the gate set must carry the exact exp/max/rescale sequence | (GPU track) |

### 2.4 Weights, models, tenants

| ID | Scenario | Mech. | Gap / fix | Test |
|---|---|---|---|---|
| W1 | Model version hot swap mid-epoch | M8 | H: one weight root per compilation → a set of roots keyed by model id, referenced per request | two versions in one epoch; each request's weights under its root |
| W2 | LoRA adapters per tenant | M8 | adapter weights under their own root; the request references base + adapter | toy adapter on one layer |
| W3 | Multiple models served (router picks a model) | M2 / M4 | model choice is public per request (client asked for it) or advice if the server chose | both |

### 2.5 Scale and the epoch

| ID | Scenario | Mech. | Gap / fix | Test |
|---|---|---|---|---|
| E1 | A day of traffic: 10^6 requests, 10^3 distinct shapes | — | description = shapes × repeats; H: compiler limits (`max_output_runs`, `max_definitions`) | bytes, compile time, kind table time at 10^4, 10^5, 10^6 requests |
| E2 | Many epochs per session; exposure aggregates | — | H: protocol has one epoch → session state (declaration count, advice bits, `U` accounting across epochs) | 100 epochs; cumulative `U`, `f_max` enforcement |
| E3 | Verifier work at scale | — | benchmarks branch | (benchmarks) |

## 3. Order of work

1. Land the in-flight branches (demo → schedule re-join, EOS, failures, `sample`; ZK backend; rate/ancestors).
2. Framework: `tests/veritor/stress/` with a `scenario(...)` fixture that records one row (ID, mechanism, advice bits charged, `U` at `λ = 40`, overhead, description bytes, verdict) to `docs/data/stress.json`; `python -m veritor.stress.report` renders section 4.
3. Mechanisms first, scenarios second: M3 (output-determined advice) and M6 (declarations) are protocol changes and unblock C1, S7, N4, N5; the gate-set union unblocks N3; the weight-root set unblocks W1–W3.
4. Then the expressiveness scenarios that need constructor work: C2, C3, C6, N2, S6.
5. GPU-track rows (N1, N8) come from `docs/hardware-semantics.md`.

## 4. Results

Generated from `docs/data/stress.json`; empty until the scenarios run.

| ID | What happened | Mechanism | Advice bits | U (λ = 40) | Overhead | Description bytes | Verdict |
|---|---|---|---|---|---|---|---|
