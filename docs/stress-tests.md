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
| M4 | **Charged compile-time advice `a`.** Names the circuit; charged `|a|` bits at admission (`A = max`), `|a|` the exact bit length `G` declares (the byte padding is checked zero). | `|a|` | Structure not determined by `(x, y)` and not decided in-circuit: MoE routing given as advice, speculative-decoding acceptance pattern given as advice, schedule when RU = step (`Schedule.encode()` v4: bit-packed, `Schedule.bit_length()` bits), a truncated request's generated length (S7). |
| M5 | **In-circuit decision (padding).** The choice is computed by gates and the structure is padded to its maximum. | wasted gates | MoE top-k with capacity-padded experts, speculative acceptance with masked candidate positions, constrained-decoding masks, token sampling from published randomness. Trades advice bits for prover cost; both must be measured. |
| M6 | **Replay-time advice (fault declarations).** After the challenge, the prover names up to `f_max` VUs it does not claim correct; the verifier skips their relation. | `f_max · u(1)`, `u(1) = W_V + log2 |S|` | Silent data corruption, bit flips, a GPU that produced a wrong token. Completeness becomes exact for any honest prover with at most `f_max` undetected faults per session. |
| M7 | **Published randomness.** Sampling randomness is part of `x`. | 0 | Temperature/top-p sampling, dropout-free inference has no other randomness. The server never chooses its own randomness. |
| M8 | **Separate commitments.** Weights, adapters and model versions each under their own root, referenced publicly per request. | 0 | Multiple models, hot swaps, LoRA multi-tenancy. |

M3 is not implemented, and nothing charges an output-determined choice at
zero: `Compilation.advice_bits` charges the exact bit length the constructor
declares for `a` (`G.advice_bits(x, a)`, else `8·len(a)`), and the compiler
checks that `a` is the canonical encoding of that many bits -- exactly
`ceil(bits / 8)` bytes with zero padding -- so the padding is not a free
channel (`veritor.core.advice`, the `Header` carries `advice_bits`, protocol
v7). What M3 would add -- charging 0 for a length or a pattern the verifier
could recompute from `(x, y, pp)` -- stays a theory question for the
architect (report flags 5 and 10): a length the server chose with no rule
to check it against carries `log2(#choices)` bits however it is conveyed,
so S7 charges it as advice and no row charges it as 0. M6 is implemented
(below): the interior message carries the declarations and `Bound` has the
`f_max` term.

### Check outputs: what a fixed word costs

A declared output that exists only so the verifier can require it to equal
a constant fixed by `G` is a *check output*: the root definition's `checks`
field names ranges of output ordinals and their constant (part of the
description, hence of the digest; `FORMAT_VERSION` 3). The verifier requires
the opened value at every check position to equal the constant, both in the
claimed outputs at admission and on the opened boundary
(`VerificationCode.CHECK_MISMATCH`), so a run whose check fails is rejected
there. A value the verifier fixes carries no information: check outputs
contribute 0 to the root's `out_bits` and 0 to every unit's `reach_bits`
and `ancestor_bits` (an erroneous VU that moves a check output is rejected,
not counted; `veritor.analysis.bound` has the soundness sentence). Three
uses in the table:

- the MoE `ok` word (C2b, C2c): `route_check` folds "the advised route is the
  router's top-k" into one word the verifier requires to be `1`; the route
  is charged as advice (exact bits), the check word is 0 bits, and the step
  bodies are one definition for every route -- the route enters at the call
  site as the parent ranges passed for the router's columns and the chosen
  experts' weights (`ToyLM.route_ports`), so the description grows by call
  sites, not by bodies per route (C2b: 42,473 → 38,397 bytes at 7 tokens;
  C2c at 128 tokens, E=2: 55,300 → 38,830);
- the speculative-decoding `ok` word (C3b): `acceptance_check` folds
  "exactly m draft tokens agree" into it;
- conditionally-absent outputs (S7): a request that stopped after `t` of
  the `max_new` tokens it asked for keeps `max_new` output slots, and the
  `max_new - t` the constructor knows are absent are check outputs required
  to equal the blank word `vocab`. They carry 0 bits; the length that
  decides where they begin is paid for once, as advice. A *presence mask*
  -- a per-slot flag the server sets and the verifier merely honours -- is
  not an alternative: the pattern of present slots is the server's choice
  and carries `log2(#patterns)` bits, the same bits the length advice pays
  for, so a mask charged nothing would be free advice (report flag 10). C1
  has no absent slots to blank: with request RUs the streamed length is in
  `x` (the client saw the stream end) and with step RUs each join's length
  is the request's output count, charged in the schedule.

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

*Charge of a declaration fixed in advance.* Let `D` be the declared set,
`E` the error set and `E' = E \ D` what stays exposed. If `D` does not
depend on the challenges, the output is a function of the correct
computation outside `E`, the values at `E'` (which must survive the
sampling as before) and the values at `D`. Enumerating `D` and its values,
`|Y_eta| <= sum_{|D| <= f} 2^kappa(D) · sum_{E' admissible} 2^kappa(E')
<= (1 + |S| 2^W_V)^f · 2^U_0`, using `kappa(D) <= sum_{v in D} kappa(v)`
(the union of downstream cuts is a downstream cut) and
`sum_{j <= f} C(|S|, j) 2^(j W_V) <= (1 + |S| 2^W_V)^f`. One declared fault
therefore costs `u(1) = log2(1 + |S| 2^W_V) = W_V + log2(|S| + 2^-W_V)`,
which is `W_V + log2 |S|` to within `2^-W_V`; `|S|` counts the VUs and
`W_V` is the widest cut over the VU kinds that have a relation
(`fault_allowance_bits(target, f_max) = f_max · u(1)`). This is value-level
advice: compile-time advice names the circuit and is charged by how many
circuits it can name; a declaration names a VU (`log2 |S|` bits) and its
contents (`W_V` bits). It is the price at `q = 1`, where the q-challenge
reveals nothing and the prover's best declaration is a function of `E`
alone.

*Numbers.* Small simulation (2 pods × 16 steps, 12 requests, 30 streamed
tokens, 26,314 gates): `|S| = 3,791`, `W_V = 64` (the `onehot` VU declares
eight 16-bit outputs, 128 bits, but its ancestors' interface cuts that to
64: `cut_bits = min(out, reach, ancestor)`), `u(1) = 75.9` bits; `Bound` at
`theta = (1/2, 1/8)` is capped at `out_bits = 480` with or without the
term, and at `theta = (1, 1)` it goes from `0` to `75.9`. One declared
fault is thus 16% of that run's entire output -- the mechanism is priced
for serving-scale runs.
GPT-2 Small's kind table (`GPT2G`, one request of 64 prompt + 64 new
tokens): `|S| = 161.3 M` VUs and `W_V = 32`, because the LayerNorm
statistics and the accumulator-width cells are 32-bit VU outputs, giving
`u(1) = 59.3` bits; under the 16-bit-word assumption `W_V = 16` and
`|S| = 176.8 M` the figure is `43.4` bits. Either way a fault costs about
three to four tokens' worth of capacity.

*Charge of a declaration chosen after `J` (what `bound` charges).* The
protocol lets the prover choose `D` after seeing `J` -- an honest server
cannot know its faults before it replays the opened RUs -- and an adversary
uses that: corrupt one VU in each of many RUs and pardon whichever `f` of
them were opened. With `N_J` corrupted VUs in the opened RUs the acceptance
probability is `sigma_f(E) = E_J[(1 - s)^max(0, N_J - f)]` instead of
`sigma_0(E) = E_J[(1 - s)^N_J]`. Two rigorous bounds: (i) `sigma_f(E) <=
(1 - s)^-f · sigma_0(E)`, so `U_f <= U_0(eta (1 - s)^f)` -- the fold at a
threshold lowered by `f · log2(1/(1 - s))` bits, worth about `1/q` extra
corrupted VUs per declaration when `s` is small, one when `q = 1`, vacuous
at `s = 1`; (ii) the best declaration is one of at most `(1 + |S|)^f` sets
and a maximum is at most a sum, so `sigma_f(E) <= sum_D sigma_0(E \ D)` and
`U_f <= U_0(eta / (1 + |S|)^f) + f · u(1)`, valid for every `s`.
`bound(..., max_faults=f)` charges the smaller of the two (and of
`U_0 + f · u(1)` at `q = 1`), capped by the interface
(`veritor.analysis.faults.declared_bits`). The adaptive price is far above
`f · u(1)` whenever `q` is small: the rate `rho` multiplies the lowered
threshold's `f · log2(1/(1 - s))` bits by about `1/(q s)`. For the headline
serving policy (`q = 1.6e-8`, `s = 8.9e-3`, `rho = 4.7e11`) one declaration
costs about `6e9` bits against `U = 1.9e13`, so `f_max = 4` is 0.13% of `U`
-- affordable, but not `u(1) ~ 100` bits. A protocol that wanted the
`f · u(1)` price would take declarations *before* the q-challenge, which an
honest server can do only for faults it detects without replaying (ECC
events, crashes, NaN checks), not for silent corruption. Which declarations
to allow where is a decision for the architect.

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
| S7 | Client disconnect / timeout truncates generation | M4 + check outputs (M3 open) | the observed length is not EOS-terminated and nothing in `x` says it: `TruncatedRequestsG` takes `t` as advice (`ceil(log2 max_new)` bits per request, charged exactly) and makes the `max_new - t` absent slots blank check outputs (0 bits); a client attestation would move `t` into `x` | truncated at *t*: outputs are the prefixes, `U` is the *t*-token run's, a token in an absent slot is `CHECK_MISMATCH` |
| S8 | Retries and duplicate execution (one result discarded) | M1 | discarded work has no observed output and is not in the circuit; document the principle | duplicate run; circuit unchanged |

### 2.2 Data-dependent control flow

| ID | Scenario | Mech. | Gap / fix | Test |
|---|---|---|---|---|
| C1 | EOS-terminated generation | M3 | H: `advice_bits` charges the length; add the verifier rule (last token EOS or `max_tokens`) and charge 0 | 1000 requests: charged advice 0; a server that stops before EOS without EOS is rejected |
| C2 | MoE routing (top-k of E experts per token) | M4 vs M5 | H: no in-circuit routing; build both: advice route (`k·ceil(log2 E)` bits/position/layer, the `ok` check word 0 bits, one step body for every route) and padded route (top-k gates, capacity factor, masked experts) | per-token bits vs padded gate overhead; `U` for both; crossover |
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
3. Mechanisms first, scenarios second: M6 (declarations) is a protocol change and unblocks N4, N5; exact advice bits and check outputs (section 1) settle the charging of C1, C2, C3, S7 without M3, which remains a theory question; the gate-set union unblocks N3; the weight-root set unblocks W1–W3.
4. Then the expressiveness scenarios that need constructor work: C2, C3, C6, N2, S6.
5. GPU-track rows (N1, N8) come from `docs/hardware-semantics.md`.

## 4. Results

Generated by `python -m veritor.stress.report` from `docs/data/stress-control-flow.json`, `docs/data/stress-protocol.json`, `docs/data/stress.json`; `U` is `Bound` at `eta = 2^-40`, overhead is `Cost(...).total` over the honest replay cost at the simulation's policy `theta = (1/2, 1/8)`.

| ID | What happened | Mechanism | Advice bits | U (λ = 40) | Overhead | Description bytes | Verdict |
|---|---|---|---|---|---|---|---|
| S1a | continual batching, ClusterG (RU = step), fcfs: 5 joins on 2x2 pods x slots | M4 | 95 | 256 | 1.39 | 43,422 | outputs = reference; the schedule is the advice and the step kinds follow it |
| S1b | continual batching, ClusterG (RU = step), reversed admission: 5 joins on 2x2 pods x slots | M4 | 95 | 256 | 1.39 | 43,161 | outputs = reference; the schedule is the advice and the step kinds follow it |
| S1c | continual batching, ClusterG (RU = step), one slot per pod: 5 joins on 2x2 pods x slots | M4 | 95 | 256 | 1.4 | 43,208 | outputs = reference; the schedule is the advice and the step kinds follow it |
| S1d | continual batching, RequestsG (RU = request): the same requests, any schedule | M1 | 0 | 256 | 0.891 | 41,874 | outputs = reference; one circuit for all three schedules; request kinds invariant under admission order |
| S2a | scale, RequestsG: 1,000 requests of 98 shapes (Zipf-like prompts, bimodal generation) | M1 | 0 | 30,392 | 0.889 | 306,134 | description 306,134 B, trace 34 ms, compile 61 ms, kind table 19 ms, bound 1136 ms |
| S2b | scale, RequestsG: 10,000 requests of 119 shapes (Zipf-like prompts, bimodal generation) | M1 | 0 | 31,530 | 0.889 | 339,452 | description 339,452 B, trace 54 ms, compile 67 ms, kind table 20 ms, bound 2270 ms |
| S2c | scale, RequestsG: 100,000 requests of 119 shapes (Zipf-like prompts, bimodal generation) | M1 | 0 | 32,435 | 0.889 | 339,808 | description 339,808 B, trace 244 ms, compile 74 ms, kind table 22 ms, bound 3701 ms |
| S3a | preemption by recompute, ClusterG: request evicted after 2 tokens, re-prefilled 3 steps later on another pod | M4 | 110 | 256 | 1.4 | 43,990 | outputs = reference; both attempts are in the circuit, the recomputed positions declared but not output |
| S3b | preemption by recompute, RequestsG: the same eviction | M1 | 0 | 256 | 0.891 | 41,874 | circuit identical with and without the eviction (digest equal); outputs = reference |
| S4a | preemption by swap, ClusterG: KV cache retained across a gap of 3 steps (Schedule v3 resume) | M4 | 110 | 256 | 1.4 | 43,091 | outputs = reference; the resumed decode step reads the KV rows declared 3 steps earlier; honest cost unchanged |
| S4b | preemption by swap, RequestsG: the same swap | M1 | 0 | 256 | 0.891 | 41,874 | circuit identical with and without the swap (digest equal) |
| S5a | chunked prefill, ClusterG: a 9-token prompt in 3 chunks of 3 over 3 steps | M4 | 38 | 80 | 1.34 | 44,404 | outputs = reference; the step-RU description differs (chunk kinds), the honest cost does not |
| S5b | chunked prefill, RequestsG: the same prompt | M1 | 0 | 80 | 0.889 | 39,387 | the request circuit is the sequential one, identical with and without chunking; values equal |
| S6a | prefill/decode disaggregation, ClusterG: prefill on pod 0, decode on pod 1 (resume across pods) | M4 | 96 | 144 | 1.35 | 34,370 | outputs = reference; a decode step reads another pod's declared KV rows through its ports |
| S6b | prefill/decode disaggregation, RequestsG | M1 | 0 | 144 | 0.893 | 33,331 | pods are not in the statement: the circuit is the sequential one |
| S7 | client disconnect / abort: 6 requests asked for max_new = 8, the clients received (8, 3, 5, 1, 6, 2) tokens (25 of 48); request RUs, the generated length as advice, the absent slots blank check outputs | M4 charged advice (exact bits) + check outputs for the absent slots; M3 open | 18 | 418 | 0.894 | 58,237 | ACCEPTED; the steps are RequestsG's for requests that asked for t tokens, and the max_new - t absent slots per request are blank (vocab) check outputs: 0 bits, U unchanged |
| S8a | retries, RequestsG: request 2 executed twice, one run discarded | M1 | 0 | 256 | 0.891 | 41,874 | the compiled circuit is identical with and without the discarded work (digest equal) |
| S8b | retries, ClusterG: the duplicate attempt declared in the schedule | M4 | 110 | 256 | 1.4 | 43,993 | outputs unchanged; the duplicate's steps are replay units, its tokens declared but not output |
| C1a | variable-length generation, request RUs: 6 requests, max_new 3..13, EOS stops | M5 in-shape: each request's circuit is its streamed length; no advice | 0 | 576 | 0.889 | 72,128 | no advice: lengths are public (the client sees the stream end), route (c) |
| C1b | variable-length generation, step RUs (ClusterG, 1 pod x 2 slots): the same requests | M4 charged advice: the schedule (Schedule.encode() v4, bit-packed, exact bits) | 118 | 694 | 1.39 | 76,955 | advice = the schedule; each request's length rides in its join (118 bits for 6 joins, charged exactly; 120 bits on the wire) |
| C2a | MoE routing, padded: E=4 experts, top-1, 1 layer, d_model 4; every position runs every expert | M5 in-circuit decision: router_topk VU (rank by lt chains) masks the experts' outputs | 0 | 112 | 0.89 | 45,592 | no advice, E/k times the expert work; lowest U at equal theta, and at equal cost while E/k <= 4 (C2c) |
| C2b | MoE routing, advised: the same shape and requests; the route is advice, only chosen experts run | M4 charged advice: ceil(log2 E) bits per chosen expert per position per layer, charged exactly; route_check VU folds into ok, a check output (0 bits) | 20 | 132 | 0.897 | 38,397 | the honest server's pick once E/k is large (from E=8, k=1 here, C2c): k/E of the expert compute for k*log2(E) bits per position, spent on a stronger theta |
| C2c | MoE crossover sweep: (E, k) in {(2,1), (4,1), (8,1), (8,2)}, 32 requests x 4 tokens, theta grid q in {1/2, 1}, s in {1/8..7/8} | M5 vs M4 compared at equal theta (equal relative overhead) and at equal absolute prover cost | 0 | 0 | 0 | 0 | at equal theta padding beats advice in U at every (E, k), by exactly \|a\| (the ok words are check outputs, 0 bits); at equal absolute prover cost advice wins from E=8, k=1 on (E=2,k=1:padding, E=4,k=1:padding, E=8,k=1:advice, E=8,k=2:padding) |
| C3a | speculative decoding, padded: gamma=2, target d_model 4, draft d_model 2 (random weights), 2 requests | M5 in-circuit acceptance: eq per position, prefix product, masked slots, V entries masked by the flags | 0 | 464 | 0.89 | 155,074 | never the honest server's pick: (gamma+1)x the target positions and the interface, max_new-1 steps |
| C3b | speculative decoding, advised: the same models and requests; m per step is advice | M4 charged advice: ceil(log2(gamma+2)) bits per step, charged exactly; acceptance_check VU folds 'exactly m agree' into ok, a check output (0 bits) | 18 | 194 | 0.893 | 81,409 | the honest server's pick: the target does exactly plain decoding's positions plus the draft's |
| C4a | constrained decoding, RequestsG with the argmax head: 3 of 6 requests ban 1-3 tokens (public, in x) | M5 | 0 | 272 | 0.892 | 60,778 | outputs = reference sampler with the mask; banned tokens never emitted; +23 gates per generated token (masked_argmax vs argmax), +2..8 gates x vocab once per request for the mask |
| C4b | constrained decoding, RequestsG with the sample head: 3 of 6 requests ban 1-3 tokens (public, in x) | M5 | 0 | 288 | 0.894 | 66,019 | outputs = reference sampler with the mask; banned tokens never emitted; +8 gates per generated token (masked_sample vs sample), +2..8 gates x vocab once per request for the mask |
| C5 | nondeterministic sampling: the LM head draws each token from a public 5-bit random word in x (sample VU, 49 gates over vocab 8, vs 49 for the argmax); 5 requests, 20 tokens, request RUs | M7 + M5 | 0 | 320 | 0.893 | 51,703 | honest run ACCEPTED; a server that biases 4 sampled tokens is RELATION_REJECTED at a head VU under theta = (1, 1), and escapes theta = (1/2, 1/8) in 299/400 fresh challenges against sigma(E) = 0.772 predicted |
| C6a | prefix caching, route A (PrefixG): 8 requests share an 8-token prefix computed by one RU | M1 | 0 | 384 | 1.38 | 38,845 | outputs = reference; the prefix RU declares W_R = 64 KV words read by 8 suffix RUs; honest replay cost 32496 vs 65928 (saves 33432) |
| C6b | prefix caching, route B (RequestsG): each of the 8 requests recomputes the 8-token prefix | M1 | 0 | 384 | 0.886 | 36,598 | outputs = reference; 8 x prefix replay (4776 each); boundary is prompts and tokens only |
| N2a | tensor parallelism, RequestsG at TP = 1: marked dots are 1 partial dots and a fixed-order reduction | M2 | 0 | 256 | 0.891 | 41,874 | outputs identical across TP = 1, 2, 4; 14 VU kinds (0 not in TP = 1); U identical (uncapped too); dot_8 keeps 15 gates, its sum tree regrouped |
| N2b | tensor parallelism, RequestsG at TP = 2: marked dots are 2 partial dots and a fixed-order reduction | M2 | 0 | 256 | 0.891 | 41,596 | outputs identical across TP = 1, 2, 4; 14 VU kinds (2 not in TP = 1); U identical (uncapped too); dot_8 keeps 15 gates, its sum tree regrouped |
| N2c | tensor parallelism, RequestsG at TP = 4: marked dots are 4 partial dots and a fixed-order reduction | M2 | 0 | 256 | 0.891 | 42,346 | outputs identical across TP = 1, 2, 4; 14 VU kinds (2 not in TP = 1); U identical (uncapped too); dot_8 keeps 15 gates, its sum tree regrouped |
| N3 | heterogeneous fleet, ClusterG: pod 0 on sm80, pod 1 on sm90 (two namespaced copies of the toy ISA in one Σ) | M2 | 96 | 144 | 1.35 | 44,884 | outputs = reference; one circuit over union_gate_set (14 gates: 6 operators x 2 + in, weight); 6 step kinds carry their architecture; prefill on sm80 decoded on sm90; a foreign gate name fails to compile |
| N4 | silent data corruption: bit 0 of one dot product's output word flipped in RU 7 of the small simulation (2 pods x 16 steps, 12 requests, 30 streamed tokens, 26314 gates, 3791 VUs); the server streamed the consequences (0 tokens changed) and finds the VU when it replays the opened RU | M6 | 271 | 751 | 1.4 | 51,367 | f_max = 0: RELATION_REJECTED at the faulty VU once the s-challenge sampled it (it escaped 10 earlier challenges at s = 1/8); f_max = 1: ACCEPTED, declared; the s-challenge did not sample it; a second undeclared corruption under the same budget, everything sampled: RELATION_REJECTED |
| N5 | a GPU produces a wrong token that was streamed: the top bit of a logit dot's output word flips, the sampler draws a different token and the request continues from it (1 and 2 tokens changed in the two runs); outputs stand as streamed | M6 | 271 | 751 | 1.4 | 51,367 | fault in RU 9, not opened by the q-challenge: nothing declared, ACCEPTED (14 of 20 RUs opened); fault in RU 7, opened but VU 1811 not sampled: declared anyway (the server cannot know the s-challenge), ACCEPTED |
| W1 | model version hot swap: two weight sets of one shape in one run, six requests alternating versions; one weights RU holds both, each request's kind is wired to its version's slice | M2 (one root over both versions; version per request public in x) | 0 | 384 | 0.894 | 41,682 | ACCEPTED, outputs equal each version's reference; a server that served version 1 to a version-0 request is RELATION_REJECTED under theta = (1, 1) |
| W2 | LoRA adapters per tenant: each request carries its tenant's merged matrix W_1 + A B (32 words, layer 0) as in gates of an adapter RU; three tenants, six requests | M1 (adapter as public input, committed in the boundary) | 0 | 384 | 0.902 | 34,776 | ACCEPTED, outputs equal each tenant's reference; a server that ran the base matrix for a tenant is RELATION_REJECTED under theta = (1, 1) |
| W3 | several models on one cluster: three weight sets of one shape, a router assigns six requests | M2 (client chose, model in x: 0 bits) / M4 (server chose: advice, exact bits) | 12 | 396 | 0.894 | 41,684 | both ACCEPTED on the same circuit (same digest); the advised route costs exactly its bits |
| E1 | scale, RequestsG: 1,000,000 requests of 119 shapes (Zipf-like prompts, bimodal generation) | M1 | 0 | 33,713 | 0.889 | 340,162 | description 340,162 B, trace 2309 ms, compile 54 ms, kind table 20 ms, bound 2667 ms |

Notes:

- **S1a**: U capped at \|Out\| = 256 bits (uncapped 759 bits); 10 RU kinds; description digest differs per schedule
- **S1b**: U capped at \|Out\| = 256 bits (uncapped 706 bits); 9 RU kinds; description digest differs per schedule
- **S1c**: U capped at \|Out\| = 256 bits (uncapped 776 bits); 10 RU kinds; description digest differs per schedule
- **S1d**: U capped at \|Out\| = 256 bits (uncapped 284 bits); the schedule is the server's business: not in x, not in a, not in the circuit
- **S2a**: 497 root output runs (one per generated position of each shape): exceeds the default 256 (compiled with max_output_runs = 4096); requests of one kind are one repeat, so the description does not grow with the requests
- **S2b**: 679 root output runs (one per generated position of each shape): exceeds the default 256 (compiled with max_output_runs = 4096); requests of one kind are one repeat, so the description does not grow with the requests
- **S2c**: 679 root output runs (one per generated position of each shape): exceeds the default 256 (compiled with max_output_runs = 4096); requests of one kind are one repeat, so the description does not grow with the requests
- **S3a**: U capped at \|Out\| = 256 bits (uncapped 1010 bits); honest replay cost 16538 vs 15198 without the eviction
- **S3b**: U capped at \|Out\| = 256 bits (uncapped 284 bits); recompute is the server's cost, not the statement's
- **S4a**: U capped at \|Out\| = 256 bits (uncapped 781 bits); Join.resume marks the attempt; the gap is wiring, not a kind: decode_c is the same kind resumed or not
- **S4b**: U capped at \|Out\| = 256 bits (uncapped 284 bits)
- **S5a**: U capped at \|Out\| = 80 bits (uncapped 268 bits); Join.chunk carries the chunk size; each chunk declares its KV rows for the next step
- **S5b**: U capped at \|Out\| = 80 bits (uncapped 91 bits)
- **S6a**: U capped at \|Out\| = 144 bits (uncapped 631 bits); the cluster is synchronous and time-major: any step may read what any earlier step of any pod declared
- **S6b**: U capped at \|Out\| = 144 bits (uncapped 161 bits)
- **S7**: t as advice: ceil(log2 max_new) = 3 bits per request, 18 bits, charged exactly (3 bytes on the wire). Every request keeps 8 output slots; the 23 absent ones are check outputs the verifier requires to equal 8, so U = 400 bits, that of the 25-token run, and a token streamed where the client saw none is CHECK_MISMATCH. A presence mask the server sets and the verifier merely honours would carry the same information (which slots are present) uncharged: it is the length, and the length is advice. Alternative, padding to max_new with the tail masked: 0 advice bits, 79% more prover compute for this batch (42934 vs 24035 honest gate-cost units) and 61526 description bytes. Whether output-determined shape (len(y) = t, no EOS rule to check) may be charged 0 like EOS termination is an open theory question for the architect; here it is charged.
- **S8a**: U capped at \|Out\| = 256 bits (uncapped 284 bits); what is not in (x, a) is not in C: a discarded execution has no gates
- **S8b**: U capped at \|Out\| = 256 bits (uncapped 818 bits); honest replay cost 17096 vs 15198 when the schedule omits it
- **C1a**: 36 tokens streamed of 48 asked; lengths (1, 5, 7, 3, 11, 9) (3 EOS stops); U = 576 bits (interface-capped); 604.1 gates/token. The lengths are in x, so the circuit has no absent slots to blank (S7 is the case where the length is the server's and is advice); a presence mask would carry the same choice uncharged.
- **C1b**: U = 576 bits (interface-capped) + 118 advice bits for 6 joins over 19 steps (a join is step and length in 5 bits each, slot and resume in 1, the request gamma-coded); 604.1 gates/token (same work, cut into steps); overhead 1.39 vs 0.89 with request RUs: the KV cache crosses every step boundary. Every output is a streamed token (a join's length is its request's output count), so there are no absent slots to blank; the lengths are charged in the joins.
- **C2a**: 7 tokens; 1326.4 gates/token vs 747.9 advised (1.77x); U = 112 bits = the interface (capped); route check 0 bits.
- **C2b**: route description 20 bits, charged exactly (3 bytes on the wire, the padding checked zero); the ok word is a check output the verifier requires to be 1, so U = 112 bits, the interface, as padded; description 38397 vs 45592 bytes at 7 tokens: the step bodies are one definition per (context, positions) whatever the route -- the route enters at the call site as the ranges passed for the router's columns and the chosen experts' weights -- so only the request bodies (their call sites) are per request (see C2c at 128 tokens).
- **C2c**: E=2,k=1: gates padded/advised 1.28x (96017 vs 74737), description 29861 vs 38830 bytes, \|a\| 160 b over 128 tokens; at theta=(1/2,1/8) capacity 2048 vs 2208 (overhead 0.889 vs 0.893); at equal absolute cost 130591: padding 2048 vs advice 2208 -> padding \| E=4,k=1: gates padded/advised 1.82x (143833 vs 79033), description 34718 vs 42548 bytes, \|a\| 320 b over 128 tokens; at theta=(1/2,1/8) capacity 2048 vs 2368 (overhead 0.889 vs 0.894); at equal absolute cost 195660: padding 2048 vs advice 2368 -> padding \| E=8,k=1: gates padded/advised 2.82x (247145 vs 87625), description 48351 vs 46740 bytes, \|a\| 480 b over 128 tokens; at theta=(1/2,1/8) capacity 2048 vs 2528 (overhead 0.892 vs 0.897); at equal absolute cost 333957: padding 2048 vs advice 1340 -> advice \| E=8,k=2: gates padded/advised 2.15x (247145 vs 114985), description 48351 vs 52104 bytes, \|a\| 960 b over 128 tokens; at theta=(1/2,1/8) capacity 2048 vs 3008 (overhead 0.892 vs 0.899); at equal absolute cost 333957: padding 2048 vs advice 3008 -> padding
- **C3a**: 11 tokens; 2313.2 gates/token vs 991.3 advised; outputs 29 slots (blank = vocab) vs 11 tokens; U = 464 bits (capped); gamma=4: 45491 gates, U 752. The blanks make each step's m output-determined (route c, not taken).
- **C3b**: acceptances ((0, 0, 0, 0, 1), (0, 0, 0, 1)): 18 bits, charged exactly (3 bytes on the wire); perfect draft (= target): acceptances ((2, 2), (2, 2)), 8 advice bits, 16563 gates; gamma=4 random draft: 27 advice bits, 10904 gates; the ok word is a check output the verifier requires to be 1: 0 bits, U = 176 = the 11 tokens' interface. The token count is output-determined; each m is not.
- **C4a**: U capped at \|Out\| = 272 bits (uncapped 305 bits); the mask is allowed_row VUs over in gates: nothing about the constraint is advice; ClusterG rejects banned lists: its step kinds carry no per-occupant mask ports (gap)
- **C4b**: U capped at \|Out\| = 288 bits (uncapped 321 bits); the mask is allowed_row VUs over in gates: nothing about the constraint is advice; ClusterG rejects banned lists: its step kinds carry no per-occupant mask ports (gap)
- **C5**: The sampling head adds 1 gate(s) per token over the argmax head: the sample VU has 49 gates vs 49 for the argmax at vocab 8, plus the in gate of the random word (13538 vs 13516 gates for the same requests, 2 sampler constants as weights); the randomness is 5 public bits per token, advice 0. Biasing a token is a relation violation of the sample VU, priced by Bound like any corruption: the server never chooses its randomness.
- **C6a**: U capped at \|Out\| = 384 bits (uncapped 818 bits); boundary 112 vs 104 words: +64 declared cache rows, -56 repeated prompt tokens; the shared prefix is the longest common prefix of the prompts, a function of x: no advice
- **C6b**: U capped at \|Out\| = 384 bits (uncapped 428 bits); crossover: route A wins on replay work for k >= 2 requests (saves (k - 1) x 4776) and pays 64 boundary words once, which costs capacity: uncapped U 818 (A) vs 428 (B) bits and overhead 1.38 vs 0.89
- **N2a**: U capped at \|Out\| = 256 bits (uncapped 284 bits); the TP degree is public configuration in the constructor's manifest, not advice
- **N2b**: U capped at \|Out\| = 256 bits (uncapped 284 bits); the TP degree is public configuration in the constructor's manifest, not advice
- **N2c**: U capped at \|Out\| = 256 bits (uncapped 284 bits); the TP degree is public configuration in the constructor's manifest, not advice
- **N3**: U capped at \|Out\| = 144 bits (uncapped 631 bits); the pod-to-architecture map is public configuration in the manifest; same honest cost as the homogeneous run (7925); the homogeneous description differs in every gate name
- **N4**: u(1) = log2(1 + \|S\| 2^W_V) = 75.89 bits (\|S\| = 3791, W_V = 64); Bound at theta = (1/2, 1/8): 480 -> 480 bits, both capped at out_bits = 480 (a 30-token run); uncapped, at theta = (1, 1): 0 -> 75.89 = f_max * u(1), the price at q = 1 where J reveals nothing. f_max from the Llama-3 SDC rate 2.83e-07/device-hour (6 events, 54 days, 16,384 GPUs): this run is 2 pods x 16 steps at 0.05 s/step = 1.6 device-seconds = 4.4e-04 device-hours, mean 1.3e-10 faults, f_max = 1 (the floor: admitting declarations at all admits one). A 16,384-GPU fleet for an hour has mean 4.63e-03 -> f_max = 2 at tail 1e-6; for a day mean 0.111 -> f_max = 4. At q < 1 the prover declares after seeing J and Bound charges the adaptive prover instead (the fold at eta (1 - s)^f_max or at eta / (1 + \|S\|)^f_max plus f_max u(1), whichever is smaller); see docs/stress-tests.md M6.
- **N5**: The honest server self-checks every opened RU against the values it streamed and declares what disagrees, before the s-challenge exists; a declaration the sample misses costs nothing beyond the f_max * u(1) already in Bound. At f_max = 0 the unopened fault is accepted too (the verifier never sees that RU) and the opened one is rejected iff sampled: completeness 1 - s per opened fault.
- **W1**: No protocol change: Header.weights is the root over the concatenation (406 weight gates, vs 203 for one version; description 41682 vs 41280 bytes). Gap vs M8: the joint root changes whenever the served set changes, so a version has no root of its own across sessions, and the request names its version through the description's wiring, not by a root id; the fix is Header.weights as a tuple of roots each covering a declared rank range of the weight gates.
- **W2**: No protocol change: 32 more boundary positions per request (232 public inputs vs 40), the request kind is unchanged (its ports are the same width; some are now in gates). Bound 384 -> 384 bits. What this is not: the adapter is public and per run, not a server-held weight under its own root (M8), and the toy LM has no low-rank path, so the merged d x hidden matrix stands in for A and B.
- **W3**: Server-chosen routing: ceil(log2 3) = 2 bits per request, 12 bits, charged exactly (2 bytes on the wire, the padding checked zero); Bound 384 -> 396. Gap: models of different shapes need one description with several kind families (the tracer builds one ToyLM family per shape) and, as for W1, a root per model rather than one root over the concatenation.
- **E1**: 679 root output runs (one per generated position of each shape): exceeds the default 256 (compiled with max_output_runs = 4096); requests of one kind are one repeat, so the description does not grow with the requests
