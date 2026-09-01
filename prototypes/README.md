# Memoized constructor-call DAG prototype

This directory is a clean-room experiment. It imports nothing from the active
`veritor` package.

The boundary under test is:

~~~text
untrusted G(x, a) -> serialized bytes
trusted Kernel.load(bytes) -> validated circuit or rejection
~~~

The producer helper uses JAX-like symbolic calls and `@circuit`-style
memoization to emit:

- local primitive leaves;
- reusable pure-function definitions;
- occurrences wired to parent inputs or earlier outputs;
- one root definition.

The trusted side independently parses the data, checks gate membership,
arities, backward references, child interfaces, counts, and costs, then assigns
fresh global positions to every occurrence. Its cache is keyed by the canonical
definition-body digest, not by an untrusted function label.

## Run it

~~~bash
uv run python -m prototypes.call_dag
uv run pytest prototypes/test_call_dag.py
~~~

The demo constructs batches with request lengths `[4, 8, 4]` and `[8, 4, 8]`.
Only the two dot-product shapes and their shared multiply-accumulate definition
need distinct bodies. Reordering the batch creates a new root but reuses all
three lower definitions.

## Initial result

The representation works mechanically:

- reused definitions receive fresh global positions and input wiring;
- kernel-computed counts and costs multiply by occurrence;
- `GateAt` agrees with brute-force flattening at every conceptual leaf;
- transitive occurrence interfaces identify exactly which external values are
  read by gates inside each occurrence;
- occurrence-tree cuts are checked to partition the conceptual gates, and
  their replay boundaries agree with a flat cross-unit-read oracle;
- memoization does not change serialized output;
- malformed gates, arities, forward references, digests, fields, advice, and
  inputs are rejected.

Canonical JSON has noticeable fixed overhead. For three equal dot products,
the measured structured/flat serialized sizes were:

- length 4: 1,894 / 1,143 bytes;
- length 16: 4,564 / 4,645 bytes;
- length 1,024: 244,492 / 323,017 bytes.

Thus this deliberately verbose encoding loses on tiny circuits, crosses over
around length 16 in the toy, and saves about 24% by length 1,024. A binary
encoding with short local definition IDs would reduce the call-record overhead
substantially. The larger expected gain comes when each reused child contains
many leaves rather than this demo's two.

## Staged-commitment bridge

The trusted kernel now computes the compiler data that Draft v4 actually uses:

- every validated definition records the subset of its root inputs that its
  gates transitively read;
- `occurrence_summary(root, path)` substitutes one occurrence's concrete
  parent wiring and returns its global gate interval, cost, external reads, and
  outputs;
- `derive_replay_plan(root, paths)` rejects gaps, overlaps, duplicate paths,
  invalid paths, and zero-gate units;
- a valid plan maps secret sampled gates to challenged units, derives each
  challenged unit's interior commitment domain, and computes expected replay
  cost;
- its boundary is computed as all root inputs, all root outputs, and the union
  of selected units' external reads.

The last rule is exactly the Draft-v4 boundary formula. An independent test
flattens small circuits, assigns every gate to its selected unit, enumerates
all cross-unit reads, and obtains the same boundary for coarse, fine, and mixed
cuts. The production derivation does not call `flatten`.

`staged_replay.py` consumes this plan in a concrete protocol-mechanics
prototype:

- indexed Merkle leaves bind the compiler-derived global position, value,
  commitment domain, and cell width;
- the boundary and every unit interior have distinct domains derived from the
  compiled root and replay cut;
- a verifier state machine enforces
  `boundary -> J -> unit interiors -> T`;
- it supports both Draft v4's pre-sampled-gate mode, where `J` is derived from
  a hidden Bernoulli gate sample, and Appendix A's two-stage mode, where units
  are sampled with probability `q` before gates inside selected units are
  sampled with probability `s`;
- sampled writes and reads are routed to either the boundary commitment or the
  owning unit commitment, then checked with `GateAt` and the trusted primitive;
- malformed commitment domains, incomplete `J` responses, invalid openings,
  and out-of-order messages are rejected.

The corresponding publication-style construction and proof are in
[`docs/two-stage-commitment-protocol.md`](../docs/two-stage-commitment-protocol.md).

~~~bash
uv run pytest prototypes/test_staged_replay.py
~~~

The adversarial test forges one gate output, computes all downstream values
consistently from it, and observes exactly the intended behavior: sampling that
gate rejects, while sampling every other gate accepts. This is the local-error
model used by the capacity argument.

For fixed per-unit incorrect-gate counts `ell_j`, the two-stage experiment's
exact survival probability is

~~~text
product_j (1 - q + q*(1 - s)**ell_j)
~~~

This is not generally `(1 - q*s)**sum(ell_j)`: the replay-unit coin correlates
the gate samples inside that unit. Staging the commitments adds no further
loss relative to this chosen `q,s` experiment.

## End-to-end research pipeline

`research_pipeline.py` connects the constructor DAG and staged verifier to the
exact explicit-DAG capacity oracle now shipped in `circuit_cut_analysis`.

~~~bash
uv run python -m prototypes.research_pipeline
uv run pytest prototypes/test_research_pipeline.py
~~~

For this first exact integration:

- `compile_circuit` returns the validated circuit, a deterministic top-level
  replay partition, and a singleton verification partition;
- `verify` draws replay units with probability \(q\), commits their interiors,
  then draws and checks singleton verification units with probability \(s\);
- `bound` enumerates every small error pattern \(E\), computes its exact
  two-stage survival probability, and queries the downstream-cut oracle for
  \(A_C(E)\);
- the returned \(U\) is
  \(\max_{E:p_\theta(E)>\eta} A_C(E)\).

The enumeration guard defaults to 20 verification units. This is an exact
finite-game seam test, not the scalable model-level algorithm. Large GPT-2
analyses use the indexed capacity oracle; Kimi-K3, DeepSeek-V4-Pro, and Inkling
currently use aggregate self-cut profiles. None of those imported model
descriptions yet supplies the concrete gate semantics and values needed for
honest execution verification.

## Advice accounting

The byte string passed to `G` in this prototype should be read as
**structural advice only**: bits that may change the compiled circuit. Those
bits select among compiled computations and are therefore charged directly in
the \(2^A\) circuit-family factor.

Other post-response choices should not be routed through `G`:

- replay-unit selection changes the commitment/replay cost, not the circuit;
- claimed boundary and unit-interior values form the assignment tested by
  sampling, so their freedom is charged through incorrect gates;
- deterministic local padding of an already-observed response supplies
  assignment values to a fixed maximum envelope; it does not select a new
  circuit.

This split is substantive. Calling all three categories “advice” and passing
them into `G` would unnecessarily charge their raw bit length at face value.
Only data that can alter `Compile(G, x, a)` belongs in structural advice.

## Not established

This prototype does not provide:

- a sandbox or proof that `G` is pure;
- compilation to intended external semantics;
- advice-dependent examples;
- a compressed repeated-call record;
- production-grade commitments, private openings, or a networked protocol;
- an inference model or cluster simulator.

Purity alone is not enough for useful memoization. Constructor helper
boundaries and cache keys must include every structural input that can affect
the emitted definition. Concrete tokens and weights should normally remain
symbolic circuit inputs; lengths, shapes, and advice that change structure must
enter the key.

## Continual-batching advice experiment

`advice_workload.py` asks how much post-response advice is needed to reconstruct
an unpadded token-inference skeleton when requests have different completion
lengths and enter and leave a continual batch.

~~~bash
uv run python -m prototypes.advice_workload
uv run pytest prototypes/test_advice_workload.py
~~~

There is no unconditional positive workload-specific lower bound. If the
model, sampling coins, and every other runtime input are in `x`, `G` can replay
inference during construction and derive an exact schedule with zero advice.
That may make compilation as expensive as inference, but the draft does not
forbid it.

The experiment instead treats completion lengths as post-`x` realizations. It
puts arrivals, prompts, token limits, and a deterministic FCFS slot scheduler
in `x`, then encodes one realized output length per request as a mixed-radix
rank. When every distinct length vector requires a distinct unpadded
structure, the exact worst-case fixed-width requirement is:

~~~text
ceil(log2(product(M_i))) bits
~~~

If zero generated tokens is allowed, replace each `M_i` by `M_i + 1`.

Once those lengths are decoded, the batch schedule is reproduced exactly with
zero additional advice. The included synthetic scenarios require 8–10 bits per
request. Allowing arbitrary client-selected batches instead would consume tens
or hundreds of thousands of additional bits in these small workloads.

A second zero-advice construction unrolls every request to its public maximum.
It requires a fixed padded output interface whose post-EOS cells are constrained
to canonical values. Keeping variable output positions while merely masking
gates does not remove the length advice.

Even if the verifier can parse lengths from the response framing, it must not
pass them to `G` as an uncharged input: they were learned after `y*` and are
therefore part of `a` under the draft's capacity argument.

The scenarios are synthetic, decode-only structural workloads, not traces from
vLLM and not GPU performance models. Prefill is excluded. The reported
zero-advice padding ratio counts token steps only; it is not a FLOP or latency
estimate. The model also assumes batch order does not change sampling or
floating-point results; otherwise schedule and token values must be modeled
jointly.

They also expose a representation issue separate from advice: every exact
ordered batch shape in the sweep was unique, while definitions keyed by one
sequence's context length had 83–93% reuse. A memoized definition DAG should
therefore cache below the batch root. Avoiding one occurrence record per decode
tick will likely require a parametric `Scan`/scheduler node rather than exact
whole-batch memoization alone.

The reuse percentages are infinite-cache shape-count proxies, not measured
serialized-byte savings. A realistic compression claim still requires passing
the trace through an inference constructor and canonical encoder.

## Sourced Kimi-2.5 cluster frontier sweep

`frontier_cluster_sweep.py` extends the advice experiment to a finite,
multi-turn inference epoch anchored on the published vLLM + Mooncake Store
Kimi-2.5 deployment: one TP4 prefill instance and one synchronized DP8+EP8
decode instance on 12 GB200 GPUs. It does not depend on the toy `call_dag.py`
representation.

~~~bash
uv run python -m prototypes.frontier_cluster_sweep
uv run python -m prototypes.frontier_cluster_sweep --trace /path/to/mooncake.jsonl
uv run pytest prototypes/test_frontier_cluster_sweep.py
~~~

The default run uses a deterministic 28-point screening design over request
count, context/output profile, turns, cache-hit target, and routing policy. It
evaluates four logical strategies (`replay_in_g`, `maximum_envelope`,
`geometric_bucket`, and `exact_logical`) and four selected physical
refinements. Pareto fronts are extracted independently for logical and physical
claims, so a semantic-only zero-advice point cannot dominate a faithful
physical replay.

Every result keeps these description components separate:

- static `G` program units and measured canonical-JSON descriptor bytes;
- finite-epoch `x` request/session metadata, prompt-token units, and block IDs;
- `a` advice bits and their exact or proxy status;
- reusable definition records;
- occurrence and routing records;
- runtime value/event cells, which are not advice;
- expanded model-token work and decode event slots;
- KV block/network units and MoE route/expert-work proxies.

The direct Mooncake FAST'25 JSONL parser accepts `timestamp`, `input_length`,
`output_length`, and 512-token `hash_ids`. Each source hash can be
conservatively namespaced into eight 64-token child IDs. Applying those older,
anonymized request shapes to Kimi-2.5 is explicitly labeled a projection, not
the measured May 2026 Codex/SWE-bench Pro benchmark.

The evidence ledger distinguishes reported facts, deployment configuration,
assumptions, exact derived counts, proxies, and unsupported metrics. In
particular, the published 3.8x/46x/8.6x outcomes are contextual observations,
not fitting targets. The cited sources do not provide absolute GB200 service
curves, cache-capacity limits, or an RDMA contention model, so offered load,
latency, throughput capacity, and GPU-hours remain unsupported rather than
being inferred from those ratios.

## No-op envelope experiment

`noop_envelope.py` isolates a direct zero-advice construction for the current
prototype primitive set: do not replay inference in `G`, do not give completion
lengths or expert choices as advice, and pad every inactive alternative with a
canonical no-op. This is not the only construction permitted by the abstract
Draft-v4 theorem, whose approved deterministic functions can be coarser than
the max-arity-2 instructions in `src/veritor/machine.py`.

~~~bash
uv run python -m prototypes.noop_envelope
uv run python -m prototypes.noop_envelope --compact --no-sweeps
uv run pytest prototypes/test_noop_envelope.py
~~~

The default Kimi-2.5 anchor has 128 requests, 900 realized output tokens per
request, a 1,024-token cap, 912,224 recomputed prefill tokens, 60 MoE layers,
384 routed experts, and top-8 routing. Its exact structural counts are:

- 1,280 bits for the unpadded `ceil(log2(M^R))` comparison, but zero length
  advice for the maximum envelope;
- 15,872 padded response tokens and exactly 63,488 materialized canonical
  token-cell padding bytes at four bytes/cell;
- 1,027,424 actual model tokens versus 1,043,296 envelope model tokens;
- 23,544,376,320 no-op expert branches, including unselected branches on
  active tokens and every branch on padded tokens.

The retained `response_padding_bytes=63,488` field means materialized
token-cell padding, not necessarily server-to-receiver communication. If the
wire response is fixed-width, those cells are also 63,488 additional wire
bytes. If the protocol instead permits a public deterministic
`Pad(y*)` after the receiver has observed ordinary variable-length `y*`, the
receiver can synthesize PAD cells locally and additional server-to-receiver
wire bytes are zero. Length remains visible in `y*`, while `C` consumes a
max-shaped canonical frame independent of that length. Local canonicalization
does not remove boundary commitment, authentication, or opening costs. Whether
this format is permitted is a protocol choice not established by the current
prototype.

The output labels five strategy dispositions. `flat_guarded_all_experts` is the
direct current-prototype construction but is huge. An
`expert_at_public_primitive` is compatible with the abstract theorem when
immutable public weights are embedded in or bound to an approved deterministic
model-specific function and the selected expert ID is a runtime router output.
It can execute top-8 without 384 explicit branches, but keeps fixed cost/static
reads and needs a realistic coarse gate/leaf definition. For mutable or
input-resident weights, `dynamic_expert_lookup` instead requires authenticated
indexed memory. `route_specialized` violates the no-route-advice constraint,
and `replay_in_g` remains inadmissible.

Gate accounting remains parameterized by `g_common`, `g_mask`, `g_router`,
`g_expert`, and `g_branch`; the prototype does not invent Kimi gate counts.
For the flat envelope, total conceptual gates are:

~~~text
T_envelope*g_common
+ D_max*g_mask
+ A_envelope*g_router
+ A_envelope*E*(g_expert + g_branch)
~~~

The report decomposes that total into useful bodies, no-op bodies, and
always-live control. No-op bodies include padded common/router work and every
unselected or all-padded expert body; active/EOS/select controls and branch
guard/aggregation remain live. Definition-DAG or serialization compression can
reduce encoded bytes without reducing conceptual `n`.

`C` consumes a fixed request-major `M`-slot canonical frame; the wire response
may be fixed-width or receiver-canonicalized as described above. `active_0` is
true, candidate tokens and the public stop/EOS policy derive `stop_t`,
`active_(t+1) = active_t AND NOT stop_t`, output uses a constrained `Select`
against canonical PAD, and state freezes after stop; the cap forces
termination. These masks are runtime circuit values, not advice or free
witnesses. The EOS/PAD boundary leaves completion length visible in `y*`, not
`a`; public padding makes `C` max-shaped but cannot erase the length already
observed by the receiver.

Protocol invariants keep even zero-cost no-ops in `n` and sampling. A false
guard does not shrink static cost/read declarations, every named read is
opened, and a repeated zero position only serves statically known outputs;
runtime-dependent positions still need selects. Live-only sampling would be a
theorem/protocol extension and must constrain mask/control gates plus their
dependency closure.

Sampling output includes Bernoulli expectations, challenged-unit probabilities,
direct Merkle-opening bytes, naive exact post-boundary padding openings, and
dense indexed-zero versus implicit-default-zero commitment counts. The
range/multiproof number is only a request-major-layout proxy; the current
index-bound Merkle implementation does not provide that optimization.

The Draft-v4 capacity object is weighted by the 32-bit wrong-gate value:

~~~text
K_B(n,L,B) = sum(comb(n,j) * 2^(jB), j=0..L)
~~~

The report computes the exact log ratio stably and includes `cell_bits`
(default 32). `L*log2(1+N0/n)` remains only a highest-`j` approximation, where
the common `2^(LB)` factor cancels.

SP1 values remain separate: 4,886 cycles / 6,859 gas for standalone no-op
execution, about 1,620 cycles per SHA path level, and about 105k cycles/s
marginal large-batch CPU throughput. The projection exposes path-only cycles
and a path-plus-floor number. Adding the standalone 4,886-cycle program floor
to every check is conservative, not a measured marginal no-op cost. There is
no measured no-op proof, and this is not a current-protocol verifier-cost
measurement.
