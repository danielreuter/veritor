# veritor

Research implementation of sampled verification for large circuits: the
paper-level `Compile` → `Verify` / `Bound` / `Cost` → `Optimize` stack, with an
executable two-stage protocol and a capacity bound that never touches a gate
of the circuit it bounds.

This is research software. Commitments are transparent SHA-256 Merkle trees
(binding, not hiding) and the sampled checks are performed in the clear by the
verifier. The transparent verifier is the specification; a zero-knowledge
construction replaces each value with a commitment and each check with a proof
at the same point in the interaction.

## Objects

- **Gate set** `Σ` (`veritor.core.GateSet`): the public operators. Each gate
  has an arity, an output width in bits, a replay cost, a proof cost, and
  modular semantics. `make_word_gate_set(B)` is `add`/`mul` over `Z_{2^B}`
  plus two zero-arity *source* gates, `in` and `weight`, whose values come
  from the environment rather than from a relation; `make_isa_gate_set(B)`
  adds `sub`, `lt`, `eq`, `shr` for the toy decoder.
- **Description**: the wire format a constructor `G` produces. A hash-consed
  sequence of definitions built from three steps, `gate`, `call`, and
  `repeat`, with relative range arguments, so a transformer-sized circuit is a
  few kilobytes. Definitions may carry a mark, `replay` or `verification`.
- **Circuit** `C` (`veritor.core.Circuit`): `C[i]` gives the operator and the
  absolute argument addresses of gate `i`. The circuit's inputs are its `in`
  gates and its weights its `weight` gates: they sit inside units like any
  other gate, so the root definition has no ports (ports remain the relative
  addressing mechanism *inside* the hierarchy). `C.inputs` and `C.weights`
  list their addresses by rank. `DescriptionCircuit` answers these queries
  lazily in `O(depth)`.
- **Index** `I` (`veritor.core.Index`): the hierarchy of copies over `C`.
  Every copy of a definition is a node; copies of the same definition are one
  *kind*. The marked nodes form two antichains: the replay units tile the
  gates -- every gate, source gates included, lies in exactly one replay unit
  and one verification unit -- and the verification units refine them.
  `I.inputs()` and `I.weights()` are the source gates by rank; `I.boundary()`
  is `In ∪ ⋃_r Out(R_r)`; `I.interior(r)` is `R_r` minus its interface minus
  its source gates. `In` and `Out` of a unit are its *declared* interfaces,
  held per kind as arithmetic runs of addresses, so the per-kind table, the
  boundary and the interiors cost by the number of runs, never by the
  addresses they span; `Out` excludes a unit's *pinned* (source) gates, which
  are boundary or `κ_W` positions already.
- **Constructor** `G` (`veritor.compile.Constructor`): the client's code, an
  object with a versioned `digest` and `G(x, a) -> (description, inputs)`:
  the description bytes for the request's public inputs `x` and advice `a`,
  and `x` as the values of the `in` gates by rank.
- **Compiled** `(C, I, digest)`: the circuit with its index. The digest binds
  the description, its marks and the gate set.
- **Compilation**: what `Compile(G, x, a)` returns and the verifier keeps:
  `compiled`, `constructor` (`G.digest`), `inputs` and `advice`, charged at
  `advice_bits = 8|a|`.
- **Policy** `θ = (q, s)`: the client's proposed sampling rates, as exact
  rationals. `η`, the acceptance threshold, belongs to the verifier
  (`VerifierParameters`), together with `U_max`, `A = max_advice_bits` and
  `W_max`. `U_max` has no default: a verifier states it, and waiving it
  (`max_capacity=None`) has to be written out.

## Trust boundary

The verifier and the channel are trusted; nothing else is. Constructors, the
tracer, the choice of marks, the advice, the proposed `θ`, cost labels and
`Optimize` are the client's. `Compile(G, x, a)` runs `G` and re-validates
every byte of the description it produces; the index checks that the marks
tile and refine; the header binds `G`'s digest and `a`; `Verify` checks the
advice against `A`, the proposal against `U_max` (by `Bound`) and the work
against `W_max` before accepting a commitment, so every accepted request has
capacity at most `U_max + A`. In this prototype `G` is ordinary Python the
verifier executes, identified by a versioned digest like the gate set; a
deployment would run it sandboxed and metered, or have the client prove
`Compile(G, x, a) = (C, I)`. A client who chooses its units badly pays in cost
or in a larger `U`, never in soundness. `docs/security-argument.md` states,
component by component, what is claimed, why the code provides it, which
attack test in `tests/veritor/security/` shows the failure mode caught, and
what is only achieved by convention.

~~~text
src/veritor/
  core/          GateSet, Circuit, Index, Compiled, VerificationPolicy   (trusted)
  compile/       Compiler: description bytes -> Compiled; Constructor,    (trusted)
                 Compilation: the record of one Compile(G, x, a)
  protocol/      the two-stage protocol, Merkle commitments, wire format  (trusted)
  analysis/      Bound, Cost, Optimize as folds over the kinds of I
  constructors/  Tracer, DemoG, MatmulG, the toy LM and ClusterG         (untrusted)
  research.py    the paper-level facade
src/circuit_cut_analysis/
                 exact downstream cuts on explicit DAGs (the Bound reference)
docs/            plans and handoffs
archive/         earlier incarnations, untouched
~~~

## Compile

A constructor is untrusted Python that traces a computation into a
description. `MatmulG` traces `Y_i = X_i W mod 2^B` with the activations and
the weights each a replay unit of source gates (every gate its own
verification unit), one replay unit per row `x_i W` and one verification unit
per output dot product; `DemoG` traces batches of multiply-accumulate chains.
`Compile(G, x, a)` is the verifier's: it runs `G` on the request's public
inputs and the client's advice, compiles the bytes `G` produced and records
what it ran on. The advice is admitted up to `max_advice_bits` and charged at
`8|a|` bits on top of `Bound`; a `G` that raises is a `CompileError`, a
rejection, never a crash.

~~~python
from veritor import Compile, MatmulCompileRequest, compile_matmul, make_word_gate_set
from veritor.constructors import MatmulG

request = MatmulCompileRequest(
    weights=((1, 2), (3, 4), (5, 6)),
    activations=(((1, 2, 3), (4, 5, 6)), ((7, 8, 9),)),
    width=8,
)

# What the verifier runs: G on (x, a), then the compiler on the bytes G produced.
gate_set = make_word_gate_set(request.width)
compilation = Compile(MatmulG(request.width), request.workload, b"", gate_set)
compiled = compilation.compiled

assert compilation.constructor == MatmulG(request.width).digest
assert compilation.inputs == request.public_inputs      # x as the `in` gates consume it
assert compilation.advice_bits == 0                     # MatmulG takes no advice
assert compiled.digest == compile_matmul(request).compiled.digest  # the same, in one call
assert compiled.index.replay_units.count == 2 + 3        # activations, weights, three rows
assert compiled.index.verification_unit_count == 9 + 6 + 6  # in gates, weight gates, dots
assert (compiled.index.input_count, compiled.index.weight_count) == (9, 6)
~~~

Writing a constructor:

~~~python
from veritor.constructors import Tracer
from veritor.core import make_word_gate_set

tracer = Tracer(make_word_gate_set(8))
add, mul = tracer.gate("add"), tracer.gate("mul")

@tracer.definition(input_count=3, key="mac", role="verification")
def mac(v):
    return add(v[0], mul(v[1], v[2]))

@tracer.definition(input_count=0, key=("dot", 4), role="replay")
def dot(_v):
    x = tracer.inputs(9)  # nine `in` gates: the accumulator, four values, four weights
    acc = x[0]
    for i in range(4):
        acc = mac(acc, x[1 + i], x[5 + i])
    return acc

@tracer.definition(input_count=0, key="batch")
def batch(_v):
    return tracer.repeat(16, dot)  # 16 copies; the root has no ports

description = tracer.serialize(batch)
~~~

Python loops unroll; `tracer.repeat` is what keeps the description small.
`tracer.inputs(n)` and `tracer.weights(n)` emit `n` source gates as one
`repeat` of a one-gate verification-marked cell, so a block of `10^9` weights
is `O(1)` description; source gates must sit inside a replay unit like every
other gate. Values must flow in as source gates or arguments, not constants:
copies are the same kind only when their values do. A definition's declared
outputs must be distinct gates: `Compile` rejects a definition whose output
ranges resolve to the same gate (source gates included) twice.

## A toy inference cluster

`make_isa_gate_set(B)` is the toy ISA: `add`, `sub`, `mul`, `lt`, `eq`, `shr`
over `Z_{2^B}` beside the `in`/`weight` sources -- what a decoder needs and
nothing that approximates real arithmetic.  `veritor.constructors.lm` traces
a shape-faithful decoder-only transformer over it (`LMShape`, `Parameters`,
the sequential oracle `reference_generate`): token embedding by one-hot
against a constant table, per layer `q, k, v` matvecs, attention over the KV
cache with the polynomial softmax `w_j = s_j^2` and a right shift, an MLP
with squares, an argmax chain for the LM head.  Constants are weights: the
grammar has no immediates.  Toy numerics, faithful structure.

`ClusterG(shape, pods, slots, steps)` runs it as a cluster with continual
batching.  The public inputs `x` are the requests; the advice `a` is a
`Schedule` -- the joins of requests to slots -- and it fixes everything about
the circuit that `x` does not: which prompts are prefilled together, how many
slots each decode step batches, where a request is cut short.  The root calls
the `weights` unit once and then one `step` per occupied `(pod, step)`; a
step is a replay unit ("replay decode step `t` of pod `p`") whose declared
outputs are the new KV-cache entries and tokens the next step reads through
ports, so the KV cache is exactly the cross-step state.  Verification units
are row-sized kinds: `dot_k`, `onehot`, `attend_head_c`, `argmax`, the
residual and square cells.  Two steps with the same occupant shapes are one
kind, so the description is `O(distinct prompt lengths + distinct context
lengths + distinct step shapes + |schedule|)`, and the circuit's outputs equal
`reference_generate` for every schedule: batching is semantically transparent.

~~~python
from veritor import Compile
from veritor.constructors import ClusterG, LMShape, Request, random_parameters, schedule_fcfs
from veritor.core import make_isa_gate_set

shape = LMShape(vocab=8, d_model=4, heads=2, layers=1, context=6, width=16)
requests = (Request((1, 2, 3), max_new=3), Request((5,), max_new=2))
advice = schedule_fcfs(requests, pods=1, slots=2, steps=3).encode()   # the client's schedule
compilation = Compile(ClusterG(shape, 1, 2, 3), requests, advice, make_isa_gate_set(16), max_advice_bits=1024)
circuit = compilation.compiled.circuit
values = circuit.evaluate(compilation.inputs, random_parameters(shape, seed=0).flatten())
~~~

## Verify

The protocol has three messages from the prover and two challenges from the
verifier. Values are committed in stages so the client never learns a
challenge before the values it constrains are fixed.

1. The prover commits the boundary `τ|∂`, `∂ = In ∪ ⋃_r Out(R_r)`, and opens
   every input gate and every output; the verifier checks them against `x`
   (by rank) and `y*` and reveals `q_seed`.
2. Replay units are selected at rate `q`. The prover commits the interior of
   each selected unit; the verifier reveals `s_seed`.
3. Verification units inside selected replay units are selected at rate `s`.
   The prover opens every value each selected unit reads or writes; the
   verifier recomputes the gate relations, checks each input gate against
   `x` and each weight gate by its opening under `κ_W` at its rank.

Selection is `Binomial(N, q)` followed by Floyd's uniform subset, so the
verifier's work is `O(K log N)` in the number `K` of selected units, never in
`N`. Model weights are the circuit's `weight` gates. They are committed once
per model under their own root `κ_W`, whose leaves are the model's weight
vector by rank: leaf `k` is the `k`-th weight, read by the `k`-th `weight`
gate in address order of whichever circuit a request compiles to. The domain
is bound to the gate set, not to a description, so one root serves every
request's circuit (continual batching compiles a different circuit per
request from the same model); a sampled weight gate is opened at its rank, a
run never carries the weights, and they are never boundary or interior
positions.

~~~python
from veritor import (
    VerificationPolicy,
    VerifierParameters,
    Verify,
    compile_matmul,
    make_verification_expectation,
    run_protocol,
)
from veritor.protocol import commit_weights, encode_transcript

compilation = compile_matmul(request)              # Compile(MatmulG, workload, b"")
compiled = compilation.compiled
values = dict(enumerate(compiled.circuit.evaluate(compilation.inputs, request.weight_values)))
outputs = tuple(values[a] for a in compiled.circuit.outputs)
weights, weight_tree = commit_weights(gate_set, request.weight_values)  # kappa_W, once per model

expectation = make_verification_expectation(       # the verifier's side of one run
    compilation,                                   # (C, I), G's digest, x by rank and a
    VerificationPolicy(q=1, s=1),                  # the client's proposal, theta
    outputs,
    parameters=VerifierParameters(eta=0, max_capacity=0),  # eta, U_max, A, W_max are the verifier's
    weights=weights,
)
run = run_protocol(compiled, expectation, values, weight_tree=weight_tree)
assert run.report.accepted
assert Verify(encode_transcript(run.transcript), expectation, compiled) == run.report
~~~

`ProverSession` and `VerifierSession` are the two state machines behind
`run_protocol`; `Verify` (`verify_transcript`) re-derives both challenges from
the verifier's seeds and checks a recorded transcript purely. The header binds
`(C, I)`, `G`'s digest, the advice `a`, `θ` and the verifier's `η`, so a
transcript recorded under another `G`, `a` or `η` is rejected; advice longer
than `max_advice_bits` is rejected at admission, before any commitment.

## Bound, Cost, Optimize

`Bound(C, I, θ)` certifies `U`, a bound in bits on the outputs an adversary
can reach with acceptance probability above `η`. It is a fold over the kinds
of `I`: every error set is assigned a cover by index nodes, the reachable
outputs of a cover are at most `2^{Σ min(out_bits, reach_bits)}` (a node's
interface and the circuit outputs it can reach are both downstream cuts; the
reach is computed at step granularity, so a replay unit (RU) or verification
unit (VU) inside one request of a serving run is charged that request's
tokens, not its interface), and the
distinct covers of admissible error sets are summed. Admissibility is a
knapsack over replay units against the budget `ln(1/η)`, solved on a cost
grid that only ever admits more; a grid-free Laplace bound is taken alongside
and the smaller reported. No copy is ever enumerated: a `10^8`-gate
transformer index bounds in milliseconds.

~~~python
from fractions import Fraction
from veritor import Bound, Capacity, Cost, CostParameters, Optimize, PolicyGrid, VerificationPolicy

theta = VerificationPolicy(Fraction(1, 2), Fraction(1, 2))
eta = Fraction(1, 100)                                  # the verifier's threshold
print(Bound(compiled, theta, eta).bits)                 # U in bits
print(Capacity(compilation, theta, eta))                # U + 8|a|: what the paper charges
print(Cost(compiled, theta, CostParameters(hash_cost=1, proof_overhead=0)).total)

best = Optimize(compiled, eta, PolicyGrid.uniform(8), max_bits=20)
~~~

`Capacity(Compile(G, x, a), θ, η)` is `Bound(C, I, θ) + 8|a|`: beyond the
degrees of freedom `Bound` leaves uncharged in the circuit, the only freedom
the client has is the advice, everything else being a deterministic function
of `(G, x, a)`. With `U_max` and `A` enforced at admission, every accepted
request has capacity at most `U_max + A`.

`Cost` is the exact per-request expectation
`h|∂| + Recompute + q Σ_r h|Int(r)| + q s Σ_v (proof(V_v) + c_0)`, with
`|∂| = |In| + Σ_r |Out(R_r)|`. `Recompute` assumes the honest prover retains
only the circuit inputs and the weights: a sampled replay unit whose ports
are all fed by source gates (a *closed* kind, `KindSummary.closed`) costs its
own `replay(R_r)`, so the term is `q Σ_r replay(R_r)` when every unit is
closed; any other sampled unit forces the re-execution of the smallest closed
kind containing it, `Σ_A copies_A (1 - (1 - q)^{m_A}) replay(A)` over closed
kinds `A` with `m_A` such units under them (nested closed kinds are charged
only when the outer one is not re-executed). `ExpectedCost` reports
`recompute` and `commit_interior` separately (`replay` is their sum); the
per-epoch weight commitment `h|W|` is reported separately
(`ExpectedCost.weights`) and is not in the total.
`Optimize` is the client's advisory grid search; the verifier only checks the
result against `U_max` and `W_max`.

## Running

~~~bash
uv sync
uv run pytest tests -q
uv run ruff check src tests
~~~

The tests in `tests/veritor/analysis` check the fold against exhaustive
enumeration of error sets on small circuits and against the explicit cut
oracle in `circuit_cut_analysis`; `tests/veritor/protocol/test_scaling.py`
checks that verifier time is flat in the number of gates.
