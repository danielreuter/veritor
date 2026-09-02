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
  modular semantics. `make_word_gate_set(B)` is `add`/`mul` over `Z_{2^B}`.
- **Description**: the wire format a constructor `G` produces. A hash-consed
  sequence of definitions built from three steps, `gate`, `call`, and
  `repeat`, with relative range arguments, so a transformer-sized circuit is a
  few kilobytes. Definitions may carry a mark, `replay` or `verification`.
- **Circuit** `C` (`veritor.core.Circuit`): `C[i]` gives the operator and the
  absolute argument addresses of gate `i`; inputs occupy the first addresses.
  `DescriptionCircuit` answers these queries lazily in `O(depth)`.
- **Index** `I` (`veritor.core.Index`): the hierarchy of copies over `C`.
  Every copy of a definition is a node; copies of the same definition are one
  *kind*. The marked nodes form two antichains: the replay units tile the
  gates and the verification units refine them. `I.boundary()` is
  `inputs ∪ ⋃_r Out(R_r)`; `I.interior(r)` is `R_r` minus its interface.
- **Compiled** `(C, I, digest)`: what `Compile` returns and everything else
  consumes. The digest binds the description, its marks and the gate set.
- **Policy** `θ = (q, s)`: the client's proposed sampling rates, as exact
  rationals. `η`, the acceptance threshold, belongs to the verifier
  (`VerifierParameters`), together with `U_max` and `W_max`.

## Trust boundary

The verifier and the channel are trusted; nothing else is. Constructors, the
tracer, the choice of marks, the proposed `θ`, cost labels and `Optimize` are
the client's. `Compile` re-validates every byte of a description; the index
checks that the marks tile and refine; `Verify` prices the proposal against
`W_max` before accepting a commitment. A client who chooses its units badly
pays in cost or in a larger `U`, never in soundness.

~~~text
src/veritor/
  core/          GateSet, Circuit, Index, Compiled, VerificationPolicy   (trusted)
  compile/       Compiler: description bytes -> Compiled                  (trusted)
  protocol/      the two-stage protocol, Merkle commitments, wire format  (trusted)
  analysis/      Bound, Cost, Optimize as folds over the kinds of I
  constructors/  Tracer, DemoG, MatmulG                                   (untrusted)
  research.py    the paper-level facade
src/circuit_cut_analysis/
                 exact downstream cuts on explicit DAGs (the Bound reference)
docs/            plans and handoffs
archive/         earlier incarnations, untouched
~~~

## Compile

A constructor is untrusted Python that traces a computation into a
description. `MatmulG` traces `Y_i = X_i W mod 2^B` with one replay unit per
row `x_i W` and one verification unit per output dot product; `DemoG` traces
batches of multiply-accumulate chains.

~~~python
from veritor import Compile, MatmulCompileRequest, compile_matmul, make_word_gate_set
from veritor.constructors import MatmulG

request = MatmulCompileRequest(
    weights=((1, 2), (3, 4), (5, 6)),
    activations=(((1, 2, 3), (4, 5, 6)), ((7, 8, 9),)),
    width=8,
)

# What the verifier runs: Compile on the bytes G produced.
description = MatmulG(request.width)(request.workload, b"")
compiled = Compile(description, request.public_inputs, make_word_gate_set(request.width))

assert compiled.digest == compile_matmul(request).digest  # the same thing, in one call
assert compiled.index.replay_units.count == 3
assert compiled.index.verification_unit_count == 6
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

@tracer.definition(input_count=1 + 2 * 4, key=("dot", 4), role="replay")
def dot(v):
    acc = v[0]
    for i in range(4):
        acc = mac(acc, v[1 + i], v[5 + i])
    return acc

@tracer.definition(input_count=9 * 16, key="batch")
def batch(v):
    return tracer.repeat(16, dot, v[0:9].by(9))  # 16 copies, one range argument

description = tracer.serialize(batch)
~~~

Python loops unroll; `tracer.repeat` is what keeps the description small.
Weights must be inputs, not constants: copies are the same kind only when
their values flow in as arguments.

## Verify

The protocol has three messages from the prover and two challenges from the
verifier. Values are committed in stages so the client never learns a
challenge before the values it constrains are fixed.

1. The prover commits the boundary `τ|∂` and opens the inputs and outputs;
   the verifier checks them against `x` and `y*` and reveals `q_seed`.
2. Replay units are selected at rate `q`. The prover commits the interior of
   each selected unit; the verifier reveals `s_seed`.
3. Verification units inside selected replay units are selected at rate `s`.
   The prover opens every value each selected unit reads or writes; the
   verifier recomputes the gate relations.

Selection is `Binomial(N, q)` followed by Floyd's uniform subset, so the
verifier's work is `O(K log N)` in the number `K` of selected units, never in
`N`. Model weights are committed once per model under their own root `κ_W`
and opened where sampled; a run never carries them.

~~~python
from veritor import (
    VerificationPolicy,
    VerifierParameters,
    Verify,
    compile_matmul,
    make_verification_expectation,
    run_protocol,
)
from veritor.protocol import encode_transcript

compiled = compile_matmul(request)
values = dict(enumerate(compiled.circuit.evaluate(request.public_inputs)))
outputs = tuple(values[a] for a in compiled.circuit.outputs)

expectation = make_verification_expectation(       # the verifier's side of one run
    compiled,
    VerificationPolicy(q=1, s=1),                  # the client's proposal, theta
    request.public_inputs,
    outputs,
    parameters=VerifierParameters(eta=0),          # eta, U_max, W_max are the verifier's
)
run = run_protocol(compiled, expectation, values)  # prover and verifier in one process
assert run.report.accepted
assert Verify(encode_transcript(run.transcript), expectation, compiled) == run.report
~~~

`ProverSession` and `VerifierSession` are the two state machines behind
`run_protocol`; `Verify` (`verify_transcript`) re-derives both challenges from
the verifier's seeds and checks a recorded transcript purely. The header binds
`θ` and the verifier's `η`, so a transcript recorded under another `η` is
rejected.

## Bound, Cost, Optimize

`Bound(C, I, θ)` certifies `U`, a bound in bits on the outputs an adversary
can reach with acceptance probability above `η`. It is a fold over the kinds
of `I`: every error set is assigned a cover by index nodes, the reachable
outputs of a cover are at most `2^{Σ out_bits}` (the downstream cut), and the
distinct covers of admissible error sets are summed. Admissibility is a
knapsack over replay units against the budget `ln(1/η)`, solved on a cost
grid that only ever admits more; a grid-free Laplace bound is taken alongside
and the smaller reported. No copy is ever enumerated: a `10^8`-gate
transformer index bounds in milliseconds.

~~~python
from fractions import Fraction
from veritor import Bound, Cost, CostParameters, Optimize, PolicyGrid, VerificationPolicy

theta = VerificationPolicy(Fraction(1, 2), Fraction(1, 2))
eta = Fraction(1, 100)                                  # the verifier's threshold
print(Bound(compiled, theta, eta).bits)                 # U in bits
print(Cost(compiled, theta, CostParameters(hash_cost=1, proof_overhead=0)).total)

best = Optimize(compiled, eta, PolicyGrid.uniform(8), max_bits=20)
~~~

`Cost` is the exact expectation
`h|∂| + q Σ_r (replay(R_r) + h|Int(r)|) + q s Σ_v (proof(V_v) + c_0)`.
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
