# Veritor research protocol API

Veritor exposes a paper-level `Compile` / `Verify` / `Bound` / `Optimize`
facade over the package's compiler, staged protocol, architecture plug-ins,
and certified analysis backends.

This is research software. The current transparent construction is useful for
conformance tests and quantitative experiments, but it is not a production
cryptographic proof system.

## Minimal use

~~~python
from fractions import Fraction

from veritor import (
    AdditiveExpectedCost,
    Bound,
    Compile,
    Optimize,
    RationalPolicyGrid,
    Unsupported,
    VerificationPolicy,
)

artifact = Compile("demo-g")
policy = VerificationPolicy(q=1, s=Fraction(1, 2), eta=Fraction(1, 4))
fixed = Bound(artifact, policy)

if isinstance(fixed, Unsupported):
    raise RuntimeError(f"{fixed.reason_code}: {fixed.detail}")

grid = RationalPolicyGrid(
    q_values=(Fraction(1, 2), 1),
    s_values=(Fraction(1, 2), 1),
    eta=Fraction(1, 4),
)
choice = Optimize(
    artifact,
    grid,
    AdditiveExpectedCost(
        boundary_cost=1,
        replay_cost=10,
        verification_cost=10,
    ),
)
~~~

Lower-case aliases `compile`, `verify`, `bound`, and `optimize` are available
from `veritor.research`. The package root reserves `veritor.compile` for the
compiler module; use top-level `Compile` for the paper-level method.

## Compile and capability scope

`Compile(architecture_id, request=None)` delegates to the immutable registry
of six built-in entries:

- `demo-g` returns `ProtocolCircuitArtifact`, a validated executable literal
  tuple `(C, R, V)`.
- `matmul` returns `ProtocolCircuitArtifact` for one or more public
  `X_i W mod 2^B` products. It stores the public shared `W` once, uses one
  replay unit per matrix multiplication, and uses one verification unit per
  output inner product.
- `gpt2` returns `IndexedStructureArtifact`, an exact indexed structural
  projection with no executable local relations.
- `kimi-k3`, `deepseek-v4-pro`, and `inkling` return
  `AggregateBoundArtifact`, counted architecture profiles without concrete
  circuit wiring.

An artifact's capability report applies only to that representation.
Structural adjacency does not imply executable semantics, and aggregate
counts do not imply a circuit, replay partition, or transcript.

DemoG and matmul support static compilation, static partitioning, static
bounding, trusted modular-word execution, and staged verification. GPT-2
supports structural compilation and conditional static bounding. The three
aggregate profiles support assumption-scoped static bounds. GPT-2, Kimi-K3,
DeepSeek-V4-Pro, and Inkling do not currently support `Execute` or `Verify`.
Their executable value schemas, rounding behavior, model weights, ordered
operand relations, and replay semantics remain deferred.

Helpers that need executable protocol semantics return `Unsupported` for
those four model artifacts. They never synthesize a tape or transcript.

## Verify and verifier-local trust

The public verifier is:

~~~python
report = Verify(transcript_bytes, expectation, trusted_context)
~~~

`Verify` is a thin wrapper around pure `verify_transcript_bytes`. It has no
mutable session state and takes all trusted inputs explicitly:

- `VerificationExpectation` fixes the session ID, compiled tuple identity,
  exact policy, exact public inputs, exact ordered claimed outputs, backend
  IDs, and both verifier-owned seeds.
- `TrustedVerificationContext` supplies a verifier-local, content-addressed
  artifact registry and allowlists of commitment and evidence backends.
- `VerificationLimits` applies parsing, transcript, proof, position, and unit
  limits. `Verify` uses the conservative package defaults when omitted.

Both `q_seed` and `s_seed` are mandatory 32-byte values in
`VerificationExpectation`. The transcript carries revealed copies, but those
copies are checked against the verifier's expectation and are never the trust
source.

`make_verification_expectation` generates independent 32-byte seeds with the
operating-system CSPRNG when they are not supplied. It also records the exact
public inputs and claimed outputs. For deployed interaction, the verifier must
generate and retain these values privately according to the phase schedule
below.

## Exact staged phase order

The protocol order is security-relevant:

1. The session, compiled tuple identity, policy, public statement, and backend
   IDs are fixed.
2. The prover fixes the boundary commitment and mandatory public-input and
   claimed-output openings.
3. Only then does the verifier reveal `q_seed`; the exact replay-unit set `J`
   is derived from the seed and boundary phase digest.
4. The prover fixes one ordered commitment root for every replay unit in `J`.
5. Only after all selected replay-unit roots are fixed does the verifier reveal
   `s_seed`; the exact verification-unit set `T` is derived from the seed and
   selected-root phase digest.
6. The prover supplies backend-tagged evidence covering exactly `T`, and the
   verifier authenticates values and checks each sampled local relation.

The verifier must keep `q_seed` secret through step 2 and `s_seed` secret
through step 4. `StagedVerifierSession` and `StagedProverSession` implement
this order in-process: the public prover context contains neither seed, and
duplicate, skipped, substituted, or out-of-order messages are rejected. A
final serialized transcript binds the resulting phase contents, but final
bytes alone cannot prove withholding. A distributed deployment must preserve
the same role separation and message order in its transport.

## Executable matmul interaction

`MatmulCompileRequest` accepts nonempty rectangular public activation
matrices, one rectangular public weight matrix, and a positive `cell_bits`
(default 8). Values must already be canonical integers in
`[0, 2^cell_bits)`. Inputs are ordered as row-major `W`, once, followed by
each row-major `X_i`; outputs are ordered by activation and then row-major.
No transpose or implicit accumulator modes are present.

Each output entry expands to exactly `k` multiply gates and `k-1` add gates.
The constructor hierarchy is
`root -> matmul -> inner-product -> scalar gate`. The trusted compiler derives
one replay unit for each top-level matmul occurrence and one verification unit
for each inner-product occurrence.

~~~python
from veritor import (
    Compile,
    MatmulCompileRequest,
    StagedProverSession,
    StagedVerifierSession,
    Unsupported,
    VerificationPolicy,
    adapt_protocol_artifact,
    create_trusted_verification_context,
    make_verification_expectation,
)

artifact = Compile(
    "matmul",
    MatmulCompileRequest(
        weights=((1, 2), (3, 4)),
        activations=(((5, 6),), ((7, 8),)),
    ),
)
resolved = adapt_protocol_artifact(artifact)
trust = create_trusted_verification_context(artifact)
expectation = make_verification_expectation(
    artifact,
    VerificationPolicy(q=1, s=1, eta=0),
)
assert not isinstance(resolved, Unsupported)
assert not isinstance(trust, Unsupported)
assert not isinstance(expectation, Unsupported)

tape = artifact.circuit.evaluate_tape(artifact.public_inputs)
verifier = StagedVerifierSession(expectation, trust)
prover = StagedProverSession(
    resolved,
    verifier.public_context,
    dict(enumerate(tape)),
)

boundary = prover.commit_boundary()
q_challenge = verifier.receive_boundary(boundary)
roots = prover.answer_replay_challenge(q_challenge)
s_challenge = verifier.receive_unit_commitments(roots)
evidence = prover.answer_sample_challenge(s_challenge)
result = verifier.receive_sample_evidence(evidence)
assert result.accepted
~~~

The default `ExecutingReplayService` discards the original full assignment
after extracting boundary values. After `q`, it evaluates only the selected
replay units gate-by-gate from those boundary values and commits only their
interiors. `replayed_unit_indices` reports the units actually replayed.
`replayed_gate_count` and `replayed_cost` expose corresponding concrete
accounting totals.

## Executable conformance utilities

The executable adaptation and local-trust helpers are:

- `adapt_protocol_artifact`
- `create_trusted_artifact_registry`
- `create_trusted_verification_context`
- `make_verification_expectation`

The adapter uses the `CallDagCircuit`'s trusted `value_codec` and
`relation_evaluator` to form `ResolvedExecutableArtifact`.

`build_executable_conformance_transcript` accepts either executable artifact,
evaluates its complete tape, forms the exact position assignment, builds
transparent staged evidence, and returns a frozen
`ExecutableConformanceTranscript` containing canonical bytes, the
expectation, and local trust. `build_demo_conformance_transcript` is the
backward-compatible DemoG wrapper:

~~~python
from veritor import Verify, build_demo_conformance_transcript

run = build_demo_conformance_transcript()
report = Verify(run.transcript_bytes, run.expectation, run.trust)
assert report.accepted
~~~

This one-shot helper is only a demo and conformance utility. It has both seeds
while constructing all phases, so it does not model an adversarially secure
interaction and must not be cited as evidence of temporal ordering.

## Transparent commitments and evidence

The bundled value commitment backend is an indexed SHA-256 Merkle tree. Its
hash frames bind values to, among other fields:

- the verifier-derived commitment domain and owner;
- the session;
- the compiled result and policy;
- the statement and phase;
- the value schema;
- the global position and local rank.

This gives position, domain, owner, session, statement, policy, and phase
binding under the usual collision-resistance assumption. The tree is
transparent and is not hiding: openings reveal values, and low-entropy values
may be guessable from roots or surrounding context.

The bundled transparent local-check evidence authenticates the exact required
values and runs trusted relations in the clear. It is not zero knowledge.

`ValueCommitmentBackend` and `SampleEvidenceBackend` are extension interfaces.
A deployment may register hiding commitments or zero-knowledge proof
backends, but the current repository does not provide or claim those
properties.

## Bound

For a policy `VerificationPolicy(q, s, eta)`, `Bound` optimizes structural
capacity over error sets whose exact two-stage survival probability is
strictly greater than `eta`. Probabilities are exact rationals; binary floats
are intentionally rejected.

### Executable call-DAG artifacts

DemoG and matmul follow their literal artifacts:

1. `StructuralCircuitCapacityOracle` constructs the exact finite downstream
   cut oracle from `C`.
2. `VerificationUnitCapacityOracle` maps attacked verification units through
   the literal `V`.
3. Exhaustive or finite branch-and-bound analysis uses the literal `R`, `V`,
   and policy.

`BoundOptions(solver="auto")` uses exhaustive analysis for small `V` and
branch-and-bound above `max_verification_units`. Complete exact-oracle runs can
return `EXACT`. A state or query limit returns a certified bracket with
`RESOURCE_LIMIT`; it is not upgraded to exact.

### GPT-2 indexed structure

GPT-2 first requests a certified gate-class catalog at the selected
granularity. If lifted-certificate conditions fail, `Bound` returns
`Unsupported` for `Capability.STATIC_BOUND`.

When a catalog is available, its `WeightedGateClassPartition` is converted to
a counted capacity schema. These classes are probability/capacity classes,
not protocol replay units. Without a separately certified replay incidence,
the counted backend uses the adversarial mega-unit relaxation described
below.

### Aggregate architecture profiles

Kimi-K3, DeepSeek-V4-Pro, and Inkling convert their weighted profile
partitions and all profile assumptions into counted schemas.

DeepSeek-V4-Pro is trace-conditional. An unbound artifact may still return an
assumption-scoped certified upper bound, but its assumptions continue to say
that selector outcomes must match an observed greedy trace. A supplied trace
digest binds provenance only; it does not execute or validate that trace and
does not add `Verify` support.

Every counted schema includes
`provenance_identity=artifact.identity.digest`. Consequently the capacity
schema and bound tuple identities change when the architecture artifact or
DeepSeek trace identity changes, even if aggregate class counts coincide.

### Counted mega-unit guarantee

When no `CountedReplayLayout` is supplied, the counted backend places every
attacked verification unit in one adversarial mega-unit. Concentrating attacks
this way can only increase survival, so the result is a certified upper
relaxation for any replay layout with the same class totals.

The result is deliberately not labeled exact for an actual protocol layout.
Its lower bound may be zero, its upper bound is capped by the designated
output frontier, and `relaxation_chain` records the capped-linear capacity
certificate and mega-unit step.

A caller may supply a separately justified `CountedReplayLayout`. If its
incidence exactly reconciles with class totals and is within the configured
materialization limit, the backend exhaustively evaluates that declared
layout. Larger layouts fall back to the certified mega-unit bracket and record
the resource limitation.

## Optimize

`Optimize` evaluates each policy in a `RationalPolicyGrid` by calling `Bound`
with the supplied `bound_options`. Expected costs use exact rational
arithmetic.

With `capacity_limit`, it minimizes expected cost among candidates whose
certified upper bound meets the limit. Without one, it minimizes certified
upper bound and then expected cost. It preserves each bound's exact,
conditional, bracketed, resource-limited, and relaxation status. If `Bound`
returns `Unsupported`, `Optimize` returns that outcome directly.

`EXACT_ON_GRID` means every objective value was solved exactly on the declared
finite grid. It is not a global continuous-policy optimum.

## Reproducing the public API checks

~~~bash
uv sync
uv run --frozen pytest tests/veritor/research_api -q
uv run --frozen ruff check src/veritor/research.py src/veritor/__init__.py tests/veritor/research_api
uv run --frozen mypy --strict src/veritor/research.py
uv run --frozen python -c "from veritor import Compile, Verify, Bound, Optimize"
~~~
