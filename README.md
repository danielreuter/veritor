# veritor

Veritor is a typed research implementation of the paper-level
`Compile` / `Verify` / `Bound` / `Optimize` stack for sampled computation
verification and capacity analysis.

The current construction is research software, not a production
cryptographic proof system. Its indexed SHA-256 Merkle commitments are
transparent and binding-oriented, not hiding. Its transparent local checks
are not zero knowledge.

## Public research stack

`Compile` delegates to a closed registry containing two executable workloads
(DemoG and shared-weight modular matmul) and four model representations
(GPT-2, Kimi-K3, DeepSeek-V4-Pro, and Inkling). Each result reports its actual
capability scope; a structural or aggregate artifact never acquires fake
execution semantics.

~~~python
from fractions import Fraction

from veritor import Bound, Compile, Unsupported, VerificationPolicy

artifact = Compile("demo-g")
policy = VerificationPolicy(
    q=1,
    s=Fraction(1, 2),
    eta=Fraction(1, 4),
)
result = Bound(artifact, policy)

if isinstance(result, Unsupported):
    raise RuntimeError(f"{result.reason_code}: {result.detail}")

print(result.claim_strength, result.lower_bound_bits, result.upper_bound_bits)
~~~

`Verify` is pure: canonical transcript bytes are checked only against a
mandatory verifier-local `VerificationExpectation`, verifier-local trusted
artifact and backend registries, and explicit resource limits.

~~~python
from veritor import Verify, build_demo_conformance_transcript

run = build_demo_conformance_transcript()
report = Verify(run.transcript_bytes, run.expectation, run.trust)
print(report.status, report.code)
~~~

That one-shot builder is a conformance utility. The phase-separated API below
uses distinct prover and verifier state machines. The verifier keeps `q_seed`
until the boundary is accepted and keeps `s_seed` until all selected replay
roots are accepted.

## Executable modular matmul

`Compile("matmul", request)` computes a nonempty list of
`Y_i = X_i W mod 2^B`. `W` is stored once in the public input view and shared
by every multiplication. There is one replay unit per `X_i W` and one
verification unit per output inner product.

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

request = MatmulCompileRequest(
    weights=((1, 2), (3, 4), (5, 6)),
    activations=(
        ((1, 2, 3), (4, 5, 6)),
        ((7, 8, 9),),
    ),
    cell_bits=8,
)
artifact = Compile("matmul", request)
resolved = adapt_protocol_artifact(artifact)
trust = create_trusted_verification_context(artifact)
expectation = make_verification_expectation(
    artifact,
    VerificationPolicy(q=1, s=1, eta=0),
)
assert not isinstance(resolved, Unsupported)
assert not isinstance(trust, Unsupported)
assert not isinstance(expectation, Unsupported)

# Initial execution fixes public outputs and boundary values.
tape = artifact.circuit.evaluate_tape(artifact.public_inputs)
assignment = dict(enumerate(tape))

verifier = StagedVerifierSession(expectation, trust)
prover = StagedProverSession(
    resolved,
    verifier.public_context,  # contains neither q_seed nor s_seed
    assignment,
)

boundary = prover.commit_boundary()
q_challenge = verifier.receive_boundary(boundary)
unit_roots = prover.answer_replay_challenge(q_challenge)
s_challenge = verifier.receive_unit_commitments(unit_roots)
evidence = prover.answer_sample_challenge(s_challenge)
result = verifier.receive_sample_evidence(evidence)

assert result.accepted
assert artifact.execute() == (22, 28, 49, 64, 76, 100)
assert prover.replayed_unit_indices == (0, 1)
~~~

The prover retains boundary values, then recomputes only replay units selected
by `q`. It retains commitment material only for those units while producing
sampled evidence. The included Merkle commitments and local checks are still
transparent and non-ZK.

`Bound` dispatches by artifact:

- DemoG and matmul use their literal executable circuits and literal replay and
  verification partitions with exact finite exhaustive or branch-and-bound
  analysis.
- GPT-2 uses a certified gate-class catalog when its lifted certificate
  conditions hold. Otherwise it returns typed `Unsupported` for static
  bounding.
- Kimi-K3, DeepSeek-V4-Pro, and Inkling use assumption-preserving counted
  capacity schemas.
- Counted artifacts default to a certified adversarial mega-unit upper
  relaxation. This is not an exact claim about an unknown replay layout.
- DeepSeek bounds remain conditional on an observed greedy sparse-execution
  trace. A trace digest binds provenance but does not execute or verify it.

Every counted schema binds the complete architecture artifact identity, so a
different architecture or trace identity changes the bound identity.
`Optimize` evaluates exact rational finite policy grids through this same
`Bound` dispatch and preserves exact, bracketed, conditional, relaxed, and
resource-limited statuses.

The full API contract, exact phase order, trust boundary, and guarantee
language are in [docs/research-protocol.md](docs/research-protocol.md).

## Running the research API

~~~bash
uv sync
uv run --frozen pytest tests/veritor/research_api -q
uv run --frozen ruff check src/veritor/research.py src/veritor/__init__.py tests/veritor/research_api
uv run --frozen mypy --strict src/veritor/research.py
uv run --frozen python -c "from veritor import Compile, Verify, Bound, Optimize"
~~~

## Legacy v0 tape API

The original sampled execution-tape protocol remains supported as a compact
teaching compatibility layer. A prover runs a scalar JAX program, commits to
its instruction and value tapes, and a verifier checks a random sample of
steps. It illustrates quantified detection and fixed-width capacity
accounting; it is not full computation correctness.

The legacy objects are:

- `tracer.py`: `jax.make_jaxpr(f)` flattened into fixed-width SSA
  instructions.
- `machine.py`: one 32-bit float32 value cell per input and instruction.
- `merkle.py`: the original domain-separated, index-bound SHA-256 tape trees.
- `protocol.py`: commit, boundary checks, random challenges, and local checks.

The **machine model** is the load-bearing choice:

- **append-only dataflow**: instruction `k` may only read cells written
  before its own write. A local check of one instruction is then meaningful
  in isolation — no global "did memory change under me" reasoning needed.
- **fixed-width cells**: every step commits exactly 32 bits. An instruction
  that escapes sampling can inject at most those 32 bits. This is what makes
  the leakage bound *exact* rather than asymptotic.
- **finite primitive library**: `const / add / sub / mul / neg / exp / tanh /
  max` (`machine.PRIMITIVES`). Opcodes are just names of functions in a
  library both sides share — the same role JAX primitives play for XLA.
- **constant materialization**: scalar literals are not inline immediates on
  the consuming instruction; each distinct constant is written to its own
  tape cell by a `const` instruction (as in MLIR's `arith.constant` or HLO's
  constant), deduplicated by bit pattern. Every other instruction reads only
  cells, and every value cell has a producing instruction.

## Legacy v0 protocol

~~~text
Prover                                     Verifier
------                                     --------
trace f -> instruction tape                trace f itself (provenance, v0)
execute  -> value tape
commit: (instr_root, value_root, y*)  -->  check instr_root == own re-trace
                                           check input cells == public x   (exact)
                                           check output cells == claimed y* (exact)
                                     <--   sample s instruction indices
open instruction k:                   -->  authenticate all openings
  instr leaf + write cell + read cells     re-apply the primitive
                                           accept iff every check passes
~~~

A cheating prover who forges `L` cells (and honestly propagates them
downstream, so exactly `L` instructions are locally inconsistent) survives
`s` uniform samples with probability

~~~text
P[miss] = (1 - L/N)^s
~~~

and every miss buys at most `32·L` unexplained bits. `experiment.py`
measures the empirical catch rate against this curve; they match.

## Running the legacy teaching API

~~~bash
uv sync
uv run python -m veritor.demo         # end-to-end run: honest prover + one cheat
uv run python -m veritor.experiment   # detection rate vs 1-(1-L/N)^s
uv run pytest                         # packaged veritor + circuit-cut suites
uv run pytest tests prototypes        # include isolated research prototypes
~~~

Write any scalar float32 function with Python/JAX ops from the library and
feed it through:

~~~python
import jax.numpy as jnp, numpy as np
from veritor import Prover, Verifier, run_protocol, trace

f = lambda x: jnp.exp(x * (x + 1.0)) + 2.0
x = np.float32(0.7)

program = trace(f, x)                     # the compiler is jax.make_jaxpr
prover = Prover(program, [x])
verifier = Verifier(trace(f, x), [x])     # only public data; commitment arrives in-protocol
print(run_protocol(prover, verifier, num_samples=8).accepted)
~~~

## Deliberately omitted from legacy v0

- **Tensors / higher-order instructions.** v0 is scalar-only so that "one
  instruction = one 32-bit cell" is literally true. The research stack now
  includes modular matrix multiplication expanded into fixed-width scalar
  gates, but the legacy JAX tape API remains scalar-only.
- **Proved compilation.** The v0 verifier re-traces `f` itself to check the
  instruction root. The real design replaces this with a proof that the
  tape is the honest compilation of the public program.
- **A proved non-interactive transform.** The staged research protocol uses
  verifier-owned interactive seeds. A secure non-interactive transform needs
  a separately specified and analyzed construction; it is not supplied here.
- **A hiding or zero-knowledge proof system.** Openings and local checks are
  transparent. `docs/sp1-benchmark-plan.md` is a benchmark plan, not an
  implemented security claim.

## Research integration

The repository also ships `circuit_cut_analysis`, the exact structural
capacity package for explicit DAGs, lazy indexed GPT-2 circuits, architecture
profiles, and finite/count-based sampling games. Its CLI remains available as
`uv run circuit-cut`.

The packaged `veritor.research` facade is the supported bridge across these
modules. DemoG and matmul compile memoized constructor DAGs into trusted
executable circuits, derive concrete replay and verification partitions, run
the staged two-stage \(q,s\) verifier, and evaluate finite error sets with the
exact downstream-cut oracle.

The prototype command remains available for exploratory comparison:

~~~bash
uv run python -m prototypes.research_pipeline
uv run pytest prototypes/test_research_pipeline.py
~~~

DemoG deliberately uses exact 8-bit word arithmetic. Matmul uses configurable
unsigned `B`-bit modular words (default `B=8`). The imported GPT-2
circuit is gate-addressable but structural only, while the Kimi-K3,
DeepSeek-V4-Pro, and Inkling integrations are aggregate capacity profiles.
Those four model artifacts currently support capacity analysis, not honest
end-to-end execution verification.

## Layout

~~~text
src/veritor/
  research.py    Compile / Verify / Bound / Optimize public facade
  plugins/       closed six-entry compile registry
  staged/        backend-neutral two-seed transcript and session protocol
  analysis/      finite and counted certified bound backends
  compile/       trusted call-DAG compiler, DemoG, and matmul semantics
  merkle.py      SHA-256 commitment layer (domain-separated, index-bound)
  machine.py     cells, primitive library, instruction encoding, interpreter
  tracer.py      jax.make_jaxpr -> Program (the "compiler")
  protocol.py    Prover, Verifier, Transcript, run_protocol
  demo.py        python -m veritor.demo
  experiment.py  python -m veritor.experiment
src/circuit_cut_analysis/
                  exact cuts, indexed model circuits, and sampling bounds
prototypes/       constructor, staged commitment, and integration experiments
tests/            packaged veritor and circuit-cut suites
docs/             protocol guarantees, plans, and handoffs
archive/         the previous incarnation of this repo, untouched
~~~
