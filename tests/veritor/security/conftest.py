"""Every protocol fixture the security tests use, in one module.

The tests in this directory attack one component of the protocol each.  All
of them build their circuits, expectations, headers, sessions, dishonest
provers and byte-level transcript mutations through the helpers here, so
that a change to the protocol API (header fields, the expectation factory,
the shape of ``kappa_W``) is a one-file adaptation.

The fixture circuit is a *chain*: a replay unit of source gates (``cells``
inputs and ``cells`` weights, each its own verification unit) followed by
``stages`` replay units of ``cells`` verification units each.  Cell ``i`` of
a stage computes ``u_i = add(mul(x_i, w_i), x_{i+1})`` from the previous
stage's outputs ``x`` and the weights ``w``; its ``mul`` is an interior
position, its ``add`` a declared output of the stage.  The last stage's
outputs are the circuit's outputs, so every other stage's outputs are
boundary positions that are *not* public I/O.

The test suite runs with ``--import-mode=importlib``, so test modules reach
this module through the ``sec`` fixture (the module itself) rather than by
importing it; stable protocol types (messages, codes) they import directly.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from types import ModuleType

import pytest

from veritor.compile import Compiler
from veritor.compile.description import (
    FORMAT_VERSION,
    canonical_description,
    definition_digest,
)
from veritor.constructors import Tracer
from veritor.core import (
    Compiled,
    GateSet,
    VerificationLimits,
    VerificationPolicy,
    canonical_json_bytes,
    iter_domain,
    make_word_gate_set,
)
from veritor.protocol import (
    BOUNDARY_OWNER,
    BoundaryMessage,
    CommitmentDomain,
    EvidenceMessage,
    Expectation,
    Header,
    InteriorMessage,
    MerkleTree,
    Opening,
    ProtocolRun,
    ProverSession,
    Reject,
    ReplayChallenge,
    SampleChallenge,
    Transcript,
    VerifierParameters,
    VerifierSession,
    Weights,
    boundary_domain,
    commit_weights,
    derive_replay_selection,
    derive_sample_selection,
    encode_transcript,
    interior_domain,
    make_expectation,
    verify_transcript,
    weight_domain,
)
from veritor.protocol.domains import leaf_schema
from veritor.protocol.phases import (
    boundary_phase,
    interior_phase,
    replay_phase,
    sample_phase,
)
from veritor.protocol.session import Replay, Values, assignment_replay, rejection_report

Q_SEED = b"Q" * 32
S_SEED = b"S" * 32
SESSION_ID = b"tests/veritor/security"
CHECK_EVERYTHING = VerificationPolicy(1, 1)
HALF = Fraction(1, 2)
HALVES = VerificationPolicy(HALF, HALF)
LIMITS = VerificationLimits()


def seed(label: str, index: int = 0) -> bytes:
    """A deterministic 32-byte seed for a test."""

    return hashlib.sha256(f"veritor/security/{label}/{index}".encode()).digest()


# -- the fixture circuit -------------------------------------------------------


def chain_description(
    stages: int = 2, cells: int = 2, width: int = 8, *, wide_units: bool = False
) -> tuple[GateSet, bytes]:
    """The chain circuit's gate set and canonical description bytes.

    With ``wide_units`` a stage's cells form one verification unit instead of
    one per cell: the same gates at the same addresses under other marks.
    """

    gate_set = make_word_gate_set(width)
    tracer = Tracer(gate_set)
    add, mul = tracer.gate("add"), tracer.gate("mul")

    def cell_gates(x, x_next, w):
        product = mul(x, w)  # interior
        return add(product, x_next)  # Out of the stage

    @tracer.definition(input_count=3, key="cell", role="verification")
    def cell(v):
        return cell_gates(v[0], v[1], v[2])

    @tracer.definition(input_count=2 * cells, key=("wide", cells), role="verification")
    def wide(v):
        x, w = v[:cells], v[cells:]
        return [cell_gates(x[i], x[(i + 1) % cells], w[i]) for i in range(cells)]

    @tracer.definition(input_count=2 * cells, key=("stage", cells, wide_units), role="replay")
    def stage(v):
        x, w = v[:cells], v[cells:]
        if wide_units:
            return wide(*x, *w)
        return [cell(x[i], x[(i + 1) % cells], w[i]) for i in range(cells)]

    @tracer.definition(input_count=0, key=("sources", cells), role="replay")
    def sources(_v):
        return tracer.inputs(cells), tracer.weights(cells)

    @tracer.definition(input_count=0, key=("root", stages, cells, wide_units))
    def root(_v):
        s = sources()
        x, w = s[0:cells], s[cells : 2 * cells]
        for _ in range(stages):
            x = stage(x, w)
        return x

    return gate_set, tracer.serialize(root)


def chain_compiled(
    stages: int = 2, cells: int = 2, width: int = 8, *, wide_units: bool = False
) -> Compiled:
    """The chain circuit described in the module docstring."""

    gate_set, description = chain_description(stages, cells, width, wide_units=wide_units)
    return Compiler(gate_set).compile(description, [0] * cells)


def random_marked_compiled(
    seed: int, marking: int, width: int = 2, gates: int = 6
) -> Compiled:
    """One random gate graph (``seed``) under one random marking (``marking``).

    The graph: ``gates`` two-argument gates over 2-3 inputs, each reading two
    earlier values.  The marking, which a client chooses freely: how the gate
    sequence is cut into replay units and those into verification units, and
    whether each unit declares only what is read outside it (a minimal
    interface) or every gate (a maximal one).  The circuit's outputs are the
    gates nothing reads.
    """

    graph = random.Random(seed)
    inputs = graph.randint(2, 3)
    ops = [graph.choice(["add", "mul"]) for _ in range(gates)]
    args = [  # one argument is recent (so units have interior values), one is anything earlier
        (graph.randrange(max(0, inputs + i - 2), inputs + i), graph.randrange(inputs + i))
        for i in range(gates)
    ]
    # value ids: 0..inputs-1 are inputs, inputs+i is gate i
    read_by = {inputs + i: set() for i in range(gates)}
    for i, (a, b) in enumerate(args):
        for v in (a, b):
            if v >= inputs:
                read_by[v].add(i)
    sinks = [inputs + i for i in range(gates) if not read_by[inputs + i]]

    marks = random.Random(marking)
    cuts = sorted(marks.sample(range(1, gates), marks.randint(0, min(2, gates - 1))))
    replay_blocks = [list(range(s, e)) for s, e in zip([0, *cuts], [*cuts, gates], strict=True)]
    layout: list[list[list[int]]] = []
    for block in replay_blocks:
        inner = sorted(marks.sample(range(1, len(block)), marks.randint(0, min(2, len(block) - 1))))
        layout.append([block[s:e] for s, e in zip([0, *inner], [*inner, len(block)], strict=True)])
    maximal = {(r, u): marks.random() < 0.3 for r, units in enumerate(layout) for u in range(len(units))}

    gate_set = make_word_gate_set(width)
    tracer = Tracer(gate_set)
    gate_of = {"add": tracer.gate("add"), "mul": tracer.gate("mul")}
    unit_of_gate = {g: (r, u) for r, units in enumerate(layout) for u, unit in enumerate(units) for g in unit}

    def outside(unit: list[int], scope) -> list[int]:
        """Value ids read by ``unit`` that are produced outside ``scope``."""

        seen: list[int] = []
        for g in unit:
            for v in args[g]:
                if v not in seen and (v < inputs or not scope(v - inputs)):
                    seen.append(v)
        return seen

    def declared(unit: list[int], scope, maximal_interface: bool) -> list[int]:
        """Value ids of ``unit`` read outside ``scope`` or sunk, or all of them."""

        produced = [inputs + g for g in unit]
        if maximal_interface:
            return produced
        return [v for v in produced if v in sinks or any(not scope(g) for g in read_by[v])]

    def verification(r: int, u: int, exported: list[int]):
        unit = layout[r][u]
        ins = outside(unit, lambda g: unit_of_gate[g] == (r, u))
        outs = declared(unit, lambda g: unit_of_gate[g] == (r, u), maximal[r, u])
        outs += [v for v in exported if v - inputs in unit and v not in outs]

        @tracer.definition(input_count=len(ins), key=("v", seed, marking, r, u), role="verification")
        def body(v):
            local = {value: v[k] for k, value in enumerate(ins)}
            for g in unit:
                local[inputs + g] = gate_of[ops[g]](local[args[g][0]], local[args[g][1]])
            return [local[value] for value in outs]

        return body, ins, outs

    def replay(r: int):
        units = layout[r]
        members = [g for unit in units for g in unit]
        ins = outside(members, lambda g: unit_of_gate[g][0] == r)
        wide = any(maximal[r, u] for u in range(len(units)))
        outs = declared(members, lambda g: unit_of_gate[g][0] == r, wide)
        pieces = [verification(r, u, outs) for u in range(len(units))]

        @tracer.definition(input_count=len(ins), key=("r", seed, marking, r), role="replay")
        def body(v):
            local = {value: v[k] for k, value in enumerate(ins)}
            for piece, piece_ins, piece_outs in pieces:
                produced = piece(*[local[value] for value in piece_ins])
                produced = [produced] if len(piece_outs) == 1 else list(produced)
                local.update(zip(piece_outs, produced, strict=True))
            return [local[value] for value in outs]

        return body, ins, outs

    @tracer.definition(input_count=0, key=("in", seed, inputs), role="replay")
    def source(_v):
        return tracer.inputs(inputs)

    blocks = [replay(r) for r in range(len(layout))]

    @tracer.definition(input_count=0, key=("root", seed, marking))
    def root(_v):
        sources = source()
        local = {k: sources[k] for k in range(inputs)}
        for body, ins, outs in blocks:
            produced = body(*[local[value] for value in ins])
            produced = [produced] if len(outs) == 1 else list(produced)
            local.update(zip(outs, produced, strict=True))
        return [local[value] for value in sinks]

    return Compiler(gate_set).compile(tracer.serialize(root), [1] * inputs)


def evaluate(
    compiled: Compiled,
    inputs: Iterable[int],
    weights: Iterable[int],
    overrides: Mapping[int, int] | None = None,
) -> dict[int, int]:
    """Every value of the circuit, with ``overrides`` injected and propagated.

    A gate whose address is overridden holds the given value instead of its
    relation's; every gate downstream is computed from what it reads, so the
    result is a full (dishonest) transcript whose incorrect gates are exactly
    the overridden addresses.
    """

    circuit = compiled.circuit
    given = {"input": iter(tuple(inputs)), "weight": iter(tuple(weights))}
    forced = {} if overrides is None else dict(overrides)
    values: dict[int, int] = {}
    for address in range(circuit.n):
        ref = circuit[address]
        if ref.is_source:
            value = next(given[ref.source])  # type: ignore[index]
        else:
            value = circuit.evaluate_gate(address, tuple(values[a] for a in ref.args))
        values[address] = forced.get(address, value)
    return values


class Model:
    """A compiled chain with honest values, ``kappa_W`` and an expectation factory."""

    def __init__(
        self,
        stages: int = 2,
        cells: int = 2,
        width: int = 8,
        *,
        inputs: tuple[int, ...] | None = None,
        weights: tuple[int, ...] | None = None,
        wide_units: bool = False,
    ) -> None:
        self.compiled = chain_compiled(stages, cells, width, wide_units=wide_units)
        self.circuit = self.compiled.circuit
        self.index = self.compiled.index
        self.width = width
        self.stages, self.cells = stages, cells
        mask = (1 << width) - 1
        self.inputs = (
            tuple((3 * i + 2) & mask for i in range(cells)) if inputs is None else inputs
        )
        self.weights = (
            tuple((5 * i + 3) & mask for i in range(cells)) if weights is None else weights
        )
        self.values: dict[int, int] = evaluate(self.compiled, self.inputs, self.weights)
        self.outputs = self.outputs_of(self.values)
        self.kappa, self.tree = commit_weights(self.compiled, self.weights)

    # -- addresses -----------------------------------------------------------

    def outputs_of(self, values: Mapping[int, int]) -> tuple[int, ...]:
        return tuple(values[address] for address in self.circuit.outputs)

    def cell_unit(self, stage: int, cell: int) -> int:
        """The global verification unit index of a stage's cell."""

        return 2 * self.cells + stage * self.cells + cell

    def cell_addresses(self, stage: int, cell: int) -> tuple[int, int]:
        """``(mul, add)`` addresses of a stage's cell."""

        node = self.index.verification_unit(self.cell_unit(stage, cell))
        return node.interval.start, node.interval.start + 1

    def replay_unit_of(self, stage: int) -> int:
        return 1 + stage

    def unit_of(self, address: int) -> int:
        """The verification unit holding ``address``."""

        replay = self.index.replay_units.owner(address)
        block = self.index.verification_units(replay)
        return block.first + block.owner(address)

    @property
    def interior_addresses(self) -> list[int]:
        return [
            int(a)
            for r in range(self.index.replay_units.count)
            for a in iter_domain(self.index.interior(r))
        ]

    @property
    def hidden_boundary_addresses(self) -> list[int]:
        """Boundary positions that are neither inputs nor circuit outputs."""

        public = set(self.circuit.inputs) | set(self.circuit.outputs)
        return [a for a in iter_domain(self.index.boundary()) if a not in public]

    # -- dishonest transcripts ----------------------------------------------

    def corrupt(self, overrides: Mapping[int, int]) -> tuple[dict[int, int], tuple[int, ...]]:
        """Values with ``overrides`` injected and propagated, and their outputs."""

        values = evaluate(self.compiled, self.inputs, self.weights, overrides)
        return values, self.outputs_of(values)

    def error_units(self, overrides: Iterable[int]) -> frozenset[int]:
        return frozenset(self.unit_of(address) for address in overrides)

    # -- the verifier's side --------------------------------------------------

    def expectation(
        self,
        policy: VerificationPolicy = CHECK_EVERYTHING,
        *,
        claimed_outputs: Iterable[int] | None = None,
        public_inputs: Iterable[int] | None = None,
        parameters: VerifierParameters | None = None,
        weights: Weights | None | str = "model",
        session_id: bytes = SESSION_ID,
        q_seed: bytes = Q_SEED,
        s_seed: bytes = S_SEED,
    ) -> Expectation:
        return make_expectation(
            self.compiled,
            policy,
            self.inputs if public_inputs is None else tuple(public_inputs),
            self.outputs if claimed_outputs is None else tuple(claimed_outputs),
            parameters=parameters,
            weights=self.kappa if weights == "model" else weights,  # type: ignore[arg-type]
            session_id=session_id,
            q_seed=q_seed,
            s_seed=s_seed,
        )

    def header(self, expectation: Expectation) -> Header:
        return VerifierSession(expectation, self.compiled).header

    def run(
        self,
        expectation: Expectation,
        values: Values,
        *,
        replay: Replay | None | str = "assignment",
        limits: VerificationLimits | None = None,
        prover: type[ProverSession] = ProverSession,
        **prover_options: object,
    ) -> ProtocolRun:
        """Run prover and verifier; the prover replays ``values`` as given by default."""

        chosen = assignment_replay(values) if replay == "assignment" else replay
        return run_sessions(
            self.compiled,
            expectation,
            values,
            replay=chosen,  # type: ignore[arg-type]
            limits=limits,
            weight_tree=self.tree,
            prover=prover,
            **prover_options,
        )

    def verify(self, transcript: Transcript | bytes, expectation: Expectation):
        data = transcript if isinstance(transcript, bytes) else encode_transcript(transcript)
        return verify_transcript(data, expectation, self.compiled)


def run_sessions(
    compiled: Compiled,
    expectation: Expectation,
    values: Values,
    *,
    replay: Replay | None = None,
    limits: VerificationLimits | None = None,
    weight_tree: MerkleTree | None = None,
    prover: type[ProverSession] = ProverSession,
    **prover_options: object,
) -> ProtocolRun:
    """``run_protocol`` with a pluggable (possibly dishonest) prover class."""

    try:
        verifier = VerifierSession(expectation, compiled, limits=limits)
    except Reject as rejection:
        return ProtocolRun(rejection_report(rejection, None), None)
    if compiled.index.weight_count == 0:
        weight_tree = None
    party = prover(
        compiled,
        verifier.header,
        values,
        replay=replay,
        limits=limits,
        weight_tree=weight_tree,
        **prover_options,  # type: ignore[arg-type]
    )
    try:
        replay_challenge = verifier.receive_boundary(party.boundary())
        sample_challenge = verifier.receive_interiors(party.interiors(replay_challenge))
        report = verifier.receive_evidence(party.evidence(sample_challenge))
    except Reject as rejection:
        return ProtocolRun(rejection_report(rejection, verifier), None)
    return ProtocolRun(report, verifier.transcript)


# -- dishonest provers -----------------------------------------------------------


class TamperingProver(ProverSession):
    """An honest prover with hooks at every point a dishonest one could deviate.

    ``raw_leaves`` commits the given bytes verbatim at an address instead of
    the canonical encoding of its value; ``domain_for`` replaces the domain
    a tree is built under (the root is still sent as if it were the right
    one); ``rewrite_opening(owner, opening, phase)`` rewrites every opening
    sent, ``rewrite_boundary`` / ``rewrite_interiors`` / ``rewrite_evidence``
    rewrite whole messages; ``recommit_boundary`` rebuilds the boundary tree
    from the given values after ``J`` has been seen.
    """

    def __init__(
        self,
        *args: object,
        raw_leaves: Mapping[int, bytes] | None = None,
        domain_for: Callable[[CommitmentDomain], CommitmentDomain] | None = None,
        rewrite_opening: Callable[[int, Opening, str], Opening] | None = None,
        rewrite_boundary: Callable[[BoundaryMessage], BoundaryMessage] | None = None,
        rewrite_interiors: Callable[[InteriorMessage], InteriorMessage] | None = None,
        rewrite_evidence: Callable[[EvidenceMessage], EvidenceMessage] | None = None,
        recommit_boundary: Values | None = None,
        **kwargs: object,
    ) -> None:
        self.raw_leaves = {} if raw_leaves is None else dict(raw_leaves)
        self.domain_for = domain_for
        self.rewrite_opening = rewrite_opening
        self.rewrite_boundary = rewrite_boundary
        self.rewrite_interiors = rewrite_interiors
        self.rewrite_evidence = rewrite_evidence
        self.recommit_boundary = recommit_boundary
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]

    def _commit(self, domain: CommitmentDomain, values: Values) -> MerkleTree:
        circuit = self._layout.circuit
        owner = domain.owner
        if self.domain_for is not None:
            domain = self.domain_for(domain)
        encoded: dict[int, bytes] = {}
        for position in iter_domain(domain.positions):
            address = int(position)
            if address in self.raw_leaves:
                encoded[address] = self.raw_leaves[address]
            else:
                encoded[address] = circuit.encode(address, values[address])
        tree = MerkleTree(domain, encoded, lambda address: leaf_schema(circuit, address))
        self._trees[owner] = tree
        return tree

    def boundary(self) -> BoundaryMessage:
        message = super().boundary()
        if self.rewrite_opening is not None:
            message = BoundaryMessage(
                message.commitment,
                tuple(
                    self.rewrite_opening(BOUNDARY_OWNER, item, "boundary")
                    for item in message.io_openings
                ),
            )
        if self.rewrite_boundary is not None:
            message = self.rewrite_boundary(message)
        return message

    def interiors(self, challenge: ReplayChallenge) -> InteriorMessage:
        if self.recommit_boundary is not None:
            self._commit(
                boundary_domain(self.header, self._layout.compiled), self.recommit_boundary
            )
        message = super().interiors(challenge)
        if self.rewrite_interiors is not None:
            message = self.rewrite_interiors(message)
        return message

    def evidence(self, challenge: SampleChallenge) -> EvidenceMessage:
        message = super().evidence(challenge)
        if self.rewrite_opening is not None:
            batches = []
            for unit, batch in zip(challenge.selected, message.units, strict=True):
                owners = [owner for owner, _ in self._layout.required(unit)]
                batches.append(
                    tuple(
                        self.rewrite_opening(owner, item, "evidence")
                        for owner, item in zip(owners, batch, strict=True)
                    )
                )
            message = EvidenceMessage(tuple(batches))
        if self.rewrite_evidence is not None:
            message = self.rewrite_evidence(message)
        return message


def flip(value: bytes) -> bytes:
    """``value`` with its first bit flipped."""

    return bytes((value[0] ^ 1,)) + value[1:]


# -- byte-level transcript surgery -----------------------------------------------


def transcript_document(transcript: Transcript) -> dict:
    return json.loads(encode_transcript(transcript))


def mutate_transcript(transcript: Transcript, mutate: Callable[[dict], None]) -> bytes:
    """Canonical bytes of ``transcript`` after ``mutate`` edits its JSON document."""

    document = transcript_document(transcript)
    mutate(document)
    return canonical_json_bytes(document)


# -- raw descriptions (format v2) --------------------------------------------------


IN, LOC = "input", "local"


def rng(space: str, start: int, count: int = 1, stride: int = 0) -> list[object]:
    return [space, start, count, stride]


def jrng(space: str, start: int, count: int = 1, stride: int = 0, jstride: int = 0) -> list[object]:
    return [space, start, count, stride, jstride]


def gate(name: str, *args: list[object]) -> dict[str, object]:
    return {"kind": "gate", "gate": name, "args": list(args)}


def call(digest: str, *args: list[object]) -> dict[str, object]:
    return {"kind": "call", "digest": digest, "args": list(args)}


def repeat(count: int, digest: str, *args: list[object]) -> dict[str, object]:
    return {"kind": "repeat", "count": count, "digest": digest, "args": list(args)}


def body(
    input_count: int,
    steps: list[dict[str, object]],
    outputs: list[list[object]],
    role: str | None = None,
) -> dict[str, object]:
    return {"input_count": input_count, "role": role, "steps": steps, "outputs": outputs}


class Doc:
    """Definitions in dependency order, serialized canonically."""

    def __init__(self) -> None:
        self.entries: list[dict[str, object]] = []

    def add(self, definition: dict[str, object]) -> str:
        digest = definition_digest(definition)
        self.entries.append({"digest": digest, "body": definition})
        return digest

    def source_cell(self, name: str) -> str:
        return self.add(body(0, [gate(name)], [rng(LOC, 0)], role="verification"))

    def value(self, root: str) -> dict[str, object]:
        return {"version": FORMAT_VERSION, "definitions": self.entries, "root": root}

    def serialize(self, root: str) -> bytes:
        return canonical_description(self.value(root))


@dataclass(frozen=True, slots=True)
class Phases:
    """The hash chain of a run, recomputed from its transcript."""

    boundary: bytes
    replay: bytes
    interior: bytes
    sample: bytes

    @staticmethod
    def of(transcript: Transcript) -> Phases:
        boundary = boundary_phase(transcript.header, transcript.boundary)
        replay = replay_phase(boundary, transcript.replay_challenge)
        interior = interior_phase(replay, transcript.interiors)
        sample = sample_phase(interior, transcript.sample_challenge)
        return Phases(boundary, replay, interior, sample)


def replay_selection(
    expectation: Expectation, header: Header, boundary: BoundaryMessage, compiled: Compiled
) -> tuple[int, ...]:
    """``J`` as the verifier derives it from ``q_seed`` and the boundary phase."""

    return derive_replay_selection(
        expectation.q_seed, boundary_phase(header, boundary), compiled, expectation.policy, LIMITS
    )


def sample_selection(
    expectation: Expectation,
    header: Header,
    boundary: BoundaryMessage,
    challenge: ReplayChallenge,
    interiors: InteriorMessage,
    compiled: Compiled,
) -> tuple[int, ...]:
    """``T`` as the verifier derives it from ``s_seed`` and the interior phase."""

    phase = interior_phase(replay_phase(boundary_phase(header, boundary), challenge), interiors)
    return derive_sample_selection(
        expectation.s_seed, phase, compiled, challenge.selected, expectation.policy, LIMITS
    )


# keep the re-exported protocol helpers referenced (they are reached through ``sec``)
_REEXPORTED = (
    BOUNDARY_OWNER,
    Header,
    ProtocolRun,
    Reject,
    SampleChallenge,
    VerifierParameters,
    VerifierSession,
    Weights,
    boundary_domain,
    commit_weights,
    interior_domain,
    make_expectation,
    sample_phase,
    weight_domain,
    encode_transcript,
    verify_transcript,
    iter_domain,
    leaf_schema,
)


# -- fixtures ----------------------------------------------------------------------


@pytest.fixture(scope="session")
def sec() -> ModuleType:
    """This module: the helpers, dishonest provers and constants of the suite."""

    return sys.modules[__name__]


@pytest.fixture(scope="session")
def model() -> Model:
    """Two stages of two cells over two inputs and two weights (12 gates)."""

    return Model(2, 2)


@pytest.fixture(scope="session")
def narrow_model() -> Model:
    """The same chain over 4-bit values: a one-byte payload can be out of range."""

    return Model(2, 2, width=4)


@pytest.fixture
def honest_run(model: Model):
    """An accepted run checking everything, with its expectation."""

    expectation = model.expectation()
    run = model.run(expectation, model.values, replay=None)
    assert run.report.accepted and run.transcript is not None
    return run, expectation
