"""Interactive prover and verifier sessions for the two-stage protocol.

The verifier owns both seeds and releases each challenge only after the
message it depends on has been received and checked.  Every lookup the
verifier performs is against the trusted :class:`Compiled` ``(C, I)``; the
prover never tells the verifier where a value lives or which addresses a unit
has.

A :class:`Claim` is what a run is about -- everything the header binds -- and
an :class:`Expectation` is a claim together with the verifier's two seeds.  A
:class:`VerifierSession` admits either: from an expectation it is the
per-session protocol as written; from a claim the seeds arrive later through
:meth:`VerifierSession.release`, after the boundary has been accepted, which
is how the epoch layer (:mod:`veritor.protocol.epoch`) commits a round of
runs before any challenge exists.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from secrets import token_bytes

from veritor.analysis import bound
from veritor.compile import Compilation
from veritor.core import (
    Compiled,
    Digest,
    ResourceLimit,
    VerificationLimits,
    VerificationPolicy,
    iter_domain,
)

from .challenge import derive_replay_selection, derive_sample_selection
from .domains import (
    BOUNDARY_OWNER,
    WEIGHT_OWNER,
    boundary_domain,
    commit_weights,
    interior_domain,
    leaf_schema,
    weight_domain,
)
from .merkle import CommitmentDomain, MerkleTree, validate_commitment, verify_opening
from .messages import (
    TRANSPARENT_BACKEND,
    BoundaryMessage,
    Commitment,
    EvidenceMessage,
    Header,
    InteriorMessage,
    Opening,
    ProtocolError,
    Reject,
    ReplayChallenge,
    SampleChallenge,
    Transcript,
    VerificationCode,
    VerificationReport,
    Weights,
)
from .parameters import VerifierParameters, expected_work, positions_per_unit
from .phases import boundary_phase, interior_phase, replay_phase, sample_phase
from .proofs import (
    BatchPlan,
    KindProgram,
    Obligation,
    ProofBackend,
    Witness,
    check_coverage,
    derive_obligations,
    encode_witness,
    make_statement,
    prove_plan,
    resolve_backend,
    statement_width,
)

type Values = Mapping[int, object]
type Replay = Callable[[int, Values], Values]
"""``replay(unit, boundary_values) -> interior values`` for one replay unit."""
type Declare = Callable[[int, Values], Iterable[int]]
"""``declare(unit, values) -> VU indices`` the prover declares incorrect in replay unit
``unit``; ``values`` are the interior values it is about to commit over its boundary
and weight values (see :func:`self_check`)."""


@dataclass(frozen=True, slots=True)
class Claim:
    """What a run is about: everything the header binds, and no randomness.

    ``compiled_digest``, ``constructor`` and ``advice`` name the
    ``Compile(G, x, a)`` the run is about; ``policy`` is the run's ``theta =
    (q, s)``; ``parameters`` hold the verifier's own ``eta``, ``U_max``,
    ``A``, ``W_max`` and ``f_max``.  ``public_inputs`` are the values of the
    circuit's ``in`` gates by rank (address order) and ``claimed_outputs``
    the outputs; the verifier encodes them with the circuit's canonical
    codec.  ``weights`` is the model's ``kappa_W``, required exactly when the
    circuit has weight gates.  ``backend`` names the proof backend the reveal
    step runs through (default: transparent openings).
    """

    session_id: bytes
    compiled_digest: Digest
    constructor: Digest
    advice: bytes
    policy: VerificationPolicy
    parameters: VerifierParameters
    public_inputs: tuple[object, ...]
    claimed_outputs: tuple[object, ...]
    weights: Weights | None = None
    backend: str = TRANSPARENT_BACKEND

    def __post_init__(self) -> None:
        _check_claim(self)


def _check_claim(claim: Claim | Expectation) -> None:
    if type(claim.advice) is not bytes:
        raise ProtocolError("advice must be bytes")
    if not isinstance(claim.parameters, VerifierParameters):
        raise ProtocolError("parameters must be VerifierParameters")
    if not isinstance(claim.policy, VerificationPolicy):
        raise ProtocolError("policy must be a VerificationPolicy")
    if claim.weights is not None and not isinstance(claim.weights, Weights):
        raise ProtocolError("weights must be Weights or None")


def check_seed(name: str, value: object) -> bytes:
    """``value`` as one of the verifier's 32-byte seeds, or a :class:`ProtocolError`."""

    if type(value) is not bytes or len(value) != 32:
        raise ProtocolError(f"expected {name} of 32 bytes")
    return value


@dataclass(frozen=True, slots=True)
class Expectation:
    """What the verifier expects and the randomness it owns: a :class:`Claim`
    (see there for the fields) with the verifier's two seeds.

    Both seeds are mandatory so a verifier can never accidentally let the
    prover choose them; a verifier that must commit to a run before its
    seeds exist admits the :attr:`claim` alone and releases the seeds later.
    """

    session_id: bytes
    compiled_digest: Digest
    constructor: Digest
    advice: bytes
    policy: VerificationPolicy
    parameters: VerifierParameters
    public_inputs: tuple[object, ...]
    claimed_outputs: tuple[object, ...]
    q_seed: bytes
    s_seed: bytes
    weights: Weights | None = None
    backend: str = TRANSPARENT_BACKEND

    def __post_init__(self) -> None:
        check_seed("q seed", self.q_seed)
        check_seed("s seed", self.s_seed)
        _check_claim(self)

    @property
    def claim(self) -> Claim:
        """The expectation without its seeds: what the header binds."""

        return Claim(
            self.session_id,
            self.compiled_digest,
            self.constructor,
            self.advice,
            self.policy,
            self.parameters,
            self.public_inputs,
            self.claimed_outputs,
            self.weights,
            self.backend,
        )


def make_expectation(
    compilation: Compilation,
    proposal: VerificationPolicy,
    claimed_outputs: Iterable[object],
    *,
    parameters: VerifierParameters,
    weights: Weights | None = None,
    session_id: bytes | None = None,
    q_seed: bytes | None = None,
    s_seed: bytes | None = None,
    backend: str = TRANSPARENT_BACKEND,
) -> Expectation:
    """The verifier's expectation for one ``Compile(G, x, a)`` and the claimed ``y*``.

    ``compilation`` supplies ``(C, I)``, ``G``'s digest, the public inputs as
    the circuit consumes them and the advice; the client's proposed ``theta``
    is admitted under the verifier's ``parameters``, which are never
    defaulted: the verifier states ``eta``, ``U_max``, ``A`` and ``W_max``.
    Fresh seeds are drawn unless given.  ``backend`` is the proof backend id
    bound into the header.
    """

    if not isinstance(compilation, Compilation):
        raise ProtocolError("make_expectation requires a Compilation from Compile")
    if not isinstance(parameters, VerifierParameters):
        raise ProtocolError("make_expectation requires the verifier's VerifierParameters")
    checked = parameters
    return Expectation(
        session_id=token_bytes(16) if session_id is None else session_id,
        compiled_digest=compilation.compiled.digest,
        constructor=compilation.constructor,
        advice=compilation.advice,
        policy=checked.policy(proposal),
        parameters=checked,
        public_inputs=tuple(compilation.inputs),
        claimed_outputs=tuple(claimed_outputs),
        q_seed=token_bytes(32) if q_seed is None else q_seed,
        s_seed=token_bytes(32) if s_seed is None else s_seed,
        weights=weights,
        backend=backend,
    )


def rejection_report(rejection: Reject, session: VerifierSession | None) -> VerificationReport:
    """The verdict for ``rejection``, with the selections made so far."""

    if session is None:
        return VerificationReport(rejection.code, rejection.detail)
    return VerificationReport(
        rejection.code,
        rejection.detail,
        session.selected_replay_units,
        session.selected_verification_units,
    )


class _Layout:
    """Address-level lookups shared by both sessions; all ``O(depth)``.

    Construction is ``O(|public I/O|)``: the input gates and the outputs are
    the only addresses either party touches wholesale.
    """

    __slots__ = ("boundary", "circuit", "compiled", "index", "io", "public_inputs", "weights")

    def __init__(self, compiled: Compiled) -> None:
        if not isinstance(compiled, Compiled):
            raise ProtocolError("sessions require a Compiled circuit")
        self.compiled = compiled
        self.circuit = compiled.circuit
        self.index = compiled.index
        self.boundary = self.index.boundary()
        self.weights = self.index.weights()
        self.public_inputs: tuple[int, ...] = tuple(self.circuit.inputs)
        """The input gate addresses by rank (ascending)."""
        addresses = set(self.public_inputs)
        addresses.update(self.circuit.outputs)
        for address in addresses:
            if not self.boundary.contains(address):
                raise ProtocolError(
                    f"circuit output at address {address} is not a boundary position "
                    "(a weight gate cannot be a claimed output)"
                )
        self.io: tuple[int, ...] = tuple(sorted(addresses, key=self.boundary.rank))
        """Distinct public I/O addresses in boundary rank order."""

    def owner(self, address: int) -> int:
        """Who commits to ``address``: ``kappa_W`` for a weight gate, the boundary
        for an input gate or a declared output, else the owning replay unit."""

        if self.weights.contains(address):
            return WEIGHT_OWNER
        if self.boundary.contains(address):
            return BOUNDARY_OWNER
        return self.index.replay_units.owner(address)

    def position(self, owner: int, address: int) -> int:
        """Where ``address`` lives in its owner's domain: its rank under ``kappa_W``,
        the address itself under the boundary or an interior."""

        return self.circuit.weight_rank(address) if owner == WEIGHT_OWNER else address

    def required(self, unit: int) -> tuple[tuple[int, int], ...]:
        """``(owner, address)`` for every value a verification unit reads or writes."""

        node = self.index.verification_unit(unit)
        replay_unit = node.replay_unit
        addresses = set(node.interval)
        addresses.update(self.circuit.In(node))
        result: list[tuple[int, int]] = []
        for address in sorted(addresses):
            owner = self.owner(address)
            if owner not in (WEIGHT_OWNER, BOUNDARY_OWNER, replay_unit):
                raise Reject(
                    VerificationCode.INVALID_COMPILED_RESULT,
                    f"address {address} is read by unit {unit} but owned by "
                    f"replay unit {owner} and is not a boundary address",
                )
            result.append((owner, address))
        return tuple(result)


def replay_unit(compiled: Compiled, unit: int, boundary_values: Values) -> dict[int, object]:
    """Honest replay: recompute ``Int(unit)`` from the boundary, in address order."""

    circuit = compiled.circuit
    known: dict[int, object] = {}
    for address in iter_domain(compiled.index.interior(unit)):
        arguments = []
        for argument in circuit[address].args:
            if argument in known:
                arguments.append(known[argument])
            else:
                try:
                    arguments.append(boundary_values[argument])
                except KeyError as error:
                    raise ProtocolError(
                        f"replay of unit {unit} needs boundary value {argument}"
                    ) from error
        known[address] = circuit.evaluate_gate(address, arguments)  # type: ignore[arg-type]
    return known


def assignment_replay(values: Values) -> Replay:
    """A replay that reads interiors from a fixed (possibly dishonest) assignment."""

    def replay(unit: int, boundary_values: Values) -> Values:
        del boundary_values
        return values

    return replay


def self_check(compiled: Compiled, unit: int, values: Values) -> tuple[int, ...]:
    """Replay-time fault detection: the VUs of replay unit ``unit`` whose values
    in ``values`` disobey their relation, as global VU indices in order.

    This is what an honest server learns when it replays an opened RU: it
    recomputes every gate from the values it actually holds (the ones it
    streamed and is about to commit) and compares.  A hardware fault that
    flipped one gate's output shows up as exactly that gate's VU -- every VU
    downstream computed correctly *from* the faulty value, so their
    relations hold against it -- and that VU is what the server declares.
    ``values`` must cover the RU's interior and everything it reads.
    """

    circuit = compiled.circuit
    units = compiled.index.verification_units(unit)
    faulty: list[int] = []
    for offset in range(units.count):
        node = units.unit(offset)
        for address in node.interval:
            gate = circuit[address]
            if gate.is_source:
                continue
            args = [values[argument] for argument in gate.args]
            if not circuit.check_gate(address, args, values[address]):  # type: ignore[arg-type]
                faulty.append(units.first + offset)
                break
    return tuple(faulty)


def honest_declare(compiled: Compiled) -> Declare:
    """The honest server's :data:`Declare`: declare whatever :func:`self_check` finds."""

    return lambda unit, values: self_check(compiled, unit, values)


class _Overlay(Mapping[int, object]):
    """``front`` over ``back``: the interior values about to be committed over the
    prover's boundary and weight values, without copying either."""

    __slots__ = ("_back", "_front")

    def __init__(self, front: Values, back: Values) -> None:
        self._front = front
        self._back = back

    def __getitem__(self, key: int) -> object:
        try:
            return self._front[key]
        except KeyError:
            return self._back[key]

    def __iter__(self) -> Iterator[int]:
        seen = set(self._front)
        yield from seen
        for key in self._back:
            if key not in seen:
                yield key

    def __len__(self) -> int:
        return len(set(self._front) | set(self._back))


class ProverSession:
    """The prover's side.  Call ``boundary``, ``interiors``, ``evidence`` in order.

    When the header binds weights, ``weight_tree`` is the model's tree from
    :func:`commit_weights`; a sampled weight gate is opened at its rank in
    that tree, which is never rebuilt.  ``backend`` is the proof backend the
    header names (defaulted for the transparent one) and ``plan`` how the
    sampled VUs' obligations are grouped into proofs (default: one proof).
    ``declare`` chooses the fault declarations of each opened RU
    (:data:`Declare`; an honest server uses :func:`honest_declare`); the
    default declares nothing.  Whatever it returns is sent as is -- a prover
    that declares more than the header's ``max_faults``, or a VU outside an
    opened RU, is rejected by the verifier, never silently trimmed here.
    """

    def __init__(
        self,
        compiled: Compiled,
        header: Header,
        values: Values,
        *,
        replay: Replay | None = None,
        limits: VerificationLimits | None = None,
        weight_tree: MerkleTree | None = None,
        backend: ProofBackend | None = None,
        plan: BatchPlan | None = None,
        declare: Declare | None = None,
    ) -> None:
        self._layout = _Layout(compiled)
        if header.compiled_digest != compiled.digest:
            raise ProtocolError("header names a different compiled circuit")
        self.header = header
        self._values = values
        self._replay = replay
        self._declare = declare
        self._limits = VerificationLimits() if limits is None else limits
        self._backend = resolve_backend(backend, header.backend, compiled, self._limits)
        self._plan = plan
        self._trees: dict[int, MerkleTree] = {}
        if header.weights is None:
            if weight_tree is not None:
                raise ProtocolError("the header binds no weights")
            if compiled.index.weight_count:
                raise ProtocolError("the circuit has weight gates but the header binds no weights")
        else:
            if weight_tree is None:
                raise ProtocolError("the header binds weights; the prover needs their tree")
            expected = weight_domain(compiled.circuit.gate_set, compiled.index.weight_count)
            if (
                weight_tree.domain.domain_id != expected.domain_id
                or weight_tree.commitment != header.weights.commitment
            ):
                raise ProtocolError("the weight tree does not match the header's weights")
            self._trees[WEIGHT_OWNER] = weight_tree
        self._phase = "boundary"
        self._boundary_phase = b""
        self._replay_phase = b""
        self._interior_phase = b""
        self._declarations: tuple[int, ...] = ()
        self.transcript_parts: list[object] = [header]

    def _expect(self, phase: str) -> None:
        if self._phase != phase:
            raise ProtocolError(f"prover is in phase {self._phase!r}, not {phase!r}")

    def _commit(self, domain: CommitmentDomain, values: Values) -> MerkleTree:
        circuit = self._layout.circuit
        encoded: dict[int, bytes] = {}
        for address in iter_domain(domain.positions):
            try:
                value = values[int(address)]
            except KeyError as error:
                raise ProtocolError(f"prover has no value for address {address}") from error
            encoded[int(address)] = circuit.encode(address, value)
        tree = MerkleTree(domain, encoded, lambda address: leaf_schema(circuit, address))
        self._trees[domain.owner] = tree
        return tree

    def boundary(self) -> BoundaryMessage:
        self._expect("boundary")
        tree = self._commit(boundary_domain(self.header, self._layout.compiled), self._values)
        message = BoundaryMessage(
            tree.commitment, tuple(tree.open(p) for p in self._layout.io)
        )
        self._boundary_phase = boundary_phase(self.header, message)
        self._phase = "interiors"
        self.transcript_parts.append(message)
        return message

    def interiors(self, challenge: ReplayChallenge) -> InteriorMessage:
        self._expect("interiors")
        self._replay_phase = replay_phase(self._boundary_phase, challenge)
        compiled = self._layout.compiled
        commitments: list[Commitment] = []
        declared: set[int] = set()
        for unit in challenge.selected:
            if unit >= compiled.index.replay_units.count:
                raise ProtocolError(f"challenge names unknown replay unit {unit}")
            interior_values = (
                replay_unit(compiled, unit, self._values)
                if self._replay is None
                else self._replay(unit, self._values)
            )
            if self._declare is not None:
                declared.update(self._declare(unit, _Overlay(interior_values, self._values)))
            domain = interior_domain(self._replay_phase, compiled, unit)
            commitments.append(self._commit(domain, interior_values).commitment)
        message = InteriorMessage(tuple(commitments), tuple(sorted(declared)))
        self._interior_phase = interior_phase(self._replay_phase, message)
        self._declarations = message.declarations
        self._phase = "evidence"
        self.transcript_parts.extend((challenge, message))
        return message

    def evidence(self, challenge: SampleChallenge) -> EvidenceMessage:
        """The reveal step: derive the sampled VUs' obligations, open their
        positions, and either hand over the openings (transparent backend) or
        prove the batches of the plan through the backend."""

        self._expect("evidence")
        sample_phase(self._interior_phase, challenge)
        commitments = {owner: (tree.domain, tree.commitment) for owner, tree in self._trees.items()}
        obligations, kinds = derive_obligations(
            self._layout, self.header, commitments, challenge.selected, set(self._declarations)
        )
        batches: list[tuple[Opening, ...]] = []
        for obligation in obligations:
            batches.append(
                tuple(
                    self._trees[obligation.commitments[ref.commitment].owner].open(ref.position)
                    for ref in obligation.positions
                )
            )
        if self._backend.backend_id == TRANSPARENT_BACKEND:
            message = EvidenceMessage(tuple(batches))
        else:
            gate_set = self._layout.circuit.gate_set
            proofs = prove_plan(
                self._backend,
                BatchPlan.single(len(obligations)) if self._plan is None else self._plan,
                obligations,
                [tuple((item.value, item.path) for item in batch) for batch in batches],
                kinds,
                gate_set.id,
                bytes.fromhex(gate_set.digest),
                statement_width(gate_set),
            )
            message = EvidenceMessage((), proofs)
        self._phase = "done"
        self.transcript_parts.extend((challenge, message))
        return message

    @property
    def transcript(self) -> Transcript:
        if self._phase != "done":
            raise ProtocolError("the protocol has not finished")
        return Transcript(*self.transcript_parts)  # type: ignore[arg-type]


class VerifierSession:
    """The verifier's side.  Feed messages in order; each returns the next challenge.

    Construction admits the run: the proposal's rates, the per-unit sizes and
    the expected work are checked against the limits and ``W_max``, the
    advice against ``A`` and ``Bound(C, I, theta)`` against ``U_max``, before
    any commitment is accepted; every verdict, including a resource limit, is
    a :class:`Reject`.

    Admitted from an :class:`Expectation`, the session holds its seeds from
    the start and :meth:`receive_boundary` answers with the replay challenge.
    Admitted from a bare :class:`Claim`, it has no seeds: the boundary is
    committed with :meth:`accept_boundary`, the seeds arrive through
    :meth:`release`, and :meth:`challenge_replay` derives the challenge --
    the run is bound to its boundary before any randomness exists.
    """

    def __init__(
        self,
        expectation: Expectation | Claim,
        compiled: Compiled,
        *,
        limits: VerificationLimits | None = None,
        backend: ProofBackend | None = None,
    ) -> None:
        if not isinstance(compiled, Compiled):
            raise ProtocolError("sessions require a Compiled circuit")
        if not isinstance(expectation, Expectation | Claim):
            raise ProtocolError("sessions require an Expectation or a Claim")
        claim = expectation.claim if isinstance(expectation, Expectation) else expectation
        if claim.compiled_digest != compiled.digest:
            raise ProtocolError("expectation names a different compiled circuit")
        self._backend = resolve_backend(backend, claim.backend, compiled, limits)
        weights = claim.weights
        weight_count = compiled.index.weight_count
        if weights is None and weight_count:
            raise Reject(
                VerificationCode.INVALID_COMPILED_RESULT,
                f"the circuit has {weight_count} weight gates but no kappa_W is bound",
            )
        if weights is not None and weights.count != weight_count:
            raise Reject(
                VerificationCode.INVALID_COMPILED_RESULT,
                f"kappa_W binds {weights.count} weights but the circuit has "
                f"{weight_count} weight gates",
            )
        self._layout = _Layout(compiled)
        self._claim = claim
        self._seeds: tuple[bytes, bytes] | None = None
        self._limits = VerificationLimits() if limits is None else limits
        self._phase = "admission"
        self.selected_replay_units: tuple[int, ...] = ()
        self.selected_verification_units: tuple[int, ...] = ()
        self._admit()
        circuit = self._layout.circuit
        try:
            inputs = tuple(
                circuit.encode(address, value)
                for address, value in zip(
                    self._layout.public_inputs, claim.public_inputs, strict=True
                )
            )
            outputs = tuple(
                circuit.encode(address, value)
                for address, value in zip(circuit.outputs, claim.claimed_outputs, strict=True)
            )
        except Exception as error:
            raise ProtocolError("expectation values do not encode canonically") from error
        self.header = Header(
            claim.session_id,
            compiled.digest,
            claim.constructor,
            claim.advice,
            claim.policy,
            claim.parameters.eta,
            inputs,
            outputs,
            weights,
            claim.backend,
            claim.parameters.max_faults,
        )
        self._commitments: dict[int, tuple[CommitmentDomain, Commitment]] = {}
        if weights is not None:
            self._commitments[WEIGHT_OWNER] = (
                weight_domain(circuit.gate_set, weight_count),
                weights.commitment,
            )
        self._phase = "boundary"
        self._boundary_phase = b""
        self._replay_phase = b""
        self._interior_phase = b""
        self.declared_verification_units: tuple[int, ...] = ()
        """The VUs the interior message declared incorrect, once accepted."""
        self.transcript_parts: list[object] = [self.header]
        if isinstance(expectation, Expectation):
            self.release(expectation.q_seed, expectation.s_seed)

    def release(self, q_seed: bytes, s_seed: bytes) -> None:
        """Give the session its seeds, once.

        A session admitted from a :class:`Claim` has none until here; the
        epoch layer calls this at round close, after every boundary of the
        round is committed, with seeds derived from the round's seal.
        """

        if self._seeds is not None:
            raise ProtocolError("the verifier's seeds are released once")
        self._seeds = (check_seed("q seed", q_seed), check_seed("s seed", s_seed))

    @property
    def released(self) -> bool:
        """Whether the session holds its seeds (it never shows them)."""

        return self._seeds is not None

    def _seed(self, index: int) -> bytes:
        if self._seeds is None:
            raise ProtocolError("the verifier's seeds have not been released")
        return self._seeds[index]

    def _admit(self) -> None:
        """Price the run before any commitment; folds over the kinds, nothing per copy.

        The limits and ``W_max`` are checked from counts alone in
        ``O(#kinds)`` and the advice against ``A`` by its length; when the
        verifier fixes ``U_max``, ``Bound(C, I, theta)`` at its ``eta`` is
        folded over the same kinds (milliseconds, independent of the number
        of copies) and must not exceed it.  Together the two caps bound the
        request's capacity by ``U_max + A``.
        """

        index = self._layout.index
        policy = self._claim.policy
        parameters = self._claim.parameters
        advice_bits = 8 * len(self._claim.advice)
        if advice_bits > parameters.max_advice_bits:
            raise self._reject(
                VerificationCode.POLICY_REJECTED,
                f"the advice is {advice_bits} bits, exceeding max_advice_bits "
                f"{parameters.max_advice_bits}",
            )
        with self._rejecting_limits():
            self._limits.enforce(
                "max_probability_denominator_bits",
                max(policy.denominator_bits, parameters.eta.denominator.bit_length()),
            )
            self._limits.enforce("max_units", index.replay_units.count)
            self._limits.enforce("max_units", index.verification_unit_count)
            for kind in index.kinds():
                self._limits.enforce("max_positions_per_unit", positions_per_unit(kind))
        work = expected_work(self._layout.compiled, policy, len(self._layout.io))
        budget = parameters.max_work
        if work > budget:
            raise self._reject(
                VerificationCode.WORK_BUDGET_EXCEEDED,
                f"expected verifier work {work} exceeds W_max {budget}",
            )
        if parameters.max_capacity is not None:
            capacity = bound(
                self._layout.compiled, policy, parameters.eta, max_faults=parameters.max_faults
            ).bits
            if capacity > parameters.max_capacity:
                raise self._reject(
                    VerificationCode.POLICY_REJECTED,
                    f"Bound(C, I, theta) is {capacity:.6g} bits, exceeding U_max "
                    f"{parameters.max_capacity}",
                )

    def _expect(self, phase: str) -> None:
        if self._phase != phase:
            raise Reject(
                VerificationCode.INVALID_PHASE,
                f"verifier is in phase {self._phase!r}, not {phase!r}",
            )

    def _reject(self, code: VerificationCode, detail: str) -> Reject:
        self._phase = "rejected"
        return Reject(code, detail)

    @contextmanager
    def _rejecting_limits(self) -> Iterator[None]:
        """Turn an exceeded resource limit into the verdict it is."""

        try:
            yield
        except ResourceLimit as error:
            raise self._reject(VerificationCode.RESOURCE_LIMIT, str(error)) from error

    def _accept_commitment(self, domain: CommitmentDomain, commitment: Commitment) -> None:
        self._limits.enforce("max_positions", domain.count)
        if not validate_commitment(domain, commitment):
            raise self._reject(
                VerificationCode.INVALID_COMMITMENT,
                f"commitment for owner {domain.owner} does not fit its domain",
            )
        self._commitments[domain.owner] = (domain, commitment)

    def _open(self, owner: int, opening: Opening, address: int) -> bytes:
        """Authenticate the opening of ``address`` under its owner's commitment.

        ``opening.position`` is the address's position in that domain (its
        rank under ``kappa_W``), already checked against the layout; the gate
        at ``address`` fixes the leaf schema.
        """

        try:
            domain, commitment = self._commitments[owner]
        except KeyError:
            raise self._reject(
                VerificationCode.INVALID_OPENING, f"no commitment for owner {owner}"
            ) from None
        if not verify_opening(
            domain,
            commitment,
            opening,
            leaf_schema(self._layout.circuit, address),
            self._limits,
        ):
            raise self._reject(
                VerificationCode.INVALID_OPENING,
                f"opening of address {address} failed under owner {owner}",
            )
        return opening.value

    def receive_boundary(self, message: BoundaryMessage) -> ReplayChallenge:
        """Accept the boundary and answer with the replay challenge."""

        self.accept_boundary(message)
        return self.challenge_replay()

    def accept_boundary(self, message: BoundaryMessage) -> None:
        """Commit the run to its boundary without deriving a challenge.

        The commitment is checked against its domain and the public I/O
        openings against the header; the boundary phase is fixed.  Nothing
        here needs a seed, so a session admitted from a :class:`Claim` can
        take the boundary at run time and be challenged at round close.
        """

        self._expect("boundary")
        with self._rejecting_limits():
            compiled = self._layout.compiled
            self._accept_commitment(
                boundary_domain(self.header, compiled), message.commitment
            )
            if tuple(item.position for item in message.io_openings) != self._layout.io:
                raise self._reject(
                    VerificationCode.COVERAGE_MISMATCH,
                    "public I/O openings must cover exactly the I/O addresses "
                    "in boundary order",
                )
            opened = {
                item.position: self._open(BOUNDARY_OWNER, item, item.position)
                for item in message.io_openings
            }
            for addresses, values, label in (
                (self._layout.public_inputs, self.header.public_inputs, "input"),
                (self._layout.circuit.outputs, self.header.claimed_outputs, "output"),
            ):
                for address, value in zip(addresses, values, strict=True):
                    if opened[address] != value:
                        raise self._reject(
                            VerificationCode.PUBLIC_IO_MISMATCH,
                            f"{label} at address {address} differs",
                        )
            self._boundary_phase = boundary_phase(self.header, message)
        self._phase = "replay"
        self.transcript_parts.append(message)

    @property
    def boundary_phase(self) -> bytes:
        """The boundary-phase digest of the accepted boundary (what the epoch
        stream links); empty until :meth:`accept_boundary`."""

        return self._boundary_phase

    def challenge_replay(self) -> ReplayChallenge:
        """The replay challenge over the accepted boundary, from the released q seed."""

        self._expect("replay")
        q_seed = self._seed(0)
        with self._rejecting_limits():
            selected = derive_replay_selection(
                q_seed,
                self._boundary_phase,
                self._layout.compiled,
                self.header.policy,
                self._limits,
            )
        challenge = ReplayChallenge(q_seed, selected)
        self._replay_phase = replay_phase(self._boundary_phase, challenge)
        self.selected_replay_units = selected
        self._phase = "interiors"
        self.transcript_parts.append(challenge)
        return challenge

    def receive_interiors(self, message: InteriorMessage) -> SampleChallenge:
        self._expect("interiors")
        with self._rejecting_limits():
            compiled = self._layout.compiled
            selected = self.selected_replay_units
            if len(message.commitments) != len(selected):
                raise self._reject(
                    VerificationCode.COVERAGE_MISMATCH,
                    f"expected {len(selected)} interior commitments, got "
                    f"{len(message.commitments)}",
                )
            for unit, commitment in zip(selected, message.commitments, strict=True):
                self._accept_commitment(
                    interior_domain(self._replay_phase, compiled, unit), commitment
                )
            self._accept_declarations(message.declarations)
            self._interior_phase = interior_phase(self._replay_phase, message)
            sampled = derive_sample_selection(
                self._seed(1),
                self._interior_phase,
                compiled,
                selected,
                self.header.policy,
                self._limits,
            )
        challenge = SampleChallenge(self._seed(1), sampled)
        sample_phase(self._interior_phase, challenge)
        self.selected_verification_units = sampled
        self._phase = "evidence"
        self.transcript_parts.extend((message, challenge))
        return challenge

    def _accept_declarations(self, declarations: tuple[int, ...]) -> None:
        """Admit the prover's fault declarations or reject the run.

        At most ``max_faults`` of them (``FAULTS_EXCEEDED``); each must name
        an existing VU inside an opened RU whose relation there is something
        to disclaim -- a VU of source gates alone has none
        (``FAULT_DECLARATION_INVALID``).  Sortedness and uniqueness are the
        message's own invariants.  Checked before the s-challenge is derived,
        so the sample is drawn over a message the verifier has accepted.
        """

        header = self.header
        if len(declarations) > header.max_faults:
            raise self._reject(
                VerificationCode.FAULTS_EXCEEDED,
                f"{len(declarations)} fault declarations exceed max_faults {header.max_faults}",
            )
        if not declarations:
            return
        index = self._layout.index
        circuit = self._layout.circuit
        blocks = [index.verification_units(unit) for unit in self.selected_replay_units]
        starts = [block.first for block in blocks]
        for unit in declarations:
            if unit >= index.verification_unit_count:
                raise self._reject(
                    VerificationCode.FAULT_DECLARATION_INVALID,
                    f"declared VU {unit} does not exist",
                )
            block = bisect_right(starts, unit) - 1
            if block < 0 or unit >= blocks[block].first + blocks[block].count:
                raise self._reject(
                    VerificationCode.FAULT_DECLARATION_INVALID,
                    f"declared VU {unit} lies in no opened replay unit",
                )
            node = blocks[block].unit(unit - blocks[block].first)
            if all(circuit[address].is_source for address in node.interval):
                raise self._reject(
                    VerificationCode.FAULT_DECLARATION_INVALID,
                    f"declared VU {unit} has no relation to disclaim",
                )
        self.declared_verification_units = declarations

    def receive_evidence(self, message: EvidenceMessage) -> VerificationReport:
        """The reveal step: derive what every sampled VU obliges the prover to
        show (from the challenge, the Index and the accepted commitments --
        never from the prover), then check that the evidence covers exactly
        those obligations through the header's proof backend.
        """

        self._expect("evidence")
        sampled = self.selected_verification_units
        transparent = self._backend.backend_id == TRANSPARENT_BACKEND
        if transparent and len(message.units) != len(sampled):
            raise self._reject(
                VerificationCode.COVERAGE_MISMATCH,
                f"expected evidence for {len(sampled)} units, got {len(message.units)}",
            )
        if message.units and not transparent:
            raise self._reject(
                VerificationCode.COVERAGE_MISMATCH,
                f"backend {self._backend.backend_id!r} takes proofs, not openings",
            )
        if message.proofs and transparent:
            raise self._reject(
                VerificationCode.COVERAGE_MISMATCH, "the transparent backend takes openings"
            )
        with self._rejecting_limits():
            demanded, kinds = derive_obligations(
                self._layout,
                self.header,
                self._commitments,
                sampled,
                set(self.declared_verification_units),
            )
            gate_set = self._layout.circuit.gate_set
            gate_set_digest = bytes.fromhex(gate_set.digest)
            width = statement_width(gate_set)
            try:
                if transparent:
                    self._check_openings(demanded, kinds, gate_set_digest, width, message)
                else:
                    check_coverage(
                        demanded,
                        kinds,
                        gate_set.id,
                        gate_set_digest,
                        width,
                        message.proofs,
                        self._backend.verify,
                        on_proof=lambda _n, proof, _s: self._limits.enforce(
                            "max_proof_bytes", len(proof.proof) + len(proof.foreign)
                        ),
                    )
            except Reject as rejection:
                raise self._reject(rejection.code, rejection.detail) from None
        self._phase = "done"
        self.transcript_parts.append(message)
        return VerificationReport(
            VerificationCode.ACCEPTED,
            sampled_replay_units=self.selected_replay_units,
            sampled_verification_units=sampled,
        )

    def _check_openings(
        self,
        demanded: tuple[Obligation, ...],
        kinds: Sequence[KindProgram],
        gate_set_digest: bytes,
        width: int,
        message: EvidenceMessage,
    ) -> None:
        """Transparent evidence: one opening batch per sampled VU, exactly its
        required positions in order, checked as a one-obligation proof."""

        gate_set = self._layout.circuit.gate_set
        opened_total = 0
        for unit, obligation, batch in zip(
            self.selected_verification_units, demanded, message.units, strict=True
        ):
            opened_total += len(batch)
            self._limits.enforce("max_openings", opened_total)
            if tuple(item.position for item in batch) != tuple(
                ref.position for ref in obligation.positions
            ):
                raise Reject(
                    VerificationCode.COVERAGE_MISMATCH,
                    f"evidence for unit {unit} must open exactly its required positions",
                )
            statement = make_statement(gate_set.id, gate_set_digest, width, kinds, (obligation,))
            witness = Witness((tuple((item.value, item.path) for item in batch),))
            if not self._backend.verify(statement, encode_witness(witness)):
                raise Reject(
                    VerificationCode.PROOF_REJECTED, f"openings for unit {unit} do not verify"
                )

    @property
    def transcript(self) -> Transcript:
        if self._phase != "done":
            raise ProtocolError("the protocol has not finished")
        return Transcript(*self.transcript_parts)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class ProtocolRun:
    report: VerificationReport
    transcript: Transcript | None


def run_protocol(
    compiled: Compiled,
    expectation: Expectation,
    values: Values,
    *,
    replay: Replay | None = None,
    limits: VerificationLimits | None = None,
    weight_tree: MerkleTree | None = None,
    backend: ProofBackend | None = None,
    plan: BatchPlan | None = None,
    declare: Declare | None = None,
) -> ProtocolRun:
    """Run prover and verifier against each other in one process.

    ``values`` supply the boundary and the weights; by default the prover
    *replays* each selected interior honestly from the boundary, so interior
    entries of ``values`` are not what gets committed.  To model a prover that
    commits a dishonest interior, pass ``replay=assignment_replay(values)``.
    When the expectation binds weights and no ``weight_tree`` is given, the
    prover commits the weights in ``values`` (an honest prover's tree).
    ``backend`` (shared by both parties) and ``plan`` (the prover's batching)
    route the reveal step through a proof backend other than the default.
    ``declare`` is the prover's fault-declaration policy; an honest server
    that streamed a faulty result passes ``replay=assignment_replay(values)``
    (it commits what it computed) and ``declare=honest_declare(compiled)``.
    """

    try:
        verifier = VerifierSession(expectation, compiled, limits=limits, backend=backend)
    except Reject as rejection:
        return ProtocolRun(rejection_report(rejection, None), None)
    if expectation.weights is not None and weight_tree is None:
        try:
            weight_values = [values[address] for address in compiled.circuit.weights]
        except KeyError as error:
            raise ProtocolError(f"prover has no value for weight gate {error.args[0]}") from None
        _, weight_tree = commit_weights(compiled.circuit.gate_set, weight_values)
    prover = ProverSession(
        compiled,
        verifier.header,
        values,
        replay=replay,
        limits=limits,
        weight_tree=weight_tree,
        backend=backend,
        plan=plan,
        declare=declare,
    )
    try:
        replay_challenge = verifier.receive_boundary(prover.boundary())
        sample_challenge = verifier.receive_interiors(prover.interiors(replay_challenge))
        report = verifier.receive_evidence(prover.evidence(sample_challenge))
    except Reject as rejection:
        return ProtocolRun(rejection_report(rejection, verifier), None)
    return ProtocolRun(report, verifier.transcript)
