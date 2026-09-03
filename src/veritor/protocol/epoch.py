"""The epoch layer: rounds of runs committed before any challenge, priced as one circuit.

An **epoch** (think: one year) is the unit of the guarantee.  Runs -- today's
sessions -- are admitted and commit their boundaries as they happen; every
admitted header and accepted boundary is appended to a hash-chained
**commitment stream**.  No seed exists for a run while its round is open.
When the verifier **closes the round** it fixes the stream's last link as the
round's **seal**, draws (or receives) one ``round_seed``, derives every run's
``(q_seed, s_seed)`` from the seed and the seal, and drives each run through
the per-run machinery of :mod:`veritor.protocol.session` exactly as today:
replay challenge, interior commitments, sample challenge, evidence, verdict.

The round's capacity is one ``Bound`` over the :func:`~veritor.analysis.union`
of its runs' kind tables at ``eta / rounds``; the epoch's is the sum over
rounds.  Sampling is Bernoulli per unit (:mod:`veritor.protocol.challenge`),
so sampling the union with the round's seeds is exactly sampling each run
with its own derived seed: the per-run challenges *are* the union's
challenge.  Soundness rests on the seal: every boundary of the round is fixed
before any seed is known, and within a run the interiors are committed after
its replay challenge and before its sample challenge, as today.

The prover side (:class:`EpochProver`) holds one :class:`ProverSession` per
run.  In a real system round close is where deterministic replay from stored
boundary values happens; the replay unit is the unit of recompute, and the
storage a round costs is the boundary data of its replay units.  No storage
machinery lives here.
"""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from secrets import token_bytes

from veritor.analysis import bound, union
from veritor.compile import Compilation
from veritor.core import (
    Compiled,
    JSONValue,
    KindTable,
    ProbabilityInput,
    VerificationLimits,
    VerificationPolicy,
    exact_fraction,
)

from .domains import commit_weights
from .merkle import MerkleTree
from .messages import (
    TRANSPARENT_BACKEND,
    BoundaryMessage,
    EvidenceMessage,
    Header,
    InteriorMessage,
    ProtocolError,
    Reject,
    ReplayChallenge,
    SampleChallenge,
    Transcript,
    VerificationCode,
    VerificationReport,
    Weights,
    rational_manifest,
    raw_digest,
)
from .parameters import DEFAULT_MAX_WORK, VerifierParameters
from .proofs import BatchPlan, ProofBackend
from .session import (
    Claim,
    Declare,
    ProverSession,
    Replay,
    Values,
    VerifierSession,
    check_seed,
    rejection_report,
)

SEED_DOMAIN = b"veritor/protocol/epoch/seed/v1"
"""Domain separation for the per-run seeds derived at round close (the one place)."""
GENESIS_TAG = "veritor/protocol/epoch/genesis/v1"
"""The first link of the stream: the epoch's parameters."""
LINK_TAG = "veritor/protocol/epoch/link/v1"
"""``link_i = H(link_{i-1} || header_i.digest || boundary_phase_i)``, as a tagged manifest."""


@dataclass(frozen=True, slots=True, init=False)
class EpochParameters:
    """What the verifier fixes for a whole epoch.

    ``eta`` is the epoch's acceptance threshold: the bound the epoch
    certifies holds at ``eta`` over the whole epoch.  ``rounds`` is how many
    challenge rounds the epoch has; each round is bounded at ``eta / rounds``
    and the sum over rounds is the epoch's bound at ``eta`` (``rounds = 1``
    is the single end-of-epoch challenge the headline estimate assumes;
    ``rounds = N`` for ``N`` runs recovers per-run challenges).  ``policy``
    is ``theta = (q, s)``, fixed by the verifier for every run of the epoch:
    the client proposes nothing.  ``max_advice_bits`` (``A``) and
    ``max_work`` (``W_max``) are the per-run caps of
    :class:`~veritor.protocol.parameters.VerifierParameters`, checked at each
    admission.  ``max_capacity`` (``U_max``) caps the *round's* capacity: a
    run is admitted only if ``Bound`` over the union of the round's tables
    with the run's stays at most ``U_max`` bits (``None`` waives the check
    and has to be written out).  ``max_faults`` is the round's fault budget:
    every header of the round carries all of it, any one run may use it, and
    the round rejects when its runs' declarations together exceed it; the
    union bound is charged with the budget once.
    """

    eta: Fraction
    rounds: int
    policy: VerificationPolicy
    max_capacity: int | None
    max_advice_bits: int
    max_work: int
    max_faults: int

    def __init__(
        self,
        eta: ProbabilityInput,
        policy: VerificationPolicy,
        *,
        max_capacity: int | None,
        rounds: int = 1,
        max_advice_bits: int = 0,
        max_work: int = DEFAULT_MAX_WORK,
        max_faults: int = 0,
    ) -> None:
        if type(rounds) is not int or rounds < 1:
            raise ProtocolError("rounds must be a positive integer")
        if not isinstance(policy, VerificationPolicy):
            raise ProtocolError("the epoch's policy must be a VerificationPolicy")
        checked = exact_fraction(eta, name="eta")
        if not 0 <= checked < 1:
            raise ProtocolError("eta must lie in [0, 1)")
        run = VerifierParameters(  # validates the caps exactly as a session would
            checked / rounds,
            max_capacity=max_capacity,
            max_advice_bits=max_advice_bits,
            max_work=max_work,
            max_faults=max_faults,
        )
        object.__setattr__(self, "eta", checked)
        object.__setattr__(self, "rounds", rounds)
        object.__setattr__(self, "policy", policy)
        object.__setattr__(self, "max_capacity", run.max_capacity)
        object.__setattr__(self, "max_advice_bits", run.max_advice_bits)
        object.__setattr__(self, "max_work", run.max_work)
        object.__setattr__(self, "max_faults", run.max_faults)

    @property
    def round_eta(self) -> Fraction:
        """The threshold each round is bounded at: ``eta / rounds``."""

        return self.eta / self.rounds

    @property
    def run_parameters(self) -> VerifierParameters:
        """What each run's header binds: the round's ``eta``, the per-run caps,
        the round's fault budget -- and no per-run ``U_max`` (the cap is the round's)."""

        return VerifierParameters(
            self.round_eta,
            max_capacity=None,
            max_advice_bits=self.max_advice_bits,
            max_work=self.max_work,
            max_faults=self.max_faults,
        )

    @property
    def manifest(self) -> dict[str, JSONValue]:
        return {
            "eta": rational_manifest(self.eta),
            "max_advice_bits": self.max_advice_bits,
            "max_capacity": self.max_capacity,
            "max_faults": self.max_faults,
            "max_work": self.max_work,
            "policy": self.policy.manifest,
            "rounds": self.rounds,
        }


def derive_run_seed(
    round_seed: bytes, seal: bytes, round_index: int, run_index: int, label: bytes
) -> bytes:
    """``HMAC-SHA256(round_seed, domain || seal || round || run || label)``: one run's seed.

    ``label`` is ``b"q"`` or ``b"s"``.  The seal is the round's; ``run_index``
    is the run's admission index within the round.  Nothing here is known
    before the seal, and the round seed is the verifier's alone.
    """

    frame = (
        SEED_DOMAIN
        + b"\0"
        + check_seed("seal", seal)
        + round_index.to_bytes(8, "big")
        + run_index.to_bytes(8, "big")
        + label
    )
    return hmac.new(
        check_seed("round seed", round_seed), frame, hashlib.sha256
    ).digest()


def stream_link(previous: bytes, header: Header, boundary_phase_digest: bytes) -> bytes:
    """The next link of the commitment stream over one run's header and boundary."""

    return raw_digest(
        LINK_TAG,
        {
            "boundary": boundary_phase_digest.hex(),
            "header": header.digest.hex(),
            "previous": previous.hex(),
        },
    )


@dataclass(frozen=True, slots=True)
class RoundChallenge:
    """What closing a round releases: the seal and one replay challenge per run.

    ``challenges`` pairs each run's header with its :class:`ReplayChallenge`
    (which carries the run's ``q_seed``), in admission order, for the runs
    whose boundary the round accepted.  The round seed itself is not here:
    the ``s`` seeds are derived from it and stay the verifier's until each
    run's interiors are committed.
    """

    round: int
    seal: bytes
    challenges: tuple[tuple[Header, ReplayChallenge], ...]


@dataclass(frozen=True, slots=True)
class RunReport:
    """One run's place in the epoch: its header and its verdict so far."""

    header: Header
    round: int
    report: VerificationReport | None
    """``None`` while the run is still being answered."""

    @property
    def accepted(self) -> bool:
        return self.report is not None and self.report.accepted


@dataclass(frozen=True, slots=True)
class RoundReport:
    """One round: its seal, its runs, its capacity and its declarations."""

    index: int
    seal: bytes | None
    """The stream's last link when the round closed; ``None`` while open."""
    runs: tuple[RunReport, ...]
    refused: tuple[VerificationReport, ...]
    """Admissions the round turned away; not committed, not part of the verdict."""
    capacity_bits: float
    """``Bound`` over the union of the round's tables at ``eta / rounds`` (0 for no runs)."""
    declarations: int
    """Fault declarations accepted across the round's runs, against the budget."""

    @property
    def closed(self) -> bool:
        return self.seal is not None

    @property
    def accepted(self) -> bool:
        return self.closed and all(run.accepted for run in self.runs)


@dataclass(frozen=True, slots=True)
class EpochReport:
    """The epoch's verdict: accepted only if every committed run was challenged and accepted."""

    code: VerificationCode
    detail: str
    parameters: EpochParameters
    rounds: tuple[RoundReport, ...]
    capacity_bits: float
    """The sum over rounds of the round bounds: the epoch's ``Bound`` at ``eta``."""

    @property
    def accepted(self) -> bool:
        return self.code is VerificationCode.ACCEPTED

    @property
    def run_count(self) -> int:
        return sum(len(round.runs) for round in self.rounds)


@dataclass(slots=True)
class _Run:
    header: Header
    session: VerifierSession
    table: KindTable
    round: int
    index: int
    """Admission index within the round."""
    boundary: BoundaryMessage | None = None
    report: VerificationReport | None = None

    def fail(self, rejection: Reject) -> Reject:
        """Record ``rejection`` as the run's verdict."""

        self.report = rejection_report(rejection, self.session)
        return rejection

    def judged(self, rejection: Reject) -> Reject:
        """A rejection out of the session: the run's verdict if the session rejected
        (its phase is ``rejected``), else a message refused out of phase, as
        :class:`VerifierSession` refuses it, with the run still live."""

        if self.session.phase == "rejected":
            self.fail(rejection)
        return rejection

    def live(self) -> None:
        if self.report is not None:
            raise Reject(
                VerificationCode.INVALID_PHASE,
                f"the run has its verdict: {self.report.code.value}",
            )


@dataclass(slots=True)
class _Round:
    index: int
    runs: list[_Run] = field(default_factory=list)
    refused: list[VerificationReport] = field(default_factory=list)
    seal: bytes | None = None
    capacity_bits: float = 0.0
    declarations: int = 0


class EpochVerifier:
    """The verifier of one epoch: admits runs, chains their boundaries, challenges at round close.

    ``limits`` and ``backend`` (a proof backend implementation) are handed to
    every per-run session.  :meth:`admit` and :meth:`receive_boundary` are
    the run-time steps; :meth:`close_round` seals the round and releases its
    challenges; :meth:`receive_interiors` and :meth:`receive_evidence` finish
    each run as :class:`VerifierSession` does; :meth:`report` is the verdict
    so far.  Every rejection is raised as a :class:`Reject`; one that judges
    the run (the session rejected, the budget is exceeded, the boundary never
    came) is recorded against it, so the report reflects it, while a message
    out of phase -- a second boundary, a boundary after the round closed, a
    message to a run with its verdict -- is refused and changes nothing.
    """

    def __init__(
        self,
        parameters: EpochParameters,
        *,
        limits: VerificationLimits | None = None,
        backend: ProofBackend | None = None,
    ) -> None:
        if not isinstance(parameters, EpochParameters):
            raise ProtocolError("an epoch needs EpochParameters")
        self.parameters = parameters
        self._limits = limits
        self._backend = backend
        self._runs: dict[bytes, _Run] = {}
        self._rounds: list[_Round] = [_Round(0)]
        self._link = raw_digest(GENESIS_TAG, {"parameters": parameters.manifest})
        self.stream: list[Header | BoundaryMessage] = []
        """Every admitted header and accepted boundary, in order: what the seal binds."""

    @property
    def round(self) -> int:
        """The index of the open round, or ``rounds`` once the epoch is complete."""

        return (
            len(self._rounds) - 1
            if self._rounds[-1].seal is None
            else len(self._rounds)
        )

    @property
    def link(self) -> bytes:
        """The stream's current last link (the next seal, if the round closed now)."""

        return self._link

    def _open_round(self) -> _Round:
        current = self._rounds[-1]
        if current.seal is not None:
            raise ProtocolError(
                f"the epoch's {self.parameters.rounds} rounds are all closed"
            )
        return current

    def _run(self, header: Header) -> _Run:
        if not isinstance(header, Header):
            raise ProtocolError("runs are named by their Header")
        run = self._runs.get(header.digest)
        if run is None or run.header != header:
            raise Reject(
                VerificationCode.EXPECTATION_MISMATCH,
                "the header names no run this epoch admitted",
            )
        return run

    def admit(
        self,
        compilation: Compilation,
        claimed_outputs: Sequence[object],
        *,
        weights: Weights | None = None,
        backend: str = TRANSPARENT_BACKEND,
        session_id: bytes | None = None,
    ) -> Header:
        """Admit a run into the open round and issue its header.

        The per-run checks are :class:`VerifierSession`'s (the advice against
        ``A``, the limits, the expected work against ``W_max``); the capacity
        check is the round's: ``Bound`` over the union of the round's tables
        and this run's, at ``eta / rounds`` with the round's fault budget,
        must not exceed ``U_max``.  A refused run is recorded in the round
        report and is not committed.  No seed exists for the run yet.
        """

        current = self._open_round()
        if not isinstance(compilation, Compilation):
            raise ProtocolError("admit requires a Compilation from Compile")
        parameters = self.parameters
        claim = Claim(
            token_bytes(16) if session_id is None else session_id,
            compilation.compiled.digest,
            compilation.constructor,
            compilation.advice,
            parameters.policy,
            parameters.run_parameters,
            tuple(compilation.inputs),
            tuple(claimed_outputs),
            weights,
            backend,
        )
        try:
            session = VerifierSession(
                claim, compilation.compiled, limits=self._limits, backend=self._backend
            )
            table = compilation.compiled.kind_table()
            tables = [run.table for run in current.runs]
            tables.append(table)
            capacity = self._capacity(tables)
            if (
                parameters.max_capacity is not None
                and capacity > parameters.max_capacity
            ):
                raise Reject(
                    VerificationCode.POLICY_REJECTED,
                    f"the round's Bound would be {capacity:.6g} bits with this run, "
                    f"exceeding U_max {parameters.max_capacity}",
                )
        except Reject as rejection:
            current.refused.append(rejection_report(rejection, None))
            raise
        if session.header.digest in self._runs:
            raise ProtocolError("a run with this header was already admitted")
        run = _Run(session.header, session, table, current.index, len(current.runs))
        current.runs.append(run)
        current.capacity_bits = capacity
        self._runs[run.header.digest] = run
        self.stream.append(run.header)
        return run.header

    def _capacity(self, tables: Sequence[KindTable]) -> float:
        if not tables:
            return 0.0
        parameters = self.parameters
        return bound(
            union(tables),
            parameters.policy,
            parameters.round_eta,
            max_faults=parameters.max_faults,
        ).bits

    def receive_boundary(self, header: Header, message: BoundaryMessage) -> None:
        """Accept a run's boundary into the stream, once, in the run's own round.

        The checks are :meth:`VerifierSession.accept_boundary`'s; nothing is
        derived.  The stream's link advances over the header's digest and
        the boundary-phase digest.
        """

        run = self._run(header)
        current = self._open_round()
        # out of phase, as a session's _expect: the message is refused, the run not judged
        if run.round != current.index:
            raise Reject(
                VerificationCode.INVALID_PHASE,
                f"the run was admitted in round {run.round}, which is closed",
            )
        run.live()
        if run.boundary is not None:
            raise Reject(
                VerificationCode.INVALID_PHASE,
                "the run's boundary was already received",
            )
        try:
            run.session.accept_boundary(message)
        except Reject as rejection:
            raise run.judged(rejection) from None
        run.boundary = message
        self._link = stream_link(self._link, run.header, run.session.boundary_phase)
        self.stream.append(message)

    def close_round(self, round_seed: bytes) -> RoundChallenge:
        """Seal the round and release its challenges.

        ``round_seed`` is 32 bytes the verifier draws fresh (or takes from a
        beacon) only now, after the seal.  Each run of the round whose
        boundary arrived gets ``q_seed, s_seed = derive_run_seed(round_seed,
        seal, round, run, b"q" | b"s")`` and its replay challenge; a run
        without a boundary is recorded as rejected.  The round's capacity is
        fixed as ``Bound`` over the union of its tables.  The next round
        opens, continuing the stream from the seal.
        """

        current = self._open_round()
        check_seed("round seed", round_seed)
        seal = self._link
        current.seal = seal
        current.capacity_bits = self._capacity([run.table for run in current.runs])
        challenges: list[tuple[Header, ReplayChallenge]] = []
        for run in current.runs:
            if run.report is not None:
                continue
            if run.boundary is None:
                run.report = VerificationReport(
                    VerificationCode.INVALID_PHASE,
                    "the boundary never arrived before the round closed",
                )
                continue
            run.session.release(
                derive_run_seed(round_seed, seal, current.index, run.index, b"q"),
                derive_run_seed(round_seed, seal, current.index, run.index, b"s"),
            )
            try:
                challenges.append((run.header, run.session.challenge_replay()))
            except Reject as rejection:
                run.fail(rejection)
        if len(self._rounds) < self.parameters.rounds:
            self._rounds.append(_Round(current.index + 1))
        return RoundChallenge(current.index, seal, tuple(challenges))

    def receive_interiors(
        self, header: Header, message: InteriorMessage
    ) -> SampleChallenge:
        """A run's interiors after its round closed; the round's fault budget is enforced first."""

        run = self._run(header)
        run.live()
        round = self._rounds[run.round]
        budget = self.parameters.max_faults
        if (
            run.session.phase == "interiors"
            and round.declarations + len(message.declarations) > budget
        ):
            raise run.fail(
                Reject(
                    VerificationCode.FAULTS_EXCEEDED,
                    f"{len(message.declarations)} declarations after {round.declarations} in the "
                    f"round exceed its budget of {budget}",
                )
            )
        try:
            challenge = run.session.receive_interiors(message)
        except Reject as rejection:
            raise run.judged(rejection) from None
        round.declarations += len(message.declarations)
        return challenge

    def receive_evidence(
        self, header: Header, message: EvidenceMessage
    ) -> VerificationReport:
        """A run's evidence: its verdict, recorded for the epoch."""

        run = self._run(header)
        run.live()
        try:
            run.report = run.session.receive_evidence(message)
        except Reject as rejection:
            raise run.judged(rejection) from None
        return run.report

    def transcript(self, header: Header) -> Transcript:
        """The per-run transcript, once the run is done (verifiable by ``verify_transcript``
        under an :class:`Expectation` carrying the run's released seeds)."""

        return self._run(header).session.transcript

    def report(self) -> EpochReport:
        """The verdict so far: accepted only when every round is closed and every
        committed run was challenged and accepted; otherwise the first failure and why."""

        rounds = tuple(
            RoundReport(
                round.index,
                round.seal,
                tuple(
                    RunReport(run.header, run.round, run.report) for run in round.runs
                ),
                tuple(round.refused),
                round.capacity_bits,
                round.declarations,
            )
            for round in self._rounds
        )
        capacity = sum(round.capacity_bits for round in rounds)
        code, detail = self._verdict(rounds)
        return EpochReport(code, detail, self.parameters, rounds, capacity)

    def _verdict(self, rounds: tuple[RoundReport, ...]) -> tuple[VerificationCode, str]:
        """The first failure in stream order, else whether the epoch is complete."""

        for round in rounds:
            for run in round.runs:
                name = f"run {run.header.session_id.hex()} of round {round.index}"
                if run.report is None:
                    return VerificationCode.INVALID_PHASE, f"{name} was not answered"
                if not run.report.accepted:
                    return run.report.code, f"{name}: {run.report.detail}"
        if len(rounds) < self.parameters.rounds or not rounds[-1].closed:
            open_round = f"round {self.round} of {self.parameters.rounds} is open"
            return VerificationCode.INVALID_PHASE, open_round
        return VerificationCode.ACCEPTED, ""


class EpochProver:
    """The prover of one epoch: one :class:`ProverSession` per run, keyed by header.

    :meth:`boundary` registers a run with everything a :class:`ProverSession`
    takes and sends its boundary at run time; :meth:`interiors` and
    :meth:`evidence` answer the run's challenges after its round closed.
    Deterministic replay from the stored boundary is :class:`ProverSession`'s
    ``replay`` (honest by default).
    """

    def __init__(
        self,
        *,
        limits: VerificationLimits | None = None,
        backend: ProofBackend | None = None,
    ) -> None:
        self._limits = limits
        self._backend = backend
        self._sessions: dict[bytes, ProverSession] = {}

    def boundary(
        self,
        compiled: Compiled,
        header: Header,
        values: Values,
        *,
        replay: Replay | None = None,
        weight_tree: MerkleTree | None = None,
        plan: BatchPlan | None = None,
        declare: Declare | None = None,
    ) -> BoundaryMessage:
        if header.digest in self._sessions:
            raise ProtocolError("the run's boundary was already sent")
        session = ProverSession(
            compiled,
            header,
            values,
            replay=replay,
            limits=self._limits,
            weight_tree=weight_tree,
            backend=self._backend,
            plan=plan,
            declare=declare,
        )
        self._sessions[header.digest] = session
        return session.boundary()

    def _session(self, header: Header) -> ProverSession:
        try:
            return self._sessions[header.digest]
        except KeyError:
            raise ProtocolError(
                "the header names no run this prover sent a boundary for"
            ) from None

    def interiors(self, header: Header, challenge: ReplayChallenge) -> InteriorMessage:
        return self._session(header).interiors(challenge)

    def evidence(self, header: Header, challenge: SampleChallenge) -> EvidenceMessage:
        return self._session(header).evidence(challenge)

    def transcript(self, header: Header) -> Transcript:
        return self._session(header).transcript


@dataclass(frozen=True, slots=True)
class Run:
    """One run of an epoch as :func:`run_epoch` drives it: what :func:`run_protocol` takes.

    ``values`` supply the boundary and the weights; ``weights`` is the
    model's ``kappa_W`` (computed from ``values`` with the honest tree when
    the circuit has weight gates and none is given); ``replay``, ``declare``,
    ``plan`` and ``weight_tree`` are the prover's, as in :class:`ProverSession`.
    """

    compilation: Compilation
    values: Values
    claimed_outputs: Sequence[object]
    weights: Weights | None = None
    weight_tree: MerkleTree | None = None
    replay: Replay | None = None
    declare: Declare | None = None
    plan: BatchPlan | None = None
    backend: str = TRANSPARENT_BACKEND
    session_id: bytes | None = None


def run_epoch(
    parameters: EpochParameters,
    runs: Sequence[Run],
    schedule: Sequence[Sequence[int]],
    seeds: Sequence[bytes],
    *,
    limits: VerificationLimits | None = None,
    backend: ProofBackend | None = None,
) -> EpochReport:
    """Run an epoch's prover and verifier against each other in one process.

    ``schedule[r]`` lists the indices into ``runs`` admitted in round ``r``
    (one entry per round of ``parameters``); ``seeds[r]`` is round ``r``'s
    seed.  Each round: every run is admitted and sends its boundary, the
    round closes, and every challenged run answers.  A run the verifier
    refuses at admission or rejects along the way is left where the verifier
    left it; the report says so.
    """

    if len(schedule) != parameters.rounds or len(seeds) != parameters.rounds:
        raise ProtocolError(
            f"the epoch has {parameters.rounds} rounds: one schedule entry and one seed each"
        )
    verifier = EpochVerifier(parameters, limits=limits, backend=backend)
    prover = EpochProver(limits=limits, backend=backend)
    for members, round_seed in zip(schedule, seeds, strict=True):
        for index in members:
            run = runs[index]
            compiled = run.compilation.compiled
            weights, weight_tree = run.weights, run.weight_tree
            if weights is None and compiled.index.weight_count:
                try:
                    weight_values = [
                        run.values[address] for address in compiled.circuit.weights
                    ]
                except KeyError as error:
                    raise ProtocolError(
                        f"run {index} has no value for weight gate {error.args[0]}"
                    ) from None
                weights, weight_tree = commit_weights(
                    compiled.circuit.gate_set, weight_values
                )
            try:
                header = verifier.admit(
                    run.compilation,
                    run.claimed_outputs,
                    weights=weights,
                    backend=run.backend,
                    session_id=run.session_id,
                )
            except Reject:
                continue
            message = prover.boundary(
                compiled,
                header,
                run.values,
                replay=run.replay,
                weight_tree=weight_tree,
                plan=run.plan,
                declare=run.declare,
            )
            try:
                verifier.receive_boundary(header, message)
            except Reject:
                continue
        challenge = verifier.close_round(round_seed)
        for header, replay_challenge in challenge.challenges:
            try:
                sample = verifier.receive_interiors(
                    header, prover.interiors(header, replay_challenge)
                )
                verifier.receive_evidence(header, prover.evidence(header, sample))
            except Reject:
                continue
    return verifier.report()


__all__ = [
    "GENESIS_TAG",
    "LINK_TAG",
    "SEED_DOMAIN",
    "EpochParameters",
    "EpochProver",
    "EpochReport",
    "EpochVerifier",
    "RoundChallenge",
    "RoundReport",
    "Run",
    "RunReport",
    "derive_run_seed",
    "run_epoch",
    "stream_link",
]
