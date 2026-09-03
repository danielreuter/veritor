"""The honest prover: what it records in production, how it replays, what it declares.

The protocol's :class:`~veritor.protocol.ProverSession` was written as if the
server kept every gate of the production run (``assignment_replay``) and found
its faults by :func:`~veritor.protocol.self_check` over that assignment.  A
real server keeps the boundary it has to commit plus whatever it chooses to
log, and *reconstructs* an opened RU's interior when the q-challenge asks for
it.  When a recomputed value disagrees with a value it recorded -- and, for a
boundary value, already committed -- the commitment is binding: the server
keeps the recorded value and declares the VU that produced it.  This module
models that server (``docs/honest-prover.md``, sections 3 and 5).

**Recording policies.**  :class:`RecordingPolicy` names what the server
keeps of a run (:func:`record` restricts the omniscient assignment to it):

* ``BOUNDARY``: the inputs, the weights and every RU's declared outputs
  ``Out(R)`` -- what the boundary commitment and the weight tree need and
  nothing more.  For ``RequestsG`` (RU = request) the recorded computed
  values of a run are its streamed tokens; for ``ClusterG`` (RU = step) the
  KV values and tokens that cross steps as well.
* ``VU_OUTPUTS``: the boundary plus every VU's declared output word, the
  interior positions of every RU -- the server logs every kernel's output.
  A VU's internal gates are recorded under no policy: they are never
  committed and never read from outside the VU.

**Pinned replay.**  :func:`replay_pinned` is honest replay from a recording.
The gates of RU ``R`` are recomputed in address order from the recorded
inputs, weights and boundary, as :func:`~veritor.protocol.replay_unit` does,
with one rule.  At an address the server *recorded*, the recorded value is
pinned: it is the value the interior commits at that address, and it is the
value every later gate of the replay reads there; the recomputed value is
compared with it and, when the two differ, the VU owning the address is
added to the pinned set.  At an address the server did not record, the
recomputed value is stored and read downstream.  The interior this yields
satisfies every VU's relation except at the pinned VUs, whose relations fail
against their own inputs: a reader of a pinned value was recomputed from that
value, so its relation holds against it.  The pinned VUs are therefore
exactly what the server must declare (M6) for the run to be accepted whatever
the s-challenge samples, and :func:`~veritor.protocol.self_check` over the
committed interior finds the same set (``tests/veritor/stress/test_honest_replay.py``
asserts it).  Two consequences shape the declaration counts of section 4: a
fault that changed no recorded value costs no declaration, since the replay
recomputes the correct interior and it agrees with the recording; and a fault
that changed recorded values costs one declaration per recorded value it
changed, whether or not the VU that produced it is the one that faulted --
with tokens-only recording the declarations name the tokens that came out
wrong, not the kernel that went wrong.  The replay consults nothing the
policy did not record: ``recorded`` is the only source of values.

**Fault classes.**  :class:`FaultClass` and the injection helpers produce
the production run of each class in the simulation's omniscient view
(:class:`Production`: the full assignment, the streamed outputs, where the
fault landed).  Stored corruptions (an output word flipped after it was
computed) are ``FaultInjector.propagate`` flips; read faults (a gate reads a
cell as another value while the stored value, and the value the boundary or
the weight root commits, stays right) are its ``misreads``.  The pinned gate
set of the toy decoder is integer arithmetic without a NaN or infinity, so
the catastrophic class corrupts every bit of an early word: the garbage is
a finite value that propagates like any other wrong value.  Not every bit
of a word is significant for the gate that reads it (the toy attention's
polynomial softmax annihilates the top bits of a key at rest);
:func:`significant_bits` says which are, and :func:`boundary_at_rest` flips
the most significant of them.

**Strategies.**  :class:`Strategy` and :func:`account` price a run under the
four strategies of ``docs/honest-prover.md``, section 5: ``P0`` declares
nothing and takes the rejections; ``P1`` records the boundary, replays the
opened RUs pinned and declares what it pins after ``J`` (the protocol as
built); ``P2`` declares before ``J`` at ``u(1)`` the pins of the RUs a
signal flagged before streaming, and the rest as ``P1``; ``P3`` replays
every RU before ``J`` and declares every pin at ``u(1)``.  A post-J
declaration is priced at ``u_post(1) = rho log2 (1 / (1 - s))``, the slope
of the fold times the threshold one adaptive declaration lowers
(``docs/notes/late-advice.md``; ``(u(1) + 1) / q`` at the scattered
channel); the protocol has no pre-J declaration message, so ``P2`` and
``P3`` are counterfactual accounting: their charge is what a protocol with
that message would bill, their verdict is the built protocol's under ``P1``
declarations.  :func:`phase_diagram` is the same accounting at fleet scale,
analytic in the fault density with Poisson counts.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction

from veritor.core import Compiled, VerificationPolicy
from veritor.core.indexed import iter_members
from veritor.protocol.session import Values

from .faults import FaultInjector, dot_units, fault_budget, poisson_tail

__all__ = [
    "Account",
    "FaultClass",
    "HonestReplay",
    "PhasePoint",
    "Production",
    "RecordingPolicy",
    "Strategy",
    "account",
    "boundary_at_rest",
    "catastrophic",
    "combine",
    "honest_replay",
    "input_read",
    "interior_flip",
    "phase_diagram",
    "pin_everything",
    "record",
    "recorded_addresses",
    "replay_pinned",
    "significant_bits",
    "token_flip",
    "vu_output_read",
    "weight_read",
]

LOG2E = math.log2(math.e)


# -- recording ------------------------------------------------------------------------


class RecordingPolicy(Enum):
    """What the server keeps of a production run."""

    BOUNDARY = "boundary"
    """Inputs, weights, every RU's ``Out(R)``: what the commitments need."""
    VU_OUTPUTS = "vu-outputs"
    """The boundary plus every VU's output word (every RU's interior positions)."""


def recorded_addresses(compiled: Compiled, policy: RecordingPolicy) -> tuple[int, ...]:
    """The addresses ``policy`` records, ascending."""

    index = compiled.index
    addresses = set(iter_members(index.boundary())) | set(iter_members(index.weights()))
    if policy is RecordingPolicy.VU_OUTPUTS:
        for unit in range(index.replay_units.count):
            addresses.update(iter_members(index.interior(unit)))
    return tuple(sorted(addresses))


def record(
    compiled: Compiled, values: Values, policy: RecordingPolicy
) -> dict[int, object]:
    """What the server holds of the production assignment ``values`` under ``policy``."""

    return {
        address: values[address] for address in recorded_addresses(compiled, policy)
    }


# -- pinned replay --------------------------------------------------------------------


def replay_pinned(
    compiled: Compiled, unit: int, recorded: Values
) -> tuple[dict[int, object], tuple[int, ...]]:
    """Honest replay of RU ``unit`` from ``recorded``: ``(interior, pinned_vus)``.

    Every computed gate of the RU is recomputed in address order from
    ``recorded`` (sources) and the values computed so far.  Where the address
    is recorded, the recorded value is what the interior holds and what later
    gates read; if it differs from the recomputation, the VU owning the
    address is pinned.  ``pinned_vus`` are global VU indices, ascending: the
    declarations the RU needs.
    """

    circuit, index = compiled.circuit, compiled.index
    interior: dict[int, object] = {}
    disagreeing: list[int] = []
    for address in index.replay_units.unit(unit).interval:
        gate = circuit[address]
        if gate.is_source:
            continue
        arguments = []
        for argument in gate.args:
            if argument in interior:
                arguments.append(interior[argument])
            else:
                try:
                    arguments.append(recorded[argument])
                except KeyError as error:
                    raise KeyError(
                        f"replay of unit {unit} needs the value of address {argument}, "
                        "which is neither recorded nor computed by the unit"
                    ) from error
        value: object = circuit.evaluate_gate(address, arguments)  # type: ignore[arg-type]
        try:
            pinned = recorded[address]
        except KeyError:
            pass
        else:
            if pinned != value:
                disagreeing.append(address)
            value = pinned
        interior[address] = value
    units = index.verification_units(unit)
    return interior, tuple(sorted({units.first + units.owner(a) for a in disagreeing}))


class HonestReplay:
    """``ProverSession``'s ``replay`` and ``declare`` for a server holding ``recorded``.

    ``replay`` is :func:`replay_pinned`'s interior and remembers what it
    pinned; ``declare`` returns it.  ``pinned`` maps every replayed RU to its
    declarations once the session has replayed it.
    """

    def __init__(self, compiled: Compiled, recorded: Values) -> None:
        self.compiled = compiled
        self.recorded = recorded
        self.pinned: dict[int, tuple[int, ...]] = {}

    def replay(self, unit: int, values: Values) -> Values:
        del values  # the session's values are the recording this replay reads
        interior, pinned = replay_pinned(self.compiled, unit, self.recorded)
        self.pinned[unit] = pinned
        return interior

    def declare(self, unit: int, values: Values) -> Iterable[int]:
        del values
        return self.pinned[unit]


def honest_replay(compiled: Compiled, recorded: Values) -> HonestReplay:
    """The honest server's replay from ``recorded``; pass ``.replay`` and ``.declare`` to the session."""

    return HonestReplay(compiled, recorded)


def pin_everything(compiled: Compiled, recorded: Values) -> dict[int, tuple[int, ...]]:
    """:func:`replay_pinned`'s pinned VUs for every RU: the declarations the whole run needs."""

    return {
        unit: replay_pinned(compiled, unit, recorded)[1]
        for unit in range(compiled.index.replay_units.count)
    }


# -- fault classes --------------------------------------------------------------------


class FaultClass(Enum):
    """Where a silent fault lands and how (``docs/honest-prover.md``, section 4)."""

    INTERIOR_FLIP = "interior VU-output bit flip"
    """One bit of an interior dot product's output word, streamed tokens unchanged."""
    TOKEN_FLIP = "token flip"
    """A flip of a dot product's output word that changed a streamed token."""
    CATASTROPHIC = "catastrophic"
    """Every bit of an early dot product's output word: a garbage tail."""
    WEIGHT_READ = "weight-source read fault"
    """A pod reads one weight cell as another value for the whole run; the root is right."""
    INPUT_READ = "input-source read fault"
    """A pod reads one input token as another value; the boundary holds the right one."""
    BOUNDARY_AT_REST = "boundary at rest"
    """``ClusterG``: a step's KV word, committed right by its producer, read corrupted by the consuming step."""
    VU_OUTPUT_READ = "VU-output read fault"
    """A VU computed right; its consumers read a corrupted copy of its output word."""


@dataclass(frozen=True, slots=True)
class Production:
    """One faulty production run in the simulation's omniscient view."""

    fault: FaultClass
    address: int
    """The word the fault hit: stored corrupted (flips) or read corrupted (read faults)."""
    replay_unit: int
    """The RU owning ``address`` (the weights' RU for a weight cell)."""
    correct: int
    """The word's correct value: what the recording and the commitments hold for a read fault."""
    corrupted: int
    """The value stored (flips) or read (read faults) instead."""
    misreaders: tuple[int, ...]
    """The gates that read ``corrupted``; empty when the stored word itself is corrupted."""
    values: Mapping[int, int]
    """The full production assignment as the server computed it."""
    outputs: tuple[int, ...]
    """The tokens the users received."""
    honest_outputs: tuple[int, ...]

    @property
    def stored(self) -> bool:
        """Whether the word itself is corrupted (else only its readers' copies are)."""

        return not self.misreaders

    @property
    def changed_outputs(self) -> int:
        return sum(
            a != b for a, b in zip(self.outputs, self.honest_outputs, strict=True)
        )


def _production(
    injector: FaultInjector,
    fault: FaultClass,
    address: int,
    corrupted: int,
    *,
    misreaders: Sequence[int] = (),
) -> Production:
    compiled = injector.compiled
    correct = injector.honest[address]
    if misreaders:
        values = injector.propagate({}, {r: {address: corrupted} for r in misreaders})
    else:
        values = injector.propagate({address: correct ^ corrupted})
    outputs = compiled.circuit.outputs
    return Production(
        fault=fault,
        address=address,
        replay_unit=compiled.index.replay_units.owner(address),
        correct=correct,
        corrupted=corrupted,
        misreaders=tuple(misreaders),
        values=values,
        outputs=tuple(values[a] for a in outputs),
        honest_outputs=tuple(injector.honest[a] for a in outputs),
    )


def combine(injector: FaultInjector, faults: Iterable[Production]) -> dict[int, int]:
    """The production assignment holding every fault of ``faults`` at once.

    Stored corruptions become flips and read faults misreads of one
    ``propagate`` call, so the cones compose as they would in one run; two
    faults at one word are refused.
    """

    flips: dict[int, int] = {}
    misreads: dict[int, dict[int, int]] = {}
    seen: set[int] = set()
    for fault in faults:
        if fault.address in seen:
            raise ValueError(f"two faults land on word {fault.address}")
        seen.add(fault.address)
        if fault.stored:
            flips[fault.address] = fault.correct ^ fault.corrupted
        else:
            for reader in fault.misreaders:
                misreads.setdefault(reader, {})[fault.address] = fault.corrupted
    return injector.propagate(flips, misreads)


def _top_bit(injector: FaultInjector, address: int) -> int:
    return 1 << (injector.compiled.circuit[address].width - 1)


def _interior_dots(injector: FaultInjector, replay_unit: int) -> list[tuple[int, int]]:
    """``(VU, output address)`` of the dot products of ``replay_unit`` whose word is no boundary position."""

    compiled = injector.compiled
    boundary = compiled.index.boundary()
    found = []
    for unit in dot_units(compiled, replay_unit):
        address = compiled.index.verification_unit(unit).interval[-1]
        if not boundary.contains(address):
            found.append((unit, address))
    return found


def interior_flip(injector: FaultInjector, replay_unit: int, *, bit: int) -> Production:
    """(a) The first interior dot product of ``replay_unit`` whose ``bit`` flips without changing a token."""

    for _unit, address in _interior_dots(injector, replay_unit):
        run = _production(
            injector,
            FaultClass.INTERIOR_FLIP,
            address,
            injector.honest[address] ^ (1 << bit),
        )
        if not run.changed_outputs:
            return run
    raise LookupError(f"every bit-{bit} flip in unit {replay_unit} changes a token")


def token_flip(injector: FaultInjector, replay_unit: int) -> Production:
    """(b) The first interior dot product of ``replay_unit`` whose top bit flips a streamed token."""

    for _unit, address in _interior_dots(injector, replay_unit):
        run = _production(
            injector,
            FaultClass.TOKEN_FLIP,
            address,
            injector.honest[address] ^ _top_bit(injector, address),
        )
        if run.changed_outputs:
            return run
    raise LookupError(f"no top-bit flip in unit {replay_unit} changes a token")


def catastrophic(injector: FaultInjector, replay_unit: int) -> Production:
    """(c) Every bit of the first interior dot product of ``replay_unit`` flipped."""

    _unit, address = _interior_dots(injector, replay_unit)[0]
    width = injector.compiled.circuit[address].width
    return _production(
        injector,
        FaultClass.CATASTROPHIC,
        address,
        injector.honest[address] ^ ((1 << width) - 1),
    )


def weight_read(injector: FaultInjector, address: int | None = None) -> Production:
    """(d) Every reader of weight cell ``address`` (default: the most-read cell) reads its top bit flipped."""

    circuit = injector.compiled.circuit
    if address is None:
        address = max(circuit.weights, key=lambda a: (len(injector.readers[a]), -a))
    if not circuit[address].is_weight:
        raise ValueError(f"address {address} is not a weight gate")
    return _production(
        injector,
        FaultClass.WEIGHT_READ,
        address,
        injector.honest[address] ^ _top_bit(injector, address),
        misreaders=injector.readers[address],
    )


def input_read(injector: FaultInjector, rank: int = 0) -> Production:
    """(e) Every reader of the input gate of ``rank`` reads its low bit flipped (another token id)."""

    address = injector.compiled.circuit.inputs[rank]
    return _production(
        injector,
        FaultClass.INPUT_READ,
        address,
        injector.honest[address] ^ 1,
        misreaders=injector.readers[address],
    )


def significant_bits(
    injector: FaultInjector, address: int, readers: Sequence[int]
) -> tuple[int, ...]:
    """The bits of the word at ``address`` whose misread by ``readers`` changes a VU output.

    A bit not listed is inert for those readers: flipping it in their copy
    changes no value any recording policy keeps, so no replay can see it
    and no declaration is needed for it.
    """

    compiled = injector.compiled
    recorded = recorded_addresses(compiled, RecordingPolicy.VU_OUTPUTS)
    honest = injector.honest
    significant = []
    for bit in range(compiled.circuit[address].width):
        corrupted = honest[address] ^ (1 << bit)
        values = injector.propagate({}, {r: {address: corrupted} for r in readers})
        if any(values[a] != honest[a] for a in recorded):
            significant.append(bit)
    return tuple(significant)


def boundary_at_rest(
    injector: FaultInjector, producer: int, consumer: int
) -> Production:
    """(f) The first output word of RU ``producer`` that RU ``consumer`` reads, read there with its
    most significant :func:`significant_bits` bit flipped (a top-bit flip may be inert)."""

    compiled = injector.compiled
    owner = compiled.index.replay_units.owner
    for address in compiled.circuit.Out(compiled.index.replay_units.unit(producer)):
        readers = tuple(r for r in injector.readers[address] if owner(r) == consumer)
        if not readers:
            continue
        significant = significant_bits(injector, address, readers)
        if not significant:
            continue
        return _production(
            injector,
            FaultClass.BOUNDARY_AT_REST,
            address,
            injector.honest[address] ^ (1 << significant[-1]),
            misreaders=readers,
        )
    raise LookupError(
        f"unit {consumer} reads no output of unit {producer} significantly"
    )


def vu_output_read(injector: FaultInjector, replay_unit: int) -> Production:
    """(g) The first interior dot product of ``replay_unit`` with readers: they read its top bit flipped."""

    for _unit, address in _interior_dots(injector, replay_unit):
        if injector.readers[address]:
            return _production(
                injector,
                FaultClass.VU_OUTPUT_READ,
                address,
                injector.honest[address] ^ _top_bit(injector, address),
                misreaders=injector.readers[address],
            )
    raise LookupError(f"no interior dot product of unit {replay_unit} is read")


# -- strategies -----------------------------------------------------------------------


class Strategy(Enum):
    """The prover's declaration strategy (``docs/honest-prover.md``, section 5)."""

    P0 = "P0"
    """No declarations; a sampled pinned VU is a rejection."""
    P1 = "P1"
    """Post-J: replay the opened RUs pinned, declare the pins (the protocol as built)."""
    P2 = "P2"
    """Pre-J at ``u(1)`` for the RUs a signal flagged before streaming, then P1."""
    P3 = "P3"
    """Pre-J at ``u(1)`` for every pin: the whole run replayed before the challenge."""


@dataclass(frozen=True, slots=True)
class Account:
    """What a strategy declared, re-executed and was charged for one run."""

    strategy: Strategy
    pre_j: int
    """Declarations priced before the q-challenge, at ``u(1)`` each (counterfactual for P2, P3)."""
    post_j: int
    """Declarations made after the q-challenge, at ``u_post(1)`` each (what the protocol saw)."""
    recompute: Fraction
    """The share of the production replay cost the prover re-executed."""
    charge_bits: float
    """``pre_j * u(1) + post_j * u_post(1)``."""

    @property
    def declarations(self) -> int:
        return self.pre_j + self.post_j


def account(
    strategy: Strategy,
    compiled: Compiled,
    pinned: Mapping[int, Sequence[int]],
    opened: Iterable[int],
    *,
    u1: float,
    u_post: float,
    flagged: Iterable[int] = (),
) -> Account:
    """Price one run under ``strategy``.

    ``pinned`` is :func:`pin_everything` (what each RU's pinned replay
    declares), ``opened`` the RUs the q-challenge opened, ``flagged`` the RUs
    a signal marked before streaming (P2 only); ``u1`` and ``u_post`` are the
    prices of one pre-J and one post-J declaration.  The recompute share
    counts the RUs the strategy replayed by their replay cost: the opened
    ones for P0 and P1 -- the protocol requires their replay whether or not
    anything is declared -- the opened and flagged ones for P2, all of them
    for P3.
    """

    circuit, index = compiled.circuit, compiled.index
    units = range(index.replay_units.count)
    opened_set, flagged_set = set(opened), set(flagged)
    pre: set[int]
    post: set[int]
    if strategy is Strategy.P0:
        pre, post, replayed = set(), set(), opened_set
    elif strategy is Strategy.P1:
        pre, post, replayed = set(), opened_set, opened_set
    elif strategy is Strategy.P2:
        pre, post = flagged_set, opened_set - flagged_set
        replayed = opened_set | flagged_set
    else:
        pre, post, replayed = set(units), set(), set(units)
    cost = {
        unit: circuit.Cost(index.replay_units.unit(unit), "replay") for unit in units
    }
    total = sum(cost.values())
    pre_j = sum(len(pinned[unit]) for unit in pre)
    post_j = sum(len(pinned[unit]) for unit in post)
    return Account(
        strategy=strategy,
        pre_j=pre_j,
        post_j=post_j,
        recompute=Fraction(sum(cost[unit] for unit in replayed), total)
        if total
        else Fraction(0),
        charge_bits=pre_j * u1 + post_j * u_post,
    )


# -- the phase diagram ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PhasePoint:
    """One fault density under one policy: what P0 risks and what P1 and P3 pay."""

    faults_per_round: float
    """``D``: the Poisson mean of pinned VUs per round, had every RU been opened."""
    q: float
    s: float
    capacity_bits: float
    """``U_0 = rho lambda + log2 e``."""
    u1: float
    """``u(1)``: one pre-J declaration."""
    u_post: float
    """``u_post(1) = rho log2 (1 / (1 - s))``: one post-J declaration."""
    declarations_mean: float
    """``q D``: the post-J declarations P1 makes, a Poisson mean."""
    f_max: int
    """The round budget P1 needs: ``fault_budget(q D, tail)``."""
    exceeded: float
    """``P[Poisson(q D) > f_max]``: the round P1 loses to its budget."""
    p0_rejected: float
    """``1 - exp(-q s D)``: some pinned VU opened and sampled, undeclared."""
    p3_f_max: int
    """The budget P3 needs before J: ``fault_budget(D, tail)``, every fault opened or not."""

    @property
    def p1_charge_bits(self) -> float:
        return self.f_max * self.u_post

    @property
    def p3_charge_bits(self) -> float:
        return self.p3_f_max * self.u1

    @property
    def p1_share(self) -> float:
        return self.p1_charge_bits / self.capacity_bits

    @property
    def p3_share(self) -> float:
        return self.p3_charge_bits / self.capacity_bits

    def p0_beats_p1(self, rounds_lost: int = 1) -> bool:
        """Whether P0's expected loss is below P1's charge, both as shares of a round's ``U_0``.

        A rejection forfeits ``rounds_lost`` rounds' capacity: one when the
        round alone is lost, the epoch's count under the epoch layer as built
        (one rejected run rejects the epoch).  For small ``s`` and ``q s D``
        the comparison is ``q D rounds_lost < log2(e) / lam``: ``s`` cancels.
        """

        return self.p0_rejected * rounds_lost < self.p1_share


def phase_diagram(
    rho: float,
    policy: VerificationPolicy,
    u1: float,
    densities: Iterable[float],
    *,
    lam: float = 40.0,
    tail: float = 1e-6,
) -> list[PhasePoint]:
    """P0, P1 and P3 at each fault density under ``policy`` for a table of slope ``rho``.

    ``U_0 = rho lam + log2 e`` is the closed form of :mod:`veritor.analysis.rate`
    (``rho`` from ``rate`` or from the fold's ``BoundResult.rho``); a post-J
    declaration lowers the threshold by ``log2 (1 / (1 - s))`` bits and so
    costs ``rho`` times that.  Pinned VUs per round are Poisson with mean
    ``D``; each lies in an opened RU with probability ``q``, so P1 declares
    ``Poisson(q D)`` and needs the budget :func:`~veritor.simulation.faults.fault_budget`
    gives at ``tail``, charged whether or not it is used; P0 is rejected
    when one is sampled as well; P3 pardons every fault before J at ``u1``
    and re-executes the whole round.
    """

    q, s = float(policy.q), float(policy.s)
    if not 0 <= s < 1 or rho < 0 or lam <= 0:
        raise ValueError("phase_diagram needs rho >= 0, 0 <= s < 1 and lam > 0")
    u_post = rho * math.log2(1 / (1 - s))
    points = []
    for density in densities:
        if density < 0:
            raise ValueError("fault densities are nonnegative")
        mean = q * density
        f_max = fault_budget(mean, tail)
        points.append(
            PhasePoint(
                faults_per_round=density,
                q=q,
                s=s,
                capacity_bits=rho * lam + LOG2E,
                u1=u1,
                u_post=u_post,
                declarations_mean=mean,
                f_max=f_max,
                exceeded=poisson_tail(mean, f_max),
                p0_rejected=-math.expm1(-s * mean),
                p3_f_max=fault_budget(density, tail),
            )
        )
    return points
