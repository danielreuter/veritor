"""The honest prover: what it records in production, how it replays, what it declares.

The protocol's :class:`~veritor.protocol.ProverSession` was written as if the
server kept every gate of the production run (``assignment_replay``) and found
its faults by :func:`~veritor.protocol.self_check` over that assignment.  A
real server keeps the boundary it has to commit plus whatever it chooses to
log, and *reconstructs* an opened RU's interior when the q-challenge asks for
it.  When a recomputed value disagrees with a value it recorded -- and, for a
boundary value, already committed -- the commitment is binding: the server
keeps the recorded value and declares the VU that produced it.  This module
models that server.

**Recording policies.**  :class:`RecordingPolicy` names what the server
keeps of a run (:func:`record` restricts the omniscient assignment to it):

* ``BOUNDARY``: the inputs, the weights and every RU's declared outputs
  ``Out(R)`` -- what the boundary commitment and the weight tree need and
  nothing more.  For ``RequestsG`` (RU = request) the recorded values of a
  run are its streamed tokens; for ``ClusterG`` (RU = step) the KV values and
  tokens that cross steps.
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
recomputed value is stored and read downstream.  The interior this yields is
recorded where the server recorded, recomputed elsewhere, and satisfies every
VU's relation except at the pinned VUs, whose relations fail against their
own (pinned or recomputed) inputs: a reader of a pinned value was recomputed
from that value, so its relation holds against it.  The pinned VUs are
therefore exactly what the server must declare (M6) for the run to be
accepted whatever the s-challenge samples, and ``len(pinned)`` is the
declaration count of the RU; :func:`~veritor.protocol.self_check` over the
committed interior finds the same set (``tests/veritor/stress/test_honest_replay.py``
asserts it).  A fault that changed no recorded value costs no declaration:
the replay recomputes the correct interior, and it agrees with the recorded
boundary.  A fault that changed a recorded value costs one declaration per
recorded value it changed, whether or not the VU that produced it is the one
that faulted: with tokens-only recording the declarations name the tokens
that came out wrong, not the kernel that went wrong.

**Fault classes.**  :class:`FaultClass` and the injection helpers produce
the production run of each class in the simulation's omniscient view
(:class:`Production`: the full assignment, the streamed outputs, where the
fault landed).  Stored corruptions (an output word flipped after it was
computed) are ``FaultInjector.propagate`` flips; read faults (a gate reads a
cell as another value while the stored value, and the value the boundary or
the weight root commits, stays right) are its ``misreads``.  The pinned gate
set of the toy decoder is integer arithmetic without a NaN or infinity, so
the catastrophic class corrupts every bit of an early word: the garbage is
a finite value that propagates like any other wrong value, and nothing
downstream can tell it apart from a legitimate one.

**Strategies.**  :class:`Strategy` and :func:`account` price a run under the
four prover strategies of ``docs/honest-prover.md``: ``P0`` declares nothing
and takes the rejections; ``P1`` records the boundary, replays the opened RUs
pinned and declares what it pins after ``J`` (the protocol as built); ``P2``
declares before ``J`` at ``u(1)`` the pins of the RUs a hardware or value
signal flagged before streaming, and the rest as ``P1``; ``P3`` replays every
RU before ``J`` and declares every pin at ``u(1)``.  The protocol has no
pre-``J`` declaration message, so ``P2`` and ``P3`` are counterfactual
accounting -- their charge is what a protocol with that message would bill,
their verdict is the built protocol's under ``P1`` declarations.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction

from veritor.analysis.bound import BoundOptions, bound
from veritor.analysis.faults import declared_bits, unit_fault_bits
from veritor.core import Compiled, KindTable, VerificationPolicy, as_kind_table
from veritor.core.indexed import iter_members
from veritor.protocol.session import Values

from .faults import FaultInjector, dot_units, fault_budget

__all__ = [
    "Account",
    "Capacity",
    "FaultClass",
    "HonestReplay",
    "PhasePoint",
    "Production",
    "RecordingPolicy",
    "Strategy",
    "account",
    "boundary_at_rest",
    "catastrophic",
    "fold_capacity",
    "honest_replay",
    "input_read",
    "interior_flip",
    "phase_boundary",
    "phase_diagram",
    "pin_everything",
    "post_j_charge_bits",
    "rate_capacity",
    "record",
    "recorded_addresses",
    "replay_pinned",
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


def boundary_at_rest(
    injector: FaultInjector, producer: int, consumer: int
) -> Production:
    """(f) The first output word of RU ``producer`` that RU ``consumer`` reads, read there with its top bit flipped."""

    compiled = injector.compiled
    owner = compiled.index.replay_units.owner
    for address in compiled.circuit.Out(compiled.index.replay_units.unit(producer)):
        readers = tuple(r for r in injector.readers[address] if owner(r) == consumer)
        if readers:
            return _production(
                injector,
                FaultClass.BOUNDARY_AT_REST,
                address,
                injector.honest[address] ^ _top_bit(injector, address),
                misreaders=readers,
            )
    raise LookupError(f"unit {consumer} reads no output of unit {producer}")


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
    """No declarations; a sampled faulty VU is a rejection."""
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
    """Declarations made after the q-challenge, priced by ``declared_bits`` (what the protocol saw)."""
    recompute: Fraction
    """The share of the production replay cost the prover re-executed."""
    charge_bits: float
    """``pre_j * u(1)`` plus the post-J price, uncapped by the interface."""


def post_j_charge_bits(
    compiled: Compiled,
    policy: VerificationPolicy,
    eta: Fraction,
    declarations: int,
    options: BoundOptions | None = None,
) -> float:
    """What ``declarations`` post-J declarations add to ``U`` at ``policy``, before the interface cap."""

    if declarations == 0:
        return 0.0
    table = as_kind_table(compiled)
    options = BoundOptions() if options is None else options
    base = bound(table, policy, eta, options)
    uncapped = min(base.knapsack_bits, base.laplace_bits)
    return declared_bits(table, policy, eta, options, declarations, uncapped) - uncapped


def account(
    strategy: Strategy,
    compiled: Compiled,
    policy: VerificationPolicy,
    eta: Fraction,
    pinned: Mapping[int, Sequence[int]],
    opened: Iterable[int],
    *,
    flagged: Iterable[int] = (),
) -> Account:
    """Price one run under ``strategy``.

    ``pinned`` is :func:`pin_everything` (what each RU's pinned replay
    declares), ``opened`` the RUs the q-challenge opened, ``flagged`` the RUs
    a signal marked before streaming (P2 only).  The recompute share counts
    the RUs the strategy replayed: the opened ones for P0 and P1 -- the
    protocol requires their replay whether or not anything is declared --
    the opened and flagged ones for P2, all of them for P3.
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
        charge_bits=pre_j * unit_fault_bits(compiled)
        + post_j_charge_bits(compiled, policy, eta, post_j),
    )


# -- the phase diagram ------------------------------------------------------------------

Capacity = Callable[[int], tuple[float, float]]
"""``f -> (U_0, charge)``: the capacity without declarations and what ``f`` post-J
declarations add to it, in bits, both uncapped by the interface."""


def fold_capacity(
    target: Compiled | KindTable,
    policy: VerificationPolicy,
    eta: Fraction,
    options: BoundOptions | None = None,
) -> Capacity:
    """The fold's price of post-J declarations (``bound`` and ``declared_bits``)."""

    table = as_kind_table(target)
    options = BoundOptions() if options is None else options
    base = bound(table, policy, eta, options)
    uncapped = min(base.knapsack_bits, base.laplace_bits)

    def capacity(declarations: int) -> tuple[float, float]:
        if declarations == 0:
            return uncapped, 0.0
        priced = declared_bits(table, policy, eta, options, declarations, uncapped)
        return uncapped, priced - uncapped

    return capacity


def rate_capacity(rho: float, s: float, lam: float = 40.0) -> Capacity:
    """The headline's closed form: ``U = rho lam + log2 e`` (:mod:`veritor.analysis.rate`).

    ``f`` post-J declarations lower the threshold by ``f log2 (1 / (1 - s))``
    bits (the first bound of :mod:`veritor.analysis.faults`, the smaller of
    the two whenever ``s < 1 - 1 / (1 + n)``), so they cost
    ``rho f log2 (1 / (1 - s))``: a fixed share ``f log2 (1 / (1 - s)) / lam``
    of ``U`` whatever the model.
    """

    if not 0 <= s < 1 or rho < 0 or lam <= 0:
        raise ValueError("rate_capacity needs rho >= 0, 0 <= s < 1 and lam > 0")
    per_declaration = rho * math.log2(1 / (1 - s))
    return lambda declarations: (rho * lam + LOG2E, declarations * per_declaration)


@dataclass(frozen=True, slots=True)
class PhasePoint:
    """One fault density under one policy: what P1 and P3 need and pay."""

    faults_per_round: float
    q: float
    s: float
    declarations_mean: float
    """``q * faults``: the expected post-J declarations of P1."""
    f_max: int
    """The budget P1's header needs: ``fault_budget(declarations_mean, tail)``."""
    capacity_bits: float
    """``U_0``, uncapped."""
    p1_charge_bits: float
    """What ``f_max`` post-J declarations add to ``U_0``."""
    u1: float
    p3_f_max: int
    """The budget P3 needs: every fault, opened or not, ``fault_budget(faults, tail)``."""
    p3_charge_bits: float
    """``p3_f_max * u(1)``: pre-J declarations, with recompute 1."""

    @property
    def p1_share(self) -> float:
        return (
            self.p1_charge_bits / self.capacity_bits if self.capacity_bits else math.inf
        )


def phase_diagram(
    capacity: Capacity,
    policy: VerificationPolicy,
    densities: Iterable[float],
    *,
    u1: float,
    tail: float = 1e-6,
) -> list[PhasePoint]:
    """P1 and P3 at each fault density (expected faults per round) under ``policy``.

    Faults per round are Poisson; each lands in an opened RU with probability
    ``q``, so P1's declarations are Poisson with mean ``q * faults`` and its
    header needs the budget :func:`~veritor.simulation.faults.fault_budget`
    gives for that mean at ``tail``; the charge is for the budget, not for
    the declarations made.  P3 pardons every fault before J at ``u(1)``.
    """

    q, s = float(policy.q), float(policy.s)
    points = []
    for density in densities:
        if density < 0:
            raise ValueError("fault densities are nonnegative")
        f_max = fault_budget(q * density, tail)
        p3_f_max = fault_budget(density, tail)
        u0, charge = capacity(f_max)
        points.append(
            PhasePoint(
                faults_per_round=density,
                q=q,
                s=s,
                declarations_mean=q * density,
                f_max=f_max,
                capacity_bits=u0,
                p1_charge_bits=charge,
                u1=u1,
                p3_f_max=p3_f_max,
                p3_charge_bits=p3_f_max * u1,
            )
        )
    return points


def phase_boundary(
    capacity: Capacity,
    policy: VerificationPolicy,
    *,
    share: float = 0.01,
    tail: float = 1e-6,
    max_declarations: int = 100_000,
) -> float:
    """The least fault density (faults per round) at which P1's charge reaches ``share`` of ``U_0``.

    ``0.0`` when the one declaration every admitting header carries already
    costs that much; ``inf`` when ``max_declarations`` do not.
    """

    u0, _ = capacity(0)
    declarations = 1
    while capacity(declarations)[1] < share * u0:
        declarations += 1
        if declarations > max_declarations:
            return math.inf
    if declarations <= 1:
        return 0.0
    q = float(policy.q)
    if q == 0:
        return math.inf
    low, high = 0.0, 1.0
    while fault_budget(q * high, tail) < declarations:
        high *= 2
    for _ in range(200):
        middle = (low + high) / 2
        if fault_budget(q * middle, tail) >= declarations:
            high = middle
        else:
            low = middle
        if high - low <= 1e-9 * high:
            break
    return high
