"""Silent data corruption in a run, and what an honest server does about it (M6).

A hardware fault flips one bit of one dot product's output word somewhere in
a request.  The server does not notice: everything downstream is computed
correctly *from* the corrupted word and the resulting tokens are streamed to
the user -- that is what happened, so that is the run's ``y*`` and the
assignment the server holds.  When the verifier's q-challenge opens the RU
holding the fault, the server replays it and finds exactly one VU whose
committed value disobeys its relation (:func:`veritor.protocol.self_check`),
and declares it (``InteriorMessage.declarations``).  Without a declaration
that VU is a rejection whenever it is sampled; with one, its openings are
authenticated and its relation check is skipped, and the VUs that read the
corrupted word are checked against it as usual.

Rates, from ``docs/notes/datacenter-realities.md`` section 7: Llama-3 405B
saw 6 silent-data-corruption events in 54 days of training on 16,384 GPUs,
about ``2.8e-7`` per device-hour (:data:`SDC_RATE_PER_DEVICE_HOUR`);
Gemini reports one every week or two on a much larger fleet; a fleet-wide
prevalence of about one faulty device per thousand.  Over a verification
window of ``devices * hours`` device-hours the number of faults is Poisson
with mean ``rate * device_hours`` (:func:`expected_faults`), and the ``f_max``
a verifier admits is the smallest count whose tail probability is below a
target (:func:`fault_budget`): exceeding it means one rejected window for
an honest server, a retry, not a soundness event.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass

from veritor.core import Compiled, IndexNode

LLAMA3_SDC_EVENTS = 6
LLAMA3_DAYS = 54
LLAMA3_GPUS = 16_384
SDC_RATE_PER_DEVICE_HOUR = LLAMA3_SDC_EVENTS / (LLAMA3_DAYS * 24 * LLAMA3_GPUS)
"""Llama-3 405B: 6 SDC events over 54 days on 16,384 GPUs, about 2.8e-7 per device-hour."""

DOT_OPS = frozenset({"mul", "add"})
"""The ops of a dot product VU: a ``repeat`` of products then a tree of sums."""


def expected_faults(device_hours: float, rate: float = SDC_RATE_PER_DEVICE_HOUR) -> float:
    """The Poisson mean of silent faults in a window of ``device_hours``."""

    if device_hours < 0 or rate < 0:
        raise ValueError("device-hours and rate must be nonnegative")
    return device_hours * rate


def fault_budget(mean: float, tail: float = 1e-6, *, at_least: int = 1) -> int:
    """The smallest ``f_max >= at_least`` with ``P[Poisson(mean) > f_max] <= tail``.

    ``at_least = 1`` by default: a verifier that admits declarations at all
    admits one, since a single fault in a window is what the mechanism is
    for; ``at_least = 0`` gives the pure quantile.
    """

    if not 0 < tail < 1:
        raise ValueError("tail must lie in (0, 1)")
    if mean < 0 or type(at_least) is not int or at_least < 0:
        raise ValueError("mean must be nonnegative and at_least a nonnegative integer")
    if mean == 0:
        return at_least
    f = at_least
    cumulative = sum(math.exp(-mean) * mean**k / math.factorial(k) for k in range(f + 1))
    while 1 - cumulative > tail:
        f += 1
        cumulative += math.exp(-mean) * mean**f / math.factorial(f)
    return f


def is_dot_unit(compiled: Compiled, node: IndexNode) -> bool:
    """Whether the VU ``node`` is a dot product: at least three gates, all ``mul``/``add``."""

    interval = node.interval
    if len(interval) < 3:
        return False
    circuit = compiled.circuit
    return all(circuit[address].op in DOT_OPS for address in interval)


def dot_units(compiled: Compiled, replay_unit: int) -> Iterator[int]:
    """The global indices of the dot product VUs inside ``replay_unit``, in order."""

    block = compiled.index.verification_units(replay_unit)
    for offset in range(block.count):
        if is_dot_unit(compiled, block.unit(offset)):
            yield block.first + offset


def evaluate_with_flips(
    compiled: Compiled,
    inputs: Iterable[int],
    weights: Iterable[int],
    flips: Mapping[int, int],
) -> dict[int, int]:
    """Every value of the circuit with the output word at each address of ``flips`` XORed
    by its mask after it is computed, and everything downstream computed from the result.

    Only the gates at the flipped addresses violate their relation.
    """

    circuit = compiled.circuit
    given = {"input": iter(tuple(inputs)), "weight": iter(tuple(weights))}
    values: dict[int, int] = {}
    for address in range(circuit.n):
        ref = circuit[address]
        if ref.is_source:
            value = next(given[ref.source])  # type: ignore[index]
        else:
            value = circuit.evaluate_gate(address, tuple(values[a] for a in ref.args))
        mask = flips.get(address)
        if mask is not None:
            if mask & ~((1 << ref.width) - 1) or not mask:
                raise ValueError(f"flip mask {mask:#x} is not a nonzero {ref.width}-bit mask")
            value ^= mask
        values[address] = value
    return values


@dataclass(frozen=True, slots=True)
class Fault:
    """One silent bit flip and the run it produced."""

    verification_unit: int
    """The VU whose output word was corrupted (global index)."""
    replay_unit: int
    """The RU holding it."""
    address: int
    """The corrupted gate: the VU's output word."""
    bit: int
    """Which bit of that word flipped."""
    honest: int
    """The word's correct value."""
    faulty: int
    """The word as the server computed (and streamed the consequences of) it."""
    values: Mapping[int, int]
    """The server's assignment: correct everywhere except at ``address``, downstream propagated."""
    outputs: tuple[int, ...]
    """The tokens the users received (``y*``)."""
    honest_outputs: tuple[int, ...]
    """What a fault-free run would have streamed."""

    @property
    def changed_outputs(self) -> int:
        """How many streamed tokens the flip changed."""

        return sum(a != b for a, b in zip(self.outputs, self.honest_outputs, strict=True))


def inject_fault(
    compiled: Compiled,
    inputs: Iterable[int],
    weights: Iterable[int],
    verification_unit: int,
    bit: int = 0,
) -> Fault:
    """A run in which bit ``bit`` of VU ``verification_unit``'s output word flipped.

    The output word is the VU's last gate (a dot product's final sum).  The
    honest assignment is computed alongside so the fault's effect on the
    streamed tokens is known.
    """

    circuit, index = compiled.circuit, compiled.index
    node = index.verification_unit(verification_unit)
    address = node.interval[-1]
    ref = circuit[address]
    if ref.is_source:
        raise ValueError(f"VU {verification_unit} ends in a source gate; nothing to corrupt")
    if type(bit) is not int or not 0 <= bit < ref.width:
        raise ValueError(f"bit must lie in [0, {ref.width})")
    replay_unit = node.replay_unit
    if replay_unit is None:
        raise ValueError(f"VU {verification_unit} lies in no replay unit")
    inputs, weights = tuple(inputs), tuple(weights)
    honest = dict(enumerate(circuit.evaluate(inputs, weights)))
    values = evaluate_with_flips(compiled, inputs, weights, {address: 1 << bit})
    return Fault(
        verification_unit=verification_unit,
        replay_unit=replay_unit,
        address=address,
        bit=bit,
        honest=honest[address],
        faulty=values[address],
        values=values,
        outputs=tuple(values[a] for a in circuit.outputs),
        honest_outputs=tuple(honest[a] for a in circuit.outputs),
    )
