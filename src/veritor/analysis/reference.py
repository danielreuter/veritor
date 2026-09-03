"""Exhaustive references for tiny circuits: what the fold is tested against.

Everything here enumerates explicitly -- every error set, every transcript
-- and is exponential in the circuit size.  It exists to pin down the
statement :func:`veritor.analysis.bound.bound` certifies, not to bound
anything real.

Source gates (the circuit's ``in`` and ``weight`` gates) are never
incorrect in an accepted transcript: the inputs are checked against the
public inputs at commit and the weights are what κ_W binds.  So a transcript
enumerates the other gates only, and a source gate is never a source of a
cut.  An error set may still name a verification unit holding nothing but
source gates; its cover has no capacity.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Iterable, Iterator, Sequence
from fractions import Fraction

import networkx as nx

from veritor.core.circuit import Circuit
from veritor.core.compiled import Compiled
from veritor.core.description import VERIFICATION, Frame
from veritor.core.index import Index, IndexNode
from veritor.core.policy import VerificationPolicy

from .probability import survival

type ErrorSet = frozenset[int]
"""Global indices of the verification units holding an incorrect gate."""


def error_sets(index: Index) -> Iterator[ErrorSet]:
    """Every subset of the verification units, the empty set first."""

    units = range(index.verification_unit_count)
    for size in range(len(units) + 1):
        for subset in itertools.combinations(units, size):
            yield frozenset(subset)


def replay_unit_of(index: Index, unit: int) -> int:
    """The replay unit containing verification unit ``unit`` (every VU lies inside one)."""

    replay_unit = index.verification_unit(unit).replay_unit
    assert replay_unit is not None
    return replay_unit


def error_counts(index: Index, errors: ErrorSet) -> list[int]:
    """``l_r = |E ∩ R_r|`` for every replay unit."""

    counts = [0] * index.replay_units.count
    for unit in errors:
        counts[replay_unit_of(index, unit)] += 1
    return counts


def unit_owner(index: Index) -> dict[int, int]:
    """Gate address -> global index of the verification unit holding it."""

    owner: dict[int, int] = {}
    for unit in range(index.verification_unit_count):
        for address in index.verification_unit(unit).interval:
            owner[address] = unit
    return owner


def check_addresses(compiled: Compiled) -> frozenset[int]:
    """The addresses of the check outputs, which the verifier fixes at their constants."""

    outputs = compiled.circuit.outputs
    return frozenset(outputs[ordinal] for ordinal, _ in compiled.check_values())


def out_bits(
    circuit: Circuit, node: IndexNode, checked: frozenset[int] = frozenset()
) -> int:
    """The width of ``Out(node)`` in bits; at the root, less its check outputs."""

    skipped = checked if node.frame.parent is None else frozenset()
    return sum(
        circuit[address].width
        for address in circuit.Out(node)
        if address not in skipped
    )


def reach_bits(
    circuit: Circuit, node: IndexNode, checked: frozenset[int] = frozenset()
) -> int:
    """The width of the circuit outputs reachable from the node's gates, in bits.

    Forward along argument reads from every gate of the node but its source
    gates (which hold their pinned values): the exact value of what
    :attr:`~veritor.core.KindSummary.reach_bits` bounds over the copies of
    a kind.  Those outputs are a downstream cut for the node, like its
    interface.  Check outputs (``checked``) are fixed by the verifier and
    count for nothing.
    """

    reached = bytearray(circuit.n)
    for address in node.interval:
        reached[address] = not circuit[address].is_source
    for address in range(node.interval.stop, circuit.n):
        reached[address] = any(reached[arg] for arg in circuit[address].args)
    return sum(
        circuit[address].width
        for address in circuit.outputs
        if reached[address] and address not in checked
    )


def ancestor_bits(
    circuit: Circuit, node: IndexNode, checked: frozenset[int] = frozenset()
) -> int:
    """The narrowest declared interface among the node's proper ancestors, in bits.

    The exact value of what :attr:`~veritor.core.KindSummary.ancestor_bits`
    bounds over the copies of a kind; the root, having no ancestor, is
    given its own interface, the whole output.  Every value inside the
    node leaves each enclosing copy through that copy's declared outputs,
    so the interface of every ancestor is a downstream cut for the node.
    """

    frame = node.frame
    if frame.parent is None:
        return out_bits(circuit, node, checked)
    narrowest = math.inf
    parent: Frame | None = frame.parent
    while parent is not None:
        narrowest = min(narrowest, out_bits(circuit, IndexNode(parent), checked))
        parent = parent.parent
    return int(narrowest)


def cover_bits(compiled: Compiled, errors: ErrorSet) -> int:
    """``kappa(E)``: the cheapest cover of ``E`` by index nodes, in bits.

    A node is covered either by itself -- charged the narrowest of its
    interface, the circuit outputs it reaches and the interfaces of the
    nodes enclosing it, all downstream cuts -- or by covering the children
    that contain errors; a verification unit (VU) is covered by itself.
    """

    owner = unit_owner(compiled.index)
    circuit = compiled.circuit
    checked = check_addresses(compiled)

    def charge(node: IndexNode, enclosing: int) -> int:
        return min(
            out_bits(circuit, node, checked),
            reach_bits(circuit, node, checked),
            enclosing,
        )

    def value(node: IndexNode, enclosing: int) -> int:
        """The cover of the errors under ``node``; ``enclosing`` is the narrowest interface above it."""

        own = charge(node, enclosing)
        if node.role == VERIFICATION:
            return own if owner[node.interval.start] in errors else 0
        inside = min(enclosing, out_bits(circuit, node, checked))
        below = sum(value(child, inside) for child in node.children())
        return min(below, own) if below else 0

    root = compiled.index.root
    return value(root, out_bits(circuit, root, checked))


def cut_bits(compiled: Compiled, errors: ErrorSet) -> int:
    """The exact minimum downstream cut of the gates of ``E``, in bits.

    A minimum vertex cut of the circuit between the erroneous gates and the
    outputs, each gate weighted by its width: split every gate into an entry
    and an exit vertex joined by an edge of capacity ``width``, give the
    wires unbounded capacity, and take the maximum flow from the erroneous
    gates' entries to the outputs' exits.  The erroneous gates may be cut
    themselves (their own width bounds their influence), and so may the
    outputs.  This is the tightest capacity the downstream-cut theorem
    allows and is never above :func:`cover_bits`.
    """

    circuit = compiled.circuit
    checked = check_addresses(compiled)
    # a check output is fixed by the verifier, so it is not a sink of the cut
    sinks = [address for address in circuit.outputs if address not in checked]
    sources = {
        address
        for unit in errors
        for address in compiled.index.verification_unit(unit).interval
        if not circuit[address].is_source  # a source gate holds its pinned value
    }
    if not sources or not sinks:
        return 0
    graph: nx.DiGraph[object] = nx.DiGraph()
    for address in range(circuit.n):
        gate = circuit[address]
        graph.add_edge(("entry", address), ("exit", address), capacity=gate.width)
        for arg in gate.args:
            graph.add_edge(("exit", arg), ("entry", address))  # no capacity: unbounded
    for address in sources:
        graph.add_edge("source", ("entry", address))
    for address in sinks:
        graph.add_edge(("exit", address), "sink")
    return int(nx.maximum_flow_value(graph, "source", "sink"))


def admissible_sets(
    compiled: Compiled, policy: VerificationPolicy, eta: Fraction
) -> list[ErrorSet]:
    """Every ``E`` with ``sigma(E) > eta``."""

    index = compiled.index
    return [
        errors
        for errors in error_sets(index)
        if survival(policy, error_counts(index, errors)) > eta
    ]


def subset_sum_bits(
    compiled: Compiled,
    policy: VerificationPolicy,
    eta: Fraction,
    kappa: Callable[[Compiled, ErrorSet], int] = cover_bits,
) -> float:
    """``log2 sum_{E admissible} 2**kappa(E)``, exactly."""

    total = sum(
        1 << kappa(compiled, errors)
        for errors in admissible_sets(compiled, policy, eta)
    )
    return math.log2(total)


type Output = tuple[int, ...]
type ErrorCounts = tuple[int, ...]


def transcript_outputs(
    compiled: Compiled, inputs: Sequence[int], weights: Sequence[int] = ()
) -> dict[Output, set[ErrorCounts]]:
    """Every output some transcript produces, with the ``(l_r)_r`` of those transcripts.

    Enumerates every assignment of every non-source gate (``2**(width *
    gates)`` transcripts) over the pinned ``inputs`` and ``weights`` (by
    rank) and derives each one's error set from the gates whose value
    disagrees with their recorded arguments.  Policy-independent, so one
    enumeration serves every ``theta``.
    """

    circuit = compiled.circuit
    index = compiled.index
    owner = unit_owner(index)
    replay_of = [replay_unit_of(index, u) for u in range(index.verification_unit_count)]
    pinned = dict(zip(circuit.inputs, inputs, strict=True))
    pinned.update(zip(circuit.weights, weights, strict=True))
    gates = [address for address in range(circuit.n) if address not in pinned]
    refs = [circuit[address] for address in gates]
    outputs: dict[Output, set[ErrorCounts]] = {}
    for values in itertools.product(*(range(1 << ref.width) for ref in refs)):
        cells = dict(pinned)
        cells.update(zip(gates, values, strict=True))
        counts = [0] * index.replay_units.count
        seen: set[int] = set()
        for address, ref in zip(gates, refs, strict=True):
            expected = circuit.evaluate_gate(address, tuple(cells[a] for a in ref.args))
            if cells[address] != expected:
                unit = owner[address]
                if unit not in seen:
                    seen.add(unit)
                    counts[replay_of[unit]] += 1
        output = tuple(cells[address] for address in circuit.outputs)
        outputs.setdefault(output, set()).add(tuple(counts))
    return outputs


def accepted_outputs(
    outputs: dict[Output, set[ErrorCounts]],
    policy: VerificationPolicy,
    eta: Fraction,
    checks: Iterable[tuple[int, int]] = (),
) -> set[Output]:
    """``Y_eta``: the outputs whose best transcript survives with probability above ``eta``.

    ``checks`` are ``(output ordinal, constant)`` pairs
    (:meth:`~veritor.core.compiled.Compiled.check_values`): the verifier
    rejects a transcript whose output differs from the constant at a check
    position before any sampling, so such outputs are never in ``Y_eta``.
    """

    required = tuple(checks)
    chances: dict[ErrorCounts, Fraction] = {}
    accepted: set[Output] = set()
    for output, variants in outputs.items():
        if any(output[ordinal] != value for ordinal, value in required):
            continue
        for counts in variants:
            chance = chances.get(counts)
            if chance is None:
                chance = chances[counts] = survival(policy, counts)
            if chance > eta:
                accepted.add(output)
                break
    return accepted


__all__ = [
    "ErrorCounts",
    "ErrorSet",
    "Output",
    "accepted_outputs",
    "admissible_sets",
    "ancestor_bits",
    "check_addresses",
    "cover_bits",
    "cut_bits",
    "error_counts",
    "error_sets",
    "subset_sum_bits",
    "transcript_outputs",
    "unit_owner",
]
