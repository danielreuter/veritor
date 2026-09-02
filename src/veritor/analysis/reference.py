"""Exhaustive references for tiny circuits: what the fold is tested against.

Everything here enumerates explicitly -- every error set, every transcript
-- and is exponential in the circuit size.  It exists to pin down the
statement :func:`veritor.analysis.bound.bound` certifies, not to bound
anything real.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Iterator, Sequence
from fractions import Fraction

from circuit_cut_analysis.capacity import GateCapacity
from circuit_cut_analysis.circuit import CircuitDAG, Gate
from circuit_cut_analysis.mincut import minimum_vertex_cut
from veritor.core.circuit import Circuit
from veritor.core.compiled import Compiled
from veritor.core.description import VERIFICATION
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


def error_counts(index: Index, errors: ErrorSet) -> list[int]:
    """``l_r = |E ∩ R_r|`` for every replay unit."""

    counts = [0] * index.replay_units.count
    for unit in errors:
        counts[index.verification_unit(unit).replay_unit] += 1
    return counts


def unit_owner(index: Index) -> dict[int, int]:
    """Gate address -> global index of the verification unit holding it."""

    owner: dict[int, int] = {}
    for unit in range(index.verification_unit_count):
        for address in index.verification_unit(unit).interval:
            owner[address] = unit
    return owner


def out_bits(circuit: Circuit, node: IndexNode) -> int:
    return sum(circuit[address].width for address in circuit.Out(node))


def cover_bits(compiled: Compiled, errors: ErrorSet) -> int:
    """``kappa(E)``: the cheapest cover of ``E`` by index nodes, in bits.

    A node is covered either by its own interface or by covering the
    children that contain errors; a verification unit is covered by itself.
    """

    owner = unit_owner(compiled.index)
    circuit = compiled.circuit

    def value(node: IndexNode) -> int:
        if node.role == VERIFICATION:
            return out_bits(circuit, node) if owner[node.interval.start] in errors else 0
        below = sum(value(child) for child in node.children())
        return min(below, out_bits(circuit, node)) if below else 0

    return value(compiled.index.root)


def cut_bits(compiled: Compiled, errors: ErrorSet) -> int:
    """The exact minimum downstream cut of the gates of ``E``, in bits.

    This is the tightest capacity the downstream-cut theorem allows and is
    never above :func:`cover_bits`.
    """

    circuit = compiled.circuit
    gates = [
        Gate(str(address), GateCapacity.values(1 << circuit[address].width), op=circuit[address].op)
        for address in range(circuit.n)
    ]
    edges = {
        (str(arg), str(address))
        for address in range(circuit.n)
        for arg in circuit[address].args
    }
    dag = CircuitDAG(gates, edges, {str(address) for address in circuit.outputs})
    sources = {
        str(address)
        for unit in errors
        for address in compiled.index.verification_unit(unit).interval
    }
    if not sources:
        return 0
    capacity = minimum_vertex_cut(dag, sources, dag.outputs).exact_capacity
    if capacity is None:  # pragma: no cover - the all-gate cut policy is exact
        raise AssertionError("the explicit min-cut is exact")
    multiplier = capacity.multiplier
    if multiplier.denominator != 1 or multiplier.numerator & (multiplier.numerator - 1):
        raise ValueError("gate widths must be whole bits")
    return multiplier.numerator.bit_length() - 1


def admissible_sets(compiled: Compiled, policy: VerificationPolicy, eta: Fraction) -> list[ErrorSet]:
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

    total = sum(1 << kappa(compiled, errors) for errors in admissible_sets(compiled, policy, eta))
    return math.log2(total)


type Output = tuple[int, ...]
type ErrorCounts = tuple[int, ...]


def transcript_outputs(compiled: Compiled, inputs: Sequence[int]) -> dict[Output, set[ErrorCounts]]:
    """Every output some transcript produces, with the ``(l_r)_r`` of those transcripts.

    Enumerates every assignment of every gate (``2**(width * gates)``
    transcripts) and derives each one's error set from the gates whose
    value disagrees with their recorded arguments.  Policy-independent, so
    one enumeration serves every ``theta``.
    """

    circuit = compiled.circuit
    index = compiled.index
    owner = unit_owner(index)
    replay_of = [index.verification_unit(u).replay_unit for u in range(index.verification_unit_count)]
    gates = range(index.input_count, circuit.n)
    refs = [circuit[address] for address in gates]
    outputs: dict[Output, set[ErrorCounts]] = {}
    for values in itertools.product(*(range(1 << ref.width) for ref in refs)):
        cells = list(inputs) + list(values)
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
) -> set[Output]:
    """``Y_eta``: the outputs whose best transcript survives with probability above ``eta``."""

    chances: dict[ErrorCounts, Fraction] = {}
    accepted: set[Output] = set()
    for output, variants in outputs.items():
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
    "cover_bits",
    "cut_bits",
    "error_counts",
    "error_sets",
    "subset_sum_bits",
    "transcript_outputs",
    "unit_owner",
]
