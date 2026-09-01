from __future__ import annotations

import pytest

from circuit_cut_analysis.capacity import LogCardinality
from circuit_cut_analysis.capacity_oracle import (
    ExplicitCircuitCapacityOracle,
    StructuralCapacityEvaluation,
)
from circuit_cut_analysis.circuit import CircuitDAG, Gate
from circuit_cut_analysis.partition import partition_gate_cuts


def _chain() -> CircuitDAG:
    return CircuitDAG(
        gates=(
            Gate("input", 16, op="input"),
            Gate("wide", 12),
            Gate("narrow", 3),
            Gate("output", 8),
        ),
        edges=(
            ("input", "wide"),
            ("wide", "narrow"),
            ("narrow", "output"),
        ),
        outputs=("output",),
    )


def test_explicit_oracle_returns_exact_capacity_evidence() -> None:
    oracle = ExplicitCircuitCapacityOracle(_chain())

    evaluation = oracle.evaluate(frozenset({"wide", "narrow"}))

    assert evaluation.is_exact
    assert evaluation.exact_capacity == LogCardinality.bits(3)
    assert evaluation.requested_support == frozenset({"wide", "narrow"})
    assert evaluation.evaluated_support == evaluation.requested_support
    assert evaluation.cut_gate_ids == frozenset({"narrow"})
    assert evaluation.method == "explicit-min-cut"


def test_explicit_oracle_can_compress_canonical_regions_exactly() -> None:
    circuit = _chain()
    partition = partition_gate_cuts(circuit)
    oracle = ExplicitCircuitCapacityOracle(
        circuit,
        canonical_partition=partition,
    )

    evaluation = oracle.evaluate(frozenset({"wide", "narrow"}))

    assert evaluation.exact_capacity == LogCardinality.bits(3)
    assert len(evaluation.evaluated_support) == 1
    assert evaluation.evaluated_support.issubset(partition.source_gates)
    assert evaluation.method == "canonical-region-compressed-explicit-min-cut"


def test_oracle_rejects_partition_for_different_outputs() -> None:
    circuit = _chain()
    partition = partition_gate_cuts(circuit, outputs=("narrow",))

    with pytest.raises(ValueError, match="identical outputs"):
        ExplicitCircuitCapacityOracle(circuit, canonical_partition=partition)


def test_capacity_interval_rejects_reversed_bounds() -> None:
    with pytest.raises(ValueError, match="lower bound"):
        StructuralCapacityEvaluation(
            lower_bound=LogCardinality.bits(4),
            upper_bound=LogCardinality.bits(3),
            requested_support="attack",
            evaluated_support="attack",
            cut_gate_ids=None,
            method="test",
        )
