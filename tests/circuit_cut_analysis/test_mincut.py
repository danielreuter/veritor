from __future__ import annotations

import itertools
import math
import random

import pytest

from circuit_cut_analysis.capacity import GateCapacity
from circuit_cut_analysis.circuit import CircuitDAG, Gate
from circuit_cut_analysis.mincut import (
    CanonicalCut,
    CutPolicy,
    CutStatus,
    minimum_vertex_cut,
)


def make_circuit(
    widths: dict[str, int],
    edges: list[tuple[str, str]],
    outputs: set[str],
) -> CircuitDAG:
    return CircuitDAG(
        (Gate(gate_id, width) for gate_id, width in widths.items()),
        edges,
        outputs,
    )


def brute_force_cut(
    circuit: CircuitDAG,
    sources: set[str],
    outputs: set[str],
    cuttable: set[str],
) -> tuple[float, set[frozenset[str]]] | None:
    best_width: float | None = None
    best_cuts: set[frozenset[str]] = set()
    candidates = sorted(cuttable)
    for size in range(len(candidates) + 1):
        for subset in itertools.combinations(candidates, size):
            cut = frozenset(subset)
            if not circuit.is_downstream_cut(sources, cut, outputs):
                continue
            width = sum(circuit.gates[gate_id].width_bits for gate_id in cut)
            if best_width is None or width < best_width:
                best_width = width
                best_cuts = {cut}
            elif width == best_width:
                best_cuts.add(cut)
    if best_width is None:
        return None
    return best_width, best_cuts


def test_equal_width_chain_returns_extremal_tied_cuts() -> None:
    circuit = make_circuit(
        {"source": 9, "middle": 4, "output": 4},
        [("source", "middle"), ("middle", "output")],
        {"output"},
    )

    result = minimum_vertex_cut(circuit, {"source"})

    assert result.status is CutStatus.FINITE
    assert result.width_bits == 4
    assert result.source_most_cut == frozenset({"middle"})
    assert result.downstream_most_cut == frozenset({"output"})
    assert result.cut == frozenset({"output"})
    assert result.tied


def test_canonical_source_most_can_be_requested() -> None:
    circuit = make_circuit(
        {"source": 4, "output": 4},
        [("source", "output")],
        {"output"},
    )

    result = minimum_vertex_cut(
        circuit,
        {"source"},
        canonical=CanonicalCut.SOURCE_MOST,
    )

    assert result.cut == frozenset({"source"})
    assert result.source_most_cut == frozenset({"source"})
    assert result.downstream_most_cut == frozenset({"output"})


def test_parallel_paths_need_a_set_valued_cut() -> None:
    circuit = make_circuit(
        {"source": 100, "left": 8, "right": 16, "output": 100},
        [
            ("source", "left"),
            ("source", "right"),
            ("left", "output"),
            ("right", "output"),
        ],
        {"output"},
    )

    result = minimum_vertex_cut(circuit, {"source"})

    assert result.width_bits == 24
    assert result.cut == frozenset({"left", "right"})
    assert not result.tied


def test_reconvergence_can_make_a_single_later_gate_cheaper() -> None:
    circuit = make_circuit(
        {"source": 100, "left": 8, "right": 8, "merge": 12, "output": 100},
        [
            ("source", "left"),
            ("source", "right"),
            ("left", "merge"),
            ("right", "merge"),
            ("merge", "output"),
        ],
        {"output"},
    )

    result = minimum_vertex_cut(circuit, {"source"})

    assert result.width_bits == 12
    assert result.cut == frozenset({"merge"})


def test_dead_source_is_reported_as_no_path() -> None:
    circuit = make_circuit(
        {"dead": 3, "live": 5, "output": 5},
        [("live", "output")],
        {"output"},
    )

    result = minimum_vertex_cut(circuit, {"dead"})

    assert result.status is CutStatus.NO_PATH
    assert result.width_bits == 0
    assert result.cut == frozenset()
    assert result.reachable_outputs == frozenset()


def test_protected_direct_path_has_no_finite_cut() -> None:
    circuit = make_circuit(
        {"source": 8, "output": 8},
        [("source", "output")],
        {"output"},
    )

    result = minimum_vertex_cut(
        circuit,
        {"source"},
        cuttable=CutPolicy.INTERNAL,
    )

    assert result.status is CutStatus.NO_FINITE_CUT
    assert result.width_bits is None
    assert result.witness_path == ("source", "output")


def test_zero_length_path_obeys_endpoint_policy() -> None:
    circuit = make_circuit({"same": 7}, [], {"same"})

    cuttable = minimum_vertex_cut(circuit, {"same"})
    protected = minimum_vertex_cut(
        circuit,
        {"same"},
        cuttable=CutPolicy.DOWNSTREAM,
    )

    assert cuttable.status is CutStatus.FINITE
    assert cuttable.cut == frozenset({"same"})
    assert cuttable.width_bits == 7
    assert protected.status is CutStatus.NO_FINITE_CUT
    assert protected.witness_path == ("same",)


def test_multiple_sources_are_cut_jointly() -> None:
    circuit = make_circuit(
        {"a": 9, "b": 9, "left": 4, "right": 4, "merge": 7, "out": 20},
        [
            ("a", "left"),
            ("b", "right"),
            ("left", "merge"),
            ("right", "merge"),
            ("merge", "out"),
        ],
        {"out"},
    )

    result = minimum_vertex_cut(circuit, {"a", "b"})

    assert result.width_bits == 7
    assert result.cut == frozenset({"merge"})


def test_exact_non_power_capacity_changes_a_near_tie_cut() -> None:
    vocabulary = 50_257
    exact = CircuitDAG(
        (
            Gate("source", 32),
            Gate("left_token", GateCapacity.values(vocabulary)),
            Gate("right_token", GateCapacity.values(vocabulary)),
            Gate("later_left", 16),
            Gate("left_output", 32),
            Gate("right_output", 32),
        ),
        (
            ("source", "left_token"),
            ("source", "right_token"),
            ("left_token", "later_left"),
            ("later_left", "left_output"),
            ("right_token", "right_output"),
        ),
        {"left_output", "right_output"},
    )
    rounded = CircuitDAG(
        (
            Gate("source", 32),
            Gate("left_token", 16),
            Gate("right_token", 16),
            Gate("later_left", 16),
            Gate("left_output", 32),
            Gate("right_output", 32),
        ),
        exact.edges,
        exact.outputs,
    )

    exact_result = minimum_vertex_cut(exact, {"source"})
    rounded_result = minimum_vertex_cut(rounded, {"source"})

    assert exact_result.exact_capacity is not None
    assert exact_result.exact_capacity.multiplier == vocabulary**2
    assert exact_result.cut == frozenset({"left_token", "right_token"})
    assert rounded_result.width_bits == 32
    assert rounded_result.source_most_cut == frozenset({"source"})
    assert rounded_result.downstream_most_cut == frozenset(
        {"later_left", "right_token"}
    )
    assert rounded_result.cut != exact_result.cut


def test_random_non_power_dags_match_exact_product_brute_force() -> None:
    rng = random.Random(50_257)
    cardinalities = (3, 5, 7, 8)
    for node_count in range(2, 7):
        for _ in range(10):
            ids = [f"g{i}" for i in range(node_count)]
            gates = {gate_id: rng.choice(cardinalities) for gate_id in ids}
            edges = [
                (ids[i], ids[j])
                for i in range(node_count)
                for j in range(i + 1, node_count)
                if rng.random() < 0.35
            ]
            circuit = CircuitDAG(
                (
                    Gate(gate_id, GateCapacity.values(cardinality))
                    for gate_id, cardinality in gates.items()
                ),
                edges,
                {ids[-1]},
            )
            best_product: int | None = None
            best_cuts: set[frozenset[str]] = set()
            for size in range(node_count + 1):
                for subset in itertools.combinations(ids, size):
                    if not circuit.is_downstream_cut({ids[0]}, subset):
                        continue
                    product = math.prod(gates[gate_id] for gate_id in subset)
                    cut = frozenset(subset)
                    if best_product is None or product < best_product:
                        best_product = product
                        best_cuts = {cut}
                    elif product == best_product:
                        best_cuts.add(cut)

            result = minimum_vertex_cut(circuit, {ids[0]})
            assert best_product is not None
            assert result.exact_capacity is not None
            assert result.exact_capacity.multiplier == best_product
            assert result.source_most_cut in best_cuts
            assert result.downstream_most_cut in best_cuts


def test_random_tiny_dags_match_brute_force() -> None:
    rng = random.Random(20260828)
    for node_count in range(2, 8):
        for _ in range(30):
            ids = [f"g{i}" for i in range(node_count)]
            widths = {gate_id: rng.randint(1, 9) for gate_id in ids}
            edges = [
                (ids[i], ids[j])
                for i in range(node_count)
                for j in range(i + 1, node_count)
                if rng.random() < 0.3
            ]
            circuit = make_circuit(widths, edges, {ids[-1]})
            sources = {ids[0]}
            cuttable = set(ids)

            exact = minimum_vertex_cut(circuit, sources)
            brute = brute_force_cut(circuit, sources, {ids[-1]}, cuttable)

            assert brute is not None
            brute_width, brute_cuts = brute
            assert exact.width_bits == brute_width
            assert exact.cut in brute_cuts
            assert exact.source_most_cut in brute_cuts
            assert exact.downstream_most_cut in brute_cuts


def test_validation_rejects_cycles_and_nonpositive_widths() -> None:
    with pytest.raises(ValueError, match="positive width"):
        Gate("bad", 0)
    with pytest.raises(ValueError, match="DAG"):
        make_circuit(
            {"a": 1, "b": 1},
            [("a", "b"), ("b", "a")],
            {"b"},
        )
