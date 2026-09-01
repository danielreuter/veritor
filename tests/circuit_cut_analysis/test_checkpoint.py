from __future__ import annotations

import itertools
import random
from collections.abc import Set as AbstractSet

from circuit_cut_analysis.checkpoint import compile_checkpoint_skeleton
from circuit_cut_analysis.circuit import CircuitDAG, Gate
from circuit_cut_analysis.mincut import minimum_vertex_cut


def make_circuit(
    widths: dict[str, int],
    edges: list[tuple[str, str]],
    outputs: set[str],
    *,
    inputs: AbstractSet[str] = frozenset(),
) -> CircuitDAG:
    return CircuitDAG(
        (
            Gate(gate_id, width, "input" if gate_id in inputs else "gate")
            for gate_id, width in widths.items()
        ),
        edges,
        outputs,
    )


def assert_every_subset_matches(circuit: CircuitDAG) -> None:
    skeleton = compile_checkpoint_skeleton(circuit)
    gate_ids = tuple(sorted(circuit.gates))
    for size in range(len(gate_ids) + 1):
        for selected in itertools.combinations(gate_ids, size):
            original = minimum_vertex_cut(circuit, selected)
            quotient = skeleton.evaluate(selected)
            assert quotient.exact_capacity == original.exact_capacity, (
                f"sources {selected!r}: skeleton {quotient.exact_capacity!r} "
                f"!= original {original.exact_capacity!r}"
            )


def test_region_contraction_counterexample_stays_exact_and_acyclic() -> None:
    """The known case where contracting canonical regions creates a cycle."""

    circuit = make_circuit(
        {"g0": 10, "g1": 6, "g2": 8},
        [("g0", "g1"), ("g0", "g2"), ("g1", "g2")],
        {"g2"},
    )

    skeleton = compile_checkpoint_skeleton(circuit)

    assert skeleton.fixed_atoms == frozenset({"g1", "g2"})
    assert skeleton.atom_frontier["g0"] == frozenset({"g2"})
    assert ("g1", "g2") in skeleton.skeleton.edges
    assert_every_subset_matches(circuit)


def test_overlapping_pair_and_singleton_cuts_do_not_double_count() -> None:
    circuit = make_circuit(
        {
            "wide": 20,
            "mid": 4,
            "p0": 3,
            "p1": 3,
            "o0": 10,
            "o1": 10,
        },
        [
            ("wide", "p0"),
            ("wide", "p1"),
            ("mid", "p0"),
            ("p0", "o0"),
            ("p1", "o1"),
        ],
        {"o0", "o1"},
    )

    skeleton = compile_checkpoint_skeleton(circuit)

    assert skeleton.atom_frontier["wide"] == frozenset({"p0", "p1"})
    assert skeleton.atom_frontier["mid"] == frozenset({"p0"})
    joint = skeleton.evaluate({"wide", "mid"})
    assert joint.exact_capacity is not None
    assert joint.exact_capacity.width_bits == 6
    assert_every_subset_matches(circuit)


def test_equal_width_chain_projects_to_the_downstream_most_atom() -> None:
    circuit = make_circuit(
        {"input": 20, "first": 4, "second": 4, "third": 4, "output": 4},
        [
            ("input", "first"),
            ("first", "second"),
            ("second", "third"),
            ("third", "output"),
        ],
        {"output"},
        inputs={"input"},
    )

    skeleton = compile_checkpoint_skeleton(circuit)

    assert skeleton.fixed_atoms == frozenset({"output"})
    assert skeleton.atom_frontier["first"] == frozenset({"output"})
    assert skeleton.atom_frontier["input"] == frozenset({"output"})
    assert len(skeleton.skeleton.gates) == 1
    assert_every_subset_matches(circuit)


def test_dead_gates_project_to_the_empty_frontier() -> None:
    circuit = make_circuit(
        {
            "live_input": 10,
            "live": 4,
            "output": 4,
            "dead_input": 10,
            "dead": 3,
        },
        [
            ("live_input", "live"),
            ("live", "output"),
            ("dead_input", "dead"),
        ],
        {"output"},
        inputs={"live_input", "dead_input"},
    )

    skeleton = compile_checkpoint_skeleton(circuit)

    assert skeleton.atom_frontier["dead"] == frozenset()
    assert skeleton.atom_frontier["dead_input"] == frozenset()
    dead_only = skeleton.evaluate({"dead", "dead_input"})
    assert dead_only.exact_capacity is not None
    assert dead_only.exact_capacity.width_bits == 0
    mixed = skeleton.evaluate({"dead", "live"})
    assert mixed.exact_capacity is not None
    assert mixed.exact_capacity.width_bits == 4
    assert_every_subset_matches(circuit)


def test_random_dags_match_the_original_oracle_on_every_source_subset() -> None:
    rng = random.Random(20260829)
    for _ in range(120):
        gate_count = rng.randint(3, 7)
        ids = tuple(f"g{index}" for index in range(gate_count))
        widths = {gate_id: rng.randint(1, 6) for gate_id in ids}
        edges = [
            (ids[source], ids[target])
            for source in range(gate_count)
            for target in range(source + 1, gate_count)
            if rng.random() < 0.4
        ]
        output_count = rng.randint(1, 2)
        outputs = set(rng.sample(ids[-max(2, gate_count // 2) :], output_count))
        inputs = {ids[0]} if rng.random() < 0.5 else set()
        circuit = make_circuit(widths, edges, outputs, inputs=inputs)
        assert_every_subset_matches(circuit)


def test_skeleton_preserves_capacities_and_never_grows() -> None:
    circuit = make_circuit(
        {
            "input": 20,
            "fork": 30,
            "left": 6,
            "right": 6,
            "merge": 9,
            "output": 20,
        },
        [
            ("input", "fork"),
            ("fork", "left"),
            ("fork", "right"),
            ("left", "merge"),
            ("right", "merge"),
            ("merge", "output"),
        ],
        {"output"},
        inputs={"input"},
    )

    skeleton = compile_checkpoint_skeleton(circuit)

    assert len(skeleton.skeleton.gates) <= len(circuit.gates)
    for atom in skeleton.fixed_atoms:
        assert skeleton.skeleton.gates[atom].capacity == circuit.gates[atom].capacity
    assert circuit.outputs.issubset(skeleton.fixed_atoms)
    assert_every_subset_matches(circuit)
