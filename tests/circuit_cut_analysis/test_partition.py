from __future__ import annotations

import itertools

from circuit_cut_analysis import (
    CanonicalCut,
    CircuitDAG,
    CutGroup,
    CutStatus,
    Gate,
    minimum_vertex_cut,
    partition_gate_cuts,
    singleton_source_cuts,
)


def make_circuit(
    widths: dict[str, int],
    edges: list[tuple[str, str]],
    outputs: set[str],
    *,
    inputs: set[str],
) -> CircuitDAG:
    return CircuitDAG(
        (
            Gate(
                gate_id,
                width,
                "input" if gate_id in inputs else "gate",
            )
            for gate_id, width in widths.items()
        ),
        edges,
        outputs,
    )


def brute_force_width(
    circuit: CircuitDAG,
    sources: frozenset[str],
    outputs: frozenset[str],
) -> float:
    gate_ids = sorted(circuit.gates)
    best: float | None = None
    for size in range(len(gate_ids) + 1):
        for candidate in itertools.combinations(gate_ids, size):
            if not circuit.is_downstream_cut(sources, candidate, outputs):
                continue
            width = sum(circuit.gates[gate_id].width_bits for gate_id in candidate)
            if best is None or width < best:
                best = width
    assert best is not None
    return best


def assert_joint_certificates(
    circuit: CircuitDAG,
    partition_outputs: frozenset[str],
    groups: tuple[CutGroup, ...],
) -> None:
    for group in groups:
        assert group.joint_cut_valid
        assert group.joint_minimum_matches_singletons
        assert group.verified
        assert circuit.is_downstream_cut(
            group.source_gates,
            group.cut,
            partition_outputs,
        )
        assert group.joint_result.width_bits == group.width_bits
        assert group.width_bits == brute_force_width(
            circuit,
            group.source_gates,
            partition_outputs,
        )


def test_chain_groups_gates_at_the_same_exact_narrow_cut() -> None:
    circuit = make_circuit(
        {"input": 20, "wide": 12, "narrow": 3, "output": 8},
        [("input", "wide"), ("wide", "narrow"), ("narrow", "output")],
        {"output"},
        inputs={"input"},
    )

    partition = partition_gate_cuts(circuit)
    group = partition.group_for_gate("wide")

    assert circuit.input_gates == frozenset({"input"})
    assert circuit.computed_gates == frozenset({"wide", "narrow", "output"})
    assert partition.source_gates == circuit.computed_gates
    assert "input" not in partition.singleton_cuts
    assert group.cut == frozenset({"narrow"})
    assert group.source_gates == frozenset({"wide", "narrow"})
    assert partition.group_for_gate("output").cut == frozenset({"output"})
    assert_joint_certificates(circuit, partition.outputs, partition.groups)


def test_equal_width_chain_uses_downstream_most_tie_break() -> None:
    circuit = make_circuit(
        {"input": 20, "first": 4, "second": 4, "output": 4},
        [("input", "first"), ("first", "second"), ("second", "output")],
        {"output"},
        inputs={"input"},
    )

    partition = partition_gate_cuts(circuit)
    first = partition.singleton_cuts["first"]

    assert first.canonical is CanonicalCut.DOWNSTREAM_MOST
    assert first.source_most_cut == frozenset({"first"})
    assert first.downstream_most_cut == frozenset({"output"})
    assert first.cut == frozenset({"output"})
    assert partition.group_for_gate("first").source_gates == frozenset(
        {"first", "second", "output"}
    )
    assert_joint_certificates(circuit, partition.outputs, partition.groups)


def test_parallel_equal_width_frontiers_choose_downstream_set() -> None:
    circuit = make_circuit(
        {
            "input": 20,
            "fork": 30,
            "left": 4,
            "right": 5,
            "left_output": 4,
            "right_output": 5,
        },
        [
            ("input", "fork"),
            ("fork", "left"),
            ("fork", "right"),
            ("left", "left_output"),
            ("right", "right_output"),
        ],
        {"left_output", "right_output"},
        inputs={"input"},
    )

    partition = partition_gate_cuts(circuit)
    fork = partition.singleton_cuts["fork"]

    assert fork.width_bits == 9
    assert fork.source_most_cut == frozenset({"left", "right"})
    assert fork.downstream_most_cut == frozenset({"left_output", "right_output"})
    assert partition.group_for_gate("fork").cut == frozenset(
        {"left_output", "right_output"}
    )
    assert_joint_certificates(circuit, partition.outputs, partition.groups)


def test_reconvergence_creates_one_shared_later_cut() -> None:
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

    partition = partition_gate_cuts(circuit)
    merge_group = partition.group_for_gate("fork")

    assert merge_group.cut == frozenset({"merge"})
    assert merge_group.width_bits == 9
    assert merge_group.source_gates == frozenset({"fork", "merge"})
    assert partition.group_for_gate("left").cut == frozenset({"left"})
    assert partition.group_for_gate("right").cut == frozenset({"right"})
    assert_joint_certificates(circuit, partition.outputs, partition.groups)


def test_dead_computation_gates_form_the_zero_width_no_path_group() -> None:
    circuit = make_circuit(
        {
            "live_input": 10,
            "live": 4,
            "output": 4,
            "dead_input": 10,
            "dead_one": 3,
            "dead_two": 2,
        },
        [
            ("live_input", "live"),
            ("live", "output"),
            ("dead_input", "dead_one"),
            ("dead_one", "dead_two"),
        ],
        {"output"},
        inputs={"live_input", "dead_input"},
    )

    partition = partition_gate_cuts(circuit)
    dead_group = partition.group_for_gate("dead_one")

    assert dead_group.cut == frozenset()
    assert dead_group.width_bits == 0
    assert dead_group.singleton_status is CutStatus.NO_PATH
    assert dead_group.source_gates == frozenset({"dead_one", "dead_two"})
    assert dead_group.joint_result.status is CutStatus.NO_PATH
    assert_joint_certificates(circuit, partition.outputs, partition.groups)


def test_multiple_outputs_are_used_in_every_singleton_query() -> None:
    circuit = make_circuit(
        {
            "input": 20,
            "fork": 30,
            "left": 4,
            "right": 5,
            "left_output": 10,
            "right_output": 10,
        },
        [
            ("input", "fork"),
            ("fork", "left"),
            ("fork", "right"),
            ("left", "left_output"),
            ("right", "right_output"),
        ],
        {"left_output", "right_output"},
        inputs={"input"},
    )

    partition = partition_gate_cuts(circuit)
    fork = partition.singleton_cuts["fork"]

    assert fork.reachable_outputs == frozenset({"left_output", "right_output"})
    assert fork.cut == frozenset({"left", "right"})
    assert fork.width_bits == 9
    assert_joint_certificates(circuit, partition.outputs, partition.groups)


def test_equal_shape_cuts_with_different_gate_ids_remain_distinct() -> None:
    circuit = make_circuit(
        {
            "input_a": 20,
            "work_a": 4,
            "output_a": 4,
            "input_b": 20,
            "work_b": 4,
            "output_b": 4,
        },
        [
            ("input_a", "work_a"),
            ("work_a", "output_a"),
            ("input_b", "work_b"),
            ("work_b", "output_b"),
        ],
        {"output_a", "output_b"},
        inputs={"input_a", "input_b"},
    )

    partition = partition_gate_cuts(circuit)
    group_a = partition.group_for_gate("work_a")
    group_b = partition.group_for_gate("work_b")

    assert group_a is not group_b
    assert group_a.cut == frozenset({"output_a"})
    assert group_b.cut == frozenset({"output_b"})
    assert len(group_a.cut) == len(group_b.cut) == 1
    assert group_a.width_bits == group_b.width_bits == 4
    assert group_a.source_gates == frozenset({"work_a", "output_a"})
    assert group_b.source_gates == frozenset({"work_b", "output_b"})
    assert_joint_certificates(circuit, partition.outputs, partition.groups)


def test_source_override_is_exact_and_may_include_fixed_inputs() -> None:
    circuit = make_circuit(
        {"input": 6, "middle": 5, "output": 4},
        [("input", "middle"), ("middle", "output")],
        {"output"},
        inputs={"input"},
    )

    partition = partition_gate_cuts(circuit, source_gates={"input", "middle"})
    singleton_defaults = singleton_source_cuts(circuit)

    assert partition.source_gates == frozenset({"input", "middle"})
    assert set(partition.singleton_cuts) == {"input", "middle"}
    assert set(singleton_defaults) == {"middle", "output"}
    assert partition.group_for_gate("input").cut == frozenset({"output"})
    assert_joint_certificates(circuit, partition.outputs, partition.groups)


def test_joint_cut_capacity_is_submodular() -> None:
    circuit = make_circuit(
        {
            "input": 20,
            "fork": 18,
            "left": 6,
            "right": 7,
            "merge": 9,
            "left_output": 5,
            "right_output": 8,
        },
        [
            ("input", "fork"),
            ("fork", "left"),
            ("fork", "right"),
            ("left", "merge"),
            ("right", "merge"),
            ("left", "left_output"),
            ("merge", "right_output"),
        ],
        {"left_output", "right_output"},
        inputs={"input"},
    )
    source_ids = tuple(sorted(circuit.computed_gates))
    source_sets = tuple(
        frozenset(selected)
        for size in range(len(source_ids) + 1)
        for selected in itertools.combinations(source_ids, size)
    )
    capacities = {
        sources: minimum_vertex_cut(circuit, sources).exact_capacity
        for sources in source_sets
    }

    for left in source_sets:
        for right in source_sets:
            left_capacity = capacities[left]
            right_capacity = capacities[right]
            union_capacity = capacities[left.union(right)]
            intersection_capacity = capacities[left.intersection(right)]
            assert left_capacity is not None
            assert right_capacity is not None
            assert union_capacity is not None
            assert intersection_capacity is not None
            assert left_capacity + right_capacity >= (
                union_capacity + intersection_capacity
            )


def test_canonical_region_compression_preserves_every_joint_capacity() -> None:
    circuit = make_circuit(
        {
            "input": 20,
            "wide": 12,
            "narrow": 3,
            "fork": 20,
            "left": 4,
            "right": 5,
            "left_output": 4,
            "right_output": 5,
        },
        [
            ("input", "wide"),
            ("wide", "narrow"),
            ("narrow", "fork"),
            ("fork", "left"),
            ("fork", "right"),
            ("left", "left_output"),
            ("right", "right_output"),
        ],
        {"left_output", "right_output"},
        inputs={"input"},
    )
    partition = partition_gate_cuts(circuit)
    source_ids = tuple(sorted(partition.source_gates))

    for size in range(len(source_ids) + 1):
        for selected in itertools.combinations(source_ids, size):
            sources = frozenset(selected)
            compressed = partition.compress_sources(sources)
            original = minimum_vertex_cut(circuit, sources)
            quotient = minimum_vertex_cut(circuit, compressed)

            assert quotient.exact_capacity == original.exact_capacity
            assert len(compressed) <= len(sources)
