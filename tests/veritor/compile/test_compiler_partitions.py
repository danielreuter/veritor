from __future__ import annotations

from collections.abc import Iterable

import pytest

from veritor.compile import (
    POSITIVE_TOP_LEVEL_OCCURRENCES,
    SINGLETON_GATES,
    WHOLE_ROOT,
    BatchInput,
    CallDagCircuit,
    DemoG,
    PartitionPolicy,
    Producer,
    compile_call_dag,
    derive_occurrence_replay_plan,
    derive_replay_boundary,
    derive_replay_partition,
    derive_verification_partition,
    make_demo_request,
    make_word_kernel,
    occurrence_paths_for_policy,
)
from veritor.core import (
    InvalidArtifact,
    ReplayPartition,
    VerificationPartition,
    iter_domain,
    validate_compiled_result,
)

CELL_BITS = 8


def make_batch(lengths: tuple[int, ...]) -> BatchInput:
    return BatchInput(
        tuple(
            make_demo_request(length, index + 1, CELL_BITS)
            for index, length in enumerate(lengths)
        )
    )


def compile_demo(
    lengths: tuple[int, ...] = (1, 2),
    *,
    replay_policy: PartitionPolicy | str = POSITIVE_TOP_LEVEL_OCCURRENCES,
    verification_policy: PartitionPolicy | str = SINGLETON_GATES,
    replay_configuration=None,
    verification_configuration=None,
):
    kernel = make_word_kernel(CELL_BITS)
    batch = make_batch(lengths)
    result = compile_call_dag(
        kernel,
        DemoG(CELL_BITS),
        batch,
        b"",
        input_cells=batch.cells(),
        advice_bound_bits=0,
        replay_policy=replay_policy,
        verification_policy=verification_policy,
        replay_configuration=replay_configuration,
        verification_configuration=verification_configuration,
    )
    return kernel, batch, result


def members(units: Iterable) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(iter_domain(unit.members)) for unit in units)


def test_compile_returns_literal_validated_tuple_with_global_positions():
    _, batch, result = compile_demo()

    assert type(result) is tuple
    assert len(result) == 3
    circuit, replay, verification = result
    assert isinstance(circuit, CallDagCircuit)
    assert isinstance(replay, ReplayPartition)
    assert isinstance(verification, VerificationPartition)
    assert circuit.input_count == len(batch.cells())
    assert circuit.gate_count == 6
    assert circuit.computed_positions.start == circuit.input_count
    assert circuit.computed_positions.stop == circuit.input_count + 6
    assert members(replay.units) == (
        (circuit.input_count, circuit.input_count + 1),
        tuple(range(circuit.input_count + 2, circuit.input_count + 6)),
    )
    assert members(verification.units) == tuple(
        (position,)
        for position in range(circuit.input_count, circuit.input_count + 6)
    )

    tuple_identity = validate_compiled_result(*result)
    assert tuple_identity.structure_digest == circuit.identity.digest
    assert tuple_identity.replay_partition_digest == replay.identity.digest
    assert tuple_identity.verification_partition_digest == (
        verification.identity.digest
    )


@pytest.mark.parametrize(
    ("policy", "expected_sizes"),
    [
        (WHOLE_ROOT, (6,)),
        (POSITIVE_TOP_LEVEL_OCCURRENCES, (2, 4)),
        (SINGLETON_GATES, (1, 1, 1, 1, 1, 1)),
    ],
)
def test_replay_policies_are_deterministic_exact_covers(policy, expected_sizes):
    _, _, (first_circuit, first_replay, _) = compile_demo(
        replay_policy=policy,
    )
    _, _, (second_circuit, second_replay, _) = compile_demo(
        replay_policy=policy,
    )

    assert tuple(unit.count for unit in first_replay.units) == expected_sizes
    assert tuple(
        first_replay.owner_of(position)
        for position in iter_domain(first_circuit.computed_positions)
    ) == tuple(
        owner
        for owner, size in enumerate(expected_sizes)
        for _ in range(size)
    )
    assert first_circuit.identity == second_circuit.identity
    assert first_replay.identity == second_replay.identity
    first_replay.validate()


def test_singleton_verification_exactly_refines_every_replay_policy():
    for replay_policy in (
        WHOLE_ROOT,
        POSITIVE_TOP_LEVEL_OCCURRENCES,
        SINGLETON_GATES,
    ):
        _, _, (circuit, replay, verification) = compile_demo(
            replay_policy=replay_policy,
        )
        assert verification.unit_count == circuit.gate_count
        for unit in verification.units:
            position = unit.members.unrank(0)
            assert unit.count == 1
            assert unit.replay_unit == replay.owner_of(position)
            assert unit.proof_relation_id == (
                circuit.executable_gate_at(position).relation_id
            )
        verification.validate(replay)


def test_coarse_verification_policy_rejects_cross_replay_units():
    kernel, batch, (circuit, replay, _) = compile_demo(
        replay_policy=POSITIVE_TOP_LEVEL_OCCURRENCES,
    )
    del kernel, batch

    with pytest.raises(InvalidArtifact, match="crosses replay units"):
        derive_verification_partition(circuit, replay, WHOLE_ROOT)


def test_policy_and_caller_configuration_are_partition_identity_bound():
    _, _, (circuit, replay_a, verification_a) = compile_demo(
        replay_configuration={"batching": 1},
        verification_configuration={"checks": "single"},
    )
    _, _, (_, replay_b, verification_b) = compile_demo(
        replay_configuration={"batching": 2},
        verification_configuration={"checks": "single"},
    )
    _, _, (_, replay_c, verification_c) = compile_demo(
        replay_policy=WHOLE_ROOT,
        replay_configuration={"batching": 1},
        verification_configuration={"checks": "single"},
    )
    replay_same = derive_replay_partition(
        circuit,
        POSITIVE_TOP_LEVEL_OCCURRENCES,
        configuration={"batching": 1},
    )
    verification_other_config = derive_verification_partition(
        circuit,
        replay_same,
        SINGLETON_GATES,
        configuration={"checks": "other"},
    )

    assert replay_a.identity == replay_same.identity
    assert replay_a.identity != replay_b.identity
    assert replay_a.identity != replay_c.identity
    assert verification_a.identity != verification_b.identity
    assert verification_a.identity != verification_c.identity
    assert verification_a.identity != verification_other_config.identity


@pytest.mark.parametrize(
    "policy",
    [WHOLE_ROOT, POSITIVE_TOP_LEVEL_OCCURRENCES, SINGLETON_GATES],
)
def test_generic_boundary_matches_kernel_occurrence_oracle(policy):
    _, _, (circuit, _, _) = compile_demo()
    replay = derive_replay_partition(circuit, policy)
    occurrence_plan = derive_occurrence_replay_plan(circuit, policy)

    assert derive_replay_boundary(circuit, replay) == occurrence_plan.boundary


def test_boundary_contains_cross_unit_writes_and_deduplicated_outputs():
    producer = Producer(CELL_BITS)

    @producer.gate(name="add")
    def add(left, right):
        return left + right

    @producer.circuit(key="root", input_count=1)
    def root(value):
        first = add(value, value)
        second = add(first, value)
        return second, second, value

    kernel = make_word_kernel(CELL_BITS)

    def constructor(_x, _a):
        return producer.serialize(root)

    circuit, replay, verification = compile_call_dag(
        kernel,
        constructor,
        None,
        b"",
        input_cells=(3,),
        advice_bound_bits=0,
        replay_policy=SINGLETON_GATES,
    )

    assert circuit.ordered_output_positions == (2, 2, 0)
    assert tuple(port.position for port in circuit.output_ports) == (2, 2, 0)
    assert derive_replay_boundary(circuit, replay) == (0, 1, 2)
    assert circuit.evaluate((3,)) == (9, 9, 3)
    validate_compiled_result(circuit, replay, verification)


def test_dead_child_gate_and_passthrough_output_have_exact_boundary():
    producer = Producer(CELL_BITS)

    @producer.gate(name="add")
    def add(left, right):
        return left + right

    @producer.circuit(key="passthrough", input_count=2)
    def passthrough(value, dead_input):
        add(dead_input, dead_input)
        return value

    @producer.circuit(key="root", input_count=2)
    def root(left, right):
        produced = add(left, left)
        passed = passthrough(produced, right)
        return add(passed, right)

    kernel = make_word_kernel(CELL_BITS)

    def constructor(_x, _a):
        return producer.serialize(root)

    circuit, replay, _ = compile_call_dag(
        kernel,
        constructor,
        None,
        b"",
        input_cells=(3, 5),
        advice_bound_bits=0,
    )
    child = circuit.occurrence_summary((1,))
    occurrence_plan = derive_occurrence_replay_plan(
        circuit,
        POSITIVE_TOP_LEVEL_OCCURRENCES,
    )

    assert child.external_reads == (1,)
    assert child.outputs == (2,)
    assert occurrence_plan.boundary == (0, 1, 2, 4)
    assert 3 not in occurrence_plan.boundary
    assert derive_replay_boundary(circuit, replay) == occurrence_plan.boundary


def test_zero_gate_passthrough_compiles_to_empty_partitions():
    _, batch, result = compile_demo((0,))
    circuit, replay, verification = result

    assert circuit.gate_count == 0
    assert circuit.computed_positions.count == 0
    assert replay.units == ()
    assert verification.units == ()
    assert circuit.ordered_output_positions == (0,)
    assert derive_replay_boundary(circuit, replay) == tuple(
        range(circuit.input_count)
    )
    assert circuit.evaluate(batch.cells()) == (batch.cells()[0],)
    validate_compiled_result(*result)


def test_zero_gate_siblings_are_omitted_from_top_level_replay_cut():
    _, _, (circuit, replay, _) = compile_demo((0, 1, 0))

    assert circuit.gate_count == 2
    assert replay.unit_count == 1
    assert replay.units[0].count == 2
    assert occurrence_paths_for_policy(
        circuit,
        POSITIVE_TOP_LEVEL_OCCURRENCES,
    ) == ((1,),)


def test_compile_validates_advice_before_running_constructor():
    calls = 0

    def constructor(_x, _a):
        nonlocal calls
        calls += 1
        raise AssertionError("must not run")

    with pytest.raises(InvalidArtifact, match="advice exceeds"):
        compile_call_dag(
            make_word_kernel(CELL_BITS),
            constructor,
            None,
            b"\x00",
            input_cells=(),
            advice_bound_bits=0,
        )
    assert calls == 0


def test_partitions_from_another_structure_are_rejected_by_boundary():
    _, _, (first_circuit, first_replay, _) = compile_demo((1,))
    _, _, (second_circuit, _, _) = compile_demo((2,))

    with pytest.raises(InvalidArtifact, match="another structure"):
        derive_replay_boundary(second_circuit, first_replay)
    assert first_circuit.identity != second_circuit.identity

