from __future__ import annotations

from collections.abc import Iterable

import pytest

from veritor.compile import (
    CallDagCircuit,
    PartitionPolicy,
    Producer,
    compile_call_dag,
    make_word_kernel,
    occurrence_paths_for_policy,
    replay_partition_from_occurrences,
    verification_partition_from_occurrences,
)
from veritor.core import (
    CompiledArtifact,
    IntervalDomain,
    InvalidArtifact,
    ReplayPartition,
    VerificationPartition,
    derive_replay_boundary,
    iter_domain,
    validate_replay_boundary,
)
from veritor.plugins import BatchInput, DemoG, make_demo_request

CELL_BITS = 8
WHOLE_ROOT = PartitionPolicy.WHOLE_ROOT
POSITIVE_TOP_LEVEL_OCCURRENCES = PartitionPolicy.POSITIVE_TOP_LEVEL_OCCURRENCES
SINGLETON_GATES = PartitionPolicy.SINGLETON_GATES


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
) -> tuple[BatchInput, CompiledArtifact]:
    batch = make_batch(lengths)
    artifact = compile_call_dag(
        make_word_kernel(CELL_BITS),
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
    return batch, artifact


def compile_producer(root, input_cells: tuple[int, ...], **options) -> CompiledArtifact:
    producer = root.producer
    return compile_call_dag(
        make_word_kernel(CELL_BITS),
        lambda _x, _a: producer.serialize(root),
        None,
        b"",
        input_cells=input_cells,
        advice_bound_bits=0,
        **options,
    )


def members(units: Iterable) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(iter_domain(unit.members)) for unit in units)


def test_compile_returns_validated_artifact_with_global_positions():
    batch, artifact = compile_demo()
    circuit = artifact.circuit

    assert isinstance(artifact, CompiledArtifact)
    assert isinstance(circuit, CallDagCircuit)
    assert isinstance(artifact.replay, ReplayPartition)
    assert isinstance(artifact.verification, VerificationPartition)
    assert isinstance(artifact.boundary, IntervalDomain)
    assert artifact.executable
    assert circuit.input_count == len(batch.cells())
    assert circuit.gate_count == 6
    assert circuit.computed_positions.start == circuit.input_count
    assert circuit.computed_positions.stop == circuit.input_count + 6
    assert members(artifact.replay.units) == (
        (circuit.input_count, circuit.input_count + 1),
        tuple(range(circuit.input_count + 2, circuit.input_count + 6)),
    )
    assert members(artifact.verification.units) == tuple(
        (position,)
        for position in range(circuit.input_count, circuit.input_count + 6)
    )
    assert artifact.identity.structure_digest == circuit.identity.digest
    assert artifact.identity.replay_partition_digest == artifact.replay.identity.digest
    assert artifact.identity.verification_partition_digest == (
        artifact.verification.identity.digest
    )
    assert artifact.identity.boundary_digest == artifact.boundary.identity_digest


@pytest.mark.parametrize(
    ("policy", "expected_sizes"),
    [
        (WHOLE_ROOT, (6,)),
        (POSITIVE_TOP_LEVEL_OCCURRENCES, (2, 4)),
        (SINGLETON_GATES, (1, 1, 1, 1, 1, 1)),
    ],
)
def test_replay_policies_are_deterministic_exact_covers(policy, expected_sizes):
    _, first = compile_demo(replay_policy=policy)
    _, second = compile_demo(replay_policy=policy)

    assert tuple(unit.count for unit in first.replay.units) == expected_sizes
    assert tuple(
        first.replay.owner_of(position)
        for position in iter_domain(first.circuit.computed_positions)
    ) == tuple(
        owner
        for owner, size in enumerate(expected_sizes)
        for _ in range(size)
    )
    assert first.circuit.identity == second.circuit.identity
    assert first.replay.identity == second.replay.identity
    assert first.identity == second.identity


def test_singleton_verification_exactly_refines_every_replay_policy():
    for replay_policy in (
        WHOLE_ROOT,
        POSITIVE_TOP_LEVEL_OCCURRENCES,
        SINGLETON_GATES,
    ):
        _, artifact = compile_demo(replay_policy=replay_policy)
        assert artifact.verification.unit_count == artifact.circuit.gate_count
        for unit in artifact.verification.units:
            position = unit.members.unrank(0)
            assert unit.count == 1
            assert unit.replay_unit == artifact.replay.owner_of(position)
            assert unit.proof_relation_id == (
                artifact.circuit.executable_gate_at(position).relation_id
            )
        assert artifact.verification.replay_partition_identity == artifact.replay.identity


def test_coarse_verification_policy_rejects_cross_replay_units():
    _, artifact = compile_demo(replay_policy=POSITIVE_TOP_LEVEL_OCCURRENCES)

    with pytest.raises(InvalidArtifact, match="crosses replay units"):
        verification_partition_from_occurrences(
            artifact.circuit,
            artifact.replay,
            occurrence_paths_for_policy(artifact.circuit, WHOLE_ROOT),
        )


def test_policy_and_caller_configuration_are_partition_identity_bound():
    _, a = compile_demo(
        replay_configuration={"batching": 1},
        verification_configuration={"checks": "single"},
    )
    _, same = compile_demo(
        replay_configuration={"batching": 1},
        verification_configuration={"checks": "single"},
    )
    _, b = compile_demo(
        replay_configuration={"batching": 2},
        verification_configuration={"checks": "single"},
    )
    _, c = compile_demo(
        replay_policy=WHOLE_ROOT,
        replay_configuration={"batching": 1},
        verification_configuration={"checks": "single"},
    )
    _, other_checks = compile_demo(
        replay_configuration={"batching": 1},
        verification_configuration={"checks": "other"},
    )

    assert a.identity == same.identity
    assert a.replay.identity != b.replay.identity
    assert a.replay.identity != c.replay.identity
    assert a.verification.identity != b.verification.identity
    assert a.verification.identity != c.verification.identity
    assert a.replay.identity == other_checks.replay.identity
    assert a.verification.identity != other_checks.verification.identity
    assert a.identity != other_checks.identity


@pytest.mark.parametrize(
    "policy",
    [WHOLE_ROOT, POSITIVE_TOP_LEVEL_OCCURRENCES, SINGLETON_GATES],
)
def test_occurrence_boundary_matches_gate_scan_reference(policy):
    _, artifact = compile_demo((1, 2, 0, 3), replay_policy=policy)
    _, plan = replay_partition_from_occurrences(
        artifact.circuit, occurrence_paths_for_policy(artifact.circuit, policy)
    )

    assert plan.boundary == artifact.boundary
    assert derive_replay_boundary(artifact.circuit, artifact.replay) == artifact.boundary
    validate_replay_boundary(artifact)


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

    artifact = compile_producer(root, (3,), replay_policy=SINGLETON_GATES)
    circuit = artifact.circuit

    assert circuit.ordered_output_positions == (2, 2, 0)
    assert tuple(port.position for port in circuit.output_ports) == (2, 2, 0)
    assert tuple(artifact.boundary) == (0, 1, 2)
    assert artifact.interior(0).count == artifact.interior(1).count == 0
    assert circuit.evaluate((3,)) == (9, 9, 3)
    validate_replay_boundary(artifact)


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

    artifact = compile_producer(root, (3, 5))
    child = artifact.circuit.occurrence_summary((1,))

    assert child.external_reads == (1,)
    assert child.outputs == (2,)
    assert tuple(artifact.boundary) == (0, 1, 2, 4)
    assert 3 not in artifact.boundary
    assert artifact.value_owner(3) == artifact.replay.owner_of(3)
    assert derive_replay_boundary(artifact.circuit, artifact.replay) == artifact.boundary


def test_zero_gate_passthrough_compiles_to_empty_partitions():
    batch, artifact = compile_demo((0,))
    circuit = artifact.circuit

    assert circuit.gate_count == 0
    assert circuit.computed_positions.count == 0
    assert artifact.replay.units == ()
    assert artifact.verification.units == ()
    assert circuit.ordered_output_positions == (0,)
    assert tuple(artifact.boundary) == tuple(range(circuit.input_count))
    assert circuit.evaluate(batch.cells()) == (batch.cells()[0],)
    validate_replay_boundary(artifact)


def test_zero_gate_siblings_are_omitted_from_top_level_replay_cut():
    _, artifact = compile_demo((0, 1, 0))

    assert artifact.circuit.gate_count == 2
    assert artifact.replay.unit_count == 1
    assert artifact.replay.units[0].count == 2
    assert occurrence_paths_for_policy(
        artifact.circuit,
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


def test_partitions_from_another_structure_are_rejected():
    _, first = compile_demo((1,))
    _, second = compile_demo((2,))

    assert first.circuit.identity != second.circuit.identity
    with pytest.raises(InvalidArtifact, match="another structure"):
        CompiledArtifact(second.circuit, first.replay, first.verification, first.boundary)
    with pytest.raises(InvalidArtifact, match="another structure"):
        verification_partition_from_occurrences(
            second.circuit,
            first.replay,
            occurrence_paths_for_policy(second.circuit, SINGLETON_GATES),
        )
