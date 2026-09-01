from __future__ import annotations

from itertools import chain

import pytest

from veritor.compile import (
    MATMUL_REPLAY_PARTITION_ALGORITHM_ID,
    MATMUL_VERIFICATION_PARTITION_ALGORITHM_ID,
    MatmulWorkload,
    compile_matmul_workload,
    expected_matmul_outputs,
)
from veritor.core import (
    BOUNDARY_OWNER,
    CompiledArtifact,
    iter_domain,
    validate_replay_boundary,
)


def _workload(*, cell_bits: int = 8) -> MatmulWorkload:
    return MatmulWorkload(
        (
            (1, 2),
            (3, 4),
            (5, 6),
        ),
        (
            (
                (1, 2, 3),
                (4, 5, 6),
            ),
            ((7, 8, 9),),
        ),
        cell_bits=cell_bits,
    )


def test_shared_weight_matmul_executes_in_canonical_order() -> None:
    workload = _workload()
    artifact = compile_matmul_workload(workload)

    assert isinstance(artifact, CompiledArtifact)
    assert artifact.executable
    assert workload.weight_shape == (3, 2)
    assert workload.activation_shapes == ((2, 3), (1, 3))
    assert workload.output_shapes == ((2, 2), (1, 2))
    assert workload.public_inputs == (
        1,
        2,
        3,
        4,
        5,
        6,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
    )
    assert expected_matmul_outputs(workload) == (22, 28, 49, 64, 76, 100)
    assert artifact.circuit.evaluate(workload.public_inputs) == expected_matmul_outputs(
        workload
    )
    assert artifact.identity.structure_digest == artifact.circuit.identity.digest


def test_partitions_are_matmuls_refined_by_inner_products() -> None:
    workload = _workload()
    artifact = compile_matmul_workload(workload)
    circuit, replay, verification = artifact.circuit, artifact.replay, artifact.verification
    inner = workload.weight_shape[0]
    columns = workload.weight_shape[1]

    assert replay.identity.algorithm_id == MATMUL_REPLAY_PARTITION_ALGORITHM_ID
    assert (
        verification.identity.algorithm_id == MATMUL_VERIFICATION_PARTITION_ALGORITHM_ID
    )
    assert replay.unit_count == len(workload.activations) == 2
    assert verification.unit_count == sum(
        rows * columns for rows, _ in workload.activation_shapes
    )
    assert tuple(unit.count for unit in replay.units) == (20, 10)
    assert all(unit.count == 2 * inner - 1 for unit in verification.units)
    assert tuple(int(unit.replay_unit) for unit in verification.units) == (
        0,
        0,
        0,
        0,
        1,
        1,
    )

    covered = tuple(
        chain.from_iterable(iter_domain(unit.members) for unit in replay.units)
    )
    assert covered == tuple(iter_domain(circuit.computed_positions))


def test_shared_weight_input_positions_fan_out_across_replay_units() -> None:
    workload = MatmulWorkload(
        ((3,), (5,)),
        (
            ((7, 11),),
            ((13, 17),),
        ),
    )
    artifact = compile_matmul_workload(workload)
    circuit = artifact.circuit
    weight_positions = {0, 1}

    reads_by_replay = []
    for unit in artifact.replay.units:
        reads = {
            int(predecessor)
            for position in iter_domain(unit.members)
            for predecessor in circuit.gate_at(position).predecessors
            if int(predecessor) in weight_positions
        }
        reads_by_replay.append(reads)

    assert workload.public_inputs[:2] == (3, 5)
    assert reads_by_replay == [weight_positions, weight_positions]


def test_replay_boundary_is_exactly_the_public_io_of_the_matmuls() -> None:
    workload = _workload()
    artifact = compile_matmul_workload(workload)
    circuit = artifact.circuit
    input_positions = {int(port.position) for port in circuit.input_ports}
    output_positions = {int(port.position) for port in circuit.output_ports}

    assert set(iter_domain(artifact.boundary)) == input_positions | output_positions
    assert artifact.replay.unit_count == len(workload.activations)
    for unit in artifact.replay.units:
        interior = artifact.interior(unit.index)
        assert interior.count == unit.count - len(
            output_positions & set(iter_domain(unit.members))
        )
        assert all(artifact.value_owner(p) == unit.index for p in iter_domain(interior))
    assert all(artifact.value_owner(p) == BOUNDARY_OWNER for p in input_positions)
    assert all(artifact.value_owner(p) == BOUNDARY_OWNER for p in output_positions)
    validate_replay_boundary(artifact)


def test_modular_overflow_is_applied_to_every_output() -> None:
    workload = MatmulWorkload(
        ((15,),),
        (
            ((15,),),
            ((14,),),
        ),
        cell_bits=4,
    )
    artifact = compile_matmul_workload(workload)

    assert expected_matmul_outputs(workload) == (1, 2)
    assert artifact.circuit.evaluate(workload.public_inputs) == (1, 2)


@pytest.mark.parametrize(
    ("weights", "activations", "match"),
    (
        ((), (((1,),),), "nonempty matrix"),
        (((1,), (2, 3)), (((1, 2),),), "rectangular"),
        (((1,),), (), "nonempty sequence"),
        (((1,), (2,)), (((1,), (2,)),), "contraction width"),
        (((256,),), (((1,),),), "8-bit word"),
    ),
)
def test_workload_rejects_malformed_shapes_and_values(
    weights: tuple[tuple[int, ...], ...],
    activations: tuple[tuple[tuple[int, ...], ...], ...],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        MatmulWorkload(weights, activations)


def test_structure_identity_depends_on_shape_and_width_not_public_values() -> None:
    first = _workload()
    second = MatmulWorkload(
        ((7, 8), (9, 10), (11, 12)),
        (
            ((2, 3, 4), (5, 6, 7)),
            ((8, 9, 10),),
        ),
    )
    wider = MatmulWorkload(
        first.weights,
        first.activations,
        cell_bits=16,
    )

    first_artifact = compile_matmul_workload(first)
    second_artifact = compile_matmul_workload(second)
    wider_artifact = compile_matmul_workload(wider)

    assert first_artifact.circuit.identity == second_artifact.circuit.identity
    assert first_artifact.replay.identity == second_artifact.replay.identity
    assert first_artifact.verification.identity == second_artifact.verification.identity
    assert first_artifact.identity == second_artifact.identity
    assert first_artifact.circuit.identity != wider_artifact.circuit.identity
    assert first_artifact.identity != wider_artifact.identity
