from __future__ import annotations

import pytest

from veritor.compile import (
    Compiler,
    MatmulG,
    MatmulWorkload,
    TracerError,
    expected_matmul_outputs,
)
from veritor.core import Compiled, make_word_gate_set


def _workload(*, width: int = 8) -> MatmulWorkload:
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
        width=width,
    )


def compile_workload(workload: MatmulWorkload) -> Compiled:
    gate_set = make_word_gate_set(workload.width)
    description = MatmulG(workload.width)(workload, b"")
    return Compiler(gate_set).compile(description, workload.public_inputs)


def outputs_of(compiled: Compiled, inputs: tuple[int, ...]) -> tuple[int, ...]:
    values = compiled.circuit.evaluate(inputs)
    return tuple(values[address] for address in compiled.circuit.outputs)


def test_shared_weight_matmul_executes_in_canonical_order() -> None:
    workload = _workload()
    compiled = compile_workload(workload)

    assert workload.weight_shape == (3, 2)
    assert workload.activation_shapes == ((2, 3), (1, 3))
    assert workload.output_shapes == ((2, 2), (1, 2))
    assert workload.public_inputs == (1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    assert expected_matmul_outputs(workload) == (22, 28, 49, 64, 76, 100)
    assert outputs_of(compiled, workload.public_inputs) == expected_matmul_outputs(workload)


def test_marks_make_rows_replay_units_and_dots_verification_units() -> None:
    workload = _workload()
    compiled = compile_workload(workload)
    index, circuit = compiled.index, compiled.circuit
    inner, columns = workload.weight_shape
    rows = sum(rows for rows, _ in workload.activation_shapes)

    assert index.replay_units.count == rows == 3
    assert index.verification_unit_count == rows * columns == 6
    for r in range(rows):
        unit = index.replay_units.unit(r)
        assert unit.role == "replay"
        assert unit.size == columns * (2 * inner - 1)
        assert index.verification_units(r).count == columns
        assert all(v.size == 2 * inner - 1 for v in index.verification_units(r))
        assert all(v.replay_unit == r for v in index.verification_units(r))
        assert len(circuit.Out(unit)) == columns
    covered = [a for r in range(rows) for a in index.replay_units.unit(r).interval]
    assert covered == list(range(circuit.input_count, circuit.n))


def test_shared_weight_inputs_fan_out_across_replay_units() -> None:
    workload = MatmulWorkload(((3,), (5,)), (((7, 11),), ((13, 17),)))
    compiled = compile_workload(workload)
    circuit, index = compiled.circuit, compiled.index
    weights = {0, 1}

    reads_by_replay = [
        set(circuit.In(index.replay_units.unit(r))) & weights
        for r in range(index.replay_units.count)
    ]

    assert workload.public_inputs[:2] == (3, 5)
    assert reads_by_replay == [weights, weights]


def test_boundary_is_exactly_the_public_io_of_the_rows() -> None:
    workload = _workload()
    compiled = compile_workload(workload)
    circuit, index = compiled.circuit, compiled.index
    boundary = index.boundary()
    io = set(circuit.inputs) | set(circuit.outputs)

    assert set(boundary) == io
    for r in range(index.replay_units.count):
        unit = index.replay_units.unit(r)
        interior = index.interior(r)
        assert set(interior) == set(unit.interval) - io
        assert all(index.replay_units.owner(a) == r for a in interior)


def test_modular_overflow_is_applied_to_every_output() -> None:
    workload = MatmulWorkload(((15,),), (((15,),), ((14,),)), width=4)
    compiled = compile_workload(workload)

    assert expected_matmul_outputs(workload) == (1, 2)
    assert outputs_of(compiled, workload.public_inputs) == (1, 2)


@pytest.mark.parametrize(
    ("weights", "activations", "match"),
    (
        ((), (((1,),),), "nonempty matrix"),
        (((1,), (2, 3)), (((1, 2),),), "rectangular"),
        (((1,),), (), "nonempty sequence"),
        (((1,), (2,)), (((1,), (2,)),), "contraction width"),
        (((256,),), (((1,),),), "8-bit value"),
    ),
)
def test_workload_rejects_malformed_shapes_and_values(
    weights: tuple[tuple[int, ...], ...],
    activations: tuple[tuple[tuple[int, ...], ...], ...],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        MatmulWorkload(weights, activations)


def test_constructor_rejects_foreign_inputs_and_advice() -> None:
    with pytest.raises(TracerError, match="expects MatmulWorkload"):
        MatmulG(8)(object(), b"")
    with pytest.raises(TracerError, match="width differs"):
        MatmulG(16)(_workload(), b"")
    with pytest.raises(TracerError, match="advice"):
        MatmulG(8)(_workload(), b"hint")


def test_digest_depends_on_shape_and_width_not_public_values() -> None:
    first = _workload()
    second = MatmulWorkload(
        ((7, 8), (9, 10), (11, 12)),
        (((2, 3, 4), (5, 6, 7)), ((8, 9, 10),)),
    )
    wider = MatmulWorkload(first.weights, first.activations, width=16)

    assert compile_workload(first).digest == compile_workload(second).digest
    assert compile_workload(first).digest != compile_workload(wider).digest


def description(rows: int, k: int, columns: int) -> bytes:
    weights = tuple((1,) * columns for _ in range(k))
    return MatmulG(8)(MatmulWorkload(weights, (tuple((1,) * k for _ in range(rows)),)), b"")


def test_description_size_does_not_grow_with_rows_or_columns() -> None:
    small, large = description(16, 8, 16), description(1024, 8, 1024)

    # Rows and columns are ``repeat`` counts: only their digits change.
    assert len(large) - len(small) < 32


def test_description_size_is_logarithmic_in_the_contraction_length() -> None:
    sizes = [len(description(4, k, 4)) for k in (64, 256, 1024)]

    # One more sum-tree level per doubling of ``k``; each level is one repeat step.
    assert sizes[1] - sizes[0] <= sizes[0] // 4
    assert sizes[2] - sizes[1] <= sizes[0] // 4
    assert sizes[2] < 4096
