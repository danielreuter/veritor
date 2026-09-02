from __future__ import annotations

import pytest

from veritor.compile import Compiler
from veritor.constructors import (
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


def outputs_of(compiled: Compiled, workload: MatmulWorkload) -> tuple[int, ...]:
    values = compiled.circuit.evaluate(workload.public_inputs, workload.weight_values)
    return tuple(values[address] for address in compiled.circuit.outputs)


def test_shared_weight_matmul_executes_in_canonical_order() -> None:
    workload = _workload()
    compiled = compile_workload(workload)

    assert workload.weight_shape == (3, 2)
    assert workload.activation_shapes == ((2, 3), (1, 3))
    assert workload.output_shapes == ((2, 2), (1, 2))
    assert workload.public_inputs == (1, 2, 3, 4, 5, 6, 7, 8, 9)  # the activations only
    assert workload.weight_values == (1, 2, 3, 4, 5, 6)
    assert workload.manifest["input_order"] == "activations-row-major"
    assert expected_matmul_outputs(workload) == (22, 28, 49, 64, 76, 100)
    assert outputs_of(compiled, workload) == expected_matmul_outputs(workload)
    assert (compiled.circuit.input_count, compiled.circuit.weight_count) == (9, 6)
    with pytest.raises(Exception, match="expected 6 weights, got 0"):
        compiled.circuit.evaluate(workload.public_inputs)


def test_marks_make_source_units_rows_replay_units_and_dots_verification_units() -> None:
    workload = _workload()
    compiled = compile_workload(workload)
    index, circuit = compiled.index, compiled.circuit
    inner, columns = workload.weight_shape
    rows = sum(rows for rows, _ in workload.activation_shapes)
    x_cells, w_cells = rows * inner, inner * columns

    assert index.root.frame.definition.input_count == 0  # the root has no ports
    assert index.replay_units.count == 2 + rows == 5
    assert index.verification_unit_count == x_cells + w_cells + rows * columns
    activations, weights = index.replay_units.unit(0), index.replay_units.unit(1)
    assert activations.interval == range(x_cells) and weights.interval == range(x_cells, x_cells + w_cells)
    assert list(circuit.inputs) == list(activations.interval)
    assert list(circuit.weights) == list(weights.interval)
    for r, unit in enumerate((activations, weights)):
        assert unit.role == "replay" and circuit.Out(unit) == () and index.interior(r).count == 0
        assert index.verification_units(r).count == unit.size
        assert all(v.size == 1 and circuit[v.interval.start].is_source for v in index.verification_units(r))
    for r in range(2, 2 + rows):
        unit = index.replay_units.unit(r)
        assert unit.role == "replay"
        assert unit.size == columns * (2 * inner - 1)
        assert index.verification_units(r).count == columns
        assert all(v.size == 2 * inner - 1 for v in index.verification_units(r))
        assert all(v.replay_unit == r for v in index.verification_units(r))
        assert len(circuit.Out(unit)) == columns
        # a row reads its activation row and all of W through ports
        assert circuit.In(unit) == tuple(range((r - 2) * inner, (r - 1) * inner)) + tuple(weights.interval)
    covered = [a for r in range(index.replay_units.count) for a in index.replay_units.unit(r).interval]
    assert covered == list(range(circuit.n))  # every gate, source gates included, is in a unit
    assert len(index.root.frame.definition.out_runs) == 1  # rows are pure dots: one run of outputs


def test_shared_weight_gates_fan_out_across_replay_units() -> None:
    workload = MatmulWorkload(((3,), (5,)), (((7, 11),), ((13, 17),)))
    compiled = compile_workload(workload)
    circuit, index = compiled.circuit, compiled.index
    weights = set(circuit.weights)

    reads_by_row = [
        set(circuit.In(index.replay_units.unit(r))) & weights for r in range(2, index.replay_units.count)
    ]

    assert workload.weight_values == (3, 5) and workload.public_inputs == (7, 11, 13, 17)
    assert weights == {4, 5}
    assert reads_by_row == [weights, weights]


def test_boundary_is_the_inputs_and_the_rows_outputs() -> None:
    workload = _workload()
    compiled = compile_workload(workload)
    circuit, index = compiled.circuit, compiled.index
    boundary = index.boundary()
    io = set(circuit.inputs) | set(circuit.outputs)

    assert set(boundary) == io  # the weight gates are under κ_W, not in the boundary
    assert set(index.weights()).isdisjoint(boundary)
    for r in range(index.replay_units.count):
        unit = index.replay_units.unit(r)
        interior = index.interior(r)
        assert set(interior) == set(unit.interval) - io - set(circuit.weights)
        assert all(index.replay_units.owner(a) == r for a in interior)


def test_modular_overflow_is_applied_to_every_output() -> None:
    workload = MatmulWorkload(((15,),), (((15,),), ((14,),)), width=4)
    compiled = compile_workload(workload)

    assert expected_matmul_outputs(workload) == (1, 2)
    assert outputs_of(compiled, workload) == (1, 2)
    assert (compiled.circuit.input_count, compiled.circuit.weight_count) == (2, 1)


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
