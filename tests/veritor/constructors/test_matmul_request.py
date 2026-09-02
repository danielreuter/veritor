from __future__ import annotations

from veritor.compile import Compilation
from veritor.constructors import (
    MatmulCompileRequest,
    MatmulG,
    compile_matmul,
    matmul_expected_matrices,
)
from veritor.core import Compiled


def _request(*, marker: int = 1, width: int = 8) -> MatmulCompileRequest:
    return MatmulCompileRequest(
        (
            (marker, 2),
            (3, 4),
        ),
        (
            ((5, 6),),
            (
                (7, 8),
                (9, 10),
            ),
        ),
        width=width,
    )


def test_matmul_request_compiles_to_a_compiled_circuit() -> None:
    request = _request()
    compilation = compile_matmul(request)
    compiled = compilation.compiled

    assert isinstance(compilation, Compilation) and isinstance(compiled, Compiled)
    assert compilation.constructor == MatmulG(request.width).digest
    assert compilation.inputs == request.public_inputs and compilation.advice_bits == 0
    assert compiled.circuit.input_count == len(request.public_inputs) == 6
    assert compiled.circuit.weight_count == len(request.weight_values) == 4
    assert request.public_inputs == (5, 6, 7, 8, 9, 10) and request.weight_values == (1, 2, 3, 4)
    values = compiled.circuit.evaluate(request.public_inputs, request.weight_values)
    assert tuple(values[o] for o in compiled.circuit.outputs) == request.expected_outputs
    assert matmul_expected_matrices(request) == (
        ((23, 34),),
        (
            (31, 46),
            (39, 58),
        ),
    )


def test_matmul_marks_rows_as_replay_and_dots_as_verification() -> None:
    request = _request()
    index = compile_matmul(request).compiled.index

    rows = sum(rows for rows, _ in request.output_shapes)
    columns = request.output_shapes[0][1]
    assert index.replay_units.count == 2 + rows  # the activations, the weights, the rows
    assert index.verification_unit_count == index.input_count + index.weight_count + rows * columns
    assert index.verification_units(0).count == index.input_count == 6
    assert index.verification_units(1).count == index.weight_count == 4
    assert all(index.verification_units(r).count == columns for r in range(2, 2 + rows))


def test_compiled_digest_binds_the_shape_and_width_but_not_the_values() -> None:
    first = compile_matmul(_request(marker=1)).compiled
    second = compile_matmul(_request(marker=2)).compiled
    wider = compile_matmul(_request(width=16)).compiled

    assert first.digest == second.digest
    assert first.digest != wider.digest
