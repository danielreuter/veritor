from __future__ import annotations

from veritor.constructors import (
    MatmulCompileRequest,
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
    compiled = compile_matmul(request)

    assert isinstance(compiled, Compiled)
    assert compiled.circuit.input_count == len(request.public_inputs)
    values = compiled.circuit.evaluate(request.public_inputs)
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
    index = compile_matmul(request).index

    rows = sum(rows for rows, _ in request.output_shapes)
    columns = request.output_shapes[0][1]
    assert index.replay_units.count == rows
    assert index.verification_unit_count == rows * columns
    assert all(index.verification_units(r).count == columns for r in range(rows))


def test_compiled_digest_binds_the_shape_and_width_but_not_the_values() -> None:
    first = compile_matmul(_request(marker=1))
    second = compile_matmul(_request(marker=2))
    wider = compile_matmul(_request(width=16))

    assert first.digest == second.digest
    assert first.digest != wider.digest
