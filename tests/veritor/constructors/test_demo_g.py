from __future__ import annotations

import pytest

from veritor.constructors import (
    BatchInput,
    DemoG,
    DemoGCompileRequest,
    DotRequest,
    TracerError,
    compile_demo_g,
    expected_dot_outputs,
    make_demo_request,
)
from veritor.core import Compiled, DescriptionCircuit


def test_demo_g_compiles_to_a_compiled_circuit() -> None:
    request = DemoGCompileRequest()
    compiled = compile_demo_g(request)

    assert isinstance(compiled, Compiled)
    assert isinstance(compiled.circuit, DescriptionCircuit)
    assert compiled.circuit.input_count == len(request.public_inputs)
    outputs = compiled.circuit.evaluate(request.public_inputs)
    assert tuple(outputs[o] for o in compiled.circuit.outputs) == request.expected_outputs
    assert request.expected_outputs == expected_dot_outputs(request.batch, request.width)


def test_demo_g_marks_dots_as_replay_and_macs_as_verification() -> None:
    request = DemoGCompileRequest()
    index = compile_demo_g(request).index

    lengths = [dot.length for dot in request.batch.requests]
    assert index.replay_units.count == len(lengths)
    # every cell is an `in` gate in its own verification unit, then one mac per step
    assert index.verification_unit_count == sum(1 + 2 * n + n for n in lengths)
    assert index.input_count == len(request.public_inputs) == sum(1 + 2 * n for n in lengths)
    assert index.weight_count == 0 and index.root.frame.definition.input_count == 0
    for unit, length in enumerate(lengths):
        node = index.replay_units.unit(unit)
        cells = 1 + 2 * length
        assert node.size == cells + 2 * length
        assert index.verification_units(unit).count == cells + length
        sizes = [v.size for v in index.verification_units(unit)]
        assert sizes == [1] * cells + [2] * length
        # the interior is the macs' gates but the declared final add; the cells are pinned
        assert index.interior(unit).count == 2 * length - 1
    cells_in_order = [
        a
        for unit in index.replay_units
        for a in range(unit.interval.start, unit.interval.start + 1 + 2 * lengths[unit.replay_unit])
    ]
    assert list(index.inputs()) == cells_in_order


def test_demo_g_digest_binds_the_batch_shape_not_its_values() -> None:
    default = compile_demo_g()
    same_shape = compile_demo_g(
        DemoGCompileRequest(
            batch=BatchInput((make_demo_request(2, 7, 8), make_demo_request(3, 9, 8)))
        )
    )
    other_shape = compile_demo_g(
        DemoGCompileRequest(batch=BatchInput((make_demo_request(2, 7, 8),)))
    )

    assert default.digest == same_shape.digest
    assert default.digest != other_shape.digest


def _batch(lengths: tuple[int, ...]) -> BatchInput:
    return BatchInput(tuple(make_demo_request(n, i, 8) for i, n in enumerate(lengths)))


def test_demo_g_description_is_one_repeat_per_run_of_equal_lengths() -> None:
    demo = DemoG(8)
    short = demo(_batch((4,) * 3 + (2,) * 5), b"")
    long = demo(_batch((4,) * 30 + (2,) * 50), b"")

    # Only the two repeat counts change: a handful of digits, not 72 more calls.
    assert len(long) - len(short) < 16

    compiled = compile_demo_g(DemoGCompileRequest(batch=_batch((4,) * 30 + (2,) * 50)))
    assert compiled.index.replay_units.count == 80
    assert compiled.index.verification_unit_count == 30 * (9 + 4) + 50 * (5 + 2)
    assert compiled.index.input_count == 30 * 9 + 50 * 5


def test_demo_g_rejects_malformed_batches() -> None:
    demo = DemoG(8)
    with pytest.raises(TracerError, match="different lengths"):
        demo(BatchInput((DotRequest(1, (1, 2), (3,)),)), b"")
    with pytest.raises(TracerError, match="nonempty"):
        demo(BatchInput((DotRequest(1, (), ()),)), b"")
    with pytest.raises(TracerError, match="at least one"):
        demo(BatchInput(()), b"")
    with pytest.raises(TracerError, match="expects BatchInput"):
        demo(object(), b"")
    with pytest.raises(ValueError, match="advice exceeds"):
        DemoGCompileRequest(advice=b"x")
