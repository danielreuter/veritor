"""A range of one value is a ``Wire``; ``Tracer.check`` marks check outputs."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from veritor.compile import CompileError, Compiler
from veritor.constructors import Tracer, TracerError
from veritor.constructors.tracer import Wire, Wires
from veritor.core import make_word_gate_set

GATES = make_word_gate_set(8)


def test_a_count_one_range_is_a_wire_however_it_is_made() -> None:
    tracer = Tracer(GATES)
    add = tracer.gate("add")
    seen: dict[str, object] = {}

    @tracer.definition(input_count=3, key="body", role="verification")
    def body(v: Wires):
        seen["gate"] = add(v[0], v[1])
        seen["slice"] = v[2:3]
        seen["strided"] = v[0:3:2][1:2]
        seen["index"] = v[2]
        seen["constructed"] = Wires(v.trace, v.space, 2, 1, 7)
        seen["sources"] = tracer.inputs(1)
        seen["many"] = tracer.inputs(2)
        seen["shifted"] = v[1].by(3)
        seen["shifted_range"] = v[1:2].by(3)
        return [seen["gate"], seen["sources"]]

    for name in (
        "gate",
        "slice",
        "strided",
        "index",
        "constructed",
        "sources",
        "shifted",
        "shifted_range",
    ):
        assert type(seen[name]) is Wire, name
    assert type(seen["many"]) is Wires and len(seen["many"]) == 2
    # equal ranges are equal objects: the stride of a one-value range is 0
    assert (
        seen["slice"]
        == seen["index"]
        == seen["constructed"]
        == Wire(seen["index"].trace, "input", 2)
    )
    assert seen["strided"] == seen["index"] and seen["constructed"].stride == 0
    assert seen["shifted"] == seen["shifted_range"] and seen["shifted"].jstride == 3
    assert seen["gate"].index == seen["gate"].start and len(seen["gate"]) == 1
    assert list(seen["gate"]) == [seen["gate"]] and seen["gate"][0] == seen["gate"]
    assert replace(seen["gate"], jstride=2) == seen["gate"].by(2)
    with pytest.raises(TracerError, match="one value"):
        Wire(seen["gate"].trace, "local", 0, 2)
    with pytest.raises(IndexError):
        seen["gate"][1]


def test_one_output_calls_and_one_gate_repeats_are_wires() -> None:
    tracer = Tracer(GATES)
    add, mul = tracer.gate("add"), tracer.gate("mul")
    double = tracer.definition(input_count=1, key="double", role="verification")(
        lambda v: add(v[0], v[0])
    )
    pair = tracer.definition(input_count=1, key="pair", role="verification")(
        lambda v: [add(v[0], v[0]), mul(v[0], v[0])]
    )
    seen: dict[str, object] = {}

    @tracer.definition(input_count=0, key="unit", role="replay")
    def unit(_v: Wires):
        x = tracer.inputs(1)
        seen["call"] = double(x)
        seen["repeat"] = tracer.repeat(1, double, x)
        seen["repeated"] = tracer.repeat(3, double, x)
        seen["pair"] = pair(x)
        return [seen["call"], seen["repeat"], seen["repeated"], seen["pair"]]

    root = tracer.definition(input_count=0, key="root")(lambda _v: unit())
    assert type(seen["call"]) is Wire and type(seen["repeat"]) is Wire
    assert type(seen["repeated"]) is Wires and len(seen["repeated"]) == 3
    assert type(seen["pair"]) is Wires and len(seen["pair"]) == 2
    compiled = Compiler(GATES).compile(tracer.serialize(root), [5])
    tape = compiled.circuit.evaluate([5])
    assert [tape[o] for o in compiled.circuit.outputs] == [10, 10, 10, 10, 10, 10, 25]


def test_iterating_and_indexing_keep_the_per_copy_shift() -> None:
    tracer = Tracer(GATES)
    add = tracer.gate("add")
    cell = tracer.definition(input_count=2, key="cell", role="verification")(
        lambda v: add(v[0], v[1])
    )

    @tracer.definition(input_count=0, key="unit", role="replay")
    def unit(_v: Wires):
        x = tracer.inputs(4)
        shifted = x.by(1)
        assert [w.jstride for w in shifted] == [1, 1, 1, 1] and shifted[2].jstride == 1
        return tracer.repeat(2, cell, shifted[0], shifted[2])

    root = tracer.definition(input_count=0, key="root")(lambda _v: unit())
    compiled = Compiler(GATES).compile(tracer.serialize(root), [1, 2, 3, 4])
    tape = compiled.circuit.evaluate([1, 2, 3, 4])
    assert [tape[o] for o in compiled.circuit.outputs] == [1 + 3, 2 + 4]


def test_check_marks_become_the_roots_checks() -> None:
    tracer = Tracer(GATES)
    add, mul = tracer.gate("add"), tracer.gate("mul")
    pair = tracer.definition(input_count=1, key="pair", role="verification")(
        lambda v: [add(v[0], v[0]), mul(v[0], v[0])]
    )

    double = tracer.definition(input_count=1, key="double", role="verification")(
        lambda v: add(v[0], v[0])
    )

    @tracer.definition(input_count=0, key="unit", role="replay")
    def unit(_v: Wires):
        x = tracer.inputs(1)
        return [pair(x), double(x), tracer.repeat(3, pair, x)]

    @tracer.definition(input_count=0, key="root")
    def root(_v: Wires):
        outputs = unit()
        p, ok, block = outputs[:2], outputs[2], outputs[3:]
        tracer.check(ok, 10)
        tracer.check(
            block[1::2], 25
        )  # the squares of the three copies: ordinals 4, 6, 8
        tracer.check(p[1], 25)  # ordinal 1
        return [p, ok, block]

    payload = tracer.serialize(root)
    document = json.loads(payload)
    body = next(
        entry["body"]
        for entry in document["definitions"]
        if entry["digest"] == document["root"]
    )
    assert body["checks"] == [[2, 1, 0, 10], [4, 3, 2, 25], [1, 1, 0, 25]]
    compiled = Compiler(GATES).compile(payload, [5])
    assert [
        (check.start, check.count, check.stride, check.value)
        for check in compiled.checks
    ] == [
        (2, 1, 0, 10),
        (4, 3, 2, 25),
        (1, 1, 0, 25),
    ]
    assert sorted(compiled.check_values()) == [
        (1, 25),
        (2, 10),
        (4, 25),
        (6, 25),
        (8, 25),
    ]
    root_definition = compiled.index.root.frame.definition
    assert (
        root_definition.checked_bits == 5 * 8
        and root_definition.out_bits == 9 * 8 - 5 * 8
    )
    tape = compiled.circuit.evaluate([5])
    assert all(
        tape[compiled.circuit.outputs[ordinal]] == value
        for ordinal, value in compiled.check_values()
    )


def test_check_marks_are_validated() -> None:
    tracer = Tracer(GATES)
    add = tracer.gate("add")

    with pytest.raises(TracerError, match="declared outputs"):

        @tracer.definition(input_count=0, key="unreturned")
        def unreturned(_v: Wires):
            x = tracer.inputs(1)
            tracer.check(add(x, x), 1)
            return add(x, x)

    with pytest.raises(TracerError, match="shift per copy"):

        @tracer.definition(input_count=0, key="shifted")
        def shifted(_v: Wires):
            x = tracer.inputs(2)
            return tracer.check(x.by(1), 0)

    with pytest.raises(TracerError, match="nonnegative integer"):

        @tracer.definition(input_count=0, key="negative")
        def negative(_v: Wires):
            x = tracer.inputs(1)
            return tracer.check(add(x, x), -1)

    other = Tracer(GATES)
    with pytest.raises(TracerError, match="current trace"):

        @tracer.definition(input_count=0, key="foreign")
        def foreign(_v: Wires):
            x = tracer.inputs(1)
            with pytest.raises(TracerError):
                other.check(x, 0)
            return tracer.check(Wire(object(), "local", 0), 0)

    # checks below the root are the compiler's to reject
    double = tracer.definition(input_count=1, key="double", role="verification")(
        lambda v: add(v[0], v[0])
    )

    @tracer.definition(input_count=0, key="inner", role="replay")
    def inner(_v: Wires):
        return tracer.check(double(tracer.inputs(1)), 0)

    root = tracer.definition(input_count=0, key="root")(lambda _v: inner())

    with pytest.raises(CompileError, match="only the root may"):
        Compiler(GATES).compile(tracer.serialize(root), [1])
