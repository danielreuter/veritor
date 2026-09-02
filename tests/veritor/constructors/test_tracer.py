"""``Tracer.inputs`` and ``Tracer.weights``: source gates as one-gate verification cells."""

from __future__ import annotations

import json

import pytest

from veritor.compile import CompileError, Compiler
from veritor.compile.description import parse_description
from veritor.constructors import Tracer, TracerError
from veritor.core import Gate, GateSet, InvalidArtifact, make_word_gate_set

GATES = make_word_gate_set(8)


def test_source_gates_are_one_repeat_of_a_canonical_verification_cell() -> None:
    tracer = Tracer(GATES)
    add = tracer.gate("add")
    pair = tracer.definition(input_count=2, key="pair", role="verification")(lambda v: add(v[0], v[1]))

    @tracer.definition(input_count=0, key="layer", role="replay")
    def layer(_v):
        x = tracer.inputs(1000)
        w = tracer.weights(3)
        one = tracer.inputs(1)
        return [pair(x[0], w[0]), pair(one, w[2])]

    payload = tracer.serialize(layer)
    document = json.loads(payload)
    bodies = {entry["digest"]: entry["body"] for entry in document["definitions"]}
    root = bodies[document["root"]]
    in_cell, weight_cell = tracer.source_cell("input"), tracer.source_cell("weight")

    assert [step["kind"] for step in root["steps"]] == ["repeat", "repeat", "call", "call", "call"]
    assert root["steps"][0] == {"kind": "repeat", "count": 1000, "digest": in_cell.digest, "args": []}
    assert root["steps"][1] == {"kind": "repeat", "count": 3, "digest": weight_cell.digest, "args": []}
    assert root["steps"][2] == {"kind": "call", "digest": in_cell.digest, "args": []}
    for cell, gate in ((in_cell, "in"), (weight_cell, "weight")):
        assert bodies[cell.digest] == {
            "input_count": 0,
            "role": "verification",
            "steps": [{"kind": "gate", "gate": gate, "args": []}],
            "outputs": [["local", 0, 1, 0]],
        }
    # the cells are hash-consed: another tracer over the same gate set produces the same digests
    assert Tracer(GATES).source_cell("weight").digest == weight_cell.digest
    assert len(payload) < 1600  # a thousand inputs is one repeat step: four definitions

    inputs = [i & 255 for i in range(1001)]
    compiled = Compiler(GATES).compile(payload, inputs)
    index = compiled.index
    assert (index.input_count, index.weight_count) == (1001, 3)
    assert list(index.inputs()) == list(range(1000)) + [1003]
    assert list(index.weights()) == [1000, 1001, 1002]
    assert index.verification_unit_count == 1001 + 3 + 2
    tape = compiled.circuit.evaluate(inputs, (10, 20, 30))
    assert [tape[o] for o in compiled.circuit.outputs] == [10, ((1000 & 255) + 30) & 255]
    with pytest.raises(InvalidArtifact, match="weight value is not a 8-bit value"):
        compiled.circuit.evaluate(inputs, (10, 20, 300))


def test_source_gates_outside_a_replay_unit_or_inside_a_verification_unit_are_rejected() -> None:
    tracer = Tracer(GATES)
    add = tracer.gate("add")

    @tracer.definition(input_count=0, key="bare")
    def bare(_v):
        x = tracer.inputs(2)
        return add(x[0], x[1])

    with pytest.raises(CompileError, match="outside any replay unit"):
        Compiler(GATES).compile(tracer.serialize(bare), (1, 2))

    @tracer.definition(input_count=0, key="nested", role="verification")
    def nested(_v):
        x = tracer.inputs(2)
        return add(x[0], x[1])

    @tracer.definition(input_count=0, key="outer", role="replay")
    def outer(_v):
        return nested()

    with pytest.raises(CompileError, match="marked verification and contains a verification mark"):
        Compiler(GATES).compile(tracer.serialize(outer), (1, 2))

    # a wider verification unit around its inputs names the gate directly
    in_gate = tracer.gate("in")

    @tracer.definition(input_count=0, key="wide", role="verification")
    def wide(_v):
        return add(in_gate(), in_gate())

    @tracer.definition(input_count=0, key="wide-root", role="replay")
    def wide_root(_v):
        return wide()

    compiled = Compiler(GATES).compile(tracer.serialize(wide_root), (3, 4))
    assert compiled.index.verification_unit_count == 1 and compiled.index.input_count == 2
    assert compiled.circuit.evaluate((3, 4))[-1] == 7


def test_source_requests_are_validated() -> None:
    tracer = Tracer(GATES)
    with pytest.raises(TracerError, match="positive integer"):
        tracer.definition(input_count=0, key="zero", role="replay")(lambda _v: tracer.inputs(0))
    with pytest.raises(TracerError, match="source must be one of"):
        tracer.source_cell("advice")
    with pytest.raises(TracerError, match="only be used while tracing"):
        tracer.inputs(1)

    plain = GateSet(
        (Gate("add", 2, 8, replay_cost=1, proof_cost=1, evaluate=lambda a: (a[0] + a[1]) & 255),),
        name="tests.plain",
        version="1",
    )
    plain_tracer = Tracer(plain)
    with pytest.raises(TracerError, match="has no input gate"):
        plain_tracer.source_cell("input")
    with pytest.raises(TracerError, match="has no weight gate"):
        plain_tracer.definition(input_count=0, key="w", role="replay")(lambda _v: plain_tracer.weights(2))


def test_the_canonical_cells_parse_as_pinned_verification_units() -> None:
    tracer = Tracer(GATES)

    @tracer.definition(input_count=0, key="w", role="replay")
    def unit(_v):
        return tracer.weights(4)

    parsed = parse_description(tracer.serialize(unit), GATES)
    assert parsed.root.role == "replay" and parsed.root.out_runs == ()  # all outputs pinned
    assert parsed.root.weight_total == 4 and parsed.root.input_total == 0
    cell = parsed.root.steps[0].child
    assert cell.role == "verification" and cell.size == 1 and cell.weight_total == 1
