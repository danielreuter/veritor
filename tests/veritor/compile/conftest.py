"""Hand-built descriptions and an O(n) reference expander for tests."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

from veritor.compile.description import (
    FORMAT_VERSION,
    canonical_description,
    definition_digest,
)
from veritor.core import INPUT_OP, FlatCircuit, GateRef, GateSet
from veritor.core.description import CallStep, Definition, Frame, GateStep, Range


def rng(space: str, start: int, count: int = 1, stride: int = 0) -> list[object]:
    return [space, start, count, stride]


def jrng(
    space: str, start: int, count: int = 1, stride: int = 0, jstride: int = 0
) -> list[object]:
    return [space, start, count, stride, jstride]


def gate(name: str, *args: list[object]) -> dict[str, object]:
    return {"kind": "gate", "gate": name, "args": list(args)}


def call(digest: str, *args: list[object]) -> dict[str, object]:
    return {"kind": "call", "digest": digest, "args": list(args)}


def repeat(count: int, digest: str, *args: list[object]) -> dict[str, object]:
    return {"kind": "repeat", "count": count, "digest": digest, "args": list(args)}


def body(
    input_count: int,
    steps: list[dict[str, object]],
    outputs: list[list[object]],
    role: str | None = None,
) -> dict[str, object]:
    return {
        "input_count": input_count,
        "role": role,
        "steps": steps,
        "outputs": outputs,
    }


class Document:
    """Accumulates definitions in dependency order and serializes them."""

    def __init__(self) -> None:
        self.entries: list[dict[str, object]] = []

    def add(self, definition: dict[str, object]) -> str:
        digest = definition_digest(definition)
        self.entries.append({"digest": digest, "body": definition})
        return digest

    def value(self, root: str) -> dict[str, object]:
        return {"version": FORMAT_VERSION, "definitions": self.entries, "root": root}

    def serialize(self, root: str) -> bytes:
        return canonical_description(self.value(root))


def single(definition: dict[str, object]) -> bytes:
    document = Document()
    return document.serialize(document.add(definition))


def expand(root: Definition) -> tuple[list[tuple[str, tuple[int, ...]]], list[int]]:
    """Unroll ``root`` into ``(gates, outputs)`` with absolute addresses.

    Gates are ``(name, args)`` at addresses ``root.input_count + index``; this
    is the reference the lazy descent is tested against.
    """

    gates: list[tuple[str, tuple[int, ...]]] = []

    def resolve(
        item: Range, element: int, copy: int, inputs: list[int], slots: list[int]
    ) -> int:
        value = item.element(element, copy)
        return slots[value] if item.space == "local" else inputs[value]

    def emit(definition: Definition, inputs: list[int]) -> list[int]:
        slots: list[int] = []
        for step in definition.steps:
            if isinstance(step, GateStep):
                args = tuple(
                    resolve(item, k, 0, inputs, slots)
                    for item in step.args
                    for k in range(item.count)
                )
                gates.append((step.gate.name, args))
                slots.append(root.input_count + len(gates) - 1)
                continue
            assert isinstance(step, CallStep)
            for copy in range(step.count):
                child_inputs = [
                    resolve(item, k, copy, inputs, slots)
                    for item in step.args
                    for k in range(item.count)
                ]
                slots.extend(emit(step.child, child_inputs))
        return [
            resolve(item, k, 0, inputs, slots)
            for item in definition.outputs
            for k in range(item.count)
        ]

    outputs = emit(root, list(range(root.input_count)))
    return gates, outputs


def flatten(root: Definition, gate_set: GateSet) -> FlatCircuit:
    """The reference :class:`FlatCircuit` of a description's expansion."""

    width = next(iter(gate_set)).width
    gates, outputs = expand(root)
    refs = [GateRef(INPUT_OP, (), width) for _ in range(root.input_count)]
    refs.extend(GateRef(name, args, width) for name, args in gates)
    return FlatCircuit(refs, outputs, gate_set)


def frames(frame: Frame):
    """Every copy below ``frame`` (inclusive), in layout order."""

    yield frame
    for index, step in enumerate(frame.definition.steps):
        if isinstance(step, CallStep):
            for copy in range(step.count):
                yield from frames(frame.child(index, copy))


def matmul_payload(k: int, cols: int, rows: int = 1) -> bytes:
    """``rows`` copies of ``x_i W`` for a ``k x cols`` matrix ``W`` (``k`` a power of two).

    Each dot product is a verification unit and each row is a replay unit; the
    description has one repeat per reduction level, so its size is
    ``O(log k)`` and independent of ``rows`` and ``cols``.
    """

    if k & (k - 1) or k < 2:
        raise ValueError("k must be a power of two >= 2")
    doc = Document()
    mul = doc.add(body(2, [gate("mul", rng("input", 0, 2, 1))], [rng("local", 0)]))
    add = doc.add(body(2, [gate("add", rng("input", 0, 2, 1))], [rng("local", 0)]))
    steps = [repeat(k, mul, jrng("input", 0, 1, 0, 1), jrng("input", k, 1, 0, 1))]
    start, width = 0, k
    while width > 1:
        steps.append(repeat(width // 2, add, jrng("local", start, 2, 1, 2)))
        start, width = start + width, width // 2
    dot = doc.add(body(2 * k, steps, [rng("local", start)], role="verification"))
    row = doc.add(
        body(
            k + k * cols,
            [repeat(cols, dot, jrng("input", 0, k, 1, 0), jrng("input", k, k, cols, 1))],
            [rng("local", 0, cols, 1)],
            role="replay",
        )
    )
    root = doc.add(
        body(
            rows * k + k * cols,
            [
                repeat(
                    rows,
                    row,
                    jrng("input", 0, k, 1, k),
                    jrng("input", rows * k, k * cols, 1, 0),
                )
            ],
            [rng("local", 0, rows * cols, 1)],
        )
    )
    return doc.serialize(root)


@pytest.fixture
def helpers() -> ModuleType:
    """The helper functions of this module, as a namespace."""

    return sys.modules[__name__]
