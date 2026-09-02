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
from veritor.core.description import CallStep, Definition, GateStep, Range


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


@pytest.fixture
def helpers() -> ModuleType:
    """The helper functions of this module, as a namespace."""

    return sys.modules[__name__]
