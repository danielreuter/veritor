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
from veritor.core import FlatCircuit, GateRef, GateSet
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


def source_cell(doc: Document, gate_name: str) -> str:
    """A verification unit of one source gate (``"in"`` or ``"weight"``)."""

    return doc.add(body(0, [gate(gate_name)], [rng("local", 0)], role="verification"))


def wrap(doc: Document, target: str, input_count: int, output_count: int) -> str:
    """A root without ports around ``target``, a definition with ``input_count`` ports.

    Addresses ``[0, input_count)`` hold an input block (a replay unit of
    one-gate ``in`` cells) feeding ``target``, whose gates therefore keep the
    addresses they would have had as a ported root; the root declares
    ``target``'s ``output_count`` outputs.
    """

    in_cell = source_cell(doc, "in")
    stride = 1 if input_count > 1 else 0
    block = doc.add(
        body(
            0,
            [repeat(input_count, in_cell)],
            [rng("local", 0, input_count, stride)],
            role="replay",
        )
    )
    return doc.add(
        body(
            0,
            [call(block), call(target, rng("local", 0, input_count, stride))],
            [rng("local", input_count, output_count, 1 if output_count > 1 else 0)],
        )
    )


def wrapped(doc: Document, definition: dict[str, object]) -> bytes:
    """Serialize ``definition`` (which may have ports) under a :func:`wrap` root."""

    outputs = definition["outputs"]
    assert isinstance(outputs, list)
    output_count = sum(int(item[2]) for item in outputs)  # type: ignore[index]
    input_count = definition["input_count"]
    assert isinstance(input_count, int)
    return doc.serialize(wrap(doc, doc.add(definition), input_count, output_count))


def expand(root: Definition) -> tuple[list[tuple[str, tuple[int, ...]]], list[int]]:
    """Unroll ``root`` (a definition without ports) into ``(gates, outputs)``.

    Gates are ``(name, args)`` at their addresses, source gates with no args;
    this is the reference the lazy descent is tested against.
    """

    assert root.input_count == 0
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
                slots.append(len(gates) - 1)
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

    outputs = emit(root, [])
    return gates, outputs


def flatten(root: Definition, gate_set: GateSet) -> FlatCircuit:
    """The reference :class:`FlatCircuit` of a description's expansion."""

    width = next(iter(gate_set)).width
    gates, outputs = expand(root)
    refs = [GateRef(name, args, width) for name, args in gates]
    return FlatCircuit(refs, outputs, gate_set)


def evaluate(root: Definition, inputs, weights=()) -> tuple[int, ...]:
    """Reference evaluation of the expansion: the root's declared outputs."""

    frame = Frame.root(root)
    given = {"input": iter(inputs), "weight": iter(weights)}
    values: list[int] = []
    for address in range(root.size):
        gate_, args = frame.gate(address)
        if gate_.source is not None:
            values.append(next(given[gate_.source]))
        else:
            values.append(gate_.evaluate(tuple(values[a] for a in args)))
    for source, rest in given.items():
        assert next(rest, None) is None, f"unused {source} values"
    return tuple(values[frame.output_address(k)] for k in range(root.output_count))


def frames(frame: Frame):
    """Every copy below ``frame`` (inclusive), in layout order."""

    yield frame
    for index, step in enumerate(frame.definition.steps):
        if isinstance(step, CallStep):
            for copy in range(step.count):
                yield from frames(frame.child(index, copy))


def matmul_payload(k: int, cols: int, rows: int = 1) -> bytes:
    """``rows`` copies of ``x_i W`` for a ``k x cols`` matrix ``W`` (``k`` a power of two).

    The activations are a replay unit of ``rows * k`` one-gate ``in`` cells at
    ``[0, rows * k)`` and the weights a replay unit of ``k * cols`` one-gate
    ``weight`` cells after them; both declare all their gates (pinned, so
    their ``Out`` is empty).  Each row is a replay unit of ``cols`` dot
    products (verification units) reading its activations and ``W`` through
    ports from the two source units' slots; rows are laid out after the
    weights.  The root has no ports.  The description has one repeat per
    reduction level, so its size is ``O(log k)`` and independent of ``rows``
    and ``cols``, and because the rows hold nothing but dots, the root's
    ``rows * cols`` outputs are one run.
    """

    if k & (k - 1) or k < 2:
        raise ValueError("k must be a power of two >= 2")
    doc = Document()
    in_cell = source_cell(doc, "in")
    weight_cell = source_cell(doc, "weight")
    mul = doc.add(body(2, [gate("mul", rng("input", 0, 2, 1))], [rng("local", 0)]))
    add = doc.add(body(2, [gate("add", rng("input", 0, 2, 1))], [rng("local", 0)]))
    steps = [repeat(k, mul, jrng("input", 0, 1, 0, 1), jrng("input", k, 1, 0, 1))]
    start, width = 0, k
    while width > 1:
        steps.append(repeat(width // 2, add, jrng("local", start, 2, 1, 2)))
        start, width = start + width, width // 2
    dot = doc.add(body(2 * k, steps, [rng("local", start)], role="verification"))
    activations = doc.add(
        body(
            0,
            [repeat(rows * k, in_cell)],
            [rng("local", 0, rows * k, 1)],
            role="replay",
        )
    )
    weights = doc.add(
        body(
            0,
            [repeat(k * cols, weight_cell)],
            [rng("local", 0, k * cols, 1)],
            role="replay",
        )
    )
    row = doc.add(
        body(
            k + k * cols,
            [
                repeat(
                    cols, dot, jrng("input", 0, k, 1, 0), jrng("input", k, k, cols, 1)
                )
            ],
            [rng("local", 0, cols, 1 if cols > 1 else 0)],
            role="replay",
        )
    )
    root = doc.add(
        body(
            0,
            [
                call(activations),
                call(weights),
                repeat(
                    rows,
                    row,
                    jrng("local", 0, k, 1, k),
                    jrng("local", rows * k, k * cols, 1, 0),
                ),
            ],
            [
                rng(
                    "local",
                    rows * k + k * cols,
                    rows * cols,
                    1 if rows * cols > 1 else 0,
                )
            ],
        )
    )
    return doc.serialize(root)


def matmul_layout(k: int, cols: int, rows: int) -> dict[str, object]:
    """Address arithmetic of :func:`matmul_payload`: unit intervals and source addresses."""

    dot_size = 2 * k - 1
    row_size = cols * dot_size
    rows_start = rows * k + k * cols
    row_ranges = [
        range(rows_start + r * row_size, rows_start + (r + 1) * row_size)
        for r in range(rows)
    ]
    return {
        "dot_size": dot_size,
        "row_size": row_size,
        "n": rows_start + rows * row_size,
        "inputs": range(rows * k),
        "weights": range(rows * k, rows * k + k * cols),
        "rows": row_ranges,
        "dots": [
            range(interval.start + c * dot_size, interval.start + (c + 1) * dot_size)
            for interval in row_ranges
            for c in range(cols)
        ],
    }


def shared_kinds_payload() -> bytes:
    """Two replay kinds reaching one verification kind through different paths.

    Replay kind ``A`` (two copies) calls an unmarked ``middle`` twice, each
    holding two copies of ``V1``; replay kind ``B`` (one copy) calls ``V1``
    once and ``V2`` twice.  Exercises sharing, an unmarked layer between the
    cuts and a kind at two depths.  The two inputs are an input block under a
    :func:`wrap` root.
    """

    IN, LOC = "input", "local"
    doc = Document()
    v1 = doc.add(
        body(
            2,
            [gate("mul", rng(IN, 0, 2, 1)), gate("add", rng(LOC, 0), rng(IN, 1))],
            [rng(LOC, 1)],
            role="verification",
        )
    )
    v2 = doc.add(
        body(2, [gate("add", rng(IN, 0, 2, 1))], [rng(LOC, 0)], role="verification")
    )
    middle = doc.add(body(2, [repeat(2, v1, jrng(IN, 0, 2, 1))], [rng(LOC, 0, 2, 1)]))
    a = doc.add(
        body(
            2,
            [call(middle, rng(IN, 0, 2, 1)), call(middle, rng(LOC, 0, 2, 1))],
            [rng(LOC, 2, 2, 1)],
            role="replay",
        )
    )
    b = doc.add(
        body(
            2,
            [call(v1, rng(IN, 0, 2, 1)), repeat(2, v2, jrng(IN, 0), jrng(LOC, 0))],
            [rng(LOC, 1, 2, 1)],
            role="replay",
        )
    )
    target = doc.add(
        body(
            2,
            [repeat(2, a, jrng(IN, 0, 2, 1)), call(b, rng(LOC, 0, 2, 1))],
            [rng(LOC, 4, 2, 1)],
        )
    )
    return doc.serialize(wrap(doc, target, 2, 2))


@pytest.fixture
def helpers() -> ModuleType:
    """The helper functions of this module, as a namespace."""

    return sys.modules[__name__]
