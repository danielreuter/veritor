"""Parse and validate a serialized description into :class:`Definition` objects.

The wire format is canonical JSON (sorted keys, no whitespace, no floats)::

    {"version": 2,
     "definitions": [{"digest": "<hex>", "body": {...}}, ...],   # dependency order
     "root": "<hex>"}

    body  = {"input_count": n, "role": null | "replay" | "verification",
             "steps": [step, ...], "outputs": [range, ...]}
    step  = {"kind": "gate", "gate": name, "args": [range, ...]}
          | {"kind": "call", "digest": "<hex>", "args": [range, ...]}
          | {"kind": "repeat", "count": n, "digest": "<hex>", "args": [jrange, ...]}
    range = [space, start, count, stride]           space = "input" | "local"
    jrange = [space, start, count, stride, jstride]

Every check here is per definition, so validation is ``O(|G|)`` regardless of
how many gates the description unrolls to.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from veritor.core import (
    CompilationLimits,
    GateSet,
    InvalidArtifact,
    JSONValue,
    canonical_json_bytes,
    identity_digest,
    tagged_sha256,
)
from veritor.core.description import (
    INPUT,
    LOCAL,
    ROLES,
    CallStep,
    Definition,
    GateStep,
    Range,
    Step,
    ranges_total,
)

FORMAT_VERSION = 2
DEFINITION_DIGEST_TAG = "veritor/definition/v2"
DESCRIPTION_DIGEST_TAG = "veritor/description/v2"


class CompileError(InvalidArtifact):
    """The compiler rejected a description."""


def canonical_description(document: JSONValue) -> bytes:
    """Serialize a description document in its canonical byte form."""

    return canonical_json_bytes(document)


def definition_digest(body: JSONValue) -> str:
    """The hash-consing identity of one definition body."""

    return identity_digest(DEFINITION_DIGEST_TAG, body)


def description_digest(payload: bytes) -> str:
    """The identity of a canonical description: ``H(C, I)`` up to the gate set."""

    return tagged_sha256(DESCRIPTION_DIGEST_TAG, payload)


@dataclass(frozen=True, slots=True)
class Description:
    """A parsed, validated description: its digest, definitions and root."""

    digest: str
    definitions: tuple[Definition, ...]
    root: Definition


# -- strict JSON -------------------------------------------------------------


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CompileError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise CompileError(f"non-finite JSON number {value!r} is forbidden")


def _object(value: object, keys: set[str], where: str) -> dict[str, object]:
    if type(value) is not dict:
        raise CompileError(f"{where} must be an object")
    if set(value) != keys:
        raise CompileError(f"{where} has keys {sorted(value)}; expected {sorted(keys)}")
    return value


def _list(value: object, where: str) -> list[object]:
    if type(value) is not list:
        raise CompileError(f"{where} must be a list")
    return value


def _int(value: object, where: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise CompileError(f"{where} must be an integer >= {minimum}")
    return value


def _digest(value: object, where: str) -> str:
    if type(value) is not str or len(value) != 64:
        raise CompileError(f"{where} must be a 64-character SHA-256 digest")
    if value.lower() != value or any(c not in "0123456789abcdef" for c in value):
        raise CompileError(f"{where} is not lowercase hexadecimal")
    return value


# -- grammar -----------------------------------------------------------------


def _range(
    value: object,
    where: str,
    *,
    input_count: int,
    slots: int,
    copies: int | None,
) -> Range:
    """Parse ``[space, start, count, stride(, jstride)]`` and bound-check it."""

    items = _list(value, where)
    width = 4 if copies is None else 5
    if len(items) != width:
        raise CompileError(f"{where} must have {width} elements")
    space = items[0]
    if space not in (INPUT, LOCAL):
        raise CompileError(f"{where} space must be {INPUT!r} or {LOCAL!r}")
    start = _int(items[1], f"{where} start")
    count = _int(items[2], f"{where} count", minimum=1)
    stride = _int(items[3], f"{where} stride")
    jstride = 0 if copies is None else _int(items[4], f"{where} jstride")
    item = Range(str(space), start, count, stride, jstride)
    bound = input_count if space == INPUT else slots
    if item.last(copies or 1) >= bound:
        raise CompileError(
            f"{where} reaches {space} coordinate {item.last(copies or 1)}; "
            f"only {bound} are available"
        )
    return item


def _ranges(
    value: object,
    where: str,
    *,
    input_count: int,
    slots: int,
    copies: int | None = None,
) -> tuple[Range, ...]:
    return tuple(
        _range(
            item,
            f"{where}[{index}]",
            input_count=input_count,
            slots=slots,
            copies=copies,
        )
        for index, item in enumerate(_list(value, where))
    )


def _step(
    value: object,
    where: str,
    *,
    input_count: int,
    slots: int,
    gate_set: GateSet,
    available: dict[str, Definition],
) -> Step:
    if type(value) is not dict or "kind" not in value:
        raise CompileError(f"{where} must be an object with a kind")
    kind = value["kind"]
    if kind == "gate":
        raw = _object(value, {"kind", "gate", "args"}, where)
        name = raw["gate"]
        if type(name) is not str or name not in gate_set:
            raise CompileError(f"{where} uses unknown gate {name!r}")
        gate = gate_set[name]
        args = _ranges(
            raw["args"], f"{where} args", input_count=input_count, slots=slots
        )
        if ranges_total(args) != gate.arity:
            raise CompileError(
                f"{where} passes {ranges_total(args)} arguments to {name}, "
                f"which has arity {gate.arity}"
            )
        return GateStep(gate, args)
    if kind == "call":
        raw = _object(value, {"kind", "digest", "args"}, where)
        count = 1
    elif kind == "repeat":
        raw = _object(value, {"kind", "count", "digest", "args"}, where)
        count = _int(raw["count"], f"{where} count", minimum=1)
    else:
        raise CompileError(f"{where} has unknown kind {kind!r}")
    child = available.get(_digest(raw["digest"], f"{where} digest"))
    if child is None:
        raise CompileError(f"{where} calls a definition that is not defined earlier")
    args = _ranges(
        raw["args"],
        f"{where} args",
        input_count=input_count,
        slots=slots,
        copies=None if kind == "call" else count,
    )
    if ranges_total(args) != child.input_count:
        raise CompileError(
            f"{where} passes {ranges_total(args)} arguments to a definition "
            f"with {child.input_count} inputs"
        )
    return CallStep.make(child, args, count)


def _definition(
    digest: str,
    body: object,
    *,
    gate_set: GateSet,
    limits: CompilationLimits,
    available: dict[str, Definition],
) -> Definition:
    where = f"definition {digest[:12]}"
    raw = _object(body, {"input_count", "role", "steps", "outputs"}, where)
    input_count = _int(raw["input_count"], f"{where} input_count")
    role = raw["role"]
    if role is not None and role not in ROLES:
        raise CompileError(f"{where} role must be null or one of {list(ROLES)}")
    raw_steps = _list(raw["steps"], f"{where} steps")
    if len(raw_steps) > limits.max_steps_per_definition:
        raise CompileError(f"{where} exceeds max_steps_per_definition")
    steps: list[Step] = []
    slots = 0
    for index, item in enumerate(raw_steps):
        step = _step(
            item,
            f"{where} step {index}",
            input_count=input_count,
            slots=slots,
            gate_set=gate_set,
            available=available,
        )
        steps.append(step)
        slots += step.slots
    outputs = _ranges(
        raw["outputs"], f"{where} outputs", input_count=input_count, slots=slots
    )
    if not outputs:
        raise CompileError(f"{where} declares no outputs")
    definition = Definition(digest, input_count, tuple(steps), outputs, role)
    for label, value, limit in (
        ("gates", definition.size, limits.max_addresses),
        ("slots", definition.slot_count, limits.max_addresses),
        ("outputs", definition.output_count, limits.max_addresses),
        ("replay cost", definition.replay_cost, limits.max_cost),
        ("proof cost", definition.proof_cost, limits.max_cost),
        ("nesting depth", definition.depth, limits.max_depth),
    ):
        if value > limit:
            raise CompileError(f"{where} has {value} {label}; the limit is {limit}")
    return definition


def parse_description(
    payload: bytes,
    gate_set: GateSet,
    limits: CompilationLimits | None = None,
) -> Description:
    """Parse canonical description bytes and validate every definition."""

    limits = CompilationLimits() if limits is None else limits
    if type(payload) is not bytes:
        raise CompileError("a description must be bytes")
    if len(payload) > limits.max_description_bytes:
        raise CompileError("description exceeds max_description_bytes")
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except CompileError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as e:
        raise CompileError("description is not valid UTF-8 JSON") from e
    try:
        canonical = canonical_description(value)
    except (TypeError, ValueError, RecursionError) as error:
        raise CompileError("description is not canonical JSON data") from error
    if canonical != payload:
        raise CompileError("description is not canonically serialized")

    document = _object(value, {"version", "definitions", "root"}, "description")
    if _int(document["version"], "version") != FORMAT_VERSION:
        raise CompileError(f"unsupported description format {document['version']}")
    entries = _list(document["definitions"], "definitions")
    if len(entries) > limits.max_definitions:
        raise CompileError("description exceeds max_definitions")
    available: dict[str, Definition] = {}
    for index, item in enumerate(entries):
        entry = _object(item, {"digest", "body"}, f"definitions[{index}]")
        digest = _digest(entry["digest"], f"definitions[{index}] digest")
        if digest in available:
            raise CompileError(f"definition {digest[:12]} appears twice")
        if definition_digest(entry["body"]) != digest:  # type: ignore[arg-type]
            raise CompileError(f"definition {digest[:12]} does not match its digest")
        available[digest] = _definition(
            digest, entry["body"], gate_set=gate_set, limits=limits, available=available
        )
    root = available.get(_digest(document["root"], "root"))
    if root is None:
        raise CompileError("root names a definition that is not defined")
    if root.input_count + root.size > limits.max_addresses:
        raise CompileError("the circuit exceeds max_addresses")
    return Description(description_digest(payload), tuple(available.values()), root)
