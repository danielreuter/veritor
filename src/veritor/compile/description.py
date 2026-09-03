"""Parse and validate a serialized description into :class:`Definition` objects.

The wire format is canonical JSON (sorted keys, no whitespace, no floats)::

    {"version": 3,
     "definitions": [{"digest": "<hex>", "body": {...}}, ...],   # dependency order
     "root": "<hex>"}

    body  = {"input_count": n, "role": null | "replay" | "verification",
             "steps": [step, ...], "outputs": [range, ...]
             (, "checks": [check, ...])}                # present iff nonempty
    step  = {"kind": "gate", "gate": name, "args": [range, ...]}
          | {"kind": "call", "digest": "<hex>", "args": [range, ...]}
          | {"kind": "repeat", "count": n, "digest": "<hex>", "args": [jrange, ...]}
    range = [space, start, count, stride]           space = "input" | "local"
    jrange = [space, start, count, stride, jstride]
    check = [start, count, stride, value]           over declared output ordinals

Besides the shape rules (arity, in-range relative references, dependency
order, limits), a definition's declared outputs, resolved to the gates a copy
owns (source gates included), must be pairwise distinct: the runs of ``Out``
(see :mod:`veritor.core.description`) and the pinned runs may not intersect.
This makes ``|Out|`` and its width sums over runs and rank/unrank inside
``Out`` prefix sums.  The root has no ports: ``input_count`` of the root must
be ``0``, the circuit's inputs being ``in`` gates (the compiler enforces this
when it builds the index).

``checks`` mark declared outputs the verifier requires to equal a constant
(:class:`~veritor.core.description.Check`).  Only the root may declare them;
the ordinals must lie within the declared outputs and be pairwise distinct,
each must resolve to a computed gate the copy owns (not a source gate, not a
port), and ``value`` must fit the width of every gate it resolves to.  The
key is omitted when there are no checks, so a body has one canonical form.

Every check here is per definition, so validation is ``O(|G|)`` regardless of
how many gates the description unrolls to; the distinctness check is
quadratic in the number of runs of a definition, never in the number of
outputs.  The runs of a definition's source gates (``input_runs``,
``weight_runs``) are bounded by the same per-definition limit as ``Out``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from veritor.core import (
    INPUT_SOURCE,
    WEIGHT_SOURCE,
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
    Check,
    Definition,
    GateStep,
    PieceKind,
    Range,
    Run,
    Step,
    progression_meet,
    ranges_total,
)

FORMAT_VERSION = 3
DEFINITION_DIGEST_TAG = "veritor/definition/v3"
DESCRIPTION_DIGEST_TAG = "veritor/description/v3"


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
    if type(body) is dict and "checks" in body:
        raw = _object(
            body, {"input_count", "role", "steps", "outputs", "checks"}, where
        )
        raw_checks = _list(raw["checks"], f"{where} checks")
        if not raw_checks:
            raise CompileError(
                f"{where} declares an empty checks list; the key must be omitted"
            )
    else:
        raw = _object(body, {"input_count", "role", "steps", "outputs"}, where)
        raw_checks = []
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
    checks = _checks(raw_checks, f"{where} checks", ranges_total(outputs))
    definition = Definition(digest, input_count, tuple(steps), outputs, role, checks)
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
    if definition.resolve_outputs(limits.max_output_runs) is None:
        raise CompileError(
            f"{where} resolves its declared outputs to more than "
            f"max_output_runs = {limits.max_output_runs} runs"
        )
    for source, label in ((INPUT_SOURCE, "input"), (WEIGHT_SOURCE, "weight")):
        if definition.resolve_source_runs(source, limits.max_output_runs) is None:
            raise CompileError(
                f"{where} lays out its {label} gates in more than "
                f"max_output_runs = {limits.max_output_runs} runs"
            )
    resolved_count = sum(run.count for _, run in definition.resolved_outputs)
    if resolved_count != definition.output_count:
        raise CompileError(
            f"{where} declares {definition.output_count} outputs but they resolve to "
            f"{resolved_count} positions (internal error in output resolution)"
        )
    owned = tuple(
        run for kind, run in definition.resolved_outputs if kind is not PieceKind.PORT
    )
    repeated = _repeated_output(owned)
    if repeated is not None:
        raise CompileError(
            f"{where} declares the gate at offset {repeated} as an output more than once; "
            "declared outputs must be distinct"
        )
    _validate_checks(definition, where)
    return definition


# -- check outputs -----------------------------------------------------------


def _checks(value: list[object], where: str, output_count: int) -> tuple[Check, ...]:
    """Parse ``[start, count, stride, value]`` checks over the declared output ordinals."""

    checks: list[Check] = []
    for index, item in enumerate(value):
        here = f"{where}[{index}]"
        items = _list(item, here)
        if len(items) != 4:
            raise CompileError(f"{here} must have 4 elements")
        check = Check(
            _int(items[0], f"{here} start"),
            _int(items[1], f"{here} count", minimum=1),
            _int(items[2], f"{here} stride"),
            _int(items[3], f"{here} value"),
        )
        if check.last >= output_count:
            raise CompileError(
                f"{here} names output ordinal {check.last}; only {output_count} are declared"
            )
        checks.append(check)
    repeated = _repeated_output(
        tuple(Run(check.start, check.count, check.stride, 0) for check in checks)
    )
    if repeated is not None:
        raise CompileError(
            f"{where} mark output ordinal {repeated} as a check more than once"
        )
    return tuple(checks)


def _validate_checks(definition: Definition, where: str) -> None:
    """Every check output is a computed gate of the copy and its value fits the gate."""

    for index, check in enumerate(definition.checks):
        here = f"{where} checks[{index}]"
        for kind, run in definition.check_pieces(check):
            if kind is not PieceKind.GATE:
                raise CompileError(
                    f"{here} marks a {kind.value} output; a check output must be a computed gate"
                )
            if check.value >= 1 << run.width:
                raise CompileError(
                    f"{here} requires the value {check.value} of a {run.width}-bit gate"
                )


# -- distinct declared outputs -----------------------------------------------


def _common_offset(first: Run, second: Run) -> int | None:
    """A gate offset in both runs, or ``None`` when they are disjoint.

    The linear congruence is solved by
    :func:`~veritor.core.description.progression_meet`; the first index of
    ``first`` that lands in ``second`` names the offset.
    """

    meet = progression_meet(first.start, first.count, first.stride, second)
    return None if meet is None else first.element(meet[0])


def _repeated_output(runs: tuple[Run, ...]) -> int | None:
    """A gate offset declared twice among ``runs``, or ``None`` if they are disjoint.

    A sweep over the runs in start order: a run is compared only with the
    earlier runs whose span still reaches its start, so runs laid out one
    after another (a token per step, a request per repeat copy) cost
    ``O(R log R)`` in all, and only runs that interleave over one span pay
    for the pairwise congruence.
    """

    ordered = []
    for run in runs:
        if run.count > 1 and run.stride == 0:
            return run.start  # the same gate ``count`` times
        ordered.append(run if run.count > 1 else Run(run.start, 1, 0, run.width))
    ordered.sort(key=lambda run: run.start)
    reaching: list[
        Run
    ] = []  # earlier runs whose last element is not before the current start
    for run in ordered:
        reaching = [other for other in reaching if other.last >= run.start]
        for other in reaching:
            common = _common_offset(other, run)
            if common is not None:
                return common
        reaching.append(run)
    return None


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
    runs = 0
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
        runs += len(available[digest].resolved_outputs)
        if runs > limits.max_output_runs_total:
            raise CompileError("description exceeds max_output_runs_total")
    root = available.get(_digest(document["root"], "root"))
    if root is None:
        raise CompileError("root names a definition that is not defined")
    if root.input_count != 0:
        raise CompileError("the root has no ports; inputs are `in` gates")
    for definition in available.values():
        if definition.checks and definition is not root:
            raise CompileError(
                f"definition {definition.digest[:12]} declares checks; only the root may"
            )
    return Description(description_digest(payload), tuple(available.values()), root)
