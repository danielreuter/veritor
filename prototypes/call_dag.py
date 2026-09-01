"""Memoized pure-constructor call DAG experiment.

This file is intentionally isolated from ``src/veritor``.  It explores one
question: can an untrusted, pure constructor describe a very large conceptual
leaf circuit by serializing reusable function definitions and their
occurrences, while a small trusted consumer validates the description and
provides random access to the conceptual flat leaves?

The trust boundary is:

    G(x, a) -> bytes                 # untrusted constructor and helpers
    Kernel.load(bytes) -> circuit    # trusted, data-only validation

The producer-side decorators and tracer are conveniences only.  The trusted
kernel does not import their objects or trust their cache keys.  Definitions
are identified on the trusted side by a SHA-256 digest of their canonical
serialized body.  A cache hit is accepted only for a body that this kernel
instance validated previously.
"""

from __future__ import annotations

import contextvars
import hashlib
import inspect
import json
from dataclasses import dataclass
from typing import Any, Callable, Hashable, Sequence


FORMAT_VERSION = 1


class ProducerError(ValueError):
    """The convenience tracer was used incorrectly."""


class KernelReject(ValueError):
    """The trusted decoder rejected constructor output."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def _definition_digest(body: object) -> str:
    return hashlib.sha256(_canonical_json(body)).hexdigest()


# =============================================================================
# UNTRUSTED PRODUCER-SIDE CONVENIENCE LIBRARY
# =============================================================================


@dataclass(frozen=True)
class _Source:
    kind: str
    index: int
    output: int = 0

    @staticmethod
    def input(index: int) -> _Source:
        return _Source("input", index)

    @staticmethod
    def step(step: int, output: int = 0) -> _Source:
        return _Source("step", step, output)

    def to_data(self) -> list[object]:
        if self.kind == "input":
            return ["input", self.index]
        return ["step", self.index, self.output]


@dataclass(frozen=True)
class Wire:
    """One symbolic value while a convenience function is being traced."""

    _trace_identity: object
    _source: _Source


@dataclass(frozen=True)
class ProducerDefinition:
    """An untrusted producer's locally cached definition."""

    producer: Producer
    digest: str
    input_count: int
    output_count: int

    def __call__(self, *args: Wire) -> Wire | tuple[Wire, ...]:
        trace = _ACTIVE_TRACE.get()
        if trace is None or trace.producer is not self.producer:
            raise ProducerError("a circuit definition may only be called while tracing")
        return trace.emit_call(self, args)


@dataclass(frozen=True)
class ProducerGate:
    """A producer-visible marker for one approved gate name and arity."""

    producer: Producer
    name: str
    arity: int

    def __call__(self, *args: Wire) -> Wire:
        trace = _ACTIVE_TRACE.get()
        if trace is None or trace.producer is not self.producer:
            raise ProducerError("a gate may only be called on symbolic wires")
        return trace.emit_leaf(self, args)


class _Trace:
    def __init__(self, producer: Producer, input_count: int) -> None:
        self.producer = producer
        self.identity = object()
        self.inputs = tuple(
            Wire(self.identity, _Source.input(i)) for i in range(input_count)
        )
        self.steps: list[dict[str, object]] = []

    def _source(self, wire: Wire) -> list[object]:
        if not isinstance(wire, Wire) or wire._trace_identity is not self.identity:
            raise ProducerError("all arguments must be wires from the current trace")
        return wire._source.to_data()

    def emit_leaf(self, gate: ProducerGate, args: Sequence[Wire]) -> Wire:
        if len(args) != gate.arity:
            raise ProducerError(
                f"{gate.name} expects {gate.arity} arguments, got {len(args)}"
            )
        step_index = len(self.steps)
        self.steps.append(
            {
                "kind": "leaf",
                "gate": gate.name,
                "args": [self._source(arg) for arg in args],
            }
        )
        return Wire(self.identity, _Source.step(step_index))

    def emit_call(
        self,
        definition: ProducerDefinition,
        args: Sequence[Wire],
    ) -> Wire | tuple[Wire, ...]:
        if len(args) != definition.input_count:
            raise ProducerError(
                f"definition expects {definition.input_count} arguments, "
                f"got {len(args)}"
            )
        step_index = len(self.steps)
        self.steps.append(
            {
                "kind": "call",
                "definition": definition.digest,
                "args": [self._source(arg) for arg in args],
            }
        )
        outputs = tuple(
            Wire(self.identity, _Source.step(step_index, i))
            for i in range(definition.output_count)
        )
        return outputs[0] if len(outputs) == 1 else outputs


_ACTIVE_TRACE: contextvars.ContextVar[_Trace | None] = contextvars.ContextVar(
    "prototype_active_trace",
    default=None,
)


class Producer:
    """Untrusted JAX-like helper for producing serialized definitions.

    ``circuit(key=..., input_count=...)`` memoizes the result of tracing one
    pure helper call.  Correct cache keys matter for the intended semantics of
    G, but not for kernel safety: the kernel validates whatever bytes G emits.
    """

    def __init__(self, cell_bits: int) -> None:
        if cell_bits <= 0:
            raise ProducerError("cell_bits must be positive")
        self.cell_bits = cell_bits
        self._gates: dict[str, ProducerGate] = {}
        self._definitions: dict[str, dict[str, object]] = {}
        self._call_cache: dict[Hashable, ProducerDefinition] = {}
        self._call_input_counts: dict[Hashable, int] = {}
        self.trace_hits = 0
        self.trace_misses = 0

    @property
    def unique_definition_count(self) -> int:
        return len(self._definitions)

    def gate(
        self,
        function: Callable[..., object] | None = None,
        *,
        name: str | None = None,
    ):
        """Register a producer-side primitive marker using decorator syntax."""

        def register(fn: Callable[..., object]) -> ProducerGate:
            gate_name = name or fn.__name__
            parameters = tuple(inspect.signature(fn).parameters.values())
            if not parameters or any(
                p.kind
                not in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
                for p in parameters
            ):
                raise ProducerError("gate functions need fixed positive positional arity")
            if gate_name in self._gates:
                raise ProducerError(f"duplicate producer gate {gate_name!r}")
            gate = ProducerGate(self, gate_name, len(parameters))
            self._gates[gate_name] = gate
            return gate

        return register(function) if function is not None else register

    def circuit(self, *, key: Hashable, input_count: int):
        """Trace or retrieve one cacheable pure constructor helper."""

        if type(input_count) is not int or input_count < 0:
            raise ProducerError("input_count must be a nonnegative integer")

        def decorate(fn: Callable[..., object]) -> ProducerDefinition:
            cached = self._call_cache.get(key)
            if cached is not None:
                if self._call_input_counts[key] != input_count:
                    raise ProducerError("a cache key was reused with a new input count")
                self.trace_hits += 1
                return cached

            self.trace_misses += 1
            trace = _Trace(self, input_count)
            token = _ACTIVE_TRACE.set(trace)
            try:
                result = fn(*trace.inputs)
            finally:
                _ACTIVE_TRACE.reset(token)

            if isinstance(result, Wire):
                outputs = (result,)
            elif isinstance(result, (tuple, list)) and result:
                outputs = tuple(result)
            else:
                raise ProducerError("a circuit must return one or more symbolic wires")
            output_data = [trace._source(output) for output in outputs]

            body: dict[str, object] = {
                "input_count": input_count,
                "steps": trace.steps,
                "outputs": output_data,
            }
            digest = _definition_digest(body)
            self._definitions.setdefault(digest, body)
            definition = ProducerDefinition(
                producer=self,
                digest=digest,
                input_count=input_count,
                output_count=len(outputs),
            )
            self._call_cache[key] = definition
            self._call_input_counts[key] = input_count
            return definition

        return decorate

    def serialize(self, root: ProducerDefinition) -> bytes:
        """Serialize the root and all transitive definitions in dependency order."""

        if root.producer is not self:
            raise ProducerError("root belongs to another producer")
        ordered: list[str] = []
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(digest: str) -> None:
            if digest in visited:
                return
            if digest in visiting:
                raise ProducerError("producer definition cycle")
            body = self._definitions.get(digest)
            if body is None:
                raise ProducerError(f"missing producer definition {digest}")
            visiting.add(digest)
            for step in body["steps"]:
                if step["kind"] == "call":
                    visit(step["definition"])
            visiting.remove(digest)
            visited.add(digest)
            ordered.append(digest)

        visit(root.digest)
        document = {
            "version": FORMAT_VERSION,
            "cell_bits": self.cell_bits,
            "definitions": [
                {"id": digest, "body": self._definitions[digest]}
                for digest in ordered
            ],
            "root": root.digest,
        }
        return _canonical_json(document)


# =============================================================================
# TRUSTED DATA MODEL AND VALIDATOR
# =============================================================================


@dataclass(frozen=True)
class GateSpec:
    name: str
    arity: int
    cost: int
    evaluate: Callable[[tuple[int, ...]], int]


@dataclass(frozen=True)
class _InputLocation:
    index: int


@dataclass(frozen=True)
class _GateLocation:
    ordinal: int


_Location = _InputLocation | _GateLocation


@dataclass(frozen=True)
class _ValidatedLeaf:
    gate: GateSpec
    args: tuple[_Location, ...]
    gate_start: int

    @property
    def gate_count(self) -> int:
        return 1


@dataclass(frozen=True)
class _ValidatedCall:
    child: ValidatedDefinition
    args: tuple[_Location, ...]
    gate_start: int

    @property
    def gate_count(self) -> int:
        return self.child.gate_count


_ValidatedStep = _ValidatedLeaf | _ValidatedCall


@dataclass(frozen=True)
class ValidatedDefinition:
    digest: str
    input_count: int
    steps: tuple[_ValidatedStep, ...]
    outputs: tuple[_Location, ...]
    required_inputs: tuple[int, ...]
    gate_count: int
    cost: int

    @property
    def output_count(self) -> int:
        return len(self.outputs)


@dataclass(frozen=True)
class FlatGate:
    ordinal: int
    write: int
    function: str
    reads: tuple[int, ...]


@dataclass(frozen=True)
class FlatCircuit:
    input_count: int
    gates: tuple[FlatGate, ...]
    outputs: tuple[int, ...]
    cost: int

    @property
    def cell_count(self) -> int:
        return self.input_count + len(self.gates)


@dataclass(frozen=True)
class LoadReport:
    root: ValidatedDefinition
    serialized_bytes: int
    new_definitions: int
    cache_hits: int


OccurrencePath = tuple[int, ...]


@dataclass(frozen=True)
class OccurrenceSummary:
    """Kernel-derived interface and interval for one occurrence-tree node.

    A path is a tuple of local step indices.  The empty path denotes the root
    definition; a nonempty path denotes either a call occurrence or one
    primitive leaf.  Reused definitions therefore have one summary per
    occurrence, with fresh global positions and input wiring.
    """

    path: OccurrencePath
    kind: str
    definition_digest: str | None
    gate_start: int
    gate_count: int
    cost: int
    external_reads: tuple[int, ...]
    outputs: tuple[int, ...]

    @property
    def gate_stop(self) -> int:
        return self.gate_start + self.gate_count


@dataclass(frozen=True)
class ReplayPlan:
    """A validated occurrence cut and its exact staged-commitment boundary."""

    root_digest: str
    root_input_count: int
    root_gate_count: int
    root_outputs: tuple[int, ...]
    units: tuple[OccurrenceSummary, ...]
    boundary: tuple[int, ...]

    @property
    def value_count(self) -> int:
        return self.root_input_count + self.root_gate_count

    def unit_index_for_gate(self, gate_ordinal: int) -> int:
        """Return the replay unit owning one conceptual gate ordinal."""

        if (
            type(gate_ordinal) is not int
            or gate_ordinal < 0
            or gate_ordinal >= self.root_gate_count
        ):
            raise KernelReject("gate ordinal is out of range")
        low = 0
        high = len(self.units)
        while low < high:
            middle = (low + high) // 2
            unit = self.units[middle]
            if gate_ordinal < unit.gate_start:
                high = middle
            elif gate_ordinal >= unit.gate_stop:
                low = middle + 1
            else:
                return middle
        raise RuntimeError("validated replay plan does not own requested gate")

    def challenged_unit_indices(
        self,
        sampled_gate_ordinals: Sequence[int],
    ) -> tuple[int, ...]:
        """Derive public J from the verifier's still-secret gate sample T."""

        return tuple(
            sorted(
                {
                    self.unit_index_for_gate(gate_ordinal)
                    for gate_ordinal in sampled_gate_ordinals
                }
            )
        )

    def interior_positions(self, unit_index: int) -> tuple[int, ...]:
        """Global value positions committed after this unit is challenged."""

        if (
            type(unit_index) is not int
            or unit_index < 0
            or unit_index >= len(self.units)
        ):
            raise KernelReject("replay unit index is out of range")
        unit = self.units[unit_index]
        boundary = set(self.boundary)
        return tuple(
            position
            for position in range(
                self.root_input_count + unit.gate_start,
                self.root_input_count + unit.gate_stop,
            )
            if position not in boundary
        )

    def expected_replay_cost(self, sampling_probability: float) -> float:
        """Draft-v4 replay cost when gates are independently pre-sampled."""

        if (
            isinstance(sampling_probability, bool)
            or not isinstance(sampling_probability, (int, float))
            or not 0 <= sampling_probability <= 1
        ):
            raise KernelReject("sampling probability must lie in [0, 1]")
        probability = float(sampling_probability)
        return sum(
            unit.cost * (1.0 - (1.0 - probability) ** unit.gate_count)
            for unit in self.units
        )

    def expected_two_stage_replay_cost(self, replay_probability: float) -> float:
        """Replay cost when each unit is independently selected with q."""

        if (
            isinstance(replay_probability, bool)
            or not isinstance(replay_probability, (int, float))
            or not 0 <= replay_probability <= 1
        ):
            raise KernelReject("replay probability must lie in [0, 1]")
        return float(replay_probability) * sum(unit.cost for unit in self.units)

    def expected_two_stage_checked_gates(
        self,
        replay_probability: float,
        within_unit_probability: float,
    ) -> float:
        """Expected number of checked gates under independent q,s sampling."""

        for name, probability in (
            ("replay", replay_probability),
            ("within-unit", within_unit_probability),
        ):
            if (
                isinstance(probability, bool)
                or not isinstance(probability, (int, float))
                or not 0 <= probability <= 1
            ):
                raise KernelReject(f"{name} probability must lie in [0, 1]")
        return (
            float(replay_probability)
            * float(within_unit_probability)
            * self.root_gate_count
        )


@dataclass(frozen=True)
class Construction:
    load: LoadReport
    input_cells: tuple[int, ...]
    constructor_output: bytes


def trusted_word_gates(cell_bits: int) -> tuple[GateSpec, ...]:
    """Independent trusted semantics for the demo's approved gate set."""

    if cell_bits <= 0:
        raise ValueError("cell_bits must be positive")
    mask = (1 << cell_bits) - 1
    return (
        GateSpec("add", 2, 1, lambda args: (args[0] + args[1]) & mask),
        GateSpec("mul", 2, 2, lambda args: (args[0] * args[1]) & mask),
    )


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise KernelReject(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _require_mapping(
    value: object,
    keys: set[str],
    where: str,
) -> dict[str, object]:
    if type(value) is not dict:
        raise KernelReject(f"{where} must be an object")
    if set(value) != keys:
        raise KernelReject(
            f"{where} has keys {sorted(value)}; expected {sorted(keys)}"
        )
    return value


def _require_list(value: object, where: str) -> list[object]:
    if type(value) is not list:
        raise KernelReject(f"{where} must be a list")
    return value


def _require_int(
    value: object,
    where: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if type(value) is not int or value < minimum:
        raise KernelReject(f"{where} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise KernelReject(f"{where} exceeds {maximum}")
    return value


def _require_digest(value: object, where: str) -> str:
    if type(value) is not str or len(value) != 64:
        raise KernelReject(f"{where} must be a 64-character SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise KernelReject(f"{where} is not hexadecimal") from error
    if value.lower() != value:
        raise KernelReject(f"{where} must use lowercase hexadecimal")
    return value


class Kernel:
    """Trusted parser, bottom-up validator, cache, and random-access decoder."""

    def __init__(
        self,
        *,
        cell_bits: int,
        gates: Sequence[GateSpec],
        max_blob_bytes: int = 10_000_000,
        max_definitions_per_blob: int = 100_000,
        max_steps_per_definition: int = 1_000_000,
        max_cells: int = (1 << 63) - 1,
        max_cost: int = (1 << 63) - 1,
    ) -> None:
        if cell_bits <= 0:
            raise ValueError("cell_bits must be positive")
        self.cell_bits = cell_bits
        self.mask = (1 << cell_bits) - 1
        self.max_blob_bytes = max_blob_bytes
        self.max_definitions_per_blob = max_definitions_per_blob
        self.max_steps_per_definition = max_steps_per_definition
        self.max_cells = max_cells
        self.max_cost = max_cost
        self._gates: dict[str, GateSpec] = {}
        for gate in gates:
            if gate.name in self._gates:
                raise ValueError(f"duplicate trusted gate {gate.name!r}")
            if gate.arity <= 0 or gate.cost < 0:
                raise ValueError("trusted gates need positive arity and nonnegative cost")
            self._gates[gate.name] = gate
        self._cache: dict[str, ValidatedDefinition] = {}

    @property
    def cached_definition_count(self) -> int:
        return len(self._cache)

    @property
    def cached_digests(self) -> frozenset[str]:
        return frozenset(self._cache)

    def _bounded_add(self, left: int, right: int, limit: int, what: str) -> int:
        total = left + right
        if total > limit:
            raise KernelReject(f"{what} exceeds configured limit {limit}")
        return total

    def _parse_source(
        self,
        value: object,
        *,
        input_count: int,
        step_outputs: Sequence[tuple[_Location, ...]],
        current_step: int,
        where: str,
    ) -> _Location:
        source = _require_list(value, where)
        if len(source) == 2 and source[0] == "input":
            index = _require_int(source[1], f"{where} input index")
            if index >= input_count:
                raise KernelReject(f"{where} references missing input {index}")
            return _InputLocation(index)
        if len(source) == 3 and source[0] == "step":
            step_index = _require_int(source[1], f"{where} step index")
            output_index = _require_int(source[2], f"{where} output index")
            if step_index >= current_step:
                raise KernelReject(f"{where} must reference an earlier step")
            if output_index >= len(step_outputs[step_index]):
                raise KernelReject(f"{where} references missing step output")
            return step_outputs[step_index][output_index]
        raise KernelReject(f"{where} is not a canonical input/step source")

    @staticmethod
    def _substitute_child_location(
        location: _Location,
        args: tuple[_Location, ...],
        gate_start: int,
    ) -> _Location:
        if isinstance(location, _InputLocation):
            return args[location.index]
        return _GateLocation(gate_start + location.ordinal)

    def _validate_definition(
        self,
        digest: str,
        body_value: object,
    ) -> ValidatedDefinition:
        body = _require_mapping(
            body_value,
            {"input_count", "steps", "outputs"},
            f"definition {digest}",
        )
        input_count = _require_int(
            body["input_count"],
            f"definition {digest} input_count",
            maximum=self.max_cells,
        )
        raw_steps = _require_list(body["steps"], f"definition {digest} steps")
        if len(raw_steps) > self.max_steps_per_definition:
            raise KernelReject(f"definition {digest} has too many local steps")

        steps: list[_ValidatedStep] = []
        step_outputs: list[tuple[_Location, ...]] = []
        required_inputs: set[int] = set()
        gate_count = 0
        cost = 0

        for step_index, raw_step in enumerate(raw_steps):
            if type(raw_step) is not dict:
                raise KernelReject(
                    f"definition {digest} step {step_index} must be an object"
                )
            kind = raw_step.get("kind")
            if kind == "leaf":
                step = _require_mapping(
                    raw_step,
                    {"kind", "gate", "args"},
                    f"definition {digest} leaf {step_index}",
                )
                gate_name = step["gate"]
                if type(gate_name) is not str or gate_name not in self._gates:
                    raise KernelReject(
                        f"definition {digest} uses unknown gate {gate_name!r}"
                    )
                gate = self._gates[gate_name]
                raw_args = _require_list(
                    step["args"],
                    f"definition {digest} leaf {step_index} args",
                )
                if len(raw_args) != gate.arity:
                    raise KernelReject(
                        f"gate {gate.name} expects {gate.arity} arguments, "
                        f"got {len(raw_args)}"
                    )
                args = tuple(
                    self._parse_source(
                        source,
                        input_count=input_count,
                        step_outputs=step_outputs,
                        current_step=step_index,
                        where=f"definition {digest} leaf {step_index} arg {arg_index}",
                    )
                    for arg_index, source in enumerate(raw_args)
                )
                validated = _ValidatedLeaf(gate, args, gate_count)
                required_inputs.update(
                    arg.index for arg in args if isinstance(arg, _InputLocation)
                )
                steps.append(validated)
                step_outputs.append((_GateLocation(gate_count),))
                gate_count = self._bounded_add(
                    gate_count, 1, self.max_cells, "gate count"
                )
                cost = self._bounded_add(cost, gate.cost, self.max_cost, "cost")
                continue

            if kind == "call":
                step = _require_mapping(
                    raw_step,
                    {"kind", "definition", "args"},
                    f"definition {digest} call {step_index}",
                )
                child_digest = _require_digest(
                    step["definition"],
                    f"definition {digest} call {step_index} child",
                )
                child = self._cache.get(child_digest)
                if child is None:
                    raise KernelReject(
                        f"definition {digest} calls unavailable definition "
                        f"{child_digest}"
                    )
                raw_args = _require_list(
                    step["args"],
                    f"definition {digest} call {step_index} args",
                )
                if len(raw_args) != child.input_count:
                    raise KernelReject(
                        f"call to {child_digest} expects {child.input_count} "
                        f"arguments, got {len(raw_args)}"
                    )
                args = tuple(
                    self._parse_source(
                        source,
                        input_count=input_count,
                        step_outputs=step_outputs,
                        current_step=step_index,
                        where=f"definition {digest} call {step_index} arg {arg_index}",
                    )
                    for arg_index, source in enumerate(raw_args)
                )
                validated = _ValidatedCall(child, args, gate_count)
                required_inputs.update(
                    args[child_input].index
                    for child_input in child.required_inputs
                    if isinstance(args[child_input], _InputLocation)
                )
                steps.append(validated)
                step_outputs.append(
                    tuple(
                        self._substitute_child_location(
                            output, args, validated.gate_start
                        )
                        for output in child.outputs
                    )
                )
                gate_count = self._bounded_add(
                    gate_count, child.gate_count, self.max_cells, "gate count"
                )
                cost = self._bounded_add(cost, child.cost, self.max_cost, "cost")
                continue

            raise KernelReject(
                f"definition {digest} step {step_index} has unknown kind {kind!r}"
            )

        raw_outputs = _require_list(
            body["outputs"],
            f"definition {digest} outputs",
        )
        if not raw_outputs:
            raise KernelReject(f"definition {digest} needs at least one output")
        outputs = tuple(
            self._parse_source(
                source,
                input_count=input_count,
                step_outputs=step_outputs,
                current_step=len(steps),
                where=f"definition {digest} output {output_index}",
            )
            for output_index, source in enumerate(raw_outputs)
        )
        self._bounded_add(input_count, gate_count, self.max_cells, "cell count")
        return ValidatedDefinition(
            digest=digest,
            input_count=input_count,
            steps=tuple(steps),
            outputs=outputs,
            required_inputs=tuple(sorted(required_inputs)),
            gate_count=gate_count,
            cost=cost,
        )

    def load(self, blob: bytes) -> LoadReport:
        """Parse and validate one constructor result, reusing local cache entries."""

        if type(blob) is not bytes:
            raise KernelReject("constructor output must be bytes")
        if len(blob) > self.max_blob_bytes:
            raise KernelReject("constructor output exceeds byte limit")
        try:
            document_value = json.loads(
                blob.decode("utf-8"),
                object_pairs_hook=_strict_object,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise KernelReject("constructor output is not valid UTF-8 JSON") from error

        document = _require_mapping(
            document_value,
            {"version", "cell_bits", "definitions", "root"},
            "document",
        )
        version = _require_int(document["version"], "document version")
        if version != FORMAT_VERSION:
            raise KernelReject(f"unsupported format version {version}")
        cell_bits = _require_int(document["cell_bits"], "document cell_bits", minimum=1)
        if cell_bits != self.cell_bits:
            raise KernelReject(
                f"document uses {cell_bits}-bit cells; kernel uses {self.cell_bits}"
            )
        raw_definitions = _require_list(document["definitions"], "definitions")
        if len(raw_definitions) > self.max_definitions_per_blob:
            raise KernelReject("too many definitions in one constructor output")

        seen_in_document: set[str] = set()
        new_definitions = 0
        cache_hits = 0
        for index, raw_entry in enumerate(raw_definitions):
            entry = _require_mapping(
                raw_entry,
                {"id", "body"},
                f"definitions[{index}]",
            )
            digest = _require_digest(entry["id"], f"definitions[{index}].id")
            if digest in seen_in_document:
                raise KernelReject(f"duplicate definition {digest} in document")
            seen_in_document.add(digest)
            actual_digest = _definition_digest(entry["body"])
            if actual_digest != digest:
                raise KernelReject(f"definition {digest} body digest does not match")
            if digest in self._cache:
                cache_hits += 1
                continue
            validated = self._validate_definition(digest, entry["body"])
            self._cache[digest] = validated
            new_definitions += 1

        root_digest = _require_digest(document["root"], "root")
        root = self._cache.get(root_digest)
        if root is None:
            raise KernelReject("root definition is unavailable")
        return LoadReport(
            root=root,
            serialized_bytes=len(blob),
            new_definitions=new_definitions,
            cache_hits=cache_hits,
        )

    @staticmethod
    def _global_location(
        location: _Location,
        *,
        input_context: tuple[int, ...],
        gate_base: int,
        root_input_count: int,
    ) -> int:
        if isinstance(location, _InputLocation):
            return input_context[location.index]
        return root_input_count + gate_base + location.ordinal

    def _flatten_definition(
        self,
        definition: ValidatedDefinition,
        *,
        input_context: tuple[int, ...],
        gate_base: int,
        root_input_count: int,
        gates: list[FlatGate],
    ) -> tuple[int, ...]:
        if len(gates) != gate_base:
            raise RuntimeError("internal flattening offset mismatch")
        for step in definition.steps:
            if isinstance(step, _ValidatedLeaf):
                ordinal = gate_base + step.gate_start
                reads = tuple(
                    self._global_location(
                        arg,
                        input_context=input_context,
                        gate_base=gate_base,
                        root_input_count=root_input_count,
                    )
                    for arg in step.args
                )
                gates.append(
                    FlatGate(
                        ordinal=ordinal,
                        write=root_input_count + ordinal,
                        function=step.gate.name,
                        reads=reads,
                    )
                )
                continue

            child_context = tuple(
                self._global_location(
                    arg,
                    input_context=input_context,
                    gate_base=gate_base,
                    root_input_count=root_input_count,
                )
                for arg in step.args
            )
            self._flatten_definition(
                step.child,
                input_context=child_context,
                gate_base=gate_base + step.gate_start,
                root_input_count=root_input_count,
                gates=gates,
            )

        return tuple(
            self._global_location(
                output,
                input_context=input_context,
                gate_base=gate_base,
                root_input_count=root_input_count,
            )
            for output in definition.outputs
        )

    def flatten(self, root: ValidatedDefinition) -> FlatCircuit:
        gates: list[FlatGate] = []
        outputs = self._flatten_definition(
            root,
            input_context=tuple(range(root.input_count)),
            gate_base=0,
            root_input_count=root.input_count,
            gates=gates,
        )
        if len(gates) != root.gate_count:
            raise RuntimeError("internal flattening count mismatch")
        return FlatCircuit(
            input_count=root.input_count,
            gates=tuple(gates),
            outputs=outputs,
            cost=root.cost,
        )

    def _definition_occurrence_summary(
        self,
        definition: ValidatedDefinition,
        *,
        path: OccurrencePath,
        kind: str,
        input_context: tuple[int, ...],
        gate_base: int,
        root_input_count: int,
    ) -> OccurrenceSummary:
        external_reads = tuple(
            sorted({input_context[index] for index in definition.required_inputs})
        )
        outputs = tuple(
            self._global_location(
                output,
                input_context=input_context,
                gate_base=gate_base,
                root_input_count=root_input_count,
            )
            for output in definition.outputs
        )
        return OccurrenceSummary(
            path=path,
            kind=kind,
            definition_digest=definition.digest,
            gate_start=gate_base,
            gate_count=definition.gate_count,
            cost=definition.cost,
            external_reads=external_reads,
            outputs=outputs,
        )

    def occurrence_summary(
        self,
        root: ValidatedDefinition,
        path: OccurrencePath,
    ) -> OccurrenceSummary:
        """Resolve one occurrence path and derive its globally wired interface."""

        if type(path) is not tuple or any(
            type(step_index) is not int or step_index < 0 for step_index in path
        ):
            raise KernelReject(
                "an occurrence path must be a tuple of nonnegative step indices"
            )

        root_input_count = root.input_count
        definition = root
        input_context = tuple(range(root_input_count))
        gate_base = 0
        if not path:
            return self._definition_occurrence_summary(
                definition,
                path=path,
                kind="root",
                input_context=input_context,
                gate_base=gate_base,
                root_input_count=root_input_count,
            )

        for depth, step_index in enumerate(path):
            if step_index >= len(definition.steps):
                raise KernelReject(f"occurrence path {path!r} references a missing step")
            step = definition.steps[step_index]
            is_target = depth == len(path) - 1
            if isinstance(step, _ValidatedLeaf):
                if not is_target:
                    raise KernelReject(
                        f"occurrence path {path!r} descends through a primitive leaf"
                    )
                reads = tuple(
                    sorted(
                        {
                            self._global_location(
                                arg,
                                input_context=input_context,
                                gate_base=gate_base,
                                root_input_count=root_input_count,
                            )
                            for arg in step.args
                        }
                    )
                )
                ordinal = gate_base + step.gate_start
                return OccurrenceSummary(
                    path=path,
                    kind="leaf",
                    definition_digest=None,
                    gate_start=ordinal,
                    gate_count=1,
                    cost=step.gate.cost,
                    external_reads=reads,
                    outputs=(root_input_count + ordinal,),
                )

            child_context = tuple(
                self._global_location(
                    arg,
                    input_context=input_context,
                    gate_base=gate_base,
                    root_input_count=root_input_count,
                )
                for arg in step.args
            )
            child_gate_base = gate_base + step.gate_start
            if is_target:
                return self._definition_occurrence_summary(
                    step.child,
                    path=path,
                    kind="call",
                    input_context=child_context,
                    gate_base=child_gate_base,
                    root_input_count=root_input_count,
                )
            definition = step.child
            input_context = child_context
            gate_base = child_gate_base

        raise RuntimeError("validated occurrence path did not resolve")

    @staticmethod
    def _validated_occurrence_paths(
        occurrence_paths: Sequence[OccurrencePath],
    ) -> tuple[OccurrencePath, ...]:
        paths: list[OccurrencePath] = []
        for path in occurrence_paths:
            if type(path) is not tuple or any(
                type(step_index) is not int or step_index < 0 for step_index in path
            ):
                raise KernelReject(
                    "every replay-unit path must be a tuple of nonnegative step indices"
                )
            paths.append(path)
        if len(set(paths)) != len(paths):
            raise KernelReject("replay-unit paths must be unique")
        return tuple(paths)

    def derive_replay_plan(
        self,
        root: ValidatedDefinition,
        occurrence_paths: Sequence[OccurrencePath],
    ) -> ReplayPlan:
        """Validate an occurrence cut and derive its boundary without flattening.

        Selected positive-size occurrence nodes must have disjoint, contiguous
        gate intervals covering the root's conceptual gates.  For such a cut,
        the exact boundary is the root inputs and outputs plus every selected
        unit's external reads.
        """

        paths = self._validated_occurrence_paths(occurrence_paths)
        units = tuple(
            sorted(
                (self.occurrence_summary(root, path) for path in paths),
                key=lambda unit: (unit.gate_start, unit.gate_stop, unit.path),
            )
        )

        cursor = 0
        total_cost = 0
        for unit in units:
            if unit.gate_count == 0:
                raise KernelReject(
                    f"replay unit {unit.path!r} contains no conceptual gates"
                )
            if unit.gate_start < cursor:
                raise KernelReject("replay-unit occurrences overlap")
            if unit.gate_start > cursor:
                raise KernelReject(
                    f"replay-unit cut leaves gate interval [{cursor}, {unit.gate_start})"
                )
            cursor = unit.gate_stop
            total_cost += unit.cost
        if cursor != root.gate_count:
            raise KernelReject(
                f"replay-unit cut leaves gate interval [{cursor}, {root.gate_count})"
            )
        if total_cost != root.cost:
            raise RuntimeError("validated replay-unit costs do not partition root cost")

        root_outputs = tuple(
            self._global_location(
                output,
                input_context=tuple(range(root.input_count)),
                gate_base=0,
                root_input_count=root.input_count,
            )
            for output in root.outputs
        )
        boundary = set(range(root.input_count))
        boundary.update(root_outputs)
        for unit in units:
            boundary.update(unit.external_reads)

        return ReplayPlan(
            root_digest=root.digest,
            root_input_count=root.input_count,
            root_gate_count=root.gate_count,
            root_outputs=root_outputs,
            units=units,
            boundary=tuple(sorted(boundary)),
        )

    def validate_replay_plan(
        self,
        root: ValidatedDefinition,
        plan: ReplayPlan,
    ) -> None:
        """Re-derive a plan and require exact circuit-relative agreement."""

        if not isinstance(root, ValidatedDefinition) or not isinstance(
            plan,
            ReplayPlan,
        ):
            raise KernelReject("replay plan validation received the wrong type")
        if type(plan.units) is not tuple or any(
            not isinstance(unit, OccurrenceSummary) for unit in plan.units
        ):
            raise KernelReject("replay plan contains malformed units")
        expected = self.derive_replay_plan(
            root,
            tuple(unit.path for unit in plan.units),
        )
        if plan != expected:
            raise KernelReject("replay plan does not match the circuit")

    def _gate_at(
        self,
        definition: ValidatedDefinition,
        local_ordinal: int,
        *,
        input_context: tuple[int, ...],
        gate_base: int,
        root_input_count: int,
    ) -> FlatGate:
        for step in definition.steps:
            if not (
                step.gate_start
                <= local_ordinal
                < step.gate_start + step.gate_count
            ):
                continue
            if isinstance(step, _ValidatedLeaf):
                ordinal = gate_base + local_ordinal
                return FlatGate(
                    ordinal=ordinal,
                    write=root_input_count + ordinal,
                    function=step.gate.name,
                    reads=tuple(
                        self._global_location(
                            arg,
                            input_context=input_context,
                            gate_base=gate_base,
                            root_input_count=root_input_count,
                        )
                        for arg in step.args
                    ),
                )
            child_context = tuple(
                self._global_location(
                    arg,
                    input_context=input_context,
                    gate_base=gate_base,
                    root_input_count=root_input_count,
                )
                for arg in step.args
            )
            return self._gate_at(
                step.child,
                local_ordinal - step.gate_start,
                input_context=child_context,
                gate_base=gate_base + step.gate_start,
                root_input_count=root_input_count,
            )
        raise RuntimeError("validated definition did not contain requested gate")

    def gate_at(self, root: ValidatedDefinition, gate_ordinal: int) -> FlatGate:
        if type(gate_ordinal) is not int or not (0 <= gate_ordinal < root.gate_count):
            raise KernelReject("gate ordinal is out of range")
        return self._gate_at(
            root,
            gate_ordinal,
            input_context=tuple(range(root.input_count)),
            gate_base=0,
            root_input_count=root.input_count,
        )

    def apply_gate(self, function: str, args: Sequence[int]) -> int:
        """Apply one approved gate using the kernel's trusted semantics."""

        spec = self._gates.get(function)
        if spec is None:
            raise KernelReject(f"unknown trusted gate {function!r}")
        if len(args) != spec.arity:
            raise KernelReject(
                f"gate {function} expects {spec.arity} arguments, got {len(args)}"
            )
        checked_args = []
        for index, value in enumerate(args):
            if type(value) is not int or not (0 <= value <= self.mask):
                raise KernelReject(
                    f"gate {function} argument {index} is not a valid cell"
                )
            checked_args.append(value)
        result = spec.evaluate(tuple(checked_args))
        if type(result) is not int or not (0 <= result <= self.mask):
            raise RuntimeError(f"trusted gate {spec.name} returned an invalid word")
        return result

    def evaluate_tape(
        self,
        root: ValidatedDefinition,
        inputs: Sequence[int],
    ) -> tuple[int, ...]:
        """Evaluate and return all input and gate-value positions."""

        if len(inputs) != root.input_count:
            raise KernelReject(
                f"circuit expects {root.input_count} inputs, got {len(inputs)}"
            )
        cells: list[int] = []
        for index, value in enumerate(inputs):
            if type(value) is not int or not (0 <= value <= self.mask):
                raise KernelReject(
                    f"input {index} is not a {self.cell_bits}-bit word"
                )
            cells.append(value)
        flat = self.flatten(root)
        for gate in flat.gates:
            if gate.write != len(cells):
                raise RuntimeError("internal flat write position mismatch")
            value = self.apply_gate(
                gate.function,
                tuple(cells[index] for index in gate.reads),
            )
            cells.append(value)
        return tuple(cells)

    def evaluate(
        self,
        root: ValidatedDefinition,
        inputs: Sequence[int],
    ) -> tuple[int, ...]:
        cells = self.evaluate_tape(root, inputs)
        outputs = tuple(
            self._global_location(
                output,
                input_context=tuple(range(root.input_count)),
                gate_base=0,
                root_input_count=root.input_count,
            )
            for output in root.outputs
        )
        return tuple(cells[index] for index in outputs)


def construct(
    kernel: Kernel,
    constructor: Callable[[object, bytes], bytes],
    x: object,
    a: bytes,
    *,
    input_cells: Sequence[int],
    advice_bound_bits: int,
) -> Construction:
    """Run the prototype ``Construct`` boundary without claiming sandboxing."""

    if type(a) is not bytes:
        raise KernelReject("advice must be bytes in this prototype")
    if type(advice_bound_bits) is not int or advice_bound_bits < 0:
        raise KernelReject("advice bound must be a nonnegative bit count")
    if len(a) * 8 > advice_bound_bits:
        raise KernelReject("advice exceeds the public bit bound")
    try:
        blob = constructor(x, a)
    except Exception as error:
        raise KernelReject("constructor execution failed") from error
    load = kernel.load(blob)
    cells = tuple(input_cells)
    if load.root.input_count != len(cells):
        raise KernelReject(
            f"root expects {load.root.input_count} inputs, got {len(cells)}"
        )
    for index, value in enumerate(cells):
        if type(value) is not int or not (0 <= value <= kernel.mask):
            raise KernelReject(f"input {index} is not a valid cell")
    return Construction(load=load, input_cells=cells, constructor_output=blob)


# =============================================================================
# DEMO G: VARIABLE-LENGTH DOT PRODUCTS BATCHED IN DIFFERENT ORDERS
# =============================================================================


@dataclass(frozen=True)
class DotRequest:
    accumulator: int
    values: tuple[int, ...]
    weights: tuple[int, ...]

    @property
    def length(self) -> int:
        return len(self.values)

    def cells(self) -> tuple[int, ...]:
        if len(self.values) != len(self.weights):
            raise ProducerError("dot-product values and weights have different lengths")
        return (self.accumulator, *self.values, *self.weights)


@dataclass(frozen=True)
class BatchInput:
    requests: tuple[DotRequest, ...]

    def cells(self) -> tuple[int, ...]:
        return tuple(cell for request in self.requests for cell in request.cells())


class DemoG:
    """A statefully memoized implementation whose emitted bytes remain pure."""

    def __init__(self, cell_bits: int = 8) -> None:
        self.cell_bits = cell_bits
        self.producer = Producer(cell_bits)

        @self.producer.gate(name="add")
        def add(left, right):
            return left + right

        @self.producer.gate(name="mul")
        def mul(left, right):
            return left * right

        self.add = add
        self.mul = mul

        @self.producer.circuit(key=("mac",), input_count=3)
        def mac(accumulator, value, weight):
            return add(accumulator, mul(value, weight))

        self.mac = mac

    def dot(self, length: int) -> ProducerDefinition:
        if type(length) is not int or length < 0:
            raise ProducerError("dot length must be a nonnegative integer")
        input_count = 1 + 2 * length

        @self.producer.circuit(key=("dot", length), input_count=input_count)
        def dot_definition(*inputs):
            accumulator = inputs[0]
            values = inputs[1 : 1 + length]
            weights = inputs[1 + length :]
            for value, weight in zip(values, weights):
                accumulator = self.mac(accumulator, value, weight)
            return accumulator

        return dot_definition

    def batch(self, lengths: tuple[int, ...]) -> ProducerDefinition:
        input_count = sum(1 + 2 * length for length in lengths)

        @self.producer.circuit(
            key=("batch", lengths),
            input_count=input_count,
        )
        def batch_definition(*inputs):
            outputs = []
            offset = 0
            for length in lengths:
                child = self.dot(length)
                child_input_count = 1 + 2 * length
                outputs.append(child(*inputs[offset : offset + child_input_count]))
                offset += child_input_count
            return tuple(outputs)

        return batch_definition

    def __call__(self, x: object, a: bytes) -> bytes:
        if not isinstance(x, BatchInput):
            raise ProducerError("DemoG expects BatchInput")
        # Advice is intentionally unused in this first constructor.
        del a
        for request in x.requests:
            request.cells()
        root = self.batch(tuple(request.length for request in x.requests))
        return self.producer.serialize(root)


def expected_dot_outputs(batch: BatchInput, cell_bits: int) -> tuple[int, ...]:
    mask = (1 << cell_bits) - 1
    outputs = []
    for request in batch.requests:
        accumulator = request.accumulator
        for value, weight in zip(request.values, request.weights):
            accumulator = (accumulator + value * weight) & mask
        outputs.append(accumulator)
    return tuple(outputs)


def make_demo_request(length: int, seed: int, cell_bits: int = 8) -> DotRequest:
    mask = (1 << cell_bits) - 1
    values = tuple((seed + 3 * index + 1) & mask for index in range(length))
    weights = tuple((2 * seed + 5 * index + 1) & mask for index in range(length))
    return DotRequest(seed & mask, values, weights)


def flat_encoding_size(flat: FlatCircuit) -> int:
    data = {
        "input_count": flat.input_count,
        "gates": [
            {
                "function": gate.function,
                "reads": list(gate.reads),
                "write": gate.write,
            }
            for gate in flat.gates
        ],
        "outputs": list(flat.outputs),
    }
    return len(_canonical_json(data))


def run_demo() -> dict[str, object]:
    cell_bits = 8
    g = DemoG(cell_bits)
    kernel = Kernel(cell_bits=cell_bits, gates=trusted_word_gates(cell_bits))

    first = BatchInput(
        (
            make_demo_request(4, 1),
            make_demo_request(8, 2),
            make_demo_request(4, 3),
        )
    )
    second = BatchInput(
        (
            make_demo_request(8, 4),
            make_demo_request(4, 5),
            make_demo_request(8, 6),
        )
    )

    first_result = construct(
        kernel,
        g,
        first,
        b"",
        input_cells=first.cells(),
        advice_bound_bits=0,
    )
    second_result = construct(
        kernel,
        g,
        second,
        b"",
        input_cells=second.cells(),
        advice_bound_bits=0,
    )
    first_flat = kernel.flatten(first_result.load.root)
    second_flat = kernel.flatten(second_result.load.root)
    first_outputs = kernel.evaluate(first_result.load.root, first.cells())
    second_outputs = kernel.evaluate(second_result.load.root, second.cells())

    if first_outputs != expected_dot_outputs(first, cell_bits):
        raise RuntimeError("first demo evaluation mismatch")
    if second_outputs != expected_dot_outputs(second, cell_bits):
        raise RuntimeError("second demo evaluation mismatch")

    return {
        "first_batch_lengths": [4, 8, 4],
        "second_batch_lengths": [8, 4, 8],
        "producer_trace_misses": g.producer.trace_misses,
        "producer_trace_hits": g.producer.trace_hits,
        "producer_unique_definitions": g.producer.unique_definition_count,
        "trusted_definitions_after_both": kernel.cached_definition_count,
        "first": {
            "conceptual_leaves": first_result.load.root.gate_count,
            "cost": first_result.load.root.cost,
            "serialized_bytes": first_result.load.serialized_bytes,
            "flat_bytes": flat_encoding_size(first_flat),
            "new_definitions": first_result.load.new_definitions,
            "trusted_cache_hits": first_result.load.cache_hits,
            "outputs": list(first_outputs),
        },
        "second": {
            "conceptual_leaves": second_result.load.root.gate_count,
            "cost": second_result.load.root.cost,
            "serialized_bytes": second_result.load.serialized_bytes,
            "flat_bytes": flat_encoding_size(second_flat),
            "new_definitions": second_result.load.new_definitions,
            "trusted_cache_hits": second_result.load.cache_hits,
            "outputs": list(second_outputs),
        },
    }


if __name__ == "__main__":
    print(json.dumps(run_demo(), indent=2, sort_keys=True))
