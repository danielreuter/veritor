"""Memoized constructor-call DAGs and their trusted data-only decoder.

The producer helpers in this module are conveniences for an untrusted
constructor ``G``.  The trusted :class:`Kernel` consumes only canonical JSON
bytes, validates every definition bottom-up, and exposes the resulting
conceptual primitive circuit without importing or executing producer code.
"""

from __future__ import annotations

import contextvars
import hashlib
import inspect
import json
from bisect import bisect_right
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import overload

from veritor.core import (
    ArtifactKind,
    ExecutableGate,
    IntervalDomain,
    InvalidArtifact,
    JSONValue,
    Port,
    RangeIndexedDomain,
    StructuralGate,
    StructureIdentity,
    identity_digest,
)

FORMAT_VERSION = 1
COMPILER_ID = "veritor.compile.call-dag"
COMPILER_VERSION = "1"
STRUCTURE_SCHEMA_VERSION = "1"


class ProducerError(ValueError):
    """The untrusted producer convenience API was used incorrectly."""


class KernelReject(InvalidArtifact):
    """The trusted call-DAG decoder rejected constructor output."""


def canonical_call_dag_json(value: object) -> bytes:
    """Return the canonical ASCII JSON encoding used by the call-DAG format."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def definition_digest(body: object) -> str:
    """Return the canonical SHA-256 identifier for one definition body."""

    return hashlib.sha256(canonical_call_dag_json(body)).hexdigest()


# ---------------------------------------------------------------------------
# Untrusted producer-side convenience API
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
class Wire:
    """One symbolic value while a producer definition is being traced."""

    _trace_identity: object
    _source: _Source


@dataclass(frozen=True, slots=True)
class ProducerDefinition:
    """An untrusted producer's locally memoized definition handle."""

    producer: Producer
    digest: str
    input_count: int
    output_count: int

    def __call__(self, *args: Wire) -> Wire | tuple[Wire, ...]:
        trace = _ACTIVE_TRACE.get()
        if trace is None or trace.producer is not self.producer:
            raise ProducerError("a circuit definition may only be called while tracing")
        return trace.emit_call(self, args)


@dataclass(frozen=True, slots=True)
class ProducerGate:
    """A producer-visible marker for one named primitive and its arity."""

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
            Wire(self.identity, _Source.input(index)) for index in range(input_count)
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
            Wire(self.identity, _Source.step(step_index, output_index))
            for output_index in range(definition.output_count)
        )
        return outputs[0] if len(outputs) == 1 else outputs


_ACTIVE_TRACE: contextvars.ContextVar[_Trace | None] = contextvars.ContextVar(
    "veritor_call_dag_active_trace",
    default=None,
)


class Producer:
    """JAX-like helpers for producing canonical, memoized call-DAG bytes.

    Producer cache keys affect constructor behavior but are never trusted by
    the kernel.  Trusted definition identity is always the digest of the
    canonical serialized body.
    """

    def __init__(self, cell_bits: int) -> None:
        if type(cell_bits) is not int or cell_bits <= 0:
            raise ProducerError("cell_bits must be a positive integer")
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

    @overload
    def gate(
        self,
        function: Callable[..., object],
        *,
        name: str | None = None,
    ) -> ProducerGate: ...

    @overload
    def gate(
        self,
        function: None = None,
        *,
        name: str | None = None,
    ) -> Callable[[Callable[..., object]], ProducerGate]: ...

    def gate(
        self,
        function: Callable[..., object] | None = None,
        *,
        name: str | None = None,
    ) -> ProducerGate | Callable[[Callable[..., object]], ProducerGate]:
        """Register a producer-side primitive marker using decorator syntax."""

        def register(fn: Callable[..., object]) -> ProducerGate:
            gate_name = name or fn.__name__
            if type(gate_name) is not str or not gate_name.strip():
                raise ProducerError("gate names must be nonempty strings")
            parameters = tuple(inspect.signature(fn).parameters.values())
            if not parameters or any(
                parameter.kind
                not in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
                for parameter in parameters
            ):
                raise ProducerError(
                    "gate functions need fixed positive positional arity"
                )
            if gate_name in self._gates:
                raise ProducerError(f"duplicate producer gate {gate_name!r}")
            gate = ProducerGate(self, gate_name, len(parameters))
            self._gates[gate_name] = gate
            return gate

        return register(function) if function is not None else register

    def circuit(
        self,
        *,
        key: Hashable,
        input_count: int,
    ) -> Callable[[Callable[..., object]], ProducerDefinition]:
        """Trace or retrieve one cacheable pure constructor helper."""

        if type(input_count) is not int or input_count < 0:
            raise ProducerError("input_count must be a nonnegative integer")
        try:
            hash(key)
        except TypeError as error:
            raise ProducerError("circuit cache keys must be hashable") from error

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
            digest = definition_digest(body)
            existing = self._definitions.get(digest)
            if existing is not None and existing != body:
                raise ProducerError("definition digest collision")
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
        """Serialize a root and all transitive definitions in dependency order."""

        if not isinstance(root, ProducerDefinition) or root.producer is not self:
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
            raw_steps = body.get("steps")
            if not isinstance(raw_steps, list):
                raise ProducerError("producer definition has malformed steps")
            for raw_step in raw_steps:
                if not isinstance(raw_step, dict):
                    raise ProducerError("producer definition has a malformed step")
                if raw_step.get("kind") == "call":
                    child_digest = raw_step.get("definition")
                    if type(child_digest) is not str:
                        raise ProducerError("producer call has a malformed definition")
                    visit(child_digest)
            visiting.remove(digest)
            visited.add(digest)
            ordered.append(digest)

        visit(root.digest)
        return canonical_call_dag_json(
            {
                "version": FORMAT_VERSION,
                "cell_bits": self.cell_bits,
                "definitions": [
                    {"id": digest, "body": self._definitions[digest]}
                    for digest in ordered
                ],
                "root": root.digest,
            }
        )


# ---------------------------------------------------------------------------
# Trusted semantics declarations
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GateSpec:
    """One trusted primitive declaration.

    ``evaluate`` is executable trusted code.  It is deliberately excluded
    from equality and all identity manifests; the explicit registry identity
    and version name the semantics instead.
    """

    name: str
    arity: int
    cost: int
    evaluate: Callable[[tuple[int, ...]], int] = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name.strip():
            raise ValueError("trusted gate names must be nonempty strings")
        if type(self.arity) is not int or self.arity <= 0:
            raise ValueError("trusted gates need positive integer arity")
        if type(self.cost) is not int or self.cost < 0:
            raise ValueError("trusted gate costs must be nonnegative integers")
        if not callable(self.evaluate):
            raise TypeError("trusted gate evaluate must be callable")


@dataclass(frozen=True, slots=True, init=False)
class SemanticRegistry:
    """Explicit identity and declarations for trusted local semantics."""

    registry_id: str
    registry_version: str
    value_schema_id: str
    value_schema_version: str
    gates: tuple[GateSpec, ...]

    def __init__(
        self,
        *,
        registry_id: str,
        registry_version: str,
        value_schema_id: str,
        value_schema_version: str,
        gates: Sequence[GateSpec],
    ) -> None:
        texts = {
            "registry_id": registry_id,
            "registry_version": registry_version,
            "value_schema_id": value_schema_id,
            "value_schema_version": value_schema_version,
        }
        for field_name, value in texts.items():
            if type(value) is not str or not value.strip():
                raise ValueError(f"{field_name} must be a nonempty string")
        checked_gates = tuple(gates)
        if any(not isinstance(gate, GateSpec) for gate in checked_gates):
            raise TypeError("semantic registry gates must be GateSpec values")
        names = tuple(gate.name for gate in checked_gates)
        if len(set(names)) != len(names):
            raise ValueError("semantic registry gate names must be unique")
        object.__setattr__(self, "registry_id", registry_id)
        object.__setattr__(self, "registry_version", registry_version)
        object.__setattr__(self, "value_schema_id", value_schema_id)
        object.__setattr__(self, "value_schema_version", value_schema_version)
        object.__setattr__(self, "gates", checked_gates)

    @property
    def semantic_scope_id(self) -> str:
        return f"{self.registry_id}@{self.registry_version}"

    def relation_id(self, gate_name: str) -> str:
        if gate_name not in {gate.name for gate in self.gates}:
            raise KeyError(gate_name)
        return f"{self.registry_id}@{self.registry_version}/relation/{gate_name}"

    def value_type_id(self, cell_bits: int) -> str:
        return f"{self.value_schema_id}@{self.value_schema_version}/u{cell_bits}"

    @property
    def operator_manifest(self) -> dict[str, JSONValue]:
        return {
            "registry_id": self.registry_id,
            "registry_version": self.registry_version,
            "gates": [
                {
                    "arity": gate.arity,
                    "cost": gate.cost,
                    "name": gate.name,
                    "relation_id": self.relation_id(gate.name),
                }
                for gate in sorted(self.gates, key=lambda item: item.name)
            ],
        }

    def value_manifest(self, cell_bits: int) -> dict[str, JSONValue]:
        return {
            "cell_bits": cell_bits,
            "encoding": "unsigned-big-endian-fixed-width/v1",
            "schema_id": self.value_schema_id,
            "schema_version": self.value_schema_version,
            "value_type_id": self.value_type_id(cell_bits),
        }


@dataclass(frozen=True, slots=True)
class CompilationLimits:
    """Explicit resource limits for parsing and validating constructor output."""

    max_blob_bytes: int = 10_000_000
    max_definitions_per_blob: int = 100_000
    max_steps_per_definition: int = 1_000_000
    max_cells: int = (1 << 63) - 1
    max_cost: int = (1 << 63) - 1
    max_nesting_depth: int = 256
    max_partition_units: int = 1_000_000

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field_name} must be a nonnegative integer")


@dataclass(frozen=True, slots=True)
class WordValueCodec:
    """Canonical fixed-width codec for one unsigned cell type."""

    cell_bits: int
    value_type_id: str

    def __post_init__(self) -> None:
        if type(self.cell_bits) is not int or self.cell_bits <= 0:
            raise ValueError("cell_bits must be positive")
        if type(self.value_type_id) is not str or not self.value_type_id.strip():
            raise ValueError("value_type_id must be nonempty")

    @property
    def byte_length(self) -> int:
        return (self.cell_bits + 7) // 8

    @property
    def cardinality(self) -> int:
        return 1 << self.cell_bits

    def validate(self, value: object, *, where: str = "value") -> int:
        if type(value) is not int or not 0 <= value < self.cardinality:
            raise KernelReject(f"{where} is not a {self.cell_bits}-bit word")
        return value

    def encode(self, value: object) -> bytes:
        return self.validate(value).to_bytes(self.byte_length, "big")

    def decode(self, payload: object) -> int:
        if type(payload) is not bytes or len(payload) != self.byte_length:
            raise KernelReject(
                f"encoded value must be exactly {self.byte_length} bytes"
            )
        return self.validate(int.from_bytes(payload, "big"), where="encoded value")


class TrustedRelationEvaluator:
    """Evaluate relation identifiers from one explicit semantic registry."""

    __slots__ = ("_by_operation", "_by_relation", "_codec", "_registry")

    def __init__(
        self,
        registry: SemanticRegistry,
        codec: WordValueCodec,
    ) -> None:
        by_operation = {gate.name: gate for gate in registry.gates}
        by_relation = {registry.relation_id(gate.name): gate for gate in registry.gates}
        self._registry = registry
        self._codec = codec
        self._by_operation = MappingProxyType(by_operation)
        self._by_relation = MappingProxyType(by_relation)

    @property
    def registry_id(self) -> str:
        return self._registry.registry_id

    @property
    def registry_version(self) -> str:
        return self._registry.registry_version

    def relation_id_for_operation(self, operation: str) -> str:
        if operation not in self._by_operation:
            raise KernelReject(f"unknown trusted gate {operation!r}")
        return self._registry.relation_id(operation)

    def _evaluate_spec(self, spec: GateSpec, args: Sequence[int]) -> int:
        if len(args) != spec.arity:
            raise KernelReject(
                f"gate {spec.name} expects {spec.arity} arguments, got {len(args)}"
            )
        checked_args = tuple(
            self._codec.validate(value, where=f"gate {spec.name} argument {index}")
            for index, value in enumerate(args)
        )
        try:
            result = spec.evaluate(checked_args)
        except Exception as error:
            raise RuntimeError(
                f"trusted gate {spec.name} evaluator raised an exception"
            ) from error
        if type(result) is not int or not 0 <= result < self._codec.cardinality:
            raise RuntimeError(f"trusted gate {spec.name} returned an invalid word")
        return result

    def evaluate(self, relation_id: str, args: Sequence[int]) -> int:
        spec = self._by_relation.get(relation_id)
        if spec is None:
            raise KernelReject(f"unknown trusted relation {relation_id!r}")
        return self._evaluate_spec(spec, args)

    def evaluate_operation(self, operation: str, args: Sequence[int]) -> int:
        spec = self._by_operation.get(operation)
        if spec is None:
            raise KernelReject(f"unknown trusted gate {operation!r}")
        return self._evaluate_spec(spec, args)


# ---------------------------------------------------------------------------
# Trusted validated representation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _InputLocation:
    index: int


@dataclass(frozen=True, slots=True)
class _GateLocation:
    ordinal: int


_Location = _InputLocation | _GateLocation


@dataclass(frozen=True, slots=True)
class ValidatedLeaf:
    gate: GateSpec
    args: tuple[_Location, ...]
    gate_start: int

    @property
    def gate_count(self) -> int:
        return 1


@dataclass(frozen=True, slots=True)
class ValidatedCall:
    child: ValidatedDefinition
    args: tuple[_Location, ...]
    gate_start: int

    @property
    def gate_count(self) -> int:
        return self.child.gate_count


_ValidatedStep = ValidatedLeaf | ValidatedCall


@dataclass(frozen=True, slots=True)
class ValidatedDefinition:
    """One trusted, recursively summarized constructor definition."""

    digest: str
    input_count: int
    steps: tuple[_ValidatedStep, ...]
    outputs: tuple[_Location, ...]
    required_inputs: tuple[int, ...]
    gate_count: int
    cost: int
    nesting_depth: int
    step_starts: tuple[int, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "step_starts", tuple(step.gate_start for step in self.steps)
        )

    @property
    def output_count(self) -> int:
        return len(self.outputs)

    def step_owning(self, local_ordinal: int) -> _ValidatedStep:
        """Return the step whose gate interval contains ``local_ordinal``."""

        index = bisect_right(self.step_starts, local_ordinal) - 1
        while index >= 0:
            step = self.steps[index]
            if step.gate_start <= local_ordinal < step.gate_start + step.gate_count:
                return step
            index -= 1
        raise RuntimeError("validated definition did not contain requested gate")


@dataclass(frozen=True, slots=True)
class _Frame:
    """One level of a lazy descent into nested calls.

    ``args`` are the call's arguments as locations in the parent frame; a
    child input resolves by walking up until it reaches a gate or a root input.
    """

    definition: ValidatedDefinition
    gate_base: int
    parent: _Frame | None
    args: tuple[_Location, ...]

    def resolve_input(self, index: int, root_input_count: int) -> int:
        frame = self
        while frame.parent is not None:
            location = frame.args[index]
            frame = frame.parent
            if not isinstance(location, _InputLocation):
                return root_input_count + frame.gate_base + location.ordinal
            index = location.index
        return index

    def resolve(self, location: _Location, root_input_count: int) -> int:
        if isinstance(location, _InputLocation):
            return self.resolve_input(location.index, root_input_count)
        return root_input_count + self.gate_base + location.ordinal


@dataclass(frozen=True, slots=True)
class FlatGate:
    ordinal: int
    write: int
    function: str
    reads: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class FlatCircuit:
    input_count: int
    gates: tuple[FlatGate, ...]
    outputs: tuple[int, ...]
    cost: int

    @property
    def cell_count(self) -> int:
        return self.input_count + len(self.gates)


@dataclass(frozen=True, slots=True)
class LoadReport:
    root: ValidatedDefinition
    serialized_bytes: int
    new_definitions: int
    cache_hits: int


OccurrencePath = tuple[int, ...]


@dataclass(frozen=True, slots=True)
class OccurrenceSummary:
    """The global interval and interface of one occurrence-tree node."""

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


@dataclass(frozen=True, slots=True)
class ReplayPlan:
    """A validated occurrence cut and the replay boundary it induces.

    ``boundary`` holds every input position, every root output position, and
    every position read across a unit boundary.  It is derived from occurrence
    interfaces alone, so its cost is proportional to the number of units and
    their interfaces rather than to the number of gates.
    """

    root_digest: str
    root_input_count: int
    root_gate_count: int
    root_outputs: tuple[int, ...]
    units: tuple[OccurrenceSummary, ...]
    boundary: IntervalDomain


@dataclass(frozen=True, slots=True)
class Construction:
    load: LoadReport
    input_cells: tuple[int, ...]
    constructor_output: bytes


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise KernelReject(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise KernelReject(f"non-finite JSON number {value!r} is forbidden")


def _require_mapping(
    value: object,
    keys: set[str],
    where: str,
) -> dict[str, object]:
    if type(value) is not dict:
        raise KernelReject(f"{where} must be an object")
    if set(value) != keys:
        raise KernelReject(f"{where} has keys {sorted(value)}; expected {sorted(keys)}")
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
    if value.lower() != value:
        raise KernelReject(f"{where} must use lowercase hexadecimal")
    if any(character not in "0123456789abcdef" for character in value):
        raise KernelReject(f"{where} is not hexadecimal")
    return value


class Kernel:
    """Trusted parser, atomic definition cache, and lazy circuit decoder."""

    def __init__(
        self,
        *,
        cell_bits: int,
        semantic_registry: SemanticRegistry,
        limits: CompilationLimits | None = None,
    ) -> None:
        if type(cell_bits) is not int or cell_bits <= 0:
            raise ValueError("cell_bits must be a positive integer")
        if not isinstance(semantic_registry, SemanticRegistry):
            raise TypeError("Kernel requires an explicit, versioned SemanticRegistry")
        if limits is not None and not isinstance(limits, CompilationLimits):
            raise TypeError("limits must be CompilationLimits")
        self.cell_bits = cell_bits
        self.mask = (1 << cell_bits) - 1
        self.semantic_registry = semantic_registry
        self.limits = CompilationLimits() if limits is None else limits
        self._gates = {gate.name: gate for gate in semantic_registry.gates}
        self._cache: dict[str, ValidatedDefinition] = {}
        self.value_codec = WordValueCodec(
            cell_bits,
            semantic_registry.value_type_id(cell_bits),
        )
        self.relation_evaluator = TrustedRelationEvaluator(
            semantic_registry,
            self.value_codec,
        )
        self.value_registry_digest = identity_digest(
            "veritor/call-dag/value-registry/v1",
            semantic_registry.value_manifest(cell_bits),
        )
        self.operator_registry_digest = identity_digest(
            "veritor/call-dag/operator-registry/v1",
            semantic_registry.operator_manifest,
        )

    @property
    def cached_definition_count(self) -> int:
        return len(self._cache)

    @property
    def cached_digests(self) -> frozenset[str]:
        return frozenset(self._cache)

    def gate_spec(self, operation: str) -> GateSpec:
        try:
            return self._gates[operation]
        except KeyError as error:
            raise KernelReject(f"unknown trusted gate {operation!r}") from error

    def require_validated_definition(
        self,
        definition: ValidatedDefinition,
    ) -> ValidatedDefinition:
        """Require an object previously validated by this exact kernel."""

        if (
            not isinstance(definition, ValidatedDefinition)
            or self._cache.get(definition.digest) is not definition
        ):
            raise KernelReject("definition was not validated by this kernel instance")
        return definition

    @staticmethod
    def _bounded_add(left: int, right: int, limit: int, what: str) -> int:
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
        available: Mapping[str, ValidatedDefinition],
    ) -> ValidatedDefinition:
        body = _require_mapping(
            body_value,
            {"input_count", "steps", "outputs"},
            f"definition {digest}",
        )
        input_count = _require_int(
            body["input_count"],
            f"definition {digest} input_count",
            maximum=self.limits.max_cells,
        )
        raw_steps = _require_list(body["steps"], f"definition {digest} steps")
        if len(raw_steps) > self.limits.max_steps_per_definition:
            raise KernelReject(f"definition {digest} has too many local steps")

        steps: list[_ValidatedStep] = []
        step_outputs: list[tuple[_Location, ...]] = []
        required_inputs: set[int] = set()
        gate_count = 0
        cost = 0
        nesting_depth = 0

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
                        where=(
                            f"definition {digest} leaf {step_index} arg {arg_index}"
                        ),
                    )
                    for arg_index, source in enumerate(raw_args)
                )
                validated_leaf = ValidatedLeaf(gate, args, gate_count)
                required_inputs.update(
                    arg.index for arg in args if isinstance(arg, _InputLocation)
                )
                steps.append(validated_leaf)
                step_outputs.append((_GateLocation(gate_count),))
                gate_count = self._bounded_add(
                    gate_count, 1, self.limits.max_cells, "gate count"
                )
                cost = self._bounded_add(cost, gate.cost, self.limits.max_cost, "cost")
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
                child = available.get(child_digest)
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
                        where=(
                            f"definition {digest} call {step_index} arg {arg_index}"
                        ),
                    )
                    for arg_index, source in enumerate(raw_args)
                )
                validated_call = ValidatedCall(child, args, gate_count)
                for child_input in child.required_inputs:
                    parent_location = args[child_input]
                    if isinstance(parent_location, _InputLocation):
                        required_inputs.add(parent_location.index)
                steps.append(validated_call)
                step_outputs.append(
                    tuple(
                        self._substitute_child_location(
                            output,
                            args,
                            validated_call.gate_start,
                        )
                        for output in child.outputs
                    )
                )
                gate_count = self._bounded_add(
                    gate_count,
                    child.gate_count,
                    self.limits.max_cells,
                    "gate count",
                )
                cost = self._bounded_add(
                    cost,
                    child.cost,
                    self.limits.max_cost,
                    "cost",
                )
                nesting_depth = max(nesting_depth, child.nesting_depth + 1)
                if nesting_depth > self.limits.max_nesting_depth:
                    raise KernelReject(
                        "definition nesting depth exceeds configured limit "
                        f"{self.limits.max_nesting_depth}"
                    )
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
        self._bounded_add(input_count, gate_count, self.limits.max_cells, "cell count")
        return ValidatedDefinition(
            digest=digest,
            input_count=input_count,
            steps=tuple(steps),
            outputs=outputs,
            required_inputs=tuple(sorted(required_inputs)),
            gate_count=gate_count,
            cost=cost,
            nesting_depth=nesting_depth,
        )

    def load(self, blob: bytes) -> LoadReport:
        """Atomically parse and validate one constructor result.

        Definitions are staged in a transaction-local mapping.  The persistent
        cache is updated only after every entry and the selected root have
        validated, so a rejected document has no cache side effects.
        """

        if type(blob) is not bytes:
            raise KernelReject("constructor output must be bytes")
        if len(blob) > self.limits.max_blob_bytes:
            raise KernelReject("constructor output exceeds byte limit")
        try:
            text = blob.decode("utf-8")
            document_value = json.loads(
                text,
                object_pairs_hook=_strict_object,
                parse_constant=_reject_json_constant,
            )
        except KernelReject:
            raise
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            RecursionError,
            ValueError,
        ) as error:
            raise KernelReject("constructor output is not valid UTF-8 JSON") from error
        try:
            canonical = canonical_call_dag_json(document_value)
        except (TypeError, ValueError, UnicodeEncodeError, RecursionError) as error:
            raise KernelReject(
                "constructor output is not canonical JSON data"
            ) from error
        if canonical != blob:
            raise KernelReject("constructor output is not canonically serialized")

        document = _require_mapping(
            document_value,
            {"version", "cell_bits", "definitions", "root"},
            "document",
        )
        version = _require_int(document["version"], "document version")
        if version != FORMAT_VERSION:
            raise KernelReject(f"unsupported format version {version}")
        cell_bits = _require_int(
            document["cell_bits"],
            "document cell_bits",
            minimum=1,
        )
        if cell_bits != self.cell_bits:
            raise KernelReject(
                f"document uses {cell_bits}-bit cells; kernel uses {self.cell_bits}"
            )
        raw_definitions = _require_list(document["definitions"], "definitions")
        if len(raw_definitions) > self.limits.max_definitions_per_blob:
            raise KernelReject("too many definitions in one constructor output")

        available = dict(self._cache)
        pending: dict[str, ValidatedDefinition] = {}
        seen_in_document: set[str] = set()
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
            actual_digest = definition_digest(entry["body"])
            if actual_digest != digest:
                raise KernelReject(f"definition {digest} body digest does not match")
            if digest in self._cache:
                cache_hits += 1
                continue
            validated = self._validate_definition(digest, entry["body"], available)
            pending[digest] = validated
            available[digest] = validated

        root_digest = _require_digest(document["root"], "root")
        root = available.get(root_digest)
        if root is None:
            raise KernelReject("root definition is unavailable")

        self._cache.update(pending)
        return LoadReport(
            root=root,
            serialized_bytes=len(blob),
            new_definitions=len(pending),
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

    def root_outputs(self, root: ValidatedDefinition) -> tuple[int, ...]:
        self.require_validated_definition(root)
        frame = _Frame(root, 0, None, ())
        return tuple(frame.resolve(output, root.input_count) for output in root.outputs)

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
            if isinstance(step, ValidatedLeaf):
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
        """Materialize the conceptual primitive circuit as a reference view."""

        self.require_validated_definition(root)
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

    @staticmethod
    def _definition_occurrence_summary(
        frame: _Frame,
        *,
        path: OccurrencePath,
        kind: str,
        root_input_count: int,
    ) -> OccurrenceSummary:
        definition = frame.definition
        external_reads = tuple(
            sorted(
                {
                    frame.resolve_input(index, root_input_count)
                    for index in definition.required_inputs
                }
            )
        )
        outputs = tuple(
            frame.resolve(output, root_input_count) for output in definition.outputs
        )
        return OccurrenceSummary(
            path=path,
            kind=kind,
            definition_digest=definition.digest,
            gate_start=frame.gate_base,
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
        """Resolve an occurrence path and derive its globally wired interface.

        Cost is ``O(depth * interface)``: argument lists of enclosing calls are
        never materialized.
        """

        self.require_validated_definition(root)
        if type(path) is not tuple or any(
            type(step_index) is not int or step_index < 0 for step_index in path
        ):
            raise KernelReject(
                "an occurrence path must be a tuple of nonnegative step indices"
            )

        root_input_count = root.input_count
        frame = _Frame(root, 0, None, ())
        if not path:
            return self._definition_occurrence_summary(
                frame, path=path, kind="root", root_input_count=root_input_count
            )

        for depth, step_index in enumerate(path):
            definition = frame.definition
            if step_index >= len(definition.steps):
                raise KernelReject(
                    f"occurrence path {path!r} references a missing step"
                )
            step = definition.steps[step_index]
            is_target = depth == len(path) - 1
            if isinstance(step, ValidatedLeaf):
                if not is_target:
                    raise KernelReject(
                        f"occurrence path {path!r} descends through a primitive leaf"
                    )
                reads = tuple(
                    sorted({frame.resolve(arg, root_input_count) for arg in step.args})
                )
                ordinal = frame.gate_base + step.gate_start
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

            frame = _Frame(
                step.child, frame.gate_base + step.gate_start, frame, step.args
            )
            if is_target:
                return self._definition_occurrence_summary(
                    frame, path=path, kind="call", root_input_count=root_input_count
                )

        raise RuntimeError("validated occurrence path did not resolve")

    def leaf_occurrence_paths(
        self,
        root: ValidatedDefinition,
    ) -> tuple[OccurrencePath, ...]:
        """Return every primitive occurrence path in conceptual gate order."""

        self.require_validated_definition(root)
        paths: list[OccurrencePath] = []

        def visit(
            definition: ValidatedDefinition,
            prefix: OccurrencePath,
        ) -> None:
            for step_index, step in enumerate(definition.steps):
                path = (*prefix, step_index)
                if isinstance(step, ValidatedLeaf):
                    paths.append(path)
                else:
                    visit(step.child, path)

        visit(root, ())
        if len(paths) != root.gate_count:
            raise RuntimeError("leaf occurrence traversal count mismatch")
        return tuple(paths)

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
        """Validate an occurrence cut and derive its boundary without flattening."""

        self.require_validated_definition(root)
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
                    "replay-unit cut leaves gate interval "
                    f"[{cursor}, {unit.gate_start})"
                )
            cursor = unit.gate_stop
            total_cost += unit.cost
        if cursor != root.gate_count:
            raise KernelReject(
                f"replay-unit cut leaves gate interval [{cursor}, {root.gate_count})"
            )
        if total_cost != root.cost:
            raise RuntimeError("validated replay-unit costs do not partition root cost")

        root_outputs = self.root_outputs(root)
        intervals: list[tuple[int, int]] = [(0, root.input_count)]
        intervals.extend((item, item + 1) for item in root_outputs)
        for unit in units:
            intervals.extend((item, item + 1) for item in unit.external_reads)
        return ReplayPlan(
            root_digest=root.digest,
            root_input_count=root.input_count,
            root_gate_count=root.gate_count,
            root_outputs=root_outputs,
            units=units,
            boundary=IntervalDomain(intervals),
        )

    def validate_replay_plan(
        self,
        root: ValidatedDefinition,
        plan: ReplayPlan,
    ) -> None:
        """Re-derive a replay plan and require exact circuit-relative agreement."""

        if not isinstance(root, ValidatedDefinition) or not isinstance(
            plan, ReplayPlan
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

    def gate_at(self, root: ValidatedDefinition, gate_ordinal: int) -> FlatGate:
        """Return one conceptual primitive without flattening the circuit.

        The descent keeps a chain of call frames instead of materializing each
        call's argument list, so a lookup costs ``O(depth * arity)`` regardless
        of how many arguments the enclosing calls take.
        """

        self.require_validated_definition(root)
        if type(gate_ordinal) is not int or not 0 <= gate_ordinal < root.gate_count:
            raise KernelReject("gate ordinal is out of range")
        root_input_count = root.input_count
        frame = _Frame(root, 0, None, ())
        local_ordinal = gate_ordinal
        while True:
            step = frame.definition.step_owning(local_ordinal)
            if isinstance(step, ValidatedLeaf):
                ordinal = frame.gate_base + local_ordinal
                return FlatGate(
                    ordinal=ordinal,
                    write=root_input_count + ordinal,
                    function=step.gate.name,
                    reads=tuple(
                        frame.resolve(arg, root_input_count) for arg in step.args
                    ),
                )
            frame = _Frame(
                step.child, frame.gate_base + step.gate_start, frame, step.args
            )
            local_ordinal -= step.gate_start

    def apply_gate(self, function: str, args: Sequence[int]) -> int:
        return self.relation_evaluator.evaluate_operation(function, args)

    def evaluate_relation(self, relation_id: str, args: Sequence[int]) -> int:
        return self.relation_evaluator.evaluate(relation_id, args)

    def evaluate_tape(
        self,
        root: ValidatedDefinition,
        inputs: Sequence[int],
    ) -> tuple[int, ...]:
        self.require_validated_definition(root)
        if len(inputs) != root.input_count:
            raise KernelReject(
                f"circuit expects {root.input_count} inputs, got {len(inputs)}"
            )
        cells = [
            self.value_codec.validate(value, where=f"input {index}")
            for index, value in enumerate(inputs)
        ]
        flat = self.flatten(root)
        for gate in flat.gates:
            if gate.write != len(cells):
                raise RuntimeError("internal flat write position mismatch")
            cells.append(
                self.apply_gate(
                    gate.function,
                    tuple(cells[index] for index in gate.reads),
                )
            )
        return tuple(cells)

    def evaluate(
        self,
        root: ValidatedDefinition,
        inputs: Sequence[int],
    ) -> tuple[int, ...]:
        cells = self.evaluate_tape(root, inputs)
        return tuple(cells[position] for position in self.root_outputs(root))


class CallDagCircuit:
    """Adapter from a validated call DAG to :class:`core.ExecutableCircuit`."""

    __slots__ = (
        "_computed_positions",
        "_input_ports",
        "_kernel",
        "_output_ports",
        "_root",
        "identity",
    )

    def __init__(self, kernel: Kernel, root: ValidatedDefinition) -> None:
        if not isinstance(kernel, Kernel) or not isinstance(root, ValidatedDefinition):
            raise TypeError("CallDagCircuit requires a Kernel and validated root")
        kernel.require_validated_definition(root)
        self._kernel = kernel
        self._root = root
        value_type = kernel.value_codec.value_type_id
        self._computed_positions = RangeIndexedDomain(
            root.input_count,
            root.input_count + root.gate_count,
        )
        self._input_ports = tuple(
            Port(f"input_{index}", index, value_type)
            for index in range(root.input_count)
        )
        self._output_ports = tuple(
            Port(f"output_{index}", position, value_type)
            for index, position in enumerate(kernel.root_outputs(root))
        )
        representation_manifest: dict[str, JSONValue] = {
            "call_dag_format_version": FORMAT_VERSION,
            "compiler_id": COMPILER_ID,
            "compiler_version": COMPILER_VERSION,
            "cost": root.cost,
            "gate_count": root.gate_count,
            "input_count": root.input_count,
            "ordered_outputs": list(kernel.root_outputs(root)),
            "root_definition_digest": root.digest,
        }
        self.identity = StructureIdentity.from_manifest(
            representation_manifest,
            schema_version=STRUCTURE_SCHEMA_VERSION,
            artifact_kind=ArtifactKind.EXECUTABLE_CIRCUIT,
            compiler_id=COMPILER_ID,
            compiler_version=COMPILER_VERSION,
            semantic_scope_id=kernel.semantic_registry.semantic_scope_id,
            value_registry_digest=kernel.value_registry_digest,
            operator_registry_digest=kernel.operator_registry_digest,
        )

    @property
    def kernel(self) -> Kernel:
        return self._kernel

    @property
    def root(self) -> ValidatedDefinition:
        return self._root

    @property
    def computed_positions(self) -> RangeIndexedDomain:
        return self._computed_positions

    @property
    def input_ports(self) -> tuple[Port, ...]:
        return self._input_ports

    @property
    def output_ports(self) -> tuple[Port, ...]:
        return self._output_ports

    @property
    def cell_bits(self) -> int:
        return self._kernel.cell_bits

    @property
    def value_codec(self) -> WordValueCodec:
        return self._kernel.value_codec

    @property
    def relation_evaluator(self) -> TrustedRelationEvaluator:
        return self._kernel.relation_evaluator

    @property
    def gate_count(self) -> int:
        return self._root.gate_count

    @property
    def input_count(self) -> int:
        return self._root.input_count

    @property
    def cost(self) -> int:
        return self._root.cost

    @property
    def ordered_output_positions(self) -> tuple[int, ...]:
        return tuple(port.position for port in self._output_ports)

    def _flat_gate_at_position(self, position: int) -> FlatGate:
        try:
            ordinal = self._computed_positions.rank(position)
        except KeyError as error:
            raise KeyError(position) from error
        return self._kernel.gate_at(self._root, ordinal)

    def gate_at(self, position: int) -> StructuralGate:
        gate = self._flat_gate_at_position(position)
        spec = self._kernel.gate_spec(gate.function)
        return StructuralGate(
            gate.write,
            gate.function,
            gate.reads,
            self.value_codec.cardinality,
            value_type=self.value_codec.value_type_id,
            metadata={"arity": spec.arity, "cost": spec.cost},
        )

    def executable_gate_at(self, position: int) -> ExecutableGate:
        gate = self._flat_gate_at_position(position)
        spec = self._kernel.gate_spec(gate.function)
        return ExecutableGate(
            gate.write,
            gate.function,
            gate.reads,
            self.value_codec.value_type_id,
            self._kernel.semantic_registry.relation_id(gate.function),
            metadata={"arity": spec.arity, "cost": spec.cost},
        )

    # -- trusted semantics (core.ExecutableCircuit) --------------------------

    def _require_value_type(self, value_type: str) -> None:
        if value_type != self.value_codec.value_type_id:
            raise KernelReject(f"unknown value type {value_type!r}")

    def encode_value(self, value_type: str, value: object) -> bytes:
        self._require_value_type(value_type)
        return self.value_codec.encode(value)

    def decode_value(self, value_type: str, payload: bytes) -> int:
        self._require_value_type(value_type)
        return self.value_codec.decode(payload)

    def evaluate_relation(self, relation_id: str, arguments: Sequence[object]) -> int:
        return self.relation_evaluator.evaluate(
            relation_id, tuple(self.value_codec.validate(item) for item in arguments)
        )

    def check_relation(
        self,
        relation_id: str,
        arguments: Sequence[object],
        output: object,
    ) -> bool:
        return self.evaluate_relation(relation_id, arguments) == output

    def flatten(self) -> FlatCircuit:
        return self._kernel.flatten(self._root)

    def evaluate_tape(self, inputs: Sequence[int]) -> tuple[int, ...]:
        return self._kernel.evaluate_tape(self._root, inputs)

    def evaluate(self, inputs: Sequence[int]) -> tuple[int, ...]:
        return self._kernel.evaluate(self._root, inputs)

    def occurrence_summary(self, path: OccurrencePath) -> OccurrenceSummary:
        return self._kernel.occurrence_summary(self._root, path)


def construct(
    kernel: Kernel,
    constructor: Callable[[object, bytes], bytes],
    x: object,
    a: bytes,
    *,
    input_cells: Sequence[int],
    advice_bound_bits: int,
) -> Construction:
    """Run the constructor boundary and validate its data-only result."""

    if type(a) is not bytes:
        raise KernelReject("advice must be bytes")
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
        kernel.value_codec.validate(value, where=f"input {index}")
    return Construction(load=load, input_cells=cells, constructor_output=blob)


# ---------------------------------------------------------------------------
# Modular word arithmetic: the trusted semantics shared by the built-in plug-ins
# ---------------------------------------------------------------------------


def trusted_word_gates(cell_bits: int) -> tuple[GateSpec, ...]:
    """Return the demo's trusted modular add/multiply declarations."""

    if type(cell_bits) is not int or cell_bits <= 0:
        raise ValueError("cell_bits must be positive")
    mask = (1 << cell_bits) - 1
    return (
        GateSpec("add", 2, 1, lambda args: (args[0] + args[1]) & mask),
        GateSpec("mul", 2, 2, lambda args: (args[0] * args[1]) & mask),
    )


def trusted_word_registry(
    cell_bits: int,
    *,
    registry_id: str = "veritor.demo.word-arithmetic",
    registry_version: str = "1",
) -> SemanticRegistry:
    """Build the explicit semantic registry for :class:`DemoG`."""

    return SemanticRegistry(
        registry_id=registry_id,
        registry_version=registry_version,
        value_schema_id="veritor.unsigned-word",
        value_schema_version="1",
        gates=trusted_word_gates(cell_bits),
    )


def make_word_kernel(
    cell_bits: int = 8,
    *,
    registry_id: str = "veritor.demo.word-arithmetic",
    registry_version: str = "1",
    limits: CompilationLimits | None = None,
) -> Kernel:
    """Construct a kernel for the self-contained modular-word demo."""

    return Kernel(
        cell_bits=cell_bits,
        semantic_registry=trusted_word_registry(
            cell_bits,
            registry_id=registry_id,
            registry_version=registry_version,
        ),
        limits=limits,
    )
