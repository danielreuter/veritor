"""The circuit ``C``: one interface, a flat and a lazy implementation.

Addresses ``0 .. n-1`` name every value of the computation.  ``C[i]`` is a
:class:`GateRef`; inputs are gates with op ``"input"`` and no arguments, so
they are members of the address space (and of units) like any other gate.

For a set ``S`` of addresses (a node of the index, or an interval for the
flat circuit):

* ``In(S)``  -- addresses outside ``S`` read by gates in ``S``;
* ``Out(S)`` -- ``S``'s interface.  For the lazy circuit this is the
  *declared* interface of the definition (its outputs resolved to the gates
  the copy owns), a superset of the addresses actually read from outside;
  the flat circuit scans and returns exactly the addresses read from outside
  plus the circuit outputs in ``S``.  The declared interface is what the
  boundary commits to, so it is the one the protocol uses;
* ``Size(S)``, ``Cost(S, kind)`` -- address count and summed gate costs.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from .capabilities import ArtifactKind
from .description import Definition, Frame
from .errors import InvalidArtifact
from .gates import GateSet, decode_value, encode_value
from .identity import JSONScalar, StructureIdentity
from .ids import (
    OperationId,
    Position,
    RelationId,
    ValueTypeId,
    nonempty_identifier,
)
from .ids import position as _as_position
from .indexed import IndexedDomain, iter_domain, position_domain

INPUT_OP = "input"
type CostKind = Literal["replay", "proof"]


@dataclass(frozen=True, slots=True)
class GateRef:
    """``C[i]``: an op name, absolute argument addresses and an output width."""

    op: str
    args: tuple[int, ...]
    width: int

    @property
    def is_input(self) -> bool:
        return self.op == INPUT_OP


def _interval(subset: object) -> range:
    if isinstance(subset, range):
        return subset
    interval = getattr(subset, "interval", None)
    if isinstance(interval, range):
        return interval
    raise TypeError("expected an index node or an address interval")


def _frame(subset: object) -> Frame:
    if isinstance(subset, Frame):
        return subset
    frame = getattr(subset, "frame", None)
    if isinstance(frame, Frame):
        return frame
    raise TypeError("the lazy circuit answers In/Out/Size/Cost for index nodes")


@runtime_checkable
class Circuit(Protocol):
    """The paper's ``C``: addresses, gates, interfaces and gate semantics."""

    @property
    def n(self) -> int: ...

    @property
    def input_count(self) -> int: ...

    @property
    def inputs(self) -> tuple[int, ...]: ...

    @property
    def outputs(self) -> Sequence[int]: ...

    def __getitem__(self, address: int) -> GateRef: ...

    def In(self, subset: object) -> tuple[int, ...]: ...

    def Out(self, subset: object) -> tuple[int, ...]: ...

    def Size(self, subset: object) -> int: ...

    def Cost(self, subset: object, kind: CostKind = "replay") -> int: ...

    def evaluate_gate(self, address: int, args: Sequence[int]) -> int: ...

    def check_gate(self, address: int, args: Sequence[int], out: int) -> bool: ...

    def encode(self, address: int, value: object) -> bytes: ...

    def decode(self, address: int, payload: bytes) -> int: ...

    def evaluate(self, inputs: Sequence[int]) -> tuple[int, ...]: ...


class _Semantics:
    """Gate semantics shared by both implementations, bound to a gate set."""

    gate_set: GateSet

    def __getitem__(self, address: int) -> GateRef:
        raise NotImplementedError

    def evaluate_gate(self, address: int, args: Sequence[int]) -> int:
        ref = self[address]
        if ref.is_input:
            raise InvalidArtifact(f"address {address} is an input, not a gate")
        return self.gate_set[ref.op].evaluate(args)

    def check_gate(self, address: int, args: Sequence[int], out: int) -> bool:
        ref = self[address]
        if ref.is_input:
            raise InvalidArtifact(f"address {address} is an input, not a gate")
        return self.gate_set[ref.op].check(args, out)

    def encode(self, address: int, value: object) -> bytes:
        return encode_value(self[address].width, value)

    def decode(self, address: int, payload: bytes) -> int:
        return decode_value(self[address].width, payload)

    def _cost(self, ref: GateRef, kind: CostKind) -> int:
        if ref.is_input:
            return 0
        gate = self.gate_set[ref.op]
        if kind == "replay":
            return gate.replay_cost
        if kind == "proof":
            return gate.proof_cost
        raise ValueError(f"unknown cost kind {kind!r}")


class FlatCircuit(_Semantics):
    """An explicit gate list; the reference implementation used in tests.

    Inputs may sit anywhere in the address space.  ``In``/``Out``/``Size``/
    ``Cost`` accept an interval or an index node and are computed by scanning.
    """

    __slots__ = ("_gates", "_inputs", "_outputs", "gate_set")

    def __init__(
        self,
        gates: Sequence[GateRef],
        outputs: Sequence[int],
        gate_set: GateSet,
    ) -> None:
        self._gates = tuple(gates)
        for address, ref in enumerate(self._gates):
            if not isinstance(ref, GateRef):
                raise TypeError("FlatCircuit gates must be GateRef values")
            if ref.is_input:
                if ref.args:
                    raise InvalidArtifact(f"input {address} cannot have arguments")
                continue
            if ref.op not in gate_set:
                raise InvalidArtifact(f"address {address} uses unknown gate {ref.op!r}")
            if len(ref.args) != gate_set[ref.op].arity:
                raise InvalidArtifact(f"address {address} has the wrong arity")
            if any(not 0 <= arg < address for arg in ref.args):
                raise InvalidArtifact(f"address {address} reads a later address")
        self._outputs = tuple(outputs)
        if any(not 0 <= out < len(self._gates) for out in self._outputs):
            raise InvalidArtifact("an output names an address outside the circuit")
        self._inputs = tuple(i for i, ref in enumerate(self._gates) if ref.is_input)
        self.gate_set = gate_set

    @property
    def n(self) -> int:
        return len(self._gates)

    @property
    def input_count(self) -> int:
        return len(self._inputs)

    @property
    def inputs(self) -> tuple[int, ...]:
        return self._inputs

    @property
    def outputs(self) -> tuple[int, ...]:
        return self._outputs

    def __getitem__(self, address: int) -> GateRef:
        if type(address) is not int or not 0 <= address < len(self._gates):
            raise IndexError(address)
        return self._gates[address]

    def In(self, subset: object) -> tuple[int, ...]:
        interval = _interval(subset)
        return tuple(
            sorted(
                {
                    arg
                    for address in interval
                    for arg in self._gates[address].args
                    if arg not in interval
                }
            )
        )

    def Out(self, subset: object) -> tuple[int, ...]:
        interval = _interval(subset)
        read = {
            arg
            for address in range(len(self._gates))
            if address not in interval
            for arg in self._gates[address].args
            if arg in interval
        }
        read.update(out for out in self._outputs if out in interval)
        return tuple(sorted(read))

    def Size(self, subset: object) -> int:
        return len(_interval(subset))

    def Cost(self, subset: object, kind: CostKind = "replay") -> int:
        return sum(self._cost(self._gates[a], kind) for a in _interval(subset))

    def evaluate(self, inputs: Sequence[int]) -> tuple[int, ...]:
        if len(inputs) != len(self._inputs):
            raise InvalidArtifact(f"expected {len(self._inputs)} inputs")
        values: list[int] = []
        given = iter(inputs)
        for ref in self._gates:
            if ref.is_input:
                values.append(next(given))
            else:
                values.append(
                    self.gate_set[ref.op].evaluate(tuple(values[a] for a in ref.args))
                )
        return tuple(values)


class _LazyOutputs(Sequence[int]):
    """The root's declared outputs as addresses, resolved on demand."""

    __slots__ = ("_frame",)

    def __init__(self, frame: Frame) -> None:
        self._frame = frame

    def __len__(self) -> int:
        return self._frame.definition.output_count

    def __getitem__(self, index):  # type: ignore[override]
        if isinstance(index, slice):
            return tuple(self[k] for k in range(*index.indices(len(self))))
        if type(index) is not int:
            raise TypeError("output ordinals are integers")
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self._frame.output_address(index)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Sequence) and tuple(self) == tuple(other)

    def __hash__(self) -> int:
        return hash(tuple(self))

    def __repr__(self) -> str:
        return f"outputs({len(self)})"


class DescriptionCircuit(_Semantics):
    """The lazy circuit of a validated description.

    ``C[i]`` descends from the root in ``O(depth * arity)``.  ``In``/``Out``/
    ``Size``/``Cost`` take an index node (any object carrying a ``frame``) and
    resolve the definition's summaries through that frame in
    ``O(depth * |interface|)``.
    """

    __slots__ = ("_outputs", "frame", "gate_set", "root", "width")

    def __init__(self, root: Definition, gate_set: GateSet) -> None:
        widths = {gate.width for gate in gate_set}
        if len(widths) != 1:
            raise InvalidArtifact("the description circuit needs gates of one width")
        self.root = root
        self.gate_set = gate_set
        self.width = widths.pop()
        self.frame = Frame.root(root)
        self._outputs = _LazyOutputs(self.frame)

    @property
    def n(self) -> int:
        return self.root.input_count + self.root.size

    @property
    def input_count(self) -> int:
        return self.root.input_count

    @property
    def inputs(self) -> tuple[int, ...]:
        return tuple(range(self.root.input_count))

    @property
    def outputs(self) -> Sequence[int]:
        return self._outputs

    def __getitem__(self, address: int) -> GateRef:
        if type(address) is not int or not 0 <= address < self.n:
            raise IndexError(address)
        if address < self.root.input_count:
            return GateRef(INPUT_OP, (), self.width)
        gate, args = self.frame.gate(address)
        return GateRef(gate.name, args, gate.width)

    def In(self, subset: object) -> tuple[int, ...]:
        frame = _frame(subset)
        return tuple(
            sorted({frame.input_address(i) for i in frame.definition.reads})
        )

    def Out(self, subset: object) -> tuple[int, ...]:
        frame = _frame(subset)
        return tuple(frame.base + offset for offset in frame.definition.local_outputs)

    def Size(self, subset: object) -> int:
        return _frame(subset).definition.size

    def Cost(self, subset: object, kind: CostKind = "replay") -> int:
        definition = _frame(subset).definition
        if kind == "replay":
            return definition.replay_cost
        if kind == "proof":
            return definition.proof_cost
        raise ValueError(f"unknown cost kind {kind!r}")

    def evaluate(self, inputs: Sequence[int]) -> tuple[int, ...]:
        if len(inputs) != self.root.input_count:
            raise InvalidArtifact(f"expected {self.root.input_count} inputs")
        values = list(inputs)
        for address in range(self.root.input_count, self.n):
            gate, args = self.frame.gate(address)
            values.append(gate.evaluate(tuple(values[a] for a in args)))
        return tuple(values)


def _metadata(
    value: Mapping[str, JSONScalar] | Iterable[tuple[str, JSONScalar]],
) -> tuple[tuple[str, JSONScalar], ...]:
    pairs = tuple(value.items()) if isinstance(value, Mapping) else tuple(value)
    result: dict[str, JSONScalar] = {}
    for pair in pairs:
        if type(pair) is not tuple or len(pair) != 2:
            raise InvalidArtifact("metadata entries must be (key, value) pairs")
        key, item = pair
        if type(key) is not str or not key:
            raise InvalidArtifact("metadata keys must be nonempty strings")
        if type(item) not in (type(None), bool, int, str):
            raise InvalidArtifact("metadata values must be canonical JSON scalars")
        if key in result:
            raise InvalidArtifact(f"duplicate metadata key {key!r}")
        result[key] = item
    return tuple(sorted(result.items()))


@dataclass(frozen=True, slots=True, init=False)
class Port:
    """A named typed circuit input or output port."""

    name: str
    position: Position
    value_type: ValueTypeId

    def __init__(self, name: str, position: int, value_type: str) -> None:
        object.__setattr__(
            self, "name", nonempty_identifier(name, field_name="port name")
        )
        object.__setattr__(
            self,
            "position",
            _as_position(position, field_name=f"port {name!r} position"),
        )
        object.__setattr__(
            self,
            "value_type",
            ValueTypeId(nonempty_identifier(value_type, field_name="value_type")),
        )


@dataclass(frozen=True, slots=True, init=False)
class StructuralGate:
    """Static dependency and value-domain metadata for one computed position.

    ``predecessors`` is a tuple so adapters may retain operand order and
    repeated reads even when a downstream structural algorithm treats it as a
    set. ``capacity_upper_bound`` is an exact finite value-cardinality bound.
    """

    position: Position
    operation: OperationId
    predecessors: tuple[Position, ...]
    capacity_upper_bound: int | None
    value_type: ValueTypeId | None
    metadata: tuple[tuple[str, JSONScalar], ...]

    def __init__(
        self,
        position: int,
        operation: str,
        predecessors: Iterable[int],
        capacity_upper_bound: int | None = None,
        *,
        value_type: str | None = None,
        metadata: Mapping[str, JSONScalar] | Iterable[tuple[str, JSONScalar]] = (),
    ) -> None:
        checked_position = _as_position(position)
        checked_operation = OperationId(
            nonempty_identifier(operation, field_name="operation")
        )
        checked_predecessors = tuple(
            _as_position(item, field_name=f"predecessors[{index}]")
            for index, item in enumerate(predecessors)
        )
        if capacity_upper_bound is not None and (
            type(capacity_upper_bound) is not int or capacity_upper_bound < 1
        ):
            raise InvalidArtifact(
                "capacity_upper_bound must be a positive integer or None"
            )
        checked_value_type = (
            None
            if value_type is None
            else ValueTypeId(nonempty_identifier(value_type, field_name="value_type"))
        )
        object.__setattr__(self, "position", checked_position)
        object.__setattr__(self, "operation", checked_operation)
        object.__setattr__(self, "predecessors", checked_predecessors)
        object.__setattr__(self, "capacity_upper_bound", capacity_upper_bound)
        object.__setattr__(self, "value_type", checked_value_type)
        object.__setattr__(self, "metadata", _metadata(metadata))

    @property
    def value_cardinality_upper_bound(self) -> int | None:
        """Descriptive alias for ``capacity_upper_bound``."""

        return self.capacity_upper_bound


@dataclass(frozen=True, slots=True, init=False)
class ExecutableGate:
    """One trusted local relation with ordered, duplicate-preserving arguments."""

    position: Position
    operation: OperationId
    arguments: tuple[Position, ...]
    output_type: ValueTypeId
    relation_id: RelationId
    metadata: tuple[tuple[str, JSONScalar], ...]

    def __init__(
        self,
        position: int,
        operation: str,
        arguments: Iterable[int],
        output_type: str,
        relation_id: str,
        *,
        metadata: Mapping[str, JSONScalar] | Iterable[tuple[str, JSONScalar]] = (),
    ) -> None:
        object.__setattr__(self, "position", _as_position(position))
        object.__setattr__(
            self,
            "operation",
            OperationId(nonempty_identifier(operation, field_name="operation")),
        )
        object.__setattr__(
            self,
            "arguments",
            tuple(
                _as_position(item, field_name=f"arguments[{index}]")
                for index, item in enumerate(arguments)
            ),
        )
        object.__setattr__(
            self,
            "output_type",
            ValueTypeId(nonempty_identifier(output_type, field_name="output_type")),
        )
        object.__setattr__(
            self,
            "relation_id",
            RelationId(nonempty_identifier(relation_id, field_name="relation_id")),
        )
        object.__setattr__(self, "metadata", _metadata(metadata))

    @property
    def predecessors(self) -> tuple[Position, ...]:
        return self.arguments


@runtime_checkable
class StructuralCircuit(Protocol):
    """A finite or lazily indexed circuit sufficient for structural analysis."""

    @property
    def identity(self) -> StructureIdentity: ...

    @property
    def computed_positions(self) -> IndexedDomain[Position]: ...

    @property
    def input_ports(self) -> tuple[Port, ...]: ...

    @property
    def output_ports(self) -> tuple[Port, ...]: ...

    def gate_at(self, position: Position) -> StructuralGate: ...


@runtime_checkable
class ExecutableCircuit(StructuralCircuit, Protocol):
    """A structural circuit that also carries its trusted local semantics.

    ``encode_value``/``decode_value`` define the canonical byte form of every
    value type the circuit uses; ``decode_value`` must reject any payload that
    ``encode_value`` would not produce.  ``evaluate_relation`` computes a gate's
    output (used by an honest prover to replay) and ``check_relation`` decides
    whether a claimed output satisfies the relation (used by the verifier).
    """

    def executable_gate_at(self, position: Position) -> ExecutableGate: ...

    def encode_value(self, value_type: str, value: object) -> bytes: ...

    def decode_value(self, value_type: str, payload: bytes) -> object: ...

    def evaluate_relation(
        self, relation_id: str, arguments: tuple[object, ...]
    ) -> object: ...

    def check_relation(
        self, relation_id: str, arguments: tuple[object, ...], output: object
    ) -> bool: ...


def ordered_output_positions(circuit: StructuralCircuit) -> tuple[Position, ...]:
    """Return outputs exactly as declared, retaining order and duplicates."""

    if type(circuit.output_ports) is not tuple:
        raise InvalidArtifact("circuit output_ports must be a tuple")
    return tuple(port.position for port in circuit.output_ports)


def validate_circuit_contract(
    circuit: StructuralCircuit,
    *,
    exhaustive: bool = False,
) -> None:
    """Validate representation-independent circuit invariants.

    Exhaustive gate lookup is opt-in so this function remains usable with very
    large lazy circuits.
    """

    if not isinstance(circuit.identity, StructureIdentity):
        raise InvalidArtifact("circuit identity must be a StructureIdentity")
    if circuit.identity.artifact_kind is ArtifactKind.CAPACITY_PROFILE:
        raise InvalidArtifact("a capacity profile is not a structural circuit")
    computed = position_domain(
        circuit.computed_positions,
        field_name="circuit computed_positions",
    )
    if type(circuit.input_ports) is not tuple:
        raise InvalidArtifact("circuit input_ports must be a tuple")
    if type(circuit.output_ports) is not tuple:
        raise InvalidArtifact("circuit output_ports must be a tuple")
    inputs: set[Position] = set()
    for port in circuit.input_ports:
        if not isinstance(port, Port):
            raise InvalidArtifact("every input port must be a Port")
        if port.position in inputs:
            raise InvalidArtifact("input positions must be unique")
        if computed.contains(port.position):
            raise InvalidArtifact("input and computed positions must be disjoint")
        inputs.add(port.position)
    for port in circuit.output_ports:
        if not isinstance(port, Port):
            raise InvalidArtifact("every output port must be a Port")
        if port.position not in inputs and not computed.contains(port.position):
            raise InvalidArtifact("an output port references an unknown position")
    if not exhaustive:
        return
    known = inputs | set(iter_domain(computed))
    for expected_position in iter_domain(computed):
        gate = circuit.gate_at(expected_position)
        if not isinstance(gate, StructuralGate):
            raise InvalidArtifact("gate_at must return StructuralGate values")
        if gate.position != expected_position:
            raise InvalidArtifact("gate_at returned a gate for another position")
        if any(predecessor not in known for predecessor in gate.predecessors):
            raise InvalidArtifact("a structural gate references an unknown position")
