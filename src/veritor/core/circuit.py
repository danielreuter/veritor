"""The circuit ``C``: one interface, a flat and a lazy implementation.

Addresses ``0 .. n-1`` name every value of the computation.  ``C[i]`` is a
:class:`GateRef`; inputs are gates with op ``"input"`` and no arguments, so
they are members of the address space (and of units) like any other gate.

For a set ``S`` of addresses (a node of the index, or an interval for the
flat circuit):

* ``In(S)``  -- addresses outside ``S`` read by gates in ``S``;
* ``Out(S)`` -- ``S``'s interface.  For the lazy circuit this is the
  *declared* interface of the definition (its outputs resolved to the gates
  the copy owns, a sequence resolved lazily from the definition's runs), a
  superset of the addresses actually read from outside; the flat circuit
  scans and returns exactly the addresses read from outside plus the circuit
  outputs in ``S``.  The declared interface is what the boundary commits to,
  so it is the one the protocol uses;
* ``Size(S)``, ``Cost(S, kind)`` -- address count and summed gate costs.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from .description import Definition, Frame
from .errors import InvalidArtifact
from .gates import GateSet, decode_value, encode_value

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

    def Out(self, subset: object) -> Sequence[int]: ...

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


class _LazyAddresses(Sequence[int]):
    """Addresses resolved on demand from their position: the root's declared
    outputs, or ``Out`` of a copy in the run order of its definition."""

    __slots__ = ("_at", "_count")

    def __init__(self, count: int, at: Callable[[int], int]) -> None:
        self._count = count
        self._at = at

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, index):  # type: ignore[override]
        if isinstance(index, slice):
            return tuple(self[k] for k in range(*index.indices(len(self))))
        if type(index) is not int:
            raise TypeError("output ordinals are integers")
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self._at(index)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Sequence) and tuple(self) == tuple(other)

    def __hash__(self) -> int:
        return hash(tuple(self))

    def __repr__(self) -> str:
        return f"addresses({len(self)})"


class DescriptionCircuit(_Semantics):
    """The lazy circuit of a validated description.

    ``C[i]`` descends from the root in ``O(depth * arity)``.  ``In``/``Out``/
    ``Size``/``Cost`` take an index node (any object carrying a ``frame``) and
    resolve the definition's summaries through that frame.  ``Out`` is a lazy
    sequence over the definition's runs (``O(log #runs)`` per address);
    ``In`` enumerates what the copy reads, ``Theta(|In|)``, so the protocol
    asks it only about sampled units.
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
        self._outputs = _LazyAddresses(root.output_count, self.frame.output_address)

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

    def Out(self, subset: object) -> Sequence[int]:
        frame = _frame(subset)
        definition = frame.definition
        return _LazyAddresses(
            definition.out_count, lambda rank: frame.base + definition.out_offset(rank)
        )

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

