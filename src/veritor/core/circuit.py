"""The circuit ``C``: one interface, a flat and a lazy implementation.

Addresses ``0 .. n-1`` name every value of the computation.  ``C[i]`` is a
:class:`GateRef`; the circuit's inputs and weights are *source* gates (the
gate set's ``in`` and ``weight`` gates) with no arguments, so they are members
of the address space (and of units) like any other gate.  ``inputs`` and
``weights`` list their addresses by rank in address order; ``evaluate``
assigns them from ``x`` and ``W`` by rank.

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

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from typing import Literal, Protocol, runtime_checkable

from .description import Definition, Frame
from .errors import InvalidArtifact
from .gates import (
    INPUT_SOURCE,
    WEIGHT_SOURCE,
    GateSet,
    check_value,
    decode_value,
    encode_value,
)

type CostKind = Literal["replay", "proof"]


@dataclass(frozen=True, slots=True)
class GateRef:
    """``C[i]``: an op name, absolute argument addresses, an output width and a source.

    ``source`` is ``"input"`` or ``"weight"`` for a source gate (no
    arguments; its value comes from ``x`` or ``W``), ``None`` otherwise.
    """

    op: str
    args: tuple[int, ...]
    width: int
    source: str | None = None

    @property
    def is_source(self) -> bool:
        return self.source is not None

    @property
    def is_input(self) -> bool:
        return self.source == INPUT_SOURCE

    @property
    def is_weight(self) -> bool:
        return self.source == WEIGHT_SOURCE


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
    def gate_set(self) -> GateSet: ...

    @property
    def n(self) -> int: ...

    @property
    def input_count(self) -> int: ...

    @property
    def weight_count(self) -> int: ...

    @property
    def inputs(self) -> Sequence[int]: ...

    @property
    def weights(self) -> Sequence[int]: ...

    @property
    def outputs(self) -> Sequence[int]: ...

    def input_rank(self, address: int) -> int: ...

    def weight_rank(self, address: int) -> int: ...

    def __getitem__(self, address: int) -> GateRef: ...

    def In(self, subset: object) -> tuple[int, ...]: ...

    def Out(self, subset: object) -> Sequence[int]: ...

    def Size(self, subset: object) -> int: ...

    def Cost(self, subset: object, kind: CostKind = "replay") -> int: ...

    def evaluate_gate(self, address: int, args: Sequence[int]) -> int: ...

    def check_gate(self, address: int, args: Sequence[int], out: int) -> bool: ...

    def encode(self, address: int, value: object) -> bytes: ...

    def decode(self, address: int, payload: bytes) -> int: ...

    def evaluate(
        self, inputs: Sequence[int], weights: Sequence[int] = ()
    ) -> tuple[int, ...]: ...


class _Semantics:
    """Gate semantics shared by both implementations, bound to a gate set."""

    gate_set: GateSet

    def __getitem__(self, address: int) -> GateRef:
        raise NotImplementedError

    def _relation(self, address: int) -> GateRef:
        ref = self[address]
        if ref.is_source:
            raise InvalidArtifact(
                f"address {address} is a source gate ({ref.op}); "
                "its value comes from the environment, not a relation"
            )
        return ref

    def evaluate_gate(self, address: int, args: Sequence[int]) -> int:
        return self.gate_set[self._relation(address).op].evaluate(args)

    def check_gate(self, address: int, args: Sequence[int], out: int) -> bool:
        return self.gate_set[self._relation(address).op].check(args, out)

    def encode(self, address: int, value: object) -> bytes:
        return encode_value(self[address].width, value)

    def decode(self, address: int, payload: bytes) -> int:
        return decode_value(self[address].width, payload)

    def _cost(self, ref: GateRef, kind: CostKind) -> int:
        gate = self.gate_set[ref.op]
        if kind == "replay":
            return gate.replay_cost
        if kind == "proof":
            return gate.proof_cost
        raise ValueError(f"unknown cost kind {kind!r}")


def _checked_sources(values: Sequence[int], count: int, what: str) -> Sequence[int]:
    if len(values) != count:
        raise InvalidArtifact(f"expected {count} {what}, got {len(values)}")
    return values


def _source_value(given: dict[str, Iterator[int]], source: str, width: int) -> int:
    """The next value of ``source`` by rank, checked to be a ``width``-bit value."""

    return check_value(width, next(given[source]), where=f"{source} value")


class FlatCircuit(_Semantics):
    """An explicit gate list; the reference implementation used in tests.

    Source gates may sit anywhere in the address space; a ``GateRef`` whose
    op is a source gate of the gate set is completed with that source.
    ``In``/``Out``/``Size``/``Cost`` accept an interval or an index node and
    are computed by scanning.
    """

    __slots__ = ("_gates", "_inputs", "_outputs", "_ranks", "_weights", "gate_set")

    def __init__(
        self,
        gates: Sequence[GateRef],
        outputs: Sequence[int],
        gate_set: GateSet,
    ) -> None:
        completed: list[GateRef] = []
        for address, ref in enumerate(gates):
            if not isinstance(ref, GateRef):
                raise TypeError("FlatCircuit gates must be GateRef values")
            if ref.op not in gate_set:
                raise InvalidArtifact(f"address {address} uses unknown gate {ref.op!r}")
            gate = gate_set[ref.op]
            if ref.source is not None and ref.source != gate.source:
                raise InvalidArtifact(
                    f"address {address} misstates the source of {ref.op!r}"
                )
            if len(ref.args) != gate.arity:
                raise InvalidArtifact(f"address {address} has the wrong arity")
            if any(not 0 <= arg < address for arg in ref.args):
                raise InvalidArtifact(f"address {address} reads a later address")
            completed.append(replace(ref, source=gate.source))
        self._gates = tuple(completed)
        self._outputs = tuple(outputs)
        if any(not 0 <= out < len(self._gates) for out in self._outputs):
            raise InvalidArtifact("an output names an address outside the circuit")
        self._inputs = tuple(i for i, ref in enumerate(self._gates) if ref.is_input)
        self._weights = tuple(i for i, ref in enumerate(self._gates) if ref.is_weight)
        self._ranks = {
            INPUT_SOURCE: {address: rank for rank, address in enumerate(self._inputs)},
            WEIGHT_SOURCE: {
                address: rank for rank, address in enumerate(self._weights)
            },
        }
        self.gate_set = gate_set

    @property
    def n(self) -> int:
        return len(self._gates)

    @property
    def input_count(self) -> int:
        return len(self._inputs)

    @property
    def weight_count(self) -> int:
        return len(self._weights)

    @property
    def inputs(self) -> tuple[int, ...]:
        return self._inputs

    @property
    def weights(self) -> tuple[int, ...]:
        return self._weights

    @property
    def outputs(self) -> tuple[int, ...]:
        return self._outputs

    def input_rank(self, address: int) -> int:
        return self._ranks[INPUT_SOURCE][address]

    def weight_rank(self, address: int) -> int:
        return self._ranks[WEIGHT_SOURCE][address]

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

    def evaluate(
        self, inputs: Sequence[int], weights: Sequence[int] = ()
    ) -> tuple[int, ...]:
        given = {
            INPUT_SOURCE: iter(_checked_sources(inputs, self.input_count, "inputs")),
            WEIGHT_SOURCE: iter(
                _checked_sources(weights, self.weight_count, "weights")
            ),
        }
        values: list[int] = []
        for ref in self._gates:
            if ref.source is not None:
                values.append(_source_value(given, ref.source, ref.width))
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

    Widths are per gate (``C[i].width``, the runs of ``Out``); ``width`` is
    the common width when every gate of the set has the same one and
    ``None`` for a mixed-width set such as
    :func:`veritor.core.ml_gates.make_ml_gate_set`.
    """

    __slots__ = ("_outputs", "frame", "gate_set", "root", "width")

    def __init__(self, root: Definition, gate_set: GateSet) -> None:
        widths = {gate.width for gate in gate_set}
        self.root = root
        self.gate_set = gate_set
        self.width: int | None = widths.pop() if len(widths) == 1 else None
        self.frame = Frame.root(root)
        self._outputs = _LazyAddresses(root.output_count, self.frame.output_address)

    @property
    def n(self) -> int:
        return self.root.size

    @property
    def input_count(self) -> int:
        return self.root.input_total

    @property
    def weight_count(self) -> int:
        return self.root.weight_total

    @property
    def inputs(self) -> Sequence[int]:
        frame = self.frame
        return _LazyAddresses(
            self.root.input_total, lambda rank: frame.source_address(INPUT_SOURCE, rank)
        )

    @property
    def weights(self) -> Sequence[int]:
        frame = self.frame
        return _LazyAddresses(
            self.root.weight_total,
            lambda rank: frame.source_address(WEIGHT_SOURCE, rank),
        )

    @property
    def outputs(self) -> Sequence[int]:
        return self._outputs

    def _source_rank(self, source: str, address: int) -> int:
        rank = None
        if type(address) is int and 0 <= address < self.n:
            rank = self.frame.source_rank(source, address)
        if rank is None:
            raise KeyError(address)
        return rank

    def input_rank(self, address: int) -> int:
        """The rank of the input gate at ``address`` (``O(depth)``)."""

        return self._source_rank(INPUT_SOURCE, address)

    def weight_rank(self, address: int) -> int:
        """The rank of the weight gate at ``address`` (``O(depth)``)."""

        return self._source_rank(WEIGHT_SOURCE, address)

    def __getitem__(self, address: int) -> GateRef:
        if type(address) is not int or not 0 <= address < self.n:
            raise IndexError(address)
        gate, args = self.frame.gate(address)
        return GateRef(gate.name, args, gate.width, gate.source)

    def In(self, subset: object) -> tuple[int, ...]:
        frame = _frame(subset)
        return tuple(sorted({frame.input_address(i) for i in frame.definition.reads}))

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

    def evaluate(
        self, inputs: Sequence[int], weights: Sequence[int] = ()
    ) -> tuple[int, ...]:
        given = {
            INPUT_SOURCE: iter(_checked_sources(inputs, self.input_count, "inputs")),
            WEIGHT_SOURCE: iter(
                _checked_sources(weights, self.weight_count, "weights")
            ),
        }
        values: list[int] = []
        for address in range(self.n):
            gate, args = self.frame.gate(address)
            if gate.source is not None:
                values.append(_source_value(given, gate.source, gate.width))
            else:
                values.append(gate.evaluate(tuple(values[a] for a in args)))
        return tuple(values)
