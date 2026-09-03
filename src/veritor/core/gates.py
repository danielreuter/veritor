"""The gate set Σ: the public registry of scalar gates a circuit may use.

A :class:`GateSet` is a public protocol parameter shared by prover and
verifier.  Every gate is a scalar relation with a fixed arity, an output width
in bits, executable semantics, and public replay/proof costs.  *Source* gates
have arity ``0`` and no relation: an ``"input"`` gate takes its value from the
circuit's public input ``x`` and a ``"weight"`` gate from the model's weights
``W``, both by rank in address order, so inputs and weights are gates of the
circuit and lie inside units like every other gate.  The value codec
(canonical bytes for a value of a given width) also lives here because it is
the only place widths are defined.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence

from .errors import InvalidArtifact
from .identity import Digest, JSONValue, identity_digest

GATE_SET_IDENTITY_TAG = "veritor/gate-set/v1"
INPUT_SOURCE = "input"
WEIGHT_SOURCE = "weight"
SOURCES = (INPUT_SOURCE, WEIGHT_SOURCE)
NAMESPACE_SEPARATOR = "@"


def value_byte_length(width: int) -> int:
    """Return the canonical encoded length of a ``width``-bit value."""

    return (width + 7) // 8


def check_value(width: int, value: object, *, where: str = "value") -> int:
    """Return ``value`` if it is an integer in ``range(2 ** width)``."""

    if type(value) is not int or not 0 <= value < (1 << width):
        raise InvalidArtifact(f"{where} is not a {width}-bit value")
    return value


def encode_value(width: int, value: object) -> bytes:
    """Encode a ``width``-bit value as fixed-width big-endian bytes."""

    return check_value(width, value).to_bytes(value_byte_length(width), "big")


def decode_value(width: int, payload: object) -> int:
    """Decode canonical fixed-width bytes into a ``width``-bit value."""

    length = value_byte_length(width)
    if type(payload) is not bytes or len(payload) != length:
        raise InvalidArtifact(f"encoded value must be exactly {length} bytes")
    return check_value(width, int.from_bytes(payload, "big"), where="encoded value")


class Gate:
    """One scalar gate: ``name``, ``arity``, output ``width``, costs and ``source``.

    ``evaluate`` and ``check`` are trusted executable semantics.  They are
    excluded from identity; the enclosing gate set's name and version name the
    semantics.  Arguments are validated against ``arg_widths``, one width per
    argument, which defaults to the gate's own ``width`` for every argument
    (a mixed-width gate such as a tensor-core step -- one 32-bit accumulator,
    then 16-bit operand words -- declares its own).  A gate with a ``source``
    (``"input"`` or ``"weight"``) has arity ``0`` and no semantics: its value
    comes from the environment, so ``evaluate`` and ``check`` raise.
    """

    __slots__ = (
        "_check",
        "_evaluate",
        "arg_widths",
        "arity",
        "name",
        "proof_cost",
        "replay_cost",
        "source",
        "width",
    )

    def __init__(
        self,
        name: str,
        arity: int,
        width: int,
        *,
        replay_cost: int,
        proof_cost: int,
        evaluate: Callable[[tuple[int, ...]], int] | None = None,
        check: Callable[[tuple[int, ...], int], bool] | None = None,
        source: str | None = None,
        arg_widths: Sequence[int] | None = None,
    ) -> None:
        if type(name) is not str or not name.strip():
            raise ValueError("gate names must be nonempty strings")
        if source is not None and source not in SOURCES:
            raise ValueError(f"gate source must be None or one of {list(SOURCES)}")
        if type(arity) is not int or arity < 0 or (arity == 0) != (source is not None):
            raise ValueError(
                "gates need positive integer arity; only source gates have arity 0"
            )
        if type(width) is not int or width <= 0:
            raise ValueError("gate widths must be positive bit counts")
        for label, cost in (("replay_cost", replay_cost), ("proof_cost", proof_cost)):
            if type(cost) is not int or cost < 0:
                raise ValueError(f"gate {label} must be a nonnegative integer")
        if source is not None:
            if evaluate is not None or check is not None:
                raise TypeError("source gates have no executable relation")
        elif not callable(evaluate) or (check is not None and not callable(check)):
            raise TypeError("gate semantics must be callable")
        if arg_widths is None:
            widths = (width,) * arity
        else:
            widths = tuple(arg_widths)
            if len(widths) != arity or any(
                type(w) is not int or w <= 0 for w in widths
            ):
                raise ValueError(
                    "arg_widths must give one positive bit count per argument"
                )
        self.name = name
        self.arity = arity
        self.width = width
        self.arg_widths: tuple[int, ...] = widths
        self.replay_cost = replay_cost
        self.proof_cost = proof_cost
        self.source = source
        self._evaluate = evaluate
        self._check = check

    def __repr__(self) -> str:
        return f"Gate({self.name!r}, arity={self.arity}, width={self.width})"

    @property
    def manifest(self) -> dict[str, JSONValue]:
        manifest: dict[str, JSONValue] = {
            "arity": self.arity,
            "name": self.name,
            "proof_cost": self.proof_cost,
            "replay_cost": self.replay_cost,
            "source": self.source,
            "width": self.width,
        }
        if self.arg_widths != (self.width,) * self.arity:
            # only a non-default declaration is part of the identity, so every
            # single-width gate set keeps the digest it had before ``arg_widths``
            manifest["arg_widths"] = list(self.arg_widths)
        return manifest

    def _checked_args(self, args: Sequence[object]) -> tuple[int, ...]:
        if self.source is not None:
            raise InvalidArtifact(
                f"gate {self.name} is a source gate; its value comes from the environment"
            )
        if len(args) != self.arity:
            raise InvalidArtifact(
                f"gate {self.name} expects {self.arity} arguments, got {len(args)}"
            )
        return tuple(
            check_value(width, value, where=f"gate {self.name} argument {index}")
            for index, (width, value) in enumerate(
                zip(self.arg_widths, args, strict=True)
            )
        )

    def evaluate(self, args: Sequence[object]) -> int:
        """Evaluate the gate on validated arguments; the result is width-checked."""

        checked = self._checked_args(args)
        assert self._evaluate is not None  # a source gate never gets this far
        try:
            result = self._evaluate(checked)
        except Exception as error:
            raise RuntimeError(f"gate {self.name} evaluator raised") from error
        if type(result) is not int or not 0 <= result < (1 << self.width):
            raise RuntimeError(f"gate {self.name} returned an invalid value")
        return result

    def check(self, args: Sequence[object], out: object) -> bool:
        """Decide whether ``out`` is a valid output for ``args``."""

        checked = self._checked_args(args)
        checked_out = check_value(self.width, out, where=f"gate {self.name} output")
        if self._check is None:
            return self.evaluate(checked) == checked_out
        try:
            return bool(self._check(checked, checked_out))
        except Exception as error:
            raise RuntimeError(f"gate {self.name} checker raised") from error


class GateSet:
    """A frozen, named registry of gates: the protocol's public parameter Σ.

    ``input_gates`` and ``weight_gates`` name the source gates of each kind.
    """

    __slots__ = ("_gates", "digest", "input_gates", "name", "version", "weight_gates")

    def __init__(self, gates: Iterable[Gate], *, name: str, version: str) -> None:
        for label, text in (("name", name), ("version", version)):
            if type(text) is not str or not text.strip():
                raise ValueError(f"gate set {label} must be a nonempty string")
        table: dict[str, Gate] = {}
        for gate in gates:
            if not isinstance(gate, Gate):
                raise TypeError("gate sets contain Gate values")
            if gate.name in table:
                raise ValueError(f"gate set declares {gate.name!r} twice")
            table[gate.name] = gate
        self.name = name
        self.version = version
        self._gates = dict(sorted(table.items()))
        self.input_gates: tuple[str, ...] = tuple(
            gate.name for gate in self._gates.values() if gate.source == INPUT_SOURCE
        )
        self.weight_gates: tuple[str, ...] = tuple(
            gate.name for gate in self._gates.values() if gate.source == WEIGHT_SOURCE
        )
        self.digest: Digest = identity_digest(GATE_SET_IDENTITY_TAG, self.manifest)

    def __repr__(self) -> str:
        return f"GateSet({self.id!r}, {list(self._gates)})"

    @property
    def id(self) -> str:
        return f"{self.name}@{self.version}"

    @property
    def manifest(self) -> dict[str, JSONValue]:
        return {
            "gates": [gate.manifest for gate in self._gates.values()],
            "name": self.name,
            "version": self.version,
        }

    def __getitem__(self, name: str) -> Gate:
        try:
            return self._gates[name]
        except KeyError:
            raise InvalidArtifact(f"unknown gate {name!r}") from None

    def __contains__(self, name: object) -> bool:
        return name in self._gates

    def __iter__(self) -> Iterator[Gate]:
        return iter(self._gates.values())

    def __len__(self) -> int:
        return len(self._gates)


def namespaced(name: str, namespace: str) -> str:
    """The name of gate ``name`` inside the member ``namespace`` of a union."""

    return f"{name}{NAMESPACE_SEPARATOR}{namespace}"


def union_gate_set(
    members: Mapping[str, GateSet], *, name: str, version: str
) -> GateSet:
    """Σ for a heterogeneous fleet: every member's gates under its namespace.

    A member's operator gate ``g`` becomes ``g@namespace`` with the same
    arity, width, costs and semantics, so kinds traced against one member
    name that member's gates and never match another's.  The source gates
    stay shared and unnamespaced: every member must declare the same ``in``
    and ``weight`` gates (same names and widths), because ``x`` and ``W`` are
    one environment for the whole fleet.  Namespaces are nonempty, contain no
    ``@`` and are distinct.
    """

    if not members:
        raise ValueError("a gate-set union needs at least one member")
    sources: dict[str, Gate] = {}
    gates: list[Gate] = []
    for namespace, member in members.items():
        if (
            type(namespace) is not str
            or not namespace
            or NAMESPACE_SEPARATOR in namespace
        ):
            raise ValueError(
                f"gate namespace {namespace!r} must be nonempty and contain no '@'"
            )
        if not isinstance(member, GateSet):
            raise TypeError("gate-set union members are GateSets")
        for gate in member:
            if gate.source is None:
                gates.append(
                    Gate(
                        namespaced(gate.name, namespace),
                        gate.arity,
                        gate.width,
                        replay_cost=gate.replay_cost,
                        proof_cost=gate.proof_cost,
                        evaluate=gate._evaluate,
                        check=gate._check,
                    )
                )
            elif gate.name in sources:
                if sources[gate.name].manifest != gate.manifest:
                    raise ValueError(
                        f"members disagree on the source gate {gate.name!r}"
                    )
            else:
                sources[gate.name] = gate
    return GateSet((*sources.values(), *gates), name=name, version=version)


def make_word_gate_set(width: int = 8) -> GateSet:
    """Modular add/multiply on ``width``-bit values, plus the ``in`` and ``weight`` sources.

    A source gate costs nothing to replay (its value is given) and the
    cheapest proof of the set to check (one opening or one comparison).
    """

    if type(width) is not int or width <= 0:
        raise ValueError("width must be a positive bit count")
    mask = (1 << width) - 1
    return GateSet(
        (
            Gate(
                "add",
                2,
                width,
                replay_cost=1,
                proof_cost=1,
                evaluate=lambda args: (args[0] + args[1]) & mask,
            ),
            Gate(
                "mul",
                2,
                width,
                replay_cost=2,
                proof_cost=2,
                evaluate=lambda args: (args[0] * args[1]) & mask,
            ),
            Gate("in", 0, width, replay_cost=0, proof_cost=1, source=INPUT_SOURCE),
            Gate("weight", 0, width, replay_cost=0, proof_cost=1, source=WEIGHT_SOURCE),
        ),
        name="veritor.word-arithmetic",
        version="2",
    )


def make_isa_gate_set(width: int = 16) -> GateSet:
    """The toy ISA: what a decoder needs, all on ``width``-bit unsigned words.

    ``add``, ``sub`` and ``mul`` are modular; ``lt`` and ``eq`` are the
    comparisons ``a < b`` and ``a == b`` as full-width words holding ``0`` or
    ``1``; ``shr`` is ``a >> b`` (``0`` once ``b >= width``) so activations
    can be scaled down.  ``in`` and ``weight`` are the sources, as in
    :func:`make_word_gate_set`.  Toy numerics in ``Z_{2^width}``: nothing
    here approximates real arithmetic, it only has the shape of it.
    """

    if type(width) is not int or width <= 0:
        raise ValueError("width must be a positive bit count")
    mask = (1 << width) - 1

    def word(name: str, cost: int, evaluate: Callable[[tuple[int, ...]], int]) -> Gate:
        return Gate(
            name, 2, width, replay_cost=cost, proof_cost=cost, evaluate=evaluate
        )

    return GateSet(
        (
            word("add", 1, lambda args: (args[0] + args[1]) & mask),
            word("sub", 1, lambda args: (args[0] - args[1]) & mask),
            word("mul", 2, lambda args: (args[0] * args[1]) & mask),
            word("lt", 1, lambda args: int(args[0] < args[1])),
            word("eq", 1, lambda args: int(args[0] == args[1])),
            word("shr", 1, lambda args: args[0] >> args[1] if args[1] < width else 0),
            Gate("in", 0, width, replay_cost=0, proof_cost=1, source=INPUT_SOURCE),
            Gate("weight", 0, width, replay_cost=0, proof_cost=1, source=WEIGHT_SOURCE),
        ),
        name="veritor.toy-isa",
        version="1",
    )
