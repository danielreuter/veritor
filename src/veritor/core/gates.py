"""The gate set Σ: the public registry of scalar gates a circuit may use.

A :class:`GateSet` is a public protocol parameter shared by prover and
verifier.  Every gate is a scalar relation with a fixed arity, an output width
in bits, executable semantics, and public replay/proof costs.  The value codec
(canonical bytes for a value of a given width) also lives here because it is
the only place widths are defined.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence

from .errors import InvalidArtifact
from .identity import Digest, JSONValue, identity_digest

GATE_SET_IDENTITY_TAG = "veritor/gate-set/v1"


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
    """One scalar gate: ``name``, ``arity``, output ``width`` and costs.

    ``evaluate`` and ``check`` are trusted executable semantics.  They are
    excluded from identity; the enclosing gate set's name and version name the
    semantics.  Arguments are validated against the gate's own width (every
    gate in a set has the same width for now).
    """

    __slots__ = (
        "_check",
        "_evaluate",
        "arity",
        "name",
        "proof_cost",
        "replay_cost",
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
        evaluate: Callable[[tuple[int, ...]], int],
        check: Callable[[tuple[int, ...], int], bool] | None = None,
    ) -> None:
        if type(name) is not str or not name.strip():
            raise ValueError("gate names must be nonempty strings")
        if type(arity) is not int or arity <= 0:
            raise ValueError("gates need positive integer arity")
        if type(width) is not int or width <= 0:
            raise ValueError("gate widths must be positive bit counts")
        for label, cost in (("replay_cost", replay_cost), ("proof_cost", proof_cost)):
            if type(cost) is not int or cost < 0:
                raise ValueError(f"gate {label} must be a nonnegative integer")
        if not callable(evaluate) or (check is not None and not callable(check)):
            raise TypeError("gate semantics must be callable")
        self.name = name
        self.arity = arity
        self.width = width
        self.replay_cost = replay_cost
        self.proof_cost = proof_cost
        self._evaluate = evaluate
        self._check = check

    def __repr__(self) -> str:
        return f"Gate({self.name!r}, arity={self.arity}, width={self.width})"

    @property
    def manifest(self) -> dict[str, JSONValue]:
        return {
            "arity": self.arity,
            "name": self.name,
            "proof_cost": self.proof_cost,
            "replay_cost": self.replay_cost,
            "width": self.width,
        }

    def _checked_args(self, args: Sequence[object]) -> tuple[int, ...]:
        if len(args) != self.arity:
            raise InvalidArtifact(
                f"gate {self.name} expects {self.arity} arguments, got {len(args)}"
            )
        return tuple(
            check_value(self.width, value, where=f"gate {self.name} argument {index}")
            for index, value in enumerate(args)
        )

    def evaluate(self, args: Sequence[object]) -> int:
        """Evaluate the gate on validated arguments; the result is width-checked."""

        checked = self._checked_args(args)
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
    """A frozen, named registry of gates: the protocol's public parameter Σ."""

    __slots__ = ("_gates", "digest", "name", "version")

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


def make_word_gate_set(width: int = 8) -> GateSet:
    """Modular add/multiply on ``width``-bit values: the built-in standard gates."""

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
        ),
        name="veritor.word-arithmetic",
        version="1",
    )
