"""``Compiled``: the compiler's output, the circuit ``C`` with its index ``I``."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace

from .circuit import Circuit
from .description import Check
from .identity import Digest, identity_digest
from .index import Index, KindTable

COMPILED_DIGEST_TAG = "veritor/compiled/v1"


@dataclass(frozen=True, slots=True)
class Compiled:
    """``Compile``'s output: the circuit ``C``, its index ``I`` and ``H(C, I)``.

    ``digest`` binds the canonical description to the gate set it was compiled
    against, so two parties holding the same digest agree on every address,
    every unit and every gate's semantics.
    """

    circuit: Circuit
    index: Index
    digest: Digest

    @staticmethod
    def digest_of(description_digest: str, gate_set_digest: str) -> Digest:
        return identity_digest(
            COMPILED_DIGEST_TAG,
            {"description": description_digest, "gate_set": gate_set_digest},
        )

    def kind_table(self) -> KindTable:
        """The index's per-kind table under ``H(C, I)``, what the analysis folds read."""

        return replace(self.index.kind_table(), digest=self.digest)

    @property
    def checks(self) -> tuple[Check, ...]:
        """The check outputs: output ordinals the verifier requires to equal a constant."""

        return self.index.checks

    def check_values(self) -> Iterator[tuple[int, int]]:
        """``(output ordinal, constant)`` for every check output."""

        return self.index.check_values()


def as_kind_table(target: Compiled | KindTable) -> KindTable:
    """The table of a :class:`Compiled` artifact, or ``target`` if it is one already."""

    if isinstance(target, Compiled):
        return target.kind_table()
    if isinstance(target, KindTable):
        return target
    raise TypeError("expected a Compiled artifact or a KindTable")
