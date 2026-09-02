"""``Compiled``: the compiler's output, the circuit ``C`` with its index ``I``."""

from __future__ import annotations

from dataclasses import dataclass

from .circuit import Circuit
from .identity import Digest, identity_digest
from .index import Index

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

