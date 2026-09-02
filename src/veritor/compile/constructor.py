"""The client's constructor ``G`` as the verifier sees it, and the record of one ``Compile``.

A constructor is public code both parties hold, named by a versioned digest
like the gate set.  The verifier runs it on a request's public inputs ``x``
and the client's advice ``a`` (:func:`veritor.research.Compile`) and keeps a
:class:`Compilation`: the ``(C, I)`` it produced, together with what it was
run on, so that the header can bind ``G`` and ``a`` and the advice can be
charged.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from veritor.core import Compiled, Digest, JSONValue, identity_digest

CONSTRUCTOR_DIGEST_TAG = "veritor/constructor/v1"


@runtime_checkable
class Constructor(Protocol):
    """What the verifier requires of a constructor ``G``.

    ``digest`` is a stable identity: the hex SHA-256 of a canonical manifest
    of class name, version and parameters (:func:`constructor_digest`).
    ``G(x, a)`` returns the description bytes and the flat circuit inputs, the
    values of the ``in`` gates in address order; ``G`` knows its own layout.
    """

    @property
    def digest(self) -> str: ...

    def __call__(self, x: object, a: bytes) -> tuple[bytes, tuple[int, ...]]: ...


def constructor_digest(name: str, version: str, parameters: Mapping[str, JSONValue]) -> Digest:
    """The identity of a constructor: its class name, version and parameters."""

    return identity_digest(
        CONSTRUCTOR_DIGEST_TAG, {"name": name, "parameters": parameters, "version": version}
    )


@dataclass(frozen=True, slots=True)
class Compilation:
    """The verifier-side record of one ``Compile(G, x, a)``.

    ``compiled`` is ``(C, I)`` with its digest; ``constructor`` is ``G``'s
    digest; ``inputs`` is ``x`` as the circuit consumes it, the values of the
    ``in`` gates by rank as ``G`` laid them out; ``advice`` is the client's
    ``a``, charged at :attr:`advice_bits`.
    """

    compiled: Compiled
    constructor: str
    inputs: tuple[int, ...]
    advice: bytes

    @property
    def advice_bits(self) -> int:
        return 8 * len(self.advice)
