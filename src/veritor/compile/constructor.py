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

from veritor.core import (
    Compiled,
    Digest,
    InvalidArtifact,
    JSONValue,
    identity_digest,
    validate_advice_bits,
)

from .description import CompileError

CONSTRUCTOR_DIGEST_TAG = "veritor/constructor/v1"


@runtime_checkable
class Constructor(Protocol):
    """What the verifier requires of a constructor ``G``.

    ``digest`` is a stable identity: the hex SHA-256 of a canonical manifest
    of class name, version and parameters (:func:`constructor_digest`).
    ``G(x, a)`` returns the description bytes and the flat circuit inputs, the
    values of the ``in`` gates in address order; ``G`` knows its own layout.

    A constructor may also define ``advice_bits(x, a) -> int``, the exact
    number of bits its advice ``a`` carries on the request ``x``.  The
    compiler charges that many (:attr:`Compilation.advice_bits`) after
    checking that ``a`` is its canonical encoding, ``ceil(bits / 8)`` bytes
    with zero padding (:func:`veritor.core.validate_advice_bits`); without
    the method every byte counts, ``8 * len(a)``.
    """

    @property
    def digest(self) -> str: ...

    def __call__(self, x: object, a: bytes) -> tuple[bytes, tuple[int, ...]]: ...


def declared_advice_bits(G: Constructor, x: object, a: bytes) -> int:
    """The bits ``G`` declares for the advice ``a`` on ``x``, once ``a`` encodes them canonically.

    ``G.advice_bits(x, a)`` when ``G`` defines it, else ``8 * len(a)``.  A
    failing or ill-typed declaration, or an ``a`` that is not the canonical
    ``ceil(bits / 8)``-byte zero-padded encoding, is a :class:`CompileError`.
    """

    declare = getattr(G, "advice_bits", None)
    if declare is None:
        return 8 * len(a)
    try:
        bits = declare(x, a)
    except (Exception, SystemExit) as error:
        raise CompileError(f"the constructor failed to declare its advice bits: {error}") from error
    try:
        return validate_advice_bits(a, bits)
    except InvalidArtifact as error:
        raise CompileError(str(error)) from error


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
    ``a``, charged at ``advice_bits``, the bits ``G`` declared for it
    (:func:`declared_advice_bits`), of which ``advice`` must be the
    canonical encoding.  The default ``0`` is right for empty advice only.
    """

    compiled: Compiled
    constructor: Digest
    inputs: tuple[int, ...]
    advice: bytes
    advice_bits: int = 0

    def __post_init__(self) -> None:
        try:
            validate_advice_bits(self.advice, self.advice_bits)
        except InvalidArtifact as error:
            raise CompileError(str(error)) from error
