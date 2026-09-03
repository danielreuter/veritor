"""Requests cut short by the client (S7): the generated length as advice.

A client that disconnects, times out or presses stop after ``t`` of the
``max_new`` tokens it asked for leaves a request whose circuit shape is
``(prompt length, t)``, and nothing in ``x`` says what ``t`` is: the client
asked for ``max_new``, the server stopped at ``t`` because the client went
away, and the verifier only sees that ``t`` tokens were streamed.  With
request replay units (:class:`~veritor.constructors.requests.RequestsG`)
the shape must reach ``Compile`` somehow.  This constructor takes the
smallest honest route: ``t`` is *advice*, ``ceil(log2 max_new)`` bits per
request (``t`` ranges over ``1..max_new``; a request that streamed nothing
has no outputs and is not in the circuit at all, S8's principle), packed
big-endian and padded to a byte, charged like any other advice.

The alternative is padding: run every request to ``max_new`` and let the
verifier ignore the tokens after ``t``.  It costs no advice but
``max_new - t`` decode steps of compute per truncated request, which the
scenario measures.  Whether a length the *output* determines
(``len(y) = t``) may be charged zero -- M3 in ``docs/stress-tests.md`` --
is a theory question for the architect: EOS termination has a
verifier-checkable rule (the last token is EOS or ``len(y) = max_tokens``),
a disconnect has none unless the client attests to it, so here it is
charged.
"""

from __future__ import annotations

from collections.abc import Sequence

from veritor.compile import constructor_digest
from veritor.core import Digest, JSONValue

from .lm import LMShape
from .requests import RequestsG
from .schedule import Request
from .tracer import TracerError


def field_width(cardinality: int) -> int:
    """``ceil(log2 cardinality)``: the bits that name one of ``cardinality`` values."""

    if type(cardinality) is not int or cardinality < 1:
        raise ValueError("cardinality must be a positive integer")
    return (cardinality - 1).bit_length()


def pack_fields(values: Sequence[int], widths: Sequence[int]) -> bytes:
    """``values[i]`` in ``widths[i]`` bits each, big-endian, zero-padded to whole bytes."""

    if len(values) != len(widths):
        raise ValueError("one width per value")
    bits = 0
    total = 0
    for value, width in zip(values, widths, strict=True):
        if type(value) is not int or type(width) is not int or width < 0:
            raise ValueError("values and widths must be integers, widths nonnegative")
        if not 0 <= value < 1 << width:
            raise ValueError(f"{value} does not fit in {width} bits")
        bits = (bits << width) | value
        total += width
    padding = -total % 8
    return ((bits << padding) & ((1 << (total + padding)) - 1)).to_bytes((total + padding) // 8, "big")


def unpack_fields(data: bytes, widths: Sequence[int]) -> tuple[int, ...]:
    """The inverse of :func:`pack_fields`; the padding must be zero and the length exact."""

    total = sum(widths)
    if type(data) is not bytes or len(data) != (total + 7) // 8:
        raise ValueError(f"expected {(total + 7) // 8} bytes of packed fields")
    bits = int.from_bytes(data, "big")
    padding = -total % 8
    if bits & ((1 << padding) - 1):
        raise ValueError("nonzero padding")
    bits >>= padding
    values: list[int] = []
    for width in reversed(widths):
        values.append(bits & ((1 << width) - 1))
        bits >>= width
    return tuple(reversed(values))


class TruncatedRequestsG:
    """The toy decoder, one replay unit per request, the generated lengths as advice.

    ``x`` is the requests as the clients issued them (``Request`` with the
    ``max_new`` they asked for, and a random word for each of those
    positions if the model samples); ``a`` is :meth:`advice` for the lengths
    actually generated.  The description is exactly what ``RequestsG`` builds
    for the truncated requests, so the kinds, the marks and ``Bound`` are
    the same as for a run that asked for ``t`` tokens in the first place.
    """

    VERSION = "1"

    def __init__(self, shape: LMShape) -> None:
        self.inner = RequestsG(shape)
        self.shape = shape
        self.digest: Digest = constructor_digest(type(self).__name__, self.VERSION, self.manifest)

    @property
    def manifest(self) -> dict[str, JSONValue]:
        return {"shape": self.shape.manifest}

    def requests(self, x: object) -> tuple[Request, ...]:
        return self.inner.requests(x)

    def widths(self, x: object) -> tuple[int, ...]:
        """The advice bits each request costs: ``ceil(log2 max_new)``."""

        return tuple(field_width(request.max_new) for request in self.requests(x))

    def advice_bits(self, x: object) -> int:
        """The bits the lengths take before padding to bytes."""

        return sum(self.widths(x))

    def advice(self, x: object, lengths: Sequence[int]) -> bytes:
        """The advice naming ``lengths[i]`` generated tokens for request ``i``."""

        requests = self.requests(x)
        if len(lengths) != len(requests):
            raise ValueError("one length per request")
        for length, request in zip(lengths, requests, strict=True):
            if type(length) is not int or not 1 <= length <= request.max_new:
                raise ValueError(f"a generated length must lie in 1..{request.max_new}")
        return pack_fields([length - 1 for length in lengths], self.widths(x))

    def lengths(self, x: object, a: bytes) -> tuple[int, ...]:
        """The generated lengths ``a`` names; a :class:`TracerError` if it is malformed."""

        try:
            fields = unpack_fields(a, self.widths(x))
        except ValueError as error:
            raise TracerError(f"malformed length advice: {error}") from error
        lengths = tuple(field + 1 for field in fields)
        for length, request in zip(lengths, self.requests(x), strict=True):
            if length > request.max_new:
                raise TracerError(f"advice names {length} tokens, more than max_new {request.max_new}")
        return lengths

    def truncated(self, x: object, a: bytes) -> tuple[Request, ...]:
        """The requests as run: ``max_new`` cut to the advised length, randomness with it."""

        return tuple(
            Request(request.prompt, length, request.randomness[:length])
            for request, length in zip(self.requests(x), self.lengths(x, a), strict=True)
        )

    def output_layout(self, x: object, a: bytes) -> tuple[tuple[int, int], ...]:
        return self.inner.output_layout(self.truncated(x, a))

    def flatten_inputs(self, x: object, a: bytes) -> tuple[int, ...]:
        return self.inner.flatten_inputs(self.truncated(x, a))

    def __call__(self, x: object, a: bytes) -> tuple[bytes, tuple[int, ...]]:
        if type(a) is not bytes:
            raise TracerError("advice must be bytes")
        return self.inner(self.truncated(x, a), b"")


__all__ = ["TruncatedRequestsG", "field_width", "pack_fields", "unpack_fields"]
