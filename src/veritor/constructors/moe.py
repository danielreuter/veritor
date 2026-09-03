"""Mixture-of-experts routes as advice: the reference routes and their codec.

With :attr:`~veritor.constructors.lm.LMShape.experts` ``= E`` every position
of every layer is routed to ``top_k`` experts (:func:`~veritor.constructors.lm.top_k_route`).
:class:`~veritor.constructors.requests.RequestsG` with ``routing="advice"``
takes those routes from the client as ``a`` and builds a circuit that runs
only the chosen experts; the compiler charges exactly :func:`advice_bits`
bits for it (the constructor declares them; the byte padding is checked to
be zero, so it is not a channel).

The codec is the description length and nothing more: request by request,
position by position (the prompt, then every generated position that is fed
back -- the last token never is), layer by layer, the ``top_k`` chosen ids
ascending, ``ceil(log2 E)`` bits each, packed most significant bit first and
zero-padded to a byte.  :func:`decode_routes` rejects anything else: a
trailing nonzero bit, an id at or above ``E``, a repeated or unsorted id.
The constructor then fails, which is a rejection.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

from .lm import Decoder, LMShape, Parameters, Route, Routes
from .schedule import Request
from .tracer import TracerError

RequestRoutes = tuple[Routes, ...]
"""One request's routes step by step: the prefill's, then one per decode step."""


def fed_positions(request: Request) -> int:
    """The positions a request feeds through the model: the prompt and all but the last token."""

    return len(request.prompt) + request.max_new - 1


def step_positions(request: Request) -> tuple[int, ...]:
    """How many positions each step of a request feeds: the prompt, then one per decode step."""

    return (len(request.prompt), *(1 for _ in range(request.max_new - 1)))


def reference_routes(
    shape: LMShape, parameters: Parameters, requests: Sequence[Request]
) -> tuple[RequestRoutes, ...]:
    """Every request's routes, as the reference decoder takes them, grouped by step."""

    if not shape.experts:
        raise ValueError("a dense shape has no routes")
    if not isinstance(parameters, Parameters) or parameters.shape != shape:
        raise ValueError("parameters must be Parameters of the given shape")
    result: list[RequestRoutes] = []
    for request in requests:
        decoder = Decoder(parameters)
        randomness = request.randomness if shape.sampling else (None,) * request.max_new
        for prompt_token in request.prompt[:-1]:
            decoder.logits(prompt_token)
        token = decoder.forward(request.prompt[-1], randomness[0])
        for position in range(1, request.max_new):
            token = decoder.forward(token, randomness[position])
        assert decoder.positions == fed_positions(request)
        steps: list[Routes] = []
        cursor = 0
        for count in step_positions(request):
            steps.append(
                tuple(tuple(decoder.routes[layer][cursor : cursor + count]) for layer in range(shape.layers))
            )
            cursor += count
        result.append(tuple(steps))
    return tuple(result)


def advice_bits(shape: LMShape, requests: Sequence[Request]) -> int:
    """The description length of the requests' routes in bits, before byte padding."""

    return sum(shape.route_advice_bits(fed_positions(request)) for request in requests)


def encode_routes(shape: LMShape, routes: Sequence[RequestRoutes]) -> bytes:
    """Pack every request's routes as the advice ``a``."""

    bits: list[str] = []
    width = shape.route_bits
    for request_routes in routes:
        for step in request_routes:
            positions = len(step[0])
            for position in range(positions):
                for layer in range(shape.layers):
                    for expert in step[layer][position]:
                        bits.append(format(expert, f"0{width}b"))
    text = "".join(bits)
    text += "0" * (-len(text) % 8)
    return bytes(int(text[i : i + 8], 2) for i in range(0, len(text), 8))


def decode_routes(shape: LMShape, requests: Sequence[Request], a: bytes) -> tuple[RequestRoutes, ...]:
    """The advice back into routes, validated: the exact length, ids below ``E``, distinct and ascending."""

    if not shape.experts:
        raise TracerError("a dense shape takes no routes")
    needed = advice_bits(shape, requests)
    if len(a) != (needed + 7) // 8:
        raise TracerError(f"the routes of these requests are {(needed + 7) // 8} bytes, got {len(a)}")
    text = "".join(format(byte, "08b") for byte in a)
    if any(bit != "0" for bit in text[needed:]):
        raise TracerError("the route advice has nonzero padding")
    width, k, experts = shape.route_bits, shape.top_k, shape.experts
    cursor = 0

    def route() -> Route:
        nonlocal cursor
        ids: list[int] = []
        for _ in range(k):
            ids.append(int(text[cursor : cursor + width], 2))
            cursor += width
        if any(e >= experts for e in ids) or any(a >= b for a, b in pairwise(ids)):
            raise TracerError(f"route {tuple(ids)} is not {k} distinct experts below {experts}, ascending")
        return tuple(ids)

    result: list[RequestRoutes] = []
    for request in requests:
        steps: list[Routes] = []
        for count in step_positions(request):
            by_position = [tuple(route() for _ in range(shape.layers)) for _ in range(count)]
            steps.append(tuple(tuple(by_position[p][layer] for p in range(count)) for layer in range(shape.layers)))
        result.append(tuple(steps))
    assert cursor == needed
    return tuple(result)


__all__ = [
    "RequestRoutes",
    "advice_bits",
    "decode_routes",
    "encode_routes",
    "fed_positions",
    "reference_routes",
    "step_positions",
]
