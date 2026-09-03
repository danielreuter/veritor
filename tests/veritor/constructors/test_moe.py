"""Advised mixture-of-experts steps: one body for every route, the route at the call site."""

from __future__ import annotations

import json

import pytest

from veritor.constructors import (
    ADVICE,
    PADDED,
    LMShape,
    RequestsG,
    random_parameters,
    reference_generate,
)
from veritor.constructors.moe import (
    RequestRoutes,
    decode_routes,
    encode_routes,
    reference_routes,
)
from veritor.constructors.schedule import Request
from veritor.constructors.tracer import TracerError
from veritor.core import as_kind_table, make_isa_gate_set
from veritor.research import Compile

WIDTH = 16
SHAPE = LMShape(
    vocab=8, d_model=4, heads=1, layers=2, context=32, width=WIDTH, experts=4, top_k=2
)
REQUESTS = (Request((1, 2, 3), 3), Request((5, 6), 2))
GATE_SET = make_isa_gate_set(WIDTH)


def definitions(description: bytes) -> dict[str, dict[str, object]]:
    """The description's definitions by digest."""

    document = json.loads(description)
    return {item["digest"]: item["body"] for item in document["definitions"]}


def bodies_below_the_requests(description: bytes) -> dict[str, dict[str, object]]:
    """Every definition but the root and the request RUs: the steps and everything they call.

    The route enters at the call sites of the steps, which live in the request
    RU's body (one per request in any case); below them nothing may mention it.
    """

    document = json.loads(description)
    return {
        digest: body
        for digest, body in definitions(description).items()
        if digest != document["root"] and body.get("role") != "replay"
    }


def other_routes(routes: tuple[RequestRoutes, ...]) -> tuple[RequestRoutes, ...]:
    """Every route replaced by another legal one: the complement's first ``k`` experts, ascending."""

    experts, k = SHAPE.experts, SHAPE.top_k
    return tuple(
        tuple(
            tuple(
                tuple(
                    tuple(e for e in range(experts) if e not in route)[:k]
                    for route in layer
                )
                for layer in step
            )
            for step in request
        )
        for request in routes
    )


def test_the_step_bodies_are_the_same_definitions_for_every_route() -> None:
    parameters = random_parameters(SHAPE, 3)
    g = RequestsG(SHAPE, ADVICE)
    honest = reference_routes(SHAPE, parameters, REQUESTS)
    lying = other_routes(honest)
    assert (
        lying != honest
        and decode_routes(SHAPE, REQUESTS, encode_routes(SHAPE, lying)) == lying
    )
    honest_description, _ = g(REQUESTS, encode_routes(SHAPE, honest))
    lying_description, _ = g(REQUESTS, encode_routes(SHAPE, lying))

    # the same step bodies, and the same everything below them: no body mentions the route ...
    assert bodies_below_the_requests(honest_description) == bodies_below_the_requests(
        lying_description
    )
    # ... only the request RUs differ, in the ranges their call sites pass, at the same length
    honest_requests = {
        d
        for d, b in definitions(honest_description).items()
        if b.get("role") == "replay"
    }
    lying_requests = {
        d
        for d, b in definitions(lying_description).items()
        if b.get("role") == "replay"
    }
    shared = honest_requests & lying_requests  # the weights unit, an RU too
    assert len(shared) == 1
    assert (
        len(honest_requests - shared) == len(lying_requests - shared) == len(REQUESTS)
    )
    assert len(honest_description) == len(lying_description)
    # the compiled kinds: every verification kind is shared; the request kinds are the routes'
    honest_c = Compile(
        g, REQUESTS, encode_routes(SHAPE, honest), GATE_SET, max_advice_bits=1 << 10
    )
    lying_c = Compile(
        g, REQUESTS, encode_routes(SHAPE, lying), GATE_SET, max_advice_bits=1 << 10
    )
    honest_table, lying_table = (
        as_kind_table(honest_c.compiled),
        as_kind_table(lying_c.compiled),
    )
    verification = {
        row.kind: (row.copies, row.size)
        for row in honest_table.rows
        if row.kind not in honest_requests and row.kind != honest_table.root
    }
    assert len(verification) > 30 and verification == {
        row.kind: (row.copies, row.size)
        for row in lying_table.rows
        if row.kind not in lying_requests and row.kind != lying_table.root
    }
    assert (
        honest_c.compiled.digest != lying_c.compiled.digest
    )  # different circuits, the same bodies


def test_the_description_grows_by_call_sites_not_by_bodies_with_more_routes() -> None:
    """One step body per ``(positions, advised)``: more requests with new routes add request RUs only."""

    parameters = random_parameters(SHAPE, 3)
    g = RequestsG(SHAPE, ADVICE)
    few = REQUESTS
    many = (*REQUESTS, *(Request((t, t + 1), 2) for t in range(1, 7)))
    routes_many = reference_routes(SHAPE, parameters, many)
    distinct = {
        route
        for request in routes_many
        for step in request
        for layer in step
        for route in layer
    }
    assert len(distinct) >= 3, distinct
    few_below = bodies_below_the_requests(g(few, g.advice(few, parameters))[0])
    many_below = bodies_below_the_requests(g(many, g.advice(many, parameters))[0])
    # the new requests are all 2-token prompts of 2 tokens: the bodies below the requests gain at
    # most the (prefill 2, decode 3) steps and their new position counts, never a body per route
    assert set(few_below) <= set(many_below)
    assert len(set(many_below) - set(few_below)) <= 4


def test_the_ok_words_are_check_outputs_and_a_lying_route_computes_zero() -> None:
    parameters = random_parameters(SHAPE, 3)
    weights = parameters.flatten()
    g = RequestsG(SHAPE, ADVICE)
    honest = reference_routes(SHAPE, parameters, REQUESTS)
    reference = tuple(
        t
        for response in reference_generate(SHAPE, parameters, REQUESTS)
        for t in response
    )
    layout = g.output_layout(REQUESTS)
    ok_ordinals = [i for i, (_r, position) in enumerate(layout) if position < 0]
    assert len(ok_ordinals) == len(REQUESTS)

    compilation = Compile(
        g, REQUESTS, encode_routes(SHAPE, honest), GATE_SET, max_advice_bits=1 << 10
    )
    assert (
        compilation.advice_bits
        == g.advice_bits(REQUESTS)
        == SHAPE.route_advice_bits(3 + 2) + SHAPE.route_advice_bits(2 + 1)
    )
    assert list(compilation.compiled.check_values()) == [
        (ordinal, 1) for ordinal in ok_ordinals
    ]
    circuit = compilation.compiled.circuit
    outputs = circuit.evaluate(compilation.inputs, weights)
    values = [outputs[a] for a in circuit.outputs]
    assert [v for i, v in enumerate(values) if i not in ok_ordinals] == list(reference)
    assert [values[i] for i in ok_ordinals] == [1, 1]
    # the ok words are outside out_bits: the root's capacity is the tokens'
    table = as_kind_table(compilation.compiled)
    root = next(row for row in table.rows if row.kind == table.root)
    assert root.out_bits == WIDTH * len(reference)

    lying = Compile(
        g,
        REQUESTS,
        encode_routes(SHAPE, other_routes(honest)),
        GATE_SET,
        max_advice_bits=1 << 10,
    )
    lying_outputs = lying.compiled.circuit.evaluate(lying.inputs, weights)
    lying_values = [lying_outputs[a] for a in lying.compiled.circuit.outputs]
    assert [lying_values[i] for i in ok_ordinals] == [
        0,
        0,
    ]  # the route check fails: the verifier rejects at ok


def test_padded_routing_takes_no_advice_and_advised_routing_needs_it() -> None:
    parameters = random_parameters(SHAPE, 3)
    padded = RequestsG(SHAPE, PADDED)
    assert (
        padded.advice(REQUESTS, parameters) == b"" and padded.advice_bits(REQUESTS) == 0
    )
    with pytest.raises(TracerError):
        RequestsG(SHAPE, ADVICE)(REQUESTS, b"")
    with pytest.raises(TracerError):
        padded(
            REQUESTS,
            encode_routes(SHAPE, reference_routes(SHAPE, parameters, REQUESTS)),
        )
