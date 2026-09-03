"""``PrefixG``: the shared prefix is its own replay unit; its KV blocks are its declared outputs.

The suffix units read the prefix's cache through ports, so the circuit's
boundary widens by ``state_size(prefix)`` words -- exactly the interface
``RequestsG`` keeps inside every request when each request recomputes the
prefix -- while the tokens are the reference's under both.
"""

from __future__ import annotations

import pytest

from veritor.compile import Compiler
from veritor.constructors import (
    LMShape,
    PrefixG,
    Request,
    RequestsG,
    TracerError,
    random_parameters,
    reference_generate,
)
from veritor.constructors.prefix import shared_prefix
from veritor.core import Compiled
from veritor.core.description import REPLAY

SHAPE = LMShape(vocab=8, d_model=4, heads=2, layers=1, context=8, width=16)
SYSTEM = (3, 1, 4)
REQUESTS = (
    Request((*SYSTEM, 1), 2),
    Request((*SYSTEM, 5, 2), 3),
    Request((*SYSTEM, 6), 2),
    Request((*SYSTEM, 7, 7), 1),
)


def compile_with(constructor, requests: tuple[Request, ...]) -> Compiled:
    description, inputs = constructor(requests, b"")
    return Compiler(constructor.gate_set).compile(description, inputs)


def generated(
    constructor, compiled: Compiled, requests, parameters
) -> tuple[tuple[int, ...], ...]:
    values = compiled.circuit.evaluate(
        constructor.flatten_inputs(requests), parameters.flatten()
    )
    outputs = [values[address] for address in compiled.circuit.outputs]
    grouped: list[list[int]] = [[] for _ in requests]
    for (request, position), token in zip(
        constructor.output_layout(requests), outputs, strict=True
    ):
        assert position == len(grouped[request])
        grouped[request].append(token)
    return tuple(tuple(tokens) for tokens in grouped)


def test_the_shared_prefix_is_the_longest_common_prefix_leaving_every_prompt_a_token() -> (
    None
):
    assert shared_prefix(REQUESTS) == 3
    assert (
        shared_prefix((Request((1, 2, 3), 1), Request((1, 2, 3, 4), 1))) == 2
    )  # the shortest keeps one
    assert shared_prefix((Request((1, 2), 1), Request((2, 2), 1))) == 0
    assert shared_prefix((Request((1, 2, 3), 1),)) == 2


def test_it_generates_what_the_reference_and_requests_g_generate() -> None:
    parameters = random_parameters(SHAPE, seed=4)
    prefix, plain = PrefixG(SHAPE), RequestsG(SHAPE)

    compiled = compile_with(prefix, REQUESTS)
    reference = reference_generate(SHAPE, parameters, REQUESTS)
    assert generated(prefix, compiled, REQUESTS, parameters) == reference
    assert (
        generated(plain, compile_with(plain, REQUESTS), REQUESTS, parameters)
        == reference
    )
    assert prefix.flatten_inputs(REQUESTS) == (
        3,
        1,
        4,
        1,
        6,
        5,
        2,
        7,
        7,
    )  # the prefix once, then the suffixes


def test_the_prefix_unit_declares_its_kv_blocks_and_the_suffixes_read_them() -> None:
    prefix = PrefixG(SHAPE)
    compiled = compile_with(prefix, REQUESTS)
    index = compiled.index

    units = {row.kind: row for row in index.kinds() if row.role == REPLAY}
    prefix_row = units[prefix.prefix_unit(3).digest]
    assert prefix_row.copies == 1 and prefix_row.out_count == SHAPE.state_size(3) == 24
    assert (
        prefix_row.input_count == SHAPE.weight_count and prefix_row.source_inputs == 3
    )
    suffixes = [
        row
        for row in units.values()
        if row.kind not in (prefix_row.kind, prefix.lm.weights_unit().digest)
    ]
    assert sorted((row.copies, row.out_count) for row in suffixes) == [
        (1, 1),
        (1, 3),
        (2, 2),
    ]
    assert all(row.input_count == SHAPE.weight_count + 24 for row in suffixes)
    # the boundary: prompts and tokens as for RequestsG, plus the prefix's declared cache
    tokens = sum(request.max_new for request in REQUESTS)
    assert index.boundary().count == len(prefix.flatten_inputs(REQUESTS)) + tokens + 24
    plain = compile_with(RequestsG(SHAPE), REQUESTS)
    assert plain.index.boundary().count == 3 * len(REQUESTS) + 6 + tokens
    assert index.replay_units.count == 1 + 1 + len(REQUESTS)


def test_it_checks_its_requests() -> None:
    constructor = PrefixG(SHAPE)

    with pytest.raises(TracerError, match="share at least one token"):
        constructor((Request((1, 2), 1), Request((2, 2), 1)), b"")
    with pytest.raises(TracerError, match="share at least one token"):
        constructor(
            (Request((1, 2), 1), Request((1,), 1)), b""
        )  # a prompt equal to the prefix
    with pytest.raises(TracerError, match="no advice"):
        constructor(REQUESTS, b"x")
    with pytest.raises(TracerError, match="unconstrained"):
        constructor((Request((1, 2), 1, banned=(3,)), Request((1, 3), 1)), b"")
    with pytest.raises(TracerError, match="the context is 8"):
        constructor((Request((1, 2), 7), Request((1, 3), 1)), b"")
