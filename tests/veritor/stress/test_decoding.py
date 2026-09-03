"""Decoding-side scenarios C4 (constrained decoding) and C6 (prefix caching).

C4 puts a request's public banned-token list into the circuit: the list is
``in`` gates, ``allowed_row`` VUs turn it into ``vocab`` flags once per
request, and ``masked_argmax`` / ``masked_sample`` decide among the allowed
tokens (mechanism M5, decided in-circuit).  C6 serves ``n`` requests sharing
a system prompt two ways: the prefix as its own RU whose declared outputs
are its KV blocks (``PrefixG``, route A: a wider boundary, less replay) and
each request recomputing the prefix inside its own RU (``RequestsG``, route
B, mechanism M1).
"""

from __future__ import annotations

import math
import random

import pytest

from veritor.constructors import PrefixG, Request, RequestsG, reference_generate
from veritor.constructors.lm import allowed_mask
from veritor.core.description import REPLAY, VERIFICATION
from veritor.stress.models import Model
from veritor.stress.rows import Recorder
from veritor.stress.serving import Served, serve


def per_request(model: Model, requests: tuple[Request, ...]) -> Served:
    constructor = RequestsG(model.shape)
    layout = constructor.output_layout(requests)
    return serve(constructor, requests, b"", model.gate_set, model.weights, layout, len(requests))


def constrained(model: Model, seed: int) -> tuple[Request, ...]:
    """Six requests, half of them banning one to three tokens -- always the token the unconstrained
    decoder would have produced first, so the mask bites; random words for a sampling shape."""

    rng = random.Random(seed)
    shape = model.shape
    plain = []
    for _ in range(6):
        prompt = tuple(rng.randrange(shape.vocab) for _ in range(rng.randint(1, 4)))
        max_new = rng.randint(2, 4)
        randomness = tuple(rng.randrange(1 << shape.random_bits) for _ in range(max_new)) if shape.sampling else ()
        plain.append(Request(prompt, max_new, randomness))
    unconstrained = reference_generate(shape, model.parameters, tuple(plain))
    requests = []
    for index, (request, tokens) in enumerate(zip(plain, unconstrained, strict=True)):
        if index % 2:
            others = [token for token in range(shape.vocab) if token != tokens[0]]
            banned = tuple(sorted({tokens[0], *rng.sample(others, rng.randint(0, 2))}))
            request = Request(request.prompt, request.max_new, request.randomness, banned)
        requests.append(request)
    return tuple(requests)


# -- C4: constrained decoding ------------------------------------------------------------


@pytest.mark.parametrize("head, letter", (("argmax", "a"), ("sample", "b")))
def test_c4_constrained_decoding(scenario: Recorder, model: Model, sampled: Model, head: str, letter: str) -> None:
    served_model = model if head == "argmax" else sampled
    shape = served_model.shape
    requests = constrained(served_model, seed=11)
    plain = tuple(Request(r.prompt, r.max_new, r.randomness) for r in requests)

    run = per_request(served_model, requests)
    unmasked = per_request(served_model, plain)

    # the circuit decides what a reference sampler with the mask decides, and never a banned token
    assert run.tokens == reference_generate(shape, served_model.parameters, requests)
    for tokens, request in zip(run.tokens, requests, strict=True):
        assert not set(tokens) & set(request.banned)
        mask = allowed_mask(shape.vocab, request.banned)
        assert mask is None or all(mask[token] for token in tokens)
    assert run.tokens != unmasked.tokens  # the mask bit on at least one position
    # the mask is in-circuit: the banned ids are inputs, the flags are VUs, the head kinds are the masked ones
    lm = RequestsG(shape).lm
    assert run.measurement.compiled.index.input_count == unmasked.measurement.compiled.index.input_count + sum(
        len(r.banned) for r in requests
    )
    kinds = {row.kind: row for row in run.measurement.compiled.index.kinds()}
    plain_head = (lm.argmax() if head == "argmax" else lm.sample()).digest
    masked_head = (lm.masked_argmax() if head == "argmax" else lm.masked_sample()).digest
    banned_positions = sum(r.max_new for r in requests if r.banned)
    assert kinds[masked_head].copies == banned_positions and kinds[masked_head].role == VERIFICATION
    assert kinds[plain_head].copies == sum(r.max_new for r in requests if not r.banned)
    per_token = kinds[masked_head].size - kinds[plain_head].size
    expected = 2 * shape.vocab + 7 if head == "argmax" else shape.vocab
    assert per_token == expected
    rows = {b: kinds[lm.allowed_row(b).digest] for b in {len(r.banned) for r in requests if r.banned}}
    assert all(row.size == 3 * b - 1 and row.role == VERIFICATION for b, row in rows.items())
    assert run.advice_bits == 0

    scenario.record(
        id=f"C4{letter}",
        what=f"constrained decoding, RequestsG with the {head} head: 3 of 6 requests ban 1-3 tokens (public, in x)",
        mechanism="M5",
        advice_bits=0,
        capacity_bits=run.capacity_bits,
        overhead=run.overhead,
        description_bytes=run.description_bytes,
        verdict=(
            f"outputs = reference sampler with the mask; banned tokens never emitted; "
            f"+{per_token} gates per generated token (masked_{head} vs {head}), +{3 * 1 - 1}..{3 * 3 - 1} gates x vocab once per request for the mask"
        ),
        notes=run.notes(
            "the mask is allowed_row VUs over in gates: nothing about the constraint is advice",
            "ClusterG rejects banned lists: its step kinds carry no per-occupant mask ports (gap)",
        ),
    )


# -- C6: prefix caching ---------------------------------------------------------------


def shared(model: Model, count: int, prefix: int, suffix: int, max_new: int) -> tuple[Request, ...]:
    rng = random.Random(prefix * 31 + count)
    system = tuple(rng.randrange(model.shape.vocab) for _ in range(prefix))
    return tuple(
        Request((*system, *(rng.randrange(model.shape.vocab) for _ in range(suffix))), max_new) for _ in range(count)
    )


def test_c6_prefix_caching_two_routes(scenario: Recorder, model: Model) -> None:
    """Eight requests sharing an 8-token system prompt, each with a 2-token suffix and 3 generated tokens."""

    count, prefix_length = 8, 8
    requests = shared(model, count, prefix_length, suffix=2, max_new=3)
    reference = reference_generate(model.shape, model.parameters, requests)
    prefix = PrefixG(model.shape)
    route_a = serve(prefix, requests, b"", model.gate_set, model.weights, prefix.output_layout(requests), count)
    route_b = per_request(model, requests)

    assert route_a.tokens == reference == route_b.tokens
    kv_words = model.shape.state_size(prefix_length)
    prefix_row = next(
        row for row in route_a.measurement.compiled.index.kinds() if row.kind == prefix.prefix_unit(prefix_length).digest
    )
    assert prefix_row.role == REPLAY and prefix_row.copies == 1 and prefix_row.out_count == kv_words
    # route A replays the prefix once; route B ``count`` times
    assert route_b.price.honest > route_a.price.honest
    saved = route_b.price.honest - route_a.price.honest
    assert saved == (count - 1) * prefix_row.replay_cost
    # route A's boundary is wider by the prefix's declared KV rows
    boundary_a = route_a.measurement.compiled.index.boundary().count
    boundary_b = route_b.measurement.compiled.index.boundary().count
    assert boundary_a == boundary_b - (count - 1) * prefix_length + kv_words

    # the crossover in replay work: with ``k`` requests route A saves ``(k - 1) * W_prefix`` replay
    # and costs ``kv_words`` boundary words; both are recorded as the rows' numbers
    scenario.record(
        id="C6a",
        what=f"prefix caching, route A (PrefixG): {count} requests share an {prefix_length}-token prefix computed by one RU",
        mechanism="M1",
        advice_bits=0,
        capacity_bits=route_a.capacity_bits,
        overhead=route_a.overhead,
        description_bytes=route_a.description_bytes,
        verdict=(
            f"outputs = reference; the prefix RU declares W_R = {kv_words} KV words read by {count} suffix RUs; "
            f"honest replay cost {route_a.price.honest} vs {route_b.price.honest} (saves {saved})"
        ),
        notes=route_a.notes(
            f"boundary {boundary_a} vs {boundary_b} words: +{kv_words} declared cache rows, -{(count - 1) * prefix_length} repeated prompt tokens",
            "the shared prefix is the longest common prefix of the prompts, a function of x: no advice",
        ),
    )
    scenario.record(
        id="C6b",
        what=f"prefix caching, route B (RequestsG): each of the {count} requests recomputes the {prefix_length}-token prefix",
        mechanism="M1",
        advice_bits=0,
        capacity_bits=route_b.capacity_bits,
        overhead=route_b.overhead,
        description_bytes=route_b.description_bytes,
        verdict=f"outputs = reference; {count} x prefix replay ({prefix_row.replay_cost} each); boundary is prompts and tokens only",
        notes=route_b.notes(
            f"crossover: route A wins on replay work for k >= 2 requests (saves (k - 1) x {prefix_row.replay_cost}) "
            f"and pays {kv_words} boundary words once, which costs capacity: uncapped U {math.ceil(route_a.price.bound.knapsack_bits)} (A) "
            f"vs {math.ceil(route_b.price.bound.knapsack_bits)} (B) bits and overhead {route_a.overhead:.2f} vs {route_b.overhead:.2f}"
        ),
    )
