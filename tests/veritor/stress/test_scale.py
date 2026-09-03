"""S2 / E1: scale.  ``RequestsG`` over 10^3 .. 10^6 requests of realistic, heavy-tailed lengths.

The finding: a request's generated positions are each their own run of the
root's ``Out``, so the verifier's ``max_output_runs = 256`` is exceeded as
soon as the distinct request shapes generate more than 256 positions between
them -- here already at 10^3 requests -- whatever the number of requests.
Grouping same-kind requests under one ``repeat`` (``RequestsG`` v2) keeps the
description and the compile time a function of the number of *shapes*, not of
requests, so 10^6 requests compile in the time 10^3 do; the run count is
``sum over shapes of max_new`` and is paid for with an explicit, bounded
raise of the limit whose cost is the ``O(R log R)`` distinctness sweep.
"""

from __future__ import annotations

import random
import time

import pytest

from veritor.compile import CompileError, Compiler
from veritor.constructors import Request, RequestsG
from veritor.core.limits import CompilationLimits
from veritor.stress.measure import compile_scenario, price
from veritor.stress.models import Model
from veritor.stress.rows import Recorder

LIMITS = CompilationLimits(max_output_runs=1 << 12, max_output_runs_total=1 << 16)
"""The simulated datacenter's limits: what the verifier accepts once the default is shown to be too small."""


def workload(count: int, context: int, seed: int = 2026) -> tuple[Request, ...]:
    """``count`` requests with Zipf-like prompt lengths and bimodal generation lengths, scaled to ``context``.

    Production traces have Zipf-distributed request lengths and bimodal
    response lengths (BurstGPT; ``docs/notes/datacenter-realities.md`` section
    12); the toy's context of 16 positions compresses that to prompts of 1 to
    ``context - 2`` and generations of 1 to 3 (short, half the time) or up to
    the room the prompt leaves (long).
    """

    rng = random.Random(seed)
    requests = []
    for _ in range(count):
        prompt = min(context - 2, int(rng.paretovariate(1.2)))  # Zipf-like: many short prompts, a heavy tail
        room = context - prompt
        max_new = rng.randint(1, min(3, room)) if rng.random() < 0.5 else rng.randint(1, room)
        requests.append(Request(tuple(rng.randrange(8) for _ in range(prompt)), max_new))
    return tuple(requests)


def test_s2_the_default_run_limit_is_exceeded_by_shapes_not_by_requests(model: Model) -> None:
    """A thousand requests of a hundred-odd shapes already exceed 256 root output runs; a thousand of one
    shape do not."""

    constructor = RequestsG(model.shape)
    varied = workload(1000, model.shape.context)
    kinds = {constructor.kind_of(request) for request in varied}
    runs = sum(max_new for _, max_new, _ in kinds)
    assert 60 < len(kinds) < 256 < runs
    description, inputs = constructor(varied, b"")
    with pytest.raises(CompileError, match="max_output_runs = 256"):
        Compiler(model.gate_set).compile(description, inputs)
    compiled = Compiler(model.gate_set, LIMITS).compile(description, inputs)
    assert 256 < len(compiled.index.root.frame.definition.out_runs) <= runs  # weaving merges a few

    uniform = tuple(Request((1, 2, 3), 4) for _ in range(1000))
    description, inputs = constructor(uniform, b"")
    assert len(Compiler(model.gate_set).compile(description, inputs).index.root.frame.definition.out_runs) == 4


@pytest.mark.parametrize("count, letter", ((1_000, "a"), (10_000, "b"), (100_000, "c")))
def test_s2_scale(scenario: Recorder, model: Model, count: int, letter: str) -> None:
    run_scale(scenario, model, count, f"S2{letter}")


@pytest.mark.slow
def test_e1_a_million_requests(scenario: Recorder, model: Model) -> None:
    run_scale(scenario, model, 1_000_000, "E1")


def run_scale(scenario: Recorder, model: Model, count: int, identifier: str) -> None:
    constructor = RequestsG(model.shape)
    requests = workload(count, model.shape.context)
    kinds = {constructor.kind_of(request) for request in requests}

    measurement = compile_scenario(constructor, requests, b"", model.gate_set, limits=LIMITS)
    started = time.perf_counter()
    table = measurement.compiled.kind_table()
    table_seconds = time.perf_counter() - started
    priced = price(table)
    root = measurement.compiled.index.root.frame.definition

    assert measurement.compiled.index.replay_units.count == count + 1
    assert measurement.compiled.index.input_count == sum(len(request.prompt) for request in requests)
    assert 256 < len(root.out_runs) <= sum(max_new for _, max_new, _ in kinds) <= LIMITS.max_output_runs
    assert measurement.description_bytes < 400_000  # a function of the shapes, not of the requests
    assert measurement.compile_seconds < 2.0

    scenario.record(
        id=identifier,
        what=f"scale, RequestsG: {count:,} requests of {len(kinds)} shapes (Zipf-like prompts, bimodal generation)",
        mechanism="M1",
        advice_bits=measurement.advice_bits,
        capacity_bits=priced.capacity_bits,
        overhead=priced.overhead,
        description_bytes=measurement.description_bytes,
        verdict=(
            f"description {measurement.description_bytes:,} B, trace {measurement.trace_seconds * 1000:.0f} ms, "
            f"compile {measurement.compile_seconds * 1000:.0f} ms, kind table {table_seconds * 1000:.0f} ms, "
            f"bound {priced.bound_seconds * 1000:.0f} ms"
        ),
        notes=(
            f"{len(root.out_runs)} root output runs (one per generated position of each shape): exceeds the default 256 "
            f"(compiled with max_output_runs = {LIMITS.max_output_runs}); requests of one kind are one repeat, "
            f"so the description does not grow with the requests"
        ),
    )
