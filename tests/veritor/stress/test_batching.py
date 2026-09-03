"""Serving-side scenarios S1, S3, S4, S5, S6 and S8: what the schedule does to the circuit.

Every scenario runs the same requests through ``ClusterG`` (RU = step; the
schedule is advice, mechanism M4) and ``RequestsG`` (RU = request; no advice,
mechanism M1) and checks the catalogue's claim: the tokens are the
reference's under every schedule, ``RequestsG``'s circuit never notices the
schedule, and ``ClusterG`` pays for the schedule in advice bits and, when the
schedule recomputes, in replay cost.
"""

from __future__ import annotations

from veritor.constructors import (
    ClusterG,
    Join,
    Request,
    RequestsG,
    Schedule,
    reference_generate,
    schedule_fcfs,
)
from veritor.core.description import REPLAY
from veritor.stress.models import Model
from veritor.stress.rows import Recorder
from veritor.stress.serving import Served, serve

REQUESTS = (
    Request((1, 2, 3), 3),
    Request((5,), 4),
    Request((7, 0), 2),
    Request((2, 2, 2, 2), 3),
    Request((4, 6), 4),
)
PODS, SLOTS, STEPS = 2, 2, 10


def cluster(model: Model, requests: tuple[Request, ...], schedule: Schedule) -> Served:
    constructor = ClusterG(model.shape, schedule.pods, schedule.slots, schedule.steps)
    layout = constructor.output_layout(requests, schedule)
    return serve(
        constructor,
        requests,
        schedule.encode(),
        model.gate_set,
        model.weights,
        layout,
        len(requests),
    )


def per_request(model: Model, requests: tuple[Request, ...]) -> Served:
    constructor = RequestsG(model.shape)
    layout = constructor.output_layout(requests)
    return serve(
        constructor, requests, b"", model.gate_set, model.weights, layout, len(requests)
    )


def reordered_fcfs(
    requests: tuple[Request, ...],
    order: tuple[int, ...],
    pods: int,
    slots: int,
    steps: int,
) -> Schedule:
    """FCFS over the requests admitted in ``order``, joins naming the original indices."""

    schedule = schedule_fcfs(
        tuple(requests[index] for index in order), pods, slots, steps
    )
    joins = tuple(
        Join(
            join.pod,
            join.step,
            join.slot,
            order[join.request],
            join.length,
            join.resume,
            join.chunk,
        )
        for join in schedule.joins
    )
    return Schedule(pods, slots, steps, tuple(sorted(joins)))


def request_kinds(
    constructor: RequestsG, requests: tuple[Request, ...]
) -> tuple[str, ...]:
    """The digest of each request's kind, request by request."""

    return tuple(
        constructor.request(*constructor.kind_of(request)).digest
        for request in requests
    )


# -- S1: continual batching ---------------------------------------------------------


def test_s1_continual_batching_under_three_schedules(
    scenario: Recorder, model: Model
) -> None:
    reference = reference_generate(model.shape, model.parameters, REQUESTS)
    schedules = {
        "fcfs": schedule_fcfs(REQUESTS, PODS, SLOTS, STEPS),
        "reversed admission": reordered_fcfs(
            REQUESTS, (4, 3, 2, 1, 0), PODS, SLOTS, STEPS
        ),
        "one slot per pod": Schedule(
            PODS, SLOTS, STEPS, schedule_fcfs(REQUESTS, PODS, 1, STEPS).joins
        ),
    }
    served = {
        name: cluster(model, REQUESTS, schedule) for name, schedule in schedules.items()
    }
    requests = per_request(model, REQUESTS)

    # identical outputs under every schedule and with request units
    assert (
        all(run.tokens == reference for run in served.values())
        and requests.tokens == reference
    )
    # three schedules, three circuits; the request circuit is one whatever the admission order
    assert len({run.digest for run in served.values()}) == 3
    assert all(
        run.advice_bits == 8 * len(schedule.encode())
        for run, schedule in zip(served.values(), schedules.values(), strict=True)
    )
    assert requests.advice_bits == 0
    # invariance: the kind of every request is the same however the schedule admits them
    kinds = request_kinds(RequestsG(model.shape), REQUESTS)
    for order in ((0, 1, 2, 3, 4), (4, 3, 2, 1, 0), (2, 0, 4, 1, 3)):
        admitted = tuple(REQUESTS[index] for index in order)
        assert request_kinds(RequestsG(model.shape), admitted) == tuple(
            kinds[index] for index in order
        )
    for run in served.values():
        assert set(kinds) <= set(requests.kinds(REPLAY)) and set(kinds).isdisjoint(
            run.kinds(REPLAY)
        )

    for letter, (name, schedule) in zip("abc", schedules.items(), strict=True):
        run = served[name]
        scenario.record(
            id=f"S1{letter}",
            what=f"continual batching, ClusterG (RU = step), {name}: {len(schedule.joins)} joins on {PODS}x{SLOTS} pods x slots",
            mechanism="M4",
            advice_bits=run.advice_bits,
            capacity_bits=run.capacity_bits,
            overhead=run.overhead,
            description_bytes=run.description_bytes,
            verdict="outputs = reference; the schedule is the advice and the step kinds follow it",
            notes=run.notes(
                f"{len(run.kinds(REPLAY))} RU kinds; description digest differs per schedule"
            ),
        )
    scenario.record(
        id="S1d",
        what="continual batching, RequestsG (RU = request): the same requests, any schedule",
        mechanism="M1",
        advice_bits=0,
        capacity_bits=requests.capacity_bits,
        overhead=requests.overhead,
        description_bytes=requests.description_bytes,
        verdict="outputs = reference; one circuit for all three schedules; request kinds invariant under admission order",
        notes=requests.notes(
            "the schedule is the server's business: not in x, not in a, not in the circuit"
        ),
    )


# -- S3: preemption by recompute -------------------------------------------------------


def test_s3_preemption_by_recompute(scenario: Recorder, model: Model) -> None:
    """Request 1 is evicted after 2 of its 4 tokens and prefilled again 3 steps later."""

    reference = reference_generate(model.shape, model.parameters, REQUESTS)
    plain = schedule_fcfs(REQUESTS, PODS, SLOTS, STEPS)
    evicted = Schedule(
        PODS,
        SLOTS,
        STEPS,
        tuple(
            sorted(
                (
                    *(join for join in plain.joins if join.request != 1),
                    Join(
                        0, 0, 1, 1, 2
                    ),  # the first attempt: 2 tokens, then the eviction
                    Join(
                        1, 5, 1, 1, 4
                    ),  # prefilled again from scratch on pod 1: positions 0, 1 recomputed, 2, 3 streamed
                )
            )
        ),
    )
    before, after = cluster(model, REQUESTS, plain), cluster(model, REQUESTS, evicted)
    requests = per_request(model, REQUESTS)

    assert after.tokens == reference == before.tokens == requests.tokens
    assert (
        after.digest != before.digest and after.price.honest > before.price.honest
    )  # the recompute is in the circuit
    second = next(
        index
        for index, join in enumerate(evicted.joins)
        if join.request == 1 and join.step == 5
    )
    assert (
        evicted.streamed_before(REQUESTS)[second] == 2
    )  # positions 0 and 1 are recomputed, not streamed again
    # with request units the eviction is invisible: the circuit is the one without it
    assert requests.digest == per_request(model, REQUESTS).digest

    scenario.record(
        id="S3a",
        what="preemption by recompute, ClusterG: request evicted after 2 tokens, re-prefilled 3 steps later on another pod",
        mechanism="M4",
        advice_bits=after.advice_bits,
        capacity_bits=after.capacity_bits,
        overhead=after.overhead,
        description_bytes=after.description_bytes,
        verdict="outputs = reference; both attempts are in the circuit, the recomputed positions declared but not output",
        notes=after.notes(
            f"honest replay cost {after.price.honest} vs {before.price.honest} without the eviction"
        ),
    )
    scenario.record(
        id="S3b",
        what="preemption by recompute, RequestsG: the same eviction",
        mechanism="M1",
        advice_bits=0,
        capacity_bits=requests.capacity_bits,
        overhead=requests.overhead,
        description_bytes=requests.description_bytes,
        verdict="circuit identical with and without the eviction (digest equal); outputs = reference",
        notes=requests.notes("recompute is the server's cost, not the statement's"),
    )


# -- S4: preemption by swap ----------------------------------------------------------


def test_s4_preemption_by_swap(scenario: Recorder, model: Model) -> None:
    """Request 1 decodes 2 tokens, its KV cache sits out 3 steps, and it resumes where it left."""

    reference = reference_generate(model.shape, model.parameters, REQUESTS)
    plain = schedule_fcfs(REQUESTS, PODS, SLOTS, STEPS)
    gap = 3
    swapped = Schedule(
        PODS,
        SLOTS,
        STEPS,
        tuple(
            sorted(
                (
                    *(join for join in plain.joins if join.request != 1),
                    Join(
                        0, 0, 1, 1, 2
                    ),  # prefill and one decode, then the cache is swapped out
                    Join(
                        0, 2 + gap, 1, 1, 2, resume=True
                    ),  # resumes: reads the KV declared ``gap`` steps earlier
                )
            )
        ),
    )
    before, after = cluster(model, REQUESTS, plain), cluster(model, REQUESTS, swapped)
    requests = per_request(model, REQUESTS)

    assert after.tokens == reference == before.tokens == requests.tokens
    assert after.digest != before.digest  # the steps are laid out differently ...
    assert after.price.honest == before.price.honest  # ... but nothing is recomputed
    assert requests.digest == per_request(model, REQUESTS).digest

    scenario.record(
        id="S4a",
        what=f"preemption by swap, ClusterG: KV cache retained across a gap of {gap} steps (Schedule v3 resume)",
        mechanism="M4",
        advice_bits=after.advice_bits,
        capacity_bits=after.capacity_bits,
        overhead=after.overhead,
        description_bytes=after.description_bytes,
        verdict="outputs = reference; the resumed decode step reads the KV rows declared 3 steps earlier; honest cost unchanged",
        notes=after.notes(
            "Join.resume marks the attempt; the gap is wiring, not a kind: decode_c is the same kind resumed or not"
        ),
    )
    scenario.record(
        id="S4b",
        what="preemption by swap, RequestsG: the same swap",
        mechanism="M1",
        advice_bits=0,
        capacity_bits=requests.capacity_bits,
        overhead=requests.overhead,
        description_bytes=requests.description_bytes,
        verdict="circuit identical with and without the swap (digest equal)",
        notes=requests.notes(),
    )


# -- S5: chunked prefill -------------------------------------------------------------


def test_s5_chunked_prefill(scenario: Recorder, model: Model) -> None:
    """A 9-token prompt prefilled 3 tokens per step over 3 steps, then decoded."""

    long = (Request((1, 2, 3, 4, 5, 6, 7, 0, 1), 3), Request((5,), 2))
    reference = reference_generate(model.shape, model.parameters, long)
    steps = 6
    plain = Schedule(1, 2, steps, (Join(0, 0, 0, 0, 3), Join(0, 0, 1, 1, 2)))
    chunked = Schedule(1, 2, steps, (Join(0, 0, 0, 0, 5, chunk=3), Join(0, 0, 1, 1, 2)))
    before, after = cluster(model, long, plain), cluster(model, long, chunked)
    requests = per_request(model, long)

    assert after.tokens == reference == before.tokens == requests.tokens
    assert (
        after.digest != before.digest
    )  # step units: chunk_3 and prefill over 6 cached positions are new kinds
    assert after.price.honest == before.price.honest  # the same gates, cut differently
    assert requests.digest == per_request(model, long).digest

    scenario.record(
        id="S5a",
        what="chunked prefill, ClusterG: a 9-token prompt in 3 chunks of 3 over 3 steps",
        mechanism="M4",
        advice_bits=after.advice_bits,
        capacity_bits=after.capacity_bits,
        overhead=after.overhead,
        description_bytes=after.description_bytes,
        verdict="outputs = reference; the step-RU description differs (chunk kinds), the honest cost does not",
        notes=after.notes(
            "Join.chunk carries the chunk size; each chunk declares its KV rows for the next step"
        ),
    )
    scenario.record(
        id="S5b",
        what="chunked prefill, RequestsG: the same prompt",
        mechanism="M1",
        advice_bits=0,
        capacity_bits=requests.capacity_bits,
        overhead=requests.overhead,
        description_bytes=requests.description_bytes,
        verdict="the request circuit is the sequential one, identical with and without chunking; values equal",
        notes=requests.notes(),
    )


# -- S6: prefill/decode disaggregation --------------------------------------------------


def test_s6_prefill_decode_disaggregation(scenario: Recorder, model: Model) -> None:
    """Every request is prefilled on pod 0 and decoded on pod 1, reading pod 0's declared KV rows."""

    requests_ = REQUESTS[:3]
    reference = reference_generate(model.shape, model.parameters, requests_)
    joins = []
    for index, request in enumerate(requests_):
        joins.append(
            Join(0, index, 0, index, 1)
        )  # prefill only, on pod 0, one request per step
        if request.max_new > 1:
            joins.append(
                Join(1, index + 1, index, index, request.max_new - 1, resume=True)
            )  # decode on pod 1
    disaggregated = Schedule(2, 3, 6, tuple(sorted(joins)))
    colocated = schedule_fcfs(requests_, 2, 3, 6)
    apart, together = (
        cluster(model, requests_, disaggregated),
        cluster(model, requests_, colocated),
    )
    requests = per_request(model, requests_)

    assert apart.tokens == reference == together.tokens == requests.tokens
    assert apart.price.honest == together.price.honest
    assert apart.digest != together.digest

    scenario.record(
        id="S6a",
        what="prefill/decode disaggregation, ClusterG: prefill on pod 0, decode on pod 1 (resume across pods)",
        mechanism="M4",
        advice_bits=apart.advice_bits,
        capacity_bits=apart.capacity_bits,
        overhead=apart.overhead,
        description_bytes=apart.description_bytes,
        verdict="outputs = reference; a decode step reads another pod's declared KV rows through its ports",
        notes=apart.notes(
            "the cluster is synchronous and time-major: any step may read what any earlier step of any pod declared"
        ),
    )
    scenario.record(
        id="S6b",
        what="prefill/decode disaggregation, RequestsG",
        mechanism="M1",
        advice_bits=0,
        capacity_bits=requests.capacity_bits,
        overhead=requests.overhead,
        description_bytes=requests.description_bytes,
        verdict="pods are not in the statement: the circuit is the sequential one",
        notes=requests.notes(),
    )


# -- S8: retries / duplicate execution --------------------------------------------------


def test_s8_a_discarded_duplicate_is_not_in_the_circuit(
    scenario: Recorder, model: Model
) -> None:
    """The server ran request 2 twice and kept the first run: with request units nothing changes; with step
    units the duplicate is in the circuit only if the schedule (the advice) declares it."""

    reference = reference_generate(model.shape, model.parameters, REQUESTS)
    plain = schedule_fcfs(REQUESTS, PODS, SLOTS, STEPS)
    declared = Schedule(
        PODS,
        SLOTS,
        STEPS,
        tuple(
            sorted((*plain.joins, Join(1, 6, 1, 2, 2)))
        ),  # a second full attempt of request 2
    )
    requests, again = per_request(model, REQUESTS), per_request(model, REQUESTS)
    without, with_duplicate = (
        cluster(model, REQUESTS, plain),
        cluster(model, REQUESTS, declared),
    )

    assert requests.digest == again.digest and requests.tokens == reference
    assert without.tokens == with_duplicate.tokens == reference
    assert (
        with_duplicate.price.honest > without.price.honest
    )  # the declared duplicate is replayed like any step
    assert (
        with_duplicate.outputs == without.outputs
    )  # its tokens are declared, not output

    scenario.record(
        id="S8a",
        what="retries, RequestsG: request 2 executed twice, one run discarded",
        mechanism="M1",
        advice_bits=0,
        capacity_bits=requests.capacity_bits,
        overhead=requests.overhead,
        description_bytes=requests.description_bytes,
        verdict="the compiled circuit is identical with and without the discarded work (digest equal)",
        notes=requests.notes(
            "what is not in (x, a) is not in C: a discarded execution has no gates"
        ),
    )
    scenario.record(
        id="S8b",
        what="retries, ClusterG: the duplicate attempt declared in the schedule",
        mechanism="M4",
        advice_bits=with_duplicate.advice_bits,
        capacity_bits=with_duplicate.capacity_bits,
        overhead=with_duplicate.overhead,
        description_bytes=with_duplicate.description_bytes,
        verdict="outputs unchanged; the duplicate's steps are replay units, its tokens declared but not output",
        notes=with_duplicate.notes(
            f"honest replay cost {with_duplicate.price.honest} vs {without.price.honest} when the schedule omits it"
        ),
    )
