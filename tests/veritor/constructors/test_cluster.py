"""The cluster constructor: batching is semantically transparent, kinds are shared, marks tile.

Default fixture: ``LMShape(vocab=8, d_model=4, heads=2, layers=1, context=6,
width=16)``, four requests (prompt lengths 1-3, ``max_new`` 1-4) on 2 pods x
2 slots for 6 steps under FCFS.  Measured on the reference machine: 6,564
gates, 8 replay units (the weights and 7 occupied steps), 1,065
verification units, 42 kinds; the description is 34.5 KB, traced in about
4 ms and compiled in about 4 ms; evaluating every gate takes about 50 ms.
"""

from __future__ import annotations

import json
import time
from fractions import Fraction

import pytest

from veritor.analysis import CostParameters, PolicyGrid
from veritor.compile import CompileError, Compiler
from veritor.constructors import (
    ClusterG,
    Join,
    LMShape,
    Parameters,
    Request,
    Schedule,
    TracerError,
    random_parameters,
    reference_generate,
    schedule_fcfs,
)
from veritor.core import Compiled, VerificationPolicy, make_isa_gate_set
from veritor.research import Bound, Cost, Optimize

SHAPE = LMShape(vocab=8, d_model=4, heads=2, layers=1, context=6, width=16)
DEEP = LMShape(vocab=8, d_model=4, heads=2, layers=2, context=6, width=16)
REQUESTS = (Request((1, 2, 3), 3), Request((5,), 2), Request((7, 0), 4), Request((2, 2, 2), 1))
GATES = make_isa_gate_set(16)


def compile_run(constructor: ClusterG, requests: tuple[Request, ...], schedule: Schedule) -> Compiled:
    description, inputs = constructor(requests, schedule.encode())
    return Compiler(GATES).compile(description, inputs)


def generated(
    constructor: ClusterG,
    compiled: Compiled,
    requests: tuple[Request, ...],
    schedule: Schedule,
    parameters: Parameters,
) -> tuple[tuple[int, ...], ...]:
    """The circuit's outputs regrouped by request through ``output_layout``."""

    inputs = constructor.flatten_inputs(requests, schedule)
    values = compiled.circuit.evaluate(inputs, parameters.flatten())
    outputs = [values[address] for address in compiled.circuit.outputs]
    layout = constructor.output_layout(requests, schedule)
    assert len(outputs) == len(layout)
    grouped: list[list[int]] = [[] for _ in requests]
    for (request, position), token in zip(layout, outputs, strict=True):
        assert position == len(grouped[request])
        grouped[request].append(token)
    return tuple(tuple(tokens) for tokens in grouped)


@pytest.fixture(scope="module")
def fcfs() -> tuple[ClusterG, Schedule, Compiled]:
    constructor = ClusterG(SHAPE, pods=2, slots=2, steps=6)
    schedule = schedule_fcfs(REQUESTS, 2, 2, 6)
    return constructor, schedule, compile_run(constructor, REQUESTS, schedule)


def test_fcfs_run_generates_what_the_reference_generates(fcfs) -> None:
    constructor, schedule, compiled = fcfs
    parameters = random_parameters(SHAPE, seed=1)

    assert schedule.active_steps(REQUESTS) == {0: 3, 1: 2, 2: 4, 3: 1}
    assert generated(constructor, compiled, REQUESTS, schedule, parameters) == reference_generate(
        SHAPE, parameters, REQUESTS
    )
    # the prompts are the in gates, laid out by (pod, step) then slot
    assert constructor.flatten_inputs(REQUESTS, schedule) == (1, 2, 3, 5, 7, 0, 2, 2, 2)
    assert compiled.circuit.input_count == 9 and compiled.circuit.weight_count == SHAPE.weight_count
    assert constructor.output_layout(REQUESTS, schedule) == (
        (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2), (2, 3), (3, 0),
    )
    # other parameters, same circuit: the weights are not part of the description
    other = random_parameters(SHAPE, seed=2)
    assert generated(constructor, compiled, REQUESTS, schedule, other) == reference_generate(SHAPE, other, REQUESTS)


def test_a_hand_written_schedule_is_semantically_transparent() -> None:
    """One slot per pod; request 1 takes request 0's slot at step 2 and cuts it to two tokens."""

    schedule = Schedule(2, 1, 6, (Join(0, 0, 0, 0), Join(0, 2, 0, 1), Join(1, 0, 0, 2), Join(1, 4, 0, 3)))
    constructor = ClusterG(SHAPE, pods=2, slots=1, steps=6)
    compiled = compile_run(constructor, REQUESTS, schedule)
    parameters = random_parameters(SHAPE, seed=7)

    assert schedule.active_steps(REQUESTS) == {0: 2, 1: 2, 2: 4, 3: 1}
    reference = reference_generate(SHAPE, parameters, REQUESTS)
    assert generated(constructor, compiled, REQUESTS, schedule, parameters) == (
        reference[0][:2], reference[1], reference[2], reference[3],
    )
    assert compiled.index.replay_units.count == 1 + 4 + 5  # weights, pod 0 steps 0-3, pod 1 steps 0-4


@pytest.mark.parametrize(
    "max_news",
    [(1, 2, 1), (1, 3, 2), (2, 1, 2, 1), (1, 1, 3), (3, 1, 1, 2)],
    ids=lambda m: "-".join(map(str, m)),
)
def test_mixed_lifetimes_in_one_pod_compile_and_decode_like_the_reference(max_news) -> None:
    """Occupants leaving at different steps make a step declare a strided subset of its tokens.

    Three or four slots in one pod, one-token prompts, `max_new` differing per
    slot: the root then declares the tokens of some occupants of a step and
    not others, a stride-`state_size + 1` run over part of one step copy.
    This is the case that once resolved to more gates than were declared.
    """

    requests = tuple(Request((1 + i,), max_new) for i, max_new in enumerate(max_news))
    constructor = ClusterG(SHAPE, pods=1, slots=len(requests), steps=max(max_news))
    schedule = schedule_fcfs(requests, 1, len(requests), max(max_news))
    compiled = compile_run(constructor, requests, schedule)
    parameters = random_parameters(SHAPE, seed=5)

    assert schedule.active_steps(requests) == dict(enumerate(max_news))
    assert generated(constructor, compiled, requests, schedule, parameters) == reference_generate(
        SHAPE, parameters, requests
    )
    assert len(compiled.circuit.outputs) == sum(max_news)


def test_two_layers_and_a_different_fcfs_shape() -> None:
    requests = (Request((3, 1), 4), Request((0,), 3), Request((6, 6, 6), 2), Request((2, 5), 1), Request((4,), 2))
    constructor = ClusterG(DEEP, pods=2, slots=2, steps=5)
    schedule = schedule_fcfs(requests, 2, 2, 5)
    compiled = compile_run(constructor, requests, schedule)
    parameters = random_parameters(DEEP, seed=3)

    active = schedule.active_steps(requests)
    reference = reference_generate(DEEP, parameters, requests)
    assert generated(constructor, compiled, requests, schedule, parameters) == tuple(
        reference[r][: active[r]] for r in range(len(requests))
    )


def definitions(description: bytes) -> list[dict]:
    return [entry["body"] for entry in json.loads(description)["definitions"]]


def test_kinds_are_shared_across_pods_and_steps() -> None:
    """Six identical requests: 1 pod x 2 slots x 9 steps and 3 pods x 2 slots x 3 steps have the same kinds."""

    requests = tuple(Request((1, 2), 3) for _ in range(6))
    runs = {}
    for pods, slots, steps in ((1, 2, 9), (3, 2, 3), (6, 1, 3)):
        constructor = ClusterG(SHAPE, pods=pods, slots=slots, steps=steps)
        schedule = schedule_fcfs(requests, pods, slots, steps)
        started = time.perf_counter()
        description, inputs = constructor(requests, schedule.encode())
        compiled = Compiler(GATES).compile(description, inputs)
        runs[pods] = (description, compiled, time.perf_counter() - started)
        assert generated(constructor, compiled, requests, schedule, random_parameters(SHAPE, 0)) == (
            reference_generate(SHAPE, random_parameters(SHAPE, 0), requests)
        )

    one, three, six = runs[1], runs[3], runs[6]
    assert len(definitions(one[0])) == len(definitions(three[0]))
    assert one[1].circuit.n == three[1].circuit.n and one[1].index.replay_units.count == 1 + 9
    assert three[1].index.replay_units.count == 1 + 9
    assert abs(len(one[0]) - len(three[0])) < 64  # the roots differ in digits only
    # six one-slot pods: three single-occupant step kinds instead of the three two-occupant ones
    assert len(definitions(six[0])) == len(definitions(three[0]))
    assert six[1].index.replay_units.count == 1 + 18
    for run, copies in ((three, 9), (six, 18)):
        replay_kinds = {row.kind: row.copies for row in run[1].index.kinds() if row.role == "replay"}
        assert len(replay_kinds) == 1 + 3 and sum(replay_kinds.values()) == 1 + copies
    step_kinds = [
        {row.kind for row in run[1].index.kinds() if row.role == "replay" and row.source_weights == 0}
        for run in (one, three, six)
    ]
    assert step_kinds[0] == step_kinds[1] and step_kinds[1].isdisjoint(step_kinds[2])
    assert max(one[2], three[2], six[2]) < 4 * min(one[2], three[2], six[2]) + 0.01
    # the weights unit and a step of prefills alone are closed (their ports are the weight cells);
    # a step with a decode occupant reads a token and a cache produced by an earlier step
    for run in (one, three, six):
        rows = {row.kind: row for row in run[1].index.kinds()}
        units = [row for row in rows.values() if row.role == "replay"]
        (weights,) = [row for row in units if row.source_weights > 0]
        steps = [row for row in units if row.source_weights == 0]
        assert weights.closed and rows[run[1].index.root.kind].closed
        assert {row.input_count == SHAPE.weight_count for row in steps} == {True, False}
        assert all(row.closed == (row.input_count == SHAPE.weight_count) for row in steps)


def test_marks_tile_every_gate_once_and_the_kv_cache_is_the_step_interface(fcfs) -> None:
    _, schedule, compiled = fcfs
    index, circuit = compiled.index, compiled.circuit
    occupancy = schedule.occupancy(REQUESTS)

    assert index.replay_units.count == 1 + len(occupancy) == 8
    weights = index.replay_units.unit(0)
    assert list(circuit.weights) == list(weights.interval) and circuit.Out(weights) == ()
    covered = [a for r in range(index.replay_units.count) for a in index.replay_units.unit(r).interval]
    assert covered == list(range(circuit.n))
    assert all(index.replay_units.owner(a) == r for r in range(8) for a in index.replay_units.unit(r).interval)
    inputs = set(circuit.inputs)
    for r, key in enumerate(sorted(occupancy), start=1):
        unit = index.replay_units.unit(r)
        occupants = occupancy[key]
        produced = sum(
            SHAPE.state_size(len(REQUESTS[o.request].prompt) if o.generated == 0 else 1) + 1 for o in occupants
        )
        assert unit.role == "replay" and len(circuit.Out(unit)) == produced
        prompts = sum(len(REQUESTS[o.request].prompt) for o in occupants if o.generated == 0)
        assert len(inputs & set(unit.interval)) == prompts
        reads = set(circuit.In(unit))
        assert set(circuit.weights) <= reads  # every step reads the whole model through ports
        for other in reads - set(circuit.weights):
            assert index.replay_units.owner(other) < r  # and only earlier steps of its pod
    verification = sum(index.verification_units(r).count for r in range(8))
    assert verification == index.verification_unit_count == 1065
    assert set(circuit.outputs) <= set(index.boundary())
    assert compiled.circuit.n == 6564


def test_bound_cost_and_optimize_run_over_the_cluster(fcfs) -> None:
    _, _, compiled = fcfs
    grid = tuple(Fraction(k, 4) for k in range(1, 5))
    for eta in (Fraction(1, 100), Fraction(1, 10)):
        table = {
            (q, s): Bound(compiled, VerificationPolicy(q, s), eta) for q in grid for s in grid
        }
        assert all(0 <= result.bits <= result.out_bits == 16 * 10 for result in table.values())
        assert table[(1, 1)].bits == 0.0
        for q in grid:
            for s in grid:
                if q < 1:
                    assert table[(q + Fraction(1, 4), s)].bits <= table[(q, s)].bits + 1e-9
                if s < 1:
                    assert table[(q, s + Fraction(1, 4))].bits <= table[(q, s)].bits + 1e-9
    assert Bound(compiled, VerificationPolicy(0, 1), Fraction(1, 2)).capped

    policy = VerificationPolicy(Fraction(1, 2), Fraction(1, 2))
    cost = Cost(compiled, policy, CostParameters(hash_cost=1, proof_overhead=0))
    assert cost.boundary == compiled.index.boundary().count == 139  # 9 prompt tokens + 130 declared outputs
    assert cost.weights == SHAPE.weight_count
    full = Cost(compiled, VerificationPolicy(1, 1), CostParameters(hash_cost=1, proof_overhead=0))
    assert full.commit_interior == 2 * cost.commit_interior and full.proof == 4 * cost.proof
    assert full.total > cost.total > 0
    # the decode steps read the caches of earlier steps, which the honest server does not keep:
    # sampling any of them re-executes the run, so the recomputation is not linear in ``q``
    rows = {row.kind: row for row in compiled.kind_table().rows}
    honest = rows[compiled.index.root.kind].replay_cost  # the replay units tile the run
    assert honest == sum(row.copies * row.replay_cost for row in rows.values() if row.role == "replay")
    assert full.recompute == honest
    assert Fraction(1, 2) * honest < cost.recompute < honest  # more than ``q`` of the run, less than all of it

    best = Optimize(compiled, Fraction(1, 100), PolicyGrid.uniform(2), max_bits=8)
    assert best is not None and best.policy == VerificationPolicy(1, 1) and best.bound.bits == 0.0
    assert best.evaluated == 9


def test_digest_names_the_constructor_and_its_parameters() -> None:
    constructor = ClusterG(SHAPE, pods=2, slots=2, steps=6)

    assert constructor.digest == ClusterG(SHAPE, 2, 2, 6).digest
    assert len(constructor.digest) == 64
    assert constructor.digest != ClusterG(SHAPE, 3, 2, 6).digest
    assert constructor.digest != ClusterG(DEEP, 2, 2, 6).digest
    assert constructor.manifest == {"pods": 2, "shape": SHAPE.manifest, "slots": 2, "steps": 6}
    with pytest.raises(ValueError, match="pods must be a positive integer"):
        ClusterG(SHAPE, 0, 2, 6)
    with pytest.raises(TypeError, match="LMShape"):
        ClusterG(object(), 1, 1, 1)  # type: ignore[arg-type]


def test_bad_requests_and_bad_advice_fail_to_trace() -> None:
    constructor = ClusterG(SHAPE, pods=2, slots=2, steps=6)
    schedule = schedule_fcfs(REQUESTS, 2, 2, 6)

    with pytest.raises(TracerError, match="malformed advice"):
        constructor(REQUESTS, b"not a schedule")
    with pytest.raises(TracerError, match="malformed advice"):
        constructor(REQUESTS, schedule.encode()[:-1])
    with pytest.raises(TracerError, match="advice must be bytes"):
        constructor(REQUESTS, "schedule")  # type: ignore[arg-type]
    with pytest.raises(TracerError, match=r"never scheduled: \[2, 3\]"):  # one pod, one step
        ClusterG(SHAPE, 1, 2, 1)(REQUESTS, schedule_fcfs(REQUESTS, 1, 2, 1).encode())
    with pytest.raises(TracerError, match="unknown request"):
        constructor(REQUESTS[:3], schedule.encode())
    with pytest.raises(TracerError, match="another cluster"):
        ClusterG(SHAPE, pods=3, slots=2, steps=6)(REQUESTS, schedule.encode())
    long = (Request((1, 2, 3), 4),)  # 3 + 4 positions in a context of 6
    with pytest.raises(TracerError, match="needs 7 positions; the context is 6"):
        ClusterG(SHAPE, 1, 1, 4)(long, schedule_fcfs(long, 1, 1, 4).encode())
    cut = Schedule(1, 1, 3, (Join(0, 0, 0, 0),))  # the run ends first: 3 + 3 fits
    assert ClusterG(SHAPE, 1, 1, 3)(long, cut.encode())[1] == (1, 2, 3)
    with pytest.raises(TracerError, match="outside the vocabulary"):
        ClusterG(SHAPE, 1, 1, 1)((Request((8,), 1),), schedule_fcfs((Request((8,), 1),), 1, 1, 1).encode())
    with pytest.raises(TracerError, match="nonempty tuple of Request"):
        constructor([REQUESTS[0]], schedule.encode())  # type: ignore[arg-type]
    with pytest.raises(TracerError, match="nonempty tuple of Request"):
        constructor((), schedule.encode())


def test_the_compiler_checks_the_input_count_against_the_prompts(fcfs) -> None:
    constructor, schedule, _ = fcfs
    description, inputs = constructor(REQUESTS, schedule.encode())

    with pytest.raises(CompileError, match="expects 9 inputs, got 8"):
        Compiler(GATES).compile(description, inputs[:-1])
