"""Section 6 of ``docs/honest-prover.md``: structural counts and the ``H4*`` rows.

The structural resolver of :mod:`veritor.simulation.systematic` is checked
against a flat walk of the toy circuit (exact agreement, weights and KV
boundary alike), run on GPT-2 Small where the flat walk cannot go, and then
the four systematic scenarios -- a corrupted weight cell, a stale weight
version and a wrong kernel path on one pod for an hour, a fleet-wide update
mid-round -- are priced on the simulated datacenter's shape under P1 and the
alternatives the plan names.
"""

from __future__ import annotations

import math
from collections import Counter
from fractions import Fraction

import pytest

from veritor.analysis.faults import unit_fault_bits
from veritor.compile import Compiler
from veritor.constructors import (
    GPT2G,
    ClusterG,
    GPT2Shape,
    Request,
    schedule_fcfs,
)
from veritor.core import Compiled
from veritor.core.description import REPLAY
from veritor.simulation.faults import (
    LLAMA3_GPUS,
    expected_faults,
    fault_budget,
    is_dot_unit,
)
from veritor.simulation.systematic import (
    Progression,
    SystematicPricing,
    brute_force_readers,
    histogram,
    kv_consumers,
    pod_scope,
    price_systematic,
    reader_count,
    readers,
    source_pardon_bits,
    weight_readers,
)
from veritor.stress.measure import POLICY, Measurement, compile_scenario, price
from veritor.stress.models import SHAPE, Model
from veritor.stress.rows import Recorder

REQUESTS = (
    Request((1, 2, 3), 3),
    Request((5,), 4),
    Request((7, 0), 2),
    Request((2, 2, 2, 2), 3),
    Request((4, 6), 4),
)
PODS, SLOTS, STEPS = 2, 2, 10
"""The simulated datacenter's shape (``small_config``), ten synchronous steps."""
STEP_SECONDS = 0.05
"""``WorkloadConfig.step_seconds``: 72,000 steps per pod-hour."""
HOURS = 1.0
POSITIONS = sum(len(r.prompt) + r.max_new - 1 for r in REQUESTS)  # 23
PREDICTIONS = sum(r.max_new for r in REQUESTS)  # 16


@pytest.fixture(scope="module")
def cluster(model: Model) -> Measurement:
    schedule = schedule_fcfs(REQUESTS, PODS, SLOTS, STEPS)
    constructor = ClusterG(SHAPE, PODS, SLOTS, STEPS)
    return compile_scenario(constructor, REQUESTS, schedule.encode(), model.gate_set)


def per_position(counts: dict[int, int], positions: int) -> dict[Fraction, int]:
    return {Fraction(readers, positions): cells for readers, cells in counts.items()}


def test_structural_readers_agree_with_the_flat_walk_on_the_toy(
    cluster: Measurement,
) -> None:
    """Every weight cell and every KV boundary value: the description-level count is the gate-level count."""

    compiled = cluster.compiled
    circuit = compiled.circuit
    structural = readers(compiled)
    sources, boundary = brute_force_readers(compiled)
    assert structural.orphans == 0  # every gate of the toy lies in a VU
    assert (
        structural.inputs == ()
    )  # the prompt tokens are ``in`` gates inside the steps
    for address in circuit.weights:
        assert reader_count(structural.weights, address) == sources.get(address, 0)
    assert weight_readers(compiled, circuit.weights[0]) == sources[circuit.weights[0]]
    for address in circuit.inputs:  # read by the embedding of the step holding it
        assert sources[address] >= 1
    consumed: Counter[int] = Counter()
    for p in structural.boundary:
        for address in p:
            consumed[address] += p.multiplicity
    assert dict(consumed) == boundary
    first = min(boundary)
    assert kv_consumers(compiled, first) == boundary[first]
    with pytest.raises(ValueError):
        weight_readers(compiled, circuit.inputs[0])


def test_toy_weight_cells_are_read_once_per_position(cluster: Measurement) -> None:
    """The layer's matrices and biases: one dot per position; the head: one per prediction."""

    compiled = cluster.compiled
    structural = readers(compiled)
    counts = histogram(
        structural.weights,
        range(compiled.circuit.weights[0], compiled.circuit.weights[-1] + 1),
    )
    assert sum(counts.values()) == compiled.index.weight_count == 201
    assert counts == {
        PREDICTIONS: 32,
        POSITIONS: 160,
        POSITIONS + PREDICTIONS: 8,
        2 * POSITIONS: 1,
    }
    # KV boundary values: read by one attention VU per later step of the request
    kv = histogram(structural.boundary)
    assert kv == {1: 59, 2: 72, 3: 24}
    assert max(kv) <= STEPS - 1


def test_histogram_paths_agree() -> None:
    """The NumPy sweep, the end-point sweep and the enumeration count the same cells."""

    pieces = (
        Progression(10, 5, 1, 2),
        Progression(12, 1, 0, 3),
        Progression(20, 3, 1, 1),
    )
    by_range = histogram(pieces, range(30))
    by_sweep = histogram(pieces)
    by_cells = histogram(pieces, list(range(10, 23)))
    assert by_sweep == {1: 3, 2: 4, 5: 1}
    assert by_range == {0: 22, 1: 3, 2: 4, 5: 1}
    assert by_cells == {0: 5, 1: 3, 2: 4, 5: 1}  # cells 15..19 lie in no progression
    strided = (Progression(0, 4, 3, 1), Progression(0, 4, 3, 2))  # 0, 3, 6, 9 twice
    assert histogram(strided) == {3: 4}
    assert histogram(strided, range(12)) == {0: 8, 3: 4}
    assert reader_count(strided, 6) == 3 and reader_count(strided, 7) == 0


def test_gpt2_small_weight_cells_are_read_once_per_position() -> None:
    """GPT-2 Small, 1.9 G gates, resolved through the description in about a second.

    Every layer parameter is read by exactly one VU per position processed;
    ``wte`` by one per position (the one-hot embedding) plus one per prediction
    (the LM head); a ``wpe`` row by one VU per request at that position.
    """

    requests = tuple(
        Request(tuple((7 * i + 3 * r) % 50257 for i in range(32)), 32) for r in range(3)
    )
    positions = sum(len(r.prompt) + r.max_new - 1 for r in requests)  # 189
    predictions = sum(r.max_new for r in requests)  # 96
    shape = GPT2Shape.small()
    constructor = GPT2G(shape)
    description, inputs = constructor(requests, b"")
    compiled: Compiled = Compiler(constructor.gate_set).compile(description, inputs)
    assert compiled.index.n == 1_924_349_881
    structural = readers(compiled)
    assert structural.orphans == 0 and structural.boundary == ()
    assert len(structural.weights) < 300  # progressions, not cells
    weights = compiled.circuit.weights
    counts = histogram(structural.weights, range(weights[0], weights[-1] + 1))
    d, layers = shape.d_model, shape.layers
    layer_parameters = layers * (
        12 * d * d + 13 * d
    )  # q k v o fc proj and their biases, two LayerNorms
    unread_wpe = (shape.context - positions // len(requests)) * d
    # ``zero`` pads every embedding dot to the padded vocabulary and every mix dot
    # to a multiple of 16 keys, so a position whose key count is a multiple of 16
    # (3 per request) has no mix pads.
    unpadded = 3 * len(requests)
    zero_readers = positions * d + (positions - unpadded) * layers * d
    assert counts == {
        0: unread_wpe,
        len(requests): (positions // len(requests)) * d,
        predictions: 2 * d + 2,
        positions: layer_parameters,
        positions + predictions: shape.vocab * d + shape.vocab,
        zero_readers: 1,
    }
    assert 12 * d * d == 7_077_888 and layer_parameters == 85_054_464


def test_h4_systematic_faults_on_one_pod_for_an_hour(
    cluster: Measurement, honest: Recorder
) -> None:
    """Rows H4a-H4d: what each systematic fault costs under P1 and under the priced alternatives."""

    compiled = cluster.compiled
    index = compiled.index
    table = compiled.kind_table()
    priced = price(compiled)
    q = POLICY.q
    u1 = unit_fault_bits(compiled)
    assert u1 == pytest.approx(64 + math.log2(index.verification_unit_count), abs=0.01)
    structural = readers(compiled)
    weight_counts = histogram(
        structural.weights,
        range(compiled.circuit.weights[0], compiled.circuit.weights[-1] + 1),
    )
    readers_per_position = per_position(weight_counts, POSITIONS)
    modal = max(readers_per_position, key=readers_per_position.__getitem__)
    assert modal == 1  # a layer matrix or bias cell: one dot per position
    worst = max(readers_per_position)
    assert worst == 2  # the attention shift, read by both heads

    units_per_position = index.verification_unit_count / POSITIONS
    dots = sum(
        is_dot_unit(compiled, index.verification_unit(unit))
        for unit in range(index.verification_unit_count)
    )
    dot_fraction = dots / index.verification_unit_count
    scope = pod_scope(
        slots=SLOTS,
        step_seconds=STEP_SECONDS,
        hours=HOURS,
        units_per_position=units_per_position,
        positions_per_request=POSITIONS / len(REQUESTS),
    )
    assert scope.steps == 72_000 and scope.positions == 144_000
    f_max = fault_budget(expected_faults(PODS * HOURS))
    assert f_max == 1
    headline_f_max = fault_budget(expected_faults(LLAMA3_GPUS * HOURS))
    steps = [row for row in table.rows if row.role == REPLAY and row.out_count > 0]
    out_bits = round(
        sum(row.out_bits * row.copies for row in steps)
        / sum(row.copies for row in steps)
    )
    pardonable = index.weight_count + index.input_count
    width = 16

    def record(
        identifier: str,
        what: str,
        mechanism: str,
        pricing: SystematicPricing,
        charge: float,
        recompute: float,
        verdict: str,
        notes: str,
    ) -> None:
        honest.record(
            id=identifier,
            what=what,
            mechanism=mechanism,
            advice_bits=cluster.advice_bits,
            capacity_bits=priced.capacity_bits + math.ceil(charge),
            overhead=priced.overhead,
            description_bytes=cluster.description_bytes,
            verdict=verdict,
            notes=notes,
            declarations=pricing.p1_declarations,
            charge_bits=math.ceil(charge),
            recompute=recompute,
        )

    # (a) one corrupted weight cell on one pod for an hour
    a = price_systematic(
        faulty_units=scope.positions * int(modal),
        q=q,
        f_max=f_max,
        unit_bits=u1,
        affected_replay_units=scope.replay_units,
        replay_units=PODS * scope.replay_units,
        out_bits=out_bits,
        reserve_fraction=1 / PODS,
        source_pardons=1,
        source_content_bits=source_pardon_bits(width, pardonable, PODS),
    )
    assert not a.p1_admissible and a.p1_declarations == 72_000
    assert a.source_bits is not None and a.source_bits < 60
    record(
        "H4a",
        "corrupted weight cell on one pod, one hour",
        "source-position pardon",
        a,
        a.source_bits,
        float(q) / PODS,
        f"source-position pardon: 1 declaration, {math.ceil(a.source_bits)} bits; "
        f"P1 needs {a.p1_declarations:,} declarations against f_max = {f_max}",
        f"the cell is read by {modal} VU per position ({worst} for the attention shift): "
        f"{a.faulty_units:,} faulty VUs in the pod-hour, {a.p1_declarations:,} in opened RUs at q = {q}; "
        f"headline f_max for {LLAMA3_GPUS:,} devices is {headline_f_max}; the pardon needs the "
        f"prover to attribute the disagreements of the opened RUs to one cell (recompute q / pods); "
        f"RU-scope pardons would cost {a.ru_scope_bits:,.0f} bits, re-serving the pod-hour "
        f"{a.reserve_fraction:.0%} of the fleet-hour",
    )

    # (b) a stale weight version on one pod: every cell differs
    b = price_systematic(
        faulty_units=scope.verification_units,
        q=q,
        f_max=f_max,
        unit_bits=u1,
        affected_replay_units=scope.replay_units,
        replay_units=PODS * scope.replay_units,
        out_bits=out_bits,
        reserve_fraction=1 / PODS,
        source_pardons=index.weight_count,
        source_content_bits=source_pardon_bits(width, pardonable, PODS),
        configurable=True,
    )
    assert (
        not b.p1_admissible
        and b.source_bits is not None
        and b.configuration_bits == 0.0
    )
    record(
        "H4b",
        "stale weight version on one pod, one hour",
        "per-pod public weight root (M2/M8)",
        b,
        0.0,
        0.0,
        "per-pod kappa_W in the constructor: 0 bits; else re-serve the pod-hour; "
        f"source-position pardons would cost {math.ceil(b.source_bits):,} bits for {index.weight_count} cells",
        f"every VU on the pod is faulty ({b.faulty_units:,}); {b.p1_declarations:,} P1 declarations; "
        f"one source-position pardon per changed cell scales with |W| (GPT-2 Small: 124,490,068 cells); "
        f"the version is knowable from the rollout log, so the constructor can carry it (M2/M8) and "
        f"the verifier checks the pod's RUs against the stale root",
    )

    # (c) a wrong kernel path on one pod: every dot's relation changes
    c = price_systematic(
        faulty_units=math.ceil(scope.verification_units * dot_fraction),
        q=q,
        f_max=f_max,
        unit_bits=u1,
        affected_replay_units=scope.replay_units,
        replay_units=PODS * scope.replay_units,
        out_bits=out_bits,
        reserve_fraction=1 / PODS,
        lowering_variants=2,
        configurable=True,
    )
    assert c.lowering_bits == scope.replay_units  # log2(2) per RU
    record(
        "H4c",
        "wrong kernel path on one pod, one hour",
        "per-pod architecture (M8)",
        c,
        0.0,
        0.0,
        "per-pod gate set in the constructor (ClusterG arches): 0 bits; a late lowering would cost "
        f"{c.lowering_bits:,.0f} bits (1 per RU) and only under the plan's conditions",
        f"{dot_fraction:.0%} of the toy's VUs are dots ({c.faulty_units:,} faulty in the pod-hour); "
        f"{c.p1_declarations:,} P1 declarations; a source-position pardon cannot express a relation change; "
        f"diagnosis without a kernel-path log: replay one opened RU under each candidate lowering",
    )

    # (d) a fleet-wide update mid-round: every pod's RUs after the update
    d = price_systematic(
        faulty_units=PODS * scope.verification_units // 2,
        q=q,
        f_max=f_max,
        unit_bits=u1,
        affected_replay_units=PODS * scope.replay_units // 2,
        replay_units=PODS * scope.replay_units,
        out_bits=out_bits,
        reserve_fraction=0.5,
        source_pardons=PODS * index.weight_count,
        source_content_bits=source_pardon_bits(width, pardonable, PODS),
        configurable=True,
    )
    record(
        "H4d",
        "fleet-wide weight update mid-round",
        "round close at the update (M2)",
        d,
        0.0,
        0.0,
        "close the round and start a run under the new kappa_W: 0 bits; else re-serve the half-round",
        f"{d.faulty_units:,} faulty VUs across {PODS} pods, {d.p1_declarations:,} P1 declarations; "
        f"the header binds one kappa_W per run, so the update is a run boundary, not a pardon; "
        f"RU-scope pardons for the half-round would cost {d.ru_scope_bits:,.0f} bits",
    )
