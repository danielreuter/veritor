"""Section 6 of ``docs/honest-prover.md``: structural counts, measured declarations, rows ``H4a``-``H4d``.

The structural resolver of :mod:`veritor.simulation.systematic` is checked
against a flat walk of the toy circuit (exact agreement, weights and KV
boundary alike) and run on GPT-2 Small where the flat walk cannot go.  The
measurement helpers -- the production run of a datacenter with one faulty
pod, pinned replay over what the server recorded -- are checked against the
honest-prover model's numbers for a weight read fault (36 gate readers, 18
VUs; 7 declarations under ``BOUNDARY`` recording, 18 under ``VU_OUTPUTS``;
``veritor.simulation.honest`` on the honest-sim branch).  Then the four
systematic scenarios -- a corrupted weight cell, a stale weight version and a
wrong kernel path on one pod, a fleet-wide update mid-round -- are measured on
the simulated datacenter under both recording policies, scaled to the
pod-hour, and priced under P1 and the alternatives the plan names.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fractions import Fraction

import pytest

from veritor.analysis.faults import unit_fault_bits
from veritor.compile import Compiler
from veritor.constructors import (
    GPT2G,
    ClusterG,
    GPT2Shape,
    Request,
    RequestsG,
    schedule_fcfs,
)
from veritor.constructors.lm import LMShape, random_parameters
from veritor.core import Compiled
from veritor.core.description import REPLAY
from veritor.simulation.faults import (
    LLAMA3_GPUS,
    FaultInjector,
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
    perturbed_run,
    pinned_units,
    pod_scope,
    post_j_unit_bits,
    price_systematic,
    reader_count,
    readers,
    recorded_positions,
    ru_scoped_source_bits,
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
WIDTH = SHAPE.width
LAYER = ("w_q", "w_k", "w_v", "w_o", "w_1", "w_2")
"""The layer's matrices: one dot reads each cell per position."""
KV_PROJECTIONS = frozenset({"w_k", "w_v"})
"""Their readers' outputs are the KV entries a step RU commits: recorded under both policies."""


def layout(shape: LMShape) -> dict[str, int]:
    """Weight rank of the ``[0][0]`` cell of every matrix, in ``Parameters.flatten`` order."""

    d, vocab, hidden = shape.d_model, shape.vocab, shape.hidden
    assert shape.layers == 1
    sizes = (
        ("embedding", vocab * d),
        ("w_q", d * d),
        ("w_k", d * d),
        ("w_v", d * d),
        ("w_o", d * d),
        ("w_1", d * hidden),
        ("w_2", hidden * d),
        ("unembedding", d * vocab),
    )
    ranks: dict[str, int] = {}
    at = 0
    for name, size in sizes:
        ranks[name] = at
        at += size
    ranks["shift"] = at + vocab  # after the constant table
    return ranks


@dataclass(frozen=True)
class Datacenter:
    """The simulated datacenter: its compiled run, its honest values, where each step RU ran."""

    measurement: Measurement
    weights: tuple[int, ...]
    placement: dict[int, tuple[int, int]]
    """Step RU -> ``(pod, step)``: ``ClusterG`` traces the occupied pairs in ``(step, pod)`` order after the weights RU."""
    honest: dict[int, int]
    recorded: dict[str, frozenset[int]]
    """The addresses the server keeps under ``BOUNDARY`` and under ``VU_OUTPUTS``."""

    @property
    def compiled(self) -> Compiled:
        return self.measurement.compiled

    @property
    def inputs(self) -> tuple[int, ...]:
        return tuple(self.measurement.compilation.inputs)

    def units(self, pod: int | None = None, since: int = 0) -> tuple[int, ...]:
        """The step RUs of ``pod`` (every pod when ``None``) at steps ``>= since``."""

        return tuple(
            unit
            for unit, (where, step) in sorted(self.placement.items())
            if (pod is None or where == pod) and step >= since
        )


@dataclass(frozen=True, slots=True)
class Work:
    """What some RUs computed and what the server recorded of them."""

    units: tuple[int, ...]
    positions: int
    """Positions processed: the readers of a layer cell among the RUs."""
    verification_units: int
    dots: int
    words: tuple[int, ...]
    """Recorded computed words of the RUs: KV entries and tokens (the ``BOUNDARY`` policy)."""
    tokens: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Pins:
    """What pinned replay declares for one fault, under each recording policy."""

    boundary: int
    vu_outputs: int
    changed_tokens: int
    """Streamed tokens of the faulty RUs that differ from the honest run (chained through the feedback)."""


@pytest.fixture(scope="module")
def datacenter(model: Model) -> Datacenter:
    schedule = schedule_fcfs(REQUESTS, PODS, SLOTS, STEPS)
    constructor = ClusterG(SHAPE, PODS, SLOTS, STEPS)
    measurement = compile_scenario(
        constructor, REQUESTS, schedule.encode(), model.gate_set
    )
    compiled = measurement.compiled
    order = sorted(schedule.occupancy(REQUESTS), key=lambda key: (key[1], key[0]))
    assert len(order) + 1 == compiled.index.replay_units.count
    weights = tuple(model.weights)
    honest = perturbed_run(compiled, measurement.compilation.inputs, weights, units=())
    assert honest == dict(
        enumerate(compiled.circuit.evaluate(measurement.compilation.inputs, weights))
    )
    return Datacenter(
        measurement,
        weights,
        {k + 1: pair for k, pair in enumerate(order)},
        honest,
        {
            "BOUNDARY": recorded_positions(compiled, interiors=False),
            "VU_OUTPUTS": recorded_positions(compiled, interiors=True),
        },
    )


@pytest.fixture(scope="module")
def cluster(datacenter: Datacenter) -> Measurement:
    return datacenter.measurement


def readers_in(compiled: Compiled, address: int, units: tuple[int, ...]) -> int:
    """VUs of the RUs ``units`` with a gate reading ``address``."""

    circuit, index = compiled.circuit, compiled.index
    count = 0
    for unit in units:
        nodes = index.verification_units(unit)
        for offset in range(nodes.count):
            node = nodes.unit(offset)
            if any(address in circuit[a].args for a in node.interval):
                count += 1
    return count


def work(dc: Datacenter, units: tuple[int, ...], cell: int) -> Work:
    compiled = dc.compiled
    circuit, index = compiled.circuit, compiled.index
    owner = index.replay_units.owner
    chosen = set(units)
    nodes = [
        index.verification_units(unit).unit(offset)
        for unit in units
        for offset in range(index.verification_units(unit).count)
    ]
    return Work(
        units=units,
        positions=readers_in(compiled, cell, units),
        verification_units=len(nodes),
        dots=sum(is_dot_unit(compiled, node) for node in nodes),
        words=tuple(
            a
            for a in sorted(dc.recorded["BOUNDARY"])
            if not circuit[a].is_source and owner(a) in chosen
        ),
        tokens=tuple(a for a in circuit.outputs if owner(a) in chosen),
    )


def pins(
    dc: Datacenter,
    faulty: Work,
    *,
    misread: Mapping[int, int] | None = None,
    corrupt: Callable[[int, int], int] | None = None,
) -> Pins:
    """Run the datacenter with ``faulty.units`` on a faulty pod, replay every RU pinned under each policy."""

    compiled = dc.compiled
    values = perturbed_run(
        compiled,
        dc.inputs,
        dc.weights,
        units=faulty.units,
        misread=misread,
        corrupt=corrupt,
    )
    counts: dict[str, int] = {}
    inside = set(faulty.units)
    for policy, recorded in dc.recorded.items():
        pinned = pinned_units(compiled, values, recorded)
        # A reader of a pinned value is recomputed from it: a fault confined to
        # these RUs pins nothing in any other RU, on the same pod or off it.
        assert not any(vus for unit, vus in pinned.items() if unit not in inside), (
            policy
        )
        counts[policy] = sum(len(vus) for vus in pinned.values())
    return Pins(
        counts["BOUNDARY"],
        counts["VU_OUTPUTS"],
        sum(values[a] != dc.honest[a] for a in faulty.tokens),
    )


def per_position(counts: dict[int, int], positions: int) -> dict[Fraction, int]:
    return {Fraction(readers, positions): cells for readers, cells in counts.items()}


# -- the structural resolver ------------------------------------------------------------


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


# -- the measurement helpers against the honest-prover model -----------------------------


def test_pinned_replay_reproduces_the_honest_model_counts(model: Model) -> None:
    """The honest model's weight read fault (``RequestsG``, two requests): 36 gate readers,
    18 VUs; every one of the 7 streamed tokens flips; 7 declarations under ``BOUNDARY``,
    18 under ``VU_OUTPUTS``."""

    requests = (Request((1, 2, 3), 3), Request((5,), 4))
    measurement = compile_scenario(RequestsG(SHAPE), requests, b"", model.gate_set)
    compiled = measurement.compiled
    circuit, index = compiled.circuit, compiled.index
    inputs, weights = measurement.compilation.inputs, tuple(model.weights)
    injector = FaultInjector(compiled, inputs, weights)
    cell = max(circuit.weights, key=lambda a: (len(injector.readers[a]), -a))
    assert len(injector.readers[cell]) == 36
    assert (
        weight_readers(compiled, cell) == 18
    )  # the attention shift: both heads, every position
    honest = perturbed_run(compiled, inputs, weights, units=())
    assert honest == injector.honest
    boundary = recorded_positions(compiled, interiors=False)
    everything = recorded_positions(compiled, interiors=True)
    assert set(circuit.outputs) <= boundary < everything
    assert set(circuit.weights) <= boundary
    for recorded in (boundary, everything):
        assert not any(pinned_units(compiled, honest, recorded).values())
    top = 1 << (circuit[cell].width - 1)
    every = tuple(range(index.replay_units.count))
    values = perturbed_run(
        compiled, inputs, weights, units=every, misread={cell: honest[cell] ^ top}
    )
    changed = sum(values[a] != honest[a] for a in circuit.outputs)
    assert 0 < changed <= len(circuit.outputs) == 7
    under_boundary = pinned_units(compiled, values, boundary)
    under_outputs = pinned_units(compiled, values, everything)
    # Every token disagrees with its pinned recompute, though a token computed from
    # a wrong earlier token can happen to agree with the honest run.
    assert sum(map(len, under_boundary.values())) == 7 >= changed
    assert sum(map(len, under_outputs.values())) == 18
    # Under tokens-only recording the declarations name the tokens that came out wrong.
    for unit, vus in under_boundary.items():
        for vu in vus:
            assert index.verification_unit(vu).interval[-1] in set(circuit.outputs), (
                unit
            )
    # ``units`` restricts the replay.
    assert pinned_units(compiled, values, boundary, units=(1,)) == {
        1: under_boundary[1]
    }
    with pytest.raises(KeyError):
        pinned_units(compiled, values, frozenset(circuit.weights))


# -- H4: systematic faults on one pod for an hour ----------------------------------------


def test_h4_systematic_faults_on_one_pod_for_an_hour(
    datacenter: Datacenter, honest: Recorder
) -> None:
    """Rows H4a-H4d: declarations under both recording policies, measured on the faulty pod
    and scaled to the pod-hour; the charge under P1 and under the priced alternatives."""

    dc = datacenter
    compiled = dc.compiled
    index, circuit = compiled.index, compiled.circuit
    table = compiled.kind_table()
    priced = price(compiled)
    q, s = POLICY.q, POLICY.s
    rho = priced.bound.rho
    u_post = post_j_unit_bits(rho, s)
    assert 140 < u_post < 150  # the note's 145.6 at its fixture
    u1 = unit_fault_bits(compiled)
    assert u1 == pytest.approx(64 + math.log2(index.verification_unit_count), abs=0.01)
    ranks = layout(SHAPE)
    assert ranks["shift"] + 1 == index.weight_count == 201

    def cell(name: str) -> int:
        return circuit.weights[ranks[name]]

    structural = readers(compiled)
    weight_counts = histogram(
        structural.weights,
        range(circuit.weights[0], circuit.weights[-1] + 1),
    )
    readers_per_position = per_position(weight_counts, POSITIONS)
    modal = max(readers_per_position, key=readers_per_position.__getitem__)
    assert modal == 1  # a layer matrix or bias cell: one dot per position
    assert max(readers_per_position) == 2  # the attention shift, read by both heads

    # The faulty pod: pod 0, four steps of the run.
    pod = work(dc, dc.units(pod=0), cell("w_q"))
    assert pod.units == (1, 3, 5, 7)
    assert pod.positions == 9 and pod.verification_units == 526 and pod.dots == 344
    assert len(pod.words) == 79 and len(pod.tokens) == 7
    for name in LAYER:
        assert weight_readers(compiled, cell(name)) == POSITIONS
        assert readers_in(compiled, cell(name), dc.units()) == POSITIONS
        assert readers_in(compiled, cell(name), pod.units) == pod.positions
    for recorded in dc.recorded.values():
        assert not any(pinned_units(compiled, dc.honest, recorded).values())

    # (a) one weight cell read wrong by every reader on the pod: the layer's
    # matrices, low bit and top bit.
    grid = {
        (name, bit): pins(
            dc, pod, misread={cell(name): dc.honest[cell(name)] ^ (1 << bit)}
        )
        for name in LAYER
        for bit in (0, WIDTH - 1)
    }
    for (name, bit), found in grid.items():
        if bit == 0:
            assert found.vu_outputs == pod.positions  # every reader's output moved
        else:
            # The toy's word is modular: bit 15 of a weight is silent for an even activation.
            assert 0 < found.vu_outputs < pod.positions
        if name in KV_PROJECTIONS:
            assert (
                found.boundary == found.vu_outputs
            )  # its outputs are recorded KV words
        else:
            assert found.boundary <= found.changed_tokens <= len(pod.tokens)
    worst = grid[("w_q", 0)]
    assert worst.vu_outputs == 9 and worst.boundary == worst.changed_tokens == 7
    flips = {
        (name, bit): Fraction(found.boundary, len(pod.tokens))
        for (name, bit), found in grid.items()
        if name not in KV_PROJECTIONS
    }
    assert max(flips.values()) == 1 and min(flips.values()) == 0

    # (b) a stale weight version on the pod: every matrix cell differs.
    stale = tuple(random_parameters(SHAPE, 8).flatten())
    changed = {
        circuit.weights[k]: stale[k]
        for k in range(len(stale))
        if stale[k] != dc.weights[k]
    }
    assert 190 <= len(changed) <= 192  # the constant table is fixed by the shape
    version = pins(dc, pod, misread=changed)
    assert version.boundary == len(pod.words)  # every KV entry and token of the pod
    assert version.vu_outputs == pod.dots  # every dot reads a changed cell

    # (c) a wrong kernel path on the pod: every dot's output word perturbed.
    dot_outputs = {
        index.verification_units(unit).unit(offset).interval[-1]
        for unit in pod.units
        for offset in range(index.verification_units(unit).count)
        if is_dot_unit(compiled, index.verification_units(unit).unit(offset))
    }
    assert len(dot_outputs) == pod.dots
    kernel = pins(dc, pod, corrupt=lambda a, v: v ^ 1 if a in dot_outputs else v)
    assert kernel.boundary == len(pod.words) and kernel.vu_outputs == pod.dots

    # (d) a fleet-wide weight update midway through the run: every step from
    # the update on, on every pod, reads the new version.
    used = 1 + max(step for _pod, step in dc.placement.values())
    update = math.ceil(used / 2)
    late = work(dc, dc.units(since=update), cell("w_q"))
    assert set(late.units) & set(pod.units) and set(late.units) - set(pod.units)
    rollout = pins(dc, late, misread=changed)
    # Every KV word is pinned; a token may agree by chance on an 8-token vocabulary.
    assert len(late.words) - len(late.tokens) <= rollout.boundary <= len(late.words)
    assert rollout.vu_outputs == late.dots

    # The pod-hour: 72,000 steps of two slots, scaled from the measured pod by positions.
    scope = pod_scope(
        slots=SLOTS,
        step_seconds=STEP_SECONDS,
        hours=HOURS,
        units_per_position=pod.verification_units / pod.positions,
        positions_per_request=POSITIONS / len(REQUESTS),
    )
    assert scope.steps == 72_000 and scope.positions == 144_000
    scale = scope.positions / pod.positions

    def hour(count: int) -> int:
        return round(count * scale)

    assert hour(pod.verification_units) == scope.verification_units
    f_max = fault_budget(expected_faults(PODS * HOURS))
    assert f_max == 1
    headline_f_max = fault_budget(expected_faults(LLAMA3_GPUS * HOURS))
    steps = [row for row in table.rows if row.role == REPLAY and row.out_count > 0]
    out_bits = round(
        sum(row.out_bits * row.copies for row in steps)
        / sum(row.copies for row in steps)
    )
    run_units = PODS * scope.replay_units
    pardonable = len(
        dc.recorded["BOUNDARY"]
    )  # every committed position: weights, inputs, KV, tokens
    source_bits = source_pardon_bits(WIDTH, pardonable, PODS)  # the pod, the cell, v'
    ru_message = source_bits + math.log2(
        run_units
    )  # ... and the RU, for an RU-scoped one

    def record(
        identifier: str,
        what: str,
        mechanism: str,
        pricing: SystematicPricing,
        charge: float,
        recompute: float,
        verdict: str,
        notes: str,
        measured: Pins,
    ) -> None:
        honest.record(
            id=identifier,
            what=what,
            mechanism=mechanism,
            advice_bits=dc.measurement.advice_bits,
            capacity_bits=priced.capacity_bits + math.ceil(charge),
            overhead=priced.overhead,
            description_bytes=dc.measurement.description_bytes,
            verdict=verdict,
            notes=notes,
            declarations=pricing.flipped.opened,
            charge_bits=math.ceil(charge),
            recompute=recompute,
            declarations_boundary=pricing.flipped.opened,
            declarations_vu_outputs=pricing.readers.opened,
            faulty_boundary=pricing.flipped.faulty,
            faulty_vu_outputs=pricing.readers.faulty,
            toy_boundary=measured.boundary,
            toy_vu_outputs=measured.vu_outputs,
            m6_bits_boundary=round(pricing.flipped.bits),
            m6_bits_vu_outputs=round(pricing.readers.bits),
        )

    def rows(found: dict[tuple[str, int], Pins], bit: int) -> str:
        return ", ".join(
            f"{name} {found[(name, bit)].boundary}/{found[(name, bit)].vu_outputs}"
            for name in LAYER
        )

    a = price_systematic(
        readers=hour(worst.vu_outputs),
        flipped=hour(worst.boundary),
        q=q,
        f_max=f_max,
        unit_post_bits=u_post,
        affected_replay_units=scope.replay_units,
        replay_units=run_units,
        out_bits=out_bits,
        reserve_fraction=1 / PODS,
        source_pardons=1,
        source_content_bits=source_bits,
        ru_scoped_bits_each=ru_scoped_source_bits(
            rho,
            s,
            math.ceil(hour(worst.vu_outputs) / scope.replay_units),
            ru_message,
            ru_message,
        ),
    )
    assert a.readers.faulty == scope.positions == 144_000
    assert a.readers.opened == 72_000 and a.flipped.opened == 56_000
    assert not a.readers.admissible and not a.flipped.admissible
    assert a.source_bits is not None and a.source_bits < 30
    assert a.ru_scoped_source_bits is not None
    assert a.ru_scoped_source_bits == pytest.approx(a.readers.bits, rel=0.01)
    record(
        "H4a",
        "corrupted weight cell on one pod, one hour",
        "source-position pardon (pod scope)",
        a,
        a.source_bits,
        float(q) / PODS,
        f"one pardon, {math.ceil(a.source_bits)} bits, pre-J from the ECC log (post-J at the same "
        f"price only under the forced-consistency argument, open); P1 needs {a.flipped.opened:,} "
        f"declarations under BOUNDARY recording or {a.readers.opened:,} under VU_OUTPUTS against "
        f"f_max = {f_max}: rejected",
        f"a layer matrix cell is read by 1 VU per position; measured on pod 0 of the fixture (4 steps, "
        f"{pod.positions} positions, {pod.verification_units} VUs, {len(pod.words)} recorded words, "
        f"{len(pod.tokens)} tokens), pinned VUs BOUNDARY/VU_OUTPUTS per cell: low bit {rows(grid, 0)}; "
        f"top bit {rows(grid, WIDTH - 1)} (the toy's word is modular, so the top bit of a weight is silent "
        f"for an even activation; W_k and W_v pin their readers under both policies because a step RU commits "
        f"its KV entries; the others pin the tokens that flipped, between {min(flips.values())} and "
        f"{max(flips.values())} of the pod's tokens on an 8-token vocabulary); the row "
        f"scales W_q low bit: {a.readers.faulty:,} readers and {a.flipped.faulty:,} flipped tokens in the pod-hour, "
        f"{a.readers.opened:,} / {a.flipped.opened:,} in opened RUs at q = {q}; M6 would charge "
        f"{a.readers.bits:,.0f} / {a.flipped.bits:,.0f} bits at u_post(1) = {u_post:.1f}; an RU-scoped source "
        f"pardon per opened RU costs the same as declaring the RU's readers ({a.ru_scoped_source_bits:,.0f} bits); "
        f"RU-scope pardons {a.ru_scope_bits:,.0f} bits pre-J and prohibited post-J; headline f_max for "
        f"{LLAMA3_GPUS:,} devices is {headline_f_max}; diagnosis: the ECC scrub log names the cell, else replay "
        f"the pod's opened RUs (q / pods of the run) and solve one opened VU for the cell",
        worst,
    )

    b = price_systematic(
        readers=hour(version.vu_outputs),
        flipped=hour(version.boundary),
        q=q,
        f_max=f_max,
        unit_post_bits=u_post,
        affected_replay_units=scope.replay_units,
        replay_units=run_units,
        out_bits=out_bits,
        reserve_fraction=1 / PODS,
        source_pardons=len(changed),
        source_content_bits=source_bits,
        configurable=True,
    )
    assert not b.flipped.admissible and b.configuration_bits == 0.0
    assert b.source_bits is not None
    record(
        "H4b",
        "stale weight version on one pod, one hour",
        "per-pod public weight root (M2/M8)",
        b,
        0.0,
        0.0,
        "per-pod kappa_W in the constructor: 0 bits; else re-serve the pod-hour; P1 needs "
        f"{b.flipped.opened:,} (BOUNDARY) or {b.readers.opened:,} (VU_OUTPUTS) declarations: rejected",
        f"{len(changed)} of {index.weight_count} cells differ between the versions; measured on pod 0: every "
        f"recorded word is pinned under BOUNDARY ({version.boundary} of {len(pod.words)}), every dot under "
        f"VU_OUTPUTS ({version.vu_outputs} of {pod.verification_units} VUs); pod-hour {b.flipped.faulty:,} words "
        f"and {b.readers.faulty:,} dots; one run-wide source-position pardon per changed cell would cost "
        f"{math.ceil(b.source_bits):,} bits here and scales with |W| (GPT-2 Small 124,490,068 cells); the version "
        f"is a public object known from the rollout log, so the constructor can carry a per-pod root and the "
        f"verifier checks the pod's RUs against it",
        version,
    )

    c = price_systematic(
        readers=hour(kernel.vu_outputs),
        flipped=hour(kernel.boundary),
        q=q,
        f_max=f_max,
        unit_post_bits=u_post,
        affected_replay_units=scope.replay_units,
        replay_units=run_units,
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
        f"{c.lowering_bits:,.0f} bits (1 per RU) and only under the note's conditions; P1 needs "
        f"{c.flipped.opened:,} (BOUNDARY) or {c.readers.opened:,} (VU_OUTPUTS) declarations: rejected",
        f"every dot of the pod ({pod.dots} of {pod.verification_units} VUs) stores a perturbed word; measured "
        f"on pod 0: every recorded word pinned under BOUNDARY ({kernel.boundary}), every dot under VU_OUTPUTS "
        f"({kernel.vu_outputs}); pod-hour {c.flipped.faulty:,} words and {c.readers.faulty:,} dots; a "
        f"source-position pardon cannot express a relation change; diagnosis without a kernel-path log: replay "
        f"one opened RU under each candidate lowering",
        kernel,
    )

    half = PODS * scope.replay_units // 2
    d = price_systematic(
        readers=round(PODS * hour(pod.dots) / 2 * rollout.vu_outputs / late.dots),
        flipped=round(
            PODS * hour(len(pod.words)) / 2 * rollout.boundary / len(late.words)
        ),
        q=q,
        f_max=f_max,
        unit_post_bits=u_post,
        affected_replay_units=half,
        replay_units=run_units,
        out_bits=out_bits,
        reserve_fraction=0.5,
        source_pardons=PODS * len(changed),
        source_content_bits=source_bits,
        configurable=True,
    )
    record(
        "H4d",
        "fleet-wide weight update mid-round",
        "round close at the update (M2)",
        d,
        0.0,
        0.0,
        "close the round and start a run under the new kappa_W: 0 bits; else re-serve the half-round; P1 "
        f"needs {d.flipped.opened:,} (BOUNDARY) or {d.readers.opened:,} (VU_OUTPUTS) declarations: rejected",
        f"measured with the update at step {update} of {used}: RUs {late.units} read the new version, "
        f"{rollout.boundary} of their {len(late.words)} recorded words are pinned under BOUNDARY (a token can "
        f"agree with the honest run by chance on an 8-token vocabulary), every dot ({rollout.vu_outputs} of "
        f"{late.verification_units} VUs) under VU_OUTPUTS, and nothing before the update; the half-round after "
        f"the update on {PODS} pods is {d.flipped.faulty:,} words and {d.readers.faulty:,} dots; the header binds "
        f"one kappa_W per run, so the update is a run boundary, not a pardon; RU-scope pardons for the "
        f"half-round would cost {d.ru_scope_bits:,.0f} bits pre-J",
        rollout,
    )
