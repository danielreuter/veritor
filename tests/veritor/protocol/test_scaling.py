"""The verifier's work is sampled work: flat in the number of units and gates.

The circuit is ``units`` identical replay tiles reading the one input, each
two verification cells deep, so the description is ``O(1)`` while ``|∂| = units
+ 1`` and ``N = 6 units + 1`` grow.  With ``q = 16 / units`` the verifier
expects sixteen replay units in ``J`` whatever ``units`` is, so its work per
phase must not move when ``units`` grows 32x.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Mapping
from fractions import Fraction

import pytest

from veritor.compile import Compiler, Tracer
from veritor.core import (
    Compiled,
    DescriptionCircuit,
    VerificationPolicy,
    make_word_gate_set,
)
from veritor.core import index as core_index
from veritor.protocol import (
    Expectation,
    ProverSession,
    VerifierSession,
    make_expectation,
    run_protocol,
)

GATE_SET = make_word_gate_set(8)
SEEDS = {"q_seed": b"q" * 32, "s_seed": b"s" * 32, "session_id": b"scaling"}
INPUT = (3,)
EXPECTED_SELECTED = 16


def tiled_description(units: int, *, tile_role: str, cell_role: str | None, root_role: str | None):
    """``units`` tiles of two cells of three ``add`` gates, with the given marks."""

    tracer = Tracer(GATE_SET)
    add = tracer.gate("add")

    @tracer.definition(input_count=1, key="cell", role=cell_role)
    def cell(v):
        doubled = add(v[0], v[0])
        tripled = add(doubled, v[0])
        return add(tripled, doubled)

    @tracer.definition(input_count=1, key="tile", role=tile_role)
    def tile(v):
        return cell(cell(v[0]))

    @tracer.definition(input_count=1, key=("root", units), role=root_role)
    def root(v):
        return tracer.repeat(units, tile, v[0])[-1]

    return tracer.serialize(root)


def tiled_compiled(units: int) -> Compiled:
    description = tiled_description(units, tile_role="replay", cell_role="verification", root_role=None)
    return Compiler(GATE_SET).compile(description, INPUT)


class TileValues(Mapping[int, object]):
    """The full assignment, lazily: every tile reads the same input as tile 0."""

    def __init__(self, compiled: Compiled) -> None:
        one = tiled_compiled(1)
        self._reference = one.circuit.evaluate(INPUT)
        tile = compiled.index.replay_units.unit(0)
        self._base = tile.interval.start
        self._size = tile.size
        self._n = compiled.circuit.n
        assert one.index.replay_units.unit(0).interval == tile.interval

    def __getitem__(self, address: int) -> object:
        if not 0 <= address < self._n:
            raise KeyError(address)
        if address < self._base:
            return self._reference[address]
        return self._reference[self._base + (address - self._base) % self._size]

    def __iter__(self) -> Iterator[int]:
        return iter(range(self._n))

    def __len__(self) -> int:
        return self._n


class Scenario:
    """A compiled tiling with its honest prover's messages recorded once."""

    def __init__(self, units: int) -> None:
        self.units = units
        self.compiled = tiled_compiled(units)
        self.values = TileValues(self.compiled)
        outputs = tuple(self.values[o] for o in self.compiled.circuit.outputs)
        policy = VerificationPolicy(Fraction(EXPECTED_SELECTED, units), 1, 0)
        self.expectation: Expectation = make_expectation(
            self.compiled, policy, INPUT, outputs, **SEEDS
        )
        verifier = VerifierSession(self.expectation, self.compiled)
        prover = ProverSession(self.compiled, verifier.header, self.values)
        self.boundary = prover.boundary()
        self.replay_challenge = verifier.receive_boundary(self.boundary)
        self.interiors = prover.interiors(self.replay_challenge)
        self.sample_challenge = verifier.receive_interiors(self.interiors)
        self.evidence = prover.evidence(self.sample_challenge)
        assert verifier.receive_evidence(self.evidence).accepted
        self.selected = len(self.replay_challenge.selected)
        self.sampled = len(self.sample_challenge.selected)

    def verifier_phase(self, phase: str) -> Callable[[], float]:
        """Seconds spent by a fresh verifier in ``phase`` alone."""

        def timed() -> float:
            verifier = VerifierSession(self.expectation, self.compiled)
            if phase == "boundary":
                start = time.perf_counter()
                verifier.receive_boundary(self.boundary)
                return time.perf_counter() - start
            verifier.receive_boundary(self.boundary)
            if phase == "interiors":
                start = time.perf_counter()
                verifier.receive_interiors(self.interiors)
                return time.perf_counter() - start
            verifier.receive_interiors(self.interiors)
            start = time.perf_counter()
            verifier.receive_evidence(self.evidence)
            return time.perf_counter() - start

        return timed


SCENARIOS: dict[int, Scenario] = {}


def scenario(units: int) -> Scenario:
    if units not in SCENARIOS:
        SCENARIOS[units] = Scenario(units)
    return SCENARIOS[units]


def fastest(timed: Callable[[], float], repetitions: int = 15) -> float:
    return min(timed() for _ in range(repetitions))


def count_calls(monkeypatch, owner: object, names: tuple[str, ...]) -> dict[str, int]:
    """Count calls of ``owner.<name>`` for each name, via ``monkeypatch``."""

    counts = dict.fromkeys(names, 0)
    for name in names:
        original = getattr(owner, name)

        def counting(*args, _name=name, _original=original, **kwargs):
            counts[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(owner, name, counting)
    return counts


SMALL, LARGE = 1024, 32768


def test_the_tiling_scales_as_described() -> None:
    small, large = scenario(SMALL), scenario(LARGE)

    assert large.compiled.index.replay_units.count == 32 * small.compiled.index.replay_units.count
    assert large.compiled.circuit.n - 1 == 32 * (small.compiled.circuit.n - 1)
    assert large.compiled.index.boundary().count == LARGE + 1
    assert 4 <= small.selected <= 40 and 4 <= large.selected <= 40
    assert small.sampled == 2 * small.selected and large.sampled == 2 * large.selected


def test_receive_interiors_derives_each_interior_lazily(monkeypatch) -> None:
    large = scenario(LARGE)
    verifier = VerifierSession(large.expectation, large.compiled)
    verifier.receive_boundary(large.boundary)
    boundary_calls = count_calls(
        monkeypatch, core_index._Boundary, ("contains", "rank", "unrank", "_locate")
    )
    index_calls = count_calls(monkeypatch, core_index.Index, ("interior", "verification_units"))
    unit_calls = count_calls(monkeypatch, core_index.Units, ("unit", "owner"))

    verifier.receive_interiors(large.interiors)

    assert index_calls["interior"] == large.selected
    assert index_calls["verification_units"] == large.selected
    assert unit_calls["unit"] == 2 * large.selected
    assert unit_calls["owner"] == 0
    assert all(count == 0 for count in boundary_calls.values())


def test_verifier_phases_are_flat_in_the_number_of_replay_units() -> None:
    small, large = scenario(SMALL), scenario(LARGE)

    for phase, per in (("boundary", 1), ("interiors", small.selected), ("evidence", small.sampled)):
        small_time = fastest(small.verifier_phase(phase)) / per
        large_per = {"boundary": 1, "interiors": large.selected, "evidence": large.sampled}[phase]
        large_time = fastest(large.verifier_phase(phase)) / large_per
        assert large_time < 4 * small_time, (phase, small_time, large_time)


def test_a_full_verifier_run_touches_only_sampled_addresses(monkeypatch) -> None:
    large = scenario(LARGE)
    io = 2
    per_unit = 3 + 1  # a cell's three gates and the one outside address it reads
    sampled_addresses = large.sampled * per_unit
    boundary_calls = count_calls(monkeypatch, core_index._Boundary, ("contains", "rank", "unrank"))
    lookups = count_calls(monkeypatch, DescriptionCircuit, ("__getitem__",))
    monkeypatch.setattr(
        DescriptionCircuit,
        "evaluate",
        lambda *_a, **_k: pytest.fail("the verifier must never evaluate the circuit"),
    )

    verifier = VerifierSession(large.expectation, large.compiled)
    verifier.receive_boundary(large.boundary)
    verifier.receive_interiors(large.interiors)
    assert verifier.receive_evidence(large.evidence).accepted

    boundary_total = sum(boundary_calls.values())
    assert boundary_total <= 3 * (io + sampled_addresses)
    assert lookups["__getitem__"] <= 4 * (io + sampled_addresses)  # schema, decode, gate, check
    assert boundary_total < LARGE / 8 and lookups["__getitem__"] < LARGE / 8


def test_run_protocol_end_to_end_on_the_large_tiling() -> None:
    large = scenario(LARGE)

    run = run_protocol(large.compiled, large.expectation, large.values)

    assert run.report.accepted
    assert run.transcript is not None
    assert len(run.transcript.replay_challenge.selected) == large.selected


# -- marks are part of what is committed -------------------------------------------


def test_changing_a_role_mark_changes_the_compiled_digest() -> None:
    cells = Compiler(GATE_SET).compile(
        tiled_description(8, tile_role="replay", cell_role="verification", root_role=None), INPUT
    )
    tiles = Compiler(GATE_SET).compile(
        tiled_description(8, tile_role="verification", cell_role=None, root_role="replay"), INPUT
    )
    again = Compiler(GATE_SET).compile(
        tiled_description(8, tile_role="replay", cell_role="verification", root_role=None), INPUT
    )

    assert cells.circuit.n == tiles.circuit.n
    assert cells.circuit.evaluate(INPUT) == tiles.circuit.evaluate(INPUT)
    assert cells.index.replay_units.count == 8 and tiles.index.replay_units.count == 1
    assert cells.digest != tiles.digest
    assert cells.index.digest != tiles.index.digest
    assert cells.digest == again.digest
