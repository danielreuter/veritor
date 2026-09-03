"""The shapes `docs/benchmarks.md` reports, asserted as ratios between sizes.

Every test compares a cost at two or more sizes and bounds the ratio: flat
means the ratio stays under 3 across decades, linear means a 10x size costs
under 30x (a quadratic would cost 100x), logarithmic means a 100x size costs
under 3x.  The constants are never asserted.
"""

from __future__ import annotations

import random
from fractions import Fraction

import pytest

from benchmarks._synthetic import (
    GATE_SET,
    INPUT,
    chain_steps,
    deep_repeat,
    many_definitions,
    unrolled_units,
)
from veritor.compile import Compiler
from veritor.compile.description import parse_description
from veritor.core import RangeIndexedDomain, VerificationLimits, encode_value
from veritor.core.index import output_reach
from veritor.protocol.challenge import bernoulli_subset
from veritor.protocol.merkle import CommitmentDomain, MerkleTree, verify_opening

from .conftest import best_of, per_call

COMPILER = Compiler(GATE_SET)
CALLS = 200
SEED = b"\x11" * 32
STAGE = b"q/replay-unit"
PHASE = b"\x22" * 32
LIMITS = VerificationLimits()


def _random_addresses(n: int, count: int = CALLS) -> list[int]:
    rng = random.Random(n)
    return [rng.randrange(n) for _ in range(count)]


# -- compile ------------------------------------------------------------------------


def test_compile_is_flat_in_n_for_a_repeat_tower() -> None:
    """`repeat(10, ...)` nested 2 and 6 deep: n grows 10^4 times, the compile stays put."""

    small = deep_repeat((10, 10))
    large = deep_repeat((10,) * 6)
    t_small = best_of(lambda: COMPILER.compile(small, INPUT))
    t_large = best_of(lambda: COMPILER.compile(large, INPUT))
    assert (
        COMPILER.compile(large, INPUT).circuit.n
        > 9_000 * COMPILER.compile(small, INPUT).circuit.n
    )
    assert t_large / t_small < 3


def test_compile_is_at_most_linear_in_the_definitions() -> None:
    t_small = best_of(lambda: COMPILER.compile(many_definitions(32), INPUT))
    t_large = best_of(lambda: COMPILER.compile(many_definitions(320), INPUT))
    assert t_large / t_small < 30


# -- lazy lookup ---------------------------------------------------------------------


def test_gate_lookup_is_flat_across_three_decades_of_n() -> None:
    """Three `repeat` levels of factor k: depth fixed, n = 3 k^3 + 1 from 3e3 to 3e9."""

    latencies = []
    for k in (10, 100, 1000):
        circuit = COMPILER.compile(deep_repeat((k, k, k)), INPUT).circuit
        latencies.append(per_call(circuit.__getitem__, _random_addresses(circuit.n)))
    assert max(latencies) / min(latencies) < 3


def test_gate_lookup_grows_at_most_linearly_in_depth() -> None:
    shallow = COMPILER.compile(deep_repeat((2,) * 4), INPUT).circuit
    deep = COMPILER.compile(deep_repeat((2,) * 16), INPUT).circuit
    t_shallow = per_call(shallow.__getitem__, _random_addresses(shallow.n))
    t_deep = per_call(deep.__getitem__, _random_addresses(deep.n))
    assert t_deep / t_shallow < 3 * 4  # 4x the depth, at most linear (with slack)


# -- kind table and address sets ----------------------------------------------------------


def test_kind_table_is_at_most_linear_in_the_definitions() -> None:
    small = COMPILER.compile(many_definitions(32), INPUT).index
    large = COMPILER.compile(many_definitions(320), INPUT).index
    assert best_of(large.kind_table) / best_of(small.kind_table) < 30


def test_kind_table_is_flat_in_n() -> None:
    small = COMPILER.compile(deep_repeat((10, 10)), INPUT).index
    large = COMPILER.compile(deep_repeat((10,) * 8), INPUT).index
    assert large.n > 900_000 * small.n
    assert best_of(large.kind_table) / best_of(small.kind_table) < 3


def test_boundary_and_unit_lookups_are_flat_in_the_replay_units() -> None:
    """`repeat(U, block)`: every lookup descends by division whatever U is."""

    ratios = {"unrank": [], "contains": [], "unit": [], "interior": []}
    for units in (100, 10_000):
        index = COMPILER.compile(deep_repeat((8, units)), INPUT).index
        boundary = index.boundary()
        rng = random.Random(units)
        ranks = [rng.randrange(boundary.count) for _ in range(CALLS)]
        addresses = [rng.randrange(index.n) for _ in range(CALLS)]
        unit_ranks = [rng.randrange(index.replay_units.count) for _ in range(CALLS)]
        ratios["unrank"].append(per_call(boundary.unrank, ranks))
        ratios["contains"].append(per_call(boundary.contains, addresses))
        ratios["unit"].append(per_call(index.replay_units.unit, unit_ranks))
        ratios["interior"].append(per_call(index.interior, unit_ranks))
    for name, (small, large) in ratios.items():
        assert large / small < 3, name


# -- challenge sampling ---------------------------------------------------------------------


def test_bernoulli_subset_is_sublinear_in_the_candidates() -> None:
    """K = 64 expected whatever N: time must not follow N over six decades."""

    times = [
        best_of(
            lambda n=n: bernoulli_subset(SEED, STAGE, PHASE, n, Fraction(64, n), LIMITS)
        )
        for n in (10**3, 10**6, 10**9)
    ]
    assert max(times) / min(times) < 3


def test_bernoulli_subset_is_at_most_linear_in_the_selections() -> None:
    n = 10**6
    t_small = best_of(
        lambda: bernoulli_subset(SEED, STAGE, PHASE, n, Fraction(100, n), LIMITS)
    )
    t_large = best_of(
        lambda: bernoulli_subset(SEED, STAGE, PHASE, n, Fraction(1000, n), LIMITS)
    )
    assert t_large / t_small < 30


# -- Merkle ---------------------------------------------------------------------------


def _tree(count: int) -> MerkleTree:
    rng = random.Random(count)
    values = {k: encode_value(16, rng.randrange(1 << 16)) for k in range(count)}
    domain = CommitmentDomain(b"\x33" * 32, 7, RangeIndexedDomain(count))
    return MerkleTree(domain, values, lambda _p: "u16")


def test_merkle_build_is_linear_in_the_leaves() -> None:
    t_small = best_of(lambda: _tree(1_000))
    t_large = best_of(lambda: _tree(10_000))
    assert t_large / t_small < 30


def test_merkle_open_and_verify_are_logarithmic_in_the_leaves() -> None:
    limits = VerificationLimits()
    opens, verifies = [], []
    for count in (1_000, 100_000):
        tree = _tree(count)
        rng = random.Random(count)
        positions = [rng.randrange(count) for _ in range(CALLS)]
        openings = [tree.open(p) for p in positions]
        opens.append(per_call(tree.open, positions))
        verifies.append(
            per_call(
                lambda o, t=tree: verify_opening(
                    t.domain, t.commitment, o, "u16", limits
                ),
                openings,
            )
        )
    assert opens[1] / opens[0] < 3
    assert verifies[1] / verifies[0] < 3


# -- reach ---------------------------------------------------------------------------


def _root(description: bytes):
    return parse_description(description, GATE_SET).root


def test_output_reach_is_at_most_linear_in_independent_steps() -> None:
    small, large = _root(unrolled_units(128)), _root(unrolled_units(1280))
    assert (
        best_of(lambda: output_reach(large)) / best_of(lambda: output_reach(small)) < 30
    )


def test_output_reach_is_at_most_linear_in_the_definitions() -> None:
    small, large = _root(many_definitions(64)), _root(many_definitions(640))
    assert (
        best_of(lambda: output_reach(large)) / best_of(lambda: output_reach(small)) < 30
    )


def test_output_reach_is_flat_in_n() -> None:
    """Same nesting, branching 4 vs 40: the description is the same size, n is 10^6 times larger."""

    small, large = _root(deep_repeat((4,) * 6)), _root(deep_repeat((40,) * 6))
    assert (
        best_of(lambda: output_reach(large)) / best_of(lambda: output_reach(small)) < 3
    )


@pytest.mark.xfail(
    reason="known: output_reach is super-quadratic in a chain of dependent steps (docs/benchmarks.md)",
    strict=False,
)
def test_output_reach_is_at_most_linear_in_chained_steps() -> None:
    small, large = _root(chain_steps(256)), _root(chain_steps(1024))
    assert (
        best_of(lambda: output_reach(large)) / best_of(lambda: output_reach(small))
        < 4 * 3
    )


# -- slow: the same shapes one or two decades further ---------------------------------------------


@pytest.mark.slow
def test_gate_lookup_is_flat_up_to_n_of_3e12() -> None:
    latencies = []
    for k in (10, 1000, 10_000):
        circuit = COMPILER.compile(deep_repeat((k, k, k)), INPUT).circuit
        latencies.append(per_call(circuit.__getitem__, _random_addresses(circuit.n)))
    assert max(latencies) / min(latencies) < 3


@pytest.mark.slow
def test_merkle_build_is_linear_up_to_a_million_leaves() -> None:
    t_small = best_of(lambda: _tree(100_000), repeats=2)
    t_large = best_of(lambda: _tree(1_000_000), repeats=1)
    assert t_large / t_small < 30


@pytest.mark.slow
def test_bernoulli_subset_is_sublinear_in_n_up_to_10_to_the_12() -> None:
    times = [
        best_of(
            lambda n=n: bernoulli_subset(
                SEED, STAGE, PHASE, n, Fraction(1000, n), LIMITS
            )
        )
        for n in (10**4, 10**8, 10**12)
    ]
    assert max(times) / min(times) < 3


@pytest.mark.slow
def test_kind_table_is_at_most_linear_in_thousands_of_definitions() -> None:
    small = COMPILER.compile(many_definitions(256), INPUT).index
    large = COMPILER.compile(many_definitions(2560), INPUT).index
    assert best_of(large.kind_table, 2) / best_of(small.kind_table, 2) < 30
