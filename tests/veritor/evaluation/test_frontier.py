"""The frontier prices partitions and policies and answers the calibration question.

At toy dimensions the numbers are not the paper's, but the mechanics are:
every point is the protocol's own ``Bound``, ``Cost`` and ``expected_work``
divided by the honest computation; ``certify`` picks the smallest capacity
within the budgets and is monotone in them; points survive a round trip
through JSON; the tables render.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from veritor.analysis import BoundOptions, PolicyGrid, bound, cost
from veritor.core import VerificationPolicy
from veritor.evaluation import (
    FRONTIER_OPTIONS,
    Point,
    ServingShape,
    calibration_table,
    certify,
    honest_cost,
    partition_table,
    price,
    serving_table,
    sweep,
)
from veritor.evaluation.frontier import load, save
from veritor.protocol.parameters import expected_work

TOY = ServingShape(
    vocab=8, d_model=4, heads=2, layers=1, prompt=2, generated=3, requests=4, batch=2
)
GRID = PolicyGrid(q=(Fraction(1, 2), Fraction(1, 8)), s=(1, Fraction(1, 4)))
ETAS = (Fraction(1, 2), Fraction(1, 100))


@pytest.fixture(scope="module")
def points() -> list[Point]:
    return sweep(TOY, grid=GRID, etas=ETAS)


def test_a_point_is_the_protocol_functions_over_the_honest_computation() -> None:
    table = serving_table(TOY, "request", "row")
    policy = VerificationPolicy(Fraction(1, 2), Fraction(1, 4))

    point = price(table, TOY, "request", "row", policy, Fraction(1, 100))

    base = honest_cost(table)
    assert (
        base
        == next(row.replay_cost for row in table.rows if row.kind == table.root)
        > table.n
    )
    assert point.bits == bound(table, policy, Fraction(1, 100), FRONTIER_OPTIONS).bits
    assert point.overhead == cost(table, policy).total / base
    assert point.recompute == cost(table, policy).recompute / base
    assert (
        point.work
        == expected_work(table, policy, TOY.input_count + TOY.output_count) / base
    )
    assert point.out_bits == TOY.output_count * TOY.width and 0 <= point.fraction <= 1
    assert point.policy == policy and point.eta == Fraction(1, 100)
    # requests are closed: the recomputation is ``q`` of the requests; a cell is not: the sampled cells
    # force their requests to be re-executed, here (a toy) with probability well below one
    requests = sum(
        row.copies * row.replay_cost for row in table.rows if row.role == "replay"
    )
    assert point.recompute == policy.q * Fraction(requests, base)
    fine = price(
        serving_table(TOY, "cell", "gate"),
        TOY,
        "cell",
        "gate",
        policy,
        Fraction(1, 100),
    )
    assert fine.recompute > point.recompute and fine.recompute < 1


def test_the_laplace_only_bound_is_the_laplace_term_and_never_below_the_full_fold() -> (
    None
):
    table = serving_table(TOY, "cell", "gate")
    policy = VerificationPolicy(Fraction(1, 2), Fraction(1, 2))

    full = bound(table, policy, Fraction(1, 100))
    laplace = bound(table, policy, Fraction(1, 100), BoundOptions(knapsack=False))

    assert laplace.knapsack_bits == float("inf")
    assert laplace.bits == pytest.approx(
        min(laplace.laplace_bits, laplace.out_bits), abs=1e-9
    )
    assert laplace.bits >= full.bits - 1e-9


def test_the_sweep_covers_every_partition_policy_and_eta(points: list[Point]) -> None:
    keys = {(p.replay, p.verification, p.q, p.s, p.eta) for p in points}
    assert len(points) == len(keys) == 12 * 4 * 2
    assert {(p.replay, p.verification) for p in points} >= {
        ("request", "cell"),
        ("row", "cell"),
        ("step", "cell"),
    }
    assert all(
        p.seconds >= 0
        and p.overhead > 0
        and p.work > 0
        and 0 <= p.recompute <= p.overhead
        for p in points
    )


def test_certify_is_monotone_in_the_budgets(points: list[Point]) -> None:
    loose = certify(points, eta=Fraction(1, 2))
    assert loose is not None
    tighter = certify(points, eta=Fraction(1, 2), max_overhead=1, max_work=20)
    assert tighter is not None and tighter.bits >= loose.bits
    assert tighter.overhead <= 1 and tighter.work <= 20
    assert certify(points, eta=Fraction(1, 2), max_overhead=Fraction(1, 10**6)) is None
    assert certify(points, eta=Fraction(1, 3)) is None  # no such eta was swept


def test_a_larger_eta_certifies_no_more(points: list[Point]) -> None:
    lenient = certify(points, eta=Fraction(1, 2), max_work=20)
    strict = certify(points, eta=Fraction(1, 100), max_work=20)
    assert lenient is not None and strict is not None
    assert lenient.bits <= strict.bits


def test_points_round_trip_through_json(points: list[Point], tmp_path: Path) -> None:
    path = tmp_path / "points.json"
    save(points, TOY, path)

    shape, back = load(path)

    assert shape == TOY and back == points


def test_the_tables_render(points: list[Point]) -> None:
    table = calibration_table(
        points, eta=Fraction(1, 2), overheads=(Fraction(1, 10), 1, 10), works=(1, 100)
    )
    lines = table.splitlines()
    assert lines[0].startswith("| verifier work") and len(lines) == 4
    assert "--" in lines[2] and "`" in lines[3]

    by_partition = partition_table(points, eta=Fraction(1, 2), max_work=100)
    assert by_partition.count("\n") == 1 + 12
    assert "of which recompute" in by_partition.splitlines()[0]
