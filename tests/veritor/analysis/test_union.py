"""The union of kind tables: what a round of runs is to ``Bound``."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import pytest

from veritor import compile_matmul
from veritor.analysis import bound, cost, union
from veritor.analysis.union import UNION_ROOT_TAG, UNION_TAG
from veritor.core import KindTable, VerificationPolicy, identity_digest
from veritor.evaluation import ServingShape, serving_table

from .conftest import build_compiled, paper_example, random_compiled

TOLERANCE = 1e-6

POLICIES = [  # (theta, eta)
    (VerificationPolicy(Fraction(1, 2), Fraction(1, 2)), Fraction(1, 4)),
    (VerificationPolicy(Fraction(1, 3), Fraction(1, 5)), Fraction(1, 100)),
    (VerificationPolicy(1, Fraction(1, 2)), Fraction(1, 8)),
    (VerificationPolicy(Fraction(1, 8), Fraction(1, 64)), Fraction(1, 2**40)),
    (VerificationPolicy(1, 1), Fraction(0)),
]

TOY = ServingShape(
    vocab=8, d_model=4, heads=2, layers=1, prompt=2, generated=3, requests=1
)


def rows_of(table: KindTable) -> dict[str, object]:
    return {row.kind: row for row in table.rows}


def test_the_union_of_a_table_with_itself_doubles_every_count() -> None:
    table = build_compiled((3, 5, 2)).kind_table()
    doubled = union([table, table])

    assert isinstance(doubled, KindTable)
    assert doubled.root != table.root
    assert (
        doubled.n,
        doubled.input_count,
        doubled.weight_count,
        doubled.replay_unit_count,
    ) == (
        2 * table.n,
        2 * table.input_count,
        2 * table.weight_count,
        2 * table.replay_unit_count,
    )
    before, after = rows_of(table), rows_of(doubled)
    assert set(after) == set(before) | {doubled.root}
    root, old_root = after[doubled.root], after[table.root]
    assert (
        root.out_bits
        == root.reach_bits
        == root.ancestor_bits
        == 2 * before[table.root].out_bits
    )
    assert (
        root.copies == 1 and root.role is None and root.input_count == 0 and root.closed
    )
    assert root.children == ((table.root, 2),)
    assert (
        root.size == doubled.n
        and root.verification_units == 2 * before[table.root].verification_units
    )
    assert dict(root.verification_kinds) == {
        kind: 2 * count for kind, count in before[table.root].verification_kinds
    }
    # the old root is enclosed by the union root now: its ancestor cut is the union's interface
    assert old_root.copies == 2 and old_root.ancestor_bits == root.out_bits
    assert old_root.cut_bits == before[table.root].cut_bits
    for kind, row in before.items():
        merged = after[kind]
        assert merged.copies == 2 * row.copies
        assert (merged.min_depth, merged.max_depth) == (
            row.min_depth + 1,
            row.max_depth + 1,
        )
        assert merged.cut_bits == row.cut_bits
        for name in ("role", "size", "out_bits", "reach_bits", "children", "closed"):
            assert getattr(merged, name) == getattr(row, name)
    assert doubled.digest == identity_digest(UNION_TAG, {"tables": [(table.digest, 2)]})
    assert doubled.root == identity_digest(UNION_ROOT_TAG, {"roots": [(table.root, 2)]})
    assert union([table, table]) == doubled  # deterministic
    assert union([table]).digest != doubled.digest


def test_the_union_accepts_compiled_artifacts_and_is_order_independent() -> None:
    first, second = build_compiled((2, 2)), paper_example(2, split=True)
    tables = union([first, second]), union([first.kind_table(), second.kind_table()])
    assert tables[0] == tables[1]
    swapped = union([second, first])
    assert swapped.digest == tables[0].digest and swapped.root == tables[0].root
    assert rows_of(swapped) == rows_of(tables[0])
    assert union([first, second, first]).digest == union([first, first, second]).digest


@pytest.mark.parametrize("copies", (1, 2, 3, 7))
@pytest.mark.parametrize("sizes", ((1,), (3, 2), (4, 3), (2, 2, 2)))
def test_a_union_of_identical_circuits_is_the_circuit_with_scaled_copies(
    sizes, copies
) -> None:
    """``N`` copies of a circuit under one root: the headline estimate's method."""

    one = build_compiled(sizes).kind_table()
    many = build_compiled(sizes * copies).kind_table()
    united = union([one] * copies)
    for policy, eta in POLICIES:
        expected = bound(many, policy, eta)
        result = bound(united, policy, eta)
        assert result.out_bits == expected.out_bits
        assert result.bits == pytest.approx(expected.bits, abs=TOLERANCE)
        assert result.knapsack_bits == pytest.approx(
            expected.knapsack_bits, abs=TOLERANCE
        )
        assert result.laplace_bits == pytest.approx(
            expected.laplace_bits, abs=TOLERANCE
        )
        assert result.capped == expected.capped
        for faults in (1, 2):
            assert bound(united, policy, eta, max_faults=faults).bits == pytest.approx(
                bound(many, policy, eta, max_faults=faults).bits, abs=TOLERANCE
            )


@pytest.mark.parametrize("requests", (2, 5))
def test_a_union_of_single_requests_prices_like_the_batch_of_requests(requests) -> None:
    """Without declarations the weights RU, the one row the two tables count differently, has no capacity."""

    one = serving_table(TOY, "request", "row")
    batch = serving_table(replace(TOY, requests=requests), "request", "row")
    united = union([one] * requests)
    assert rows_of(united)[united.root].out_bits == rows_of(batch)[batch.root].out_bits
    for policy, eta in POLICIES:
        result, expected = bound(united, policy, eta), bound(batch, policy, eta)
        assert result.bits == pytest.approx(expected.bits, abs=TOLERANCE)
        assert result.knapsack_bits == pytest.approx(
            expected.knapsack_bits, abs=TOLERANCE
        )
        assert result.laplace_bits == pytest.approx(
            expected.laplace_bits, abs=TOLERANCE
        )


def test_the_union_of_one_table_prices_like_the_table() -> None:
    for table in (
        build_compiled((3, 2)).kind_table(),
        compile_matmul().compiled.kind_table(),
    ):
        alone = union([table])
        for policy, eta in POLICIES:
            assert bound(alone, policy, eta).bits == bound(table, policy, eta).bits
            assert bound(alone, policy, eta, max_faults=2).bits == pytest.approx(
                bound(table, policy, eta, max_faults=2).bits, abs=TOLERANCE
            )


SHAPES = [
    build_compiled((3, 2)).kind_table(),
    build_compiled((1,)).kind_table(),
    paper_example(2, split=False).kind_table(),
    paper_example(2, split=True).kind_table(),
    compile_matmul().compiled.kind_table(),
    serving_table(TOY, "request", "row"),
    serving_table(replace(TOY, requests=2, batch=2), "step", "row"),
] + [random_compiled(seed).kind_table() for seed in range(6)]


@pytest.mark.parametrize("seed", range(8))
def test_the_bound_is_monotone_under_adding_runs(seed) -> None:
    """Adding a run to a round never lowers its capacity: every sub-union is priced at most as high."""

    import random

    rng = random.Random(seed)
    tables = [rng.choice(SHAPES) for _ in range(rng.randint(2, 5))]
    for policy, eta in POLICIES:
        for max_faults in (0, 1):
            previous = -1.0
            for prefix in range(1, len(tables) + 1):
                result = bound(
                    union(tables[:prefix]), policy, eta, max_faults=max_faults
                )
                assert result.bits + TOLERANCE >= previous, (
                    prefix,
                    policy,
                    eta,
                    max_faults,
                )
                previous = result.bits
            whole = bound(union(tables), policy, eta, max_faults=max_faults).bits
            for table in tables:
                assert (
                    bound(table, policy, eta, max_faults=max_faults).bits
                    <= whole + TOLERANCE
                )
            # any sub-multiset, not just prefixes
            subset = [table for table in tables if rng.random() < 0.5] or tables[:1]
            assert (
                bound(union(subset), policy, eta, max_faults=max_faults).bits
                <= whole + TOLERANCE
            )


def test_nested_unions_price_like_the_flat_union() -> None:
    a, b, c = SHAPES[0], SHAPES[2], SHAPES[4]
    nested, flat = union([union([a, b]), c]), union([a, b, c])
    assert nested.digest != flat.digest  # a different table ...
    assert (nested.n, nested.replay_unit_count) == (flat.n, flat.replay_unit_count)
    for policy, eta in POLICIES:  # ... with the same capacity
        assert bound(nested, policy, eta).bits == pytest.approx(
            bound(flat, policy, eta).bits, abs=TOLERANCE
        )
        assert bound(nested, policy, eta, max_faults=1).bits == pytest.approx(
            bound(flat, policy, eta, max_faults=1).bits, abs=TOLERANCE
        )


def test_cost_and_work_are_additive_over_the_union() -> None:
    table = serving_table(TOY, "request", "row")
    policy = VerificationPolicy(Fraction(1, 2), Fraction(1, 8))
    assert cost(union([table, table]), policy).total == 2 * cost(table, policy).total
    other = build_compiled((3, 2)).kind_table()
    assert (
        cost(union([table, other]), policy).total
        == cost(table, policy).total + cost(other, policy).total
    )


def test_the_union_checks_its_inputs() -> None:
    table = build_compiled((2, 3)).kind_table()
    with pytest.raises(ValueError, match="no tables"):
        union([])
    with pytest.raises(TypeError):
        union([table.rows])  # type: ignore[list-item]
    # the same digest over different rows is a misuse, not a table to count twice
    other = replace(table, replay_unit_count=table.replay_unit_count + 1)
    with pytest.raises(ValueError, match="share the digest"):
        union([table, other])
    # a kind shared by two tables must be the same definition
    kind = next(row.kind for row in table.rows if row.role == "verification")
    rows = tuple(
        replace(row, out_bits=row.out_bits + 8) if row.kind == kind else row
        for row in table.rows
    )
    conflicting = replace(
        table, rows=rows, digest=identity_digest("test/union/other", {})
    )
    with pytest.raises(ValueError, match="disagree on out_bits"):
        union([table, conflicting])
