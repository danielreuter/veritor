"""The folds read the kind table and nothing else.

``Bound``, ``Cost`` and ``expected_work`` give the same answer on a compiled
artifact and on its table, so a table written from a model of a circuit is
priced exactly like a profiled one; the table checks its own consistency.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from veritor.analysis import PolicyGrid, bound, cost, optimize
from veritor.core import KindTable, VerificationPolicy, as_kind_table
from veritor.protocol.parameters import expected_work

from .conftest import build_compiled, paper_example

POLICIES = (VerificationPolicy(1, 1), VerificationPolicy(Fraction(1, 2), Fraction(1, 3)), VerificationPolicy(Fraction(1, 5), 1))


@pytest.mark.parametrize("compiled", (build_compiled((3, 5, 2)), paper_example(split=True)), ids=("units", "paper"))
@pytest.mark.parametrize("policy", POLICIES)
def test_the_folds_agree_on_the_artifact_and_its_table(compiled, policy: VerificationPolicy) -> None:
    table = compiled.kind_table()

    assert isinstance(table, KindTable)
    assert table.rows == compiled.index.kinds() and table.digest == compiled.digest
    assert (table.n, table.input_count, table.weight_count, table.replay_unit_count) == (
        compiled.index.n,
        compiled.index.input_count,
        compiled.index.weight_count,
        compiled.index.replay_units.count,
    )
    for eta in (Fraction(1, 2), Fraction(1, 1000)):
        assert bound(table, policy, eta) == bound(compiled, policy, eta)
    assert cost(table, policy) == cost(compiled, policy)
    io = compiled.index.input_count + len(compiled.circuit.outputs)
    assert expected_work(table, policy, io) == expected_work(compiled, policy, io)


def test_optimize_accepts_the_table() -> None:
    compiled = build_compiled((4, 4))

    grid = PolicyGrid(q=(Fraction(1, 2), 1), s=(Fraction(1, 2), 1))

    from_table = optimize(compiled.kind_table(), Fraction(1, 2), grid, max_bits=8)
    from_artifact = optimize(compiled, Fraction(1, 2), grid, max_bits=8)

    assert from_table == from_artifact


def test_as_kind_table_passes_tables_through_and_rejects_the_rest() -> None:
    compiled = build_compiled((2,))
    table = compiled.kind_table()

    assert as_kind_table(table) is table
    assert as_kind_table(compiled) == table
    with pytest.raises(TypeError):
        as_kind_table(compiled.index)  # type: ignore[arg-type]


def test_a_table_checks_its_kinds() -> None:
    table = build_compiled((2, 3)).kind_table()
    rows = table.rows

    with pytest.raises(ValueError, match="distinct kinds"):
        KindTable(rows + rows[:1], table.root, table.n, table.input_count, table.weight_count, 2, table.digest)
    with pytest.raises(ValueError, match="root must be one of its rows"):
        KindTable(rows, "nope", table.n, table.input_count, table.weight_count, 2, table.digest)
    # drop a kind the root calls: the root then calls an unknown kind
    root = next(row for row in rows if row.kind == table.root)
    dropped = root.children[0][0]
    with pytest.raises(ValueError, match="calls unknown kind"):
        KindTable(
            tuple(row for row in rows if row.kind != dropped),
            table.root,
            table.n,
            table.input_count,
            table.weight_count,
            2,
            table.digest,
        )
