"""The disjoint union of circuits as one kind table: what a round of runs is to ``Bound``.

A round of the epoch layer (:mod:`veritor.protocol.epoch`) challenges several
runs at once, each RU of each run selected independently with ``q`` and
each VU of a selected RU with ``s``.  That is exactly the sampling of one
circuit -- the disjoint union of the runs' circuits -- so the round's
capacity is ``Bound`` of the union's kind table, and :func:`union` writes
that table from the runs' own.

The union circuit has a fresh root whose steps call each constituent root
once (``count`` times for a table that occurs ``count`` times); nothing
below the roots changes.  Rows therefore merge by kind digest: a kind that
occurs in several tables is one kind with the copies summed, its
definition-determined fields (size, costs, interface, children, ...)
identical -- the digest names the definition -- and its context-dependent
bounds merged the way the profiler would have computed them over all the
copies at once: ``reach_bits`` and ``ancestor_bits`` are maxima over
copies, so they take the maximum; ``min_depth``/``max_depth`` shift by the
new root and take the extremes; ``closed`` holds iff every call site feeds
retained values, so it takes the conjunction.  A constituent root becomes
an ordinary row enclosed by the union root, whose interface is the sum of
the roots' ``out_bits``: its ``ancestor_bits`` becomes that sum (it has an
ancestor now), which leaves its bottleneck ``min(out, reach, ancestor)``
and every descendant's unchanged.  The union root reaches, and is enclosed
by, its own interface, as every root is.

``Bound`` of the union is monotone under adding tables (every fold term is
nonnegative and the interface cap grows), and for ``N`` copies of one
table it equals the bound of that circuit with every copy count scaled by
``N`` -- the headline estimate's method of treating a year as one circuit
(``docs/global-estimate.md``); ``tests/veritor/analysis/test_union.py``
checks both.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

from veritor.core import (
    Compiled,
    Digest,
    KindSummary,
    KindTable,
    as_kind_table,
    identity_digest,
)

UNION_TAG = "veritor/analysis/union/v1"
"""The table digest of a union: the multiset of its constituents' digests."""
UNION_ROOT_TAG = "veritor/analysis/union/root/v1"
"""The kind of the union root: the multiset of its constituents' root kinds."""

_DEFINITION_FIELDS = (
    "role",
    "size",
    "replay_cost",
    "proof_cost",
    "input_count",
    "out_count",
    "out_bits",
    "source_inputs",
    "source_weights",
    "children",
    "verification_units",
    "verification_kinds",
)
"""What the kind digest determines: rows of one kind must agree on these."""


def union(tables: Iterable[Compiled | KindTable]) -> KindTable:
    """The kind table of the disjoint union of the circuits behind ``tables``.

    Identical tables (equal digests) are counted rather than merged one by
    one, so ``union([T] * N)`` costs ``O(rows + N)``.  Raises ``ValueError``
    for no tables, for tables sharing a digest but differing, and for rows
    sharing a kind but disagreeing on what the kind determines.
    """

    counted = _counted(tables)
    if not counted:
        raise ValueError("the union of no tables is not a table")
    total_out_bits = sum(count * _root(table).out_bits for table, count in counted)
    rows: dict[str, KindSummary] = {}
    for table, count in counted:
        for row in table.rows:
            shifted = replace(
                row,
                copies=row.copies * count,
                min_depth=row.min_depth + 1,
                max_depth=row.max_depth + 1,
                ancestor_bits=total_out_bits
                if row.kind == table.root
                else row.ancestor_bits,
            )
            found = rows.get(row.kind)
            rows[row.kind] = shifted if found is None else _merge(found, shifted)
    root = _union_root(counted, total_out_bits)
    if root.kind in rows:  # pragma: no cover - a tagged digest never names a definition
        raise ValueError("the union root collides with a constituent kind")
    return KindTable(
        rows=(root, *rows.values()),
        root=root.kind,
        n=root.size,
        input_count=sum(count * table.input_count for table, count in counted),
        weight_count=sum(count * table.weight_count for table, count in counted),
        replay_unit_count=sum(
            count * table.replay_unit_count for table, count in counted
        ),
        digest=identity_digest(
            UNION_TAG,
            {"tables": sorted((table.digest, count) for table, count in counted)},
        ),
    )


def _counted(tables: Iterable[Compiled | KindTable]) -> list[tuple[KindTable, int]]:
    """The distinct tables in first-seen order, each with its multiplicity."""

    order: list[tuple[KindTable, int]] = []
    slot: dict[Digest, int] = {}
    for target in tables:
        table = as_kind_table(target)
        index = slot.get(table.digest)
        if index is None:
            slot[table.digest] = len(order)
            order.append((table, 1))
            continue
        known, count = order[index]
        if known != table:
            raise ValueError(
                f"two tables share the digest {table.digest[:12]} but differ"
            )
        order[index] = (known, count + 1)
    return order


def _root(table: KindTable) -> KindSummary:
    return next(row for row in table.rows if row.kind == table.root)


def _merge(first: KindSummary, second: KindSummary) -> KindSummary:
    """One row for a kind two constituents share: copies summed, bounds over all copies."""

    for name in _DEFINITION_FIELDS:
        if getattr(first, name) != getattr(second, name):
            raise ValueError(f"rows of kind {first.kind[:12]} disagree on {name}")
    return replace(
        first,
        copies=first.copies + second.copies,
        reach_bits=max(first.reach_bits, second.reach_bits),
        ancestor_bits=max(first.ancestor_bits, second.ancestor_bits),
        min_depth=min(first.min_depth, second.min_depth),
        max_depth=max(first.max_depth, second.max_depth),
        closed=first.closed and second.closed,
    )


def _multiset(pairs: Iterable[tuple[str, int]]) -> tuple[tuple[str, int], ...]:
    counts: dict[str, int] = {}
    for kind, count in pairs:
        counts[kind] = counts.get(kind, 0) + count
    return tuple(sorted(counts.items()))


def _union_root(counted: list[tuple[KindTable, int]], out_bits: int) -> KindSummary:
    """The fresh root: one call per constituent root, no ports, the summed interface."""

    roots = [(_root(table), count) for table, count in counted]
    return KindSummary(
        kind=identity_digest(
            UNION_ROOT_TAG,
            {"roots": sorted((root.kind, count) for root, count in roots)},
        ),
        role=None,
        copies=1,
        size=sum(count * root.size for root, count in roots),
        replay_cost=sum(count * root.replay_cost for root, count in roots),
        proof_cost=sum(count * root.proof_cost for root, count in roots),
        input_count=0,
        out_count=sum(count * root.out_count for root, count in roots),
        out_bits=out_bits,
        reach_bits=out_bits,
        ancestor_bits=out_bits,
        source_inputs=sum(count * root.source_inputs for root, count in roots),
        source_weights=sum(count * root.source_weights for root, count in roots),
        min_depth=0,
        max_depth=0,
        children=_multiset((root.kind, count) for root, count in roots),
        verification_units=sum(
            count * root.verification_units for root, count in roots
        ),
        verification_kinds=_multiset(
            (kind, count * inner)
            for root, count in roots
            for kind, inner in root.verification_kinds
        ),
        closed=True,
    )


__all__ = ["UNION_ROOT_TAG", "UNION_TAG", "union"]
