from __future__ import annotations

import pytest

from circuit_cut_analysis.capacity import GateCapacity
from circuit_cut_analysis.indexed import (
    CausalPairsDomain,
    CausalReductionStepsDomain,
    ExplicitDomain,
    GateFamily,
    GateRef,
    IndexedCircuit,
    ProductDomain,
    RectangularDomain,
)
from veritor.plugins import (
    GateRefIndexedDomain,
    rank_index,
    supports_rank_unrank,
    unrank_index,
)


@pytest.mark.parametrize(
    "domain",
    (
        RectangularDomain((2, 3, 2)),
        CausalPairsDomain(5),
        CausalPairsDomain(5, strict=True),
        CausalReductionStepsDomain(5),
        ProductDomain(
            (
                RectangularDomain((2,)),
                CausalPairsDomain(3),
                ExplicitDomain(((1,), (4,))),
            )
        ),
        ExplicitDomain(((2, 7), (0, 1), (9, 3))),
    ),
)
def test_supported_index_domains_round_trip_exhaustively(domain) -> None:
    expected = tuple(domain.iter_indices())
    assert supports_rank_unrank(domain)
    assert len(expected) == domain.count
    for rank, index in enumerate(expected):
        assert rank_index(domain, index) == rank
        assert unrank_index(domain, rank) == index


def test_index_rank_errors_are_strict() -> None:
    domain = RectangularDomain((2, 3))
    with pytest.raises(KeyError):
        rank_index(domain, (2, 0))
    with pytest.raises(IndexError):
        unrank_index(domain, -1)
    with pytest.raises(IndexError):
        unrank_index(domain, domain.count)
    with pytest.raises(TypeError):
        unrank_index(domain, True)


def _small_indexed_circuit() -> IndexedCircuit:
    families = (
        GateFamily(
            "input",
            RectangularDomain((2,)),
            ("position",),
            GateCapacity.bits(2),
            "input",
        ),
        GateFamily(
            "computed",
            ProductDomain(
                (
                    RectangularDomain((2,)),
                    CausalPairsDomain(3),
                )
            ),
            ("layer", "query", "key"),
            GateCapacity.bits(3),
            "test",
            primitive="test",
        ),
    )
    return IndexedCircuit(
        families,
        (),
        (GateRef("computed", (0, 0, 0)),),
    )


@pytest.mark.parametrize("computed_only", (False, True))
def test_gate_ref_domain_round_trips_family_order_exhaustively(
    computed_only: bool,
) -> None:
    circuit = _small_indexed_circuit()
    domain = GateRefIndexedDomain(circuit, computed_only=computed_only)
    expected = tuple(
        GateRef(family.name, index)
        for family in circuit.families.values()
        if not computed_only or family.op != "input"
        for index in family.domain.iter_indices()
    )
    assert domain.count == len(expected)
    assert tuple(domain) == expected
    for rank, ref in enumerate(expected):
        assert domain.contains(ref)
        assert ref in domain
        assert domain.rank(ref) == rank
        assert domain.unrank(rank) == ref
        assert domain.at_rank(rank) == ref
    assert (
        GateRefIndexedDomain(
            circuit,
            computed_only=computed_only,
        ).identity_digest
        == domain.identity_digest
    )


def test_gate_ref_adapter_is_lazy_for_large_rectangular_family() -> None:
    family = GateFamily(
        "huge",
        RectangularDomain((1_000_000_000,)),
        ("index",),
        GateCapacity.bits(1),
        "input",
    )
    circuit = IndexedCircuit(
        (family,),
        (),
        (GateRef("huge", (0,)),),
    )
    domain = GateRefIndexedDomain(circuit)
    last = GateRef("huge", (999_999_999,))
    assert domain.count == 1_000_000_000
    assert domain.rank(last) == 999_999_999
    assert domain.unrank(999_999_999) == last
