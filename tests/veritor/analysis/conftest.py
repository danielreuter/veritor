from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pytest

from circuit_cut_analysis.capacity import LogCardinality
from veritor.analysis import CapacityEvidence
from veritor.core import (
    ArtifactKind,
    RangeIndexedDomain,
    ReplayPartition,
    ReplayUnit,
    StructureIdentity,
    VerificationPartition,
    VerificationUnit,
    identity_digest,
)


def build_partitions(
    replay_sizes: Iterable[int],
    *,
    label: str = "analysis-test",
) -> tuple[ReplayPartition, VerificationPartition]:
    sizes = tuple(replay_sizes)
    total = sum(sizes)
    structure = StructureIdentity(
        schema_version="1",
        artifact_kind=ArtifactKind.STRUCTURAL_CIRCUIT,
        compiler_id="tests.analysis",
        compiler_version="1",
        semantic_scope_id="finite-bound-test",
        representation_digest=identity_digest(
            "tests/analysis/structure",
            {"label": label, "replay_sizes": list(sizes)},
        ),
    )
    eligible = RangeIndexedDomain(10, 10 + total)
    replay_units: list[ReplayUnit] = []
    verification_units: list[VerificationUnit] = []
    position = 10
    verification_index = 0
    for replay_index, size in enumerate(sizes):
        replay_units.append(
            ReplayUnit(
                replay_index,
                range(position, position + size),
                replay_cost=size + 1,
            )
        )
        for offset in range(size):
            verification_units.append(
                VerificationUnit(
                    verification_index,
                    replay_index,
                    (position + offset,),
                )
            )
            verification_index += 1
        position += size
    replay = ReplayPartition(structure, eligible, replay_units)
    verification = VerificationPartition(
        structure,
        replay,
        eligible,
        verification_units,
    )
    return replay, verification


@dataclass(frozen=True, slots=True)
class AdditiveExactOracle:
    weights: tuple[int, ...]
    frontier: int = 64
    assumptions: tuple[str, ...] = ()

    def evaluate(
        self,
        attack_support: frozenset[int],
    ) -> CapacityEvidence[frozenset[int]]:
        bits = min(self.frontier, sum(self.weights[index] for index in attack_support))
        capacity = LogCardinality.bits(bits)
        return CapacityEvidence(
            lower_bound=capacity,
            upper_bound=capacity,
            requested_support=attack_support,
            evaluated_support=attack_support,
            method="test-exact-additive",
            assumptions=self.assumptions,
        )


@pytest.fixture
def make_partitions():
    return build_partitions


@pytest.fixture
def exact_oracle_type():
    return AdditiveExactOracle
