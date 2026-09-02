from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pytest

from circuit_cut_analysis.capacity import LogCardinality
from veritor.analysis import CapacityEvidence
from veritor.compile import Compiler, Tracer
from veritor.core import Compiled, make_word_gate_set

GATE_SET = make_word_gate_set(8)


def build_compiled(replay_sizes: Iterable[int]) -> Compiled:
    """A circuit whose replay unit ``r`` holds ``replay_sizes[r]`` one-gate verification units.

    Every verification unit doubles one of its own inputs, so the units are
    independent and the circuit's outputs are exactly the unit outputs.
    """

    sizes = tuple(replay_sizes)
    total = sum(sizes)
    tracer = Tracer(GATE_SET)
    add = tracer.gate("add")

    @tracer.definition(input_count=1, key="double", role="verification")
    def double(v):
        return add(v[0], v[0])

    def replay(size: int):
        @tracer.definition(input_count=size, key=("replay", size), role="replay")
        def unit(v):
            return tracer.repeat(size, double, v[0].by(1))

        return unit

    @tracer.definition(input_count=total, key="root")
    def root(v):
        outputs = []
        offset = 0
        for size in sizes:
            outputs.append(replay(size)(v[offset : offset + size]))
            offset += size
        return outputs

    return Compiler(GATE_SET).compile(tracer.serialize(root), [1] * total)


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
def make_compiled():
    return build_compiled


@pytest.fixture
def make_index():
    return lambda replay_sizes: build_compiled(replay_sizes).index


@pytest.fixture
def exact_oracle_type():
    return AdditiveExactOracle
