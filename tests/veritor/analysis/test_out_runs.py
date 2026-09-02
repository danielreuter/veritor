"""The run-based interfaces agree with enumeration on every traced example."""

from __future__ import annotations

import pytest

from veritor import compile_demo_g, compile_matmul


@pytest.mark.parametrize(
    "replay_sizes", [(1,), (2,), (1, 2, 3), (3, 3), (4, 1, 2, 1)], ids=lambda sizes: "x".join(map(str, sizes))
)
def test_replay_chains_match_enumeration(make_compiled, check_interfaces, replay_sizes):
    compiled = make_compiled(replay_sizes)
    check_interfaces(compiled.index, compiled.circuit)


@pytest.mark.parametrize("split", [False, True])
def test_paper_example_matches_enumeration(make_paper_example, check_interfaces, split):
    compiled = make_paper_example(2, split)
    check_interfaces(compiled.index, compiled.circuit)


@pytest.mark.parametrize("seed", range(12))
def test_random_circuits_match_enumeration(make_random_compiled, check_interfaces, seed):
    compiled = make_random_compiled(seed)
    check_interfaces(compiled.index, compiled.circuit)


@pytest.mark.parametrize("compile", [compile_demo_g, compile_matmul], ids=["demo-g", "matmul"])
def test_constructors_match_enumeration(check_interfaces, compile):
    compiled = compile()
    check_interfaces(compiled.index, compiled.circuit)
