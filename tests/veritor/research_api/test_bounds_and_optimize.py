from __future__ import annotations

from fractions import Fraction

import pytest

from veritor import (
    ArchitectureId,
    Bound,
    BoundOptions,
    BoundResult,
    Compile,
    Cost,
    CostParameters,
    ExpectedCost,
    MatmulCompileRequest,
    Optimization,
    Optimize,
    PolicyGrid,
    Unsupported,
    VerificationPolicy,
)
from veritor.core import Capability, Compiled
from veritor.plugins import NO_CONSTRUCTOR
from veritor.protocol import expected_work

POLICY = VerificationPolicy(Fraction(1, 2), Fraction(1, 2), Fraction(1, 4))


def test_demo_bound_folds_over_the_compiled_index() -> None:
    artifact = Compile(ArchitectureId.DEMO_G)
    assert isinstance(artifact, Compiled)

    result = Bound(artifact, POLICY)

    assert isinstance(result, BoundResult)
    assert result.digest == artifact.digest
    assert result.policy == POLICY
    assert 0 <= result.bits <= result.out_bits
    assert result.bits == min(result.knapsack_bits, result.laplace_bits, result.out_bits)


def test_matmul_bound_is_zero_under_full_checking_and_capped_under_none() -> None:
    artifact = Compile(ArchitectureId.MATMUL, MatmulCompileRequest(((3,),), (((5,),),)))
    assert isinstance(artifact, Compiled)

    exact = Bound(artifact, VerificationPolicy(1, 1, 0))
    nothing = Bound(artifact, VerificationPolicy(0, 1, Fraction(1, 2)))

    assert isinstance(exact, BoundResult) and isinstance(nothing, BoundResult)
    assert 0 <= exact.bits < 1e-9 and not exact.capped  # one honest output, rounded up
    assert nothing.bits == nothing.out_bits == 8 and nothing.capped


def test_bound_options_control_the_grid() -> None:
    artifact = Compile(ArchitectureId.MATMUL)
    coarse = Bound(artifact, POLICY, BoundOptions(max_buckets=4))
    fine = Bound(artifact, POLICY, BoundOptions(resolution=64))

    assert isinstance(coarse, BoundResult) and isinstance(fine, BoundResult)
    assert coarse.buckets == 4 < fine.buckets
    assert coarse.cost_step > fine.cost_step
    assert fine.bits <= coarse.bits + 1e-9


def test_cost_and_optimize_share_the_compiled_index() -> None:
    artifact = Compile(ArchitectureId.MATMUL)
    assert isinstance(artifact, Compiled)
    parameters = CostParameters(2, 1)

    expected = Cost(artifact, POLICY, parameters)
    chosen = Optimize(
        artifact,
        POLICY.eta,
        PolicyGrid.uniform(4),
        max_bits=30,
        parameters=parameters,
        accept=lambda policy: expected_work(artifact, policy, 6) <= 400,
    )

    assert isinstance(expected, ExpectedCost)
    assert expected.total == expected.boundary + expected.replay + expected.proof
    assert isinstance(chosen, Optimization)
    assert chosen.bound.bits <= 30
    assert chosen.cost == Cost(artifact, chosen.policy, parameters)
    assert expected_work(artifact, chosen.policy, 6) <= 400
    assert Optimize(artifact, POLICY.eta, PolicyGrid.uniform(1), max_bits=-1) is None


@pytest.mark.parametrize(
    "architecture_id",
    (
        ArchitectureId.GPT2,
        ArchitectureId.KIMI_K3,
        ArchitectureId.DEEPSEEK_V4_PRO,
        ArchitectureId.INKLING,
    ),
)
def test_analysis_is_unsupported_without_a_compiled_description(
    architecture_id: ArchitectureId,
) -> None:
    artifact = Compile(architecture_id)
    assert isinstance(artifact, Unsupported)

    outcomes = (
        Bound(artifact, POLICY),
        Cost(artifact, POLICY),
        Optimize(artifact, POLICY.eta, PolicyGrid.uniform(1), max_bits=1),
    )

    for outcome in outcomes:
        assert isinstance(outcome, Unsupported)
        assert outcome.capability is Capability.STATIC_BOUND
        assert outcome.reason_code == NO_CONSTRUCTOR
        assert outcome.plugin_id == artifact.plugin_id


def test_analysis_rejects_things_that_are_not_compile_results() -> None:
    with pytest.raises(TypeError, match="Compile result"):
        Bound(object(), POLICY)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Compile result"):
        Cost(object(), POLICY)  # type: ignore[arg-type]
