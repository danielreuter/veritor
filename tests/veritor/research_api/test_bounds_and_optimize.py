from __future__ import annotations

from fractions import Fraction

from veritor import (
    Bound,
    BoundOptions,
    BoundResult,
    Cost,
    CostParameters,
    ExpectedCost,
    MatmulCompileRequest,
    Optimization,
    Optimize,
    PolicyGrid,
    VerificationPolicy,
    compile_demo_g,
    compile_matmul,
)
from veritor.protocol import expected_work

POLICY = VerificationPolicy(Fraction(1, 2), Fraction(1, 2))
ETA = Fraction(1, 4)


def test_demo_bound_folds_over_the_compiled_index() -> None:
    compiled = compile_demo_g().compiled

    result = Bound(compiled, POLICY, ETA)

    assert isinstance(result, BoundResult)
    assert result.digest == compiled.digest
    assert result.policy == POLICY and result.eta == ETA
    assert 0 <= result.bits <= result.out_bits
    raw = min(result.knapsack_bits, result.laplace_bits, result.out_bits)
    assert raw - 1e-9 <= result.bits <= raw  # tightened to an integer count of outputs


def test_matmul_bound_is_zero_under_full_checking_and_capped_under_none() -> None:
    compiled = compile_matmul(MatmulCompileRequest(((3,),), (((5,),),))).compiled

    exact = Bound(compiled, VerificationPolicy(1, 1), 0)
    nothing = Bound(compiled, VerificationPolicy(0, 1), Fraction(1, 2))

    assert exact.bits == 0.0 and not exact.capped  # one honest output, an exact count
    assert nothing.bits == nothing.out_bits == 8 and nothing.capped


def test_bound_options_control_the_grid() -> None:
    compiled = compile_matmul().compiled
    coarse = Bound(compiled, POLICY, ETA, BoundOptions(max_buckets=4))
    fine = Bound(compiled, POLICY, ETA, BoundOptions(resolution=64))

    assert coarse.buckets == 4 < fine.buckets
    assert coarse.cost_step > fine.cost_step
    assert fine.bits <= coarse.bits + 1e-9


def test_cost_and_optimize_share_the_compiled_index() -> None:
    compiled = compile_matmul().compiled
    parameters = CostParameters(2, 1)

    expected = Cost(compiled, POLICY, parameters)
    chosen = Optimize(
        compiled,
        ETA,
        PolicyGrid.uniform(4),
        max_bits=30,
        parameters=parameters,
        accept=lambda policy: expected_work(compiled, policy, 6) <= 400,
    )

    assert isinstance(expected, ExpectedCost)
    assert expected.total == expected.boundary + expected.replay + expected.proof
    assert isinstance(chosen, Optimization)
    assert chosen.bound.bits <= 30 and chosen.bound.eta == ETA
    assert chosen.cost == Cost(compiled, chosen.policy, parameters)
    assert expected_work(compiled, chosen.policy, 6) <= 400
    assert Optimize(compiled, ETA, PolicyGrid.uniform(1), max_bits=-1) is None
