from fractions import Fraction

import pytest

from veritor import compile_matmul
from veritor.analysis import CostParameters, ExpectedCost, cost
from veritor.core import Compiled, VerificationPolicy


def explicit_cost(compiled: Compiled, policy: VerificationPolicy, parameters: CostParameters) -> ExpectedCost:
    """The formula evaluated unit by unit over the explicit index."""

    index, circuit = compiled.index, compiled.circuit
    h, c0 = parameters.hash_cost, parameters.proof_overhead
    replay = sum(
        circuit.Cost(index.replay_units.unit(r), "replay") + h * index.interior(r).count
        for r in range(index.replay_units.count)
    )
    proof = sum(
        circuit.Cost(index.verification_unit(v), "proof") + c0
        for v in range(index.verification_unit_count)
    )
    return ExpectedCost(
        h * index.boundary().count,
        policy.q * replay,
        policy.q * policy.s * proof,
        h * index.weights().count,
    )


@pytest.mark.parametrize("sizes", [(1,), (3, 2), (2, 2, 2)])
@pytest.mark.parametrize(
    "policy",
    [
        VerificationPolicy(Fraction(1, 2), Fraction(1, 3)),
        VerificationPolicy(1, 1),
        VerificationPolicy(0, 1),
    ],
)
def test_cost_fold_matches_the_unit_by_unit_sum(make_compiled, sizes, policy):
    compiled = make_compiled(sizes)
    parameters = CostParameters(Fraction(3, 2), 5)

    assert cost(compiled, policy, parameters) == explicit_cost(compiled, policy, parameters)


def test_cost_fold_matches_on_nested_indices(make_paper_example):
    policy = VerificationPolicy(Fraction(2, 3), Fraction(1, 5))
    parameters = CostParameters(2, 1)
    for compiled in (make_paper_example(2, False), make_paper_example(2, True), compile_matmul()):
        assert isinstance(compiled, Compiled)
        assert cost(compiled, policy, parameters) == explicit_cost(compiled, policy, parameters)


def test_cost_terms_and_defaults(make_compiled):
    compiled = make_compiled((3, 2))  # 5 (in, add) units, replay interfaces are the unit outputs
    full = cost(compiled, VerificationPolicy(1, 1))
    nothing = cost(compiled, VerificationPolicy(0, 0))

    assert full.boundary == 5 + 5  # the input gates plus every replay unit's Out
    assert full.replay == 5 * 1  # replaying an ``in`` gate is free; the add costs one
    assert full.proof == 5 * 2  # both gates of a unit are proved
    assert full.total == 25 and full.weights == 0  # no weight gates: nothing to commit per epoch
    assert nothing == ExpectedCost(Fraction(10), Fraction(0), Fraction(0), Fraction(0))
    assert cost(compiled, VerificationPolicy(Fraction(1, 2), Fraction(1, 2))).total == 10 + Fraction(5, 2) + Fraction(10, 4)


def test_the_weight_commitment_is_priced_per_epoch_not_per_request():
    compiled = compile_matmul()  # 3 rows of 3 activations, a 3x2 weight matrix, 3x2 outputs
    index = compiled.index
    full = cost(compiled, VerificationPolicy(1, 1), CostParameters(2, 0))

    assert (index.input_count, index.weight_count) == (9, 6)
    # boundary: the input gates, the activations and weights units' Out (nothing:
    # their gates are pinned) and each row's dots
    assert full.boundary == 2 * (9 + 0 + 0 + 6)
    assert full.weights == 2 * 6
    assert full.total == full.boundary + full.replay + full.proof  # ``weights`` is not in the total
    # replaying the source units costs nothing: pinned gates are not in an interior
    assert full.replay == sum(
        compiled.circuit.Cost(index.replay_units.unit(r), "replay") + 2 * index.interior(r).count
        for r in range(index.replay_units.count)
    )
    assert index.interior(0).count == index.interior(1).count == 0


def test_cost_validates_its_inputs(make_compiled):
    compiled = make_compiled((1,))
    with pytest.raises(TypeError, match="Compiled"):
        cost(compiled.circuit, VerificationPolicy(1, 1))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="VerificationPolicy"):
        cost(compiled, (1, 1))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="hash_cost"):
        CostParameters(-1)
    with pytest.raises(TypeError, match="proof_overhead"):
        CostParameters(1, 0.5)
    assert CostParameters("3/2").hash_cost == Fraction(3, 2)
