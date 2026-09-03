"""``cost``: the boundary, the recomputation, the interior commitments, the proofs, the weights.

The recomputation term charges each sampled replay unit the re-execution of
its recomputation unit, the smallest closed kind containing it.  When every
replay unit is closed that is ``q Cost_replay`` per unit, the formula the
explicit unit-by-unit sum below evaluates; when replay units read values the
prover does not retain, the closed kind above them is re-executed with
probability ``1 - (1 - q)^m``.
"""

from __future__ import annotations

import math
from fractions import Fraction

import pytest

from veritor import compile_matmul
from veritor.analysis import CostParameters, ExpectedCost, cost
from veritor.analysis.cost import (
    CERTAIN_MASS,
    EXACT_BITS,
    recomputation_units,
    survival,
)
from veritor.compile import Compiler
from veritor.constructors import Tracer
from veritor.core import Compiled, VerificationPolicy, make_word_gate_set
from veritor.core.description import REPLAY, VERIFICATION
from veritor.evaluation import ServingShape, serving_table
from veritor.evaluation.frontier import honest_cost

GATES = make_word_gate_set(8)


def explicit_cost(
    compiled: Compiled, policy: VerificationPolicy, parameters: CostParameters
) -> ExpectedCost:
    """The formula evaluated unit by unit over the explicit index, every replay unit taken as closed."""

    index, circuit = compiled.index, compiled.circuit
    h, c0, alpha = (
        parameters.hash_cost,
        parameters.proof_overhead,
        parameters.proof_factor,
    )
    recompute = sum(
        circuit.Cost(index.replay_units.unit(r), "replay")
        for r in range(index.replay_units.count)
    )
    interior = sum(h * index.interior(r).count for r in range(index.replay_units.count))
    proof = sum(
        alpha * circuit.Cost(index.verification_unit(v), "proof") + c0
        for v in range(index.verification_unit_count)
    )
    return ExpectedCost(
        h * index.boundary().count,
        policy.q * recompute,
        policy.q * interior,
        policy.q * policy.s * proof,
        h * index.weights().count,
    )


def all_closed(compiled: Compiled) -> bool:
    return all(row.closed for row in compiled.kind_table().rows if row.role == REPLAY)


@pytest.mark.parametrize("sizes", [(1,), (3, 2), (2, 2, 2)])
@pytest.mark.parametrize(
    "policy",
    [
        VerificationPolicy(Fraction(1, 2), Fraction(1, 3)),
        VerificationPolicy(1, 1),
        VerificationPolicy(0, 1),
    ],
)
def test_cost_fold_matches_the_unit_by_unit_sum_over_closed_units(
    make_compiled, sizes, policy
):
    compiled = make_compiled(sizes)
    parameters = CostParameters(Fraction(3, 2), 5)

    assert all_closed(compiled)  # every unit holds its own ``in`` gates
    assert cost(compiled, policy, parameters) == explicit_cost(
        compiled, policy, parameters
    )


def test_cost_fold_matches_on_nested_closed_indices(make_paper_example):
    policy = VerificationPolicy(Fraction(2, 3), Fraction(1, 5))
    parameters = CostParameters(2, 1)
    for compiled in (make_paper_example(2, False), compile_matmul().compiled):
        assert isinstance(compiled, Compiled)
        assert all_closed(compiled)
        assert cost(compiled, policy, parameters) == explicit_cost(
            compiled, policy, parameters
        )


def test_a_replay_unit_fed_by_another_is_charged_through_the_root(make_paper_example):
    """The split paper example: ``rest`` reads ``first``'s outputs, so sampling it re-executes the run."""

    compiled = make_paper_example(2, True)
    policy = VerificationPolicy(Fraction(2, 3), Fraction(1, 5))
    parameters = CostParameters(2, 1)
    table = compiled.kind_table()
    rows = {row.kind: row for row in table.rows}
    first, rest = (rows[compiled.index.replay_units.unit(r).kind] for r in range(2))
    root = rows[table.root]
    q = policy.q

    assert (first.closed, rest.closed, root.closed) == (True, False, True)
    assert recomputation_units(table) == {first.kind: 1, root.kind: 1}
    expected = explicit_cost(compiled, policy, parameters)
    actual = cost(compiled, policy, parameters)
    # the root is re-executed when ``rest`` is sampled; ``first`` on its own only when ``rest`` is not
    assert actual.recompute == q * root.replay_cost + (1 - q) * q * first.replay_cost
    assert (
        actual.recompute
        > expected.recompute
        == q * (first.replay_cost + rest.replay_cost)
    )
    assert (actual.boundary, actual.commit_interior, actual.proof, actual.weights) == (
        expected.boundary,
        expected.commit_interior,
        expected.proof,
        expected.weights,
    )
    assert actual.replay == actual.recompute + actual.commit_interior
    assert actual.total == actual.boundary + actual.replay + actual.proof


def stages(m: int) -> Compiled:
    """A closed ``feed`` unit whose computed output ``m`` open ``stage`` units read, under a closed root."""

    tracer = Tracer(GATES)
    add = tracer.gate("add")
    double = tracer.definition(input_count=1, key="double", role="verification")(
        lambda v: add(v[0], v[0])
    )

    @tracer.definition(input_count=0, key="feed", role="replay")
    def feed(_v):
        return double(
            tracer.inputs(1)[0]
        )  # one verification unit over the input gate: a computed output

    @tracer.definition(input_count=1, key="stage", role="replay")
    def stage(v):
        return double(v[0])

    @tracer.definition(input_count=0, key="root")
    def root(_v):
        value = feed()
        return [stage(value) for _ in range(m)]

    return Compiler(GATES).compile(tracer.serialize(root), [1])


@pytest.mark.parametrize("m", [1, 3, 7])
@pytest.mark.parametrize("q", [Fraction(1, 2), Fraction(1, 5), 1, 0])
def test_open_units_under_a_closed_parent_are_charged_its_replay_with_probability_of_any_hit(
    m, q
):
    compiled = stages(m)
    table = compiled.kind_table()
    rows = {row.kind: row for row in table.rows}
    root = rows[table.root]
    units = [
        rows[compiled.index.replay_units.unit(r).kind]
        for r in range(compiled.index.replay_units.count)
    ]
    feed = next(row for row in units if row.closed)
    stage = next(row for row in units if not row.closed)
    policy = VerificationPolicy(q, 1)

    assert (
        stage.copies == m
        and root.replay_cost == feed.replay_cost + m * stage.replay_cost
    )
    assert recomputation_units(table) == {feed.kind: 1, root.kind: m}
    hit = 1 - (1 - Fraction(q)) ** m
    expected = hit * root.replay_cost + (1 - hit) * Fraction(q) * feed.replay_cost
    assert cost(compiled, policy).recompute == expected == cost(table, policy).recompute
    if q == 0:
        assert expected == 0
    if q == 1:
        assert expected == root.replay_cost


def test_with_many_open_units_the_recomputation_is_the_whole_honest_computation():
    """A serving run with one replay unit per dot product: every request is re-executed."""

    shape = ServingShape(
        vocab=64,
        d_model=64,
        heads=4,
        layers=4,
        prompt=16,
        generated=16,
        requests=64,
        batch=8,
    )
    fine = serving_table(shape, "cell", "gate")
    coarse = serving_table(shape, "request", "row")
    q = Fraction(
        1, 512
    )  # some 70 thousand replay units per request: q m is well over a hundred
    honest = honest_cost(fine)

    units = recomputation_units(fine)
    closed = [
        row
        for row in fine.rows
        if row.role != REPLAY and row.closed and row.copies == shape.requests
    ]
    request, prefill = sorted(
        closed, key=lambda row: -row.replay_cost
    )  # the request holds the prefill
    assert request.out_count == shape.generated and prefill.out_count > shape.generated
    assert (
        units[request.kind] * q > CERTAIN_MASS
    )  # far past the regime where a request could be skipped
    assert units[prefill.kind] * q > CERTAIN_MASS
    expected = cost(fine, VerificationPolicy(q, Fraction(1, 8)))
    # exactly: the request's survival is taken as 0, and a re-executed request covers its prefill
    assert expected.recompute == shape.requests * request.replay_cost
    assert expected.recompute / honest > Fraction(99, 100)
    # the same run with closed request units costs ``q`` of that
    closed = cost(coarse, VerificationPolicy(q, Fraction(1, 8)))
    assert closed.recompute == q * shape.requests * request.replay_cost
    assert honest_cost(coarse) == honest


def test_cost_terms_and_defaults(make_compiled):
    compiled = make_compiled(
        (3, 2)
    )  # 5 (in, add) units, replay interfaces are the unit outputs
    full = cost(compiled, VerificationPolicy(1, 1))
    nothing = cost(compiled, VerificationPolicy(0, 0))

    assert full.boundary == 5 + 5  # the input gates plus every replay unit's Out
    assert (
        full.recompute == 5 * 1
    )  # replaying an ``in`` gate is free; the add costs one
    assert full.commit_interior == 0  # every gate is an input or an output: no interior
    assert full.replay == full.recompute + full.commit_interior
    assert full.proof == 5 * 2  # both gates of a unit are proved
    assert (
        full.total == 25 and full.weights == 0
    )  # no weight gates: nothing to commit per epoch
    assert nothing == ExpectedCost(
        Fraction(10), Fraction(0), Fraction(0), Fraction(0), Fraction(0)
    )
    assert cost(
        compiled, VerificationPolicy(Fraction(1, 2), Fraction(1, 2))
    ).total == 10 + Fraction(5, 2) + Fraction(10, 4)


def test_the_weight_commitment_is_priced_per_epoch_not_per_request():
    compiled = compile_matmul().compiled  # 3 rows of 3 activations, a 3x2 weight matrix
    index = compiled.index
    full = cost(compiled, VerificationPolicy(1, 1), CostParameters(2, 0))

    assert (index.input_count, index.weight_count) == (9, 6)
    # boundary: the input gates, the activations and weights units' Out (nothing:
    # their gates are pinned) and each row's dots
    assert full.boundary == 2 * (9 + 0 + 0 + 6)
    assert full.weights == 2 * 6
    assert (
        full.total == full.boundary + full.replay + full.proof
    )  # ``weights`` is not in the total
    # replaying the source units costs nothing: pinned gates are not in an interior
    assert full.replay == sum(
        compiled.circuit.Cost(index.replay_units.unit(r), "replay")
        + 2 * index.interior(r).count
        for r in range(index.replay_units.count)
    )
    assert index.interior(0).count == index.interior(1).count == 0


def test_the_fold_agrees_on_the_artifact_and_its_table(make_paper_example):
    policy = VerificationPolicy(Fraction(1, 3), Fraction(1, 2))
    for compiled in (make_paper_example(2, True), stages(4), compile_matmul().compiled):
        assert cost(compiled, policy) == cost(compiled.kind_table(), policy)


def test_the_proving_factor_scales_the_proofs_and_nothing_else(
    make_compiled, make_paper_example
):
    """``alpha`` multiplies ``Cost_proof`` of every sampled VU; ``c_0``, the replay and the boundary are untouched."""

    policy = VerificationPolicy(Fraction(1, 3), Fraction(1, 5))
    for compiled in (
        make_compiled((3, 2)),
        make_paper_example(2, True),
        compile_matmul().compiled,
    ):
        table = compiled.kind_table()
        plain = cost(table, policy, CostParameters(2, 1))
        assert (
            cost(table, policy, CostParameters(2, 1, 1)) == plain
        )  # the default is one native execution
        scaled = cost(table, policy, CostParameters(2, 1, Fraction(7, 2)))
        assert (
            scaled.boundary,
            scaled.recompute,
            scaled.commit_interior,
            scaled.weights,
        ) == (
            plain.boundary,
            plain.recompute,
            plain.commit_interior,
            plain.weights,
        )
        proofs = (
            policy.q
            * policy.s
            * sum(
                row.copies * row.proof_cost
                for row in table.rows
                if row.role == VERIFICATION
            )
        )
        assert scaled.proof - plain.proof == Fraction(5, 2) * proofs
        assert scaled.proof == Fraction(7, 2) * proofs + policy.q * policy.s * sum(
            row.copies for row in table.rows if row.role == VERIFICATION
        )
        # zero prices the proofs at their fixed overhead alone; the explicit sum agrees at every alpha
        assert (
            cost(table, policy, CostParameters(2, 1, 0)).proof == plain.proof - proofs
        )
        if all_closed(compiled):
            for alpha in (0, Fraction(7, 2), 100):
                parameters = CostParameters(2, 1, alpha)
                assert cost(compiled, policy, parameters) == explicit_cost(
                    compiled, policy, parameters
                )


def test_survival_is_exact_then_certain_then_a_float():
    # exact: the result's denominator stays within EXACT_BITS
    assert survival(Fraction(1, 2), 10) == Fraction(1, 1024)
    assert survival(Fraction(1, 3), 2048) == Fraction(2**2048, 3**2048)
    assert 2048 * Fraction(2, 3).denominator.bit_length() == EXACT_BITS
    assert survival(Fraction(0), 10**12) == 1 and survival(Fraction(1), 10**12) == 0
    assert survival(Fraction(1, 2), 0) == 1
    q = Fraction(1, 2048)
    assert survival(q, 341) == Fraction(2047, 2048) ** 341  # 341 * 12 bits
    # certain: q m > CERTAIN_MASS, the survival is 0 exactly
    assert survival(q, 10**10) == 0 and survival(Fraction(1, 3), 2049) == 0
    assert (
        q * (CERTAIN_MASS * 2048 + 1) > CERTAIN_MASS
        and survival(q, CERTAIN_MASS * 2048 + 1) == 0
    )
    # float: too big for exact arithmetic, too likely to round to 0
    m = 1000
    approximate = survival(q, m)
    exact = Fraction(2047, 2048) ** m
    assert (
        approximate.denominator.bit_length() <= 1075 < exact.denominator.bit_length()
    )  # a binary float
    assert abs(float(approximate - exact)) < 1e-12
    assert float(approximate) == pytest.approx(math.exp(m * math.log1p(-1 / 2048)))
    with pytest.raises(ValueError, match="nonnegative"):
        survival(q, -1)


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
    with pytest.raises(ValueError, match="proof_factor"):
        CostParameters(1, 0, -1)
    assert CostParameters("3/2").hash_cost == Fraction(3, 2)
    assert CostParameters().proof_factor == 1
    assert CostParameters(proof_factor="5/2").proof_factor == Fraction(5, 2)
