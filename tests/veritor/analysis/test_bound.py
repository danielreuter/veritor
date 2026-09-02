"""The fold against exhaustive references and brute-forced unions."""

from __future__ import annotations

import math
import time
from fractions import Fraction

import pytest

from veritor import compile_matmul
from veritor.analysis import BoundOptions, BoundResult, bound
from veritor.analysis.reference import (
    accepted_outputs,
    admissible_sets,
    cover_bits,
    cut_bits,
    error_counts,
    error_sets,
    subset_sum_bits,
    transcript_outputs,
)
from veritor.compile import Compiler
from veritor.constructors import Tracer
from veritor.core import Compiled, VerificationPolicy, make_word_gate_set

TOLERANCE = 1e-6

POLICIES = [  # (theta, eta)
    (VerificationPolicy(Fraction(1, 2), Fraction(1, 2)), Fraction(1, 4)),
    (VerificationPolicy(Fraction(1, 3), Fraction(1, 5)), Fraction(1, 100)),
    (VerificationPolicy(Fraction(9, 10), Fraction(9, 10)), Fraction(1, 10)),
    (VerificationPolicy(1, Fraction(1, 2)), Fraction(1, 8)),
    (VerificationPolicy(1, 1), Fraction(0)),
    (VerificationPolicy(0, 1), Fraction(1, 2)),
    (VerificationPolicy(Fraction(1, 2), Fraction(1, 2)), Fraction(0)),
    (VerificationPolicy(1, 1), Fraction(1, 2)),
]

FAMILIES = [(1,), (2,), (3, 2), (2, 2, 2), (4, 3)]


def relaxed(eta: Fraction, result: BoundResult, replay_units: int) -> Fraction:
    """The threshold the grid actually enforces: ``eta`` lowered by one step per replay unit."""

    if eta == 0 or math.isinf(result.cost_step):
        return eta
    return eta * Fraction(math.exp(-replay_units * result.cost_step))


@pytest.mark.parametrize("sizes", FAMILIES)
@pytest.mark.parametrize(("policy", "eta"), POLICIES)
def test_fold_sits_between_the_union_and_the_relaxed_per_set_sum(make_compiled, sizes, policy, eta):
    compiled = make_compiled(sizes)
    result = bound(compiled, policy, eta)

    assert result.digest == compiled.digest
    assert result.policy == policy and result.eta == eta
    assert 0 <= result.bits <= result.out_bits == 8 * sum(sizes)
    raw = min(result.knapsack_bits, result.laplace_bits, result.out_bits)
    assert raw - 1e-9 <= result.bits <= raw  # tightened to an integer count of outputs
    assert result.capped == (result.bits == result.out_bits)
    # The grid admits at most the sets admissible at the relaxed threshold,
    # and distinct covers never weigh more than the per-set sum ...
    per_set = subset_sum_bits(
        compiled, policy, relaxed(eta, result, compiled.index.replay_units.count)
    )
    assert result.knapsack_bits <= per_set + TOLERANCE
    # ... while every set admissible at eta is admitted.  Here one-gate units
    # make every cover distinct unless error counts are lumped (the lumped
    # subsets share the unit's interface), so without lumping both the
    # knapsack and the Laplace bound sit above the exact per-set sum.
    if result.errors_limit >= max(sizes):
        exact = subset_sum_bits(compiled, policy, eta)
        assert result.knapsack_bits >= exact - TOLERANCE
        assert result.laplace_bits >= exact - TOLERANCE


@pytest.mark.parametrize("sizes", FAMILIES)
def test_grid_is_exact_when_fine_enough(make_compiled, sizes):
    """Away from knife edges a fine grid admits nothing extra."""

    compiled = make_compiled(sizes)
    policy, eta = VerificationPolicy(Fraction(1, 2), Fraction(1, 2)), Fraction(1, 5)
    result = bound(compiled, policy, eta, BoundOptions(resolution=256, max_buckets=1 << 16))

    assert result.errors_limit >= max(sizes)
    assert result.knapsack_bits == pytest.approx(
        subset_sum_bits(compiled, policy, eta), abs=TOLERANCE
    )


def test_knife_edge_is_admitted_by_the_grid_only(make_compiled):
    """Three errors cost exactly ``Lambda = 3 ln 2``: inadmissible, but on the grid."""

    compiled = make_compiled((3, 2))
    policy, eta = VerificationPolicy(1, Fraction(1, 2)), Fraction(1, 8)
    result = bound(compiled, policy, eta)

    exact = subset_sum_bits(compiled, policy, eta)
    admitted = subset_sum_bits(compiled, policy, relaxed(eta, result, 2))
    assert exact < result.knapsack_bits <= admitted + TOLERANCE
    assert result.knapsack_bits == pytest.approx(admitted, abs=TOLERANCE)
    assert result.cost_step <= math.log(2) / 16 * (1 + 1e-9)


def test_cover_by_index_nodes_is_never_below_the_exact_cut(make_compiled):
    compiled = make_compiled((3, 2))
    for errors in error_sets(compiled.index):
        assert cut_bits(compiled, errors) <= cover_bits(compiled, errors)
        assert cover_bits(compiled, errors) == 8 * len(errors)
    assert error_counts(compiled.index, frozenset({0, 1, 4})) == [2, 1]


@pytest.fixture(scope="module")
def paper_outputs(make_paper_example):
    """Every transcript of the 8-gate fan-in circuit over 2-bit cells, once per marking."""

    return {split: transcript_outputs(make_paper_example(2, split), [1, 2, 3]) for split in (False, True)}


@pytest.mark.parametrize("split", [False, True])
def test_paper_fanin_example_union_is_below_the_fold(make_paper_example, paper_outputs, split):
    compiled = make_paper_example(2, split)
    outputs = paper_outputs[split]
    for policy, eta in POLICIES[:4]:
        union = len(accepted_outputs(outputs, policy, eta))
        result = bound(compiled, policy, eta)
        assert math.log2(union) <= result.bits + TOLERANCE
        # both h's and the tail together are covered by the replay unit's own
        # interface: one cover, far below the per-set sum
        if not split:
            assert result.knapsack_bits < subset_sum_bits(compiled, policy, eta) - 0.5


@pytest.mark.parametrize("seed", range(6))
def test_random_small_circuits_union_is_below_the_fold(make_random_compiled, seed):
    compiled = make_random_compiled(seed)
    inputs = list(range(1, compiled.index.input_count + 1))
    outputs = transcript_outputs(compiled, inputs)
    for policy, eta in POLICIES[:3]:
        union = len(accepted_outputs(outputs, policy, eta))
        result = bound(compiled, policy, eta)
        assert math.log2(union) <= result.bits + TOLERANCE
        # the per-set sum may cover an error set by the root; the fold only
        # caps its total by the root, which is never more
        per_set = subset_sum_bits(
            compiled, policy, relaxed(eta, result, compiled.index.replay_units.count)
        )
        assert result.bits <= per_set + TOLERANCE


def test_all_outputs_are_reachable_when_nothing_is_checked(make_paper_example, paper_outputs):
    compiled = make_paper_example(2, False)
    outputs = paper_outputs[False]
    everything = accepted_outputs(outputs, VerificationPolicy(0, 1), Fraction(1, 2))
    honest = accepted_outputs(outputs, VerificationPolicy(1, 1), Fraction(0))

    assert len(everything) == 1 << 4
    assert len(honest) == 1
    assert len(admissible_sets(compiled, VerificationPolicy(1, 1), Fraction(0))) == 1


def test_whole_unit_corruption_is_cheap_and_covered_once(make_compiled):
    """Mega-unit: with ``s = 1`` every error count in a unit costs ``-ln(1 - q)``."""

    compiled = make_compiled((6,))
    policy, eta = VerificationPolicy(Fraction(1, 2), 1), Fraction(1, 4)
    result = bound(compiled, policy, eta)

    assert result.errors_limit == 1
    # every nonempty subset is admissible; the unit's interface covers them all
    assert result.bits == pytest.approx(48.0, abs=0.01)
    assert result.bits < subset_sum_bits(compiled, policy, eta)


def source_only_units(inputs: int, weights: int) -> Compiled:
    """A replay unit of nothing but source gates beside one two-gate unit reading them."""

    gate_set = make_word_gate_set(8)
    tracer = Tracer(gate_set)
    add = tracer.gate("add")
    pair = tracer.definition(input_count=2, key="pair", role="verification")(lambda v: add(v[0], v[1]))

    @tracer.definition(input_count=0, key=("sources", inputs, weights), role="replay")
    def sources(_v):
        return tracer.inputs(inputs), tracer.weights(weights)

    @tracer.definition(input_count=2, key="work", role="replay")
    def work(v):
        return pair(v[0], v[1])

    @tracer.definition(input_count=0, key="root")
    def root(_v):
        cells = sources()
        return work(cells[0], cells[inputs])

    return Compiler(gate_set).compile(tracer.serialize(root), [1] * inputs)


def test_a_unit_of_source_gates_has_no_capacity(make_compiled):
    compiled = source_only_units(3, 2)
    index = compiled.index
    rows = {row.kind: row for row in index.kinds()}
    sources = index.replay_units.unit(0)
    cells = list(index.verification_units(0))

    assert rows[sources.kind].out_bits == 0 and rows[sources.kind].out_count == 0
    assert all(rows[cell.kind].out_bits == 0 for cell in cells) and len(cells) == 5
    assert index.interior(0).count == 0
    # the error set may name the source cells: they cover nothing, and the exact cut is empty
    for errors in error_sets(index):
        expected = 8 if any(index.verification_unit(u).replay_unit == 1 for u in errors) else 0
        assert cover_bits(compiled, errors) == cut_bits(compiled, errors) == expected
    for policy, eta in POLICIES[:4]:
        result = bound(compiled, policy, eta)
        assert result.out_bits == 8 and 0 <= result.bits <= 8
        # the same bound as a circuit without the source unit: one 8-bit unit, one replay unit
        # (the knapsack sum still ranges over the source unit's error counts, each weighing
        # 2**0; the root's interface caps that multiplicity away here)
        alone = bound(make_compiled((1,)), policy, eta)
        assert result.bits == pytest.approx(alone.bits, abs=TOLERANCE)
        assert result.knapsack_bits >= alone.knapsack_bits - TOLERANCE
    outputs = transcript_outputs(compiled, [1, 2, 3], [4, 5])
    everything = accepted_outputs(outputs, VerificationPolicy(0, 1), Fraction(1, 2))
    honest = accepted_outputs(outputs, VerificationPolicy(1, 1), Fraction(0))
    assert len(everything) == 256 and honest == {(5,)}  # 1 + 4: the sources hold their values


def test_bound_on_the_matmul_counts_the_dots_and_not_the_source_units():
    compiled = compile_matmul()  # activations and weights units, 3 rows of 2 dots
    index = compiled.index
    rows = {row.kind: row for row in index.kinds()}
    activations, weights = index.replay_units.unit(0), index.replay_units.unit(1)

    assert rows[activations.kind].out_bits == rows[weights.kind].out_bits == 0
    assert all(rows[index.replay_units.unit(r).kind].out_bits == 16 for r in range(2, 5))
    for policy, eta in POLICIES[:4]:
        result = bound(compiled, policy, eta)
        assert result.out_bits == 6 * 8 and 0 <= result.bits <= 48
        # a policy checking everything leaves nothing; sampling half the rows leaves at least a row
        if policy.q == policy.s == 1 and eta == 0:
            assert result.bits == 0
    leaky = bound(compiled, VerificationPolicy(Fraction(1, 2), 1), Fraction(1, 4))
    assert 16 <= leaky.bits <= 48 and not leaky.capped


def test_bound_rejects_foreign_inputs_and_bad_options(make_compiled):
    compiled = make_compiled((1,))
    policy, eta = POLICIES[0]
    with pytest.raises(TypeError, match="Compiled"):
        bound(compiled.circuit, policy, eta)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="VerificationPolicy"):
        bound(compiled, (1, 1), eta)  # type: ignore[arg-type]
    for bad_eta in (1, Fraction(5, 4), "-1/8"):
        with pytest.raises(ValueError, match=r"eta must lie in \[0, 1\)"):
            bound(compiled, policy, bad_eta)
    with pytest.raises(TypeError, match="eta"):
        bound(compiled, policy, 0.25)  # type: ignore[arg-type]
    assert bound(compiled, policy, "1/4") == bound(compiled, policy, eta)
    for field in ("max_buckets", "resolution", "max_errors"):
        with pytest.raises(ValueError, match=field):
            BoundOptions(**{field: 0})
        with pytest.raises(ValueError, match=field):
            BoundOptions(**{field: 2.0})


def synthetic_transformer_shape() -> Compiled:
    """12 layers x 1024 blocks x 64 heads x 128 gates: ``10**8`` gates, 4 kinds."""

    gate_set = make_word_gate_set(16)
    tracer = Tracer(gate_set)
    add, mul = tracer.gate("add"), tracer.gate("mul")
    width = 64

    @tracer.definition(input_count=width, key="head", role="verification")
    def head(v):
        return [add(mul(v[i], v[(i + 1) % width]), v[(i + 2) % width]) for i in range(width)]

    @tracer.definition(input_count=width, key="block", role="replay")
    def block(v):
        current = list(v)
        for _ in range(64):
            current = list(head(*current))
        return current

    @tracer.definition(input_count=width, key="layer")
    def layer(v):
        current = list(v)
        for _ in range(1024):
            current = list(block(*current))
        return current

    @tracer.definition(input_count=0, key="inputs", role="replay")
    def source(_v):
        return tracer.inputs(width)

    @tracer.definition(input_count=0, key="root")
    def root(_v):
        current = list(source())
        for _ in range(12):
            current = list(layer(*current))
        return current

    return Compiler(gate_set).compile(tracer.serialize(root), [1] * width)


def test_fold_never_enumerates_copies():
    compiled = synthetic_transformer_shape()
    assert compiled.circuit.n > 10**8
    assert compiled.index.verification_unit_count == 12 * 1024 * 64 + 64  # the heads and the input cells

    for policy, eta in (
        (VerificationPolicy(Fraction(1, 2), Fraction(1, 2)), Fraction(1, 4)),
        (VerificationPolicy(Fraction(1, 10), Fraction(1, 10)), Fraction(1, 10**6)),
        (VerificationPolicy(1, Fraction(1, 2)), Fraction(1, 10**6)),
    ):
        started = time.perf_counter()
        result = bound(compiled, policy, eta)
        assert time.perf_counter() - started < 2.0
        assert result.capped and result.bits == 1024
        assert result.knapsack_bits > 1024 and result.laplace_bits > 1024
