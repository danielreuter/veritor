"""GPT-2's structure: gate counts, marks, determinism, the legacy explicit DAG, and GPT-2 Small.

``GPT2G`` writes GPT-2's shape over the structural gate set.  At a tiny
shape the compiled table is checked gate for gate against the closed forms
(``gate_budget``: MACs per token, inner products, cells), every gate is in
exactly one replay unit (RU) and one verification unit (VU), the request and
the weights are closed, no VU is wider than the accumulator, and the per
component totals reconcile with the legacy ``circuit_cut_analysis`` GPT-2
DAG built at the same shape, up to four documented structural deltas.  At
``GPT2Shape.small()`` a three-request run (prompt 32, 32 generated tokens)
compiles in well under a second and its table carries the numbers of
``docs/gpt2-structure.md``.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from circuit_cut_analysis.models.gpt2 import GPT2Config
from circuit_cut_analysis.models.gpt2_circuit import build_gpt2_indexed_circuit
from veritor.analysis.bound import BoundOptions, bound, cut_bits
from veritor.analysis.cost import cost
from veritor.compile import Compiler
from veritor.constructors import GPT2G, GPT2Shape, Request, TracerError, gate_budget
from veritor.core import Compiled, VerificationPolicy
from veritor.core.description import REPLAY, VERIFICATION
from veritor.core.index import KindTable

TINY = GPT2Shape(layers=2, d_model=8, heads=2, d_ff=16, vocab=11, context=8)
REQUESTS = (Request((1, 2, 3), 3), Request((4, 5), 2))
LEGACY_REQUEST = REQUESTS[0]
ETA = Fraction(1, 2**40)


def compile_gpt2(constructor: GPT2G, requests: tuple[Request, ...]) -> Compiled:
    description, inputs = constructor(requests, b"")
    return Compiler(constructor.gate_set).compile(description, inputs)


def positions_of(requests: tuple[Request, ...]) -> int:
    return sum(len(r.prompt) + r.max_new - 1 for r in requests)


def predictions_of(requests: tuple[Request, ...]) -> int:
    return sum(r.max_new for r in requests)


def keys_of(requests: tuple[Request, ...]) -> int:
    """The number of (query, key) pairs attended: ``sum_p (p + 1)`` per request."""

    return sum(sum(range(1, len(r.prompt) + r.max_new)) for r in requests)


def by_name(constructor: GPT2G, table: KindTable) -> dict[str, tuple[int, int]]:
    """Kind name -> ``(copies, size)``; names are the keys ``gpt2.py`` traces under."""

    names = constructor.model.kind_names()
    return {names[row.kind]: (row.copies, row.size) for row in table.rows if row.kind in names}


def gates_of(kinds: dict[str, tuple[int, int]], *names: str) -> int:
    return sum(copies * size for name, (copies, size) in kinds.items() if name in names)


def gates_with_prefix(kinds: dict[str, tuple[int, int]], prefix: str) -> int:
    return sum(copies * size for name, (copies, size) in kinds.items() if name.startswith(prefix))


def components(constructor: GPT2G, table: KindTable, requests: tuple[Request, ...]) -> dict[str, int]:
    """The computed gates of a compiled run by component, read off the kind table by kind name."""

    shape = constructor.shape
    d, f, vocab = shape.d_model, shape.d_ff, shape.vocab
    kinds = by_name(constructor, table)
    return {
        "embedding": gates_of(kinds, "embed"),
        "layer_norm": gates_of(kinds, "layer_norm"),
        "attention": gates_of(kinds, f"matvec({d},{d},True,False)") + gates_with_prefix(kinds, "attend_head("),
        "mlp": gates_of(kinds, f"matvec({d},{f},True,False)", f"matvec({f},{d},True,False)", "gelu_cell"),
        "residual": gates_of(kinds, "add_cell") - positions_of(requests) * d,  # minus the embedding adds
        "lm_head": gates_of(kinds, f"matvec({d},{vocab},False,True)"),
        "argmax": gates_of(kinds, "argmax"),
    }


@pytest.fixture(scope="module")
def run() -> tuple[GPT2G, Compiled]:
    constructor = GPT2G(TINY)
    return constructor, compile_gpt2(constructor, REQUESTS)


# -- shape and weights ------------------------------------------------------------------


def test_the_shape_validates_and_small_is_gpt2_small() -> None:
    small = GPT2Shape.small()
    assert (small.layers, small.d_model, small.heads, small.d_ff, small.vocab, small.context) == (
        12,
        768,
        12,
        3072,
        50257,
        1024,
    )
    assert (small.width, small.acc_width, small.d_head) == (16, 32, 64)
    # 124,439,808 parameters (tied embedding), plus the token table and the seven scalars
    assert small.weight_count == 124_439_808 + 50257 + 7
    assert dict(small.layout())["wte"] == 50257 * 768 and dict(small.layout())["layer11.w_proj"] == 3072 * 768
    with pytest.raises(ValueError, match="multiple of heads"):
        GPT2Shape(layers=1, d_model=6, heads=4, d_ff=8, vocab=5, context=4)
    with pytest.raises(ValueError, match="positive"):
        GPT2Shape(layers=0, d_model=4, heads=2, d_ff=8, vocab=5, context=4)
    with pytest.raises(ValueError, match="acc_width"):
        GPT2Shape(layers=1, d_model=4, heads=2, d_ff=8, vocab=5, context=4, width=32, acc_width=16)
    with pytest.raises(ValueError, match="vocab <= 2"):
        GPT2Shape(layers=1, d_model=4, heads=2, d_ff=8, vocab=300, context=4, width=8)


def test_the_flat_weight_layout_is_documented_and_consistent() -> None:
    layout = TINY.layout()
    names = [name for name, _ in layout]
    assert names[:2] == ["wte", "wpe"] and names[-10:-7] == ["lnf_g", "lnf_b", "tokens"]
    assert names[-7:] == ["inv_d", "eps", "scale", "gelu_c3", "gelu_k", "one", "half"]
    assert names[2:18] == [f"layer0.{field}" for field, _ in TINY.layer_layout()]
    assert sum(count for _, count in layout) == TINY.weight_count
    assert TINY.layer_weights == 2 * 8 + 3 * (64 + 8) + (64 + 8) + 2 * 8 + (128 + 16) + (128 + 8)
    assert TINY.state_size(3) == 2 * 2 * 3 * 8


# -- gate counts ------------------------------------------------------------------------


def test_the_closed_forms_account_for_every_computed_gate(run) -> None:
    constructor, compiled = run
    index = compiled.index

    budget = constructor.gate_budget(REQUESTS)
    assert budget["total"] == sum(v for k, v in budget.items() if k != "total")
    assert budget["total"] == sum(gate_budget(TINY, len(r.prompt), r.max_new)["total"] for r in REQUESTS)
    assert index.n == budget["total"] + index.input_count + index.weight_count
    assert index.input_count == sum(len(r.prompt) for r in REQUESTS) == 5
    assert index.weight_count == TINY.weight_count == 1386
    assert index.n == 27_783


def test_macs_per_token_match_the_multiply_count(run) -> None:
    """MACs: ``4 d^2 + 2 d d_ff`` per layer (``12 d^2`` when ``d_ff = 4 d``), ``2 d c`` for attention, ``vocab d`` for the head.

    Every dot-product multiply is a copy of the one-gate ``acc_mul`` kind (the
    other ``acc_mul`` gates -- statistics, scale, GELU -- are inline), so its
    copy count is the MAC count.  The embedding one-hot adds ``vocab d`` MACs
    per position that a gather would not.
    """

    constructor, compiled = run
    kinds = by_name(constructor, compiled.kind_table())
    d, f, vocab, layers = TINY.d_model, TINY.d_ff, TINY.vocab, TINY.layers
    positions, predictions, keys = positions_of(REQUESTS), predictions_of(REQUESTS), keys_of(REQUESTS)

    projections = positions * layers * (4 * d * d + 2 * d * f)
    attention = layers * 2 * d * keys
    head = predictions * vocab * d
    embedding = positions * vocab * d
    assert kinds["acc_mul"] == (projections + attention + head + embedding, 1)
    small = GPT2Shape.small()  # d_ff = 4 d: the familiar 12 d^2 MACs per layer and token
    assert 4 * small.d_model**2 + 2 * small.d_model * small.d_ff == 12 * small.d_model**2
    # one ``narrow`` per inner product: the projections, the scores, the mix, the embedding and the head
    inner_products = (
        positions * layers * (4 * d + f + d) + layers * keys * TINY.heads + layers * positions * d + positions * d
    ) + predictions * vocab
    assert kinds[f"dot({d},True)"] == (positions * layers * (4 * d + f), 2 * d + 1)  # q, k, v, o and fc
    assert kinds[f"dot({f},True)"] == (positions * layers * d, 2 * f + 1)
    assert kinds[f"dot({d},False)"] == (predictions * vocab, 2 * d)
    assert kinds[f"dot({vocab},False)"] == (positions * d, 2 * vocab)
    assert kinds["score"] == (layers * TINY.heads * keys, 2 * TINY.d_head + 1)
    assert sum(copies for name, (copies, _) in kinds.items() if name.startswith("dot(") or name == "score") == (
        inner_products
    )


def test_the_components_match_the_closed_forms(run) -> None:
    constructor, compiled = run
    budget = constructor.gate_budget(REQUESTS)

    found = components(constructor, compiled.kind_table(), REQUESTS)

    assert found == {
        "embedding": budget["embedding"],
        "layer_norm": budget["layer_norm"],
        "attention": budget["attention"] + budget["softmax"],  # the head kind holds its softmax
        "mlp": budget["mlp"],
        "residual": budget["residual"],
        "lm_head": budget["lm_head"],
        "argmax": budget["argmax"],
    }
    kinds = by_name(constructor, compiled.kind_table())
    assert kinds["layer_norm"] == (2 * TINY.layers * positions_of(REQUESTS) + predictions_of(REQUESTS), 7 * 8 + 2)
    assert kinds["gelu_cell"] == (TINY.layers * positions_of(REQUESTS) * TINY.d_ff, 9)
    assert kinds["argmax"] == (predictions_of(REQUESTS), 3 * (TINY.vocab - 1))


# -- marks ------------------------------------------------------------------------------


def test_marks_tile_every_gate(run) -> None:
    """Every gate is in exactly one replay unit and exactly one verification unit."""

    _, compiled = run
    index, n = compiled.index, compiled.circuit.n

    replay = [index.replay_units.unit(r).interval for r in range(index.replay_units.count)]
    assert sorted(a for interval in replay for a in interval) == list(range(n))
    verification = [index.verification_unit(u).interval for u in range(index.verification_unit_count)]
    assert sorted(a for interval in verification for a in interval) == list(range(n))
    for address in range(0, n, 97):
        r = index.replay_units.owner(address)
        assert address in replay[r]
        block = index.verification_units(r)
        assert address in verification[block.first + block.owner(address)]
    # the two unit antichains, as the table sums them
    table = compiled.kind_table()
    assert sum(row.copies * row.size for row in table.rows if row.role == REPLAY) == n
    assert sum(row.copies * row.size for row in table.rows if row.role == VERIFICATION) == n
    assert index.replay_units.count == 1 + len(REQUESTS)
    assert index.verification_unit_count == sum(row.copies for row in table.rows if row.role == VERIFICATION)


def test_interfaces_match_enumeration_at_a_smaller_shape(check_interfaces) -> None:
    shape = GPT2Shape(layers=1, d_model=4, heads=2, d_ff=8, vocab=5, context=4)
    constructor = GPT2G(shape)
    compiled = compile_gpt2(constructor, (Request((1, 2), 2),))

    check_interfaces(compiled.index, compiled.circuit)


def test_the_request_and_the_weights_are_closed_and_the_decode_steps_are_not(run) -> None:
    constructor, compiled = run
    table = compiled.kind_table()
    rows = {row.kind: row for row in table.rows}
    names = {name: rows[digest] for digest, name in constructor.model.kind_names().items()}

    units = [row for row in table.rows if row.role == REPLAY]
    assert len(units) == 3 and all(row.closed for row in units)
    weights = names["weights"]
    assert (weights.source_weights, weights.out_count, weights.copies) == (TINY.weight_count, 0, 1)
    for prompt, new in ((3, 3), (2, 2)):
        request = names[f"request({prompt},{new})"]
        assert request.closed and request.copies == 1 and request.input_count == TINY.weight_count
        assert request.out_count == new and request.out_bits == request.reach_bits == new * TINY.width
        assert names[f"prefill({prompt})"].closed
    assert not names["decode(4)"].closed and not names["decode(5)"].closed and not names["decode(3)"].closed
    assert not names["layer(3,0)"].closed  # a layer is fed activations, not sources
    # the boundary is the prompts and the tokens: the KV caches stay inside their requests
    assert compiled.index.boundary().count == 5 + predictions_of(REQUESTS)
    assert rows[table.root].out_bits == predictions_of(REQUESTS) * TINY.width


def test_no_verification_unit_is_wider_than_the_accumulator(run) -> None:
    """``kappa_V = min(out_bits, reach_bits)`` is 16 or 32 bits for every computed gate.

    This is the paper's bottleneck claim read off the table: a dot product,
    a score, a probability, a GELU, an argmax and a scaled-and-shifted
    coordinate leave through one 16-bit value; a mean, a variance, a
    softmax maximum or denominator, a centred coordinate or an exponential
    through one 32-bit value; an equality of the one-hot through one bit.
    """

    constructor, compiled = run
    table = compiled.kind_table()
    names = constructor.model.kind_names()
    cells = [row for row in table.rows if row.role == VERIFICATION]

    widths = {}
    for row in cells:
        widths.setdefault(cut_bits(row), set()).add(names[row.kind].split("(")[0])
    assert widths == {
        0: {"veritor.source"},
        1: {"eq_cell"},
        16: {"add_cell", "argmax", "dot", "gelu_cell", "ln_out", "prob_cell", "score"},
        32: {"exp_cell", "ln_center", "ln_mean", "ln_var", "softmax_denominator", "softmax_max"},
    }
    assert all(row.out_bits == cut_bits(row) for row in cells)  # the interface, not the reach, is the cut
    sources = table.input_count + table.weight_count
    gates_at = {k: sum(r.copies * r.size for r in cells if cut_bits(r) == k) for k in widths}
    assert sum(gates_at.values()) == table.n
    computed = table.n - sources
    assert gates_at[0] == sources
    d, layers, heads = TINY.d_model, TINY.layers, TINY.heads
    positions, predictions, keys = positions_of(REQUESTS), predictions_of(REQUESTS), keys_of(REQUESTS)
    # 32-bit cells: mean, centre and variance of each LayerNorm (4 d + 2), and per query and head the
    # shifted exponentials (2 c), the maximum (c - 1) and the denominator (c): 4 c - 1
    assert gates_at[32] == (2 * layers * positions + predictions) * (4 * d + 2) + layers * heads * (4 * keys - positions)
    assert gates_at[32] == 1562 and gates_at[1] == positions * TINY.vocab == 88
    assert gates_at[1] + gates_at[16] + gates_at[32] == computed == 26_392
    assert (gates_at[1] + gates_at[16]) / computed == pytest.approx(0.94082, abs=1e-5)


# -- determinism and the constructor protocol --------------------------------------------


def test_compilation_is_deterministic(run) -> None:
    constructor, compiled = run

    again = GPT2G(TINY)
    description, inputs = again(REQUESTS, b"")
    assert (description, inputs) == constructor(REQUESTS, b"")
    recompiled = Compiler(again.gate_set).compile(description, inputs)
    assert recompiled.digest == compiled.digest and recompiled.index.digest == compiled.index.digest
    assert again.digest == constructor.digest
    other = GPT2Shape(layers=2, d_model=8, heads=4, d_ff=16, vocab=11, context=8)
    assert GPT2G(other).digest != constructor.digest
    # requests of one shape are one kind; the same run in another order is another description
    same = compile_gpt2(constructor, (Request((1, 2), 2), Request((3, 4), 2), Request((5, 6), 2)))
    units = [row for row in same.index.kinds() if row.role == REPLAY and row.out_count > 0]
    assert len(units) == 1 and units[0].copies == 3


def test_it_takes_no_advice_and_checks_its_requests() -> None:
    constructor = GPT2G(TINY)

    with pytest.raises(TracerError, match="no advice"):
        constructor(REQUESTS, b"x")
    with pytest.raises(TracerError, match="outside the vocabulary"):
        constructor((Request((11,), 1),), b"")
    with pytest.raises(TracerError, match="context is 8"):
        constructor((Request((1, 2, 3, 4), 5),), b"")
    with pytest.raises(TracerError, match="nonempty tuple"):
        constructor((), b"")
    assert constructor.output_layout(REQUESTS) == ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1))
    assert constructor.flatten_inputs(REQUESTS) == (1, 2, 3, 4, 5)
    assert constructor.manifest["shape"] == TINY.manifest


# -- the legacy explicit DAG ----------------------------------------------------------------


def legacy_components(shape: GPT2Shape, request: Request) -> dict[str, int]:
    """The legacy GPT-2 DAG at ``shape`` for one greedy request, its computed primitives by component.

    Families tagged ``inner-product-output`` are the legacy ``write`` nodes
    (no primitive: they are not counted as gates there) and ``embedding``
    lookups are zero-work wiring; both are returned separately.
    """

    config = GPT2Config(
        model_id="tiny",
        layers=shape.layers,
        hidden_size=shape.d_model,
        heads=shape.heads,
        intermediate_size=shape.d_ff,
        vocabulary_size=shape.vocab,
        max_context=shape.context,
    )
    circuit = build_gpt2_indexed_circuit(len(request.prompt), request.max_new, config=config).circuit
    totals = {
        "embedding": 0,
        "layer_norm": 0,
        "attention": 0,
        "mlp": 0,
        "residual": 0,
        "lm_head": 0,
        "argmax": 0,
        "writes": 0,
        "lookups": 0,
    }
    for family in circuit.families.values():
        tags = set(family.tags)
        if family.primitive is None:
            totals["writes" if "inner-product-output" in tags else "lookups"] += family.count
            continue
        if "layernorm" in tags:
            component = "layer_norm"
        elif "attention" in tags:
            component = "attention"
        elif "mlp" in tags:
            component = "mlp"
        elif "lm-head" in tags:
            component = "lm_head"
        elif "token" in tags:
            component = "argmax"
        elif "embedding" in tags:
            component = "embedding"
        else:
            assert "residual" in tags, family
            component = "residual"
        totals[component] += family.count
    assert sum(totals.values()) == circuit.gate_count
    assert sum(v for k, v in totals.items() if k not in ("writes", "lookups")) == sum(circuit.primitive_counts.values())
    return totals


def test_the_legacy_explicit_dag_agrees_up_to_four_structural_deltas() -> None:
    """Both structures at ``TINY`` for one request (prompt 3, 3 tokens): 16,627 gates here, 14,988 primitives there.

    The deltas, each a modelling choice and each exact:

    1. *Embedding gather.*  The legacy DAG looks a row up for free (a
       ``lookup`` node without a primitive); the grammar has no gather, so
       here it is a one-hot (``vocab`` equalities) and ``d_model`` dots of
       length ``vocab``: ``vocab + 2 vocab d_model`` gates per position.
    2. *Inner-product write-out.*  Every dot product here ends in a
       ``narrow`` gate (the 32 -> 16 bit rounding); the legacy DAG has the
       same node as a ``write`` without a primitive.  One per inner product,
       and the counts agree exactly.
    3. *Final LayerNorm.*  The legacy DAG normalises every processed
       position; here only positions that predict a token are normalised
       (``prompt - 1`` fewer copies of ``7 d_model + 2`` gates).
    4. *Argmax.*  One atomic ``vocab``-ary gate there; a tournament of
       ``vocab - 1`` compare-and-select nodes of three gates here.

    Layer norm cells, softmax (``5 c - 1`` per query and head), GELU
    (``9`` per hidden unit), the residual adds and every multiply-accumulate
    agree gate for gate.
    """

    constructor = GPT2G(TINY)
    compiled = compile_gpt2(constructor, (LEGACY_REQUEST,))
    table = compiled.kind_table()
    ours = components(constructor, table, (LEGACY_REQUEST,))
    legacy = legacy_components(TINY, LEGACY_REQUEST)
    d, vocab = TINY.d_model, TINY.vocab
    positions, predictions = positions_of((LEGACY_REQUEST,)), predictions_of((LEGACY_REQUEST,))
    kinds = by_name(constructor, table)
    narrows = kinds["narrow"][0] if "narrow" in kinds else None
    assert narrows is None  # narrow is inline in the dot kinds: count it through them
    inner_products = sum(copies for name, (copies, _) in kinds.items() if name.startswith("dot(") or name == "score")

    assert legacy == {
        "embedding": 40,
        "layer_norm": 1450,
        "attention": 6280,
        "mlp": 6560,
        "residual": 160,
        "lm_head": 495,
        "argmax": 3,
        "writes": 733,
        "lookups": 40,
    }
    assert ours == {
        "embedding": 975,
        "layer_norm": 1334,
        "attention": 6740,
        "mlp": 6800,
        "residual": 160,
        "lm_head": 528,
        "argmax": 90,
    }
    # the deltas, exactly
    assert ours["embedding"] == legacy["embedding"] + positions * (vocab + 2 * vocab * d)
    assert legacy["lookups"] == positions * d  # a free row lookup per coordinate
    assert ours["layer_norm"] == legacy["layer_norm"] - (positions - predictions) * (7 * d + 2)
    writes = {"attention": 4 * d * positions * TINY.layers + TINY.heads * TINY.layers * keys_of((LEGACY_REQUEST,))}
    writes["attention"] += d * positions * TINY.layers  # the mix
    writes["mlp"] = (TINY.d_ff + d) * positions * TINY.layers
    writes["lm_head"] = vocab * predictions
    assert sum(writes.values()) == legacy["writes"] == 733
    assert inner_products == legacy["writes"] + d * positions  # plus the embedding dots the legacy looks up
    assert ours["attention"] == legacy["attention"] + writes["attention"]
    assert ours["mlp"] == legacy["mlp"] + writes["mlp"]
    assert ours["lm_head"] == legacy["lm_head"] + writes["lm_head"]
    assert ours["residual"] == legacy["residual"]
    assert ours["argmax"] == 3 * (vocab - 1) * predictions and legacy["argmax"] == predictions
    computed = table.n - table.input_count - table.weight_count
    assert computed == sum(ours.values()) == 16_627
    legacy_primitives = sum(v for k, v in legacy.items() if k not in ("writes", "lookups"))
    assert legacy_primitives == 14_988
    assert computed == (
        legacy_primitives
        + legacy["writes"]
        + positions * (vocab + 2 * vocab * d)
        - (positions - predictions) * (7 * d + 2)
        + (3 * (vocab - 1) - 1) * predictions
    )


# -- GPT-2 Small ----------------------------------------------------------------------------


SMALL_REQUESTS = tuple(Request(tuple((7 * i + 3 * r) % 50257 for i in range(32)), 32) for r in range(3))


@pytest.fixture(scope="module")
def small() -> tuple[GPT2G, Compiled]:
    constructor = GPT2G(GPT2Shape.small())
    return constructor, compile_gpt2(constructor, SMALL_REQUESTS)


def test_gpt2_small_compiles_and_the_table_has_the_documented_numbers(small) -> None:
    constructor, compiled = small
    index = compiled.index
    table = compiled.kind_table()

    description, _ = constructor(SMALL_REQUESTS, b"")
    assert len(description) == 773_587  # count-one ranges carry stride 0
    assert index.n == 54_589_340_261
    assert index.weight_count == 124_490_072 and index.input_count == 96
    assert index.replay_units.count == 4 and index.verification_unit_count == 176_763_749
    assert len(table.rows) == 348
    assert index.n == constructor.gate_budget(SMALL_REQUESTS)["total"] + index.input_count + index.weight_count
    cells = [row for row in table.rows if row.role == VERIFICATION]
    assert max(row.out_bits for row in cells) == 32 and max(cut_bits(row) for row in cells) == 32
    computed = table.n - table.input_count - table.weight_count
    narrow = sum(row.copies * row.size for row in cells if cut_bits(row) <= 16) - table.input_count - table.weight_count
    assert narrow / computed == pytest.approx(0.999675, abs=1e-6)
    assert sum(row.copies * row.size for row in cells if cut_bits(row) == 32) == 17_695_200
    requests = [row for row in table.rows if row.role == REPLAY and row.out_count > 0]
    assert len(requests) == 1 and requests[0].copies == 3
    assert requests[0].closed and requests[0].out_bits == requests[0].reach_bits == 32 * 16
    assert requests[0].size == 18_154_950_063


def test_gpt2_small_bound_and_cost_fold_over_the_table(small) -> None:
    """Three requests are too few for ``U`` to bite below the whole output (``(1 - q)^3 > eta``); the fold still runs."""

    _, compiled = small
    policy = VerificationPolicy(Fraction(1, 2), Fraction(1, 10))

    result = bound(compiled, policy, ETA, BoundOptions(knapsack=False))
    assert result.capped and result.bits == result.out_bits == 3 * 32 * 16
    assert result.laplace_bits > result.out_bits
    full = bound(compiled, VerificationPolicy(1, 1), ETA, BoundOptions(knapsack=False))
    assert full.bits == 0.0
    strict = bound(compiled, VerificationPolicy(1, Fraction(3, 4)), ETA, BoundOptions(knapsack=False))
    assert not strict.capped and 1000 < strict.bits < 1100  # 1023.96: two of three requests' worth
    expected = cost(compiled, policy)
    assert expected.total > 0 and expected.weights == 124_490_072
    assert expected.boundary == 96 + 96  # the prompts and the tokens, at h = 1


def test_a_thousand_gpt2_small_requests_are_one_repeat_and_the_bound_bites() -> None:
    """1000 requests of one shape: one ``repeat`` (one output run), 18.2 T gates, ``U`` well below the output."""

    constructor = GPT2G(GPT2Shape.small())
    requests = tuple(Request(tuple((11 * i + r) % 50257 for i in range(32)), 32) for r in range(1000))
    compiled = compile_gpt2(constructor, requests)
    table = compiled.kind_table()

    assert table.n == 18_155_074_553_072 and table.replay_unit_count == 1001
    (request,) = [row for row in table.rows if row.role == REPLAY and row.out_count > 0]
    assert request.copies == 1000 and request.out_bits == 512
    root = {row.kind: row for row in table.rows}[table.root]
    assert root.out_count == 32_000 and root.out_bits == 512_000
    assert constructor.output_layout(requests)[:3] == ((0, 0), (0, 1), (0, 2))
    assert compiled.circuit.outputs[32] == compiled.circuit.Out(compiled.index.replay_units.unit(2))[0]
    loose = bound(compiled, VerificationPolicy(Fraction(1, 10), Fraction(1, 10)), ETA, BoundOptions(knapsack=False))
    firm = bound(compiled, VerificationPolicy(Fraction(1, 2), Fraction(1, 10)), ETA, BoundOptions(knapsack=False))
    assert not loose.capped and 208_000 < loose.bits < 208_500  # 40.7% of the output
    assert not firm.capped and 35_500 < firm.bits < 36_000  # 7.0%
