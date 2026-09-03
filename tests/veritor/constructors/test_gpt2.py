"""GPT-2's structure over the pinned gate set: gate counts, marks, determinism, the legacy DAG, GPT-2 Small.

``GPT2G`` writes GPT-2's shape over ``make_pinned_gate_set()``: tensor-core
chains for every dot product and explicit IEEE fp32 sequences for the rest.
At a tiny shape the compiled table is checked gate for gate against the
closed forms (``gate_budget``), every gate is in exactly one replay unit
(RU) and one verification unit (VU), the request and the weights are
closed, no VU is wider than the argmax block's ``(best, index)`` pair, and
the dot products and MACs reconcile with the legacy ``circuit_cut_analysis``
GPT-2 DAG built at the same shape.  At ``GPT2Shape.small()`` a three-request
run (prompt 32, 32 generated tokens) compiles in well under a second and its
table carries the numbers of ``docs/gpt2-structure.md``.  The values the
circuit computes are tested in ``test_gpt2_reference.py`` (a tiny model
end to end, including the protocol) and ``test_gpt2_capture.py`` (VUs of
the RTX 4090 run).
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
from veritor.constructors.gpt2 import STEP, padded
from veritor.core import Compiled, VerificationPolicy
from veritor.core.description import REPLAY, VERIFICATION
from veritor.core.index import KindTable

TINY = GPT2Shape(layers=2, d_model=32, heads=2, d_ff=64, vocab=11, context=8)
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
    forwards = predictions_of(requests)
    return {
        "embedding": gates_of(kinds, "embed"),
        "constants": 2 * forwards,  # the two widened scalars per forward, ``widen_cell`` copies outside ``embed``
        "layer_norm": gates_of(kinds, "layer_norm"),
        "attention": gates_of(kinds, f"matvec({d},{d},True,True,False)", f"matvec({d},{d},True,False,False)")
        + gates_with_prefix(kinds, "attend_head("),
        "mlp": gates_of(kinds, f"matvec({d},{f},True,False,False)", f"matvec({f},{d},True,False,False)", "gelu_cell"),
        "residual": gates_of(kinds, "add_cell") - positions_of(requests) * d,  # minus the embedding adds
        "lm_head": gates_of(kinds, f"matvec({d},{vocab},False,False,True)"),
        "argmax": gates_with_prefix(kinds, "argmax_block(") + gates_with_prefix(kinds, "argmax_top("),
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
    assert (small.width, small.acc_width, small.d_head, small.argmax_block) == (16, 32, 64, 64)
    # 124,439,808 parameters (tied embedding), plus the token table and the three BF16 scalars
    assert small.weight_count == 124_439_808 + 50257 + 3
    assert dict(small.layout())["wte"] == 50257 * 768 and dict(small.layout())["layer11.w_proj"] == 3072 * 768
    assert small.vocab_padded == 50272 and small.scalar_words() == {"n": 0x4440, "scale": 0x3E00, "zero": 0}
    with pytest.raises(ValueError, match="multiple of heads"):
        GPT2Shape(layers=1, d_model=48, heads=5, d_ff=16, vocab=5, context=4)
    with pytest.raises(ValueError, match="positive"):
        GPT2Shape(layers=0, d_model=16, heads=1, d_ff=16, vocab=5, context=4)
    with pytest.raises(ValueError, match=f"multiples of {STEP}"):
        GPT2Shape(layers=1, d_model=8, heads=1, d_ff=16, vocab=5, context=4)
    with pytest.raises(ValueError, match=f"multiples of {STEP}"):
        GPT2Shape(layers=1, d_model=32, heads=4, d_ff=16, vocab=5, context=4)  # d_head 8
    with pytest.raises(ValueError, match="vocab <= 2"):
        GPT2Shape(layers=1, d_model=16, heads=1, d_ff=16, vocab=1 << 17, context=4)
    with pytest.raises(ValueError, match="not exact in BF16"):
        GPT2Shape(layers=1, d_model=16 * 257, heads=1, d_ff=16, vocab=5, context=4)  # 4112 = 2^12 (1 + 2^-8)
    assert padded(50257) == 50272 and padded(16) == 16 and padded(17) == 32


def test_the_flat_weight_layout_is_documented_and_consistent() -> None:
    layout = TINY.layout()
    names = [name for name, _ in layout]
    assert names[:2] == ["wte", "wpe"] and names[-6:-3] == ["lnf_g", "lnf_b", "tokens"]
    assert names[-3:] == ["n", "scale", "zero"]
    assert names[2:18] == [f"layer0.{field}" for field, _ in TINY.layer_layout()]
    assert sum(count for _, count in layout) == TINY.weight_count
    d, f = TINY.d_model, TINY.d_ff
    assert TINY.layer_weights == 2 * d + 3 * (d * d + d) + (d * d + d) + 2 * d + (d * f + f) + (f * d + d)
    assert TINY.state_size(3) == 2 * 2 * 3 * d
    assert TINY.scalar_words() == {"n": 0x4200, "scale": 0x3E00, "zero": 0}  # 32.0, 0.125, 0.0


# -- gate counts ------------------------------------------------------------------------


def test_the_closed_forms_account_for_every_computed_gate(run) -> None:
    constructor, compiled = run
    index = compiled.index

    budget = constructor.gate_budget(REQUESTS)
    assert budget["total"] == sum(v for k, v in budget.items() if k != "total")
    assert budget["total"] == sum(gate_budget(TINY, len(r.prompt), r.max_new)["total"] for r in REQUESTS)
    assert index.n == budget["total"] + index.input_count + index.weight_count
    assert index.input_count == sum(len(r.prompt) for r in REQUESTS) == 5
    assert index.weight_count == TINY.weight_count == 17_774
    assert index.n == 48_748 and budget["total"] == 30_969


def test_dot_products_are_tensor_core_chains_of_k_over_16_steps(run) -> None:
    """Every inner product is a ``dot(k, biased, rounded)`` VU of ``k/16`` ``tc_dot16`` steps plus its bias and rounding.

    The MACs of a run are ``16 x`` the tensor-core steps; the classical
    count (``4 d^2 + 2 d d_ff`` per layer and position, ``2 d c`` for
    attention at context ``c``, ``vocab d`` per prediction) is recovered
    from the unpadded chain lengths, and the embedding's one-hot adds
    ``vocab_padded d`` per position that a gather would not.
    """

    constructor, compiled = run
    kinds = by_name(constructor, compiled.kind_table())
    d, f, dh, heads, layers = TINY.d_model, TINY.d_ff, TINY.d_head, TINY.heads, TINY.layers
    positions, predictions, keys = positions_of(REQUESTS), predictions_of(REQUESTS), keys_of(REQUESTS)

    # (copies, gates): the widened bias, k/16 steps, the rounding
    assert kinds[f"dot({d},True,True)"] == (3 * d * positions * layers, 1 + d // STEP + 1)  # q, k, v
    assert kinds[f"dot({d},True,False)"] == ((d + f) * positions * layers, 1 + d // STEP)  # o and fc (length d)
    assert kinds[f"dot({f},True,False)"] == (d * positions * layers, 1 + f // STEP)  # proj (length d_ff)
    assert kinds[f"dot({dh},False,False)"] == (layers * heads * keys + positions * d, dh // STEP)  # scores + embedding
    assert kinds[f"dot({dh},False,True)"] == (layers * heads * dh * positions, 1 + 1)  # the mix: ceil(c/16) = 1 step
    assert kinds[f"dot({d},False,False)"] == (predictions * TINY.vocab, d // STEP)  # the tied head
    inner_products = sum(copies for name, (copies, _) in kinds.items() if name.startswith("dot("))
    assert inner_products == positions * layers * (4 * d + f + d) + layers * heads * keys + layers * positions * d + (
        positions * d + predictions * TINY.vocab
    )
    # unpadded MACs: the classical count plus the one-hot
    macs = (
        positions * layers * (4 * d * d + 2 * d * f)
        + layers * 2 * d * keys
        + predictions * TINY.vocab * d
        + positions * TINY.vocab_padded * d
    )
    steps = 0
    for name, (copies, size) in kinds.items():
        if name.startswith("dot("):
            k, biased, rounded = name[4:-1].split(",")
            steps += copies * int(k) // STEP
            assert size == int(k) // STEP + int(biased == "True") + int(rounded == "True")
    padded_macs = STEP * steps
    padding = layers * heads * dh * (positions * padded(1) - keys)  # the mix pads each context to 16 keys
    assert padded_macs == macs + padding
    small = GPT2Shape.small()  # d_ff = 4 d: the familiar 12 d^2 MACs per layer and token
    assert 4 * small.d_model**2 + 2 * small.d_model * small.d_ff == 12 * small.d_model**2


def test_the_components_match_the_closed_forms(run) -> None:
    constructor, compiled = run
    budget = constructor.gate_budget(REQUESTS)

    found = components(constructor, compiled.kind_table(), REQUESTS)

    assert found == {k: v for k, v in budget.items() if k != "total"}
    kinds = by_name(constructor, compiled.kind_table())
    d = TINY.d_model
    normalised = 2 * TINY.layers * positions_of(REQUESTS) + predictions_of(REQUESTS)
    assert kinds["ln_mean"] == (normalised, d)  # d - 1 adds and the division
    assert kinds["ln_var"] == (normalised, 2 * d + 1)  # d squares, d - 1 adds, the division, ln_rstd
    assert kinds["sub_cell"] == (normalised * d, 1) and kinds["ln_out"] == (normalised * d, 6)
    assert kinds["gelu_cell"] == (TINY.layers * positions_of(REQUESTS) * TINY.d_ff, 2)
    assert kinds["argmax_block(11)"] == (predictions_of(REQUESTS), 2 * (TINY.vocab - 1))  # one block covers the vocab
    assert kinds["eq_cell"] == (positions_of(REQUESTS) * TINY.vocab, 1)
    assert kinds["widen_cell"] == (2 * predictions_of(REQUESTS) + positions_of(REQUESTS) * d, 1)
    for c in range(2, 6):
        assert kinds[f"softmax_max({c})"] == kinds[f"softmax_sum({c})"]
        assert kinds[f"softmax_max({c})"][1] == c - 1
    for cell in ("scale_cell", "exp_cell", "prob_cell"):
        assert kinds[cell][0] == TINY.layers * TINY.heads * keys_of(REQUESTS)


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
    assert index.verification_unit_count == sum(row.copies for row in table.rows if row.role == VERIFICATION) == 27_675


def test_interfaces_match_enumeration_at_a_smaller_shape(check_interfaces) -> None:
    shape = GPT2Shape(layers=1, d_model=16, heads=1, d_ff=16, vocab=5, context=4, argmax_block=2)
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


def test_no_verification_unit_is_wider_than_an_argmax_block(run) -> None:
    """``kappa_V = min(out_bits, reach_bits)`` is 16, 32 or 48 bits for every computed gate.

    The paper's bottleneck claim read off the table: a rounded dot product,
    a probability, a GELU, a LayerNorm output or the argmax leave through
    one BF16 word or a token id; an fp32 dot product, a residual, a
    statistic, an exponential or a scaled score through one fp32 word; an
    argmax block through its ``(best, index)`` pair.  The one-hot's
    equalities are 16-bit words too (BF16 ``1.0`` or ``0``).
    """

    constructor, compiled = run
    table = compiled.kind_table()
    names = constructor.model.kind_names()
    cells = [row for row in table.rows if row.role == VERIFICATION]

    widths: dict[int, set[str]] = {}
    for row in cells:
        widths.setdefault(cut_bits(row), set()).add(names[row.kind].split("(")[0])
    assert widths == {
        0: {"veritor.source"},
        16: {"dot", "eq_cell", "gelu_cell", "ln_out", "prob_cell"},
        32: {
            "add_cell",
            "dot",
            "exp_cell",
            "ln_mean",
            "ln_var",
            "scale_cell",
            "softmax_max",
            "softmax_sum",
            "sub_cell",
            "widen_cell",
        },
        48: {"argmax_block"},
    }
    assert all(row.out_bits == cut_bits(row) for row in cells)  # the interface, not the reach, is the cut
    sources = table.input_count + table.weight_count
    gates_at = {k: sum(r.copies * r.size for r in cells if cut_bits(r) == k) for k in widths}
    assert sum(gates_at.values()) == table.n
    assert gates_at[0] == sources
    assert gates_at == {0: 17_779, 16: 16_576, 32: 14_293, 48: 100}
    assert gates_at[48] == predictions_of(REQUESTS) * 2 * (TINY.vocab - 1)


# -- determinism and the constructor protocol --------------------------------------------


def test_compilation_is_deterministic(run) -> None:
    constructor, compiled = run

    again = GPT2G(TINY)
    description, inputs = again(REQUESTS, b"")
    assert (description, inputs) == constructor(REQUESTS, b"")
    recompiled = Compiler(again.gate_set).compile(description, inputs)
    assert recompiled.digest == compiled.digest and recompiled.index.digest == compiled.index.digest
    assert again.digest == constructor.digest
    other = GPT2Shape(layers=2, d_model=32, heads=1, d_ff=64, vocab=11, context=8)
    assert GPT2G(other).digest != constructor.digest
    assert constructor.manifest["gate_set"] == constructor.gate_set.digest  # the semantics are part of the identity
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


def legacy_inner_products(shape: GPT2Shape, request: Request) -> tuple[dict[str, int], dict[str, int]]:
    """The legacy GPT-2 DAG at ``shape`` for one greedy request: inner products and multiplies by family.

    Families tagged ``inner-product-output`` are the legacy ``write`` nodes,
    one per inner product; the ``mul`` primitives of the same families are
    its MACs (the score family also holds the ``1/sqrt(d_head)`` scaling,
    one per score).
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
    families = ("q-projection", "k-projection", "v-projection", "output-projection", "score", "value-reduction", "expansion", "contraction", "lm-head")
    products = dict.fromkeys(families, 0)
    muls = dict.fromkeys(families, 0)
    for family in circuit.families.values():
        tags = set(family.tags)
        name = next((f for f in families if f in tags), None)
        if name is None:
            continue
        if family.primitive is None and "inner-product-output" in tags:
            products[name] += family.count
        elif family.primitive == "mul":
            muls[name] += family.count
    return products, muls


def test_the_legacy_explicit_dag_agrees_on_inner_products_and_macs() -> None:
    """Both structures at ``TINY`` for one request (prompt 3, 3 tokens): the same 2,653 inner products and 84,896 MACs.

    Gates are not comparable (the legacy DAG has one ``mul`` and one ``add``
    per MAC; here a ``tc_dot16`` step is 16 MACs), so the cross-check is
    on what both count: every dot product of the projections, scores, mix
    and LM head, and the multiplies inside them.  The deltas are the
    embedding (a free row lookup there, a one-hot times ``wte`` here: ``d``
    dots of ``vocab_padded`` per position) and the tensor-core padding of
    the value mix to 16 keys, both exact.
    """

    constructor = GPT2G(TINY)
    compiled = compile_gpt2(constructor, (LEGACY_REQUEST,))
    kinds = by_name(constructor, compiled.kind_table())
    products, muls = legacy_inner_products(TINY, LEGACY_REQUEST)
    d, dh, heads, layers, vocab = TINY.d_model, TINY.d_head, TINY.heads, TINY.layers, TINY.vocab
    positions, predictions, keys = positions_of((LEGACY_REQUEST,)), predictions_of((LEGACY_REQUEST,)), keys_of((LEGACY_REQUEST,))

    assert products == {
        "q-projection": 320,
        "k-projection": 320,
        "v-projection": 320,
        "output-projection": 320,
        "score": 60,
        "value-reduction": 320,
        "expansion": 640,
        "contraction": 320,
        "lm-head": 33,
    }
    assert muls["score"] == 60 * dh + 60  # the MACs and the scaling
    assert kinds[f"dot({d},True,True)"][0] == products["q-projection"] + products["k-projection"] + products["v-projection"]
    assert kinds[f"dot({d},True,False)"][0] == products["output-projection"] + products["expansion"]
    assert kinds[f"dot({TINY.d_ff},True,False)"][0] == products["contraction"]
    assert kinds[f"dot({dh},False,False)"][0] == products["score"] + positions * d  # plus the embedding dots
    assert kinds[f"dot({dh},False,True)"][0] == products["value-reduction"]
    assert kinds[f"dot({d},False,False)"][0] == products["lm-head"]
    ours = sum(copies for name, (copies, _) in kinds.items() if name.startswith("dot("))
    assert ours == sum(products.values()) + positions * d == 2_653 + positions * d
    # MACs: the legacy multiplies of the dot families; ours from the unpadded chain lengths
    legacy_macs = sum(muls.values()) - products["score"]
    assert legacy_macs == 84_896
    projections = (kinds[f"dot({d},True,True)"][0] + kinds[f"dot({d},True,False)"][0]) * d
    projections += kinds[f"dot({TINY.d_ff},True,False)"][0] * TINY.d_ff
    scores = products["score"] * dh
    mix = layers * heads * dh * keys
    head = predictions * vocab * d
    assert projections + scores + mix + head == legacy_macs
    assert kinds["scale_cell"][0] == products["score"]  # the scaling is a VU of its own here
    # what the tensor cores actually execute: 16 MACs per step, the mix padded to 16 keys, the one-hot embedding
    steps = sum(copies * int(name[4:-1].split(",")[0]) // STEP for name, (copies, _) in kinds.items() if name.startswith("dot("))
    assert STEP * steps == legacy_macs + layers * heads * dh * (positions * STEP - keys) + positions * d * TINY.vocab_padded


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
    assert len(description) == 1_090_248
    assert index.n == 1_924_349_881
    assert index.weight_count == 124_490_068 and index.input_count == 96
    assert index.replay_units.count == 4 and index.verification_unit_count == 177_855_025
    assert len(table.rows) == 291
    assert index.n == constructor.gate_budget(SMALL_REQUESTS)["total"] + index.input_count + index.weight_count
    cells = [row for row in table.rows if row.role == VERIFICATION]
    assert max(row.out_bits for row in cells) == 48 and max(cut_bits(row) for row in cells) == 48
    computed = table.n - table.input_count - table.weight_count
    at = {k: sum(row.copies * row.size for row in cells if cut_bits(row) == k) for k in (16, 32, 48)}
    assert at == {16: 313_998_477, 32: 1_476_362_808, 48: 9_498_432}
    assert sum(at.values()) == computed
    assert at[32] / computed == pytest.approx(0.820265, abs=1e-6)
    requests = [row for row in table.rows if row.role == REPLAY and row.out_count > 0]
    assert len(requests) == 1 and requests[0].copies == 3
    assert requests[0].closed and requests[0].out_bits == requests[0].reach_bits == 32 * 16
    assert requests[0].size == 599_953_271
    budget = constructor.gate_budget(SMALL_REQUESTS)
    assert budget["embedding"] == 465_856_461 and budget["lm_head"] == 231_584_256 and budget["mlp"] == 691_504_128


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
    assert not strict.capped and 1200 < strict.bits < 1250  # 1224.6: two and a half requests' worth
    expected = cost(compiled, policy)
    assert expected.total > 0 and expected.weights == 124_490_068
    assert expected.boundary == 96 + 96  # the prompts and the tokens, at h = 1


def test_a_thousand_gpt2_small_requests_are_one_repeat_and_the_bound_bites() -> None:
    """1000 requests of one shape: one ``repeat`` (one output run), 600 G gates, ``U`` well below the output."""

    constructor = GPT2G(GPT2Shape.small())
    requests = tuple(Request(tuple((11 * i + r) % 50257 for i in range(32)), 32) for r in range(1000))
    compiled = compile_gpt2(constructor, requests)
    table = compiled.kind_table()

    assert table.n == 600_077_761_068 and table.replay_unit_count == 1001
    (request,) = [row for row in table.rows if row.role == REPLAY and row.out_count > 0]
    assert request.copies == 1000 and request.out_bits == 512
    root = {row.kind: row for row in table.rows}[table.root]
    assert root.out_count == 32_000 and root.out_bits == 512_000
    assert constructor.output_layout(requests)[:3] == ((0, 0), (0, 1), (0, 2))
    assert compiled.circuit.outputs[32] == compiled.circuit.Out(compiled.index.replay_units.unit(2))[0]
    loose = bound(compiled, VerificationPolicy(Fraction(1, 10), Fraction(1, 10)), ETA, BoundOptions(knapsack=False))
    firm = bound(compiled, VerificationPolicy(Fraction(1, 2), Fraction(1, 10)), ETA, BoundOptions(knapsack=False))
    assert not loose.capped and 230_000 < loose.bits < 231_000  # 45.1% of the output
    assert not firm.capped and 40_000 < firm.bits < 41_000  # 7.9%
