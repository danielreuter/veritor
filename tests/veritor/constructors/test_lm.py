"""The toy decoder: shape and parameters, the sequential reference, and its kinds.

The single-request circuit (one pod, one slot) is the LM itself: it must
decode exactly like :func:`reference_generate`, token by token, over the
toy ISA.  Default fixture ``LMShape(vocab=8, d_model=4, heads=2, layers=1,
context=6, width=16)``, one request of prompt length 2 generating 3 tokens:
about 2,000 gates, traced and compiled in a few milliseconds.
"""

from __future__ import annotations

import random

import pytest

from veritor.compile import Compiler
from veritor.constructors import (
    ClusterG,
    LayerParameters,
    LMShape,
    Parameters,
    Request,
    ToyLM,
    TracerError,
    random_parameters,
    reference_generate,
    schedule_fcfs,
)
from veritor.constructors.lm import Decoder, argmax_token, concat, sample_token
from veritor.core import Compiled, make_isa_gate_set

SHAPE = LMShape(vocab=8, d_model=4, heads=2, layers=1, context=6, width=16)
DEEP = LMShape(vocab=8, d_model=4, heads=2, layers=2, context=6, width=16)


def single_request(shape: LMShape, request: Request, parameters: Parameters) -> tuple[int, ...]:
    """Run one request alone on a one-slot cluster and return its generated tokens."""

    constructor = ClusterG(shape, pods=1, slots=1, steps=request.max_new)
    schedule = schedule_fcfs((request,), 1, 1, request.max_new)
    description, inputs = constructor((request,), schedule.encode())
    compiled: Compiled = Compiler(make_isa_gate_set(shape.width)).compile(description, inputs)
    values = compiled.circuit.evaluate(inputs, parameters.flatten())
    assert constructor.output_layout((request,), schedule) == tuple((0, g) for g in range(request.max_new))
    return tuple(values[address] for address in compiled.circuit.outputs)


def test_shape_derives_head_and_hidden_sizes_and_the_weight_count() -> None:
    assert (SHAPE.d_head, SHAPE.hidden) == (2, 8)
    # E, 4 square matrices, W_1 and W_2, U, the constant table and the shift
    assert SHAPE.weight_count == 8 * 4 + 4 * 16 + 2 * 32 + 4 * 8 + 8 + 1 == 201
    assert DEEP.weight_count == 201 + 128
    assert SHAPE.state_size(3) == 2 * 1 * 3 * 4
    assert SHAPE.manifest == {"context": 6, "d_model": 4, "heads": 2, "layers": 1, "vocab": 8, "width": 16}


@pytest.mark.parametrize(
    ("fields", "match"),
    (
        ({"d_model": 5}, "multiple of heads"),
        ({"vocab": 1}, "at least 2"),
        ({"vocab": 32, "width": 4}, "vocab <= 2"),
        ({"layers": 0}, "positive integer"),
        ({"context": -1}, "positive integer"),
        ({"heads": 1.0}, "positive integer"),
    ),
)
def test_shape_rejects_bad_dimensions(fields: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        LMShape(**{"vocab": 8, "d_model": 4, "heads": 2, "layers": 1, "context": 6, "width": 16, **fields})


def test_parameters_flatten_in_weight_gate_order() -> None:
    parameters = random_parameters(SHAPE, seed=3)
    flat = parameters.flatten()

    assert len(flat) == SHAPE.weight_count
    assert flat[:4] == parameters.embedding[0] and flat[28:32] == parameters.embedding[7]
    layer = parameters.layers[0]
    assert flat[32:36] == layer.w_q[0] and flat[48:52] == layer.w_k[0]
    assert flat[96:104] == layer.w_1[0] and flat[128:132] == layer.w_2[0]
    assert flat[160:168] == parameters.unembedding[0]
    assert flat[192:200] == tuple(range(8)) == parameters.constants
    assert flat[200] == parameters.shift == 4
    assert all(0 <= value < 1 << 16 for value in flat)
    assert random_parameters(SHAPE, seed=3) == parameters
    assert random_parameters(SHAPE, seed=4) != parameters
    assert len(random_parameters(DEEP, seed=3).layers) == 2


def test_parameters_are_validated() -> None:
    good = random_parameters(SHAPE, seed=0)
    with pytest.raises(ValueError, match="embedding must be a tuple of 8 rows"):
        Parameters(SHAPE, good.embedding[:7], good.layers, good.unembedding, good.shift)
    with pytest.raises(ValueError, match=r"unembedding\[1\] must be a tuple of 8"):
        Parameters(SHAPE, good.embedding, good.layers, (good.unembedding[0], (1, 2)) + good.unembedding[2:], 0)
    with pytest.raises(ValueError, match="16-bit value"):
        Parameters(SHAPE, good.embedding, good.layers, good.unembedding, 1 << 16)
    with pytest.raises(ValueError, match="layers must be a tuple of 1"):
        Parameters(SHAPE, good.embedding, (), good.unembedding, 0)
    bad_layer = LayerParameters(good.layers[0].w_q, good.layers[0].w_k, good.layers[0].w_v, good.layers[0].w_o, good.layers[0].w_2, good.layers[0].w_1)
    with pytest.raises(ValueError, match=r"layers\[0\].w_1 must be a tuple of 4 rows"):
        Parameters(SHAPE, good.embedding, (bad_layer,), good.unembedding, 0)
    with pytest.raises(TypeError, match="LMShape"):
        random_parameters(object(), 0)  # type: ignore[arg-type]


def test_reference_is_deterministic_and_stays_in_the_vocabulary() -> None:
    parameters = random_parameters(SHAPE, seed=11)
    requests = (Request((1, 2, 3), 3), Request((5,), 2), Request((7, 0), 4))

    first = reference_generate(SHAPE, parameters, requests)
    assert first == reference_generate(SHAPE, parameters, requests)
    assert [len(tokens) for tokens in first] == [3, 2, 4]
    assert all(0 <= token < 8 for tokens in first for token in tokens)
    # a request is decoded alone: its tokens do not depend on its neighbours
    assert reference_generate(SHAPE, parameters, requests[1:2]) == first[1:2]
    with pytest.raises(ValueError, match="below vocab"):
        reference_generate(SHAPE, parameters, (Request((8,), 1),))
    with pytest.raises(ValueError, match="of the given shape"):
        reference_generate(DEEP, parameters, requests)


def test_zero_weights_generate_the_first_token_by_the_tie_rule() -> None:
    zero = tuple(tuple(0 for _ in range(4)) for _ in range(8))
    square = tuple(tuple(0 for _ in range(4)) for _ in range(4))
    up, down = tuple((0,) * 8 for _ in range(4)), tuple((0,) * 4 for _ in range(8))
    parameters = Parameters(
        SHAPE, zero, (LayerParameters(square, square, square, square, up, down),), tuple((0,) * 8 for _ in range(4)), 0
    )
    request = Request((3, 5), 3)

    assert reference_generate(SHAPE, parameters, (request,)) == ((0, 0, 0),)  # all logits tie: token 0
    assert single_request(SHAPE, request, parameters) == (0, 0, 0)


def test_the_unembedding_alone_picks_the_argmax_with_first_tie_kept() -> None:
    """With ``x`` forced constant the LM head is a lookup: the circuit's argmax is the reference's."""

    base = random_parameters(SHAPE, seed=5)
    # Zero the layer so x is the embedding row; E rows all (1, 0, 0, 0): logits are U[0].
    square = tuple((0,) * 4 for _ in range(4))
    layer = LayerParameters(square, square, square, square, tuple((0,) * 8 for _ in range(4)), tuple((0,) * 4 for _ in range(8)))
    embedding = tuple((1, 0, 0, 0) for _ in range(8))
    for row, expected in (((3, 9, 9, 1, 0, 2, 9, 4), 1), ((0, 0, 0, 0, 0, 0, 0, 7), 7), ((5, 5, 5, 5, 5, 5, 5, 5), 0)):
        unembedding = (row,) + base.unembedding[1:]
        parameters = Parameters(SHAPE, embedding, (layer,), unembedding, 0)
        request = Request((2,), 2)
        assert reference_generate(SHAPE, parameters, (request,)) == ((expected, expected),)
        assert single_request(SHAPE, request, parameters) == (expected, expected)


@pytest.mark.parametrize("shape", (SHAPE, DEEP))
@pytest.mark.parametrize("prompt", (Request((1, 2), 3), Request((4,), 4), Request((6, 7, 0), 2)))
def test_single_request_circuit_decodes_like_the_reference(shape: LMShape, prompt: Request) -> None:
    parameters = random_parameters(shape, seed=10 * shape.layers + prompt.max_new)

    assert single_request(shape, prompt, parameters) == reference_generate(shape, parameters, (prompt,))[0]


def test_kinds_are_row_sized_and_shared_across_positions() -> None:
    lm = ToyLM(SHAPE)
    prefill, decode = lm.prefill(3), lm.decode(4)

    assert prefill.input_count == SHAPE.weight_count and prefill.output_count == SHAPE.state_size(3) + 1
    assert decode.input_count == SHAPE.weight_count + 1 + SHAPE.state_size(3)
    assert decode.output_count == SHAPE.state_size(1) + 1
    assert lm.dot(4).role == "verification" and lm.dot(4, marked=False).role is None
    assert lm.dot(4).digest != lm.dot(4, marked=False).digest
    assert lm.attend_head(3).role == lm.argmax().role == lm.onehot().role == "verification"
    assert lm.attend_head(3).input_count == 2 + 2 * 3 * 2 + 1 and lm.attend_head(3).output_count == 2
    assert lm.matvec(4, 8).role is None and lm.embed_row().role is None
    assert lm.weights_unit().role == "replay" and lm.weights_unit().output_count == SHAPE.weight_count
    # tracing a second prefill of the same length adds nothing
    before = lm.tracer.definition_count
    assert lm.prefill(3) is prefill and lm.decode(4) is decode
    assert lm.tracer.definition_count == before
    with pytest.raises(TracerError, match="cached position"):
        lm.decode(1)
    with pytest.raises(TracerError, match="prompt length"):
        lm.prefill(0)
    with pytest.raises(TypeError, match="LMShape"):
        ToyLM(object())  # type: ignore[arg-type]


SAMPLED = LMShape(vocab=8, d_model=4, heads=2, layers=1, context=8, width=16, sampling=True)


def test_a_sampling_shape_has_a_bit_budget_and_two_more_constants() -> None:
    assert (SAMPLED.vocab_bits, SAMPLED.score_bits, SAMPLED.random_bits) == (3, 4, 5)
    assert SAMPLED.vocab_bits + 2 * SAMPLED.score_bits + SAMPLED.random_bits == SAMPLED.width
    assert SAMPLED.score_shift == 12 and SAMPLED.sampler_constants == (12, 5)
    assert SAMPLED.weight_count == SHAPE.weight_count + 2 and SHAPE.sampler_constants == ()
    assert SAMPLED.manifest == {**SHAPE.manifest, "context": 8, "sampling": True}
    parameters = random_parameters(SAMPLED, seed=3)
    assert parameters.flatten()[-3:] == (4, 12, 5)  # shift, score_shift, random_bits
    wide = LMShape(vocab=32, d_model=4, heads=2, layers=1, context=8, width=16, sampling=True)
    assert (wide.vocab_bits, wide.score_bits, wide.random_bits) == (5, 3, 5)
    with pytest.raises(ValueError, match="sampling needs width"):
        LMShape(vocab=8, d_model=4, heads=2, layers=1, context=8, width=5, sampling=True)
    with pytest.raises(ValueError, match="sampling must be a bool"):
        LMShape(vocab=8, d_model=4, heads=2, layers=1, context=8, width=16, sampling=1)  # type: ignore[arg-type]


def test_the_reference_sampler_draws_by_the_squared_score_cdf() -> None:
    logits = [0xF000, 0x0000, 0x8000, 0x1000, 0x0000, 0x0000, 0x0000, 0x2FFF]
    # scores 15, 0, 8, 1, 0, 0, 0, 2 -> weights 226, 1, 65, 2, 1, 1, 1, 5; total 302
    cdf = [226, 227, 292, 294, 295, 296, 297, 302]
    thresholds = [(r * 302) >> 5 for r in range(32)]
    drawn = [sample_token(SAMPLED, logits, r) for r in range(32)]
    assert all(0 <= t < 302 for t in thresholds)
    assert drawn == [sum(entry <= t for entry in cdf) for t in thresholds]  # the first j with cdf_j > t
    assert drawn.count(0) == 24 and drawn[24] == 1 and set(drawn) == {0, 1, 2, 3}  # 226/302 of the mass on token 0
    # all-zero logits: every weight is one, the token is r * vocab >> random_bits
    assert [sample_token(SAMPLED, [0] * 8, r) for r in (0, 4, 31)] == [0, 1, 7]
    with pytest.raises(ValueError, match="5-bit word"):
        sample_token(SAMPLED, logits, 32)


def test_randomness_is_checked_against_the_shape() -> None:
    with pytest.raises(ValueError, match="one random word per generated position"):
        SAMPLED.check_randomness(Request((1,), 2))
    with pytest.raises(ValueError, match="at most 5 bits"):
        SAMPLED.check_randomness(Request((1,), 1, (32,)))
    with pytest.raises(ValueError, match="argmax model takes no randomness"):
        SHAPE.check_randomness(Request((1,), 1, (3,)))
    with pytest.raises(ValueError, match="argmax model takes no randomness"):
        reference_generate(SHAPE, random_parameters(SHAPE, 0), (Request((1,), 1, (3,)),))
    with pytest.raises(ValueError, match="sampling model needs a random word"):
        Decoder(random_parameters(SAMPLED, 0)).forward(1)


@pytest.mark.parametrize("seed", (0, 1, 2))
def test_the_sample_unit_computes_the_reference_sampler(seed: int) -> None:
    """One request through a one-slot cluster with sampling: the circuit draws what Python draws."""

    rng = random.Random(seed)
    parameters = random_parameters(SAMPLED, seed=seed)
    request = Request(tuple(rng.randrange(8) for _ in range(rng.randint(1, 3))), 4, tuple(rng.randrange(32) for _ in range(4)))
    reference = reference_generate(SAMPLED, parameters, (request,))[0]

    assert single_request(SAMPLED, request, parameters) == reference
    # the same logits pass through the incremental decoder and the sampler by hand
    decoder = Decoder(parameters)
    for token in request.prompt[:-1]:
        decoder.logits(token)
    logits = decoder.logits(request.prompt[-1])
    assert sample_token(SAMPLED, logits, request.randomness[0]) == reference[0]
    assert argmax_token(logits) == max(range(8), key=lambda k: (logits[k], -k))
    # different randomness, different tokens (with overwhelming probability over 4 positions)
    other = Request(request.prompt, 4, tuple((r + 16) % 32 for r in request.randomness))
    assert reference_generate(SAMPLED, parameters, (other,))[0] != reference


def test_the_sample_kind_is_a_verification_unit_reading_the_random_word() -> None:
    lm = ToyLM(SAMPLED)
    sample = lm.sample()

    assert sample.role == "verification" and sample.input_count == 8 + 4 and sample.output_count == 1
    assert lm.prefill(2).output_count == SAMPLED.state_size(2) + 1 and lm.decode(3).output_count == SAMPLED.state_size(1) + 1
    request = Request((1, 2), 3, (0, 1, 2))
    constructor = ClusterG(SAMPLED, pods=1, slots=1, steps=3)
    schedule = schedule_fcfs((request,), 1, 1, 3)
    description, inputs = constructor((request,), schedule.encode())
    assert inputs == (1, 2, 0, 1, 2)  # the prompt, then one random word per position, step by step
    compiled = Compiler(make_isa_gate_set(16)).compile(description, inputs)
    assert compiled.circuit.input_count == 5
    kinds = {row.kind: row for row in compiled.index.kinds()}
    assert kinds[sample.digest].copies == 3 and kinds[sample.digest].out_bits == 16
    with pytest.raises(TracerError, match="one random word per generated position"):
        constructor((Request((1, 2), 3),), schedule.encode())


def test_concat_requires_consecutive_ranges() -> None:
    lm = ToyLM(SHAPE)

    @lm.tracer.definition(input_count=4, key="probe")
    def probe(v):
        a = lm.tracer.repeat(2, lm.add_pair, v[0].by(1), v[2].by(1))
        b = lm.tracer.repeat(2, lm.add_pair, v[0].by(1), v[2].by(1))
        joined = concat([a, b])
        assert (joined.start, joined.count, joined.stride) == (a.start, 4, 1)
        with pytest.raises(TracerError, match="consecutive"):
            concat([b, a])
        with pytest.raises(TracerError, match="consecutive"):
            concat([a, v[0:2]])
        return joined

    assert probe.output_count == 4
