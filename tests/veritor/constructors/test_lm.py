"""The toy decoder: shape and parameters, the sequential reference, and its kinds.

The single-request circuit (one pod, one slot) is the LM itself: it must
decode exactly like :func:`reference_generate`, token by token, over the
toy ISA.  Default fixture ``LMShape(vocab=8, d_model=4, heads=2, layers=1,
context=6, width=16)``, one request of prompt length 2 generating 3 tokens:
about 2,000 gates, traced and compiled in a few milliseconds.
"""

from __future__ import annotations

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
from veritor.constructors.lm import concat
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
