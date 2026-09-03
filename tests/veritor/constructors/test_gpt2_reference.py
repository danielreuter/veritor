"""The pinned GPT-2 circuit *runs*: the numpy reference forward, the circuit and the protocol agree on every word.

A tiny model with random BF16 weights is decoded greedily by
:func:`veritor.constructors.gpt2_reference.forward` (vectorised numpy with
the pinned fp32 semantics, every intermediate recorded at gate
granularity); the same request is compiled by ``GPT2G``; ``address_map``
places every recorded word at its circuit address.  Then: every VU of the
run re-executes gate by gate (``Circuit.evaluate_gate`` / ``check_gate``)
to the recorded words, the framework's ``replay_unit`` of the request RU
reproduces every recorded interior word and the claimed tokens, and
``run_protocol`` accepts the honest run, rejects one flipped bit of one
dot output with ``RELATION_REJECTED`` and a wrong claim with
``PUBLIC_IO_MISMATCH``.  The GPU side of the same statement is
``test_gpt2_capture.py``.
"""

from __future__ import annotations

from fractions import Fraction
from typing import cast

import numpy as np
import pytest

from veritor.constructors import GPT2G, GPT2Shape, Request
from veritor.constructors.gpt2_reference import (
    GPT2Weights,
    NumpyOps,
    SparseValues,
    address_map,
    check_unit,
    evaluate_unit,
    forward,
    request_frames,
)
from veritor.core import VerificationPolicy
from veritor.core.description import VERIFICATION
from veritor.protocol import (
    VerificationCode,
    VerifierParameters,
    commit_weights,
    encode_transcript,
    make_expectation,
    replay_unit,
    run_protocol,
)
from veritor.research import Compile

SHAPE = GPT2Shape(layers=2, d_model=32, heads=2, d_ff=64, vocab=11, context=8)
REQUEST = Request((1, 2, 3), 3)
PARAMETERS = VerifierParameters(Fraction(1, 2**40), max_capacity=None)
SEEDS = {"session_id": b"gpt2-reference-s".ljust(16, b"\0"), "q_seed": b"q" * 32, "s_seed": b"s" * 32}


@pytest.fixture(scope="module")
def run():
    """Weights, the reference run, the compilation and the recorded words by address."""

    weights = GPT2Weights.random(SHAPE, seed=1)
    result = forward(weights, REQUEST.prompt, REQUEST.max_new, NumpyOps())
    constructor = GPT2G(SHAPE)
    compilation = Compile(constructor, (REQUEST,), b"", constructor.gate_set)
    compiled = compilation.compiled
    addresses = address_map(compiled, constructor.model, request_frames(compiled)[0], len(REQUEST.prompt), REQUEST.max_new)
    values = SparseValues(compiled, result.capture, addresses, weights.flat(), compilation.inputs)
    return weights, result, compilation, addresses, values


def test_the_reference_forward_records_every_gate_granular_tensor(run) -> None:
    weights, result, compilation, addresses, values = run
    compiled = compilation.compiled

    assert result.tokens == (3, 3, 3) and len(result.tokens) == REQUEST.max_new
    assert set(result.capture) == set(addresses)
    for name, address in addresses.items():
        assert result.capture[name].shape == address.shape, name
    assert values.recorded_count == 8_133 and compiled.circuit.n == 37_147
    # every recorded address lies in the request RU; all are computed gates but the prompt's ``in`` gates
    request_unit = compiled.index.replay_units.unit(1).interval
    for name, address in addresses.items():
        for a in address[address >= 0].reshape(-1).tolist():
            assert a in request_unit, name
            assert compiled.circuit[a].is_source == (name == "tokens" and a in addresses["tokens"][: len(REQUEST.prompt)]), name
    assert weights.flat().shape == (SHAPE.weight_count,)


def test_every_verification_unit_re_executes_to_the_recorded_words(run) -> None:
    _, _, compilation, _, values = run
    compiled = compilation.compiled
    index = compiled.index

    checked = agreeing = 0
    kinds = set()
    for u in range(index.verification_unit_count):
        node = index.verification_unit(u)
        c, a = check_unit(compiled, node, values)
        checked += c
        agreeing += a
        if c:
            kinds.add(node.kind)
    assert checked == agreeing == values.recorded_count - 3  # the three tokens are read as inputs, not checked here
    computed = {row.kind for row in compiled.kind_table().rows if row.role == VERIFICATION and row.source_weights == 0 and row.source_inputs == 0}
    assert kinds == computed  # every computed VU kind of the run has a recorded output
    # ``Circuit.evaluate`` on a VU's definition, from its inputs alone, equals the recorded output
    for u in (index.verification_units(1).first, index.verification_unit_count - 1):
        node = index.verification_unit(u)
        known = evaluate_unit(compiled, node, values)
        assert all(known[a] == values[a] for a in known if a in values)


def test_the_request_replay_reproduces_the_recorded_interior_and_the_tokens(run) -> None:
    weights, result, compilation, addresses, values = run
    compiled = compilation.compiled

    interior = replay_unit(compiled, 1, values)
    recorded = [a for a in interior if a in values]
    assert len(recorded) == values.recorded_count - 2 * 3  # minus the prompt (``in`` gates) and the tokens (outputs)
    assert all(interior[a] == values[a] for a in recorded)
    outputs = compiled.circuit.outputs
    assert tuple(values[a] for a in outputs) == result.tokens and not any(a in interior for a in outputs)
    # and the whole circuit, from the prompt and the weights alone
    assignment = compiled.circuit.evaluate(compilation.inputs, weights.flat().tolist())
    assert tuple(assignment[a] for a in outputs) == result.tokens
    assert all(assignment[a] == values[a] for a in addresses["L1.x2"].reshape(-1).tolist())


def test_the_protocol_accepts_the_run_and_rejects_one_flipped_dot_bit() -> None:
    """A smaller model (one layer, ``d`` 16, five tokens) so that ``s = 1`` opens every VU in a second."""

    shape = GPT2Shape(layers=1, d_model=16, heads=1, d_ff=32, vocab=5, context=4, argmax_block=2)
    request = Request((1,), 3)
    weights = GPT2Weights.random(shape, seed=3)
    result = forward(weights, request.prompt, request.max_new, NumpyOps())
    constructor = GPT2G(shape)
    compilation = Compile(constructor, (request,), b"", constructor.gate_set)
    compiled = compilation.compiled
    addresses = address_map(compiled, constructor.model, request_frames(compiled)[0], 1, request.max_new)
    values = SparseValues(compiled, result.capture, addresses, weights.flat(), compilation.inputs)
    kappa, tree = commit_weights(constructor.gate_set, weights.flat().tolist())
    assert result.tokens == (1, 1, 2) and compiled.circuit.n == 5_352

    def expectation(policy: VerificationPolicy, claimed: tuple[int, ...]):
        return make_expectation(compilation, policy, claimed, parameters=PARAMETERS, weights=kappa, **SEEDS)

    honest = run_protocol(compiled, expectation(VerificationPolicy(1, 1), result.tokens), values, weight_tree=tree)
    assert honest.report.code is VerificationCode.ACCEPTED
    assert honest.report.sampled_replay_units == (0, 1)  # q = 1: the weights and the request
    assert len(honest.report.sampled_verification_units) == compiled.index.verification_unit_count  # s = 1: every VU
    assert honest.transcript is not None and len(encode_transcript(honest.transcript)) > 1 << 20
    half = run_protocol(compiled, expectation(VerificationPolicy(1, Fraction(1, 2)), result.tokens), values, weight_tree=tree)
    assert half.report.code is VerificationCode.ACCEPTED
    assert 0 < len(half.report.sampled_verification_units) < len(honest.report.sampled_verification_units)

    # one flipped bit in one dot output: the attention mix at position 0, coordinate 3 (a rounded BF16 word)
    target = int(addresses["L0.mix"][0, 3])
    assert compiled.circuit[target].op == "f32_to_bf16"

    def flipped(unit: int, boundary):
        known = dict(replay_unit(compiled, unit, boundary))
        if target in known:
            known[target] = cast(int, known[target]) ^ 1
        return known

    bad = run_protocol(compiled, expectation(VerificationPolicy(1, 1), result.tokens), values, weight_tree=tree, replay=flipped)
    assert bad.report.code is VerificationCode.RELATION_REJECTED
    assert f"address {target}" in bad.report.detail and "f32_to_bf16" in bad.report.detail
    # and a wrong claim about the tokens
    wrong = tuple((t + 1) % shape.vocab for t in result.tokens)
    lie = run_protocol(compiled, expectation(VerificationPolicy(1, 1), wrong), values, weight_tree=tree)
    assert lie.report.code is VerificationCode.PUBLIC_IO_MISMATCH


def test_random_weights_are_bf16_words_in_layout_order() -> None:
    weights = GPT2Weights.random(SHAPE, seed=0)
    flat = weights.flat()
    assert flat.dtype == np.uint16 and flat.shape == (SHAPE.weight_count,)
    again = GPT2Weights.from_flat(SHAPE, flat)
    assert np.array_equal(again.flat(), flat)
    assert flat[-3:].tolist() == [0x4200, 0x3E00, 0]  # n = 32.0, scale = 0.125, zero
    assert flat[-3 - SHAPE.vocab : -3].tolist() == list(range(SHAPE.vocab))  # the token table
