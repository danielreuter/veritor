"""Truncated requests (S7): the length as advice, the absent slots as blank check outputs."""

from __future__ import annotations

import pytest

from veritor.constructors import (
    LMShape,
    RequestsG,
    random_parameters,
    reference_generate,
)
from veritor.constructors.schedule import Request
from veritor.constructors.tracer import TracerError
from veritor.constructors.truncation import (
    TruncatedRequestsG,
    field_width,
    pack_fields,
    unpack_fields,
)
from veritor.core import VerificationPolicy, as_kind_table, make_isa_gate_set
from veritor.core.description import VERIFICATION
from veritor.protocol import (
    VerificationCode,
    VerifierParameters,
    assignment_replay,
    commit_weights,
    make_expectation,
    run_protocol,
)
from veritor.research import Compile

WIDTH = 16
SHAPE = LMShape(vocab=8, d_model=4, heads=1, layers=1, context=32, width=WIDTH)
REQUESTS = (Request((1, 2, 3), 4), Request((5, 6), 4), Request((7,), 3))
LENGTHS = (2, 4, 1)
GATE_SET = make_isa_gate_set(WIDTH)


def test_the_advice_is_the_lengths_in_exact_bits() -> None:
    g = TruncatedRequestsG(SHAPE)
    a = g.advice(REQUESTS, LENGTHS)
    assert g.widths(REQUESTS) == (2, 2, 2) and g.advice_bits(
        REQUESTS
    ) == 6 == g.advice_bits(REQUESTS, a)
    assert len(a) == 1 and a == bytes(
        [0b01_11_00_00]
    )  # lengths - 1 = 1, 3, 0 then zero padding
    assert g.lengths(REQUESTS, a) == LENGTHS
    for bad in (a + b"\0", b"", bytes([a[0] | 1])):
        with pytest.raises(TracerError, match="malformed length advice"):
            g.lengths(REQUESTS, bad)
    with pytest.raises(TracerError, match="more than max_new"):
        g.lengths(
            REQUESTS, bytes([0b01_11_11_00])
        )  # 4 tokens for a request that asked for 3
    with pytest.raises(ValueError):
        g.advice(REQUESTS, (2, 5, 1))
    assert unpack_fields(pack_fields([3, 0, 5], [2, 1, 3]), [2, 1, 3]) == (3, 0, 5)
    assert [field_width(n) for n in (1, 2, 3, 4, 5, 8, 9)] == [0, 1, 2, 2, 3, 3, 4]


def test_the_absent_slots_are_blank_check_outputs_and_the_steps_are_the_asked_for_run() -> (
    None
):
    parameters = random_parameters(SHAPE, 5)
    weights = parameters.flatten()
    g = TruncatedRequestsG(SHAPE)
    a = g.advice(REQUESTS, LENGTHS)
    compilation = Compile(g, REQUESTS, a, GATE_SET, max_advice_bits=8)
    assert compilation.advice_bits == 6

    # every request keeps its max_new slots: the t tokens, then blanks the root's checks fix at vocab
    layout = g.output_layout(REQUESTS, a)
    assert len(layout) == sum(r.max_new for r in REQUESTS) == 11
    blanks = g.blank_positions(REQUESTS, a)
    assert len(blanks) == 11 - sum(LENGTHS) == 4
    assert list(compilation.compiled.check_values()) == [
        (i, SHAPE.vocab) for i in blanks
    ]
    circuit = compilation.compiled.circuit
    values = circuit.evaluate(compilation.inputs, weights)
    outputs = [values[address] for address in circuit.outputs]
    reference = reference_generate(SHAPE, parameters, REQUESTS)
    for ordinal, (r, position) in enumerate(layout):
        expected = reference[r][position] if position < LENGTHS[r] else SHAPE.vocab
        assert outputs[ordinal] == expected, (ordinal, r, position)

    # the blanks carry no bits: out_bits and Bound are those of the run that asked for t tokens
    table = as_kind_table(compilation.compiled)
    root = next(row for row in table.rows if row.kind == table.root)
    assert root.out_bits == WIDTH * sum(LENGTHS)
    asked = Compile(
        RequestsG(SHAPE), g.truncated(REQUESTS, a), b"", GATE_SET, max_advice_bits=0
    )
    asked_table = as_kind_table(asked.compiled)
    asked_root = next(row for row in asked_table.rows if row.kind == asked_table.root)
    assert asked_root.out_bits == root.out_bits
    # the same verification kinds (the request RUs differ: theirs hold the blank cells); one gate per blank
    verification = {row.kind for row in table.rows if row.role == VERIFICATION}
    assert verification == {
        row.kind for row in asked_table.rows if row.role == VERIFICATION
    }
    assert circuit.n == asked.compiled.circuit.n + len(blanks)


def test_a_request_that_streamed_everything_has_no_blanks_and_a_lie_names_another_circuit() -> (
    None
):
    g = TruncatedRequestsG(SHAPE)
    full = g.advice(REQUESTS, (4, 4, 3))
    compilation = Compile(g, REQUESTS, full, GATE_SET, max_advice_bits=8)
    assert (
        list(compilation.compiled.check_values()) == []
        and g.blank_positions(REQUESTS, full) == ()
    )
    asked = Compile(RequestsG(SHAPE), REQUESTS, b"", GATE_SET, max_advice_bits=0)
    assert compilation.compiled.circuit.n == asked.compiled.circuit.n
    honest = Compile(
        g, REQUESTS, g.advice(REQUESTS, LENGTHS), GATE_SET, max_advice_bits=8
    )
    lying = Compile(
        g, REQUESTS, g.advice(REQUESTS, (3, 4, 1)), GATE_SET, max_advice_bits=8
    )
    assert (
        honest.compiled.digest != lying.compiled.digest != compilation.compiled.digest
    )


def test_the_verifier_rejects_a_token_streamed_into_a_blank_slot() -> None:
    parameters = random_parameters(SHAPE, 5)
    weights = parameters.flatten()
    g = TruncatedRequestsG(SHAPE)
    a = g.advice(REQUESTS, LENGTHS)
    compilation = Compile(g, REQUESTS, a, GATE_SET, max_advice_bits=8)
    circuit = compilation.compiled.circuit
    values = dict(enumerate(circuit.evaluate(compilation.inputs, weights)))
    outputs = tuple(values[address] for address in circuit.outputs)
    kappa, tree = commit_weights(GATE_SET, weights)
    parameters_ = VerifierParameters(max_capacity=1 << 20, max_advice_bits=6)

    def run(values_, outputs_):
        expectation = make_expectation(
            compilation,
            VerificationPolicy(1, 1),
            outputs_,
            parameters=parameters_,
            weights=kappa,
        )
        return run_protocol(
            compilation.compiled,
            expectation,
            values_,
            replay=assignment_replay(values_),
            weight_tree=tree,
        )

    assert run(values, outputs).report.accepted
    # a server that computed a token into an absent slot: the opened blank is not vocab
    blank = g.blank_positions(REQUESTS, a)[0]
    moved = dict(values)
    moved[circuit.outputs[blank]] = 3
    assert run(moved, outputs).report.code is VerificationCode.CHECK_MISMATCH
    # ... and claiming it as computed fails at admission for the same reason
    claimed = list(outputs)
    claimed[blank] = 3
    assert run(moved, tuple(claimed)).report.code is VerificationCode.CHECK_MISMATCH


def test_requests_of_one_truncated_kind_are_one_repeat_with_a_check_run_per_copy() -> (
    None
):
    parameters = random_parameters(SHAPE, 5)
    weights = parameters.flatten()
    g = TruncatedRequestsG(SHAPE)
    requests = (
        Request((1, 2, 3), 4),
        Request((5, 6, 7), 4),
        Request((2, 2, 2), 4),
        Request((7,), 3),
    )
    lengths = (2, 2, 2, 3)
    a = g.advice(requests, lengths)
    assert g.groups(requests, a) == (((3, 2, 0, 4), (0, 1, 2)), ((1, 3, 0, 3), (3,)))
    compilation = Compile(g, requests, a, GATE_SET, max_advice_bits=8)
    circuit = compilation.compiled.circuit
    values = circuit.evaluate(compilation.inputs, weights)
    outputs = [values[address] for address in circuit.outputs]
    reference = reference_generate(SHAPE, parameters, requests)
    for ordinal, (r, position) in enumerate(g.output_layout(requests, a)):
        assert outputs[ordinal] == (
            reference[r][position] if position < lengths[r] else SHAPE.vocab
        )
    blanks = g.blank_positions(requests, a)
    assert len(blanks) == 6 and list(compilation.compiled.check_values()) == [
        (i, SHAPE.vocab) for i in blanks
    ]


def test_the_shape_must_leave_room_for_the_blank_word() -> None:
    with pytest.raises(ValueError, match="vocab"):
        TruncatedRequestsG(
            LMShape(vocab=8, d_model=4, heads=1, layers=1, context=16, width=3)
        )
