"""Fault declarations through a proof backend: the declared VU's obligation is the vacuous
program, its openings are still in the witness, and an undeclared corruption still fails
the proof.  Also the statement bytes of a declared obligation decode to what was encoded."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

from veritor.core import Compiled, VerificationPolicy
from veritor.protocol import (
    VerificationCode,
    VerifierParameters,
    assignment_replay,
    honest_declare,
    run_protocol,
)
from veritor.protocol.proofs import (
    DECLARED_KIND,
    DECLARED_PROGRAM,
    decode_statement,
    encode_statement,
)

from .conftest import RECORDING_BACKEND, RecordingBackend

FAULTY_UNIT = 10
OTHER_UNIT = 12
CHECK_EVERYTHING = VerificationPolicy(1, 1)


def corrupt(compiled: Compiled, honest_values: dict[int, object], *units: int):
    values = dict(honest_values)
    for unit in units:
        address = compiled.index.verification_unit(unit).interval[-1]
        values[address] = values[address] ^ 1  # type: ignore[operator]
    return values, tuple(values[a] for a in compiled.circuit.outputs)


def test_a_declared_unit_is_proved_under_the_vacuous_program(
    compiled: Compiled, expect, honest_values, model_weights, recording: RecordingBackend
) -> None:
    values, outputs = corrupt(compiled, honest_values, FAULTY_UNIT)
    expectation = expect(
        CHECK_EVERYTHING,
        backend=RECORDING_BACKEND,
        parameters=VerifierParameters(max_capacity=None, max_faults=1),
    )
    expectation = replace(expectation, claimed_outputs=outputs)
    run = run_protocol(
        compiled,
        expectation,
        values,
        replay=assignment_replay(values),
        weight_tree=model_weights[1],
        backend=recording,
        declare=honest_declare(compiled),
    )
    assert run.report.code is VerificationCode.ACCEPTED
    assert run.transcript is not None
    assert run.transcript.interiors.declarations == (FAULTY_UNIT,)
    assert run.transcript.evidence.units == () and len(run.transcript.evidence.proofs) == 1
    (statement, witness), = recording.proved
    assert statement in recording.verified
    assert DECLARED_PROGRAM in statement.kinds
    declared = [o for o in statement.obligations if o.kind == DECLARED_KIND]
    assert [o.unit for o in declared] == [FAULTY_UNIT]
    (obligation,) = declared
    assert obligation.gates == () and obligation.inputs == ()
    checked = next(o for o in statement.obligations if o.unit == FAULTY_UNIT + 1)
    assert len(obligation.positions) == len(checked.positions)  # same openings, no relation
    slot = statement.obligations.index(obligation)
    assert len(witness.obligations[slot]) == len(obligation.positions)
    assert decode_statement(encode_statement(statement)) == statement


def test_a_proof_cannot_hide_an_undeclared_corruption(
    compiled: Compiled, expect, honest_values, model_weights, recording: RecordingBackend
) -> None:
    values, outputs = corrupt(compiled, honest_values, FAULTY_UNIT, OTHER_UNIT)
    expectation = expect(
        CHECK_EVERYTHING,
        backend=RECORDING_BACKEND,
        parameters=VerifierParameters(max_capacity=None, max_faults=2),
    )
    expectation = replace(expectation, claimed_outputs=outputs)
    run = run_protocol(
        compiled,
        expectation,
        values,
        replay=assignment_replay(values),
        weight_tree=model_weights[1],
        backend=recording,
        declare=lambda unit, _values: (FAULTY_UNIT,) if unit == 2 else (),
    )
    assert run.report.code is VerificationCode.RELATION_REJECTED
    assert run.transcript is None


def test_sampling_some_units_still_derives_declared_obligations_on_both_sides(
    compiled: Compiled, expect, honest_values, model_weights, recording: RecordingBackend
) -> None:
    """Under partial sampling both parties derive the same statement, declared or not."""

    values, outputs = corrupt(compiled, honest_values, FAULTY_UNIT)
    expectation = expect(
        VerificationPolicy(1, Fraction(1, 2)),
        backend=RECORDING_BACKEND,
        parameters=VerifierParameters(max_capacity=None, max_faults=1),
    )
    expectation = replace(expectation, claimed_outputs=outputs)
    run = run_protocol(
        compiled,
        expectation,
        values,
        replay=assignment_replay(values),
        weight_tree=model_weights[1],
        backend=recording,
        declare=honest_declare(compiled),
    )
    assert run.report.accepted and run.transcript is not None
    assert run.transcript.interiors.declarations == (FAULTY_UNIT,)
    for (proved, _), verified in zip(recording.proved, recording.verified, strict=True):
        assert proved == verified
