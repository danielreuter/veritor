"""M6 fault declarations on the matmul fixture.

An honest server whose run holds one silently corrupted VU (a bit flip in a
dot product's output word, streamed as it happened) is rejected as soon as
that VU is sampled -- unless the header admits declarations (``max_faults >=
1``) and the server declares the VU when it replays the opened RU.  The
declarations are bound into the s-challenge, capped by ``max_faults``,
validated against the opened RUs and the index, priced into ``Bound`` at
admission, and never let an *undeclared* corruption through.  With
``max_faults = 0`` and no declarations the wire bytes are exactly what they
were before the mechanism existed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from fractions import Fraction

import pytest

from veritor.analysis.bound import bound
from veritor.analysis.faults import fault_allowance_bits, unit_fault_bits
from veritor.core import Compiled, VerificationPolicy
from veritor.protocol import (
    Header,
    InteriorMessage,
    Opening,
    ProtocolError,
    ProverSession,
    Reject,
    VerificationCode,
    VerifierParameters,
    VerifierSession,
    assignment_replay,
    decode_transcript,
    encode_transcript,
    honest_declare,
    run_protocol,
    self_check,
    verify_transcript,
)
from veritor.protocol.phases import interior_phase
from veritor.protocol.proofs import DECLARED_KIND, DECLARED_PROGRAM, derive_obligations
from veritor.protocol.wire import MalformedTranscript, NoncanonicalTranscript

FAULTY_UNIT = 10
"""The first VU with a relation: ``mul, mul, add`` in RU 2 (RUs 0 and 1 hold the sources)."""
OTHER_UNIT = 12
"""A second such VU, in RU 3."""
SOURCE_UNIT = 0
"""An ``in`` gate's VU: nothing to declare."""

BASELINE_TRANSCRIPT_SHA256 = "7eb5a9da19f296b56e9a077c73653dc5cd77eda0daf4dbb471ed639b0ab08997"
"""SHA-256 of the fixture's honest transcript as the code before M6 encoded it.

Re-pinned when the interior moved to VU-output granularity (protocol v7: evidence
opens a unit's inputs and outputs only, interior domain tag v2)."""


def faults(max_faults: int = 0) -> VerifierParameters:
    return VerifierParameters(max_capacity=None, max_faults=max_faults)


def flip_output(compiled: Compiled, values: dict[int, object], unit: int, bit: int = 0) -> int:
    """Flip ``bit`` of VU ``unit``'s output word in ``values`` (in place); return the address.

    The matmul VUs' outputs are circuit outputs, so this is a fault the users saw.
    """

    address = compiled.index.verification_unit(unit).interval[-1]
    assert compiled.circuit[address].op == "add"
    values[address] = values[address] ^ (1 << bit)  # type: ignore[operator]
    return address


def faulty_run(compiled: Compiled, honest_values: dict[int, object], *units: int) -> tuple[dict[int, object], tuple[int, ...]]:
    """The assignment an honest server holds after faults at ``units``, and its outputs."""

    values = dict(honest_values)
    for unit in units:
        flip_output(compiled, values, unit)
    return values, tuple(values[a] for a in compiled.circuit.outputs)  # type: ignore[misc]


# -- messages and wire ----------------------------------------------------------------


def test_header_binds_max_faults_and_omits_the_default(compiled, expect) -> None:
    verifier = VerifierSession(expect(), compiled)
    header = verifier.header
    assert header.max_faults == 0
    same = Header(
        header.session_id,
        header.compiled_digest,
        header.constructor,
        header.advice,
        header.policy,
        header.eta,
        header.public_inputs,
        header.claimed_outputs,
        header.weights,
        header.backend,
        0,
    )
    assert same.digest == header.digest
    admitting = VerifierSession(expect(parameters=faults(2)), compiled).header
    assert admitting.max_faults == 2 and admitting.digest != header.digest
    with pytest.raises(ProtocolError):
        replace(header, max_faults=-1)


def test_declarations_are_sorted_unique_and_bound_into_the_s_challenge(compiled, expect) -> None:
    verifier = VerifierSession(expect(), compiled)
    commitments = ()
    plain = InteriorMessage(commitments)
    assert "declarations" not in plain.manifest
    declared = InteriorMessage(commitments, (3, 7))
    assert declared.manifest["declarations"] == [3, 7]
    previous = verifier.header.digest
    assert interior_phase(previous, plain) != interior_phase(previous, declared)
    assert interior_phase(previous, declared) != interior_phase(previous, InteriorMessage(commitments, (3,)))
    for bad in ((7, 3), (3, 3), (-1,), [3]):
        with pytest.raises(ProtocolError):
            InteriorMessage(commitments, bad)  # type: ignore[arg-type]


def test_wire_round_trips_declarations_and_rejects_noncanonical_defaults(
    compiled, honest_values, expect
) -> None:
    values, outputs = faulty_run(compiled, honest_values, FAULTY_UNIT)
    expectation = expect(parameters=faults(2), claimed_outputs=outputs)
    run = run_protocol(
        compiled,
        expectation,
        values,
        replay=assignment_replay(values),
        declare=honest_declare(compiled),
    )
    assert run.report.accepted and run.transcript is not None
    data = encode_transcript(run.transcript)
    assert b'"max_faults":2' in data and b'"declarations":[%d]' % FAULTY_UNIT in data
    assert decode_transcript(data) == run.transcript
    assert verify_transcript(data, expectation, compiled).accepted

    document = json.loads(data)
    document["header"]["max_faults"] = 0
    with pytest.raises(NoncanonicalTranscript):
        decode_transcript(json.dumps(document, sort_keys=True, separators=(",", ":")).encode())
    document = json.loads(data)
    document["interiors"]["declarations"] = []
    with pytest.raises(NoncanonicalTranscript):
        decode_transcript(json.dumps(document, sort_keys=True, separators=(",", ":")).encode())
    for header_value in (-1, "2", 2.0):
        document = json.loads(data)
        document["header"]["max_faults"] = header_value
        with pytest.raises(MalformedTranscript):
            decode_transcript(json.dumps(document, sort_keys=True, separators=(",", ":")).encode())
    for declarations in ([12, 10], [10, 10], [-1], "10"):
        document = json.loads(data)
        document["interiors"]["declarations"] = declarations
        with pytest.raises(MalformedTranscript):
            decode_transcript(json.dumps(document, sort_keys=True, separators=(",", ":")).encode())


def test_transcripts_without_faults_are_byte_identical_to_before(compiled, honest_values, expect) -> None:
    """The fixture's honest transcript under fixed seeds, as the pre-M6 code encoded it."""

    run = run_protocol(compiled, expect(), honest_values)
    assert run.transcript is not None
    data = encode_transcript(run.transcript)
    assert b"max_faults" not in data and b"declarations" not in data
    assert hashlib.sha256(data).hexdigest() == BASELINE_TRANSCRIPT_SHA256


# -- the honest server with a fault ---------------------------------------------------


def test_self_check_finds_exactly_the_faulty_unit(compiled, honest_values) -> None:
    values, _ = faulty_run(compiled, honest_values, FAULTY_UNIT)
    index = compiled.index
    replay_unit = index.verification_unit(FAULTY_UNIT).replay_unit
    assert replay_unit is not None
    assert self_check(compiled, replay_unit, values) == (FAULTY_UNIT,)
    assert all(
        self_check(compiled, unit, values) == ()
        for unit in range(index.replay_units.count)
        if unit != replay_unit
    )
    assert self_check(compiled, replay_unit, honest_values) == ()


def test_faulty_run_is_rejected_at_zero_faults_and_accepted_when_declared(
    compiled, honest_values, expect
) -> None:
    values, outputs = faulty_run(compiled, honest_values, FAULTY_UNIT)
    address = compiled.index.verification_unit(FAULTY_UNIT).interval[-1]

    rejected = run_protocol(
        compiled,
        expect(claimed_outputs=outputs),
        values,
        replay=assignment_replay(values),
        declare=honest_declare(compiled),
    )
    assert rejected.report.code is VerificationCode.FAULTS_EXCEEDED
    silent = run_protocol(
        compiled, expect(claimed_outputs=outputs), values, replay=assignment_replay(values)
    )
    assert silent.report.code is VerificationCode.RELATION_REJECTED
    assert f"address {address}" in silent.report.detail

    for max_faults in (1, 3):
        expectation = expect(parameters=faults(max_faults), claimed_outputs=outputs)
        accepted = run_protocol(
            compiled,
            expectation,
            values,
            replay=assignment_replay(values),
            declare=honest_declare(compiled),
        )
        assert accepted.report.code is VerificationCode.ACCEPTED
        assert accepted.transcript is not None
        assert accepted.transcript.interiors.declarations == (FAULTY_UNIT,)
        assert FAULTY_UNIT in accepted.report.sampled_verification_units
        assert accepted.transcript.header.max_faults == max_faults


def test_declared_unit_is_obliged_under_the_vacuous_program(compiled, honest_values, expect) -> None:
    values, outputs = faulty_run(compiled, honest_values, FAULTY_UNIT)
    verifier = VerifierSession(expect(parameters=faults(1), claimed_outputs=outputs), compiled)
    prover = ProverSession(
        compiled,
        verifier.header,
        values,
        replay=assignment_replay(values),
        weight_tree=_weight_tree(compiled, values),
        declare=honest_declare(compiled),
    )
    replay_challenge = verifier.receive_boundary(prover.boundary())
    sample_challenge = verifier.receive_interiors(prover.interiors(replay_challenge))
    assert verifier.declared_verification_units == (FAULTY_UNIT,)
    # the accepted roots and the address layout, as the verifier holds them
    commitments = verifier._commitments
    layout = verifier._layout
    checked, kinds = derive_obligations(layout, verifier.header, commitments, sample_challenge.selected)
    skipped, kinds_with_declared = derive_obligations(
        layout, verifier.header, commitments, sample_challenge.selected, {FAULTY_UNIT}
    )
    assert DECLARED_PROGRAM in kinds_with_declared and DECLARED_PROGRAM not in kinds
    for plain, declared in zip(checked, skipped, strict=True):
        if plain.unit != FAULTY_UNIT:
            assert plain == declared
            continue
        assert declared.kind == DECLARED_KIND and declared.outputs == () and declared.inputs == ()
        assert declared.positions == plain.positions and declared.commitments == plain.commitments
        assert plain.outputs and plain.kind != DECLARED_KIND
    assert verifier.receive_evidence(prover.evidence(sample_challenge)).accepted


def _weight_tree(compiled, values):
    from veritor.protocol import commit_weights

    return commit_weights(
        compiled.circuit.gate_set, [values[address] for address in compiled.circuit.weights]
    )[1]


# -- what the verifier refuses ----------------------------------------------------------


def test_declaring_more_than_max_faults_is_rejected(compiled, honest_values, expect) -> None:
    values, outputs = faulty_run(compiled, honest_values, FAULTY_UNIT, OTHER_UNIT)
    both = run_protocol(
        compiled,
        expect(parameters=faults(1), claimed_outputs=outputs),
        values,
        replay=assignment_replay(values),
        declare=honest_declare(compiled),
    )
    assert both.report.code is VerificationCode.FAULTS_EXCEEDED
    assert "2 fault declarations exceed max_faults 1" in both.report.detail
    admitted = run_protocol(
        compiled,
        expect(parameters=faults(2), claimed_outputs=outputs),
        values,
        replay=assignment_replay(values),
        declare=honest_declare(compiled),
    )
    assert admitted.report.accepted and admitted.transcript is not None
    assert admitted.transcript.interiors.declarations == (FAULTY_UNIT, OTHER_UNIT)


@pytest.mark.parametrize(
    ("declared", "detail"),
    [
        ((16,), "does not exist"),  # the fixture has VUs 0..15
        ((10**6,), "does not exist"),
        ((SOURCE_UNIT,), "no relation to disclaim"),
    ],
)
def test_invalid_declarations_are_rejected(compiled, honest_values, expect, declared, detail) -> None:
    run = run_protocol(
        compiled,
        expect(parameters=faults(4)),
        honest_values,
        declare=lambda unit, values: declared if unit == 0 else (),
    )
    assert run.report.code is VerificationCode.FAULT_DECLARATION_INVALID
    assert detail in run.report.detail
    assert run.transcript is None


def test_a_declaration_outside_the_opened_rus_is_rejected(compiled, honest_values, expect) -> None:
    """Some RUs are opened; a VU of an unopened one is not the prover's to declare."""

    verifier = VerifierSession(
        expect(VerificationPolicy(Fraction(1, 2), 1), parameters=faults(4)), compiled
    )
    prover = ProverSession(
        compiled, verifier.header, honest_values, weight_tree=_weight_tree(compiled, honest_values)
    )
    replay_challenge = verifier.receive_boundary(prover.boundary())
    opened = set(replay_challenge.selected)
    index = compiled.index
    compute = [
        unit
        for unit in range(index.replay_units.count)
        if not compiled.circuit[index.replay_units.unit(unit).interval.start].is_source
    ]
    assert opened & set(compute) and set(compute) - opened, "the seeds open some compute RUs, not all"
    outside = index.verification_units(next(u for u in compute if u not in opened)).first
    interiors = prover.interiors(replay_challenge)
    with pytest.raises(Reject) as caught:
        verifier.receive_interiors(InteriorMessage(interiors.commitments, (outside,)))
    assert caught.value.code is VerificationCode.FAULT_DECLARATION_INVALID
    assert "lies in no opened replay unit" in caught.value.detail


def test_declarations_cannot_hide_an_undeclared_corruption(compiled, honest_values, expect) -> None:
    values, outputs = faulty_run(compiled, honest_values, FAULTY_UNIT, OTHER_UNIT)
    other_address = compiled.index.verification_unit(OTHER_UNIT).interval[-1]

    hiding = run_protocol(
        compiled,
        expect(parameters=faults(2), claimed_outputs=outputs),
        values,
        replay=assignment_replay(values),
        declare=lambda unit, _values: (FAULTY_UNIT,) if unit == 2 else (),
    )
    assert hiding.report.code is VerificationCode.RELATION_REJECTED
    assert f"address {other_address}" in hiding.report.detail

    misdirected = run_protocol(
        compiled,
        expect(parameters=faults(2), claimed_outputs=outputs),
        values,
        replay=assignment_replay(values),
        declare=lambda unit, _values: (OTHER_UNIT + 1,) if unit == 3 else (),
    )
    assert misdirected.report.code is VerificationCode.RELATION_REJECTED


def test_a_declared_units_openings_are_still_authenticated(compiled, honest_values, expect) -> None:
    values, outputs = faulty_run(compiled, honest_values, FAULTY_UNIT)
    verifier = VerifierSession(expect(parameters=faults(1), claimed_outputs=outputs), compiled)
    prover = ProverSession(
        compiled,
        verifier.header,
        values,
        replay=assignment_replay(values),
        weight_tree=_weight_tree(compiled, values),
        declare=honest_declare(compiled),
    )
    replay_challenge = verifier.receive_boundary(prover.boundary())
    sample_challenge = verifier.receive_interiors(prover.interiors(replay_challenge))
    evidence = prover.evidence(sample_challenge)
    slot = sample_challenge.selected.index(FAULTY_UNIT)
    batch = evidence.units[slot]
    tampered_opening = batch[0]
    forged = Opening(
        tampered_opening.position,
        bytes((tampered_opening.value[0] ^ 1,)) + tampered_opening.value[1:],
        tampered_opening.path,
    )
    tampered = replace(
        evidence,
        units=evidence.units[:slot] + ((forged,) + batch[1:],) + evidence.units[slot + 1 :],
    )
    with pytest.raises(Reject) as caught:
        verifier.receive_evidence(tampered)
    assert caught.value.code is VerificationCode.INVALID_OPENING


# -- pricing ------------------------------------------------------------------------------


def test_bound_grows_by_the_fault_allowance_under_full_verification(compiled) -> None:
    """At ``q = 1`` the q-challenge reveals nothing, so a declaration is priced as one VU,
    ``u(1)``; at ``s < 1`` the lowered-threshold bound may be tighter still, never looser."""

    unit = unit_fault_bits(compiled)
    table = compiled.kind_table()
    assert unit == pytest.approx(8 + 4, abs=1e-3)  # W_V = 8 (one output word), |S| = 16 VUs
    full = VerificationPolicy(1, 1)
    base = bound(table, full, Fraction(1, 2**40))
    assert base.bits == 0.0
    for max_faults in (1, 2):
        widened = bound(table, full, Fraction(1, 2**40), max_faults=max_faults)
        assert widened.bits == pytest.approx(max_faults * unit, abs=1e-9)
        assert fault_allowance_bits(table, max_faults) == max_faults * unit
    half = VerificationPolicy(1, Fraction(1, 2))
    loose = bound(table, half, Fraction(1, 4))
    assert not loose.capped
    widened = bound(table, half, Fraction(1, 4), max_faults=1)
    assert loose.bits < widened.bits <= loose.bits + unit
    assert fault_allowance_bits(table, 0) == 0.0
    assert bound(table, full, 0, max_faults=8).bits == 48.0  # capped at the interface


def test_admission_prices_the_fault_allowance(compiled, honest_values, expect) -> None:
    exact = run_protocol(
        compiled,
        expect(parameters=VerifierParameters(max_capacity=0, max_faults=0)),
        honest_values,
    )
    assert exact.report.accepted
    widened = run_protocol(
        compiled,
        expect(parameters=VerifierParameters(max_capacity=0, max_faults=1)),
        honest_values,
    )
    assert widened.report.code is VerificationCode.POLICY_REJECTED
    assert "exceeding U_max 0" in widened.report.detail
    admitted = run_protocol(
        compiled,
        expect(parameters=VerifierParameters(max_capacity=13, max_faults=1)),
        honest_values,
    )
    assert admitted.report.accepted
