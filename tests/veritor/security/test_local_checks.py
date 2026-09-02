"""Component 4: local checks (session._check_unit, circuit.check_gate / decode).

Every non-source gate of a sampled verification unit is checked against
values opened under their owners; input gates are compared with the public
inputs exhaustively at the boundary; weights are accepted only as kappa_W's
leaves; non-canonical or out-of-range encodings are INVALID_VALUE; the
evidence must open exactly the required addresses, in order.
"""

from __future__ import annotations

import pytest

from veritor.protocol import (
    BOUNDARY_OWNER,
    BoundaryMessage,
    EvidenceMessage,
    Opening,
    VerificationCode,
)


def non_source_addresses(model) -> list[int]:
    return [a for a in range(model.circuit.n) if not model.circuit[a].is_source]


@pytest.mark.parametrize("offset", range(8))
def test_every_non_source_gate_of_a_sampled_unit_is_checked(model, offset):
    """Corrupting any one gate (outputs propagated consistently) names that gate."""

    address = non_source_addresses(model)[offset]
    values, outputs = model.corrupt({address: (model.values[address] + 1) % (1 << model.width)})
    # every gate downstream is consistent with the corrupted value; the outputs are claimed as is
    expectation = model.expectation(claimed_outputs=outputs)
    report = model.run(expectation, values).report
    assert report.code == VerificationCode.RELATION_REJECTED
    assert f"address {address} " in report.detail
    assert model.unit_of(address) in report.sampled_verification_units


@pytest.mark.parametrize("rank", range(2))
def test_wrong_input_value_is_caught_at_the_boundary_before_any_sampling(model, sec, rank):
    """All gates satisfied over a wrong ``x``: PUBLIC_IO_MISMATCH at the boundary, even with q = 0."""

    inputs = list(model.inputs)
    inputs[rank] = (inputs[rank] + 1) % (1 << model.width)
    values = sec.evaluate(model.compiled, inputs, model.weights)
    outputs = model.outputs_of(values)
    for policy in (sec.CHECK_EVERYTHING, sec.HALVES, sec.VerificationPolicy(0, 0)):
        expectation = model.expectation(policy, claimed_outputs=outputs)  # the verifier's x
        report = model.run(expectation, values).report
        assert report.code == VerificationCode.PUBLIC_IO_MISMATCH
        assert report.sampled_replay_units == ()  # before J
        assert f"address {model.circuit.inputs[rank]} " in report.detail


def test_wrong_claimed_output_with_honest_values_is_caught_at_the_boundary(model):
    claimed = ((model.outputs[0] + 1) % (1 << model.width), *model.outputs[1:])
    report = model.run(model.expectation(claimed_outputs=claimed), model.values).report
    assert report.code == VerificationCode.PUBLIC_IO_MISMATCH
    assert report.sampled_replay_units == ()


def test_altered_weight_in_the_run_is_caught_only_when_a_reader_is_sampled(model, sec):
    """The run uses ``w'``; kappa_W opens ``w``.  Caught iff a gate reading it is sampled."""

    weights = list(model.weights)
    weights[0] = (weights[0] + 1) % (1 << model.width)
    values = sec.evaluate(model.compiled, model.inputs, weights)
    outputs = model.outputs_of(values)
    assert outputs != model.outputs
    weight = model.circuit.weights[0]
    readers = {model.unit_of(a) for a in non_source_addresses(model) if weight in model.circuit[a].args}
    assert len(readers) == 2

    # checking everything: the reader's relation fails against kappa_W's leaf
    report = model.run(model.expectation(claimed_outputs=outputs), values).report
    assert report.code == VerificationCode.RELATION_REJECTED

    outcomes = {}
    for trial in range(64):
        expectation = model.expectation(
            sec.HALVES,
            claimed_outputs=outputs,
            q_seed=sec.seed("weight/q", trial),
            s_seed=sec.seed("weight/s", trial),
        )
        report = model.run(expectation, values).report
        sampled = set(report.sampled_verification_units)
        if report.accepted:
            assert not (sampled & readers)  # the false output passed: no reader was sampled
            outcomes.setdefault("accepted", report)
        else:
            assert report.code == VerificationCode.RELATION_REJECTED
            assert sampled & readers
            outcomes.setdefault("rejected", report)
        if len(outcomes) == 2:
            break
    assert set(outcomes) == {"accepted", "rejected"}  # both outcomes occur


def test_noncanonical_encoding_of_a_committed_value_is_invalid_value(model, sec):
    """The leaf authenticates (it is what was committed) but does not decode."""

    for address in (model.interior_addresses[0], model.hidden_boundary_addresses[0]):
        canonical = model.circuit.encode(address, model.values[address])
        padded = b"\0" + canonical  # the same integer, one byte too long
        run = model.run(
            model.expectation(), model.values, prover=sec.TamperingProver, raw_leaves={address: padded}
        )
        assert run.report.code == VerificationCode.INVALID_VALUE
        assert f"address {address} " in run.report.detail
    # a public input encoded non-canonically differs bytewise from the header's encoding
    address = model.circuit.inputs[0]
    padded = b"\0" + model.circuit.encode(address, model.values[address])
    run = model.run(
        model.expectation(), model.values, prover=sec.TamperingProver, raw_leaves={address: padded}
    )
    assert run.report.code == VerificationCode.PUBLIC_IO_MISMATCH


def test_value_outside_the_gate_width_is_invalid_value(narrow_model, sec):
    """A 4-bit circuit: a one-byte payload above 15 is rejected by the codec."""

    model = narrow_model
    address = model.interior_addresses[0]
    assert len(model.circuit.encode(address, model.values[address])) == 1
    run = model.run(
        model.expectation(), model.values, prover=sec.TamperingProver, raw_leaves={address: b"\x1f"}
    )
    assert run.report.code == VerificationCode.INVALID_VALUE
    with pytest.raises(Exception, match="4-bit"):
        model.circuit.encode(address, 16)


def test_evidence_must_open_exactly_the_required_addresses_in_order(model, sec):
    def reversed_batch(message: EvidenceMessage) -> EvidenceMessage:
        *rest, last = message.units
        return EvidenceMessage((*rest, tuple(reversed(last))))

    def dropped_opening(message: EvidenceMessage) -> EvidenceMessage:
        *rest, last = message.units
        return EvidenceMessage((*rest, last[1:]))

    def dropped_batch(message: EvidenceMessage) -> EvidenceMessage:
        return EvidenceMessage(message.units[1:])

    def extra_batch(message: EvidenceMessage) -> EvidenceMessage:
        return EvidenceMessage((*message.units, message.units[-1]))

    def foreign_address(message: EvidenceMessage) -> EvidenceMessage:
        *rest, last = message.units
        item = last[0]
        moved = Opening(item.position + 1, item.value, item.path)
        return EvidenceMessage((*rest, (moved, *last[1:])))

    for rewrite in (reversed_batch, dropped_opening, dropped_batch, extra_batch, foreign_address):
        run = model.run(
            model.expectation(), model.values, prover=sec.TamperingProver, rewrite_evidence=rewrite
        )
        assert run.report.code == VerificationCode.COVERAGE_MISMATCH, rewrite.__name__


def test_boundary_must_open_exactly_the_public_io_in_order(model, sec):
    def reordered(message: BoundaryMessage) -> BoundaryMessage:
        return BoundaryMessage(message.commitment, tuple(reversed(message.io_openings)))

    def dropped(message: BoundaryMessage) -> BoundaryMessage:
        return BoundaryMessage(message.commitment, message.io_openings[1:])

    def hidden_extra(message: BoundaryMessage) -> BoundaryMessage:
        return BoundaryMessage(message.commitment, (*message.io_openings, message.io_openings[0]))

    for rewrite in (reordered, dropped, hidden_extra):
        run = model.run(
            model.expectation(), model.values, prover=sec.TamperingProver, rewrite_boundary=rewrite
        )
        assert run.report.code == VerificationCode.COVERAGE_MISMATCH, rewrite.__name__


def test_gate_arguments_are_the_owners_committed_values_not_the_provers_claims(model, sec):
    """A corrupted ``mul`` cannot be shown a correct argument: the leaf is the committed one."""

    mul = model.cell_addresses(0, 0)[0]
    values, outputs = model.corrupt({mul: 0})
    honest_mul = model.circuit.encode(mul, model.values[mul])

    def show_honest_argument(owner: int, opening: Opening, phase: str) -> Opening:
        if phase == "evidence" and owner != BOUNDARY_OWNER and opening.position == mul:
            return Opening(mul, honest_mul, opening.path)
        return opening

    run = model.run(
        model.expectation(claimed_outputs=outputs),
        values,
        prover=sec.TamperingProver,
        rewrite_opening=show_honest_argument,
    )
    assert run.report.code == VerificationCode.INVALID_OPENING
    honest = model.run(model.expectation(claimed_outputs=outputs), values).report
    assert honest.code == VerificationCode.RELATION_REJECTED and f"address {mul} " in honest.detail
