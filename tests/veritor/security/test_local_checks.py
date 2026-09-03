"""Component 4: local checks (session._check_unit, circuit.check_gate / decode).

Every gate of a sampled verification unit is recomputed from the values
opened under their owners (its inputs) and every opened output (a declared
output) must equal what was recomputed; input gates are compared with the
public inputs exhaustively at the boundary; weights are accepted only as
kappa_W's leaves; non-canonical or out-of-range encodings are INVALID_VALUE;
the evidence must open exactly the required addresses -- inputs and
outputs, never an internal gate -- in order.
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
from veritor.protocol.session import _Layout


def non_source_addresses(model) -> list[int]:
    return [a for a in range(model.circuit.n) if not model.circuit[a].is_source]


@pytest.mark.parametrize("offset", range(8))
def test_every_non_source_gate_of_a_sampled_unit_is_checked(model, offset):
    """Corrupting any one gate (outputs propagated consistently) names that gate."""

    address = non_source_addresses(model)[offset]
    values, outputs = model.corrupt(
        {address: (model.values[address] + 1) % (1 << model.width)}
    )
    # every gate downstream is consistent with the corrupted value; the outputs are claimed as is
    expectation = model.expectation(claimed_outputs=outputs)
    report = model.run(expectation, values).report
    assert report.code == VerificationCode.RELATION_REJECTED
    assert f"address {address} " in report.detail
    assert model.unit_of(address) in report.sampled_verification_units


@pytest.mark.parametrize("marks", ["whole", "wide"])
@pytest.mark.parametrize("cell", [(0, 0), (1, 1)])
def test_corrupting_an_internal_gate_is_caught_at_the_output_that_reads_it(
    sec, marks, cell
):
    """A ``mul`` internal to its unit is never opened: the recomputation stands for it.

    The prover's transcript holds a wrong product and the sum that follows
    from it.  The unit's check recomputes the product from the opened ``x``
    and ``w``, adds ``x_next`` and finds the opened sum wrong: the rejection
    names the *output*, the only committed value the corruption reaches.
    """

    model = sec.Model(2, 2, wide_units=marks == "wide", split_cells=False)
    mul, add = model.cell_addresses(*cell)
    assert model.unit_of(mul) == model.unit_of(add)
    values, outputs = model.corrupt({mul: (model.values[mul] + 1) % (1 << model.width)})
    assert values[add] != model.values[add]
    report = model.run(model.expectation(claimed_outputs=outputs), values).report
    assert report.code == VerificationCode.RELATION_REJECTED
    assert f"address {add} " in report.detail and f"address {mul} " not in report.detail
    assert model.unit_of(mul) in report.sampled_verification_units
    # the same corruption with the product declared (split cells): the product's own
    # unit is the one rejected, at the product's address
    split = sec.Model(2, 2)
    values, outputs = split.corrupt({mul: (split.values[mul] + 1) % (1 << split.width)})
    report = split.run(split.expectation(claimed_outputs=outputs), values).report
    assert report.code == VerificationCode.RELATION_REJECTED
    assert f"address {mul} " in report.detail


def test_a_recomputation_disagreeing_with_an_opened_output_is_relation_rejected(
    model, sec
):
    """Honest inputs, a wrong declared output: whether interior (a product) or boundary (a sum)."""

    for address in (model.interior_addresses[0], model.hidden_boundary_addresses[0]):
        forged = dict(model.values)
        forged[address] = (forged[address] + 1) % (1 << model.width)
        # the downstream values stay honest, so the claim is the honest output: only the
        # sampled check of the unit computing (or reading) the address can notice
        report = model.run(model.expectation(), forged).report
        assert report.code == VerificationCode.RELATION_REJECTED
        assert f"address {address} " in report.detail
        assert model.unit_of(address) in report.sampled_verification_units
        # nothing sampled: the forged value is committed and accepted
        report = model.run(
            model.expectation(sec.VerificationPolicy(1, 0)), forged
        ).report
        assert report.accepted and report.sampled_verification_units == ()


@pytest.mark.parametrize("marks", ["whole", "wide"])
def test_evidence_opening_an_internal_gate_instead_of_an_output_is_rejected(sec, marks):
    """No domain holds an internal gate, so no opening of it authenticates or is even required.

    The prover swaps a unit's output opening for one at its internal ``mul``
    (the address of the ``add`` is what the verifier requires there), or
    appends such an opening: COVERAGE_MISMATCH before any hash is checked.
    An internal gate under the *boundary's* address with the output's leaf
    is INVALID_OPENING: the leaf binds its position.
    """

    model = sec.Model(2, 2, wide_units=marks == "wide", split_cells=False)
    mul, add = model.cell_addresses(0, 0)
    unit = model.unit_of(add)
    layout = _Layout(model.compiled)
    assert (BOUNDARY_OWNER, add) in layout.required(unit)
    assert all(address != mul for _, address in layout.required(unit))

    def find(message: EvidenceMessage, selected: tuple[int, ...]) -> tuple[int, int]:
        batch = selected.index(unit)
        slot = next(
            i for i, (_, address) in enumerate(layout.required(unit)) if address == add
        )
        return batch, slot

    class Swapping(sec.TamperingProver):
        def evidence(self, challenge):
            message = super().evidence(challenge)
            batch, slot = find(message, challenge.selected)
            units = list(message.units)
            openings = list(units[batch])
            item = openings[slot]
            if self.mode == "swap":
                openings[slot] = Opening(mul, item.value, item.path)
            elif self.mode == "append":
                openings.append(Opening(mul, item.value, item.path))
            else:  # the internal gate's value shown under the output's address
                encoded = self._layout.circuit.encode(mul, self._values[mul])
                openings[slot] = Opening(add, encoded, item.path)
            units[batch] = tuple(openings)
            return EvidenceMessage(tuple(units))

    for mode, code in (
        ("swap", VerificationCode.COVERAGE_MISMATCH),
        ("append", VerificationCode.COVERAGE_MISMATCH),
        ("relabel", VerificationCode.INVALID_OPENING),
    ):
        Swapping.mode = mode
        run = model.run(model.expectation(), model.values, prover=Swapping)
        assert run.report.code == code, mode


@pytest.mark.parametrize("rank", range(2))
def test_wrong_input_value_is_caught_at_the_boundary_before_any_sampling(
    model, sec, rank
):
    """All gates satisfied over a wrong ``x``: PUBLIC_IO_MISMATCH at the boundary, even with q = 0."""

    inputs = list(model.inputs)
    inputs[rank] = (inputs[rank] + 1) % (1 << model.width)
    values = sec.evaluate(model.compiled, inputs, model.weights)
    outputs = model.outputs_of(values)
    for policy in (sec.CHECK_EVERYTHING, sec.HALVES, sec.VerificationPolicy(0, 0)):
        expectation = model.expectation(
            policy, claimed_outputs=outputs
        )  # the verifier's x
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
    readers = {
        model.unit_of(a)
        for a in non_source_addresses(model)
        if weight in model.circuit[a].args
    }
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
            assert not (
                sampled & readers
            )  # the false output passed: no reader was sampled
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
            model.expectation(),
            model.values,
            prover=sec.TamperingProver,
            raw_leaves={address: padded},
        )
        assert run.report.code == VerificationCode.INVALID_VALUE
        assert f"address {address} " in run.report.detail
    # a public input encoded non-canonically differs bytewise from the header's encoding
    address = model.circuit.inputs[0]
    padded = b"\0" + model.circuit.encode(address, model.values[address])
    run = model.run(
        model.expectation(),
        model.values,
        prover=sec.TamperingProver,
        raw_leaves={address: padded},
    )
    assert run.report.code == VerificationCode.PUBLIC_IO_MISMATCH


def test_value_outside_the_gate_width_is_invalid_value(narrow_model, sec):
    """A 4-bit circuit: a one-byte payload above 15 is rejected by the codec."""

    model = narrow_model
    address = model.interior_addresses[0]
    assert len(model.circuit.encode(address, model.values[address])) == 1
    run = model.run(
        model.expectation(),
        model.values,
        prover=sec.TamperingProver,
        raw_leaves={address: b"\x1f"},
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

    for rewrite in (
        reversed_batch,
        dropped_opening,
        dropped_batch,
        extra_batch,
        foreign_address,
    ):
        run = model.run(
            model.expectation(),
            model.values,
            prover=sec.TamperingProver,
            rewrite_evidence=rewrite,
        )
        assert run.report.code == VerificationCode.COVERAGE_MISMATCH, rewrite.__name__


def test_boundary_must_open_exactly_the_public_io_in_order(model, sec):
    def reordered(message: BoundaryMessage) -> BoundaryMessage:
        return BoundaryMessage(message.commitment, tuple(reversed(message.io_openings)))

    def dropped(message: BoundaryMessage) -> BoundaryMessage:
        return BoundaryMessage(message.commitment, message.io_openings[1:])

    def hidden_extra(message: BoundaryMessage) -> BoundaryMessage:
        return BoundaryMessage(
            message.commitment, (*message.io_openings, message.io_openings[0])
        )

    for rewrite in (reordered, dropped, hidden_extra):
        run = model.run(
            model.expectation(),
            model.values,
            prover=sec.TamperingProver,
            rewrite_boundary=rewrite,
        )
        assert run.report.code == VerificationCode.COVERAGE_MISMATCH, rewrite.__name__


def test_gate_arguments_are_the_owners_committed_values_not_the_provers_claims(
    model, sec
):
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
    assert (
        honest.code == VerificationCode.RELATION_REJECTED
        and f"address {mul} " in honest.detail
    )
