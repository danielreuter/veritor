"""Component 9: transcript verification (verify.py).

A third party holding the expectation (seeds) and ``(C, I)`` recomputes
every challenge from the recorded messages; anything the prover could alter
after the fact is caught.  Every message of a recorded transcript is altered
in turn and the verdict code checked.
"""

from __future__ import annotations

import pytest

from veritor.protocol import VerificationCode

OTHER_ROOT = "ab" * 32
OTHER_SEED = "cd" * 32


def interior_index(document: dict) -> int:
    """The first interior commitment with positions (the source unit has none)."""

    for index, commitment in enumerate(document["interiors"]["commitments"]):
        if commitment["count"]:
            return index
    raise AssertionError("no interior positions")


def opening_at(document: dict, model, owner: str) -> dict:
    """An evidence opening under the boundary, kappa_W or an interior root."""

    wanted = {
        "boundary": set(model.hidden_boundary_addresses),
        "weight": set(model.circuit.weights),
        "interior": set(model.interior_addresses),
    }[owner]
    for batch in document["evidence"]["units"]:
        for opening in batch:
            if opening["position"] in wanted:
                return opening
    raise AssertionError(owner)


MUTATIONS = {
    # the header: everything the verifier fixed itself
    "header.session_id": (
        lambda d, m: d["header"].__setitem__("session_id", "ff"),
        VerificationCode.EXPECTATION_MISMATCH,
    ),
    "header.compiled_digest": (
        lambda d, m: d["header"].__setitem__("compiled_digest", OTHER_ROOT),
        VerificationCode.EXPECTATION_MISMATCH,
    ),
    "header.policy": (
        lambda d, m: d["header"]["policy"].__setitem__("q", [1, 3]),
        VerificationCode.EXPECTATION_MISMATCH,
    ),
    "header.eta": (
        lambda d, m: d["header"].__setitem__("eta", [1, 3]),
        VerificationCode.EXPECTATION_MISMATCH,
    ),
    "header.public_inputs": (
        lambda d, m: d["header"]["public_inputs"].__setitem__(0, "00"),
        VerificationCode.EXPECTATION_MISMATCH,
    ),
    "header.claimed_outputs": (
        lambda d, m: d["header"]["claimed_outputs"].__setitem__(0, "00"),
        VerificationCode.EXPECTATION_MISMATCH,
    ),
    "header.weights": (
        lambda d, m: d["header"]["weights"].__setitem__("root", OTHER_ROOT),
        VerificationCode.EXPECTATION_MISMATCH,
    ),
    # the boundary message
    "boundary.commitment.root": (
        lambda d, m: d["boundary"]["commitment"].__setitem__("root", OTHER_ROOT),
        VerificationCode.INVALID_OPENING,
    ),
    "boundary.commitment.count": (
        lambda d, m: d["boundary"]["commitment"].__setitem__("count", 7),
        VerificationCode.INVALID_COMMITMENT,
    ),
    "boundary.io_openings.value": (
        lambda d, m: d["boundary"]["io_openings"][0].__setitem__("value", "00"),
        VerificationCode.INVALID_OPENING,
    ),
    "boundary.io_openings.path": (
        lambda d, m: d["boundary"]["io_openings"][0]["path"].__setitem__(0, OTHER_ROOT),
        VerificationCode.INVALID_OPENING,
    ),
    "boundary.io_openings.position": (
        lambda d, m: d["boundary"]["io_openings"][0].__setitem__("position", 1),
        VerificationCode.COVERAGE_MISMATCH,
    ),
    "boundary.io_openings.dropped": (
        lambda d, m: d["boundary"].__setitem__("io_openings", d["boundary"]["io_openings"][1:]),
        VerificationCode.COVERAGE_MISMATCH,
    ),
    # the replay challenge
    "replay_challenge.seed": (
        lambda d, m: d["replay_challenge"].__setitem__("seed", OTHER_SEED),
        VerificationCode.EXPECTATION_MISMATCH,
    ),
    "replay_challenge.selected": (
        lambda d, m: d["replay_challenge"].__setitem__("selected", [0]),
        VerificationCode.CHALLENGE_MISMATCH,
    ),
    # the interiors: with s = 1 the sample is everything whatever the roots, so the openings fail
    # (with s < 1 the changed root changes T first: test_altered_interior_root_changes_the_sample)
    "interiors.root": (
        lambda d, m: d["interiors"]["commitments"][interior_index(d)].__setitem__("root", OTHER_ROOT),
        VerificationCode.INVALID_OPENING,
    ),
    "interiors.count": (
        lambda d, m: d["interiors"]["commitments"][interior_index(d)].__setitem__("count", 3),
        VerificationCode.INVALID_COMMITMENT,
    ),
    "interiors.dropped": (
        lambda d, m: d["interiors"].__setitem__("commitments", d["interiors"]["commitments"][1:]),
        VerificationCode.COVERAGE_MISMATCH,
    ),
    # the sample challenge
    "sample_challenge.seed": (
        lambda d, m: d["sample_challenge"].__setitem__("seed", OTHER_SEED),
        VerificationCode.EXPECTATION_MISMATCH,
    ),
    "sample_challenge.selected": (
        lambda d, m: d["sample_challenge"].__setitem__("selected", d["sample_challenge"]["selected"][:-1]),
        VerificationCode.CHALLENGE_MISMATCH,
    ),
    # the evidence, under each owner
    "evidence.boundary_value": (
        lambda d, m: opening_at(d, m, "boundary").__setitem__("value", "00"),
        VerificationCode.INVALID_OPENING,
    ),
    "evidence.weight_value": (
        lambda d, m: opening_at(d, m, "weight").__setitem__("value", "00"),
        VerificationCode.INVALID_OPENING,
    ),
    "evidence.interior_value": (
        lambda d, m: opening_at(d, m, "interior").__setitem__("value", "00"),
        VerificationCode.INVALID_OPENING,
    ),
    "evidence.interior_path": (
        lambda d, m: opening_at(d, m, "interior")["path"].__setitem__(0, OTHER_ROOT),
        VerificationCode.INVALID_OPENING,
    ),
    "evidence.position": (
        lambda d, m: opening_at(d, m, "interior").__setitem__("position", 0),
        VerificationCode.COVERAGE_MISMATCH,
    ),
    "evidence.dropped_batch": (
        lambda d, m: d["evidence"].__setitem__("units", d["evidence"]["units"][1:]),
        VerificationCode.COVERAGE_MISMATCH,
    ),
    "evidence.dropped_opening": (
        lambda d, m: d["evidence"]["units"][-1].pop(),
        VerificationCode.COVERAGE_MISMATCH,
    ),
}


@pytest.mark.parametrize("label", sorted(MUTATIONS))
def test_altering_a_recorded_message_is_caught_with_the_expected_code(honest_run, model, sec, label):
    run, expectation = honest_run
    mutate, code = MUTATIONS[label]
    data = sec.mutate_transcript(run.transcript, lambda document: mutate(document, model))
    assert data != sec.encode_transcript(run.transcript)
    report = model.verify(data, expectation)
    assert report.code == code, (label, report)


def test_altered_interior_root_changes_the_sample(model, sec):
    """With ``s < 1`` the sample challenge depends on the interiors: a new root, a new ``T``."""

    codes = set()
    for trial in range(8):
        expectation = model.expectation(
            sec.VerificationPolicy(1, sec.HALF), q_seed=sec.seed("i/q", trial), s_seed=sec.seed("i/s", trial)
        )
        run = model.run(expectation, model.values)
        assert run.report.accepted

        def swap_root(document: dict) -> None:
            document["interiors"]["commitments"][interior_index(document)]["root"] = OTHER_ROOT

        codes.add(model.verify(sec.mutate_transcript(run.transcript, swap_root), expectation).code)
    assert VerificationCode.CHALLENGE_MISMATCH in codes
    assert codes <= {VerificationCode.CHALLENGE_MISMATCH, VerificationCode.INVALID_OPENING}


def test_the_recorded_transcript_verifies_only_under_its_own_expectation(honest_run, model, sec):
    run, expectation = honest_run
    assert model.verify(run.transcript, expectation).code == VerificationCode.ACCEPTED
    for other in (
        model.expectation(q_seed=sec.seed("other-q")),
        model.expectation(s_seed=sec.seed("other-s")),
        model.expectation(session_id=b"other-session"),
        model.expectation(sec.HALVES),
        model.expectation(claimed_outputs=[(y + 1) % 256 for y in model.outputs]),
    ):
        report = model.verify(run.transcript, other)
        assert report.code == VerificationCode.EXPECTATION_MISMATCH, report


def test_a_rejected_interaction_leaves_no_transcript(model):
    mul = model.cell_addresses(0, 0)[0]
    values, outputs = model.corrupt({mul: 0})
    run = model.run(model.expectation(claimed_outputs=outputs), values)
    assert run.report.code == VerificationCode.RELATION_REJECTED and run.transcript is None


def test_transcript_verdict_equals_the_interactive_verdict(honest_run, model, sec):
    """The pure verifier replays the interaction: same sampled units, same verdict."""

    run, expectation = honest_run
    report = model.verify(run.transcript, expectation)
    assert report == run.report
    # ... and a dishonest transcript that the interactive verifier accepted (an error set that
    # escaped sampling) is accepted offline as well, with the same sampled units
    mul = model.cell_addresses(1, 1)[0]
    values, outputs = model.corrupt({mul: 0})
    for trial in range(64):
        escaped = model.expectation(
            sec.HALVES, claimed_outputs=outputs, q_seed=sec.seed("e/q", trial), s_seed=sec.seed("e/s", trial)
        )
        run = model.run(escaped, values)
        if run.report.accepted:
            assert model.verify(run.transcript, escaped) == run.report
            assert model.unit_of(mul) not in run.report.sampled_verification_units
            break
    else:  # pragma: no cover - probability 4**-64
        raise AssertionError("an error in one unit escapes sampling at q = s = 1/2 three times in four")
