import dataclasses
import random

import pytest

import prototypes.call_dag as dag
import prototypes.staged_replay as staged


CELL_BITS = 8


def compiled_batch(
    *,
    lengths: tuple[int, ...] = (2,),
    cut: tuple[dag.OccurrencePath, ...] = ((0, 0), (0, 1)),
):
    constructor = dag.DemoG(CELL_BITS)
    kernel = dag.Kernel(
        cell_bits=CELL_BITS,
        gates=dag.trusted_word_gates(CELL_BITS),
    )
    batch = dag.BatchInput(
        tuple(
            dag.make_demo_request(length, index + 1, CELL_BITS)
            for index, length in enumerate(lengths)
        )
    )
    construction = dag.construct(
        kernel,
        constructor,
        batch,
        b"",
        input_cells=batch.cells(),
        advice_bound_bits=0,
    )
    root = construction.load.root
    plan = kernel.derive_replay_plan(root, cut)
    return kernel, root, plan, batch.cells()


def honest_parties(*, cut=((0, 0), (0, 1))):
    kernel, root, plan, inputs = compiled_batch(cut=cut)
    assignment = staged.evaluate_assignment(kernel, root, inputs)
    claimed_outputs = tuple(assignment[position] for position in plan.root_outputs)
    client = staged.StagedClient(
        kernel=kernel,
        root=root,
        plan=plan,
        assignment=assignment,
    )
    verifier = staged.StagedVerifier(
        kernel=kernel,
        root=root,
        plan=plan,
        inputs=inputs,
        claimed_outputs=claimed_outputs,
    )
    return client, verifier, assignment


def test_indexed_commitment_binds_domain_position_and_value():
    domain = b"test-domain"
    positions = (2, 9, 15)
    tree = staged.IndexedValueTree.build(
        domain=domain,
        positions=positions,
        values={2: 3, 9: 5, 15: 7},
        cell_bits=CELL_BITS,
    )

    for position in positions:
        opening = tree.open(position)
        assert staged.verify_value_opening(
            tree.commitment,
            domain=domain,
            positions=positions,
            opening=opening,
            cell_bits=CELL_BITS,
        )

    opening = tree.open(9)
    assert not staged.verify_value_opening(
        tree.commitment,
        domain=b"another-domain",
        positions=positions,
        opening=opening,
        cell_bits=CELL_BITS,
    )
    assert not staged.verify_value_opening(
        tree.commitment,
        domain=domain,
        positions=positions,
        opening=dataclasses.replace(opening, value=6),
        cell_bits=CELL_BITS,
    )
    assert not staged.verify_value_opening(
        tree.commitment,
        domain=domain,
        positions=positions,
        opening=dataclasses.replace(opening, position=15),
        cell_bits=CELL_BITS,
    )


def test_empty_indexed_domain_has_one_canonical_commitment():
    tree = staged.IndexedValueTree.build(
        domain=b"empty-test",
        positions=(),
        values={},
        cell_bits=CELL_BITS,
    )

    staged.validate_commitment_domain(
        tree.commitment,
        domain=b"empty-test",
        positions=(),
    )
    with pytest.raises(staged.ProtocolReject, match="noncanonical"):
        staged.validate_commitment_domain(
            staged.ValueCommitment(root=bytes(32), value_count=0),
            domain=b"empty-test",
            positions=(),
        )


@pytest.mark.parametrize(
    "cut",
    [
        ((),),
        ((0,),),
        ((0, 0), (0, 1)),
        ((0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)),
    ],
)
def test_honest_staged_protocol_accepts_every_gate_for_every_cut(cut):
    client, verifier, _ = honest_parties(cut=cut)

    transcript = staged.run_staged_protocol(
        client,
        verifier,
        sampled_gates=(0, 1, 2, 3),
    )

    assert transcript.accepted
    assert transcript.sampled_gates == (0, 1, 2, 3)
    assert transcript.challenged_units == tuple(range(len(verifier.plan.units)))
    assert set(transcript.unit_commitments) == set(transcript.challenged_units)


def test_empty_sample_requires_no_unit_commitments():
    client, verifier, _ = honest_parties()

    transcript = staged.run_staged_protocol(
        client,
        verifier,
        sampled_gates=(),
    )

    assert transcript.accepted
    assert transcript.challenged_units == ()
    assert transcript.unit_commitments == {}


def test_two_stage_probability_uses_replay_then_within_unit_coins():
    probability = staged.two_stage_acceptance_probability(
        (1, 2),
        replay_probability=0.2,
        within_unit_probability=0.5,
    )

    assert probability == pytest.approx((0.8 + 0.2 * 0.5) * (0.8 + 0.2 * 0.25))
    assert staged.two_stage_acceptance_probability(
        (3,),
        replay_probability=0.2,
        within_unit_probability=0.5,
    ) == pytest.approx(0.8 + 0.2 * 0.5**3)
    assert 0.8 + 0.2 * 0.5**3 > (1 - 0.2 * 0.5) ** 3


def test_honest_two_stage_protocol_accepts_selected_gate_checks():
    client, verifier, _ = honest_parties()

    transcript = staged.run_two_stage_protocol(
        client,
        verifier,
        challenged_units=(0, 1),
        sampled_gates=(1, 2),
    )

    assert transcript.accepted
    assert transcript.challenged_units == (0, 1)
    assert transcript.sampled_gates == (1, 2)


def test_two_stage_verifier_samples_gates_only_after_interior_roots():
    client, verifier, _ = honest_parties()
    boundary = client.commit_boundary()
    verifier.receive_boundary_commitment(boundary)
    challenged = verifier.lock_replay_units((0,))

    with pytest.raises(staged.ProtocolReject, match="after unit commitments"):
        verifier.lock_within_replay_unit_sample((0,))

    commitments = client.commit_units(challenged)
    verifier.receive_unit_commitments(commitments)
    assert verifier.lock_within_replay_unit_sample((0,)) == (0,)


@pytest.mark.parametrize(
    ("q", "expected_units"),
    [
        (0.0, ()),
        (1.0, (0, 1)),
    ],
)
def test_two_stage_replay_probability_boundaries(q, expected_units):
    client, verifier, _ = honest_parties()
    verifier.receive_boundary_commitment(client.commit_boundary())

    selected = verifier.sample_replay_units(q, rng=random.Random(1))

    assert selected == expected_units


@pytest.mark.parametrize(
    ("s", "expected_gates"),
    [
        (0.0, ()),
        (1.0, (0, 1, 2, 3)),
    ],
)
def test_two_stage_within_unit_probability_boundaries(s, expected_gates):
    client, verifier, _ = honest_parties()
    verifier.receive_boundary_commitment(client.commit_boundary())
    selected = verifier.lock_replay_units((0, 1))
    verifier.receive_unit_commitments(client.commit_units(selected))

    sampled = verifier.sample_within_replay_units(s, rng=random.Random(1))

    assert sampled == expected_gates


def test_two_stage_draws_one_coin_per_unit_then_one_per_selected_gate():
    class CountingRandom:
        def __init__(self):
            self.calls = 0

        def random(self):
            self.calls += 1
            return 0.0

    client, verifier, _ = honest_parties()
    verifier.receive_boundary_commitment(client.commit_boundary())
    unit_rng = CountingRandom()
    selected = verifier.sample_replay_units(1.0, rng=unit_rng)
    verifier.receive_unit_commitments(client.commit_units(selected))
    gate_rng = CountingRandom()
    sampled = verifier.sample_within_replay_units(1.0, rng=gate_rng)

    assert unit_rng.calls == len(verifier.plan.units)
    assert gate_rng.calls == verifier.plan.root_gate_count
    assert sampled == tuple(range(verifier.plan.root_gate_count))


def test_zero_gate_circuit_uses_empty_cut_and_only_boundary_commitment():
    kernel, root, plan, inputs = compiled_batch(lengths=(0,), cut=())
    assignment = staged.evaluate_assignment(kernel, root, inputs)
    client = staged.StagedClient(
        kernel=kernel,
        root=root,
        plan=plan,
        assignment=assignment,
    )
    verifier = staged.StagedVerifier(
        kernel=kernel,
        root=root,
        plan=plan,
        inputs=inputs,
        claimed_outputs=(assignment[plan.root_outputs[0]],),
    )

    transcript = staged.run_staged_protocol(
        client,
        verifier,
        sampled_gates=(),
    )

    assert transcript.accepted
    assert plan.units == ()
    assert transcript.unit_commitments == {}
    assert transcript.sampled_gates == ()


def test_staged_parties_reject_a_plan_mixed_with_the_wrong_circuit_metadata():
    kernel, root, plan, inputs = compiled_batch()
    assignment = staged.evaluate_assignment(kernel, root, inputs)
    tampered = dataclasses.replace(plan, root_outputs=(0,))

    with pytest.raises(staged.ProtocolReject, match="does not match"):
        staged.StagedClient(
            kernel=kernel,
            root=root,
            plan=tampered,
            assignment=assignment,
        )
    with pytest.raises(staged.ProtocolReject, match="does not match"):
        staged.StagedVerifier(
            kernel=kernel,
            root=root,
            plan=tampered,
            inputs=inputs,
            claimed_outputs=(assignment[0],),
        )


def test_verifier_enforces_boundary_J_interiors_T_order():
    client, verifier, _ = honest_parties()

    with pytest.raises(staged.ProtocolReject, match="before unit commitments"):
        verifier.reveal_sample()

    boundary = client.commit_boundary()
    verifier.receive_boundary_commitment(boundary)
    with pytest.raises(staged.ProtocolReject, match="out of order"):
        verifier.receive_unit_commitments({})

    challenged = verifier.lock_sample((0,))
    assert challenged == (0,)
    with pytest.raises(staged.ProtocolReject, match="before unit commitments"):
        verifier.reveal_sample()

    commitments = client.commit_units(challenged)
    verifier.receive_unit_commitments(commitments)
    assert verifier.reveal_sample() == (0,)


def test_one_forged_write_is_caught_exactly_when_its_gate_is_sampled():
    kernel, root, plan, inputs = compiled_batch()
    honest = staged.evaluate_assignment(kernel, root, inputs)
    forged_position = root.input_count
    forged_value = honest[forged_position] ^ 1
    assignment = staged.evaluate_assignment(
        kernel,
        root,
        inputs,
        overrides={forged_position: forged_value},
    )
    claimed_outputs = tuple(assignment[position] for position in plan.root_outputs)

    def run(sample):
        client = staged.StagedClient(
            kernel=kernel,
            root=root,
            plan=plan,
            assignment=assignment,
        )
        verifier = staged.StagedVerifier(
            kernel=kernel,
            root=root,
            plan=plan,
            inputs=inputs,
            claimed_outputs=claimed_outputs,
        )
        return staged.run_staged_protocol(
            client,
            verifier,
            sampled_gates=sample,
        )

    caught = run((0,))
    missed = run((1, 2, 3))

    assert not caught.accepted
    assert any(check.name == "gate 0" and not check.ok for check in caught.checks)
    assert missed.accepted


def test_two_stage_forgery_survives_unselected_unit_and_fails_when_sampled():
    kernel, root, plan, inputs = compiled_batch()
    honest = staged.evaluate_assignment(kernel, root, inputs)
    forged_position = root.input_count
    assignment = staged.evaluate_assignment(
        kernel,
        root,
        inputs,
        overrides={forged_position: honest[forged_position] ^ 1},
    )
    claimed_outputs = tuple(assignment[position] for position in plan.root_outputs)

    def run(challenged_units, sampled_gates):
        client = staged.StagedClient(
            kernel=kernel,
            root=root,
            plan=plan,
            assignment=assignment,
        )
        verifier = staged.StagedVerifier(
            kernel=kernel,
            root=root,
            plan=plan,
            inputs=inputs,
            claimed_outputs=claimed_outputs,
        )
        return staged.run_two_stage_protocol(
            client,
            verifier,
            challenged_units=challenged_units,
            sampled_gates=sampled_gates,
        )

    assert run((), ()).accepted
    assert run((0,), ()).accepted
    assert not run((0,), (0,)).accepted


def test_public_output_is_pinned_before_sampling():
    client, verifier, assignment = honest_parties()
    verifier.claimed_outputs = (assignment[verifier.plan.root_outputs[0]] ^ 1,)

    transcript = staged.run_staged_protocol(
        client,
        verifier,
        sampled_gates=(),
    )

    assert not transcript.accepted
    assert any(check.name == "output 0" and not check.ok for check in transcript.checks)


def test_tampering_with_an_opening_breaks_authentication():
    client, verifier, _ = honest_parties()
    boundary = client.commit_boundary()
    verifier.receive_boundary_commitment(boundary)
    challenged = verifier.lock_sample((0,))
    commitments = client.commit_units(challenged)
    verifier.receive_unit_commitments(commitments)
    verifier.reveal_sample()
    opening = client.open_gate(0)
    tampered = dataclasses.replace(
        opening,
        write=dataclasses.replace(opening.write, value=opening.write.value ^ 1),
    )

    check = verifier.check_gate(tampered)

    assert not check.ok
    assert check.detail == "write opening is invalid"


def test_finalization_rejects_a_missing_sampled_gate_check():
    client, verifier, _ = honest_parties()
    verifier.receive_boundary_commitment(client.commit_boundary())
    selected = verifier.lock_replay_units((0,))
    verifier.receive_unit_commitments(client.commit_units(selected))
    verifier.lock_within_replay_unit_sample((0, 1))
    assert verifier.check_gate(client.open_gate(0)).ok

    coverage = verifier.finalize()

    assert not coverage.ok
    assert "missing" in coverage.detail


def test_unit_commitments_must_match_revealed_J_exactly():
    client, verifier, _ = honest_parties()
    boundary = client.commit_boundary()
    verifier.receive_boundary_commitment(boundary)
    challenged = verifier.lock_sample((0, 2))
    commitments = client.commit_units(challenged)

    with pytest.raises(staged.ProtocolReject, match="exactly match"):
        verifier.receive_unit_commitments({0: commitments[0]})
