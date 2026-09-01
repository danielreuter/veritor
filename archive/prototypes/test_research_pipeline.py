import dataclasses
import random
from fractions import Fraction

import pytest

from prototypes.call_dag import (
    BatchInput,
    DemoG,
    Kernel,
    make_demo_request,
    trusted_word_gates,
)
from prototypes.research_pipeline import (
    VerificationPolicy,
    bound,
    claimed_outputs,
    compile_circuit,
    survival_probability,
    to_explicit_circuit,
    verify,
)
from prototypes.staged_replay import evaluate_assignment

CELL_BITS = 8


def compiled_demo():
    kernel = Kernel(
        cell_bits=CELL_BITS,
        gates=trusted_word_gates(CELL_BITS),
    )
    constructor = DemoG(CELL_BITS)
    batch = BatchInput(
        (
            make_demo_request(1, 1, CELL_BITS),
            make_demo_request(2, 2, CELL_BITS),
        )
    )
    circuit, replay_partition, verification_partition = compile_circuit(
        kernel,
        constructor,
        batch,
        b"",
        input_cells=batch.cells(),
        advice_bound_bits=0,
    )
    return (
        kernel,
        batch,
        circuit,
        replay_partition,
        verification_partition,
    )


def test_compile_returns_circuit_and_nested_partitions():
    kernel, batch, circuit, replay_partition, verification_partition = compiled_demo()

    assert circuit.input_count == len(batch.cells())
    assert circuit.gate_count == 6
    assert len(replay_partition.units) == 2
    assert tuple(unit.gate_count for unit in replay_partition.units) == (2, 4)
    assert tuple(unit.gate_ordinal for unit in verification_partition.units) == tuple(
        range(6)
    )
    assert tuple(unit.replay_unit_index for unit in verification_partition.units) == (
        0,
        0,
        1,
        1,
        1,
        1,
    )

    structural = to_explicit_circuit(kernel, circuit)
    assert len(structural.input_gates) == circuit.input_count
    assert len(structural.computed_gates) == circuit.gate_count
    assert len(structural.outputs) == 2


def test_honest_two_stage_verify_accepts_every_checked_gate():
    kernel, batch, circuit, replay_partition, verification_partition = compiled_demo()
    assignment = evaluate_assignment(kernel, circuit, batch.cells())
    policy = VerificationPolicy(1.0, 1.0, 0.0)

    transcript = verify(
        kernel,
        circuit,
        replay_partition,
        verification_partition,
        policy,
        assignment,
        inputs=batch.cells(),
        claimed_outputs=claimed_outputs(replay_partition, assignment),
        rng=random.Random(1),
    )

    assert transcript.accepted
    assert transcript.challenged_units == (0, 1)
    assert transcript.sampled_gates == tuple(range(circuit.gate_count))


def test_forged_gate_is_rejected_exactly_when_sampled():
    kernel, batch, circuit, replay_partition, verification_partition = compiled_demo()
    forged_position = circuit.input_count
    honest = evaluate_assignment(kernel, circuit, batch.cells())
    forged = evaluate_assignment(
        kernel,
        circuit,
        batch.cells(),
        overrides={forged_position: (honest[forged_position] + 1) % (1 << CELL_BITS)},
    )
    outputs = claimed_outputs(replay_partition, forged)

    checked = verify(
        kernel,
        circuit,
        replay_partition,
        verification_partition,
        VerificationPolicy(1.0, 1.0, 0.0),
        forged,
        inputs=batch.cells(),
        claimed_outputs=outputs,
        rng=random.Random(1),
    )
    missed = verify(
        kernel,
        circuit,
        replay_partition,
        verification_partition,
        VerificationPolicy(0.0, 1.0, 0.0),
        forged,
        inputs=batch.cells(),
        claimed_outputs=outputs,
        rng=random.Random(1),
    )

    assert not checked.accepted
    assert missed.accepted
    assert missed.sampled_gates == ()


def test_verify_does_not_underflow_exact_rational_sampling_probabilities():
    class ZeroRandom:
        def randrange(self, stop):
            assert stop > 0
            return 0

    kernel, batch, circuit, replay_partition, verification_partition = compiled_demo()
    honest = evaluate_assignment(kernel, circuit, batch.cells())
    forged_position = circuit.input_count
    forged = evaluate_assignment(
        kernel,
        circuit,
        batch.cells(),
        overrides={forged_position: honest[forged_position] ^ 1},
    )

    transcript = verify(
        kernel,
        circuit,
        replay_partition,
        verification_partition,
        VerificationPolicy(Fraction(1, 10**400), 1, 0),
        forged,
        inputs=batch.cells(),
        claimed_outputs=claimed_outputs(replay_partition, forged),
        rng=ZeroRandom(),
    )

    assert transcript.challenged_units == (0, 1)
    assert transcript.sampled_gates == tuple(range(circuit.gate_count))
    assert not transcript.accepted


def test_survival_profile_respects_replay_unit_correlation():
    _, _, _, replay_partition, verification_partition = compiled_demo()
    policy = VerificationPolicy(0.5, 1.0, 0.0)

    assert survival_probability(
        replay_partition,
        verification_partition,
        policy,
        (0, 1),
    ) == pytest.approx(0.5)
    assert survival_probability(
        replay_partition,
        verification_partition,
        policy,
        (0, 2),
    ) == pytest.approx(0.25)


def test_survival_profile_is_exact_at_the_strict_eta_boundary():
    _, _, _, replay_partition, verification_partition = compiled_demo()
    policy = VerificationPolicy(0.1, 0.3, 0.8866507)
    probability = survival_probability(
        replay_partition,
        verification_partition,
        policy,
        (0, 1, 2, 3, 4),
    )

    assert probability == Fraction(8_866_507, 10_000_000)
    assert probability == policy.acceptance_threshold


def test_bound_interpolates_between_full_and_no_verification():
    kernel, _, circuit, replay_partition, verification_partition = compiled_demo()

    full = bound(
        kernel,
        circuit,
        replay_partition,
        verification_partition,
        VerificationPolicy(1.0, 1.0, 0.0),
    )
    none = bound(
        kernel,
        circuit,
        replay_partition,
        verification_partition,
        VerificationPolicy(0.0, 1.0, 0.0),
    )
    one_replay_unit = bound(
        kernel,
        circuit,
        replay_partition,
        verification_partition,
        VerificationPolicy(0.5, 1.0, 0.4),
    )

    assert full.capacity_bits == 0
    assert full.witness_error_units == ()
    assert none.capacity_bits == 2 * CELL_BITS
    assert one_replay_unit.capacity_bits == CELL_BITS
    assert one_replay_unit.witness_survival_probability == pytest.approx(0.5)


def test_bound_refuses_accidental_exponential_expansion():
    kernel, _, circuit, replay_partition, verification_partition = compiled_demo()

    with pytest.raises(ValueError, match="at most 5"):
        bound(
            kernel,
            circuit,
            replay_partition,
            verification_partition,
            VerificationPolicy(0.5, 0.5, 0.1),
            max_enumerated_units=5,
        )


def test_bound_rejects_a_replay_partition_from_another_compiled_tuple():
    kernel, _, circuit, replay_partition, verification_partition = compiled_demo()
    tampered = dataclasses.replace(replay_partition, root_outputs=(0,))

    with pytest.raises(ValueError, match="does not match"):
        bound(
            kernel,
            circuit,
            tampered,
            verification_partition,
            VerificationPolicy(0.5, 0.5, 0.1),
        )
