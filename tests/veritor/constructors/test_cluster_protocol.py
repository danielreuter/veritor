"""The cluster through the protocol: the model committed once, a run per schedule.

``Compile(ClusterG, requests, schedule)`` is the verifier's; the parameters
are committed under ``kappa_W`` before any request exists; an honest run is
accepted at every policy and its transcript round-trips; a prover who alters
a generated token is caught at the boundary and one who corrupts an interior
value is caught when everything is sampled.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from veritor.constructors import (
    ClusterG,
    Join,
    LMShape,
    Request,
    Schedule,
    random_parameters,
    reference_generate,
    schedule_fcfs,
)
from veritor.core import VerificationPolicy, make_isa_gate_set
from veritor.protocol import (
    VerificationCode,
    VerifierParameters,
    assignment_replay,
    commit_weights,
    decode_transcript,
    encode_transcript,
    make_expectation,
    run_protocol,
    verify_transcript,
)
from veritor.research import Compile

SHAPE = LMShape(vocab=8, d_model=4, heads=2, layers=1, context=6, width=16)
REQUESTS = (Request((1, 2, 3), 3), Request((5,), 2), Request((7, 0), 4), Request((2, 2, 2), 1))
GATE_SET = make_isa_gate_set(16)
SEEDS = {"session_id": b"cluster-run", "q_seed": b"Q" * 32, "s_seed": b"S" * 32}
CHECK_EVERYTHING = VerificationPolicy(1, 1)
SAMPLE_SOME = VerificationPolicy(Fraction(1, 2), Fraction(1, 3))


class Deployment:
    """A model committed once and one cluster run compiled by the verifier."""

    def __init__(self, schedule: Schedule, pods: int, slots: int, steps: int) -> None:
        self.parameters = random_parameters(SHAPE, seed=42)
        self.weights, self.tree = commit_weights(GATE_SET, self.parameters.flatten())
        self.constructor = ClusterG(SHAPE, pods, slots, steps)
        self.schedule = schedule
        self.compilation = Compile(
            self.constructor, REQUESTS, schedule.encode(), GATE_SET, max_advice_bits=4096
        )
        self.compiled = self.compilation.compiled
        self.circuit = self.compiled.circuit
        self.values = dict(
            enumerate(self.circuit.evaluate(self.compilation.inputs, self.parameters.flatten()))
        )
        self.outputs = tuple(self.values[address] for address in self.circuit.outputs)

    def expectation(self, policy: VerificationPolicy = CHECK_EVERYTHING, **overrides):
        arguments = {
            "weights": self.weights,
            "claimed_outputs": self.outputs,
            "parameters": VerifierParameters(max_advice_bits=4096, max_capacity=None),
            **SEEDS,
            **overrides,
        }
        return make_expectation(self.compilation, policy, arguments.pop("claimed_outputs"), **arguments)


@pytest.fixture(scope="module")
def deployment() -> Deployment:
    return Deployment(schedule_fcfs(REQUESTS, 2, 2, 6), 2, 2, 6)


def test_the_compilation_binds_the_constructor_the_prompts_and_the_schedule(deployment: Deployment) -> None:
    compilation = deployment.compilation

    assert compilation.constructor == deployment.constructor.digest
    assert compilation.inputs == deployment.constructor.flatten_inputs(REQUESTS, deployment.schedule)
    assert compilation.advice == deployment.schedule.encode()
    assert compilation.advice_bits == 8 * len(deployment.schedule.encode()) > 0
    assert deployment.weights.count == SHAPE.weight_count == deployment.compiled.index.weight_count
    # the claimed outputs are what sequential decoding gives
    layout = deployment.constructor.output_layout(REQUESTS, deployment.schedule)
    reference = reference_generate(SHAPE, deployment.parameters, REQUESTS)
    assert deployment.outputs == tuple(reference[r][g] for r, g in layout)


@pytest.mark.parametrize("policy", (CHECK_EVERYTHING, SAMPLE_SOME))
def test_an_honest_cluster_run_is_accepted_and_its_transcript_round_trips(
    deployment: Deployment, policy: VerificationPolicy
) -> None:
    expectation = deployment.expectation(policy)

    run = run_protocol(deployment.compiled, expectation, deployment.values, weight_tree=deployment.tree)

    assert run.report.accepted and run.report.code is VerificationCode.ACCEPTED
    assert run.transcript is not None
    header = run.transcript.header
    assert header.weights == deployment.weights and header.advice == deployment.schedule.encode()
    assert header.constructor == deployment.constructor.digest
    boundary = deployment.compiled.index.boundary()
    assert run.transcript.boundary.commitment.count == boundary.count == 9 + 130
    opened = {item.position for item in run.transcript.boundary.io_openings}
    assert opened == set(deployment.circuit.inputs) | set(deployment.circuit.outputs)
    assert opened.isdisjoint(deployment.circuit.weights)
    replayed = run.report.sampled_replay_units
    if policy == CHECK_EVERYTHING:
        assert len(replayed) == deployment.compiled.index.replay_units.count == 8
        assert len(run.report.sampled_verification_units) == 1065
    else:
        assert len(replayed) < 8 and len(run.report.sampled_verification_units) < 1065

    data = encode_transcript(run.transcript)
    assert decode_transcript(data) == run.transcript
    assert verify_transcript(data, expectation, deployment.compiled) == run.report


def test_the_advice_must_be_admitted_by_the_verifier(deployment: Deployment) -> None:
    expectation = deployment.expectation(parameters=VerifierParameters(max_advice_bits=8, max_capacity=None))

    run = run_protocol(deployment.compiled, expectation, deployment.values, weight_tree=deployment.tree)

    assert run.report.code is VerificationCode.POLICY_REJECTED
    assert "exceeding max_advice_bits 8" in run.report.detail


def test_an_altered_generated_token_is_rejected_at_the_boundary(deployment: Deployment) -> None:
    """The prover claims a different token for one request but computed honestly."""

    claimed = list(deployment.outputs)
    claimed[4] = (claimed[4] + 1) % SHAPE.vocab
    expectation = deployment.expectation(claimed_outputs=tuple(claimed))

    run = run_protocol(deployment.compiled, expectation, deployment.values, weight_tree=deployment.tree)

    assert run.report.code is VerificationCode.PUBLIC_IO_MISMATCH
    assert run.transcript is None

    # ... and one who also changes the output gate's value to match is caught at its relation
    lying = dict(deployment.values)
    lying[deployment.circuit.outputs[4]] = claimed[4]
    run = run_protocol(deployment.compiled, expectation, lying, weight_tree=deployment.tree)
    assert run.report.code is VerificationCode.RELATION_REJECTED


def test_a_corrupted_interior_value_is_rejected_when_everything_is_sampled(deployment: Deployment) -> None:
    """A dot product inside one decode step is off by one; the outputs are the honest ones.

    The prover commits the corrupted interior (``assignment_replay``: the
    default prover replays honestly from the boundary, which would hide the
    corruption); at ``q = s = 1`` the dot's own relation is checked and fails.
    """

    index = deployment.compiled.index
    step = index.replay_units.unit(3)  # pod 0, step 2: decodes
    interior = index.interior(3)
    address = int(interior.unrank(interior.count // 2))
    assert step.interval.start <= address < step.interval.stop
    assert address not in set(deployment.circuit.outputs) and deployment.circuit[address].op == "add"
    corrupted = dict(deployment.values)
    corrupted[address] = (corrupted[address] + 1) % (1 << SHAPE.width)

    run = run_protocol(
        deployment.compiled,
        deployment.expectation(),
        corrupted,
        replay=assignment_replay(corrupted),
        weight_tree=deployment.tree,
    )

    assert run.report.code is VerificationCode.RELATION_REJECTED
    assert run.transcript is None
    # the honest prover recomputes interiors from the boundary: the same dict is harmless
    honest = run_protocol(deployment.compiled, deployment.expectation(), corrupted, weight_tree=deployment.tree)
    assert honest.report.accepted
    # sampled sparsely the corruption may slip through: that is what Bound charges for
    sparse = run_protocol(
        deployment.compiled,
        deployment.expectation(SAMPLE_SOME),
        corrupted,
        replay=assignment_replay(corrupted),
        weight_tree=deployment.tree,
    )
    assert sparse.report.code in (VerificationCode.ACCEPTED, VerificationCode.RELATION_REJECTED)


def test_a_hand_written_schedule_runs_under_the_same_weight_root(deployment: Deployment) -> None:
    """Another schedule of the same requests: a different circuit, the model's one ``kappa_W``."""

    schedule = Schedule(2, 1, 6, (Join(0, 0, 0, 0), Join(0, 2, 0, 1), Join(1, 0, 0, 2), Join(1, 4, 0, 3)))
    other = Deployment(schedule, 2, 1, 6)

    assert other.weights == deployment.weights and other.compiled.digest != deployment.compiled.digest
    run = run_protocol(
        other.compiled, other.expectation(weights=deployment.weights), other.values, weight_tree=deployment.tree
    )
    assert run.report.accepted
    reference = reference_generate(SHAPE, other.parameters, REQUESTS)
    assert other.outputs == (*reference[0][:2], *reference[1], *reference[2], *reference[3])
