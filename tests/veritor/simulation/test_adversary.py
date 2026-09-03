"""The exfiltration adversary against a small compiled cluster run."""

from __future__ import annotations

from fractions import Fraction

import pytest

from veritor.analysis.probability import survival
from veritor.constructors import ClusterG, LMShape, random_parameters
from veritor.core import VerificationPolicy, make_isa_gate_set
from veritor.protocol import VerifierParameters, commit_weights
from veritor.research import Compile
from veritor.simulation.adversary import (
    carriers,
    decode_secret,
    evaluate_with_overrides,
    plan_attack,
    predicted_survival,
    protocol_trials,
    random_secret,
    survival_trials,
)
from veritor.simulation.workload import WorkloadConfig, simulate

SHAPE = LMShape(
    vocab=8, d_model=4, heads=2, layers=1, context=16, width=16, sampling=True
)
GATE_SET = make_isa_gate_set(16)


class Run:
    def __init__(self) -> None:
        self.parameters = random_parameters(SHAPE, seed=5)
        self.weights = self.parameters.flatten()
        self.kappa, self.tree = commit_weights(GATE_SET, self.weights)
        config = WorkloadConfig(
            pods=1, slots=2, steps=8, arrivals=5, seed=2, forced_failures=((0, 3),)
        )
        self.simulation = simulate(config, SHAPE, self.parameters)
        constructor = ClusterG(SHAPE, 1, 2, 8)
        advice = self.simulation.schedule.encode()
        self.compilation = Compile(
            constructor,
            self.simulation.requests,
            advice,
            GATE_SET,
            max_advice_bits=8 * len(advice),
        )
        self.compiled = self.compilation.compiled
        self.layout = constructor.output_layout(
            self.simulation.requests, self.simulation.schedule
        )
        self.honest = evaluate_with_overrides(
            self.compiled, self.compilation.inputs, self.weights, {}
        )
        self.outputs = tuple(self.honest[a] for a in self.compiled.circuit.outputs)
        assert self.outputs == tuple(
            t for response in self.simulation.streamed for t in response
        )


@pytest.fixture(scope="module")
def run() -> Run:
    return Run()


def test_carriers_are_the_last_tokens_round_robin_over_requests():
    layout = [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0), (2, 1)]
    assert carriers(layout, 0) == ()
    assert carriers(layout, 3) == (2, 3, 5)  # the last token of each request
    assert carriers(layout, 4) == (
        1,
        2,
        3,
        5,
    )  # then the second to last of the first request
    assert carriers(layout, 6) == (0, 1, 2, 3, 4, 5)
    with pytest.raises(ValueError):
        carriers(layout, 7)


def test_the_secret_is_written_into_the_head_vus_and_read_back(run):
    secret = random_secret(4 * SHAPE.vocab_bits, seed=1)
    assert len(set(secret)) == 2 and random_secret(12, seed=1) == secret
    attack = plan_attack(
        run.compiled,
        run.compilation.inputs,
        run.weights,
        run.layout,
        secret,
        SHAPE.vocab_bits,
    )
    assert attack.bits == 12 and len(attack.carriers) == 4
    assert attack.addresses == tuple(
        run.compiled.circuit.outputs[c] for c in attack.carriers
    )
    assert set(attack.corrupted) <= set(attack.addresses)
    assert len(set(attack.verification_units)) == len(
        attack.corrupted
    )  # one head VU per corrupted token
    assert decode_secret(attack.outputs, attack.carriers, SHAPE.vocab_bits) == secret
    # a carrier whose honest token already spells its chunk is not corrupted
    for address, carrier in zip(attack.addresses, attack.carriers, strict=True):
        assert (address in attack.corrupted) == (
            attack.outputs[carrier] != run.outputs[carrier]
        )
    for index, token in enumerate(attack.outputs):
        if index not in attack.carriers:
            assert token == run.outputs[index]
    # only the corrupted gates violate their relation
    circuit = run.compiled.circuit
    wrong = [
        address
        for address in range(circuit.n)
        if not circuit[address].is_source
        and not circuit.check_gate(
            address,
            tuple(attack.values[a] for a in circuit[address].args),
            attack.values[address],
        )
    ]
    assert wrong == sorted(attack.corrupted)
    with pytest.raises(ValueError):
        plan_attack(
            run.compiled,
            run.compilation.inputs,
            run.weights,
            run.layout,
            "01",
            SHAPE.vocab_bits,
        )


def test_the_predicted_survival_is_the_product_over_replay_units(run):
    policy = VerificationPolicy(Fraction(1, 2), Fraction(1, 3))
    honest = format(run.outputs[carriers(run.layout, 1)[0]], "03b")
    chunk = "000" if honest != "000" else "111"
    one = plan_attack(
        run.compiled, run.compilation.inputs, run.weights, run.layout, chunk, 3
    )
    assert (
        len(one.corrupted) == 1
        and predicted_survival(policy, one) == 1 - policy.q * policy.s
    )
    same = plan_attack(
        run.compiled, run.compilation.inputs, run.weights, run.layout, honest, 3
    )
    assert same.corrupted == () and predicted_survival(policy, same) == 1
    everything = plan_attack(
        run.compiled,
        run.compilation.inputs,
        run.weights,
        run.layout,
        "0" * (3 * len(run.layout)),
        3,
    )
    assert predicted_survival(policy, everything) == survival(
        policy, everything.errors_per_replay_unit
    )
    assert predicted_survival(policy, everything) < predicted_survival(policy, one)


def test_the_extreme_policies_detect_everything_or_nothing(run):
    honest = format(run.outputs[carriers(run.layout, 1)[0]], "03b")
    chunk = "110" if honest != "110" else "001"
    attack = plan_attack(
        run.compiled, run.compilation.inputs, run.weights, run.layout, chunk, 3
    )
    assert len(attack.corrupted) == 1
    assert survival_trials(run.compiled, VerificationPolicy(1, 1), attack, 20) == 0
    assert survival_trials(run.compiled, VerificationPolicy(0, 1), attack, 20) == 20
    out_bits = len(run.compiled.circuit.outputs) * SHAPE.width
    parameters = VerifierParameters(
        Fraction(1, 2**40), max_capacity=out_bits, max_advice_bits=1 << 16
    )
    caught = protocol_trials(
        run.compilation,
        VerificationPolicy(1, 1),
        parameters,
        run.kappa,
        run.tree,
        attack,
        1,
    )
    assert not caught[0].accepted and caught[0].code.name == "RELATION_REJECTED"
    assert set(caught[0].sampled_verification_units) >= set(attack.verification_units)
    escaped = protocol_trials(
        run.compilation,
        VerificationPolicy(0, 1),
        parameters,
        run.kappa,
        run.tree,
        attack,
        1,
    )
    assert escaped[0].accepted


def test_the_selection_law_tracks_the_prediction_over_many_seeds(run):
    policy = VerificationPolicy(Fraction(1, 2), Fraction(1, 2))
    secret = random_secret(3 * len(run.layout), seed=9)
    attack = plan_attack(
        run.compiled, run.compilation.inputs, run.weights, run.layout, secret, 3
    )
    trials = 600
    expected = float(predicted_survival(policy, attack))
    observed = (
        survival_trials(run.compiled, policy, attack, trials, label="law") / trials
    )
    sigma = (expected * (1 - expected) / trials) ** 0.5
    assert abs(observed - expected) <= 4 * sigma
