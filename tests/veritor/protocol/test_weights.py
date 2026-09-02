"""Weights live under their own per-model root; runs never carry them wholesale."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import replace

import pytest

from veritor.compile import Compiler
from veritor.constructors import MatmulCompileRequest, MatmulG
from veritor.core import VerificationPolicy, make_word_gate_set
from veritor.protocol import (
    BOUNDARY_OWNER,
    WEIGHT_OWNER,
    Expectation,
    Opening,
    ProtocolError,
    ProverSession,
    VerificationCode,
    VerifierSession,
    Weights,
    commit_weights,
    decode_transcript,
    encode_transcript,
    make_expectation,
    merkle,
    run_protocol,
    verify_transcript,
)
from veritor.protocol.domains import public_boundary

CHECK_EVERYTHING = VerificationPolicy(1, 1)
SEEDS = {"session_id": b"weights", "q_seed": b"Q" * 32, "s_seed": b"S" * 32}
GATE_SET = make_word_gate_set(8)


def matmul_request(n: int, rows: int) -> MatmulCompileRequest:
    weights = tuple(tuple((i * j + 1) % 256 for j in range(n)) for i in range(n))
    activation = tuple(tuple((r + 3 * c) % 256 for c in range(n)) for r in range(rows))
    return MatmulCompileRequest(weights, (activation,))


class Model:
    """A compiled matmul with its weights committed once."""

    def __init__(self, n: int = 4, rows: int = 2) -> None:
        self.request = matmul_request(n, rows)
        workload = self.request.workload
        self.compiled = Compiler(GATE_SET).compile(
            MatmulG(8)(workload, b""), workload.public_inputs
        )
        self.values = dict(enumerate(self.compiled.circuit.evaluate(workload.public_inputs)))
        span = self.request.weight_addresses
        self.weights, self.tree = commit_weights(self.compiled, span.start, span.stop, self.values)

    def expectation(self, **overrides) -> Expectation:
        arguments = {
            "weights": self.weights,
            "claimed_outputs": self.request.expected_outputs,
            **SEEDS,
            **overrides,
        }
        return make_expectation(
            self.compiled,
            CHECK_EVERYTHING,
            self.request.activation_inputs,
            arguments.pop("claimed_outputs"),
            **arguments,
        )

    def tampered(self, address: int) -> tuple[Expectation, dict[int, object]]:
        """A prover who computed with one weight changed and claims that result."""

        inputs = list(self.request.public_inputs)
        inputs[address] = (inputs[address] + 1) % 256
        values = dict(enumerate(self.compiled.circuit.evaluate(inputs)))
        outputs = tuple(values[o] for o in self.compiled.circuit.outputs)
        return self.expectation(claimed_outputs=outputs), values


MODELS: dict[tuple[int, int], Model] = {}


def cached_model(n: int, rows: int) -> Model:
    if (n, rows) not in MODELS:
        MODELS[n, rows] = Model(n, rows)
    return MODELS[n, rows]


@pytest.fixture(scope="module")
def model() -> Model:
    return cached_model(4, 2)


def test_matmul_request_names_its_weight_inputs() -> None:
    request = matmul_request(3, 2)

    assert request.weight_addresses == range(9)
    assert request.public_inputs[:9] == tuple(v for row in request.weights for v in row)
    assert request.activation_inputs == request.public_inputs[9:]
    assert len(request.activation_inputs) == 6


def test_honest_run_under_a_weight_root_accepts_and_round_trips(model: Model) -> None:
    expectation = model.expectation()

    run = run_protocol(model.compiled, expectation, model.values, weight_tree=model.tree)

    assert run.report.accepted
    assert run.transcript is not None
    header = run.transcript.header
    assert header.weights == model.weights
    assert len(header.public_inputs) == len(model.request.activation_inputs)
    weight_count = model.weights.count
    boundary = model.compiled.index.boundary()
    assert run.transcript.boundary.commitment.count == boundary.count - weight_count
    opened = {item.position for item in run.transcript.boundary.io_openings}
    assert opened.isdisjoint(model.request.weight_addresses)
    data = encode_transcript(run.transcript)
    assert decode_transcript(data) == run.transcript
    assert verify_transcript(data, expectation, model.compiled) == run.report

    without_tree = run_protocol(model.compiled, expectation, model.values)
    assert without_tree.report.accepted


def test_sampled_evidence_opens_weights_under_kappa_w(model: Model) -> None:
    run = run_protocol(model.compiled, model.expectation(), model.values, weight_tree=model.tree)
    assert run.transcript is not None
    weight_openings = [
        item
        for batch in run.transcript.evidence.units
        for item in batch
        if item.position in model.weights
    ]

    assert weight_openings
    assert all(len(item.path) == merkle.merkle_depth(model.weights.count) for item in weight_openings)
    assert {len(item.path) for item in weight_openings} == {
        merkle.merkle_depth(model.weights.count)
    }


def test_a_tampered_weight_is_caught_at_a_sampled_gate(model: Model) -> None:
    expectation, values = model.tampered(address=5)

    run = run_protocol(model.compiled, expectation, values, weight_tree=model.tree)

    assert run.report.code is VerificationCode.RELATION_REJECTED
    assert run.transcript is None


def test_a_prover_cannot_substitute_its_own_weight_root(model: Model) -> None:
    expectation, values = model.tampered(address=5)
    verifier = VerifierSession(expectation, model.compiled)
    span = model.request.weight_addresses
    _, own_tree = commit_weights(model.compiled, span.start, span.stop, values)

    with pytest.raises(ProtocolError, match="does not match"):
        ProverSession(model.compiled, verifier.header, values, weight_tree=own_tree)
    with pytest.raises(ProtocolError, match="does not match"):
        run_protocol(model.compiled, expectation, values)
    with pytest.raises(ProtocolError, match="needs their tree"):
        ProverSession(model.compiled, verifier.header, values)


def test_a_forged_weight_opening_fails_under_kappa_w(model: Model) -> None:
    expectation = model.expectation()
    run = run_protocol(model.compiled, expectation, model.values, weight_tree=model.tree)
    assert run.transcript is not None
    units = list(run.transcript.evidence.units)
    batch = list(units[0])
    position = next(i for i, item in enumerate(batch) if item.position in model.weights)
    item = batch[position]
    batch[position] = Opening(item.position, bytes((item.value[0] ^ 1,)) + item.value[1:], item.path)
    units[0] = tuple(batch)
    tampered = replace(run.transcript, evidence=replace(run.transcript.evidence, units=tuple(units)))

    report = verify_transcript(encode_transcript(tampered), expectation, model.compiled)

    assert report.code is VerificationCode.INVALID_OPENING
    assert f"owner {WEIGHT_OWNER}" in report.detail


def test_a_weight_range_outside_the_inputs_is_rejected(model: Model) -> None:
    inputs = model.compiled.index.input_count
    beyond = Weights(0, inputs + 1, model.weights.root)

    run = run_protocol(model.compiled, model.expectation(weights=beyond), model.values)

    assert run.report.code is VerificationCode.INVALID_COMPILED_RESULT
    with pytest.raises(ProtocolError, match="nonnegative"):
        Weights(3, 2, model.weights.root)


def test_public_boundary_is_the_boundary_without_the_weights(model: Model) -> None:
    index = model.compiled.index
    full = list(index.boundary())
    span = model.request.weight_addresses
    public = public_boundary(index, model.weights)
    expected = [address for address in full if address not in span]

    assert public.count == len(expected)
    assert [public.unrank(rank) for rank in range(public.count)] == expected
    assert all(public.rank(address) == rank for rank, address in enumerate(expected))
    assert all(not public.contains(address) for address in span)
    assert all(public.contains(address) for address in expected)
    assert not public.contains(index.n - 1) or (index.n - 1) in expected
    with pytest.raises(KeyError):
        public.rank(span.start)
    with pytest.raises(IndexError):
        public.unrank(public.count)


def test_ownership_rule_weights_then_boundary_then_interior(model: Model) -> None:
    session = VerifierSession(model.expectation(), model.compiled)
    layout = session._layout
    index = model.compiled.index
    span = model.request.weight_addresses

    assert all(layout.owner(address) == WEIGHT_OWNER for address in span)
    assert all(layout.owner(address) == BOUNDARY_OWNER for address in layout.public_inputs)
    assert all(layout.owner(address) == BOUNDARY_OWNER for address in model.compiled.circuit.outputs)
    interior = int(index.interior(1).unrank(0))
    assert layout.owner(interior) == 1


# -- the verifier's work does not grow with |W| ----------------------------------


class BoundaryPhase:
    """The verifier's setup and boundary phase for one ``n``, ready to rerun."""

    def __init__(self, n: int, *, weights: bool) -> None:
        model = cached_model(n, 1)
        if weights:
            expectation = model.expectation()
            tree = model.tree
        else:
            expectation = make_expectation(
                model.compiled,
                CHECK_EVERYTHING,
                model.request.public_inputs,
                model.request.expected_outputs,
                **SEEDS,
            )
            tree = None
        header = VerifierSession(expectation, model.compiled).header
        prover = ProverSession(model.compiled, header, model.values, weight_tree=tree)
        self.boundary = prover.boundary()
        self.compiled = model.compiled
        self.expectation = expectation
        self.io = len(header.public_inputs) + len(header.claimed_outputs)
        self.weight_count = model.weights.count
        self.hashes_expected = self.io * (1 + merkle.merkle_depth(self.boundary.commitment.count))
        """Leaf plus path hashes for the public I/O openings, the only per-run hashing."""

    def __call__(self) -> object:
        return VerifierSession(self.expectation, self.compiled).receive_boundary(self.boundary)


def count_hashes(monkeypatch, action: Callable[[], object]) -> int:
    calls = 0
    original = merkle._hash

    def counting(*parts):
        nonlocal calls
        calls += 1
        return original(*parts)

    monkeypatch.setattr(merkle, "_hash", counting)
    action()
    return calls


def test_verifier_hashes_follow_the_public_io_not_the_weights(monkeypatch) -> None:
    small = BoundaryPhase(16, weights=True)
    large = BoundaryPhase(128, weights=True)
    assert large.weight_count == 64 * small.weight_count and large.io == 8 * small.io

    hashes_small = count_hashes(monkeypatch, small)
    hashes_large = count_hashes(monkeypatch, large)

    domains = 2  # the domain ids of kappa_W and of the boundary
    assert small.hashes_expected <= hashes_small <= small.hashes_expected + domains
    assert large.hashes_expected <= hashes_large <= large.hashes_expected + domains
    assert hashes_large < 16 * hashes_small

    public = BoundaryPhase(128, weights=False)
    assert public.io == large.io + large.weight_count
    assert count_hashes(monkeypatch, public) > 64 * hashes_large


def fastest(action: Callable[[], object], repetitions: int = 5) -> float:
    best = float("inf")
    for _ in range(repetitions):
        start = time.perf_counter()
        action()
        best = min(best, time.perf_counter() - start)
    return best


def test_verifier_time_drops_when_weights_leave_the_boundary() -> None:
    with_root = BoundaryPhase(64, weights=True)
    public = BoundaryPhase(64, weights=False)
    assert with_root.weight_count == 32 * with_root.io
    assert public.io == with_root.io + with_root.weight_count

    assert fastest(with_root) < fastest(public) / 4
