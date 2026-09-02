"""Weights live under their own per-model root; runs never carry them wholesale."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import replace

import pytest

from veritor.compile import Compiler
from veritor.constructors import MatmulCompileRequest, MatmulG, TracedDefinition
from veritor.core import Gate, GateSet, VerificationPolicy, make_word_gate_set
from veritor.protocol import (
    BOUNDARY_OWNER,
    WEIGHT_OWNER,
    Expectation,
    Opening,
    ProtocolError,
    ProtocolRun,
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
from veritor.protocol.domains import weight_domain

CHECK_EVERYTHING = VerificationPolicy(1, 1)
SEEDS = {"session_id": b"weights", "q_seed": b"Q" * 32, "s_seed": b"S" * 32}
GATE_SET = make_word_gate_set(8)


def matmul_request(n: int, rows: int) -> MatmulCompileRequest:
    weights = tuple(tuple((i * j + 1) % 256 for j in range(n)) for i in range(n))
    activation = tuple(tuple((r + 3 * c) % 256 for c in range(n)) for r in range(rows))
    return MatmulCompileRequest(weights, (activation,))


class PublicWeightsG(MatmulG):
    """The same matmul with the weights as ``in`` gates: every value is public I/O."""

    def weights_unit(self, count: int) -> TracedDefinition:
        return self.tracer.definition(
            input_count=0, key=("public-weights", count), role="replay"
        )(lambda _v: self.tracer.inputs(count))


class Model:
    """A compiled matmul with its weights committed once."""

    def __init__(self, n: int = 4, rows: int = 2) -> None:
        self.request = matmul_request(n, rows)
        workload = self.request.workload
        self.compiled = Compiler(GATE_SET).compile(
            MatmulG(8)(workload, b""), workload.public_inputs
        )
        self.circuit = self.compiled.circuit
        self.weight_addresses = frozenset(self.circuit.weights)
        self.values = dict(
            enumerate(self.circuit.evaluate(workload.public_inputs, workload.weight_values))
        )
        self.weights, self.tree = commit_weights(GATE_SET, workload.weight_values)

    def weight_openings(self, run: ProtocolRun) -> list[tuple[int, int, int, Opening]]:
        """``(unit, index, address, opening)`` for every ``kappa_W`` opening in the evidence."""

        assert run.transcript is not None
        layout = VerifierSession(self.expectation(), self.compiled)._layout
        return [
            (unit, index, address, item)
            for unit, batch in zip(
                run.report.sampled_verification_units, run.transcript.evidence.units, strict=True
            )
            for index, ((owner, address), item) in enumerate(
                zip(layout.required(unit), batch, strict=True)
            )
            if owner == WEIGHT_OWNER
        ]

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
            self.request.public_inputs,
            arguments.pop("claimed_outputs"),
            **arguments,
        )

    def tampered(self, rank: int) -> tuple[Expectation, dict[int, object]]:
        """A prover who computed with one weight changed and claims that result."""

        weights = list(self.request.weight_values)
        weights[rank] = (weights[rank] + 1) % 256
        values = dict(enumerate(self.circuit.evaluate(self.request.public_inputs, weights)))
        outputs = tuple(values[o] for o in self.circuit.outputs)
        return self.expectation(claimed_outputs=outputs), values


MODELS: dict[tuple[int, int], Model] = {}


def cached_model(n: int, rows: int) -> Model:
    if (n, rows) not in MODELS:
        MODELS[n, rows] = Model(n, rows)
    return MODELS[n, rows]


@pytest.fixture(scope="module")
def model() -> Model:
    return cached_model(4, 2)


def test_matmul_request_separates_activations_from_weights() -> None:
    request = matmul_request(3, 2)

    assert request.weight_values == tuple(v for row in request.weights for v in row)
    assert len(request.weight_values) == 9 and len(request.public_inputs) == 6
    assert request.public_inputs == tuple(v for row in request.activations[0] for v in row)


def test_honest_run_under_a_weight_root_accepts_and_round_trips(model: Model) -> None:
    expectation = model.expectation()

    run = run_protocol(model.compiled, expectation, model.values, weight_tree=model.tree)

    assert run.report.accepted
    assert run.transcript is not None
    header = run.transcript.header
    assert header.weights == model.weights == Weights(16, model.weights.root)
    assert len(header.public_inputs) == len(model.request.public_inputs) == 8
    boundary = model.compiled.index.boundary()
    assert run.transcript.boundary.commitment.count == boundary.count == 8 + 8
    opened = {item.position for item in run.transcript.boundary.io_openings}
    assert opened.isdisjoint(model.weight_addresses)
    assert opened == set(model.circuit.inputs) | set(model.circuit.outputs)
    data = encode_transcript(run.transcript)
    assert decode_transcript(data) == run.transcript
    assert verify_transcript(data, expectation, model.compiled) == run.report

    without_tree = run_protocol(model.compiled, expectation, model.values)
    assert without_tree.report.accepted


def test_sampled_evidence_opens_weights_under_kappa_w_at_their_ranks(model: Model) -> None:
    run = run_protocol(model.compiled, model.expectation(), model.values, weight_tree=model.tree)
    weight_openings = model.weight_openings(run)

    # every weight cell opens itself and every dot opens the column it reads
    assert len(weight_openings) == 16 + 2 * 4 * 4
    assert {len(item.path) for *_, item in weight_openings} == {
        merkle.merkle_depth(model.weights.count)
    }
    # a weight opening names the weight's rank, the position in kappa_W, not its address
    assert all(item.position == model.circuit.weight_rank(a) for _, _, a, item in weight_openings)
    assert {a for _, _, a, _ in weight_openings} == model.weight_addresses
    assert {item.position for *_, item in weight_openings} == set(range(16))
    assert all(
        item.value == bytes((model.request.weight_values[item.position],))
        for *_, item in weight_openings
    )


def test_one_weight_root_serves_every_batch_shape_of_the_model() -> None:
    """``kappa_W`` is per model: requests of different shapes compile different circuits
    from the same weights and all open them under the one root committed for the model."""

    two_rows, one_row = cached_model(4, 2), cached_model(4, 1)
    assert two_rows.compiled.digest != one_row.compiled.digest
    assert two_rows.request.weight_values == one_row.request.weight_values
    assert two_rows.weights == one_row.weights == Weights(16, two_rows.weights.root)
    assert two_rows.tree.domain == one_row.tree.domain

    for model in (two_rows, one_row):
        run = run_protocol(
            model.compiled,
            model.expectation(weights=two_rows.weights),
            model.values,
            weight_tree=two_rows.tree,
        )
        assert run.report.accepted and run.transcript is not None
        assert run.transcript.header.weights == two_rows.weights
        assert len(model.weight_openings(run)) == 16 + model.request.output_shapes[0][0] * 16

    # the root is the model's under this gate set: other words, other root
    wider, _ = commit_weights(make_word_gate_set(16), two_rows.request.weight_values)
    assert wider.count == 16 and wider.root != two_rows.weights.root


def test_a_tampered_weight_is_caught_at_a_sampled_gate(model: Model) -> None:
    """The dot used a weight that differs from the kappa_W leaf it must open."""

    expectation, values = model.tampered(rank=5)

    run = run_protocol(model.compiled, expectation, values, weight_tree=model.tree)

    assert run.report.code is VerificationCode.RELATION_REJECTED
    assert run.transcript is None


def test_a_prover_cannot_substitute_its_own_weight_root(model: Model) -> None:
    expectation, values = model.tampered(rank=5)
    verifier = VerifierSession(expectation, model.compiled)
    own_weights = [values[address] for address in model.circuit.weights]
    _, own_tree = commit_weights(GATE_SET, own_weights)

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
    unit, position, address, item = model.weight_openings(run)[0]
    batch = list(units[unit])
    batch[position] = Opening(item.position, bytes((item.value[0] ^ 1,)) + item.value[1:], item.path)
    units[unit] = tuple(batch)
    tampered = replace(run.transcript, evidence=replace(run.transcript.evidence, units=tuple(units)))

    report = verify_transcript(encode_transcript(tampered), expectation, model.compiled)

    assert report.code is VerificationCode.INVALID_OPENING
    assert f"address {address}" in report.detail and f"owner {WEIGHT_OWNER}" in report.detail

    # ... and so does an opening moved to another rank
    batch[position] = Opening((item.position + 1) % 16, item.value, item.path)
    units[unit] = tuple(batch)
    moved = replace(run.transcript, evidence=replace(run.transcript.evidence, units=tuple(units)))
    report = verify_transcript(encode_transcript(moved), expectation, model.compiled)
    assert report.code is VerificationCode.COVERAGE_MISMATCH


def test_kappa_w_must_bind_exactly_the_circuits_weight_gates(model: Model) -> None:
    count = model.compiled.index.weight_count
    wrong_count = Weights(count + 1, model.weights.root)

    run = run_protocol(model.compiled, model.expectation(weights=wrong_count), model.values)
    assert run.report.code is VerificationCode.INVALID_COMPILED_RESULT
    assert "binds 17 weights" in run.report.detail

    unbound = run_protocol(model.compiled, model.expectation(weights=None), model.values)
    assert unbound.report.code is VerificationCode.INVALID_COMPILED_RESULT
    assert "no kappa_W" in unbound.report.detail

    with pytest.raises(ProtocolError, match="nonnegative"):
        Weights(-1, model.weights.root)
    # a weight vector of the wrong length commits fine (the model is committed before any
    # request) and is rejected by every circuit whose weight gates it does not count
    short, short_tree = commit_weights(GATE_SET, model.request.weight_values[:-1])
    assert short.count == 15
    rejected = run_protocol(
        model.compiled, model.expectation(weights=short), model.values, weight_tree=short_tree
    )
    assert rejected.report.code is VerificationCode.INVALID_COMPILED_RESULT
    assert "binds 15 weights" in rejected.report.detail
    header = VerifierSession(model.expectation(), model.compiled).header
    with pytest.raises(ProtocolError, match="does not match"):
        ProverSession(model.compiled, header, model.values, weight_tree=short_tree)


def test_the_weight_domain_is_the_rank_space_and_the_boundary_excludes_the_weight_gates(
    model: Model,
) -> None:
    index = model.compiled.index
    domain = weight_domain(GATE_SET, index.weight_count)
    boundary = index.boundary()
    weights = list(model.circuit.weights)

    assert domain.owner == WEIGHT_OWNER and domain.count == 16
    assert list(domain.positions) == list(range(16))  # ranks, not addresses
    assert domain == model.tree.domain == weight_domain(GATE_SET, 16)
    assert domain.domain_id != weight_domain(GATE_SET, 17).domain_id
    assert domain.domain_id != weight_domain(make_word_gate_set(16), 16).domain_id
    assert all(not boundary.contains(address) for address in weights)
    assert list(boundary)[: index.input_count] == list(model.circuit.inputs)
    assert list(index.weights()) == weights
    assert all(
        index.weights().rank(address) == model.circuit.weight_rank(address) == rank
        for rank, address in enumerate(weights)
    )
    with pytest.raises(KeyError):
        index.weights().rank(model.circuit.inputs[0])
    with pytest.raises(IndexError):
        domain.positions.unrank(16)
    with pytest.raises(ProtocolError, match="nonnegative"):
        weight_domain(GATE_SET, -1)
    with pytest.raises(ProtocolError, match="GateSet"):
        weight_domain(model.compiled, 16)  # type: ignore[arg-type]
    weightless = GateSet(
        (Gate("add", 2, 8, replay_cost=1, proof_cost=1, evaluate=lambda a: (a[0] + a[1]) % 256),),
        name="weightless",
        version="1",
    )
    with pytest.raises(ProtocolError, match="one width"):
        commit_weights(weightless, (1, 2, 3))


def test_ownership_rule_weights_then_boundary_then_interior(model: Model) -> None:
    session = VerifierSession(model.expectation(), model.compiled)
    layout = session._layout
    index = model.compiled.index

    assert all(layout.owner(address) == WEIGHT_OWNER for address in model.circuit.weights)
    assert all(layout.owner(address) == BOUNDARY_OWNER for address in layout.public_inputs)
    assert all(layout.owner(address) == BOUNDARY_OWNER for address in model.circuit.outputs)
    assert index.interior(0).count == index.interior(1).count == 0  # the source units
    interior = int(index.interior(2).unrank(0))
    assert layout.owner(interior) == 2


# -- the verifier's work does not grow with |W| ----------------------------------


class BoundaryPhase:
    """The verifier's setup and boundary phase for one ``n``, ready to rerun.

    With ``weights`` the model is the matmul under ``kappa_W``; without, the
    same product with the weights as ``in`` gates, so they are public inputs
    the verifier opens one by one.
    """

    def __init__(self, n: int, *, weights: bool) -> None:
        model = cached_model(n, 1)
        workload = model.request.workload
        if weights:
            compiled = model.compiled
            expectation = model.expectation()
            values, tree = model.values, model.tree
        else:
            public_inputs = (*workload.public_inputs, *workload.weight_values)
            compiled = Compiler(GATE_SET).compile(PublicWeightsG(8)(workload, b""), public_inputs)
            assert compiled.index.weight_count == 0
            values = dict(enumerate(compiled.circuit.evaluate(public_inputs)))
            expectation = make_expectation(
                compiled, CHECK_EVERYTHING, public_inputs, model.request.expected_outputs, **SEEDS
            )
            tree = None
        header = VerifierSession(expectation, compiled).header
        prover = ProverSession(compiled, header, values, weight_tree=tree)
        self.boundary = prover.boundary()
        self.compiled = compiled
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
