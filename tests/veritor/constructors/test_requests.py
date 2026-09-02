"""The per-request constructor: no advice, one unit per request, the same tokens as the reference.

``RequestsG`` serves the requests of :mod:`test_cluster` in their own replay
units.  It generates what sequential decoding generates, shares kinds
between requests of the same shape, keeps every KV-cache value inside its
request (the boundary is the prompts, the tokens and the weights), takes no
advice, and goes through the protocol like the cluster does.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from veritor.compile import Compiler
from veritor.constructors import (
    LMShape,
    Request,
    RequestsG,
    TracerError,
    random_parameters,
    reference_generate,
)
from veritor.core import Compiled, VerificationPolicy, make_isa_gate_set
from veritor.core.description import REPLAY
from veritor.protocol import (
    VerificationCode,
    VerifierParameters,
    assignment_replay,
    commit_weights,
    make_expectation,
    run_protocol,
)
from veritor.research import Compile

SHAPE = LMShape(vocab=8, d_model=4, heads=2, layers=1, context=6, width=16)
REQUESTS = (Request((1, 2, 3), 3), Request((5,), 2), Request((7, 0), 4), Request((2, 2, 2), 1))
GATES = make_isa_gate_set(16)
SEEDS = {"session_id": b"requests-run", "q_seed": b"Q" * 32, "s_seed": b"S" * 32}


def compile_requests(constructor: RequestsG, requests: tuple[Request, ...]) -> Compiled:
    description, inputs = constructor(requests, b"")
    return Compiler(GATES).compile(description, inputs)


def generated(constructor: RequestsG, compiled: Compiled, requests, parameters) -> tuple[tuple[int, ...], ...]:
    values = compiled.circuit.evaluate(constructor.flatten_inputs(requests), parameters.flatten())
    outputs = [values[address] for address in compiled.circuit.outputs]
    grouped: list[list[int]] = [[] for _ in requests]
    for (request, position), token in zip(constructor.output_layout(requests), outputs, strict=True):
        assert position == len(grouped[request])
        grouped[request].append(token)
    return tuple(tuple(tokens) for tokens in grouped)


@pytest.fixture(scope="module")
def run() -> tuple[RequestsG, Compiled]:
    constructor = RequestsG(SHAPE)
    return constructor, compile_requests(constructor, REQUESTS)


def test_it_generates_what_the_reference_generates(run) -> None:
    constructor, compiled = run
    parameters = random_parameters(SHAPE, seed=1)

    assert generated(constructor, compiled, REQUESTS, parameters) == reference_generate(SHAPE, parameters, REQUESTS)


def test_one_replay_unit_per_request_and_the_weights(run) -> None:
    _, compiled = run
    index = compiled.index

    assert index.replay_units.count == 1 + len(REQUESTS)
    assert index.input_count == sum(len(r.prompt) for r in REQUESTS) == 9
    assert index.weight_count == SHAPE.weight_count
    # the boundary is the prompts, the tokens and nothing else: the cache stays inside
    assert index.boundary().count == 9 + sum(r.max_new for r in REQUESTS)
    requests = [row for row in index.kinds() if row.role == REPLAY and row.out_count > 0]
    assert sorted(row.out_count for row in requests) == [1, 2, 3, 4]
    assert all(row.input_count == SHAPE.weight_count for row in requests)


def test_requests_of_one_shape_are_one_kind() -> None:
    constructor = RequestsG(SHAPE)
    same = (Request((1, 2), 3), Request((4, 5), 3), Request((0, 7), 3))

    compiled = compile_requests(constructor, same)

    units = [row for row in compiled.index.kinds() if row.role == REPLAY and row.out_count > 0]
    assert len(units) == 1 and units[0].copies == 3


def test_it_takes_no_advice_and_checks_its_requests() -> None:
    constructor = RequestsG(SHAPE)

    with pytest.raises(TracerError, match="no advice"):
        constructor(REQUESTS, b"x")
    with pytest.raises(TracerError, match="outside the vocabulary"):
        constructor((Request((8,), 1),), b"")
    with pytest.raises(TracerError, match="the context is 6"):
        constructor((Request((1, 2, 3), 4),), b"")
    with pytest.raises(TracerError, match="nonempty tuple"):
        constructor((), b"")


def test_the_digest_names_the_shape() -> None:
    assert RequestsG(SHAPE).digest == RequestsG(SHAPE).digest
    other = LMShape(vocab=8, d_model=4, heads=2, layers=2, context=6, width=16)
    assert RequestsG(SHAPE).digest != RequestsG(other).digest


@pytest.fixture(scope="module")
def deployment():
    parameters = random_parameters(SHAPE, seed=42)
    weights, tree = commit_weights(GATES, parameters.flatten())
    constructor = RequestsG(SHAPE)
    compilation = Compile(constructor, REQUESTS, b"", GATES)
    circuit = compilation.compiled.circuit
    values = dict(enumerate(circuit.evaluate(compilation.inputs, parameters.flatten())))
    outputs = tuple(values[address] for address in circuit.outputs)
    return constructor, compilation, weights, tree, values, outputs


class TestProtocol:
    def expectation(self, deployment, policy, outputs=None):
        _, compilation, weights, _, _, honest = deployment
        return make_expectation(
            compilation,
            policy,
            honest if outputs is None else outputs,
            weights=weights,
            parameters=VerifierParameters(max_capacity=None),
            **SEEDS,
        )

    @pytest.mark.parametrize("policy", (VerificationPolicy(1, 1), VerificationPolicy(Fraction(1, 2), Fraction(1, 3))))
    def test_an_honest_run_is_accepted_with_empty_advice(self, deployment, policy) -> None:
        constructor, compilation, _, tree, values, _ = deployment

        run = run_protocol(compilation.compiled, self.expectation(deployment, policy), values, weight_tree=tree)

        assert run.report.accepted and run.transcript is not None
        assert run.transcript.header.advice == b"" and compilation.advice_bits == 0
        assert run.transcript.header.constructor == constructor.digest
        assert run.transcript.boundary.commitment.count == 9 + 10

    def test_a_corrupted_token_is_caught_at_the_boundary(self, deployment) -> None:
        _, compilation, _, tree, values, outputs = deployment
        claimed = list(outputs)
        claimed[3] = (claimed[3] + 1) % SHAPE.vocab

        run = run_protocol(
            compilation.compiled,
            self.expectation(deployment, VerificationPolicy(1, 1), tuple(claimed)),
            values,
            weight_tree=tree,
        )

        assert not run.report.accepted and run.report.code is VerificationCode.PUBLIC_IO_MISMATCH

    def test_a_corrupted_interior_value_is_caught_when_everything_is_checked(self, deployment) -> None:
        _, compilation, _, tree, values, _ = deployment
        circuit = compilation.compiled.circuit
        interior = next(
            address
            for address in range(circuit.n)
            if address not in set(circuit.inputs) | set(circuit.outputs) | set(circuit.weights)
        )
        tampered = dict(values)
        tampered[interior] = (tampered[interior] + 1) % (1 << 16)

        run = run_protocol(
            compilation.compiled,
            self.expectation(deployment, VerificationPolicy(1, 1)),
            values,
            weight_tree=tree,
            replay=assignment_replay(tampered),
        )

        assert not run.report.accepted and run.report.code is VerificationCode.RELATION_REJECTED
