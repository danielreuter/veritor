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


def test_the_requests_and_the_weights_are_closed_and_the_decode_steps_are_not(run) -> None:
    """A request is handed the weights and holds its prompt: replayable from what the server retains.

    A decode step reads the previous token and the KV cache, which the
    server does not keep; so does everything under it.
    """

    constructor, compiled = run
    kinds = compiled.index.kinds()
    by_kind = {row.kind: row for row in kinds}
    by_key = constructor.lm.tracer._by_key  # the toy's kinds by the keys ``lm.py`` gives them

    def closed(*keys: object) -> list[bool]:
        return [by_kind[by_key[key].digest].closed for key in keys]

    requests = [row for row in kinds if row.role == REPLAY and row.out_count > 0]
    (weights,) = [row for row in kinds if row.role == REPLAY and row.source_weights > 0]
    assert weights.closed and all(row.closed for row in requests)
    assert closed("weights", ("request", 3, 3), ("request", 1, 2), ("prefill", 3), ("prefill", 1)) == [True] * 5
    assert closed(("decode", 4), ("decode", 5), ("decode", 2)) == [False] * 3
    # the kinds shared between the prefill and the decode steps (an embedding row, a matvec) are open
    assert closed("embed_row", ("matvec", SHAPE.d_model, SHAPE.d_model), "argmax") == [False] * 3


def test_requests_of_one_shape_are_one_kind() -> None:
    constructor = RequestsG(SHAPE)
    same = (Request((1, 2), 3), Request((4, 5), 3), Request((0, 7), 3))

    compiled = compile_requests(constructor, same)

    units = [row for row in compiled.index.kinds() if row.role == REPLAY and row.out_count > 0]
    assert len(units) == 1 and units[0].copies == 3


def test_requests_are_grouped_by_kind_so_the_root_has_one_output_run_per_shape() -> None:
    """Kinds in order of first appearance, each group one ``repeat``; the layouts follow the circuit order."""

    constructor = RequestsG(SHAPE)
    mixed = (Request((1, 2), 3), Request((5,), 2), Request((4, 4), 3), Request((0,), 2), Request((7, 7, 7), 1))
    parameters = random_parameters(SHAPE, seed=5)

    assert constructor.groups(mixed) == (((2, 3, 0), (0, 2)), ((1, 2, 0), (1, 3)), ((3, 1, 0), (4,)))
    assert constructor.order(mixed) == (0, 2, 1, 3, 4)
    assert constructor.output_layout(mixed) == (
        (0, 0), (0, 1), (0, 2), (2, 0), (2, 1), (2, 2), (1, 0), (1, 1), (3, 0), (3, 1), (4, 0),
    )
    assert constructor.flatten_inputs(mixed) == (1, 2, 4, 4, 5, 0, 7, 7, 7)
    compiled = compile_requests(constructor, mixed)
    assert generated(constructor, compiled, mixed, parameters) == reference_generate(SHAPE, parameters, mixed)
    # one run per generated position of each kind (3 + 2 + 1), not per request (3 + 2 + 3 + 2 + 1)
    runs = compiled.index.root.frame.definition.out_runs
    assert len(runs) == 6 and sorted(run.count for run in runs) == [1, 2, 2, 2, 2, 2]
    units = {row.copies for row in compiled.index.kinds() if row.role == REPLAY and row.out_count > 0}
    assert units == {2, 1}


def test_banned_tokens_are_masked_in_circuit_and_never_generated() -> None:
    """Constrained requests are their own kinds; the mask is ``allowed_row`` units over public ``in`` gates."""

    constructor = RequestsG(SHAPE)
    parameters = random_parameters(SHAPE, seed=9)
    requests = (Request((1, 2, 3), 3, banned=(0, 4, 5)), Request((5,), 2), Request((2, 6), 3, banned=(7,)))

    compiled = compile_requests(constructor, requests)
    tokens = generated(constructor, compiled, requests, parameters)
    assert tokens == reference_generate(SHAPE, parameters, requests)
    assert not set(tokens[0]) & {0, 4, 5} and 7 not in tokens[2]
    assert constructor.flatten_inputs(requests) == (0, 4, 5, 1, 2, 3, 5, 7, 2, 6)  # banned ids precede the prompt
    kinds = {row.kind: row for row in compiled.index.kinds()}
    lm = constructor.lm
    assert kinds[lm.allowed_row(3).digest].copies == SHAPE.vocab and kinds[lm.allowed_row(1).digest].copies == SHAPE.vocab
    assert kinds[lm.masked_argmax().digest].copies == 6 and kinds[lm.argmax().digest].copies == 2
    assert lm.allowed_row(3).role == "verification"
    assert {(row.out_count, row.input_count) for row in kinds.values() if row.role == REPLAY and row.out_count} == {
        (3, SHAPE.weight_count), (2, SHAPE.weight_count),
    }
    with pytest.raises(TracerError, match="below vocab"):
        constructor((Request((1,), 1, banned=(8,)),), b"")
    with pytest.raises(TracerError, match="at least one allowed"):
        constructor((Request((1,), 1, banned=tuple(range(8))),), b"")


@pytest.mark.parametrize("tensor_parallel", (2, 4))
def test_tensor_parallel_changes_the_dot_kinds_and_nothing_the_verifier_sees_in_the_tokens(tensor_parallel: int) -> None:
    parameters = random_parameters(SHAPE, seed=11)
    plain, sharded = RequestsG(SHAPE), RequestsG(SHAPE, tensor_parallel=tensor_parallel)

    compiled_plain, compiled_sharded = compile_requests(plain, REQUESTS), compile_requests(sharded, REQUESTS)
    assert generated(sharded, compiled_sharded, REQUESTS, parameters) == generated(plain, compiled_plain, REQUESTS, parameters)
    assert compiled_sharded.digest != compiled_plain.digest and sharded.digest != plain.digest
    assert sharded.manifest == {"shape": SHAPE.manifest, "tensor_parallel": tensor_parallel}
    verification = lambda compiled: {row.kind for row in compiled.index.kinds() if row.role == "verification"}
    assert verification(compiled_sharded).isdisjoint({plain.lm.dot(k).digest for k in (SHAPE.d_model, SHAPE.hidden, SHAPE.vocab)})
    # the units that do not contain a marked dot are unchanged
    assert {plain.lm.argmax().digest, plain.lm.onehot().digest} <= verification(compiled_sharded)


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
