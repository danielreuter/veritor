from __future__ import annotations

import dataclasses
import json

import pytest

from veritor.compile import (
    FORMAT_VERSION,
    CallDagCircuit,
    CompilationLimits,
    FlatGate,
    GateSpec,
    Kernel,
    KernelReject,
    OccurrenceSummary,
    Producer,
    SemanticRegistry,
    ValidatedCall,
    ValidatedLeaf,
    canonical_call_dag_json,
    construct,
    definition_digest,
    make_word_kernel,
    trusted_word_registry,
)
from veritor.core import ExecutableCircuit, validate_circuit_contract
from veritor.plugins import BatchInput, DemoG, expected_dot_outputs, make_demo_request

CELL_BITS = 8


def make_batch(lengths: tuple[int, ...], seed: int = 1) -> BatchInput:
    return BatchInput(
        tuple(
            make_demo_request(length, seed + index, CELL_BITS)
            for index, length in enumerate(lengths)
        )
    )


def load_batch(
    kernel: Kernel,
    constructor: DemoG,
    lengths: tuple[int, ...],
    *,
    seed: int = 1,
):
    batch = make_batch(lengths, seed)
    construction = construct(
        kernel,
        constructor,
        batch,
        b"",
        input_cells=batch.cells(),
        advice_bound_bits=0,
    )
    return batch, construction.load.root


def encode_document(
    definitions: list[tuple[str, dict[str, object]]],
    root: str,
    *,
    cell_bits: int = CELL_BITS,
) -> bytes:
    return canonical_call_dag_json(
        {
            "version": FORMAT_VERSION,
            "cell_bits": cell_bits,
            "definitions": [
                {"id": digest, "body": body} for digest, body in definitions
            ],
            "root": root,
        }
    )


def encode_single_definition(
    body: dict[str, object],
    *,
    cell_bits: int = CELL_BITS,
) -> bytes:
    digest = definition_digest(body)
    return encode_document([(digest, body)], digest, cell_bits=cell_bits)


def test_memoized_constructor_reuses_definitions_without_changing_bytes():
    constructor = DemoG(CELL_BITS)
    kernel = make_word_kernel(CELL_BITS)
    first_batch = make_batch((4, 8, 4))
    first_blob = constructor(first_batch, b"")

    first = kernel.load(first_blob)
    repeated = constructor(first_batch, b"")
    second = kernel.load(constructor(make_batch((8, 4, 8), seed=10), b""))

    assert repeated == first_blob == DemoG(CELL_BITS)(first_batch, b"")
    assert first.root.gate_count == 32
    assert first.root.cost == 48
    assert first.new_definitions == 4
    assert second.root.gate_count == 40
    assert second.new_definitions == 1
    assert second.cache_hits == 3
    assert constructor.producer.trace_misses == 5
    assert constructor.producer.trace_hits == 5
    assert kernel.cached_definition_count == 5


@pytest.mark.parametrize("lengths", [(1,), (4, 8, 4), (0, 3, 0)])
def test_evaluation_and_lazy_gate_lookup_match_flat_reference(lengths):
    kernel = make_word_kernel(CELL_BITS)
    batch, root = load_batch(kernel, DemoG(CELL_BITS), lengths)
    flat = kernel.flatten(root)

    assert kernel.evaluate(root, batch.cells()) == expected_dot_outputs(
        batch,
        CELL_BITS,
    )
    assert tuple(kernel.gate_at(root, ordinal) for ordinal in range(root.gate_count)) == (
        flat.gates
    )


def test_gate_at_is_lazy_and_preserves_fresh_occurrence_wiring(monkeypatch):
    kernel = make_word_kernel(CELL_BITS)
    _, root = load_batch(kernel, DemoG(CELL_BITS), (2, 2))
    flat = kernel.flatten(root)
    input_count = root.input_count

    assert flat.gates[0] == FlatGate(0, input_count, "mul", (1, 3))
    assert flat.gates[1] == FlatGate(1, input_count + 1, "add", (0, input_count))
    assert flat.gates[4] == FlatGate(4, input_count + 4, "mul", (6, 8))

    def fail_if_flattened(*_args, **_kwargs):
        raise AssertionError("gate_at called flatten")

    monkeypatch.setattr(Kernel, "flatten", fail_if_flattened)
    for ordinal in (0, root.gate_count // 2, root.gate_count - 1):
        assert kernel.gate_at(root, ordinal).ordinal == ordinal


def test_recursive_occurrence_summaries_track_exact_interfaces():
    kernel = make_word_kernel(CELL_BITS)
    _, root = load_batch(kernel, DemoG(CELL_BITS), (1, 1))
    first = kernel.occurrence_summary(root, (0,))
    second = kernel.occurrence_summary(root, (1,))

    assert first.definition_digest == second.definition_digest
    assert first.gate_start == 0
    assert second.gate_start == 2
    assert first.external_reads == (0, 1, 2)
    assert second.external_reads == (3, 4, 5)
    assert first.outputs == (7,)
    assert second.outputs == (9,)
    assert len(kernel.leaf_occurrence_paths(root)) == root.gate_count


def test_definition_summary_ignores_output_only_inputs():
    body = {
        "input_count": 3,
        "steps": [
            {
                "kind": "leaf",
                "gate": "add",
                "args": [["input", 0], ["input", 1]],
            }
        ],
        "outputs": [["input", 2], ["step", 0, 0]],
    }
    kernel = make_word_kernel(CELL_BITS)
    root = kernel.load(encode_single_definition(body)).root

    assert root.required_inputs == (0, 1)
    assert kernel.occurrence_summary(root, ()) == OccurrenceSummary(
        path=(),
        kind="root",
        definition_digest=root.digest,
        gate_start=0,
        gate_count=1,
        cost=1,
        external_reads=(0, 1),
        outputs=(2, 3),
    )


def test_replay_plan_is_fully_rederived_and_identity_bound_to_root():
    kernel = make_word_kernel(CELL_BITS)
    _, root = load_batch(kernel, DemoG(CELL_BITS), (2,))
    plan = kernel.derive_replay_plan(root, ((0,),))

    kernel.validate_replay_plan(root, plan)
    with pytest.raises(KernelReject, match="does not match"):
        kernel.validate_replay_plan(
            root,
            dataclasses.replace(plan, root_outputs=(0,)),
        )
    with pytest.raises(KernelReject, match="overlap"):
        kernel.derive_replay_plan(root, ((0,), (0, 0), (0, 1)))
    with pytest.raises(KernelReject, match="leaves gate interval"):
        kernel.derive_replay_plan(root, ((0, 0),))
    with pytest.raises(KernelReject, match="unique"):
        kernel.derive_replay_plan(root, ((0,), (0,)))


def test_adapter_implements_executable_contract_and_trusted_codec():
    producer = Producer(CELL_BITS)

    @producer.gate(name="add")
    def add(left, right):
        return left + right

    @producer.circuit(key="duplicate-read", input_count=1)
    def root(value):
        doubled = add(value, value)
        return doubled, doubled, value

    kernel = make_word_kernel(CELL_BITS)
    circuit = CallDagCircuit(kernel, kernel.load(producer.serialize(root)).root)

    assert isinstance(circuit, ExecutableCircuit)
    validate_circuit_contract(circuit, exhaustive=True)
    assert circuit.computed_positions.start == 1
    assert circuit.computed_positions.stop == 2
    assert circuit.gate_at(1).predecessors == (0, 0)
    assert circuit.gate_at(1).capacity_upper_bound == 2**CELL_BITS
    executable = circuit.executable_gate_at(1)
    assert executable.arguments == (0, 0)
    assert circuit.evaluate_relation(executable.relation_id, (7, 7)) == 14
    assert circuit.value_codec.decode(circuit.value_codec.encode(255)) == 255
    assert circuit.ordered_output_positions == (1, 1, 0)
    assert circuit.evaluate((7,)) == (14, 14, 7)


def test_semantic_identity_binds_declarations_not_python_callable_objects():
    producer = Producer(CELL_BITS)

    @producer.gate(name="add")
    def add(left, right):
        return left + right

    @producer.circuit(key="root", input_count=1)
    def root(value):
        return add(value, value)

    blob = producer.serialize(root)

    def registry(evaluator, *, version="1", cost=1):
        return SemanticRegistry(
            registry_id="tests.explicit-semantics",
            registry_version=version,
            value_schema_id="tests.word",
            value_schema_version="1",
            gates=(GateSpec("add", 2, cost, evaluator),),
        )

    first_kernel = Kernel(
        cell_bits=CELL_BITS,
        semantic_registry=registry(lambda args: sum(args) & 255),
    )
    first = CallDagCircuit(first_kernel, first_kernel.load(blob).root)
    same_declaration_kernel = Kernel(
        cell_bits=CELL_BITS,
        semantic_registry=registry(lambda args: (args[0] - args[1]) & 255),
    )
    same_declaration = CallDagCircuit(
        same_declaration_kernel,
        same_declaration_kernel.load(blob).root,
    )
    new_version_kernel = Kernel(
        cell_bits=CELL_BITS,
        semantic_registry=registry(lambda args: sum(args) & 255, version="2"),
    )
    new_version = CallDagCircuit(new_version_kernel, new_version_kernel.load(blob).root)
    new_cost_kernel = Kernel(
        cell_bits=CELL_BITS,
        semantic_registry=registry(lambda args: sum(args) & 255, cost=9),
    )
    new_cost = CallDagCircuit(new_cost_kernel, new_cost_kernel.load(blob).root)

    assert first.identity == same_declaration.identity
    assert first.identity != new_version.identity
    assert first.identity != new_cost.identity


def test_adapter_rejects_definition_from_another_kernel_instance():
    producer = Producer(CELL_BITS)

    @producer.gate(name="add")
    def add(left, right):
        return left + right

    @producer.circuit(key="root", input_count=1)
    def root(value):
        return add(value, value)

    source_kernel = make_word_kernel(CELL_BITS)
    foreign_root = source_kernel.load(producer.serialize(root)).root

    with pytest.raises(KernelReject, match="this kernel instance"):
        CallDagCircuit(make_word_kernel(CELL_BITS), foreign_root)


def test_cell_width_is_identity_bound_even_for_identical_definition_body():
    circuits = []
    for cell_bits in (8, 16):
        producer = Producer(cell_bits)

        @producer.gate(name="add")
        def add(left, right):
            return left + right

        @producer.circuit(key="root", input_count=1)
        def root(value):
            return add(value, value)

        kernel = Kernel(
            cell_bits=cell_bits,
            semantic_registry=trusted_word_registry(cell_bits),
        )
        validated = kernel.load(producer.serialize(root)).root
        circuits.append(CallDagCircuit(kernel, validated))

    assert circuits[0].root.digest == circuits[1].root.digest
    assert circuits[0].identity.representation_digest == (
        circuits[1].identity.representation_digest
    )
    assert circuits[0].identity.value_registry_digest != (
        circuits[1].identity.value_registry_digest
    )
    assert circuits[0].identity != circuits[1].identity


def test_load_rejection_is_atomic_after_validating_new_definitions():
    kernel = make_word_kernel(CELL_BITS)
    warm_body = {
        "input_count": 1,
        "steps": [],
        "outputs": [["input", 0]],
    }
    kernel.load(encode_single_definition(warm_body))
    before = kernel.cached_digests

    staged_body = {
        "input_count": 1,
        "steps": [
            {
                "kind": "leaf",
                "gate": "add",
                "args": [["input", 0], ["input", 0]],
            }
        ],
        "outputs": [["step", 0, 0]],
    }
    rejected_body = {
        "input_count": 1,
        "steps": [
            {
                "kind": "leaf",
                "gate": "not-trusted",
                "args": [["input", 0]],
            }
        ],
        "outputs": [["step", 0, 0]],
    }
    staged_digest = definition_digest(staged_body)
    rejected_digest = definition_digest(rejected_body)
    blob = encode_document(
        [(staged_digest, staged_body), (rejected_digest, rejected_body)],
        staged_digest,
    )

    with pytest.raises(KernelReject, match="unknown gate"):
        kernel.load(blob)
    assert kernel.cached_digests == before
    assert staged_digest not in kernel.cached_digests


def test_unavailable_root_rejection_does_not_commit_staged_definition():
    kernel = make_word_kernel(CELL_BITS)
    body = {
        "input_count": 1,
        "steps": [],
        "outputs": [["input", 0]],
    }
    digest = definition_digest(body)

    with pytest.raises(KernelReject, match="root definition is unavailable"):
        kernel.load(encode_document([(digest, body)], "0" * 64))
    assert kernel.cached_definition_count == 0


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            {
                "input_count": 1,
                "steps": [
                    {
                        "kind": "leaf",
                        "gate": "unknown",
                        "args": [["input", 0]],
                    }
                ],
                "outputs": [["step", 0, 0]],
            },
            "unknown gate",
        ),
        (
            {
                "input_count": 1,
                "steps": [
                    {
                        "kind": "leaf",
                        "gate": "add",
                        "args": [["step", 0, 0], ["input", 0]],
                    }
                ],
                "outputs": [["step", 0, 0]],
            },
            "earlier step",
        ),
        (
            {
                "input_count": 1,
                "steps": [
                    {
                        "kind": "leaf",
                        "gate": "add",
                        "args": [["input", 0]],
                    }
                ],
                "outputs": [["step", 0, 0]],
            },
            "expects 2 arguments",
        ),
        (
            {
                "input_count": 1,
                "steps": [],
                "outputs": [["input", 0]],
                "untrusted_cost": 0,
            },
            "expected",
        ),
    ],
)
def test_malformed_definition_data_is_rejected(body, message):
    with pytest.raises(KernelReject, match=message):
        make_word_kernel(CELL_BITS).load(encode_single_definition(body))


def test_digest_duplicate_key_canonical_form_and_cell_width_are_enforced():
    kernel = make_word_kernel(CELL_BITS)
    body = {
        "input_count": 1,
        "steps": [],
        "outputs": [["input", 0]],
    }
    digest = definition_digest(body)
    document = json.loads(encode_document([(digest, body)], digest))
    document["definitions"][0]["body"]["input_count"] = 2
    with pytest.raises(KernelReject, match="digest does not match"):
        kernel.load(canonical_call_dag_json(document))

    noncanonical = json.dumps(
        {
            "version": FORMAT_VERSION,
            "cell_bits": CELL_BITS,
            "definitions": [{"id": digest, "body": body}],
            "root": digest,
        },
        indent=2,
    ).encode()
    with pytest.raises(KernelReject, match="canonically serialized"):
        kernel.load(noncanonical)

    duplicate_key = (
        b'{"cell_bits":8,"definitions":[],"root":"'
        + b"0" * 64
        + b'","root":"'
        + b"0" * 64
        + b'","version":1}'
    )
    with pytest.raises(KernelReject, match="duplicate JSON key"):
        kernel.load(duplicate_key)

    whitespace_digest = ("0 " * 32)
    with pytest.raises(KernelReject, match="not hexadecimal"):
        kernel.load(encode_document([(whitespace_digest, body)], whitespace_digest))

    with pytest.raises(KernelReject, match="16-bit cells"):
        kernel.load(encode_single_definition(body, cell_bits=16))


def test_resource_limits_reject_before_cache_commit():
    limits = CompilationLimits(max_steps_per_definition=0)
    kernel = make_word_kernel(CELL_BITS, limits=limits)
    body = {
        "input_count": 1,
        "steps": [
            {
                "kind": "leaf",
                "gate": "add",
                "args": [["input", 0], ["input", 0]],
            }
        ],
        "outputs": [["step", 0, 0]],
    }

    with pytest.raises(KernelReject, match="too many local steps"):
        kernel.load(encode_single_definition(body))
    assert kernel.cached_definition_count == 0


def test_zero_gate_definition_evaluates_passthrough_and_has_no_leaves():
    body = {
        "input_count": 1,
        "steps": [],
        "outputs": [["input", 0], ["input", 0]],
    }
    kernel = make_word_kernel(CELL_BITS)
    root = kernel.load(encode_single_definition(body)).root

    assert root.gate_count == 0
    assert kernel.flatten(root).gates == ()
    assert kernel.leaf_occurrence_paths(root) == ()
    assert kernel.evaluate(root, (13,)) == (13, 13)
    assert tuple(kernel.derive_replay_plan(root, ()).boundary) == (0,)
    with pytest.raises(KernelReject, match="contains no conceptual gates"):
        kernel.derive_replay_plan(root, ((),))


def test_validated_call_and_leaf_types_are_publicly_inspectable():
    kernel = make_word_kernel(CELL_BITS)
    _, root = load_batch(kernel, DemoG(CELL_BITS), (1,))

    assert isinstance(root.steps[0], ValidatedCall)
    dot = root.steps[0].child
    assert isinstance(dot.steps[0], ValidatedCall)
    mac = dot.steps[0].child
    assert isinstance(mac.steps[0], ValidatedLeaf)
    assert isinstance(mac.steps[1], ValidatedLeaf)

