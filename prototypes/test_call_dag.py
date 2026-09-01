import dataclasses
import hashlib
import json
from itertools import product

import pytest

import prototypes.call_dag as dag


CELL_BITS = 8


def make_kernel() -> dag.Kernel:
    return dag.Kernel(
        cell_bits=CELL_BITS,
        gates=dag.trusted_word_gates(CELL_BITS),
    )


def make_batch(lengths: tuple[int, ...], seed: int = 1) -> dag.BatchInput:
    return dag.BatchInput(
        tuple(
            dag.make_demo_request(length, seed + index, CELL_BITS)
            for index, length in enumerate(lengths)
        )
    )


def construct_batch(
    kernel: dag.Kernel,
    constructor: dag.DemoG,
    batch: dag.BatchInput,
) -> dag.Construction:
    return dag.construct(
        kernel,
        constructor,
        batch,
        b"",
        input_cells=batch.cells(),
        advice_bound_bits=0,
    )


def encode_single_definition(body: dict[str, object]) -> bytes:
    digest = hashlib.sha256(dag._canonical_json(body)).hexdigest()
    return dag._canonical_json(
        {
            "version": dag.FORMAT_VERSION,
            "cell_bits": CELL_BITS,
            "definitions": [{"id": digest, "body": body}],
            "root": digest,
        }
    )


def all_leaf_paths(
    definition: dag.ValidatedDefinition,
    prefix: dag.OccurrencePath = (),
) -> tuple[dag.OccurrencePath, ...]:
    paths = []
    for step_index, step in enumerate(definition.steps):
        path = (*prefix, step_index)
        if isinstance(step, dag._ValidatedLeaf):
            paths.append(path)
        else:
            paths.extend(all_leaf_paths(step.child, path))
    return tuple(paths)


def occurrence_cuts(
    definition: dag.ValidatedDefinition,
    prefix: dag.OccurrencePath = (),
) -> tuple[tuple[dag.OccurrencePath, ...], ...]:
    if definition.gate_count == 0:
        return ((),)

    child_options = []
    for step_index, step in enumerate(definition.steps):
        if step.gate_count == 0:
            continue
        path = (*prefix, step_index)
        if isinstance(step, dag._ValidatedLeaf):
            child_options.append(((path,),))
        else:
            child_options.append(occurrence_cuts(step.child, path))

    decompositions = tuple(
        tuple(path for child_cut in choices for path in child_cut)
        for choices in product(*child_options)
    )
    return ((prefix,), *decompositions)


def flat_boundary_from_occurrence_paths(
    root: dag.ValidatedDefinition,
    flat: dag.FlatCircuit,
    selected_paths: tuple[dag.OccurrencePath, ...],
) -> tuple[int, ...]:
    leaf_paths = all_leaf_paths(root)
    assert len(leaf_paths) == len(flat.gates)
    owner_by_gate = {}
    for ordinal, leaf_path in enumerate(leaf_paths):
        owners = tuple(
            path
            for path in selected_paths
            if leaf_path[: len(path)] == path
        )
        assert len(owners) == 1
        owner_by_gate[ordinal] = owners[0]

    boundary = set(range(flat.input_count))
    boundary.update(flat.outputs)
    for gate in flat.gates:
        consumer = owner_by_gate[gate.ordinal]
        for read in gate.reads:
            if read < flat.input_count:
                boundary.add(read)
                continue
            producer = owner_by_gate[read - flat.input_count]
            if producer != consumer:
                boundary.add(read)
    return tuple(sorted(boundary))


def test_variable_batch_orders_reuse_definitions_on_both_sides():
    constructor = dag.DemoG(CELL_BITS)
    kernel = make_kernel()

    first = construct_batch(kernel, constructor, make_batch((4, 8, 4)))
    second = construct_batch(kernel, constructor, make_batch((8, 4, 8), seed=10))

    assert first.load.root.gate_count == 32
    assert first.load.root.cost == 48
    assert first.load.new_definitions == 4
    assert first.load.cache_hits == 0

    assert second.load.root.gate_count == 40
    assert second.load.root.cost == 60
    assert second.load.new_definitions == 1
    assert second.load.cache_hits == 3  # MAC, DOT4, and DOT8

    assert constructor.producer.trace_misses == 5
    assert constructor.producer.trace_hits == 4
    assert constructor.producer.unique_definition_count == 5
    assert kernel.cached_definition_count == 5


def test_constructor_output_is_pure_despite_memoization():
    batch = make_batch((4, 8, 4))
    warm_constructor = dag.DemoG(CELL_BITS)

    first = warm_constructor(batch, b"")
    second = warm_constructor(batch, b"")
    from_fresh_constructor = dag.DemoG(CELL_BITS)(batch, b"")

    assert first == second == from_fresh_constructor


@pytest.mark.parametrize("lengths", [(1,), (4, 8, 4), (8, 4, 8), (0, 3, 0)])
def test_structured_evaluation_matches_direct_modular_dot(lengths):
    constructor = dag.DemoG(CELL_BITS)
    kernel = make_kernel()
    batch = make_batch(lengths)
    construction = construct_batch(kernel, constructor, batch)

    assert kernel.evaluate(construction.load.root, batch.cells()) == (
        dag.expected_dot_outputs(batch, CELL_BITS)
    )


@pytest.mark.parametrize("lengths", [(4, 8, 4), (8, 4, 8), (0, 3, 0)])
def test_gate_at_matches_every_leaf_of_brute_force_flattening(lengths):
    constructor = dag.DemoG(CELL_BITS)
    kernel = make_kernel()
    construction = construct_batch(kernel, constructor, make_batch(lengths))
    flat = kernel.flatten(construction.load.root)

    assert tuple(
        kernel.gate_at(construction.load.root, ordinal)
        for ordinal in range(construction.load.root.gate_count)
    ) == flat.gates


def test_gate_at_does_not_require_flattening(monkeypatch):
    constructor = dag.DemoG(CELL_BITS)
    kernel = make_kernel()
    construction = construct_batch(kernel, constructor, make_batch((64, 128, 64)))
    root = construction.load.root

    def fail_if_flattened(*args, **kwargs):
        raise AssertionError("gate_at called flatten")

    monkeypatch.setattr(dag.Kernel, "flatten", fail_if_flattened)
    for ordinal in (0, root.gate_count // 2, root.gate_count - 1):
        gate = kernel.gate_at(root, ordinal)
        assert gate.ordinal == ordinal
        assert gate.write == root.input_count + ordinal


def test_reused_occurrences_receive_fresh_positions_and_input_contexts():
    constructor = dag.DemoG(CELL_BITS)
    kernel = make_kernel()
    construction = construct_batch(kernel, constructor, make_batch((2, 2)))
    flat = kernel.flatten(construction.load.root)
    q = flat.input_count

    assert q == 10
    assert len(flat.gates) == 8
    assert flat.gates[0] == dag.FlatGate(0, q, "mul", (1, 3))
    assert flat.gates[1] == dag.FlatGate(1, q + 1, "add", (0, q))
    assert flat.gates[4] == dag.FlatGate(4, q + 4, "mul", (6, 8))
    assert flat.gates[5] == dag.FlatGate(5, q + 5, "add", (5, q + 4))


def test_large_call_dag_contains_two_primitive_leaf_records():
    constructor = dag.DemoG(CELL_BITS)
    blob = constructor(make_batch((64, 128, 64)), b"")
    document = json.loads(blob)

    local_leaf_records = sum(
        step["kind"] == "leaf"
        for definition in document["definitions"]
        for step in definition["body"]["steps"]
    )
    root = make_kernel().load(blob).root

    assert local_leaf_records == 2  # mul and add live only in the MAC definition
    assert root.gate_count == 512  # but occurrences denote 512 conceptual leaves


def test_definition_summaries_track_only_inputs_read_by_internal_gates():
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
    kernel = make_kernel()
    root = kernel.load(encode_single_definition(body)).root

    assert root.required_inputs == (0, 1)
    assert kernel.occurrence_summary(root, ()) == dag.OccurrenceSummary(
        path=(),
        kind="root",
        definition_digest=root.digest,
        gate_start=0,
        gate_count=1,
        cost=1,
        external_reads=(0, 1),
        outputs=(2, 3),
    )


def test_dead_gate_and_pass_through_output_have_exact_nonroot_interface():
    producer = dag.Producer(CELL_BITS)

    @producer.gate(name="add")
    def add(left, right):
        return left + right

    @producer.circuit(key=("pass-through",), input_count=2)
    def pass_through(value, dead_input):
        add(dead_input, dead_input)
        return value

    @producer.circuit(key=("root",), input_count=2)
    def root_definition(left, right):
        produced = add(left, left)
        passed = pass_through(produced, right)
        return add(passed, right)

    kernel = make_kernel()
    root = kernel.load(producer.serialize(root_definition)).root
    child = root.steps[1]
    assert isinstance(child, dag._ValidatedCall)
    assert child.child.required_inputs == (1,)

    summary = kernel.occurrence_summary(root, (1,))
    plan = kernel.derive_replay_plan(root, ((0,), (1,), (2,)))

    assert summary.external_reads == (1,)
    assert summary.outputs == (2,)
    assert plan.boundary == (0, 1, 2, 4)
    assert 3 not in plan.boundary  # dead gate write remains inside the child
    assert plan.boundary == flat_boundary_from_occurrence_paths(
        root,
        kernel.flatten(root),
        ((0,), (1,), (2,)),
    )


def test_duplicate_and_pass_through_root_outputs_preserve_order_not_membership():
    body = {
        "input_count": 1,
        "steps": [
            {
                "kind": "leaf",
                "gate": "add",
                "args": [["input", 0], ["input", 0]],
            }
        ],
        "outputs": [
            ["step", 0, 0],
            ["step", 0, 0],
            ["input", 0],
        ],
    }
    kernel = make_kernel()
    root = kernel.load(encode_single_definition(body)).root

    plan = kernel.derive_replay_plan(root, ((),))

    assert plan.root_outputs == (1, 1, 0)
    assert plan.boundary == (0, 1)


def test_replay_plan_is_bound_to_the_exact_compiled_circuit():
    constructor = dag.DemoG(CELL_BITS)
    kernel = make_kernel()
    root = construct_batch(kernel, constructor, make_batch((2,))).load.root
    plan = kernel.derive_replay_plan(root, ((0,),))

    assert plan.root_digest == root.digest
    kernel.validate_replay_plan(root, plan)
    with pytest.raises(dag.KernelReject, match="does not match"):
        kernel.validate_replay_plan(
            root,
            dataclasses.replace(plan, root_outputs=(0,)),
        )


@pytest.mark.parametrize(
    "cut",
    [
        ((),),
        ((0,),),
        ((0, 0), (0, 1)),
        ((0, 0), (0, 1, 0), (0, 1, 1)),
    ],
)
def test_hierarchical_replay_boundary_matches_flat_cross_unit_oracle(cut):
    constructor = dag.DemoG(CELL_BITS)
    kernel = make_kernel()
    root = construct_batch(kernel, constructor, make_batch((2,))).load.root

    plan = kernel.derive_replay_plan(root, cut)

    assert plan.boundary == flat_boundary_from_occurrence_paths(
        root,
        kernel.flatten(root),
        cut,
    )


def test_all_primitive_occurrences_form_an_exact_replay_cut():
    constructor = dag.DemoG(CELL_BITS)
    kernel = make_kernel()
    root = construct_batch(kernel, constructor, make_batch((2, 1))).load.root
    paths = all_leaf_paths(root)

    plan = kernel.derive_replay_plan(root, paths)

    assert len(plan.units) == root.gate_count
    assert all(unit.kind == "leaf" and unit.gate_count == 1 for unit in plan.units)
    assert plan.boundary == flat_boundary_from_occurrence_paths(
        root,
        kernel.flatten(root),
        paths,
    )


def test_every_small_mixed_depth_cut_matches_independent_flat_oracle():
    constructor = dag.DemoG(CELL_BITS)
    kernel = make_kernel()
    root = construct_batch(kernel, constructor, make_batch((2, 1))).load.root
    flat = kernel.flatten(root)
    cuts = occurrence_cuts(root)

    assert len(cuts) == 16
    for cut in cuts:
        plan = kernel.derive_replay_plan(root, cut)
        assert plan.boundary == flat_boundary_from_occurrence_paths(root, flat, cut)


def test_unordered_valid_paths_are_canonicalized_by_gate_interval():
    constructor = dag.DemoG(CELL_BITS)
    kernel = make_kernel()
    root = construct_batch(kernel, constructor, make_batch((1, 1))).load.root
    cut = ((1,), (0,))

    plan = kernel.derive_replay_plan(root, cut)

    assert tuple(unit.path for unit in plan.units) == ((0,), (1,))
    assert plan.boundary == flat_boundary_from_occurrence_paths(
        root,
        kernel.flatten(root),
        cut,
    )


def test_zero_gate_siblings_need_no_replay_unit():
    constructor = dag.DemoG(CELL_BITS)
    kernel = make_kernel()
    root = construct_batch(kernel, constructor, make_batch((0, 1, 0))).load.root

    plan = kernel.derive_replay_plan(root, ((1,),))

    assert root.gate_count == 2
    assert tuple(unit.path for unit in plan.units) == ((1,),)
    assert plan.units[0].gate_count == 2


def test_reused_call_occurrences_have_fresh_interface_positions():
    constructor = dag.DemoG(CELL_BITS)
    kernel = make_kernel()
    root = construct_batch(kernel, constructor, make_batch((1, 1))).load.root

    first = kernel.occurrence_summary(root, (0,))
    second = kernel.occurrence_summary(root, (1,))

    assert first.definition_digest == second.definition_digest
    assert first.gate_start == 0
    assert second.gate_start == 2
    assert first.external_reads == (0, 1, 2)
    assert second.external_reads == (3, 4, 5)
    assert first.outputs == (7,)
    assert second.outputs == (9,)


def test_replay_plan_derivation_does_not_flatten(monkeypatch):
    constructor = dag.DemoG(CELL_BITS)
    kernel = make_kernel()
    root = construct_batch(kernel, constructor, make_batch((64, 128, 64))).load.root

    def fail_if_flattened(*args, **kwargs):
        raise AssertionError("derive_replay_plan called flatten")

    monkeypatch.setattr(dag.Kernel, "flatten", fail_if_flattened)
    plan = kernel.derive_replay_plan(root, ((0,), (1,), (2,)))

    assert tuple(unit.gate_count for unit in plan.units) == (128, 256, 128)
    assert plan.root_gate_count == 512


def test_replay_granularity_changes_expected_replay_not_the_gate_population():
    constructor = dag.DemoG(CELL_BITS)
    kernel = make_kernel()
    root = construct_batch(kernel, constructor, make_batch((2,))).load.root
    coarse = kernel.derive_replay_plan(root, ((),))
    fine = kernel.derive_replay_plan(root, all_leaf_paths(root))

    assert coarse.root_gate_count == fine.root_gate_count == 4
    assert coarse.expected_replay_cost(0.1) == pytest.approx(6 * (1 - 0.9**4))
    assert fine.expected_replay_cost(0.1) == pytest.approx(0.1 * 6)
    assert coarse.expected_replay_cost(0) == fine.expected_replay_cost(0) == 0
    assert coarse.expected_replay_cost(1) == fine.expected_replay_cost(1) == 6
    assert coarse.expected_two_stage_replay_cost(0.25) == pytest.approx(1.5)
    assert fine.expected_two_stage_replay_cost(0.25) == pytest.approx(1.5)
    assert coarse.expected_two_stage_checked_gates(0.25, 0.4) == pytest.approx(0.4)
    assert fine.expected_two_stage_checked_gates(0.25, 0.4) == pytest.approx(0.4)


def test_replay_plan_routes_secret_samples_and_derives_unit_interiors():
    constructor = dag.DemoG(CELL_BITS)
    kernel = make_kernel()
    root = construct_batch(kernel, constructor, make_batch((2,))).load.root
    plan = kernel.derive_replay_plan(root, ((0, 0), (0, 1)))

    assert tuple(plan.unit_index_for_gate(i) for i in range(4)) == (0, 0, 1, 1)
    assert plan.challenged_unit_indices((3, 0, 3)) == (0, 1)
    assert plan.interior_positions(0) == (5,)
    assert plan.interior_positions(1) == (7,)


@pytest.mark.parametrize(
    ("paths", "message"),
    [
        (((0,),), "leaves gate interval"),
        (((0,), (0, 0), (0, 1), (1,)), "overlap"),
        (((99,),), "missing step"),
        (((0,), (0,)), "unique"),
        (((0, 0, 0, 0),), "descends through"),
    ],
)
def test_invalid_replay_unit_cuts_are_rejected(paths, message):
    constructor = dag.DemoG(CELL_BITS)
    kernel = make_kernel()
    root = construct_batch(kernel, constructor, make_batch((2, 1))).load.root

    with pytest.raises(dag.KernelReject, match=message):
        kernel.derive_replay_plan(root, paths)


def test_empty_circuit_has_empty_unit_cut_and_public_io_boundary():
    constructor = dag.DemoG(CELL_BITS)
    kernel = make_kernel()
    root = construct_batch(kernel, constructor, make_batch((0,))).load.root

    plan = kernel.derive_replay_plan(root, ())

    assert plan.units == ()
    assert plan.root_gate_count == 0
    assert plan.root_outputs == (0,)
    assert plan.boundary == (0,)
    with pytest.raises(dag.KernelReject, match="contains no conceptual gates"):
        kernel.derive_replay_plan(root, ((),))


def test_advice_bound_is_checked_before_constructor_execution():
    constructor = dag.DemoG(CELL_BITS)
    kernel = make_kernel()
    batch = make_batch((2,))

    with pytest.raises(dag.KernelReject, match="advice exceeds"):
        dag.construct(
            kernel,
            constructor,
            batch,
            b"\x00",
            input_cells=batch.cells(),
            advice_bound_bits=0,
        )
    assert constructor.producer.trace_misses == 1  # only MAC from initialization


def test_unknown_gate_is_rejected():
    body = {
        "input_count": 1,
        "steps": [
            {
                "kind": "leaf",
                "gate": "arbitrary_python",
                "args": [["input", 0]],
            }
        ],
        "outputs": [["step", 0, 0]],
    }

    with pytest.raises(dag.KernelReject, match="unknown gate"):
        make_kernel().load(encode_single_definition(body))


def test_future_step_reference_is_rejected():
    body = {
        "input_count": 1,
        "steps": [
            {
                "kind": "leaf",
                "gate": "add",
                "args": [["step", 1, 0], ["input", 0]],
            },
            {
                "kind": "leaf",
                "gate": "add",
                "args": [["input", 0], ["input", 0]],
            },
        ],
        "outputs": [["step", 1, 0]],
    }

    with pytest.raises(dag.KernelReject, match="earlier step"):
        make_kernel().load(encode_single_definition(body))


def test_wrong_gate_arity_is_rejected():
    body = {
        "input_count": 1,
        "steps": [
            {
                "kind": "leaf",
                "gate": "add",
                "args": [["input", 0]],
            }
        ],
        "outputs": [["step", 0, 0]],
    }

    with pytest.raises(dag.KernelReject, match="expects 2 arguments"):
        make_kernel().load(encode_single_definition(body))


def test_tampered_body_with_stale_digest_is_rejected():
    constructor = dag.DemoG(CELL_BITS)
    document = json.loads(constructor(make_batch((2,)), b""))
    document["definitions"][0]["body"]["input_count"] += 1
    tampered = dag._canonical_json(document)

    with pytest.raises(dag.KernelReject, match="digest does not match"):
        make_kernel().load(tampered)


def test_noncanonical_extra_fields_are_rejected():
    body = {
        "input_count": 1,
        "steps": [],
        "outputs": [["input", 0]],
        "client_claimed_cost": 0,
    }

    with pytest.raises(dag.KernelReject, match="expected"):
        make_kernel().load(encode_single_definition(body))


def test_input_count_and_cell_width_are_enforced():
    constructor = dag.DemoG(CELL_BITS)
    kernel = make_kernel()
    batch = make_batch((2,))
    blob = constructor(batch, b"")
    root = kernel.load(blob).root

    with pytest.raises(dag.KernelReject, match="expects"):
        kernel.evaluate(root, batch.cells()[:-1])
    with pytest.raises(dag.KernelReject, match="8-bit word"):
        kernel.evaluate(root, (*batch.cells()[:-1], 256))
