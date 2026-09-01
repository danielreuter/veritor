import json
import math

import pytest

from prototypes.noop_envelope import (
    ANCHOR_WORKLOAD,
    CELL_BITS,
    IMPLEMENTATION_SCOPE,
    PROTOCOL_INVARIANTS,
    RUNTIME_FAILURE_HANDLING,
    SP1_CALIBRATION,
    STRATEGIES,
    TOKEN_ENVELOPE_SEMANTICS,
    EnvelopeError,
    GateCoefficients,
    SamplingMerkleParameters,
    WorkloadShape,
    K_B,
    canonical_json,
    capacity_bound_penalty,
    capacity_bound_penalty_bits,
    challenged_unit_probability,
    conceptual_gate_accounting,
    default_anchor_sweeps,
    direct_merkle_opening_bytes,
    envelope_counts,
    expected_sampled_noop_gates,
    length_advice_comparison_bits,
    main,
    naive_exact_padding_opening_bytes,
    padding_communication_semantics,
    range_multiproof_proxy_bytes,
    run_experiment,
    sampling_sweep,
    sparse_capacity_penalty_approximation_bits,
    sp1_constant_noop_local_check_projection,
    zero_commitment_comparison,
)


def tiny_workload() -> WorkloadShape:
    return WorkloadShape(
        requests=2,
        actual_output_tokens_per_request=3,
        max_output_tokens_per_request=5,
        prefill_recomputation_tokens=7,
        moe_layers=2,
        routed_experts=4,
        top_k_experts=1,
    )


def tiny_merkle(*, p: float = 0.25) -> SamplingMerkleParameters:
    return SamplingMerkleParameters(
        p=p,
        instruction_bytes=11,
        cell_bytes=4,
        hash_bytes=32,
        instruction_depth=2,
        value_depth=3,
        arity=2,
    )


def test_anchor_exact_counts_match_hand_calculation():
    counts = envelope_counts(ANCHOR_WORKLOAD)

    assert counts.length_advice_comparison_bits == 1_280
    assert counts.actual_decode_tokens == 128 * 900 == 115_200
    assert counts.max_decode_tokens == 128 * 1_024 == 131_072
    assert counts.padded_decode_tokens == 15_872
    assert counts.response_padding_bytes == 15_872 * 4 == 63_488
    assert counts.materialized_token_cell_padding_bytes == 63_488
    assert (
        counts.server_to_receiver_padding_bytes_if_fixed_width_wire == 63_488
    )
    assert (
        counts.server_to_receiver_padding_bytes_if_local_canonicalization == 0
    )
    assert counts.actual_model_tokens == 912_224 + 115_200 == 1_027_424
    assert counts.envelope_model_tokens == 912_224 + 131_072 == 1_043_296
    assert counts.total_model_token_inflation_tokens == 15_872

    active_layer_tokens = 1_027_424 * 60
    padded_layer_tokens = 15_872 * 60
    assert counts.active_moe_layer_token_occurrences == active_layer_tokens
    assert counts.padded_moe_layer_token_occurrences == padded_layer_tokens
    assert counts.active_top_k_calls == active_layer_tokens * 8
    assert (
        counts.unselected_branches_on_active_tokens
        == active_layer_tokens * (384 - 8)
    )
    assert (
        counts.all_branches_on_padded_tokens
        == padded_layer_tokens * 384
    )
    assert counts.total_noop_expert_branches == 23_544_376_320
    assert (
        counts.total_envelope_expert_branches
        == counts.active_top_k_calls + counts.total_noop_expert_branches
    )


def test_tiny_workload_formulas_are_hand_checkable():
    counts = envelope_counts(tiny_workload())

    assert length_advice_comparison_bits(5, 2) == math.ceil(math.log2(5**2)) == 5
    assert counts.actual_decode_tokens == 6
    assert counts.max_decode_tokens == 10
    assert counts.padded_decode_tokens == 4
    assert counts.response_padding_bytes == 16
    assert counts.materialized_token_cell_padding_bytes == 16
    assert counts.server_to_receiver_padding_bytes_if_fixed_width_wire == 16
    assert counts.server_to_receiver_padding_bytes_if_local_canonicalization == 0
    assert counts.actual_model_tokens == 13
    assert counts.envelope_model_tokens == 17
    assert counts.active_moe_layer_token_occurrences == 26
    assert counts.padded_moe_layer_token_occurrences == 8
    assert counts.active_top_k_calls == 26
    assert counts.unselected_branches_on_active_tokens == 78
    assert counts.all_branches_on_padded_tokens == 32
    assert counts.total_noop_expert_branches == 110
    assert counts.total_envelope_expert_branches == 136
    assert counts.noop_fraction == pytest.approx(110 / 136)
    assert counts.noop_branches_per_useful_expert_call == pytest.approx(110 / 26)
    assert counts.expansion_over_useful_expert_calls == pytest.approx(136 / 26)


def test_count_monotonicity_for_caps_requests_and_experts():
    base = tiny_workload()
    larger_cap = WorkloadShape(
        **{
            **base.__dict__,
            "max_output_tokens_per_request": 6,
        }
    )
    more_requests = WorkloadShape(
        **{
            **base.__dict__,
            "requests": 4,
        }
    )
    more_experts = WorkloadShape(
        **{
            **base.__dict__,
            "routed_experts": 8,
        }
    )

    base_counts = envelope_counts(base)
    assert (
        envelope_counts(larger_cap).total_noop_expert_branches
        > base_counts.total_noop_expert_branches
    )
    assert (
        envelope_counts(more_requests).total_noop_expert_branches
        > base_counts.total_noop_expert_branches
    )
    assert (
        envelope_counts(more_experts).total_noop_expert_branches
        > base_counts.total_noop_expert_branches
    )


def test_strategy_labels_make_compatibility_and_rejections_explicit():
    strategies = {strategy.name: strategy for strategy in STRATEGIES}

    flat = strategies["flat_guarded_all_experts"]
    assert flat.compatibility_label == "direct_current_prototype_construction"
    assert flat.admissible_under_user_constraints is True
    assert flat.compatible_with_current_prototype_primitives is True
    assert flat.compatible_with_abstract_theorem is True
    assert flat.length_advice == flat.route_advice == "zero"
    assert "huge" in flat.explanation
    assert "not the only construction" in flat.explanation

    public = strategies["expert_at_public_primitive"]
    assert public.route_advice == "zero"
    assert public.compatible_with_current_prototype_primitives is False
    assert public.compatible_with_abstract_theorem is True
    assert "immutable public model weights" in public.explanation
    assert "runtime router output" in public.explanation
    assert "top-8 without 384 explicit branches" in public.explanation
    assert "cost and static read list remain fixed" in public.explanation

    dynamic = strategies["dynamic_expert_lookup"]
    assert dynamic.route_advice == "zero"
    assert dynamic.compatible_with_current_prototype_primitives is False
    assert dynamic.compatible_with_abstract_theorem is False
    assert "mutable or input-resident weights" in dynamic.explanation
    assert "authenticated indexed memory/lookup" in dynamic.explanation

    specialized = strategies["route_specialized"]
    assert specialized.admissible_under_user_constraints is False
    assert specialized.compatibility_label == "inadmissible_no_route_advice"
    assert "no-route-advice" in specialized.explanation

    replay = strategies["replay_in_g"]
    assert replay.admissible_under_user_constraints is False
    assert replay.compatibility_label == "inadmissible_no_replay_in_g"
    assert "forbids replaying inference in G" in replay.explanation


@pytest.mark.parametrize(
    "kwargs, message",
    (
        ({"requests": 0}, "positive"),
        ({"actual_output_tokens_per_request": 0}, "positive"),
        ({"prefill_recomputation_tokens": -1}, "nonnegative"),
        ({"moe_layers": True}, "positive"),
        ({"top_k_experts": 5}, "cannot exceed"),
        ({"actual_output_tokens_per_request": 6}, "cannot exceed"),
    ),
)
def test_workload_validation(kwargs, message):
    values = tiny_workload().__dict__ | kwargs
    with pytest.raises(EnvelopeError, match=message):
        WorkloadShape(**values)


def test_sampling_and_gate_coefficient_validation():
    with pytest.raises(EnvelopeError, match="probability"):
        SamplingMerkleParameters(p=1.01)
    with pytest.raises(EnvelopeError, match="nonnegative"):
        SamplingMerkleParameters(instruction_depth=-1)
    with pytest.raises(EnvelopeError, match="nonnegative"):
        GateCoefficients(g_branch=-1)
    with pytest.raises(EnvelopeError, match="nonnegative"):
        GateCoefficients(g_common=True)


def test_conceptual_gate_formulas_remain_unknown_until_supplied():
    counts = envelope_counts(tiny_workload())
    unknown = conceptual_gate_accounting(counts)
    assert unknown["complete"] is False
    assert unknown["evaluated"] == {
        "useful_body_gates": None,
        "noop_body_gates": None,
        "always_live_control_gates": None,
        "n_live_plus_control": None,
        "N0_noop_body": None,
        "total_conceptual_gates": None,
    }
    assert all(value is None for value in unknown["coefficients"].values())
    assert "compression cannot reduce conceptual n" in unknown["note"]

    supplied = conceptual_gate_accounting(
        counts,
        GateCoefficients(
            g_common=1,
            g_mask=1,
            g_router=1,
            g_expert=1,
            g_branch=1,
        ),
    )
    assert supplied["complete"] is True
    evaluated = supplied["evaluated"]
    assert supplied["total_multipliers"] == {
        "g_common": 17,
        "g_mask": 10,
        "g_router": 34,
        "g_expert": 136,
        "g_branch": 136,
    }
    assert evaluated["useful_body_gates"] == 13 + 26 == 39
    assert evaluated["noop_body_gates"] == 4 + 8 + 110 == 122
    assert evaluated["always_live_control_gates"] == 10 + 26 + 136 == 172
    assert evaluated["n_live_plus_control"] == 211
    assert evaluated["N0_noop_body"] == 122
    assert evaluated["total_conceptual_gates"] == 17 + 10 + 34 + 136 + 136 == 333
    assert (
        evaluated["useful_body_gates"]
        + evaluated["noop_body_gates"]
        + evaluated["always_live_control_gates"]
        == evaluated["total_conceptual_gates"]
    )


def test_parameterized_gate_capacity_uses_noop_body_and_cell_bits():
    coefficients = GateCoefficients(
        g_common=1,
        g_mask=1,
        g_router=1,
        g_expert=1,
        g_branch=1,
    )
    result = run_experiment(
        tiny_workload(),
        tiny_merkle(),
        coefficients,
        cell_bits=7,
        include_sweeps=False,
    )
    projection = result["capacity_bound_gate_projection"]
    assert projection["n"] == 211
    assert projection["N0"] == 122
    assert projection["cell_bits"] == 7
    assert projection["unit"] == "conceptual gate"
    assert projection["exact_penalty_bits"] >= 0


def test_sampling_and_direct_merkle_formulas():
    parameters = tiny_merkle()

    assert expected_sampled_noop_gates(0.25, 40) == 10
    assert challenged_unit_probability(0.25, 4) == pytest.approx(1 - 0.75**4)
    assert challenged_unit_probability(0, 100) == 0
    assert challenged_unit_probability(1, 100) == 1

    opening = direct_merkle_opening_bytes(parameters)
    assert opening["instruction_opening_bytes"] == 11 + 2 * 32 == 75
    assert opening["one_value_opening_bytes"] == 4 + 3 * 32 == 100
    assert opening["value_openings"] == 3
    assert opening["total_bytes"] == 375

    assert naive_exact_padding_opening_bytes(5, parameters) == 5 * 100
    assert range_multiproof_proxy_bytes(2, parameters) == 2 * (
        2 * 4 + 2 * 3 * 32
    )


def test_sampling_sweep_is_monotone_and_zero_commitment_is_labeled():
    sweep = sampling_sweep(100, {"unit": 4}, (0, 0.1, 0.5, 1))
    assert [point["p"] for point in sweep] == [0.0, 0.1, 0.5, 1.0]
    expected = [point["expected_sampled_noop_gates"] for point in sweep]
    challenged = [
        point["challenged_unit_probability"]["unit"] for point in sweep
    ]
    assert expected == sorted(expected)
    assert challenged == sorted(challenged)

    comparison = zero_commitment_comparison(5, configured_depth=2)
    assert comparison["minimum_depth"] == 3
    assert comparison["configured_depth_sufficient"] is False
    assert comparison["dense_indexed_zero_leaf_hashes"] == 5
    assert comparison["dense_full_binary_tree_hashes"] == 15
    assert comparison["default_zero_precomputed_hashes"] == 4
    assert comparison["default_zero_explicit_occurrence_hashes"] == 0
    assert comparison["default_zero_capability_required"] is True


def test_default_sampling_sweep_clamps_tiny_and_zero_populations():
    for population in (0, 1, 2):
        sweep = sampling_sweep(population, {"unit": 1})
        probabilities = [point["p"] for point in sweep]
        assert probabilities == sorted(set(probabilities))
        assert all(0.0 <= probability <= 1.0 for probability in probabilities)
    assert sampling_sweep(0, {"unit": 0})[-1]["p"] == 1.0


def test_capacity_bound_exact_value_nonnegativity_and_monotonicity():
    assert K_B(4, 2, cell_bits=1) == 1 + 4 * 2 + 6 * 4 == 33
    assert K_B(6, 2, cell_bits=1) == 1 + 6 * 2 + 15 * 4 == 73
    assert capacity_bound_penalty_bits(
        4, 2, 2, cell_bits=1
    ) == pytest.approx(math.log2(73 / 33))
    assert capacity_bound_penalty_bits(100, 0, 3) == 0
    assert capacity_bound_penalty_bits(100, 10, 0) == 0
    with pytest.raises(EnvelopeError, match="positive"):
        capacity_bound_penalty_bits(100, 10, 3, cell_bits=0)

    penalties = [
        capacity_bound_penalty_bits(1_000, noop_gates, 4)
        for noop_gates in (0, 1, 10, 100)
    ]
    assert all(penalty >= 0 for penalty in penalties)
    assert penalties == sorted(penalties)


def test_sparse_capacity_approximation_converges_in_sparse_regime():
    n = 1_000_000
    noop_gates = 1_000
    limit = 4
    exact = capacity_bound_penalty_bits(n, noop_gates, limit)
    approximation = sparse_capacity_penalty_approximation_bits(
        n, noop_gates, limit
    )
    report = capacity_bound_penalty(n, noop_gates, limit)

    assert exact >= 0
    assert approximation >= 0
    assert approximation == pytest.approx(exact, rel=2e-5)
    assert report["cell_bits"] == CELL_BITS == 32
    assert report["exact_penalty_bits"] == exact
    assert report["sparse_approximation_bits"] == approximation
    assert "2**(j*cell_bits)" in report["K_B_definition"]
    assert "j=L weighted term" in report["approximation_regime"]


def test_sp1_projection_keeps_measurements_and_projection_separate():
    projection = sp1_constant_noop_local_check_projection(tiny_merkle())

    assert SP1_CALIBRATION["standalone_noop_execution"]["cycles"] == 4_886
    assert SP1_CALIBRATION["standalone_noop_execution"]["gas"] == 6_859
    assert (
        SP1_CALIBRATION["sha_precompile_path_folding"][
            "cycles_per_level_approx"
        ]
        == 1_620
    )
    assert (
        SP1_CALIBRATION["large_batch_cpu_marginal"][
            "cycles_per_second_approx"
        ]
        == 105_000
    )
    assert projection["path_levels"] == 5
    assert projection["path_only_cycles_approx"] == 5 * 1_620
    assert projection["standalone_program_floor_cycles"] == 4_886
    assert (
        projection["projected_cycles_with_conservative_floor_approx"]
        == 4_886 + 5 * 1_620
    )
    assert projection["projected_gas"] is None
    assert projection["measured_noop_proof"] is None
    assert projection["is_current_protocol_verifier_cost"] is False
    assert projection["evidence_kind"].startswith("derived_projection")
    assert "no measured no-op proof" in projection["note"].lower()
    assert "not a measured marginal no-op cost" in projection["note"]


def test_token_envelope_semantics_keep_masks_and_lengths_constrained():
    semantics = TOKEN_ENVELOPE_SEMANTICS
    assert "request-major" in semantics["frame"]
    assert semantics["initial_activity"] == "active_0 = 1 for each admitted request"
    assert "candidate token" in semantics["stop_derivation"]
    assert "public stop/EOS policy" in semantics["stop_derivation"]
    assert semantics["monotone_chain"] == "active_(t+1) = active_t AND NOT stop_t"
    assert "canonical_PAD" in semantics["output_select"]
    assert "Select" in semantics["state_freeze"]
    assert "forces termination at M" in semantics["forced_cap"]
    assert "not advice and not free witnesses" in semantics["mask_status"]
    assert "receiver-visible in y*" in semantics["length_visibility"]
    assert "public Pad(y*)" in semantics["length_visibility"]
    assert "receiver already observed" in semantics["length_visibility"]


def test_padding_bytes_separate_materialization_from_wire_choice():
    semantics = padding_communication_semantics(envelope_counts(ANCHOR_WORKLOAD))

    legacy = semantics["legacy_field"]
    assert legacy["name"] == "response_padding_bytes"
    assert legacy["value"] == 63_488
    assert "materialized canonical token-cell padding" in legacy["meaning"]
    assert "not an unconditional wire-communication claim" in legacy["meaning"]

    frame = semantics["materialized_canonical_frame"]
    assert frame["padding_token_cells"] == 15_872
    assert frame["bytes"] == 63_488
    wire = semantics["server_to_receiver_wire_modes"]
    assert wire["fixed_width_response"]["additional_padding_bytes"] == 63_488
    assert (
        wire["receiver_local_canonicalization"]["additional_padding_bytes"] == 0
    )
    assert "public deterministic Pad(y*)" in wire[
        "receiver_local_canonicalization"
    ]["condition"]
    assert "length remains visible in y*" in wire[
        "receiver_local_canonicalization"
    ]["effect"]
    assert "C consumes a max-shaped canonical frame" in wire[
        "receiver_local_canonicalization"
    ]["effect"]
    assert "does not make boundary commitment" in semantics["boundary_costs"]
    assert "not established by the current prototype" in semantics[
        "prototype_status"
    ]


def test_protocol_invariants_and_implementation_scope_are_explicit():
    assert "conceptual position in n" in PROTOCOL_INVARIANTS["zero_cost_noops"]
    assert "sampling population" in PROTOCOL_INVARIANTS["zero_cost_noops"]
    assert "does not shrink" in PROTOCOL_INVARIANTS[
        "guard_static_cost_and_reads"
    ]
    assert "Every read named" in PROTOCOL_INVARIANTS["all_named_reads_opened"]
    assert "runtime-dependent output positions" in PROTOCOL_INVARIANTS[
        "one_zero_position_limit"
    ]
    assert "does not change conceptual n" in PROTOCOL_INVARIANTS[
        "definition_compression"
    ]
    assert "theorem/protocol extension" in PROTOCOL_INVARIANTS[
        "live_only_sampling"
    ]
    assert "dependency closure" in PROTOCOL_INVARIANTS["live_only_sampling"]

    prototype = IMPLEMENTATION_SCOPE["current_prototype"]
    abstract = IMPLEMENTATION_SCOPE["draft_v4_abstract_theorem"]
    assert prototype["path"] == "src/veritor/machine.py"
    assert prototype["max_arity"] == 2
    assert "No model-specific expert_at primitive" in prototype["limitation"]
    assert "arbitrary approved deterministic functions" in abstract["model"]
    assert "does not imply support in machine.py" in abstract["distinction"]


def test_runtime_failure_dispositions_do_not_make_asserted_failures_free():
    fixed = RUNTIME_FAILURE_HANDLING["availability_fixed_in_x_or_epoch"]
    authenticated = RUNTIME_FAILURE_HANDLING["post_x_authenticated_failure"]
    self_asserted = RUNTIME_FAILURE_HANDLING["self_asserted_failure"]

    assert fixed["advice_bits"] == 0
    assert "authenticated event input" in authenticated["disposition"]
    assert "canonical reroute" in authenticated["disposition"]
    assert self_asserted["advice_bits"] == "not free"
    assert "cannot be treated as a free input" in self_asserted["disposition"]


def test_default_sweeps_include_anchor_and_are_monotone():
    sweeps = default_anchor_sweeps()
    sequence = sweeps["sequence_length"]
    requests = sweeps["request_count"]

    assert any(
        point["workload"]["actual_output_tokens_per_request"] == 900
        for point in sequence
    )
    assert any(point["workload"]["requests"] == 128 for point in requests)
    request_noops = [
        point["metrics"]["total_noop_expert_branches"] for point in requests
    ]
    assert request_noops == sorted(request_noops)


def test_json_and_cli_output_are_deterministic(capsys):
    first = run_experiment(include_sweeps=False)
    second = run_experiment(include_sweeps=False)
    serialized = canonical_json(first)
    assert serialized == canonical_json(second)
    assert json.loads(serialized) == first

    assert main(["--compact", "--no-sweeps"]) == 0
    first_cli = capsys.readouterr().out
    assert main(["--compact", "--no-sweeps"]) == 0
    second_cli = capsys.readouterr().out
    assert first_cli == second_cli
    cli_result = json.loads(first_cli)
    assert cli_result["counts"]["total_noop_expert_branches"] == 23_544_376_320
    assert (
        cli_result["padding_communication_semantics"][
            "server_to_receiver_wire_modes"
        ]["receiver_local_canonicalization"]["additional_padding_bytes"]
        == 0
    )
    assert cli_result["capacity_bound_branch_unit_projection"]["cell_bits"] == 32
    range_proxy = cli_result["sampling_and_merkle"]["range_multiproof"]
    assert "request-major" in range_proxy["layout_assumption"]
    assert range_proxy["available_in_current_index_bound_merkle"] is False
    assert "does not provide this optimization" in range_proxy["note"]
    assert "sweeps" not in cli_result
