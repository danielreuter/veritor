import itertools

import pytest

from prototypes.advice_workload import (
    LengthAdvice,
    RequestInput,
    Scenario,
    WorkloadError,
    WorkloadInput,
    analyze_scenario,
    cache_shape_metrics,
    continual_batch_schedule,
    decode_lengths,
    encode_lengths,
    explicit_schedule_choice_bits,
    fixed_field_length_bits,
    length_space_size,
    minimum_length_advice_bits,
)


def small_workload() -> WorkloadInput:
    return WorkloadInput(
        requests=(
            RequestInput(0, 0, 3, 2),
            RequestInput(1, 0, 5, 3),
            RequestInput(2, 1, 7, 2),
        ),
        decode_slots=2,
    )


def test_mixed_radix_encoding_is_bijective_over_small_length_space():
    workload = small_workload()
    payloads = set()

    for values in itertools.product(range(1, 3), range(1, 4), range(1, 3)):
        lengths = dict(enumerate(values))
        advice = encode_lengths(workload, lengths)
        payloads.add(advice.payload)
        assert decode_lengths(workload, advice) == lengths

    assert len(payloads) == length_space_size(workload) == 12
    assert minimum_length_advice_bits(workload) == 4
    assert fixed_field_length_bits(workload) == 4


def test_unused_mixed_radix_codes_are_rejected():
    workload = small_workload()
    # Four bits represent 16 values, while this workload has only 12 length vectors.
    with pytest.raises(WorkloadError, match="unused"):
        decode_lengths(workload, LengthAdvice(payload=b"\x0f", bit_length=4))


def test_joint_rank_can_beat_independent_fixed_width_fields():
    workload = WorkloadInput(
        requests=tuple(RequestInput(i, 0, 4, 3) for i in range(3)),
        decode_slots=2,
    )

    assert length_space_size(workload) == 27
    assert minimum_length_advice_bits(workload) == 5
    assert fixed_field_length_bits(workload) == 6


def test_scheduler_reconstruction_needs_only_lengths():
    workload = small_workload()
    lengths = {0: 2, 1: 3, 2: 1}
    decoded = decode_lengths(workload, encode_lengths(workload, lengths))

    assert continual_batch_schedule(workload, decoded) == continual_batch_schedule(
        workload, lengths
    )


def test_continual_batching_replaces_completed_requests_deterministically():
    schedule = continual_batch_schedule(
        small_workload(),
        {0: 2, 1: 3, 2: 1},
    )

    assert [step.request_ids for step in schedule.steps] == [
        (0, 1),
        (0, 1),
        (2, 1),
    ]
    assert [step.context_lengths for step in schedule.steps] == [
        (3, 5),
        (4, 6),
        (7, 7),
    ]
    assert schedule.token_occurrences == 6
    assert dict(schedule.completion_ticks) == {0: 1, 1: 2, 2: 2}


def test_golden_trace_preserves_ordered_empty_slots():
    workload = WorkloadInput(
        requests=(
            RequestInput(0, 0, 16, 4),
            RequestInput(1, 0, 64, 4),
            RequestInput(2, 0, 16, 4),
            RequestInput(3, 1, 64, 4),
            RequestInput(4, 1, 16, 4),
            RequestInput(5, 1, 16, 4),
            RequestInput(6, 2, 32, 3),
            RequestInput(7, 2, 64, 3),
        ),
        decode_slots=3,
    )
    schedule = continual_batch_schedule(
        workload,
        {0: 1, 1: 1, 2: 1, 3: 2, 4: 4, 5: 3, 6: 1, 7: 2},
    )

    assert [step.context_lengths for step in schedule.steps] == [
        (16, 64, 16),
        (64, 16, 16),
        (65, 17, 17),
        (32, 18, 18),
        (64, 19, None),
        (65, None, None),
    ]
    assert [step.request_ids for step in schedule.steps[-2:]] == [
        (7, 4, None),
        (7, None, None),
    ]
    assert schedule.token_occurrences == 15


def test_explicit_schedule_metrics_are_extra_to_deterministic_policy():
    schedule = continual_batch_schedule(
        small_workload(),
        {0: 2, 1: 3, 2: 1},
    )
    subset_bits, ordered_bits = explicit_schedule_choice_bits(schedule)

    assert subset_bits == 2
    assert ordered_bits >= subset_bits


def test_shape_metrics_count_occurrences_and_reuse():
    schedule = continual_batch_schedule(
        small_workload(),
        {0: 2, 1: 3, 2: 1},
    )
    metrics = cache_shape_metrics(schedule)

    assert metrics["token_occurrences"] == 6
    assert metrics["batch_steps"] == 3
    assert metrics["batch_composition_changes"] == 1
    assert 0 <= metrics["single_context_definition_reuse"] < 1


def test_scenario_analysis_separates_worst_case_and_empirical_metrics():
    result = analyze_scenario(
        Scenario(
            name="test",
            num_requests=16,
            decode_slots=4,
            max_new_tokens=16,
            arrival_rate_per_tick=2.0,
            prompt_min=4,
            prompt_max=32,
            mean_output_tokens=5,
            seed=123,
        )
    )

    assert result["minimum_length_advice_bits"] == 64
    assert result["fixed_field_length_bits"] == 64
    assert result["deterministic_scheduler_advice_bits"] == 0
    assert result["actual_output_tokens"] == result["token_occurrences"]
    assert result["zero_advice_padded_token_step_inflation"] >= 1
    assert (
        result["empirical_length_entropy_bits"]
        <= result["minimum_length_advice_bits"]
    )


def test_invalid_length_realization_is_rejected():
    workload = small_workload()
    with pytest.raises(WorkloadError, match="every request"):
        encode_lengths(workload, {0: 1, 1: 1})
    with pytest.raises(WorkloadError, match=r"\[1, 2\]"):
        encode_lengths(workload, {0: 0, 1: 1, 2: 1})
