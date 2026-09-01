"""Advice accounting for variable-length continual-batching inference.

The experiment separates three kinds of information:

* ``x`` fixes request arrivals, prompt lengths, and per-request token limits.
* generated token values are circuit values, not constructor advice;
* realized completion lengths determine the unpadded circuit skeleton and are
  encoded in advice when they are not derivable from ``x``.

There is no unconditional positive advice lower bound. If the model, sampling
coins, and all other runtime inputs are in ``x``, G can replay inference during
construction and derive the lengths with zero advice, at potentially enormous
compilation cost. This experiment instead treats lengths as post-``x``
realizations and measures the bits needed to avoid that replay.

Given completion lengths, the scheduler below is deterministic. Therefore its
continual batch composition—including replacements and unusual request
orderings—requires zero additional advice. Alternative metrics estimate how
many bits would be spent if the client were instead allowed to choose each
eligible subset or ordered batch explicitly.

This is a decode-only structural simulation, not a latency or GPU-throughput
model.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter, deque
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence


class WorkloadError(ValueError):
    """The workload, realization, or advice is malformed."""


def ceil_log2(value: int) -> int:
    if type(value) is not int or value < 1:
        raise WorkloadError("ceil_log2 expects a positive integer")
    return (value - 1).bit_length()


@dataclass(frozen=True)
class RequestInput:
    request_id: int
    arrival_tick: int
    prompt_length: int
    max_new_tokens: int


@dataclass(frozen=True)
class WorkloadInput:
    requests: tuple[RequestInput, ...]
    decode_slots: int


@dataclass(frozen=True)
class LengthAdvice:
    """Mixed-radix encoding of one completion length per request."""

    payload: bytes
    bit_length: int


@dataclass(frozen=True)
class BatchStep:
    tick: int
    request_ids: tuple[int | None, ...]
    context_lengths: tuple[int | None, ...]
    eligible_count: int

    @property
    def active_request_ids(self) -> tuple[int, ...]:
        return tuple(
            request_id for request_id in self.request_ids if request_id is not None
        )

    @property
    def active_context_lengths(self) -> tuple[int, ...]:
        return tuple(
            context
            for context in self.context_lengths
            if context is not None
        )

    @property
    def active_count(self) -> int:
        return len(self.active_request_ids)


@dataclass(frozen=True)
class Schedule:
    steps: tuple[BatchStep, ...]
    completion_ticks: tuple[tuple[int, int], ...]

    @property
    def token_occurrences(self) -> int:
        return sum(step.active_count for step in self.steps)


@dataclass(frozen=True)
class Scenario:
    name: str
    num_requests: int
    decode_slots: int
    max_new_tokens: int
    arrival_rate_per_tick: float
    prompt_min: int
    prompt_max: int
    mean_output_tokens: float
    seed: int


def validate_workload(workload: WorkloadInput) -> None:
    if type(workload.decode_slots) is not int or workload.decode_slots <= 0:
        raise WorkloadError("decode_slots must be positive")
    ids: set[int] = set()
    for request in workload.requests:
        if type(request.request_id) is not int or request.request_id < 0:
            raise WorkloadError("request IDs must be nonnegative integers")
        if request.request_id in ids:
            raise WorkloadError("request IDs must be unique")
        ids.add(request.request_id)
        if type(request.arrival_tick) is not int or request.arrival_tick < 0:
            raise WorkloadError("arrival ticks must be nonnegative integers")
        if type(request.prompt_length) is not int or request.prompt_length <= 0:
            raise WorkloadError("prompt lengths must be positive integers")
        if (
            type(request.max_new_tokens) is not int
            or request.max_new_tokens <= 0
        ):
            raise WorkloadError("max_new_tokens must be positive")


def validate_lengths(
    workload: WorkloadInput,
    lengths: Mapping[int, int],
) -> None:
    validate_workload(workload)
    expected_ids = {request.request_id for request in workload.requests}
    if set(lengths) != expected_ids:
        raise WorkloadError("length realization must contain every request exactly once")
    for request in workload.requests:
        length = lengths[request.request_id]
        if type(length) is not int or not (1 <= length <= request.max_new_tokens):
            raise WorkloadError(
                f"request {request.request_id} length must lie in "
                f"[1, {request.max_new_tokens}]"
            )


def length_space_size(workload: WorkloadInput) -> int:
    validate_workload(workload)
    return math.prod(request.max_new_tokens for request in workload.requests)


def minimum_length_advice_bits(workload: WorkloadInput) -> int:
    """Code size when every length vector needs a distinct unpadded structure."""

    return ceil_log2(length_space_size(workload))


def fixed_field_length_bits(workload: WorkloadInput) -> int:
    """Bits used by an independent fixed-width field for each request."""

    validate_workload(workload)
    return sum(ceil_log2(request.max_new_tokens) for request in workload.requests)


def encode_lengths(
    workload: WorkloadInput,
    lengths: Mapping[int, int],
) -> LengthAdvice:
    """Encode the complete length vector as one mixed-radix integer."""

    validate_lengths(workload, lengths)
    rank = 0
    for request in workload.requests:
        rank = rank * request.max_new_tokens + lengths[request.request_id] - 1
    bit_length = minimum_length_advice_bits(workload)
    payload_length = (bit_length + 7) // 8
    return LengthAdvice(rank.to_bytes(payload_length, "big"), bit_length)


def decode_lengths(
    workload: WorkloadInput,
    advice: LengthAdvice,
) -> dict[int, int]:
    expected_bits = minimum_length_advice_bits(workload)
    expected_bytes = (expected_bits + 7) // 8
    if advice.bit_length != expected_bits or len(advice.payload) != expected_bytes:
        raise WorkloadError("length advice has the wrong canonical size")
    rank = int.from_bytes(advice.payload, "big")
    if rank >= length_space_size(workload):
        raise WorkloadError("length advice uses an unused mixed-radix code")

    decoded_reversed: list[tuple[int, int]] = []
    for request in reversed(workload.requests):
        rank, digit = divmod(rank, request.max_new_tokens)
        decoded_reversed.append((request.request_id, digit + 1))
    if rank != 0:
        raise WorkloadError("length advice rank did not decode completely")
    return dict(reversed(decoded_reversed))


def continual_batch_schedule(
    workload: WorkloadInput,
    lengths: Mapping[int, int],
) -> Schedule:
    """Deterministic FCFS slot-replacement continual batching.

    At each tick, arrived requests fill free slots in request-arrival order.
    Every resident request emits one token. Completed requests leave after that
    tick and the next waiting requests enter on the following tick.
    """

    validate_lengths(workload, lengths)
    ordered = sorted(
        workload.requests,
        key=lambda request: (request.arrival_tick, request.request_id),
    )
    by_id = {request.request_id: request for request in workload.requests}
    generated = {request.request_id: 0 for request in workload.requests}
    waiting: deque[int] = deque()
    slots: list[int | None] = [None] * workload.decode_slots
    completion_ticks: dict[int, int] = {}
    steps: list[BatchStep] = []
    next_arrival = 0
    tick = ordered[0].arrival_tick if ordered else 0

    while len(completion_ticks) < len(ordered):
        while (
            next_arrival < len(ordered)
            and ordered[next_arrival].arrival_tick <= tick
        ):
            waiting.append(ordered[next_arrival].request_id)
            next_arrival += 1

        for slot_index, occupant in enumerate(slots):
            if occupant is None and waiting:
                slots[slot_index] = waiting.popleft()

        residents = tuple(request_id for request_id in slots if request_id is not None)
        if not residents:
            if next_arrival >= len(ordered):
                raise RuntimeError("scheduler lost an unfinished request")
            tick = max(tick + 1, ordered[next_arrival].arrival_tick)
            continue

        eligible_count = len(residents) + len(waiting)
        contexts = tuple(
            (
                None
                if request_id is None
                else by_id[request_id].prompt_length + generated[request_id]
            )
            for request_id in slots
        )
        steps.append(
            BatchStep(
                tick=tick,
                request_ids=tuple(slots),
                context_lengths=contexts,
                eligible_count=eligible_count,
            )
        )

        for slot_index, request_id in enumerate(slots):
            if request_id is None:
                continue
            generated[request_id] += 1
            if generated[request_id] == lengths[request_id]:
                completion_ticks[request_id] = tick
                slots[slot_index] = None
        tick += 1

    return Schedule(
        steps=tuple(steps),
        completion_ticks=tuple(sorted(completion_ticks.items())),
    )


def explicit_schedule_choice_bits(schedule: Schedule) -> tuple[int, int]:
    """Return subset-choice and ordered-choice bits, conditioned on lengths.

    These are deliberately conservative comparison metrics. They encode each
    observed batch as a fresh choice from all currently eligible requests.
    A real stateful schedule code could be smaller. A fixed public scheduler,
    as used by ``continual_batch_schedule``, needs zero such bits.
    """

    subset_log2 = 0.0
    ordered_log2 = 0.0
    for step in schedule.steps:
        eligible = step.eligible_count
        selected = step.active_count
        subset_log2 += math.log2(math.comb(eligible, selected))
        ordered_log2 += math.log2(math.perm(eligible, selected))
    return math.ceil(subset_log2), math.ceil(ordered_log2)


def empirical_entropy_bits(values: Sequence[int]) -> float:
    """In-sample entropy estimate; not a worst-case advice guarantee."""

    if not values:
        return 0.0
    counts = Counter(values)
    total = len(values)
    entropy_per_value = -sum(
        (count / total) * math.log2(count / total) for count in counts.values()
    )
    return total * entropy_per_value


def cache_shape_metrics(schedule: Schedule) -> dict[str, int | float]:
    token_occurrences = schedule.token_occurrences
    single_context_shapes = {
        context
        for step in schedule.steps
        for context in step.active_context_lengths
    }
    ordered_batch_shapes = {step.context_lengths for step in schedule.steps}
    canonical_batch_shapes = {
        tuple(sorted(step.active_context_lengths)) for step in schedule.steps
    }
    padded_batch_shapes = {
        (step.active_count, max(step.active_context_lengths))
        for step in schedule.steps
    }
    composition_changes = sum(
        left.request_ids != right.request_ids
        for left, right in zip(schedule.steps, schedule.steps[1:])
    )
    batch_steps = len(schedule.steps)
    max_slots = max((step.active_count for step in schedule.steps), default=0)
    return {
        "token_occurrences": token_occurrences,
        "batch_steps": batch_steps,
        "mean_batch_size": token_occurrences / batch_steps if batch_steps else 0.0,
        "max_batch_size": max_slots,
        "batch_composition_changes": composition_changes,
        "unique_single_context_shapes": len(single_context_shapes),
        "single_context_definition_reuse": (
            1.0 - len(single_context_shapes) / token_occurrences
            if token_occurrences
            else 0.0
        ),
        "unique_ordered_batch_shapes": len(ordered_batch_shapes),
        "unique_canonical_batch_shapes": len(canonical_batch_shapes),
        "unique_padded_batch_shapes": len(padded_batch_shapes),
    }


def generate_scenario(
    scenario: Scenario,
) -> tuple[WorkloadInput, dict[int, int]]:
    if scenario.num_requests <= 0:
        raise WorkloadError("scenario must contain requests")
    if scenario.arrival_rate_per_tick <= 0:
        raise WorkloadError("arrival rate must be positive")
    if not (0 < scenario.prompt_min <= scenario.prompt_max):
        raise WorkloadError("invalid prompt range")
    if scenario.mean_output_tokens <= 0:
        raise WorkloadError("mean output length must be positive")

    rng = random.Random(scenario.seed)
    arrival_time = 0.0
    requests: list[RequestInput] = []
    lengths: dict[int, int] = {}
    log_prompt_min = math.log(scenario.prompt_min)
    log_prompt_max = math.log(scenario.prompt_max)
    for request_id in range(scenario.num_requests):
        if request_id:
            arrival_time += rng.expovariate(scenario.arrival_rate_per_tick)
        prompt = round(math.exp(rng.uniform(log_prompt_min, log_prompt_max)))
        realized = max(1, math.ceil(rng.expovariate(1 / scenario.mean_output_tokens)))
        realized = min(realized, scenario.max_new_tokens)
        requests.append(
            RequestInput(
                request_id=request_id,
                arrival_tick=int(arrival_time),
                prompt_length=prompt,
                max_new_tokens=scenario.max_new_tokens,
            )
        )
        lengths[request_id] = realized
    workload = WorkloadInput(tuple(requests), scenario.decode_slots)
    validate_lengths(workload, lengths)
    return workload, lengths


def analyze_scenario(scenario: Scenario) -> dict[str, object]:
    workload, lengths = generate_scenario(scenario)
    advice = encode_lengths(workload, lengths)
    decoded = decode_lengths(workload, advice)
    if decoded != lengths:
        raise RuntimeError("mixed-radix advice failed to reconstruct lengths")

    schedule = continual_batch_schedule(workload, decoded)
    replayed_schedule = continual_batch_schedule(workload, lengths)
    if schedule != replayed_schedule:
        raise RuntimeError("decoded advice did not reconstruct the schedule")
    if schedule.token_occurrences != sum(lengths.values()):
        raise RuntimeError("scheduler token count does not match realized lengths")

    subset_bits, ordered_bits = explicit_schedule_choice_bits(schedule)
    minimum_bits = minimum_length_advice_bits(workload)
    fixed_bits = fixed_field_length_bits(workload)
    actual_tokens = sum(lengths.values())
    padded_tokens = sum(
        request.max_new_tokens for request in workload.requests
    )
    output_lengths = tuple(lengths[request.request_id] for request in workload.requests)
    result: dict[str, object] = {
        "scenario": asdict(scenario),
        "requests": len(workload.requests),
        "actual_output_tokens": actual_tokens,
        "maximum_output_tokens": padded_tokens,
        "length_advice_condition": (
            "each length vector requires a distinct unpadded structure"
        ),
        "minimum_length_advice_bits": minimum_bits,
        "fixed_field_length_bits": fixed_bits,
        "minimum_length_advice_bytes": (minimum_bits + 7) // 8,
        "advice_bits_per_request": minimum_bits / len(workload.requests),
        "advice_bits_per_output_token": minimum_bits / actual_tokens,
        "empirical_length_entropy_bits": empirical_entropy_bits(output_lengths),
        "deterministic_scheduler_advice_bits": 0,
        "explicit_subset_schedule_bits": subset_bits,
        "explicit_ordered_schedule_bits": ordered_bits,
        "zero_advice_padded_token_step_inflation": padded_tokens / actual_tokens,
        "realized_output_length": {
            "min": min(output_lengths),
            "median": sorted(output_lengths)[len(output_lengths) // 2],
            "mean": actual_tokens / len(output_lengths),
            "max": max(output_lengths),
        },
    }
    result.update(cache_shape_metrics(schedule))
    return result


SCENARIOS = (
    Scenario(
        name="interactive",
        num_requests=256,
        decode_slots=32,
        max_new_tokens=256,
        arrival_rate_per_tick=4.0,
        prompt_min=32,
        prompt_max=1_024,
        mean_output_tokens=64,
        seed=7,
    ),
    Scenario(
        name="mixed",
        num_requests=256,
        decode_slots=32,
        max_new_tokens=512,
        arrival_rate_per_tick=8.0,
        prompt_min=8,
        prompt_max=4_096,
        mean_output_tokens=128,
        seed=11,
    ),
    Scenario(
        name="long_form",
        num_requests=128,
        decode_slots=16,
        max_new_tokens=1_024,
        arrival_rate_per_tick=2.0,
        prompt_min=128,
        prompt_max=8_192,
        mean_output_tokens=384,
        seed=17,
    ),
)


def run_scenarios() -> dict[str, object]:
    return {
        "model": {
            "scope": "decode-only structural simulation",
            "load_bearing_assumption": (
                "completion lengths are post-x realizations not recomputed by G"
            ),
            "known_in_x": [
                "request identity and arrival tick",
                "prompt tokens/length",
                "max_new_tokens",
                "deterministic FCFS slot scheduler",
            ],
            "encoded_in_advice": ["one realized completion length per request"],
            "not_advice": [
                "generated token values, with model randomness fixed in x",
                "batch choices implied by the fixed scheduler and lengths",
            ],
            "other_zero_advice_options": [
                "G replays deterministic inference from x during compilation",
                "fixed maximum-envelope circuit with canonical padded outputs",
            ],
        },
        "scenarios": [analyze_scenario(scenario) for scenario in SCENARIOS],
    }


if __name__ == "__main__":
    print(json.dumps(run_scenarios(), indent=2, sort_keys=True))
