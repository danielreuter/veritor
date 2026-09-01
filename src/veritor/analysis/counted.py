"""Scalable counted two-stage bounds with explicit layout separation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from fractions import Fraction
from typing import cast

from circuit_cut_analysis.capacity import LogCardinality
from circuit_cut_analysis.models.capacity_profile import ModelCapacityProfile
from circuit_cut_analysis.weighted_sampling import (
    WeightedGateClassPartition,
    weighted_partition_from_capacity_profile,
)
from veritor.analysis.capacity import CapacityEvidence
from veritor.analysis.probability import survival_from_replay_error_counts
from veritor.analysis.result import (
    BoundClaimStrength,
    BoundWitness,
    FixedPolicyBoundResult,
    SurvivalComparison,
    TerminationStatus,
    counted_bound_identities,
    derive_claim_strength,
)
from veritor.core.errors import ResourceLimit
from veritor.core.identity import Digest, JSONValue, identity_digest, validate_digest
from veritor.core.policy import VerificationPolicy


class CountedCapacitySemantics(StrEnum):
    """Whether the capped-linear counted capacity is exact or only an upper."""

    EXACT_CAPPED_LINEAR = "exact_capped_linear"
    CERTIFIED_CAPPED_LINEAR_UPPER = "certified_capped_linear_upper"


@dataclass(frozen=True, slots=True)
class CountedCapacityClass:
    """One capacity class, independent of any replay-unit layout."""

    id: str
    verification_unit_count: int
    singleton_upper_bound: LogCardinality
    aggregate_upper_bound: LogCardinality | None = None
    singleton_lower_bound: LogCardinality = field(
        default_factory=LogCardinality.zero
    )
    description: str = ""
    source_class_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.id) is not str or not self.id:
            raise ValueError("counted capacity class id must be nonempty")
        if (
            type(self.verification_unit_count) is not int
            or self.verification_unit_count <= 0
        ):
            raise ValueError("counted capacity class size must be positive")
        zero = LogCardinality.zero()
        if self.singleton_upper_bound < zero:
            raise ValueError("singleton capacity upper bound cannot be negative")
        if not zero <= self.singleton_lower_bound <= self.singleton_upper_bound:
            raise ValueError("singleton capacity interval is invalid")
        if self.aggregate_upper_bound is not None:
            if self.aggregate_upper_bound < zero:
                raise ValueError("aggregate capacity upper bound cannot be negative")
            if self.singleton_lower_bound > self.aggregate_upper_bound:
                raise ValueError(
                    "singleton lower bound exceeds the full-class upper bound"
                )
        if len(self.source_class_ids) != len(set(self.source_class_ids)):
            raise ValueError("source capacity class ids must be unique")

    @property
    def gate_count(self) -> int:
        """Compatibility alias for weighted scalar-gate schemas."""

        return self.verification_unit_count

    @property
    def unit_count(self) -> int:
        return self.verification_unit_count

    @property
    def singleton_capacity(self) -> LogCardinality:
        return self.singleton_upper_bound

    @property
    def aggregate_capacity(self) -> LogCardinality | None:
        return self.aggregate_upper_bound


@dataclass(frozen=True, slots=True)
class CountedCapacitySchema:
    """Count totals and capacity evidence, with no replay-layout assertion."""

    model_id: str
    classes: tuple[CountedCapacityClass, ...]
    output_frontier: LogCardinality
    semantics: CountedCapacitySemantics
    certificate: str
    assumptions: tuple[str, ...] = ()
    provenance_identity: Digest | None = None
    identity: Digest = field(init=False)

    def __post_init__(self) -> None:
        if type(self.model_id) is not str or not self.model_id:
            raise ValueError("counted schema model_id must be nonempty")
        if not self.classes:
            raise ValueError("counted schema needs at least one capacity class")
        ids = tuple(item.id for item in self.classes)
        if len(ids) != len(set(ids)):
            raise ValueError("counted capacity class ids must be unique")
        if self.output_frontier < LogCardinality.zero():
            raise ValueError("output frontier cannot be negative")
        if type(self.certificate) is not str or not self.certificate:
            raise ValueError("counted capacity certificate must be nonempty")
        if any(type(item) is not str or not item for item in self.assumptions):
            raise ValueError("counted assumptions must be nonempty strings")
        if self.provenance_identity is not None:
            validate_digest(self.provenance_identity, "provenance_identity")
        semantics = CountedCapacitySemantics(self.semantics)
        object.__setattr__(self, "semantics", semantics)
        if semantics is CountedCapacitySemantics.EXACT_CAPPED_LINEAR:
            for item in self.classes:
                if item.singleton_lower_bound != item.singleton_upper_bound:
                    raise ValueError(
                        "exact capped-linear classes require exact singleton capacity"
                    )
        manifest = cast(
            JSONValue,
            {
                "assumptions": list(self.assumptions),
                "certificate": self.certificate,
                "classes": [
                    {
                        "aggregate_upper": (
                            None
                            if item.aggregate_upper_bound is None
                            else {
                                "denominator": (
                                    item.aggregate_upper_bound.multiplier.denominator
                                ),
                                "numerator": (
                                    item.aggregate_upper_bound.multiplier.numerator
                                ),
                            }
                        ),
                        "count": item.verification_unit_count,
                        "description": item.description,
                        "id": item.id,
                        "singleton_lower": {
                            "denominator": (
                                item.singleton_lower_bound.multiplier.denominator
                            ),
                            "numerator": (
                                item.singleton_lower_bound.multiplier.numerator
                            ),
                        },
                        "singleton_upper": {
                            "denominator": (
                                item.singleton_upper_bound.multiplier.denominator
                            ),
                            "numerator": (
                                item.singleton_upper_bound.multiplier.numerator
                            ),
                        },
                        "source_class_ids": list(item.source_class_ids),
                    }
                    for item in self.classes
                ],
                "model_id": self.model_id,
                "output_frontier": {
                    "denominator": self.output_frontier.multiplier.denominator,
                    "numerator": self.output_frontier.multiplier.numerator,
                },
                "provenance_identity": self.provenance_identity,
                "semantics": semantics.value,
            },
        )
        object.__setattr__(
            self,
            "identity",
            identity_digest("veritor/analysis/counted-capacity-schema/v1", manifest),
        )

    @property
    def class_ids(self) -> tuple[str, ...]:
        return tuple(item.id for item in self.classes)

    @property
    def class_sizes(self) -> tuple[int, ...]:
        return tuple(item.verification_unit_count for item in self.classes)

    @property
    def total_verification_units(self) -> int:
        return sum(self.class_sizes)

    @property
    def total_gate_count(self) -> int:
        return self.total_verification_units

    @property
    def identity_digest(self) -> Digest:
        return self.identity


def counted_schema_from_weighted(
    partition: WeightedGateClassPartition,
    *,
    assumptions: tuple[str, ...] = (),
    provenance_identity: Digest | None = None,
) -> CountedCapacitySchema:
    """Preserve a weighted partition's certified capped-linear envelope."""

    if not isinstance(partition, WeightedGateClassPartition):
        raise TypeError("partition must be a WeightedGateClassPartition")
    return CountedCapacitySchema(
        model_id=partition.model_id,
        classes=tuple(
            CountedCapacityClass(
                id=item.id,
                verification_unit_count=item.gate_count,
                singleton_upper_bound=(
                    item.singleton_capacity
                    if item.aggregate_capacity is None
                    else min(item.singleton_capacity, item.aggregate_capacity)
                ),
                aggregate_upper_bound=item.aggregate_capacity,
                singleton_lower_bound=LogCardinality.zero(),
                description=item.description,
                source_class_ids=item.source_class_ids or (item.id,),
            )
            for item in partition.classes
        ),
        output_frontier=partition.output_frontier,
        semantics=CountedCapacitySemantics.CERTIFIED_CAPPED_LINEAR_UPPER,
        certificate=partition.certificate,
        assumptions=assumptions,
        provenance_identity=provenance_identity,
    )


def counted_schema_from_capacity_profile(
    profile: ModelCapacityProfile,
    *,
    provenance_identity: Digest | None = None,
) -> CountedCapacitySchema:
    """Convert an aggregate model profile without dropping its assumptions."""

    return counted_schema_from_weighted(
        weighted_partition_from_capacity_profile(profile),
        assumptions=profile.assumptions,
        provenance_identity=provenance_identity,
    )


@dataclass(frozen=True, slots=True)
class CountedReplayType:
    """A multiplicity of replay units with identical class incidence."""

    id: str
    multiplicity: int
    class_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.id) is not str or not self.id:
            raise ValueError("counted replay type id must be nonempty")
        if type(self.multiplicity) is not int or self.multiplicity <= 0:
            raise ValueError("counted replay multiplicity must be positive")
        if not self.class_counts or any(
            type(count) is not int or count < 0 for count in self.class_counts
        ):
            raise ValueError("replay class incidence must be nonnegative integers")
        if sum(self.class_counts) == 0:
            raise ValueError("counted replay units must not be empty")

    @property
    def verification_units_per_replay(self) -> int:
        return sum(self.class_counts)

    @property
    def replay_unit_count(self) -> int:
        return self.multiplicity

    @property
    def incidence(self) -> tuple[int, ...]:
        return self.class_counts


@dataclass(frozen=True, slots=True)
class CountedReplayLayout:
    """Actual replay incidence, separately identified from capacity classes."""

    model_id: str
    class_ids: tuple[str, ...]
    replay_types: tuple[CountedReplayType, ...]
    assumptions: tuple[str, ...] = ()
    identity: Digest = field(init=False)

    def __post_init__(self) -> None:
        if type(self.model_id) is not str or not self.model_id:
            raise ValueError("counted replay layout model_id must be nonempty")
        if not self.class_ids or len(self.class_ids) != len(set(self.class_ids)):
            raise ValueError("replay layout class ids must be nonempty and unique")
        if not self.replay_types:
            raise ValueError("counted replay layout needs at least one replay type")
        type_ids = tuple(item.id for item in self.replay_types)
        if len(type_ids) != len(set(type_ids)):
            raise ValueError("counted replay type ids must be unique")
        if any(
            len(item.class_counts) != len(self.class_ids)
            for item in self.replay_types
        ):
            raise ValueError("replay incidence vectors must align with class ids")
        if any(type(item) is not str or not item for item in self.assumptions):
            raise ValueError("layout assumptions must be nonempty strings")
        object.__setattr__(
            self,
            "identity",
            identity_digest(
                "veritor/analysis/counted-replay-layout/v1",
                {
                    "assumptions": list(self.assumptions),
                    "class_ids": list(self.class_ids),
                    "model_id": self.model_id,
                    "replay_types": [
                        {
                            "class_counts": list(item.class_counts),
                            "id": item.id,
                            "multiplicity": item.multiplicity,
                        }
                        for item in self.replay_types
                    ],
                },
            ),
        )

    @property
    def class_totals(self) -> tuple[int, ...]:
        return tuple(
            sum(
                item.multiplicity * item.class_counts[index]
                for item in self.replay_types
            )
            for index in range(len(self.class_ids))
        )

    @property
    def replay_unit_count(self) -> int:
        return sum(item.multiplicity for item in self.replay_types)

    @property
    def total_verification_units(self) -> int:
        return sum(self.class_totals)

    @property
    def identity_digest(self) -> Digest:
        return self.identity


def reconcile_counted_layout(
    schema: CountedCapacitySchema,
    layout: CountedReplayLayout,
) -> None:
    """Require replay incidence to reproduce capacity-class totals exactly."""

    if schema.model_id != layout.model_id:
        raise ValueError("capacity schema and replay layout name different models")
    if schema.class_ids != layout.class_ids:
        raise ValueError(
            "capacity classes and replay incidence are separate but must reconcile "
            "by the same ordered class ids"
        )
    if schema.class_sizes != layout.class_totals:
        raise ValueError(
            "replay incidence totals do not match counted capacity class totals"
        )


@dataclass(frozen=True, slots=True)
class CountedSolverLimits:
    """Explicit materialization and exact-integer limits for counted solvers."""

    max_actual_verification_units: int = 24
    max_exact_exponent: int = 100_000
    max_exact_power_bits: int = 4_000_000

    def __post_init__(self) -> None:
        for name in (
            "max_actual_verification_units",
            "max_exact_exponent",
            "max_exact_power_bits",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")


DEFAULT_COUNTED_SOLVER_LIMITS = CountedSolverLimits()


def _ordered_unique(*groups: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for group in groups:
        for item in group:
            if item not in seen:
                seen.add(item)
                result.append(item)
    return tuple(result)


def counted_capacity_upper_bound(
    schema: CountedCapacitySchema,
    attacked_counts: tuple[int, ...],
) -> LogCardinality:
    """Evaluate the declared capped-linear envelope exactly."""

    if len(attacked_counts) != len(schema.classes):
        raise ValueError("attacked counts do not align with capacity classes")
    total = LogCardinality.zero()
    for item, attacked in zip(schema.classes, attacked_counts, strict=True):
        if type(attacked) is not int or not (
            0 <= attacked <= item.verification_unit_count
        ):
            raise ValueError("attacked count is outside its capacity class")
        contribution = item.singleton_upper_bound.scale(attacked)
        if item.aggregate_upper_bound is not None:
            contribution = min(contribution, item.aggregate_upper_bound)
        total += contribution
        if total >= schema.output_frontier:
            return schema.output_frontier
    return total


def counted_capacity_evidence(
    schema: CountedCapacitySchema,
    attacked_counts: tuple[int, ...],
) -> CapacityEvidence[tuple[int, ...]]:
    """Return exact or certified-upper evidence for one class-count attack."""

    upper = counted_capacity_upper_bound(schema, attacked_counts)
    if schema.semantics is CountedCapacitySemantics.EXACT_CAPPED_LINEAR:
        lower = upper
    else:
        lower = LogCardinality.zero()
        for item, attacked in zip(schema.classes, attacked_counts, strict=True):
            if attacked:
                lower = max(lower, item.singleton_lower_bound)
        lower = min(lower, upper)
    return CapacityEvidence(
        lower_bound=lower,
        upper_bound=upper,
        requested_support=attacked_counts,
        evaluated_support=attacked_counts,
        method=schema.semantics.value,
        certificate=schema.certificate,
        assumptions=schema.assumptions,
    )


def _compare_scaled(
    base: LogCardinality,
    count: int,
    target: LogCardinality,
    limits: CountedSolverLimits,
) -> int | None:
    """Compare ``count * base`` with ``target`` without unsafe giant powers."""

    if count == 0 or base.is_zero:
        return (LogCardinality.zero() > target) - (
            LogCardinality.zero() < target
        )
    base_bits = base.integral_width_bits
    target_bits = target.integral_width_bits
    if base_bits is not None and target_bits is not None:
        scaled_bits = base_bits * count
        return (scaled_bits > target_bits) - (scaled_bits < target_bits)
    if base_bits is not None and base_bits >= 0:
        exponent = base_bits * count
        numerator = target.multiplier.numerator
        denominator = target.multiplier.denominator
        left_bits = denominator.bit_length() + exponent
        right_bits = numerator.bit_length()
        if left_bits != right_bits:
            return (left_bits > right_bits) - (left_bits < right_bits)
        if left_bits > limits.max_exact_power_bits:
            return None
        left = denominator << exponent
        return (left > numerator) - (left < numerator)
    estimated_bits = count * max(
        base.multiplier.numerator.bit_length(),
        base.multiplier.denominator.bit_length(),
    )
    if (
        count > limits.max_exact_exponent
        or estimated_bits > limits.max_exact_power_bits
    ):
        return None
    scaled = base.scale(count)
    return (scaled > target) - (scaled < target)


def _minimum_count_to_reach(
    base: LogCardinality,
    target: LogCardinality,
    maximum: int,
    limits: CountedSolverLimits,
) -> tuple[int | None, bool]:
    """Return (least count, ambiguous); ``None, False`` means unreachable."""

    zero = LogCardinality.zero()
    if target <= zero:
        return 0, False
    if maximum <= 0 or base.is_zero:
        return None, False
    comparison = _compare_scaled(base, maximum, target, limits)
    if comparison is None:
        return None, True
    if comparison < 0:
        return None, False
    low = 1
    high = maximum
    while low < high:
        middle = (low + high) // 2
        comparison = _compare_scaled(base, middle, target, limits)
        if comparison is None:
            return None, True
        if comparison >= 0:
            high = middle
        else:
            low = middle + 1
    return low, False


@dataclass(frozen=True, slots=True)
class _CapacitySegment:
    gain: LogCardinality
    count: int
    class_index: int


@dataclass(frozen=True, slots=True)
class CappedLinearAllocation:
    """Best class allocation under an attack-count budget."""

    attacked_counts: tuple[int, ...]
    capacity_upper_bound: LogCardinality
    used_attack_count: int
    numerically_conservative: bool
    output_frontier_fallback: bool
    segment_count: int


def capped_linear_allocation(
    schema: CountedCapacitySchema,
    attack_budget: int,
    *,
    limits: CountedSolverLimits = DEFAULT_COUNTED_SOLVER_LIMITS,
) -> CappedLinearAllocation:
    """Maximize the capped-linear upper envelope with at most ``attack_budget``.

    Each class contributes a run of equal singleton gains, at most one residual
    gain at its aggregate cap, and then zero.  Sorting those exact marginal
    gains is the discrete capped-linear optimum.
    """

    if type(attack_budget) is not int or not (
        0 <= attack_budget <= schema.total_verification_units
    ):
        raise ValueError("attack budget is outside the counted schema")
    zero = LogCardinality.zero()
    if attack_budget == 0 or schema.output_frontier.is_zero:
        return CappedLinearAllocation(
            attacked_counts=(0,) * len(schema.classes),
            capacity_upper_bound=zero,
            used_attack_count=0,
            numerically_conservative=False,
            output_frontier_fallback=False,
            segment_count=0,
        )
    segments: list[_CapacitySegment] = []
    for index, item in enumerate(schema.classes):
        singleton = item.singleton_upper_bound
        if singleton.is_zero:
            continue
        local_cap = min(
            schema.output_frontier,
            item.aggregate_upper_bound
            if item.aggregate_upper_bound is not None
            else schema.output_frontier,
        )
        if local_cap.is_zero:
            continue
        reaches, ambiguous = _minimum_count_to_reach(
            singleton,
            local_cap,
            item.verification_unit_count,
            limits,
        )
        if ambiguous:
            return CappedLinearAllocation(
                attacked_counts=(0,) * len(schema.classes),
                capacity_upper_bound=schema.output_frontier,
                used_attack_count=0,
                numerically_conservative=True,
                output_frontier_fallback=True,
                segment_count=len(segments),
            )
        if reaches is None:
            segments.append(
                _CapacitySegment(
                    singleton,
                    item.verification_unit_count,
                    index,
                )
            )
            continue
        full_count = max(0, reaches - 1)
        if full_count:
            segments.append(_CapacitySegment(singleton, full_count, index))
        singleton_bits = singleton.integral_width_bits
        local_cap_bits = local_cap.integral_width_bits
        if singleton_bits is not None and local_cap_bits is not None:
            residual = LogCardinality.bits(
                local_cap_bits - singleton_bits * full_count
            )
        else:
            residual = local_cap - singleton.scale(full_count)
        if reaches <= item.verification_unit_count and residual > zero:
            segments.append(_CapacitySegment(residual, 1, index))

    segments.sort(key=lambda item: (item.gain, -item.class_index), reverse=True)
    attacked = [0] * len(schema.classes)
    capacity = zero
    remaining = attack_budget
    conservative = False
    fallback = False
    for segment in segments:
        if remaining == 0 or capacity >= schema.output_frontier:
            break
        take = min(remaining, segment.count)
        residual_frontier = schema.output_frontier - capacity
        reaches, ambiguous = _minimum_count_to_reach(
            segment.gain,
            residual_frontier,
            take,
            limits,
        )
        if ambiguous:
            capacity = schema.output_frontier
            conservative = True
            fallback = True
            break
        if reaches is not None:
            attacked[segment.class_index] += reaches
            remaining -= reaches
            capacity = schema.output_frontier
            break
        gain_bits = segment.gain.integral_width_bits
        capacity_bits = capacity.integral_width_bits
        if (
            gain_bits is not None
            and capacity_bits is not None
            and capacity_bits + gain_bits * take > limits.max_exact_power_bits
        ):
            capacity = schema.output_frontier
            conservative = True
            fallback = True
            break
        attacked[segment.class_index] += take
        remaining -= take
        capacity += segment.gain.scale(take)

    return CappedLinearAllocation(
        attacked_counts=tuple(attacked),
        capacity_upper_bound=min(capacity, schema.output_frontier),
        used_attack_count=sum(attacked),
        numerically_conservative=conservative,
        output_frontier_fallback=fallback,
        segment_count=len(segments),
    )


@dataclass(frozen=True, slots=True)
class MegaUnitThreshold:
    """Largest conservatively admissible attack count in the mega-unit game."""

    max_attack_count: int
    exact: bool
    numerically_conservative: bool
    method: str


def _can_evaluate_power(
    base: Fraction,
    exponent: int,
    limits: CountedSolverLimits,
) -> bool:
    estimated = exponent * max(
        base.numerator.bit_length(),
        base.denominator.bit_length(),
    )
    return (
        exponent <= limits.max_exact_exponent
        and estimated <= limits.max_exact_power_bits
    )


def _mega_survival_if_exact(
    policy: VerificationPolicy,
    attacked: int,
    limits: CountedSolverLimits,
) -> Fraction | None:
    if attacked == 0 or policy.q == 0 or policy.s == 0:
        return Fraction(1)
    if policy.s == 1:
        return 1 - policy.q
    base = 1 - policy.s
    if not _can_evaluate_power(base, attacked, limits):
        return None
    return 1 - policy.q + policy.q * base**attacked


def mega_unit_threshold(
    total_verification_units: int,
    policy: VerificationPolicy,
    *,
    limits: CountedSolverLimits = DEFAULT_COUNTED_SOLVER_LIMITS,
) -> MegaUnitThreshold:
    """Compare mega-unit survival exactly when bounded, conservatively otherwise."""

    if (
        type(total_verification_units) is not int
        or total_verification_units < 0
    ):
        raise ValueError("total verification-unit count must be nonnegative")
    total = total_verification_units
    if total == 0:
        return MegaUnitThreshold(0, True, False, "empty")
    if policy.q == 0 or policy.s == 0:
        return MegaUnitThreshold(total, True, False, "zero-sampling-endpoint")
    if policy.s == 1:
        admissible = 1 - policy.q > policy.eta
        return MegaUnitThreshold(
            total if admissible else 0,
            True,
            False,
            "unit-within-sampling-endpoint",
        )

    asymptote = 1 - policy.q
    if policy.eta < asymptote:
        return MegaUnitThreshold(total, True, False, "threshold-below-asymptote")
    if policy.eta == asymptote:
        return MegaUnitThreshold(
            total,
            True,
            False,
            "finite-count-strictly-above-asymptote",
        )

    probe = min(total, limits.max_exact_exponent)
    while probe > 0 and _mega_survival_if_exact(policy, probe, limits) is None:
        probe //= 2
    if probe == 0:
        return MegaUnitThreshold(
            total,
            False,
            True,
            "exact-power-budget-unavailable",
        )
    probe_survival = _mega_survival_if_exact(policy, probe, limits)
    if probe_survival is None:
        raise AssertionError("selected mega-unit probe must be exact")
    if probe == total and probe_survival > policy.eta:
        return MegaUnitThreshold(total, True, False, "exact-total-comparison")
    if probe < total and probe_survival > policy.eta:
        return MegaUnitThreshold(
            total,
            False,
            True,
            "ambiguous-beyond-exact-power-budget",
        )

    low = 0
    high = probe
    while low + 1 < high:
        middle = (low + high) // 2
        survival = _mega_survival_if_exact(policy, middle, limits)
        if survival is None:
            raise AssertionError("binary threshold search escaped exact probe")
        if survival > policy.eta:
            low = middle
        else:
            high = middle
    return MegaUnitThreshold(low, True, False, "exact-binary-threshold")


def _empty_counted_witness(
    schema: CountedCapacitySchema,
) -> BoundWitness:
    counts = (0,) * len(schema.classes)
    evidence = counted_capacity_evidence(schema, counts)
    return BoundWitness(
        error_units=(),
        attacked_positions=(),
        attack_support=counts,
        survival_probability=Fraction(1),
        survival_comparison=SurvivalComparison.STRICTLY_ABOVE,
        capacity_evidence=cast(CapacityEvidence[object], evidence),
        attacked_counts=counts,
    )


def mega_unit_relaxation_bound(
    schema: CountedCapacitySchema | WeightedGateClassPartition,
    policy: VerificationPolicy,
    *,
    assumptions: tuple[str, ...] = (),
    limits: CountedSolverLimits = DEFAULT_COUNTED_SOLVER_LIMITS,
    replay_layout_identity: Digest | None = None,
    resource_limited_actual_layout: bool = False,
) -> FixedPolicyBoundResult:
    """Return the always-available adversarial one-replay-unit upper bound.

    Concentrating every attacked verification unit in one mega-unit can only
    increase survival:

    ``f(a+b) >= f(a) f(b)`` for ``f(k)=1-q+q(1-s)^k``.

    Consequently this result is a certified upper relaxation for every replay
    layout matching the class totals.  It is deliberately never labeled exact
    for an actual protocol layout.
    """

    if isinstance(schema, WeightedGateClassPartition):
        schema = counted_schema_from_weighted(schema, assumptions=assumptions)
        assumptions = ()
    if not isinstance(schema, CountedCapacitySchema):
        raise TypeError("schema must be counted or weighted capacity classes")
    combined_assumptions = _ordered_unique(schema.assumptions, assumptions)
    threshold = mega_unit_threshold(
        schema.total_verification_units,
        policy,
        limits=limits,
    )
    allocation = capped_linear_allocation(
        schema,
        threshold.max_attack_count,
        limits=limits,
    )
    numeric = (
        threshold.numerically_conservative
        or allocation.numerically_conservative
    )
    used_survival = _mega_survival_if_exact(
        policy,
        allocation.used_attack_count,
        limits,
    )
    if used_survival is not None and used_survival > policy.eta:
        comparison = SurvivalComparison.STRICTLY_ABOVE
    elif used_survival is not None:
        comparison = SurvivalComparison.CONSERVATIVELY_INCLUDED
        numeric = True
    else:
        comparison = SurvivalComparison.CONSERVATIVELY_INCLUDED
        numeric = True

    upper_evidence = CapacityEvidence(
        lower_bound=LogCardinality.zero(),
        upper_bound=allocation.capacity_upper_bound,
        requested_support=allocation.attacked_counts,
        evaluated_support=allocation.attacked_counts,
        method="adversarial-mega-unit-capped-linear-allocation",
        certificate=schema.certificate,
        assumptions=combined_assumptions,
        notes=(
            "capacity value is an upper envelope, not an attained-layout claim",
        ),
    )
    upper_witness = BoundWitness(
        error_units=(),
        attacked_positions=(),
        attack_support=allocation.attacked_counts,
        survival_probability=used_survival,
        survival_comparison=comparison,
        capacity_evidence=cast(CapacityEvidence[object], upper_evidence),
        attacked_counts=allocation.attacked_counts,
        note="relaxed mega-unit attack allocation",
    )
    if resource_limited_actual_layout:
        termination = TerminationStatus.RESOURCE_LIMIT
    elif numeric:
        termination = TerminationStatus.NUMERICALLY_CONSERVATIVE
    else:
        termination = TerminationStatus.COMPLETE
    relaxation = [
        "capped-linear structural capacity certificate",
        "adversarial mega-unit replay relaxation",
        "designated-output frontier cap",
    ]
    if allocation.output_frontier_fallback:
        relaxation.append("numerically conservative output-frontier fallback")
    if resource_limited_actual_layout:
        relaxation.append("actual incidence exceeded materialization limit")
    return FixedPolicyBoundResult(
        lower_bound=LogCardinality.zero(),
        upper_bound=allocation.capacity_upper_bound,
        claim_strength=BoundClaimStrength.CERTIFIED_UPPER,
        termination_status=termination,
        method="counted-adversarial-mega-unit",
        witness=_empty_counted_witness(schema),
        upper_witness=upper_witness,
        identities=counted_bound_identities(
            policy,
            schema.identity,
            replay_layout_identity,
            relaxation="adversarial-mega-unit",
        ),
        assumptions=combined_assumptions,
        relaxation_chain=tuple(relaxation),
        state_count=allocation.segment_count,
        numerically_conservative=numeric,
    )


def actual_counted_layout_bound(
    schema: CountedCapacitySchema,
    layout: CountedReplayLayout,
    policy: VerificationPolicy,
    *,
    limits: CountedSolverLimits = DEFAULT_COUNTED_SOLVER_LIMITS,
    assumptions: tuple[str, ...] = (),
) -> FixedPolicyBoundResult:
    """Exhaustively evaluate supplied replay incidence when it is tractable."""

    reconcile_counted_layout(schema, layout)
    total_units = layout.total_verification_units
    if total_units > limits.max_actual_verification_units:
        raise ResourceLimit(
            "actual_counted_verification_units",
            limit=limits.max_actual_verification_units,
            observed=total_units,
        )

    atoms: list[tuple[int, int]] = []
    replay_index = 0
    for replay_type in layout.replay_types:
        for _ in range(replay_type.multiplicity):
            for class_index, count in enumerate(replay_type.class_counts):
                atoms.extend((replay_index, class_index) for _ in range(count))
            replay_index += 1
    if len(atoms) != total_units:
        raise AssertionError("expanded replay incidence changed its total")

    zero = LogCardinality.zero()
    empty_witness = _empty_counted_witness(schema)
    lower = zero
    upper = zero
    lower_witness = empty_witness
    upper_witness = empty_witness
    feasible = 0
    pattern_count = 1 << total_units
    for mask in range(pattern_count):
        replay_counts = [0] * layout.replay_unit_count
        class_counts = [0] * len(schema.classes)
        error_units: list[int] = []
        for atom_index, (owner, class_index) in enumerate(atoms):
            if mask & (1 << atom_index):
                replay_counts[owner] += 1
                class_counts[class_index] += 1
                error_units.append(atom_index)
        survival = survival_from_replay_error_counts(policy, replay_counts)
        if survival <= policy.eta:
            continue
        feasible += 1
        counts = tuple(class_counts)
        evidence = counted_capacity_evidence(schema, counts)
        witness = BoundWitness(
            error_units=tuple(error_units),
            attacked_positions=(),
            attack_support=counts,
            survival_probability=survival,
            survival_comparison=SurvivalComparison.STRICTLY_ABOVE,
            capacity_evidence=cast(CapacityEvidence[object], evidence),
            attacked_counts=counts,
        )
        if evidence.lower_bound > lower:
            lower = evidence.lower_bound
            lower_witness = witness
        if evidence.upper_bound > upper:
            upper = evidence.upper_bound
            upper_witness = witness

    combined_assumptions = _ordered_unique(
        schema.assumptions,
        layout.assumptions,
        assumptions,
    )
    upper_relaxation = (
        schema.semantics
        is CountedCapacitySemantics.CERTIFIED_CAPPED_LINEAR_UPPER
    )
    termination = TerminationStatus.COMPLETE
    claim = derive_claim_strength(
        lower,
        upper,
        termination,
        assumptions=combined_assumptions,
        certified_upper_relaxation=upper_relaxation,
    )
    relaxation_chain = (
        ("capped-linear structural capacity certificate",)
        if upper_relaxation
        else ()
    )
    return FixedPolicyBoundResult(
        lower_bound=lower,
        upper_bound=upper,
        claim_strength=claim,
        termination_status=termination,
        method="actual-counted-layout-exhaustive",
        witness=lower_witness,
        upper_witness=upper_witness,
        identities=counted_bound_identities(
            policy,
            schema.identity,
            layout.identity,
        ),
        assumptions=combined_assumptions,
        relaxation_chain=relaxation_chain,
        state_count=pattern_count,
        feasible_state_count=feasible,
        pruned_infeasible_count=pattern_count - feasible,
    )


def counted_fixed_policy_bound(
    schema: CountedCapacitySchema | WeightedGateClassPartition,
    policy: VerificationPolicy,
    *,
    replay_layout: CountedReplayLayout | None = None,
    assumptions: tuple[str, ...] = (),
    limits: CountedSolverLimits = DEFAULT_COUNTED_SOLVER_LIMITS,
) -> FixedPolicyBoundResult:
    """Use actual incidence when tractable, otherwise the mega-unit upper."""

    if isinstance(schema, WeightedGateClassPartition):
        schema = counted_schema_from_weighted(schema, assumptions=assumptions)
        assumptions = ()
    if replay_layout is None:
        return mega_unit_relaxation_bound(
            schema,
            policy,
            assumptions=assumptions,
            limits=limits,
        )
    reconcile_counted_layout(schema, replay_layout)
    if (
        replay_layout.total_verification_units
        <= limits.max_actual_verification_units
    ):
        return actual_counted_layout_bound(
            schema,
            replay_layout,
            policy,
            limits=limits,
            assumptions=assumptions,
        )
    return mega_unit_relaxation_bound(
        schema,
        policy,
        assumptions=_ordered_unique(replay_layout.assumptions, assumptions),
        limits=limits,
        replay_layout_identity=replay_layout.identity,
        resource_limited_actual_layout=True,
    )


# Stable concise spellings.
solve_counted_bound = counted_fixed_policy_bound
mega_unit_bound = mega_unit_relaxation_bound


__all__ = [
    "DEFAULT_COUNTED_SOLVER_LIMITS",
    "CappedLinearAllocation",
    "CountedCapacityClass",
    "CountedCapacitySchema",
    "CountedCapacitySemantics",
    "CountedReplayLayout",
    "CountedReplayType",
    "CountedSolverLimits",
    "MegaUnitThreshold",
    "actual_counted_layout_bound",
    "capped_linear_allocation",
    "counted_capacity_evidence",
    "counted_capacity_upper_bound",
    "counted_fixed_policy_bound",
    "counted_schema_from_capacity_profile",
    "counted_schema_from_weighted",
    "mega_unit_bound",
    "mega_unit_relaxation_bound",
    "mega_unit_threshold",
    "reconcile_counted_layout",
    "solve_counted_bound",
]
