"""Scalable all-adversary evaluation for class-weighted scalar sampling.

The finite solver in :mod:`circuit_cut_analysis.class_sampling` enumerates
actual gates.  This module instead accepts a certified capped-linear
structural bound

``lambda(E) <= min(frontier, sum_i min(e_i * w_i, A_i))``,

where ``e_i`` is the number of attacked gates in class ``i`` and ``w_i`` is a
certified upper bound on every singleton cut in that class. ``A_i`` is an
optional certified full-class union-of-cuts cap.  The union of singleton cuts
proves the linear term for any circuit, including when those cuts overlap.

For one deterministic exact-budget class quota, the adversarial best response
is solved without enumerating gates.  A Pareto dynamic program considers every
attack count vector that can matter before reaching the output frontier and
uses exact hypergeometric miss probabilities.  It therefore quantifies over
all scalar-gate attacks under the declared linear capacity certificate.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction

from circuit_cut_analysis.capacity import GateCapacity, LogCardinality
from circuit_cut_analysis.class_sampling import (
    GateClassPartition,
    GateProbabilityClass,
)
from circuit_cut_analysis.models.capacity_profile import ModelCapacityProfile
from circuit_cut_analysis.sampling_study import RegionUnit


@dataclass(frozen=True, slots=True)
class WeightedGateClass:
    """One scalar-gate probability class and its singleton capacity bound."""

    id: str
    gate_count: int
    singleton_capacity: LogCardinality
    aggregate_capacity: LogCardinality | None = None
    description: str = ""
    source_class_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("weighted class id must be non-empty")
        if self.gate_count <= 0:
            raise ValueError("weighted class gate count must be positive")
        if self.singleton_capacity < LogCardinality.zero():
            raise ValueError("singleton capacity cannot be negative")
        if (
            self.aggregate_capacity is not None
            and self.aggregate_capacity < LogCardinality.zero()
        ):
            raise ValueError("aggregate capacity cannot be negative")
        if len(self.source_class_ids) != len(set(self.source_class_ids)):
            raise ValueError("source class ids must be unique")

    @property
    def full_class_capacity(self) -> LogCardinality:
        """Return the certified cap for attacking every gate in this class."""

        if self.aggregate_capacity is not None:
            return self.aggregate_capacity
        return self.singleton_capacity.scale(self.gate_count)

    @property
    def effective_singleton_capacity(self) -> LogCardinality:
        """Return the strongest declared bound for one attacked class member."""

        return min(self.singleton_capacity, self.full_class_capacity)


@dataclass(frozen=True, slots=True)
class WeightedGateClassPartition:
    """A counted class partition with a certified capped-linear bound."""

    model_id: str
    classes: tuple[WeightedGateClass, ...]
    output_frontier: LogCardinality
    certificate: str

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model id must be non-empty")
        if not self.classes:
            raise ValueError("weighted partition needs at least one class")
        ids = [gate_class.id for gate_class in self.classes]
        if len(ids) != len(set(ids)):
            raise ValueError("weighted class ids must be unique")
        if self.output_frontier <= LogCardinality.zero():
            raise ValueError("output frontier must be positive")
        if not self.certificate:
            raise ValueError("capacity certificate must be non-empty")

    @property
    def total_gate_count(self) -> int:
        return sum(gate_class.gate_count for gate_class in self.classes)

    @property
    def class_sizes(self) -> tuple[int, ...]:
        return tuple(gate_class.gate_count for gate_class in self.classes)

    def abstract_gate_partition(self) -> GateClassPartition:
        """Return a count-equivalent tiny identity shell for strategy validation.

        This method is intentionally unavailable for large classes because
        materializing synthetic gate IDs would defeat the counted
        representation.  Strategies for this module are built directly by
        :func:`fixed_quota_strategy`.
        """

        if self.total_gate_count > 100_000:
            raise ValueError("abstract scalar partition is too large to materialize")
        offset = 0
        classes: list[GateProbabilityClass] = []
        all_gates: list[str] = []
        for gate_class in self.classes:
            gates = tuple(
                f"{gate_class.id}/abstract-{index}"
                for index in range(offset, offset + gate_class.gate_count)
            )
            offset += gate_class.gate_count
            all_gates.extend(gates)
            classes.append(GateProbabilityClass(gate_class.id, gates))
        return GateClassPartition(classes, all_gates)


def coalesce_equal_capacity_classes(
    partition: WeightedGateClassPartition,
) -> WeightedGateClassPartition:
    """Merge equal-width classes into a sound upper-payoff relaxation.

    For equal singleton capacity ``c``,
    ``sum_i min(e_i*c, A_i) <= min(sum_i e_i*c, sum_i A_i)``.  The merged class
    therefore may overstate capacity when individual aggregate caps differ,
    but never understates it.  This is valid for verifier upper bounds and is
    exact when every member's cap is its linear gate-count cap.  Global lower
    certificates must be evaluated on the unmerged partition.
    """

    grouped: dict[LogCardinality, list[WeightedGateClass]] = {}
    for gate_class in partition.classes:
        grouped.setdefault(gate_class.effective_singleton_capacity, []).append(
            gate_class
        )

    classes: list[WeightedGateClass] = []
    for capacity, members in sorted(
        grouped.items(),
        key=lambda item: item[0],
    ):
        width = capacity.width_bits
        label = f"{width}-bit" if isinstance(width, int) else f"{float(width):.12g}-bit"
        aggregate = LogCardinality.zero()
        for member in members:
            aggregate += member.full_class_capacity
            if aggregate >= partition.output_frontier:
                aggregate = partition.output_frontier
                break
        classes.append(
            WeightedGateClass(
                id=f"singleton-capacity/{label}",
                gate_count=sum(member.gate_count for member in members),
                singleton_capacity=capacity,
                aggregate_capacity=aggregate,
                description="; ".join(member.id for member in members),
                source_class_ids=tuple(
                    source_id
                    for member in members
                    for source_id in (member.source_class_ids or (member.id,))
                ),
            )
        )
    return WeightedGateClassPartition(
        model_id=partition.model_id,
        classes=tuple(classes),
        output_frontier=partition.output_frontier,
        certificate=(
            partition.certificate
            + "; equal-singleton-capacity classes coalesced under the linear bound"
        ),
    )


def coalesce_frontier_equivalent_classes(
    partition: WeightedGateClassPartition,
) -> WeightedGateClassPartition:
    """Exactly merge classes whose local caps cannot bind before the frontier.

    Classes with the same singleton width and full-class capacity at least the
    global output frontier have contribution
    ``min(F, sum_i e_i*w)`` after the global cap, independent of their original
    labels. Zero-capacity classes are likewise exactly interchangeable.
    Classes with a genuine sub-frontier aggregate cap remain separate.

    This preserves not only every attack payoff but the full minimax value:
    any verifier strategy on the finer classes can be averaged over
    permutations within an exact merged union. Every permuted attack has the
    same payoff, so threshold feasibility is preserved, while exact checked
    budgets remain exact.
    """

    grouped: dict[
        tuple[str, str, LogCardinality],
        list[WeightedGateClass],
    ] = {}
    for gate_class in partition.classes:
        capacity = gate_class.effective_singleton_capacity
        if capacity.is_zero:
            key = ("zero", "", capacity)
        elif gate_class.full_class_capacity >= partition.output_frontier:
            key = ("frontier", "", capacity)
        else:
            key = ("separate", gate_class.id, capacity)
        grouped.setdefault(key, []).append(gate_class)

    classes: list[WeightedGateClass] = []
    used_ids = {gate_class.id for gate_class in partition.classes}
    for group_index, ((kind, _, capacity), members) in enumerate(
        sorted(
            grouped.items(),
            key=lambda item: (item[0][2], item[0][0], item[0][1]),
        )
    ):
        if kind == "zero":
            aggregate = LogCardinality.zero()
        elif kind == "frontier":
            aggregate = partition.output_frontier
        elif kind == "separate":
            if len(members) != 1:
                raise AssertionError("sub-frontier classes cannot be merged")
            aggregate = members[0].full_class_capacity
        else:
            raise AssertionError(f"unexpected coalescing key {kind!r}")
        width = capacity.width_bits
        label = f"{width}-bit" if isinstance(width, int) else f"{float(width):.12g}-bit"
        if kind in {"zero", "frontier"}:
            class_id = f"minimax-equivalent/{group_index}/{label}"
            suffix = 1
            while class_id in used_ids:
                class_id = f"minimax-equivalent/{group_index}/{label}/{suffix}"
                suffix += 1
            used_ids.add(class_id)
        else:
            class_id = members[0].id
        classes.append(
            WeightedGateClass(
                id=class_id,
                gate_count=sum(member.gate_count for member in members),
                singleton_capacity=capacity,
                aggregate_capacity=aggregate,
                description="; ".join(member.id for member in members),
                source_class_ids=tuple(
                    source_id
                    for member in members
                    for source_id in (member.source_class_ids or (member.id,))
                ),
            )
        )
    return WeightedGateClassPartition(
        model_id=partition.model_id,
        classes=tuple(classes),
        output_frontier=partition.output_frontier,
        certificate=(
            partition.certificate
            + "; exactly coalesced equal-width classes whose local caps cannot "
            "bind below the output frontier"
        ),
    )


def _capacity_from_width(
    width_bits: float,
    *,
    logical_vocabulary_size: int,
) -> LogCardinality:
    """Recover exact integral or token capacities from a profile width."""

    if not math.isfinite(width_bits) or width_bits < 0:
        raise ValueError("capacity width must be finite and nonnegative")
    integral = round(width_bits)
    if float(integral) >= width_bits and math.isclose(
        width_bits,
        integral,
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        return LogCardinality.bits(integral)
    token_width = math.log2(logical_vocabulary_size)
    for token_count in range(1, 5):
        candidate = LogCardinality.cardinality(logical_vocabulary_size**token_count)
        if float(candidate.width_bits) >= width_bits and math.isclose(
            width_bits,
            token_count * token_width,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            return candidate
    # A power-of-two ceiling is a sound recovery when a profile supplied an
    # otherwise inexact floating width.
    return LogCardinality.bits(math.ceil(width_bits))


def _upper_capacity_from_width(
    width_bits: float,
    *,
    logical_vocabulary_size: int,
) -> LogCardinality:
    """Recover a sound exact upper capacity from an aggregate float width."""

    if width_bits < 0:
        raise ValueError("capacity width cannot be negative")
    if not math.isfinite(width_bits):
        raise ValueError("capacity width must be finite")
    integral = round(width_bits)
    if float(integral) >= width_bits and math.isclose(
        width_bits,
        integral,
        rel_tol=0.0,
        abs_tol=1e-8,
    ):
        return LogCardinality.bits(integral)
    token_width = math.log2(logical_vocabulary_size)
    token_count = round(width_bits / token_width)
    if token_count >= 0:
        candidate = LogCardinality.cardinality(logical_vocabulary_size**token_count)
        if float(candidate.width_bits) >= width_bits and math.isclose(
            width_bits,
            token_count * token_width,
            rel_tol=1e-12,
            abs_tol=1e-7,
        ):
            return candidate
    return LogCardinality.bits(math.ceil(width_bits))


def _scale_capped_at(
    capacity: LogCardinality,
    count: int,
    cap: LogCardinality,
) -> LogCardinality:
    """Scale exactly while avoiding construction beyond the declared cap."""

    if (
        _minimum_attack_count_to_reach(
            capacity,
            cap,
            maximum=count,
        )
        is not None
    ):
        return cap
    return capacity.scale(count)


def weighted_partition_from_capacity_profile(
    profile: ModelCapacityProfile,
) -> WeightedGateClassPartition:
    """Use each architectural region as one scalar probability class."""

    frontier = GateCapacity.values(profile.logical_vocabulary_size).log_value.scale(
        profile.generated_tokens
    )
    classes = tuple(
        WeightedGateClass(
            id=region.id,
            gate_count=region.gate_count,
            singleton_capacity=(
                singleton := (
                    LogCardinality.cardinality(region.value_cardinality_upper_bound)
                    if region.value_cardinality_upper_bound is not None
                    else _capacity_from_width(
                        region.self_cut_bits_per_gate,
                        logical_vocabulary_size=profile.logical_vocabulary_size,
                    )
                )
            ),
            aggregate_capacity=_scale_capped_at(
                singleton,
                region.gate_count,
                frontier,
            ),
            description=region.description,
            source_class_ids=(region.id,),
        )
        for region in profile.regions
        if region.gate_count > 0
    )
    return WeightedGateClassPartition(
        model_id=profile.model_id,
        classes=classes,
        output_frontier=frontier,
        certificate=(
            "self-cut union: each attacked gate is a valid singleton self-cut; "
            "the union of those cuts and the designated-output frontier are valid"
        ),
    )


def weighted_partition_from_region_units(
    *,
    model_id: str,
    units: Iterable[RegionUnit],
    output_frontier: LogCardinality,
    token_cardinality: int,
) -> WeightedGateClassPartition:
    """Convert certified canonical-region units to scalar probability classes.

    ``max_single_cut_bits`` is used—not the full-unit capacity—so arbitrary
    subsets inside a class remain soundly bounded.
    """

    classes = tuple(
        WeightedGateClass(
            id=unit.id,
            gate_count=unit.checked_gate_count,
            singleton_capacity=_capacity_from_width(
                unit.max_single_cut_bits,
                logical_vocabulary_size=token_cardinality,
            ),
            aggregate_capacity=min(
                output_frontier,
                _upper_capacity_from_width(
                    unit.capacity_upper_bits,
                    logical_vocabulary_size=token_cardinality,
                ),
            ),
            description=", ".join(unit.row_ids),
            source_class_ids=(unit.id,),
        )
        for unit in units
        if unit.checked_gate_count > 0
    )
    return WeightedGateClassPartition(
        model_id=model_id,
        classes=classes,
        output_frontier=output_frontier,
        certificate=(
            "union of certified singleton canonical cuts, upper-bounded within "
            "each class by max_single_cut_bits; capped by output frontier"
        ),
    )


@dataclass(frozen=True, slots=True)
class CountedQuotaStrategy:
    """One deterministic class quota with exact scalar inclusion marginals."""

    class_ids: tuple[str, ...]
    class_sizes: tuple[int, ...]
    checked_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        if not (
            len(self.class_ids) == len(self.class_sizes) == len(self.checked_counts)
        ):
            raise ValueError("quota vectors must align")
        if any(size <= 0 for size in self.class_sizes):
            raise ValueError("class sizes must be positive")
        if any(
            not 0 <= checked <= size
            for checked, size in zip(self.checked_counts, self.class_sizes, strict=True)
        ):
            raise ValueError("checked quota is outside a class")

    @property
    def checked_gate_budget(self) -> int:
        return sum(self.checked_counts)

    @property
    def inclusion_probabilities(self) -> tuple[Fraction, ...]:
        return tuple(
            Fraction(checked, size)
            for checked, size in zip(self.checked_counts, self.class_sizes, strict=True)
        )


def fixed_quota_strategy(
    partition: WeightedGateClassPartition,
    checked_counts: Sequence[int],
) -> CountedQuotaStrategy:
    """Build one exact-budget stratified sampling strategy."""

    return CountedQuotaStrategy(
        class_ids=tuple(gate_class.id for gate_class in partition.classes),
        class_sizes=partition.class_sizes,
        checked_counts=tuple(checked_counts),
    )


def capacity_upper_bound_for_counts(
    partition: WeightedGateClassPartition,
    attacked_counts: Sequence[int],
) -> LogCardinality:
    """Evaluate the partition's certified capped-linear attack envelope."""

    if len(attacked_counts) != len(partition.classes):
        raise ValueError("attack count vector does not match weighted classes")
    total = LogCardinality.zero()
    for gate_class, attacked in zip(partition.classes, attacked_counts, strict=True):
        if not 0 <= attacked <= gate_class.gate_count:
            raise ValueError("attacked count is outside a weighted class")
        full_cap = gate_class.full_class_capacity
        singleton = gate_class.effective_singleton_capacity
        contribution = min(singleton.scale(attacked), full_cap)
        total += contribution
        if total >= partition.output_frontier:
            return partition.output_frontier
    return total


def _minimum_attack_count_to_reach(
    singleton: LogCardinality,
    target: LogCardinality,
    *,
    maximum: int,
) -> int | None:
    """Return the least exact count whose linear capacity reaches ``target``."""

    if maximum <= 0 or singleton.is_zero:
        return None
    low = 1
    singleton_bits = float(singleton.width_bits)
    target_bits = float(target.width_bits)
    if singleton_bits > 0 and math.isfinite(singleton_bits):
        high = min(
            maximum,
            max(1, math.ceil(target_bits / singleton_bits) + 2),
        )
    else:
        high = 1
    while high < maximum and singleton.scale(high) < target:
        high = min(maximum, high * 2)
    if singleton.scale(high) < target:
        return None
    while low < high:
        middle = (low + high) // 2
        if singleton.scale(middle) >= target:
            high = middle
        else:
            low = middle + 1
    return low


def _scalable_miss_probability(
    population: int,
    checked: int,
    attacked: int,
) -> Fraction:
    """Compute ``C(n-e,k)/C(n,k)`` in O(e), even for trillion-scale n."""

    if not 0 <= checked <= population:
        raise ValueError("checked count is outside population")
    if not 0 <= attacked <= population:
        raise ValueError("attacked count is outside population")
    if checked > population - attacked:
        return Fraction()
    result = Fraction(1)
    for index in range(attacked):
        result *= Fraction(
            population - checked - index,
            population - index,
        )
    return result


@dataclass(frozen=True, slots=True)
class WeightedClassAttack:
    """An all-location class-count attack under the certified linear bound."""

    attacked_counts: tuple[int, ...]
    detection_probability: Fraction
    capacity_upper_bound: LogCardinality

    @property
    def capacity_upper_bits(self) -> int | float:
        return self.capacity_upper_bound.width_bits


@dataclass(frozen=True, slots=True)
class UniversalMinimaxLowerBound:
    """A verifier-independent lower certificate for the structural-bound game.

    A pool is chosen so every ``attack_size`` subset has certified payoff at
    least ``capacity_lower_bound``.  Even if a verifier spends its entire
    exact budget inside that pool, the average miss probability of a uniformly
    random attack from the pool is ``miss_probability``.  When that value is
    strictly above ``1 - detection_threshold``, at least one concrete attack
    is admissible against every verifier strategy.
    """

    capacity_lower_bound: LogCardinality
    checked_gate_budget: int
    detection_threshold: Fraction
    attack_size: int
    pool_gate_count: int
    pool_class_ids: tuple[str, ...]
    miss_probability: Fraction

    @property
    def capacity_lower_bits(self) -> int | float:
        return self.capacity_lower_bound.width_bits


def universal_minimax_lower_bound(
    partition: WeightedGateClassPartition,
    *,
    checked_gate_budget: int,
    detection_threshold: int | float | Fraction = Fraction(99, 100),
) -> UniversalMinimaxLowerBound:
    """Certify a lower bound against every exact-budget verifier strategy.

    The search considers every distinct effective singleton width ``w`` and
    attack size up to ``ceil(frontier / w)``.  A class joins the attack pool
    only when both its singleton bound is at least ``w`` and its aggregate
    cap is at least the target ``t*w``.  Therefore every size-``t`` subset of
    the pool earns at least that target in the assumed-attained structural
    bound game, regardless of how its gates are distributed among classes.
    """

    if not 0 <= checked_gate_budget <= partition.total_gate_count:
        raise ValueError("checked-gate budget is outside partition")
    threshold = (
        Fraction(str(detection_threshold))
        if isinstance(detection_threshold, float)
        else Fraction(detection_threshold)
    )
    if not 0 < threshold <= 1:
        raise ValueError("detection threshold must lie in (0, 1]")
    admissible_miss = 1 - threshold

    best = UniversalMinimaxLowerBound(
        capacity_lower_bound=LogCardinality.zero(),
        checked_gate_budget=checked_gate_budget,
        detection_threshold=threshold,
        attack_size=0,
        pool_gate_count=0,
        pool_class_ids=(),
        miss_probability=Fraction(1),
    )
    widths = sorted(
        {
            gate_class.effective_singleton_capacity
            for gate_class in partition.classes
            if not gate_class.effective_singleton_capacity.is_zero
        }
    )
    for width in widths:
        max_attack_size = _minimum_attack_count_to_reach(
            width,
            partition.output_frontier,
            maximum=partition.total_gate_count,
        )
        if max_attack_size is None:
            max_attack_size = partition.total_gate_count
        for attack_size in range(1, max_attack_size + 1):
            target = min(
                width.scale(attack_size),
                partition.output_frontier,
            )
            pool = tuple(
                gate_class
                for gate_class in partition.classes
                if gate_class.effective_singleton_capacity >= width
                and gate_class.full_class_capacity >= target
            )
            pool_size = sum(gate_class.gate_count for gate_class in pool)
            if pool_size < attack_size:
                continue
            pool_checked = min(checked_gate_budget, pool_size)
            miss = _scalable_miss_probability(
                pool_size,
                pool_checked,
                attack_size,
            )
            if miss <= admissible_miss:
                continue
            if target > best.capacity_lower_bound:
                best = UniversalMinimaxLowerBound(
                    capacity_lower_bound=target,
                    checked_gate_budget=checked_gate_budget,
                    detection_threshold=threshold,
                    attack_size=attack_size,
                    pool_gate_count=pool_size,
                    pool_class_ids=tuple(gate_class.id for gate_class in pool),
                    miss_probability=miss,
                )
    return best


@dataclass(frozen=True, slots=True)
class ClassMomentRequirement:
    """Necessary expected checks for suppressing one class's attacks."""

    class_id: str
    attack_size: int
    gate_count: int
    required_expected_checks: Fraction


@dataclass(frozen=True, slots=True)
class PureClassMomentLowerBound:
    """Global minimax lower certificate from per-class attack moments."""

    capacity_lower_bound: LogCardinality
    checked_gate_budget: int
    detection_threshold: Fraction
    required_expected_checks: Fraction
    requirements: tuple[ClassMomentRequirement, ...]

    @property
    def capacity_lower_bits(self) -> int | float:
        return self.capacity_lower_bound.width_bits

    @property
    def budget_shortfall(self) -> Fraction:
        return self.required_expected_checks - self.checked_gate_budget


def _minimum_expected_checks_for_miss(
    population: int,
    attacked: int,
    maximum_miss: Fraction,
) -> Fraction:
    """Invert the convex hypergeometric miss curve exactly."""

    if not 0 <= maximum_miss < 1:
        raise ValueError("maximum miss must lie in [0, 1)")
    lower = 0
    upper = population
    while lower < upper:
        middle = (lower + upper) // 2
        if _scalable_miss_probability(population, middle, attacked) <= maximum_miss:
            upper = middle
        else:
            lower = middle + 1
    first_feasible = lower
    if first_feasible == 0:
        return Fraction()
    below = first_feasible - 1
    miss_below = _scalable_miss_probability(population, below, attacked)
    miss_above = _scalable_miss_probability(
        population,
        first_feasible,
        attacked,
    )
    fraction_above = Fraction(
        miss_below - maximum_miss,
        miss_below - miss_above,
    )
    return below + fraction_above


def pure_class_moment_lower_bound(
    partition: WeightedGateClassPartition,
    *,
    checked_gate_budget: int,
    detection_threshold: int | float | Fraction = Fraction(99, 100),
) -> PureClassMomentLowerBound:
    """Prove a global lower bound by summing necessary class quotas.

    To keep the game value below a target ``T``, every pure-class attack with
    capacity at least ``T`` must have miss probability at most ``alpha``.
    Averaging uniformly over all ``t``-subsets in a class gives

    ``E[choose(n - K, t) / choose(n, t)] <= alpha``.

    The hypergeometric miss curve is decreasing and discretely convex in
    ``K``.  Its piecewise-linear inverse therefore gives an exact lower bound
    on ``E[K]``.  If these necessary expected quotas sum to more than the
    exact budget, some target-reaching attack is admissible against every
    verifier strategy.
    """

    if not 0 <= checked_gate_budget <= partition.total_gate_count:
        raise ValueError("checked-gate budget is outside partition")
    threshold = (
        Fraction(str(detection_threshold))
        if isinstance(detection_threshold, float)
        else Fraction(detection_threshold)
    )
    if not 0 < threshold <= 1:
        raise ValueError("detection threshold must lie in (0, 1]")
    maximum_miss = 1 - threshold

    targets: set[LogCardinality] = set()
    for gate_class in partition.classes:
        width = gate_class.effective_singleton_capacity
        if width.is_zero:
            continue
        class_cap = min(
            gate_class.full_class_capacity,
            partition.output_frontier,
        )
        for attacked in range(1, gate_class.gate_count + 1):
            target = min(width.scale(attacked), class_cap)
            targets.add(target)
            if target == class_cap:
                break

    best = PureClassMomentLowerBound(
        capacity_lower_bound=LogCardinality.zero(),
        checked_gate_budget=checked_gate_budget,
        detection_threshold=threshold,
        required_expected_checks=Fraction(),
        requirements=(),
    )
    for target in sorted(targets):
        requirements: list[ClassMomentRequirement] = []
        required_total = Fraction()
        for gate_class in partition.classes:
            width = gate_class.effective_singleton_capacity
            class_cap = min(
                gate_class.full_class_capacity,
                partition.output_frontier,
            )
            if width.is_zero or class_cap < target:
                continue
            attack_size = _minimum_attack_count_to_reach(
                width,
                target,
                maximum=gate_class.gate_count,
            )
            if attack_size is None:
                continue
            required = _minimum_expected_checks_for_miss(
                gate_class.gate_count,
                attack_size,
                maximum_miss,
            )
            requirements.append(
                ClassMomentRequirement(
                    class_id=gate_class.id,
                    attack_size=attack_size,
                    gate_count=gate_class.gate_count,
                    required_expected_checks=required,
                )
            )
            required_total += required
        if required_total > checked_gate_budget and target > best.capacity_lower_bound:
            best = PureClassMomentLowerBound(
                capacity_lower_bound=target,
                checked_gate_budget=checked_gate_budget,
                detection_threshold=threshold,
                required_expected_checks=required_total,
                requirements=tuple(requirements),
            )
    return best


@dataclass(frozen=True, slots=True)
class _ParetoState:
    capacity: LogCardinality
    miss_probability: Fraction
    attacked_counts: tuple[int, ...]


def _prune_states(states: Iterable[_ParetoState]) -> list[_ParetoState]:
    """Remove states dominated in both capacity and miss probability."""

    best_by_capacity: dict[LogCardinality, _ParetoState] = {}
    for state in states:
        previous = best_by_capacity.get(state.capacity)
        if previous is None or state.miss_probability > previous.miss_probability:
            best_by_capacity[state.capacity] = state

    ordered = sorted(
        best_by_capacity.values(),
        key=lambda state: state.capacity,
        reverse=True,
    )
    retained: list[_ParetoState] = []
    largest_miss = Fraction(-1)
    for state in ordered:
        if state.miss_probability > largest_miss:
            retained.append(state)
            largest_miss = state.miss_probability
    return retained


def stratified_quota_best_response(
    partition: WeightedGateClassPartition,
    strategy: CountedQuotaStrategy,
    *,
    detection_threshold: int | float | Fraction = Fraction(99, 100),
) -> WeightedClassAttack | None:
    """Return the exact worst class-count attack under the linear certificate.

    The dynamic program is exact: options enumerate every per-class attack
    count that can improve payoff before the frontier cap, exact
    hypergeometric probabilities enforce the strict threshold, and Pareto
    pruning removes only states dominated in both payoff and admissibility.
    """

    if strategy.class_ids != tuple(gate_class.id for gate_class in partition.classes):
        raise ValueError("strategy classes do not match weighted partition")
    if strategy.class_sizes != partition.class_sizes:
        raise ValueError("strategy class sizes do not match weighted partition")
    threshold = (
        Fraction(str(detection_threshold))
        if isinstance(detection_threshold, float)
        else Fraction(detection_threshold)
    )
    if not 0 < threshold <= 1:
        raise ValueError("detection threshold must lie in (0, 1]")
    admissible_miss = 1 - threshold

    states = [
        _ParetoState(
            capacity=LogCardinality.zero(),
            miss_probability=Fraction(1),
            attacked_counts=(),
        )
    ]
    for gate_class, checked in zip(
        partition.classes, strategy.checked_counts, strict=True
    ):
        options: tuple[tuple[int, LogCardinality, Fraction], ...]
        singleton = gate_class.effective_singleton_capacity
        if singleton.is_zero:
            options = ((0, LogCardinality.zero(), Fraction(1)),)
        else:
            max_attacked = _minimum_attack_count_to_reach(
                singleton,
                partition.output_frontier,
                maximum=gate_class.gate_count,
            )
            if max_attacked is None:
                max_attacked = gate_class.gate_count
            options = tuple(
                (
                    attacked,
                    min(
                        singleton.scale(attacked),
                        gate_class.full_class_capacity,
                    ),
                    _scalable_miss_probability(
                        gate_class.gate_count,
                        checked,
                        attacked,
                    ),
                )
                for attacked in range(max_attacked + 1)
            )

        expanded: list[_ParetoState] = []
        for state in states:
            for attacked, capacity, class_miss in options:
                miss = state.miss_probability * class_miss
                if miss <= admissible_miss:
                    continue
                total_capacity = state.capacity + capacity
                if total_capacity > partition.output_frontier:
                    total_capacity = partition.output_frontier
                expanded.append(
                    _ParetoState(
                        capacity=total_capacity,
                        miss_probability=miss,
                        attacked_counts=(*state.attacked_counts, attacked),
                    )
                )
        states = _prune_states(expanded)
        if not states:
            return None

    nonempty = [
        state
        for state in states
        if any(state.attacked_counts) and state.miss_probability > admissible_miss
    ]
    if not nonempty:
        return None
    best = max(
        nonempty,
        key=lambda state: (
            state.capacity,
            state.miss_probability,
            state.attacked_counts,
        ),
    )
    return WeightedClassAttack(
        attacked_counts=best.attacked_counts,
        detection_probability=1 - best.miss_probability,
        capacity_upper_bound=best.capacity,
    )


def equalized_log_quota_strategy(
    partition: WeightedGateClassPartition,
    checked_gate_budget: int,
) -> CountedQuotaStrategy:
    """Construct the fractional-adversary equalized fixed-quota strategy.

    In the with-replacement/independent relaxation, one attacked gate in class
    ``i`` consumes detection budget ``-log(1-p_i)`` and earns ``w_i`` bits.
    Minimizing the strongest fractional ratio sets

    ``p_i = 1 - exp(-alpha * w_i)``

    for one ``alpha`` chosen to spend the gate budget.  We round the resulting
    class quotas by largest remainder, preserving the exact total.  Zero-cut
    classes receive budget only after every positive class is fully checked.

    This is an optimized, scalable candidate—not a global optimality theorem
    for the discrete hypergeometric minimax game.
    """

    if not 0 <= checked_gate_budget <= partition.total_gate_count:
        raise ValueError("checked-gate budget is outside partition")
    positive = [
        index
        for index, gate_class in enumerate(partition.classes)
        if not gate_class.effective_singleton_capacity.is_zero
    ]
    positive_gates = sum(partition.classes[index].gate_count for index in positive)
    targets = [0.0] * len(partition.classes)

    if checked_gate_budget >= positive_gates:
        for index in positive:
            targets[index] = float(partition.classes[index].gate_count)
        remaining = checked_gate_budget - positive_gates
        for index, gate_class in enumerate(partition.classes):
            if index in positive:
                continue
            assigned = min(remaining, gate_class.gate_count)
            targets[index] = float(assigned)
            remaining -= assigned
    elif checked_gate_budget > 0:
        low = 0.0
        high = 1.0

        def spend(alpha: float) -> float:
            return sum(
                partition.classes[index].gate_count
                * (
                    1
                    - math.exp(
                        -alpha
                        * float(
                            partition.classes[
                                index
                            ].effective_singleton_capacity.width_bits
                        )
                    )
                )
                for index in positive
            )

        while spend(high) < checked_gate_budget:
            high *= 2
        for _ in range(100):
            middle = (low + high) / 2
            if spend(middle) < checked_gate_budget:
                low = middle
            else:
                high = middle
        alpha = (low + high) / 2
        for index in positive:
            gate_class = partition.classes[index]
            targets[index] = gate_class.gate_count * (
                1
                - math.exp(
                    -alpha * float(gate_class.effective_singleton_capacity.width_bits)
                )
            )

    checked_counts = [
        min(gate_class.gate_count, math.floor(target))
        for gate_class, target in zip(partition.classes, targets, strict=True)
    ]
    difference = checked_gate_budget - sum(checked_counts)
    if difference > 0:
        order = sorted(
            range(len(targets)),
            key=lambda index: (
                targets[index] - checked_counts[index],
                float(partition.classes[index].effective_singleton_capacity.width_bits),
            ),
            reverse=True,
        )
        for index in order:
            if difference == 0:
                break
            if checked_counts[index] < partition.classes[index].gate_count:
                checked_counts[index] += 1
                difference -= 1
    elif difference < 0:
        order = sorted(
            range(len(targets)),
            key=lambda index: (
                targets[index] - checked_counts[index],
                float(partition.classes[index].effective_singleton_capacity.width_bits),
            ),
        )
        for index in order:
            if difference == 0:
                break
            if checked_counts[index] > 0:
                checked_counts[index] -= 1
                difference += 1
    if difference:
        raise AssertionError("largest-remainder allocation did not preserve budget")
    return fixed_quota_strategy(partition, checked_counts)


def uniform_quota_strategy(
    partition: WeightedGateClassPartition,
    checked_gate_budget: int,
) -> CountedQuotaStrategy:
    """Allocate an exact gate budget proportionally across all classes."""

    if not 0 <= checked_gate_budget <= partition.total_gate_count:
        raise ValueError("checked-gate budget is outside partition")
    total = partition.total_gate_count
    numerators = [
        checked_gate_budget * gate_class.gate_count for gate_class in partition.classes
    ]
    checked = [numerator // total for numerator in numerators]
    remaining = checked_gate_budget - sum(checked)
    order = sorted(
        range(len(checked)),
        key=lambda index: numerators[index] % total,
        reverse=True,
    )
    for index in order[:remaining]:
        checked[index] += 1
    return fixed_quota_strategy(partition, checked)


@dataclass(frozen=True, slots=True)
class FixedQuotaOptimization:
    """Best exactly replayed quota found by deterministic exchange search."""

    strategy: CountedQuotaStrategy
    worst_attack: WeightedClassAttack | None
    checked_gate_budget: int
    detection_threshold: Fraction
    evaluated_quota_count: int
    globally_optimal: bool = False

    @property
    def certified_upper_bound(self) -> LogCardinality:
        if self.worst_attack is None:
            return LogCardinality.zero()
        return self.worst_attack.capacity_upper_bound


def _quota_score(
    attack: WeightedClassAttack | None,
) -> tuple[LogCardinality, Fraction]:
    if attack is None:
        return LogCardinality.zero(), Fraction(-1)
    return attack.capacity_upper_bound, -attack.detection_probability


def _quota_with_fully_checked_classes(
    partition: WeightedGateClassPartition,
    checked_gate_budget: int,
    fully_checked: frozenset[int],
) -> CountedQuotaStrategy | None:
    fixed_budget = sum(partition.classes[index].gate_count for index in fully_checked)
    residual_budget = checked_gate_budget - fixed_budget
    residual_classes = tuple(
        gate_class
        for index, gate_class in enumerate(partition.classes)
        if index not in fully_checked
    )
    residual_gate_count = sum(gate_class.gate_count for gate_class in residual_classes)
    if residual_budget < 0 or residual_budget > residual_gate_count:
        return None
    residual_counts: dict[str, int] = {}
    if residual_classes:
        residual_partition = WeightedGateClassPartition(
            model_id=partition.model_id,
            classes=residual_classes,
            output_frontier=partition.output_frontier,
            certificate=partition.certificate,
        )
        residual_strategy = equalized_log_quota_strategy(
            residual_partition,
            residual_budget,
        )
        residual_counts = dict(
            zip(
                residual_strategy.class_ids,
                residual_strategy.checked_counts,
                strict=True,
            )
        )
    return fixed_quota_strategy(
        partition,
        tuple(
            gate_class.gate_count
            if index in fully_checked
            else residual_counts[gate_class.id]
            for index, gate_class in enumerate(partition.classes)
        ),
    )


def optimize_fixed_quota_strategy(
    partition: WeightedGateClassPartition,
    *,
    checked_gate_budget: int,
    detection_threshold: int | float | Fraction = Fraction(99, 100),
    max_evaluations: int = 2_000,
) -> FixedQuotaOptimization:
    """Search exact class quotas, certifying every candidate against all attacks.

    The search starts from uniform and proportional-exponent allocations, then
    performs budget-preserving pairwise exchanges at successively finer powers
    of two.  Every reported upper bound is rigorous because the exact
    all-adversary DP replays the winning quota.  The combinatorial outer search
    is heuristic and therefore does not claim global optimality.
    """

    if max_evaluations <= 0:
        raise ValueError("max_evaluations must be positive")
    threshold = (
        Fraction(str(detection_threshold))
        if isinstance(detection_threshold, float)
        else Fraction(detection_threshold)
    )
    if not 0 < threshold <= 1:
        raise ValueError("detection threshold must lie in (0, 1]")

    seed_candidates = [
        uniform_quota_strategy(partition, checked_gate_budget),
        equalized_log_quota_strategy(partition, checked_gate_budget),
    ]
    small_class_limit = max(10_000, partition.total_gate_count // 1_000_000)
    small_indices = tuple(
        index
        for index, gate_class in enumerate(partition.classes)
        if gate_class.gate_count <= small_class_limit
    )
    if small_indices:
        seed_candidate = _quota_with_fully_checked_classes(
            partition,
            checked_gate_budget,
            frozenset(small_indices),
        )
        if seed_candidate is not None:
            seed_candidates.append(seed_candidate)
    for index in small_indices:
        seed_candidate = _quota_with_fully_checked_classes(
            partition,
            checked_gate_budget,
            frozenset((index,)),
        )
        if seed_candidate is not None:
            seed_candidates.append(seed_candidate)
    seeds = tuple(
        {strategy.checked_counts: strategy for strategy in seed_candidates}.values()
    )[:max_evaluations]
    cache: dict[tuple[int, ...], WeightedClassAttack | None] = {}

    def evaluate(counts: tuple[int, ...]) -> WeightedClassAttack | None:
        cached = cache.get(counts)
        if cached is not None or counts in cache:
            return cached
        if len(cache) >= max_evaluations:
            raise RuntimeError("fixed-quota search exhausted its evaluation budget")
        strategy = fixed_quota_strategy(partition, counts)
        result = stratified_quota_best_response(
            partition,
            strategy,
            detection_threshold=threshold,
        )
        cache[counts] = result
        return result

    current = min(
        seeds,
        key=lambda strategy: _quota_score(evaluate(strategy.checked_counts)),
    )
    current_attack = evaluate(current.checked_counts)

    largest_movable = max(
        (
            min(
                current.checked_counts[donor],
                partition.classes[receiver].gate_count
                - current.checked_counts[receiver],
            )
            for donor in range(len(partition.classes))
            for receiver in range(len(partition.classes))
            if donor != receiver
        ),
        default=0,
    )
    step = 1 << (largest_movable.bit_length() - 1) if largest_movable else 0
    while step and len(cache) < max_evaluations:
        best_counts = current.checked_counts
        best_attack = current_attack
        for donor in range(len(partition.classes)):
            for receiver in range(len(partition.classes)):
                if donor == receiver:
                    continue
                amount = min(
                    step,
                    current.checked_counts[donor],
                    partition.classes[receiver].gate_count
                    - current.checked_counts[receiver],
                )
                if amount <= 0:
                    continue
                candidate = list(current.checked_counts)
                candidate[donor] -= amount
                candidate[receiver] += amount
                candidate_tuple = tuple(candidate)
                if candidate_tuple in cache:
                    attack = cache[candidate_tuple]
                elif len(cache) >= max_evaluations:
                    break
                else:
                    attack = evaluate(candidate_tuple)
                if _quota_score(attack) < _quota_score(best_attack):
                    best_counts = candidate_tuple
                    best_attack = attack
            if len(cache) >= max_evaluations:
                break
        if best_counts != current.checked_counts:
            current = fixed_quota_strategy(partition, best_counts)
            current_attack = best_attack
        else:
            step //= 2

    return FixedQuotaOptimization(
        strategy=current,
        worst_attack=current_attack,
        checked_gate_budget=checked_gate_budget,
        detection_threshold=threshold,
        evaluated_quota_count=len(cache),
    )
