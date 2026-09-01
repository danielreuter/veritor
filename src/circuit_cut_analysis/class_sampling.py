"""Exact class-symmetric, fixed-count verification-sampling games.

This module implements the verifier game in which scalar gates remain the
sampling and attack atoms, while a partition only ties gates that must receive
the same treatment.  A verifier outcome chooses an integer quota ``k_i`` for
each probability class ``C_i``, with ``sum_i k_i = B``, and then samples
``k_i`` gates uniformly without replacement inside that class.  Randomizing
over quota vectors represents every exact-``B`` strategy invariant under
permutations within the declared classes.

For an attack containing ``e_i`` gates from class ``i``, the probability that
one quota outcome misses every attacked gate is exactly

``product_i choose(|C_i| - e_i, k_i) / choose(|C_i|, k_i)``.

The finite optimizer below enumerates actual attacked gate sets, not merely
class counts, so gates incorrectly grouped into one class remain distinguishable
to the adversary.  This provides the exact small-instance oracle against which
adaptive, indexed large-model solvers can be tested.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType

from scipy.optimize import linprog

from circuit_cut_analysis.capacity import LogCardinality
from circuit_cut_analysis.circuit import CircuitDAG, GateId
from circuit_cut_analysis.mincut import CutResult, minimum_vertex_cut


def _probability(value: int | float | Fraction) -> Fraction:
    result = Fraction(str(value)) if isinstance(value, float) else Fraction(value)
    if not 0 <= result <= 1:
        raise ValueError(f"probability must lie in [0, 1], got {value!r}")
    return result


@dataclass(frozen=True, slots=True, init=False)
class GateProbabilityClass:
    """A set of scalar gates constrained to symmetric verifier treatment."""

    id: str
    gates: frozenset[GateId]

    def __init__(self, id: str, gates: Iterable[GateId]) -> None:
        gate_set = frozenset(gates)
        if not id:
            raise ValueError("probability-class id must be non-empty")
        if not gate_set:
            raise ValueError(f"probability class {id!r} must contain a gate")
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "gates", gate_set)

    @property
    def size(self) -> int:
        return len(self.gates)


@dataclass(frozen=True, slots=True, init=False)
class GateClassPartition:
    """A disjoint exhaustive probability-class partition of scalar gates."""

    classes: tuple[GateProbabilityClass, ...]
    eligible_gates: frozenset[GateId]
    class_by_id: Mapping[str, GateProbabilityClass]
    class_for_gate: Mapping[GateId, str]

    def __init__(
        self,
        classes: Iterable[GateProbabilityClass],
        eligible_gates: Iterable[GateId],
    ) -> None:
        class_tuple = tuple(classes)
        eligible = frozenset(eligible_gates)
        if not class_tuple:
            raise ValueError("gate-class partition needs at least one class")
        if not eligible:
            raise ValueError("gate-class partition needs eligible gates")

        by_id: dict[str, GateProbabilityClass] = {}
        gate_to_class: dict[GateId, str] = {}
        for gate_class in class_tuple:
            if gate_class.id in by_id:
                raise ValueError(f"duplicate probability-class id: {gate_class.id!r}")
            by_id[gate_class.id] = gate_class
            for gate_id in gate_class.gates:
                previous = gate_to_class.setdefault(gate_id, gate_class.id)
                if previous != gate_class.id:
                    raise ValueError(
                        f"gate {gate_id!r} belongs to both {previous!r} "
                        f"and {gate_class.id!r}"
                    )

        covered = frozenset(gate_to_class)
        if covered != eligible:
            missing = sorted(eligible.difference(covered))
            extra = sorted(covered.difference(eligible))
            raise ValueError(
                "probability classes must partition eligible gates exactly; "
                f"missing={missing[:3]!r}, extra={extra[:3]!r}"
            )

        object.__setattr__(self, "classes", class_tuple)
        object.__setattr__(self, "eligible_gates", eligible)
        object.__setattr__(self, "class_by_id", MappingProxyType(by_id))
        object.__setattr__(self, "class_for_gate", MappingProxyType(gate_to_class))

    @classmethod
    def singleton_gates(cls, gates: Iterable[GateId]) -> GateClassPartition:
        """Create one probability class per scalar gate."""

        gate_set = frozenset(gates)
        return cls(
            (GateProbabilityClass(gate_id, (gate_id,)) for gate_id in sorted(gate_set)),
            gate_set,
        )

    @property
    def gate_count(self) -> int:
        return len(self.eligible_gates)

    @property
    def class_ids(self) -> tuple[str, ...]:
        return tuple(gate_class.id for gate_class in self.classes)

    @property
    def class_sizes(self) -> tuple[int, ...]:
        return tuple(gate_class.size for gate_class in self.classes)

    def attack_counts(self, error_gates: Iterable[GateId]) -> tuple[int, ...]:
        """Count attacked scalar gates in each probability class."""

        errors = frozenset(error_gates)
        unknown = errors.difference(self.eligible_gates)
        if unknown:
            raise ValueError(f"attack names ineligible gates: {sorted(unknown)!r}")
        counts = {class_id: 0 for class_id in self.class_by_id}
        for gate_id in errors:
            counts[self.class_for_gate[gate_id]] += 1
        return tuple(counts[gate_class.id] for gate_class in self.classes)


def quota_miss_probability(
    class_sizes: Sequence[int],
    checked_counts: Sequence[int],
    attacked_counts: Sequence[int],
) -> Fraction:
    """Return the exact probability one class-quota outcome misses an attack."""

    if not (len(class_sizes) == len(checked_counts) == len(attacked_counts)):
        raise ValueError("class, checked, and attacked count vectors must align")
    miss = Fraction(1)
    for size, checked, attacked in zip(
        class_sizes, checked_counts, attacked_counts, strict=True
    ):
        if size <= 0:
            raise ValueError("class sizes must be positive")
        if not 0 <= checked <= size:
            raise ValueError("checked count is outside its probability class")
        if not 0 <= attacked <= size:
            raise ValueError("attacked count is outside its probability class")
        if checked > size - attacked:
            return Fraction()
        miss *= Fraction(
            math.comb(size - attacked, checked),
            math.comb(size, checked),
        )
    return miss


@dataclass(frozen=True, slots=True)
class ClassQuotaOutcome:
    """One exact checked-gate quota vector and its strategy probability."""

    checked_counts: tuple[int, ...]
    probability: Fraction

    def __init__(
        self,
        checked_counts: Iterable[int],
        probability: int | float | Fraction,
    ) -> None:
        counts = tuple(checked_counts)
        parsed = _probability(probability)
        if parsed == 0:
            raise ValueError("zero-probability outcomes must be omitted")
        if any(count < 0 for count in counts):
            raise ValueError("checked counts cannot be negative")
        object.__setattr__(self, "checked_counts", counts)
        object.__setattr__(self, "probability", parsed)


@dataclass(frozen=True, slots=True, init=False)
class ClassSymmetricStrategy:
    """A distribution over exact-budget class quota vectors."""

    class_ids: tuple[str, ...]
    class_sizes: tuple[int, ...]
    checked_gate_budget: int
    outcomes: tuple[ClassQuotaOutcome, ...]

    def __init__(
        self,
        partition: GateClassPartition,
        checked_gate_budget: int,
        outcomes: Iterable[ClassQuotaOutcome],
    ) -> None:
        if not 0 <= checked_gate_budget <= partition.gate_count:
            raise ValueError("checked-gate budget is outside eligible gates")
        outcome_tuple = tuple(outcomes)
        if not outcome_tuple:
            raise ValueError("class-symmetric strategy needs at least one outcome")
        seen: set[tuple[int, ...]] = set()
        total = Fraction()
        for outcome in outcome_tuple:
            if len(outcome.checked_counts) != len(partition.classes):
                raise ValueError("quota vector does not match probability classes")
            if outcome.checked_counts in seen:
                raise ValueError(f"duplicate quota outcome: {outcome.checked_counts!r}")
            seen.add(outcome.checked_counts)
            if any(
                checked > gate_class.size
                for checked, gate_class in zip(
                    outcome.checked_counts, partition.classes, strict=True
                )
            ):
                raise ValueError("quota exceeds a probability-class size")
            if sum(outcome.checked_counts) != checked_gate_budget:
                raise ValueError(
                    "every quota outcome must check exactly the declared budget"
                )
            total += outcome.probability
        if total != 1:
            raise ValueError(f"strategy probabilities must sum to one, got {total}")

        object.__setattr__(self, "class_ids", partition.class_ids)
        object.__setattr__(self, "class_sizes", partition.class_sizes)
        object.__setattr__(self, "checked_gate_budget", checked_gate_budget)
        object.__setattr__(self, "outcomes", outcome_tuple)

    def validate(self, partition: GateClassPartition) -> None:
        if self.class_ids != partition.class_ids:
            raise ValueError("strategy probability classes do not match partition")
        if self.class_sizes != partition.class_sizes:
            raise ValueError("strategy probability-class sizes do not match partition")

    def detection_probability_from_counts(
        self,
        attacked_counts: Sequence[int],
    ) -> Fraction:
        """Return exact detection probability for one attack count vector."""

        miss = sum(
            (
                outcome.probability
                * quota_miss_probability(
                    self.class_sizes,
                    outcome.checked_counts,
                    attacked_counts,
                )
                for outcome in self.outcomes
            ),
            Fraction(),
        )
        return 1 - miss

    def detection_probability(
        self,
        error_gates: Iterable[GateId],
        partition: GateClassPartition,
    ) -> Fraction:
        """Return exact probability that at least one attacked gate is checked."""

        self.validate(partition)
        return self.detection_probability_from_counts(
            partition.attack_counts(error_gates)
        )

    @property
    def class_inclusion_probabilities(self) -> Mapping[str, Fraction]:
        """Return the scalar-gate inclusion probability for each class."""

        expected_counts = [
            sum(
                (
                    outcome.probability * outcome.checked_counts[index]
                    for outcome in self.outcomes
                ),
                Fraction(),
            )
            for index in range(len(self.class_ids))
        ]
        return MappingProxyType(
            {
                class_id: expected / size
                for class_id, size, expected in zip(
                    self.class_ids,
                    self.class_sizes,
                    expected_counts,
                    strict=True,
                )
            }
        )


def enumerate_class_quotas(
    class_sizes: Sequence[int],
    checked_gate_budget: int,
    *,
    max_actions: int = 100_000,
) -> tuple[tuple[int, ...], ...]:
    """Enumerate all bounded quota vectors summing to an exact gate budget."""

    sizes = tuple(class_sizes)
    if any(size <= 0 for size in sizes):
        raise ValueError("class sizes must be positive")
    if not 0 <= checked_gate_budget <= sum(sizes):
        raise ValueError("checked-gate budget is outside class sizes")
    if max_actions <= 0:
        raise ValueError("max_actions must be positive")

    suffix_sizes = [0] * (len(sizes) + 1)
    for index in range(len(sizes) - 1, -1, -1):
        suffix_sizes[index] = suffix_sizes[index + 1] + sizes[index]

    quotas: list[tuple[int, ...]] = []
    prefix: list[int] = []

    def visit(index: int, remaining: int) -> None:
        if len(quotas) > max_actions:
            return
        if index == len(sizes):
            if remaining == 0:
                quotas.append(tuple(prefix))
            return
        lower = max(0, remaining - suffix_sizes[index + 1])
        upper = min(sizes[index], remaining)
        for checked in range(lower, upper + 1):
            prefix.append(checked)
            visit(index + 1, remaining - checked)
            prefix.pop()

    visit(0, checked_gate_budget)
    if len(quotas) > max_actions:
        raise ValueError(
            f"exact budget has more than {max_actions:,} class-quota actions"
        )
    return tuple(quotas)


@dataclass(frozen=True, slots=True)
class ClassSymmetricAttack:
    """An actual scalar-gate attack and its exact finite-circuit evidence."""

    error_gates: frozenset[GateId]
    attacked_counts: tuple[int, ...]
    detection_probability: Fraction
    cut_result: CutResult

    @property
    def log2_reachable_bound(self) -> LogCardinality:
        capacity = self.cut_result.exact_capacity
        if capacity is None:
            raise AssertionError("all-gate cut policy must yield exact capacity")
        return capacity


@dataclass(frozen=True, slots=True)
class FiniteClassSamplingOptimization:
    """Numerical LP candidate with exact strategy and adversary replay."""

    strategy: ClassSymmetricStrategy
    checked_gate_budget: int
    detection_threshold: Fraction
    worst_attack: ClassSymmetricAttack | None
    replay_certified_log2_reachable_bound: LogCardinality
    lp_candidate_log2_reachable_bound: LogCardinality
    candidate_quota_action_count: int
    adversarial_gate_set_count: int
    global_optimality_certified: bool = False


@dataclass(frozen=True, slots=True)
class _FiniteAttackScenario:
    error_gates: frozenset[GateId]
    attacked_counts: tuple[int, ...]
    capacity: LogCardinality
    cut_result: CutResult


def _strategy_from_solution(
    partition: GateClassPartition,
    checked_gate_budget: int,
    quotas: Sequence[tuple[int, ...]],
    solution: Iterable[float],
    *,
    tolerance: float,
) -> ClassSymmetricStrategy:
    retained = [
        (quota, Fraction(float(max(probability, 0.0))).limit_denominator(10**9))
        for quota, probability in zip(quotas, solution, strict=True)
        if probability > tolerance
    ]
    if not retained:
        raise RuntimeError("linear program returned an empty strategy")
    total = sum((probability for _, probability in retained), Fraction())
    return ClassSymmetricStrategy(
        partition,
        checked_gate_budget,
        (
            ClassQuotaOutcome(quota, probability / total)
            for quota, probability in retained
        ),
    )


def _best_response_from_scenarios(
    strategy: ClassSymmetricStrategy,
    scenarios: Sequence[_FiniteAttackScenario],
    threshold: Fraction,
) -> ClassSymmetricAttack | None:
    best: ClassSymmetricAttack | None = None
    for scenario in scenarios:
        detection = strategy.detection_probability_from_counts(scenario.attacked_counts)
        if detection >= threshold:
            continue
        candidate = ClassSymmetricAttack(
            error_gates=scenario.error_gates,
            attacked_counts=scenario.attacked_counts,
            detection_probability=detection,
            cut_result=scenario.cut_result,
        )
        if best is None:
            best = candidate
            continue
        candidate_key = (
            candidate.log2_reachable_bound,
            -candidate.detection_probability,
            tuple(sorted(candidate.error_gates)),
        )
        best_key = (
            best.log2_reachable_bound,
            -best.detection_probability,
            tuple(sorted(best.error_gates)),
        )
        if candidate_key > best_key:
            best = candidate
    return best


def optimize_class_sampling_finite(
    circuit: CircuitDAG,
    partition: GateClassPartition,
    *,
    checked_gate_budget: int,
    detection_threshold: int | float | Fraction = Fraction(99, 100),
    outputs: Iterable[GateId] | None = None,
    attackable_gates: Iterable[GateId] | None = None,
    max_attackable_gates: int = 18,
    max_quota_actions: int = 100_000,
    numerical_tolerance: float = 1e-9,
) -> FiniteClassSamplingOptimization:
    """Solve the complete class-symmetric exact-count game on a small circuit.

    Every nonempty subset of ``attackable_gates`` is evaluated with the exact
    multi-source minimum-cut oracle.  Every exact-budget class quota vector is
    a verifier action.  For each candidate capacity ``K``, a linear program
    asks whether a distribution over quota vectors detects every attack with
    ``lambda(E) > K`` at or above the threshold.

    The returned strategy and worst attack are replayed in exact rational
    arithmetic.  LP infeasibility is numerical, so the result deliberately
    does not claim a formal global-optimality certificate.
    """

    if not 0 <= checked_gate_budget <= partition.gate_count:
        raise ValueError("checked-gate budget is outside eligible gates")
    threshold = _probability(detection_threshold)
    if not 0 < threshold <= 1:
        raise ValueError("detection threshold must be positive")
    if numerical_tolerance <= 0:
        raise ValueError("numerical tolerance must be positive")

    circuit.require_gates(partition.eligible_gates)
    attackable = (
        partition.eligible_gates
        if attackable_gates is None
        else circuit.require_gates(attackable_gates)
    )
    if not attackable.issubset(partition.eligible_gates):
        raise ValueError("every attackable gate must belong to a probability class")
    if len(attackable) > max_attackable_gates:
        raise ValueError(
            f"attack enumeration has {len(attackable)} gates; "
            f"limit is {max_attackable_gates}"
        )

    quotas = enumerate_class_quotas(
        partition.class_sizes,
        checked_gate_budget,
        max_actions=max_quota_actions,
    )
    attack_ids = tuple(sorted(attackable))
    scenarios: list[_FiniteAttackScenario] = []
    for attacked_count in range(1, len(attack_ids) + 1):
        for attacked_tuple in itertools.combinations(attack_ids, attacked_count):
            errors = frozenset(attacked_tuple)
            cut_result = minimum_vertex_cut(circuit, errors, outputs)
            capacity = cut_result.exact_capacity
            if capacity is None:
                raise AssertionError("all-gate cut policy must yield exact capacity")
            scenarios.append(
                _FiniteAttackScenario(
                    error_gates=errors,
                    attacked_counts=partition.attack_counts(errors),
                    capacity=capacity,
                    cut_result=cut_result,
                )
            )

    candidate_bounds = sorted(
        {LogCardinality.zero(), *(scenario.capacity for scenario in scenarios)}
    )
    selected_strategy: ClassSymmetricStrategy | None = None
    selected_bound: LogCardinality | None = None
    for candidate_bound in candidate_bounds:
        bad_scenarios = tuple(
            scenario for scenario in scenarios if scenario.capacity > candidate_bound
        )
        detection_rows = [
            [
                float(
                    1
                    - quota_miss_probability(
                        partition.class_sizes,
                        quota,
                        scenario.attacked_counts,
                    )
                )
                for quota in quotas
            ]
            for scenario in bad_scenarios
        ]
        result = linprog(
            c=[0.0] * len(quotas),
            A_ub=[[-coefficient for coefficient in row] for row in detection_rows]
            or None,
            b_ub=[-float(threshold)] * len(detection_rows) or None,
            A_eq=[[1.0] * len(quotas)],
            b_eq=[1.0],
            bounds=(0.0, None),
            method="highs",
            options={
                "dual_feasibility_tolerance": numerical_tolerance,
                "primal_feasibility_tolerance": numerical_tolerance,
            },
        )
        if not result.success or result.x is None:
            continue
        strategy = _strategy_from_solution(
            partition,
            checked_gate_budget,
            quotas,
            result.x,
            tolerance=numerical_tolerance,
        )
        if any(
            strategy.detection_probability_from_counts(scenario.attacked_counts)
            < threshold
            for scenario in bad_scenarios
        ):
            continue
        selected_strategy = strategy
        selected_bound = candidate_bound
        break

    if selected_strategy is None or selected_bound is None:
        raise RuntimeError("no replay-certified exact-budget strategy found")

    worst_attack = _best_response_from_scenarios(
        selected_strategy,
        scenarios,
        threshold,
    )
    achieved_bound = (
        LogCardinality.zero()
        if worst_attack is None
        else worst_attack.log2_reachable_bound
    )
    if achieved_bound > selected_bound:
        raise AssertionError("LP candidate disagrees with exact adversarial replay")
    return FiniteClassSamplingOptimization(
        strategy=selected_strategy,
        checked_gate_budget=checked_gate_budget,
        detection_threshold=threshold,
        worst_attack=worst_attack,
        replay_certified_log2_reachable_bound=achieved_bound,
        lp_candidate_log2_reachable_bound=selected_bound,
        candidate_quota_action_count=len(quotas),
        adversarial_gate_set_count=len(scenarios),
    )
