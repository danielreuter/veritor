"""Verification-sampling games on safely small explicit circuits.

The circuit and input are fixed.  A verification partition divides the
eligible computed gates into units.  A sampling strategy is an arbitrary
distribution over sets of units to verify.  If an adversary corrupts any gate
in a sampled unit, that outcome detects the attack.

For one attacked set of units, corrupting every gate in those units weakly
maximizes the structural bound because bottleneck width is monotone in the
source set.  The exact finite oracle therefore enumerates attacked unit sets,
unions their gates, and evaluates ``lambda(E)`` with the existing min-cut
solver.  This module is deliberately exponential: it supplies exact attack
replay and a finite-action numerical LP baseline for compressed and
model-specific optimizers, not the full-model algorithm.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from types import MappingProxyType

from scipy.optimize import linprog

from circuit_cut_analysis.capacity import LogCardinality
from circuit_cut_analysis.capacity_oracle import StructuralCapacityEvaluation
from circuit_cut_analysis.circuit import CircuitDAG, GateId
from circuit_cut_analysis.mincut import CutResult, minimum_vertex_cut

Probability = Fraction
_UNCHECKED_ATTACK_UNIT_ID = "__unchecked_attackable_gates__"


def _probability(value: int | float | Fraction) -> Fraction:
    if isinstance(value, float):
        result = Fraction(str(value))
    else:
        result = Fraction(value)
    if not 0 <= result <= 1:
        raise ValueError(f"probability must lie in [0, 1], got {value!r}")
    return result


@dataclass(frozen=True, slots=True)
class VerificationUnit:
    """One indivisible sampling unit and the scalar gates it verifies."""

    id: str
    gates: frozenset[GateId]

    def __init__(self, id: str, gates: Iterable[GateId]) -> None:
        gate_set = frozenset(gates)
        if not id:
            raise ValueError("verification-unit id must be non-empty")
        if not gate_set:
            raise ValueError(f"verification unit {id!r} must contain a gate")
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "gates", gate_set)

    @property
    def checked_gate_count(self) -> int:
        """Number of scalar gates checked when this unit is selected."""

        return len(self.gates)


@dataclass(frozen=True, slots=True)
class VerificationPartition:
    """A disjoint, exhaustive partition of eligible computed gates."""

    units: tuple[VerificationUnit, ...]
    eligible_gates: frozenset[GateId]
    unit_by_id: Mapping[str, VerificationUnit]
    unit_for_gate: Mapping[GateId, str]

    def __init__(
        self,
        units: Iterable[VerificationUnit],
        eligible_gates: Iterable[GateId],
    ) -> None:
        unit_tuple = tuple(units)
        eligible = frozenset(eligible_gates)
        if not unit_tuple:
            raise ValueError("verification partition needs at least one unit")
        if not eligible:
            raise ValueError("verification partition needs eligible gates")

        by_id: dict[str, VerificationUnit] = {}
        gate_to_unit: dict[GateId, str] = {}
        for unit in unit_tuple:
            if unit.id in by_id:
                raise ValueError(f"duplicate verification-unit id: {unit.id!r}")
            by_id[unit.id] = unit
            for gate_id in unit.gates:
                previous = gate_to_unit.setdefault(gate_id, unit.id)
                if previous != unit.id:
                    raise ValueError(
                        f"gate {gate_id!r} belongs to both {previous!r} and {unit.id!r}"
                    )

        covered = frozenset(gate_to_unit)
        if covered != eligible:
            missing = sorted(eligible.difference(covered))
            extra = sorted(covered.difference(eligible))
            raise ValueError(
                "verification units must partition eligible gates exactly; "
                f"missing={missing[:3]!r}, extra={extra[:3]!r}"
            )

        object.__setattr__(self, "units", unit_tuple)
        object.__setattr__(self, "eligible_gates", eligible)
        object.__setattr__(self, "unit_by_id", MappingProxyType(by_id))
        object.__setattr__(self, "unit_for_gate", MappingProxyType(gate_to_unit))

    @classmethod
    def singleton_gates(
        cls,
        gates: Iterable[GateId],
    ) -> VerificationPartition:
        """Create the ideal one-verification-unit-per-gate partition."""

        gate_set = frozenset(gates)
        return cls(
            (VerificationUnit(gate_id, (gate_id,)) for gate_id in sorted(gate_set)),
            gate_set,
        )

    @property
    def checked_gate_count(self) -> int:
        return len(self.eligible_gates)

    def require_unit_ids(self, unit_ids: Iterable[str]) -> frozenset[str]:
        result = frozenset(unit_ids)
        unknown = result.difference(self.unit_by_id)
        if unknown:
            raise ValueError(f"unknown verification units: {sorted(unknown)!r}")
        return result

    def gates_for_units(self, unit_ids: Iterable[str]) -> frozenset[GateId]:
        selected = self.require_unit_ids(unit_ids)
        return frozenset(
            gate_id
            for unit_id in selected
            for gate_id in self.unit_by_id[unit_id].gates
        )


@dataclass(frozen=True, slots=True)
class SamplingOutcome:
    """One sampled unit set in a finite strategy distribution."""

    sampled_units: frozenset[str]
    probability: Fraction

    def __init__(
        self,
        sampled_units: Iterable[str],
        probability: int | float | Fraction,
    ) -> None:
        parsed = _probability(probability)
        if parsed == 0:
            raise ValueError("zero-probability outcomes must be omitted")
        object.__setattr__(self, "sampled_units", frozenset(sampled_units))
        object.__setattr__(self, "probability", parsed)


@dataclass(frozen=True, slots=True)
class SamplingStrategy:
    """An arbitrary finite distribution over sets of verification units."""

    outcomes: tuple[SamplingOutcome, ...]

    def __init__(self, outcomes: Iterable[SamplingOutcome]) -> None:
        outcome_tuple = tuple(outcomes)
        if not outcome_tuple:
            raise ValueError("sampling strategy needs at least one outcome")
        support: set[frozenset[str]] = set()
        total = Fraction()
        for outcome in outcome_tuple:
            if outcome.sampled_units in support:
                raise ValueError(
                    f"duplicate sampled-unit outcome: {sorted(outcome.sampled_units)!r}"
                )
            support.add(outcome.sampled_units)
            total += outcome.probability
        if total != 1:
            raise ValueError(f"sampling probabilities must sum to one, got {total}")
        object.__setattr__(self, "outcomes", outcome_tuple)

    def validate(self, partition: VerificationPartition) -> None:
        for outcome in self.outcomes:
            partition.require_unit_ids(outcome.sampled_units)

    def detection_probability(
        self,
        attacked_units: Iterable[str],
        partition: VerificationPartition,
    ) -> Fraction:
        """Probability that at least one attacked unit is sampled."""

        self.validate(partition)
        attacked = partition.require_unit_ids(attacked_units)
        return sum(
            (
                outcome.probability
                for outcome in self.outcomes
                if attacked.intersection(outcome.sampled_units)
            ),
            Fraction(),
        )

    def expected_checked_gates(self, partition: VerificationPartition) -> Fraction:
        """Expected number of scalar gates checked in one verifier run."""

        self.validate(partition)
        return sum(
            (
                outcome.probability
                * sum(
                    partition.unit_by_id[unit_id].checked_gate_count
                    for unit_id in outcome.sampled_units
                )
                for outcome in self.outcomes
            ),
            Fraction(),
        )

    def expected_checked_fraction(
        self,
        partition: VerificationPartition,
    ) -> Fraction:
        return self.expected_checked_gates(partition) / partition.checked_gate_count


def independent_unit_sampling(
    probabilities: Mapping[str, int | float | Fraction],
    partition: VerificationPartition,
    *,
    max_units: int = 20,
) -> SamplingStrategy:
    """Enumerate independent Bernoulli unit sampling for a small partition."""

    unit_ids = tuple(sorted(partition.unit_by_id))
    if len(unit_ids) > max_units:
        raise ValueError(
            f"independent enumeration has {len(unit_ids)} units; limit is {max_units}"
        )
    unknown = set(probabilities).difference(unit_ids)
    if unknown:
        raise ValueError(f"probabilities name unknown units: {sorted(unknown)!r}")
    parsed = {
        unit_id: _probability(probabilities.get(unit_id, 0)) for unit_id in unit_ids
    }

    outcomes: list[SamplingOutcome] = []
    for selected_count in range(len(unit_ids) + 1):
        for selected_tuple in itertools.combinations(unit_ids, selected_count):
            selected = frozenset(selected_tuple)
            probability = Fraction(1)
            for unit_id in unit_ids:
                unit_probability = parsed[unit_id]
                probability *= (
                    unit_probability if unit_id in selected else 1 - unit_probability
                )
            if probability:
                outcomes.append(SamplingOutcome(selected, probability))
    return SamplingStrategy(outcomes)


def uniform_fixed_count_sampling(
    partition: VerificationPartition,
    sampled_unit_count: int,
) -> SamplingStrategy:
    """Sample exactly ``sampled_unit_count`` units uniformly without replacement."""

    unit_ids = tuple(sorted(partition.unit_by_id))
    if not 0 <= sampled_unit_count <= len(unit_ids):
        raise ValueError("sampled unit count is outside the partition")
    selections = tuple(itertools.combinations(unit_ids, sampled_unit_count))
    probability = Fraction(1, len(selections))
    return SamplingStrategy(
        SamplingOutcome(selection, probability) for selection in selections
    )


@dataclass(frozen=True, slots=True)
class AdversarialAttack:
    """One admissible attacked-unit set and its exact structural bound.

    ``attacked_units`` normally names verification units.  If attackable gates
    lie outside the verification partition, one reserved synthetic unit groups
    all of those never-checked gates; corrupting all of them has no additional
    detection cost and weakly maximizes structural capacity.
    """

    attacked_units: frozenset[str]
    error_gates: frozenset[GateId]
    detection_probability: Fraction
    cut_result: CutResult

    @property
    def log2_reachable_bound(self) -> LogCardinality:
        capacity = self.cut_result.exact_capacity
        if capacity is None:
            raise AssertionError("all-gate cuts must have finite exact capacity")
        return capacity


def _resolve_attackable_gates(
    circuit: CircuitDAG,
    partition: VerificationPartition,
    attackable_gates: Iterable[GateId] | None,
) -> frozenset[GateId]:
    if attackable_gates is None:
        return partition.eligible_gates
    return circuit.require_gates(attackable_gates)


def _attack_units(
    partition: VerificationPartition,
    attackable_gates: frozenset[GateId],
) -> Mapping[str, frozenset[GateId]]:
    if _UNCHECKED_ATTACK_UNIT_ID in partition.unit_by_id:
        raise ValueError(
            f"verification-unit id {_UNCHECKED_ATTACK_UNIT_ID!r} is reserved"
        )
    units = {
        unit.id: unit.gates.intersection(attackable_gates)
        for unit in partition.units
        if unit.gates.intersection(attackable_gates)
    }
    unchecked = attackable_gates.difference(partition.eligible_gates)
    if unchecked:
        units[_UNCHECKED_ATTACK_UNIT_ID] = unchecked
    return MappingProxyType(units)


def _attack_detection_probability(
    strategy: SamplingStrategy,
    attacked_units: frozenset[str],
) -> Fraction:
    return sum(
        (
            outcome.probability
            for outcome in strategy.outcomes
            if attacked_units.intersection(outcome.sampled_units)
        ),
        Fraction(),
    )


def adversarial_best_response(
    circuit: CircuitDAG,
    partition: VerificationPartition,
    strategy: SamplingStrategy,
    *,
    detection_threshold: int | float | Fraction = Fraction(99, 100),
    outputs: Iterable[GateId] | None = None,
    attackable_gates: Iterable[GateId] | None = None,
    max_units: int = 20,
) -> AdversarialAttack | None:
    """Return the exact best attack detected with probability below a threshold.

    The adversary chooses a nonempty set of attack units.  A checked attack
    unit is the intersection of one verification unit with ``attackable_gates``;
    all attackable gates outside the partition form one never-detected unit.
    Because ``lambda(A)`` is monotone in ``A``, the strongest attack touching a
    unit corrupts every attackable gate in it.  Ties in exact cut capacity are
    resolved by lower detection probability and then lexicographic unit IDs.
    """

    threshold = _probability(detection_threshold)
    strategy.validate(partition)
    circuit.require_gates(partition.eligible_gates)
    attackable = _resolve_attackable_gates(circuit, partition, attackable_gates)
    attack_units = _attack_units(partition, attackable)
    unit_ids = tuple(sorted(attack_units))
    if len(unit_ids) > max_units:
        raise ValueError(
            f"attack enumeration has {len(unit_ids)} units; limit is {max_units}"
        )

    best: AdversarialAttack | None = None
    for attacked_count in range(1, len(unit_ids) + 1):
        for attacked_tuple in itertools.combinations(unit_ids, attacked_count):
            attacked = frozenset(attacked_tuple)
            detection = _attack_detection_probability(strategy, attacked)
            if detection >= threshold:
                continue
            error_gates = frozenset(
                gate_id for unit_id in attacked for gate_id in attack_units[unit_id]
            )
            cut = minimum_vertex_cut(circuit, error_gates, outputs)
            candidate = AdversarialAttack(
                attacked_units=attacked,
                error_gates=error_gates,
                detection_probability=detection,
                cut_result=cut,
            )
            if best is None:
                best = candidate
                continue
            candidate_key = (
                candidate.log2_reachable_bound,
                -candidate.detection_probability,
                tuple(sorted(candidate.attacked_units)),
            )
            best_key = (
                best.log2_reachable_bound,
                -best.detection_probability,
                tuple(sorted(best.attacked_units)),
            )
            if candidate_key > best_key:
                best = candidate
    return best


class SamplingBudgetMode(StrEnum):
    """How a checked-gate budget constrains the verifier's distribution."""

    EXPECTED = "expected"
    HARD = "hard"


class SamplingOptimizationStatus(StrEnum):
    """Strength of the optimization claim returned by a sampling solver."""

    NUMERICAL_LP_REPLAY_CERTIFIED = "numerical-lp-replay-certified"


@dataclass(frozen=True, slots=True)
class FiniteSamplingOptimization:
    """Finite-action numerical LP candidate with exact adversarial replay."""

    strategy: SamplingStrategy
    budget_mode: SamplingBudgetMode
    checked_gate_budget: Fraction
    detection_threshold: Fraction
    worst_attack: AdversarialAttack | None
    replay_certified_log2_reachable_bound: LogCardinality
    lp_candidate_log2_reachable_bound: LogCardinality
    candidate_sampling_action_count: int
    adversarial_action_count: int
    status: SamplingOptimizationStatus = (
        SamplingOptimizationStatus.NUMERICAL_LP_REPLAY_CERTIFIED
    )
    global_optimality_certified: bool = False


@dataclass(frozen=True, slots=True)
class _AttackScenario:
    attacked_units: frozenset[str]
    capacity: LogCardinality


def _unit_subsets(unit_ids: tuple[str, ...]) -> tuple[frozenset[str], ...]:
    return tuple(
        frozenset(selection)
        for size in range(len(unit_ids) + 1)
        for selection in itertools.combinations(unit_ids, size)
    )


def _action_checked_gates(
    action: frozenset[str],
    partition: VerificationPartition,
) -> int:
    return sum(partition.unit_by_id[unit_id].checked_gate_count for unit_id in action)


def _sampling_strategy_from_lp(
    actions: tuple[frozenset[str], ...],
    solution: Iterable[float],
    *,
    tolerance: float,
) -> SamplingStrategy:
    retained = [
        (action, Fraction(str(max(probability, 0.0))))
        for action, probability in zip(actions, solution, strict=True)
        if probability > tolerance
    ]
    if not retained:
        raise RuntimeError("linear program returned an empty strategy")
    total = sum((probability for _, probability in retained), Fraction())
    return SamplingStrategy(
        SamplingOutcome(action, probability / total) for action, probability in retained
    )


def _solve_bound_minimizing_lp(
    actions: tuple[frozenset[str], ...],
    costs: tuple[int, ...],
    scenarios: tuple[_AttackScenario, ...],
    *,
    budget: Fraction,
    threshold: Fraction,
    budget_mode: SamplingBudgetMode,
    numerical_tolerance: float,
    expected_cost: Callable[[SamplingStrategy], Fraction],
) -> tuple[SamplingStrategy, LogCardinality]:
    """Find the least scenario-capacity bound with a validated feasible strategy.

    For each candidate bound ``K`` every scenario with capacity above ``K``
    must be detected with at least ``threshold`` probability.  The returned
    strategy is re-validated with exact rational arithmetic after the
    floating-point solve.
    """

    candidate_bounds = sorted(
        {LogCardinality.zero(), *(scenario.capacity for scenario in scenarios)}
    )
    for candidate_bound in candidate_bounds:
        bad_scenarios = tuple(
            scenario for scenario in scenarios if scenario.capacity > candidate_bound
        )
        upper_rows: list[list[float]] = []
        upper_bounds: list[float] = []
        if budget_mode is SamplingBudgetMode.EXPECTED:
            upper_rows.append([float(cost) for cost in costs])
            upper_bounds.append(float(budget))
        for scenario in bad_scenarios:
            upper_rows.append(
                [
                    -1.0 if action.intersection(scenario.attacked_units) else 0.0
                    for action in actions
                ]
            )
            upper_bounds.append(-float(threshold))

        lp_result = linprog(
            c=[float(cost) for cost in costs],
            A_ub=upper_rows or None,
            b_ub=upper_bounds or None,
            A_eq=[[1.0] * len(actions)],
            b_eq=[1.0],
            bounds=(0.0, None),
            method="highs",
            options={
                "dual_feasibility_tolerance": numerical_tolerance,
                "primal_feasibility_tolerance": numerical_tolerance,
            },
        )
        if not lp_result.success:
            continue
        if lp_result.x is None:
            raise RuntimeError("successful linear program returned no solution")
        strategy = _sampling_strategy_from_lp(
            actions,
            lp_result.x,
            tolerance=numerical_tolerance,
        )
        if budget_mode is SamplingBudgetMode.EXPECTED and expected_cost(strategy) > (
            budget
        ):
            continue
        if any(
            _attack_detection_probability(strategy, scenario.attacked_units) < threshold
            for scenario in bad_scenarios
        ):
            continue
        return strategy, candidate_bound
    raise RuntimeError("no feasible sampling strategy found")


def optimize_sampling_strategy_finite(
    circuit: CircuitDAG,
    partition: VerificationPartition,
    *,
    checked_gate_budget: int | float | Fraction,
    detection_threshold: int | float | Fraction = Fraction(99, 100),
    budget_mode: SamplingBudgetMode = SamplingBudgetMode.EXPECTED,
    outputs: Iterable[GateId] | None = None,
    attackable_gates: Iterable[GateId] | None = None,
    max_units: int = 12,
    numerical_tolerance: float = 1e-9,
) -> FiniteSamplingOptimization:
    """Optimize the complete finite action space with a numerical LP.

    For a candidate logarithmic output bound ``K``, every attack with
    ``lambda(E) > K`` must be detected with at least ``detection_threshold``
    probability.  Those conditions, normalization, and the expected checked
    gate budget are linear in an arbitrary distribution over sampled unit
    sets.  We enumerate the finitely many distinct attack capacities and solve
    one LP feasibility problem per candidate, yielding the global optimum over
    all correlated sampling distributions on the supplied partition.

    Enumeration is exponential in the number of verification units.  This is
    an exact oracle for each attack's structural capacity and for replaying the
    returned rationalized strategy.  SciPy's LP solve is floating point, so
    failure to find a lower candidate is not a formal rational infeasibility
    certificate.  The result is therefore replay-certified but does not claim
    formally certified global optimality.
    """

    budget = (
        Fraction(str(checked_gate_budget))
        if isinstance(checked_gate_budget, float)
        else Fraction(checked_gate_budget)
    )
    if budget < 0:
        raise ValueError("checked-gate budget cannot be negative")
    if budget > partition.checked_gate_count:
        raise ValueError("checked-gate budget exceeds all eligible gates")
    threshold = _probability(detection_threshold)
    if numerical_tolerance <= 0:
        raise ValueError("numerical tolerance must be positive")

    circuit.require_gates(partition.eligible_gates)
    attackable = _resolve_attackable_gates(circuit, partition, attackable_gates)
    attack_units = _attack_units(partition, attackable)
    sampling_unit_ids = tuple(sorted(partition.unit_by_id))
    attack_unit_ids = tuple(sorted(attack_units))
    if len(sampling_unit_ids) > max_units:
        raise ValueError(
            "finite optimization has "
            f"{len(sampling_unit_ids)} sampling units; limit is {max_units}"
        )
    if len(attack_unit_ids) > max_units:
        raise ValueError(
            "finite optimization has "
            f"{len(attack_unit_ids)} attack units; limit is {max_units}"
        )

    all_subsets = _unit_subsets(sampling_unit_ids)
    action_costs = tuple(
        _action_checked_gates(action, partition) for action in all_subsets
    )
    if budget_mode is SamplingBudgetMode.HARD:
        actions_with_costs = tuple(
            (action, cost)
            for action, cost in zip(all_subsets, action_costs, strict=True)
            if cost <= budget
        )
    else:
        actions_with_costs = tuple(zip(all_subsets, action_costs, strict=True))
    actions = tuple(action for action, _ in actions_with_costs)
    costs = tuple(cost for _, cost in actions_with_costs)
    if not actions:
        raise ValueError("checked-gate budget permits no sampling action")

    scenarios: list[_AttackScenario] = []
    for attacked in _unit_subsets(attack_unit_ids):
        if not attacked:
            continue
        error_gates = frozenset(
            gate_id for unit_id in attacked for gate_id in attack_units[unit_id]
        )
        cut_result = minimum_vertex_cut(circuit, error_gates, outputs)
        capacity = cut_result.exact_capacity
        if capacity is None:
            raise AssertionError("all-gate cuts must have finite exact capacity")
        scenarios.append(_AttackScenario(attacked, capacity))

    selected_strategy, selected_bound = _solve_bound_minimizing_lp(
        actions,
        costs,
        tuple(scenarios),
        budget=budget,
        threshold=threshold,
        budget_mode=budget_mode,
        numerical_tolerance=numerical_tolerance,
        expected_cost=lambda strategy: strategy.expected_checked_gates(partition),
    )
    worst_attack = adversarial_best_response(
        circuit,
        partition,
        selected_strategy,
        detection_threshold=threshold,
        outputs=outputs,
        attackable_gates=attackable,
        max_units=max_units,
    )
    achieved_bound = (
        LogCardinality.zero()
        if worst_attack is None
        else worst_attack.log2_reachable_bound
    )
    if achieved_bound > selected_bound:
        raise AssertionError("LP certificate disagrees with exact adversarial replay")
    return FiniteSamplingOptimization(
        strategy=selected_strategy,
        budget_mode=budget_mode,
        checked_gate_budget=budget,
        detection_threshold=threshold,
        worst_attack=worst_attack,
        replay_certified_log2_reachable_bound=achieved_bound,
        lp_candidate_log2_reachable_bound=selected_bound,
        candidate_sampling_action_count=len(actions),
        adversarial_action_count=len(scenarios),
    )


CapacityEvaluator = Callable[
    [frozenset[GateId]],
    StructuralCapacityEvaluation[frozenset[GateId]],
]
"""Certified-interval ``lambda`` evaluator over explicit attacked gate sets."""


@dataclass(frozen=True, slots=True)
class RobustAdversarialAttack:
    """One sub-threshold attack together with its certified capacity interval."""

    attacked_units: frozenset[str]
    error_gates: frozenset[GateId]
    detection_probability: Fraction
    evaluation: StructuralCapacityEvaluation[frozenset[GateId]]


@dataclass(frozen=True, slots=True)
class RobustSamplingOptimization:
    """Sound sampling optimization against certified capacity upper bounds.

    ``certified_upper_log2_reachable_bound`` is the sound guarantee: every
    attack the strategy detects with probability below the threshold has a
    certified capacity upper bound at most this value.  When some scenario is
    only interval-certified, the optimization is conservative rather than
    provably optimal; ``residual_exact_lower_bound`` is the largest certified
    lower bound among sub-threshold attacks and quantifies the remaining gap.
    """

    strategy: SamplingStrategy
    budget_mode: SamplingBudgetMode
    checked_gate_budget: Fraction
    detection_threshold: Fraction
    worst_attack: RobustAdversarialAttack | None
    certified_upper_log2_reachable_bound: LogCardinality
    residual_exact_lower_bound: LogCardinality
    lp_candidate_log2_reachable_bound: LogCardinality
    candidate_sampling_action_count: int
    adversarial_action_count: int
    bounded_scenario_count: int

    @property
    def all_scenarios_exact(self) -> bool:
        return self.bounded_scenario_count == 0


def optimize_sampling_strategy_robust(
    evaluate_capacity: CapacityEvaluator,
    partition: VerificationPartition,
    *,
    checked_gate_budget: int | float | Fraction,
    detection_threshold: int | float | Fraction = Fraction(99, 100),
    budget_mode: SamplingBudgetMode = SamplingBudgetMode.EXPECTED,
    unchecked_attackable_gates: Iterable[GateId] = (),
    max_units: int = 12,
    numerical_tolerance: float = 1e-9,
) -> RobustSamplingOptimization:
    """Optimize sampling against certified capacity intervals soundly.

    Scenario capacities are the evaluators' certified *upper* bounds, so any
    attack left below the detection threshold provably reaches at most the
    reported bound.  When every evaluation is exact this coincides with the
    exact finite optimizer; otherwise it is sound but possibly conservative.
    Gates in ``unchecked_attackable_gates`` are attackable but never sampled;
    they form one reserved always-undetected attack unit.
    """

    budget = (
        Fraction(str(checked_gate_budget))
        if isinstance(checked_gate_budget, float)
        else Fraction(checked_gate_budget)
    )
    if budget < 0:
        raise ValueError("checked-gate budget cannot be negative")
    if budget > partition.checked_gate_count:
        raise ValueError("checked-gate budget exceeds all eligible gates")
    threshold = _probability(detection_threshold)
    if numerical_tolerance <= 0:
        raise ValueError("numerical tolerance must be positive")

    unchecked = frozenset(unchecked_attackable_gates).difference(
        partition.eligible_gates
    )
    attack_units: dict[str, frozenset[GateId]] = {
        unit.id: unit.gates for unit in partition.units
    }
    if unchecked:
        if _UNCHECKED_ATTACK_UNIT_ID in attack_units:
            raise ValueError(
                f"verification-unit id {_UNCHECKED_ATTACK_UNIT_ID!r} is reserved"
            )
        attack_units[_UNCHECKED_ATTACK_UNIT_ID] = unchecked

    sampling_unit_ids = tuple(sorted(partition.unit_by_id))
    attack_unit_ids = tuple(sorted(attack_units))
    if len(sampling_unit_ids) > max_units:
        raise ValueError(
            "robust optimization has "
            f"{len(sampling_unit_ids)} sampling units; limit is {max_units}"
        )
    if len(attack_unit_ids) > max_units:
        raise ValueError(
            "robust optimization has "
            f"{len(attack_unit_ids)} attack units; limit is {max_units}"
        )

    all_actions = _unit_subsets(sampling_unit_ids)
    action_costs = tuple(
        _action_checked_gates(action, partition) for action in all_actions
    )
    if budget_mode is SamplingBudgetMode.HARD:
        actions_with_costs = tuple(
            (action, cost)
            for action, cost in zip(all_actions, action_costs, strict=True)
            if cost <= budget
        )
    else:
        actions_with_costs = tuple(zip(all_actions, action_costs, strict=True))
    actions = tuple(action for action, _ in actions_with_costs)
    costs = tuple(cost for _, cost in actions_with_costs)
    if not actions:
        raise ValueError("checked-gate budget permits no sampling action")

    evaluations: dict[frozenset[str], StructuralCapacityEvaluation[frozenset[GateId]]]
    evaluations = {}
    scenarios: list[_AttackScenario] = []
    bounded_scenarios = 0
    for attacked in _unit_subsets(attack_unit_ids):
        if not attacked:
            continue
        error_gates = frozenset(
            gate_id for unit_id in attacked for gate_id in attack_units[unit_id]
        )
        evaluation = evaluate_capacity(error_gates)
        evaluations[attacked] = evaluation
        bounded_scenarios += not evaluation.is_exact
        scenarios.append(_AttackScenario(attacked, evaluation.upper_bound))

    selected_strategy, selected_bound = _solve_bound_minimizing_lp(
        actions,
        costs,
        tuple(scenarios),
        budget=budget,
        threshold=threshold,
        budget_mode=budget_mode,
        numerical_tolerance=numerical_tolerance,
        expected_cost=lambda strategy: strategy.expected_checked_gates(partition),
    )

    worst_attack: RobustAdversarialAttack | None = None
    residual_lower = LogCardinality.zero()
    achieved_upper = LogCardinality.zero()
    for attacked, evaluation in evaluations.items():
        detection = _attack_detection_probability(selected_strategy, attacked)
        if detection >= threshold:
            continue
        if evaluation.lower_bound > residual_lower:
            residual_lower = evaluation.lower_bound
        candidate = RobustAdversarialAttack(
            attacked_units=attacked,
            error_gates=frozenset(
                gate_id for unit_id in attacked for gate_id in attack_units[unit_id]
            ),
            detection_probability=detection,
            evaluation=evaluation,
        )
        if evaluation.upper_bound > achieved_upper:
            achieved_upper = evaluation.upper_bound
        if (
            worst_attack is None
            or evaluation.upper_bound > worst_attack.evaluation.upper_bound
        ):
            worst_attack = candidate
    if achieved_upper > selected_bound:
        raise AssertionError("LP certificate disagrees with certified replay")
    return RobustSamplingOptimization(
        strategy=selected_strategy,
        budget_mode=budget_mode,
        checked_gate_budget=budget,
        detection_threshold=threshold,
        worst_attack=worst_attack,
        certified_upper_log2_reachable_bound=achieved_upper,
        residual_exact_lower_bound=residual_lower,
        lp_candidate_log2_reachable_bound=selected_bound,
        candidate_sampling_action_count=len(actions),
        adversarial_action_count=len(scenarios),
        bounded_scenario_count=bounded_scenarios,
    )
