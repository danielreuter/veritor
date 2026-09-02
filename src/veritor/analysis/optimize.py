"""Finite rational policy-grid optimization over certified bound results."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from typing import Any

from circuit_cut_analysis.capacity import LogCardinality
from veritor.analysis.capacity import CapacityOracle
from veritor.analysis.finite import (
    branch_and_bound_finite_bound,
    exhaustive_finite_bound,
)
from veritor.analysis.result import FixedPolicyBoundResult
from veritor.core.identity import Digest, identity_digest
from veritor.core.index import Index
from veritor.core.policy import (
    ProbabilityInput,
    VerificationPolicy,
    exact_fraction,
    rational_manifest,
)


def _nonnegative_rational(
    value: ProbabilityInput,
    *,
    name: str,
) -> Fraction:
    result = exact_fraction(value, name=name)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


class GridOptimizationStatus(StrEnum):
    """Whether the chosen optimum is proved on the declared finite grid."""

    EXACT_ON_GRID = "exact_on_grid"
    HEURISTIC = "heuristic"


@dataclass(frozen=True, slots=True, init=False)
class RationalPolicyGrid:
    """A content-identified Cartesian grid of exact ``q,s`` candidates."""

    q_values: tuple[Fraction, ...]
    s_values: tuple[Fraction, ...]
    eta: Fraction
    identity: Digest

    def __init__(
        self,
        q_values: Iterable[ProbabilityInput],
        s_values: Iterable[ProbabilityInput],
        eta: ProbabilityInput,
    ) -> None:
        checked_q = tuple(
            dict.fromkeys(
                exact_fraction(value, name="q candidate") for value in q_values
            )
        )
        checked_s = tuple(
            dict.fromkeys(
                exact_fraction(value, name="s candidate") for value in s_values
            )
        )
        checked_eta = exact_fraction(eta, name="eta")
        if not checked_q or not checked_s:
            raise ValueError("policy grid axes must be nonempty")
        if any(not 0 <= value <= 1 for value in (*checked_q, *checked_s)):
            raise ValueError("q and s candidates must lie in [0, 1]")
        if not 0 <= checked_eta < 1:
            raise ValueError("eta must lie in [0, 1)")
        object.__setattr__(self, "q_values", checked_q)
        object.__setattr__(self, "s_values", checked_s)
        object.__setattr__(self, "eta", checked_eta)
        object.__setattr__(
            self,
            "identity",
            identity_digest(
                "veritor/analysis/rational-policy-grid/v1",
                {
                    "eta": rational_manifest(checked_eta),
                    "q_values": [rational_manifest(value) for value in checked_q],
                    "s_values": [rational_manifest(value) for value in checked_s],
                },
            ),
        )

    @property
    def policies(self) -> tuple[VerificationPolicy, ...]:
        return tuple(
            VerificationPolicy(q, s, self.eta)
            for q in self.q_values
            for s in self.s_values
        )

    @property
    def candidate_count(self) -> int:
        return len(self.q_values) * len(self.s_values)


PolicyGrid = RationalPolicyGrid


@dataclass(frozen=True, slots=True, init=False)
class AdditiveExpectedCost:
    """Exact additive cost ``c_boundary + q*r + q*s*v``."""

    boundary_cost: Fraction
    replay_cost: Fraction
    verification_cost: Fraction

    def __init__(
        self,
        boundary_cost: ProbabilityInput,
        replay_cost: ProbabilityInput,
        verification_cost: ProbabilityInput,
    ) -> None:
        object.__setattr__(
            self,
            "boundary_cost",
            _nonnegative_rational(boundary_cost, name="boundary_cost"),
        )
        object.__setattr__(
            self,
            "replay_cost",
            _nonnegative_rational(replay_cost, name="replay_cost"),
        )
        object.__setattr__(
            self,
            "verification_cost",
            _nonnegative_rational(
                verification_cost,
                name="verification_cost",
            ),
        )

    def evaluate(self, policy: VerificationPolicy) -> Fraction:
        return (
            self.boundary_cost
            + policy.q * self.replay_cost
            + policy.q * policy.s * self.verification_cost
        )

    @property
    def c_boundary(self) -> Fraction:
        return self.boundary_cost

    expected_cost = evaluate
    __call__ = evaluate


ExpectedCostModel = AdditiveExpectedCost


@dataclass(frozen=True, slots=True)
class PolicyGridEvaluation:
    """One grid policy, its exact cost, and its certified bound."""

    policy: VerificationPolicy
    expected_cost: Fraction
    bound: FixedPolicyBoundResult

    def __post_init__(self) -> None:
        if self.bound.policy_identity != self.policy.digest:
            raise ValueError("grid evaluation bound identifies another policy")


@dataclass(frozen=True, slots=True)
class PolicyGridOptimizationResult:
    """Best certified choice(s) on one declared finite policy grid."""

    grid: RationalPolicyGrid
    cost_model: AdditiveExpectedCost
    evaluations: tuple[PolicyGridEvaluation, ...]
    chosen: tuple[PolicyGridEvaluation, ...]
    status: GridOptimizationStatus
    objective: str
    capacity_limit: LogCardinality | None = None
    maximum_expected_cost: Fraction | None = None
    reason: str = ""

    @property
    def exact_on_grid(self) -> bool:
        return self.status is GridOptimizationStatus.EXACT_ON_GRID

    @property
    def chosen_policy(self) -> VerificationPolicy | None:
        return self.chosen[0].policy if len(self.chosen) == 1 else None

    @property
    def chosen_bound(self) -> FixedPolicyBoundResult | None:
        return self.chosen[0].bound if len(self.chosen) == 1 else None


def optimize_policy_grid(
    grid: RationalPolicyGrid,
    cost_model: AdditiveExpectedCost,
    evaluate_bound: Callable[[VerificationPolicy], FixedPolicyBoundResult],
    *,
    capacity_limit: LogCardinality | None = None,
    maximum_expected_cost: ProbabilityInput | None = None,
) -> PolicyGridOptimizationResult:
    """Optimize a finite grid, never upgrading bracketed bounds to exactness.

    With ``capacity_limit``, the objective is minimum expected cost among
    policies whose certified upper bound meets the limit.  Otherwise the
    objective is minimum certified upper bound, then minimum expected cost.
    """

    if not isinstance(grid, RationalPolicyGrid):
        raise TypeError("grid must be a RationalPolicyGrid")
    if not isinstance(cost_model, AdditiveExpectedCost):
        raise TypeError("cost_model must be AdditiveExpectedCost")
    if capacity_limit is not None and not isinstance(
        capacity_limit,
        LogCardinality,
    ):
        raise TypeError("capacity_limit must be a LogCardinality")
    max_cost = (
        None
        if maximum_expected_cost is None
        else _nonnegative_rational(
            maximum_expected_cost,
            name="maximum_expected_cost",
        )
    )

    evaluations: list[PolicyGridEvaluation] = []
    for policy in grid.policies:
        bound = evaluate_bound(policy)
        if not isinstance(bound, FixedPolicyBoundResult):
            raise TypeError("evaluate_bound must return FixedPolicyBoundResult")
        evaluations.append(
            PolicyGridEvaluation(
                policy=policy,
                expected_cost=cost_model.evaluate(policy),
                bound=bound,
            )
        )

    eligible = [
        item
        for item in evaluations
        if max_cost is None or item.expected_cost <= max_cost
    ]
    objective: str
    if capacity_limit is not None:
        objective = "minimum_expected_cost_subject_to_certified_upper_bound"
        eligible = [
            item for item in eligible if item.bound.upper_bound <= capacity_limit
        ]
        cost_primary = (
            min(
                (
                    item.expected_cost,
                    item.bound.upper_bound,
                )
                for item in eligible
            )
            if eligible
            else None
        )
        chosen = tuple(
            item
            for item in eligible
            if (item.expected_cost, item.bound.upper_bound) == cost_primary
        )
    else:
        objective = "minimum_certified_upper_bound_then_expected_cost"
        bound_primary = (
            min(
                (
                    item.bound.upper_bound,
                    item.expected_cost,
                )
                for item in eligible
            )
            if eligible
            else None
        )
        chosen = tuple(
            item
            for item in eligible
            if (item.bound.upper_bound, item.expected_cost) == bound_primary
        )

    all_exact = all(item.bound.is_exact for item in evaluations)
    status = (
        GridOptimizationStatus.EXACT_ON_GRID
        if all_exact
        else GridOptimizationStatus.HEURISTIC
    )
    if not chosen:
        reason = "no grid policy satisfies the declared certified constraints"
    elif all_exact:
        reason = "every policy objective was solved exactly on the finite grid"
    else:
        reason = (
            "selection uses certified upper bounds, but at least one grid "
            "objective remains bracketed or relaxed"
        )
    return PolicyGridOptimizationResult(
        grid=grid,
        cost_model=cost_model,
        evaluations=tuple(evaluations),
        chosen=chosen,
        status=status,
        objective=objective,
        capacity_limit=capacity_limit,
        maximum_expected_cost=max_cost,
        reason=reason,
    )


def optimize_finite_policy_grid(
    index: Index,
    capacity_oracle: CapacityOracle[Any],
    grid: RationalPolicyGrid,
    cost_model: AdditiveExpectedCost,
    *,
    solver: str = "branch-and-bound",
    solver_options: Mapping[str, Any] | None = None,
    capacity_limit: LogCardinality | None = None,
    maximum_expected_cost: ProbabilityInput | None = None,
) -> PolicyGridOptimizationResult:
    """Evaluate every finite-grid policy with a certified finite backend."""

    options = dict(solver_options or {})
    if solver == "branch-and-bound":

        def evaluate(policy: VerificationPolicy) -> FixedPolicyBoundResult:
            return branch_and_bound_finite_bound(
                index,
                policy,
                capacity_oracle,
                **options,
            )

    elif solver == "exhaustive":

        def evaluate(policy: VerificationPolicy) -> FixedPolicyBoundResult:
            return exhaustive_finite_bound(
                index,
                policy,
                capacity_oracle,
                **options,
            )

    else:
        raise ValueError("solver must be 'branch-and-bound' or 'exhaustive'")
    return optimize_policy_grid(
        grid,
        cost_model,
        evaluate,
        capacity_limit=capacity_limit,
        maximum_expected_cost=maximum_expected_cost,
    )


optimize_grid = optimize_policy_grid


__all__ = [
    "AdditiveExpectedCost",
    "ExpectedCostModel",
    "GridOptimizationStatus",
    "PolicyGrid",
    "PolicyGridEvaluation",
    "PolicyGridOptimizationResult",
    "RationalPolicyGrid",
    "optimize_finite_policy_grid",
    "optimize_grid",
    "optimize_policy_grid",
]
