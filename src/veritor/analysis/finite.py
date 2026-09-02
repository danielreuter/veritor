"""Reference exhaustive and exact finite branch-and-bound backends."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, cast

from circuit_cut_analysis.capacity import LogCardinality
from veritor.analysis.capacity import (
    CapacityEvidence,
    CapacityOracle,
    coerce_capacity_evidence,
    zero_capacity_evidence,
)
from veritor.analysis.probability import (
    AttackSetKind,
    survival_from_replay_error_counts,
)
from veritor.analysis.result import (
    BoundWitness,
    FixedPolicyBoundResult,
    SurvivalComparison,
    TerminationStatus,
    derive_claim_strength,
    finite_bound_identities,
)
from veritor.core.errors import ResourceLimit
from veritor.core.index import Index
from veritor.core.policy import VerificationPolicy


def _ordered_unique(*groups: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for group in groups:
        for item in group:
            if item not in seen:
                seen.add(item)
                result.append(item)
    return tuple(result)


def _support_kind(value: AttackSetKind | str) -> AttackSetKind:
    if value in ("units", "verification_unit_ids"):
        return AttackSetKind.VERIFICATION_UNITS
    try:
        return AttackSetKind(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"unknown capacity support kind {value!r}") from error


@dataclass(frozen=True, slots=True)
class _FiniteContext:
    """The finite game over an index: every verification unit is materialized."""

    index: Index
    policy: VerificationPolicy
    oracle: CapacityOracle[Any]
    support_kind: AttackSetKind
    owners: tuple[int, ...]
    positions_by_unit: tuple[tuple[int, ...], ...]

    @classmethod
    def build(
        cls,
        index: Index,
        policy: VerificationPolicy,
        oracle: CapacityOracle[Any],
        support_kind: AttackSetKind | str,
    ) -> _FiniteContext:
        if not isinstance(index, Index):
            raise TypeError("index must be an Index")
        if not isinstance(policy, VerificationPolicy):
            raise TypeError("policy must be a VerificationPolicy")
        units = [
            index.verification_unit(unit) for unit in range(index.verification_unit_count)
        ]
        owners = []
        for unit in units:
            owner = unit.replay_unit
            assert owner is not None
            owners.append(owner)
        return cls(
            index=index,
            policy=policy,
            oracle=oracle,
            support_kind=_support_kind(support_kind),
            owners=tuple(owners),
            positions_by_unit=tuple(tuple(unit.interval) for unit in units),
        )

    def replay_counts(self, error_units: tuple[int, ...]) -> tuple[int, ...]:
        counts = [0] * self.index.replay_units.count
        for unit_index in error_units:
            counts[self.owners[unit_index]] += 1
        return tuple(counts)

    def survival(self, error_units: tuple[int, ...]) -> Fraction:
        return survival_from_replay_error_counts(
            self.policy,
            self.replay_counts(error_units),
        )

    def attacked_positions(self, error_units: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(
            sorted(
                {
                    position
                    for unit_index in error_units
                    for position in self.positions_by_unit[unit_index]
                }
            )
        )

    def support(self, error_units: tuple[int, ...]) -> frozenset[int]:
        if self.support_kind is AttackSetKind.VERIFICATION_UNITS:
            return frozenset(error_units)
        return frozenset(self.attacked_positions(error_units))

    def evaluate(
        self,
        error_units: tuple[int, ...],
    ) -> tuple[CapacityEvidence[Any], bool]:
        support = self.support(error_units)
        if not support:
            return zero_capacity_evidence(support), False
        return coerce_capacity_evidence(
            self.oracle.evaluate(support),
            requested_support=support,
        ), True

    def witness(
        self,
        error_units: tuple[int, ...],
        survival: Fraction | None,
        evidence: CapacityEvidence[Any],
        *,
        comparison: SurvivalComparison = SurvivalComparison.STRICTLY_ABOVE,
        note: str = "",
    ) -> BoundWitness:
        return BoundWitness(
            error_units=error_units,
            attacked_positions=self.attacked_positions(error_units),
            attack_support=self.support(error_units),
            survival_probability=survival,
            survival_comparison=comparison,
            capacity_evidence=cast(CapacityEvidence[object], evidence),
            note=note,
        )


def exhaustive_finite_bound(
    index: Index,
    policy: VerificationPolicy,
    capacity_oracle: CapacityOracle[Any],
    *,
    support_kind: AttackSetKind | str = AttackSetKind.VERIFICATION_UNITS,
    max_verification_units: int = 24,
    assumptions: tuple[str, ...] = (),
    relaxation_chain: tuple[str, ...] = (),
) -> FixedPolicyBoundResult:
    """Enumerate every finite error set and return its certified optimum bracket."""

    context = _FiniteContext.build(index, policy, capacity_oracle, support_kind)
    unit_count = index.verification_unit_count
    if type(max_verification_units) is not int or max_verification_units < 0:
        raise ValueError("max_verification_units must be nonnegative")
    if unit_count > max_verification_units:
        raise ResourceLimit(
            "verification_units",
            limit=max_verification_units,
            observed=unit_count,
        )

    zero = LogCardinality.zero()
    empty_evidence, _ = context.evaluate(())
    empty_witness = context.witness((), Fraction(1), empty_evidence)
    best_lower = zero
    best_upper = zero
    lower_witness = empty_witness
    upper_witness = empty_witness
    query_count = 0
    feasible_count = 0
    observed_assumptions = assumptions
    pattern_count = 1 << unit_count

    for mask in range(pattern_count):
        error_units = tuple(
            index for index in range(unit_count) if mask & (1 << index)
        )
        survival = context.survival(error_units)
        if survival <= policy.eta:
            continue
        feasible_count += 1
        evidence, queried = context.evaluate(error_units)
        query_count += int(queried)
        observed_assumptions = _ordered_unique(
            observed_assumptions,
            evidence.assumptions,
        )
        witness = context.witness(error_units, survival, evidence)
        if evidence.lower_bound > best_lower:
            best_lower = evidence.lower_bound
            lower_witness = witness
        if evidence.upper_bound > best_upper:
            best_upper = evidence.upper_bound
            upper_witness = witness

    termination = TerminationStatus.COMPLETE
    claim = derive_claim_strength(
        best_lower,
        best_upper,
        termination,
        assumptions=observed_assumptions,
        certified_upper_relaxation=bool(relaxation_chain),
    )
    return FixedPolicyBoundResult(
        lower_bound=best_lower,
        upper_bound=best_upper,
        claim_strength=claim,
        termination_status=termination,
        method="reference-exhaustive-finite",
        witness=lower_witness,
        upper_witness=upper_witness,
        identities=finite_bound_identities(index, policy),
        assumptions=observed_assumptions,
        relaxation_chain=relaxation_chain,
        state_count=pattern_count,
        capacity_query_count=query_count,
        feasible_state_count=feasible_count,
        pruned_infeasible_count=pattern_count - feasible_count,
    )


@dataclass(frozen=True, slots=True)
class _SearchNode:
    depth: int
    included: tuple[int, ...]
    replay_counts: tuple[int, ...]
    survival: Fraction
    envelope_upper: LogCardinality
    envelope_evidence: CapacityEvidence[Any]
    envelope_units: tuple[int, ...]
    exact_for_subtree_full_support: bool


def branch_and_bound_finite_bound(
    index: Index,
    policy: VerificationPolicy,
    capacity_oracle: CapacityOracle[Any],
    *,
    support_kind: AttackSetKind | str = AttackSetKind.VERIFICATION_UNITS,
    max_states: int = 1_000_000,
    max_capacity_queries: int = 1_000_000,
    assumptions: tuple[str, ...] = (),
) -> FixedPolicyBoundResult:
    """Solve the finite game with exact survival and monotone capacity pruning.

    Resource exhaustion returns a certified bracket.  Every unresolved subtree
    retains an upper envelope queried at a superset of all attacks in that
    subtree, so the returned upper bound continues to contain the optimum.
    """

    if type(max_states) is not int or max_states < 0:
        raise ValueError("max_states must be a nonnegative integer")
    if type(max_capacity_queries) is not int or max_capacity_queries < 0:
        raise ValueError("max_capacity_queries must be a nonnegative integer")
    context = _FiniteContext.build(index, policy, capacity_oracle, support_kind)
    unit_count = index.verification_unit_count
    if unit_count and max_capacity_queries == 0:
        raise ValueError(
            "at least one capacity query is required for a certified root envelope"
        )

    zero = LogCardinality.zero()
    empty_evidence, _ = context.evaluate(())
    empty_witness = context.witness((), Fraction(1), empty_evidence)
    best_lower = zero
    lower_witness = empty_witness
    completed_upper = zero
    completed_upper_witness = empty_witness
    observed_assumptions = assumptions
    query_count = 0

    all_units = tuple(range(unit_count))
    if all_units:
        root_evidence, queried = context.evaluate(all_units)
        query_count += int(queried)
        observed_assumptions = _ordered_unique(
            observed_assumptions,
            root_evidence.assumptions,
        )
    else:
        root_evidence = empty_evidence
    root_counts = (0,) * index.replay_units.count
    stack = [
        _SearchNode(
            depth=0,
            included=(),
            replay_counts=root_counts,
            survival=Fraction(1),
            envelope_upper=root_evidence.upper_bound,
            envelope_evidence=root_evidence,
            envelope_units=all_units,
            exact_for_subtree_full_support=True,
        )
    ]

    state_count = 0
    feasible_state_count = 0
    pruned_infeasible = 0
    pruned_all_feasible = 0
    pruned_capacity = 0
    unresolved: list[_SearchNode] = []
    resource_limited = False

    def update_candidate(
        error_units: tuple[int, ...],
        survival: Fraction,
        evidence: CapacityEvidence[Any],
    ) -> None:
        nonlocal best_lower
        nonlocal lower_witness
        nonlocal completed_upper
        nonlocal completed_upper_witness
        witness = context.witness(error_units, survival, evidence)
        if evidence.lower_bound > best_lower:
            best_lower = evidence.lower_bound
            lower_witness = witness
        if evidence.upper_bound > completed_upper:
            completed_upper = evidence.upper_bound
            completed_upper_witness = witness

    while stack:
        if state_count >= max_states:
            unresolved = list(stack)
            resource_limited = True
            break
        node = stack.pop()
        state_count += 1

        if node.survival <= policy.eta:
            pruned_infeasible += 1
            continue
        if node.envelope_upper <= best_lower:
            pruned_capacity += 1
            continue

        subtree_full = (*node.included, *range(node.depth, unit_count))
        full_counts = list(node.replay_counts)
        for unit_index in range(node.depth, unit_count):
            full_counts[context.owners[unit_index]] += 1
        full_survival = survival_from_replay_error_counts(
            policy,
            full_counts,
        )

        evidence = node.envelope_evidence
        exact_for_full = node.exact_for_subtree_full_support
        if not exact_for_full:
            if query_count >= max_capacity_queries:
                unresolved = [node, *stack]
                resource_limited = True
                break
            evidence, queried = context.evaluate(subtree_full)
            query_count += int(queried)
            observed_assumptions = _ordered_unique(
                observed_assumptions,
                evidence.assumptions,
            )
            node = _SearchNode(
                depth=node.depth,
                included=node.included,
                replay_counts=node.replay_counts,
                survival=node.survival,
                envelope_upper=evidence.upper_bound,
                envelope_evidence=evidence,
                envelope_units=subtree_full,
                exact_for_subtree_full_support=True,
            )

        if evidence.upper_bound <= best_lower:
            pruned_capacity += 1
            continue

        if full_survival > policy.eta:
            feasible_state_count += 1
            pruned_all_feasible += 1
            update_candidate(subtree_full, full_survival, evidence)
            continue

        if node.depth == unit_count:
            feasible_state_count += 1
            update_candidate(node.included, node.survival, evidence)
            continue

        decision_unit = node.depth
        next_depth = decision_unit + 1
        excluded = _SearchNode(
            depth=next_depth,
            included=node.included,
            replay_counts=node.replay_counts,
            survival=node.survival,
            envelope_upper=evidence.upper_bound,
            envelope_evidence=evidence,
            envelope_units=subtree_full,
            exact_for_subtree_full_support=False,
        )
        included_counts = list(node.replay_counts)
        included_counts[context.owners[decision_unit]] += 1
        included_survival = survival_from_replay_error_counts(
            policy,
            included_counts,
        )
        included = _SearchNode(
            depth=next_depth,
            included=(*node.included, decision_unit),
            replay_counts=tuple(included_counts),
            survival=included_survival,
            envelope_upper=evidence.upper_bound,
            envelope_evidence=evidence,
            envelope_units=subtree_full,
            exact_for_subtree_full_support=True,
        )
        stack.append(excluded)
        stack.append(included)

    if resource_limited:
        upper_bound = completed_upper
        upper_witness = completed_upper_witness
        for node in unresolved:
            if node.envelope_upper > upper_bound:
                upper_bound = node.envelope_upper
                envelope_survival = context.survival(node.envelope_units)
                comparison = (
                    SurvivalComparison.STRICTLY_ABOVE
                    if envelope_survival > policy.eta
                    else SurvivalComparison.AT_OR_BELOW
                )
                upper_witness = context.witness(
                    node.envelope_units,
                    envelope_survival,
                    node.envelope_evidence,
                    comparison=comparison,
                    note=(
                        "capacity envelope for an unresolved subtree; "
                        "the support itself need not be admissible"
                    ),
                )
        if best_lower > upper_bound:
            raise AssertionError("search lower bound escaped its root envelope")
        termination = TerminationStatus.RESOURCE_LIMIT
    else:
        upper_bound = max(best_lower, completed_upper)
        upper_witness = (
            lower_witness
            if best_lower > completed_upper
            else completed_upper_witness
        )
        termination = TerminationStatus.COMPLETE

    claim = derive_claim_strength(
        best_lower,
        upper_bound,
        termination,
        assumptions=observed_assumptions,
    )
    return FixedPolicyBoundResult(
        lower_bound=best_lower,
        upper_bound=upper_bound,
        claim_strength=claim,
        termination_status=termination,
        method="exact-finite-branch-and-bound",
        witness=lower_witness,
        upper_witness=upper_witness,
        identities=finite_bound_identities(index, policy),
        assumptions=observed_assumptions,
        state_count=state_count,
        capacity_query_count=query_count,
        feasible_state_count=feasible_state_count,
        pruned_infeasible_count=pruned_infeasible,
        pruned_all_feasible_count=pruned_all_feasible,
        pruned_capacity_dominated_count=pruned_capacity,
    )


# Stable concise spellings.
exhaustive_bound = exhaustive_finite_bound
branch_and_bound = branch_and_bound_finite_bound
fixed_policy_bound = branch_and_bound_finite_bound
exhaustive_fixed_policy_bound = exhaustive_finite_bound
branch_and_bound_fixed_policy_bound = branch_and_bound_finite_bound


__all__ = [
    "branch_and_bound",
    "branch_and_bound_finite_bound",
    "branch_and_bound_fixed_policy_bound",
    "exhaustive_bound",
    "exhaustive_finite_bound",
    "exhaustive_fixed_policy_bound",
    "fixed_policy_bound",
]
