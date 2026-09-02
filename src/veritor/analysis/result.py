"""Typed identities, witnesses, and guarantee levels for bound backends."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction

from circuit_cut_analysis.capacity import LogCardinality
from veritor.analysis.capacity import CapacityEvidence
from veritor.core.identity import Digest, identity_digest, validate_digest
from veritor.core.index import Index
from veritor.core.policy import VerificationPolicy


class BoundClaimStrength(StrEnum):
    """The strongest honest interpretation of a bound result."""

    EXACT = "exact"
    CERTIFIED_BRACKET = "certified_bracket"
    CERTIFIED_UPPER = "certified_upper"
    CONDITIONAL = "conditional"
    HEURISTIC = "heuristic"


class TerminationStatus(StrEnum):
    """How a backend stopped."""

    COMPLETE = "complete"
    RESOURCE_LIMIT = "resource_limit"
    NUMERICALLY_CONSERVATIVE = "numerically_conservative"


class SurvivalComparison(StrEnum):
    """Evidence for an attack's strict survival-threshold comparison."""

    STRICTLY_ABOVE = "strictly_above"
    AT_OR_BELOW = "at_or_below"
    CONSERVATIVELY_INCLUDED = "conservatively_included"


@dataclass(frozen=True, slots=True)
class BoundIdentities:
    """Content identities of the exact policy and analyzed artifact tuple."""

    policy_identity: Digest
    tuple_identity: Digest
    index_identity: Digest | None = None
    capacity_schema_identity: Digest | None = None
    replay_layout_identity: Digest | None = None

    def __post_init__(self) -> None:
        for name in (
            "policy_identity",
            "tuple_identity",
            "index_identity",
            "capacity_schema_identity",
            "replay_layout_identity",
        ):
            value = getattr(self, name)
            if value is not None:
                validate_digest(value, name)


@dataclass(frozen=True, slots=True)
class BoundWitness:
    """One attack and the evidence attached to it."""

    error_units: tuple[int, ...]
    attacked_positions: tuple[int, ...]
    attack_support: object
    survival_probability: Fraction | None
    survival_comparison: SurvivalComparison
    capacity_evidence: CapacityEvidence[object]
    attacked_counts: tuple[int, ...] | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if tuple(sorted(set(self.error_units))) != self.error_units:
            raise ValueError("witness error units must be sorted and unique")
        if tuple(sorted(set(self.attacked_positions))) != self.attacked_positions:
            raise ValueError("witness positions must be sorted and unique")
        if self.survival_probability is not None and not (
            0 <= self.survival_probability <= 1
        ):
            raise ValueError("witness survival probability must lie in [0, 1]")
        if self.attacked_counts is not None and any(
            type(count) is not int or count < 0 for count in self.attacked_counts
        ):
            raise ValueError("witness attacked counts must be nonnegative integers")

    @property
    def capacity_lower_bound(self) -> LogCardinality:
        return self.capacity_evidence.lower_bound

    @property
    def capacity_upper_bound(self) -> LogCardinality:
        return self.capacity_evidence.upper_bound

    @property
    def survival(self) -> Fraction | None:
        return self.survival_probability


@dataclass(frozen=True, slots=True)
class FixedPolicyBoundResult:
    """A certified interval for the fixed-policy adversarial optimum."""

    lower_bound: LogCardinality
    upper_bound: LogCardinality
    claim_strength: BoundClaimStrength
    termination_status: TerminationStatus
    method: str
    witness: BoundWitness
    identities: BoundIdentities
    assumptions: tuple[str, ...] = ()
    relaxation_chain: tuple[str, ...] = ()
    upper_witness: BoundWitness | None = None
    state_count: int = 0
    capacity_query_count: int = 0
    feasible_state_count: int = 0
    pruned_infeasible_count: int = 0
    pruned_all_feasible_count: int = 0
    pruned_capacity_dominated_count: int = 0
    numerically_conservative: bool = False

    def __post_init__(self) -> None:
        zero = LogCardinality.zero()
        if self.lower_bound < zero or self.upper_bound < zero:
            raise ValueError("bound result capacities cannot be negative")
        if self.lower_bound > self.upper_bound:
            raise ValueError("bound result lower bound exceeds upper bound")
        if type(self.method) is not str or not self.method:
            raise ValueError("bound result method must be nonempty")
        for name in (
            "state_count",
            "capacity_query_count",
            "feasible_state_count",
            "pruned_infeasible_count",
            "pruned_all_feasible_count",
            "pruned_capacity_dominated_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if any(type(item) is not str or not item for item in self.assumptions):
            raise ValueError("bound assumptions must be nonempty strings")
        if any(type(item) is not str or not item for item in self.relaxation_chain):
            raise ValueError("relaxation steps must be nonempty strings")
        if self.witness.capacity_lower_bound != self.lower_bound:
            raise ValueError("primary witness must certify the reported lower bound")
        if (
            self.upper_witness is not None
            and self.upper_witness.capacity_upper_bound != self.upper_bound
        ):
            raise ValueError("upper witness must certify the reported upper bound")
        if self.claim_strength is BoundClaimStrength.EXACT:
            if self.lower_bound != self.upper_bound:
                raise ValueError("an exact claim requires equal certified bounds")
            if self.termination_status is not TerminationStatus.COMPLETE:
                raise ValueError("an exact claim requires complete termination")
            if self.numerically_conservative:
                raise ValueError("numerically conservative results are not exact")
            if self.assumptions or self.relaxation_chain:
                raise ValueError(
                    "an unconditional exact claim cannot carry assumptions "
                    "or relaxations"
                )

    @property
    def exact_capacity(self) -> LogCardinality | None:
        if self.claim_strength is BoundClaimStrength.EXACT:
            return self.lower_bound
        return None

    @property
    def is_exact(self) -> bool:
        return self.exact_capacity is not None

    @property
    def certified_lower_bound(self) -> LogCardinality:
        return self.lower_bound

    @property
    def certified_upper_bound(self) -> LogCardinality:
        return self.upper_bound

    @property
    def policy_identity(self) -> Digest:
        return self.identities.policy_identity

    @property
    def tuple_identity(self) -> Digest:
        return self.identities.tuple_identity

    @property
    def query_count(self) -> int:
        return self.capacity_query_count

    @property
    def lower_bound_bits(self) -> int | float:
        return self.lower_bound.width_bits

    @property
    def upper_bound_bits(self) -> int | float:
        return self.upper_bound.width_bits

    @property
    def termination(self) -> TerminationStatus:
        return self.termination_status


def finite_bound_identities(index: Index, policy: VerificationPolicy) -> BoundIdentities:
    """Bind a finite result to the literal index ``I`` it was computed over."""

    tuple_identity = identity_digest(
        "veritor/analysis/finite-bound-tuple/v2", {"index_digest": index.digest}
    )
    return BoundIdentities(
        policy_identity=policy.digest,
        tuple_identity=tuple_identity,
        index_identity=index.digest,
    )
def counted_bound_identities(
    policy: VerificationPolicy,
    capacity_schema_identity: Digest,
    replay_layout_identity: Digest | None,
    *,
    relaxation: str | None = None,
) -> BoundIdentities:
    """Bind a counted result to its schema, optional layout, and relaxation."""

    manifest = {
        "capacity_schema_identity": capacity_schema_identity,
        "replay_layout_identity": replay_layout_identity,
        "relaxation": relaxation,
    }
    tuple_identity = identity_digest(
        "veritor/analysis/counted-bound-tuple/v1",
        manifest,
    )
    return BoundIdentities(
        policy_identity=policy.digest,
        tuple_identity=tuple_identity,
        capacity_schema_identity=capacity_schema_identity,
        replay_layout_identity=replay_layout_identity,
    )


def derive_claim_strength(
    lower_bound: LogCardinality,
    upper_bound: LogCardinality,
    termination_status: TerminationStatus,
    *,
    assumptions: tuple[str, ...] = (),
    certified_upper_relaxation: bool = False,
    heuristic: bool = False,
) -> BoundClaimStrength:
    """Derive a claim label without upgrading heuristics or relaxations."""

    if heuristic:
        return BoundClaimStrength.HEURISTIC
    if certified_upper_relaxation:
        return BoundClaimStrength.CERTIFIED_UPPER
    if assumptions:
        return BoundClaimStrength.CONDITIONAL
    if termination_status is TerminationStatus.COMPLETE and lower_bound == upper_bound:
        return BoundClaimStrength.EXACT
    return BoundClaimStrength.CERTIFIED_BRACKET


__all__ = [
    "BoundClaimStrength",
    "BoundIdentities",
    "BoundWitness",
    "FixedPolicyBoundResult",
    "SurvivalComparison",
    "TerminationStatus",
    "counted_bound_identities",
    "derive_claim_strength",
    "finite_bound_identities",
]
