"""``Cost(C, I, theta)``: the expected cost of one run of the protocol.

Committing the boundary ``∂ = In ∪ ⋃_r Out(R_r)`` costs ``h`` per position; a
replay unit, selected with probability ``q``, costs its replay and ``h`` per
interior position it commits (its gates but ``Out`` and its pinned source
gates); a verification unit, selected with probability ``q s``, costs its
proof and a fixed ``c_0``::

    Cost = h |∂| + q sum_r (Cost_replay(R_r) + h |Int(r)|) + q s sum_v (Cost_proof(V_v) + c_0)

The weights are committed once per epoch under κ_W, ``h |W|``, reported
separately from the per-run total.  Everything is a count per kind weighted
by copies, so the fold is ``O(#kinds)`` and exact.  The verifier's own
expected work is priced by :func:`veritor.protocol.expected_work`.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from veritor.core.compiled import Compiled
from veritor.core.description import REPLAY, VERIFICATION
from veritor.core.policy import ProbabilityInput, VerificationPolicy, exact_fraction


@dataclass(frozen=True, slots=True, init=False)
class CostParameters:
    """``h``, the cost of committing one position, and ``c_0``, the fixed cost of one proof."""

    hash_cost: Fraction
    proof_overhead: Fraction

    def __init__(self, hash_cost: ProbabilityInput = 1, proof_overhead: ProbabilityInput = 0) -> None:
        for name, value in (("hash_cost", hash_cost), ("proof_overhead", proof_overhead)):
            checked = exact_fraction(value, name=name)
            if checked < 0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, checked)


@dataclass(frozen=True, slots=True)
class ExpectedCost:
    """The three per-run terms of the expected cost, exactly, and the per-epoch weight commitment.

    ``weights`` is ``h |W|``: paid once per epoch when κ_W is built, not per
    run, so it is not part of ``total``.
    """

    boundary: Fraction
    replay: Fraction
    proof: Fraction
    weights: Fraction = Fraction(0)

    @property
    def total(self) -> Fraction:
        return self.boundary + self.replay + self.proof


def cost(
    compiled: Compiled,
    policy: VerificationPolicy,
    parameters: CostParameters | None = None,
) -> ExpectedCost:
    """Fold the expected cost of ``policy`` over the kinds of ``compiled.index``."""

    if not isinstance(compiled, Compiled):
        raise TypeError("cost needs a Compiled artifact")
    if not isinstance(policy, VerificationPolicy):
        raise TypeError("policy must be a VerificationPolicy")
    parameters = CostParameters() if parameters is None else parameters
    h, c0 = parameters.hash_cost, parameters.proof_overhead
    index = compiled.index
    boundary = Fraction(index.input_count)  # the input gates, then every unit's Out
    replay = Fraction(0)
    proof = Fraction(0)
    for kind in index.kinds():
        if kind.role == REPLAY:
            boundary += kind.copies * kind.out_count
            interior = kind.size - kind.out_count - kind.source_inputs - kind.source_weights
            replay += kind.copies * (kind.replay_cost + h * interior)
        elif kind.role == VERIFICATION:
            proof += kind.copies * (kind.proof_cost + c0)
    return ExpectedCost(
        h * boundary,
        policy.q * replay,
        policy.q * policy.s * proof,
        h * index.weight_count,
    )


__all__ = ["CostParameters", "ExpectedCost", "cost"]
