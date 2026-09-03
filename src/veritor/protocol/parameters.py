"""What the verifier fixes on its own: ``eta``, ``U_max``, ``A``, ``W_max`` and ``f_max``.

The client proposes the sampling rates ``theta = (q, s)`` and the advice
``a``.  The acceptance threshold ``eta`` defines the security statement, so
it is never taken from the client; ``U_max`` bounds the capacity ``U =
Bound(C, I, theta)`` the verifier will underwrite and ``A = max_advice_bits``
the advice it will accept, so every admitted request has capacity at most
``U_max + A``; ``W_max`` bounds the verifier's own expected work, priced by
:func:`expected_work` from counts alone before any commitment is accepted.
``f_max = max_faults`` is how many VUs the prover may declare incorrect after
the q-challenge (fault declarations, mechanism M6); it is bound into the
header and priced into ``U`` by ``bound(..., max_faults=f_max)``
(:mod:`veritor.analysis.faults`).
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from veritor.core import (
    Compiled,
    KindSummary,
    KindTable,
    ProbabilityInput,
    VerificationPolicy,
    as_kind_table,
    exact_fraction,
)
from veritor.core.description import VERIFICATION

from .merkle import merkle_depth
from .messages import ProtocolError

DEFAULT_MAX_WORK = 1 << 32
"""Default ``W_max``: operations (leaves, path hashes, gate checks) per run."""


@dataclass(frozen=True, slots=True, init=False)
class VerifierParameters:
    """The verifier's side of the acceptance statement.

    ``eta`` is the strict acceptance-probability threshold of the bound;
    ``max_capacity`` is ``U_max``: a run is admitted only when ``Bound(C, I,
    theta)`` at this ``eta`` is at most ``U_max`` bits.  It has no default:
    a verifier states its capacity bound, and ``None`` -- which waives the
    check and admits ``theta = (0, 0)`` -- has to be written out.
    ``max_advice_bits`` is ``A``, the longest advice admitted, in bits;
    ``max_work`` is ``W_max``; ``max_faults`` is ``f_max``, the most VUs a
    prover may declare incorrect in one run (default 0: no declarations, the
    protocol exactly as it was before M6).

    The paper charges a request ``Bound(C, I, theta) + |a|`` bits: everything
    the client did beyond the advice is a deterministic function of ``(G, x,
    a)``.  With ``U_max`` and ``A`` both enforced at admission, every accepted
    request has capacity at most ``U_max + A``; ``Bound`` includes the
    ``f_max`` fault allowance, so admitting declarations widens ``U``, never
    the acceptance probability of an undeclared incorrect VU.
    """

    eta: Fraction
    max_capacity: int | None
    max_advice_bits: int
    max_work: int
    max_faults: int

    def __init__(
        self,
        eta: ProbabilityInput = 0,
        *,
        max_capacity: int | None,
        max_advice_bits: int = 0,
        max_work: int = DEFAULT_MAX_WORK,
        max_faults: int = 0,
    ) -> None:
        checked = exact_fraction(eta, name="eta")
        if not 0 <= checked < 1:
            raise ProtocolError("eta must lie in [0, 1)")
        if max_capacity is not None and (
            type(max_capacity) is not int or max_capacity < 0
        ):
            raise ProtocolError("max_capacity must be None or a nonnegative integer")
        if type(max_advice_bits) is not int or max_advice_bits < 0:
            raise ProtocolError("max_advice_bits must be a nonnegative integer")
        if type(max_work) is not int or max_work < 0:
            raise ProtocolError("max_work must be a nonnegative integer")
        if type(max_faults) is not int or max_faults < 0:
            raise ProtocolError("max_faults must be a nonnegative integer")
        object.__setattr__(self, "eta", checked)
        object.__setattr__(self, "max_capacity", max_capacity)
        object.__setattr__(self, "max_advice_bits", max_advice_bits)
        object.__setattr__(self, "max_work", max_work)
        object.__setattr__(self, "max_faults", max_faults)

    def policy(self, proposal: VerificationPolicy) -> VerificationPolicy:
        """The run's policy: the client's ``theta = (q, s)``, validated.

        The client chooses how much is sampled, never what the verifier's
        acceptance means: ``eta`` is this object's and is bound into the
        header alongside the proposal.
        """

        if not isinstance(proposal, VerificationPolicy):
            raise ProtocolError("the proposal must be a VerificationPolicy")
        return proposal


def positions_per_unit(kind: KindSummary) -> int:
    """The most positions the verifier handles for one copy of ``kind``.

    A sampled verification unit is opened at its declared outputs and at the
    outside addresses it reads, at most its declared inputs; a replay unit's
    interior commitment covers the outputs of the verification units inside
    it that are not its own (:attr:`KindSummary.interior_count`).
    """

    if kind.role == VERIFICATION:
        return kind.out_count + kind.input_count
    return kind.interior_count


def expected_work(
    target: Compiled | KindTable, policy: VerificationPolicy, io_count: int
) -> Fraction:
    """The verifier's expected work for one run, from counts alone.

    One operation is counted per opened leaf, per Merkle path hash, per gate
    check and per commitment received::

        W = (|IO| + q s A) (1 + d) + q s S + 1 + q R

    with ``R`` replay units, ``A = sum_k m_k (out_k + in_k)`` and ``S = sum_k
    m_k size_k`` over the verification kinds (``m_k`` copies of ``size_k``
    gates with ``out_k`` declared outputs and ``in_k`` declared inputs, which
    bound the outside addresses a copy reads; a copy is sampled with
    probability ``q s`` and its gates are recomputed from the opened inputs),
    and ``d = merkle_depth(n)`` bounding every path length.  Evaluated in
    ``O(#kinds)`` from the per-kind table alone: nothing here enumerates an
    interface, so a client cannot make admission cost depend on the size of
    its inputs.
    """

    table = as_kind_table(target)
    openings = 0
    gates = 0
    for kind in table.rows:
        if kind.role == VERIFICATION:
            openings += kind.copies * (kind.out_count + kind.input_count)
            gates += kind.copies * kind.size
    sampled = policy.q * policy.s
    depth = merkle_depth(table.n)
    return (
        (io_count + sampled * openings) * (1 + depth)
        + sampled * gates
        + 1
        + policy.q * table.replay_unit_count
    )
