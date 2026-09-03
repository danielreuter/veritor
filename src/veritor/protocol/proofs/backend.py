"""The pluggable proof interface and the verifier's coverage check.

A :class:`ProofBackend` proves and verifies *batches*: a proof is about a
:class:`Statement` (the canonical, sorted set of obligations it covers plus
their kinds' programs), and its public output is bound to
``sha256(encode_statement(statement))``.  How obligations are grouped into
proofs is the prover's business (:class:`BatchPlan`); the verifier only
demands that every obligation it derived is covered by exactly one verified
proof whose statement it recomputed itself, and rejects anything else.

Batches may span sessions.  A proof message then carries, besides the indices
of *this* session's demands it covers, the canonical statement of the
*foreign* obligations in the same batch (other sessions' public statements,
nothing secret); the verifier admits them only if none claims this session
and re-derives the whole batch statement before verifying.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from veritor.protocol.messages import (
    ProofMessage,
    ProtocolError,
    Reject,
    VerificationCode,
)

from .statement import KindProgram, Obligation, Statement, Witness, make_statement
from .wire import decode_statement, encode_statement

type Proof = bytes
"""An opaque proof: whatever the backend's ``verify`` needs besides the statement."""


@runtime_checkable
class ProofBackend(Protocol):
    """One proof system behind the protocol's reveal step.

    ``backend_id`` is bound into the header so both parties run the same
    backend.  ``prove`` takes the batch statement and its witness; ``verify``
    decides a proof against the statement the *verifier* derived.  A backend
    may raise :class:`Reject` from ``verify`` to give a precise verdict code
    instead of returning ``False``.
    """

    @property
    def backend_id(self) -> str: ...

    def prove(self, statement: Statement, witness: Witness) -> Proof: ...

    def verify(self, statement: Statement, proof: Proof) -> bool: ...


@dataclass(frozen=True, slots=True)
class ForeignBatch:
    """Other sessions' obligations (and their witness) the prover folds into a proof."""

    statement: Statement
    witness: Witness

    def __post_init__(self) -> None:
        self.witness.for_statement(self.statement)


@dataclass(frozen=True, slots=True)
class BatchPlan:
    """How a prover groups the obligations of one session into proofs.

    ``groups`` partitions ``range(len(obligations))`` (the indices of the
    sampled VUs in ``T`` order); each group becomes one proof.  A group may be
    joined with a :class:`ForeignBatch` (``foreign[g]``: other sessions'
    obligations with their witness) so one proof amortizes across sessions.
    """

    groups: tuple[tuple[int, ...], ...]
    foreign: tuple[ForeignBatch | None, ...] = ()

    def __post_init__(self) -> None:
        seen: set[int] = set()
        for group in self.groups:
            if type(group) is not tuple or not group:
                raise ProtocolError("batch plan groups must be nonempty tuples")
            if tuple(sorted(set(group))) != group:
                raise ProtocolError("batch plan groups must be sorted and unique")
            if seen & set(group):
                raise ProtocolError("batch plan groups must be disjoint")
            seen.update(group)
        if self.foreign and len(self.foreign) != len(self.groups):
            raise ProtocolError("batch plan foreign batches must match its groups")

    def foreign_of(self, group: int) -> ForeignBatch | None:
        return self.foreign[group] if self.foreign else None

    @staticmethod
    def one_per_unit(count: int) -> BatchPlan:
        return BatchPlan(tuple((index,) for index in range(count)))

    @staticmethod
    def single(count: int) -> BatchPlan:
        return BatchPlan((tuple(range(count)),) if count else ())

    @staticmethod
    def chunked(count: int, size: int) -> BatchPlan:
        if size <= 0:
            raise ProtocolError("batch size must be positive")
        return BatchPlan(
            tuple(
                tuple(range(start, min(start + size, count)))
                for start in range(0, count, size)
            )
        )

    def covers(self, count: int) -> bool:
        return sorted(index for group in self.groups for index in group) == list(
            range(count)
        )


def merge_statement(
    gate_set_id: str,
    gate_set_digest: bytes,
    width: int,
    kinds: Iterable[KindProgram],
    obligations: Iterable[Obligation],
    foreign: Statement | None,
) -> Statement:
    """The batch statement of ``obligations`` joined with a foreign statement."""

    kinds = list(kinds)
    obligations = list(obligations)
    if foreign is not None:
        if (
            foreign.gate_set_id != gate_set_id
            or foreign.gate_set_digest != gate_set_digest
            or foreign.width != width
        ):
            raise ProtocolError("foreign obligations use another gate set")
        kinds.extend(foreign.kinds)
        obligations.extend(foreign.obligations)
    return make_statement(gate_set_id, gate_set_digest, width, kinds, obligations)


type Openings = tuple[tuple[bytes, tuple[bytes, ...]], ...]
"""One obligation's witness: ``(value, path)`` per position, in position order."""


def prove_plan(
    backend: ProofBackend,
    plan: BatchPlan,
    obligations: Sequence[Obligation],
    openings: Sequence[Openings],
    kinds: Sequence[KindProgram],
    gate_set_id: str,
    gate_set_digest: bytes,
    width: int,
) -> tuple[ProofMessage, ...]:
    """The prover's side: one proof per group of ``plan`` over this session's obligations."""

    if not plan.covers(len(obligations)):
        raise ProtocolError(
            "the batch plan does not cover every sampled VU exactly once"
        )
    messages: list[ProofMessage] = []
    for number, group in enumerate(plan.groups):
        foreign = plan.foreign_of(number)
        own = [obligations[index] for index in group]
        statement = merge_statement(
            gate_set_id,
            gate_set_digest,
            width,
            kinds,
            own,
            None if foreign is None else foreign.statement,
        )
        witness_of = {obligations[index].key: openings[index] for index in group}
        if foreign is not None:
            for item, opened in zip(
                foreign.statement.obligations, foreign.witness.obligations, strict=True
            ):
                witness_of[item.key] = opened
        witness = Witness(tuple(witness_of[item.key] for item in statement.obligations))
        proof = backend.prove(statement, witness)
        messages.append(
            ProofMessage(
                group,
                proof,
                b"" if foreign is None else encode_statement(foreign.statement),
            )
        )
    return tuple(messages)


def check_coverage(
    demanded: Sequence[Obligation],
    kinds: Sequence[KindProgram],
    gate_set_id: str,
    gate_set_digest: bytes,
    width: int,
    proofs: Sequence[ProofMessage],
    verify: Callable[[Statement, Proof], bool],
    *,
    on_proof: Callable[[int, ProofMessage, Statement], None] | None = None,
) -> None:
    """Every demanded obligation is covered by exactly one verified proof.

    ``verify`` is the backend's verifier; ``on_proof`` is called with each
    proof's statement before verification (for resource limits).  Raises
    :class:`Reject` with the precise code on any failure.
    """

    covered: list[int] = []
    for message in proofs:
        covered.extend(message.units)
    if sorted(covered) != list(range(len(demanded))):
        raise Reject(
            VerificationCode.COVERAGE_MISMATCH,
            f"proofs cover VU slots {sorted(covered)}, demanded {list(range(len(demanded)))}",
        )
    session = demanded[0].session if demanded else None
    for number, message in enumerate(proofs):
        own = [demanded[index] for index in message.units]
        foreign: Statement | None = None
        if message.foreign:
            try:
                foreign = decode_statement(message.foreign)
            except ProtocolError as error:
                raise Reject(
                    VerificationCode.MALFORMED_TRANSCRIPT, f"proof {number}: {error}"
                ) from error
            if any(item.session == session for item in foreign.obligations):
                raise Reject(
                    VerificationCode.COVERAGE_MISMATCH,
                    f"proof {number} smuggles an obligation claiming this session as foreign",
                )
        try:
            statement = merge_statement(
                gate_set_id, gate_set_digest, width, kinds, own, foreign
            )
        except ProtocolError as error:
            raise Reject(
                VerificationCode.COVERAGE_MISMATCH, f"proof {number}: {error}"
            ) from error
        if on_proof is not None:
            on_proof(number, message, statement)
        if not verify(statement, message.proof):
            raise Reject(
                VerificationCode.PROOF_REJECTED,
                f"proof {number} does not verify for the demanded statement",
            )
