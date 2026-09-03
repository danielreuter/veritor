"""The transparent backend: the proof *is* the openings.

This is the protocol as it stood before proofs were pluggable, expressed
through :class:`ProofBackend`: the prover hands over the witness (values and
Merkle paths), and the verifier authenticates every opening under the roots
the statement names, compares pinned public inputs, decodes each value with
the gate set's canonical codec and recomputes every relation with the gate
set's pinned semantics.  It gives the verifier every value it checks, so it is
not zero-knowledge; it is the reference every other backend must agree with.
"""

from __future__ import annotations

from veritor.core import Compiled, GateSet, VerificationLimits
from veritor.protocol.domains import WEIGHT_OWNER

# The framing primitives are private to merkle.py on purpose: nothing but the
# commitment code and this reference checker may spell out a leaf or node hash.
from veritor.protocol.merkle import _hash as framed_hash
from veritor.protocol.merkle import _uint, merkle_depth
from veritor.protocol.messages import ProtocolError, Reject, VerificationCode

from .statement import LOCAL, Obligation, PositionRef, Statement, Witness
from .wire import decode_witness, encode_witness

TRANSPARENT_BACKEND = "transparent"


def fold_path(
    domain_id: bytes, rank: int, leaf: bytes, path: tuple[bytes, ...]
) -> bytes:
    """Fold an authentication path to its root with the ``merkle.py`` node framing."""

    digest = leaf
    cursor = rank
    for level, sibling in enumerate(path):
        pair = (digest, sibling) if cursor % 2 == 0 else (sibling, digest)
        digest = framed_hash(
            b"node", domain_id, _uint(level), _uint(cursor >> 1), *pair
        )
        cursor >>= 1
    return digest


def leaf_hash(
    domain_id: bytes, rank: int, position: int, schema: str, value: bytes
) -> bytes:
    """``CommitmentDomain.leaf`` from the domain id alone."""

    return framed_hash(
        b"leaf", domain_id, _uint(rank), _uint(position), schema.encode("utf-8"), value
    )


class TransparentBackend:
    """Openings as the proof; Merkle paths and relations checked in Python.

    ``gate_set`` supplies the pinned semantics and must be the statement's.
    ``compiled``, when given, lets rejection details name addresses (a weight
    position is a rank under ``kappa_W``; every other position is its address).
    """

    backend_id = TRANSPARENT_BACKEND

    def __init__(
        self,
        gate_set: GateSet,
        compiled: Compiled | None = None,
        limits: VerificationLimits | None = None,
    ) -> None:
        if not isinstance(gate_set, GateSet):
            raise ProtocolError("TransparentBackend requires a GateSet")
        self._gate_set = gate_set
        self._compiled = compiled
        self._limits = VerificationLimits() if limits is None else limits

    def prove(self, statement: Statement, witness: Witness) -> bytes:
        witness.for_statement(statement)
        return encode_witness(witness)

    def verify(self, statement: Statement, proof: bytes) -> bool:
        self.check(statement, proof)
        return True

    def _where(self, obligation: Obligation, ref: PositionRef) -> str:
        owner = obligation.commitments[ref.commitment].owner
        if owner == WEIGHT_OWNER:
            compiled = self._compiled
            if (
                compiled is not None
                and bytes.fromhex(compiled.digest) == obligation.compiled
            ):
                return f"address {compiled.index.weights().unrank(ref.rank)}"
            return f"weight rank {ref.rank}"
        return f"address {ref.position}"

    def check(self, statement: Statement, proof: bytes) -> None:
        """Verify, raising :class:`Reject` with the precise code on failure."""

        if statement.gate_set_digest != bytes.fromhex(self._gate_set.digest):
            raise Reject(
                VerificationCode.EXPECTATION_MISMATCH,
                "the statement names another gate set than the transparent checker's",
            )
        try:
            witness = decode_witness(proof)
            witness.for_statement(statement)
        except ProtocolError as error:
            raise Reject(VerificationCode.MALFORMED_TRANSCRIPT, str(error)) from error
        for obligation, openings in zip(
            statement.obligations, witness.obligations, strict=True
        ):
            values = self._open_all(obligation, openings)
            self._check_relations(statement, obligation, values)

    def _open_all(
        self,
        obligation: Obligation,
        openings: tuple[tuple[bytes, tuple[bytes, ...]], ...],
    ) -> list[int]:
        values: list[int] = []
        for ref, (value, path) in zip(obligation.positions, openings, strict=True):
            commitment = obligation.commitments[ref.commitment]
            where = self._where(obligation, ref)
            failed = Reject(
                VerificationCode.INVALID_OPENING,
                f"opening of {where} failed under owner {commitment.owner}",
            )
            if ref.rank >= commitment.count or len(path) != merkle_depth(
                commitment.count
            ):
                raise failed
            self._limits.enforce("max_proof_bytes", len(value) + 32 * len(path))
            leaf = leaf_hash(
                commitment.domain_id, ref.rank, ref.position, ref.schema, value
            )
            if fold_path(commitment.domain_id, ref.rank, leaf, path) != commitment.root:
                raise failed
            if ref.expected is not None and ref.expected != value:
                raise Reject(
                    VerificationCode.PUBLIC_IO_MISMATCH,
                    f"input at {where} differs from the public input",
                )
            try:
                values.append(self._decode(ref.schema, value))
            except ProtocolError as error:
                raise Reject(
                    VerificationCode.INVALID_VALUE,
                    f"value at {where} is not canonical: {error}",
                ) from error
        return values

    def _decode(self, schema: str, value: bytes) -> int:
        if not schema.startswith("u"):
            raise ProtocolError(f"unknown leaf schema {schema!r}")
        try:
            width = int(schema[1:])
        except ValueError:
            raise ProtocolError(f"unknown leaf schema {schema!r}") from None
        if len(value) != (width + 7) // 8:
            raise ProtocolError(
                f"encoded value must be exactly {(width + 7) // 8} bytes"
            )
        decoded = int.from_bytes(value, "big")
        if decoded >= 1 << width:
            raise ProtocolError(f"encoded value is not a {width}-bit value")
        return decoded

    def _check_relations(
        self, statement: Statement, obligation: Obligation, values: list[int]
    ) -> None:
        program = statement.program(obligation.kind)
        for offset, gate in enumerate(program.gates):
            try:
                semantics = self._gate_set[gate.op]
            except Exception as error:
                raise Reject(
                    VerificationCode.INVALID_COMPILED_RESULT, f"unknown gate {gate.op}"
                ) from error
            if semantics.source is not None:
                continue
            args = tuple(
                values[
                    obligation.gates[value]
                    if space == LOCAL
                    else obligation.inputs[value]
                ]
                for space, value in gate.args
            )
            out_ref = obligation.positions[obligation.gates[offset]]
            where = self._where(obligation, out_ref)
            try:
                satisfied = semantics.check(args, values[obligation.gates[offset]])
            except Exception as error:
                raise Reject(
                    VerificationCode.TRUSTED_SERVICE_FAILURE,
                    f"gate {gate.op} raised at {where}: {error}",
                ) from error
            if not satisfied:
                raise Reject(
                    VerificationCode.RELATION_REJECTED,
                    f"gate at {where} violates {gate.op}",
                )
