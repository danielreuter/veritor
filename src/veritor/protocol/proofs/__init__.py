"""The pluggable, batchable proof layer behind the protocol's reveal step.

* :class:`Obligation` -- the public statement for one sampled verification
  unit (VU), derived by the verifier from the challenge and the Index
  (:func:`derive_obligations`); :class:`KindProgram` -- the relation of a
  kind in copy-relative coordinates; :class:`Statement` -- a batch of both,
  canonical (:func:`encode_statement`, :func:`statement_digest`).
* :class:`ProofBackend` -- ``prove(statement, witness) -> proof`` and
  ``verify(statement, proof) -> bool`` behind a ``backend_id`` the header
  binds.  :class:`BatchPlan` is the prover's grouping; :func:`check_coverage`
  is the verifier's "every demand covered exactly once" check.
* Backends: :class:`TransparentBackend` (openings as proof, the reference),
  :class:`SP1Backend` (the SP1 zkVM guest in ``zk/sp1``), :class:`OpenVMBackend`
  (adapter stub for ``openvm-tc-bench``).
* :mod:`veritor.protocol.proofs.costs` -- measured constants and the estimate
  of prover seconds and ``alpha`` for a batch.
"""

from __future__ import annotations

from veritor.core import Compiled, VerificationLimits

from ..messages import TRANSPARENT_BACKEND, ProtocolError
from .backend import (
    BatchPlan,
    ForeignBatch,
    Openings,
    Proof,
    ProofBackend,
    check_coverage,
    merge_statement,
    prove_plan,
)
from .derive import (
    DECLARED_KIND,
    DECLARED_PROGRAM,
    Layout,
    derive_obligation,
    derive_obligations,
    kind_program,
    statement_width,
)
from .openvm import OPENVM_BACKEND, OpenVMBackend
from .sp1 import SP1_BACKEND, SP1Backend, SP1Unavailable, sp1_toolchain_available
from .statement import (
    LOCAL,
    PORT,
    Arg,
    CommitmentRef,
    GateOp,
    KindProgram,
    Obligation,
    PositionRef,
    Statement,
    Witness,
    make_statement,
)
from .transparent import TransparentBackend
from .wire import (
    STATEMENT_MAGIC,
    WITNESS_MAGIC,
    decode_statement,
    decode_witness,
    encode_obligation,
    encode_statement,
    encode_witness,
    statement_digest,
)


def resolve_backend(
    backend: ProofBackend | None,
    backend_id: str,
    compiled: Compiled,
    limits: VerificationLimits | None = None,
) -> ProofBackend:
    """The backend a session runs: ``backend`` if given (its id must match), else the default.

    Only the transparent backend can be defaulted; a zkVM backend needs a
    configured host binary and must be passed in.
    """

    if backend is not None:
        if backend.backend_id != backend_id:
            raise ProtocolError(
                f"the header binds backend {backend_id!r} but {backend.backend_id!r} was supplied"
            )
        return backend
    if backend_id == TRANSPARENT_BACKEND:
        return TransparentBackend(compiled.circuit.gate_set, compiled, limits)
    raise ProtocolError(f"no proof backend configured for {backend_id!r}")


__all__ = [
    "DECLARED_KIND",
    "DECLARED_PROGRAM",
    "LOCAL",
    "OPENVM_BACKEND",
    "PORT",
    "SP1_BACKEND",
    "STATEMENT_MAGIC",
    "TRANSPARENT_BACKEND",
    "WITNESS_MAGIC",
    "Arg",
    "BatchPlan",
    "CommitmentRef",
    "ForeignBatch",
    "GateOp",
    "KindProgram",
    "Layout",
    "Obligation",
    "OpenVMBackend",
    "Openings",
    "PositionRef",
    "Proof",
    "ProofBackend",
    "SP1Backend",
    "SP1Unavailable",
    "Statement",
    "TransparentBackend",
    "Witness",
    "check_coverage",
    "decode_statement",
    "decode_witness",
    "derive_obligation",
    "derive_obligations",
    "encode_obligation",
    "encode_statement",
    "encode_witness",
    "kind_program",
    "make_statement",
    "merge_statement",
    "prove_plan",
    "resolve_backend",
    "sp1_toolchain_available",
    "statement_digest",
    "statement_width",
]
