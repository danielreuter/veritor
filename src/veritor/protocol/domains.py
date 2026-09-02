"""The commitment domains both parties derive from trusted data alone.

Three kinds of commitment cover the address space, and every address has
exactly one owner:

* ``W = [start, stop)``, the weight inputs, under the per-model root
  ``kappa_W`` (:func:`weight_domain`, owner ``WEIGHT_OWNER``);
* ``∂ \\ W``, the rest of the boundary, under the boundary root bound to the
  header (:func:`boundary_domain`, owner ``BOUNDARY_OWNER``);
* ``Int(r)`` for each replay unit ``r``, under its interior root bound to the
  replay phase (:func:`interior_domain`, owner ``r``).

All position domains are lazy: membership, rank and unrank cost ``O(depth)``
through the index, never a scan.
"""

from __future__ import annotations

from collections.abc import Mapping

from veritor.core import (
    Circuit,
    Compiled,
    Index,
    IndexedDomain,
    RangeIndexedDomain,
)

from .merkle import CommitmentDomain, MerkleTree
from .messages import Header, ProtocolError, Weights, raw_digest

BOUNDARY_OWNER = -1
WEIGHT_OWNER = -2
WEIGHT_BINDING = raw_digest("veritor/protocol/weights/v1", {"owner": WEIGHT_OWNER})
"""Weight roots outlive sessions, so their domain is bound to a fixed tag."""


def leaf_schema(circuit: Circuit, address: int) -> str:
    """The leaf schema of an address: its value width."""

    return f"u{circuit[address].width}"


def public_boundary(index: Index, weights: Weights | None) -> IndexedDomain[int]:
    """``∂ \\ W`` straight from the index; ``∂`` itself when there are no weights."""

    return index.boundary(
        exclude=None if weights is None else range(weights.start, weights.stop)
    )


def weight_domain(start: int, stop: int) -> CommitmentDomain:
    """The domain of ``kappa_W``: the inputs ``[start, stop)``, per model."""

    return CommitmentDomain(WEIGHT_BINDING, WEIGHT_OWNER, RangeIndexedDomain(start, stop))


def boundary_domain(header: Header, compiled: Compiled) -> CommitmentDomain:
    """The boundary commitment covers ``∂ \\ W`` and is bound to the header."""

    return CommitmentDomain(
        header.digest, BOUNDARY_OWNER, public_boundary(compiled.index, header.weights)
    )


def interior_domain(
    replay_phase_digest: bytes, compiled: Compiled, replay_unit: int
) -> CommitmentDomain:
    """The interior commitment of ``r`` covers ``Int(r)`` and is bound to ``J``."""

    return CommitmentDomain(
        replay_phase_digest, replay_unit, compiled.index.interior(replay_unit)
    )


def commit_weights(
    compiled: Compiled, start: int, stop: int, values: Mapping[int, object]
) -> tuple[Weights, MerkleTree]:
    """Commit the weights at inputs ``[start, stop)`` once for a model.

    The verifier holds the :class:`Weights`; the prover keeps the tree to open
    sampled weights under ``kappa_W``.
    """

    circuit = compiled.circuit
    if type(start) is not int or type(stop) is not int or not 0 <= start <= stop:
        raise ProtocolError("weights must be a range of nonnegative addresses")
    if stop > compiled.index.input_count:
        raise ProtocolError("weights must be circuit inputs")
    domain = weight_domain(start, stop)
    try:
        encoded = {
            address: circuit.encode(address, values[address]) for address in range(start, stop)
        }
    except KeyError as error:
        raise ProtocolError(f"no value for weight {error.args[0]}") from None
    tree = MerkleTree(domain, encoded, lambda address: leaf_schema(circuit, address))
    return Weights(start, stop, tree.commitment.root), tree
