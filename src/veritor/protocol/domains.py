"""The commitment domains both parties derive from trusted data alone.

Three kinds of commitment cover the address space, and every address has
exactly one owner:

* ``W``, the circuit's ``weight`` gates, under the per-model root ``kappa_W``
  (:func:`weight_domain`, owner ``WEIGHT_OWNER``);
* ``∂ = In ∪ ⋃_r Out(R_r)``, the boundary -- the ``in`` gates and every
  replay unit's declared, unpinned outputs -- under the boundary root bound to
  the header (:func:`boundary_domain`, owner ``BOUNDARY_OWNER``);
* ``Int(r)`` for each replay unit ``r``, under its interior root bound to the
  replay phase (:func:`interior_domain`, owner ``r``).

Source gates are never in an interior: an ``in`` gate is a boundary position
and a ``weight`` gate a ``kappa_W`` position.  All position domains are lazy:
membership, rank and unrank cost ``O(depth)`` through the index, never a scan.
"""

from __future__ import annotations

from collections.abc import Sequence

from veritor.core import Circuit, Compiled

from .merkle import CommitmentDomain, MerkleTree
from .messages import Header, ProtocolError, Weights, raw_digest

BOUNDARY_OWNER = -1
WEIGHT_OWNER = -2
WEIGHT_BINDING = raw_digest("veritor/protocol/weights/v2", {"owner": WEIGHT_OWNER})
"""Weight roots outlive sessions, so their domain is bound to a fixed tag."""


def leaf_schema(circuit: Circuit, address: int) -> str:
    """The leaf schema of an address: its value width."""

    return f"u{circuit[address].width}"


def weight_domain(compiled: Compiled) -> CommitmentDomain:
    """The domain of ``kappa_W``: the circuit's weight gates, per model."""

    return CommitmentDomain(WEIGHT_BINDING, WEIGHT_OWNER, compiled.index.weights())


def boundary_domain(header: Header, compiled: Compiled) -> CommitmentDomain:
    """The boundary commitment covers ``∂`` and is bound to the header."""

    return CommitmentDomain(header.digest, BOUNDARY_OWNER, compiled.index.boundary())


def interior_domain(
    replay_phase_digest: bytes, compiled: Compiled, replay_unit: int
) -> CommitmentDomain:
    """The interior commitment of ``r`` covers ``Int(r)`` and is bound to ``J``."""

    return CommitmentDomain(
        replay_phase_digest, replay_unit, compiled.index.interior(replay_unit)
    )


def commit_weights(compiled: Compiled, values: Sequence[object]) -> tuple[Weights, MerkleTree]:
    """Commit the circuit's weight gates, ``values`` by rank, once for a model.

    The verifier holds the :class:`Weights`; the prover keeps the tree to open
    sampled weights under ``kappa_W``.
    """

    circuit = compiled.circuit
    domain = weight_domain(compiled)
    if len(values) != domain.count:
        raise ProtocolError(f"expected {domain.count} weight values, got {len(values)}")
    encoded = {
        address: circuit.encode(address, value)
        for address, value in zip(circuit.weights, values, strict=True)
    }
    tree = MerkleTree(domain, encoded, lambda address: leaf_schema(circuit, address))
    return Weights(domain.count, tree.commitment.root), tree
