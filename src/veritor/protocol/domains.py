"""The commitment domains both parties derive from trusted data alone.

Three kinds of commitment cover the address space, and every address has
exactly one owner:

* ``W``, the model's weight vector, under the per-model root ``kappa_W``
  (:func:`weight_domain`, owner ``WEIGHT_OWNER``).  Its positions are ranks:
  position ``k`` is the ``k``-th weight gate in address order, whatever the
  circuit.  The domain is bound to a fixed tag and the gate set, never to a
  description, so a model is committed once and the same root serves every
  circuit compiled from it;
* ``∂ = In ∪ ⋃_r Out(R_r)``, the boundary -- the ``in`` gates and every
  replay unit's declared, unpinned outputs -- under the boundary root bound to
  the header (:func:`boundary_domain`, owner ``BOUNDARY_OWNER``);
* ``Int(r)`` for each replay unit ``r``, under its interior root bound to the
  replay phase (:func:`interior_domain`, owner ``r``).

Source gates are never in an interior: an ``in`` gate is a boundary position
and the ``weight`` gate at address ``a`` is the ``kappa_W`` position
``weight_rank(a)``.  All position domains are lazy: membership, rank and
unrank cost ``O(depth)`` through the index, never a scan.
"""

from __future__ import annotations

from collections.abc import Sequence

from veritor.core import Circuit, Compiled, GateSet, RangeIndexedDomain, encode_value

from .merkle import CommitmentDomain, MerkleTree
from .messages import Header, ProtocolError, Weights, raw_digest

BOUNDARY_OWNER = -1
WEIGHT_OWNER = -2
WEIGHT_TAG = "veritor/protocol/weights/v3"
"""Weight roots outlive sessions and descriptions: their domain is bound to this tag."""


def leaf_schema(circuit: Circuit, address: int) -> str:
    """The leaf schema of an address: its value width."""

    return f"u{circuit[address].width}"


def weight_width(gate_set: GateSet) -> int:
    """The width of the gate set's weight gates, which fixes the ``kappa_W`` leaf schema."""

    widths = {gate_set[name].width for name in gate_set.weight_gates}
    if len(widths) != 1:
        raise ProtocolError("kappa_W needs a gate set with weight gates of one width")
    return widths.pop()


def weight_domain(gate_set: GateSet, count: int) -> CommitmentDomain:
    """The domain of ``kappa_W``: the ranks ``0 .. count - 1`` of a model's weight vector.

    Position ``k`` is the ``k``-th weight gate in address order of whichever
    circuit is being verified; the binding is a fixed tag and the gate set
    (whose weight width fixes the leaf schema), so the root does not depend
    on the description and is committed once per model.
    """

    if not isinstance(gate_set, GateSet):
        raise ProtocolError("the weight domain requires a GateSet")
    if type(count) is not int or count < 0:
        raise ProtocolError("weight count must be a nonnegative integer")
    binding = raw_digest(WEIGHT_TAG, {"gate_set": gate_set.digest, "owner": WEIGHT_OWNER})
    return CommitmentDomain(binding, WEIGHT_OWNER, RangeIndexedDomain(count))


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


def commit_weights(gate_set: GateSet, values: Sequence[object]) -> tuple[Weights, MerkleTree]:
    """Commit a model's weight vector, once per model: leaf ``k`` holds ``values[k]``.

    No circuit is needed: the model is committed before any request exists.
    The verifier holds the :class:`Weights`; the prover keeps the tree to open
    sampled weight gates under ``kappa_W`` at their ranks.
    """

    if not isinstance(gate_set, GateSet):
        raise ProtocolError("commit_weights requires a GateSet")
    width = weight_width(gate_set)
    domain = weight_domain(gate_set, len(values))
    encoded = {rank: encode_value(width, value) for rank, value in enumerate(values)}
    tree = MerkleTree(domain, encoded, lambda _rank: f"u{width}")
    return Weights(domain.count, tree.commitment.root), tree
