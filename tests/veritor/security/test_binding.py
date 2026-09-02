"""Component 1: position binding and domain separation (merkle.py, domains.py).

Every address has one owner; a leaf binds (domain, rank, position, schema,
value); a domain binds (phase digest, owner, position set, count); nodes
bind (level, index); padding is domain-bound; roots do not travel across
sessions, phases, owners or position sets; the verifier derives every domain
from trusted data alone.
"""

from __future__ import annotations

import pytest

from veritor.core import iter_domain, position_domain
from veritor.protocol import (
    BOUNDARY_OWNER,
    WEIGHT_OWNER,
    BoundaryMessage,
    Commitment,
    CommitmentDomain,
    InteriorMessage,
    MerkleTree,
    Opening,
    ProtocolError,
    ProverSession,
    VerificationCode,
    Weights,
    boundary_domain,
    interior_domain,
    merkle,
    verify_opening,
    weight_domain,
)
from veritor.protocol.domains import leaf_schema


def honest_boundary(model, expectation) -> tuple[MerkleTree, CommitmentDomain]:
    header = model.header(expectation)
    domain = boundary_domain(header, model.compiled)
    encoded = {
        int(a): model.circuit.encode(int(a), model.values[int(a)])
        for a in iter_domain(domain.positions)
    }
    return MerkleTree(domain, encoded, lambda a: leaf_schema(model.circuit, a)), domain


# -- what a root is bound to -----------------------------------------------------------


@pytest.mark.parametrize("field", ["binding", "owner", "positions", "count"])
def test_root_is_bound_to_binding_owner_position_set_and_count(model, sec, field):
    """The same values under a domain differing in one component authenticate nothing."""

    expectation = model.expectation()
    tree, domain = honest_boundary(model, expectation)
    address = model.hidden_boundary_addresses[0]
    opening = tree.open(address)
    positions = list(iter_domain(domain.positions))
    if field == "binding":
        other = CommitmentDomain(bytes(32), domain.owner, domain.positions)
    elif field == "owner":
        other = CommitmentDomain(domain.binding, model.replay_unit_of(0), domain.positions)
    elif field == "positions":  # same count, one position swapped for an interior address
        swapped = [model.interior_addresses[0] if p == positions[-1] else p for p in positions]
        other = CommitmentDomain(domain.binding, domain.owner, position_domain(swapped))
    else:  # one position fewer: the count changes with the position set
        other = CommitmentDomain(domain.binding, domain.owner, position_domain(positions[:-1]))
    assert other.domain_id != domain.domain_id
    schema = leaf_schema(model.circuit, address)
    # the honest opening under the honest root, but checked with the other domain
    assert not verify_opening(other, tree.commitment, opening, schema, sec.LIMITS)
    if field != "count":
        # and a tree honestly built under the other domain has another root
        encoded = {
            int(a): model.circuit.encode(int(a), model.values[int(a)])
            for a in iter_domain(other.positions)
        }
        foreign = MerkleTree(other, encoded, lambda a: leaf_schema(model.circuit, a))
        assert foreign.commitment.root != tree.commitment.root
        assert not verify_opening(domain, tree.commitment, foreign.open(address), schema, sec.LIMITS)
    assert verify_opening(domain, tree.commitment, opening, schema, sec.LIMITS)


def test_leaf_binds_rank_position_schema_and_value(model):
    """Changing any leaf component (or the value's schema) changes the leaf hash."""

    expectation = model.expectation()
    _, domain = honest_boundary(model, expectation)
    address = model.hidden_boundary_addresses[0]
    rank = domain.positions.rank(address)
    schema = leaf_schema(model.circuit, address)
    value = model.circuit.encode(address, model.values[address])
    reference = domain.leaf(rank, address, schema, value)
    variants = {
        domain.leaf(rank + 1, address, schema, value),
        domain.leaf(rank, address + 1, schema, value),
        domain.leaf(rank, address, schema + "x", value),
        domain.leaf(rank, address, schema, bytes([value[0] ^ 1]) + value[1:]),
        domain.node(0, 0, reference, reference),
    }
    assert reference not in variants and len(variants) == 5


def test_padding_leaves_are_domain_bound_and_cannot_be_opened(model, sec):
    """The root is the spec's, with ``pad(domain, rank)`` leaves; no position reaches them."""

    expectation = model.expectation()
    tree, domain = honest_boundary(model, expectation)
    positions = list(iter_domain(domain.positions))
    count = len(positions)
    width = 1 << merkle.merkle_depth(count)
    assert count < width  # 6 leaves in a tree of 8: two padding leaves

    def root_of(leaves: list[bytes]) -> bytes:
        level, depth = leaves, 0
        while len(level) > 1:
            level = [
                domain.node(depth, i // 2, level[i], level[i + 1]) for i in range(0, len(level), 2)
            ]
            depth += 1
        return level[0]

    leaves = [
        domain.leaf(
            rank, p, leaf_schema(model.circuit, p), model.circuit.encode(p, model.values[p])
        )
        for rank, p in enumerate(positions)
    ]
    padding = [merkle._hash(merkle._PAD, domain.domain_id, merkle._uint(r)) for r in range(count, width)]
    assert root_of(leaves + padding) == tree.commitment.root
    # the same leaves with another domain's padding do not hash to this root
    other = CommitmentDomain(bytes(32), domain.owner, domain.positions)
    foreign = [merkle._hash(merkle._PAD, other.domain_id, merkle._uint(r)) for r in range(count, width)]
    assert root_of(leaves + foreign) != tree.commitment.root
    # an opening naming a position outside the domain (an address past n, an interior address)
    path = tree.open(positions[-1]).path
    for position in (model.circuit.n, model.circuit.n + 1, model.interior_addresses[0]):
        forged = Opening(position, b"\0", path)
        assert not verify_opening(domain, tree.commitment, forged, "x", sec.LIMITS)
    # a path of the wrong length, even with the right leaf
    honest = tree.open(positions[0])
    schema = leaf_schema(model.circuit, honest.position)
    for wrong in (honest.path[:-1], honest.path + (honest.path[0],)):
        forged = Opening(honest.position, honest.value, wrong)
        assert not verify_opening(domain, tree.commitment, forged, schema, sec.LIMITS)


# -- protocol-level attacks -------------------------------------------------------------


def test_interior_committed_under_the_boundary_domain_is_invalid_opening(model, sec):
    """A prover that builds an interior tree as if it were the boundary's cannot open it."""

    expectation = model.expectation()
    header = model.header(expectation)

    def as_boundary(domain: CommitmentDomain) -> CommitmentDomain:
        if domain.owner == BOUNDARY_OWNER or domain.count == 0:
            return domain
        return CommitmentDomain(header.digest, BOUNDARY_OWNER, domain.positions)

    run = model.run(expectation, model.values, prover=sec.TamperingProver, domain_for=as_boundary)
    assert run.report.code == VerificationCode.INVALID_OPENING
    assert run.report.sampled_replay_units  # rejected at the evidence, where owners are checked

    # even the empty commitment (the source unit has no interior) is domain-bound
    def empty_as_boundary(domain: CommitmentDomain) -> CommitmentDomain:
        if domain.owner == BOUNDARY_OWNER or domain.count:
            return domain
        return CommitmentDomain(header.digest, BOUNDARY_OWNER, domain.positions)

    run = model.run(
        expectation, model.values, prover=sec.TamperingProver, domain_for=empty_as_boundary
    )
    assert run.report.code == VerificationCode.INVALID_COMMITMENT


def test_boundary_committed_under_an_interior_domain_is_invalid_opening(model, sec):
    """...and vice versa: the boundary tree keyed as a replay unit opens nothing."""

    expectation = model.expectation()
    header = model.header(expectation)

    def as_interior(domain: CommitmentDomain) -> CommitmentDomain:
        if domain.owner != BOUNDARY_OWNER:
            return domain
        return CommitmentDomain(header.digest, model.replay_unit_of(0), domain.positions)

    run = model.run(expectation, model.values, prover=sec.TamperingProver, domain_for=as_interior)
    assert run.report.code == VerificationCode.INVALID_OPENING
    assert run.report.sampled_replay_units == ()  # the I/O openings fail at the boundary phase


def test_boundary_message_replayed_from_another_session_is_invalid_opening(model, sec):
    """Session 2 (another session id, same seeds and values) rejects session 1's boundary."""

    first = model.expectation(session_id=b"session-1")
    recorded = model.run(first, model.values).transcript
    assert recorded is not None
    second = model.expectation(session_id=b"session-2")
    assert model.header(second).digest != recorded.header.digest
    run = model.run(
        second,
        model.values,
        prover=sec.TamperingProver,
        rewrite_boundary=lambda _message: recorded.boundary,
    )
    assert run.report.code == VerificationCode.INVALID_OPENING
    assert run.report.sampled_replay_units == ()


def test_commitment_count_disagreeing_with_the_domain_is_invalid_commitment(model, sec):
    expectation = model.expectation()

    def inflate(message: BoundaryMessage) -> BoundaryMessage:
        root, count = message.commitment.root, message.commitment.count
        return BoundaryMessage(Commitment(root, count + 1), message.io_openings)

    run = model.run(expectation, model.values, prover=sec.TamperingProver, rewrite_boundary=inflate)
    assert run.report.code == VerificationCode.INVALID_COMMITMENT

    def shrink(message: InteriorMessage) -> InteriorMessage:
        *rest, last = message.commitments  # the last stage's interior has two positions
        return InteriorMessage((*rest, Commitment(last.root, last.count - 1)))

    run = model.run(expectation, model.values, prover=sec.TamperingProver, rewrite_interiors=shrink)
    assert run.report.code == VerificationCode.INVALID_COMMITMENT


def test_equivocating_on_a_boundary_value_between_phases_is_invalid_opening(model, sec):
    """Commit ``v`` at a hidden boundary address, open ``v'`` there in the evidence."""

    expectation = model.expectation()
    address = model.hidden_boundary_addresses[0]
    forged = dict(model.values)
    forged[address] = (model.values[address] + 1) % (1 << model.width)
    run = model.run(expectation, model.values, prover=sec.TamperingProver, recommit_boundary=forged)
    assert run.report.code == VerificationCode.INVALID_OPENING
    assert run.report.sampled_verification_units  # at the evidence: the boundary root was fixed


def test_two_units_reading_one_address_cannot_be_shown_different_values(model, sec):
    """A hidden boundary address read by two sampled units is one leaf under one root."""

    expectation = model.expectation()
    address = model.hidden_boundary_addresses[0]
    readers = [
        unit
        for unit in range(model.index.verification_unit_count)
        if address in model.circuit.In(model.index.verification_unit(unit))
    ]
    assert len(readers) == 2
    seen: list[int] = []

    def second_reader_sees_another_value(owner: int, opening: Opening, phase: str) -> Opening:
        if phase == "evidence" and owner == BOUNDARY_OWNER and opening.position == address:
            seen.append(opening.position)
            if len(seen) == 2:
                other = (model.values[address] + 1) % (1 << model.width)
                return Opening(opening.position, model.circuit.encode(address, other), opening.path)
        return opening

    run = model.run(
        expectation,
        model.values,
        prover=sec.TamperingProver,
        rewrite_opening=second_reader_sees_another_value,
    )
    assert run.report.code == VerificationCode.INVALID_OPENING
    honest = model.run(expectation, model.values).transcript
    assert honest is not None
    values = {
        item.value for batch in honest.evidence.units for item in batch if item.position == address
    }
    assert len(values) == 1  # one owner, one leaf, one value


def test_every_address_has_exactly_one_owner(model):
    """kappa_W for weights, the boundary for inputs and declared outputs, else one replay unit."""

    circuit, index = model.circuit, model.index
    weights = set(circuit.weights)
    boundary = set(iter_domain(index.boundary()))
    interiors = {
        r: {int(a) for a in iter_domain(index.interior(r))} for r in range(index.replay_units.count)
    }
    assert not (weights & boundary)
    for r, interior in interiors.items():
        assert not (interior & boundary) and not (interior & weights)
        for other, theirs in interiors.items():
            assert other == r or not (interior & theirs)
    covered = weights | boundary | set().union(*interiors.values())
    assert covered == set(range(circuit.n))


def test_the_wire_carries_no_prover_described_domain(honest_run, sec):
    """Commitments are (root, count) only: the verifier derives every domain itself."""

    run, _ = honest_run
    document = sec.transcript_document(run.transcript)
    assert set(document["boundary"]["commitment"]) == {"count", "root"}
    for commitment in document["interiors"]["commitments"]:
        assert set(commitment) == {"count", "root"}
    assert set(document["header"]["weights"]) == {"count", "root"}


# -- kappa_W ----------------------------------------------------------------------------


def test_kappa_w_is_bound_to_the_gate_set_and_the_vector_not_the_description(model, sec):
    """One model, one root: the same weights under another description share kappa_W.

    The domain is the rank space of the weight vector bound to the gate set, so
    a model is committed once per epoch and serves every circuit compiled from
    it.  What the root does not travel across: another gate set (the leaf
    schema and the binding change), another vector length, another vector.
    """

    other = sec.Model(3, 2)  # the same weights under another description
    assert other.weights == model.weights and other.compiled.digest != model.compiled.digest
    assert other.kappa == model.kappa
    domain = weight_domain(model.gate_set, model.kappa.count)
    assert domain.domain_id == weight_domain(other.gate_set, other.kappa.count).domain_id
    rank = 0
    schema = leaf_schema(model.circuit, model.circuit.weights[rank])
    opening = other.tree.open(rank)
    assert verify_opening(domain, model.kappa.commitment, opening, schema, sec.LIMITS)
    # another gate set: a different binding, and the opening no longer authenticates
    wider = sec.Model(2, 2, width=16)
    foreign = weight_domain(wider.gate_set, model.kappa.count)
    assert foreign.domain_id != domain.domain_id
    assert not verify_opening(foreign, model.kappa.commitment, opening, schema, sec.LIMITS)
    # another vector length is another domain
    assert weight_domain(model.gate_set, model.kappa.count + 1).domain_id != domain.domain_id
    # another vector: the honest prover refuses to run under a header binding its kappa_W
    different = sec.Model(2, 2, weights=tuple(w ^ 1 for w in model.weights))
    assert different.kappa != model.kappa
    expectation = model.expectation(weights=different.kappa)
    header = model.header(expectation)
    with pytest.raises(ProtocolError, match="weight tree"):
        ProverSession(model.compiled, header, model.values, weight_tree=model.tree)


def test_kappa_w_with_another_count_is_rejected_before_any_commitment(model):
    expectation = model.expectation(weights=Weights(model.kappa.count + 1, model.kappa.root))
    run = model.run(expectation, model.values)
    assert run.report.code == VerificationCode.INVALID_COMPILED_RESULT
    assert run.transcript is None
    missing = model.expectation(weights=None)
    assert model.run(missing, model.values).report.code == VerificationCode.INVALID_COMPILED_RESULT


def test_weight_opened_with_another_value_is_invalid_opening(model, sec):
    """A weight is accepted only as kappa_W's leaf: any other value fails to open."""

    expectation = model.expectation()
    address = model.circuit.weights[0]
    rank = model.circuit.weight_rank(address)  # kappa_W positions are ranks

    def substitute(owner: int, opening: Opening, phase: str) -> Opening:
        if owner == WEIGHT_OWNER and opening.position == rank:
            other = (model.weights[0] + 1) % (1 << model.width)
            return Opening(opening.position, model.circuit.encode(address, other), opening.path)
        return opening

    run = model.run(expectation, model.values, prover=sec.TamperingProver, rewrite_opening=substitute)
    assert run.report.code == VerificationCode.INVALID_OPENING


def test_interior_domain_is_bound_to_the_replay_phase(model):
    """An interior root from one replay phase (one J) is another domain under another."""

    unit = model.replay_unit_of(0)
    first = interior_domain(bytes(32), model.compiled, unit)
    second = interior_domain(bytes([1]) + bytes(31), model.compiled, unit)
    assert first.domain_id != second.domain_id
    assert first.positions.identity_digest == second.positions.identity_digest
    assert interior_domain(bytes(32), model.compiled, unit + 1).domain_id != first.domain_id
