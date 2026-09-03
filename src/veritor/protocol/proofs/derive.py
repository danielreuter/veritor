"""Deriving obligations and kind programs from the trusted ``(C, I)``.

The verifier never takes the prover's word for what a proof must cover: the
obligation for a sampled VU is computed here from the Index, the header and
the commitments the verifier itself accepted, and the kind's program is read
off the definition the VU is a copy of.  The prover runs the same code on the
same trusted data, so both sides produce identical statement bytes.

A sampled VU the prover *declared* incorrect (``InteriorMessage.declarations``,
validated by the verifier before the s-challenge) is obliged to open exactly
the same positions -- its values stay authenticated under the accepted roots,
and its ``in`` gates stay pinned to the public inputs -- but under the
:data:`DECLARED_KIND` program, which has no gates: the relation check is
vacuous.  Both parties derive this from the declarations they hold, so a
proof over a statement that skips a relation can only exist for a VU the
transcript says was declared.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import Protocol

from veritor.core import Circuit, Compiled, GateSet, Index, IndexNode, Position
from veritor.core.description import LOCAL, CallStep, Frame
from veritor.protocol.domains import leaf_schema
from veritor.protocol.merkle import CommitmentDomain
from veritor.protocol.messages import Commitment, Header, ProtocolError, raw_digest

from .statement import LOCAL as LOCAL_ARG
from .statement import (
    PORT,
    Arg,
    CommitmentRef,
    GateOp,
    KindProgram,
    Obligation,
    PositionRef,
)


class Layout(Protocol):
    """What obligation derivation needs from a session's address layout."""

    @property
    def compiled(self) -> Compiled: ...

    @property
    def circuit(self) -> Circuit: ...

    @property
    def index(self) -> Index: ...

    def position(self, owner: int, address: int) -> int: ...

    def required(self, unit: int) -> tuple[tuple[int, int], ...]: ...


def _resolve(root: Frame, frame: Frame, space: str, value: int) -> Arg:
    """Resolve a reference made inside ``frame`` relative to the copy ``root``."""

    while True:
        if space == LOCAL:
            is_gate, resolved = frame.definition.slot_source(value)
            if is_gate:
                return (LOCAL_ARG, frame.base + resolved - root.base)
            value = resolved
        if frame.depth == root.depth:
            return (PORT, value)
        parent = frame.parent
        assert parent is not None
        step = parent.definition.steps[frame.step]
        assert isinstance(step, CallStep)
        item, element = step.arg_at(value)
        value = item.element(element, frame.j)
        space = item.space
        frame = parent


def kind_program(node: IndexNode) -> KindProgram:
    """The program of ``node``'s kind, read off its definition in relative coordinates."""

    root = node.frame
    gates: list[tuple[str, list[Arg]]] = []
    ports: set[int] = set()
    for address in root.interval:
        frame, step = root.locate(address)
        args: list[Arg] = []
        for item in step.args:
            for k in range(item.count):
                arg = _resolve(root, frame, item.space, item.element(k))
                if arg[0] == PORT:
                    ports.add(arg[1])
                args.append(arg)
        gates.append((step.gate.name, args))
    ordered = tuple(sorted(ports))
    if ordered != tuple(root.definition.reads):
        raise ProtocolError(
            "the kind's program reads ports other than its definition declares"
        )
    slot = {ordinal: k for k, ordinal in enumerate(ordered)}
    return KindProgram(
        bytes.fromhex(node.kind),
        len(gates),
        ordered,
        tuple(
            GateOp(op, tuple((PORT, slot[v]) if s == PORT else (s, v) for s, v in args))
            for op, args in gates
        ),
    )


DECLARED_KIND = raw_digest("veritor/protocol/proofs/declared/v1", {"gates": 0})
"""The kind digest of a declared VU's obligation: a program of no gates and no ports."""

DECLARED_PROGRAM = KindProgram(DECLARED_KIND, 0, (), ())
"""The vacuous relation every declared VU is checked against (its openings still are)."""


def statement_width(gate_set: GateSet) -> int:
    """The word width of a statement over ``gate_set``: its widest gate.

    The transparent backend decodes every opened value by its own leaf
    schema, so a set of mixed widths (the pinned GPT-2 gates: 1-, 16- and
    32-bit) needs no single word.  A zkVM guest resolves the gate set it
    implements by ``(id, width)`` and rejects any other, so the width is
    also what tells a guest which machine word a batch is in.
    """

    widths = {gate.width for gate in gate_set}
    if not widths:
        raise ProtocolError("proof statements need a gate set with at least one gate")
    return max(widths)


def derive_obligation(
    layout: Layout,
    header: Header,
    commitments: Mapping[int, tuple[CommitmentDomain, Commitment]],
    unit: int,
    program: KindProgram | None = None,
    *,
    declared: bool = False,
) -> Obligation:
    """The obligation for sampled VU ``unit`` under the given accepted commitments.

    ``commitments`` maps an owner (``WEIGHT_OWNER``, ``BOUNDARY_OWNER`` or a
    replay unit) to its domain and root; every owner the VU touches must be
    present.  ``program`` may be supplied when already derived for the kind.
    With ``declared`` the obligation opens the same positions under
    :data:`DECLARED_PROGRAM` instead: authenticated, relation unchecked.
    """

    circuit = layout.circuit
    node = layout.index.verification_unit(unit)
    if declared:
        if program is not None and program != DECLARED_PROGRAM:
            raise ProtocolError("a declared VU is obliged under the declared program")
        program = DECLARED_PROGRAM
    elif program is None:
        program = kind_program(node)
    elif program.kind != bytes.fromhex(node.kind):
        raise ProtocolError("program is for another kind")
    required = layout.required(unit)
    owners = sorted({owner for owner, _ in required})
    refs: list[CommitmentRef] = []
    ref_index: dict[int, int] = {}
    for owner in owners:
        try:
            domain, commitment = commitments[owner]
        except KeyError:
            raise ProtocolError(
                f"sampled VU {unit} needs a commitment from owner {owner}"
            ) from None
        ref_index[owner] = len(refs)
        refs.append(
            CommitmentRef(owner, domain.domain_id, commitment.root, commitment.count)
        )
    positions: list[PositionRef] = []
    slot_of: dict[int, int] = {}
    for owner, address in required:
        domain = commitments[owner][0]
        position = layout.position(owner, address)
        try:
            rank = domain.positions.rank(Position(position))
        except KeyError:
            raise ProtocolError(
                f"address {address} is not in owner {owner}'s domain"
            ) from None
        gate = circuit[address]
        expected = (
            header.public_inputs[circuit.input_rank(address)] if gate.is_input else None
        )
        slot_of[address] = len(positions)
        positions.append(
            PositionRef(
                ref_index[owner],
                rank,
                position,
                leaf_schema(circuit, address),
                expected,
            )
        )
    frame = node.frame
    try:
        inputs = tuple(
            slot_of[frame.input_address(ordinal)] for ordinal in program.ports
        )
        gates = () if declared else tuple(slot_of[address] for address in node.interval)
    except KeyError as error:
        raise ProtocolError(
            f"VU {unit} touches address {error.args[0]} outside its openings"
        ) from None
    replay_unit = node.replay_unit
    if replay_unit is None:
        raise ProtocolError(f"VU {unit} lies in no replay unit")
    return Obligation(
        header.digest,
        bytes.fromhex(layout.compiled.digest),
        unit,
        replay_unit,
        program.kind,
        tuple(refs),
        tuple(positions),
        inputs,
        gates,
    )


def derive_obligations(
    layout: Layout,
    header: Header,
    commitments: Mapping[int, tuple[CommitmentDomain, Commitment]],
    units: Sequence[int],
    declared: Collection[int] = (),
) -> tuple[tuple[Obligation, ...], tuple[KindProgram, ...]]:
    """Obligations for the sampled VUs in ``units`` order, and the programs of their kinds.

    ``declared`` are the VUs the interior message declared incorrect (already
    validated against the header's ``max_faults`` and the opened RUs); a
    sampled one is obliged under :data:`DECLARED_PROGRAM`.
    """

    programs: dict[str, KindProgram] = {}
    obligations: list[Obligation] = []
    for unit in units:
        if unit in declared:
            programs.setdefault(DECLARED_KIND.hex(), DECLARED_PROGRAM)
            obligations.append(
                derive_obligation(layout, header, commitments, unit, declared=True)
            )
            continue
        node = layout.index.verification_unit(unit)
        program = programs.get(node.kind)
        if program is None:
            program = programs[node.kind] = kind_program(node)
        obligations.append(
            derive_obligation(layout, header, commitments, unit, program)
        )
    return tuple(obligations), tuple(programs.values())
