"""Draft-v4 staged value commitments over the memoized call-DAG compiler.

This is a protocol-mechanics prototype, not a production commitment library.
It implements the load-bearing reveal order:

    boundary commitment -> challenged units J -> unit commitments -> sample T

The verifier derives every commitment domain from ``ReplayPlan``.  The client
never supplies boundary indices, unit interiors, or gate routing.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import random
import struct
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from prototypes.call_dag import (
    Kernel,
    KernelReject,
    ReplayPlan,
    ValidatedDefinition,
)


_LEAF_TAG = b"veritor-staged-value-leaf-v1"
_PADDING_TAG = b"veritor-staged-value-padding-v1"
_NODE_TAG = b"veritor-staged-value-node-v1"
_EMPTY_TAG = b"veritor-staged-value-empty-v1"


class ProtocolReject(ValueError):
    """A staged-protocol message is invalid or arrives out of order."""


def _hash_parts(tag: bytes, *parts: bytes) -> bytes:
    digest = hashlib.sha256()
    digest.update(tag)
    for part in parts:
        digest.update(struct.pack(">Q", len(part)))
        digest.update(part)
    return digest.digest()


def _tree_depth(item_count: int) -> int:
    depth = 0
    while (1 << depth) < item_count:
        depth += 1
    return depth


def _cell_payload(value: int, cell_bits: int) -> bytes:
    if type(value) is not int or not 0 <= value < (1 << cell_bits):
        raise ProtocolReject(f"value is not a {cell_bits}-bit cell")
    return value.to_bytes((cell_bits + 7) // 8, "big")


def _leaf_hash(domain: bytes, position: int, value: int, cell_bits: int) -> bytes:
    return _hash_parts(
        _LEAF_TAG,
        domain,
        struct.pack(">Q", position),
        _cell_payload(value, cell_bits),
    )


def _padding_hash(domain: bytes, local_index: int) -> bytes:
    return _hash_parts(_PADDING_TAG, domain, struct.pack(">Q", local_index))


def _node_hash(left: bytes, right: bytes) -> bytes:
    return _hash_parts(_NODE_TAG, left, right)


def _empty_root(domain: bytes) -> bytes:
    return _hash_parts(_EMPTY_TAG, domain)


def _canonical_positions(positions: Sequence[int]) -> tuple[int, ...]:
    result = tuple(positions)
    if any(
        type(position) is not int or not 0 <= position < (1 << 64)
        for position in result
    ):
        raise ProtocolReject("commitment positions must be unsigned 64-bit integers")
    if tuple(sorted(set(result))) != result:
        raise ProtocolReject("commitment positions must be sorted and unique")
    return result


@dataclass(frozen=True)
class ValueCommitment:
    root: bytes
    value_count: int


@dataclass(frozen=True)
class ValueOpening:
    position: int
    value: int
    path: tuple[bytes, ...]


@dataclass(frozen=True)
class IndexedValueTree:
    """Private Merkle material for one verifier-derived indexed domain."""

    domain: bytes
    positions: tuple[int, ...]
    values: tuple[int, ...]
    cell_bits: int
    levels: tuple[tuple[bytes, ...], ...]

    @staticmethod
    def build(
        *,
        domain: bytes,
        positions: Sequence[int],
        values: Mapping[int, int],
        cell_bits: int,
    ) -> IndexedValueTree:
        if type(domain) is not bytes or not domain:
            raise ProtocolReject("commitment domain must be nonempty bytes")
        if type(cell_bits) is not int or cell_bits <= 0:
            raise ProtocolReject("cell_bits must be positive")
        canonical = _canonical_positions(positions)
        if set(values) != set(canonical):
            raise ProtocolReject("committed values do not exactly match the domain")
        ordered_values = tuple(values[position] for position in canonical)
        for value in ordered_values:
            _cell_payload(value, cell_bits)

        if not canonical:
            return IndexedValueTree(
                domain=domain,
                positions=canonical,
                values=ordered_values,
                cell_bits=cell_bits,
                levels=(),
            )

        width = 1 << _tree_depth(len(canonical))
        leaves = [
            _leaf_hash(domain, position, value, cell_bits)
            for position, value in zip(canonical, ordered_values)
        ]
        leaves.extend(
            _padding_hash(domain, local_index)
            for local_index in range(len(leaves), width)
        )
        levels = [tuple(leaves)]
        level = leaves
        while len(level) > 1:
            level = [
                _node_hash(level[index], level[index + 1])
                for index in range(0, len(level), 2)
            ]
            levels.append(tuple(level))
        return IndexedValueTree(
            domain=domain,
            positions=canonical,
            values=ordered_values,
            cell_bits=cell_bits,
            levels=tuple(levels),
        )

    @property
    def commitment(self) -> ValueCommitment:
        root = _empty_root(self.domain) if not self.levels else self.levels[-1][0]
        return ValueCommitment(root=root, value_count=len(self.positions))

    def open(self, position: int) -> ValueOpening:
        local_index = bisect.bisect_left(self.positions, position)
        if (
            local_index >= len(self.positions)
            or self.positions[local_index] != position
        ):
            raise ProtocolReject(f"position {position} is outside this commitment")
        index = local_index
        path = []
        for level in self.levels[:-1]:
            path.append(level[index ^ 1])
            index >>= 1
        return ValueOpening(
            position=position,
            value=self.values[local_index],
            path=tuple(path),
        )


def validate_commitment_domain(
    commitment: ValueCommitment,
    *,
    domain: bytes,
    positions: Sequence[int],
) -> None:
    canonical = _canonical_positions(positions)
    if (
        not isinstance(commitment, ValueCommitment)
        or type(commitment.root) is not bytes
        or len(commitment.root) != hashlib.sha256().digest_size
        or type(commitment.value_count) is not int
        or commitment.value_count != len(canonical)
    ):
        raise ProtocolReject("value commitment has the wrong shape")
    if not canonical and commitment.root != _empty_root(domain):
        raise ProtocolReject("empty value commitment has a noncanonical root")


def verify_value_opening(
    commitment: ValueCommitment,
    *,
    domain: bytes,
    positions: Sequence[int],
    opening: ValueOpening,
    cell_bits: int,
) -> bool:
    try:
        if (
            not isinstance(opening, ValueOpening)
            or type(opening.position) is not int
            or opening.position < 0
            or type(opening.path) is not tuple
        ):
            return False
        validate_commitment_domain(
            commitment,
            domain=domain,
            positions=positions,
        )
        canonical = _canonical_positions(positions)
        local_index = bisect.bisect_left(canonical, opening.position)
        if (
            local_index >= len(canonical)
            or canonical[local_index] != opening.position
            or len(opening.path) != _tree_depth(len(canonical))
        ):
            return False
        value_hash = _leaf_hash(
            domain,
            opening.position,
            opening.value,
            cell_bits,
        )
    except ProtocolReject:
        return False

    index = local_index
    for sibling in opening.path:
        if type(sibling) is not bytes or len(sibling) != 32:
            return False
        value_hash = (
            _node_hash(value_hash, sibling)
            if index % 2 == 0
            else _node_hash(sibling, value_hash)
        )
        index >>= 1
    return value_hash == commitment.root


def _plan_fingerprint(
    root: ValidatedDefinition,
    plan: ReplayPlan,
    cell_bits: int,
) -> bytes:
    if (
        plan.root_digest != root.digest
        or plan.root_gate_count != root.gate_count
        or plan.root_input_count != root.input_count
    ):
        raise ProtocolReject("replay plan does not identify this circuit")
    descriptor = {
        "cell_bits": cell_bits,
        "root": root.digest,
        "root_gate_count": plan.root_gate_count,
        "root_input_count": plan.root_input_count,
        "units": [list(unit.path) for unit in plan.units],
    }
    payload = json.dumps(
        descriptor,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).digest()


def boundary_domain(
    root: ValidatedDefinition,
    plan: ReplayPlan,
    cell_bits: int,
) -> bytes:
    return b"boundary:" + _plan_fingerprint(root, plan, cell_bits)


def unit_domain(
    root: ValidatedDefinition,
    plan: ReplayPlan,
    cell_bits: int,
    unit_index: int,
) -> bytes:
    if type(unit_index) is not int or not 0 <= unit_index < len(plan.units):
        raise ProtocolReject("unit index is out of range")
    return (
        b"unit:"
        + _plan_fingerprint(root, plan, cell_bits)
        + struct.pack(">Q", unit_index)
    )


def evaluate_assignment(
    kernel: Kernel,
    root: ValidatedDefinition,
    inputs: Sequence[int],
    *,
    overrides: Mapping[int, int] | None = None,
) -> tuple[int, ...]:
    """Evaluate a complete assignment, optionally forging selected writes.

    Later gates consume overridden values.  Consequently each override creates
    one locally incorrect gate while downstream gates can remain consistent.
    """

    if len(inputs) != root.input_count:
        raise ProtocolReject("assignment input count does not match the circuit")
    cells = list(inputs)
    for value in cells:
        _cell_payload(value, kernel.cell_bits)
    overrides = overrides or {}
    for position, value in overrides.items():
        if (
            type(position) is not int
            or position < root.input_count
            or position >= root.input_count + root.gate_count
        ):
            raise ProtocolReject("an override must name a gate output position")
        _cell_payload(value, kernel.cell_bits)

    flat = kernel.flatten(root)
    for gate in flat.gates:
        value = kernel.apply_gate(
            gate.function,
            tuple(cells[position] for position in gate.reads),
        )
        value = overrides.get(gate.write, value)
        cells.append(value)
    return tuple(cells)


@dataclass(frozen=True)
class GateOpening:
    gate_ordinal: int
    write: ValueOpening
    reads: tuple[ValueOpening, ...]


@dataclass(frozen=True)
class ProtocolCheck:
    name: str
    ok: bool
    detail: str


@dataclass
class StagedTranscript:
    boundary_commitment: ValueCommitment
    challenged_units: tuple[int, ...] = ()
    unit_commitments: dict[int, ValueCommitment] = field(default_factory=dict)
    sampled_gates: tuple[int, ...] = ()
    checks: list[ProtocolCheck] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return all(check.ok for check in self.checks)


class StagedClient:
    """Client-side commitment material for one complete claimed assignment."""

    def __init__(
        self,
        *,
        kernel: Kernel,
        root: ValidatedDefinition,
        plan: ReplayPlan,
        assignment: Sequence[int],
    ) -> None:
        expected_values = root.input_count + root.gate_count
        try:
            kernel.validate_replay_plan(root, plan)
        except KernelReject as error:
            raise ProtocolReject("replay plan does not match the circuit") from error
        if len(assignment) != expected_values:
            raise ProtocolReject("assignment length does not match the circuit")
        self.kernel = kernel
        self.root = root
        self.plan = plan
        self.assignment = tuple(assignment)
        for value in self.assignment:
            _cell_payload(value, kernel.cell_bits)
        self._boundary_tree: IndexedValueTree | None = None
        self._unit_trees: dict[int, IndexedValueTree] = {}

    @classmethod
    def honest(
        cls,
        *,
        kernel: Kernel,
        root: ValidatedDefinition,
        plan: ReplayPlan,
        inputs: Sequence[int],
    ) -> StagedClient:
        return cls(
            kernel=kernel,
            root=root,
            plan=plan,
            assignment=evaluate_assignment(kernel, root, inputs),
        )

    def commit_boundary(self) -> ValueCommitment:
        if self._boundary_tree is not None:
            raise ProtocolReject("boundary was already committed")
        values = {position: self.assignment[position] for position in self.plan.boundary}
        self._boundary_tree = IndexedValueTree.build(
            domain=boundary_domain(self.root, self.plan, self.kernel.cell_bits),
            positions=self.plan.boundary,
            values=values,
            cell_bits=self.kernel.cell_bits,
        )
        return self._boundary_tree.commitment

    def open_boundary(self, position: int) -> ValueOpening:
        if self._boundary_tree is None:
            raise ProtocolReject("boundary has not been committed")
        return self._boundary_tree.open(position)

    def commit_units(
        self,
        challenged_units: Sequence[int],
    ) -> dict[int, ValueCommitment]:
        if self._boundary_tree is None:
            raise ProtocolReject("boundary must be committed before replay units")
        unit_indices = tuple(challenged_units)
        if (
            any(type(index) is not int for index in unit_indices)
            or tuple(sorted(set(unit_indices))) != unit_indices
        ):
            raise ProtocolReject("challenged unit indices must be sorted and unique")
        commitments = {}
        for unit_index in unit_indices:
            if unit_index in self._unit_trees:
                raise ProtocolReject("a challenged unit was already committed")
            positions = self.plan.interior_positions(unit_index)
            values = {position: self.assignment[position] for position in positions}
            tree = IndexedValueTree.build(
                domain=unit_domain(
                    self.root,
                    self.plan,
                    self.kernel.cell_bits,
                    unit_index,
                ),
                positions=positions,
                values=values,
                cell_bits=self.kernel.cell_bits,
            )
            self._unit_trees[unit_index] = tree
            commitments[unit_index] = tree.commitment
        return commitments

    def _open_for_unit(self, unit_index: int, position: int) -> ValueOpening:
        if position in self.plan.boundary:
            return self.open_boundary(position)
        tree = self._unit_trees.get(unit_index)
        if tree is None:
            raise ProtocolReject("unit interior has not been committed")
        return tree.open(position)

    def open_gate(self, gate_ordinal: int) -> GateOpening:
        unit_index = self.plan.unit_index_for_gate(gate_ordinal)
        if unit_index not in self._unit_trees:
            raise ProtocolReject("sampled gate belongs to an uncommitted unit")
        gate = self.kernel.gate_at(self.root, gate_ordinal)
        return GateOpening(
            gate_ordinal=gate_ordinal,
            write=self._open_for_unit(unit_index, gate.write),
            reads=tuple(
                self._open_for_unit(unit_index, position) for position in gate.reads
            ),
        )


class StagedVerifier:
    """State machine that keeps T hidden until challenged interiors are bound."""

    _READY = "ready"
    _BOUNDARY = "boundary"
    _UNITS_REVEALED = "units_revealed"
    _INTERIORS = "interiors"
    _SAMPLE_REVEALED = "sample_revealed"

    def __init__(
        self,
        *,
        kernel: Kernel,
        root: ValidatedDefinition,
        plan: ReplayPlan,
        inputs: Sequence[int],
        claimed_outputs: Sequence[int],
    ) -> None:
        if len(inputs) != root.input_count:
            raise ProtocolReject("public input count does not match the circuit")
        try:
            kernel.validate_replay_plan(root, plan)
        except KernelReject as error:
            raise ProtocolReject("replay plan does not match the circuit") from error
        if len(claimed_outputs) != len(plan.root_outputs):
            raise ProtocolReject("claimed output count does not match the circuit")
        self.kernel = kernel
        self.root = root
        self.plan = plan
        self.inputs = tuple(inputs)
        self.claimed_outputs = tuple(claimed_outputs)
        for value in (*self.inputs, *self.claimed_outputs):
            _cell_payload(value, kernel.cell_bits)
        self.phase = self._READY
        self.boundary_commitment: ValueCommitment | None = None
        self.unit_commitments: dict[int, ValueCommitment] = {}
        self._sample: tuple[int, ...] | None = None
        self._challenged_units: tuple[int, ...] = ()
        self._sampling_mode: str | None = None
        self._checked_gates: set[int] = set()

    def receive_boundary_commitment(self, commitment: ValueCommitment) -> None:
        if self.phase != self._READY:
            raise ProtocolReject("boundary commitment arrived out of order")
        validate_commitment_domain(
            commitment,
            domain=boundary_domain(self.root, self.plan, self.kernel.cell_bits),
            positions=self.plan.boundary,
        )
        self.boundary_commitment = commitment
        self.phase = self._BOUNDARY

    def _verify_position(
        self,
        *,
        unit_index: int | None,
        position: int,
        opening: ValueOpening,
    ) -> bool:
        if position in self.plan.boundary:
            if self.boundary_commitment is None:
                return False
            return verify_value_opening(
                self.boundary_commitment,
                domain=boundary_domain(
                    self.root,
                    self.plan,
                    self.kernel.cell_bits,
                ),
                positions=self.plan.boundary,
                opening=opening,
                cell_bits=self.kernel.cell_bits,
            ) and opening.position == position
        if unit_index is None:
            return False
        commitment = self.unit_commitments.get(unit_index)
        if commitment is None:
            return False
        return verify_value_opening(
            commitment,
            domain=unit_domain(
                self.root,
                self.plan,
                self.kernel.cell_bits,
                unit_index,
            ),
            positions=self.plan.interior_positions(unit_index),
            opening=opening,
            cell_bits=self.kernel.cell_bits,
        ) and opening.position == position

    def check_public_io(
        self,
        openings: Mapping[int, ValueOpening],
    ) -> tuple[ProtocolCheck, ...]:
        if self.phase != self._BOUNDARY:
            raise ProtocolReject("public I/O must be checked after the boundary commitment")
        checks = []
        for position, expected in enumerate(self.inputs):
            opening = openings.get(position)
            authentic = opening is not None and self._verify_position(
                unit_index=None,
                position=position,
                opening=opening,
            )
            ok = authentic and opening.value == expected
            checks.append(
                ProtocolCheck(
                    name=f"input {position}",
                    ok=ok,
                    detail="matches public input" if ok else "input opening is invalid",
                )
            )
        for output_index, (position, expected) in enumerate(
            zip(self.plan.root_outputs, self.claimed_outputs)
        ):
            opening = openings.get(position)
            authentic = opening is not None and self._verify_position(
                unit_index=None,
                position=position,
                opening=opening,
            )
            ok = authentic and opening.value == expected
            checks.append(
                ProtocolCheck(
                    name=f"output {output_index}",
                    ok=ok,
                    detail="matches claimed output" if ok else "output opening is invalid",
                )
            )
        return tuple(checks)

    def sample_gates(
        self,
        probability: float,
        *,
        rng: random.Random | None = None,
    ) -> tuple[int, ...]:
        if (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not 0 <= probability <= 1
        ):
            raise ProtocolReject("sampling probability must lie in [0, 1]")
        source = rng if rng is not None else random.SystemRandom()
        sample = tuple(
            ordinal
            for ordinal in range(self.root.gate_count)
            if source.random() < probability
        )
        return self.lock_sample(sample)

    def sample_replay_units(
        self,
        probability: float,
        *,
        rng: random.Random | None = None,
    ) -> tuple[int, ...]:
        """First stage of q,s sampling: select replay units with probability q."""

        if (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not 0 <= probability <= 1
        ):
            raise ProtocolReject("replay probability must lie in [0, 1]")
        source = rng if rng is not None else random.SystemRandom()
        challenged_units = tuple(
            unit_index
            for unit_index in range(len(self.plan.units))
            if source.random() < probability
        )
        return self.lock_replay_units(challenged_units)

    def lock_replay_units(
        self,
        challenged_units: Sequence[int],
    ) -> tuple[int, ...]:
        """Fix and reveal J before any selected-unit interior commitment."""

        if self.phase != self._BOUNDARY:
            raise ProtocolReject(
                "replay units must be sampled after the boundary commitment"
            )
        unit_indices = tuple(challenged_units)
        if (
            any(
                type(unit_index) is not int
                or unit_index < 0
                or unit_index >= len(self.plan.units)
                for unit_index in unit_indices
            )
            or tuple(sorted(set(unit_indices))) != unit_indices
        ):
            raise ProtocolReject(
                "challenged unit indices must be sorted, unique, and in range"
            )
        self._sample = None
        self._challenged_units = unit_indices
        self._sampling_mode = "two_stage"
        self.phase = self._UNITS_REVEALED
        return self._challenged_units

    def lock_sample(
        self,
        sampled_gates: Sequence[int],
    ) -> tuple[int, ...]:
        """Pre-sample gates and reveal only their owning units (Draft-v4 mode)."""

        if self.phase != self._BOUNDARY:
            raise ProtocolReject("gate sample must be fixed after the boundary commitment")
        sample = tuple(sampled_gates)
        if (
            any(
                type(ordinal) is not int
                or ordinal < 0
                or ordinal >= self.root.gate_count
                for ordinal in sample
            )
            or tuple(sorted(set(sample))) != sample
        ):
            raise ProtocolReject("sampled gates must be sorted, unique, and in range")
        self._sample = sample
        self._challenged_units = self.plan.challenged_unit_indices(sample)
        self._sampling_mode = "derived_units"
        self.phase = self._UNITS_REVEALED
        return self._challenged_units

    def receive_unit_commitments(
        self,
        commitments: Mapping[int, ValueCommitment],
    ) -> None:
        if self.phase != self._UNITS_REVEALED:
            raise ProtocolReject("unit commitments arrived out of order")
        if set(commitments) != set(self._challenged_units):
            raise ProtocolReject("unit commitments do not exactly match challenged units")
        for unit_index in self._challenged_units:
            validate_commitment_domain(
                commitments[unit_index],
                domain=unit_domain(
                    self.root,
                    self.plan,
                    self.kernel.cell_bits,
                    unit_index,
                ),
                positions=self.plan.interior_positions(unit_index),
            )
        self.unit_commitments = dict(commitments)
        self.phase = self._INTERIORS

    def reveal_sample(self) -> tuple[int, ...]:
        if self.phase != self._INTERIORS:
            raise ProtocolReject("sample cannot be revealed before unit commitments")
        if self._sampling_mode != "derived_units":
            raise ProtocolReject(
                "two-stage sampling must draw gates after unit commitments"
            )
        assert self._sample is not None
        self.phase = self._SAMPLE_REVEALED
        return self._sample

    def sample_within_replay_units(
        self,
        probability: float,
        *,
        rng: random.Random | None = None,
    ) -> tuple[int, ...]:
        """Second stage of q,s sampling: sample gates after interiors are bound."""

        if (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not 0 <= probability <= 1
        ):
            raise ProtocolReject("within-unit probability must lie in [0, 1]")
        source = rng if rng is not None else random.SystemRandom()
        sample = tuple(
            gate_ordinal
            for unit_index in self._challenged_units
            for gate_ordinal in range(
                self.plan.units[unit_index].gate_start,
                self.plan.units[unit_index].gate_stop,
            )
            if source.random() < probability
        )
        return self.lock_within_replay_unit_sample(sample)

    def lock_within_replay_unit_sample(
        self,
        sampled_gates: Sequence[int],
    ) -> tuple[int, ...]:
        """Fix and reveal T after all roots for the already-revealed J exist."""

        if self.phase != self._INTERIORS:
            raise ProtocolReject(
                "within-unit gates must be sampled after unit commitments"
            )
        if self._sampling_mode != "two_stage":
            raise ProtocolReject("replay units were not selected in two-stage mode")
        sample = tuple(sampled_gates)
        challenged = set(self._challenged_units)
        if (
            any(
                type(gate_ordinal) is not int
                or gate_ordinal < 0
                or gate_ordinal >= self.root.gate_count
                or self.plan.unit_index_for_gate(gate_ordinal) not in challenged
                for gate_ordinal in sample
            )
            or tuple(sorted(set(sample))) != sample
        ):
            raise ProtocolReject(
                "sampled gates must be sorted, unique, and inside challenged units"
            )
        self._sample = sample
        self.phase = self._SAMPLE_REVEALED
        return sample

    def check_gate(self, opening: GateOpening) -> ProtocolCheck:
        if self.phase != self._SAMPLE_REVEALED:
            raise ProtocolReject("gate checks require the revealed sample")
        if (
            not isinstance(opening, GateOpening)
            or type(opening.gate_ordinal) is not int
            or type(opening.reads) is not tuple
            or self._sample is None
            or opening.gate_ordinal not in self._sample
        ):
            return ProtocolCheck("gate", False, "opening is not for a sampled gate")
        if opening.gate_ordinal in self._checked_gates:
            return ProtocolCheck(
                f"gate {opening.gate_ordinal}",
                False,
                "sampled gate was checked more than once",
            )
        self._checked_gates.add(opening.gate_ordinal)
        gate = self.kernel.gate_at(self.root, opening.gate_ordinal)
        unit_index = self.plan.unit_index_for_gate(opening.gate_ordinal)
        if len(opening.reads) != len(gate.reads):
            return ProtocolCheck(
                f"gate {opening.gate_ordinal}",
                False,
                "wrong number of read openings",
            )
        if not self._verify_position(
            unit_index=unit_index,
            position=gate.write,
            opening=opening.write,
        ):
            return ProtocolCheck(
                f"gate {opening.gate_ordinal}",
                False,
                "write opening is invalid",
            )
        args = []
        unit = self.plan.units[unit_index]
        for position, read_opening in zip(gate.reads, opening.reads):
            if position not in self.plan.boundary:
                producer_ordinal = position - self.root.input_count
                if not unit.gate_start <= producer_ordinal < unit.gate_stop:
                    raise RuntimeError(
                        "compiler boundary omitted a cross-unit gate read"
                    )
            if not self._verify_position(
                unit_index=unit_index,
                position=position,
                opening=read_opening,
            ):
                return ProtocolCheck(
                    f"gate {opening.gate_ordinal}",
                    False,
                    f"read opening for position {position} is invalid",
                )
            args.append(read_opening.value)
        expected = self.kernel.apply_gate(gate.function, args)
        ok = opening.write.value == expected
        return ProtocolCheck(
            name=f"gate {opening.gate_ordinal}",
            ok=ok,
            detail="local gate equation holds" if ok else "local gate equation fails",
        )

    def finalize(self) -> ProtocolCheck:
        """Require one check attempt for every gate in the revealed sample."""

        if self.phase != self._SAMPLE_REVEALED or self._sample is None:
            raise ProtocolReject("protocol cannot finish before the gate sample")
        expected = set(self._sample)
        ok = self._checked_gates == expected
        return ProtocolCheck(
            name="sample coverage",
            ok=ok,
            detail="every sampled gate was checked exactly once"
            if ok
            else "sampled gate checks were missing or duplicated",
        )


def public_io_positions(plan: ReplayPlan) -> tuple[int, ...]:
    return tuple(sorted(set(range(plan.root_input_count)) | set(plan.root_outputs)))


def two_stage_acceptance_probability(
    incorrect_gates_per_unit: Sequence[int],
    *,
    replay_probability: float,
    within_unit_probability: float,
) -> float:
    """Exact q,s survival probability for fixed per-unit error counts."""

    for name, probability in (
        ("replay", replay_probability),
        ("within-unit", within_unit_probability),
    ):
        if (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not 0 <= probability <= 1
        ):
            raise ProtocolReject(f"{name} probability must lie in [0, 1]")
    counts = tuple(incorrect_gates_per_unit)
    if any(type(count) is not int or count < 0 for count in counts):
        raise ProtocolReject("incorrect-gate counts must be nonnegative integers")
    q = float(replay_probability)
    s = float(within_unit_probability)
    probability = 1.0
    for count in counts:
        probability *= 1.0 - q + q * (1.0 - s) ** count
    return probability


def _commit_boundary_and_check_io(
    client: StagedClient,
    verifier: StagedVerifier,
) -> StagedTranscript:
    boundary_commitment = client.commit_boundary()
    verifier.receive_boundary_commitment(boundary_commitment)
    io_positions = public_io_positions(verifier.plan)
    io_openings = {
        position: client.open_boundary(position) for position in io_positions
    }
    transcript = StagedTranscript(boundary_commitment=boundary_commitment)
    transcript.checks.extend(verifier.check_public_io(io_openings))
    return transcript


def run_staged_protocol(
    client: StagedClient,
    verifier: StagedVerifier,
    *,
    sampled_gates: Sequence[int],
) -> StagedTranscript:
    """Drive one run while exposing only J before interior commitments."""

    transcript = _commit_boundary_and_check_io(client, verifier)

    challenged_units = verifier.lock_sample(sampled_gates)
    transcript.challenged_units = challenged_units
    unit_commitments = client.commit_units(challenged_units)
    verifier.receive_unit_commitments(unit_commitments)
    transcript.unit_commitments = unit_commitments

    revealed_sample = verifier.reveal_sample()
    transcript.sampled_gates = revealed_sample
    for gate_ordinal in revealed_sample:
        transcript.checks.append(
            verifier.check_gate(client.open_gate(gate_ordinal))
        )
    transcript.checks.append(verifier.finalize())
    return transcript


def run_two_stage_protocol(
    client: StagedClient,
    verifier: StagedVerifier,
    *,
    challenged_units: Sequence[int],
    sampled_gates: Sequence[int],
) -> StagedTranscript:
    """Drive boundary -> q-stage units -> interiors -> s-stage gates."""

    transcript = _commit_boundary_and_check_io(client, verifier)

    revealed_units = verifier.lock_replay_units(challenged_units)
    transcript.challenged_units = revealed_units
    unit_commitments = client.commit_units(revealed_units)
    verifier.receive_unit_commitments(unit_commitments)
    transcript.unit_commitments = unit_commitments

    revealed_sample = verifier.lock_within_replay_unit_sample(sampled_gates)
    transcript.sampled_gates = revealed_sample
    for gate_ordinal in revealed_sample:
        transcript.checks.append(
            verifier.check_gate(client.open_gate(gate_ordinal))
        )
    transcript.checks.append(verifier.finalize())
    return transcript
