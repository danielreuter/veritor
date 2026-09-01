"""Deterministic partitions and verifier-derived replay boundaries."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

from veritor.core import (
    InvalidArtifact,
    JSONValue,
    RangeIndexedDomain,
    ReplayPartition,
    ReplayUnit,
    StructuralCircuit,
    StructuralGate,
    VerificationPartition,
    VerificationUnit,
    domains_equal,
    iter_domain,
    ordered_output_positions,
    validate_circuit_contract,
)

from .call_dag import CallDagCircuit, KernelReject, OccurrencePath, ReplayPlan

PARTITION_POLICY_VERSION = "1"
REPLAY_PARTITION_ALGORITHM_ID = "veritor.compile.call-dag.replay-partition"
VERIFICATION_PARTITION_ALGORITHM_ID = "veritor.compile.call-dag.verification-partition"


class PartitionPolicy(StrEnum):
    """Supported deterministic occurrence-based partition granularities."""

    WHOLE_ROOT = "whole-root"
    POSITIVE_TOP_LEVEL_OCCURRENCES = "positive-top-level-occurrences"
    SINGLETON_GATES = "singleton-gates"

    # Concise aliases remain useful at call sites.
    POSITIVE_TOP_LEVEL = "positive-top-level-occurrences"
    SINGLETON = "singleton-gates"


DEFAULT_REPLAY_POLICY = PartitionPolicy.POSITIVE_TOP_LEVEL_OCCURRENCES
DEFAULT_VERIFICATION_POLICY = PartitionPolicy.SINGLETON_GATES

# Descriptive constants form a stable, discoverable policy surface.
WHOLE_ROOT = PartitionPolicy.WHOLE_ROOT
POSITIVE_TOP_LEVEL_OCCURRENCES = PartitionPolicy.POSITIVE_TOP_LEVEL_OCCURRENCES
SINGLETON_GATES = PartitionPolicy.SINGLETON_GATES


def _checked_policy(policy: PartitionPolicy | str) -> PartitionPolicy:
    try:
        return PartitionPolicy(policy)
    except (TypeError, ValueError) as error:
        raise InvalidArtifact(
            f"unknown call-DAG partition policy {policy!r}"
        ) from error


def occurrence_paths_for_policy(
    circuit: CallDagCircuit,
    policy: PartitionPolicy | str,
) -> tuple[OccurrencePath, ...]:
    """Derive one deterministic exact occurrence cut."""

    if not isinstance(circuit, CallDagCircuit):
        raise TypeError("occurrence policies require a CallDagCircuit")
    selected = _checked_policy(policy)
    root = circuit.root
    kernel = circuit.kernel
    if root.gate_count == 0:
        return ()
    if selected is PartitionPolicy.WHOLE_ROOT:
        paths: tuple[OccurrencePath, ...] = ((),)
    elif selected is PartitionPolicy.POSITIVE_TOP_LEVEL_OCCURRENCES:
        paths = tuple(
            (step_index,)
            for step_index in range(len(root.steps))
            if kernel.occurrence_summary(root, (step_index,)).gate_count > 0
        )
    else:
        if root.gate_count > kernel.limits.max_partition_units:
            raise KernelReject(
                "singleton partition needs "
                f"{root.gate_count} units, exceeding configured limit "
                f"{kernel.limits.max_partition_units}"
            )
        paths = kernel.leaf_occurrence_paths(root)
    if len(paths) > kernel.limits.max_partition_units:
        raise KernelReject(
            f"partition needs {len(paths)} units, exceeding configured limit "
            f"{kernel.limits.max_partition_units}"
        )
    return paths


def derive_occurrence_replay_plan(
    circuit: CallDagCircuit,
    policy: PartitionPolicy | str = DEFAULT_REPLAY_POLICY,
) -> ReplayPlan:
    """Derive and fully revalidate the kernel's occurrence-relative plan."""

    paths = occurrence_paths_for_policy(circuit, policy)
    plan = circuit.kernel.derive_replay_plan(circuit.root, paths)
    circuit.kernel.validate_replay_plan(circuit.root, plan)
    return plan


def _identity_configuration(
    policy: PartitionPolicy,
    configuration: JSONValue | None,
) -> dict[str, JSONValue]:
    return {
        "configuration": {} if configuration is None else configuration,
        "policy": policy.value,
        "policy_version": PARTITION_POLICY_VERSION,
    }


def derive_replay_partition(
    circuit: CallDagCircuit,
    policy: PartitionPolicy | str = DEFAULT_REPLAY_POLICY,
    *,
    configuration: JSONValue | None = None,
) -> ReplayPartition:
    """Derive a core exact-cover replay partition from an occurrence policy."""

    selected = _checked_policy(policy)
    return derive_replay_partition_from_occurrences(
        circuit,
        occurrence_paths_for_policy(circuit, selected),
        algorithm_id=REPLAY_PARTITION_ALGORITHM_ID,
        algorithm_version=PARTITION_POLICY_VERSION,
        configuration=_identity_configuration(selected, configuration),
    )


def derive_replay_partition_from_occurrences(
    circuit: CallDagCircuit,
    occurrence_paths: Sequence[OccurrencePath],
    *,
    algorithm_id: str,
    algorithm_version: str = PARTITION_POLICY_VERSION,
    configuration: JSONValue | None = None,
) -> ReplayPartition:
    """Build an exact replay partition from a validated occurrence cut."""

    if not isinstance(circuit, CallDagCircuit):
        raise TypeError("occurrence partitions require a CallDagCircuit")
    plan = circuit.kernel.derive_replay_plan(circuit.root, occurrence_paths)
    circuit.kernel.validate_replay_plan(circuit.root, plan)
    units = tuple(
        ReplayUnit(
            index,
            RangeIndexedDomain(
                circuit.input_count + summary.gate_start,
                circuit.input_count + summary.gate_stop,
            ),
            replay_cost=summary.cost,
        )
        for index, summary in enumerate(plan.units)
    )
    return ReplayPartition(
        circuit.identity,
        circuit.computed_positions,
        units,
        algorithm_id=algorithm_id,
        algorithm_version=algorithm_version,
        configuration=configuration,
    )


def derive_verification_partition(
    circuit: CallDagCircuit,
    replay_partition: ReplayPartition,
    policy: PartitionPolicy | str = DEFAULT_VERIFICATION_POLICY,
    *,
    configuration: JSONValue | None = None,
) -> VerificationPartition:
    """Derive an exact verification refinement of ``replay_partition``."""

    selected = _checked_policy(policy)
    return derive_verification_partition_from_occurrences(
        circuit,
        replay_partition,
        occurrence_paths_for_policy(circuit, selected),
        algorithm_id=VERIFICATION_PARTITION_ALGORITHM_ID,
        algorithm_version=PARTITION_POLICY_VERSION,
        configuration=_identity_configuration(selected, configuration),
    )


def derive_verification_partition_from_occurrences(
    circuit: CallDagCircuit,
    replay_partition: ReplayPartition,
    occurrence_paths: Sequence[OccurrencePath],
    *,
    algorithm_id: str,
    algorithm_version: str = PARTITION_POLICY_VERSION,
    configuration: JSONValue | None = None,
) -> VerificationPartition:
    """Build an exact verification refinement from an occurrence cut."""

    if not isinstance(circuit, CallDagCircuit):
        raise TypeError("occurrence partitions require a CallDagCircuit")
    if not isinstance(replay_partition, ReplayPartition):
        raise InvalidArtifact("verification derivation requires a ReplayPartition")
    if replay_partition.structure_identity != circuit.identity:
        raise InvalidArtifact("replay partition belongs to another structure")
    plan = circuit.kernel.derive_replay_plan(circuit.root, occurrence_paths)
    circuit.kernel.validate_replay_plan(circuit.root, plan)
    units: list[VerificationUnit] = []
    for index, summary in enumerate(plan.units):
        start = circuit.input_count + summary.gate_start
        stop = circuit.input_count + summary.gate_stop
        members = RangeIndexedDomain(start, stop)
        replay_owner = replay_partition.owner_of(start)
        if any(
            replay_partition.owner_of(position) != replay_owner
            for position in iter_domain(members)
        ):
            raise InvalidArtifact("verification occurrence cut crosses replay units")
        proof_relation_id = (
            circuit.executable_gate_at(start).relation_id
            if members.count == 1
            else None
        )
        units.append(
            VerificationUnit(
                index,
                replay_owner,
                members,
                proof_relation_id=proof_relation_id,
            )
        )
    return VerificationPartition(
        circuit.identity,
        replay_partition,
        circuit.computed_positions,
        units,
        algorithm_id=algorithm_id,
        algorithm_version=algorithm_version,
        configuration=configuration,
    )


def derive_replay_boundary(
    circuit: StructuralCircuit,
    replay_partition: ReplayPartition,
) -> tuple[int, ...]:
    """Derive the exact finite replay boundary from verifier-visible contracts.

    The result contains every input position, the set of ordered output
    positions, and each computed predecessor whose replay owner differs from
    the owner of the gate that reads it.
    """

    try:
        validate_circuit_contract(circuit, exhaustive=False)
    except (AttributeError, TypeError) as error:
        raise InvalidArtifact(
            "boundary derivation requires a structural circuit"
        ) from error
    if not isinstance(replay_partition, ReplayPartition):
        raise InvalidArtifact("boundary derivation requires a ReplayPartition")
    if replay_partition.structure_identity != circuit.identity:
        raise InvalidArtifact("replay partition belongs to another structure")
    if not domains_equal(
        circuit.computed_positions,
        replay_partition.eligible_positions,
    ):
        raise InvalidArtifact(
            "replay partition does not cover the circuit's computed positions"
        )
    replay_partition.validate()

    input_positions = {port.position for port in circuit.input_ports}
    boundary = set(input_positions)
    boundary.update(ordered_output_positions(circuit))
    computed = circuit.computed_positions
    owner_by_position = {
        position: unit.index
        for unit in replay_partition.units
        for position in iter_domain(unit.members)
    }
    for position in iter_domain(computed):
        gate = circuit.gate_at(position)
        if not isinstance(gate, StructuralGate) or gate.position != position:
            raise InvalidArtifact("gate_at returned a malformed structural gate")
        consumer_owner = owner_by_position[position]
        for predecessor in gate.predecessors:
            if predecessor in input_positions:
                boundary.add(predecessor)
            elif computed.contains(predecessor):
                if owner_by_position[predecessor] != consumer_owner:
                    boundary.add(predecessor)
            else:
                raise InvalidArtifact(
                    f"gate at position {position} references unknown position "
                    f"{predecessor}"
                )
    return tuple(sorted(boundary))
