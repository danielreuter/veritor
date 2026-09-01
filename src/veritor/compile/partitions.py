"""Deterministic occurrence-based partitions of a call-DAG circuit."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

from veritor.core import (
    CompiledArtifact,
    InvalidArtifact,
    JSONValue,
    RangeIndexedDomain,
    ReplayPartition,
    ReplayUnit,
    VerificationPartition,
    VerificationUnit,
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


DEFAULT_REPLAY_POLICY = PartitionPolicy.POSITIVE_TOP_LEVEL_OCCURRENCES
DEFAULT_VERIFICATION_POLICY = PartitionPolicy.SINGLETON_GATES


def _checked_policy(policy: PartitionPolicy | str) -> PartitionPolicy:
    try:
        return PartitionPolicy(policy)
    except (TypeError, ValueError) as error:
        raise InvalidArtifact(f"unknown call-DAG partition policy {policy!r}") from error


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


def _identity_configuration(
    policy: PartitionPolicy,
    configuration: JSONValue | None,
) -> dict[str, JSONValue]:
    return {
        "configuration": {} if configuration is None else configuration,
        "policy": policy.value,
        "policy_version": PARTITION_POLICY_VERSION,
    }


def _plan(circuit: CallDagCircuit, paths: Sequence[OccurrencePath]) -> ReplayPlan:
    if not isinstance(circuit, CallDagCircuit):
        raise TypeError("occurrence partitions require a CallDagCircuit")
    plan = circuit.kernel.derive_replay_plan(circuit.root, paths)
    circuit.kernel.validate_replay_plan(circuit.root, plan)
    return plan


def replay_partition_from_occurrences(
    circuit: CallDagCircuit,
    occurrence_paths: Sequence[OccurrencePath],
    *,
    algorithm_id: str = REPLAY_PARTITION_ALGORITHM_ID,
    algorithm_version: str = PARTITION_POLICY_VERSION,
    configuration: JSONValue | None = None,
) -> tuple[ReplayPartition, ReplayPlan]:
    """Build an exact replay partition (and its plan) from an occurrence cut."""

    plan = _plan(circuit, occurrence_paths)
    base = circuit.input_count
    units = tuple(
        ReplayUnit(
            index,
            RangeIndexedDomain(base + summary.gate_start, base + summary.gate_stop),
            replay_cost=summary.cost,
        )
        for index, summary in enumerate(plan.units)
    )
    partition = ReplayPartition(
        circuit.identity,
        circuit.computed_positions,
        units,
        algorithm_id=algorithm_id,
        algorithm_version=algorithm_version,
        configuration=configuration,
    )
    return partition, plan


def verification_partition_from_occurrences(
    circuit: CallDagCircuit,
    replay_partition: ReplayPartition,
    occurrence_paths: Sequence[OccurrencePath],
    *,
    algorithm_id: str = VERIFICATION_PARTITION_ALGORITHM_ID,
    algorithm_version: str = PARTITION_POLICY_VERSION,
    configuration: JSONValue | None = None,
) -> VerificationPartition:
    """Build an exact verification refinement from an occurrence cut."""

    if not isinstance(replay_partition, ReplayPartition):
        raise InvalidArtifact("verification derivation requires a ReplayPartition")
    if replay_partition.structure_identity != circuit.identity:
        raise InvalidArtifact("replay partition belongs to another structure")
    plan = _plan(circuit, occurrence_paths)
    base = circuit.input_count
    units: list[VerificationUnit] = []
    for index, summary in enumerate(plan.units):
        start = base + summary.gate_start
        stop = base + summary.gate_stop
        units.append(
            VerificationUnit(
                index,
                replay_partition.owner_of(start),
                RangeIndexedDomain(start, stop),
                proof_relation_id=(
                    circuit.executable_gate_at(start).relation_id
                    if stop - start == 1
                    else None
                ),
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


def compile_partitions(
    circuit: CallDagCircuit,
    replay_paths: Sequence[OccurrencePath],
    verification_paths: Sequence[OccurrencePath],
    *,
    replay_algorithm_id: str = REPLAY_PARTITION_ALGORITHM_ID,
    verification_algorithm_id: str = VERIFICATION_PARTITION_ALGORITHM_ID,
    replay_configuration: JSONValue | None = None,
    verification_configuration: JSONValue | None = None,
) -> CompiledArtifact:
    """Assemble the compiled artifact from two occurrence cuts.

    The replay boundary comes from the kernel's occurrence interfaces, so no
    gate is visited here.
    """

    replay, plan = replay_partition_from_occurrences(
        circuit,
        replay_paths,
        algorithm_id=replay_algorithm_id,
        configuration=replay_configuration,
    )
    verification = verification_partition_from_occurrences(
        circuit,
        replay,
        verification_paths,
        algorithm_id=verification_algorithm_id,
        configuration=verification_configuration,
    )
    return CompiledArtifact(circuit, replay, verification, plan.boundary)


def compile_partitions_for_policies(
    circuit: CallDagCircuit,
    replay_policy: PartitionPolicy | str = DEFAULT_REPLAY_POLICY,
    verification_policy: PartitionPolicy | str = DEFAULT_VERIFICATION_POLICY,
    *,
    replay_configuration: JSONValue | None = None,
    verification_configuration: JSONValue | None = None,
) -> CompiledArtifact:
    """Assemble the compiled artifact from two named partition policies."""

    replay_selected = _checked_policy(replay_policy)
    verification_selected = _checked_policy(verification_policy)
    return compile_partitions(
        circuit,
        occurrence_paths_for_policy(circuit, replay_selected),
        occurrence_paths_for_policy(circuit, verification_selected),
        replay_configuration=_identity_configuration(
            replay_selected, replay_configuration
        ),
        verification_configuration=_identity_configuration(
            verification_selected, verification_configuration
        ),
    )
