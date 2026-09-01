"""Validation for the literal compiled result tuple."""

from __future__ import annotations

from .circuit import StructuralCircuit, validate_circuit_contract
from .errors import InvalidArtifact
from .identity import CompiledResultIdentity
from .indexed import domains_equal
from .partitions import (
    ReplayPartition,
    VerificationPartition,
    validate_verification_refines_replay,
)


def validate_compiled_result(
    circuit: StructuralCircuit,
    replay_partition: ReplayPartition,
    verification_partition: VerificationPartition,
) -> CompiledResultIdentity:
    """Validate ``(C, R, V)`` and return its canonical tuple identity.

    The objects themselves remain an ordinary three-item tuple at API
    boundaries; this function deliberately introduces no result wrapper.
    """

    if not isinstance(replay_partition, ReplayPartition):
        raise InvalidArtifact("compiled replay value is not a ReplayPartition")
    if not isinstance(verification_partition, VerificationPartition):
        raise InvalidArtifact(
            "compiled verification value is not a VerificationPartition"
        )
    try:
        validate_circuit_contract(circuit, exhaustive=False)
    except (AttributeError, TypeError) as error:
        raise InvalidArtifact(
            "compiled circuit does not satisfy the circuit contract"
        ) from error
    structure_identity = circuit.identity
    if replay_partition.structure_identity != structure_identity:
        raise InvalidArtifact("replay partition belongs to another structure")
    if verification_partition.structure_identity != structure_identity:
        raise InvalidArtifact("verification partition belongs to another structure")
    if replay_partition.identity.structure_digest != structure_identity.digest:
        raise InvalidArtifact("replay partition identity names another structure")
    if verification_partition.identity.structure_digest != structure_identity.digest:
        raise InvalidArtifact("verification partition identity names another structure")
    if not domains_equal(
        circuit.computed_positions,
        replay_partition.eligible_positions,
    ):
        raise InvalidArtifact(
            "replay partition does not cover the circuit's computed positions"
        )
    if not domains_equal(
        circuit.computed_positions,
        verification_partition.eligible_positions,
    ):
        raise InvalidArtifact(
            "verification partition does not cover the circuit's computed positions"
        )
    replay_partition.validate()
    verification_partition.validate(replay_partition)
    validate_verification_refines_replay(
        replay_partition,
        verification_partition,
    )
    return CompiledResultIdentity.from_components(
        structure_identity,
        replay_partition.identity,
        verification_partition.identity,
    )


compiled_result_identity = validate_compiled_result
