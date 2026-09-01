from dataclasses import dataclass

import pytest

from veritor import core
from veritor.core import (
    ArtifactKind,
    CompiledResultIdentity,
    ExplicitIndexedDomain,
    InvalidArtifact,
    Port,
    RangeIndexedDomain,
    ReplayPartition,
    ReplayUnit,
    StructuralGate,
    StructureIdentity,
    VerificationPartition,
    VerificationUnit,
    identity_digest,
    validate_compiled_result,
)


def structure(label="main"):
    return StructureIdentity(
        schema_version="1",
        artifact_kind=ArtifactKind.EXECUTABLE_CIRCUIT,
        compiler_id="tests.compiler",
        compiler_version="1",
        semantic_scope_id="tiny",
        representation_digest=identity_digest(
            "tests/structure-representation", {"label": label}
        ),
        value_registry_digest=identity_digest("tests/values", {"bits": 8}),
        operator_registry_digest=identity_digest("tests/operators", {"version": 1}),
    )


@dataclass(frozen=True, slots=True)
class ContractCircuit:
    identity: StructureIdentity
    computed_positions: object
    input_ports: tuple[Port, ...]
    output_ports: tuple[Port, ...]
    gates: tuple[StructuralGate, ...]

    def gate_at(self, position):
        return self.gates[self.computed_positions.rank(position)]


def circuit(identity=None, computed=None):
    identity = structure() if identity is None else identity
    computed = RangeIndexedDomain(1, 5) if computed is None else computed
    gates = tuple(
        StructuralGate(position, "copy", (position - 1,), 256)
        for position in range(1, 5)
    )
    return ContractCircuit(
        identity,
        computed,
        (Port("x", 0, "u8"),),
        (Port("y", 4, "u8"), Port("y_again", 4, "u8")),
        gates,
    )


def partitions(identity=None, *, replay_configuration=None):
    identity = structure() if identity is None else identity
    eligible = RangeIndexedDomain(1, 5)
    replay = ReplayPartition(
        identity,
        eligible,
        (
            ReplayUnit(0, (1, 2), replay_cost=3, replay_relation_id="replay.v1"),
            ReplayUnit(1, RangeIndexedDomain(3, 5), replay_cost=5),
        ),
        algorithm_id="tests.replay",
        configuration=replay_configuration,
    )
    verification = VerificationPartition(
        identity,
        replay,
        ExplicitIndexedDomain((1, 2, 3, 4)),
        (
            VerificationUnit(0, 0, (1,), "check.copy"),
            VerificationUnit(1, 0, (2,), "check.copy"),
            VerificationUnit(2, 1, (3, 4), "check.copy-pair"),
        ),
        algorithm_id="tests.verification",
    )
    return replay, verification


def test_partitions_validate_exact_cover_ownership_and_refinement():
    identity = structure()
    replay, verification = partitions(identity)

    assert replay.unit_count == 2
    assert replay.owner_of(1) == 0
    assert replay.owner_of(4) == 1
    assert verification.owner_of(3) == 2
    assert verification.units_in_replay_unit(0).items == (0, 1)
    assert verification.units_in_replay_unit(1).items == (2,)
    with pytest.raises(KeyError):
        verification.units_in_replay_unit(2)
    assert replay.identity.structure_digest == identity.digest
    assert verification.replay_partition_identity == replay.identity


def test_partition_identities_are_deterministic_and_bind_configuration():
    identity = structure()
    first_replay, first_verification = partitions(
        identity, replay_configuration={"coarsening": 2}
    )
    second_replay, second_verification = partitions(
        identity, replay_configuration={"coarsening": 2}
    )
    other_replay, _ = partitions(identity, replay_configuration={"coarsening": 3})

    assert first_replay.identity == second_replay.identity
    assert first_verification.identity == second_verification.identity
    assert first_replay.identity != other_replay.identity


@pytest.mark.parametrize(
    "units",
    [
        (ReplayUnit(0, (1, 2)), ReplayUnit(1, (2, 3, 4))),
        (ReplayUnit(0, (1,)), ReplayUnit(1, (3, 4))),
        (ReplayUnit(0, (1, 2)), ReplayUnit(1, (3, 4, 5))),
        (ReplayUnit(1, (1, 2)), ReplayUnit(0, (3, 4))),
    ],
)
def test_replay_partition_rejects_overlap_gap_ineligible_and_bad_order(units):
    with pytest.raises(InvalidArtifact):
        ReplayPartition(structure(), RangeIndexedDomain(1, 5), units)


def test_units_reject_empty_members_and_boolean_indices():
    with pytest.raises(InvalidArtifact, match="must not be empty"):
        ReplayUnit(0, ())
    with pytest.raises(InvalidArtifact):
        VerificationUnit(True, 0, (1,))
    with pytest.raises(InvalidArtifact):
        VerificationUnit(0, True, (1,))


def test_verification_partition_rejects_units_crossing_replay_units():
    identity = structure()
    replay, _ = partitions(identity)

    with pytest.raises(InvalidArtifact, match="crosses replay units"):
        VerificationPartition(
            identity,
            replay,
            RangeIndexedDomain(1, 5),
            (
                VerificationUnit(0, 0, (1, 2, 3)),
                VerificationUnit(1, 1, (4,)),
            ),
        )


def test_zero_computed_positions_require_and_support_zero_units():
    identity = structure("empty")
    empty = RangeIndexedDomain(0)
    replay = ReplayPartition(identity, empty, ())
    verification = VerificationPartition(identity, replay, empty, ())
    empty_circuit = ContractCircuit(
        identity,
        empty,
        (Port("x", 0, "u8"),),
        (Port("y", 0, "u8"),),
        (),
    )

    result_identity = validate_compiled_result(
        empty_circuit,
        replay,
        verification,
    )

    assert replay.unit_count == 0
    assert verification.unit_count == 0
    assert isinstance(result_identity, CompiledResultIdentity)
    with pytest.raises(InvalidArtifact, match="zero units"):
        ReplayPartition(identity, empty, (ReplayUnit(0, (0,)),))


def test_validate_compiled_result_returns_identity_for_literal_components():
    identity = structure()
    replay, verification = partitions(identity)

    result = validate_compiled_result(circuit(identity), replay, verification)

    assert isinstance(result, CompiledResultIdentity)
    assert result.structure_digest == identity.digest
    assert result.replay_partition_digest == replay.identity.digest
    assert result.verification_partition_digest == verification.identity.digest
    assert not hasattr(core, "Gamma")


def test_validate_compiled_result_rejects_mixed_structure_identities():
    circuit_identity = structure("circuit")
    partition_identity = structure("partition")
    replay, verification = partitions(partition_identity)

    with pytest.raises(InvalidArtifact, match="another structure"):
        validate_compiled_result(
            circuit(circuit_identity),
            replay,
            verification,
        )


def test_validate_compiled_result_rejects_verification_for_another_replay():
    identity = structure()
    original_replay, verification = partitions(
        identity, replay_configuration={"version": 1}
    )
    replacement_replay, _ = partitions(identity, replay_configuration={"version": 2})

    assert original_replay.identity != replacement_replay.identity
    with pytest.raises(InvalidArtifact, match="different replay partition"):
        validate_compiled_result(
            circuit(identity),
            replacement_replay,
            verification,
        )
