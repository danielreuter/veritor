from dataclasses import dataclass

import pytest

from veritor.core import (
    ArtifactKind,
    ExecutableGate,
    InvalidArtifact,
    Port,
    RangeIndexedDomain,
    StructuralGate,
    StructureIdentity,
    identity_digest,
    ordered_output_positions,
    validate_circuit_contract,
)


def structure(kind=ArtifactKind.EXECUTABLE_CIRCUIT):
    return StructureIdentity(
        schema_version="1",
        artifact_kind=kind,
        compiler_id="tests.compiler",
        compiler_version="1",
        semantic_scope_id="tiny-word-circuit",
        representation_digest=identity_digest("tests/circuit", {"kind": kind.value}),
        value_registry_digest=identity_digest("tests/values", {"bits": 8}),
        operator_registry_digest=identity_digest(
            "tests/operators", {"operations": ["copy", "add"]}
        ),
    )


@dataclass(frozen=True, slots=True)
class TinyCircuit:
    identity: StructureIdentity
    computed_positions: RangeIndexedDomain
    input_ports: tuple[Port, ...]
    output_ports: tuple[Port, ...]
    gates: tuple[StructuralGate, ...]

    def gate_at(self, position):
        return self.gates[self.computed_positions.rank(position)]


def tiny_circuit():
    return TinyCircuit(
        identity=structure(),
        computed_positions=RangeIndexedDomain(1, 3),
        input_ports=(Port("x", 0, "u8"),),
        output_ports=(
            Port("first", 2, "u8"),
            Port("duplicate", 2, "u8"),
            Port("input_passthrough", 0, "u8"),
        ),
        gates=(
            StructuralGate(
                1,
                "copy",
                (0,),
                256,
                value_type="u8",
                metadata={"cost": 1},
            ),
            StructuralGate(2, "add", (1, 1), 256, value_type="u8"),
        ),
    )


def test_structural_contract_preserves_ordered_duplicate_outputs_and_reads():
    circuit = tiny_circuit()

    validate_circuit_contract(circuit, exhaustive=True)

    assert ordered_output_positions(circuit) == (2, 2, 0)
    assert circuit.gate_at(2).predecessors == (1, 1)
    assert circuit.gate_at(1).metadata == (("cost", 1),)
    assert circuit.gate_at(1).value_cardinality_upper_bound == 256


def test_executable_gate_preserves_argument_order_and_relation_metadata():
    gate = ExecutableGate(
        position=7,
        operation="subtract",
        arguments=(2, 2, 1),
        output_type="u8",
        relation_id="word.subtract.v1",
        metadata={"rounding": "wrap"},
    )

    assert gate.arguments == (2, 2, 1)
    assert gate.predecessors == (2, 2, 1)
    assert gate.relation_id == "word.subtract.v1"


def test_circuit_contract_rejects_unknown_ports_and_gate_dependencies():
    circuit = tiny_circuit()
    bad_output = TinyCircuit(
        identity=circuit.identity,
        computed_positions=circuit.computed_positions,
        input_ports=circuit.input_ports,
        output_ports=(Port("missing", 99, "u8"),),
        gates=circuit.gates,
    )
    bad_gate = TinyCircuit(
        identity=circuit.identity,
        computed_positions=circuit.computed_positions,
        input_ports=circuit.input_ports,
        output_ports=circuit.output_ports,
        gates=(circuit.gates[0], StructuralGate(2, "add", (99,), 256)),
    )

    with pytest.raises(InvalidArtifact, match="unknown position"):
        validate_circuit_contract(bad_output)
    with pytest.raises(InvalidArtifact, match="unknown position"):
        validate_circuit_contract(bad_gate, exhaustive=True)


def test_capacity_profile_identity_cannot_masquerade_as_a_circuit():
    circuit = tiny_circuit()
    profile = TinyCircuit(
        identity=structure(ArtifactKind.CAPACITY_PROFILE),
        computed_positions=circuit.computed_positions,
        input_ports=circuit.input_ports,
        output_ports=circuit.output_ports,
        gates=circuit.gates,
    )

    with pytest.raises(InvalidArtifact, match="not a structural circuit"):
        validate_circuit_contract(profile)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Port("bad", True, "u8"),
        lambda: StructuralGate(1, "op", (True,), 2),
        lambda: StructuralGate(1, "op", (), 0),
        lambda: ExecutableGate(1, "op", (), "", "relation"),
    ],
)
def test_circuit_records_reject_ambiguous_identifiers(factory):
    with pytest.raises(InvalidArtifact):
        factory()
