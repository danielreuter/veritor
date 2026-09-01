from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from veritor.commitment import MerkleSha256Backend, ValueCommitmentRegistry
from veritor.core import (
    ArtifactKind,
    ExecutableGate,
    ExplicitIndexedDomain,
    Port,
    ReplayPartition,
    ReplayUnit,
    StructuralGate,
    StructureIdentity,
    VerificationPartition,
    VerificationPolicy,
    VerificationUnit,
)
from veritor.staged import (
    ResolvedExecutableArtifact,
    ResolvedStructuralArtifact,
    SampleEvidenceRegistry,
    StagedProtocolBuilder,
    TransparentLocalCheckBackend,
    TrustedArtifactRegistry,
    TrustedVerificationContext,
    VerificationExpectation,
)


@dataclass(frozen=True, slots=True)
class TinyExecutableCircuit:
    identity: StructureIdentity
    computed_positions: ExplicitIndexedDomain
    input_ports: tuple[Port, ...]
    output_ports: tuple[Port, ...]
    structural_gates: tuple[StructuralGate, ...]
    executable_gates: tuple[ExecutableGate, ...]

    def gate_at(self, position: int) -> StructuralGate:
        return self.structural_gates[self.computed_positions.rank(position)]

    def executable_gate_at(self, position: int) -> ExecutableGate:
        return self.executable_gates[self.computed_positions.rank(position)]


@dataclass(frozen=True, slots=True)
class U8ValueService:
    def encode(self, value_type: str, value: object) -> bytes:
        if value_type != "tests/u8" or type(value) is not int or not 0 <= value < 256:
            raise ValueError("not a canonical u8")
        return value.to_bytes(1, "big")

    def decode(self, value_type: str, payload: bytes) -> int:
        if value_type != "tests/u8" or type(payload) is not bytes or len(payload) != 1:
            raise ValueError("not an encoded u8")
        return int.from_bytes(payload, "big")


@dataclass(frozen=True, slots=True)
class TinyRelationService:
    def check(
        self,
        relation_id: str,
        arguments: tuple[object, ...],
        output: object,
    ) -> bool:
        if relation_id == "tests/relation/add":
            return (
                len(arguments) == 2
                and output == (int(arguments[0]) + int(arguments[1])) % 256
            )
        if relation_id == "tests/relation/copy":
            return len(arguments) == 1 and output == arguments[0]
        raise ValueError("unknown trusted relation")


def _identity(
    label: str,
    kind: ArtifactKind = ArtifactKind.EXECUTABLE_CIRCUIT,
) -> StructureIdentity:
    return StructureIdentity.from_manifest(
        {"label": label},
        schema_version="1",
        artifact_kind=kind,
        compiler_id="tests.staged.compiler",
        compiler_version="1",
        semantic_scope_id="tests.staged.u8-v1",
        value_registry_digest="11" * 32,
        operator_registry_digest="22" * 32,
    )


@dataclass(frozen=True, slots=True)
class ProtocolCase:
    artifact: ResolvedExecutableArtifact
    assignment: dict[int, int]
    q_seed: bytes = b"Q" * 32
    s_seed: bytes = b"S" * 32

    @property
    def circuit(self) -> TinyExecutableCircuit:
        return self.artifact.circuit  # type: ignore[return-value]

    def expectation(
        self,
        *,
        policy: VerificationPolicy | None = None,
        assignment: dict[int, int] | None = None,
        session_id: bytes = b"session/fixture/v1",
    ) -> VerificationExpectation:
        values = self.assignment if assignment is None else assignment
        return VerificationExpectation(
            session_id=session_id,
            compiled_result_digest=self.artifact.compiled_result_digest,
            policy=VerificationPolicy(1, 1, 0) if policy is None else policy,
            public_inputs=tuple(
                values[int(port.position)] for port in self.artifact.circuit.input_ports
            ),
            claimed_outputs=tuple(
                values[int(port.position)]
                for port in self.artifact.circuit.output_ports
            ),
            q_seed=self.q_seed,
            s_seed=self.s_seed,
        )

    def builder(self) -> StagedProtocolBuilder:
        return StagedProtocolBuilder(
            self.artifact,
            MerkleSha256Backend(),
            TransparentLocalCheckBackend(),
        )

    def trust(self, *, executable: bool = True) -> TrustedVerificationContext:
        artifact: Any = self.artifact
        if not executable:
            artifact = ResolvedStructuralArtifact(
                self.artifact.circuit,
                self.artifact.replay_partition,
                self.artifact.verification_partition,
            )
        return TrustedVerificationContext(
            TrustedArtifactRegistry((artifact,)),
            ValueCommitmentRegistry.with_defaults(),
            SampleEvidenceRegistry.with_defaults(),
        )


def make_protocol_case() -> ProtocolCase:
    identity = _identity("three-gate")
    computed = ExplicitIndexedDomain((30, 40, 50))
    inputs = (Port("left", 10, "tests/u8"), Port("right", 20, "tests/u8"))
    outputs = (
        Port("result", 50, "tests/u8"),
        Port("duplicate_result", 50, "tests/u8"),
        Port("left_passthrough", 10, "tests/u8"),
    )
    structural = (
        StructuralGate(30, "add", (10, 20), 256, value_type="tests/u8"),
        StructuralGate(40, "add", (30, 10), 256, value_type="tests/u8"),
        StructuralGate(50, "copy", (40,), 256, value_type="tests/u8"),
    )
    executable = (
        ExecutableGate(
            30,
            "add",
            (10, 20),
            "tests/u8",
            "tests/relation/add",
        ),
        ExecutableGate(
            40,
            "add",
            (30, 10),
            "tests/u8",
            "tests/relation/add",
        ),
        ExecutableGate(
            50,
            "copy",
            (40,),
            "tests/u8",
            "tests/relation/copy",
        ),
    )
    circuit = TinyExecutableCircuit(
        identity,
        computed,
        inputs,
        outputs,
        structural,
        executable,
    )
    replay = ReplayPartition(
        identity,
        computed,
        (
            ReplayUnit(0, (30, 40)),
            ReplayUnit(1, (50,)),
        ),
        algorithm_id="tests.replay",
    )
    verification = VerificationPartition(
        identity,
        replay,
        computed,
        (
            VerificationUnit(0, 0, (30,), "tests/relation/add"),
            VerificationUnit(1, 0, (40,), "tests/relation/add"),
            VerificationUnit(2, 1, (50,), "tests/relation/copy"),
        ),
        algorithm_id="tests.verification",
    )
    artifact = ResolvedExecutableArtifact(
        circuit,
        replay,
        verification,
        U8ValueService(),
        TinyRelationService(),
    )
    return ProtocolCase(
        artifact,
        {10: 2, 20: 3, 30: 5, 40: 7, 50: 7},
    )


def make_empty_case() -> ProtocolCase:
    identity = _identity("empty")
    computed = ExplicitIndexedDomain(())
    circuit = TinyExecutableCircuit(
        identity,
        computed,
        (Port("input", 10, "tests/u8"),),
        (
            Port("output", 10, "tests/u8"),
            Port("duplicate", 10, "tests/u8"),
        ),
        (),
        (),
    )
    replay = ReplayPartition(
        identity,
        computed,
        (),
        algorithm_id="tests.replay",
    )
    verification = VerificationPartition(
        identity,
        replay,
        computed,
        (),
        algorithm_id="tests.verification",
    )
    artifact = ResolvedExecutableArtifact(
        circuit,
        replay,
        verification,
        U8ValueService(),
        TinyRelationService(),
    )
    return ProtocolCase(artifact, {10: 9})


def make_structural_empty_artifact() -> ResolvedStructuralArtifact:
    identity = _identity("structural-empty", ArtifactKind.STRUCTURAL_CIRCUIT)
    computed = ExplicitIndexedDomain(())
    circuit = TinyExecutableCircuit(
        identity,
        computed,
        (Port("input", 10, "tests/u8"),),
        (Port("output", 10, "tests/u8"),),
        (),
        (),
    )
    replay = ReplayPartition(
        identity,
        computed,
        (),
        algorithm_id="tests.replay",
    )
    verification = VerificationPartition(
        identity,
        replay,
        computed,
        (),
        algorithm_id="tests.verification",
    )
    return ResolvedStructuralArtifact(circuit, replay, verification)


@pytest.fixture
def protocol_case() -> ProtocolCase:
    return make_protocol_case()


@pytest.fixture
def empty_case() -> ProtocolCase:
    return make_empty_case()


@pytest.fixture
def structural_empty_artifact() -> ResolvedStructuralArtifact:
    return make_structural_empty_artifact()
