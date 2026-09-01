from __future__ import annotations

from circuit_cut_analysis.capacity import LogCardinality
from veritor.core import Capability, CompiledArtifact
from veritor.plugins import (
    ArchitectureId,
    CapacityClaimKind,
    MatmulCompileRequest,
    ProtocolCircuitArtifact,
    compile_architecture,
    compile_matmul,
    matmul_expected_matrices,
    matmul_expected_outputs,
    matmul_public_inputs,
)


def _request(*, marker: int = 1, cell_bits: int = 8) -> MatmulCompileRequest:
    return MatmulCompileRequest(
        (
            (marker, 2),
            (3, 4),
        ),
        (
            ((5, 6),),
            (
                (7, 8),
                (9, 10),
            ),
        ),
        cell_bits=cell_bits,
    )


def test_matmul_plugin_exposes_the_compiled_artifact() -> None:
    request = _request()
    artifact = compile_matmul(request)

    assert isinstance(artifact, ProtocolCircuitArtifact)
    assert artifact.architecture_id is ArchitectureId.MATMUL
    assert isinstance(artifact.compiled, CompiledArtifact)
    assert artifact.compiled_identity == artifact.compiled.identity
    assert artifact.public_inputs == matmul_public_inputs(request)
    assert artifact.expected_outputs == matmul_expected_outputs(request)
    assert artifact.execute() == request.expected_outputs
    assert matmul_expected_matrices(request) == (
        ((23, 34),),
        (
            (31, 46),
            (39, 58),
        ),
    )
    for capability in (
        Capability.STATIC_COMPILE,
        Capability.STATIC_PARTITION,
        Capability.STATIC_BOUND,
        Capability.EXECUTE,
        Capability.VERIFY,
    ):
        assert artifact.capabilities.supports(capability)


def test_registry_compiles_matmul_and_binds_public_values() -> None:
    first = compile_architecture(ArchitectureId.MATMUL, _request(marker=1))
    second = compile_architecture("matmul", _request(marker=2))

    assert isinstance(first, ProtocolCircuitArtifact)
    assert isinstance(second, ProtocolCircuitArtifact)
    assert first.circuit.identity == second.circuit.identity
    assert first.compiled_identity == second.compiled_identity
    assert first.identity.request_digest != second.identity.request_digest
    assert first.identity.digest != second.identity.digest


def test_matmul_capacity_provider_is_exact() -> None:
    artifact = compile_matmul(_request())
    provider = artifact.bound_provider
    output_positions = tuple(
        int(port.position) for port in artifact.circuit.output_ports
    )
    result = provider.evaluate(output_positions)

    assert provider.claim_kind is CapacityClaimKind.EXACT
    assert provider.output_frontier > LogCardinality.zero()
    assert result.claim_kind is CapacityClaimKind.EXACT
    assert result.is_exact
    assert result.cut_gate_ids
