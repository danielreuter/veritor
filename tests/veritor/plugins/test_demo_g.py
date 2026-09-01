from __future__ import annotations

from circuit_cut_analysis.capacity import LogCardinality
from veritor.compile import CallDagCircuit
from veritor.core import CompiledArtifact
from veritor.plugins import (
    BatchInput,
    CapacityClaimKind,
    DemoGCompileRequest,
    ProtocolCircuitArtifact,
    compile_demo_g,
    demo_expected_outputs,
    demo_public_inputs,
    make_demo_request,
)


def test_demo_g_exposes_the_compiled_artifact() -> None:
    request = DemoGCompileRequest()
    artifact = compile_demo_g(request)
    assert isinstance(artifact, ProtocolCircuitArtifact)
    assert isinstance(artifact.compiled, CompiledArtifact)
    assert artifact.verification_access() is artifact.compiled
    assert artifact.circuit is artifact.compiled.circuit
    assert artifact.replay_partition is artifact.compiled.replay
    assert artifact.verification_partition is artifact.compiled.verification
    assert artifact.compiled_identity == artifact.compiled.identity
    assert isinstance(artifact.circuit, CallDagCircuit)


def test_demo_g_request_binds_batch_into_identity() -> None:
    default = compile_demo_g()
    other = compile_demo_g(
        DemoGCompileRequest(batch=BatchInput((make_demo_request(2, 7, 8),)))
    )

    assert default.identity.request_digest != other.identity.request_digest
    assert default.compiled.identity != other.compiled.identity
    assert other.execute() == other.expected_outputs


def test_demo_g_public_inputs_expected_outputs_and_execution() -> None:
    request = DemoGCompileRequest()
    artifact = compile_demo_g(request)
    assert artifact.public_inputs == request.public_inputs
    assert artifact.public_inputs == demo_public_inputs(request)
    assert artifact.public_inputs == demo_public_inputs(request.batch)
    assert artifact.expected_outputs == request.expected_outputs
    assert artifact.expected_outputs == demo_expected_outputs(request)
    assert artifact.expected_outputs == demo_expected_outputs(
        request.batch,
        request.cell_bits,
    )
    assert artifact.execute() == request.expected_outputs
    assert artifact.execution_access() is artifact.circuit


def test_demo_g_bound_provider_is_exact() -> None:
    artifact = compile_demo_g()
    provider = artifact.bound_provider
    assert provider.claim_kind is CapacityClaimKind.EXACT
    assert provider.output_frontier > LogCardinality.zero()
    output_positions = tuple(port.position for port in artifact.circuit.output_ports)
    result = provider.evaluate(output_positions)
    assert result.claim_kind is CapacityClaimKind.EXACT
    assert result.is_exact
    assert result.lower_bound == result.upper_bound
    assert result.cut_gate_ids
