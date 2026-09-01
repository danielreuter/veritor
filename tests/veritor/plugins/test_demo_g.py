from __future__ import annotations

from circuit_cut_analysis.capacity import LogCardinality
from veritor.compile import CallDagCircuit
from veritor.core import validate_compiled_result
from veritor.plugins import (
    CapacityClaimKind,
    DemoGCompileRequest,
    ProtocolCircuitArtifact,
    compile_demo_g,
    demo_expected_outputs,
    demo_public_inputs,
)


def test_demo_g_exposes_literal_validated_protocol_tuple() -> None:
    request = DemoGCompileRequest()
    artifact = compile_demo_g(request)
    assert isinstance(artifact, ProtocolCircuitArtifact)
    assert artifact.literal_tuple is artifact.compiled_tuple
    assert artifact.as_protocol_tuple() is artifact.literal_tuple
    assert artifact.circuit is artifact.literal_tuple[0]
    assert artifact.replay_partition is artifact.literal_tuple[1]
    assert artifact.verification_partition is artifact.literal_tuple[2]
    assert artifact.verification_access() is artifact.literal_tuple
    assert artifact.compiled_identity == validate_compiled_result(
        *artifact.literal_tuple
    )
    assert isinstance(artifact.circuit, CallDagCircuit)


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
