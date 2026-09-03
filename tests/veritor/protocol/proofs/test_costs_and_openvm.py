"""The cost table's estimates and the OpenVM adapter's wire mapping (no GPU needed)."""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import pytest

from veritor.analysis.cost import CostParameters
from veritor.protocol import ProtocolError, run_protocol
from veritor.protocol.proofs import (
    OPENVM_BACKEND,
    OpenVMBackend,
    Statement,
    TransparentBackend,
    Witness,
    encode_statement,
    encode_witness,
)
from veritor.protocol.proofs.costs import (
    LAPTOP_M_SERIES,
    POD_B_CPU,
    SP1_BATCHING_CURVE,
    SP1_CYCLES_PER_GATE,
    SP1_CYCLES_PER_LEAF,
    SP1_CYCLES_PER_MERKLE_LEVEL,
    alpha_dot,
    alpha_toy_isa,
    estimate_cycles,
    estimate_prover_seconds,
    prover_seconds,
    sp1_cost_parameters,
)
from veritor.protocol.proofs.openvm import openvm_input, reveals_to_public_values

from .conftest import RECORDING_BACKEND, RecordingBackend


@pytest.fixture
def batch(compiled, expect, honest_values, model_weights) -> tuple[Statement, Witness]:
    recording = RecordingBackend(
        TransparentBackend(compiled.circuit.gate_set, compiled)
    )
    run = run_protocol(
        compiled,
        expect(backend=RECORDING_BACKEND),
        honest_values,
        weight_tree=model_weights[1],
        backend=recording,
    )
    assert run.report.accepted
    ((statement, witness),) = recording.proved
    return statement, witness


class TestCosts:
    def test_the_estimate_counts_the_batch(self, batch) -> None:
        statement, witness = batch
        estimate = estimate_cycles(statement)
        assert estimate.obligations == len(statement.obligations) == 16
        assert estimate.positions == sum(
            len(item.positions) for item in statement.obligations
        )
        assert estimate.gates == sum(statement.program(item.kind).size for item in statement.obligations)
        assert estimate.statement_bytes == len(encode_statement(statement))
        # the implied witness size is within a few bytes of the real one
        assert abs(estimate.witness_bytes - len(encode_witness(witness))) < 64
        assert estimate.merkle == (
            SP1_CYCLES_PER_LEAF * estimate.positions
            + SP1_CYCLES_PER_MERKLE_LEVEL * estimate.merkle_levels
        )
        assert estimate.total == sum(
            (
                estimate.batch,
                estimate.per_obligation,
                estimate.parse,
                estimate.digest,
                estimate.merkle,
                estimate.relations,
            )
        )
        # the measured value for this very batch (SP1 6.4.0 execute): 879,613 cycles
        assert 0.95 < estimate.total / 879_613 < 1.05

    def test_seconds_follow_the_machine_profile(self, batch) -> None:
        statement, _ = batch
        cycles = estimate_cycles(statement).total
        assert prover_seconds(0) == LAPTOP_M_SERIES.floor_seconds
        assert prover_seconds(cycles) == pytest.approx(
            LAPTOP_M_SERIES.floor_seconds + cycles / (LAPTOP_M_SERIES.khz * 1000)
        )
        assert estimate_prover_seconds(statement) == prover_seconds(cycles)
        assert estimate_prover_seconds(statement, POD_B_CPU) > estimate_prover_seconds(
            statement
        )
        # the measured proof of this batch took 18.1 s on the laptop profile's machine class
        assert 14 < estimate_prover_seconds(statement) < 22

    def test_the_batching_curve_is_amortizing(self) -> None:
        per_check = [seconds / checks for checks, _, seconds, _ in SP1_BATCHING_CURVE]
        assert per_check == sorted(per_check, reverse=True)
        _, (k4, _, s4, _), (k16, _, s16, _) = SP1_BATCHING_CURVE
        marginal = (s16 - s4) / (k16 - k4)
        assert marginal == pytest.approx(24.7, abs=0.05)
        # the pod profile reproduces the curve to ~10%
        for checks, cycles, seconds, _ in SP1_BATCHING_CURVE:
            assert abs(prover_seconds(cycles, POD_B_CPU) - seconds) / seconds < 0.15

    def test_alpha_values_and_provenance(self, batch) -> None:
        statement, _ = batch
        assert alpha_dot("openvm-tc-matmul") == pytest.approx(7.1e7, rel=0.05)
        assert alpha_dot("openvm-tc-dot") == pytest.approx(1.5e8, rel=0.05)
        assert alpha_dot("sp1-tc-dot") == pytest.approx(1.8e8, rel=0.05)
        assert alpha_dot("sp1-tc-dot", native_mac_per_second=3.3e14) > alpha_dot(
            "sp1-tc-dot"
        )
        with pytest.raises(ValueError, match="unknown dot backend"):
            alpha_dot("groth16")
        # the toy ISA opens a leaf per scalar: alpha is dominated by hashing
        assert 1e9 < alpha_toy_isa(statement) < 1e10

    def test_gate_table_is_complete(self) -> None:
        assert set(SP1_CYCLES_PER_GATE) == {
            "add",
            "eq",
            "lt",
            "mul",
            "shr",
            "sub",
            "in",
            "weight",
        }
        assert SP1_CYCLES_PER_GATE["in"] == SP1_CYCLES_PER_GATE["weight"] == 0
        assert (
            min(v for k, v in SP1_CYCLES_PER_GATE.items() if k not in ("in", "weight"))
            > 100
        )

    def test_cost_parameters_price_h_and_c0_in_seconds(self) -> None:
        shallow = sp1_cost_parameters(depth=4, batch_size=1)
        deep = sp1_cost_parameters(depth=20, batch_size=1)
        batched = sp1_cost_parameters(depth=20, batch_size=1000)
        assert isinstance(shallow, CostParameters)
        assert deep.hash_cost > shallow.hash_cost
        assert deep.proof_overhead == shallow.proof_overhead
        assert batched.proof_overhead < deep.proof_overhead / 100
        assert batched.hash_cost == deep.hash_cost
        assert float(deep.hash_cost) == pytest.approx(
            (SP1_CYCLES_PER_LEAF + 20 * SP1_CYCLES_PER_MERKLE_LEVEL + 11.7 * (40 + 640))
            / (LAPTOP_M_SERIES.khz * 1000)
        )
        with pytest.raises(ValueError):
            sp1_cost_parameters(depth=-1, batch_size=1)


class TestOpenVMAdapter:
    def test_the_hint_stream_is_our_canonical_bytes(self, batch) -> None:
        statement, witness = batch
        mapped = openvm_input(statement, witness)
        assert mapped.hint_stream == (
            encode_statement(statement),
            encode_witness(witness),
        )
        digest = hashlib.sha256(encode_statement(statement)).digest()
        assert mapped.reveals[:8] == struct.unpack("<8I", digest)
        assert mapped.reveals[8] == 1
        assert openvm_input(statement, witness, verdict=False).reveals[8] == 0

    def test_reveals_rechunk_to_sp1_public_values(self, batch) -> None:
        statement, witness = batch
        mapped = openvm_input(statement, witness)
        public_values = reveals_to_public_values(mapped.reveals)
        assert (
            public_values
            == hashlib.sha256(encode_statement(statement)).digest() + b"\x01"
        )
        with pytest.raises(ProtocolError):
            reveals_to_public_values(mapped.reveals[:8])
        with pytest.raises(ProtocolError):
            reveals_to_public_values((*mapped.reveals[:8], 2))

    def test_the_backend_is_a_host_binary_away(self, tmp_path: Path) -> None:
        backend = OpenVMBackend(tmp_path / "openvm-host")
        assert backend.backend_id == OPENVM_BACKEND
        assert backend.build is False
        with pytest.raises(ProtocolError, match="does not exist"):
            backend.binary()
