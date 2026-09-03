"""The SP1 backend on real obligations: execute (exact cycles) and one real core proof.

These build the guest and run the host binary, so they are ``slow`` and ``zk``
and skip cleanly when ``cargo prove`` is not installed.  Everything but one
test uses ``execute`` (no proof); the proof test keeps its batch small.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from veritor.core import VerificationPolicy
from veritor.protocol import VerificationCode, run_protocol
from veritor.protocol.proofs import (
    SP1_BACKEND,
    SP1Backend,
    Statement,
    TransparentBackend,
    Witness,
    make_statement,
    sp1_toolchain_available,
    statement_digest,
)
from veritor.protocol.proofs.sp1 import ExecutionReport, ProofReport, describe_batch
from veritor.protocol.proofs.statement import select

from .conftest import RecordingBackend, SmallCluster

pytestmark = [
    pytest.mark.slow,
    pytest.mark.zk,
    pytest.mark.skipif(
        not sp1_toolchain_available(), reason="the SP1 toolchain is not installed"
    ),
]

PHASES = ("io", "digest", "parse", "merkle", "gates")


@pytest.fixture(scope="module")
def sp1() -> SP1Backend:
    backend = SP1Backend(mode="core")
    backend.binary()  # build once for the module
    return backend


@pytest.fixture(scope="module")
def cluster_batch(small_cluster: SmallCluster) -> tuple[Statement, Witness]:
    """Every obligation of a sampled small-cluster run, as one batch."""

    recording = RecordingBackend(
        TransparentBackend(
            small_cluster.compiled.circuit.gate_set, small_cluster.compiled
        )
    )
    run = run_protocol(
        small_cluster.compiled,
        small_cluster.expectation(VerificationPolicy(Fraction(1, 2), Fraction(1, 3))),
        small_cluster.values,
        weight_tree=small_cluster.tree,
        backend=recording,
    )
    assert run.report.accepted
    ((statement, witness),) = recording.proved
    return statement, witness


def test_the_host_identifies_the_guest(sp1: SP1Backend) -> None:
    info = sp1.info()
    assert info["backend"] == SP1_BACKEND
    assert info["sp1_version"] == "6.4.0"
    assert len(str(info["elf_sha256"])) == 64
    assert str(info["vk_hash"]).startswith("0x")
    assert info["public_values_len"] == 33


def test_execute_accepts_real_cluster_obligations_with_exact_cycles(
    sp1: SP1Backend, cluster_batch: tuple[Statement, Witness]
) -> None:
    statement, witness = cluster_batch
    report = sp1.execute(statement, witness)
    assert isinstance(report, ExecutionReport)
    assert report.verdict is True
    assert report.statement_digest == statement_digest(statement)
    assert set(report.cycle_tracker) == set(PHASES)
    assert report.total_cycles >= sum(report.cycle_tracker.values())
    assert report.syscalls.get("SHA_COMPRESS", 0) > 0, (
        "the sha2 precompile patch must apply"
    )
    sizes = describe_batch(statement)
    assert sizes["obligations"] == len(statement.obligations) > 1
    # every Merkle level is at least one compression and the guest paid for it
    assert report.cycle_tracker["merkle"] > 200 * (
        sizes["merkle_levels"] + sizes["positions"]
    )


def test_execute_rejects_a_corrupted_witness(
    sp1: SP1Backend, cluster_batch: tuple[Statement, Witness]
) -> None:
    statement, witness = cluster_batch
    openings = [list(items) for items in witness.obligations]
    value, path = openings[0][0]
    openings[0][0] = (bytes(byte ^ 1 for byte in value), path)
    corrupted = Witness(tuple(tuple(items) for items in openings))
    report = sp1.execute(statement, corrupted)
    assert report.verdict is False
    assert report.statement_digest == statement_digest(statement)


def test_execute_cost_grows_with_the_batch(
    sp1: SP1Backend, cluster_batch: tuple[Statement, Witness]
) -> None:
    statement, witness = cluster_batch
    half = statement.obligations[: len(statement.obligations) // 2]
    smaller = make_statement(
        statement.gate_set_id,
        statement.gate_set_digest,
        statement.width,
        statement.kinds,
        half,
    )
    full = sp1.execute(statement, witness)
    part = sp1.execute(smaller, select(statement, witness, half))
    assert part.verdict and full.verdict
    assert part.cycle_tracker["merkle"] < full.cycle_tracker["merkle"]
    assert part.cycle_tracker["gates"] < full.cycle_tracker["gates"]


def test_one_real_core_proof_of_a_small_batch_round_trips(
    compiled, expect, honest_values, model_weights
) -> None:
    """The whole protocol with the SP1 backend: prove one batch, verify it, and reject a swap."""

    sp1 = SP1Backend(mode="core")
    expectation = expect(backend=SP1_BACKEND)
    run = run_protocol(
        compiled,
        expectation,
        honest_values,
        weight_tree=model_weights[1],
        backend=sp1,
        plan=None,
    )
    assert run.report.code is VerificationCode.ACCEPTED, run.report
    assert run.transcript is not None
    (proof,) = run.transcript.evidence.proofs
    report = sp1.last_report
    assert isinstance(report, ProofReport)
    assert report.mode == "core" and report.shards >= 1
    assert report.proof_bytes == len(proof.proof) > 0
    assert report.execution.verdict is True
    print(
        f"\nSP1 core proof: {report.execution.total_cycles} cycles, {report.shards} shard(s), "
        f"{report.proof_bytes} bytes, setup {report.setup_seconds:.1f}s, "
        f"prove {report.prove_seconds:.1f}s, verify {report.verify_seconds:.2f}s"
    )
    # the proof is bound to its statement: the same proof does not verify another statement
    statement, _ = _statement_of(compiled, expectation, honest_values, model_weights)
    assert sp1.verify(statement, proof.proof) is True
    other = _other_statement(statement)
    assert sp1.verify(other, proof.proof) is False
    assert sp1.verify(statement, b"not a proof") is False


def _statement_of(compiled, expectation, honest_values, model_weights):
    """The statement the SP1 run proved, recomputed with the recording backend under SP1's id."""

    recording = RecordingBackend(
        TransparentBackend(compiled.circuit.gate_set, compiled)
    )
    recording.backend_id = SP1_BACKEND  # type: ignore[misc]
    run = run_protocol(
        compiled,
        expectation,
        honest_values,
        weight_tree=model_weights[1],
        backend=recording,
    )
    assert run.report.accepted
    return recording.proved[0]


def _other_statement(statement: Statement) -> Statement:
    return make_statement(
        statement.gate_set_id,
        statement.gate_set_digest,
        statement.width,
        statement.kinds,
        statement.obligations[:-1],
    )
