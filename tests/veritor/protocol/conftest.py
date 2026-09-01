from __future__ import annotations

from collections.abc import Callable

import pytest

from veritor.compile import (
    MatmulWorkload,
    compile_matmul_workload,
    expected_matmul_outputs,
)
from veritor.core import CompiledArtifact, VerificationPolicy
from veritor.protocol import Expectation, make_expectation

Q_SEED = b"Q" * 32
S_SEED = b"S" * 32
SESSION_ID = b"tests/veritor/protocol"
CHECK_EVERYTHING = VerificationPolicy(1, 1, 0)

type ExpectationFactory = Callable[..., Expectation]


@pytest.fixture(scope="session")
def workload() -> MatmulWorkload:
    return MatmulWorkload(((1, 2), (3, 4)), (((1, 0), (0, 1)), ((2, 2),)))


@pytest.fixture(scope="session")
def artifact(workload: MatmulWorkload) -> CompiledArtifact:
    return compile_matmul_workload(workload)


@pytest.fixture
def honest_values(artifact: CompiledArtifact, workload: MatmulWorkload) -> dict[int, object]:
    return dict(enumerate(artifact.circuit.evaluate_tape(workload.public_inputs)))


@pytest.fixture
def expect(artifact: CompiledArtifact, workload: MatmulWorkload) -> ExpectationFactory:
    """Build an expectation for ``artifact`` with fixed seeds and honest I/O by default."""

    def build(
        policy: VerificationPolicy = CHECK_EVERYTHING,
        *,
        claimed_outputs: tuple[int, ...] | None = None,
        session_id: bytes = SESSION_ID,
        q_seed: bytes = Q_SEED,
        s_seed: bytes = S_SEED,
    ) -> Expectation:
        return make_expectation(
            artifact,
            policy,
            workload.public_inputs,
            expected_matmul_outputs(workload) if claimed_outputs is None else claimed_outputs,
            session_id=session_id,
            q_seed=q_seed,
            s_seed=s_seed,
        )

    return build
