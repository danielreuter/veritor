from __future__ import annotations

from collections.abc import Callable

import pytest

from veritor.compile import Compiler
from veritor.constructors import MatmulG, MatmulWorkload, expected_matmul_outputs
from veritor.core import Compiled, VerificationPolicy, make_word_gate_set
from veritor.protocol import (
    Expectation,
    MerkleTree,
    VerifierParameters,
    Weights,
    commit_weights,
    make_expectation,
)

Q_SEED = b"Q" * 32
S_SEED = b"S" * 32
SESSION_ID = b"tests/veritor/protocol"
CHECK_EVERYTHING = VerificationPolicy(1, 1)

type ExpectationFactory = Callable[..., Expectation]


@pytest.fixture(scope="session")
def workload() -> MatmulWorkload:
    return MatmulWorkload(((1, 2), (3, 4)), (((1, 0), (0, 1)), ((2, 2),)))


@pytest.fixture(scope="session")
def compiled(workload: MatmulWorkload) -> Compiled:
    gate_set = make_word_gate_set(workload.width)
    return Compiler(gate_set).compile(
        MatmulG(workload.width)(workload, b""), workload.public_inputs
    )


@pytest.fixture(scope="session")
def model_weights(compiled: Compiled, workload: MatmulWorkload) -> tuple[Weights, MerkleTree]:
    """The model's ``kappa_W`` over the circuit's weight gates, committed once."""

    return commit_weights(compiled, workload.weight_values)


@pytest.fixture
def honest_values(compiled: Compiled, workload: MatmulWorkload) -> dict[int, object]:
    return dict(
        enumerate(compiled.circuit.evaluate(workload.public_inputs, workload.weight_values))
    )


@pytest.fixture
def expect(
    compiled: Compiled, workload: MatmulWorkload, model_weights: tuple[Weights, MerkleTree]
) -> ExpectationFactory:
    """Build an expectation for ``compiled`` with fixed seeds and honest I/O by default."""

    def build(
        policy: VerificationPolicy = CHECK_EVERYTHING,
        *,
        parameters: VerifierParameters | None = None,
        public_inputs: tuple[int, ...] | None = None,
        claimed_outputs: tuple[int, ...] | None = None,
        weights: Weights | None = model_weights[0],
        session_id: bytes = SESSION_ID,
        q_seed: bytes = Q_SEED,
        s_seed: bytes = S_SEED,
    ) -> Expectation:
        return make_expectation(
            compiled,
            policy,
            workload.public_inputs if public_inputs is None else public_inputs,
            expected_matmul_outputs(workload) if claimed_outputs is None else claimed_outputs,
            parameters=parameters,
            weights=weights,
            session_id=session_id,
            q_seed=q_seed,
            s_seed=s_seed,
        )

    return build
