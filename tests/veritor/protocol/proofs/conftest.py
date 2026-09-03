"""Fixtures for the proof layer: a recording backend and small runs to feed it."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

import pytest

from veritor.constructors import (
    ClusterG,
    LMShape,
    Request,
    expected_matmul_outputs,
    random_parameters,
    schedule_fcfs,
)
from veritor.core import Compiled, GateSet, VerificationPolicy, make_isa_gate_set
from veritor.protocol import (
    Expectation,
    VerifierParameters,
    commit_weights,
    make_expectation,
)
from veritor.protocol.proofs import Statement, TransparentBackend, Witness
from veritor.research import Compile

RECORDING_BACKEND = "recording"


@dataclass
class RecordingBackend:
    """A fake zk backend: the transparent checker behind a non-transparent id.

    It records every ``(statement, witness)`` it is asked to prove, so tests
    can feed real obligations from a run to the SP1 host, and every statement
    it verifies.  Set ``reject`` to make ``verify`` fail.
    """

    transparent: TransparentBackend
    proved: list[tuple[Statement, Witness]] = field(default_factory=list)
    verified: list[Statement] = field(default_factory=list)
    reject: bool = False

    backend_id = RECORDING_BACKEND

    def prove(self, statement: Statement, witness: Witness) -> bytes:
        self.proved.append((statement, witness))
        return self.transparent.prove(statement, witness)

    def verify(self, statement: Statement, proof: bytes) -> bool:
        self.verified.append(statement)
        if self.reject:
            return False
        return self.transparent.verify(statement, proof)


@pytest.fixture
def recording(compiled: Compiled) -> RecordingBackend:
    return RecordingBackend(TransparentBackend(compiled.circuit.gate_set, compiled))


@pytest.fixture
def expect(compilation, workload, model_weights):
    """The protocol fixture's expectation factory, plus a ``backend`` argument."""

    def build(
        policy: VerificationPolicy = CHECK_EVERYTHING,
        *,
        backend: str = "transparent",
        session_id: bytes = b"tests/veritor/protocol/proofs",
        parameters: VerifierParameters | None = None,
    ) -> Expectation:
        return make_expectation(
            compilation,
            policy,
            expected_matmul_outputs(workload),
            parameters=parameters or VerifierParameters(max_capacity=None),
            weights=model_weights[0],
            session_id=session_id,
            q_seed=b"Q" * 32,
            s_seed=b"S" * 32,
            backend=backend,
        )

    return build


SHAPE = LMShape(vocab=8, d_model=4, heads=2, layers=1, context=6, width=16)
REQUESTS = (Request((1, 2, 3), 3), Request((5,), 2))
CLUSTER_GATE_SET: GateSet = make_isa_gate_set(16)
SAMPLE_SOME = VerificationPolicy(Fraction(1, 2), Fraction(1, 3))
CHECK_EVERYTHING = VerificationPolicy(1, 1)


class SmallCluster:
    """A tiny cluster run on the toy ISA: real obligations for the zk tests."""

    def __init__(self) -> None:
        self.parameters = random_parameters(SHAPE, seed=7)
        self.weights, self.tree = commit_weights(
            CLUSTER_GATE_SET, self.parameters.flatten()
        )
        schedule = schedule_fcfs(REQUESTS, 1, 2, 5)
        self.constructor = ClusterG(SHAPE, 1, 2, 5)
        self.compilation = Compile(
            self.constructor,
            REQUESTS,
            schedule.encode(),
            CLUSTER_GATE_SET,
            max_advice_bits=4096,
        )
        self.compiled = self.compilation.compiled
        self.values = dict(
            enumerate(
                self.compiled.circuit.evaluate(
                    self.compilation.inputs, self.parameters.flatten()
                )
            )
        )
        self.outputs = tuple(
            self.values[address] for address in self.compiled.circuit.outputs
        )

    def expectation(
        self,
        policy: VerificationPolicy = SAMPLE_SOME,
        *,
        backend: str = RECORDING_BACKEND,
        session_id: bytes = b"proofs/small-cluster",
    ) -> Expectation:
        return make_expectation(
            self.compilation,
            policy,
            self.outputs,
            parameters=VerifierParameters(max_advice_bits=4096, max_capacity=None),
            weights=self.weights,
            session_id=session_id,
            q_seed=b"Q" * 32,
            s_seed=b"S" * 32,
            backend=backend,
        )


@pytest.fixture(scope="session")
def small_cluster() -> SmallCluster:
    return SmallCluster()
