"""Public orchestration for data-only call-DAG compilation."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from veritor.core import (
    JSONValue,
    ReplayPartition,
    VerificationPartition,
    validate_compiled_result,
)

from .call_dag import CallDagCircuit, Kernel, construct
from .partitions import (
    DEFAULT_REPLAY_POLICY,
    DEFAULT_VERIFICATION_POLICY,
    PartitionPolicy,
    derive_replay_partition,
    derive_verification_partition,
)

type CompiledCallDag = tuple[
    CallDagCircuit,
    ReplayPartition,
    VerificationPartition,
]


def compile_call_dag(
    kernel: Kernel,
    constructor: Callable[[object, bytes], bytes],
    x: object,
    a: bytes,
    *,
    input_cells: Sequence[int],
    advice_bound_bits: int,
    replay_policy: PartitionPolicy | str = DEFAULT_REPLAY_POLICY,
    verification_policy: PartitionPolicy | str = DEFAULT_VERIFICATION_POLICY,
    replay_configuration: JSONValue | None = None,
    verification_configuration: JSONValue | None = None,
) -> CompiledCallDag:
    """Run ``G`` and return the validated literal ``(C, replay, verification)``.

    Constructor code is outside the trust boundary.  Its only trusted output
    is the canonical byte document accepted by ``kernel.load``.
    """

    construction = construct(
        kernel,
        constructor,
        x,
        a,
        input_cells=input_cells,
        advice_bound_bits=advice_bound_bits,
    )
    circuit = CallDagCircuit(kernel, construction.load.root)
    replay = derive_replay_partition(
        circuit,
        replay_policy,
        configuration=replay_configuration,
    )
    verification = derive_verification_partition(
        circuit,
        replay,
        verification_policy,
        configuration=verification_configuration,
    )
    validate_compiled_result(circuit, replay, verification)
    return circuit, replay, verification

