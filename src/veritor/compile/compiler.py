"""Public orchestration for data-only call-DAG compilation."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from veritor.core import CompiledArtifact, JSONValue

from .call_dag import CallDagCircuit, Kernel, construct
from .partitions import (
    DEFAULT_REPLAY_POLICY,
    DEFAULT_VERIFICATION_POLICY,
    PartitionPolicy,
    compile_partitions_for_policies,
)


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
) -> CompiledArtifact:
    """Run ``G`` and return the compiled ``(C, replay, verification, boundary)``.

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
    return compile_partitions_for_policies(
        CallDagCircuit(kernel, construction.load.root),
        replay_policy,
        verification_policy,
        replay_configuration=replay_configuration,
        verification_configuration=verification_configuration,
    )
