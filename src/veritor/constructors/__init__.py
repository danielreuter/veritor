"""Untrusted, client-side code: the tracer and the built-in constructors ``G``.

Nothing in this package is trusted.  A constructor turns a workload into
canonical description bytes; :class:`veritor.compile.Compiler` re-validates
every byte before anything downstream sees a circuit.
"""

from .cluster import ClusterG
from .demo_g import (
    BatchInput,
    DemoG,
    DemoGCompileRequest,
    DotRequest,
    compile_demo_g,
    expected_dot_outputs,
    make_demo_request,
)
from .gpt2 import GPT2, GPT2G, GPT2Shape, gate_budget
from .lm import (
    ADVICE,
    PADDED,
    ExpertParameters,
    LayerParameters,
    LMShape,
    Parameters,
    ToyLM,
    random_parameters,
    reference_generate,
    top_k_route,
)
from .matmul import (
    MatmulCompileRequest,
    MatmulG,
    MatmulWorkload,
    WordMatrix,
    compile_matmul,
    expected_matmul_outputs,
    matmul_expected_matrices,
)
from .prefix import PrefixG
from .requests import RequestsG
from .schedule import (
    Join,
    Occupant,
    Request,
    Schedule,
    ScheduleError,
    schedule_fcfs,
)
from .speculative import SpeculativeG, SpeculativeTrace, reference_speculative
from .tracer import (
    TracedDefinition,
    Tracer,
    TracerError,
    TracerGate,
    Wire,
    Wires,
)

__all__ = [
    "ADVICE",
    "GPT2",
    "GPT2G",
    "PADDED",
    "BatchInput",
    "ClusterG",
    "DemoG",
    "DemoGCompileRequest",
    "DotRequest",
    "ExpertParameters",
    "GPT2Shape",
    "Join",
    "LMShape",
    "LayerParameters",
    "MatmulCompileRequest",
    "MatmulG",
    "MatmulWorkload",
    "Occupant",
    "Parameters",
    "PrefixG",
    "Request",
    "RequestsG",
    "Schedule",
    "ScheduleError",
    "SpeculativeG",
    "SpeculativeTrace",
    "ToyLM",
    "TracedDefinition",
    "Tracer",
    "TracerError",
    "TracerGate",
    "Wire",
    "Wires",
    "WordMatrix",
    "compile_demo_g",
    "compile_matmul",
    "expected_dot_outputs",
    "expected_matmul_outputs",
    "gate_budget",
    "make_demo_request",
    "matmul_expected_matrices",
    "random_parameters",
    "reference_generate",
    "reference_speculative",
    "schedule_fcfs",
    "top_k_route",
]
