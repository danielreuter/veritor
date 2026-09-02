"""Untrusted, client-side code: the tracer and the built-in constructors ``G``.

Nothing in this package is trusted.  A constructor turns a workload into
canonical description bytes; :class:`veritor.compile.Compiler` re-validates
every byte before anything downstream sees a circuit.
"""

from .demo_g import (
    BatchInput,
    DemoG,
    DemoGCompileRequest,
    DotRequest,
    compile_demo_g,
    expected_dot_outputs,
    make_demo_request,
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
from .tracer import (
    TracedDefinition,
    Tracer,
    TracerError,
    TracerGate,
    Wire,
    Wires,
)

__all__ = [
    "BatchInput",
    "DemoG",
    "DemoGCompileRequest",
    "DotRequest",
    "MatmulCompileRequest",
    "MatmulG",
    "MatmulWorkload",
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
    "make_demo_request",
    "matmul_expected_matrices",
]
