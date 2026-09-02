"""Built-in architecture plug-ins."""

from .deepseek_v4_pro import (
    DEEPSEEK_V4_PRO_ARCHITECTURE_ID,
    DEEPSEEK_V4_PRO_NUMERICAL_PROFILE_ID,
    DEEPSEEK_V4_PRO_PLUGIN,
    DeepSeekV4ProCompileRequest,
)
from .demo_g import (
    DEMO_G_ARCHITECTURE_ID,
    DEMO_G_PLUGIN,
    BatchInput,
    DemoG,
    DemoGCompileRequest,
    DemoGPlugin,
    DotRequest,
    compile_demo_g,
    expected_dot_outputs,
    make_demo_request,
)
from .gpt2 import GPT2_ARCHITECTURE_ID, GPT2_PLUGIN, GPT2CompileRequest
from .inkling import (
    INKLING_ARCHITECTURE_ID,
    INKLING_NUMERICAL_PROFILE_ID,
    INKLING_PLUGIN,
    InklingCompileRequest,
)
from .kimi_k3 import (
    KIMI_K3_ARCHITECTURE_ID,
    KIMI_K3_NUMERICAL_PROFILE_ID,
    KIMI_K3_PLUGIN,
    KimiK3CompileRequest,
)
from .matmul import (
    MATMUL_ARCHITECTURE_ID,
    MATMUL_PLUGIN,
    MatmulCompileRequest,
    MatmulPlugin,
    compile_matmul,
    matmul_expected_matrices,
)

__all__ = [
    "DEEPSEEK_V4_PRO_ARCHITECTURE_ID",
    "DEEPSEEK_V4_PRO_NUMERICAL_PROFILE_ID",
    "DEEPSEEK_V4_PRO_PLUGIN",
    "DEMO_G_ARCHITECTURE_ID",
    "DEMO_G_PLUGIN",
    "GPT2_ARCHITECTURE_ID",
    "GPT2_PLUGIN",
    "INKLING_ARCHITECTURE_ID",
    "INKLING_NUMERICAL_PROFILE_ID",
    "INKLING_PLUGIN",
    "KIMI_K3_ARCHITECTURE_ID",
    "KIMI_K3_NUMERICAL_PROFILE_ID",
    "KIMI_K3_PLUGIN",
    "MATMUL_ARCHITECTURE_ID",
    "MATMUL_PLUGIN",
    "BatchInput",
    "DeepSeekV4ProCompileRequest",
    "DemoG",
    "DemoGCompileRequest",
    "DemoGPlugin",
    "DotRequest",
    "GPT2CompileRequest",
    "InklingCompileRequest",
    "KimiK3CompileRequest",
    "MatmulCompileRequest",
    "MatmulPlugin",
    "compile_demo_g",
    "compile_matmul",
    "expected_dot_outputs",
    "make_demo_request",
    "matmul_expected_matrices",
]
