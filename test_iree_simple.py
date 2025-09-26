#!/usr/bin/env python3
"""
Simple test of IREE for StableHLO transformation.
"""

import jax
import jax.numpy as jnp
import iree.compiler as compiler
import iree.runtime as runtime
import tempfile


def test_iree_compilation():
    """Test compiling StableHLO with IREE."""

    print("=" * 80)
    print("TESTING IREE STABLEHLO COMPILATION")
    print("=" * 80)

    # Simple function
    def simple_add(x, y):
        return x + y

    # Get StableHLO
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([4.0, 5.0, 6.0])

    jitted = jax.jit(simple_add)
    lowered = jitted.lower(x, y)
    stablehlo_text = lowered.as_text()

    print("\n1. Original StableHLO:")
    print(stablehlo_text[:500] + "...")

    # Save to file (IREE prefers files)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as f:
        f.write(stablehlo_text)
        input_file = f.name

    print(f"\n2. Saved StableHLO to: {input_file}")

    # Compile with IREE
    print("\n3. Compiling with IREE...")

    try:
        # IREE compilation
        compiled = compiler.compile_file(
            input_file,
            target_backends=["llvm-cpu"],
            input_type=compiler.InputType.STABLEHLO,
        )

        print("   ✓ Compilation successful!")
        print(f"   Bytecode size: {len(compiled)} bytes")

        # Create runtime and execute
        print("\n4. Executing with IREE runtime...")

        config = runtime.Config("local-task")
        ctx = runtime.SystemContext(config=config)

        vm_module = runtime.VmModule.from_buffer(ctx.instance, compiled)
        ctx.add_vm_module(vm_module)

        # Execute
        result = ctx.modules.module.main(x, y)
        print(f"   Result: {result}")

        # Compare with JAX
        jax_result = jitted(x, y)
        print(f"   JAX result: {jax_result}")
        print(f"   Match: {jnp.allclose(result, jax_result)}")

    except Exception as e:
        print(f"   ✗ Error: {e}")

    return True


if __name__ == "__main__":
    test_iree_compilation()