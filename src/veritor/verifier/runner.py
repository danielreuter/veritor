"""
Verification runner - IREE/JAX re-execution engine.
"""

from typing import Any, Dict

import jax.numpy as jnp
import numpy as np


class ExecutionEngine:
    """Engine for executing bound graphs using IREE or JAX"""

    def __init__(self, backend: str = "iree", device: str = "local-task"):
        """
        Initialize execution engine.

        Args:
            backend: Execution backend ('iree' or 'jax')
            device: IREE device to use ('local-task', 'cuda', 'vulkan', etc.)
        """
        self.backend = backend
        self.device = device
        self._compiled_modules = {}  # Cache compiled modules

    def _compile_stablehlo_to_iree(self, stablehlo_text: str) -> Any:
        """Compile StableHLO text to IREE module."""
        try:
            import iree.compiler as iree_compiler
            import iree.runtime as iree_runtime
        except ImportError:
            raise ImportError(
                "IREE not installed. Install with: pip install iree-compiler iree-runtime"
            )

        # Compile StableHLO to IREE bytecode
        compile_options = [
            "--iree-hal-target-backends=llvm-cpu"
            if "cpu" in self.device or "task" in self.device
            else "--iree-hal-target-backends=cuda"
            if "cuda" in self.device
            else "--iree-hal-target-backends=vulkan-spirv",
        ]

        try:
            bytecode = iree_compiler.compile_str(
                stablehlo_text, input_type="stablehlo", extra_args=compile_options
            )
        except Exception as e:
            raise RuntimeError(f"Failed to compile StableHLO: {e}")

        # Create runtime config and module
        config = iree_runtime.Config(self.device)
        vm_module = iree_runtime.VmModule.from_flatbuffer(
            config.vm_instance, bytecode, warn_if_copy=False
        )

        # Create a session to hold the module
        session = iree_runtime.SystemContext(config=config)
        session.add_vm_module(vm_module)

        return session

    def _execute_with_iree(
        self, ir_text: str, inputs: Dict[str, jnp.ndarray]
    ) -> Dict[str, jnp.ndarray]:
        """Execute StableHLO using IREE."""
        # Get or compile module
        module_hash = hash(ir_text)
        if module_hash not in self._compiled_modules:
            self._compiled_modules[module_hash] = self._compile_stablehlo_to_iree(
                ir_text
            )

        session = self._compiled_modules[module_hash]

        # Find our compiled module (not 'hal')
        module = None
        for key in session.modules:
            if key != "hal":
                module = session.modules[key]
                break

        if module is None:
            raise RuntimeError("Could not find compiled module in session")

        # Get the main function
        if not hasattr(module, "main"):
            raise RuntimeError("Module does not have a 'main' function")

        main_function = module.main

        # Convert inputs to numpy arrays (IREE works with numpy)
        # Note: StableHLO from JAX has weights baked in, so we only pass the actual input
        # We need to identify which is the actual input vs. weights
        np_inputs = []
        for key, value in inputs.items():
            # Assume keys starting with 'w' or 'b' are weights (already baked into the module)
            if not (key.startswith("w") or key.startswith("b")):
                np_inputs.append(np.array(value))

        if not np_inputs:
            # If no obvious input found, just take the first one
            np_inputs = [np.array(list(inputs.values())[0])]

        # Execute
        result = main_function(*np_inputs)

        # Convert result back to JAX arrays
        if isinstance(result, (list, tuple)):
            outputs = {
                f"output_{i}": jnp.array(np.array(r)) for i, r in enumerate(result)
            }
        else:
            outputs = {"output_0": jnp.array(np.array(result))}

        return outputs

    def _execute_with_jax(
        self, ir_text: str, inputs: Dict[str, jnp.ndarray]
    ) -> Dict[str, jnp.ndarray]:
        """Execute using JAX directly (fallback)."""
        # This would require parsing the StableHLO and reconstructing JAX operations
        # For now, we'll raise an error
        raise NotImplementedError(
            "Direct JAX execution from StableHLO not yet implemented"
        )

    def execute(
        self, ir_text: str, inputs: Dict[str, jnp.ndarray]
    ) -> Dict[str, jnp.ndarray]:
        """
        Execute a computational graph.

        Args:
            ir_text: StableHLO IR text
            inputs: Input tensors

        Returns:
            Output tensors
        """
        # Execute based on backend
        if self.backend == "iree":
            return self._execute_with_iree(ir_text, inputs)
        elif self.backend == "jax":
            return self._execute_with_jax(ir_text, inputs)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")


class Verifier:
    """High-level verification interface"""

    def __init__(self, database):
        """
        Initialize verifier.

        Args:
            database: WorkloadDatabase instance
        """
        self.database = database
        self.engine = ExecutionEngine()

    def verify_execution(
        self, trace_id: str, tolerance: float = 1e-5
    ) -> Dict[str, Any]:
        """
        Verify a claimed execution by replaying it.

        Returns verification results including any discrepancies found.
        """
        # Get trace and associated data
        trace = self.database.get_trace(trace_id)
        if not trace:
            return {"error": "Trace not found"}

        graph = self.database.get_graph(trace.graph_id)
        if not graph:
            return {"error": "Graph not found"}

        data_bundles = self.database.get_data_for_graph(graph.id)
        if not data_bundles:
            return {"error": "No data found for graph"}

        data = data_bundles[0]  # Use first bundle for now

        # Get IR text from graph metadata or IR store
        ir_text = graph.metadata.get("ir_text")
        if not ir_text and graph.ir_blob_id:
            from veritor.db.ir_store import IRRole

            ir_text = self.database.get_graph_ir(graph.id, IRRole.LOGICAL)
            if ir_text:
                ir_text = ir_text.decode("utf-8")

        if not ir_text:
            return {"error": "No IR found for graph"}

        # Prepare inputs
        jax_inputs = {}
        for input_id, tensor_data in data.inputs.items():
            jax_inputs[input_id] = tensor_data.to_array()

        # Add weights if needed
        for weight_id, tensor_data in data.weights.items():
            jax_inputs[weight_id] = tensor_data.to_array()

        # Execute
        computed_outputs = self.engine.execute(ir_text, jax_inputs)

        # Compare outputs
        discrepancies = []
        for output_id, computed in computed_outputs.items():
            # Find corresponding claimed output
            claimed_id = output_id.replace("output_", "output")
            if claimed_id in data.outputs:
                claimed = data.outputs[claimed_id]
                computed_array = computed
                claimed_array = claimed.to_array()

                if not jnp.allclose(computed_array, claimed_array, rtol=tolerance):
                    max_diff = float(jnp.max(jnp.abs(computed_array - claimed_array)))
                    discrepancies.append({"output_id": output_id, "max_diff": max_diff})

        return {
            "trace_id": trace_id,
            "graph_id": graph.id,
            "verified": len(discrepancies) == 0,
            "discrepancies": discrepancies,
        }

    def verify_challenges(self, trace_id: str) -> Dict[str, Any]:
        """Verify challenge responses in a trace"""
        trace = self.database.get_trace(trace_id)
        if not trace:
            return {"error": "Trace not found"}

        from veritor.db.models import EventType

        challenges = [e for e in trace.events if e.event_type == EventType.CHALLENGE]

        # Verify each challenge
        failures = []
        for challenge_event in challenges:
            # This would verify LSH projections, timing constraints, etc.
            # For now, just collect the challenges
            pass

        return {
            "trace_id": trace_id,
            "num_challenges": len(challenges),
            "verified": len(failures) == 0,
            "failures": failures,
        }
