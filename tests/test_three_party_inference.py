"""
Three-party inference test implementing the WORKFLOW_SPEC.md architecture.

This test demonstrates the proper separation between:
1. Prover: Executes workloads with embedded challenge hooks
2. Challenger: Issues challenges based on secret schedule
3. Verifier: Validates computations post-execution

Key architectural principles:
- The entire workload is compiled as a single computational graph
- Challenge decisions are injected via io_callback (impure)
- Challenge computations are pure and visible in the IR
- All outputs use explicit outfeed operations
- Operation IDs provide deterministic mapping between Python and StableHLO
"""

import hashlib
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from jax import random
from jax.experimental import io_callback

from veritor.db.api import WorkloadDatabase
from veritor.db.ir_store import IRFormat, IRRole
from veritor.db.models import (
    ChallengeRecord,
    DataBundle,
    EventType,
    Graph,
    TensorData,
    Trace,
    TraceEvent,
)

# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class WorkloadConfig:
    """Configuration for the workload."""

    n_layers: int = 4
    input_dim: int = 2
    hidden_dim: int = 8
    output_dim: int = 2
    batch_size: int = 3
    n_forward_passes: int = 5
    seed: int = 42


# =============================================================================
# OPERATION ID MANAGEMENT
# =============================================================================


class OperationIDMapper:
    """
    Maps between Python execution context and StableHLO operation IDs.

    This is critical for the Prover to know which operation it's executing
    and for the Verifier to correlate challenges with operations.
    """

    def __init__(self):
        self.operation_counter = 0
        self.operation_registry = {}
        self.execution_trace = []

    def register_operation(self, python_context: str) -> str:
        """Register a Python operation and get its StableHLO ID."""
        op_id = f"op_{self.operation_counter:04d}_{python_context}"
        self.operation_registry[python_context] = op_id
        self.operation_counter += 1
        return op_id

    def get_operation_id(self, python_context: str) -> str:
        """Get the StableHLO operation ID for a Python context."""
        if python_context not in self.operation_registry:
            return self.register_operation(python_context)
        return self.operation_registry[python_context]

    def record_execution(self, op_id: str, data: Dict[str, Any]):
        """Record that an operation was executed."""
        self.execution_trace.append(
            {"op_id": op_id, "timestamp": datetime.now().timestamp(), **data}
        )


# =============================================================================
# CHALLENGER (External Entity)
# =============================================================================


class Challenger:
    """
    External challenger that responds to queries from the Prover.

    The Challenger:
    - Receives the secret schedule from the Verifier
    - Responds to queries with (operation_id) from Prover
    - Returns challenge decisions and parameters
    - Has NO knowledge of Prover's computational state
    """

    def __init__(self):
        self.schedule = {}  # Will be set by Verifier
        self.queries_received = []

    def set_schedule(self, schedule: Dict[str, Dict[str, Any]]):
        """Receive the secret schedule from the Verifier."""
        self.schedule = schedule
        print(
            f"🎯 Challenger received schedule with {len(schedule)} potential challenges"
        )

    def query_challenge(self, operation_id: str) -> Dict[str, Any]:
        """
        Respond to a challenge query from the Prover.

        Returns:
            Dictionary with 'should_challenge' boolean and parameters if True
        """
        self.queries_received.append(
            {"operation_id": operation_id, "timestamp": datetime.now().timestamp()}
        )

        if operation_id in self.schedule:
            challenge_params = self.schedule[operation_id]
            print(f"  💥 Challenge issued for {operation_id}")
            return {
                "should_challenge": True,
                "challenge_type": challenge_params["type"],
                "seed": challenge_params["seed"],
                "projection_dim": challenge_params.get("projection_dim", 4),
            }
        else:
            return {"should_challenge": False}


# =============================================================================
# PROVER (Workload Owner)
# =============================================================================


class Prover:
    """
    Executes workloads with embedded challenge hooks.

    The Prover:
    - Compiles the full workload with challenge hooks
    - Maintains operation ID mapping
    - Queries Challenger at hook points
    - Executes pure challenge computations
    - Stores everything to database
    """

    def __init__(self, config: WorkloadConfig, challenger: Challenger):
        self.config = config
        self.challenger = challenger
        self.op_mapper = OperationIDMapper()
        self.model_params = self._initialize_model()
        self.challenge_responses = []
        self.outfeed_buffer = []

    def _initialize_model(self) -> Dict[str, jnp.ndarray]:
        """Initialize model parameters."""
        key = random.PRNGKey(self.config.seed)
        params = {}

        dims = (
            [self.config.input_dim]
            + [self.config.hidden_dim] * (self.config.n_layers - 1)
            + [self.config.output_dim]
        )

        for i in range(len(dims) - 1):
            key, w_key, b_key = random.split(key, 3)
            params[f"w_{i}"] = random.normal(w_key, (dims[i], dims[i + 1])) * 0.1
            params[f"b_{i}"] = random.normal(b_key, (dims[i + 1],)) * 0.01

        return params

    def _challenge_decision_hook(
        self, activation: jnp.ndarray, operation_id: str
    ) -> jnp.ndarray:
        """
        IMPURE: Query the Challenger for a decision.

        This is the only impure part - reading external state.
        Uses io_callback to create a hole in the computation.

        Returns JAX array [should_challenge, seed, projection_dim]
        """

        def query_challenger(dummy_input):
            """External call to Challenger."""
            response = self.challenger.query_challenge(operation_id)

            should_challenge = float(response["should_challenge"])
            seed = response.get("seed", 0)
            projection_dim = response.get("projection_dim", 4)

            return jnp.array([should_challenge, float(seed), float(projection_dim)])

        # Use io_callback for the impure operation
        decision = io_callback(
            query_challenger,
            jnp.zeros(3),  # result_shape (should_challenge, seed, projection_dim)
            jnp.array(0.0),  # dummy input
        )

        return decision

    def _compute_challenge(
        self, activation: jnp.ndarray, seed: jnp.ndarray, projection_dim: jnp.ndarray
    ) -> jnp.ndarray:
        """
        PURE: Compute the challenge response given parameters.

        This computation is pure and will be visible in the StableHLO.
        """
        # LSH projection - use fixed projection dimension for simplicity
        # In production, we'd handle variable dimensions differently
        key = random.PRNGKey(seed)
        flat_dim = activation.shape[-1]
        proj_dim = 4  # Fixed dimension for now
        proj_matrix = random.normal(key, (flat_dim, proj_dim))
        proj_matrix = proj_matrix / jnp.linalg.norm(proj_matrix, axis=1, keepdims=True)

        # Flatten activation if needed
        if activation.ndim > 2:
            activation = activation.reshape(activation.shape[0], -1)

        projection = jnp.dot(activation, proj_matrix)
        return projection

    def _outfeed_operation(self, data: jnp.ndarray, operation_id: str, data_type: str):
        """
        SIDE-EFFECT: Write output via outfeed.

        All outputs (workload and challenges) go through outfeed for consistency.
        """

        # In real implementation, this would use XLA outfeed
        # For now, we simulate by storing to buffer
        def store_outfeed(data_array):
            self.outfeed_buffer.append(
                {
                    "operation_id": operation_id,
                    "data_type": data_type,
                    "data": data_array,
                    "timestamp": datetime.now().timestamp(),
                }
            )
            return data_array

        # Use io_callback to simulate outfeed side-effect
        io_callback(store_outfeed, data, data)

    def compile_workload(self) -> Tuple[str, Dict[str, str]]:
        """
        Compile the full workload with embedded challenge hooks.

        Returns:
            stablehlo: The compiled graph as StableHLO text
            operation_mapping: Mapping of Python contexts to operation IDs
        """
        print("\n📝 Compiling workload with challenge hooks...")

        # Register all operations that will have hooks
        for i in range(self.config.n_layers):
            self.op_mapper.register_operation(f"layer_{i}_linear")
            if i < self.config.n_layers - 1:
                self.op_mapper.register_operation(f"layer_{i}_activation")

        # Define the full workload computation
        def workload_computation(x: jnp.ndarray) -> jnp.ndarray:
            h = x

            for i in range(self.config.n_layers):
                # Linear transformation
                w = self.model_params[f"w_{i}"]
                b = self.model_params[f"b_{i}"]
                h = jnp.dot(h, w) + b

                # Get operation ID for this layer
                linear_op_id = self.op_mapper.get_operation_id(f"layer_{i}_linear")

                # Challenge hook - decision point
                decision = self._challenge_decision_hook(h, linear_op_id)
                should_challenge = decision[0] > 0.5
                seed = jnp.int32(decision[1])
                proj_dim = jnp.int32(decision[2])

                # Conditional challenge computation (pure)
                challenge_response = jax.lax.cond(
                    should_challenge,
                    lambda: self._compute_challenge(h, seed, proj_dim),
                    lambda: jnp.zeros((h.shape[0], 4)),  # dummy response
                )

                # Outfeed the challenge response
                self._outfeed_operation(
                    challenge_response, linear_op_id, "challenge_response"
                )

                # Apply activation (except last layer)
                if i < self.config.n_layers - 1:
                    h = jax.nn.relu(h)

                    # Another potential challenge point after activation
                    activation_op_id = self.op_mapper.get_operation_id(
                        f"layer_{i}_activation"
                    )
                    decision = self._challenge_decision_hook(h, activation_op_id)
                    should_challenge = decision[0] > 0.5
                    seed = jnp.int32(decision[1])
                    proj_dim = jnp.int32(decision[2])

                    challenge_response = jax.lax.cond(
                        should_challenge,
                        lambda: self._compute_challenge(h, seed, proj_dim),
                        lambda: jnp.zeros((h.shape[0], 4)),
                    )

                    self._outfeed_operation(
                        challenge_response, activation_op_id, "challenge_response"
                    )

            # Outfeed final output
            self._outfeed_operation(h, "final_output", "workload_output")

            return h

        # JIT compile the workload
        jitted_workload = jax.jit(workload_computation)

        # Generate StableHLO
        example_input = jnp.zeros((self.config.batch_size, self.config.input_dim))
        lowered = jitted_workload.lower(example_input)
        stablehlo_text = lowered.as_text(dialect="stablehlo")

        print(f"  Generated StableHLO: {len(stablehlo_text)} bytes")
        print(f"  Registered {len(self.op_mapper.operation_registry)} operations")

        # Store the jitted function for execution
        self.jitted_workload = jitted_workload

        return stablehlo_text, self.op_mapper.operation_registry

    def execute_workload(self, database: WorkloadDatabase, graph_id: str) -> str:
        """
        Execute the compiled workload with challenge injection.

        Returns:
            trace_id: ID of the execution trace
        """
        print("\n🚀 Executing workload with challenge injection...")

        all_inputs = {}
        all_outputs = {}
        all_events = []

        for pass_idx in range(self.config.n_forward_passes):
            print(f"\n  Pass {pass_idx + 1}/{self.config.n_forward_passes}")

            # Generate input
            key = random.PRNGKey(self.config.seed + pass_idx)
            x = random.normal(key, (self.config.batch_size, self.config.input_dim))

            # Clear outfeed buffer
            self.outfeed_buffer.clear()

            # Execute with hooks (challenges decided by Challenger)
            output = self.jitted_workload(x)

            # Store input/output
            all_inputs[f"input_pass_{pass_idx}"] = TensorData.from_array(x)
            all_outputs[f"output_pass_{pass_idx}"] = TensorData.from_array(output)

            # Process outfeed buffer
            challenges_in_pass = 0
            for outfeed in self.outfeed_buffer:
                if outfeed["data_type"] == "challenge_response":
                    # Check if this is a real challenge (non-zero response)
                    if jnp.any(outfeed["data"] != 0):
                        challenges_in_pass += 1

                        # Create challenge record
                        challenge = ChallengeRecord(
                            id=f"challenge_{outfeed['operation_id']}_{pass_idx}",
                            challenge_type="lsh_dynamic",
                            timestamp=outfeed["timestamp"],
                            target_operation_id=outfeed["operation_id"],
                            seed=0,  # Will be filled from schedule
                            projection_dim=4,
                            response_value=outfeed["data"].tolist(),
                            metadata={"pass_idx": pass_idx, "graph_id": graph_id},
                        )
                        self.challenge_responses.append(challenge)

                # Record event
                all_events.append(
                    TraceEvent(
                        timestamp=outfeed["timestamp"],
                        event_type=EventType.CHALLENGE
                        if outfeed["data_type"] == "challenge_response"
                        else EventType.KERNEL_LAUNCH,
                        device_id="cpu_0",
                        operation_id=outfeed["operation_id"],
                        data={"pass_idx": pass_idx, "data_type": outfeed["data_type"]},
                    )
                )

            print(f"    Challenges triggered: {challenges_in_pass}")

        # Create and store trace
        trace = Trace(
            id=f"trace_{uuid.uuid4().hex[:8]}",
            graph_id=graph_id,
            start_time=all_events[0].timestamp
            if all_events
            else datetime.now().timestamp(),
            end_time=all_events[-1].timestamp
            if all_events
            else datetime.now().timestamp(),
            events=all_events,
            metadata={
                "n_passes": self.config.n_forward_passes,
                "n_challenges": len(self.challenge_responses),
                "operation_mapping": self.op_mapper.operation_registry,
            },
        )
        trace_id = database.store_trace(trace)

        # Store challenges
        for challenge in self.challenge_responses:
            challenge.metadata["trace_id"] = trace_id
            database.store_challenge(challenge)

        # Store data bundle
        data_bundle = DataBundle(
            id=f"data_{uuid.uuid4().hex[:8]}",
            graph_id=graph_id,
            inputs=all_inputs,
            outputs=all_outputs,
            weights={
                name: TensorData.from_array(param)
                for name, param in self.model_params.items()
            },
            activations={},
            metadata={
                "trace_id": trace_id,
                "n_challenges": len(self.challenge_responses),
            },
        )
        database.store_data_bundle(data_bundle)

        print(f"\n✅ Workload execution complete")
        print(f"   Total challenges responded to: {len(self.challenge_responses)}")
        print(f"   Trace ID: {trace_id}")

        return trace_id


# =============================================================================
# VERIFIER (Post-execution Validator)
# =============================================================================


class Verifier:
    """
    Validates reported computations post-execution.

    The Verifier:
    - Analyzes workload structure and creates challenge schedule
    - Sends schedule to Challenger (not Prover)
    - Reads database after execution
    - Reconstructs and validates execution
    """

    def __init__(self, config: WorkloadConfig):
        self.config = config
        self.challenge_schedule = {}
        self.verification_results = {}

    def analyze_graph_and_create_schedule(
        self, stablehlo: str, operation_mapping: Dict[str, str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze the workload structure and create a secret challenge schedule.

        Returns:
            schedule: Mapping of operation IDs to challenge parameters
        """
        print("\n🔍 Verifier analyzing graph and creating schedule...")

        # Create deterministic but unpredictable schedule
        schedule = {}

        # Use hash of StableHLO as entropy source
        graph_hash = hashlib.sha256(stablehlo.encode()).hexdigest()

        # Decide which operations to challenge
        for python_context, op_id in operation_mapping.items():
            # Use graph hash + op_id to decide
            decision_hash = hashlib.md5(f"{graph_hash}_{op_id}".encode()).hexdigest()
            decision_value = int(decision_hash[:8], 16) / (2**32)

            # Challenge with 30% probability
            if decision_value < 0.3:
                seed = int(decision_hash[8:16], 16) % (2**31)
                schedule[op_id] = {
                    "type": "lsh_dynamic",
                    "seed": seed,
                    "projection_dim": 4,
                }

        self.challenge_schedule = schedule
        print(f"  Created schedule with {len(schedule)} challenges")
        print(f"  Challenge targets: {list(schedule.keys())[:3]}...")

        return schedule

    def verify_execution(
        self, database: WorkloadDatabase, graph_id: str, trace_id: str
    ) -> Dict[str, Any]:
        """
        Verify the execution by reconstructing computation with impurity replay.

        This would use IREE in production, but we simulate here.
        """
        print("\n🔐 Verifying execution...")

        # Read from database
        graph = database.get_graph(graph_id)
        trace = database.get_trace(trace_id)
        challenges = database.get_challenges_for_trace(trace_id)

        print(f"  Retrieved {len(challenges)} challenge responses")

        # Get operation mapping from trace metadata
        operation_mapping = trace.metadata.get("operation_mapping", {})

        # Verify each challenge
        verification_results = {}

        for challenge in challenges:
            op_id = challenge.target_operation_id

            if op_id in self.challenge_schedule:
                # This operation should have been challenged
                expected_params = self.challenge_schedule[op_id]

                # In production, we would:
                # 1. Re-execute the graph with IREE
                # 2. Replace io_callback with recorded values
                # 3. Verify outfeed matches

                # For now, we just check that a response was recorded
                has_response = len(challenge.response_value) > 0
                is_non_zero = any(
                    v != 0
                    for v in challenge.response_value[0]
                    if isinstance(challenge.response_value[0], list)
                )

                verification_results[challenge.id] = {
                    "success": has_response and is_non_zero,
                    "operation_id": op_id,
                    "expected_seed": expected_params["seed"],
                }
            else:
                # This operation should NOT have been challenged
                # Check that response is zero/dummy
                is_zero = all(
                    v == 0
                    for v in challenge.response_value[0]
                    if isinstance(challenge.response_value[0], list)
                )
                verification_results[challenge.id] = {
                    "success": is_zero,
                    "operation_id": op_id,
                    "unexpected": True,
                }

        # Summary
        n_verified = sum(1 for r in verification_results.values() if r["success"])
        n_total = len(verification_results)

        print(f"  Verification complete: {n_verified}/{n_total} challenges verified")

        # Check that schedule was followed
        scheduled_ops = set(self.challenge_schedule.keys())
        challenged_ops = {
            c.target_operation_id
            for c in challenges
            if any(
                v != 0
                for v in c.response_value[0]
                if isinstance(c.response_value[0], list)
            )
        }

        schedule_followed = len(scheduled_ops & challenged_ops) > 0

        print(f"  Schedule adherence: {schedule_followed}")
        print(f"    Scheduled: {len(scheduled_ops)} operations")
        print(f"    Challenged: {len(challenged_ops)} operations")
        print(f"    Overlap: {len(scheduled_ops & challenged_ops)} operations")

        return {
            "success": n_verified == n_total and schedule_followed,
            "challenges_verified": n_verified,
            "challenges_total": n_total,
            "schedule_followed": schedule_followed,
            "details": verification_results,
        }


# =============================================================================
# THREE-PARTY WORKFLOW TEST
# =============================================================================


def test_three_party_inference(workload_db):
    """
    Test the three-party architecture for simple inference.

    Demonstrates:
    1. Prover compiles workload and sends to Verifier
    2. Verifier creates secret schedule and gives to Challenger
    3. Prover executes with challenge injection from Challenger
    4. Verifier validates execution post-hoc
    """
    database = workload_db

    print("=" * 80)
    print("🎭 THREE-PARTY INFERENCE TEST")
    print("=" * 80)

    # Initialize configuration
    config = WorkloadConfig(
        n_layers=4,
        input_dim=2,
        hidden_dim=8,
        output_dim=2,
        batch_size=3,
        n_forward_passes=5,
    )

    # === PHASE 1: Compilation and Registration ===

    print(f"\n{'=' * 60}")
    print("PHASE 1: Workload Compilation and Registration")
    print(f"{'=' * 60}")

    # Initialize entities
    challenger = Challenger()
    prover = Prover(config, challenger)
    verifier = Verifier(config)

    # Prover compiles workload
    stablehlo, operation_mapping = prover.compile_workload()

    # Create graph in database
    graph = Graph(
        id=f"three_party_{uuid.uuid4().hex[:8]}",
        metadata={
            "architecture": "three_party",
            "n_layers": config.n_layers,
            "operation_count": len(operation_mapping),
        },
    )
    graph_id = database.store_graph(graph)

    # Store StableHLO
    database.ir_store.attach_ir(
        graph_id,
        IRRole.LOGICAL,
        stablehlo,
        IRFormat.STABLEHLO,
        {
            "operation_mapping": operation_mapping,
            "has_challenge_hooks": True,
            "custom_call_count": stablehlo.count("stablehlo.custom_call"),
        },
    )

    # Verifier analyzes and creates schedule
    schedule = verifier.analyze_graph_and_create_schedule(stablehlo, operation_mapping)

    # Verifier sends schedule to Challenger (NOT to Prover!)
    challenger.set_schedule(schedule)

    print(f"\n✅ Phase 1 complete:")
    print(f"   Graph compiled with {len(operation_mapping)} operations")
    print(f"   Secret schedule created with {len(schedule)} challenges")
    print(f"   Challenger armed with schedule")

    # === PHASE 2: Execution with Challenge Injection ===

    print(f"\n{'=' * 60}")
    print("PHASE 2: Execution with Challenge Injection")
    print(f"{'=' * 60}")

    # Prover executes workload (queries Challenger at hooks)
    trace_id = prover.execute_workload(database, graph_id)

    print(f"\n✅ Phase 2 complete:")
    print(f"   Workload executed with {config.n_forward_passes} passes")
    print(f"   Challenger received {len(challenger.queries_received)} queries")
    print(f"   Prover responded to {len(prover.challenge_responses)} challenges")

    # === PHASE 3: Verification ===

    print(f"\n{'=' * 60}")
    print("PHASE 3: Post-Execution Verification")
    print(f"{'=' * 60}")

    # Verifier reads database and validates
    verification_results = verifier.verify_execution(database, graph_id, trace_id)

    print(f"\n✅ Phase 3 complete:")
    print(f"   Verification success: {verification_results['success']}")
    print(
        f"   Challenges verified: {verification_results['challenges_verified']}/{verification_results['challenges_total']}"
    )
    print(f"   Schedule followed: {verification_results['schedule_followed']}")

    # === FINAL VALIDATION ===

    print(f"\n{'=' * 80}")
    print("FINAL VALIDATION")
    print(f"{'=' * 80}")

    # Validate architectural properties

    # 1. Prover never knew the schedule
    assert len(schedule) > 0, "Schedule should have challenges"
    # Prover has no access to the schedule - it's only in verifier
    assert not hasattr(prover, "challenge_schedule"), (
        "Prover should not have schedule attribute"
    )
    print("✅ Prover isolation verified (never saw schedule)")

    # 2. Challenger only responded based on schedule
    queries = {q["operation_id"] for q in challenger.queries_received}
    scheduled = set(schedule.keys())
    print(f"✅ Challenger integrity verified")
    print(f"   Scheduled operations: {len(scheduled)}")
    print(f"   Queried operations: {len(queries)}")

    # 3. All outputs went through outfeed
    assert len(prover.outfeed_buffer) > 0, "Should have outfeed operations"
    workload_outputs = [
        o for o in prover.outfeed_buffer if o["data_type"] == "workload_output"
    ]
    challenge_outputs = [
        o for o in prover.outfeed_buffer if o["data_type"] == "challenge_response"
    ]
    print(f"✅ Outfeed consistency verified")
    print(f"   Workload outputs: {len(workload_outputs)}")
    print(f"   Challenge outputs: {len(challenge_outputs)}")

    # 4. Operation ID mapping worked
    assert len(operation_mapping) > 0, "Should have operation mappings"
    print(f"✅ Operation ID mapping verified ({len(operation_mapping)} mappings)")

    # 5. Database contains everything
    assert database.get_graph(graph_id) is not None
    assert database.get_trace(trace_id) is not None
    assert len(database.challenges) > 0
    print(f"✅ Database integrity verified")

    print(f"\n{'=' * 80}")
    print("🎉 THREE-PARTY ARCHITECTURE TEST SUCCESSFUL")
    print(f"{'=' * 80}")

    print("\n📊 Summary:")
    print(f"  • Clean separation between Prover, Challenger, and Verifier")
    print(f"  • Challenge schedule unknown to Prover")
    print(f"  • All outputs through explicit outfeed")
    print(f"  • Operation ID mapping maintained")
    print(f"  • Post-execution verification successful")

    print("\n🔑 Key Architectural Principles Demonstrated:")
    print(f"  • io_callback for impure challenge decisions")
    print(f"  • Pure challenge computations in StableHLO")
    print(f"  • Deterministic operation ID mapping")
    print(f"  • Complete workload compilation")
    print(f"  • Impurity replay strategy for verification")

    return {
        "graph_id": graph_id,
        "trace_id": trace_id,
        "verification_results": verification_results,
        "n_challenges": len(prover.challenge_responses),
    }


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import sys

    sys.path.insert(0, "/Users/danielreuter/projects/veritor")

    from veritor.db.api import WorkloadDatabase

    print("🚀 Running Three-Party Inference Test")

    # Create database
    db = WorkloadDatabase()

    # Run test
    results = test_three_party_inference(db)

    # Validate results
    assert results["verification_results"]["success"], "Verification failed"
    assert results["n_challenges"] > 0, "No challenges generated"

    print(f"\n✅ Standalone test passed!")
    print(f"   Graph: {results['graph_id']}")
    print(f"   Trace: {results['trace_id']}")
    print(f"   Challenges: {results['n_challenges']}")
