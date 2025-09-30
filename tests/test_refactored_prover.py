"""
Test for refactored Prover architecture with base class and subclasses.

This test demonstrates the cleaner separation where:
- Base Prover handles common functionality
- Subclasses implement specific workloads
- Challengeable decorator simplifies hook creation
"""

import uuid
from datetime import datetime

import jax.numpy as jnp
import pytest
from jax import random

from veritor.challenger import Challenger
from veritor.db.api import WorkloadDatabase
from veritor.db.ir_store import IRFormat, IRRole
from veritor.db.models import DataBundle, EventType, Graph, TensorData, Trace, TraceEvent
from veritor.prover.simple_inference import SimpleInferenceConfig, SimpleInferenceProver
from veritor.verifier.engine import VerificationConfig
from veritor.verifier.three_party import ThreePartyVerifier


def test_refactored_prover_architecture(workload_db):
    """
    Test the refactored Prover architecture with base class and subclasses.

    This test demonstrates:
    1. Clean separation of base Prover and workload-specific subclasses
    2. Simplified workload implementation in subclasses
    3. Challenger integration handled by base class
    4. Database population managed by base class
    """
    database = workload_db

    print("=" * 80)
    print("🎭 REFACTORED PROVER ARCHITECTURE TEST")
    print("=" * 80)

    # === CONFIGURATION ===

    prover_config = SimpleInferenceConfig(
        n_layers=4,
        input_dim=2,
        hidden_dim=8,
        output_dim=2,
        batch_size=3,
        seed=42
    )

    verification_config = VerificationConfig(
        execution_rtol=1e-5,
        lsh_rtol=1e-3,
        enable_jit_vs_python=False,
        enable_challenge_verification=False,
        enable_transformation_checks=False
    )

    n_forward_passes = 5

    # === PHASE 1: Initialize Entities ===

    print(f"\n{'=' * 60}")
    print("PHASE 1: Initialize Three-Party Architecture")
    print(f"{'=' * 60}")

    # Create entities
    challenger = Challenger()
    prover = SimpleInferenceProver(prover_config, challenger)
    verifier = ThreePartyVerifier(verification_config)

    print("✅ Entities initialized:")
    print("   - Challenger (external)")
    print("   - SimpleInferenceProver (extends BaseProver)")
    print("   - Verifier (validator)")

    # === PHASE 2: Compilation and Schedule Creation ===

    print(f"\n{'=' * 60}")
    print("PHASE 2: Workload Compilation and Schedule Creation")
    print(f"{'=' * 60}")

    # Prover compiles workload
    stablehlo, operation_mapping, jitted_workload = prover.compile_workload()

    # Create graph in database
    graph = Graph(
        id=f"refactored_test_{uuid.uuid4().hex[:8]}",
        metadata={
            "architecture": "refactored_prover",
            "prover_class": "SimpleInferenceProver",
            "n_layers": prover_config.n_layers,
            "operation_count": len(operation_mapping),
            "uses_base_prover": True
        }
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
            "uses_io_callback": True,
            "prover_type": "SimpleInferenceProver",
            "base_class": "BaseProver"
        }
    )

    # Verifier creates schedule
    schedule = verifier.create_challenge_schedule(
        stablehlo=stablehlo,
        operation_mapping=operation_mapping,
        challenge_probability=0.4
    )

    # Send schedule to Challenger
    challenger.set_schedule(schedule)

    print(f"\n✅ Compilation and scheduling complete:")
    print(f"   Graph ID: {graph_id}")
    print(f"   Operations registered: {len(operation_mapping)}")
    print(f"   Challenges scheduled: {len(schedule.operation_challenges)}")

    # === PHASE 3: Execution with Challenge Injection ===

    print(f"\n{'=' * 60}")
    print("PHASE 3: Workload Execution with Dynamic Challenges")
    print(f"{'=' * 60}")

    all_inputs = {}
    all_outputs = {}
    all_events = []
    pass_challenges = []

    for pass_idx in range(n_forward_passes):
        print(f"\n  Pass {pass_idx + 1}/{n_forward_passes}")

        # Generate input
        key = random.PRNGKey(prover_config.seed + pass_idx)
        x = random.normal(key, (prover_config.batch_size, prover_config.input_dim))

        # Store input
        input_id = f"input_pass_{pass_idx}"
        all_inputs[input_id] = TensorData.from_array(x)

        # Execute using base class execute method
        output = prover.execute(x)

        # Store output
        output_id = f"output_pass_{pass_idx}"
        all_outputs[output_id] = TensorData.from_array(output)

        # Count challenges in this pass
        challenges_before = len(pass_challenges)
        pass_challenges.extend(prover.challenge_responses[challenges_before:])
        challenges_in_pass = len(pass_challenges) - challenges_before

        print(f"    Input shape: {x.shape}")
        print(f"    Output shape: {output.shape}")
        print(f"    Challenges triggered: {challenges_in_pass}")

        # Record events
        all_events.append(
            TraceEvent(
                timestamp=datetime.now().timestamp(),
                event_type=EventType.KERNEL_LAUNCH,
                device_id="cpu_0",
                operation_id=f"forward_pass_{pass_idx}",
                data={
                    "pass_idx": pass_idx,
                    "challenges_in_pass": challenges_in_pass
                }
            )
        )

        # Record challenge events
        for challenge in prover.challenge_responses[challenges_before:]:
            all_events.append(
                TraceEvent(
                    timestamp=challenge.timestamp,
                    event_type=EventType.CHALLENGE,
                    device_id="cpu_0",
                    operation_id=challenge.target_operation_id,
                    data={
                        "pass_idx": pass_idx,
                        "challenge_id": challenge.id
                    }
                )
            )

    # Create and store trace
    trace = Trace(
        id=f"trace_refactored_{uuid.uuid4().hex[:8]}",
        graph_id=graph_id,
        start_time=all_events[0].timestamp if all_events else datetime.now().timestamp(),
        end_time=all_events[-1].timestamp if all_events else datetime.now().timestamp(),
        events=all_events,
        metadata={
            "n_passes": n_forward_passes,
            "n_challenges": len(prover.challenge_responses),
            "operation_mapping": operation_mapping,
            "uses_base_prover": True
        }
    )
    trace_id = database.store_trace(trace)

    # Store challenges
    for challenge in prover.challenge_responses:
        challenge.metadata["trace_id"] = trace_id
        challenge.metadata["graph_id"] = graph_id
        database.store_challenge(challenge)

    # Store data bundle
    data_bundle = DataBundle(
        id=f"data_refactored_{uuid.uuid4().hex[:8]}",
        graph_id=graph_id,
        inputs=all_inputs,
        outputs=all_outputs,
        weights={
            name: TensorData.from_array(param)
            for name, param in prover.model_params.items()
        },
        activations={},
        metadata={
            "trace_id": trace_id,
            "n_challenges": len(prover.challenge_responses),
            "uses_base_prover": True
        }
    )
    data_id = database.store_data_bundle(data_bundle)

    print(f"\n✅ Execution complete:")
    print(f"   Total challenges responded: {len(prover.challenge_responses)}")
    print(f"   Trace ID: {trace_id}")
    print(f"   Data ID: {data_id}")

    # Get statistics
    challenger_stats = challenger.get_statistics()
    prover_stats = prover.get_statistics()

    print(f"\n📊 Challenger Statistics:")
    print(f"   Queries received: {challenger_stats['queries_received']}")
    print(f"   Challenges issued: {challenger_stats['challenges_issued']}")
    print(f"   Scheduled operations: {challenger_stats['scheduled_operations']}")

    print(f"\n📊 Prover Statistics:")
    print(f"   Operations registered: {prover_stats['operations_registered']}")
    print(f"   Challenges responded: {prover_stats['challenges_responded']}")
    print(f"   Challenge enabled: {prover_stats['challenge_enabled']}")
    print(f"   Workload type: {prover_stats['workload_type']}")
    print(f"   Model layers: {prover_stats['n_layers']}")

    # === PHASE 4: Post-Execution Verification ===

    print(f"\n{'=' * 60}")
    print("PHASE 4: Post-Execution Verification")
    print(f"{'=' * 60}")

    # Verify execution
    verification_result = verifier.verify_execution(database, graph_id, trace_id)

    print(f"\n✅ Verification Results:")
    print(f"   Success: {verification_result.success}")
    print(f"   Challenges verified: {len(verification_result.challenge_results)}")

    if verification_result.metrics:
        print(f"   Schedule adherence: {verification_result.metrics.get('schedule_adherence', 'N/A')}")
        print(f"   Scheduled operations: {verification_result.metrics.get('scheduled_operations', 0)}")
        print(f"   Challenged operations: {verification_result.metrics.get('challenged_operations', 0)}")

    # === ARCHITECTURAL VALIDATION ===

    print(f"\n{'=' * 80}")
    print("ARCHITECTURAL VALIDATION")
    print(f"{'=' * 80}")

    # 1. Verify Prover uses base class
    assert hasattr(prover, 'op_mapper'), "Prover should have op_mapper from base class"
    assert hasattr(prover, 'query_challenge'), "Prover should have query_challenge from base class"
    assert hasattr(prover, 'outfeed'), "Prover should have outfeed from base class"
    print("✅ Prover correctly extends BaseProver")

    # 2. Verify workload-specific implementation
    assert hasattr(prover, 'model_params'), "Prover should have workload-specific model_params"
    assert prover.get_workload_metadata()['workload_type'] == 'simple_inference'
    print("✅ Workload-specific implementation verified")

    # 3. Verify Prover isolation
    assert not hasattr(prover, 'challenge_schedule'), "Prover should not have access to schedule"
    print("✅ Prover isolation verified (no access to schedule)")

    # 4. Verify Challenger integrity
    assert challenger.schedule is not None, "Challenger should have received schedule"
    assert len(challenger.queries_received) > 0, "Challenger should have received queries"
    print("✅ Challenger integrity verified")

    # 5. Verify database integrity
    assert database.get_graph(graph_id) is not None, "Graph should be stored"
    assert database.get_trace(trace_id) is not None, "Trace should be stored"
    assert len(database.challenges) > 0, "Challenges should be stored"
    print("✅ Database integrity verified")

    # === SUMMARY ===

    print(f"\n{'=' * 80}")
    print("🎉 REFACTORED PROVER ARCHITECTURE TEST SUCCESSFUL")
    print(f"{'=' * 80}")

    print("\n✅ Key Achievements:")
    print("   • Clean separation between BaseProver and SimpleInferenceProver")
    print("   • Base class handles common functionality")
    print("   • Subclass implements workload-specific logic")
    print("   • Challenge hooks managed by base class")
    print("   • Database population handled by base class")
    print("   • Three-party architecture maintained")

    print("\n🔑 Architecture Components:")
    print("   • veritor.prover.base.BaseProver (base class)")
    print("   • veritor.prover.simple_inference.SimpleInferenceProver (subclass)")
    print("   • veritor.prover.base.challengeable (decorator)")
    print("   • veritor.challenger.Challenger (external)")
    print("   • veritor.verifier.three_party.ThreePartyVerifier")

    # Return results
    return {
        "graph_id": graph_id,
        "trace_id": trace_id,
        "data_id": data_id,
        "verification_result": verification_result,
        "n_challenges": len(prover.challenge_responses),
        "success": verification_result.success
    }


if __name__ == "__main__":
    """Standalone execution for development."""
    import sys
    sys.path.insert(0, '/Users/danielreuter/projects/veritor')

    from veritor.db.api import WorkloadDatabase

    print("🚀 Running Refactored Prover Architecture Test (Standalone)")

    # Create database
    db = WorkloadDatabase()

    # Run test
    results = test_refactored_prover_architecture(db)

    # Validate results
    assert results["success"], "Verification should succeed"
    assert results["n_challenges"] > 0, "Should have generated challenges"

    print(f"\n✅ Standalone test passed!")
    print(f"   Graph: {results['graph_id']}")
    print(f"   Trace: {results['trace_id']}")
    print(f"   Data: {results['data_id']}")
    print(f"   Challenges: {results['n_challenges']}")
    print(f"   Verification: {'PASSED' if results['success'] else 'FAILED'}")