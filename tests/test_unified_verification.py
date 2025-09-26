"""
Test unified verification engine across all test types.

This test demonstrates the unified verification engine's ability to:
1. Verify entire WorkloadDatabase contents
2. Handle all verification patterns (inference, training, autoregressive, distributed)
3. Provide a single entry point for complete verification
"""

import pytest

from veritor.verifier.engine import (
    verify_workload_database,
    verify_single_execution,
    VerificationConfig,
    UnifiedVerificationEngine
)


def test_database_wide_verification(workload_db):
    """Test that the unified engine can verify an entire database."""
    database = workload_db

    # First, run some tests to populate the database
    from test_simple_inference import test_simple_inference_with_real_stablehlo
    from test_simple_training import test_simple_training_with_real_gradients

    # Run a few tests to get some data
    test_simple_inference_with_real_stablehlo(database)
    test_simple_training_with_real_gradients(database)

    print(f"Database populated with {len(database.graphs)} graphs")
    print(f"Graphs: {list(database.graphs.keys())}")

    # === Test Database-Wide Verification ===

    # Configure verification
    config = VerificationConfig(
        enable_jit_vs_python=True,
        enable_challenge_verification=False,  # Mixed workload - disable for gradient challenges
        enable_transformation_checks=True,
        execution_rtol=1e-5,
        lsh_rtol=1e-2,  # More lenient for training gradients
    )

    # Verify entire database
    results = verify_workload_database(database, config=config, sample_rate=1.0)

    print(f"\n📊 Database Verification Results:")
    print(f"   Total executions verified: {len(results)}")

    # Check results
    all_success = True
    for execution_id, result in results.items():
        success_symbol = "✅" if result.success else "❌"
        print(f"   {success_symbol} {execution_id}: {result.metadata.execution_type}")

        if not result.success:
            print(f"      Errors: {result.errors}")
            all_success = False
        else:
            challenges = len(result.challenge_results)
            if challenges > 0:
                print(f"      Challenges verified: {challenges}")
            if result.metrics:
                max_diff = result.metrics.get('max_difference', 'N/A')
                print(f"      Max execution difference: {max_diff}")

    # Overall assertions
    assert all_success, f"Some verifications failed: {[k for k, v in results.items() if not v.success]}"
    assert len(results) >= 2, f"Expected at least 2 executions, got {len(results)}"

    # Check that we got different verification types
    execution_types = {result.metadata.execution_type for result in results.values()}
    assert "direct_execution" in execution_types, "Should have direct execution verification"

    print(f"\n✅ Database-wide verification completed successfully!")
    print(f"   Execution types verified: {execution_types}")


def test_single_execution_verification(workload_db):
    """Test verifying a single execution."""
    database = workload_db

    # Run one test to get some data
    from test_simple_inference import test_simple_inference_with_real_stablehlo
    test_simple_inference_with_real_stablehlo(database)

    # Get the most recent graph and trace from the database
    graph_id = list(database.graphs.keys())[-1]
    trace_id = list(database.traces.keys())[-1]

    # Test single execution verification
    config = VerificationConfig(
        enable_jit_vs_python=True,
        enable_challenge_verification=True,
        execution_rtol=1e-5,
    )

    result = verify_single_execution(database, graph_id, trace_id, config)

    assert result.success, f"Single execution verification failed: {result.errors}"
    assert result.metadata.graph_id == graph_id
    assert result.metadata.trace_id == trace_id

    print(f"✅ Single execution verification passed for {graph_id}")


def test_verification_engine_strategy_selection(workload_db):
    """Test that the engine selects appropriate strategies."""
    database = workload_db

    # Create engine
    engine = UnifiedVerificationEngine(database)

    # Test strategy selection with different graph metadata
    from veritor.db.models import Graph
    from veritor.verifier.engine import DirectExecutionStrategy, TransformationVerificationStrategy, MultiVariantStrategy

    # Mock graph for testing strategy selection
    inference_graph = Graph(id="test_inference", metadata={"test_type": "inference"})
    ar_graph = Graph(id="test_ar", metadata={"generation_type": "autoregressive"})
    distributed_graph = Graph(id="test_dist", metadata={"test_type": "distributed_inference"})

    # Test strategy selection logic
    from veritor.verifier.engine import GraphExecutionData

    # Inference -> Direct
    inference_data = GraphExecutionData(graph=inference_graph)
    strategy = engine._select_strategy(inference_data)
    assert isinstance(strategy, DirectExecutionStrategy)

    # Autoregressive -> Transformation
    ar_data = GraphExecutionData(graph=ar_graph)
    strategy = engine._select_strategy(ar_data)
    assert isinstance(strategy, TransformationVerificationStrategy)

    # Distributed -> MultiVariant
    dist_data = GraphExecutionData(graph=distributed_graph)
    strategy = engine._select_strategy(dist_data)
    assert isinstance(strategy, MultiVariantStrategy)

    print("✅ Strategy selection working correctly")


def test_verification_config_options():
    """Test different verification configuration options."""

    # Test default config
    default_config = VerificationConfig()
    assert default_config.enable_jit_vs_python == True
    assert default_config.enable_challenge_verification == True
    assert default_config.execution_rtol == 1e-5

    # Test custom config
    custom_config = VerificationConfig(
        enable_jit_vs_python=False,
        enable_challenge_verification=True,
        execution_rtol=1e-3,
        lsh_rtol=1e-2,
        backend="jax",
        device="cpu"
    )

    assert custom_config.enable_jit_vs_python == False
    assert custom_config.execution_rtol == 1e-3
    assert custom_config.lsh_rtol == 1e-2
    assert custom_config.backend == "jax"

    print("✅ Verification configuration options working correctly")


def test_error_handling(workload_db):
    """Test error handling in verification engine."""
    database = workload_db

    # Test non-existent graph
    result = verify_single_execution(database, "nonexistent_graph", "nonexistent_trace")

    assert not result.success
    assert len(result.errors) > 0
    assert "not found" in result.errors[0].lower()

    print("✅ Error handling working correctly")


if __name__ == "__main__":
    # Run unified verification tests directly
    from veritor.db.api import WorkloadDatabase

    db = WorkloadDatabase()

    print("Running unified verification tests...")

    try:
        # Test database-wide verification
        results = test_database_wide_verification(db)

        # Test single execution verification
        single_result = test_single_execution_verification(db)

        # Test strategy selection
        test_verification_engine_strategy_selection(db)

        # Test config options
        test_verification_config_options()

        # Test error handling
        test_error_handling(db)

        print(f"\n🎉 All unified verification tests passed!")
        print(f"   Database executions: {len(results)}")
        print(f"   All verification patterns tested")

    except Exception as e:
        print(f"❌ Unified verification test failed: {e}")
        raise