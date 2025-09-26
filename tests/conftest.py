"""
Shared pytest fixtures and configuration.
"""

import pytest
import tempfile
import shutil
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Provide a temporary directory that's cleaned up after the test."""
    temp = tempfile.mkdtemp(prefix="veritor_test_")
    yield Path(temp)
    shutil.rmtree(temp, ignore_errors=True)


@pytest.fixture
def ir_store():
    """Provide a fresh IRStore instance."""
    from src.veritor.ir_store import IRStore
    return IRStore()


@pytest.fixture
def workload_db():
    """Provide a fresh WorkloadDatabase instance."""
    from src.veritor.api import WorkloadDatabase
    return WorkloadDatabase()


@pytest.fixture
def sample_stablehlo():
    """Provide sample StableHLO IR."""
    return """
    module @test {
        func.func @main(%arg0: tensor<32x64xf32>) -> tensor<32x10xf32> {
            %0 = stablehlo.constant dense<0.0> : tensor<64x10xf32>
            %1 = stablehlo.dot_general %arg0, %0,
                contracting_dims = [1] x [0] : (tensor<32x64xf32>, tensor<64x10xf32>) -> tensor<32x10xf32>
            return %1 : tensor<32x10xf32>
        }
    }
    """


@pytest.fixture
def sample_hlo():
    """Provide sample HLO IR (distributed)."""
    return """
    HloModule test, entry_computation_layout={(f32[32,64]{1,0})->f32[32,10]{1,0}}

    ENTRY main {
        %arg0 = f32[32,64]{1,0} parameter(0), sharding={devices=[2,1]0,1}
        %const = f32[64,10]{1,0} constant({...}), sharding={replicated}
        ROOT %dot = f32[32,10]{1,0} dot(%arg0, %const),
            lhs_contracting_dims={1}, rhs_contracting_dims={0},
            sharding={devices=[2,1]0,1}
    }
    """