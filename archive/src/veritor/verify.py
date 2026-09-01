"""
Verification functionality for the Compute Accounting Protocol.

This module implements the core verify() function that executes claims
via IREE and checks them using verification tests.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import iree.compiler
import iree.runtime
import numpy as np

from veritor.accounting import Claim


@dataclass
class VerificationResult:
    """Result of a verification test."""

    passed: bool
    scores: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        """Allow using result in boolean context."""
        return self.passed


class VerificationTest(ABC):
    """
    Base class for verification tests.

    A verification test takes two lists of tensors (computed vs claimed) and
    determines if they are consistent according to some criteria.
    """

    @abstractmethod
    def check(
        self,
        computed_outputs: list[np.ndarray],
        claimed_outputs: list[np.ndarray],
        config: dict[str, Any],
    ) -> VerificationResult:
        """
        Check consistency between computed and claimed outputs.

        Args:
            computed_outputs: What the verifier computed
            claimed_outputs: What the prover claimed
            config: Test-specific configuration

        Returns:
            VerificationResult indicating pass/fail and detailed scores
        """
        pass

    @staticmethod
    @abstractmethod
    def default_config() -> dict[str, Any]:
        """Return default configuration for this test."""
        pass


class BitExactTest(VerificationTest):
    """
    Test for deterministic computations.

    For floating point: uses allclose with configurable tolerances
    For integers: requires exact equality
    """

    @staticmethod
    def default_config() -> dict[str, Any]:
        return {
            "rtol": 1e-6,
            "atol": 1e-6,
        }

    def check(
        self,
        computed_outputs: list[np.ndarray],
        claimed_outputs: list[np.ndarray],
        config: dict[str, Any],
    ) -> VerificationResult:
        """Check if outputs match within tolerance."""
        if len(computed_outputs) != len(claimed_outputs):
            return VerificationResult(
                passed=False,
                details={
                    "error": f"Output count mismatch: {len(computed_outputs)} vs {len(claimed_outputs)}"
                },
            )

        rtol = config.get("rtol", 1e-6)
        atol = config.get("atol", 1e-6)

        per_output_scores = []
        all_passed = True

        for i, (computed, claimed) in enumerate(zip(computed_outputs, claimed_outputs)):
            # For integer types, require exact equality
            if np.issubdtype(computed.dtype, np.integer):
                exact_match = bool(np.array_equal(computed, claimed))
                per_output_scores.append(
                    {
                        "output_idx": i,
                        "dtype": str(computed.dtype),
                        "exact_match": exact_match,
                        "passed": exact_match,
                    }
                )
                if not exact_match:
                    all_passed = False

            # For float types, use allclose
            else:
                max_abs_diff = float(np.max(np.abs(computed - claimed)))
                max_rel_diff = float(
                    np.max(np.abs((computed - claimed) / (np.abs(claimed) + 1e-10)))
                )
                within_tolerance = bool(
                    np.allclose(computed, claimed, rtol=rtol, atol=atol)
                )

                per_output_scores.append(
                    {
                        "output_idx": i,
                        "dtype": str(computed.dtype),
                        "max_abs_diff": max_abs_diff,
                        "max_rel_diff": max_rel_diff,
                        "atol": atol,
                        "rtol": rtol,
                        "passed": within_tolerance,
                    }
                )
                if not within_tolerance:
                    all_passed = False

        return VerificationResult(
            passed=all_passed,
            scores={"per_output": per_output_scores},
            details={"num_outputs": len(computed_outputs)},
        )


class StampTest(VerificationTest):
    """
    Test for STAMP protocol (inference verification via LSH).

    Expected outputs:
    - Output 0: logits at each position
    - Outputs 1+: LSH fingerprints of hidden states

    Claimed outputs:
    - Output 0: tokens (for surprisal check)
    - Outputs 1+: LSH fingerprints
    """

    @staticmethod
    def default_config() -> dict[str, Any]:
        return {
            "lsh_tolerance": 0.1,
            "surprisal_threshold": 10.0,
        }

    def check(
        self,
        computed_outputs: list[np.ndarray],
        claimed_outputs: list[np.ndarray],
        config: dict[str, Any],
    ) -> VerificationResult:
        """Check LSH fingerprints and token surprisal."""
        lsh_tolerance = config.get("lsh_tolerance", 0.1)
        surprisal_threshold = config.get("surprisal_threshold", 10.0)

        # TODO: Implement full STAMP logic
        # For now, return placeholder
        return VerificationResult(
            passed=False, details={"error": "STAMP test not yet implemented"}
        )


class TrainingReplayTest(VerificationTest):
    """
    Test for training verification via checkpoint replay.

    Expected outputs:
    - Outputs 0..N-1: Gradient LSH fingerprints
    - Output N (optional): Updated checkpoint weights
    """

    @staticmethod
    def default_config() -> dict[str, Any]:
        return {
            "gradient_lsh_tolerance": 0.1,
            "checkpoint_rtol": 1e-3,
            "checkpoint_atol": 1e-5,
        }

    def check(
        self,
        computed_outputs: list[np.ndarray],
        claimed_outputs: list[np.ndarray],
        config: dict[str, Any],
    ) -> VerificationResult:
        """Check gradient LSH and optional checkpoint weights."""
        # TODO: Implement full training verification logic
        # For now, return placeholder
        return VerificationResult(
            passed=False,
            details={"error": "TrainingReplay test not yet implemented"},
        )


# Registry of available tests
TEST_REGISTRY: dict[str, type[VerificationTest]] = {
    "bit_exact": BitExactTest,
    "stamp": StampTest,
    "training_replay": TrainingReplayTest,
}


def get_test(name: str) -> VerificationTest:
    """Get a verification test instance by name."""
    if name not in TEST_REGISTRY:
        raise ValueError(
            f"Unknown test: {name}. Available: {list(TEST_REGISTRY.keys())}"
        )
    return TEST_REGISTRY[name]()


def run_test(
    test_name: str,
    computed_outputs: list[np.ndarray],
    claimed_outputs: list[np.ndarray],
    config: dict[str, Any] | None = None,
) -> VerificationResult:
    """
    Convenience function to run verification test.

    Args:
        test_name: Name of test to use
        computed_outputs: What the verifier computed
        claimed_outputs: What the prover claimed
        config: Optional configuration (uses defaults if not provided)

    Returns:
        VerificationResult
    """
    test = get_test(test_name)
    if config is None:
        config = test.default_config()
    return test.check(computed_outputs, claimed_outputs, config)


def extract_module_name(graph: str) -> Optional[str]:
    """Extract module name from StableHLO MLIR text."""
    match = re.search(r"module @(\w+)", graph)
    return match.group(1) if match else None


def verify(claim: Claim) -> VerificationResult:
    """
    Verify a claim by compiling and executing its StableHLO graph via IREE.

    This follows the verification workflow from SPEC.md:
    1. Compile claim.graph using IREE
    2. Execute with claim.inputs
    3. Compare computed outputs against claim.outputs using the specified test
    4. Return VerificationResult with pass/fail and detailed scores

    Args:
        claim: The Claim to verify

    Returns:
        VerificationResult indicating pass/fail and detailed scores

    Raises:
        ValueError: If the claim has an unknown test or invalid structure
    """
    # Validate claim first
    if not claim.validate():
        raise ValueError("Invalid claim structure")

    try:
        # Compile StableHLO to IREE bytecode
        compiled = iree.compiler.compile_str(
            claim.graph, input_type="stablehlo", target_backends=["llvm-cpu"]
        )

        # Execute via IREE runtime
        config = iree.runtime.Config("local-task")
        ctx = iree.runtime.SystemContext(config=config)
        vm_module = iree.runtime.VmModule.copy_buffer(ctx.instance, compiled)
        ctx.add_vm_module(vm_module)

        # Extract module name from MLIR
        module_name = extract_module_name(claim.graph)
        if not module_name:
            raise ValueError("Could not find module name in graph")

        # Invoke specified entry point
        function = ctx.modules[module_name][claim.entry_point]
        results = function(*claim.inputs)
        if not isinstance(results, tuple):
            results = (results,)
        computed_outputs = [np.asarray(r) for r in results]

        # Get verification test and config
        test = get_test(claim.test_spec["name"])
        config = claim.test_spec.get("config", test.default_config())

        # Run verification test
        return test.check(computed_outputs, claim.outputs, config)

    except Exception as e:
        # Log error for debugging and return failed result
        print(f"Verification failed with error: {e}")
        return VerificationResult(passed=False, details={"error": str(e)})