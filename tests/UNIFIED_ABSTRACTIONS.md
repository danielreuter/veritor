# Unified Abstractions for Veritor Tests

## Core Abstractions

### 1. **ChallengeOracle** - External Unpredictable Challenge Source
```python
class ChallengeOracle:
    """External source of verification challenges that the prover cannot predict."""
```

**Key Features:**
- Uses cryptographic randomness and execution-time state
- Challenges are unknowable to prover during JIT compilation
- Adaptive challenge probabilities based on execution history
- Complete isolation from prover logic via JAX `pure_callback`

**Why It's Better:**
- **OLD**: Static challenge schedules passed at initialization (prover can predict)
- **NEW**: Dynamic decisions made during execution via IO callbacks (unpredictable)

### 2. **ProverSystem** - Unified Model Execution with Hooks
```python
class ProverSystem:
    """Unified prover system that executes models with embedded verification hooks."""
```

**Key Features:**
- Contains JAX `pure_callback` hooks that query external ChallengeOracle
- Supports all workload types (inference, training, autoregressive)
- Model-agnostic (feedforward, transformer, conv, etc.)
- Clean separation from verification logic

**Why It's Better:**
- **OLD**: Each test had its own model implementation with static hooks
- **NEW**: Single abstraction for all models with dynamic hooks

### 3. **VerifierSystem** - Independent Verification Logic
```python
class VerifierSystem:
    """Unified verifier system that coordinates challenge generation and verification."""
```

**Key Features:**
- Completely separate from prover
- Re-computes challenges using reference data
- Supports cross-variant verification (AR vs TF, distributed vs single)
- Handles all challenge types uniformly

**Why It's Better:**
- **OLD**: Verification logic mixed with prover code
- **NEW**: Clean separation of concerns

### 4. **UnifiedWorkflow** - Complete Orchestration
```python
class UnifiedWorkflow:
    """Orchestrates the complete prover/verifier workflow for any workload type."""
```

**Key Features:**
- Coordinates model execution with dynamic hooks
- Manages challenge generation and collection
- Handles reference execution for verification
- Manages database storage with proper IR linking

**Why It's Better:**
- **OLD**: Each test duplicated workflow logic
- **NEW**: Single orchestrator for all workflows

## Key Improvements Over Current Tests

### Dynamic Hooks with JAX IO Callbacks

**Current Issue:**
```python
# OLD: Static schedule passed at initialization
schedule = hook_system.generate_challenge_schedule(config.n_forward_passes)
# Prover knows exactly when challenges will occur!
```

**Unified Solution:**
```python
# NEW: Dynamic decisions via pure_callback
challenge_signal = jax.pure_callback(
    oracle_callback,  # External function called at runtime
    result_shape_dtypes,
    activation,
    context,
    vmap_method='sequential'
)
# Prover cannot predict when challenges occur!
```

### External Challenge Oracle

**Current Issue:**
```python
# OLD: Challenge decisions are deterministic from config
if random.bernoulli(subkey, p=self.config.challenge_prob):
    # Predictable based on seed and probability
```

**Unified Solution:**
```python
# NEW: Cryptographic unpredictability
entropy_sources = [
    secrets.token_bytes(32),  # Cryptographic randomness
    str(time.time()).encode(),  # Current time
    execution_state,  # Runtime state
]
# Completely unpredictable to prover!
```

### Unified Configuration

**Current Issue:**
- Each test has different config dataclasses
- No standardization across tests
- Duplication of parameters

**Unified Solution:**
```python
@dataclass
class UnifiedConfig:
    # Model architecture
    model_type: str = "feedforward"
    # Execution parameters
    workload_type: str = "inference"
    # Challenge configuration
    challenge_strategies: List[str]
    # Hook configuration
    enable_dynamic_hooks: bool = True
    # Security
    isolation_mode: bool = True
```

## Migration Guide

To migrate existing tests to unified abstractions:

### Step 1: Replace Static Hooks
```python
# OLD
class ChallengeHookSystem:
    def generate_challenge_schedule(self, n_passes):
        # Static schedule generation

# NEW
oracle = ChallengeOracle(config)
# Dynamic decisions at runtime
```

### Step 2: Use ProverSystem
```python
# OLD
model = SimpleModel(config)
output, activations = model.forward(x)

# NEW
prover = ProverSystem(config, oracle)
output = prover.forward_with_hooks(x, step=0)
```

### Step 3: Separate Verification
```python
# OLD
# Verification mixed with execution
assert jnp.allclose(jitted_output, python_output)

# NEW
verifier = VerifierSystem(config)
results = verifier.verify_challenge_responses(challenges, reference_data)
```

### Step 4: Use UnifiedWorkflow
```python
# OLD
# Manual orchestration in each test

# NEW
workflow = UnifiedWorkflow(config, database)
graph_id, trace_id, results = workflow.execute_workload()
```

## Benefits of Unified Abstractions

1. **Security**: Prover cannot predict challenges (cryptographic isolation)
2. **Modularity**: Clean separation of prover/verifier/orchestration
3. **Reusability**: Same abstractions work for all workload types
4. **Maintainability**: Single source of truth for core logic
5. **Extensibility**: Easy to add new model types or challenge strategies
6. **Testability**: Each component can be tested independently

## Example Usage

```python
# Create configuration
config = UnifiedConfig(
    model_type="transformer",
    workload_type="autoregressive",
    challenge_strategies=["lsh_dynamic", "activation_hash"],
    enable_dynamic_hooks=True,
    isolation_mode=True
)

# Execute workflow
workflow = UnifiedWorkflow(config, database)
graph_id, trace_id, results = workflow.execute_workload()

# Verify results
assert results["total_challenges"] > 0
assert results["execution_consistency"] == True
```

## Next Steps

1. **Rollout**: Apply unified abstractions to all existing tests
2. **Extensions**: Add support for more challenge strategies
3. **Performance**: Optimize pure_callback overhead
4. **Documentation**: Create detailed API documentation
5. **Testing**: Add unit tests for each abstraction

## Key Insight

The fundamental improvement is moving from **static, predictable challenges** to **dynamic, unpredictable challenges** using JAX IO callbacks. This ensures the prover cannot game the verification system while maintaining compatibility with JAX's compilation model.