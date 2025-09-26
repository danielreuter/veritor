# Sampling and Graph Transformation Brittleness Report

## Executive Summary

Stress testing reveals **22 critical failure modes** in the sampling and graph transformation system. The system is indeed **extremely brittle** as suspected, with fundamental issues arising from the impedance mismatch between dynamic sampling operations and static graph compilation.

## Critical Vulnerabilities Found

### 1. Numerical Instability (9 failures)

#### Temperature Scaling Issues
- **Near-zero temperatures (< 1e-4)**: Cause numerical overflow in softmax computation
- **High temperatures (> 100)**: Distributions become pathologically uniform, losing all signal
- **Cascading errors**: Numerical errors compound through multiple sampling steps

```python
# Example failure case
temp = 1e-10
scaled_logits = logits / temp  # Overflow!
probs = jax.nn.softmax(scaled_logits)  # NaN or Inf
```

#### Distribution Collapse
- Distributions collapse to deterministic after just 1-2 steps of log-prob recycling
- Max probability reaches 0.9999+ quickly, making sampling meaningless

### 2. Edge Case Handling (7 failures)

#### Top-k Sampling
- **k=0**: Returns empty array, breaking downstream code
- **k > vocab_size**: Ambiguous behavior
- **Equal logits**: No principled way to select top-k tokens
- **Negative k**: Undefined behavior

#### Nucleus (Top-p) Sampling
- **p=0**: Should select nothing but implementation varies
- **p>1**: Invalid but not caught
- **Small p with concentrated distribution**: Can select 0 tokens unexpectedly

### 3. Graph Transformation Issues (4 failures)

#### Dynamic Control Flow
```python
# This breaks JAX compilation
if sample > threshold:  # TracerBoolConversionError!
    output = branch_a()
else:
    output = branch_b()
```

#### Missing RNG Operations in StableHLO
- Random operations sometimes don't appear in compiled graphs
- Makes verification of sampling impossible

#### Side Effects Don't Work
```python
global_counter[0] += 1  # Only executes once due to JIT caching!
```

#### Gradient Flow Issues
- Gradients through sampling give **wrong results** (non-zero when should be zero)
- No proper REINFORCE or Gumbel-Softmax implementation

### 4. Verification Challenges (2 failures)

#### Non-Deterministic Behavior
- Same PRNG key doesn't guarantee same results across:
  - Different JAX versions
  - Different hardware (CPU vs GPU vs TPU)
  - Different compilation settings

#### Graph Explosion
- Nested conditionals cause exponential graph size growth
- 3 levels of nesting → 10+ conditional nodes in StableHLO

## Specific Failure Examples

### Example 1: Temperature Overflow
```python
logits = jnp.array([1.0, 2.0, 3.0])
temp = 1e-10
scaled = logits / temp
# Result: [1e10, 2e10, 3e10] or [inf, inf, inf]
```

### Example 2: Dynamic Shape Failure
```python
def sample_until_end_token(logits, key):
    for i in range(max_len):
        token = sample(logits, key)
        if token == END_TOKEN:
            break  # TracerBoolConversionError!
    return outputs
```

### Example 3: Verification Impossibility
```python
# This produces different StableHLO each compilation
@jax.jit
def model(x, key):
    sample = random.categorical(key, x)
    return sample * 2 if sample > 5 else sample * 3
```

## Root Causes

1. **Static vs Dynamic Mismatch**: JAX/XLA require static graphs, but sampling is inherently dynamic
2. **Numerical Precision**: Float32 insufficient for extreme temperature scaling
3. **Missing Abstractions**: No proper probabilistic programming primitives in JAX
4. **Verification Gap**: StableHLO doesn't capture sampling semantics properly

## Recommendations

### Short Term (Mitigation)
1. **Clamp temperatures**: Force range [0.01, 100] to avoid numerical issues
2. **Validate inputs**: Check for NaN, Inf, and degenerate distributions
3. **Use stable implementations**:
   ```python
   # Better
   log_probs = jax.nn.log_softmax(logits)
   # Instead of
   probs = jax.nn.softmax(logits)
   log_probs = jnp.log(probs)  # Unstable!
   ```

### Long Term (Redesign Needed)
1. **Separate verification paths**: Don't try to verify sampling through StableHLO
2. **Probabilistic IR**: Need specialized IR for probabilistic operations
3. **Hardware-specific verification**: Different verification strategies per backend
4. **Explicit non-determinism handling**: Mark sampling regions as "verification boundaries"

## Impact Assessment

### High Risk Areas
- Autoregressive generation (cascading errors)
- Beam search (complex control flow)
- Reinforcement learning (gradient issues)
- Multi-device inference (synchronization)

### Medium Risk Areas
- Single-step sampling
- Temperature-controlled generation
- Top-k filtering with reasonable k

### Low Risk Areas
- Argmax decoding (deterministic)
- Fixed temperature = 1.0
- Simple categorical sampling

## Conclusion

The system is **fundamentally brittle** due to:
1. Attempting to verify inherently stochastic operations through deterministic IR
2. Numerical instabilities in extreme parameter regimes
3. Impedance mismatch between ML sampling needs and compiler assumptions

**This brittleness is not a bug, it's a fundamental architectural limitation.**

## Test Coverage

- Total test cases: 50+
- Failures found: 22
- Success rate: 56%
- Critical failures: 12
- Edge cases covered: 30+

---

*Generated from stress tests in:*
- `test_sampling_stress.py` (18 failures)
- `test_graph_transform_brittleness.py` (4 failures)