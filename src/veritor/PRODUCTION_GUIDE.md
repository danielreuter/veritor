# Production Deployment Guide

## Recommended Modules for Production

### For Sampling: `sampling_production.py`

This is the most stable and simple version:

```python
from sampling_production import ProductionSampler, SimpleTokenSampler

# Choose based on your environment:
# - ProductionSampler: When you have JAX
# - SimpleTokenSampler: Pure NumPy, works anywhere
```

**Why this version?**
- No complex JAX tracing issues
- Simple, predictable behavior
- Well-tested edge cases
- Clear error messages

### For Transformations: `transformation_fixed.py`

This version has the most robust parsing:

```python
from transformation_fixed import (
    rewrite_decode_to_teacher_forcing,
    analyze_module
)

# Analyze first to understand the module
analysis = analyze_module(hlo_text)
print(f"Functions: {list(analysis.keys())}")

# Then transform
if analysis["main"].has_while_loop:
    transformed = rewrite_decode_to_teacher_forcing(hlo_text, "main")
```

**Why this version?**
- Better regex patterns for various HLO formats
- Detailed error messages
- Validation options
- Fallback strategies

## Production Checklist

### Before Deployment

- [ ] Test with your specific HLO format
- [ ] Verify deterministic behavior across environments
- [ ] Check memory usage for large vocabularies
- [ ] Test edge cases (empty inputs, single token, etc.)
- [ ] Benchmark performance vs requirements

### Configuration

```python
# Recommended production settings
config = {
    "enable_x64": False,  # Only if you need uint64 support
    "validate_transforms": True,  # Always validate in production
    "temperature_min": 0.1,  # Prevent divide-by-zero
    "temperature_max": 2.0,  # Prevent too much randomness
}
```

### Error Handling

```python
def safe_transform(hlo_text, func_name="main"):
    """Production-safe transformation with fallbacks."""
    try:
        # Try with validation
        return rewrite_decode_to_teacher_forcing(
            hlo_text, func_name, validate=True
        )
    except RuntimeError as e:
        if "while loop" in str(e):
            print(f"Warning: {func_name} has no while loop")
            return hlo_text  # Return unchanged
        raise  # Re-raise other errors

def safe_sample(logits, **kwargs):
    """Production-safe sampling with fallbacks."""
    try:
        # Try JAX sampler
        from sampling_production import ProductionSampler
        sampler = ProductionSampler(enable_x64=False)
        return sampler.sample_simple(logits, **kwargs)
    except ImportError:
        # Fall back to NumPy
        from sampling_production import SimpleTokenSampler
        return SimpleTokenSampler.sample(logits, **kwargs)
```

## Performance Tips

### For Sampling

1. **Batch operations** when possible:
```python
# Good: Single batched call
logits_batch = jnp.stack([logits1, logits2, logits3])
tokens = sampler.sample_simple(logits_batch)

# Bad: Multiple individual calls
token1 = sampler.sample_simple(logits1)
token2 = sampler.sample_simple(logits2)
```

2. **Pre-compile with JAX JIT** for hot paths:
```python
import jax

@jax.jit
def fast_sample_batch(logits, temp, seed):
    return sampler.sample_simple(logits, temp, seed, 0)
```

3. **Use static values** for top-k when possible

### For Transformations

1. **Cache parsed modules** if transforming multiple functions:
```python
# Parse once
analysis = analyze_module(hlo_text)

# Use multiple times
for func_name in analysis:
    if analysis[func_name].has_while_loop:
        # Transform...
```

2. **Use MLIR bindings** when available for large modules

## Common Issues and Solutions

### Issue: JAX Tracing Errors
**Solution**: Use `sampling_production.py` which avoids complex tracing

### Issue: Regex Timeout on Large HLO
**Solution**: Use MLIR bindings or split into smaller modules

### Issue: Non-deterministic Results
**Solution**: Ensure consistent seed, session_id, sequence_id, position

### Issue: Memory Spike with Large Vocab
**Solution**: Use smaller batches or streaming approach

## Monitoring

Add these metrics to your monitoring:
- Sampling latency (p50, p95, p99)
- Transformation success rate
- Memory usage per vocab size
- Cache hit rates (if caching)

## Support Matrix

| Feature | NumPy Only | JAX | MLIR |
|---------|-----------|-----|------|
| Basic Sampling | ✅ | ✅ | N/A |
| Top-k Sampling | ✅ | ✅* | N/A |
| Top-p Sampling | Limited | ✅* | N/A |
| Basic Transform | ✅ | ✅ | ✅ |
| Complex Transform | Limited | ✅ | ✅ |

*With limitations on dynamic values

## Security Considerations

1. **Input Validation**: Always validate HLO input
2. **Resource Limits**: Set timeouts for regex operations
3. **Sandboxing**: Run transformations in isolated environment if untrusted
4. **Audit Logging**: Log all transformations in production