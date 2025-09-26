# Veritor Module

Core production modules for StableHLO transformations and deterministic sampling.

## Quick Usage

```python
import veritor

# Transform decode to teacher forcing
transformed_hlo = veritor.rewrite_decode_to_teacher_forcing(hlo_text, "main")

# Deterministic sampling
sampler = veritor.ProductionSampler()
token = sampler.sample_simple(logits, temperature=0.8)

# Simple sampling without JAX
token = veritor.SimpleTokenSampler.sample(logits_numpy, temperature=0.8)
```

## Module Structure

- `transformation.py` - StableHLO graph transformations
- `sampling.py` - Deterministic token sampling
- `api.py` - Core Veritor API (existing)
- `ir_store.py` - IR storage (existing)
- `data_models.py` - Data models (existing)
- `prover.py` - Proving utilities (existing)

## See Also

- `PRODUCTION_GUIDE.md` - Detailed deployment guide
- `tests/` - Unit tests for all modules