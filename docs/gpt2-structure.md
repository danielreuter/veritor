# GPT-2's structure in the ontology

Sep 2, 2026. GPT-2 Small, the paper's worked real architecture, is now a description with marks compiled by `veritor.compile.Compiler`: `veritor.constructors.gpt2.GPT2G` writes the structure of a greedy serving run (prefill of each prompt, then one decode forward per generated token, over a per-request KV cache) over the *structural* gate set `veritor.core.make_ml_gate_set(width=16, acc_width=32)`, and the compiled artifact's per-kind table (`Compiled.kind_table()`) is what `Bound` and `Cost` fold over. Nothing here runs: every gate of the ML set has declared output width and costs and an evaluator that raises `NotImplementedError("structural gate set: no executable semantics")`; the compile, index and analysis path never calls one. At `GPT2Shape.small()` with three requests of 32 prompt + 32 generated tokens the description is 774 KB, compiles in 0.1 s to a circuit of 54.6 billion gates with 176.8 million verification units (VUs), and the table reproduces the paper's bottleneck claim exactly: every computed gate lies in a VU whose downstream cut `min(out_bits, reach_bits)` is at most 32 bits, and 99.97% of them in one of at most 16 bits. The gate counts reconcile with the legacy explicit-DAG analysis (`circuit_cut_analysis.models.gpt2_circuit`) at a matched tiny shape up to four documented structural deltas, each exact.

Terminology as in the rest of the repo: *replay units* (RUs) are the coarse partition the prover commits and, when sampled at rate `q`, recomputes and commits the interior of; *verification units* (VUs) refine them and are sampled at rate `s` inside a sampled RU. Here the RUs are the `weights` unit and one `request` per request; the VUs are every dot product and every nonlinearity cell (below). Tests: `tests/veritor/core/test_ml_gates.py`, `tests/veritor/constructors/test_gpt2.py` (all of it runs in about 3 s, including the GPT-2 Small compiles; nothing needed a `slow` mark).

## The gate set

`make_ml_gate_set(width=16, acc_width=32)`, `veritor.core.ml_gates`. Two widths: `width` is the activation boundary (weights, activations, KV-cache entries, softmax probabilities, residuals, logits and token ids: the `vllm-fp16-reference` profile of the legacy analysis), `acc_width` the accumulator and reduction width (dot-product accumulators, LayerNorm and softmax statistics, the transcendentals). A gate's width is the width of its *output*.

| width | gates | arity | replay = proof cost |
|---|---|---|---|
| 16 | `add`, `sub`, `max` | 2 | 1 |
| 16 | `mul` | 2 | 2 |
| 16 | `select(c, a, b)` | 3 | 1 |
| 16 | `narrow(x)` (round an accumulator to the activation width) | 1 | 1 |
| 1 | `lt`, `eq` | 2 | 1 |
| 32 | `acc_add`, `acc_sub`, `acc_max` | 2 | 2 |
| 32 | `acc_mul` | 2 | 4 |
| 32 | `exp`, `recip`, `rsqrt`, `tanh` | 1 | 16 |
| 16 | `in`, `weight` (sources) | 0 | 0 (proof 1) |

Costs are declared, not measured; they enter `Cost` and the VU proof cap only. The one change outside the new files: `DescriptionCircuit` (`veritor.core.circuit`) required every gate of a set to have one width and now accepts a mixed-width set (`circuit.width` is `None` for one; every consumer reads `C[i].width` per gate, and the runs of `Out` already carried a width each). `make_isa_gate_set` and everything under `compile/` and `index.py` are untouched.

## The structure

`GPT2Shape(layers, d_model, heads, d_ff, vocab, context, width=16, acc_width=32)`, `GPT2Shape.small() = (12, 768, 12, 3072, 50257, 1024)`. The traced `GPT2` mirrors `ToyLM`: one `Tracer` over the ML set, hash-consed definitions keyed by shape, marks on definitions. Per position: token embedding (a one-hot of `vocab` equalities against a constant token table, then `d_model` dots of length `vocab` against `wte`, then the position row added), and per layer LayerNorm, `q`/`k`/`v` biased projections, per head the scaled scores over the cached and fresh keys, a max-shifted softmax (`exp`, summed denominator, one `recip`), the value mix, the biased output projection, a residual add, LayerNorm, the `gelu_new` MLP (tanh form, 9 gates per hidden unit) and a residual add; at positions that predict a token, a final LayerNorm, the LM head tied to `wte` (`vocab` unbiased dots) and a tournament argmax (`vocab - 1` compare-and-select nodes of `lt`, `select`, `select`). Every dot product accumulates in `acc_mul`/`acc_add` at 32 bits and ends in one `narrow` to 16 bits.

Marks. Every dot product is a VU: `dot(k, biased)` (one output of a projection or of the LM head: `k` products, a sum tree, the bias, the `narrow`; `2k + 1` gates, `2k` unbiased), `score` (one scaled query-key score, `2 d_head + 1`) and the value mix (`dot(c, False)` over the `c` probabilities). Every nonlinearity cell is a VU: `ln_mean`, `ln_var` (one each per normalised vector), `ln_center`, `ln_out` (per coordinate), `softmax_max(c)`, `softmax_denominator(c)` (one each per query and head), `exp_cell`, `prob_cell` (per key), `gelu_cell`, the residual and embedding `add_cell`, the one-hot's `eq_cell`; the `argmax` is one VU; the source gates are the tracer's one-gate cells. The RUs are `weights` (every `weight` gate) and one `request(prompt, max_new)` per request, holding its prefill and decode steps and its KV cache, which never leaves it (the boundary is the prompts, the generated tokens and nothing else), as `RequestsG` does for the toy. Requests of one shape are one kind; consecutive requests of one shape are one `repeat`, so their tokens are one output run of the root.

Weights. `GPT2Shape.layout()` is the flat order of the `weight` gates: `wte` (`vocab x d_model`), `wpe` (`context x d_model`), per layer `ln1_g, ln1_b, w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o, ln2_g, ln2_b, w_fc, b_fc, w_proj, b_proj` (row-major in the `x @ W` orientation), `lnf_g, lnf_b`, the token table `0 .. vocab - 1`, and the seven scalar constants `inv_d, eps, scale, gelu_c3, gelu_k, one, half` (the grammar has no immediates). For GPT-2 Small that is 124,439,808 parameters (the tied embedding counted once, as in the released checkpoint) + 50,257 + 7 = 124,490,072 `weight` gates.

Closed forms. `gate_budget(shape, prompt, max_new)` gives the computed gates of one request by component; the compiled `n` equals its total plus the source gates at every shape compiled (tests). MACs per processed position: `4 d^2 + 2 d d_ff` per layer (`12 d^2` at GPT-2 Small's `d_ff = 4 d`) for the projections and the MLP, `2 d c` for attention at context `c`, `vocab d` for the LM head per prediction, and `vocab d` for the embedding one-hot; the copy count of the one-gate `acc_mul` kind is exactly this sum.

## Compile at scale

| run | requests | description bytes | trace | compile | kind table | n (gates) | weights | kinds | RUs | VUs |
|---|---|---|---|---|---|---|---|---|---|---|
| tiny (2L, d 8, 2 heads, d_ff 16, vocab 11, ctx 8) | (3+3), (2+2) | 46,806 | 4 ms | 6 ms | 2 ms | 27,783 | 1,386 | 64 | 3 | 4,177 |
| reduced (4L, d 128, 4 heads, d_ff 512, vocab 1024, ctx 128) | 2 x (16+16) | 222,161 | 20 ms | 29 ms | 10 ms | 127,362,567 | 941,831 | 188 | 3 | 1,747,719 |
| GPT-2 Small | 3 x (32+32) | 773,651 | 75 ms | 106 ms | 56 ms | 54,589,340,261 | 124,490,072 | 348 | 4 | 176,763,749 |
| GPT-2 Small | 1000 x (32+32) | 773,657 | 367 ms | 105 ms | 56 ms | 18,155,074,553,072 | 124,490,072 | 348 | 1001 | 17,549,049,072 |

"Trace" is `GPT2G.__call__` (Python tracing plus canonical JSON), "compile" is `Compiler.compile` (parse, validate, `Index`), "kind table" is `Compiled.kind_table()`; wall time on one core, `CompilationLimits()` defaults. Nothing dominates: the compile is `O(|description|)` and the description is 348 definitions at GPT-2 Small, whatever the number of gates. The 1000-request run costs 0.3 s more tracing (1000 `Request` objects go through `requests()`, `flatten_inputs` and the grouping) and six more bytes of description (one `repeat` step in place of three calls). Per request at GPT-2 Small, 63 processed positions and 32 predictions: 18,154,950,063 gates, of which embedding 4,866,483,951 (26.8%: the one-hot, 77.2 M per position), attention 3,644,186,112 (20.1%), softmax 1,442,448, MLP 7,158,316,032 (39.4%), LayerNorm 8,303,632, residual 1,161,216, LM head 2,470,232,064 (13.6%), argmax 4,824,576.

Two limits mattered. The argmax is a tournament tree rather than a chain so that its definition stays within `max_steps_per_definition` (a flat chain over 50,257 logits is 150 K steps and a 6 MB definition; the tree is 17 `repeat` steps). Many requests must be a `repeat`: the root's declared outputs resolve to one run per call, and `max_output_runs = 256` per definition would otherwise stop a run at 257 requests of distinct calls (this is the `CompileError` a 1000-call root raises; a `repeat` of 1000 copies is one run). No change to `compile/` or `index.py` was needed.

## Bottleneck statistics

`kappa_V = min(out_bits, reach_bits)` of the VU that owns each gate, weighted by copies, from the kind table. Every VU's interface is its own cut here (`out_bits <= reach_bits` for every VU kind: a request reaches its 512 token bits, every VU inside it far less), so the two cuts of the paper coincide with the VU interfaces `W_V`.

GPT-2 Small, three requests (the 1000-request run has the same fractions to six digits):

| `kappa_V` = `W_V` | VU kinds | VUs | gates | of computed gates |
|---|---|---|---|---|
| 0 (source cells: `in`, `weight`) | 2 | 124,490,168 | 124,490,168 | (sources) |
| 1 (`eq_cell`) | 1 | 9,498,573 | 9,498,573 | 0.0174% |
| 16 (`dot`, `score`, `prob_cell`, `gelu_cell`, `ln_out`, `add_cell`, `argmax`) | 73 | 38,283,456 | 54,437,656,320 | 99.9501% |
| 32 (`ln_mean`, `ln_var`, `ln_center`, `exp_cell`, `softmax_max`, `softmax_denominator`) | 129 | 4,491,552 | 17,695,200 | 0.0325% |

Fraction of computed gates in a VU with `kappa_V <= 16`: **99.9675%**; with `kappa_V <= 32`: **100%**. The 73 kinds at 16 bits and 129 at 32 are the per-context instances of `dot(c, False)`, `softmax_max(c)`, `softmax_denominator(c)` for `c = 1 .. 63` plus the fixed cells. No VU of the model is wider than the accumulator; the widest declared VU interface is the `argmax` at 16 bits out of 50,257 logits (had the one-hot been marked as one VU it would have had a 50,257-bit interface, which is why its equalities are cells).

The same at the tiny shape (two requests, 26,392 computed gates): 88 gates at 1 bit (0.33%), 24,742 at 16 (93.75%), 1,562 at 32 (5.92%); `<= 16`: 94.08%, `<= 32`: 100%. The 32-bit share is `(2 L P + G)(4 d + 2) + L h (4 K - P)` gates for `P` positions, `G` predictions and `K` attended keys (the LayerNorm statistics and the softmax's exponentials, maxima and denominators), which is `O(d)` against the `O(d^2)` of the dots and so vanishes with the width.

Per-RU interfaces `W_R`: `request(32, 32)` has `out_count = 32`, `W_R = out_bits = 512` bits (its 32 generated tokens) and `reach_bits = 512`; `weights` has `W_R = 0` (its gates are pinned under the weight commitment) and reaches the whole output; the root's `out_bits` is `1536` for three requests and `512,000` for a thousand. Every RU is closed (`request` reads only its prompt and the weights; `prefill(p)` is closed too; every `decode(c)`, `layer`, `attend_head` and cell kind is not), so a sampled request replays at `q x replay_cost` with no re-execution of anything larger.

## Cross-check against the legacy explicit DAG

`test_the_legacy_explicit_dag_agrees_up_to_four_structural_deltas` builds both structures at the tiny shape for one request (prompt 3, 3 generated tokens: 5 processed positions, 3 predictions, 15 attended keys): `GPT2G` compiled, and `circuit_cut_analysis.models.gpt2_circuit.build_gpt2_indexed_circuit(3, 3, config=GPT2Config(2, 8, 2, 16, 11, 8))`, its families summed by tag. The legacy DAG counts 14,988 primitives (its `gate_count` of 15,761 also includes 733 `write` nodes and 40 embedding `lookup` nodes that carry no primitive); the description has 16,627 computed gates.

| component | legacy primitives | legacy uncounted nodes | this description | delta |
|---|---|---|---|---|
| embedding | 40 (`add`) | 40 lookups | 975 | + 935 = 5 x (11 + 2 x 11 x 8): one-hot and gather dots |
| LayerNorm | 1,450 (25 copies of 58) | | 1,334 (23 copies) | - 116: final LN only where a token is predicted |
| attention (projections, scores, softmax, mix) | 6,280 | 460 writes | 6,740 | + 460 `narrow` = the writes |
| MLP (two projections, GELU) | 6,560 | 240 writes | 6,800 | + 240 `narrow` = the writes |
| residual | 160 | | 160 | 0 |
| LM head | 495 | 33 writes | 528 | + 33 `narrow` = the writes |
| argmax | 3 (atomic) | | 90 | + 87: 3 x 3 (11 - 1) tournament nodes |
| total | 14,988 | 773 | 16,627 | 16,627 = 14,988 + 733 + 935 - 116 + 87 |

The deltas, each exact and each a modelling choice:

1. *Embedding gather.* The legacy DAG looks an embedding row up for free (a `lookup` node without a primitive). The grammar has no gather, so the description spends a one-hot (`vocab` one-bit `eq` cells) and `d_model` dots of length `vocab`: `vocab + 2 vocab d_model` gates per position, 77.2 M at GPT-2 Small, 27% of a request. A `gather` gate over a `weight` table (arity 1, output `width`) would remove it and is the natural first extension of the ML set once value semantics are decided.
2. *Inner-product write-out.* Every dot product here ends in a `narrow` (the 32 to 16 bit rounding); the legacy DAG has the same node as a `write` without a primitive. One per inner product, and the counts agree exactly (733 = 4 projections x 8 x 10 + 60 scores + 80 mix + (16 + 8) x 10 MLP + 33 logits).
3. *Final LayerNorm.* The legacy DAG normalises every processed position; the description only positions that predict a token (`prompt - 1` fewer copies of `7 d_model + 2`).
4. *Argmax.* One atomic `vocab`-ary gate there, a tournament of `vocab - 1` three-gate nodes here.

LayerNorm cells (`7 d + 2` each: `add` `4d - 1`, `mul` `3d + 2`, one `rsqrt`), softmax (`5 c - 1` per query and head: `c - 1` maxima, `c` shifts, `c` exponentials, `c - 1` sums, one reciprocal, `c` products), GELU (9 per hidden unit: 6 `mul`, 2 `add`, one `tanh`), the residual adds and every multiply-accumulate agree gate for gate, and so does the classification: the legacy `vllm-fp16-reference` profile puts every computed gate under a 16- or 32-bit downstream cut, which is the `kappa_V` table above.

## Bound and cost at GPT-2 Small

`veritor.analysis.bound` with `eta = 2^-40` and `BoundOptions(knapsack=False)` (Laplace only: the table has 348 kinds and a request holds 1.7e7 computed VUs, so the knapsack grid would round a single VU's cost to zero) and `veritor.analysis.cost` with `CostParameters()` (`h = 1`, `c_0 = 0`). Prover overhead is `cost.total` over the honest replay cost (the sum of the RUs' `replay_cost`, 163.4e9 for three requests: the cost-weighted gate count, about `3 n`). Each `bound` call takes 0.15 to 0.5 s.

Three requests (output 96 tokens, 1536 bits):

| `q` | `s` | `U` (bits) | of output | prover overhead | of which recompute | commit interior | proof |
|---|---|---|---|---|---|---|---|
| 1/10 | 1/10 | 1536 (cap) | 100% | 0.143 | 0.100 | 0.033 | 0.010 |
| 1/2 | 1/10 | 1536 (cap) | 100% | 0.717 | 0.500 | 0.167 | 0.050 |
| 1/2 | 1/2 | 1536 (cap) | 100% | 0.917 | 0.500 | 0.167 | 0.250 |
| 1 | 1/10 | 1536 (cap) | 100% | 1.433 | 1.000 | 0.333 | 0.100 |
| 1 | 1/2 | 1536 (cap) | 100% | 1.834 | 1.000 | 0.333 | 0.500 |
| 1 | 3/4 | 1023.96 | 66.7% | 2.084 | 1.000 | 0.333 | 0.750 |
| 1 | 9/10 | 625.5 | 40.7% | 2.234 | 1.000 | 0.333 | 0.900 |
| 1 | 1 | 0 | 0% | 2.334 | 1.000 | 0.333 | 1.000 |

With three RUs the bound cannot bite below the whole output unless `q = 1`: an unsampled request survives with probability `1 - q`, so all three are corrupted with probability `(1 - q)^3 > 2^-40` for any `q < 1`, and each contributes its 512-bit interface. At `q = 1` the RU channel closes and only the VU channel remains; a request's `V(l)` saturates at `2^512` after about 14 corrupted 16-bit VUs (`14 x 16 + log2 C(1.7e7, 14) > 512`), which cost `-14 ln(1 - s)` nats against the budget `ln 2^40 = 27.7`, so `s >= 1/2` is needed before a whole request stops being affordable and `s = 3/4` certifies two requests' worth. The bound is a statement about a run, and three requests is a small run.

A thousand requests (output 32,000 tokens, 512,000 bits; `n = 1.8e13`):

| `q` | `s` | `U` (bits) | of output | bits per generated token | prover overhead |
|---|---|---|---|---|---|
| 1/10 | 1/10 | 208,190 | 40.7% | 6.5 | 0.143 |
| 1/2 | 1/10 | 35,786 | 7.0% | 1.1 | 0.717 |
| 1/2 | 1/2 | 20,957 | 4.1% | 0.65 | 0.917 |
| 1 | 1/10 | 14,706 | 2.9% | 0.46 | 1.433 |
| 1 | 1/2 | 2,344 | 0.46% | 0.073 | 1.833 |
| 1 | 3/4 | 1,192 | 0.23% | 0.037 | 2.083 |
| 1 | 9/10 | 726 | 0.14% | 0.023 | 2.233 |
| 1 | 1 | 0 | 0% | 0 | 2.333 |

The RU channel is `Lambda_q x 512` bits with `Lambda_q = ln(2^40) / -ln(1 - q)` whole requests (263 at `q = 1/10`: 134,656 bits; 40 at `q = 1/2`: 20,480), and the rest is the VU channel at rate `q s` inside the sampled requests; at `q = 1` it is the VU channel alone. Prover overhead is independent of the number of requests (every request is a closed RU: `q x` its cost to recompute, `q x` its interior to commit, `q s x` its proof cost, and the boundary is negligible), so the per-token capacity falls as `1 / N` at fixed policy: the protocol's leverage on GPT-2 comes from the many-request run, exactly as in the frontier report.

## What is not here

Value semantics: the gate set is structural and nothing evaluates; whether values are fixed-point words or floats, and what `narrow`, `exp` or `rsqrt` compute, is the open decision the set is designed not to prejudge. The one-hot embedding and the tournament argmax are the two places where the grammar, not GPT-2, dictates the gate count; a `gather` gate and an `argmax` gate would bring both to the legacy accounting. Requests are served one RU each with no batching (the toy's `ClusterG` schedule is not ported); a batched schedule changes the RU partition, the closed set and the reach, not the VU table. `docs/paper/**` is untouched: the numbers above are for it to cite.
