# GPT-2's structure in the ontology

Sep 3, 2026 (supersedes the Sep 2 structural version). GPT-2 Small, the paper's worked real architecture, is a description with marks compiled by `veritor.compile.Compiler`: `veritor.constructors.gpt2.GPT2G` writes the structure of a greedy serving run (prefill of each prompt, then one decode forward per generated token, over a per-request KV cache) over the *pinned* gate set `veritor.core.make_pinned_gate_set()`, and the compiled artifact's per-kind table (`Compiled.kind_table()`) is what `Bound` and `Cost` fold over. Unlike the structural version this circuit *runs*: every gate has executable semantics (Ada's tensor-core `mma.sync` step or an explicit IEEE binary32 sequence, `docs/gpt2-silicon.md`), so `Circuit.evaluate` on it is the model and an RTX 4090 running the same fixed-order kernels produces the same word at every address. At `GPT2Shape.small()` with three requests of 32 prompt + 32 generated tokens the description is 1.09 MB, compiles in 0.14 s to a circuit of 1.92 billion gates with 177.9 million verification units (VUs), and every computed gate lies in a VU whose downstream cut `min(out_bits, reach_bits)` is at most 48 bits (the argmax block's `(best, index)` pair), 99.5% in one of at most 32 bits. The dot products and MACs reconcile exactly with the inner-product and MAC counts recorded from the legacy explicit-DAG analysis (the retired `circuit_cut_analysis` package) at a matched tiny shape.

Terminology as in the rest of the repo: *replay units* (RUs) are the coarse partition the prover commits and, when sampled at rate `q`, recomputes and commits the interior of; *verification units* (VUs) refine them and are sampled at rate `s` inside a sampled RU. Here the RUs are the `weights` unit and one `request` per request; the VUs are every dot product and every nonlinearity cell (below). Tests: `tests/veritor/constructors/test_gpt2.py` (structure, 7 s including the GPT-2 Small compiles), `test_gpt2_reference.py` (a tiny model run end to end through the circuit and the protocol), `test_gpt2_capture.py` (VUs of the GPU run), `tests/veritor/core/test_pinned.py` (the gates against GPU golden vectors).

## The gate set

`make_pinned_gate_set()`, `veritor.core.silicon`; digest `5786b505...`. Two word widths: 16 (BF16 weights and activations, token ids, the one-hot) and 32 (fp32 accumulators, residual stream, statistics, logits). A gate's `width` is its output's; `Gate.arg_widths` (new, default `(width,) * arity`) gives its arguments' widths, so the tensor-core step takes an fp32 accumulator and 32 BF16 operands.

| gate | arity | in -> out (bits) | semantics | replay = proof cost |
|---|---|---|---|---|
| `tc_dot16(acc, a[16], b[16])` | 33 | 32 + 32 x 16 -> 32 | one `mma.sync m16n8k16` BF16 step: Ada's pipeline (`docs/hardware-semantics.md`) | 16 |
| `tc_dot16_0(a[16], b[16])` | 32 | 32 x 16 -> 32 | the same with a `+0.0` accumulator (unbiased chains) | 16 |
| `bf16_to_f32` | 1 | 16 -> 32 | widen (`word << 16`) | 1 |
| `f32_to_bf16` | 1 | 32 -> 16 | round to nearest even; NaN to `0x7FC0` | 1 |
| `f32_add`, `f32_sub`, `f32_mul`, `f32_div` | 2 | 32 -> 32 | IEEE binary32, round to nearest even, no FTZ | 1 |
| `f32_max(a, b)` | 2 | 32 -> 32 | `b if b > a else a` (ties keep `a`) | 1 |
| `f32_exp` | 1 | 32 -> 32 | `rint` by magic-number add, Cody-Waite `ln 2` split, degree-5 Horner polynomial, scale by `2^k` (no libm/libdevice) | 24 |
| `f32_tanh` | 1 | 32 -> 32 | `sign(x) (1 - 2 / (exp(2|x|) + 1))` over `f32_exp`, saturating at `|x| >= 9` | 30 |
| `gelu_tanh` | 1 | 32 -> 32 | HF `gelu_new`: `(0.5 x) (1 + tanh(c0 (x + c1 x^3)))` in a fixed order over `f32_tanh` | 38 |
| `ln_rstd` | 1 | 32 -> 32 | `1 / sqrt(var + 1e-5)` with correctly rounded `sqrt` and `div` | 3 |
| `argmax_select(la, lb, ia, ib)` | 4 | 2 x 32 + 2 x 16 -> 16 | `ib if lb > la else ia` (ties keep the earlier index) | 1 |
| `token_eq` | 2 | 16 -> 16 | BF16 `1.0` if the token words are equal, else `0` | 1 |
| `in`, `weight` (sources) | 0 | -> 16 | | 0 (proof 1) |

Every fp32 gate's relation also requires NaN-free operands and results (`docs/gpt2-silicon.md`, section 2), so a prover cannot hide behind NaN payload freedom. Costs are declared (a `tc_dot16` step counts as its 16 MACs; the transcendentals as their operation counts) and enter `Cost` and the VU proof cap only. `core/gates.py` changed only by `arg_widths` (validation and the manifest, which is unchanged for single-width gates, so every existing gate-set digest is preserved).

## The structure

`GPT2Shape(layers, d_model, heads, d_ff, vocab, context, argmax_block=64)`, `GPT2Shape.small() = (12, 768, 12, 3072, 50257, 1024)`; `d_model`, `d_head` and `d_ff` must be multiples of 16 (chain lengths) and `d_model` exact in BF16 (it is a weight). The traced `GPT2` mirrors `ToyLM`: one `Tracer` over the pinned set, hash-consed definitions keyed by shape, marks on definitions. Per position: token embedding (a one-hot of `vocab` `token_eq` cells against a constant token table, then `d_model` `tc_dot16_0` chains of `vocab_padded / 16` steps against `wte` -- provably the gather -- plus the widened position row), and per layer LayerNorm (fp32 statistics by fixed pairwise trees, BF16 output), `q`/`k`/`v` biased projections rounded to BF16, per head the scores over the cached and fresh keys (`d_head / 16` steps each), the `1/sqrt(d_head)` scaling, a max-shifted softmax (`f32_exp`, a sum tree, `f32_div`, rounding to BF16), the value mix (a `ceil(c/16)`-step chain over the `c` probabilities, zero-padded), the biased output projection, an fp32 residual add, LayerNorm, the `gelu_new` MLP (`gelu_tanh` then rounding) and a residual add; at positions that predict a token, a final LayerNorm, the LM head tied to `wte` (`vocab` unbiased chains of `d_model / 16` steps, fp32 logits) and a blocked tournament argmax (`argmax_block` logits per block, then one tournament over the block winners). The dataflow is fixed in `docs/gpt2-silicon.md`, section 3.

Marks. Every dot product is a VU: `dot(k, biased, rounded)` (the widened bias when biased, `k/16` tensor-core steps, the rounding when rounded: `k/16 + [biased] + [rounded]` gates). Every nonlinearity cell is a VU: `ln_mean` (`d - 1` adds and the division), `ln_var` (`d` squares, `d - 1` adds, the division, `ln_rstd`), `sub_cell` (centring, per coordinate), `ln_out` (six gates per coordinate), `softmax_max(c)`, `softmax_sum(c)` (trees of `c - 1`), `scale_cell`, `exp_cell` (2), `prob_cell` (2), `gelu_cell` (2), the residual and embedding `add_cell`, the `widen_cell` of the position row and the two per-forward constants, the one-hot's `eq_cell`, `argmax_block(n)` (`2 (n - 1)`, outputs `(best, index)`) and `argmax_top(m)`; the source gates are the tracer's one-gate cells. The RUs are `weights` (every `weight` gate) and one `request(prompt, max_new)` per request, holding its prefill and decode steps and its KV cache, which never leaves it (the boundary is the prompts, the generated tokens and nothing else), as `RequestsG` does for the toy. Requests of one shape are one kind; consecutive requests of one shape are one `repeat`, so their tokens are one output run of the root.

Weights. `GPT2Shape.layout()` is the flat order of the `weight` gates: `wte` (`vocab x d_model`), `wpe` (`context x d_model`), per layer `ln1_g, ln1_b, w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o, ln2_g, ln2_b, w_fc, b_fc, w_proj, b_proj` (row-major in the `x @ W` orientation), `lnf_g, lnf_b`, the token table `0 .. vocab - 1`, and the three BF16 scalars `n = d_model`, `scale = 0.125`, `zero` (the grammar has no immediates; the LayerNorm `eps` and the GELU constants are part of the gate semantics instead, which is why they are not weights). For GPT-2 Small that is 124,439,808 parameters (the tied embedding counted once, as in the released checkpoint) + 50,257 + 3 = 124,490,068 `weight` gates, the HF fp32 checkpoint rounded once to BF16.

Closed forms. `gate_budget(shape, prompt, max_new)` gives the computed gates of one request by component; the compiled `n` equals its total plus the source gates at every shape compiled (tests). MACs per processed position: `4 d^2 + 2 d d_ff` per layer (`12 d^2` at GPT-2 Small's `d_ff = 4 d`) for the projections and the MLP, `2 d c` for attention at context `c` (the mix padded to `16 ceil(c/16)` keys on the tensor cores), `vocab d` for the LM head per prediction, and `vocab_padded d` for the embedding one-hot; the tensor-core steps of a run are exactly this sum over 16.

## Compile at scale

Description bytes are as of format 3 (count-one ranges carry stride 0, which took 64 bytes off GPT-2 Small); the timings are from the format-2 measurement.

| run | requests | description bytes | trace | compile | kind table | n (gates) | weights | kinds | RUs | VUs |
|---|---|---|---|---|---|---|---|---|---|---|
| tiny (2L, d 32, 2 heads, d_ff 64, vocab 11, ctx 8) | (3+3), (2+2) | 45,103 | 5 ms | 8 ms | 2 ms | 48,748 | 17,774 | 59 | 3 | 27,675 |
| reduced (4L, d 128, 4 heads, d_ff 512, vocab 1024, ctx 128) | 2 x (16+16) | 211,392 | 19 ms | 25 ms | 9 ms | 6,498,675 | 941,827 | 160 | 3 | 1,772,067 |
| GPT-2 Small | 3 x (32+32) | 1,090,184 | 113 ms | 144 ms | 60 ms | 1,924,349,881 | 124,490,068 | 291 | 4 | 177,855,025 |
| GPT-2 Small | 1000 x (32+32) | 1,090,190 | 420 ms | 173 ms | 54 ms | 600,077,761,068 | 124,490,068 | 291 | 1001 | 17,912,809,068 |
| GPT-2 Small (the GPU run) | 1 x (27+8) | | | 131 ms | | 423,850,313 | 124,490,068 | | 2 | 133,318,577 |

"Trace" is `GPT2G.__call__` (Python tracing plus canonical JSON), "compile" is `Compiler.compile` (parse, validate, `Index`), "kind table" is `Compiled.kind_table()`; wall time on one core (Apple M-series), `CompilationLimits()` defaults. Nothing dominates: the compile is `O(|description|)` and the description is 291 definitions at GPT-2 Small, whatever the number of gates. Against the structural version, `tc_dot16` cuts the gate count 28x (54.6 G -> 1.92 G for the three-request run: a 768-long dot product is 48 steps instead of 1,537 gates) while the VU count is unchanged up to the finer LayerNorm cells (177.9 M). Per request at GPT-2 Small, 63 processed positions and 32 predictions: 599,953,239 computed gates, of which embedding 155,285,487 (25.9%: the one-hot, 2.46 M per position, of which 50,257 equalities and 2,412,336 tensor-core steps), attention 120,734,496 (20.1%, softmax included), MLP 230,501,376 (38.4%), LayerNorm 11,859,464 (2.0%), residual 1,161,216 (0.2%), LM head 77,194,752 (12.9%), argmax 3,216,384 (0.5%), constants 64.

Two limits mattered, as before. The argmax is blocked (786 blocks of 64 logits and a top tournament of 786) rather than one flat chain so that no definition exceeds `max_steps_per_definition`. Many requests must be a `repeat`: the root's declared outputs resolve to one run per call, and `max_output_runs = 256` per definition would otherwise stop a run at 257 requests of distinct calls; a `repeat` of 1000 copies is one run. No change to `compile/` or `index.py` was needed.

## Bottleneck statistics

`kappa_V = min(out_bits, reach_bits)` of the VU that owns each gate, weighted by copies, from the kind table. Every VU's interface is its own cut here (`out_bits <= reach_bits` for every VU kind: a request reaches its 512 token bits, every VU inside it far less), so the two cuts of the paper coincide with the VU interfaces `W_V`.

GPT-2 Small, three requests (the 1000-request run has the same fractions to six digits):

| `kappa_V` = `W_V` | VU kinds | VUs | gates | of computed gates |
|---|---|---|---|---|
| 0 (source cells: `in`, `weight`) | 2 | 124,490,164 | 124,490,164 | (sources) |
| 16 (`dot(.., rounded)`, `eq_cell`, `gelu_cell`, `ln_out`, `prob_cell`, `argmax_top`) | 10 | 27,861,549 | 313,998,477 | 17.45% |
| 32 (`dot(.., fp32)`, `add_cell`, `sub_cell`, `widen_cell`, `scale_cell`, `exp_cell`, `ln_mean`, `ln_var`, `softmax_max`, `softmax_sum`) | 136 | 25,427,856 | 1,476,362,808 | 82.03% |
| 48 (`argmax_block`: the `(best, index)` pair) | 2 | 75,456 | 9,498,432 | 0.53% |

Fraction of computed gates in a VU with `kappa_V <= 16`: **17.4%**; `<= 32`: **99.5%**; `<= 48`: **100%**. The shift from the structural version's 99.97% at 16 bits is the dataflow, not the structure: the fp32 dot products (the output and MLP projections, the LM head's logits, the unrounded `q`/`k`/`v` accumulators inside `dot(768,True,True)` which count under the 16-bit rounded VU) leave through one fp32 word, and the two MLP projections alone are 38% of all gates. No VU of the model is wider than 48 bits; the one-hot's equalities are cells for the same reason as before (marked as one VU the embedding would have a 50,272 x 16-bit interface). The 136 kinds at 32 bits are the per-context instances of `softmax_max(c)`, `softmax_sum(c)` and `dot(16 ceil(c/16), False, True)` for `c = 1 .. 63` plus the fixed cells.

The same at the tiny shape (two requests, 30,969 computed gates): 16,576 gates at 16 bits (53.5%), 14,293 at 32 (46.2%), 100 at 48 (0.3%).

Per-RU interfaces `W_R`: `request(32, 32)` has `out_count = 32`, `W_R = out_bits = 512` bits (its 32 generated tokens) and `reach_bits = 512`; `weights` has `W_R = 0` (its gates are pinned under the weight commitment) and reaches the whole output; the root's `out_bits` is `1536` for three requests and `512,000` for a thousand. Every RU is closed (`request` reads only its prompt and the weights; `prefill(p)` is closed too; every `decode(c)`, `layer`, `attend_head` and cell kind is not), so a sampled request replays at `q x replay_cost` with no re-execution of anything larger.

## Cross-check against the legacy explicit DAG

`test_the_legacy_explicit_dag_agrees_on_inner_products_and_macs` compares `GPT2G` at the tiny shape for one request (prompt 3, 3 generated tokens: 5 processed positions, 3 predictions, 15 attended keys) against the counts recorded from the legacy explicit DAG, `build_gpt2_indexed_circuit(3, 3, config=GPT2Config(2, 32, 2, 64, 11, 8))` of the retired `circuit_cut_analysis` package. Gates are not comparable between the two (the legacy DAG spends one `mul` and one `add` per MAC; here a `tc_dot16` step is 16 MACs and a gate), so the comparison is on what both count: inner products (the legacy `inner-product-output` write nodes, one per dot product; the `dot(...)` VU copies here) and the multiplies inside them.

| family | legacy inner products | legacy `mul` | this description | delta |
|---|---|---|---|---|
| `q`, `k`, `v` projections | 3 x 320 | 3 x 10,240 | `dot(32,True,True)`: 960 copies, 2 steps | 0 |
| output projection, MLP up | 320 + 640 | 10,240 + 20,480 | `dot(32,True,False)`: 960 copies, 2 steps | 0 |
| MLP down | 320 | 20,480 | `dot(64,True,False)`: 320 copies, 4 steps | 0 |
| scores | 60 | 1,020 = 960 + 60 scalings | `dot(16,False,False)`: 60 copies, 1 step; `scale_cell`: 60 | the scaling is its own VU |
| value mix | 320 | 960 | `dot(16,False,True)`: 320 copies, 1 step | padded to 16 keys: 5,120 tensor-core MACs for 960 |
| LM head | 33 | 1,056 | `dot(32,False,False)`: 33 copies, 2 steps | 0 |
| embedding | 0 (free lookups) | 0 | `dot(16,False,False)`: 160 copies, 1 step | + 160 dots, 2,560 MACs: the one-hot |
| total | 2,653 | 84,896 MACs | 2,813 dots; 84,896 unpadded MACs; 5,632 steps = 90,112 tensor-core MACs | |

The deltas, each exact and each a modelling choice: the embedding gather (the legacy DAG looks a row up for free; the grammar has no gather, so the description spends a one-hot and `d_model` chains of `vocab_padded / 16` steps: 2.46 M gates per position at GPT-2 Small, 26% of a request -- a `gather` gate over a committed table remains the natural first extension), the padding of the value mix to whole tensor-core steps, and the scaling of the scores as a cell rather than a multiply inside the score. The LayerNorm, softmax and GELU no longer agree gate for gate with the legacy DAG (a `gelu_tanh` gate is one gate where the legacy spends nine; `f32_exp` one where it spends one `exp`), which is the point of pinning: the legacy count is a cost model, this one is an executable definition.

## Bound and cost at GPT-2 Small

`veritor.analysis.bound` with `eta = 2^-40` and `BoundOptions(knapsack=False)` (Laplace only: the table has 291 kinds and a request holds 5.9e7 computed VUs, so the knapsack grid would round a single VU's cost to zero) and `veritor.analysis.cost` with `CostParameters()` (`h = 1`, `c_0 = 0`). Prover overhead is `cost.total` over the honest replay cost (the sum of the RUs' `replay_cost`, 27.56e9 for three requests: the cost-weighted gate count, 14.3 gates' worth per gate because a `tc_dot16` step costs 16). Each `bound` call takes 0.15 to 0.7 s.

Three requests (output 96 tokens, 1536 bits):

| `q` | `s` | `U` (bits) | of output | prover overhead | of which recompute | commit interior | proof |
|---|---|---|---|---|---|---|---|
| 1/10 | 1/10 | 1536 (cap) | 100% | 0.117 | 0.100 | 0.007 | 0.010 |
| 1/2 | 1/10 | 1536 (cap) | 100% | 0.583 | 0.500 | 0.033 | 0.050 |
| 1/2 | 1/2 | 1536 (cap) | 100% | 0.784 | 0.500 | 0.033 | 0.251 |
| 1 | 1/10 | 1536 (cap) | 100% | 1.166 | 1.000 | 0.065 | 0.100 |
| 1 | 1/2 | 1536 (cap) | 100% | 1.568 | 1.000 | 0.065 | 0.502 |
| 1 | 3/4 | 1224.6 | 79.7% | 1.819 | 1.000 | 0.065 | 0.753 |
| 1 | 9/10 | 747.2 | 48.6% | 1.969 | 1.000 | 0.065 | 0.904 |
| 1 | 1 | 0 | 0% | 2.070 | 1.000 | 0.065 | 1.005 |

With three RUs the bound cannot bite below the whole output unless `q = 1`: an unsampled request survives with probability `1 - q`, so all three are corrupted with probability `(1 - q)^3 > 2^-40` for any `q < 1`, and each contributes its 512-bit interface. At `q = 1` the RU channel closes and only the VU channel remains; a request's `V(l)` saturates at `2^512` after about 11 corrupted 48-bit VUs or 16 corrupted 32-bit ones, which cost `-l ln(1 - s)` nats against the budget `ln 2^40 = 27.7`, so `s >= 1/2` is needed before a whole request stops being affordable. The bound is a statement about a run, and three requests is a small run. The commit-interior share fell from 0.33 to 0.065 of the honest cost with the 28x smaller gate count (one interior position per gate, 16 MACs per tensor-core position).

A thousand requests (output 32,000 tokens, 512,000 bits; `n = 6.0e11`):

| `q` | `s` | `U` (bits) | of output | bits per generated token | prover overhead |
|---|---|---|---|---|---|
| 1/10 | 1/10 | 230,649 | 45.0% | 7.2 | 0.117 |
| 1/2 | 1/10 | 40,593 | 7.9% | 1.27 | 0.583 |
| 1/2 | 1/2 | 20,957 | 4.1% | 0.65 | 0.783 |
| 1 | 1/10 | 17,367 | 3.4% | 0.54 | 1.165 |
| 1 | 1/2 | 2,748 | 0.54% | 0.086 | 1.565 |
| 1 | 3/4 | 1,394 | 0.27% | 0.044 | 1.815 |
| 1 | 9/10 | 848 | 0.17% | 0.027 | 1.965 |
| 1 | 1 | 0 | 0% | 0 | 2.065 |

The RU channel is `Lambda_q x 512` bits with `Lambda_q = ln(2^40) / -ln(1 - q)` whole requests (263 at `q = 1/10`: 134,656 bits; 40 at `q = 1/2`: 20,480), and the rest is the VU channel at rate `q s` inside the sampled requests; at `q = 1` it is the VU channel alone. The VU channel is 10-20% wider than in the structural version (230.6 K against 208.2 K bits at `q = s = 1/10`) because the widest VUs are now 48 bits and most gates sit in 32-bit VUs. Prover overhead is independent of the number of requests (every request is a closed RU), so the per-token capacity falls as `1 / N` at fixed policy: the protocol's leverage on GPT-2 comes from the many-request run, exactly as in the frontier report.

## What is not here

Requests are served one RU each with no batching (the toy's `ClusterG` schedule is not ported); a batched schedule changes the RU partition, the closed set and the reach, not the VU table. Sampling instead of argmax, FP8, other architectures and fused attention orders are listed in `docs/gpt2-silicon.md`. `docs/paper/**` is untouched: the numbers above are for it to cite.
