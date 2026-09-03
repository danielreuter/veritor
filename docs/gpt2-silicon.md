# GPT-2 Small on silicon: a pinned circuit that an RTX 4090 and a CPU compute word for word

Sep 3, 2026. Deliverable 4: the real model, HF `gpt2` (124M), run through fixed-order GPU kernels so that the run *is* a circuit of pinned gates -- gates whose semantics are part of their definition, either the tensor core's `mma.sync` step (recovered in `docs/hardware-semantics.md`) or an explicit sequence of IEEE binary32 operations -- and the framework's sampled verification of that circuit exercised end to end. The result: the GPU's greedy decode of a 27-token prompt for 8 tokens (" is the tallest building in the world.", the same tokens HF's fp32 model produces) was captured at gate granularity, 11,664,980 words in 363 tensors, and the CPU evaluation of the same circuit reproduced every one of them: 0 mismatches, on Apple arm64 and on the pod's x86. 37,225 VUs drawn with the framework's own challenge derivation re-executed through `Circuit.evaluate_gate`/`check_gate` to the GPU's words with no failure; `run_protocol` on the 12-layer model accepted an honest one-token run and rejected a single flipped bit of one dot output (numbers in section 7). Code: `src/veritor/core/silicon.py` (gates), `src/veritor/constructors/gpt2.py` (structure), `src/veritor/constructors/gpt2_reference.py` (numpy reference, addressing), `gpu/gpt2/` (kernels, drivers, results), `docs/gpt2-structure.md` (the structure and its tables).

## 1. Hardware and software identity

One RunPod on-demand pod, community cloud: NVIDIA GeForce RTX 4090 (Ada Lovelace, sm_89, `GPU-123812ef-87c2-f233-64d8-fb86dd5e364b`), 24,564 MiB, SM clock 3105 MHz, memory 10,501 MHz, 450 W; driver 550.127.05; CUDA 12.4 (`cuda_12.4.r12.4/compiler.34097967_0`); Ubuntu 22.04.5, g++ 11.4.0, Linux 6.8.0-60 x86_64; Python 3.12.14, numpy 2.5.2, PyTorch 2.4.1+cu124, transformers 4.46.3 (only to load the checkpoint and produce the HF fp32 reference tokens). Kernels: `gpu/tensor-core-semantics/gemm_chain.cu` (the fixed-order tensor-core GEMM of deliverable 3, unchanged) and `gpu/gpt2/pinned_ops.cu` (this deliverable's elementwise and reduction kernels), both compiled with `nvcc -O3 -arch=sm_89 -fmad=false -prec-div=true -prec-sqrt=true -ftz=false`; the CPU chain `gpu/tensor-core-semantics/ref_chain.cpp` with `g++ -O2 -ffp-contract=off`. The CPU side of every comparison is `veritor.constructors.gpt2_reference.NumpyOps` (numpy float32; IEEE-exact for `+ - * / sqrt`) with the chains in `ref_chain.cpp` or in pure numpy (`PythonGemm`, the gate's own evaluator).

Pod accounting (RunPod billing API, `GET /v1/billing/pods`): pod `2tz42uwnaa3ptg`, 3,050,077 ms = **50.8 minutes**, **$0.63** (RTX 4090 at $0.74/h plus disk). The account was billed for one more pod on the same UTC day (`ogb239kwmxj1kw`, 14.6 min, $0.18) which I cannot attribute with certainty -- my SSH known-hosts file for this work holds one host key, so it is either an unrelated pod or a first attempt that never reached SSH; the upper bound for this deliverable is 65.4 minutes, $0.81. `myself { pods }` and `GET /v1/pods` both return an empty list: no pod remains.

## 2. The pinned gate set

`make_pinned_gate_set()` in `veritor.core.silicon` (digest `5786b505...`), the full table in `docs/gpt2-structure.md`. Design decisions:

**Words, not numbers.** A value is a 16- or 32-bit word; 32-bit words hold binary32 patterns, 16-bit words hold BF16 patterns or token ids, and which one a word is follows from the gate that consumes it. `Gate.arg_widths` (the minimal extension of `core/gates.py`: a tuple defaulting to `(width,) * arity`, used wherever argument widths are validated, absent from the manifest when it equals the default so that every existing gate-set digest is preserved) lets `tc_dot16` take one fp32 accumulator and 32 BF16 operands.

**Two tensor-core gates.** `tc_dot16(acc, a[16], b[16])` is one Ada BF16 `mma.sync m16n8k16` step, the `ADA_BF16_M16N8K16` pipeline of deliverable 3 (products exact in 25 bits, aligned to the maximum exponent, summed with the accumulator in a wide fixed-point adder, one round-to-nearest-even to fp32); `tc_dot16_0(a[16], b[16])` is the same with an implicit `+0.0` accumulator, so that unbiased chains (the tied LM head, the attention scores, the value mix, the embedding) do not need a zero constant at every chain start. A dot product of length `K` is a `K/16`-step chain, `tc_dot16(...(tc_dot16_0(a_0, b_0), a_1, b_1)...)`, and the GEMM kernel executes exactly that chain per output element (no split-K, one warp per 16-row tile stepping through `K`), which is why it matches.

**Explicit fp32 sequences for everything else.** `+ - * /` and `sqrt` are correctly rounded on any IEEE machine when FMA contraction, fast math and flush-to-zero are off, so `f32_add`, `f32_sub`, `f32_mul`, `f32_div` and `ln_rstd` (`1 / sqrt(v + 1e-5)`) are bit-exact by construction; the GPU kernels use the `__fadd_rn`-family intrinsics so the guarantee does not even depend on the compiler flags. `exp` and `tanh` are not library calls: `f32_exp` is a fixed sequence (`t = x log2e`; `k = rint(t)` by the `1.5 * 2^23` magic-number add and subtract; `r = x - k ln2_hi - k ln2_lo` with a 12-bit `ln2_hi` so the first product is exact; a degree-5 Horner polynomial with Cephes' coefficients; `y = ((p r) r + r) + 1`; `y * 2^k` with `2^k` built from its bit pattern; `x < -86.5 -> +0`, `x > 88 -> +inf`) written once in numpy (`silicon.f32_exp`) and once in CUDA (`pinned_ops.cu`), 24 operations, one gate. `f32_tanh(x) = sign(x) (1 - 2 / (exp(2|x|) + 1))` over `f32_exp`, saturating to `+-1` at `|x| >= 9` (30 operations); `gelu_tanh(x) = (0.5 x) (1 + tanh(c0 (x + c1 ((x x) x))))` with `c0 = 0x3F4C422A`, `c1 = 0x3D372713` (HF `gelu_new`; 38 operations). These are *definitions*: any implementation that reproduces the sequence reproduces the model, and one that uses libm's `expf` does not (it differs in the last bit on a few inputs per thousand, which the goldens would catch).

**Comparisons and selection.** `f32_max(a, b) = b if b > a else a` (ties keep `a`), `argmax_select(la, lb, ia, ib) = ib if lb > la else ia`; a tournament node is one `f32_max` and one `argmax_select`, and ties resolve to the earlier index, as `torch.argmax` does. `token_eq(t, j)` gives BF16 `1.0` or `0`: the one-hot of a token against the committed token table.

**NaN.** Every fp32 gate's `check` requires NaN-free arguments and output. An honest run has no NaN anywhere, and NaN payloads are the one thing IEEE leaves to the implementation (the GPU and numpy do produce different payloads for `inf - inf`, which the golden tests treat as equal); making them a relation violation removes the prover's only degree of freedom.

**Constants.** The grammar has no immediates. BF16-exact constants (`n = d_model = 768.0`, the attention scale `0.125`, `zero`) are `weight` gates at the end of the weight vector and widened once per forward; fp32-only constants (`eps = 1e-5`, the GELU coefficients, the `exp` polynomial) are inside the gate semantics, where they belong.

**Costs.** `replay_cost = proof_cost` = the operations of the sequence (16 for a tensor-core step, 24/30/38/3 for the transcendentals, 1 otherwise), entering `Cost` and the per-VU proof cap only.

## 3. The dataflow

Weights: the HF fp32 checkpoint rounded once, round-to-nearest-even, to BF16 (`as_bf16_words`), `wte` and `wpe` included, biases and LayerNorm affine parameters included; the `Conv1D` weights kept in HF's `x @ W` orientation (`in x out`, row-major), so a dot product of `x` against column `j` of `W` reads `W[:, j]`; 124,490,068 `weight` gates in `GPT2Shape.layout()` order, committed as the weight root (`WeightTree`, section 7).

Per forward (a prefill of `m` positions or a decode of one), for each position `p` with token `t`:

1. `onehot[j] = token_eq(t, j)` for `j < vocab` (`1.0`/`0` BF16), padded with zeros to `vocab_padded = 50,272`; `emb[i] = tc_dot16_0` chain of 3,142 steps over `onehot` and `wte[:, i]` (fp32; exactly `wte[t, i]` widened, since one product is nonzero and it is exact); `x0[i] = f32_add(emb[i], bf16_to_f32(wpe[p, i]))`. The residual stream is fp32 from here on.
2. Per layer: `ln1` (below) gives BF16 `h`; `q32/k32/v32[j] = tc_dot16` chain of 48 steps over `h` and `W[:, j]` starting from `bf16_to_f32(b[j])` (fp32); `q, k, v = f32_to_bf16(.)`; `k`, `v` are appended to the request's KV cache (BF16). Per head, for each cached position `c'` up to `p`: `scores[c'] = tc_dot16_0` chain of 4 steps (`d_head = 64`) over `q_head` and `k_head[c']` (fp32); `u = f32_mul(scores, 0.125)`; `m = ` fixed pairwise `f32_max` tree over `u[0..p]`; `e[c'] = f32_exp(f32_sub(u[c'], m))`; `S = ` fixed pairwise `f32_add` tree over `e`; `prob[c'] = f32_to_bf16(f32_div(e[c'], S))`; `mix32[i] = tc_dot16_0` chain of `ceil((p+1)/16)` steps over `prob` (zero-padded to a multiple of 16) and `v_head[:, i]` (fp32); `mix = f32_to_bf16(mix32)`. Then `proj[i] = ` 48-step biased chain over the concatenated `mix` and `W_o[:, i]` (fp32); `x1 = f32_add(x0, proj)`; `ln2` gives BF16 `h2`; `fc[j] = ` 48-step biased chain (fp32); `gelu = f32_to_bf16(gelu_tanh(fc))`; `mlp[i] = ` 192-step biased chain over `gelu` and `W_proj[:, i]` (fp32); `x2 = f32_add(x1, mlp)`.
3. LayerNorm (`ln1`, `ln2`, `lnf`) in fp32: `mean = f32_div(sum_tree(x), n)`; `center[i] = f32_sub(x[i], mean)`; `var = f32_div(sum_tree(center^2), n)` (squares by `f32_mul`); `rstd = ln_rstd(var)`; `out[i] = f32_to_bf16(f32_add(f32_mul(f32_mul(center[i], rstd), bf16_to_f32(g[i])), bf16_to_f32(b[i])))`. The sum trees are pairwise: `((x0 + x1) + (x2 + x3)) + ...`, the last odd element carried up (`tree_reduce`), the same tree in the kernels.
4. At a predicting position: `lnf` then `logits[j] = tc_dot16_0` chain of 48 steps over `h` and `wte[j, :]` (fp32, tied head, no bias); argmax as 786 blocks of 64 (785 x 64 + 17) each a tournament of `f32_max`/`argmax_select` over `(logit, token)` pairs producing `(best, index)`, then one tournament of 786 over the block winners; the winning token is the next input.

The BF16 rounding points are therefore: LayerNorm outputs, `q`/`k`/`v`, the probabilities, the head mix, the GELU output -- exactly the tensors a BF16 GEMM consumes; fp32 everywhere a GEMM produces or a statistic lives (residual stream, scores, `exp`, sums, logits). `capture_gpu.npz` records all of them: 363 tensors named as above (`L{l}.q32`, `L{l}.q`, `L{l}.scores`, ..., `lnf.mean`, `logits`, `argmax.best/idx/token`), 11,664,980 words for the 34 positions of the run (27 prompt + 7 decode-step inputs; the eighth generated token is the last argmax); the same names index `address_map`, which places every recorded word at its circuit address (0.48 s at GPT-2 Small).

## 4. The GPU run

`gpu/gpt2/dump_hf.py` (on the pod: load HF `gpt2` fp32, tokenize "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and", 27 tokens; HF greedy for 8 tokens gives `[318, 262, 38760, 2615, 287, 262, 995, 13]` = " is the tallest building in the world." in 0.36 s; dump the weights and cuBLAS timings). `gpu/gpt2/run_gpt2.py capture --backend cuda`: `CudaOps` (ctypes over `libgemm_chain.so` and `libpinned_ops.so`; PyTorch is not used) drives `gpt2_reference.forward`, the one forward implementation shared by GPU and CPU, which records every intermediate. 25,672 kernel launches, 9.99 s wall including the host copies of every tensor (the GEMM chains are about 35 ms of it by the table below, 15 ms of that the one-hot embedding). Tokens: `[318, 262, 38760, 2615, 287, 262, 995, 13]`, identical to HF fp32 -- BF16 operands with fp32 accumulation did not change a greedy choice on this prompt.

Kernel cost against cuBLAS (`torch.matmul`, BF16 in, BF16 out) on the run's GEMM shapes, RTX 4090, microseconds per call (`bench.json`, best of repeated launches):

| shape `M x K x N` | fixed-order chain | cuBLAS | ratio |
|---|---|---|---|
| 1 x 768 x 768 (decode `q/k/v/o`) | 23.3 | 9.8 | 2.4x |
| 1 x 768 x 3072 (decode `fc`) | 23.3 | 9.6 | 2.4x |
| 1 x 3072 x 768 (decode `proj`) | 87.1 | 9.6 | 9.1x |
| 1 x 50272 x 768 (decode embedding one-hot) | 1,829 | 38.6 | 47x |
| 27 x 768 x 768 (prefill `q/k/v/o`) | 23.7 | 13.1 | 1.8x |
| 27 x 768 x 3072 (prefill `fc`) | 23.8 | 13.2 | 1.8x |
| 27 x 3072 x 768 (prefill `proj`) | 88.4 | 13.1 | 6.7x |
| 27 x 50272 x 768 (prefill embedding) | 1,766 | 62.5 | 28x |
| 1 x 768 x 50257 (LM head) | 114 | 40.6 | 2.8x |

The chain is serial in `K` by construction (no split-K, one accumulator per output), so its time grows with `K` where cuBLAS parallelises the reduction: 1.8-2.8x at `K = 768` (deliverable 3's 1.2-3.7x), 7-9x at `K = 3072`, 28-47x for the one-hot embedding at `K = 50,272` -- a 3,142-step serial chain per output that a `gather` gate would replace with a table lookup. Over the whole run (one prefill of 27, seven decodes) the chains are about 35 ms of GEMM; the elementwise kernels are unoptimised single-pass launches and the captured run is launch- and copy-bound.

## 5. Whole-forward bit-exactness

`run_gpt2.py cpu` runs the same `forward` with `NumpyOps(CppGemm(libref_chain.so))` on the CPU from the same BF16 weights and prompt, then `match` compares every recorded tensor word for word (`match.json`; `match_pod_x86.json` is the same on the pod's x86 CPU, 205 s, `capture_cpu.json` the Apple arm64 run, 9.75 s). Tensors grouped by the gate kind that produces them; "elements" counts stored words (the attention tensors are stored dense per head, so they include the causal upper triangle, zero on both sides: 85,680 of the 166,464 score words are gates):

| producing gate(s) | tensors | elements | mismatches |
|---|---|---|---|
| `token_eq` (one-hot, `vocab_padded` per position) | 1 | 1,708,738 | 0 |
| `tc_dot16_0` chain, K = 50,272 (embedding) | 1 | 26,112 | 0 |
| `bf16_to_f32` (position rows, constants) | 3 | 26,128 | 0 |
| `f32_add` (embedding add, residuals) | 25 | 652,800 | 0 |
| `ln_mean`: `f32_add` tree, `f32_div` | 25 | 824 | 0 |
| `f32_sub` (centring) | 25 | 632,832 | 0 |
| `ln_var`: `f32_mul`, `f32_add` tree, `f32_div`, `ln_rstd` | 25 | 824 | 0 |
| `ln_out`: `f32_mul`, `f32_mul`, `f32_add`, `f32_to_bf16` | 25 | 632,832 | 0 |
| `tc_dot16` chain, K = 768, biased (`q32`, `k32`, `v32`) | 36 | 940,032 | 0 |
| `f32_to_bf16` (`q`, `k`, `v`) | 36 | 940,032 | 0 |
| `tc_dot16_0` chain, K = 64 (scores) | 12 | 166,464 | 0 |
| `f32_mul` (scale) | 12 | 166,464 | 0 |
| `softmax_max`: `f32_max` tree | 12 | 4,896 | 0 |
| `exp_cell`: `f32_sub`, `f32_exp` | 12 | 166,464 | 0 |
| `softmax_sum`: `f32_add` tree | 12 | 4,896 | 0 |
| `prob_cell`: `f32_div`, `f32_to_bf16` | 12 | 166,464 | 0 |
| `tc_dot16_0` chain, K = 16 ceil(c/16) (mix) | 12 | 313,344 | 0 |
| `f32_to_bf16` (mix) | 12 | 313,344 | 0 |
| `tc_dot16` chain, K = 768, biased (`proj`) | 12 | 313,344 | 0 |
| `tc_dot16` chain, K = 768, biased (`fc`) | 12 | 1,253,376 | 0 |
| `gelu_tanh` | 12 | 1,253,376 | 0 |
| `f32_to_bf16` (GELU) | 12 | 1,253,376 | 0 |
| `tc_dot16` chain, K = 3072, biased (`mlp`) | 12 | 313,344 | 0 |
| `tc_dot16_0` chain, K = 768 (logits) | 1 | 402,056 | 0 |
| `argmax_block`, `argmax_top`: `f32_max`, `argmax_select` | 3 | 12,584 | 0 |
| tokens | 1 | 34 | 0 |
| **total** | **363** | **11,664,980** | **0** |

Nothing had to be fixed after the kernels were written to the sequences above; the one adjustment during development was to the goldens (section 6), where NaN payloads of `inf - inf` differ between the GPU and numpy and are compared as "both NaN". The whole-forward CPU run takes 9.75 s on the laptop (numpy over 11.7 M words plus the C++ chains) against some 35 ms of GEMM chains on the GPU: a few hundred times slower, which is the ratio the protocol is built to avoid paying on everything.

## 6. Golden vectors

`tests/veritor/core/golden/ada_pinned_f32.json` (504 KB) and `tests/veritor/core/test_pinned.py`: 400 GPU records per pinned elementwise gate (`f32_exp`, `f32_tanh`, `gelu_tanh`, `ln_rstd`, `f32_add`, `f32_sub`, `f32_mul`, `f32_div`, `f32_max`, `f32_to_bf16`, `argmax_select`, `token_eq`) over inputs that exercise every branch (random magnitudes, `+-0`, subnormals, `+-inf`, NaN, the saturation and overflow thresholds, ties); 26 fixed-tree sum and max reductions at lengths 1 to 1000; 24 LayerNorm rows, 34 softmax rows and 400 GELU inputs taken from the GPT-2 Small capture with the GPU's outputs. The gate evaluators of `make_pinned_gate_set()` and `NumpyOps` reproduce every record (NaN compared as NaN); the tensor-core goldens of deliverable 3 (`ada_bf16_m16n8k16.json`, `ada_e4m3_m16n8k32.json`) are unchanged. `tests/veritor/constructors/golden/gpt2_small_capture_slice.json` (681 KB) is the part of the GPU capture that fits in the repository: 262 VUs of 88 kinds with their input and output words by circuit address (section 7). The 28 MB captures and the 497 MB checkpoint dump stay out of git (`.gitignore`); `gpu/gpt2/results/` keeps the JSON summaries.

## 7. The circuit run through the framework

### 7.1 Sampled VUs (the framework's challenge derivation)

`gpu/gpt2/verify_capture.py sample`: compile GPT-2 Small for `Request(prompt, 8)` (0.13 s: 423,850,313 gates, 2 RUs, 133,318,577 VUs of which 8,828,509 computed VUs in the request RU), `address_map` (0.48 s), `SparseValues` over the capture's 11,341,549 recorded words (the 34-position tensors; the dense attention padding is dropped), the 124.5 M weights and the 27 inputs. `derive_sample_selection` with a fixed verifier seed over the request RU at `s = 20000 / 8828509` selected 19,822 VUs uniformly; 17,403 more were targeted (every VU of the rare kinds: LayerNorm statistics, softmax statistics, the argmax) to cover every kind. Each VU was re-executed with `check_unit`: every non-source gate evaluated in address order from the recorded words through `Circuit.evaluate_gate`, and every gate whose output is a recorded word compared and checked with `check_gate`. 156 s for 37,225 VUs, 3,514,992 gates, 49,162 recorded outputs compared, **0 disagreements**; `evaluate_unit` (the VU from its inputs alone, `Circuit.evaluate_gate` over its definition) equals the GPU's recorded output for one VU of every kind.

| VU kind | VUs | uniform | targeted | gates re-executed | recorded outputs compared | agreeing |
|---|---|---|---|---|---|---|
| `dot(768,True,False)` (`o`, `fc`) | 3,486 | 3,486 | 0 | 170,814 | 3,486 | 3,486 |
| `dot(768,True,True)` (`q/k/v`) | 2,100 | 2,100 | 0 | 105,000 | 4,200 | 4,200 |
| `dot(768,False,False)` (logits) | 870 | 870 | 0 | 41,760 | 870 | 870 |
| `dot(3072,True,False)` (`mlp`) | 716 | 716 | 0 | 138,188 | 716 | 716 |
| `dot(50272,False,False)` (embedding) | 54 | 54 | 0 | 169,668 | 54 | 54 |
| `dot(64,False,False)` (scores) | 202 | 202 | 0 | 808 | 202 | 202 |
| `dot(16/32/48,False,True)` (mix) | 660 | 660 | 0 | 1,716 | 1,320 | 1,320 |
| `eq_cell` | 3,804 | 3,804 | 0 | 3,804 | 3,804 | 3,804 |
| `add_cell` | 1,504 | 1,504 | 0 | 1,504 | 1,504 | 1,504 |
| `sub_cell` | 1,420 | 1,420 | 0 | 1,420 | 1,420 | 1,420 |
| `ln_out` | 1,413 | 1,413 | 0 | 8,478 | 1,413 | 1,413 |
| `ln_mean` | 824 | 1 | 823 | 632,832 | 824 | 824 |
| `ln_var` | 824 | 3 | 821 | 1,266,488 | 824 | 824 |
| `gelu_cell` | 2,889 | 2,889 | 0 | 5,778 | 5,778 | 5,778 |
| `scale_cell` | 191 | 191 | 0 | 191 | 191 | 191 |
| `exp_cell` | 200 | 200 | 0 | 400 | 200 | 200 |
| `prob_cell` | 211 | 211 | 0 | 422 | 211 | 211 |
| `softmax_max(c)`, c = 2..34 | 4,752 | 11 | 4,741 | 80,784 | 4,752 | 4,752 |
| `softmax_sum(c)`, c = 2..34 | 4,752 | 12 | 4,740 | 80,784 | 4,752 | 4,752 |
| `widen_cell` | 57 | 57 | 0 | 57 | 57 | 57 |
| `argmax_block(64)` | 6,280 | 18 | 6,262 | 791,280 | 12,560 | 12,560 |
| `argmax_block(17)` | 8 | 0 | 8 | 256 | 16 | 16 |
| `argmax_top(786)` | 8 | 0 | 8 | 12,560 | 8 | 8 |
| **total** | **37,225** | **19,822** | **17,403** | **3,514,992** | **49,162** | **49,162** |

A dot VU's chain steps are not recorded (only the chain's output is, as on the GPU), so a `dot(768,...)` VU checks its 48 steps through the one output they determine; `dot(768,True,True)` has two recorded outputs (the fp32 accumulator and its BF16 rounding), `gelu_cell` two, `argmax_block` two (`best`, `index`). The uniform sample is dominated by the kinds with the most VUs (the one-hot's `eq_cell`, the projections' dots), as it should be.

`tests/veritor/constructors/test_gpt2_capture.py` repeats this from the committed slice: 262 VUs of 88 kinds (every kind of the run but the embedding dot, whose 100,544 input words would double the file) re-executed against the GPU's words, 283 outputs compared, and the compiled description's digest checked against the one the slice was cut from (1.5 s). `test_gpt2_reference.py` does the same for a tiny random model with *every* VU and the whole circuit (`Circuit.evaluate` from prompt and weights gives the recorded tokens).

### 7.2 `run_protocol`

`gpu/gpt2/verify_capture.py protocol --layers L --prompt 1 --new 1`: the client's run is the CPU reference forward (bit-identical to the GPU capture on the full prompt) on a one-token prompt (the first token of the prompt, 464 = "The") generating one token, so the circuit is one prefill of one position. Two slices were run: the first layer alone (`--layers 1`: the HF weights of layer 0 and the final LayerNorm and `lm_head`, a legitimate 1-layer GPT-2 at full width), and the whole 12-layer model. The weights are committed by `gpu/gpt2/weight_tree.py`'s `WeightTree`, a numpy-backed, process-parallel construction of the same Merkle tree as `commit_weights` (root-identical, checked at 300 K and 3 M words), exposing the `MerkleTree` interface the prover opens weights through. The policy is `q = 1`, `s = 1/500`; the verifier seeds come from `make_expectation` (`session_id`, `q_seed`, `s_seed` derived from a fixed label per session, never from the prover); the prover's `replay` callback is the framework's `replay_unit` over the request RU, computed once and cached so that the rejection sessions pay only the commitment. `VerificationLimits` were raised (the request RU has 5.5 M and 10.7 M interior positions against the default caps). Times are wall-clock on the M-series laptop with the other slice running concurrently.

| | 1 layer (`protocol_L1_p1_n1.json`) | 12 layers (`protocol_L12_p1_n1.json`) |
|---|---|---|
| gates (with the weights) / weights / RUs / VUs | 51,985,127 / 46,523,476 / 2 / 46,644,034 | 135,190,689 / 124,490,068 / 2 / 124,780,158 |
| request RU interior positions committed | 5,461,649 | 10,700,619 |
| forward (numpy + `ref_chain`) / `Compile` / `WeightTree` | 0.30 s / 0.12 s / 39.9 s | 0.62 s / 0.13 s / 91.5 s |
| honest replay of the request RU (`replay_unit`, pure Python) | 876.5 s | 1,698.9 s |
| prover (boundary, interior commitment, evidence; replay excluded) | 747.7 s | 1,443.4 s |
| verifier (header, challenges, evidence check) | 18.9 s | 44.9 s |
| VUs sampled at `s = 1/500` / openings | 93,409 / 273,099 | 249,669 / 896,685 |
| transcript | 470,492,422 B | 1,599,259,965 B |
| verdict | `accepted` | `accepted` |
| rejection: one flipped bit | `relation_rejected` ("gate at address 49117656 violates tc_dot16"), 801.2 s | `relation_rejected` ("gate at address 132339192 violates tc_dot16"), 1,263.3 s |

The rejection test flips bit 0 of one BF16-rounded attention-mix word (`f32_to_bf16` at address 49,111,233 in the 1-layer slice and 132,316,795 in the 12-layer model, the output of a `dot(16,False,True)` VU of the last layer) in the prover's replay, so the committed interior differs from the honest one at one position: the flipped VU's own relation no longer holds, and the 768 `o`-projection `tc_dot16` chains that read the word are each inconsistent with it. A verifier sampling at `s = 1/500` sees at least one of these 769 VUs with probability `1 - (499/500)^769 = 78.6%` per session, and the first session caught it in both runs: the verifier opened one of the `o` dots, checked `tc_dot16` on the opened operands and accumulator against the opened output, and returned `RELATION_REJECTED` naming the address. The honest sessions accepted; the honest and the rejecting session of a run differ only in the verifier's seeds and the one flipped bit. (`test_gpt2_reference.py` does the same at a tiny shape with `s = 1`, where detection is certain, and also rejects a wrong claimed token with the honest interior.)

The transcripts are large because at `s = 1/500` the verifier opens 93 K and 250 K VUs, each with its inputs' Merkle paths (23-24 levels for the request RU, 26-27 for the weight tree, 32 B per node) and the boundary and interior commitment headers; it is dominated by the openings (about 1.7 KB each). `docs/benchmarks.md`'s 57 us per committed position was measured on a smaller circuit; here the pure-Python prover spends about 135 us per interior position on encoding and hashing plus the `IntervalDifferenceDomain` bookkeeping, and 160 us per replayed gate (a `tc_dot16` step is 33 word conversions and a fixed-point sum in Python). A native prover is out of scope for this deliverable; `WeightTree` shows its shape for the weight commitment (124.5 M words in 91.5 s against an estimated 4 minutes and 22 GB for the pure-Python tree).

## 8. What did not match, and what is not done

Nothing in the run failed to match. One thing was adjusted so that the *tests* express the semantics correctly: NaN payloads differ between the GPU and numpy on the golden edge cases (`+-inf` inputs to the reductions and to `f32_exp`/`f32_tanh`), so the goldens compare NaN to NaN and every gate's `check` rejects NaN outright; the GPT-2 run itself has no NaN at any address. `f32_max` is pinned as `b if b > a else a` (not `fmaxf`, which returns the non-NaN argument) so that the tournament's tie rule is the gate's rule: this was a choice, not a fix.

Not done, or not here:

- **Hopper and Ampere.** The pinned set is instantiated for `sm_89` BF16; `make_pinned_gate_set("sm_90", ...)` would take the Hopper pipeline of deliverable 3 (`HOPPER_BF16_M16N8K16`, not yet reproduced on hardware). The fp32 gates are architecture-independent.
- **FP8.** `ADA_E4M3_M16N8K32` is recovered; an E4M3 GPT-2 needs per-tensor scales in the dataflow and a `tc_dot32` gate with 64 8-bit operands, which `arg_widths` supports.
- **Fused attention orders.** FlashAttention-style kernels compute the softmax online in blocks with a different reduction tree and a different rounding point; the gate set expresses any fixed order, but the order must be that of the kernel actually run. This run uses the unfused order above.
- **Sampling instead of argmax.** Temperature sampling needs a committed random tape and a `select-by-cdf` gate; greedy only here.
- **The embedding.** A `gather` gate over the committed table would replace 26% of the gates and the 28-47x GEMM overhead of the one-hot. Left as a modelling choice made visible.
- **Prover speed.** The pure-Python prover replays 5.5 M gates in 877 s and commits and opens the 5.5 M interior positions in 748 s (section 7.2); a native prover is a separate deliverable. `WeightTree` shows the shape of it for the weight commitment.
- **Batching.** One request per RU; the toy's `ClusterG` schedule is not ported.
