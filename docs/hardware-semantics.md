# Tensor-core semantics on real silicon

**Question.** On a GPU we can rent, can a tensor-core matmul be expressed as a
chain of gates with pinned, CPU-reproducible semantics, bit-exactly, at real
model shapes? And does the synthetic FP8 contract of the openvm-tc-bench spike
(`hawkeye_ampere_groupsum_fp8e4m3_v0`) match any real hardware?

**Answer.** Yes, and no.

- On an NVIDIA RTX 4090 (Ada Lovelace, sm_89) every `mma.sync` output element
  in BF16 (`m16n8k16`) and FP8 E4M3 (`m16n8k32`) is reproduced bit-exactly by a
  ~100-line pure-integer Python model (`veritor.core.silicon.tc_dot`):
  **24,898,304 / 24,898,304 elements per dtype** over 194,518 tiles including
  random, subnormal, cancellation, mixed-magnitude, near-overflow, signed-zero
  and nonzero-accumulator families. A fixed-order GEMM kernel built from these
  steps matches a CPU evaluation of the same gate chain bit-exactly on all 30
  GPT-2 Small shapes with real GPT-2 weights and activations (60 of 60 cases
  incl. random data; 241.9 M output elements checked, 0 mismatches), at 1.2-3.7x
  (BF16) and 0.5-2.3x (FP8) cuBLAS time with a deliberately naive kernel.
- `hawkeye_ampere_groupsum_fp8e4m3_v0` matches **no** real hardware we know of.
  It reproduces Ada's FP8 instruction on only **18.31 %** of 256,000 single-tile
  outputs and **8.69 %** of 256,000 two-step (K = 32) outputs. Ada's FP8 path
  keeps 14 significand bits in the adder, not 25; the spec's group structure
  (two groups of eight over K = 16) is also not what the instruction does
  (two groups of sixteen over K = 32). It does pass 7 of its own 10 golden
  tiles on the GPU, because those are dominated by small exact values.
- Hawkeye's published models are confirmed for BF16 (Ada BF16 `mma.sync` is
  parameter-identical to Hawkeye's Ampere model: 12,800,000 / 12,800,000
  random-tile elements bit-exact against `Ampere_simulator`), but Hawkeye's
  Hopper FP8 simulator does **not** describe Ada's FP8 `mma.sync`
  (17.60 % mismatching elements on random tiles); Ada uses the same 14-bit
  adder but two accumulation groups of 16 instead of one of 32.
- Two behaviours had to be added beyond Hawkeye's model to get to 100 %:
  per-group saturation to a sticky infinity when a group sum leaves the FP32
  range (Hawkeye's simulators wrap the exponent there), and the FP8
  accumulator truncation to 14 bits *even when all products are zero*.

Everything below is measured on the hardware named in section 1 unless it says
otherwise. Raw results are under `gpu/tensor-core-semantics/results/`.

## 1. Hardware identity and cost

| Item | Value |
| --- | --- |
| Provider / pod | RunPod on-demand, pod id `2tz42uwnaa3ptg`, name `veritor-hw-semantics` |
| GPU | NVIDIA GeForce RTX 4090, 24564 MiB, compute capability 8.9 (Ada Lovelace, sm_89) |
| GPU UUID | `GPU-d0b01642-2923-2228-f309-ea9e8d1a8979` |
| Driver / CUDA | driver 580.159.03; `nvcc` 12.4.131; PyTorch 2.4.1+cu124 |
| Image | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` (Ubuntu 22.04.5) |
| Price | $0.74 / h |
| Lifetime | 2026-09-03 00:52:49Z to 01:43:25Z = 50 min 36 s = 0.84 pod-hours |
| Cost | ≈ $0.63 |
| Cleanup | `podTerminate` issued at 01:43:25Z; `myself { pods }` returned `[]` afterwards |

Only one pod was ever alive. Wall-clock for the whole workstream (setup,
measurements, port, tests, this report) was about 1.5 hours, of which 51
minutes had the pod running; the GPU was idle while the Python model was being
debugged for at most a few minutes at a time.

## 2. Method

All GPU measurements go through one CUDA kernel per instruction that executes
**exactly one `mma.sync` per warp** with raw bit patterns in and out
(`gpu/tensor-core-semantics/mma_tiles.cu`, driven through `ctypes` by
`tcs_gpu.py`). Each output element `D[i][j]` is therefore the hardware's own
function of `(C[i][j], A[i][0..K), B[0..K)[j])` with nothing in between:

~~~
mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32   (BF16, K = 16)
mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32   (E4M3, K = 32, sm_89+)
~~~

The fragment layout was verified first with exact small-integer tiles (65,536
elements per dtype, 0 mismatches, `results/characterize_*.json:layout`).
Then, following Hawkeye's methodology (`experiments/GPU_reproduction_fp8.py` in
the gpu-simulator repo), `characterize.py`:

1. finds the **neutral subgroups** — which sets of k-slots (plus the
   accumulator) can be cancelled exactly without disturbing a tiny residual
   elsewhere — which exposes the accumulation groups;
2. measures the **internal adder width** with a cancellation + tiny-residual
   probe per k-slot;
3. fits the **exponent floor** by scanning `(width, zero_exponent)` of the
   Python model against 6,000 tiny-valued and mixed tiles.

`validate_tiles.py` then runs the families of section 3 through the GPU and the
model, and (for zero-accumulator random tiles, which is all it can express) the
Hawkeye C++ simulator, and writes the golden vectors. `spec_vectors.py` writes
the same FP8 tiles as text for the Rust `tc-dot-spec` crate. `gemm_chain.cu` /
`ref_chain.cpp` / `gemm_chain.py` are the fixed-order GEMM and its CPU
reference (section 5).

## 3. Reproducing Hawkeye on Ada: match rates

Families (per dtype; `n` = number of 16x8 tiles, 128 output elements each):

| Family | n tiles | Construction |
| --- | --- | --- |
| `randn_zero_acc` | 100,000 | standard-normal operands, zero accumulator (Hawkeye's setting) |
| `randn_random_acc` | 25,000 | standard-normal operands, accumulator `N(0,1) * 2^U[-20,20]` |
| `uniform_bits` | 25,000 | uniformly random finite encodings for operands and accumulator (all exponents) |
| `subnormal` | 12,500 | operands with exponent field 0..3 (BF16) / 0..1 (E4M3); subnormal accumulators |
| `cancellation` | 12,500 | pairs of equal-magnitude opposite products at 2^16, 15 % of slots replaced by 2^-6 residuals, small accumulators |
| `mixed_magnitude` | 12,500 | random exponents over the whole operand range; accumulator `N(0,1) * 2^U[-60,60]` |
| `near_overflow` | 6,250 | accumulator exponent 250..254 with maximal operands |
| `signed_zero` | 256 | -0 / +0 accumulators, all-zero products, exact cancellations to zero |
| `tiny_acc_zero_products` | 512 | accumulator exponent 0..30, all products zero (exponent-floor probe) |

Results (`results/validate_bf16_final.json`, `results/validate_e4m3_final.json`,
Hawkeye columns from `results/validate_bf16.json`, `results/validate_e4m3.json`):

| dtype / instruction | Elements | vs `silicon.py` model | vs Hawkeye simulator (random tiles, zero acc) |
| --- | --- | --- | --- |
| BF16 `m16n8k16` | 24,898,304 | **24,898,304 bit-exact (100 %)** | `Ampere_simulator`: 12,800,000 / 12,800,000 (100 %) |
| E4M3 `m16n8k32` | 24,898,304 | **24,898,304 bit-exact (100 %)** | `Hopper_fp8_simulator`: 10,547,091 / 12,800,000 (82.40 %; 2,252,909 mismatches) |

The released gpu-simulator repository has simulators for Ampere (BF16/FP16),
Hopper (BF16/FP16) and Hopper FP8 only; there is no Ada simulator, so Ada is
compared against the two closest published models. Ada BF16 turns out to be
parameter-identical to the Ampere model; Ada FP8 `mma.sync` matches neither.

**Mismatch vs Hawkeye, FP8.** A GPU-measured reproducing tile
(acc = 0, K = 32 bytes as hex):

~~~
a = b08234bbb8a83aabc0beb2be9635bd35b19bc13b2eb93327ad31a8b8b5b636ba
b = 3b382f3d1a352fb3b928ab3033b63cb62f402ead34bab221bbc0292c3039a7bb
GPU (RTX 4090) = c0c9d800     Hopper_fp8_simulator = c0c9d400     silicon.py (ada) = c0c9d800
~~~

The minimal tile separating the two structures (derived from the validated
model; the neutral-subgroup measurement in `results/characterize_e4m3.json` is
the direct hardware evidence for the grouping): products 0 and 1 are `+256*256`
and `-256*256`, product 16 is `1*1`, everything else zero. One group of 32
aligns the `1` to exponent 16 inside a 14-bit window and truncates it to zero
(Hopper model → `0x00000000`); Ada cancels the first group to zero and then
adds `1` in the second (→ `0x3f800000`). This is
`tests/veritor/core/test_silicon.py::test_ada_fp8_has_two_groups_where_hopper_has_one`.

**Mismatch vs Hawkeye's model, BF16 (overflow).** Before the saturation rule
was added, the model disagreed with the GPU on 352,221 / 3,200,000
`uniform_bits`, 176,747 / 1,600,000 `mixed_magnitude` and 199,965 / 800,000
`near_overflow` elements (`results/validate_bf16.json`), always by producing an
infinity of the wrong sign or a finite value where the GPU had an infinity.
Reproducing tile (BF16 words as hex): acc `b07e8ecc`,
`a = 44380e42c0f160f0fb492db220eda741544eb84c1ba72f43a68387b922cd16b2`,
`b = 358dfea91fc467f69c454637a0fc8e32ff20b9bc28cec926530b332c5b30ef0f`;
GPU `7f800000`, unsaturated model `ff800000`. With saturation the three
families are 5,600,000 / 5,600,000
(`results/validate_bf16_overflow_families.json`). Minimal tile, measured
directly (`results/probe_inf.txt`): `2^64 * 2^64` in group 1 and `-(2^64 * 2^64)`
in group 2 gives `+inf`; the same two products both in group 1 give `0`.

Hawkeye's own simulators versus the Python port, on the CPU
(`results/hawkeye_vs_port.json`, 3,000 tiles each of `randn` and uniformly
random finite encodings): 0 mismatches on every `randn` tile for all three
simulators, and 0 mismatches anywhere for `Hopper_fp8_simulator`; the 12,409
(`Ampere_simulator`) and 12,544 (`Hopper_simulator`) mismatching elements are
exactly the elements whose true result lies outside the FP32 range, where the
C++ `Gfloat -> float` cast wraps the exponent field and the port saturates as
the GPU does. On the finite domain the port is a faithful reproduction of the
simulators.

## 4. The recovered pipeline (Ada Lovelace, sm_89)

For one output element, with `acc` the incoming FP32 accumulator and `K`
products `p_k = a_k * b_k` computed **exactly** (BF16: 8x8-bit significands,
16-bit product; E4M3: 4x4-bit, 8-bit product; subnormal inputs keep their
unnormalised significand at the minimum exponent, so no input is ever
normalised or flushed):

~~~
parameters   BF16 m16n8k16:  groups = (8, 8),   width = 25, floor = -132
             E4M3 m16n8k32:  groups = (16, 16), width = 14, floor = -139 (any value <= -126 is equivalent, see below)

tc_dot(acc, a[0..K), b[0..K)):
    t = acc                                   # 24-bit significand at its own exponent
    for each group g (in k order):
        terms = [t] + [p_k for k in g]
        E = max(floor, max exponent over nonzero terms)
        S = 0
        for each nonzero term x in terms:
            m = x.magnitude rescaled to `width` bits    # left shift for BF16 (25 > 24), RIGHT shift (truncation) for E4M3 (14 < 24)
            m = m >> (E - x.exponent)                   # alignment by truncating right shift (no guard/sticky bits)
            S += sign(x) * m                            # exact integer sum
        if S == 0: t = +0 ; continue
        normalise |S| to exactly `width` bits by shift; exponent = E + bitlength(|S|) - width   # truncation = round toward zero
        if exponent < -126: shift right by (-126 - exponent); exponent = -126             # denormalise, no flush
        truncate the `width`-bit significand back to 24 bits (E4M3: zero-extend)
        if exponent > 127: return +-inf with the sign of S                                # saturate; sticky for later groups
        t = (sign, exponent, 24-bit significand)
    return pack(t) as FP32   # +0 for zero; a 24-bit significand without its top bit at exponent -126 is a subnormal
~~~

Observed consequences that matter for a gate definition:

- The incoming accumulator is a *term of the first group*, not an addend of the
  group result. `acc + sum(products)` computed in FP32 does not reproduce the
  instruction; `tc_dot(acc, a, b)` does. The gate must take the accumulator as
  an input.
- E4M3: a pass-through accumulator with all-zero products is truncated to 14
  significand bits (`0x3f800001 -> 0x3f800000`); the FP8 path never returns the
  accumulator unchanged unless it already fits in 14 bits.
- The exponent floor is only observable when every term's exponent is below it.
  For BF16 the floor is pinned by the fit (`-132`: 0 mismatches on the 6,000
  fit records vs 2 at `-133` and 3 at `-131`). For E4M3 no term can have
  exponent below `-126` (E4M3 products are at least 2^-18 and FP32 subnormal
  accumulators sit at exponent `-126`), so `-139` is Hawkeye's Hopper constant
  carried over and any floor `<= -126` gives identical results; the model's
  agreement with the GPU does not depend on it.
- Overflow: a group whose exact sum exceeds the FP32 range yields `+-inf` and
  later groups cannot bring it back. `2^200 - 2^200` inside one group is exactly
  zero (no intermediate overflow); the FP8 path cannot overflow from a finite
  accumulator (`maxfloat + 448*448 -> 0x7f7ffc00`, i.e. the 14-bit-truncated
  maximum).
- Non-finite inputs are **outside the modelled domain**; `tc_dot` raises and the
  gate's `check` returns false. What the GPU does (`results/probe_inf.txt`,
  single probes, not modelled): `+inf` accumulator stays `+inf` whatever the
  products; a NaN accumulator, `inf * 0`, `inf + (-inf)`, any NaN operand and any
  E4M3 `0x7f`/`0xff` byte produce the canonical NaN `0x7fffffff`; `inf * 1` is
  `+inf`. This is IEEE-like with a canonical NaN and would be a small extension.

## 5. Fixed-order GEMM as a gate chain

`gemm_chain.cu`: grid `(ceil(N/64), ceil(M/128))`, 128 threads = 4 warps per
block; warp `w` owns rows `[32w, 32w+32)` of the block's 128 rows and all 64
columns (2 m16 x 8 n8 tiles = 16 `mma.sync` per k-step). Every output element
is one chain

~~~
D[i][j] = tc_dot(... tc_dot(tc_dot(C[i][j], A[i][0:KS], B[0:KS][j]), A[i][KS:2KS], B[KS:2KS][j]) ...)
~~~

with `KS = 16` (BF16) or `32` (E4M3), consumed by one warp, in one thread's
accumulator registers, in increasing k. No split-K, no software reduction, no
reordering. Operands are read straight from global memory into fragments; the
kernel is intentionally the simplest correct one, so its throughput is a lower
bound for fixed-order kernels.

`ref_chain.cpp` is a C++/OpenMP transcription of `silicon.py` (same integer
algorithm) that evaluates the identical chain for every output element;
`gemm_chain.py` additionally re-evaluates 400 random elements per case with the
Python model itself. Real-data inputs: GPT-2 Small (`gpt2` from Hugging Face,
`transformers` 4.44.2): block-0 `ln_1` output (input of `c_attn`, reused for
`attn.c_proj`), block-0 `ln_2` output (input of `c_fc`), block-0 GELU output
(input of `mlp.c_proj`), and `ln_f` output (input of `lm_head`), on 1024 tokens
of text; weights are the corresponding `Conv1D`/`lm_head` matrices, `lm_head`
zero-padded from 50257 to 50304 columns. Casting: BF16 is round-to-nearest-even
from FP32; E4M3 is per-tensor absmax scaling `q = rne_e4m3(x * 448 / max|x|)`
(scales recorded per case in `results/gemm_chain_gpt2.json`, e.g. `c_attn`
input scale 524.26, weight scale 157.54). Random inputs are `N(0,1)` (x32 for
E4M3), with a nonzero random initial accumulator for every `M = 32` case.

All 60 cases (30 random, 30 GPT-2) are bit-exact against the CPU chain
(`mismatches_vs_cpp_chain = 0`, Python subset 400/400 each); for BF16 with zero
accumulator Hawkeye's `Ampere_simulator` also agreed on every element where it
was run. GPT-2 results (`results/gemm_chain_gpt2.json`; cuBLAS = `torch.matmul`
bf16 for BF16, `torch._scaled_mm` e4m3 with FP32 output for FP8; times are the
mean of 20 timed repetitions after 3 warm-ups, CUDA events, same tensors):

| Layer (K x N) | dtype | M | Elements | Bit-exact | Kernel ms | cuBLAS ms | Kernel / cuBLAS |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| c_attn (768 x 2304) | bf16 | 1 | 2,304 | yes | 0.023 | 0.013 | 1.7x |
| | bf16 | 32 | 73,728 | yes | 0.023 | 0.018 | 1.3x |
| | bf16 | 1024 | 2,359,296 | yes | 0.053 | 0.033 | 1.6x |
| | e4m3 | 1 | 2,304 | yes | 0.013 | 0.025 | 0.5x |
| | e4m3 | 32 | 73,728 | yes | 0.014 | 0.024 | 0.6x |
| | e4m3 | 1024 | 2,359,296 | yes | 0.036 | 0.024 | 1.5x |
| attn.c_proj (768 x 768) | bf16 | 1 | 768 | yes | 0.022 | 0.015 | 1.5x |
| | bf16 | 32 | 24,576 | yes | 0.023 | 0.017 | 1.4x |
| | bf16 | 1024 | 786,432 | yes | 0.031 | 0.015 | 2.0x |
| | e4m3 | 1 | 768 | yes | 0.013 | 0.023 | 0.6x |
| | e4m3 | 32 | 24,576 | yes | 0.014 | 0.028 | 0.5x |
| | e4m3 | 1024 | 786,432 | yes | 0.020 | 0.024 | 0.8x |
| c_fc (768 x 3072) | bf16 | 1 | 3,072 | yes | 0.023 | 0.013 | 1.7x |
| | bf16 | 32 | 98,304 | yes | 0.023 | 0.016 | 1.4x |
| | bf16 | 1024 | 3,145,728 | yes | 0.063 | 0.037 | 1.7x |
| | e4m3 | 1 | 3,072 | yes | 0.013 | 0.027 | 0.5x |
| | e4m3 | 32 | 98,304 | yes | 0.014 | 0.023 | 0.6x |
| | e4m3 | 1024 | 3,145,728 | yes | 0.043 | 0.025 | 1.8x |
| mlp.c_proj (3072 x 768) | bf16 | 1 | 768 | yes | 0.084 | 0.014 | 6.2x |
| | bf16 | 32 | 24,576 | yes | 0.087 | 0.023 | 3.8x |
| | bf16 | 1024 | 786,432 | yes | 0.103 | 0.043 | 2.4x |
| | e4m3 | 1 | 768 | yes | 0.047 | 0.027 | 1.7x |
| | e4m3 | 32 | 24,576 | yes | 0.049 | 0.032 | 1.5x |
| | e4m3 | 1024 | 786,432 | yes | 0.057 | 0.025 | 2.3x |
| lm_head (768 x 50304) | bf16 | 1 | 50,304 | yes | 0.100 | 0.039 | 2.6x |
| | bf16 | 32 | 1,609,728 | yes | 0.287 | 0.077 | 3.7x |
| | bf16 | 1024 | 51,511,296 | yes | 2.446 | 0.748 | 3.3x |
| | e4m3 | 1 | 50,304 | yes | 0.041 | 0.024 | 1.7x |
| | e4m3 | 32 | 1,609,728 | yes | 0.046 | 0.027 | 1.7x |
| | e4m3 | 1024 | 51,511,296 | yes | 0.607 | 0.291 | 2.1x |

The random-data runs at the same 30 shapes (`results/gemm_chain_random.log`)
are also all bit-exact with the same timing picture (BF16 1.2-6.0x, E4M3
0.5-2.3x cuBLAS time). Peak observed: 130 TFLOP/s (E4M3, lm_head M = 1024) and
77 TFLOP/s (BF16, c_fc M = 1024) against the 4090's ~330 (FP8) / ~165 (BF16)
TFLOP/s dense peaks with FP32 accumulation; the sub-1x FP8 ratios at small M reflect `torch._scaled_mm` launch
overhead rather than a fast kernel. Reading operands through shared memory and
`ldmatrix` would close most of the gap without changing a single accumulation
order; the M = 1 cases are latency-bound in both implementations.

Cost of the fixed order, then: **none in correctness** (every element of every
shape is reproduced by the CPU chain) and **1.2-3.7x in time** with this naive
kernel at the shapes that matter (M = 1024), before any optimisation that keeps
the order.

## 6. `hawkeye_ampere_groupsum_fp8e4m3_v0` versus real hardware

Verdict: **the synthetic contract does not describe Ada Lovelace's FP8 tensor
core, and by construction (Ampere has no FP8 tensor cores; Hopper uses a single
group of 32 with a 14-bit adder) it describes no NVIDIA hardware.** Measured
with the crate's own `tc_dot_spec::tile` on GPU vectors
(`results/tcspec_check.txt`; 2,000 random E4M3 tiles = 256,000 elements each):

| Comparison | Bit-exact | Rate |
| --- | ---: | ---: |
| One `tile(a[0..16), b[0..16), acc)` vs the GPU with `a[16..32) = b[16..32) = 0` | 46,883 / 256,000 | **18.31 %** |
| Two chained `tile` calls vs one GPU `m16n8k32` on the full K = 32 | 22,243 / 256,000 | **8.69 %** |
| The crate's 10 golden tiles (`golden.rs`) on the GPU (`results/spec_e4m3_golden.json`) | 7 / 10 | 70 % |

The three failing goldens are exactly those that depend on 24/25-bit alignment
(`one_bit_closer_product_survives`, `normalization_is_not_round_to_nearest`,
`two_stage_cancellation_preserves_second_half`): Ada keeps 14 bits,
so `0x48000000 + 2^-18 -> 0x48000000` (spec: `0x47ffffff`), `0x4b000000 + 1
-> 0x4b000000` (spec: `0x4b000001`) and the two-group probe gives `0x41000000`
(spec: `0x41000040`). The 18 % / 9 % agreement on random tiles is the fraction
of outputs where none of the truncated bits happened to matter. Both the
crate's semantics and Ada's are retained in `silicon.py`
(`HAWKEYE_AMPERE_GROUPSUM_E4M3_V0` reproduces all 10 goldens;
`test_synthetic_v0_contract_differs_from_ada_fp8_silicon` pins the difference).
A zkVM precompile for the v0 contract proves a relation no GPU computes; the
E4M3 gate that should replace it is `ADA_E4M3_M16N8K32` (or Hopper's
`(32,)`/14-bit variant once measured on an H100).

## 7. Proposed gate definition

`src/veritor/core/silicon.py` exposes, per `(arch, dtype)`, a `Pipeline` and
`make_tensor_core_gate_set(arch, dtype)` returning a `GateSet` with three gates:

~~~
tc_dot{K}   arity 1 + 2K, width 32, replay_cost = proof_cost = K
            args = (acc: fp32 word, a[0..K): operand words, b[0..K): operand words)
            evaluate(args) = tc_dot(pipeline, acc, a, b)          # section 4, integer-exact
            check(args, out) = evaluate(args) == out  (False outside the finite domain)
in          arity 0, width 32, source INPUT_SOURCE
weight      arity 0, width 32, source WEIGHT_SOURCE
~~~

with `K = 16` for `sm_89`/`bf16` (`ada_bf16_m16n8k16`) and `K = 32` for
`sm_89`/`e4m3` (`ada_e4m3_m16n8k32`). A GEMM output element with reduction
length `K_total` is a chain of `K_total / K` such gates, the first taking the
initial accumulator (a constant `+0` gate or a bias), each subsequent one taking
the previous gate's output; `tc_dot_chain` is that fold and is what
`ref_chain.cpp` and the CUDA kernel compute. The gate is exactly one hardware
instruction's worth of work per output element, the semantics are the silicon's
and the CPU reproduces them cheaply: the C++ reference evaluated the 2.5 G gate
steps of the `lm_head` M = 1024 chain in 16 s on the pod's CPU cores, and the
Python model runs at roughly 0.5 M gates/s per process.

Fit with the current `Gate` model: `Gate(name, arity, width, ...)` carries a
single `width` that is validated against **every** argument. Operand words are
8 or 16 bits, the accumulator 32; today the gate is declared at width 32 and the
evaluator range-checks operand words itself (`check` returns false for a 17-bit
"BF16" word). The smallest clean extension is an optional per-argument width on
`Gate`, e.g. `arg_widths: tuple[int, ...] | None` defaulting to `(width,) *
arity`, used by `FlatCircuit`/value validation wherever `width` is used today.
Nothing else in the model needs to change: arity 33 / 65 is just a number, and
the sampling/proof-cost accounting already takes `replay_cost`/`proof_cost` per
gate. A second, optional refinement is to let a gate declare its operand
*dtype* so a compiler can pack 16 BF16 words into 8 committed 32-bit words; that
is a commitment-layout question, not a semantics question, and is left open.

## 8. What was ported and how it is tested

- `src/veritor/core/silicon.py`: `Term`, `bf16_product`, `e4m3_product`,
  `fp32_term`, `pack_fp32`, `group_sum`, `Pipeline`, `tc_dot`, `tc_dot_chain`,
  the six pipelines (`ADA_*` measured here; `AMPERE_BF16_M16N8K16`,
  `HOPPER_BF16_M16N8K16`, `HOPPER_E4M3_K32` transcribed from Hawkeye and
  cross-checked against its simulators on the CPU but not against an A100/H100
  here; `HAWKEYE_AMPERE_GROUPSUM_E4M3_V0` synthetic), `make_tensor_core_gate_set`.
  Ported from `src/Ampere_simulator.cpp`, `src/Hopper_simulator.cpp`,
  `src/Hopper_fp8_simulator.cpp`, `utils/utils.cpp`, `src/gfloat.cpp` of
  https://github.com/badasherez/gpu-simulator at commit
  `30703fcb309c943a6df5eee0277cb81815deb8f4`.
- `tests/veritor/core/golden/ada_{bf16_m16n8k16,e4m3_m16n8k32}.json`: 360
  GPU-measured records each (40 per family, ~73 KB each), hex `acc`, `a`, `b`,
  `d`.
- `tests/veritor/core/test_silicon.py`: goldens reproduce bit-exactly (both
  directly and through the `GateSet`), the ten `tc-dot-spec` goldens reproduce
  under the synthetic pipeline and differ under Ada's, the minimal Ada/Hopper
  FP8 separating tile, sticky overflow, 14-bit accumulator truncation, chain =
  sequential fold, domain errors; and a `slow` test
  (`HAWKEYE_GPU_SIMULATOR_DIR=/path/to/built/gpu-simulator`, optional
  `HAWKEYE_TILES`) that runs Hawkeye's three simulators against the port on
  random tiles (skipped when the simulator is not importable).
- `gpu/tensor-core-semantics/`: `mma_tiles.cu`, `tcs_gpu.py`, `characterize.py`,
  `validate_tiles.py`, `spec_vectors.py`, `probe_inf.py`, `hawkeye_vs_port.py`,
  `gemm_chain.cu`, `ref_chain.cpp`, `gemm_chain.py`, and `results/` (JSON and
  logs named above; the 256,000-line spec vector files are not committed, they
  regenerate from `spec_vectors.py` with the seed in the script).

## 9. What remains

1. **Hopper (H100) and Ampere (A100) on silicon.** The `HOPPER_*` and
   `AMPERE_*` pipelines are Hawkeye's, checked only against Hawkeye's
   simulators. An H100 run of the same harness (`wgmma` for FP8, `mma.sync`
   for BF16/FP8) would pin them, and would settle whether Hopper's FP8
   `mma.sync` (as opposed to `wgmma`) also uses two groups of 16. Add
   Blackwell when available.
2. **Non-finite domain.** Measured but not modelled (section 4): sticky
   infinities, canonical NaN `0x7fffffff`. Needed before the gate is total;
   a few hundred targeted probes plus a 10-line extension of `tc_dot`.
3. **FP16 operands** (`m16n8k16.f16`) and **FP16/BF16 accumulators**
   (`.f16.f16.f16`): not measured. Hawkeye reports FP16 uses the same pipeline
   as BF16 per architecture; the accumulator-in-FP16 variant needs its own
   rounding study.
4. **Kernel performance with the fixed order preserved:** shared-memory staging,
   `ldmatrix`, double buffering; none of these change the per-element chain.
   Also confirm the claim that cuBLAS/cuBLASLt itself is *not* reproducible by
   the chain at these shapes (it uses split-K and different k orders; the chain
   is the price of pinning).
5. **Elementwise and reduction ops on CUDA cores**, whose semantics also need
   pinning before a whole transformer is a circuit. What they depend on:
   - *Softmax*: `expf` implementation (libdevice `__expf` vs `expf` vs a
     framework's polynomial), the max-subtraction and the **reduction order**
     of the row sum (warp shuffles, block size, PyTorch's vectorised reduction
     tree), FP32 vs mixed precision, and whether `--use_fast_math` /
     `-ftz=true` / `-prec-div` were on. Exact FP32 `expf` is available (CUDA's
     `expf` is 2 ulp, not correctly rounded, so the *specific* implementation
     must be pinned, e.g. by porting the PTX sequence).
   - *GELU*: tanh-approximation vs erf form; `tanhf`/`erff` implementations
     (both non-correctly-rounded libdevice functions), constant folding by the
     compiler, FMA contraction (`-fmad=true` by default changes results).
   - *LayerNorm*: two-pass vs Welford variance, reduction order across the
     hidden dimension, `rsqrtf` (approximate, 2 ulp) vs `1/sqrtf`, epsilon
     placement, FMA contraction, whether the output is rounded to BF16 before
     the next matmul.
   - *Residual adds, casts, scaling*: FP32 add order is fixed per element
     (single op), casts are RNE per IEEE except for FP8 saturation behaviour
     (`float8_e4m3fn` saturates vs NaN) which must be stated per framework.
   - *Attention `QK^T` and `PV`*: tensor-core matmuls again (this gate), but
     fused kernels (FlashAttention) rescale the softmax online — the fused
     order is a different, and undocumented, chain.

   The common pattern: each of these is a short fixed sequence of FP32 ops whose
   result is determined by (a) the exact libdevice function versions,
   (b) compiler flags (FMA contraction, fast-math, FTZ), and (c) the reduction
   tree. Pinning them means either committing to a specific PTX and porting it
   instruction-by-instruction (as done here for `mma.sync`), or replacing them
   with correctly-rounded implementations whose semantics are mathematical
   rather than implementation-defined.
6. **Commitment layout**: how 8/16-bit operand words are packed into the
   32-bit words the protocol commits to (section 7).

## 10. Files

~~~
docs/hardware-semantics.md                       this report
src/veritor/core/silicon.py                      the semantics and gate set
src/veritor/core/__init__.py                     exports (appended lines only)
tests/veritor/core/test_silicon.py               tests
tests/veritor/core/conftest.py                   `slow` marker
tests/veritor/core/golden/ada_bf16_m16n8k16.json 360 GPU records
tests/veritor/core/golden/ada_e4m3_m16n8k32.json 360 GPU records
gpu/tensor-core-semantics/mma_tiles.cu           one-instruction probe kernel
gpu/tensor-core-semantics/tcs_gpu.py             ctypes driver, bit casts
gpu/tensor-core-semantics/characterize.py        layout / groups / width / floor recovery
gpu/tensor-core-semantics/validate_tiles.py      >= 100k tiles + edge families vs model (+ Hawkeye)
gpu/tensor-core-semantics/spec_vectors.py        vectors for tc-dot-spec, spec goldens on GPU
gpu/tensor-core-semantics/probe_inf.py           non-finite probes
gpu/tensor-core-semantics/hawkeye_vs_port.py     Hawkeye C++ simulators vs the port (CPU)
gpu/tensor-core-semantics/gemm_chain.cu          fixed-order GEMM
gpu/tensor-core-semantics/ref_chain.cpp          C++/OpenMP reference chain
gpu/tensor-core-semantics/gemm_chain.py          GEMM harness: random + GPT-2, cuBLAS timing
gpu/tensor-core-semantics/results/               measurements (JSON, logs)
~~~
