# Datacenter realities for Verity: literature brief (2026-09-02)

Mechanism tags used in the implications: INV = invariant (does not change the circuit), CFG = public configuration, ADV = server advice charged in bits, CIRC = decided inside the circuit (padding/masking/comparators), FAULT = replay-time fault declaration. RU = replay unit (e.g. one request), VU = verification unit (e.g. one dot product).

## 1. Continuous batching and scheduling

| Fact | Number / mechanism | Source (year) |
|---|---|---|
| Iteration-level scheduling | Orca schedules per decode iteration and "selectively batches" only non-attention ops across requests; 36.9x throughput vs FasterTransformer at equal latency | [Orca, OSDI (2022)](https://www.usenix.org/conference/osdi22/presentation/yu) |
| Batch composition changes every step | vLLM caps `max_num_seqs` (default 256) and `max_num_batched_tokens`; requests join and leave at any iteration | [vLLM engine args (2024)](https://docs.vllm.ai/en/v0.6.1.post2/models/engine_args.html) |
| No padding in production batches | TensorRT-LLM in-flight batching packs tokens of all requests into one padding-free tensor; `max_num_tokens` default 8192 per forward; `max_batch_size` tunable at runtime | [TensorRT-LLM attention docs](https://nvidia.github.io/TensorRT-LLM/advanced/gpt-attention.html), [TensorRT-LLM tuning guide (2025)](https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/tuning-max-batch-size-and-max-num-tokens.html) |
| Root cause of run-to-run nondeterminism is batch-dependent kernel choice, not atomics | cuBLAS/attention pick tile sizes and split-K by M, so the same request in a different batch gets a different reduction order; 1000 identical temperature-0 completions (Qwen3-235B, vLLM) gave 80 distinct outputs | [Thinking Machines (2025)](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) |
| Cost of batch-invariant kernels (TML) | Invariant matmul ~20% slower than cuBLAS; end-to-end vLLM on 1000 sequences: 26 s default, 55 s naive invariant, 42 s with improved attention kernel; distinct outputs 80 -> 1 | [Thinking Machines (2025)](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) |
| Cost in SGLang | Deterministic mode: 25-45% slowdown, mean 34.35% on FlashInfer/FA3 backends | [LMSYS (2025)](https://lmsys.org/blog/2025-09-22-sglang-deterministic/) |
| Cost in vLLM | `VLLM_BATCH_INVARIANT=1`; FA2+LoRA support PR reports ~50% throughput loss; without it, 1000 requests at concurrency 100 produced 26-36 distinct outputs (Qwen3-32B, Qwen2.5-7B) | [vLLM PR #30018 (2025)](https://github.com/vllm-project/vllm/pull/30018), [vLLM docs](https://docs.vllm.ai/en/latest/features/batch_invariance/) |
| Tile quantization | A 257-token prefill chunk costs 32% more than 256 (tile boundary), so schedulers align chunk sizes to tiles | [Sarathi-Serve, OSDI (2024)](https://arxiv.org/abs/2403.02310) |

Implication for Verity:
- Treat each step's batch composition (which RUs, how many tokens each) as ADV or CFG committed per iteration; the kernel/reduction-order choice is then a deterministic function of (M, N, K, hardware) that replay must re-derive, not trust.
- A test RU must be interleavable with arbitrary other RUs at every step: compute the same token position of one request under 3+ batch sizes, and either the circuit reproduces each (reduction order as ADV) or the server runs batch-invariant kernels (CFG) and the outputs are bit-identical.
- If the deployment runs batch-invariant kernels, the 20-50% throughput cost is the price of INV for matmul/attention/RMSNorm; test suites should include both regimes.
- Packed (no-padding) batches make the token-to-request mapping a ragged index map; padding-based CIRC must be explicit because production tensors carry none.
- Sweep M across tile boundaries (255/256/257 tokens) in scenario generators and check the circuit's declared reduction tree changes where the kernel's does.

## 2. Preemption and KV-cache management

| Fact | Number / mechanism | Source (year) |
|---|---|---|
| PagedAttention | KV stored in fixed blocks (vLLM default 16 tokens); fragmentation waste <4% vs 60-80% in prior systems; 2-4x throughput | [vLLM, SOSP (2023)](https://arxiv.org/abs/2309.06180) |
| Preemption modes | Swap (KV copied to CPU and back) vs recompute (KV discarded, request re-prefilled); vLLM V1 uses recompute (swap was a V0 option); a preempted request's re-prefill may hit the prefix cache | [vLLM optimization docs (2025)](https://docs.vllm.ai/en/latest/configuration/optimization/) |
| Eviction rate in practice | No public fleet-wide preemption rate; vLLM logs a warning per preemption and operators tune `gpu_memory_utilization`/`max_num_seqs` to avoid it (uncertain) | [vLLM optimization docs (2025)](https://docs.vllm.ai/en/latest/configuration/optimization/) |
| Prefix sharing (RadixAttention) | Radix tree over token prefixes, LRU eviction, cache-aware scheduling; up to 6.4x throughput; reported hit rates roughly 50-99% across its workloads | [SGLang, NeurIPS (2024)](https://arxiv.org/abs/2312.07104) |
| Hit rates in production | DeepSeek: 608B input tokens/day, 342B (56.3%) served from on-disk KV cache; agentic coding traces: 94.2% hit rate, 131:1 input:output tokens, ~2,242 new tokens/turn, contexts grow 12K -> 80K (max >180K) | [DeepSeek (2025)](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md), [vLLM blog (2026)](https://vllm.ai/blog/2026-05-06-mooncake-store) |
| API-level caching | OpenAI: automatic for prompts >=1024 tokens in 128-token increments, evicted after 5-10 min idle (<=1 h); Anthropic: explicit `cache_control`, 5 min TTL (1 h at 2x), writes 1.25x, reads 0.1x, min 1024 tokens (Sonnet) | [OpenAI docs (2025)](https://platform.openai.com/docs/guides/prompt-caching), [Anthropic docs (2025)](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) |
| KV bytes per token (BF16, all layers) | DeepSeek-V3 (MLA) 70.272 KB; Qwen-2.5 72B 327.68 KB; Llama-3.1 405B 516.096 KB | [DeepSeek hardware insights (2025)](https://arxiv.org/abs/2505.09343) |

Implication for Verity:
- A request must be able to join twice: swap preemption keeps its KV (the circuit resumes on the same K/V values); recompute preemption re-prefills (new prefill gates whose outputs equal the earlier ones only under INV kernels). Model it as an ADV event "RU r left at step t, rejoined at step t'", not a FAULT.
- Prefix-cache hits mean an RU's K/V for the shared prefix were produced by another RU, possibly on another node or day; the commitment must link the consuming RU to the producing RU's gates, or the cache read is ADV ("these blocks came from RU r'") charged in bits.
- Block granularity (16 tokens) is CFG; the trailing partial block of a prompt is recomputed, so hit boundaries in tests must be block-aligned.
- With 56-94% of input tokens served from cache, most "prefill" gates of a production RU do not exist; headline cost estimates must count only computed tokens.
- LRU eviction makes re-prefill of an evicted prefix a legitimate second computation of the same tokens; tests should include it and require bit-equality only under INV kernels.

## 3. Chunked prefill and prefill/decode disaggregation

| Fact | Number / mechanism | Source (year) |
|---|---|---|
| Chunked prefill | Prompt split into chunks sized by a token budget derived from the TBT SLO; decode tokens ride along with prefill chunks ("stall-free batching") | [Sarathi-Serve (2024)](https://arxiv.org/abs/2403.02310) |
| Default in vLLM V1 | Chunked prefill on by default; decode requests scheduled first, prefill fills the remaining `max_num_batched_tokens` | [vLLM docs (2025)](https://docs.vllm.ai/en/latest/configuration/optimization/) |
| DistServe | Separate prefill and decode GPU pools; 7.4x more requests or 12.6x tighter SLO at the same goodput; KV moved over NVLink/IB per request | [DistServe, OSDI (2024)](https://arxiv.org/abs/2401.09670) |
| Splitwise KV transfer | Per-layer transfer overlapped with prefill; non-overlapped tail ~8 ms (A100) / ~5 ms (H100); overhead <7% of prompt compute | [Splitwise, ISCA (2024)](https://arxiv.org/abs/2311.18677) |
| Mooncake (Kimi) | KVCache-centric disaggregation with a CPU-DRAM/SSD KV pool; up to 525% throughput in simulation, ~75% more requests in production | [Mooncake, FAST (2025)](https://arxiv.org/abs/2407.00079) |
| DeepSeek-V3 serving layout | Prefill: 4 nodes / 32 H800, attention TP4+SP+DP8, MoE EP32 with 32 redundant experts; decode: 40 nodes / 320 GPUs, EP320 (1 routed expert per GPU, 64 GPUs hold redundant + shared experts); all-to-all over IB | [DeepSeek-V3 (2024)](https://arxiv.org/abs/2412.19437) |
| DeepSeek production rates | ~73.7k input tok/s per H800 node in prefill, ~14.8k output tok/s per node in decode; average 226.75 nodes online over 24 h | [DeepSeek (2025)](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md) |

Implication for Verity:
- The prefill and decode gates of one RU can run on different GPUs, nodes, parallel layouts (TP4/DP8 vs EP320) and hardware; "which circuit" is declared per phase (CFG per pool) and the KV handoff is an ADV edge between two sub-circuits.
- Chunked prefill turns a prompt's attention into chunk-sized calls with different kernel shapes from whole-prompt prefill; run the same prompt as 1 chunk and as k chunks and confirm the circuit declares the chunking.
- KV transfer is a pure copy (INV) when bit-exact; inject one corrupted KV block during transfer and confirm the consuming decode gates fail verification.
- A decode step mixes tokens from many RUs plus prefill chunks; the RU boundary must be defined over this ragged mix, not as "one request = one kernel call".

## 4. Speculative decoding

| Fact | Number / mechanism | Source (year) |
|---|---|---|
| Speculative sampling | Draft k tokens with a cheap model; target scores k+1 positions in one forward; accept token i with prob min(1, p_i/q_i), on first rejection resample from norm(max(0, p-q)); distribution equals the target's "within hardware numerics"; 2-2.5x speedup on Chinchilla-70B | [Chen et al. (2023)](https://arxiv.org/abs/2302.01318), [Leviathan et al. (2023)](https://arxiv.org/abs/2211.17192) |
| Medusa | Extra decoding heads on the target plus tree attention over candidates; 2.2-3.6x speedup | [Medusa (2024)](https://arxiv.org/abs/2401.10774) |
| EAGLE-3 | Feature-level draft trained on the target's multi-layer features; speedup up to 6.5x (~1.4x over EAGLE-2) | [EAGLE-3 (2025)](https://arxiv.org/abs/2503.01840) |
| DeepSeek-V3 MTP | One extra MTP module predicts token t+2 and serves as the draft at inference: 85-90% acceptance of the 2nd token, ~1.8x TPS | [DeepSeek-V3 (2024)](https://arxiv.org/abs/2412.19437) |
| vLLM guarantee scope | Lossless only "up to hardware numerics"; docs state batch-size changes alter logprobs ("non-deterministic behavior in batched operations"); greedy-equality tests are the acceptance criterion | [vLLM spec-decode docs (2025)](https://docs.vllm.ai/en/latest/features/speculative_decoding/) |
| Per-position logits are not bit-identical | The verification forward processes k+1 tokens per sequence (prefill-shaped kernels) instead of 1, changing M and hence kernel/reduction order unless batch-invariant kernels are used | [Thinking Machines (2025)](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) |

Implication for Verity:
- A speculative step's target circuit has k+1 query positions per RU; the accept/reject gate consumes draft probabilities q, target probabilities p and a uniform u, so q (draft gates, or ADV if the draft is untrusted) and the RNG stream must both be inside the commitment.
- Rejected positions are computed-then-discarded gates that still cost compute and can leak bits; the token output is a function of (p, q, u), not of p alone.
- Decode the same prompt with k=0 and k=3: under non-invariant kernels the logits differ bitwise, so k and the draft must be declared per step (CFG/ADV) rather than assumed equivalent.
- MTP-style drafts share trunk activations with the target, so the draft head is part of the same circuit and its VUs are ordinary VUs.
- Tree drafts (Medusa/EAGLE) use custom attention masks; the mask is CIRC, reconstructed from the tree shape declared as ADV.

## 5. Mixture-of-experts routing and capacity

| Fact | Number / mechanism | Source (year) |
|---|---|---|
| DeepSeek-V3/R1 routing | 256 routed + 1 shared expert per MoE layer; top-8 by sigmoid affinity plus a learned bias (aux-loss-free balancing); node-limited to <=4 nodes per token; no token dropping in training or inference; 61 layers, first 3 dense; 671B total / 37B active | [DeepSeek-V3 (2024)](https://arxiv.org/abs/2412.19437) |
| Mixtral 8x7B | 8 experts, top-2 softmax gating; 47B total / 13B active per token | [Mixtral (2024)](https://arxiv.org/abs/2401.04088) |
| Qwen3-235B-A22B | 128 experts, 8 activated, no shared expert, 94 layers | [Qwen3 (2025)](https://arxiv.org/abs/2505.09388) |
| Kimi K2 | 384 experts, 8 selected + 1 shared, 61 layers (1 dense), 1.04T total / 32B active; MLA with 64 heads | [Kimi K2 (2025)](https://arxiv.org/abs/2507.20534) |
| Capacity factor and dropping | Switch Transformer: capacity = tokens/experts x CF (CF 1.0-1.25); tokens over capacity are dropped and pass through the residual only | [Switch Transformer (2021)](https://arxiv.org/abs/2101.03961) |
| Routing entropy per token (derived from configs above) | log2 C(256,8) = 48.5 bits/layer -> ~2.8 kbit/token over 58 MoE layers (DeepSeek-V3); Kimi K2 log2 C(384,8) = 53.3 -> ~3.2 kbit; Qwen3-235B 40.4 -> ~3.8 kbit; Mixtral 4.8 -> ~154 bits; gate weights extra if not recomputed | derived |
| Expert-parallel all-to-all | DeepEP normal kernels ~153 GB/s intranode NVLink, ~43-47 GB/s internode RDMA; low-latency decode kernels: dispatch ~163-194 us, combine ~318-369 us for 8-256 experts | [DeepEP (2025)](https://github.com/deepseek-ai/DeepEP) |
| Grouped GEMM shapes depend on the batch | Each expert's GEMM has M = tokens routed to it in this batch (contiguous/masked grouped GEMM layouts) | [DeepGEMM (2025)](https://github.com/deepseek-ai/DeepGEMM) |

Implication for Verity:
- Routing is a top-k over gate logits: decidable in-circuit (CIRC) at 0 advice bits if the gate GEMM is a committed gate; if the server supplies routing as ADV it costs ~2.8-3.8 kbit/token on frontier MoEs, a useful benchmark for "advice is expensive".
- Near-ties in top-k are where a batch-dependent 1-ulp difference flips an expert; construct near-tie gates and confirm the comparator reads committed values, not values recomputed elsewhere.
- Under capacity factors or node-limited routing the experts a token gets depend on the other tokens in the batch; DeepSeek-V3 drops nothing, but Switch-style dropping must be modeled as batch-composition ADV.
- Expert GEMM M varies per batch so per-expert reduction order varies; batch-invariant MoE kernels are rarer than dense ones (vLLM's flag only overrides aten matmuls), so mark MoE layers non-invariant unless proven.
- All-to-all is a permutation (INV) with per-batch sizes; corrupting one dispatched token in transit is caught by the receiving expert's VU only if the dispatch itself is committed.

## 6. Parallelism and numerics

| Fact | Number / mechanism | Source (year) |
|---|---|---|
| TP all-reduce order | NCCL Ring order is fixed by topology and bit-exact across runs; Tree order is not user-controllable; NVLS (in-switch reduction) was nondeterministic on Hopper until CUDA 12.8 / driver 550.144.03 and is always deterministic on Blackwell | [NCCL issue #1497 (2024-25)](https://github.com/NVIDIA/nccl/issues/1497), [Megatron-Core determinism (2026)](https://docs.nvidia.com/megatron-core/developer-guide/nightly/user-guide/deterministic-training.html) |
| Overlap breaks reproducibility | Megatron deterministic mode rejects TP comm overlap and the fused cross-entropy kernel; `NCCL_ALGO=Ring` pins the algorithm but not the physical ring across allocations | [Megatron-Core op catalog (2026)](https://docs.nvidia.com/megatron-core/developer-guide/nightly/developer/determinism/op-catalog.html) |
| Sequence parallel | Replaces the all-reduce with reduce-scatter + all-gather around LayerNorm/dropout, i.e. a different partial-sum grouping than TP all-reduce | [Korthikanti et al. (2022)](https://arxiv.org/abs/2205.05198) |
| Tensor-core accumulation | PTX: for f16/bf16 mma "the accumulation order, rounding and handling of subnormal inputs are unspecified"; Hopper FP8 GEMM keeps ~14 mantissa bits per aligned product (max rel. error ~2% at K=4096) | [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma), [DeepSeek-V3 (2024)](https://arxiv.org/abs/2412.19437) |
| DeepSeek-V3 FP8 scheme | E4M3 throughout; activations quantized per 1x128 tile, weights per 128x128 block; partial sums promoted to FP32 on CUDA cores every 128 K-elements (N_C = 128) | [DeepSeek-V3 (2024)](https://arxiv.org/abs/2412.19437) |
| Serving FP8/FP4 formats | vLLM W8A8: per-tensor or per-channel weight scales with dynamic per-token activation scales; NVFP4: 16-element micro-blocks with an E4M3 scale plus an FP32 per-tensor scale | [vLLM FP8 docs (2025)](https://docs.vllm.ai/en/latest/features/quantization/fp8/), [NVIDIA NVFP4 (2025)](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/) |
| FlashAttention numerics | Online softmax with running max/sum over KV blocks (block-sequential reduction order); forward deterministic, backward uses atomics unless the deterministic flag is set; FA3 evaluates exp2 on the MUFU unit (3.9 TFLOPS vs 989 TFLOPS matmul on H100) | [FlashAttention-2 (2023)](https://arxiv.org/abs/2307.08691), [FlashAttention-3 (2024)](https://arxiv.org/abs/2407.08608) |
| Fast-math transcendentals | `__expf` max error 2 + floor(1.16*abs(x)) ulp; `-use_fast_math` rewrites `expf` to `__expf`; GELU ships in erf, tanh-approximation and sigmoid ("quick") variants | [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#intrinsic-functions), [HF activations.py](https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py) |

Implication for Verity:
- Every reduction gate (dot product, all-reduce, softmax denominator, RMSNorm variance) needs a declared tree; TP degree, collective algorithm (Ring/Tree/NVLS) and physical ring are CFG per deployment and must be fixed before replay.
- The dot-product VU must model the tensor core's actual accumulation (Hawkeye: 24/25-bit internal mantissa, truncation, block structure), not IEEE fp32 sequential sums, or honest servers will be rejected.
- Quantization scales are data-dependent gates (max-abs per 1x128 tile) and must be committed values; a wrong scale changes every downstream product.
- `__expf`/tanh-GELU vs exact variants are different circuits; the SASS-level approximation is CFG and must be pinned (bit-accurate tables such as sass-math).
- Flip one bit of a partial sum inside a K=4096 FP8 GEMM and verify detection: honest ~14-bit accumulation and adversarial corruption must be distinguishable by the gate model.

## 7. Hardware faults and silent data corruption

| Fact | Number / mechanism | Source (year) |
|---|---|---|
| Llama 3 405B pre-training | 466 interruptions in 54 days on 16K H100s: 47 planned, 419 unexpected; 78% of unexpected were hardware; GPU faults 58.7% of unexpected (148 faulty GPU/NVLink = 30.1%, 72 HBM3 = 17.2%); 6 SDC events (1.4%); ~90% effective training time | [Llama 3 (2024)](https://arxiv.org/abs/2407.21783) |
| Gemini Ultra training | SDC events expected "every week or two"; deterministic replay isolates bad hardware, plus SDC scanners and hot standbys; goodput 85% -> 97% via in-memory redundant state | [Gemini (2023)](https://arxiv.org/abs/2312.11805) |
| Fleet SDC rate | SDC now ~1 fault per 1,000 devices vs the historical 1-per-million soft-error rate; detection via Fleetscanner, Ripple, Hardware Sentinel | [Meta engineering (2025)](https://engineering.fb.com/2025/07/22/data-infrastructure/how-meta-keeps-its-ai-hardware-reliable/), [Meta SDC at scale (2021)](https://arxiv.org/abs/2102.11245) |
| Mercurial cores | "A few mercurial cores per several thousand machines" compute wrong results reproducibly for specific inputs | [Google, HotOS (2021)](https://research.google/pubs/cores-that-dont-count/) |
| GPU memory errors | HBM ECC corrects single-bit errors; double-bit errors raise Xid 48 and row remapping (Xid 63/64); uncorrectable errors kill the context rather than continue silently | [NVIDIA Xid errors](https://docs.nvidia.com/deploy/xid-errors/index.html) |
| NaN/Inf in serving | Unmasked/uninitialized KV blocks produced NaN attention rows and `probability tensor contains inf/nan` at sampling (vLLM bugs fixed by masking/padding) | [vLLM issue #641 (2023)](https://github.com/vllm-project/vllm/issues/641) |
| Retries and duplicates | OpenAI SDK retries 2x by default (408/409/429/5xx, timeouts) with an idempotency-key header; a timed-out request that actually completed yields duplicate server execution | [openai-python README (2025)](https://github.com/openai/openai-python/blob/main/README.md) |
| Client disconnect | vLLM aborts the engine request on the ASGI `http.disconnect` event and frees KV; a middleware bug made `is_disconnected()` return False so requests ran to `max_tokens` | [vLLM issue #10087 (2024)](https://github.com/vllm-project/vllm/issues/10087), [vLLM PR #11190 (2024)](https://github.com/vllm-project/vllm/pull/11190) |

Implication for Verity:
- Honest SDC at ~1e-3 per device means a 10^4-GPU fleet has ~10 silently wrong devices at any time; the leakage bound must treat a small background rate of corrupted VUs as baseline, and FAULT declarations must be cheap enough that honest servers use them (after a failed self-check, NaN, Xid).
- A NaN/Inf in logits is an in-circuit detectable event (CIRC): the sampler gate needs a defined behavior (abort RU or FAULT) rather than undefined output.
- Duplicate execution: one request id can produce two RUs; include a retried request whose first attempt completed and confirm both RUs are separately committed or the duplicate is declared.
- Disconnect/abort truncates an RU mid-decode; the RU boundary must accept "terminated at step t by abort" as a legal ending distinct from EOS/`max_tokens`.
- ECC and row-remapping events are host-visible (Xid) with timestamps; deployments can surface them as FAULT declarations that tests replay.

## 8. Sampling and randomness

| Fact | Number / mechanism | Source (year) |
|---|---|---|
| Order of logit processing | Logits -> penalties (repetition/frequency/presence) -> temperature divide -> top-k/top-p/min-p masks -> softmax -> sample; all elementwise/sort ops on the [B, V] logits after the transformer | [vLLM sampler source](https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/sampler.py) |
| Sampling trick | vLLM samples by exponential race, argmax(probs / Exp(1)) (equivalent to Gumbel-max), using a per-request `torch.Generator` when a seed is given | [vLLM topk_topp_sampler](https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/ops/topk_topp_sampler.py) |
| RNG engines | PyTorch CUDA: Philox4x32-10 counter-based, state = (seed, offset) advanced per kernel launch; JAX: Threefry2x32 keys, `categorical` uses Gumbel-max and its default "low" mode is biased for probabilities <1e-7 in float32 | [PyTorch CUDAGeneratorImpl](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cuda/CUDAGeneratorImpl.cpp), [JAX docs](https://docs.jax.dev/en/latest/_autosummary/jax.random.categorical.html) |
| Penalty formulas | OpenAI: logit -= count x alpha_frequency + 1[count>0] x alpha_presence (coefficients -2..2); HF `repetition_penalty` divides positive / multiplies negative logits of seen tokens; `logit_bias` adds -100..100 per token | [OpenAI advanced usage (2025)](https://developers.openai.com/api/docs/guides/advanced-usage) |
| Provider determinism | OpenAI `seed` is best-effort; `system_fingerprint` changes when backend numerics change ("a few times a year"); outputs then "mostly identical" | [OpenAI cookbook (2023)](https://developers.openai.com/cookbook/examples/reproducible_outputs_with_the_seed_parameter) |
| Constrained decoding | Grammar/JSON schema compiled to a per-step token mask (Outlines FSM index; XGrammar adaptive token-mask cache, ~100x faster masking); OpenAI Structured Outputs: 100% schema adherence vs <40% by prompting | [Outlines (2023)](https://arxiv.org/abs/2307.09702), [XGrammar (2024)](https://arxiv.org/abs/2411.15100), [OpenAI (2024)](https://openai.com/index/introducing-structured-outputs-in-the-api/) |
| Stop conditions | `stop` up to 4 strings (matched on detokenized text), `max_tokens`, EOS; `n` parallel samples share one prefill | [OpenAI API parameters](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb) |

Implication for Verity:
- The transformer circuit is unchanged by sampling parameters; only the sampler tail ([B, V] elementwise ops, one sort for top-p, one RNG draw) differs, so temperature/top-p/top-k/penalties/logit_bias/grammar mask are CFG or public per-request parameters, not advice.
- The random draw must be committed: either a seeded counter-based RNG (Philox seed+offset as CFG, making the uniform CIRC) or the server's uniform is ADV charged at roughly log2(V) bits per token; include unseeded requests in tests to see that cost.
- Top-p needs a sort whose tie-breaking is algorithm-dependent; equal probabilities occur after bf16 rounding, so the circuit must define tie order (CIRC).
- Grammar masks and stop-string detection are functions of the token history (INV given history); repetition penalties add a data-dependent [B, V] gate that must read the committed history.
- Provider fingerprints change a few times a year; replay must pin kernel/library versions as CFG per RU.

## 9. Request lifecycle

| Fact | Number / mechanism | Source (year) |
|---|---|---|
| Termination reasons | `finish_reason` in {stop, length, tool_calls, content_filter}; streaming emits deltas and a final usage chunk | [OpenAI Python reference (2025)](https://developers.openai.com/api/reference/python/) |
| Multi-turn / tool calls | Each turn re-sends the whole conversation and only the suffix is new; agentic traces: 33 LLM calls per task, 131:1 input:output, 94.2% of input tokens cached, average output 520 tokens/call (median 246) | [Codex/SWE-bench Pro traces (2026)](https://huggingface.co/datasets/DiscoPosse/codex_swebenchpro_traces), [vLLM blog (2026)](https://vllm.ai/blog/2026-05-06-mooncake-store) |
| Cache lifetimes | OpenAI 5-10 min idle (<=1 h off-peak); Anthropic 5 min (1 h paid); after expiry the same prefix is re-prefilled | [OpenAI docs (2025)](https://platform.openai.com/docs/guides/prompt-caching), [Anthropic docs (2025)](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) |
| Retries | 2 automatic retries with exponential backoff and an idempotency header; proxy upstream timeouts default to 6000 s in LiteLLM | [openai-python (2025)](https://github.com/openai/openai-python/blob/main/README.md), [LiteLLM PR #30223 (2026)](https://github.com/BerriAI/litellm/pull/30223) |
| Cache-aware load balancing | SGLang router keeps an approximate radix tree per worker and routes to the longest prefix match; NVIDIA Dynamo routes by KV overlap and disaggregates prefill/decode | [SGLang model gateway docs (2026)](https://docs.sglang.ai/advanced_features/sgl_model_gateway.html), [NVIDIA Dynamo (2025)](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/) |
| Model versions | Dated snapshots plus `system_fingerprint` identifying weights + infrastructure + numerics config; changes "a few times a year" | [OpenAI cookbook (2023)](https://developers.openai.com/cookbook/examples/reproducible_outputs_with_the_seed_parameter) |
| LoRA multi-tenancy | S-LoRA: thousands of adapters per GPU, up to 4x throughput vs PEFT and 30x vs vLLM-packed; Punica SGMV kernel: 12x throughput at +2 ms/token; adapter chosen per request | [S-LoRA (2023)](https://arxiv.org/abs/2311.03285), [Punica (2023)](https://arxiv.org/abs/2310.18547) |

Implication for Verity:
- Define the RU as one API request (one prefill + decode run on one replica); conversation continuity is a chain of RUs linked by cache-read ADV edges, and a tool-call turn is a new RU whose prompt includes the previous RU's output.
- Tests need all four terminations (EOS, `max_tokens`, stop string, tool_calls) plus abort; `length` truncation means the last committed logits were never sampled to EOS.
- Replica choice is public per RU (replica id, model snapshot, fingerprint); hot swaps mid-conversation mean consecutive RUs can run different circuits, so model version is per RU, not per conversation.
- LoRA adds per-request low-rank GEMMs (SGMV) whose grouping depends on which adapters share a batch; adapter id is public per RU and the SGMV grouping is batch-composition ADV.
- Retries create duplicate RUs with identical inputs; RU sampling must prevent a server from substituting the "good" duplicate for the sampled one.

## 10. Bit-exact reproducibility of GPU math

| Fact | Number / mechanism | Source (year) |
|---|---|---|
| Hawkeye | Reverse-engineers tensor-core MMA: Ampere FP16/BF16 = two-stage pyramid (C + P1..P8, then + P9..P16), 24-bit internal mantissa, round-toward-zero, single normalization at the end; Hopper = one stage over 16 products with a 25-bit mantissa; 100% bit-exact on 100,000 random 16x16 tiles across Ampere/Hopper/Lovelace and FP16/BF16/FP8 | [Hawkeye, MLSys (2026)](https://arxiv.org/abs/2603.20421), [gpu-simulator](https://github.com/badasherez/gpu-simulator) |
| Hawkeye CPU cost | 4096x4096 matmul emulated in 40.6-52.5 s on an Apple M4 Pro (FP8/Hopper fastest) | [Hawkeye (2026)](https://arxiv.org/abs/2603.20421) |
| NVIDIA specification | PTX leaves mma accumulation order/rounding/subnormals unspecified; Hopper FP8 accumulation retains ~14 bits | [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma), [DeepSeek-V3 (2024)](https://arxiv.org/abs/2412.19437) |
| cuBLAS reproducibility | Same toolkit + same architecture + same SM count -> bit-wise identical runs; not guaranteed across toolkit versions or SM counts; multi-stream use needs one workspace per stream | [cuBLAS docs](https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility) |
| PyTorch deterministic mode | `torch.use_deterministic_algorithms(True)` plus `CUBLAS_WORKSPACE_CONFIG=:4096:8`; some ops raise, others switch to slower deterministic kernels | [PyTorch randomness notes](https://docs.pytorch.org/docs/stable/notes/randomness.html) |
| XLA/JAX and TPU | `--xla_gpu_deterministic_ops=true` disables atomics-based reductions; Google's TPU stack is "fully deterministic", enabling replay-based SDC isolation | [XLA flags](https://github.com/openxla/xla/blob/main/xla/debug_options_flags.cc), [Gemini (2023)](https://arxiv.org/abs/2312.11805) |
| Non-associativity | Atomics-based reductions produce run-to-run variation in HPC and DL codes; FP16 example: (65504 + -65504) + 1 = 1 but 65504 + (-65504 + 1) = 0 | [Shanmugavelu et al. (2024)](https://arxiv.org/abs/2408.05148), [Hawkeye (2026)](https://arxiv.org/abs/2603.20421) |

Implication for Verity:
- A VU "one dot product" is well-defined only once (architecture, precision, tile order) are CFG; Hawkeye shows the spec is recoverable and emulatable, so the zkVM gate should implement the Hawkeye model per architecture, not IEEE fp32.
- Reproducibility is per SM-count and toolkit: a replica on another GPU SKU is a different circuit, so the RU must carry hardware and library identifiers.
- Deterministic modes exist in every stack (PyTorch/XLA/Megatron/vLLM/SGLang) at 25-50% cost; the protocol can require them (CFG) for all RUs, since the server cannot know which RUs will be sampled.
- Validate the circuit emulator against real hardware on random tiles (as Hawkeye did) before adversarial tests; a 1-in-100,000 honest mismatch would swamp the fault budget.

## 11. Verifiable / audited inference prior work

| Approach | Number / mechanism | Source (year) |
|---|---|---|
| TOPLOC (heuristic commitment) | Commits top-k activation indices/values per 32 tokens in 258 bytes; detected model/prompt/precision substitution with 100% accuracy in tests; robust to GPU type, TP and batch changes; no proof, the verifier recomputes | [TOPLOC (2025)](https://arxiv.org/abs/2501.16007) |
| Spot-checking / refereed delegation | Proof of Sampling verifies a random fraction and makes honesty the Nash equilibrium; Verde uses refereed delegation with RepOps (bitwise-reproducible operators) so disputes are decidable by bisection | [PoSP (2024)](https://arxiv.org/abs/2405.00295), [Verde (2025)](https://arxiv.org/abs/2502.19405) |
| zkLLM | Full-inference proof for LLaMA-2 13B in <15 min on GPU, proof <200 kB, verification 1-3 s; perplexity increase <0.01 from discretization | [zkLLM, CCS (2024)](https://arxiv.org/abs/2404.16109) |
| zkPyTorch / Expander | Llama-3 8B: ~150 s per token on one CPU thread, 99.32% cosine similarity to floating-point outputs; VGG-16 2.2 s/image in the paper vs 6.3 s in Polyhedra's later blog (sources disagree; blog origin was unreachable on 2026-09-02) | [zkPyTorch (2025)](https://eprint.iacr.org/2025/535), [press summary (2025)](https://cryptonews.net/news/blockchain/31051709/) |
| EZKL / ZKTorch / Modulus | EZKL (Halo2 from ONNX): 65.88x faster and 98% less memory than RISC Zero on small models; ZKTorch compiles models to ~61 basic blocks with proof accumulation; Modulus's "Cost of Intelligence" benchmarked the first generation | [EZKL benchmarks (2024)](https://blog.ezkl.xyz/post/benchmarks/), [ZKTorch (2025)](https://arxiv.org/abs/2507.07031) |
| zkVMs with precompiles | SP1 Hypercube proves ~93% of Ethereum blocks in <12 s on 16 GPUs; RISC Zero and OpenVM expose precompiles/extensions; no published ML-operator (GEMM/attention) proving throughput for any of them | [Succinct (2025)](https://blog.succinct.xyz/sp1-hypercube/) |
| Trusted execution | H100 confidential-computing mode: <7% throughput overhead at batch 1 and ~19% TTFT overhead on an 8B model; 45-70% throughput gap when models are swapped; trust moves to NVIDIA attestation and absence of side channels | [Zhu et al. (2024)](https://arxiv.org/abs/2409.03992), [CC GPUs (2025)](https://arxiv.org/abs/2505.16501) |

Gap Verity fills:
- zkML proves whole inferences at 10^2-10^3 s per token; Verity proves only sampled VUs inside sampled RUs, so cost scales with the sample, not the model.
- TOPLOC and PoSP tolerate numerics by thresholds and re-execution; Verity needs the bit-exact gate model (Hawkeye) so a VU proof is decisive rather than statistical.
- Verde/RepOps show kernel-level bit-reproducibility is achievable; Verity's addition is charging non-reproducible or server-chosen behavior in bits instead of forbidding it.
- TEEs attest code, not arithmetic correctness under SDC; Verity's leakage bound is the alternative when the hardware itself is untrusted.

## 12. Scale facts for the headline estimate

| Fact | Number / mechanism | Source (year) |
|---|---|---|
| Installed AI compute | ~20M H100-equivalents sold cumulatively by end-2025; ~12-16M operational depending on install lag; five hyperscalers hold 71% | [Epoch AI (2026)](https://epoch.ai/gradient-updates/frontier-labs-dont-use-most-ai-compute), [Epoch AI (2026)](https://epoch.ai/data-insights/hyperscalers-control-most-compute) |
| Peak FLOP/s (derived) | 16M H100e x 1.979e15 dense FP8 FLOP/s = ~3.2e22 FLOP/s installed peak; fleet-wide utilization is not public | [NVIDIA H100 specs](https://www.nvidia.com/en-us/data-center/h100/), derived |
| Tokens per day, providers | Google 1.3 quadrillion tokens/month (~43T/day, Oct 2025; 480T/month in May 2025); OpenAI API 6B tokens/min (~8.6T/day) and 800M weekly ChatGPT users (Oct 2025) | [Google I/O (2025)](https://blog.google/technology/ai/io-2025-keynote/), [PYMNTS (2025, press)](https://www.pymnts.com/news/artificial-intelligence/2025/openai-bests-google-in-race-for-consumer-ai-token-consumption/) |
| DeepSeek 24 h profile | 608B input tokens (56.3% cache hits) + 168B output tokens per day on average 226.75 H800 nodes (1,814 GPUs); derived: 776B tok/day x 74 GFLOP/tok (2 x 37B) = ~6.6e17 FLOP/s = ~18% of FP8 peak (ignores attention FLOPs and cache savings) | [DeepSeek (2025)](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md), derived |
| Chat trace lengths | ShareGPT: mean 161 input / 338 output tokens; Alpaca: 19 / 58; LMSYS-Chat-1M: 69.5 prompt / 214.5 response tokens on average | [vLLM paper (2023)](https://arxiv.org/abs/2309.06180), [LMSYS-Chat-1M (2023)](https://arxiv.org/abs/2309.11998) |
| Azure production traces | Coding: median 1,500 prompt / 13 output tokens; conversation: median 1,020 / 129 | [Splitwise (2024)](https://arxiv.org/abs/2311.18677) |
| BurstGPT (Azure OpenAI regional service) | Request lengths Zipf-distributed; ChatGPT response lengths bimodal and growing linearly with request length; 5.29M requests over 121 days in v1 (later releases larger) | [BurstGPT (2024)](https://arxiv.org/abs/2401.17644) |
| Long-context production (Kimi/Mooncake trace) | Average input 7,590 / output 182 tokens; conversation 12,035 / 343; tool-and-agent 8,596 / 182 | [Mooncake (2025)](https://arxiv.org/abs/2407.00079) |

Implication for Verity:
- Input tokens outnumber output tokens 8-130x in production traces and 56-94% of inputs are cache hits; RU compute is dominated by decode gates plus cache-miss prefill, so per-RU gate counts should come from traces, not from "prompt length x layers".
- At ~1e13-1e14 tokens/day per major provider and ~74 GFLOP/token for a 37B-active MoE, a provider runs ~1e21-1e22 FLOP/day of forward passes; RU/VU sampling rates and proving budgets must be set against these totals.
- Fleet utilization near 15-30% of peak (DeepSeek-derived) leaves headroom for batch-invariant or deterministic kernels (25-50% cost) without new hardware.

## Top facts

1. Nondeterminism in serving is caused by batch-size-dependent kernel selection (tile/split-K), not atomics; fixing it costs ~20% per matmul and 34-50% end to end (TML 2025, LMSYS 2025, vLLM 2025).
2. Production batches are packed with no padding and change composition every iteration (Orca 2022, TensorRT-LLM); the same token is computed under a different M each step.
3. A 257-token chunk costs 32% more than 256: schedulers align to tiles, so tests must straddle tile boundaries (Sarathi-Serve 2024).
4. 56.3% (DeepSeek) to 94.2% (Codex agent traces) of input tokens are served from KV cache; most prefill gates of a production RU were computed by another RU.
5. Agentic workloads are 131:1 input:output with ~2,242 new tokens per turn and contexts to 80K-180K tokens (vLLM 2026).
6. Preemption is recompute in vLLM V1: a request can be re-prefilled mid-life, and its re-prefill may itself hit the cache.
7. One RU's prefill and decode can run on different GPU pools with different parallel layouts (DeepSeek-V3: TP4+DP8/EP32 prefill vs EP320 decode).
8. KV per token spans 70 KB (DeepSeek-V3 MLA) to 516 KB (Llama-3.1 405B); KV handoff is a copy but its size and node vary per RU.
9. Speculative decoding verifies k+1 positions per step; DeepSeek MTP accepts the 2nd token 85-90% of the time; per-position logits are bit-identical to plain decoding only under batch-invariant kernels.
10. MoE routing is ~48-53 bits per token per layer (top-8 of 256/384), ~2.8-3.8 kbit per token per model if taken as advice; zero bits if decided in-circuit from committed gate logits.
11. Expert GEMMs have batch-dependent M (tokens routed to that expert), so MoE layers are non-invariant even when dense layers are fixed.
12. Hopper FP8 tensor cores accumulate with ~14 mantissa bits (max rel. error ~2% at K=4096); DeepSeek promotes to FP32 every 128 K-elements, which changes the gate structure.
13. Tensor-core accumulation order is unspecified by NVIDIA but recoverable: Ampere sums in two 8-product stages with 24-bit mantissa and truncation, Hopper in one 16-product stage with 25 bits (Hawkeye 2026, 100% bit-exact on 100,000 tiles).
14. cuBLAS is bit-reproducible only for the same toolkit, architecture and SM count; NCCL Ring is reproducible, Tree and (pre-CUDA-12.8 Hopper) NVLS are not.
15. Fleet SDC is ~1 per 1,000 devices (Meta 2025); Gemini training saw an SDC "every week or two"; Llama-3 405B had 6 SDC events among 419 unexpected interruptions in 54 days.
16. Sampling is a [B, V] tail after the transformer: exponential-race/Gumbel-max with Philox (seed, offset) state; the uniform must be committed or charged at ~log2(V) bits per token.
17. Constrained decoding and penalties are token masks/biases computed from committed history (XGrammar, Outlines); they never change transformer gates.
18. Requests end by EOS, `max_tokens`, stop string, tool_calls or client abort; SDKs retry twice by default, so duplicate RUs with identical inputs occur.
19. LoRA multi-tenancy (S-LoRA, Punica SGMV) adds per-request GEMMs whose grouping depends on which adapters share the batch.
20. Whole-inference zkML costs ~150 s/token for an 8B model on CPU and <15 min per 13B inference on GPU; installed compute is ~12-16M H100e (~3e22 FLOP/s peak) serving ~1e13-1e14 tokens/day per major provider, which fixes the scale Verity's sampling must cover.
