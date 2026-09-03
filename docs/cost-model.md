# Cost model

Sep 3, 2026. A drop-in for section 6.5 of the paper. Every number is measured in this repository, measured on the VM this section was written on, or cited; the Provenance table at the end maps each one to its source. The body has no file paths.

## 6.5 Cost model

The prover's work beyond the computation itself is of four kinds: **retaining** values between production and challenge, **committing** to values, **replaying** the interior of an opened replay unit, and **proving** the sampled verification units. The verifier **checks openings**, **verifies proofs**, and in the transparent variant **re-executes** the sampled units. Write $\alpha$ for the prover's overhead and $\beta$ for the verifier's cost, both as fractions of the original computation $C$; $\gamma$ for the cost of replaying a unit once relative to computing it originally; $h$ for the cost of committing one position (its leaf hash plus its share of the internal nodes) in the same unit as $C$; $\pi$ for the cost of proving a verification unit relative to computing it; $q$ and $s$ for the sampling rates. With $|B|$ boundary positions and $|I|$ interior positions per replay unit,

$$\alpha \;=\; \underbrace{\frac{h\,|B|}{C}}_{\text{commit boundary}} \;+\; q\Big(\underbrace{\gamma}_{\text{replay}} + \underbrace{\frac{h\,|I|}{C}}_{\text{commit interior}}\Big) \;+\; \underbrace{q\,s\,\pi}_{\text{prove}} .$$

Retention is bytes, not work, and is accounted separately. The rest of this section prices each term.

### What is committed, and how many leaves

**The commitment.** A transcript commitment is a binary Merkle tree whose leaf for position $i$ with value $v_i$ is $H(\mathsf{leaf}, D, i, \mathsf{schema}, v_i)$, where $D$ is a domain identifier binding the session, the phase, the owner (weights, boundary, or one replay unit) and the ordered position set, and $\mathsf{schema}$ is the value width. Internal nodes are $H(\mathsf{node}, D, \ell, j, \text{left}, \text{right})$ with their level and index; positions beyond the last leaf are padding hashes. The domain separation means a root cannot be replayed for another session, phase, owner or position set, and a leaf cannot be moved. A tree of $L$ leaves has $L-1$ internal nodes: committing a position costs one leaf hash and, asymptotically, one node hash.

**How many leaves.** The boundary of every replay unit is always committed; the interior of a unit is committed only when the unit is opened, and then at verification-unit granularity: the committed interior positions of a unit are the declared outputs of the verification units inside it, less the unit's own outputs. Every gate value is *not* a leaf. With $|B|$ the boundary positions of the whole computation and $|I_r|$ the interior positions of replay unit $r$,

$$\mathbb{E}[L] \;=\; |B| \;+\; q\sum_r |I_r| .$$

The boundary is not negligible by fiat; it depends on what the constructor puts on it. With one request per replay unit the boundary is the token stream. With one decode step per replay unit the KV state crosses the boundary at every step and is committed at production time.

| | GPT-2 Small, one request of 32 + 32 tokens | 70B-class decoder, one request of 512 + 512 tokens |
|---|---:|---:|
| gates in the request $C$ (scalar-op equivalents) | $9.19 \times 10^{9}$ | $2.01 \times 10^{14}$ cost units ($6.70 \times 10^{13}$ MACs) |
| flops | $1.8 \times 10^{10}$ | $1.34 \times 10^{14}$ |
| gate values (every non-source gate) | $6.0 \times 10^{8}$ (a tensor-core step is one gate) | $1.34 \times 10^{14}$ (two per MAC) |
| tensor-core outputs (16-term dot products) | $5.66 \times 10^{8}$ | $4.19 \times 10^{12}$ |
| verification units | $1.78 \times 10^{7}$ | $1.69 \times 10^{10}$ |
| interior positions $\lvert I \rvert$ (VU outputs) | $17{,}813{,}407$ | $16{,}936{,}947{,}712$ |
| boundary $\lvert B \rvert$, RU = request (tokens) | 64 | 1,024 |
| boundary, RU = decode step (KV state + tokens) | $1.18 \times 10^{6}$ | $1.34 \times 10^{9}$ |
| Merkle depth of the interior tree | 25 | 34 |

The 70B row is a serving model (a toy ISA with one `mul` and one `add` per MAC, so two gate values per MAC and a `cell` verification unit per dot product); the GPT-2 row is the compiled circuit of the repository's GPT-2 Small constructor, whose gate set has a 16-term tensor-core step as one gate. The interior at VU granularity is $1/34$ of the gate values for GPT-2 Small and $1/7{,}900$ for the 70B model, where every dot product has $8{,}192$ terms.

**Leaf size.** Committed values are 16-bit words (BF16 tensors, tokens) or 32-bit words (fp32 accumulators, statistics). The payload is 2 or 4 bytes; the hash's input is the frame around it. In the minimal design ($\mathsf{leaf}$ tag, index, value) the leaf preimage fits in one block of every algorithm below and the 64-byte node preimage fits in one block of every algorithm except SHA-256, whose 64-byte block leaves no room for padding. The repository's frames are wider: a 124-byte leaf preimage (33-byte frame constant, tag, five length-prefixed parts including the 32-byte domain identifier and the schema string) and a 179-byte node preimage, three SHA-256 blocks each, so a committed position costs six compressions where the minimal SHA-256 design costs three and a BLAKE3 design two. Nothing in the protocol depends on this; it is a constant to fix in a native prover.

Implications:

- The count that matters is $q\sum_r |I_r|$, and $|I_r|$ is the number of verification units, not gates: $5.4 \times 10^{-2}$ of the gates at GPT-2 Small, $1.3 \times 10^{-4}$ at the 70B shape.
- The boundary is $10^{-5}$ to $10^{-7}$ of the interior when the replay unit is a request. When it is a decode step, the boundary is the KV state, $1.3 \times 10^{9}$ positions per 70B request, $8\%$ of the interior, and it is paid at $q = 1$.
- The leaf payload is irrelevant to the hash cost; the frame and the internal nodes are the cost. Packing 32 16-bit values into one 64-byte leaf divides $h$ by 32 at the price of opening 64 bytes instead of 2.

### Hash cost

**Block sizes, corrected.** The algorithms do not all process 32-byte blocks. SHA-256 compresses 64-byte blocks (a message of at most 55 bytes is one block), SHA-512 and BLAKE2b 128-byte blocks (at most 111 and 128 bytes in one block), BLAKE3 64-byte blocks inside 1 KiB chunks (at most 64 bytes in one compression), and SHA3-256 absorbs 136 bytes per Keccak permutation (at most 135 in one). A tiny payload costs one compression whatever its size; the per-compression cost differs by an order of magnitude across algorithms and by another order of magnitude across hardware.

| algorithm | block | 32-bit ops per compression (from the round structure) | cycles/byte, long messages, SHA-NI CPU (eBACS `alder`) | cycles/byte, long, no SHA-NI (eBACS `samba`) | cycles per 64-byte message (eBACS `alder`) | ns per 64-byte message, Cascade Lake (BLAKE3 paper data) |
|---|---:|---:|---:|---:|---:|---:|
| BLAKE3 | 64 B | $\approx 800$ (7 rounds $\times$ 8 G $\times$ 14) | 1.73 | 0.95 | 200 | 94 |
| SHA-256 | 64 B | $\approx 2{,}200$ (64 rounds, 48-word schedule) | 2.05 | 7.69 | 300 | 478 |
| BLAKE2b | 128 B | $\approx 2{,}700$ (12 rounds of 64-bit G) | 3.36 | 3.15 | 451 | 154 |
| SHA-512 | 128 B | $\approx 5{,}800$ (80 rounds, 64-bit) | 5.02 | 5.24 | 719 | 401 |
| SHA3-256 | 136 B | $\approx 7{,}400$ (24 rounds of Keccak-f[1600]) | 5.96 | 7.69 | 835 | 581 |

The operation counts are the author's table re-derived: they are per compression, and the table's "payloads $\le 32$ B" column was these numbers under a wrong block size. The cycle figures are SUPERCOP medians on a 2022 Golden Cove desktop with SHA-NI (`alder`), a 2015 Skylake without it (`samba`), and the BLAKE3 authors' Criterion data on a Cascade Lake-SP server (which has AVX-512 but no SHA-NI, and whose small-message SHA numbers include OpenSSL's per-call setup). SHA-NI is the whole story for SHA-256: 2.0 cycles per byte with it, 7.7 without; the other algorithms have no such instruction. BLAKE3's long-message figure is its 16-lane AVX-512 rate over independent chunks, exactly the shape of a Merkle build over many small leaves; single-lane, at one 64-byte message, it is 200 cycles against SHA-NI's 300.

**Measured here.** The same five functions were timed on this VM (Python 3.12, OpenSSL 3.0.13, `blake3` 1.0.9; a 4-vCPU Intel Xeon, CPUID family 6 model 207, 2.4 GHz nominal, SHA-NI and AVX-512), one hash object per call, median of seven runs. The small-payload columns are dominated by the interpreter (a bare call costs 58 ns); the 64 B $\to$ 1 KiB slope prices one more block without that overhead.

| algorithm | 32 B | 64 B | 1 KiB | 1 MiB (MB/s) | marginal ns per block | cycles per block at 2.4 GHz |
|---|---:|---:|---:|---:|---:|---:|
| SHA-256 | 329 ns | 338 ns | 840 ns | 538 µs (1,949) | 33 | 80 |
| BLAKE3 | 330 ns | 330 ns | 1,017 ns | 128 µs (8,213) | 46 (single lane); 7.8 per 64 B at 16 lanes | 110 |
| BLAKE2b | 328 ns | 328 ns | 1,371 ns | 1,232 µs (851) | 139 | 334 |
| SHA-512 | 477 ns | 480 ns | 1,676 ns | 1,221 µs (859) | 160 | 383 |
| SHA3-256 | 591 ns | 596 ns | 2,252 ns | 1,808 µs (580) | 235 | 563 |

The repository's own frames, timed the same way: 1,333 ns for a leaf and 1,302 ns for a node, 2.6 µs per committed position in pure Python against about 0.2 µs of SHA-256 work inside it ($6 \times 33$ ns). The repository's Merkle benchmark on an Apple M-series laptop builds at $1.1 \times 10^{6}$ hashes per second, $5.2 \times 10^{5}$ positions per second, the same order.

**On the GPU.** CPU numbers are a proxy; the prover's arithmetic runs on GPUs and a Merkle build over billions of independent leaves is the workload GPU hashing kernels are good at. hashcat's benchmark mode hashes short independent messages, one compression each: on an A100-SXM4-40GB, SHA-256 at $9.56 \times 10^{9}$ hashes/s (A100 PCIe: SHA-256 $9.41 \times 10^{9}$, BLAKE2b-512 $5.53 \times 10^{9}$, SHA-512 $3.08 \times 10^{9}$, SHA3-256 $2.14 \times 10^{9}$); on an H100 80GB HBM3, SHA-256 at $1.62 \times 10^{10}$ and SHA-512 at $5.41 \times 10^{9}$. No comparable BLAKE3 figure is published for GPUs (the public GPU ports are single-stream proofs of concept); by operation count it should sit 2 to 3 times above SHA-256. Against the same GPUs' dense tensor-core rates (A100 312 TFLOPS FP16, H100 989 TFLOPS), one SHA-256 compression is worth $3.3 \times 10^{4}$ flops on the A100 and $6.1 \times 10^{4}$ on the H100.

**Per committed position.** One leaf hash plus one node hash: two compressions for BLAKE3, BLAKE2b, SHA-512 and SHA3-256, three for SHA-256 (the 64-byte node needs two blocks), six for the repository's frames.

| cost of one committed position | BLAKE3 | SHA-256 | repository frames (SHA-256) |
|---|---:|---:|---:|
| compressions | 2 | 3 | 6 |
| 32-bit operations | $\approx 1{,}600$ | $\approx 6{,}600$ | $\approx 13{,}000$ |
| CPU, this VM (marginal) | 92 ns single lane; 16 ns at 16 lanes | 100 ns | 200 ns (2.6 µs in Python) |
| CPU, SHA-NI desktop (eBACS, incl. call overhead) | 400 cycles | 900 cycles | |
| flop-equivalents on this CPU core (fp32 peak $1.5 \times 10^{11}$ flop/s) | $1.4 \times 10^{4}$ | $1.5 \times 10^{4}$ | $3.1 \times 10^{4}$ |
| flop-equivalents on an A100 (hashcat rate vs 312 TFLOPS) | not published | $9.8 \times 10^{4}$ | $2.0 \times 10^{5}$ |
| flop-equivalents on an H100 (vs 989 TFLOPS) | not published | $1.8 \times 10^{5}$ | $3.7 \times 10^{5}$ |

Implications:

- The two drafts disagree on hash costs because they measure different things. Operation counts (800 per BLAKE3 compression, 2,200 per SHA-256) are right as counts and are the right metric for a GPU's integer units. Wall-clock on a CPU inverts the order: SHA-NI makes a SHA-256 block (33 ns, 80 cycles) cheaper than a single-lane BLAKE3 block (46 ns), and BLAKE3 wins only by hashing 16 leaves at once. The per-position figures the second draft gives (1,600 BLAKE3, 6,600 SHA-256, 6,000 BLAKE2b, 16,000 SHA3-256) are the compression counts times the operation counts and are correct.
- Committing one 16-bit value costs $10^{4}$ to $10^{5}$ flop-equivalents of the serving hardware's time. Relative to a 16-MAC tensor-core output (32 flops) that is $3 \times 10^{3}$ to $6 \times 10^{3}$; relative to a scalar operation, $10^{4}$ to $10^{5}$. No hash function changes this by more than a small factor; only the leaf count does.
- The repository's $h$ assumes $10^{5}$ MACs per SHA-256 compression, one compression per 32-value leaf and no internal-node term: $3{,}125$ MACs per committed value. The measured GPU ratio is $1.6 \times 10^{4}$ (A100) to $3.1 \times 10^{4}$ (H100) MACs per compression, 3 to 6 times less per compression; charging the leaf and its node (three SHA-256 compressions) at the H100 ratio gives $2{,}870$ MACs per committed value, so the constant is about right for a packed SHA-256 commitment and 2 times conservative on an A100. The 32-values-per-leaf packing is a layout the protocol code does not yet implement; without it, and with the six-compression frames, $h$ is $1.8 \times 10^{5}$ MACs per value at the H100 ratio and $6 \times 10^{5}$ at the repository's constant, 59 and 192 times the assumed value. The effect on the optimum is in the fourth digit (below).

### Replay cost $\gamma$

**Same hardware, same kernels: $\gamma \approx 1$.** Replay re-runs the replay unit from its committed inputs. If the replay machine runs the production kernels, the replay costs what the production run cost, plus the interior commitment of the opened unit. Bit-exact replay needs the same kernel semantics: the same reduction order, the same rounding points, the same intermediate precision. The repository established that these can be pinned. The BF16 `mma.sync m16n8k16` instruction of an RTX 4090 was reproduced bit-exactly in software (every output element is a 16-term dot product accumulated into fp32 with a fixed internal alignment and rounding, recovered from the instruction's behaviour on adversarial inputs), and a full GPT-2 Small forward pass run through fixed-order GPU kernels built from that instruction was reproduced word for word on a CPU (363 recorded tensors, 11.7 M words, every word equal).

**What pinning costs on the GPU.** The fixed-order kernels are slower than cuBLAS on the same GPU because they serialise the reduction over $K$ (one accumulator per output, no split-$K$): 1.8 to 2.8 times at $K = 768$, 6.7 to 9.1 times at $K = 3{,}072$, 28 to 47 times for the one-hot embedding at $K = 50{,}272$. This is a cost of *pinning*, paid on the production run as well as the replay if the production kernels are the pinned ones, and it is a property of these first kernels (a `gather` gate removes the embedding case; split-$K$ with a declared tree removes the rest), not of the protocol.

**Replay on other hardware.** If the replay machine has no tensor cores, or the semantics cannot be matched on them, the same arithmetic runs on CUDA cores or a CPU. Peak-throughput ratios from the datasheets: A100 FP16 tensor core 312 TFLOPS against 78 TFLOPS non-tensor FP16 and 19.5 TFLOPS FP32, so $\gamma = 4$ for FP16 and $\gamma = 16$ for FP32 CUDA-core replay; H100 989 TFLOPS dense FP16 tensor core against 67 TFLOPS FP32, $\gamma = 15$. CPU replay of the GPT-2 forward pass measured $9.75$ s on a laptop against about 35 ms of GEMM chains on the GPU, $\gamma \approx 300$. Pure-Python replay of the pinned gates costs 160 µs per tensor-core step and is not a replay strategy; a native emulator of the pinned `mma.sync` semantics has not been timed (not measured).

**Memory-bound replay.** Decode is bound by weight bandwidth, not arithmetic: a 70B model in 16-bit reads 140 GB of weights per decode step, 42 ms at the H100's 3.35 TB/s, and production amortises that over a batch of 32 requests. A replay of one sampled request alone pays the full read for one request, so in device time $\gamma$ can approach the production batch size unless the replay machine batches sampled units together. At the headline $q$ the fleet samples $4.6 \times 10^{5}$ requests a year, one every 68 seconds, so batching them is a scheduling choice, not a constraint.

**The replay share.** The replay term of $\alpha$ is $q\,(\gamma + h|I|/C)$. At the 70B shape $h|I|/C$, the cost of committing an opened request's interior relative to replaying it, is $0.79$ with the repository's $h$ (32 values per leaf, one compression each), $0.73$ with the measured H100 ratio and a leaf plus node of three SHA-256 compressions per 32 values, $46$ with the repository's six-compression frames and one value per leaf at the measured ratio, and $152$ for those frames at the repository's conservative constant. At the headline $q = 1.57 \times 10^{-8}$ the whole term is between $2.7 \times 10^{-8}$ and $2.4 \times 10^{-6}$: never a visible share of a 1% budget.

Implications:

- $\gamma$ is a hardware and kernel decision, between 1 (production kernels, same silicon) and 16 (FP32 CUDA cores); it multiplies $q$, which the optimum drives to $10^{-8}$.
- Pinned semantics are established for one instruction on one architecture. Every new tensor-core instruction, precision or GPU generation needs the same recovery; that is the maintenance cost of $\gamma \approx 1$.
- The interior commitment of an opened unit costs from three quarters of a replay to 150 replays, depending on leaf packing and hash implementation; it rides on $q$ and is invisible at the optimum.

### Proving cost $\pi$

**What was measured.** The proof layer is a generic checker for the protocol's statements running in a RISC-V zkVM (SP1 6.4.0): it authenticates each opened position against its Merkle root inside the proof, recomputes the verification unit's gates from the opened inputs, and compares with the opened outputs. Exact execution cycle counts: $3{,}296$ cycles per opened position plus $2{,}668$ per Merkle level (three SHA-256 precompile calls per level, because of the three-block node frame), 125 to 238 cycles per recomputed toy-ISA gate, $5{,}800$ per obligation and $50{,}000$ per batch. Merkle authentication is 60% to 81% of the cycles of real batches, gate recomputation 1% to 3%. Proving throughput on CPUs is 105 to 118 kHz (cycles proved per second) with a fixed floor of 10 to 40 s per proof; the one end-to-end proof in the repository covered 16 obligations, $761{,}710$ cycles, 2 shards, a 2.80 MB core proof, 16.9 s to prove and 0.07 s to verify on a laptop.

**The tensor-core precompile.** For the dot-product relations of the ML workloads the relevant backend is a zkVM extension that checks whole tensor-core tiles as one instruction. Measured on an RTX 4090: OpenVM with a `TC_MATMUL_4X4X16` extension proves $2.52 \times 10^{6}$ MACs per second at its batching plateau (64 tiles per proof, 1.4 to 2 s proofs), a `TC_DOT` extension $1.20 \times 10^{6}$, a TC-DOT-patched SP1 fork $0.98 \times 10^{6}$. Against a native rate of $1.8 \times 10^{14}$ fp8 MACs per second on the same GPU (55% of its dense peak, the achieved rate the report's figures imply), $\pi = 7.1 \times 10^{7}$ for the matmul precompile, $1.5 \times 10^{8}$ and $1.8 \times 10^{8}$ for the dot variants. This $\pi$ is what the headline estimate uses.

**Without the precompile.** In plain RISC-V a fixed-point MAC costs 16 zkVM cycles and an f32 MAC 130 to 140 (soft float). At the fastest published general-purpose GPU proving rate, $1.2 \times 10^{6}$ cycles/s on an RTX 4090 (RISC Zero; the repository's CPU figure is $10^{5}$), a MAC proves in $1.3 \times 10^{-5}$ s fixed-point or $1.1 \times 10^{-4}$ s in f32, against $5.6 \times 10^{-15}$ s natively on tensor cores: $\pi = 2.4 \times 10^{9}$ and $2 \times 10^{10}$. That is where the "$10^{10}\times$ native" figure comes from, and what it is relative to: a zkVM interpreting scalar arithmetic against tensor-core throughput. Relative to a CPU running the same scalar program the published overhead is $5 \times 10^{5}$ (Jolt) to "millions" (deployed zkVMs). The precompile recovers three orders of magnitude of the four that separate scalar from tensor-core execution. The state of the art at the system level is real-time proving of Ethereum blocks: 93% of blocks in under 12 s on about 160 RTX 4090s in May 2025, 99.7% on 16 RTX 5090s in November 2025; those workloads are precompile-heavy (hashing, signatures), as ours would be.

**Proof size and verification.** Core proofs grow with the batch (2.80 MB for the 16-obligation batch, two shards); compressed proofs are a constant 1.27 MB; the verifier's time was 0.07 s for the core proof (measured) and is not measured for OpenVM. A proof carries a fixed cost (the 10 to 40 s floor and $50{,}000$ cycles of setup), so verification units are batched $k$ to a proof and the per-unit cost $c_0$ is the floor divided by $k$ plus the marginal cycles; it plateaus at $k = 64$ to 256 for the tensor-core extension (1.4 to 2 s proofs) and at $k = 2{,}048$ to $16{,}384$ (35 to 69 s) for the SP1 fork.

**The transparent alternative.** Without proofs the verifier re-executes each sampled verification unit against the opened values. Nothing is paid on the prover's side beyond the openings; the verifier pays $q\,s$ of the computation plus the openings, and must own hardware whose semantics match the pinned ones. At the headline point $q\,s = 1.4 \times 10^{-10}$ of the year's arithmetic: $5.5 \times 10^{17}$ flops, four minutes of one H100. The transparent design is not the bottleneck; its cost is the openings (next subsection), and its constraint is that the verifier sees the opened values in the clear.

Implications:

- $\pi$ spans $7 \times 10^{7}$ (tensor-core precompile, GPU prover) to $2 \times 10^{10}$ (soft-float scalar zkVM). The headline uses the smaller; the sensitivity table shows $U$ scaling linearly with $\pi$ ($4.98$ TB at $1.5 \times 10^{8}$, $33$ TB at $10^{9}$).
- Per-scalar Merkle authentication, not arithmetic, is the measured cost of a proof: $3{,}296 + 2{,}668\,d$ cycles per opened position against 125 to 238 per gate. Leaves must be tiles, not scalars, for the ML relations; the toy ISA's per-scalar inputs are the wrong shape and the tile kind is the fix.
- Batching is the lever on the fixed cost, and memory caps the batch: 84 GB of RAM at 42 M cycles on the CPU prover.

### Verifier cost $\beta$

**Openings.** An opening of a position in a tree of depth $d$ is $d$ sibling hashes ($32d$ bytes) and the value; checking it is $d + 1$ hashes: 26 for a GPT-2 Small request interior ($d = 25$, 800-byte path), 35 for a 70B request ($d = 34$, 1,088 bytes), 27 to 28 for the GPT-2 weight tree. With the repository's frames that is $3(d+1)$ SHA-256 blocks, $3.5$ µs at 33 ns per block on this VM ($105$ blocks at $d = 34$); the repository's Python verifier measured 20 µs at $d = 20$.

**Proof verification.** In the zero-knowledge design the Merkle paths are checked inside the proof and the verifier checks only the proof, one per batch of $k$ verification units: $\beta_{\mathrm{zk}} = (q\,s/k)\,v_\pi$ per verification unit, with $v_\pi$ the verification time (0.07 s measured for an SP1 core proof; OpenVM not measured). At the headline the fleet proves $6.9 \times 10^{13}$ verification units a year; at $k = 64$ that is $1.1 \times 10^{12}$ proofs, $7.6 \times 10^{10}$ core-seconds at 0.07 s each, $2{,}400$ core-years against $3.5 \times 10^{5}$ GPU-years of serving: $0.7\%$ in device time, $3 \times 10^{-6}$ in flops (a core at $1.5 \times 10^{11}$ flop/s against $3.9 \times 10^{27}$ flops a year). At $k = 256$, a quarter of that.

**Transparent re-execution.** $\beta_{\mathrm{tr}} = q\,s\,(1 + p_V\,(d+1)\,c_H / c_V)$, where $p_V$ positions are opened per verification unit, $c_H$ is one hash and $c_V$ the unit's arithmetic. With scalar leaves a row dot product of the 70B model opens $2 \times 8{,}192 + 1$ positions, $1.7 \times 10^{6}$ SHA-256 blocks with the repository's frames, 57 ms on this CPU against $0.1$ µs for its 8,192 MACs: the openings are $5 \times 10^{5}$ times the arithmetic and $\beta_{\mathrm{tr}} \approx 7.6 \times 10^{-5}$ in CPU time at the headline $q\,s$. With tile leaves (one leaf per tensor-core tile, as the tile kinds of the proof layer commit) $p_V$ falls to a few and the factor to $\approx 10^{2}$, $\beta_{\mathrm{tr}} \approx 10^{-8}$. Measured on GPT-2 Small in Python at $q = 1$, $s = 1/500$ on a one-token prompt: the verifier opened 249,669 verification units through 896,685 openings in 44.9 s (50 µs per opening) against a 0.62 s CPU forward pass; the same openings are $2.4$ s of SHA-256 blocks natively.

Implications:

- The verifier's cost is openings, and openings are $d + 1$ hashes per opened *position*. Scalar leaves make a dot product cost $10^{5}$ times its arithmetic to check; tile leaves make it cost $10^{2}$.
- With proofs, $\beta$ is one proof verification per $k$ units, $10^{-6}$ of the computation in flops and under $1\%$ in device time at the headline; the number of proofs, not their size, is what the verifier pays for.
- Transparent verification is cheaper for the prover (no $\pi$) and for the verifier in flops; it trades the zero-knowledge property for the openings' bandwidth ($1.7$ KB per opened scalar at $d = 34$).

### Retention

**What the prover keeps.** The prover must be able to answer a challenge after production. Three recording policies span the range: tokens only (the boundary under request replay units), the KV boundary (the boundary under decode-step replay units), and every verification-unit output (what a fault-diagnosing prover keeps to name the faulty kernel rather than re-serve the request). The interior of an opened unit is not retained under the protocol's default: it is recomputed from the boundary, which is why the replay term exists.

| retention per request | GPT-2 Small, 32 + 32 | toy at GPT-2 dims, 100 + 100 | 70B, 512 + 512 |
|---|---:|---:|---:|
| tokens only | 128 B | 400 B | 2 KB |
| KV boundary | 2.36 MB | 7.4 MB | 2.68 GB |
| every VU output | 35.6 MB | 175 MB | 33.9 GB |
| flops of the request | $1.8 \times 10^{10}$ | $5.8 \times 10^{10}$ | $1.34 \times 10^{14}$ |
| VU outputs, bytes per flop | $2.0 \times 10^{-3}$ | $3.0 \times 10^{-3}$ | $2.5 \times 10^{-4}$ |

Bytes per flop fall with model width (a dot product of $d$ terms yields one value), so the largest models retain the least per unit of work. At the fleet scale of the headline ($2.9 \times 10^{13}$ requests a year) tokens-only retention is $60$ PB a year, the prompts and outputs themselves; KV retention would be $8 \times 10^{22}$ bytes and every-VU-output retention $10^{24}$, neither storable. The weights are committed once per model ($1.2 \times 10^{8}$ words for GPT-2 Small, $6.5 \times 10^{10}$ for the 70B shape; the GPT-2 weight tree took 91.5 s in the repository's parallel builder).

Implications:

- Retention is the term that forces replay units to be requests and interiors to be recomputed: anything finer is either unstorable at scale or paid at $q = 1$.
- The retention window is the round: values are held until the round closes and the challenge is answered, not for a year.

### The operating points

**GPT-2 Small.** The compiled circuit of three requests of 32 + 32 tokens: $1{,}924{,}349{,}881$ gates ($124{,}490{,}068$ weights, 96 inputs, $1.80 \times 10^{9}$ computed), 291 kinds, 4 replay units (the weights and three requests), $177{,}855{,}025$ verification units of which $53{,}364{,}861$ are computed, a description of $1{,}090{,}184$ bytes, compiled in 0.53 s on this VM (0.32 s on the repository's laptop). Per request: $9.19 \times 10^{9}$ cost units, $9.06 \times 10^{9}$ of them tensor-core MACs in $5.66 \times 10^{8}$ steps; $17{,}813{,}407$ interior positions; a boundary of 64 tokens. Committing the whole interior at $q = 1$ costs, per request, $1.78 \times 10^{7} \times h$: with BLAKE3's $1{,}600$ operations per position, $2.9 \times 10^{10}$ operations against $1.8 \times 10^{10}$ flops of arithmetic (157%); in A100 time with SHA-256 one value per leaf, 96 times the arithmetic; with 32 values per leaf, 3 times. Replaying it costs $\gamma$ times the arithmetic. Proving a 768-term dot product with the measured SP1 constants and scalar leaves costs $1{,}537 \times (3{,}296 + 2{,}668 \times 25)$ cycles, $1.1 \times 10^{8}$ cycles, 15 minutes at 118 kHz: per-scalar leaves are unusable for ML relations, and the tile kind is required before a GPT-2 verification unit can be proved at $\pi \sim 10^{8}$. The bound at this size: with three requests $U$ cannot fall below the whole output unless $q = 1$ (an unsampled request survives with probability $1 - q$); at $q = 1$, $s = 3/4$, $U$ is 1,225 of 1,536 output bits and the repository's cost fold gives overhead 1.82 with $h = 1$ and $\pi = 1$ gate-equivalents (1.000 replay, 0.065 interior commitment, 0.753 proof). GPT-2 Small is the structural point, not the budget point: at a real $\pi$ the 1% budget admits $q\,s \le 1.4 \times 10^{-10}$, less than one proved verification unit per 400 requests, and the bound bites only at fleet scale.

**The 70B serving shape at the headline.** A dense decoder with $d = 8{,}192$, 80 layers, 64 heads, vocabulary 32,768, 16-bit values, serving requests of 512 + 512 tokens in batches of 32; $3 \times 10^{16}$ tokens a year, $2.93 \times 10^{13}$ requests, $4.96 \times 10^{23}$ verification units. The estimate fixes a budget $\alpha = 1\%$ and chooses $(q, s)$ to minimise the capacity $U$ at $\lambda = 40$. Its cost parameters: $\pi = 7.14 \times 10^{7}$ (the matmul precompile); $h$ from $10^{5}$ MACs per SHA-256 compression and 32 values per leaf, $9{,}375$ cost units per position; the interior at VU granularity ($1.69 \times 10^{10}$ positions per request); the boundary the token stream. The optimum is $q = 1.57 \times 10^{-8}$, $s = 8.91 \times 10^{-3}$, $U = 1.90 \times 10^{13}$ bits $= 2.37$ TB a year, and the budget splits as follows.

| term of $\alpha$, 70B headline | repository constants (one compression per 32-value leaf) | measured H100 rate, leaf + node, 32 values per leaf | one value per leaf, repository frames, repository constant |
|---|---:|---:|---:|
| $h$, MACs per committed position | 3,125 | 2,869 | 600,000 |
| commit boundary $h\lvert B\rvert / C$ | $7.2 \times 10^{-8}$ | $6.6 \times 10^{-8}$ | $1.4 \times 10^{-5}$ |
| replay $q\gamma$ | $1.6 \times 10^{-8}$ | $1.6 \times 10^{-8}$ | $1.6 \times 10^{-8}$ |
| commit interior $q h \lvert I \rvert / C$ | $1.2 \times 10^{-8}$ | $1.1 \times 10^{-8}$ | $2.4 \times 10^{-6}$ |
| prove $q s \pi$ | $0.0099999$ | $0.0099999$ | $0.0099839$ |
| $\alpha$ | $0.0100$ | $0.0100$ | $0.0100$ |
| $q$ | $1.5707 \times 10^{-8}$ | $1.5707 \times 10^{-8}$ | $1.5682 \times 10^{-8}$ |
| $U$ | 2.3696 TB | 2.3696 TB | 2.3734 TB |
| retention (tokens only), per request | 2 KB | 2 KB | 2 KB |
| $\beta$, proofs at $k = 64$ | $3 \times 10^{-6}$ flops, $0.7\%$ device time | same | same |

Re-running the estimate with the measured hash cost leaves $q$ and $U$ unchanged to five digits. Even the 192-fold worse $h$ of one-value-per-leaf six-compression frames at the repository's constant moves $U$ by 0.16% (0.05% at the measured H100 ratio, where $h = 1.8 \times 10^{5}$). The reason is the granularity: at gate granularity (every gate value a leaf, as the prototype committed before the interior moved to VU outputs) the same worst-case $h$ puts 63% of the budget into interior commitment and raises $U$ to 6.47 TB (34% and 3.65 TB at the measured ratio), and even the repository's packed $h$ takes 1% of the budget and raises $U$ to 2.39 TB.

**The granularity table, corrected.** The author's table charged every gate value as a leaf. The rows below are per GPT-2 Small request at the drafts' $q = 10^{-4}$, with $1{,}600$ operations per committed position for BLAKE3 (operation count) and $9.8 \times 10^{4}$ flop-equivalents per position for SHA-256 on an A100 (device time, one value per leaf); the arithmetic is $1.8 \times 10^{10}$ flops.

| granularity of the interior | positions in a complete transcript | expected interior leaves at $q = 10^{-4}$ | hash work vs arithmetic, BLAKE3 operations | hash work vs arithmetic, SHA-256 on the A100 |
|---|---:|---:|---:|---:|
| scalar operation (a multiply and an add each yield a value) | $1.8 \times 10^{10}$ | $1.8 \times 10^{6}$ | 16% | $9.8\times$ |
| tensor-core output (16-term accumulated dot product, 32 flops) | $5.7 \times 10^{8}$ | $5.7 \times 10^{4}$ | 0.50% | 30% |
| verification-unit output (the protocol) | $1.8 \times 10^{7}$ | $1.8 \times 10^{3}$ | 0.016% | 0.96% |

At the 70B shape the ratios at $q = 1$ are 1,600, 50 and 0.20 (BLAKE3 operations) and $9.8 \times 10^{4}$, $3{,}100$ and 12 (SHA-256 on the A100) for the three rows; at the headline $q$ they are $3 \times 10^{-9}$ and $2 \times 10^{-7}$ for the protocol's row.

Implications:

- The second draft's 16% for scalar gates at $q = 10^{-4}$ is right as an operation count; in device time on the GPU the same commitment costs ten times the arithmetic. The first draft's Tensor Core granularity is wrong: an `mma.sync m16n8k16` exposes 128 outputs, each a 16-term dot product whose partial products and sums are not observable, so the finest replayable gate is one 16-MAC output (32 flops), not a $16^{3}$ tile; the whole instruction is $16 \times 8 \times 16 = 2{,}048$ MACs. The repository's gate set has exactly that gate.
- Verification-unit granularity is 32 times coarser than tensor-core outputs and $10^{3}$ times coarser than scalars, and it is a free choice: the verifier re-executes or proves the unit from its inputs, so its internal values need not be committed.
- The budget point is proofs. Everything hashed and replayed at the optimum is $10^{-7}$ of the computation with the repository's constants and below $2 \times 10^{-5}$ in the worst variant of $h$ examined.

### Drivers

- **Commitment:** the leaf count $|B| + q\sum_r |I_r|$, set by the boundary size (tokens, or the KV state if replay units are decode steps), by $q$, and by the verification-unit granularity of the interior; the hash algorithm (a factor of 3 in operations between BLAKE3 and SHA-256, inverted on SHA-NI CPUs); the leaf payload and frame (one value or 32 per leaf, two or six compressions per position); the internal nodes (one per leaf).
- **Replay:** $q$; $\gamma$ (1 on the production kernels, 4 to 16 on CUDA cores, 300 on a CPU); the pinned semantics that make $\gamma \approx 1$ available at all; memory-boundness (a lone replayed decode pays the batch factor in device time unless sampled units are batched).
- **Proving:** $s$; the verification-unit size (leaves must be tiles; per-scalar Merkle authentication is 60% to 81% of a proof's cycles); the proof system's fixed cost per statement and the batch size that amortises it (plateau at $k = 64$ to 256 for the tensor-core extension); $\pi$ itself, $7 \times 10^{7}$ with the precompile, $10^{10}$ without.
- **Verifier:** openings ($d + 1$ hashes per opened position, $10^{5}$ times a dot product's arithmetic with scalar leaves); proof verification (one per batch, 0.07 s measured); transparent re-execution ($q\,s$ of the computation, on matched hardware).
- **Retention:** the recording policy (tokens only, KV boundary, every VU output: $2$ KB, $2.68$ GB, $33.9$ GB per 70B request) and the round length.

**Is this tolerable?** At the headline the optimum puts $99.999\%$ of the 1% budget into proofs; replay, interior commitment and the boundary together are $10^{-7}$ of the computation with the repository's constants and under $2 \times 10^{-5}$ with the worst hash implementation examined, because a replay unit is opened with probability $1.6 \times 10^{-8}$ and its interior has $1.3 \times 10^{-4}$ as many positions as gates. What the 1% buys is $U = 2.37$ TB a year against $60$ PB of output ($4 \times 10^{-5}$ of the output bits), with $U \propto \pi / \text{budget}$: $23.7$ TB at $0.1\%$, $0.24$ TB at $10\%$. The naive design that materialises and hashes every gate value would cost $1{,}600$ times the arithmetic in operations and $10^{5}$ times in GPU time; the design that commits only replay-unit boundaries at runtime, recomputes an opened unit's interior on a separate machine and commits it at verification-unit granularity reduces that by $q\,|I|/|G|$, twelve orders of magnitude at the headline. The costs that remain are the proving factor and the openings, both of which are set by the shape of a verification unit: a tensor-core tile, not a scalar.

## Provenance

Numbers in the order they appear. "Repo" is this repository at the commit this document was added; "VM" is the measurement made for this document on the cloud VM described in the hash-cost table; URLs are external.

| number | source |
|---|---|
| Leaf and node preimages, domain identifier, padding, `leaf`/`node` framing; 124-byte leaf and 179-byte node preimages, 3 SHA-256 blocks each | repo `src/veritor/protocol/merkle.py` (`_FRAME`, `_hash`, `CommitmentDomain.leaf/node`), `src/veritor/protocol/domains.py` (`leaf_schema` = `u16`/`u32`); preimage bytes measured by `benchmarks/hash_cost.py` (`veritor_frames` in `docs/data/hash-cost.json`) |
| Interior at VU-output granularity: declared outputs of the VUs inside an RU less the RU's outputs | repo `src/veritor/core/index.py` (`KindSummary.interior_count`, `Index.interior`), `docs/stress-tests.md`, `docs/global-estimate.md` |
| GPT-2 Small, 3 x (32 + 32): 1,924,349,881 gates; 124,490,068 weights; 96 inputs; 291 kinds; 4 RUs; 177,855,025 VUs; description 1,090,184 bytes; request `size` 599,953,271; laptop timings 113/144/60 ms | repo `docs/gpt2-structure.md` (compilation table), `tests/veritor/constructors/test_gpt2.py::test_gpt2_small_compiles_and_the_table_has_the_documented_numbers` |
| GPT-2 Small per request: `replay_cost` 9,186,259,303; `interior_count` 17,813,407; `out_count` 32; `source_inputs` 32; tensor-core steps 1,698,739,200 / 3 requests (MACs = 16 x steps); computed VUs 53,364,861; non-dot computed gates 78,476,805; interior depth 25; compile 0.53 s | VM: `GPT2G(GPT2Shape.small())` compiled with the test's `SMALL_REQUESTS`, kind table read as in the test; `MACs` and `steps` from the `dot(k, ...)` rows (`k/16` steps per copy) |
| GPT-2 Small KV boundary 2.36 MB = 12 layers x 2 x 768 x 64 positions x 2 B; tokens 128 B | arithmetic on `GPT2Shape.small()` (`d_model` 768, 12 layers) |
| GPT-2 Small cost fold at (q, s) = (1, 3/4): overhead 1.819 = 1.000 + 0.065 + 0.753; U = 1,224.6 of 1,536 bits; three requests cannot bite below the output unless q = 1 | repo `docs/gpt2-structure.md` ("Bound and cost at GPT-2 Small"); recomputed on the VM with `veritor.analysis.cost.cost(table, VerificationPolicy(1, 3/4), CostParameters())`: recompute 27,558,777,909, commit_interior 53,440,221, proof 83,049,804,219/4 |
| 70B shape (d 8192, 80 layers, 64 heads, vocab 32,768, 512 + 512, batch 32, width 16); 2.93e13 requests/yr; 4.962e23 VUs; request `replay_cost` 201,018,047,303,168 units (3 per MAC); `interior_count` 16,936,947,712; `out_count` 512; `source_inputs` 512; gate positions 1.34e14; interior depth 34; `state_size(1024)` = 1,342,177,280 | repo `docs/data/global-estimate.json`, `src/veritor/evaluation/global_estimate.py` (`Inputs`, `estimate`, `UNITS_PER_MAC`), `src/veritor/evaluation/serving.py` (`ServingShape.state_size`, `serving_table`); recomputed on the VM |
| Decode-step RU boundary at the 70B shape: `out_count` 41,943,072 per step for 32 requests = 32 x (80 x 2 x 8192 + 1) | VM: `serving_table(shape, "step", "cell")` step rows |
| 33.9 GB VU outputs, 2.68 GB KV, 2 KB tokens per 70B request; 2.53e-4 B/flop; 175 MB / 7.4 MB / 3.03e-3 B/flop for the toy at GPT-2 dims, 100 + 100 | repo branch `late-advice`, `tests/veritor/analysis/test_late_advice.py::test_recording_costs_at_the_two_shapes` (commit 06f01c1); 70B values recomputed on the VM (2 x `interior_count`, 2 x `state_size`) |
| Recording policies (tokens only / `BOUNDARY`, KV boundary under step RUs, `VU_OUTPUTS`); interior recomputed, not retained; weights committed once | repo `docs/honest-prover.md` sections 3 and 9, `src/veritor/analysis/cost.py` (recompute-honest model) |
| Operation counts per compression (BLAKE3 ~800, SHA-256 ~2,200, BLAKE2b ~2,700, SHA-512 ~5,800, SHA3-256 ~7,400) | derived from the round structures: BLAKE3 7 rounds x 8 G x 14 ops (FIPS-style count of adds, xors, rotates); SHA-256 64 rounds x ~25 + 48 schedule words x 13; BLAKE2b 12 x 8 x 14 64-bit ops at 2 32-bit ops each; SHA-512 80 x ~26 + 64 x 13, 64-bit; Keccak-f[1600] 24 rounds x ~155 64-bit ops. Specifications: https://github.com/BLAKE3-team/BLAKE3-specs/blob/master/blake3.pdf, https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.180-4.pdf, https://www.rfc-editor.org/rfc/rfc7693, https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.202.pdf |
| Block sizes: SHA-256 64 B, SHA-512/BLAKE2b 128 B, BLAKE3 64-byte blocks in 1 KiB chunks, SHA3-256 rate 136 B | the same specifications |
| eBACS/SUPERCOP cycles per byte, machine `alder` (amd64; Golden Cove; 2022 Intel Core i3-12100; 3300 MHz; supercop-20260831): long messages blake3 1.73, sha256 2.05, blake2b 3.36, sha512 5.02, sha3-256 5.96; 64-byte messages blake3 3.12, sha256 4.69, blake2b 7.04, sha512 11.24, sha3-256 13.04 cycles/byte | https://bench.cr.yp.to/results-hash/amd64-alder.html (medians) |
| eBACS machine `samba` (amd64; Skylake; 2015 Intel Xeon E3-1220 v5; 3000 MHz; no SHA-NI): long messages blake3 0.95, blake2b 3.15, sha512 5.24, sha256 7.69, sha3-256 7.69 | https://bench.cr.yp.to/results-hash/amd64-samba.html |
| eBACS machine `hertz` (Zen 4, SHA-NI): sha256 2.02, blake3 0.60 cycles/byte long | https://bench.cr.yp.to/results-hash/amd64-hertz.html |
| Intel SHA extensions (the `sha256rnds2` etc. instructions behind the SHA-NI rows) | https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sha-extensions.html |
| BLAKE3 paper benchmark data, Cascade Lake-SP 8275CL, AVX-512, single thread, Criterion slope estimates in ns: 64 B blake3 93.8, blake2b 154, sha256 478, sha512 401, sha3-256 581; 1 MiB blake3 205,167, blake2b 978,740, sha256 2,688,487, sha512 1,803,976, sha3-256 3,299,640; chart at 16 KiB: BLAKE3 6866, BLAKE2b 1312, SHA-512 720, SHA-256 484, SHA3-256 394 MiB/s | https://github.com/BLAKE3-team/BLAKE3-specs/tree/master/benchmarks/results/avx512_metal_native/bench_group (`<hash>/<size>/new/estimates.json`, `Slope.point_estimate`); https://github.com/BLAKE3-team/BLAKE3-specs/blob/master/benchmarks/bar_chart.py; paper section 5, https://github.com/BLAKE3-team/BLAKE3-specs/blob/master/blake3.pdf |
| Measured hash costs on this VM (Intel Xeon, CPUID family 6 model 207, 4 vCPU, 2.4 GHz nominal, sha_ni/avx512f; Python 3.12.3, OpenSSL 3.0.13, blake3 1.0.9): all ns, MB/s, marginal ns per block, cycles at nominal clock, 58 ns call floor, leaf 1,333 ns, node 1,302 ns | VM: `uv run --with blake3 python -m benchmarks.hash_cost` -> `docs/data/hash-cost.json` (`hashes`, `marginal_block`, `veritor_frames`, `machine`) |
| Repository Merkle build 1.1e6 hashes/s, 5.2e5 positions/s; `verify_opening` 20 µs at depth 20; opening 2 + 32 d bytes (Apple M-series) | repo `docs/data/benchmarks.json` (`merkle`, `build_vs_leaves.hashes_per_s`, `values_per_s`; `open_verify_vs_leaves.verify_s`), `docs/benchmarks.md` |
| hashcat GPU rates: A100-SXM4-40GB SHA-256 9,558 MH/s (v6.1.1); A100-PCIE-40GB SHA-256 9,409, BLAKE2b-512 5,528, SHA-512 3,081, SHA3-256 2,138 MH/s (v6.1.1); H100 80GB HBM3 SHA-256 16,166, SHA-512 5,409 MH/s (v6.2.6) | https://gist.github.com/rarecoil/2baf9f335faa7ad044a46281773ec5b3 ; https://gist.github.com/Chick3nman/d65bcd5c137626c0fcb05078bba9ca89 ; https://gist.github.com/laluka/2ee80e40ad485adc26934109f4ab75ef |
| No published GPU BLAKE3 throughput for batched independent messages; public GPU ports are single-stream | https://github.com/BLAKE3-team/BLAKE3/pull/80 (Vulkan port, slower than one CPU core) |
| A100: FP32 19.5 TFLOPS, FP16/BF16 Tensor Core 312 TFLOPS dense (624 sparse), memory 1,935 / 2,039 GB/s; non-tensor FP16 78 TFLOPS, BF16 39 TFLOPS | https://www.nvidia.com/en-us/data-center/a100/ (specifications); https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf (table "Peak FP16 (non-Tensor) 78 TFLOPS") |
| H100 SXM: FP32 67 TFLOPS, FP16/BF16 Tensor Core 1,979 TFLOPS with sparsity (989 dense), 3.35 TB/s | https://www.nvidia.com/en-us/data-center/h100/ (product specifications) |
| Flop-equivalents per compression: A100 312e12 / 9.558e9 = 3.3e4; H100 989e12 / 16.166e9 = 6.1e4; MACs per compression 1.6e4 / 3.1e4 | arithmetic on the two rows above |
| CPU flop-equivalents: one core at 2 x 512-bit FMA x 16 fp32 lanes x 2 flops x 2.4 GHz = 1.5e11 flop/s | assumption stated in the text (two AVX-512 FMA units), nominal clock from `docs/data/hash-cost.json` |
| Repository hash constant: `hash_macs` 1e5 (1e15 MAC/s against 1e10 compressions/s), `values_per_leaf` 32, `hash_units` = `hash_macs` x `UNITS_PER_MAC` / `values_per_leaf` = 9,375 cost units per position (one compression per leaf, no internal-node term); `h|I|/C` = 0.79 at the 70B shape | repo `src/veritor/evaluation/global_estimate.py` (`Inputs` docstring, `hash_units`); the 0.79 also asserted in `test_late_advice.py` |
| Measured `h` variants: 3 compressions (leaf + 2-block node) x 3.06e4 = 9.18e4 MACs per 32-value leaf = 2,869 per value (H100), 3 x 1.63e4 = 1,528 per value (A100); 6 compressions x 3.06e4 = 1.84e5 per value (repository frames, H100), 6 x 1.63e4 = 9.8e4 (A100), 6 x 1e5 = 6e5 (repository constant); `h|I|/C` 0.73 / 46 / 152 | arithmetic on the hashcat and datasheet rows and `docs/data/hash-cost.json` (`veritor_frames.*_sha256_blocks`) |
| `mma.sync m16n8k16` BF16 reproduced bit-exactly on an RTX 4090 (128 outputs of 16-term dot products; partial sums unobservable); GPT-2 Small forward reproduced word for word on CPU (363 tensors, 11.7 M words) | repo `docs/hardware-semantics.md`, `docs/gpt2-silicon.md` sections 4-5, `src/veritor/core/silicon.py` |
| Fixed-order chains vs cuBLAS on the RTX 4090: 1.8-2.8x at K = 768, 6.7-9.1x at K = 3,072, 28-47x at K = 50,272; ~35 ms of GEMM chains per run; CPU forward 9.75 s; pure-Python replay 160 µs per gate, 135 µs per interior position | repo `docs/gpt2-silicon.md` sections 4, 5 and 7 (`bench.json`) |
| Native emulator of `mma.sync` semantics: throughput not measured | repo: no such measurement exists (`docs/gpt2-silicon.md` section 8 lists a native prover as not done) |
| 140 GB of weights per decode step, 42 ms at 3.35 TB/s; 4.6e5 sampled requests a year = q x 2.93e13 | arithmetic: 70e9 parameters x 2 bytes; H100 bandwidth above; headline q |
| SP1 cycle constants: 3,296 per opened position, 2,668 per Merkle level (3 `SHA_COMPRESS` per level), 125-238 per gate, 5,800 per obligation, 50,000 per batch; Merkle 60-81% of cycles; proving 105-118 kHz, floors 10 s / 40 s, 84 GB RSS at 42 M cycles; one real proof: 16 obligations, 761,710 cycles, 2 shards, 2,804,880-byte core proof, prove 16.9 s, verify 0.07 s; compressed proof 1,272,577 bytes | repo `src/veritor/protocol/proofs/costs.py`, `docs/zk-backend.md` sections 2-3 |
| Tensor-core precompile proving rates on an RTX 4090: OpenVM `TC_MATMUL_4X4X16` 2,519,884 MAC/s (k = 64, 1.4-2 s proofs), `TC_DOT` 1,198,237, SP1 `TC_DOT` fork 982,066 (plateau k >= 2,048); native 1.8e14 fp8 MAC/s; pi = 7.1e7 / 1.5e8 / 1.8e8; toy-ISA alpha 2-3e9 | repo `docs/zk-backend.md` section 4, `src/veritor/protocol/proofs/costs.py` (`alpha_dot`, `NATIVE_FP8_MAC_PER_SECOND_4090`) |
| Plain RISC-V: 16 cycles per fixed-point MAC, 130-140 per f32 MAC | repo `src/veritor/protocol/proofs/costs.py` (`SP1_CYCLES_PER_FIXED_MAC`, sp1-op-bench section B) |
| RISC Zero GPU proving 1,207 kHz (fib, RTX 4090, po2 = 22; 808 kHz baseline) | https://github.com/risc0/risc0/pull/3761 |
| SP1 Hypercube: 93% of Ethereum blocks under 12 s on ~160 RTX 4090s (May 2025); 99.7% under 12 s on 16 RTX 5090s (Nov 2025) | https://blog.succinct.xyz/sp1-hypercube/ ; https://blog.succinct.xyz/real-time-proving-16-gpus/ |
| zkVM overhead relative to native CPU execution: Jolt ~500,000x; deployed zkVMs "millions of times"; "5 million and up" by CPU time | https://a16zcrypto.com/posts/article/a-new-era-in-snark-design-releasing-jolt/ ; https://a16zcrypto.com/posts/article/faqs-on-jolts-initial-implementation/ |
| pi = 2.4e9 (fixed-point) and 2e10 (f32) without a precompile: 16 or 135 cycles per MAC at 1.2e6 cycles/s against 1.8e14 MAC/s | arithmetic on the three rows above |
| Headline: q = 1.5707e-8, s = 8.9125e-3, U = 1.90e13 bits = 2.37 TB, split 1.57e-8 / 1.24e-8 / 0.0099999 / 7.16e-8; sensitivity: alpha 1.5e8 -> 4.98 TB, 1e9 -> 33.18 TB; budget 0.001 -> 23.70 TB, 0.1 -> 0.24 TB; interior "gate" -> 2.39 TB with 0.01% commit | repo `docs/global-estimate.md`, `docs/data/global-estimate.json` |
| Re-runs of `estimate()` with (`hash_macs`, `values_per_leaf`) = (1e5, 32) repository, (9.18e4, 32) and (4.89e4, 32) measured with nodes, (1.84e5, 1), (9.78e4, 1) and (6e5, 1) repository frames one value per leaf; interior "vu": q 1.5682e-8 to 1.5707e-8, U 2.3696 to 2.3734 TB (boundary 3.5e-8 to 1.4e-5, commit 6.1e-9 to 2.4e-6, replay 1.57e-8); interior "gate" with (6e5, 1): commit 0.00626, proof 0.00373, U 6.47 TB; "gate" with (1.84e5, 1): commit 0.00339, U 3.65 TB; "gate" with repository constants: commit 9.7e-5, U 2.39 TB | VM: `estimate(Inputs(hash_macs=..., values_per_leaf=..., interior=...))` |
| Year totals: 6.9e13 proved VUs = q s x 4.962e23; 3.9e27 flops = 2.93e13 x 1.34e14; 3,460 RTX-4090-years of proving at 2.52e6 MAC/s vs 3.5e5 GPU-years at 1.8e14 MAC/s (ratio 1.00%); proofs a year 1.1e12 at k = 64; 7.6e10 core-seconds at 0.07 s | arithmetic on the rows above |
| Transparent openings at the 70B shape: 16,385 positions x 35 levels x 3 blocks = 1.7e6 SHA-256 blocks, 57 ms at 33 ns; VU arithmetic 8,192 MACs at 1.5e11 flop/s = 0.1 µs; factor 5.4e5; beta_tr = 7.6e-5 at q s = 1.4e-10 | arithmetic on the measured block time and the CPU peak assumption |
| GPT-2 protocol run in Python (12 layers, one-token prompt, q = 1, s = 1/500): 249,669 VUs, 896,685 openings, verifier 44.9 s, forward 0.62 s, paths 23-27 levels, ~1.7 KB per opening; 1-layer slice 93,409 / 273,099 / 18.9 s; `WeightTree` 91.5 s | repo `docs/gpt2-silicon.md` section 7 |
| Output of the year: 3e16 tokens x 16 bits = 4.8e17 bits = 60 PB; U / output = 4e-5 | arithmetic on the headline inputs |
| SHA-256 1 hash of a 64-byte message needs 2 blocks; a 55-byte message 1 block (9 bytes of padding and length) | FIPS 180-4 padding rule |

Reproduce the measurements with `uv run --with blake3 python -m benchmarks.hash_cost` (writes `docs/data/hash-cost.json`) and the operating-point numbers with `python -m veritor.evaluation.global_estimate` and the GPT-2 constructor as in the tests named above.
