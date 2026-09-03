# 4. Costs, parameters, open problems

The security analysis in §3 is conditional on parameters the auditor sets
and on the honest developer being able to run the instrumented computation at
all. This section prices the protocol on realistic hardware and shapes,
works one parameter set, and lists what would have to be solved to make the
guarantee both tight and cheap. The cost findings are not favorable, and
§4.1 is the most important part of this document after §3.9.

## 4.1. The instrumentation overhead is not $1+o(1)$ on GPUs

PoUW's overhead claim is asymptotic in $n$ for square matrices with unit-cost
hashing. Two constants that the asymptotics hide dominate on accelerators.

**Scalar work costs ~30 tensor MACs.** An H100 does about $10^{15}$ INT8
tensor-core MACs per second and about $3\times10^{13}$ INT32 or FP32 scalar
operations per second on its CUDA cores. Any per-entry operation on a tile
(a hash round, a byte extraction, a nonlinearity) is therefore worth about 30
MAC-equivalents, and a light hash of $c'$ scalar ops per entry costs
$\approx30c'$ MAC-equivalents per entry. Call this $c\approx60$ for $c'=2$.

**The two overhead terms pull in opposite directions.** For a gate with
operands $n\times k$ and $k\times m$ and tile/noise rank $r$:

- *Hashing.* Running-sum tiles expose $nkm/r$ entries to the hash, so the
  hash costs $c/r$ of the gate's MACs. Hashing only the final output tiles
  would cost $c/k$ instead, but then a prover who has cached $AB$ (repeated
  inputs, or a model whose activations ignore the prompt) computes only the
  low-rank cross terms and earns tickets at a fraction of the cost (§3.10);
  intermediates are what make cached work useless.
- *Noise and decode.* $E=E_LE_R$, $F=F_LF_R$, $(AF_L)F_R$, $E_L(E_RB')$ cost
  $2r(nk+nm+km)$ MACs, i.e. $2r(1/m+1/n+1/k)$ of the gate. The noise rank
  must equal the tile size (§2.2), so this term grows with $r$.

Total: $\mathrm{ovh}(r)\approx c/r+b\,r$ with $b=2(1/n+1/k+1/m)$. The minimum
is $2\sqrt{cb}$ at $r^*=\sqrt{c/b}$.

| Gate (n, k, m) | $b$ | $c=60$ (scalar hash) | $c=16$ (tensor sketch, below) |
|---|---|---|---|
| MLP up, (8192, 8192, 32768) | $5.5\times10^{-4}$ | $r^*\approx330$, **36 %** | $r^*\approx170$, **19 %** |
| MLP up, (65536, 8192, 32768) | $3.3\times10^{-4}$ | $r^*\approx430$, **28 %** | $r^*\approx220$, **15 %** |
| Attention $QK^\top$, (8192, 128, 8192) | $\approx1/64$ | any $r\le128$: decode alone $\ge2\times$ | same |
| Attention $PV$, (8192, 8192, 128) | $\approx1/64$ | same | same |

**Attention is not ticketable by this scheme.** Its contraction dimension is
$d_{\rm head}=128$. With $r=k$ the correction terms $(AF_L)F_R$ and
$E_L(E_RB')$ are each $n\,r\,m=nkm$: the decode costs twice the product. With
$r<k$ the hash term $c/r\ge c/128\approx47\,\%$. And hashing only outputs
(cost $c/k$, still $47\,\%$ for scalar hashing) is insecure against cached
operands. At long context attention is 30–50 % of inference FLOPs, so a large
fraction of an honest developer's compute cannot earn tickets, and must be
treated like the non-matmul work of §3.9(iv): either credited by trusting the
architecture's ratio, or covered by filler.

**A tensor-core-friendly hash.** The hash must apply a nonlinearity to every
tile entry (any *linear* sketch $TG$ equals $A'_{i\ell}(B'_{\ell j}G)$ and costs
$2r^2s$ instead of $r^3$; a *quadratic* one, $\|T\|_F^2=\langle A'^\top A',B'B'^\top\rangle$,
costs $r^2$ per tile after $O(nkr+kmr)$ shared work; §3.1 A8). One
nonlinearity that tensor cores can then consume: reinterpret the INT32
accumulator tile as an $r\times4r$ byte matrix $T_b$ (byte extraction is
non-algebraic in the entries; carries propagate) and compute $Y=T_bG$ for a
random $4r\times s$ matrix $G$ from $\zeta_g$, then hash the $rs$ entries of
$Y$ with a scalar hash. Cost $4r^2s$ tensor MACs plus $\approx60\,rs$
scalar; with $s=4$ that is $c\approx16$ in the table. Whether byte
reinterpretation of the accumulator is free in a fused epilogue is a kernel
question; whether $T_b\mapsto T_bG$ is unpredictable without $T$ is a
cryptanalytic one. Neither is settled here.

The honest conclusion: **with the paper's noise scheme, instrumenting the
matmuls of a transformer costs 15–40 % of their compute, not $o(1)$, and
attention cannot be instrumented at all.** The route to low single digits is
a self-canceling transform with no decode (PoUW Appendix A), which removes the
$br$ term and lets $r$ grow until $c/r$ is small — but that scheme is
conjectural, needs integer-exact transforms (randomized Walsh–Hadamard over
$\mathbb Z_p$ rather than real rotations), and is broken by degenerate
operands (zero or low-rank blocks, which real post-ReLU activations
approximate), so it would need to be combined with at least low-rank additive
noise, reintroducing part of the decode term. Open problem (a) below.

## 4.2. Ticket proofs are cheap

A ticket proof recomputes $\le k/r$ tile products ($\le kr^2$ MACs, about
$1.3\times10^8$ at $k=8192,r=128$) and opens $2k/r$ blocks of $r^2$ values.
The statement is exact-arithmetic matmul plus Merkle paths, for which
sumcheck-based systems achieve prover overheads in the $10^2$–$10^3$ range;
take $10^3$. A ticket then costs $\sim10^{11}$ MAC-equivalents to prove, or
0.1 ms of one H100. With $K=10^8$ tickets per day the proving cost is
$\sim10^4$ device-seconds per day, which is $\sim10^{-7}$ of a
$10^6$-device fleet. Verification at $\sim10$ ms per proof is $10^6$ CPU-seconds
per day, or one proof per request after recursive aggregation. Even
$K=10^{10}$ is affordable. **Statistical precision is not the bottleneck;
§4.1 is.**

## 4.3. Statistical precision and granularity

$K\sim\mathrm{Bin}(M,p)$ with $M$ the number of tiles computed. The
threshold $K_{\min}$ of §1.8 with $z$ standard deviations of honest slack and
$z'$ of adversarial slack resolves a deficit of
$\Delta\approx(z+z')/\sqrt{pM}$. At $pM=K=10^8$ and $z=z'=5$, $\Delta=10^{-3}$;
at $K=10^{10}$, $10^{-4}$. Per-device accounting (each device must present
its own quota) changes the granularity, not the total: at $K=10^8$ over $10^6$
devices each device has $\sim100$ tickets per day, so per-device deficits of
$\sim50\,\%$ are visible in a day and $\sim5\,\%$ in a hundred days. A
thinly spread undeclared workload (attack 21 of §3.8) is still bounded by the fleet-wide
$\delta$.

Statistical slack $\Delta$ enters the free-compute bound additively with
$\delta$, $\varepsilon_{\rm alg}$ and $\varepsilon_\alpha$ (§3.7). It is the only
one of the four the auditor can drive to zero by spending money.

## 4.4. A worked parameter set

Fleet of $10^6$ H100-class devices, one-day epoch, INT8 tensor peak
$10^{15}$ MAC/s per device: $\Phi_T\approx8.6\times10^{25}$ MAC-equivalents.

| Parameter | Value | Note |
|---|---|---|
| $r$ (tile and noise rank) | 256 | near $r^*$ for MLP shapes with $c\approx60$ |
| $\alpha$ (tile fraction of instrumented MLP work) | 0.72 | $1/(1+0.36)$ from §4.1 |
| non-matmul + attention fraction of honest inference | 0.3–0.5 | unevidenced; filler or trust |
| tiles per epoch at full utilization, $\alpha\Phi_T/r^3$ | $3.7\times10^{18}$ | |
| $p$ | $2.7\times10^{-11}$ | for $K\approx10^8$ |
| $\Delta$ at $z=z'=5$ | $10^{-3}$ | |
| $\varepsilon_{\rm alg}$ | 0–0.05 | tensor-core cost model; assumption |
| $\varepsilon_\alpha$ | 0.02–0.05 | how well the auditor knows the best implementation |
| $\delta$ (policy) | 0.01 | |
| resulting free-compute bound, §3.7 | $\approx(0.01+0.001+0.05+0.05)\Phi_T\approx0.1\,\Phi_T$ | before $\lambda$ and residuals |
| $(\Gamma,X,Y,N)$ | PBA's $(10\ \text{H100h},10\ \text{MB},1\ \text{MB},\text{per day})$ | one task per request |
| $\lambda\Phi_T$ | $N\cdot5\times10^9$ MACs $\approx5\times10^{16}$ at $N=10^7$ | negligible against $\delta\Phi_T$ |
| Verity $(q,s,\eta)$, $U$ per request | as in the Verity draft's calibration (kilobits per request) | unchanged |

Read the free-compute line carefully: the bound is $\approx10\,\%$ of the
fleet, dominated by the two non-cryptographic terms, and $10\,\%$ of a
$10^6$-device fleet is $10^5$ devices. The cryptographic and statistical
parts of the protocol are tight; what is loose is knowledge of how fast the
adversary's tiles can be. The honest developer, meanwhile, pays 15–40 % on
MLP matmuls and either fills or forfeits credit for attention and the rest.

## 4.5. Open problems

Ordered by how much they would change the picture.

**(a) A cheap, provably unpredictable tile hash, and a decode-free noise.**
The $c/r+br$ tension of §4.1 is the whole cost story. Two directions: prove
(or break) the byte-sketch hash so $c$ drops to $\sim16$; and make PoUW
Appendix A's self-canceling transform exact over $\mathbb Z_p$ and robust to
degenerate operands so $b\to0$. Either alone gives $\sim15\,\%$; both give
low single digits. A related question is whether a hash chain *through* the
running sums (noise for block $\ell+1$ seeded by the hash of block $\ell$)
could replace per-tile hashing with per-row-block hashing at no loss.

**(b) Attention.** No variant of the scheme prices attention below $2\times$.
Options: leave it unevidenced (then §3.9(iv) grows to 30–50 % at long
context); trust the architecture ratio; or find a proof of work whose cost
scales with output entries rather than contraction length. The latter is
what a proof of *memory traffic* would be, and none is known.

**(c) Freshness for non-matmul work.** Softmax, norms, activations, KV-cache
reads are memory-bound and produce no tickets. For adversarial circuits they
need not be fresh (attacks 11 and 12 of §3.8). Any scheme that evidences bytes moved rather
than MACs performed would close this and (b) together.

**(d) The heavy-MAC theorem for restructured training.** §3.5 counts MACs
whose results leave the task through the same matmul. A lower bound of the
form "a $(\Gamma,X,Y)$-bounded task advances a training run by at most
$f(X,Y)$ bits of progress, for any restructuring" — a communication
complexity statement — is what "inference-shaped" needs to rest on. PBA's
Remark A.5 and Assumption A.7 are the current placeholders.

**(e) The small-state escape.** Adapters of $\le\min(X,Y)$ bits train at full
declared compute (§3.9 i). Structurally, a prompt-derived matrix reused as a
weight across the token dimension is distinguishable from attention's
per-position $K,V$ — but so is any in-context-learning trick a developer
might legitimately use. Whether to charge "dynamic weights" as heavy, and
what that breaks, is a modeling decision that should be made with a
concrete list of legitimate inference patterns in hand.

**(f) Composition with wardens and wipes.** The residual $\delta\Phi_T$ is
where PBA's other two mechanisms act: training in the residual must survive
wipes through $R$ bits and must synchronize through interconnect limits.
Whether the three-way composition yields a slowdown factor that multiplies,
adds, or saturates is not worked out anywhere.

**(g) Filler that co-runs with memory-bound decode.** If tensor cores can
run filler tiles while decode streams weights, the honest developer's fill
cost is power rather than throughput. If they cannot, filling means
displacing real work, and $\delta$ has a floor set by the decode fraction.

**(h) Exact arithmetic for the serving stack.** INT8 with 7-bit operands
works; MXFP8/NVFP4 need block-integer accumulation; FP16 does not work. The
quality cost of the extra quantization bit and the engineering cost of
block-exact kernels are borne by the honest party.

**(i) The reference efficiency $\alpha$.** $\varepsilon_\alpha$ is one-for-one
free compute. An open, competitively optimized reference implementation per
hardware generation is the only mitigation; the auditor's threshold should
track it.

**(j) Per-device or per-pod attribution.** Tickets carry a request and gate
but not a device. Binding tickets to devices (a device key in $\zeta_g$'s
derivation, attested at admission) would let the auditor localize deficits
and would also catch undeclared hardware more directly than $K_{\max}$.

## 4.6. What is settled

Against this list, what the analysis does establish:

- Under A1–A3 the ticket count is a sound lower bound on tiles computed
  under fresh noise on operands fixed before the noise (S1), with statistical
  precision the auditor can buy.
- Committing the header before $\zeta$ removes the cheap-operand attack
  without hashing operands (§3.3); making every matmul operand a Verity
  boundary value makes the ticket proofs and Verity's sampling open the same
  leaves, which is what turns "matmuls happened" into "this circuit's
  matmuls happened" (S2).
- Marking partial dot products rather than tiles as verification units keeps
  `Bound` unchanged, and the noise gates have zero capacity (S4).
- The $(\Gamma,X,Y,N)$ predicate is a fold over the index using the replay-unit
  boundary as the task interface; for independent inference it is one task
  per request; Loomis–Whitney gives the heavy-MAC bound with the stated
  hypotheses (S3).
- Each of the three ingredients is necessary (§3.10).
- The residual is $(\delta+\Delta+\varepsilon_{\rm alg}+\varepsilon_\alpha)\Phi_T$
  plus the small-state escape, and at fleet scale that is a lot of compute in
  absolute terms; and the honest cost is 15–40 % on the matmuls that can be
  instrumented, with attention outside the scheme.
