# SP1 Operator-Cost Benchmark Plan

**Status:** plan only — no code written yet.
**Purpose:** measure what it costs to prove ML-relevant primitive operators inside the SP1 zkVM, so we can choose the instruction granularity for the sampled-execution-tape verification protocol (e.g. "is a 4096-element f32 dot product affordable as one sampled leaf check?").
**Target machine:** macOS arm64 (Apple Silicon), CPU proving only, no CUDA.
**SP1 version:** v6.4.0 ("Hypercube", RV64IM over the KoalaBear field). The local clone at `/Users/danielreuter/projects/sp1` is checked out at the `v6.4.0` tag, and `sp1-sdk 6.4.0` is the newest release on crates.io, so we pin `6.4.0` everywhere.

Every SP1 claim below is grounded in a file from the local clone; paths are relative to `/Users/danielreuter/projects/sp1`.

---

## 0. Decisions (2026-08-22)

Answers to the open questions in §8:

1. **Budget:** start *minimal* and iterate — smallest useful matrix first (noop, f32 dot, fixed dot, nonlinearities; execution-only), report numbers, then expand. Phase B starts with 1–2 calibration proofs, grows only as needed.
2. **Proof modes:** core proofs for calibration, plus 1–2 compressed data points. No Groth16/Plonk.
3. **Prover network:** deferred; keep `SP1_DUMP=1` artifacts replayable.
4. **Fixed-point:** f32 is primary; fixed-point (O7) stays in the matrix as a comparison point only.
5. **f64:** skipped entirely.
6. **Merkle leaf granularity (O9):** block leaves primary (one operand block per leaf); per-cell leaves as a secondary data point only.

---

## 1. Ground truth about SP1 (from the repo)

### 1.1 Guest ISA and floating point

- The executor's opcode enum (`crates/core/executor/src/opcode.rs`) contains only RV64IM: integer ALU, mul/div, loads/stores, branches, jumps. There is **no F/D extension**, so every f32 operation in guest code is compiled by LLVM into soft-float integer routines (`__addsf3`, `__mulsf3`, … from compiler-builtins).
- Transcendentals: the zkVM runtime exports `#[no_mangle] extern "C"` shims for the whole libm surface (`exp`, `expf`, `tanhf`, `sigmoid` building blocks, etc.) that forward to the pure-Rust `libm` crate — see `crates/zkvm/entrypoint/src/libm.rs`. So `x.exp()` on an `f32` in the guest resolves to `libm::expf` compiled to soft-float RV64IM. Calling `libm` directly in the guest is equivalent and makes the dependency explicit.
- Precompiles exist only for hashes/curves/bigint: `SHA_EXTEND`, `SHA_COMPRESS`, `KECCAK_PERMUTE`, `POSEIDON2`, curve ops, u256/u320 ops (`crates/core/executor/src/syscall_code.rs`). **There are no float precompiles.** Guest-side raw syscalls for SHA-256 have RV64 signatures `syscall_sha256_extend(w: *mut [u64; 64])` and `syscall_sha256_compress(w: *mut [u64; 64], state: *mut [u64; 8])` (`crates/zkvm/lib/src/lib.rs`).
- Guests have std: the `examples/sha` guest uses `println!` and `rand`. Guest I/O: `sp1_zkvm::io::read::<T>()` (bincode-deserializes — costs cycles), `read_vec()` (raw bytes into uninitialized memory — cheap), `commit()` / `commit_slice()` (`crates/zkvm/lib/src/io.rs`).

### 1.2 Execution reports: what we get for free (no proving)

`client.execute(elf, stdin).await` returns `(SP1PublicValues, ExecutionReport)` (`crates/sdk/src/prover/execute.rs`). The report (`crates/core/executor/src/report.rs`) exposes:

- `opcode_counts: EnumMap<Opcode, u64>` — per-opcode instruction counts, and `total_instruction_count()` (this is what SP1's own perf tooling calls "cycles", see `crates/perf/src/bin/perf.rs`).
- `syscall_counts: EnumMap<SyscallCode, u64>` and `total_syscall_count()`.
- `cycle_tracker: HashMap<String, u64>` + `invocation_tracker` — populated when the guest prints `cycle-tracker-report-start: <label>` / `cycle-tracker-report-end: <label>` (parsed in `crates/core/executor/src/minimal/write.rs`). **This lets one guest binary report per-phase cycle counts (io vs. merkle vs. operator) in a single run.**
- `touched_memory_addresses`, `exit_code`.
- `gas: Option<u64>` via `report.gas()` — computed when `.calculate_gas(true)` (the default; `crates/sdk/src/prover/execute.rs`).

Gas model (`crates/core/executor/src/vm/gas.rs`): raw gas = `(3 * trace_area + complexity) / 10`, then `gas()` normalizes by `* 10 / 191` (`GAS_NORMALIZATION_FACTOR`, `crates/core/executor/src/report.rs`). Net: `gas ≈ (3·trace_area + complexity) / 191`. Trace area is `Σ (rows of chip × per-row cost)` with per-chip costs in `crates/core/executor/src/artifacts/rv64im_costs.json`. Interesting entries (trace-area cost per row):

| chip | cost | chip | cost |
|---|---|---|---|
| Add / Addi | 33 / 30 | ShaCompress | 206 |
| Mul | 82 | ShaExtend | 128 |
| DivRem | 246 | KeccakPermute | 2640 |
| Load* / Store* | ~39–50 | Global (per touched addr) | 241 |
| Branch / Jal / Jalr | 45 / 31 / 35 | MemoryLocal | 20 |

Gas is a **proving-cost proxy calibrated by Succinct** — it weights a Mul at ~2.5× an Add, a DivRem at ~7.5× — so we report both raw instruction counts and gas, and calibrate wall-clock against gas.

### 1.3 Sharding thresholds (when a program stops fitting in one shard)

From `crates/core/executor/src/opts.rs`:

- `ELEMENT_THRESHOLD = (1 << 28) + (1 << 27)` ≈ **4.03 × 10⁸ trace elements** per shard (total trace area).
- `HEIGHT_THRESHOLD = 1 << 22` = **4,194,304 rows** max for any single chip table per shard.
- Both are overridable via env vars (`ELEMENT_THRESHOLD`, `HEIGHT_THRESHOLD`) or `SP1CoreOpts` passed to `ProverClient::builder().cpu().core_opts(opts)` (`crates/sdk/src/cpu/builder.rs`).

Implication: a program dominated by one ALU chip splits at ~4.19M instructions of that op; mixed programs hit the element threshold somewhere in the millions of cycles. All of our single-operator benchmarks except possibly `n = 16384` dot products should fit in **one shard**; we record the actual shard count per proof (a core proof is `SP1Proof::Core(Vec<ShardProof>)` — `crates/sdk/src/proof.rs`, match arm in `crates/sdk/src/prover.rs`).

### 1.4 SDK API (v6, async)

From `crates/sdk/src/client.rs`, `crates/sdk/src/prover/prove.rs`, `crates/sdk/src/env/mod.rs`, and `examples/fibonacci/script/bin/*.rs`:

~~~rust
let client = ProverClient::builder().cpu().build().await;   // or ::from_env(): SP1_PROVER ∈ {cpu (default), cuda, mock, light, network}
let (pv, report) = client.execute(elf, stdin).await?;        // ExecutionReport; .calculate_gas(true) is default
let pk = client.setup(elf).await?;                           // proving key (time this separately)
let proof = client.prove(&pk, stdin).core().await?;          // default mode is Core; also .compressed()/.groth16()/.plonk()
client.verify(&proof, pk.verifying_key(), None)?;
proof.save("proof.bin")?;                                    // bincode; file size = proof size metric
~~~

Proof modes (`crates/sdk/src/prover/prove.rs` doc comments): **Core** — proof size grows linearly with cycles (default); **Compressed** — constant size, recursively aggregated, slower; **Groth16/Plonk** — onchain-verifiable wrappers (~100k/~300k verify gas). The prover client is initialized once and reused (doc comment in `crates/sdk/src/client.rs`).

### 1.5 Building guest programs

From `examples/fibonacci/{program,script}` and `crates/build/src/lib.rs`:

- Guest crate: depends on `sp1-zkvm`; `#![no_main]` + `sp1_zkvm::entrypoint!(main)`.
- Host crate: `build.rs` calls `sp1_build::build_program("../program")`; embed the ELF with `include_elf!("<program-crate-name>")` → `Elf::Static`. Alternatively load from disk with `Elf::Dynamic(bytes)` (used by `crates/perf/src/bin/perf.rs`).
- The build helper spawns cargo with `RUSTUP_TOOLCHAIN=succinct` (`crates/build/src/command/local.rs`), targeting `riscv64im-succinct-zkvm-elf` (`crates/build/src/lib.rs`, `DEFAULT_TARGET`). **The Succinct rustc toolchain must be installed** via `sp1up`, which installs the `cargo prove` CLI and runs `cargo prove install-toolchain` (`sp1up/sp1up`). No C toolchain needed for pure-Rust guests.
- `SP1_SKIP_PROGRAM_BUILD=true` skips guest rebuilds (`crates/build/src/lib.rs`).
- SHA-256 precompile via patched crate: add a workspace-root `[patch.crates-io]` entry, exactly as `examples/Cargo.toml` and `patch-testing/Cargo.toml` do. Newest patch tag in the repo: `patch-sha2-0.10.9-sp1-6.2.0` of `sha2` from `sp1-patches/RustCrypto-hashes` (`patch-testing/Cargo.toml`).

### 1.6 SP1's own benchmarking tooling (what to reuse, what to avoid)

- **Avoid the root `eval.sh`** — it is stale: it references `examples/<p>/program` layouts and a `riscv32im-succinct-zkvm-elf` ELF path plus a root-level `eval/` dir and v4-era flags (`--shard-size`, `--hashfn`) that don't match v6.
- `crates/perf` is the current tooling: `sp1-perf` (execute + setup + prove + verify with timings, takes `--program program.bin --stdin stdin.bin --mode cpu`), `sp1-perf-executor` (execution-only; modes `gas`, `minimal`, `minimal_trace`, `node`; prints cycles, MHz, gas — `crates/perf/src/bin/executor.rs`), and the `sp1-perf-prover` REPL that logs `cycles, gas, …, prove_secs, khz, mgas_per_s` rows to CSV (`crates/perf/README.md`). We mirror that metric set.
- Any `sp1-sdk` binary run with `SP1_DUMP=1` dumps `program.bin` + `stdin.bin` (`crates/sdk/src/cpu/prove.rs`, `crates/perf/README.md`), so our workloads can later be replayed by SP1's own tools or shipped to the prover network unchanged.

---

## 2. Benchmark matrix

Two independent axes: **operator** (what the sampled leaf recomputes) and **measurement tier** (execution-only vs. real proof — §3).

### 2.1 Operators

| # | family | variants | sizes n | why |
|---|--------|----------|---------|-----|
| O0 | `noop` baseline | read spec, commit constant | — | fixed program overhead (runtime init, io) to subtract from everything |
| O1 | `io_read` | `read_vec` raw bytes | 1 KiB, 32 KiB, 256 KiB | cost of feeding operands through stdin — the real protocol reads tape data as hints; separates I/O from compute |
| O2 | `f32_dot` | data from in-guest PRNG; separately data from stdin | 64, 256, 1024, 4096, 16384 | the headline question: per-element soft-float FMA cost; slope = marginal cost, intercept = overhead |
| O3 | `f32_elementwise` | `y[i] = a[i]*b[i] + c` chain, depth 1 and 4 | 1024, 4096 | elementwise map cost without the reduction dependency chain |
| O4 | `f32_sum` | sequential sum reduction | 1024, 4096, 16384 | reductions; also isolates add-only soft-float cost |
| O5 | `nonlin` | `libm::expf`, `tanhf`, sigmoid (`1/(1+expf(-x))`) | 64, 1024, 4096 calls | per-call transcendental cost; these dominate softmax/attention leaves |
| O6 | `softmax_row` | max, sub, expf, sum, div | 64, 256, 1024, 4096 | a realistic composite leaf: mixes O4 + O5 |
| O7 | `fixed_dot` | i32×i32→i64-accumulate dot (Q-format) | 64, 1024, 4096, 16384 | comparison point: RV64IM `Mul` costs 82 trace-area units vs. dozens of instructions per soft-float mul; quantifies the "quantize the tape" option |
| O8 | `merkle_sha256` | (a) patched `sha2` crate → precompile; (b) vendored pure-Rust SHA-256 | depth 20, 30 | the other half of every sampled check; precompile-vs-soft gap decides whether we require the patched crate |
| O9 | `leaf_check` | full protocol shape: read operands + auth path from stdin, verify Merkle path (precompile sha2), recompute operator, commit | operator ∈ {f32_dot 1024, f32_dot 4096, softmax_row 1024, fixed_dot 4096} × depth ∈ {20, 30} | the derived quantity the protocol design actually needs |

Optional (pending answers to open questions): `f64_dot`, multi-leaf batching in O9 (k = 4, 16 sampled checks per proof) to measure amortization of the fixed overhead.

Design notes:

- **PRNG vs. stdin data:** O2 runs both ways. In-guest xorshift PRNG (seeded from stdin) isolates pure compute; stdin-fed operands (`read_vec` + byte reinterpretation, not bincode `read::<Vec<f32>>`) match the real protocol. O1 quantifies the difference so we don't conflate I/O with arithmetic.
- **Dead-code protection:** every operator commits a checksum of its result (`io::commit`); inputs are runtime data, so LLVM cannot constant-fold. `std::hint::black_box` around accumulators as a belt-and-suspenders measure.
- **Phase attribution:** each phase is wrapped in `cycle-tracker-report-start/end: <label>` prints (`io`, `merkle`, `op`), which the executor sums into `report.cycle_tracker` (§1.2). One binary, one execution, clean attribution.
- **Pure-Rust SHA-256 (O8b):** `[patch.crates-io]` applies workspace-wide, so an unpatched `sha2` can't coexist with the patched one in one binary. We vendor a minimal (~100-line) pure-Rust SHA-256 compression in the guest as the soft baseline and assert it agrees with the patched crate's output.
- **Merkle arithmetic:** hashing a 64-byte parent input (`H(left ‖ right)`) requires 2 compression blocks with standard SHA-256 padding, so a depth-d path costs ~2d extend+compress pairs — depth 30 ⇒ ~60 precompile invocations. Leaf hashing adds ~`ceil(leaf_bytes/64)` blocks; a 4096-f32 operand leaf (16 KiB) adds ~256 blocks, which likely rivals the path cost — O9 measures leaf hashing and path hashing under separate cycle-tracker labels so granularity trade-offs are visible.

### 2.2 Priors (to validate, not trust)

Order-of-magnitude expectations that shaped the matrix — the whole point of the benchmark is to replace these with measurements:

- soft-float f32 add/mul: ~30–150 RV64IM instructions each ⇒ 4096-dot ≈ 8192 flops ≈ 0.3–1.5M cycles → fits one shard (§1.3), so a 4096-dot leaf check is *plausibly* affordable.
- `libm::expf`/`tanhf`: ~0.5–2K cycles per call ⇒ a 1024-softmax row ≈ 1–3M cycles.
- SHA-256 precompile: ~10² cycles-equivalent per block (ShaCompress row cost 206 + control, §1.2) vs. pure-Rust ~3–8K instructions per block ⇒ depth-30 path: trivial with precompile, ~0.2–0.5M cycles without.
- Local CPU core proving on Apple Silicon: expect O(10⁴–10⁵) cycles/s effective throughput (unknown for v6 on this machine — this is exactly what Phase B calibrates). A 1M-cycle program might take ~10 s–2 min to prove locally.

---

## 3. Metrics and methodology

### Phase A — execution-only sweep (cheap, exact, run the full matrix)

For every cell in the matrix, run `client.execute(...)` once (execution is deterministic; no repeats needed) with `.calculate_gas(true)` and record:

| metric | source |
|---|---|
| total instructions ("cycles") | `report.total_instruction_count()` |
| per-opcode counts (top 10 + full JSON) | `report.opcode_counts` |
| syscall counts (SHA_EXTEND / SHA_COMPRESS etc.) | `report.syscall_counts` |
| gas | `report.gas()` |
| per-phase cycles: `io` / `merkle` / `op` | `report.cycle_tracker` |
| touched memory addresses | `report.touched_memory_addresses` |

Derived per operator: **marginal cost per element** — fit `cycles(n) = a + b·n` across the size sweep (slope b = cycles/element, intercept a = fixed overhead); same for gas. Also compute predicted trace area offline from `opcode_counts × rv64im_costs.json` and predicted shard count vs. the §1.3 thresholds.

### Phase B — proving calibration (expensive, run a selected subset)

Real local core proofs on this machine for ~8–12 configs spanning ~10⁵ → ~10⁷ cycles (including at least one config that crosses the 1-shard boundary, e.g. `f32_dot 16384` or a repeated-op loop scaled up). Per config, 3 repetitions (median reported; min/max recorded), timing each stage separately as `crates/perf/src/bin/perf.rs` does:

- prover init (once), `setup(elf)`, `prove(...).core()`, `verify` — wall-clock each.
- shard count (`SP1Proof::Core(shards).len()`), proof size (bytes of `proof.save()` output).
- Derived: **kHz = cycles / prove_seconds** and **Mgas/s = gas / prove_seconds** (same metrics as `sp1-perf-prover`'s CSV, `crates/perf/README.md`).

Fit `prove_seconds ≈ α + β·gas` (and vs. cycles) over the calibration set. Environment control: plugged in, minimal background load, record chip model / RAM / macOS version / rustc + sp1 versions in every CSV row. Optionally repeat 1–2 configs as `.compressed()` proofs to measure the constant-size wrap cost and proof size (open question Q2).

### Phase C — derived quantities (the protocol answer)

For each O9 `leaf_check` config: measured cycles/gas (Phase A) × calibrated seconds-per-gas (Phase B) ⇒ **estimated proving seconds and proof bytes per sampled instruction check**, as a function of operator size and Merkle depth. Present as a table plus a short "cost model" section:

~~~text
T_prove(op, n, depth) ≈ α + β · [gas_io(n) + gas_merkle(depth, leaf_bytes) + gas_op(n)]
~~~

with the measured coefficient table, so the protocol design can extrapolate to operator sizes we didn't run.

---

## 4. Project layout

New standalone workspace at `/Users/danielreuter/projects/sp1-op-bench` (nothing inside the sp1 clone):

~~~text
sp1-op-bench/
├── Cargo.toml              # workspace = ["guest", "host"]; [patch.crates-io] sha2 → sp1-patches (root-level, as examples/Cargo.toml does)
├── rust-toolchain.toml     # stable (host); guest uses the `succinct` toolchain automatically via sp1-build
├── guest/
│   ├── Cargo.toml          # sp1-zkvm = "6.4.0", libm = "0.2", sha2 = "0.10.9" (patched), serde
│   └── src/main.rs         # single dispatcher: reads OpSpec from stdin, runs op, commits checksum
├── host/
│   ├── Cargo.toml          # sp1-sdk = "6.4.0", sp1-build (build-dep), tokio = "1", clap, serde, serde_json, csv
│   ├── build.rs            # sp1_build::build_program("../guest")
│   └── src/main.rs         # subcommands: execute (Phase A), prove (Phase B), report (Phase C)
└── results/                # execution.csv, proving.csv, report.md (gitignored artifacts + committed summaries)
~~~

Version pins: `sp1-sdk = "6.4.0"`, `sp1-zkvm = "6.4.0"`, `sp1-build = "6.4.0"` (repo HEAD tag and crates.io max version agree). sha2 patch tag `patch-sha2-0.10.9-sp1-6.2.0` (newest in `patch-testing/Cargo.toml`; the patched crates degrade gracefully to pure Rust when compiled for the host, per `patch-testing/build-host`).

Guest dispatcher sketch (single ELF, operator + size from stdin ⇒ **no rebuilds across the matrix**):

~~~rust
#![no_main]
sp1_zkvm::entrypoint!(main);

#[derive(serde::Serialize, serde::Deserialize)]
pub enum OpSpec {
    Noop,
    IoRead { bytes: u32 },
    F32Dot { n: u32, from_stdin: bool, seed: u64 },
    F32Elementwise { n: u32, depth: u32, seed: u64 },
    F32Sum { n: u32, seed: u64 },
    Nonlin { kind: NonlinKind, calls: u32, seed: u64 },
    SoftmaxRow { n: u32, seed: u64 },
    FixedDot { n: u32, seed: u64 },
    MerkleSha256 { depth: u32, precompile: bool },
    LeafCheck { op: Box<OpSpec>, depth: u32 },
}

pub fn main() {
    let spec: OpSpec = sp1_zkvm::io::read();
    println!("cycle-tracker-report-start: op");
    let checksum = run(&spec); // each arm wraps sub-phases in their own tracker labels
    println!("cycle-tracker-report-end: op");
    sp1_zkvm::io::commit(&checksum);
}
~~~

Host runner sketch (Phase A loop; Phase B analogous with `setup`/`prove`/`verify` timing):

~~~rust
let client = ProverClient::builder().cpu().build().await; // init once, reuse
for spec in matrix() {
    let mut stdin = SP1Stdin::new();
    stdin.write(&spec);
    if let Some(bytes) = operand_bytes(&spec) { stdin.write_vec(bytes); }
    let (_pv, report) = client.execute(ELF, stdin).await?; // .calculate_gas(true) is default
    csv.write(row_from(&spec, &report))?;
}
~~~

### Toolchain install steps (one-time, ~10 min + downloads)

~~~sh
# 1. Install sp1up, then the cargo-prove CLI + `succinct` rustc toolchain
curl -L https://sp1up.succinct.xyz | bash
sp1up

# 2. Verify
cargo prove --version
rustup toolchain list | grep succinct

# 3. Scaffold check (optional): `cargo prove new --bare tmp-check` pulls
#    succinctlabs/sp1-project-template; we hand-roll instead, mirroring examples/fibonacci.
~~~

No C toolchain needed (pure-Rust guest; `sp1up --c-toolchain` exists but is only for C FFI guests, `sp1up/sp1up`). No Docker needed unless we later want reproducible ELF builds (`BuildArgs::docker`, `crates/build/src/lib.rs`) or Groth16/Plonk wrapping.

---

## 5. Runbook

| step | what | est. time |
|---|---|---|
| 0 | Install toolchain (`sp1up`), scaffold workspace, first guest build | 30–60 min (mostly downloads/compile) |
| 1 | Smoke test: `noop` executes; sha precompile counts appear in `syscall_counts`; vendored SHA-256 agrees with patched crate | 15 min |
| 2 | Phase A full matrix (~45 execution runs) + `execution.csv` | ~15–30 min wall-clock |
| 3 | Sanity-check Phase A: linear fits, io-vs-op attribution, per-opcode sanity (soft-float ⇒ no float opcodes, all IM) | 30 min |
| 4 | Phase B calibration proofs (~10 configs × 3 reps, core mode) | unknown until first proof; budget 0.5–3 h (question Q1) |
| 5 | Phase C: fit cost model, generate `report.md` with the leaf-check answer table | 30 min |
| 6 | Optional: compressed-mode data points, network data point, f64/batching variants | per answers to open questions |

Fallbacks if local proving is slower than expected: cut Phase B to 5 configs × 2 reps; the execution-only matrix (Phase A) plus SP1's published gas calibration still yields relative operator costs, and `SP1_DUMP=1` artifacts let us re-run the exact workloads on beefier hardware or the prover network later.

---

## 6. Deliverables

1. `results/execution.csv` — one row per matrix cell: spec fields, total instructions, gas, per-phase cycles, syscall counts, top opcode counts, predicted shards.
2. `results/proving.csv` — one row per calibration proof: spec, cycles, gas, setup_s, prove_s, verify_s, shards, proof_bytes, khz, mgas_per_s, machine metadata.
3. `results/report.md` — the answer document: per-operator cycles/element and gas/element tables, the calibrated seconds-per-gas coefficient, and the O9 table "estimated proving time + proof size per sampled leaf check vs. operator size and Merkle depth", with explicit guidance on the 4096-f32-dot question.

---

## 7. Risks and caveats

- **Absolute times are machine-relative.** Apple-Silicon CPU proving is supported but not SP1's optimized deployment path (CUDA/network). We treat local seconds as calibration for *relative* costs; production numbers need a GPU or network data point.
- **v6 gas is calibrated to Succinct's infra**, normalized to match v4 op-succinct blocks (`GAS_NORMALIZATION_FACTOR` comment, `crates/core/executor/src/report.rs`). Our local seconds-per-gas fit may differ from theirs; we report our own fit.
- **Proving memory:** default guest memory limit is 24 GB (`DEFAULT_MEMORY_LIMIT`, `crates/core/executor/src/opts.rs`) and host-side prover memory for multi-shard proofs may pressure smaller Macs; we record RAM and watch for swapping during Phase B.
- **Groth16/Plonk locally** requires circuit artifacts and (on arm64) likely the Docker gnark path (`Dockerfile.gnark-ffi` at repo root); excluded unless explicitly wanted.
- **Patched-crate version skew:** sha2 patch tags say `sp1-6.2.0`; the v6.4.0 repo's own `patch-testing` uses them, so they're current, but we assert precompile syscall counts are non-zero in the smoke test rather than assuming.
- **stdin encoding overhead:** `io::read::<Vec<f32>>` bincode-decodes element-wise; we use `read_vec` + byte reinterpretation for operands and measure I/O explicitly (O1), so operator numbers aren't polluted.

---

## 8. Open questions

Tracked here; also posed to the user directly:

1. **Local proving budget** per full benchmark run — quick (~30 min, ~5 calibration configs) vs. thorough (2–3 h, ~12 configs incl. multi-shard)?
2. **Proof modes:** core-only calibration, or also compressed (the mode you'd actually ship/aggregate; constant size but slower)? Groth16/Plonk needed at all?
3. **Prover network data point** (real $/proof + latency; needs `NETWORK_PRIVATE_KEY` + funds) — now or later?
4. **Fixed-point variants:** is quantized (i32/i64) arithmetic a live design option for the tape, or is f32 semantics non-negotiable? (Priors say fixed-point could be ~10–100× cheaper; determines how much matrix space it deserves.)
5. **f64:** does the primitive library need f64 anywhere (e.g. accumulators for long dots)?
6. **Merkle leaf granularity for O9:** default assumption is one operator's full operand block per leaf (e.g. 16 KiB leaf for a 4096-f32 dot, hashed then opened along a depth-20/30 path). If you envision 32-byte or chunked leaves with multiple openings per check, the hash budget changes materially — which shape should be primary?
