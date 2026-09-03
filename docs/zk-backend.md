# The proof layer: obligations, batches, backends

The protocol's reveal step used to be *transparent*: for every sampled verification unit (VU) the prover opened the values it reads and writes (Merkle openings) and the verifier recomputed the VU's relation in Python. The paper replaces that reveal with zero-knowledge proofs. This document describes the pluggable, batchable proof layer that now sits behind the reveal step (`src/veritor/protocol/proofs/`), the SP1 zkVM backend that implements it (`zk/sp1/`), the measured costs (`proofs/costs.py`) and what a production OpenVM backend would change.

Terminology as in the README: a *replay unit* (RU) is a coarse unit whose interior the prover commits when it is sampled at rate `q`; a *verification unit* (VU) refines an RU and is checked gate by gate when sampled at rate `s`; a VU is a copy of a definition, its *kind*.

## 1. Architecture

### 1.1 Obligation: the public statement for one sampled VU

`Obligation` (`proofs/statement.py`) is what the verifier demands for one sampled VU and nothing more:

- `session`: the header digest (binds session id, `H(C, I)`, policy, public I/O, `kappa_W`, and the backend id); `compiled`: `H(C, I)` again, explicitly.
- `unit`, `replay_unit`: the VU index and the RU it lies in.
- `kind`: the digest of the definition the VU is a copy of; its relation must hold.
- `commitments`: every commitment the relation opens against, as `(owner, domain_id, root, count)`; owners are `-2` (weights, `kappa_W`), `-1` (boundary) and RU indices (interiors).
- `positions`: every coordinate the relation touches, `(commitment, rank, position, schema[, expected])`, in address order; `expected` pins the public input an `in` gate must hold.
- `inputs[k]`, `gates[j]`: which position slot is the kind's `k`-th read port and the copy's `j`-th gate.

No value appears in an obligation: the values are the *witness*. A `KindProgram` gives the kind's relation once in copy-relative coordinates (gate `j` is `op(arg, arg)` where an argument is `("port", k)` or `("local", i < j)`), so one program serves every copy of the kind.

### 1.2 Statement: a batch

A `Statement` is the public statement of one proof: the gate set (`id`, digest, word width), the programs of every kind that occurs (sorted by digest) and the obligations (sorted by `(session, compiled, unit)`). Sorting makes a batch a *set* and its encoding unique. `wire.py` gives the canonical bytes:

~~~text
Statement   = MAGIC str gate_set_id digest gate_set_digest u32 width
              list<KindProgram> kinds  list<Obligation> obligations
KindProgram = digest kind u32 size list<u32> ports list<GateOp> gates
GateOp      = str op list<Arg> args ;  Arg = u8 space (0 port, 1 local) u32 value
Obligation  = digest session digest compiled u64 unit u64 replay_unit digest kind
              list<CommitmentRef> list<PositionRef> list<u32> inputs list<u32> gates
Witness     = MAGIC list< list< bytes value list<digest> path > >
~~~

(big-endian integers, `u32` length prefixes, raw 32-byte digests; `decode_statement` accepts exactly the outputs of `encode_statement`). `statement_digest = sha256(encode_statement(s))` is the public value a proof commits to. The Rust mirror is `zk/sp1/common/src/codec.rs`; `zk/sp1/common/tests/vectors.json`, generated from Python by `tests/veritor/protocol/proofs/test_vectors.py`, pins both to each other.

### 1.3 Backends and coverage

~~~python
class ProofBackend(Protocol):
    backend_id: str                                    # bound into the header
    def prove(self, statement: Statement, witness: Witness) -> bytes: ...
    def verify(self, statement: Statement, proof: bytes) -> bool: ...
~~~

The header carries `backend`; `resolve_backend` refuses a backend whose id differs from the header's, so both parties run the same one. The default `"transparent"` leaves the header manifest and digest exactly as before (the field is omitted when default), so existing transcripts are byte-identical.

Batching is the prover's choice. A `BatchPlan` partitions the sampled VUs (in challenge order) into groups; `prove_plan` builds one `Statement` per group and asks the backend for one proof each, producing `ProofMessage(units, proof, foreign)` entries in `EvidenceMessage.proofs`. A group may be joined with a `ForeignBatch` (other sessions' obligations and their witness) so one proof spans sessions; `foreign` then carries the canonical statement of those obligations (public, no values).

The verifier never trusts the prover's list. `VerifierSession.receive_evidence` derives the obligations itself from the challenge, the Index and the commitments it accepted (`derive_obligations`), then `check_coverage` requires that the proofs' `units` cover `range(len(demanded))` exactly once (`COVERAGE_MISMATCH` otherwise), decodes any foreign statement (`MALFORMED_TRANSCRIPT` if not canonical; `COVERAGE_MISMATCH` if a "foreign" obligation claims this session), recomputes each proof's statement by merging its own demands with the foreign ones, enforces `max_proof_bytes`, and calls `backend.verify` (`PROOF_REJECTED` on failure).

### 1.4 The backends

- `TransparentBackend` (`proofs/transparent.py`): the reference. The "proof" is the encoded witness; `verify` re-derives every leaf and node hash with the exact framing of `merkle.py`, folds each path to its root, checks public inputs and decodes values, then interprets each kind's program with the gate set's pinned semantics (`GateSet.evaluate`). Under the transparent id the session still sends `EvidenceMessage.units` (one opening batch per sampled VU), and the verifier checks each as a one-obligation statement through this backend, so the wire format is unchanged.
- `SP1Backend` (`proofs/sp1.py`): drives `zk/sp1/target/release/veritor-zk-host` over a subprocess protocol (one JSON object on stdout per command): `info`, `execute --statement S --witness W` (the SP1 executor, no proof, exact cycle counts per phase), `prove ... --out P --mode core|compressed`, `verify --proof P --statement S`. Builds the host on demand; skips cleanly when `cargo prove` is absent.
- `OpenVMBackend` (`proofs/openvm.py`): the same subprocess protocol against a remote-built OpenVM host (section 5).

### 1.5 The SP1 guest

`zk/sp1/guest` is a *generic checker* for the protocol, not a program per workload. It reads the statement and witness bytes (`read_vec` twice), hashes the statement with the SP1 `sha2` precompile (workspace-wide `[patch.crates-io]` on `sp1-patches/RustCrypto-hashes`, as in `sp1-op-bench`), parses both strictly (`codec.rs`), resolves the gate set from `(id, width)` and checks its digest against the pinned manifest (`gateset.rs`, a mirror of `veritor/core/gates.py` including the canonical JSON manifest and `tagged_sha256`), authenticates every opening with our exact framing (`frame.rs`: `_FRAME`, 4-byte tag length, 8-byte part lengths, `domain_id`, `leaf(rank, position, schema, value)`, `node(level, index, left, right)`), checks public inputs, decodes values, interprets every kind program with the modular semantics (`check.rs`), and commits `sha256(statement) || verdict` (33 bytes) as public values. Phases are bracketed with `cycle-tracker-report-start/end` (`io`, `digest`, `parse`, `merkle`, `gates`). A rejected batch still produces a valid execution (verdict `0`), so a proof of a false statement is impossible to pass off: `verify` requires `ok && statement_match && verdict`.

## 2. How batching amortizes

Proving has a fixed floor per proof and a marginal rate per cycle. sp1-op-bench measured the curve for its 2.6M-cycle "leaf check" (SP1 6.4.0, core proofs, 15-core cloud CPU pod):

| checks per proof | cycles | prove s | s per check | marginal s per check | peak RSS |
|---:|---:|---:|---:|---:|---:|
| 1 | 2,595,299 | 63.45 | 63.5 | — | 20.4 GB |
| 4 | 10,395,153 | 153.78 | 38.4 | 30.1 | 38.4 GB |
| 16 | 41,577,643 | 450.14 | 28.1 | 24.7 | 83.8 GB |

The marginal cost converges to ~105 kHz (the Apple Silicon laptop does 110–118 kHz asymptotically with a ~10 s floor). Batching amortizes the floor, not the linear term: `proofs/costs.py` linearizes each machine as `seconds = floor + cycles / kHz` (`MachineProfile`), and `sp1_cost_parameters(depth, batch_size)` prices the paper's `c_0` as `(per-obligation cycles + per-batch cycles / k) / kHz + floor / k`. Memory grows with the batch (roughly linearly in cycles; ~84 GB at 41.6M cycles), which caps the batch a given prover can take; compressed mode adds ~1.3× time at that size for a constant 1.27 MB proof and is the right shape when many checks share one artifact.

Cross-session batching (`ForeignBatch`) exists for the same reason: an operator proving many small sessions amortizes one floor over all of them.

## 3. Measured cycles for our obligations

All numbers are exact cycle counts from `veritor-zk-host execute` (SP1 6.4.0; `zk/sp1/bench/measure.py`, synthetic but valid batches: real Merkle trees with our framing, real gate semantics; two-point slopes).

| constant | cycles | how |
|---|---:|---|
| `add` gate (relation check) | 116 | `gates` slope, 8 vs 136 gates of the op in one kind |
| `eq` | 116 | same |
| `lt` | 159 | same |
| `mul` | 162 | same |
| `shr` | 184 | same |
| `sub` | 212 | same |
| `in`, `weight` | 0 | no relation; `in` adds one 2-byte compare to `merkle` |
| Merkle level (`node` frame) | 2,694 | `merkle` slope, depth 5 vs 13 at 18 positions; 3 `SHA_COMPRESS` per level |
| opened position (leaf frame + checks + decode) | 3,419 | `merkle` slope per position at depth 8 (24,971) minus 8 levels |
| parse, per input byte | 11.7 | `parse` slope vs statement + witness bytes |
| statement digest, per byte | 3.53 | `digest` slope (~226 cycles per 64-byte block) |
| per obligation, fixed | ~5,800 | residual of the 1-vs-65 obligation fit |
| per batch, fixed | ~50,000 | floor of a 1-obligation batch (94,647 total, 48,583 untracked) |

Two real batches, both verdict `true`:

| batch | obligations | positions | Merkle levels | gates | total cycles | `merkle` | `parse` | `gates` | `digest` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| matmul fixture, every VU | 16 | 52 | 152 | 28 | 879,613 | 558,796 | 196,676 | 10,819 | 23,593 |
| small cluster run, `q = 1/2, s = 1/3` | 162 | 1,506 | 14,771 | 789 | 52,867,791 | 45,111,487 | 5,934,508 | 203,675 | 327,335 |

`estimate_cycles` predicts 892k and 53.2M for these (+1.4%, +0.5%). The picture is unambiguous: with the toy ISA every scalar is its own committed leaf, so **Merkle authentication is 64–85% of the cycles and the relation itself is ~1%**. A position in a depth-`d` domain costs `3,419 + 2,694 d` cycles; the gate whose output it is costs 116–212. Three notes on what moves these numbers:

- Our `leaf`/`node` frames are 3 SHA-256 blocks (the framing adds ~115 bytes to a 64-byte `left || right`). sp1-op-bench measured 1,620 cycles for the 2-block bare hash; we are on the same ~800-cycles-per-block line, of which ~240 is the precompile and the rest is the `sha2` crate's per-call init/finalize. A hand-rolled SHA-256 over the `syscall_sha256_extend/compress` pair would bring a level to ~1,000 cycles.
- The parser allocates a `Vec` per list; ~11.7 cycles/byte is mostly that. Zero-copy parsing would remove most of `parse` (11% of the cluster batch).
- The relation is cheap because the guest interprets small programs; a kind with thousands of gates per opened value (a dot product over a committed *tile*, as in sp1-op-bench's leaf check: 16 cycles per fixed-point MAC, 32 KiB leaf) amortizes one leaf over many operations, which is exactly why the paper's ML workloads are feasible and the toy ISA's per-scalar commitment is not the shape to scale.

### The one real proof

`tests/veritor/protocol/proofs/test_sp1.py::test_one_real_core_proof_of_a_small_batch_round_trips` runs the whole protocol on the matmul fixture with `SP1Backend(mode="core")`: 16 obligations in one batch, 879,613 cycles, **2 shards, 2,806,928-byte core proof, prove 16.5–18.1 s, verify 0.07–0.13 s, setup 0.7–0.85 s** on this Apple Silicon laptop (31.8 s wall for the host process, 4.2 GiB peak RSS). The proof verifies against the verifier's recomputed statement, fails `statement_match` against any other statement, and the Python `verify` returns `False` for both a swapped statement and non-proof bytes. The laptop profile in `costs.py` (10 s floor + 112 kHz) predicts 18.0 s for this batch.

## 4. `alpha`: proving cost over native cost

`alpha_dot(backend)` returns native MAC/s over proving MAC/s for the dot relations of the ML workloads, from `openvm-tc-bench/docs/backend-decision.md` (RTX 4090, warm paired medians, OpenVM v2.0.2 vs a TC_DOT-patched SP1 6.4.0, same GPU):

| backend | proving MAC/s | `alpha` (native 1.8e14 MAC/s) | provenance |
|---|---:|---:|---|
| OpenVM `TC_MATMUL_4X4X16`, n=4096, k=64 | 2,519,884 | 7.1e7 | report's "~7.5e7 at the matmul plateau" |
| OpenVM `TC_DOT`, n=4096, k=256 | 1,198,237 | 1.5e8 | |
| SP1 `TC_DOT` fork, n=4096, k=16384 | 982,066 | 1.8e8 | report's "prior SP1 headline ~1.8e8" |

The native rate `1.8e14` fp8 MAC/s is the one the report's two `alpha` figures imply (7.5e7 × 2.52e6 ≈ 1.9e14; 1.8e8 × 0.98e6 ≈ 1.8e14), about 55% of the 4090's 3.3e14 dense fp8 peak; using the peak multiplies every `alpha` by 1.83. For the toy ISA, `alpha_toy_isa(statement)` (prover seconds over one 3 GHz ALU cycle per gate) is 2–3e9 for the real batches above: three orders of magnitude worse than the tensor-core relations, entirely because of per-scalar Merkle authentication.

## 5. What a production OpenVM backend changes

`openvm-tc-bench` recommends OpenVM: 1.2× the matched TC_DOT throughput of the SP1 fork, 3.8× with `TC_MATMUL`, plateau at k=64–256 (1.4–2 s proofs) instead of k=2,048–16,384, an extension fully out of tree against unmodified OpenVM v2.0.2, and a constraint system that extracts to Lean. GPU proving is remote-only, so this repo holds an *adapter*, `OpenVMBackend`, tested for the wire mapping (`tests/veritor/protocol/proofs/test_costs_and_openvm.py`):

- **Inputs.** The OpenVM guest reads one hint-stream item per `read_vec`; ours are the two canonical encodings above, byte for byte what the SP1 guest consumes (`openvm_input(statement, witness).hint_stream`). `codec.rs` is the parser to port; the guest crate is `no_std`.
- **Public values.** OpenVM reveals `u32` words by index (`reveal_u32`): `sha256(statement)` as eight little-endian words at indices 0..8 and the verdict at index 8; `reveals_to_public_values` re-chunks them into the same 33 bytes SP1 commits, so `verify` is unchanged.
- **Host protocol.** The same `info | execute | prove | verify` JSON protocol as `veritor-zk-host`; `OpenVMBackend(host=Path(...))` differs from `SP1Backend` only in the binary (no local build).
- **The relation.** What OpenVM adds is a *tile kind*: a definition whose gates are `TC_DOT` / `TC_MATMUL_4X4X16` groups over fp8 e4m3 operands under the pinned `hawkeye_ampere_groupsum_fp8e4m3_v0` contract (the `tc-dot-batch` guest's `u32 dot_count, (u32 acc, u32 tile_count, tiles)*` payload is the witness shape for one such kind). Its positions are tiles, not scalars, so one leaf covers thousands of MACs and the Merkle share of section 3 collapses. That waits on a gate set carrying the fp8 contract; nothing in the proof layer presumes it (a kind program is `op(args)` over ports and locals; the guest's `GateSet::resolve` is where a new set with its digest would be registered).

## 6. Running it

~~~sh
uv sync
.venv/bin/pytest -q tests/veritor/protocol/proofs -m "not zk"     # codec, coverage, cross-session, costs, OpenVM mapping
.venv/bin/pytest -q tests/veritor/protocol/proofs -m zk -s        # builds the guest; execute on real obligations; one real core proof (~80 s)
VERITOR_WRITE_VECTORS=1 .venv/bin/pytest -q tests/veritor/protocol/proofs/test_vectors.py   # regenerate cross-language vectors
(cd zk/sp1 && cargo test -p veritor-zk-common)                    # Rust unit tests + the vectors
.venv/bin/python zk/sp1/bench/measure.py --json /tmp/measure.json # the cycle table
~~~

Toolchain: `rust-toolchain.toml` pins stable Rust; `sp1-sdk`, `sp1-zkvm` and `sp1-build` are pinned to `=6.4.0`; the `sha2` patch tag is `patch-sha2-0.10.9-sp1-6.2.0`. `~/.sp1/bin/cargo-prove` must be installed for the guest build. Keep local proofs under ~10M cycles (the 162-obligation cluster batch at 53M cycles is an `execute`-only object here).

## 7. Open items

- Cheaper hashing in the guest (hand-rolled SHA-256 over the syscalls; ~2.5× fewer cycles on the dominant term) and a zero-copy parser.
- `verify` in the host constructs the CPU prover to derive the verifying key (~10 s); caching the vk or a standalone verifier would make verification sub-second end to end.
- Compressed-mode proving is wired (`--mode compressed`) but not measured here.
- The OpenVM host binary and the fp8 tile kind (section 5).
