# Lean collaborator brief: private optimized macros over public instructions

## The problem

Assume we already have a public registry of low-level instructions. Each
instruction has:

- a public input/output relation;
- a proof verifier or a constraint system whose accepted proofs are intended
  to establish that relation; and
- a public identity, version, and security assumptions.

We want a lab to take many of these public instructions and build a much larger
macro-circuit—for example, a fused attention block or a tensor-core inner
product—with a proof system specialized to that composition and to the lab's
hardware.

The specialized prover may exploit facts unavailable to a generic VM:

- the same opcode is repeated thousands of times;
- operands are reused;
- memory can be laid out specifically for a GPU;
- several primitive steps can share range checks or lookup arguments; and
- the whole computation can be represented by one custom AIR or circuit rather
  than thousands of generic CPU instructions.

The desired outcome is:

> A verifier that already trusts the public low-level instruction set can
> accept a proof from the optimized macro without rechecking every primitive
> proof and without blindly trusting the lab's custom proof system.

Ideally the lab can keep at least its prover implementation and hardware
optimization private. A stronger version would also keep the macro's internal
circuit topology private.

This note is only about the design space for that composition and
certification problem. It does not cover the surrounding compiler or sampled
execution protocol.

## First distinction: what is supposed to remain private?

There are three materially different goals:

1. **Private witness.** Inputs and intermediate values are hidden, but the
   macro circuit and verifier are public. Ordinary zero-knowledge systems
   already target this.
2. **Private implementation.** The macro relation, AIR, or verifier is public,
   but the prover implementation, CUDA kernel, scheduling, and other
   optimizations are proprietary. This is compatible with current custom-chip
   systems and is the easiest useful target.
3. **Private circuit topology.** The verifier learns only a digest and public
   interface, not which primitives were composed or how. This requires a
   universal circuit/interpreter, a zero-knowledge registration proof, or a
   trusted certifier. It is substantially harder.

An entirely private proof system is not enough. If an external party is to
trust its proofs, the verification relation and soundness-critical parameters
must be public, formally certified, trusted by governance, or recursively
wrapped by a proof system whose soundness is already accepted. Keeping only the
prover implementation private is straightforward; keeping the verifier and
its security argument private is not.

## Why generic instruction-by-instruction proving is not enough

We implemented the same fp8 E4M3 tensor-core inner product as a custom
instruction in both SP1 and OpenVM and measured it on one RTX 4090.

| Configuration | Observed GPU proving result | What it shows |
|---|---:|---|
| One TC_DOT per proof (`k=1`) | roughly 0.4–0.9 s/proof; only 4k–39k MAC/s in singleton bracketing observations | Fixed proof and CUDA startup dominate a single instruction. |
| SP1 guest-code implementation | custom precompile was 14.6× faster in the prior SP1 campaign | A generic CPU instruction expansion leaves large performance on the table. |
| SP1 batched TC_DOT precompile | 0.95–0.98M MAC/s; required `k=2,048–16,384` and 35–69 s proofs to plateau | Batching amortizes the floor, but latency is high. |
| OpenVM batched TC_DOT custom chip | 1.15–1.20M MAC/s; `k=64–256`, about 21–22% faster than SP1 | A modular custom chip can outperform the forked SP1 precompile and reaches steady state at much smaller batches. |
| OpenVM `TC_MATMUL_4X4X16` macro | 1.55–2.52M MAC/s; 1.35–2.10× OpenVM TC_DOT per MAC | A circuit specialized to operand reuse creates another large gain that batching independent dots cannot recover. |

The independent-dot OpenVM result still has about `1.4×10^8` proving overhead
relative to the 4090's native fp8 tensor-core throughput; SP1 is about
`1.7×10^8`. The operand-reuse macro lowers the best observed ratio to roughly
`7.5×10^7`. Native throughput is only a performance denominator—the hardware
operation is not a bit-exact semantic oracle for our synthetic
Ampere-GroupSum/E4M3 contract.

The practical lesson is not merely "batch proofs." It is:

> Developers need a way to replace a large composition of public low-level
> instructions with a certified macro whose proof system is specialized to
> that composition.

## The core formal object

At a high level, a registered macro needs to bind:

- a public macro identifier and version;
- its input and output interface;
- either a public primitive expansion or a commitment to one;
- its public denotation, if that is not defined solely by the expansion;
- the custom constraint system, verifier, or verification-key digest;
- the backend and cryptographic assumptions under which proofs are accepted;
- a refinement claim connecting the macro to the public primitives; and
- any public cost, leaf-count, or sampling metadata.

The logical goal is:

$$
\mathsf{BackendSound}
\;\land\;
\mathsf{MacroConstraintsRefinePrimitiveExpansion}
\;\Longrightarrow\;
\mathsf{AcceptedMacroProofHasPrimitiveSemantics}.
$$

A rough Lean interface might look like:

~~~lean
structure CertifiedMacro where
  macroId       : Digest
  backendId     : BackendId
  inputType     : Interface
  outputType    : Interface
  denotation    : Inputs inputType → Outputs outputType
  expansionRoot : Digest
  verify        : Proof → Inputs inputType → Outputs outputType → Bool

  verifier_sound :
    ∀ proof input output,
      verify proof input output = true →
      output = denotation input

  refines_public_instructions :
    ∀ input,
      denotation input =
      evalCommittedPrimitiveExpansion expansionRoot input
~~~

This is only a sketch. In particular, `verifier_sound` will usually depend on
explicit cryptographic assumptions rather than being an unconditional theorem
inside Lean.

## Option 1: one common proof backend with public custom chips

The macro is implemented as a custom AIR, circuit, or precompile inside one
public proof system. The macro's constraints and verifier are public; the
optimized prover and hardware implementation can remain private.

Lean's role is to prove:

1. satisfying the custom constraints implies the macro's public denotation;
2. the macro denotation equals the composition of public primitive
   instructions; and
3. the opcode, bus, memory, and public-input bindings prevent proving the right
   arithmetic under the wrong instruction identity or inputs.

**Advantages**

- Best performance and lowest runtime complexity.
- No need to compose heterogeneous cryptographic proof systems.
- One recursive aggregation mechanism can combine primitive and macro chips.
- The proprietary value—the prover, CUDA kernel, witness generation, and
  scheduling—can remain private.

**Limitations**

- The macro circuit/AIR is public, so this does not hide topology.
- Every custom chip must fit the common backend's algebra and commitment
  scheme.
- A formal semantic proof is required for each registered macro.

**SP1/OpenVM evidence**

- SP1 supports custom precompiles, but our implementation required a fork
  touching the executor, machine, and GPU server.
- OpenVM supported the same work as an out-of-tree extension with custom CPU
  and CUDA trace generation.
- In paired warm measurements, OpenVM TC_DOT achieved `1.220×` and `1.205×`
  SP1 throughput at `n=4096` and `n=16384`.
- We deterministically extracted the complete OpenVM AIR—1,393 columns and
  4,448 constraints—into Lean. Lean elaborated the 3.2 MB extraction in 162 s
  using about 3.4 GB peak memory, and an opcode/ProgramBus binding theorem,
  axiom audit, and hygiene checks passed.
- We have not yet proved the entire extracted arithmetic AIR semantically
  correct in Lean.

**Assessment:** This is the strongest near-term option and the one our current
results support. OpenVM is the better base than SP1 for this architecture.

## Option 2: publish a primitive expansion, then certify an optimized macro

The developer publishes the canonical low-level expansion that defines the
macro's meaning, plus a separate optimized AIR or circuit. Lean proves once
that the optimized representation refines the expansion.

At runtime, only the optimized macro proof is produced. The primitive
expansion is a specification and certification artifact, not something the
prover executes instruction by instruction.

**Advantages**

- Cleanest semantic story: the public instruction composition remains the
  source of truth.
- Lean checks an ordinary refinement theorem.
- Runtime performance can equal a hand-written custom chip.
- The optimized prover implementation remains private.

**Limitations**

- The expansion and optimized circuit topology are normally public.
- Huge expansions need a compressed or committed representation.
- Certification must bind exact versions of the expansion, AIR, verifier,
  interface, and backend.

**Assessment:** This should probably be the default certification layer on top
of Option 1. It says precisely what a custom chip is allowed to replace.

## Option 3: committed private macro plus one-time certification

The developer commits to a private macro circuit `M` with digest `d`. A
registration process establishes that `M` is a composition of allowed public
instructions and that its runtime verifier proves the same relation. Repeated
execution proofs refer only to `d`.

There are two ways to perform registration:

- a zero-knowledge proof of refinement checked by a universal public verifier;
  or
- an offline Lean proof/audit whose result registers `d`.

**Advantages**

- Can hide circuit topology after registration.
- Expensive certification is paid once; repeated runtime proofs can use the
  optimized system.
- The public registry can expose only interface, cost, metadata, and digests.

**Hard problem**

Proving that an *arbitrary proof system is sound* is not a normal finite
zero-knowledge statement. Soundness quantifies over adversaries and generally
depends on cryptographic assumptions. A registration proof can show that a
private circuit was correctly compiled into a **previously certified backend
family**; it cannot magically certify an unrestricted new proof system.

Therefore this option should restrict macros to:

- one already-certified backend;
- a small family of verified transformations; or
- a custom verifier whose soundness has been established separately in Lean
  and whose exact digest is registered.

**Assessment:** This is the most direct route if circuit topology truly must
remain private, but it is a second phase. The tractable version certifies
private circuits within a public proof-system family, not arbitrary private
proof systems.

## Option 4: universal circuit or zkVM interprets the private macro

Instead of registering a custom circuit, a public universal circuit or zkVM
interprets committed macro bytecode. The proof can hide the bytecode and
witness while showing that the committed program produced the claimed output.

**Advantages**

- Naturally supports private program topology.
- One public verifier and one backend soundness theorem.
- Arbitrary macros require no new trusted verifier.

**Limitations**

- Pays interpretation overhead and loses many circuit-specific optimizations.
- Reintroduces the generic-instruction performance problem demonstrated by our
  benchmarks.
- Adding high-performance private macros eventually requires JIT
  certification or custom chips, returning to Options 2 or 3.

**Assessment:** Best generic fallback for uncommon or small private macros, not
the likely steady-state solution for large repeated inference kernels.

## Option 5: recursively aggregate proofs of public primitives

Each primitive or public subcircuit produces its own proof. A recursive circuit
verifies and aggregates those proofs into one macro proof. Zero knowledge can
hide witnesses and potentially the aggregation structure.

**Advantages**

- Strong modularity: each primitive proof system can be reused unchanged.
- Composition soundness is conceptually direct if every component verifier is
  already trusted.
- Does not require proving that a fused AIR implements all primitive
  arithmetic.

**Limitations**

- One proof per low-level instruction is completely impractical.
- Recursion only becomes plausible when each child proof already covers a
  large batch or subcircuit.
- Heterogeneous fields, commitments, and verifier circuits create substantial
  interoperability costs.
- It cannot exploit shared lookups, memory arguments, or operand reuse as
  effectively as a fused macro.

**Assessment:** Useful for composing already-large certified blocks, not for
the leaves. The batching measurements are strong evidence against using this
at instruction granularity.

## Option 6: arbitrary heterogeneous proof systems wrapped by one outer proof

A lab invents its own proof system. A common outer SNARK or zkVM proves that
the lab's verifier accepted.

This establishes only:

> The inner verifier accepted.

It does **not** establish:

> The inner verifier is sound for the claimed public-instruction composition.

That second statement still requires:

- a Lean proof of the inner verifier and proof system;
- an explicit accepted cryptographic assumption; or
- governance that registers the verifier as trusted.

The outer proof can hide the inner proof and amortize verification, but it
cannot repair an unsound inner system.

**Assessment:** Maximum flexibility, maximum formal and cryptographic burden.
This should not be the default. It is reasonable only for a small number of
important proof systems with serious independent certification.

## Option 7: trusted certification or governance registry

A trusted committee, auditor, or TEE inspects a private macro and registers its
digest, interface, verifier, and refinement claim. Runtime proofs are then
checked against the registry.

**Advantages**

- Supports private topology and highly optimized proprietary systems.
- Far simpler than a zero-knowledge proof of circuit refinement.
- Certification cost is one-time.

**Limitations**

- Introduces an explicit trust and governance assumption.
- Updates, revocation, and versioning become security-critical.
- Does not provide the end-to-end formal assurance we ultimately want.

**Assessment:** A plausible deployment bridge, but not the clean research
answer.

## Comparison

| Approach | Circuit topology private? | Prover implementation private? | Expected performance | Main Lean burden | Recommendation |
|---|---|---|---|---|---|
| Common backend + public custom chip | No | Yes | Best | Prove chip constraints and primitive refinement | **Recommended now** |
| Public primitive expansion + optimized chip | No | Yes | Best | Prove optimized chip equals public expansion | **Recommended certification model** |
| Committed private macro + registration proof | Yes | Yes | Potentially best | Certify a hidden macro inside a restricted public backend family | **Promising second phase** |
| Universal zkVM/interpreter | Yes | Yes | Generic but slower | Prove interpreter/backend once | Useful fallback |
| Recursive primitive-proof aggregation | Possibly | Yes | Poor at leaf granularity | Modular verifier-composition theorem | Use only for large blocks |
| Arbitrary proof system + outer wrapper | Possibly | Yes | Variable | Prove or assume each inner system's soundness | Avoid by default |
| Trusted certification registry | Yes | Yes | Best | Little formal work; governance replaces it | Deployment bridge only |

## Recommended architecture

The best current design is a two-level system.

### Level 1: common public backend

Use OpenVM as the common proof backend. Register public primitive instructions
and permit out-of-tree custom chips/macros. A macro registration binds:

- primitive-expansion digest;
- macro interface and public denotation;
- AIR/circuit and verification-key digests;
- OpenVM and proof-backend versions;
- a Lean refinement theorem or certificate;
- cost and sampling summaries; and
- cryptographic assumptions.

The lab may keep its witness generator, CUDA kernels, scheduling, and prover
optimizations private. Runtime proofs use the custom chip and can be aggregated
with ordinary OpenVM execution.

### Level 2: optional circuit privacy

If hiding topology is important, add a separate registration layer:

1. commit to the private macro;
2. prove or attest that it is an allowed composition within the already
   certified OpenVM/AIR family;
3. register the macro and verification-key digests; and
4. use the fast specialized proof system for repeated executions.

Do not initially allow arbitrary private proof systems. First solve private
circuits within one public backend family. Heterogeneous proof systems can be
added later through explicit, separately certified verifier adapters.

## What I would like help with in Lean

The useful Lean questions are narrowly about certification and composition:

1. What should the formal interface for a `CertifiedInstruction` and
   `CertifiedMacro` be?
2. How should we separate:
   - denotational refinement;
   - constraint-system soundness;
   - opcode/public-input binding; and
   - assumed cryptographic soundness?
3. Can we prove one generic theorem saying that backend soundness plus macro
   refinement yields sound replacement of a public primitive composition?
4. Can this theorem compose recursively, so a certified macro may invoke other
   certified macros?
5. What exact information must remain public if the expansion or topology is
   hidden behind a digest?
6. Is there a workable proof-carrying registration object whose checking does
   not reveal the private expansion?
7. How should we connect extracted AIR constraints to hand-written semantic
   refinement proofs modularly, rather than proving one monolithic theorem?
8. Which parts of proof-system soundness can realistically be established in
   Lean, and which should appear as named cryptographic assumptions?

The most useful concrete case study is the existing OpenVM TC_DOT chip:

- the public specification is a composition of low-level fp8 decode,
  multiplication, GroupSum, normalization, packing, and memory operations;
- the optimized implementation is one custom CPU/GPU AIR chip;
- the complete AIR already extracts deterministically into Lean; and
- the missing formal step is proving that satisfying that extracted AIR implies
  exactly the public composed semantics.

That case is small enough to study, but it exercises the same architecture we
would need for private, hardware-optimized inference macros.

