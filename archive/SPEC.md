This is a rough overview/spec for the `veritor` library. 

# Compute Accounting Protocol

The Compute Accounting Protocol (CAP) is a procedure for recording and reporting computational workloads in a format that enables independent auditing and verification. Similar to financial accounting—where transactions are recorded in structured ledgers that auditors can inspect—CAP provides a lightly structured representation of computation that verifiers can query, replay, and audit. This enables AI developers to make verifiable claims about how they are using their compute clusters. 

## Requirements

### 1. Enable replay of selected subgraphs

The verifier must be able to:

- Select arbitrary subgraphs from the reported computation (e.g., "step 573, batch element 32")
- Bind concrete data to the inputs/outputs of those subgraphs
- Execute those subgraphs (potentially transformed for verification purposes, e.g., autoregressive → teacher-forcing)
- Compare execution results against claimed behavior

This requires:

- A representation of the full computation
- A way to identify which parts of the computation correspond to which data
- Availability of transformed versions when verification strategies require them
- Enough structure to enable slicing and data binding

### 2. Provide security for the verifier

The data exchange format must:

- Not allow the prover to inject malicious code that executes in the verifier
- Be semantically simple enough that verifiers (or verifier agents) can understand what's being claimed
- Be sanitizable (similar to SQL with parameterized queries—here, we defend against jailbreaks by avoiding plaintext code)

Counter-examples of what doesn't work:

- Raw Python code (can contain subtle exploits, jailbreaks)—though maybe raw Python could provide the verifier with supplementary information about what’s going on
- Opaque binaries (can't be inspected)

### 3. Minimize prover burden

The format must:

- Require minimal changes to existing prover code—generally requiring just additions/annotations
- Work naturally with common ML frameworks (JAX, PyTorch, etc.)
- Work naturally with common inference engines (vLLM, Triton, etc.)

The prover should ideally:

- Run their workload normally with minimal instrumentation
- Export a standard representation with straightforward tooling
- Not need deep knowledge of verification protocols

## Solution Sketch

At a high level, CAP requires the prover to maintain a **ClaimDatabase**—a queryable collection of Claims about computations that were performed. Each Claim is a self-contained, verifiable assertion about a specific computation.

### The Claim Object

A Claim packages everything needed to verify a computation. The key insight: **the graph in a Claim is a reference implementation optimized for verification**, not necessarily the actual production code. For STAMP, the prover's production system runs autoregressive inference, but the Claim contains a teacher-forcing graph that's easier to verify. The prover is responsible for ensuring these are semantically equivalent—this is testable in the prover's own CI/CD pipeline.

The `entry_point` field specifies which function in the StableHLO module should be executed during verification. Since IREE has no required naming convention (unlike C programs with main()), this field makes it explicit which function serves as the verification entry point. The default value "main" is a convention for simple single-function graphs, but multi-function graphs should specify the appropriate function name.

**Entry point requirements:**

- Must match a public function in the graph (e.g., func.func @main → entry_point="main")
- Functions marked private cannot be used as entry points
- Module name is extracted from the graph's module @module_name declaration

The `strategy` field identifies how this Claim should be verified. Different strategies impose different requirements on the Claim structure:

- **`bit_exact`**: Basic deterministic replay. Verifier re-executes the graph with provided inputs and checks for exact output match. Note: likely only works on CPU backends.
- **`stamp`**: Inference verification via locality-sensitive hashing. The graph is a teacher-forcing implementation that takes the full input-output sequence and produces both logits at each position and LSH fingerprints of hidden states (e.g. a random projection, scaled a la JL, with a known seed). The verifier checks: (1) L2 distance between computed and claimed LSH fingerprints is within tolerance, and (2) the claimed output tokens have low surprisal under the verifier's reproduced logits.
- **`training_replay`**: Training verification via checkpoint replay. The graph represents one or more training steps. Inputs include initial weights and minibatch data. Outputs include LSH fingerprints of minibatch gradients and (optionally) updated checkpoint weights. The verifier checks: (1) L2 distance between computed and claimed gradient LSH values, and (2) if replaying a full batch, L2 distance between computed and claimed updated weights.

### Verification Workflow

To verify a workload:

1. **Verifier queries** the prover's ClaimDatabase for Claims of interest
2. **For each Claim**, verifier calls `verify(claim) -> bool`
3. **The verify function**:
    - Compiles `claim.graph` using IREE
    - Executes with `claim.inputs`
    - Compares computed outputs against `claim.outputs` using strategy-specific logic
    - Returns `True` if verification passes
4. **Verifier aggregates results** to assess overall compliance

### Key Design Principles

**The Claim contains reference implementations**: The graph in a Claim is designed for verification, not production performance. It may differ significantly from the prover's actual execution (autoregressive → teacher-forcing, distributed → single-device, optimized kernels → standard ops). The prover ensures semantic equivalence through testing.

**The Claim is atomic**: Each Claim represents a single verifiable unit. The prover decides granularity—one Claim per inference request, per training step, per batch. Claims are self-contained with no external dependencies.

**The prover structures Claims for strategies**: Provers must understand available strategies to generate appropriate Claims. Supporting a new strategy means creating reference implementations that produce the required outputs (LSH fingerprints, logits, etc.).

**The verifier is deterministic**: No AI agents, no transformation inference, no binding heuristics in v0. The verifier compiles and executes StableHLO, then applies well-defined comparison logic.

**Transport agnostic**: The protocol specifies Claim structure, not storage or transmission. Provers can use databases, files, APIs—whatever fits their infrastructure.

## Current Thoughts on Implementation (v0)

### Division of Responsibilities

**Prover provides:**

- **ClaimDatabase**: A collection of Claims with query support (by time range, strategy type, computation ID)
- **Reference implementations**: For each computation type and strategy, the prover creates StableHLO graphs optimized for verification:
- **Claim generation**: After executing production workload, prover creates corresponding Claim with reference graph

**Verifier implements:**

- **verify(claim: Claim) -> bool**: Core verification using IREE
- **Strategy checkers**: One checker per supported strategy with specific comparison logic
- **Sampling logic**: Decide which Claims to verify and how to aggregate results

### Why This Approach for v0

**Separates production from verification**: Provers optimize production code for performance (distributed, quantized, fused kernels). Verification code prioritizes reproducibility (single-device, standard precision, explicit operations). This separation lets each optimize for its purpose.

**Prover controls equivalence**: Only the prover knows how to create semantically equivalent reference implementations. They test this equivalence before deployment. Verifier just checks the reference matches claimed behavior.

**Deterministic verification**: IREE on CPU backends provides reproducible execution. Strategy-specific checkers use clear numerical tolerances. No ambiguity about what "verification passed" means.

**Practical for real deployments**: Production inference engines (vLLM, TensorRT-LLM) use complex optimizations that are hard to verify directly. Reference implementations in standard StableHLO are much more tractable.

### What v0 Punts On

**Trace integration**: How should Claims incorporate execution traces showing device placement, sharding decisions, and timing? This is necessary for protocols like memory spot checks that verify compute actually happened on claimed hardware. v0 Claims are stateless—they don't capture "when" or "where" computation occurred, only "what" was computed.

**Device API**: Related to traces, we need a way to represent device topology, memory capacities, and network characteristics. Memory verification protocols require knowing device capabilities to validate that responses are feasible given hardware constraints.

**Claim granularity standards**: How should workloads be divided into Claims? One per request? Per batch? Per training step? v0 leaves this to prover discretion.

**Strategy standardization**: What strategies should exist and what are their precise specifications? v0 implements examples but needs formal registry with schemas.

**Security against malicious provers**: v0 assumes cooperative provers. No cryptographic commitments, no challenge-response, no trusted execution. Production needs mechanisms to detect provers providing fake Claims or cherry-picking which to expose.

**Large tensor handling**: For big models, storing full tensors in Claims is impractical. Future versions might use content-addressed storage with hashes in Claims, or streaming verification.

**Cross-hardware determinism**: v0 uses CPU for deterministic verification, but production runs on GPUs. How do we handle legitimate floating-point differences? Tolerances in strategy checkers may need hardware-specific tuning.

### Open Questions for v0

**Extending to traces**: The most important open question is how to extend Claims to include execution traces. Options:

1. Add optional `trace: ExecutionTrace` field to Claim with device IDs, timestamps, memory measurements
2. Separate Trace objects linked to Claims by ID
3. Embed trace data in `metadata` field (unstructured for v0)

This affects verification protocols that need to know compute provenance, not just computational results.

**Device representation**: Related to traces—how do we represent device topology? Need some `Device` API that captures:

- Device capabilities (memory, compute)
- Network topology (which devices can communicate, bandwidth/latency)
- Used for memory spot check verification and distributed training validation

**Claim serialization**: Should Claims use JSON, Protocol Buffers, custom format? NumPy arrays don't serialize to JSON cleanly—need binary encoding or separate files for tensors.

**Entry point conventions**: Must graphs have `@main` function? Or should Claims include `entry_point: str`? v0 assumes `@main` exists.

**Error handling**: Should verify() return bool or richer `VerificationResult` with status and diagnostic info?

- Maybe “verificationtest” is a bad name… clashes with our pytests… idk not important
- Probably should serialize/deserialize the database as it gets passed from prover → verifier (so that we can stand this up as a server)
- Right now the VerificationTest is tightly coupled with the Claim — e.g. the Claim’s graph is just whatever the verifier needs to run to verify the claim, so in the case of inference it will be teacher-forcing instead of decode. These should be decoupled — so you should be able to attach something like a VerificationMechanism or something, which is some pre-determined contract between prover/verifier that allows the verifier to check that claim… And maybe you can have multiple? Not sure then what the top-level attributes of Claim are then… maybe just the data, and then each VerificationMechanism knows how to consume that data? hmm
- Is there really no clean way to assign nice stable IDs to each operator in the prover’s graphs—like actually include those IDs (or some sort of generator…) in the IR itself?
    - [Dex](https://github.com/google-research/dex-lang?tab=readme-ov-file) seems to represent complex control flow in a format that enables addressable subgraphs
        - But it’s an abandoned research project…