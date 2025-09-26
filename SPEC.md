# Design goals

- Make it easy for a compute cluster to report its workloads in a format that can be easily and securely verified
- Support interactive protocols that allow for better workload verification

# Roles

**Prover**. Performs computations and reports them. 

**Verifier**. Checks reported computations. 

# Workload database: data models

We capture information about the execution of ML workloads in a database managed by the prover and rendered accessible to the verifier. The design separates the computation itself from its execution history and concrete data, creating a queryable database of the prover's activities that verifiers can replay and validate at different granularities.

## **Computational graph**

- **Definition:** The specification of a computation that was performed, expressed as a graph of operations and their dependencies.
- **Two primary forms:**
    - **Logical graph:** Pure computational semantics without any hardware or distribution information. Captures only the mathematical operations and their dataflow. (Form: StableHLO or similar hardware-agnostic IR)
    - **Distributed graph:** The same computation annotated with the distribution strategy that was actually used — device placement, sharding specifications, replication factors, and communication patterns. Has a stable unique identifier incorporating all distribution choices. (Form: HLO after SPMD partitioning or equivalent IR with distribution annotations)
- **Other possible forms:**
    - **Verification graph**: Sometime it will be necessary to transform the computational graph into another representation that is easier for the verifier to work with—e.g. to verify inference, the verifier will likely need to use teacher-forcing (instead of decode), which is semantically equivalent but better controls for non-determinism (and also runs ~100x faster).
        - Note: it’s possible this should be done by the verifier, e.g. the verifier can just directly transform the StableHLO representation…
- **Purpose:** Records what computation the prover claims to have performed in a format that can be independently executed by any verifier using their own trusted compilation toolchain. For distributed graphs, the distribution metadata documents how the computation was distributed across devices, which will help for verification strategies that require the verifier knows the actual execution path. We currently ignore lower-level details of the prover’s execution path, e.g. its layout.
- **Key design decision:** Both forms share the same interface since they represent the same executed computation at different levels of detail. The distribution annotations enable device-specific verification challenges without constraining how the verifier reproduces the computation.

## **Trace**

- **Definition:** A detailed log of what happened during the actual execution of a distributed graph.
- **Form:** Timeline of runtime events. Depending on the application, this might include metadata about kernel launches, collective operations, memory transfers, and synchronization points, all annotated with timestamps, device identifiers, and operation references.
- **Purpose:** Provides forensic evidence that the reported computation actually ran with the specified distribution strategy. Links runtime behavior back to specific operations in the distributed graph.
- **Relationship to graphs:** Traces always reference a distributed graph (not logical) because device information is necessary to interpret the events. Each event must be traceable to specific operations and devices claimed in the distributed graph.

## **Data**

- **Definition:** The concrete tensors that flowed through the executed computation—e.g. inputs, outputs, model weights, and RNG seeds.
- **Form:** Serialized tensors with associated metadata (shape, type, layout) and cryptographic hashes for efficient verification.
- **Purpose:** Enables replay verification of the reported computation. A verifier can bind this data to the reported computational graph and re-execute to check consistency.
- **Key properties:**
    - Data was **emitted by** the prover's execution of the graph
    - Data can be **bound to** the reported graph for verification by the verifier
    - The same data can be bound to either a logical or distributed graph, enabling different verification strategies

## **Device**

- **Definition:** The hardware inventory and topology of the compute cluster where workloads execute.
- **Form:** Registry of accelerators, hosts, and network elements with their resource capacities (memory, compute, bandwidth) and interconnection topology. May follow existing formats like SLURM's topology/gres configs.
- **Purpose:** Enables verification protocols to validate resource claims and feasibility constraints. For example, memory spot-check challenges need to know device memory capacities and network latencies to verify that responses are only feasible if the claimed workload was actually running.
- **Relationship to other data:** Distributed graphs reference devices for placement claims. Traces tag events with device IDs. Verification protocols query device properties to establish constraints (e.g., "was there enough memory to run this workload?").
- **Note:** The exact schema is still being refined, but will likely include basic resource inventories (GPU types, memory sizes) and network topology (which devices can communicate, with what latency/bandwidth).

# API

The API provides capabilities for retrieving, transforming, and verifying reported workloads. These capabilities are organized around the core data types: Graphs, Traces, and Data.

## **Query**

The query system allows verifiers to retrieve information from the reported workload database.

### **What can be queried**

- Graphs (both logical and distributed forms)
- Traces
- Data—note that certain data types (e.g. tensors) may need to be canonicalized so that the verifier can interpret them appropriately.

### Joins

- Navigate between related objects (e.g., from a distributed graph to its logical form)
- Join across data types (e.g., find all traces for a graph, or all data for a trace)
- Filter by time ranges, device sets, or other criteria
- Discover what data is available for verification

## **Graph transformation**

Once retrieved, graphs can be transformed to focus verification on specific portions.

### **Slicing and selection**

- Extract subgraphs containing specific operations of interest
- Select computation paths between sets of operations
- Isolate individual batch elements for fine-grained verification
- Split batched computations into separate examples

### **Inspection**

- Examine graph structure and operations
- Access distribution metadata for distributed graphs
- Understand data dependencies and flow

## **Data binding**

The system can combine graphs with concrete data to create verifiable computation packages.

### **Binding modes**

- Attach full tensor data to all edges for complete replay
- Bind only inputs and outputs for black-box verification
- Include model weights and RNG states for exact reproduction

### **Validation**

- Verify type compatibility between data and graph edges
- Identify missing data needed for verification
- Support partial binding when complete data isn't available

## **Interpretation**

Bound graphs can be executed and verified on the verifier's infrastructure.

### **Execution**

- Run any graph slice on verifier's available hardware by interpreting them using the Python bindings of IREE
- Execute regardless of original distribution strategy

### **Verification**

- Compare execution results against claimed outputs with configurable tolerance
- Validate consistency between execution and reported trace events
- Support challenge-response protocols for device-specific verification
- Enable property-based testing without exact replay

# Applications

Below are some concrete applications of Veritor. 

## **Inference verification via locality-sensitive hashing (STAMP)**

### Protocol overview

When users query LLM inference providers, they cannot verify whether the provider ran the claimed model and configuration. A provider might secretly use quantized models to save costs, alter sampling parameters, or even substitute smaller models entirely. STAMP enables lightweight verification that the correct inference occurred by combining activation fingerprints with sampler attestation.

The key insight: while GPU non-determinism prevents bit-exact reproducibility, the structure of neural network computations remains consistent enough to verify through compact, noise-robust fingerprints. By recording both what the model computed (activations) and how tokens were selected (sampling), we can detect various forms of dishonest inference.

### **Prover workflow**

1. **Execute inference normally**
    - Run the claimed model on user input
    - Sample output tokens using declared parameters (temperature, top-p, seed)
    - During generation, periodically record activation fingerprints via locality-sensitive hashing
    - Log the PRNG seed used for sampling
2. **Generate proof transcript**
    - Compute k-dimensional projections of hidden states at regular intervals
    - Include sampling configuration and seed
    - Total overhead: ~0.5-8 bytes per token depending on verification requirements
3. **Report to Veritor**
    - Submit the computational graph (model architecture and config)
    - Submit trace including the proof transcript
    - Submit input/output data

### **Verifier workflow**

1. **Retrieve claimed execution**
    - Query for the model configuration, inputs, outputs, and proof transcript
    - Extract activation fingerprints and sampling parameters
2. **Re-execute in teacher-forcing mode**
    - Run the full input-output sequence through the trusted model in a single prefill pass
    - Compute activation fingerprints at the same positions
    - Note: This is much faster than the original autoregressive generation
3. **Verify activation matching**
    - Compare fingerprints using L2 distance under quantization noise tolerance
    - Detect if a different model or precision was used
4. **Verify sampler consistency**
    - Reconstruct the sampling distribution at each position
    - Check that output tokens are **not surprising under the verifier’s logits**—you can inject information about the realization of the RNG, but this is not strictly necessary
        - Note that here we do not *directly compare the prover’s tokens to the verifier’s tokens*—this is because small FP differences will drive errors. There is still enough statistical power just looking at the surprisal of the prover’s token choices.
    - Detect teacher-forcing attacks where arbitrary outputs were fed through the model
5. **Decision**
    - Accept if both activation distances and sampling likelihoods fall within thresholds
    - Reject if significant deviations indicate dishonest inference

### **What this enables**

- **Detect cost-cutting shortcuts**: Identify when providers use quantized models or reduced precision without disclosure
- **Verify sampling integrity**: Ensure outputs were actually sampled, not injected via teacher-forcing
- **Maintain efficiency**: Validation runs faster than generation due to prefill-only execution
- **Handle GPU non-determinism**: Robust to legitimate implementation differences while detecting meaningful changes

This protocol composes naturally with Veritor's broader verification framework - the proof transcript becomes part of the trace, and verification involves comparing re-executed computations against the claimed behavior.

### **Extensions needed for Veritor**

To fully implement STAMP, Veritor needs the following capabilities:

1. **Autoregressive-to-prefill graph transformation**
    - The prover's graph represents autoregressive generation where each token sampling feeds into the next position
    - We need to transformation this into a teacher-forced graph for verification
    - This allows the verifier to process all tokens in parallel (prefill mode) rather than sequentially
2. **Explicit sampling operation representation**
    - Current IR likely represents sampling as an opaque "sample" operation
    - (Perhaps) need to decompose this into explicit operations: softmax → temperature scaling → cumulative sum → binary search with PRNG value
    - This enables the verifier to reconstruct the exact sampling distribution and verify token consistency

## **Training verification via unpredictable checkpointing**

### Protocol overview

Verifying that AI models were trained only on declared data is crucial for AI governance. However, existing approaches either require trusted firmware on all GPUs globally (politically infeasible) or cannot reliably detect small amounts of undeclared training data due to GPU non-determinism. This protocol enables lightweight verification by giving the verifier control over which training checkpoints are persisted, facilitating i.i.d. sampling of declared training steps for verification.

The key insight: by maintaining a buffer of recent checkpoints and responding to unpredictable verifier requests, the prover cannot feasibly compute alternative checkpoints that would be consistent with committed gradients while secretly training on undeclared data.

### **Prover workflow**

1. **Execute training with commitments**
    - During training, each replica commits to the contents of each mini-batch
    - Compute and commit to locality-sensitive hashes (LSH) of each gradient (e.g., random down-projection seeded with verifier nonce)
    - Store each all-reduce result (new model checkpoint) in a rolling buffer of size K (e.g., K=10)
    - Continue normal training while maintaining these commitments
2. **Respond to checkpoint requests**
    - When verifier requests, provide checkpoint(s) from the buffer
    - Include associated mini-batch commitments and gradient LSHs
    - Must respond within time bound to prevent computing fake checkpoints
3. **Report to Veritor**
    - Submit computational graph of the training procedure
    - Submit trace including all commitments and checkpoint events
    - Submit requested checkpoints and associated data

### **Verifier workflow**

1. **Issue unpredictable challenges**
    - Periodically request random checkpoint(s) from the prover's buffer
    - May request single checkpoints or consecutive pairs
    - Time requests unpredictably to ensure i.i.d. sampling
2. **Retrieve verification data**
    - Query for requested checkpoints and commitments
    - Extract mini-batch data and gradient LSHs
    - Obtain computational graph for the training step
3. **Re-execute training segments**
    - For single checkpoints: re-run any mini-batch and verify gradient LSH consistency
    - For checkpoint pairs: re-run full batch between checkpoints
    - Execute on trusted hardware to establish ground truth
4. **Verify consistency**
    - Compare recomputed gradients against committed LSHs
    - For checkpoint pairs, verify the model update matches claimed training
    - Check that timing constraints make alternative checkpoint computation infeasible
5. **Decision**
    - Accept if gradients match commitments and checkpoints are consistent
    - Reject if discrepancies suggest undeclared data insertion
    - Request additional checkpoints if suspicious patterns detected

## **Compute verification on untrusted hardware via memory spot checks**

### Protocol overview

Lightweight "pinger" devices issue unpredictable challenges to accelerators at random times, asking for locality-sensitive hashes (LSH) of their current identifying state (e.g., activations). The key insight: if accelerators have limited free memory (< Y) and must respond within latency bound X, then the only feasible way to pass is to already have the requested state in memory. Undeclared workloads would require memory evictions that get detected through spot checks.

The protocol enforces that all storage within X-latency of accelerators (the "proximal memory pool") must contain expensive state that takes ≥ Z time to recompute. Random challenges verify this invariant.

### **Prover workflow**

1. **Execute workload with challenge support**
    - Run computation normally
    - When pinged, respond with LSH (seeded via a challenge nonce) of latest identifying state (e.g., activations) within X latency
    - Continue execution while capturing all runtime behavior
2. **Report to Veritor**
    - Submit logical graph, distributed graph, trace, and data
    - The trace contains references to all data including challenge events and responses
    - Note: Devices must be registered beforehand in a device registry that tracks resource capacities

### **Verifier workflow**

1. **Examine cluster behavior**
    - Review top-level cluster view to understand overall execution pattern
    - Identify trace slices of interest for detailed verification
2. **Sample and retrieve**
    - Select trace slices to verify full I/O behavior (memory measurements are one type of output)
    - Query for relevant graphs and associated data
3. **Verify trace slice behavior**
    - Extract subgraph for the selected trace window
    - Re-execute with corresponding inputs
    - Compare all outputs: computed values, memory states, and LSH responses
4. **Verify protocol compliance**
    - Response time: verify challenges answered within X latency bound
    - Memory pressure: confirm free memory stayed < Y during window
    - Feasibility check: verify no room for undeclared work given (X,Y,Z) constraints
5. **Decision**
    - Accept if outputs match and protocol constraints satisfied
    - Reject or request additional evidence if violations detected

### Extensions to Veritor this implementation requires

We will need to fill up the memory of all devices near each accelerator (according to some latency metric) with memory stuffing—some expensive state, e.g. using proof-of-spacetime. This prevents the prover from caching its identifying state and then fetching it from the cache at challenge-time. 

# Open questions

**Precision representation.** It is unclear where numerical precision should formally “live” in the data model. On one hand, the logical graph might specify intended dtypes and numerical semantics, while the distributed graph records the effective dtypes and kernel-level choices that were actually used. Verification may need both, but we don’t yet know the cleanest abstraction for capturing intended versus realized precision, or how much detail is necessary to make verification meaningful without overcomplicating the graph.

**Sampling representation.** It is unclear how best to represent stochastic sampling within the computational graph. In principle, PRNG operations could be modeled as graph nodes, but different frameworks and kernels consume random numbers in inconsistent ways, making reproducibility and verification difficult. We may need a canonical “sampler dialect” that fixes both PRNG semantics and sampling composites (e.g. top-p, top-k), but we don’t yet know how hard this will be to define or how portable such an abstraction would be across hardware and frameworks.

# Implementation plan

## Step 0

- De-risk by checking that this translation from prover graph → verification/teacher-forcing graph will work in JAXPR
- De-risk by checking how sampler operators might work, and how these need to be translated to be reproducible

## Step 1

- Single device (just laptop CPU)
- No notion of a Device
- Workloads:
    - Simple deterministic inference (random models, not autoregressive, no separate verification graph)
    - Simple deterministic training (just save every checkpoint, tiny random models)
        - Here I’m mainly curious here to get a feel for what these graphs look like… like I’m not entirely sure where the logical end of one of these is…
- Measurements
    - Static LSHs (e.g. fixed beforehand)
    - Dynamic LSHs (e.g. via memory challenges)
- Show we can run arbitrary forms of this workload and the prover can naturally capture all of its data in our abstractions, and the verifier can pick things out and verify them
- Set up a nice dev harness
- Implement memory challenges via io hooks — make sure that measurements get logged too

## Step 2

- Mock distribution of workloads across many devices (again, all on CPU)
- Show logical graph ↔ distributed graph

## Step 3

- Stand up random model checkpointing

## Step 4

- Introduce autoregressive workloads (still no sampling)

## Step 5

- Introduce RNG + sampling (might need to canonicalize sampling somehow? not sure)
- Run training/inference verification on a tiny, deterministic transformer

## Step 6

- Test on cloud GPUs
- Get some LLMs to run in inference/training
- Get a handle on non-determinism

## Step 7

- Test on cloud GPUs
- Get some LLMs to run in inference/training
- Get a handle on non-determinism
- **Write the training verification paper and the inference verification paper**

## Step 8

- Go back to on-laptop CPU for dev
- Build out Device API
- Convert computational graphs → dataflow graphs, including I/O operators (probably using Ray Data)
- Simulate a cluster—implement the memory spot check protocol, filling up the devices etc.

## Step 9

- Simulate an actual cluster
- **Write the compute verification paper (memory spot checks)**