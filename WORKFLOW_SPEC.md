Workflow
The verification workflow involves three distinct entities with well-defined roles and interfaces. The core design principle is that the Prover cannot predict the challenge schedule, while maintaining a clean separation between computation and verification.
Entities
Prover: Executes workloads with embedded challenge hooks. Owns workload compilation, execution, and database writes.
Challenger: Issues challenges based on a secret schedule during execution. Maintains complete isolation from the Prover's computational state.
Verifier: Validates reported computations post-execution. Owns verification logic and database reads, using IREE for graph interpretation without JAX dependencies.
Design Decisions & Motivation
Compilation Granularity: The entire workload is compiled as a single computational graph. This choice ensures:

The Verifier can see the complete computation structure
Challenge hooks are embedded at known positions in the graph
The workload's resource usage patterns are fully visible
All outputs (both intermediate and final) are explicitly marked with outfeed operations for consistency

Challenge Decomposition: Challenges involve three operations, only one of which is truly impure:

(Impure) Read challenge decision + parameters from Challenger via io_callback
(Pure) Compute challenge given decision (LSH projection, hash, etc.)
(Side-effect) Write results via outfeed

This decomposition means most challenge logic remains pure and verifiable in the IR, with only the decision-making being external.
Operation Identification: A critical requirement is mapping between Python execution context and StableHLO operations:

Each StableHLO operation has a deterministic handle/ID
The Prover must reliably map its current Python execution point to these handles
Challenges are issued to specific StableHLO operation IDs
The Prover's runtime maintains this operation ID ↔ execution context mapping

IR Representation: The StableHLO contains:

The full workload computation with custom_call operations for challenge decision points
Pure challenge computation logic (conditional on decisions)
Explicit outfeed operations for all outputs (workload results and challenge responses)

Custom ops are not opaque - they contain metadata about what type of challenge computation they represent.
Workflow Phases
Phase 1: Workload Compilation and Registration

Prover compiles full workload

Creates computational graph for entire workload (training loop, inference batch, etc.)
Embeds challenge hooks at predetermined positions
Hook points reference operations via deterministic StableHLO operation IDs
All outputs marked with explicit outfeed operations
Outputs: StableHLO with custom_call operations for challenges


Prover sends graph to Verifier

Complete workload graph including challenge points
Graph is immutable once sent - defines the computation contract
Contains metadata for interpreting custom operations
Includes mapping scheme for operation IDs


Verifier analyzes and creates schedule

Examines workload structure and challenge hook positions
Creates secret challenge schedule targeting specific StableHLO operation IDs
Schedule specifies which challenges fire and with what parameters
Sends schedule to Challenger (not to Prover)



Phase 2: Execution with Challenge Injection

Prover executes compiled workload

Runs the full compiled graph
Maintains mapping of current execution to StableHLO operation IDs
At each custom_call, provides operation ID to Challenger
Executes pure challenge computation based on response
All outputs (workload and challenges) go through outfeed


Challenger responds to queries

Receives queries with StableHLO operation IDs
Returns decisions based on Verifier's secret schedule for those IDs
Provides challenge parameters (seeds, dimensions, etc.)
Maintains no knowledge of Prover's computational state


Prover completes and stores

Finishes full workload execution
Stores complete graph, trace, and all data to database
Trace includes mapping of events to StableHLO operation IDs
Database becomes immutable record of execution



Phase 3: Verification

Verifier reads database

Retrieves workload graph, trace, and data
Extracts challenge responses and parameters from trace
Maps trace events back to StableHLO operation IDs


Verifier reconstructs execution

Impurity Replay Strategy: The Verifier's execution engine must handle:

Replacing custom_call ops with recorded (decision, seed) values
Intercepting outfeed operations to compare with recorded outputs
Maintaining operation ID tracking for correlation


Executes pure challenge computations to verify responses
Runs full graph through IREE with impurity replay support
No JAX dependency - pure IR interpretation with controlled replay


Verifier validates compliance

Checks challenge responses match expected values
Verifies all outfeed operations produced expected outputs
Confirms operation ID sequence matches expected execution order
Validates main computation outputs



Key Architectural Choices
Why io_callback: JAX needs to compile pure functions, but challenges require reading external state unknown at compile time. The io_callback creates "holes" in the computation where external decisions can be injected without JAX trying to optimize across them.
Why Full Workload Compilation: Compiling the complete workload ensures the Verifier sees the actual computation structure. This applies to any computational pattern - training loops, inference batches, or complex pipelines. The challenge schedule can be unknown-but-static across the workload.
Why Explicit Outfeed Everywhere: Using outfeed for all outputs (not just challenges) creates a consistent side-effect pattern in the IR. This makes verification cleaner - the Verifier expects all results to flow through outfeed operations rather than mixing returns and side-effects.
Why Operation ID Mapping is Critical: The Prover must reliably know which StableHLO operation it's executing to respond to challenges correctly. This requires deterministic operation naming/numbering that both Prover and Verifier can compute independently from the graph structure.
Why Verifier Needs Impurity Replay: While most computation is pure, the Verifier's execution engine must handle replaying recorded impure decisions (challenge responses) and comparing side effects (outfeed operations). This replay mechanism is the key addition to pure IREE interpretation.
This architecture ensures challenge unpredictability while maintaining a verifiable computation record that can be validated without requiring identical hardware or software stacks.





## Implementation Status

### Completed Abstractions (test_three_party_inference.py)

**OperationIDMapper**: Maintains deterministic mapping between Python execution context and StableHLO operation IDs. Critical for correlating challenges with specific operations.

**Challenger**: External entity that receives secret schedule from Verifier and responds to Prover queries with challenge decisions. Has no knowledge of computational state.

**Prover**: Executes workload with embedded hooks. Uses `io_callback` for impure challenge decisions, pure computations for challenge responses, and outfeed for all outputs.

**Verifier**: Creates secret schedule from graph structure, provides to Challenger, and validates execution post-hoc using database records.

### Key Implementation Details

1. **Challenge Decomposition**:
   - Impure: `io_callback` to query Challenger for decision
   - Pure: Challenge computation (LSH projection) visible in StableHLO
   - Side-effect: Outfeed operation for all outputs

2. **Operation ID Tracking**: Deterministic naming scheme (`op_0000_layer_0_linear`) enables correlation between execution and graph structure.

3. **Fixed Projection Dimensions**: Current implementation uses fixed dimension (4) for LSH projections to avoid dynamic shape issues in JAX.

4. **Schedule Creation**: Verifier uses hash of StableHLO content as entropy source for unpredictable but deterministic schedule.

### Next Steps

1. **IREE Integration**: Replace simulation with actual IREE execution for verification
2. **Variable Dimensions**: Handle dynamic projection dimensions properly
3. **Impurity Replay**: Implement proper replay mechanism in Verifier
4. **Training Support**: Extend to training workloads with gradient challenges
5. **Distributed Execution**: Add support for multi-device scenarios 