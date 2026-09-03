"""Measured proof-cost constants and the estimates built on them.

Every constant here was measured, and its docstring says how.  Three sources:

* **This repo's SP1 guest in execute mode** (exact cycle counts, no proving;
  ``zk/sp1/bench/measure.py``, SP1 6.4.0, guest built from the ``zk/sp1``
  workspace at the commit that introduced these numbers, ``sha2`` precompile
  patch on).  Synthetic but valid batches isolate one cost at a time; the
  fit is a two-point slope, exact for these linear costs.  Re-measured after
  the interior moved to VU-output granularity (a sampled VU opens its inputs
  and outputs and the guest recomputes the gates between) and cross-checked
  on real batches: a 186-obligation small-cluster batch (1,040 positions,
  8,018 Merkle levels, 854 gates) predicted 30.67M cycles against 30.84M
  measured (-0.5%); the 16-obligation matmul batch (40 positions, 128 levels,
  28 gates) 761,283 against 761,710 (-0.1%).
* **sp1-op-bench** (``results/report.md``): core proving throughput and the
  batching curve of SP1 6.4.0 on an Apple Silicon laptop and on a cloud CPU
  pod, and the per-MAC cycle costs of dot products in the RISC-V ISA.
* **openvm-tc-bench** (``docs/backend-decision.md``): GPU proving throughput
  of the ``TC_DOT`` / ``TC_MATMUL_4X4X16`` precompiles under OpenVM v2.0.2
  and a TC_DOT-patched SP1 6.4.0 on one RTX 4090, and the implied cost of
  verifiability against native fp8 tensor cores.

The estimates: :func:`estimate_cycles` (a batch statement to SP1 cycles,
by phase), :func:`prover_seconds` (cycles to seconds on a machine profile,
including the per-proof floor), :func:`estimate_prover_seconds`,
:func:`alpha_toy_isa` (proving cost over native cost for a toy-ISA batch)
and :func:`alpha_dot` (the same ratio for the dot relations of the ML
workloads, from the GPU numbers).  ``Cost`` in ``veritor.analysis`` counts
abstract units (``h`` per committed position, ``c_0`` per proof); the
:func:`sp1_cost_parameters` helper prices those units in prover seconds.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from veritor.analysis.cost import CostParameters

from .statement import Statement
from .wire import encode_statement

# -- SP1 guest, execute mode (this repo) -------------------------------------

SP1_CYCLES_PER_GATE: Mapping[str, int] = {
    "add": 125,
    "eq": 126,
    "lt": 177,
    "mul": 180,
    "shr": 206,
    "sub": 238,
    "in": 0,
    "weight": 0,
}
"""Recomputation cycles per toy-ISA gate, by op (the ``gates`` tracker).

Measured as the slope of the ``gates`` phase between one obligation whose kind
is a chain of 8 gates and one of 136 gates of the same op, each opening only
its last gate (``add``: 125.0, ``eq``: 126.0, ``lt``: 177.0, ``mul``: 180.0,
``shr``: 206.0, ``sub``: 238.0 cycles/gate; 116-212 before the guest
recomputed, when it read every gate's opened value instead of pushing the
result onto the copy's local values).  These are the pinned modular semantics
of ``veritor/core/gates.py`` interpreted by ``zk/sp1/common/src/check.rs``:
reading two operands, the ``wrapping_*`` op and mask, the output cursor and
the compare when the gate is opened.  Source gates (``in``, ``weight``) have
no relation; ``in`` costs one 2-byte compare in the ``merkle`` phase.  Only a
copy's inputs and declared outputs are opened *positions*; a gate between
costs its recomputation and nothing else, where before it also cost a leaf
plus depth levels of Merkle authentication (~25k cycles at depth 8).
"""

SP1_CYCLES_PER_MERKLE_LEVEL = 2_668
"""Cycles per authentication-path level (one ``node`` frame hash).

Slope of the ``merkle`` phase between a 16-gate obligation opening all 18 of
its positions in a depth-5 domain and the same in a depth-13 domain: 2,668.0
cycles/level (2,846 in total cycles; 2,694 before the re-measurement, the
difference being run-to-run allocator noise).  Each level is exactly 3 ``SHA_COMPRESS`` syscalls
(measured 3.0/level): our ``node`` frame is ``FRAME || u32 len || "node" ||
(u64 len || part)*5`` = 179 bytes, i.e. three 64-byte blocks after padding,
against 2 blocks for a bare ``left || right`` hash.  sp1-op-bench measured
1,620 cycles/level for that 2-block hash, i.e. ~800 cycles/block effective
with the sha2 crate's per-call init/finalize; our 3-block frame sits on the
same line.  A hand-rolled SHA-256 over the syscalls (no ``block-buffer``)
would cut this to ~3 x 240 + ~200 cycles; not done here.
"""

SP1_CYCLES_PER_LEAF = 3_296
"""Cycles per opened position besides its path: leaf frame hash and decode.

The ``merkle`` slope per position at depth 8, between a 16-gate chain opening
every gate (18 positions) and the same opening only its last (3 positions),
is 24,640 cycles; minus 8 x 2,668 for the path leaves 3,296 for the ``leaf``
frame hash (3 blocks), the rank/depth checks, the optional public-input
compare and the value decode (3,419 before the re-measurement).
"""

SP1_CYCLES_PER_PARSE_BYTE = 12.26
"""Cycles per byte of statement + witness in the ``parse`` phase.

Slope of ``parse`` against total input bytes between a 1-obligation and a
65-obligation batch of one-gate kinds at depth 8: 12.26 cycles/byte (11.70
before).  The witness dominates (each opening carries ``32 x depth`` path
bytes); the parser allocates a ``Vec`` per list, which is most of this.
"""

SP1_CYCLES_PER_DIGEST_BYTE = 3.53
"""Cycles per statement byte for ``sha256(statement)`` (the ``digest`` phase): 3.53.

Bulk hashing through the precompile: ~226 cycles per 64-byte block, in line
with sp1-op-bench's 231 cycles/block.
"""

SP1_CYCLES_PER_OBLIGATION = 5_800
"""Fixed cycles per obligation beyond its positions, gates and bytes.

From the 1-vs-65 obligation fit (1-gate kinds at depth 8): 93,465 cycles per
obligation in total, of which 3 positions x 24,640 = 73,919 are Merkle,
1,106 bytes x 12.26 = 13,563 are parse, 304 statement bytes x 3.53 = 1,073
are digest and 929 are the ``gates`` pass; the remaining ~4,000 is the
per-obligation constant of that synthetic batch.  Real batches carry more
kinds and positions per obligation and fit ~5,700-6,600 (the matmul and
small-cluster cross-checks), so 5,800 is kept.
"""

SP1_CYCLES_PER_BATCH = 50_000
"""Fixed cycles per batch: SP1 entry and exit, reading the two inputs, the
33-byte commit and the tracker gaps.

A one-obligation, one-gate, depth-2 batch runs 94,647 cycles in total with
48,583 outside every tracker; the intercept of the per-obligation fit is
53,343.  ``io`` is 435 cycles for both inputs.
"""

# -- SP1 proving throughput (sp1-op-bench, results/report.md) ----------------


@dataclass(frozen=True, slots=True)
class MachineProfile:
    """Core-proof throughput of one machine: a fixed floor plus a marginal rate."""

    name: str
    floor_seconds: float
    khz: float
    peak_rss_gib_per_million_cycles: float

    def seconds(self, cycles: float) -> float:
        return self.floor_seconds + cycles / (self.khz * 1_000)


LAPTOP_M_SERIES = MachineProfile("apple-silicon-laptop", 10.0, 112.0, 4.6)
"""sp1-op-bench section C: Apple Silicon laptop, SP1 6.4.0 core proofs, CPU.

f32_dot n=4096 (2.26M cycles) proved in 20.57 s and n=16384 (9.02M) in
76.67 s: asymptotic 110-118 kHz; a 175k-cycle program still took 10.4 s, the
per-proof floor.  This repo's 16-obligation matmul batch (879,613 cycles,
2 shards) proved in 18.1 s core (2,806,928-byte proof, verified in 0.13 s,
4.2 GiB peak RSS) on the same class of machine: consistent with the floor
plus ~8 s of marginal work.  Memory: ~20 GiB at 2.6M cycles, ~84 GiB at
41.6M (pod B); ~4.6 GiB per million cycles is a conservative planning figure
for RSS growth beyond the floor.
"""

POD_B_CPU = MachineProfile("cloud-cpu-pod-b", 40.0, 105.0, 2.0)
"""sp1-op-bench sections I-K: a 15-core cloud CPU pod, SP1 6.4.0 core proofs.

leaf_check k=1 (2.60M cycles) 63.45 s, k=4 (10.4M) 153.78 s, k=16 (41.6M)
450.14 s: 92.4 kHz overall on the largest, ~105 kHz marginal from k=4 to
k=16 (24.7 s per added 2.6M-cycle check), a ~40 s floor.  Peak RSS 20.4 /
38.4 / 83.8 GiB.  Compressed k=16 took 1.28x the core time for a constant
1,272,577-byte proof.
"""

SP1_BATCHING_CURVE: tuple[tuple[int, int, float, float], ...] = (
    (1, 2_595_299, 63.45, 20.4),
    (4, 10_395_153, 153.78, 38.4),
    (16, 41_577_643, 450.14, 83.8),
)
"""sp1-op-bench section J: ``(checks per proof, cycles, core prove seconds, peak RSS GiB)``.

Seconds per check fall 63.5 -> 38.4 -> 28.1 as the fixed floor is amortized;
the marginal cost converges to 24.7 s per 2.6M-cycle check (~105 kHz).  The
curve is what :class:`MachineProfile` linearizes: ``floor + cycles / kHz``.
"""

SP1_CYCLES_PER_FIXED_MAC = 16.0
"""sp1-op-bench section B: fixed-point dot product, 16.0-16.3 cycles per MAC
(n = 1,024-16,384) in plain RISC-V, no precompile; f32 dot is 130-140."""

# -- GPU tensor-core precompiles (openvm-tc-bench, docs/backend-decision.md) --

OPENVM_TC_MATMUL_MAC_PER_SECOND = 2_519_884.0
"""OpenVM v2.0.2 + ``TC_MATMUL_4X4X16`` extension, RTX 4090, warm paired
medians, n=4096 tiles, batch k=64: 3.97e-7 s/MAC.  (n=16384, k=8: 1.55M MAC/s.)"""

OPENVM_TC_DOT_MAC_PER_SECOND = 1_198_237.0
"""OpenVM ``TC_DOT`` precompile, RTX 4090, n=4096, k=256: 8.35e-7 s/MAC."""

SP1_TC_DOT_MAC_PER_SECOND = 982_066.0
"""TC_DOT-patched SP1 6.4.0 fork, RTX 4090 (GPU prover), n=4096, k=16384:
1.02e-6 s/MAC; the plateau needs k >= 2,048 (35-69 s proofs)."""

NATIVE_FP8_MAC_PER_SECOND_4090 = 1.8e14
"""The native fp8 tensor-core rate the decision report's ``alpha`` figures imply.

The report states a cost of verifiability of ~7.5e7 at the TC_MATMUL plateau
and ~1.8e8 for the prior SP1 TC_DOT headline; dividing those by the measured
proving rates (2.52e6 and 0.98e6 MAC/s) gives 1.89e14 and 1.77e14 native
MAC/s, i.e. ~1.8e14: roughly 55% of the 4090's 3.3e14 dense fp8 peak, a
realistic achieved matmul rate.  Using the peak instead scales every
``alpha_dot`` by 1.83.
"""

NATIVE_TOY_GATE_SECONDS = 1 / 3.0e9
"""One toy-ISA gate natively: one 16-bit ALU op, one cycle at 3 GHz."""

# -- estimates ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CycleEstimate:
    """SP1 guest cycles for a batch, by phase."""

    obligations: int
    positions: int
    merkle_levels: int
    gates: int
    statement_bytes: int
    witness_bytes: int
    batch: int
    per_obligation: int
    parse: int
    digest: int
    merkle: int
    relations: int

    @property
    def total(self) -> int:
        return (
            self.batch
            + self.per_obligation
            + self.parse
            + self.digest
            + self.merkle
            + self.relations
        )


def _depth(count: int) -> int:
    return 0 if count <= 1 else (count - 1).bit_length()


def estimate_cycles(statement: Statement) -> CycleEstimate:
    """Predict the SP1 guest's cycles for ``statement`` from the measured constants.

    The witness size is implied by the statement (one value of the schema's
    width and ``depth`` 32-byte siblings per position).
    """

    positions = merkle_levels = gates = relations = witness_bytes = 0
    for obligation in statement.obligations:
        program = statement.program(obligation.kind)
        witness_bytes += 4
        for ref in obligation.positions:
            depth = _depth(obligation.commitments[ref.commitment].count)
            positions += 1
            merkle_levels += depth
            witness_bytes += 4 + (statement.width + 7) // 8 + 4 + 32 * depth
        gates += len(program.gates)
        relations += sum(
            SP1_CYCLES_PER_GATE.get(gate.op, SP1_CYCLES_PER_GATE["sub"])
            for gate in program.gates
        )
    statement_bytes = len(encode_statement(statement))
    return CycleEstimate(
        obligations=len(statement.obligations),
        positions=positions,
        merkle_levels=merkle_levels,
        gates=gates,
        statement_bytes=statement_bytes,
        witness_bytes=witness_bytes,
        batch=SP1_CYCLES_PER_BATCH,
        per_obligation=SP1_CYCLES_PER_OBLIGATION * len(statement.obligations),
        parse=round(SP1_CYCLES_PER_PARSE_BYTE * (statement_bytes + witness_bytes)),
        digest=round(SP1_CYCLES_PER_DIGEST_BYTE * statement_bytes),
        merkle=SP1_CYCLES_PER_LEAF * positions
        + SP1_CYCLES_PER_MERKLE_LEVEL * merkle_levels,
        relations=relations,
    )


def prover_seconds(cycles: float, machine: MachineProfile = LAPTOP_M_SERIES) -> float:
    """Core-proof seconds for ``cycles`` on ``machine``: the floor plus the marginal rate."""

    return machine.seconds(cycles)


def estimate_prover_seconds(
    statement: Statement, machine: MachineProfile = LAPTOP_M_SERIES
) -> float:
    """Estimated seconds to prove one batch with the SP1 backend on ``machine``."""

    return prover_seconds(estimate_cycles(statement).total, machine)


def alpha_toy_isa(
    statement: Statement, machine: MachineProfile = LAPTOP_M_SERIES
) -> float:
    """``alpha`` = proving seconds / native seconds for a toy-ISA batch.

    Native cost is one ALU cycle per non-source gate.  A copy's inputs and
    declared outputs are committed leaves (~25k cycles each at depth 8) while
    the gates between cost only their recomputation (~125-240 cycles), so
    this alpha is dominated by Merkle authentication of the opened positions
    unless the kinds are deep.
    """

    estimate = estimate_cycles(statement)
    native_gates = sum(
        1
        for obligation in statement.obligations
        for gate in statement.program(obligation.kind).gates
        if gate.op not in ("in", "weight")
    )
    if native_gates == 0:
        raise ValueError("a batch with no relation has no native cost")
    return prover_seconds(estimate.total, machine) / (
        native_gates * NATIVE_TOY_GATE_SECONDS
    )


DOT_BACKENDS: Mapping[str, float] = {
    "openvm-tc-matmul": OPENVM_TC_MATMUL_MAC_PER_SECOND,
    "openvm-tc-dot": OPENVM_TC_DOT_MAC_PER_SECOND,
    "sp1-tc-dot": SP1_TC_DOT_MAC_PER_SECOND,
}


def alpha_dot(
    backend: str = "openvm-tc-matmul",
    native_mac_per_second: float = NATIVE_FP8_MAC_PER_SECOND_4090,
) -> float:
    """``alpha`` for the dot relations: native MAC/s over proving MAC/s.

    ``"openvm-tc-matmul"`` -> ~7.1e7 (the report's ~7.5e7), ``"openvm-tc-dot"``
    -> ~1.5e8, ``"sp1-tc-dot"`` -> ~1.8e8 with the implied native rate.
    """

    try:
        proving = DOT_BACKENDS[backend]
    except KeyError:
        raise ValueError(
            f"unknown dot backend {backend!r}; one of {sorted(DOT_BACKENDS)}"
        ) from None
    return native_mac_per_second / proving


def sp1_cost_parameters(
    depth: int, batch_size: int, machine: MachineProfile = LAPTOP_M_SERIES
) -> CostParameters:
    """``h`` and ``c_0`` in prover seconds for the SP1 backend.

    ``h`` is the marginal cost of one opened position in a domain of the given
    Merkle ``depth`` (leaf, path, its parse bytes); ``c_0`` is the fixed cost
    of one proof obligation when ``batch_size`` obligations share a proof (the
    per-obligation cycles plus the batch cycles and the machine's floor,
    amortized).  ``Cost``'s gate units stay in ``kind.proof_cost``; the
    relation cycles are a rounding error next to ``h``.
    """

    if depth < 0 or batch_size <= 0:
        raise ValueError("depth must be nonnegative and batch_size positive")
    per_second = 1 / (machine.khz * 1_000)
    position_cycles = (
        SP1_CYCLES_PER_LEAF
        + SP1_CYCLES_PER_MERKLE_LEVEL * depth
        + SP1_CYCLES_PER_PARSE_BYTE * (8 + 2 + 32 * depth + 30)
    )
    fixed_cycles = SP1_CYCLES_PER_OBLIGATION + SP1_CYCLES_PER_BATCH / batch_size
    fixed = fixed_cycles * per_second + machine.floor_seconds / batch_size
    return CostParameters(
        hash_cost=Fraction(position_cycles * per_second).limit_denominator(10**12),
        proof_overhead=Fraction(fixed).limit_denominator(10**12),
    )


__all__ = [
    "DOT_BACKENDS",
    "LAPTOP_M_SERIES",
    "NATIVE_FP8_MAC_PER_SECOND_4090",
    "NATIVE_TOY_GATE_SECONDS",
    "OPENVM_TC_DOT_MAC_PER_SECOND",
    "OPENVM_TC_MATMUL_MAC_PER_SECOND",
    "POD_B_CPU",
    "SP1_BATCHING_CURVE",
    "SP1_CYCLES_PER_BATCH",
    "SP1_CYCLES_PER_DIGEST_BYTE",
    "SP1_CYCLES_PER_FIXED_MAC",
    "SP1_CYCLES_PER_GATE",
    "SP1_CYCLES_PER_LEAF",
    "SP1_CYCLES_PER_MERKLE_LEVEL",
    "SP1_CYCLES_PER_OBLIGATION",
    "SP1_CYCLES_PER_PARSE_BYTE",
    "SP1_TC_DOT_MAC_PER_SECOND",
    "CycleEstimate",
    "MachineProfile",
    "alpha_dot",
    "alpha_toy_isa",
    "estimate_cycles",
    "estimate_prover_seconds",
    "prover_seconds",
    "sp1_cost_parameters",
]
