"""The protocol: commit to both tapes, then sample and locally check instructions.

Roles:

  Prover  -- runs the program, commits to the instruction tape and the value
             tape (two Merkle roots), and answers openings.
  Verifier -- re-traces the public program (v0 instruction provenance),
             checks the boundary cells exactly (inputs and outputs), draws
             random instruction indices *after* the roots are fixed, and
             checks each sampled instruction locally.

Message flow (one round trip each way):

  message 1 (prover -> verifier): the Commitment -- both roots and the
            claimed outputs. After this, nothing can change.
  message 2 (verifier -> prover): the challenges -- boundary indices plus
            s uniform instruction indices, drawn only AFTER message 1.
  message 3 (prover -> verifier): the openings -- authenticated cells and
            instructions, which the verifier checks locally.

What a sampled check establishes: the committed write cell of instruction k
equals the primitive applied to the committed cells it reads. A dishonest
tape with L locally-inconsistent instructions survives s independent samples
with probability (1 - L/N)^s, and every instruction that escapes checking can
inject at most 32 unexplained bits (one cell). That capacity bound -- not
full correctness -- is the point of the design.
"""

from __future__ import annotations

import random
import secrets
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from veritor.machine import (
    Program,
    apply_primitive,
    decode_cell,
    decode_instruction,
    encode_cell,
    encode_instruction,
    execute,
)
from veritor.merkle import LEAF_INSTRUCTION, LEAF_VALUE, MerkleTree, verify_leaf

# ---------------------------------------------------------------------------
# Messages exchanged between prover and verifier.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Commitment:
    """The prover's first message. After this, nothing can change."""

    instruction_root: bytes
    value_root: bytes
    num_inputs: int
    num_instructions: int
    claimed_outputs: tuple[float, ...]  # the claimed result y*, one per output cell

    @property
    def num_cells(self) -> int:
        return self.num_inputs + self.num_instructions


@dataclass(frozen=True)
class ValueOpening:
    """One value-tape cell plus its authentication path."""

    index: int
    payload: bytes  # 4-byte encoded float32 cell
    path: tuple[bytes, ...]


@dataclass(frozen=True)
class InstructionOpening:
    """Everything needed to locally check instruction k:

    the instruction itself, the cell it wrote, and the cells it read
    (in operand order; literal operands need no opening).
    """

    index: int  # instruction index k (writes value cell num_inputs + k)
    instruction_payload: bytes
    instruction_path: tuple[bytes, ...]
    write: ValueOpening
    reads: tuple[ValueOpening, ...]


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


@dataclass
class Transcript:
    """A record of one protocol run, for humans to read."""

    commitment: Commitment
    challenges: list[int] = field(default_factory=list)
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return all(c.ok for c in self.checks)

    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.ok]


# ---------------------------------------------------------------------------
# Prover.
# ---------------------------------------------------------------------------


class Prover:
    """Executes the program and commits to both tapes.

    `overrides` makes the prover dishonest: {cell_index: forged_value}. The
    forged value replaces the computed one, and later instructions honestly
    consume it (the canonical adversary of the counting argument -- exactly
    one instruction per override becomes locally inconsistent).
    """

    def __init__(
        self,
        program: Program,
        inputs: Sequence[float | np.float32],
        overrides: Mapping[int, float | np.float32] | None = None,
    ) -> None:
        self.program = program
        self.inputs = [np.float32(x) for x in inputs]
        self.cells = execute(program, self.inputs, overrides=overrides)
        self.instruction_tree = MerkleTree.build(
            LEAF_INSTRUCTION, [encode_instruction(i) for i in program.instructions]
        )
        self.value_tree = MerkleTree.build(
            LEAF_VALUE, [encode_cell(c) for c in self.cells]
        )

    def commit(self) -> Commitment:
        return Commitment(
            instruction_root=self.instruction_tree.root,
            value_root=self.value_tree.root,
            num_inputs=self.program.num_inputs,
            num_instructions=len(self.program.instructions),
            claimed_outputs=tuple(
                float(self.cells[i]) for i in self.program.output_indices
            ),
        )

    def open_value(self, index: int) -> ValueOpening:
        return ValueOpening(
            index=index,
            payload=encode_cell(self.cells[index]),
            path=self.value_tree.prove(index),
        )

    def open_instruction(self, k: int) -> InstructionOpening:
        instr = self.program.instructions[k]
        reads = tuple(
            self.open_value(op.index) for op in instr.operands if not op.is_literal
        )
        return InstructionOpening(
            index=k,
            instruction_payload=encode_instruction(instr),
            instruction_path=self.instruction_tree.prove(k),
            write=self.open_value(self.program.num_inputs + k),
            reads=reads,
        )


# ---------------------------------------------------------------------------
# Verifier.
# ---------------------------------------------------------------------------


class Verifier:
    """Holds the verifier's own re-trace of the public program and the public
    input. The prover's commitment arrives later, as the first protocol
    message (`receive_commitment`). All checks are local: they touch only
    the roots and individual openings.
    """

    def __init__(
        self,
        program: Program,
        inputs: Sequence[float | np.float32],
    ) -> None:
        self.program = program  # the verifier's OWN re-trace, not the prover's
        self.inputs = [np.float32(x) for x in inputs]
        self.commitment: Commitment | None = None  # set by receive_commitment
        # v0 provenance baseline: the verifier can recompute the entire
        # instruction root from its re-trace. (In the full design this exact
        # recomputation is replaced by a proved compilation / hierarchy.)
        self.expected_instruction_root = MerkleTree.build(
            LEAF_INSTRUCTION, [encode_instruction(i) for i in program.instructions]
        ).root

    # -- message 1: the prover's commitment ----------------------------------

    def receive_commitment(self, commitment: Commitment) -> None:
        """Store the prover's first message. Everything the verifier later
        checks is checked AGAINST this -- once received, the prover is bound.
        """
        self.commitment = commitment

    # -- global checks ------------------------------------------------------

    def check_provenance(self) -> CheckResult:
        assert self.commitment is not None, "no commitment received"
        ok = self.commitment.instruction_root == self.expected_instruction_root
        ok = ok and self.commitment.num_inputs == self.program.num_inputs
        ok = ok and self.commitment.num_instructions == len(self.program.instructions)
        return CheckResult(
            name="instruction provenance",
            ok=ok,
            detail="committed instruction root matches verifier's own re-trace"
            if ok
            else "committed instruction tape is NOT the compilation of the public program",
        )

    def check_boundaries(self, openings: dict[int, ValueOpening]) -> list[CheckResult]:
        """Exact (not sampled) checks of the input and output views."""
        assert self.commitment is not None, "no commitment received"
        results = []
        n = self.commitment.num_cells

        for i, x in enumerate(self.inputs):
            ok, why = self._check_value_opening(openings.get(i), i, n)
            if ok and openings[i].payload != encode_cell(x):
                ok, why = False, f"input cell {i} does not equal the public input"
            results.append(CheckResult(f"input cell {i}", ok, why))

        for j, out_index in enumerate(self.program.output_indices):
            claimed = np.float32(self.commitment.claimed_outputs[j])
            ok, why = self._check_value_opening(openings.get(out_index), out_index, n)
            if ok and openings[out_index].payload != encode_cell(claimed):
                ok, why = False, f"output cell {out_index} does not equal the claimed output"
            results.append(CheckResult(f"output cell {out_index}", ok, why))
        return results

    # -- sampling -----------------------------------------------------------

    def draw_challenges(
        self,
        num_samples: int,
        rng: random.Random | None = None,
    ) -> list[int]:
        """Uniform iid instruction indices. MUST be drawn after the
        commitment is received -- unpredictability is what makes the
        (1 - L/N)^s bound real. `rng` (a random.Random) is for reproducible
        experiments; default is the OS CSPRNG.
        """
        assert self.commitment is not None, "no commitment received"
        n = self.commitment.num_instructions
        if rng is None:
            return [secrets.randbelow(n) for _ in range(num_samples)]
        return [rng.randrange(n) for _ in range(num_samples)]

    # -- the local check ----------------------------------------------------

    def check_instruction(self, opening: InstructionOpening) -> CheckResult:
        assert self.commitment is not None, "no commitment received"
        name = f"instruction {opening.index}"
        k = opening.index
        m = self.commitment.num_inputs
        n_instr = self.commitment.num_instructions
        n_cells = self.commitment.num_cells

        # 1. The opened instruction is leaf k of the committed instruction tape.
        if not verify_leaf(
            self.commitment.instruction_root,
            LEAF_INSTRUCTION,
            k,
            opening.instruction_payload,
            opening.instruction_path,
            n_instr,
        ):
            return CheckResult(name, False, "instruction opening fails authentication")
        try:
            # decode_instruction enforces the operand invariant: literals may
            # appear only as the sole operand of `const`. A malicious
            # instruction smuggling a literal into any other primitive is
            # rejected here, before its semantics are ever consulted.
            instr = decode_instruction(opening.instruction_payload)
        except AssertionError as e:
            return CheckResult(name, False, f"malformed instruction: {e}")

        # 2. The write cell (index m + k) is on the committed value tape.
        write_index = m + k
        ok, why = self._check_value_opening(opening.write, write_index, n_cells)
        if not ok:
            return CheckResult(name, False, f"write cell: {why}")

        # 3. Every read is an authentic, EARLIER cell (append-only discipline).
        # Only `const` reaches the literal branch (decode enforced that); its
        # check below then verifies the write cell equals the immediate
        # bit-for-bit, via apply_primitive's identity.
        args: list[np.float32] = []
        reads = iter(opening.reads)
        for op in instr.operands:
            if op.is_literal:
                args.append(np.float32(op.literal))
                continue
            read = next(reads, None)
            if read is None or read.index != op.index:
                return CheckResult(name, False, "read openings do not match operands")
            if read.index >= write_index:
                return CheckResult(name, False, "instruction reads a future cell")
            ok, why = self._check_value_opening(read, read.index, n_cells)
            if not ok:
                return CheckResult(name, False, f"read cell {read.index}: {why}")
            args.append(decode_cell(read.payload))

        # 4. Recompute and compare bit-for-bit.
        expected = encode_cell(apply_primitive(instr.prim, args))
        if expected != opening.write.payload:
            return CheckResult(
                name,
                False,
                f"cell {write_index} is {opening.write.payload.hex()} but "
                f"{instr.describe()} gives {expected.hex()}",
            )
        return CheckResult(name, True, f"cell {write_index} = {instr.describe()} verified")

    def _check_value_opening(
        self, opening: ValueOpening | None, expected_index: int, num_cells: int
    ) -> tuple[bool, str]:
        assert self.commitment is not None, "no commitment received"
        if opening is None or opening.index != expected_index:
            return False, f"missing opening for cell {expected_index}"
        if not verify_leaf(
            self.commitment.value_root,
            LEAF_VALUE,
            opening.index,
            opening.payload,
            opening.path,
            num_cells,
        ):
            return False, f"cell {expected_index} fails authentication"
        return True, "ok"


# ---------------------------------------------------------------------------
# One full protocol run.
# ---------------------------------------------------------------------------


def run_protocol(
    prover: Prover,
    verifier: Verifier,
    num_samples: int,
    rng: random.Random | None = None,
    challenges: list[int] | None = None,
) -> Transcript:
    """Orchestrate the full interaction, from the top:

    message 1: the prover commits, the verifier receives the commitment;
    message 2: the verifier's challenges (boundary indices, then s sampled
               instruction indices -- drawn only after the roots are fixed);
    message 3: the prover's openings, which the verifier checks locally
               (provenance, exact boundary checks, sampled instruction checks).

    `challenges` may be supplied explicitly (tests); otherwise the verifier
    draws them itself.
    """
    commitment = prover.commit()
    verifier.receive_commitment(commitment)
    transcript = Transcript(commitment=commitment)

    transcript.checks.append(verifier.check_provenance())

    boundary_indices = list(range(commitment.num_inputs)) + list(
        verifier.program.output_indices
    )
    boundary_openings = {i: prover.open_value(i) for i in boundary_indices}
    transcript.checks.extend(verifier.check_boundaries(boundary_openings))

    if challenges is None:
        challenges = verifier.draw_challenges(num_samples, rng=rng)
    transcript.challenges = list(challenges)
    for k in challenges:
        transcript.checks.append(verifier.check_instruction(prover.open_instruction(k)))

    return transcript
