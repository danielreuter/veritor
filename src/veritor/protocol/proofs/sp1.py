"""The SP1 zkVM backend: a generic checker guest driven through a host binary.

The Rust workspace in ``zk/sp1`` holds the guest (``guest``), the shared
codec/checker (``common``) and a host CLI (``host``).  This module drives the
host over a subprocess protocol -- one JSON object on stdout per command --
in three modes:

* ``execute``: run the guest in the SP1 executor without proving and report
  the verdict and *exact* cycle counts (per phase via ``cycle-tracker``).
  Fast; this is what tests and the cost table use.
* ``prove``: produce one core or compressed proof of a batch (minutes and
  tens of GB of RAM on a laptop; keep batches under ~10M cycles locally).
* ``verify``: check a saved proof against the guest's verifying key and the
  statement digest the verifier recomputed.

The guest commits ``sha256(encode_statement(statement)) || verdict`` as public
values, so ``verify`` is: the proof verifies under the pinned verifying key,
its statement digest is ours, and its verdict byte is ``1``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

from ..messages import ProtocolError
from .statement import Statement, Witness
from .wire import encode_statement, encode_witness, statement_digest

SP1_BACKEND = "sp1"
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_WORKSPACE = REPO_ROOT / "zk" / "sp1"
HOST_BINARY = "veritor-zk-host"

type ProveMode = Literal["core", "compressed"]


class SP1Unavailable(ProtocolError):
    """The SP1 toolchain or the host binary is not available."""


def _toolchain_path() -> str:
    home = Path.home()
    extra = [str(home / ".sp1" / "bin"), str(home / ".cargo" / "bin")]
    return os.pathsep.join([*extra, os.environ.get("PATH", "")])


def _toolchain_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PATH"] = _toolchain_path()
    return env


def sp1_toolchain_available() -> bool:
    """Whether ``cargo prove --version`` runs (the SP1 toolchain is installed)."""

    cargo = shutil.which("cargo", path=_toolchain_path())
    if cargo is None:
        return False
    try:
        result = subprocess.run(
            [cargo, "prove", "--version"],
            capture_output=True,
            env=_toolchain_env(),
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


@dataclass(frozen=True, slots=True)
class ExecutionReport:
    """What ``execute`` measured for one batch."""

    verdict: bool
    statement_digest: bytes
    total_cycles: int
    cycle_tracker: Mapping[str, int]
    gas: int | None
    syscalls: Mapping[str, int]
    execute_seconds: float

    @property
    def phase(self) -> Mapping[str, int]:
        """Cycles per guest phase: ``io``, ``digest``, ``parse``, ``merkle``, ``gates``."""

        return self.cycle_tracker


@dataclass(frozen=True, slots=True)
class ProofReport:
    """What ``prove`` measured for one batch, besides the proof itself."""

    execution: ExecutionReport
    mode: str
    proof_bytes: int
    shards: int
    vk_hash: str
    setup_seconds: float
    prove_seconds: float
    verify_seconds: float


@dataclass
class SP1Backend:
    """Drive the ``veritor-zk-host`` binary; ``prove`` returns the saved proof file's bytes.

    ``host`` is the binary; when ``None`` it is looked up under
    ``workspace/target/release`` and built with ``cargo build --release`` if
    ``build`` is set.  ``vk_hash`` pins the guest's verifying key: ``verify``
    rejects a proof whose key differs.  ``last_report`` keeps the most recent
    execution/proof report for callers that measure.
    """

    host: Path | None = None
    workspace: Path = DEFAULT_WORKSPACE
    mode: ProveMode = "core"
    build: bool = True
    vk_hash: str | None = None
    timeout: float | None = None
    last_report: ExecutionReport | ProofReport | None = field(
        default=None, init=False, repr=False
    )

    backend_id = SP1_BACKEND

    # -- toolchain --------------------------------------------------------------

    def binary(self) -> Path:
        """The host binary, building it on demand."""

        if self.host is not None:
            if not Path(self.host).exists():
                raise SP1Unavailable(f"host binary {self.host} does not exist")
            return Path(self.host)
        candidate = self.workspace / "target" / "release" / HOST_BINARY
        if candidate.exists():
            return candidate
        if not self.build:
            raise SP1Unavailable(f"{candidate} is not built")
        if not sp1_toolchain_available():
            raise SP1Unavailable("the SP1 toolchain (cargo prove) is not installed")
        result = subprocess.run(
            ["cargo", "build", "--release", "-p", HOST_BINARY],
            cwd=self.workspace,
            env=_toolchain_env(),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0 or not candidate.exists():
            raise SP1Unavailable(
                f"building the SP1 host failed:\n{result.stderr[-4000:]}"
            )
        return candidate

    def _run(self, *args: str) -> dict[str, object]:
        command = [str(self.binary()), *args]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                env=_toolchain_env(),
                timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise SP1Unavailable(
                f"{args[0]} timed out after {self.timeout}s"
            ) from error
        if result.returncode != 0:
            raise ProtocolError(
                f"veritor-zk-host {args[0]} failed ({result.returncode}):\n{result.stderr[-4000:]}"
            )
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            raise ProtocolError(f"veritor-zk-host {args[0]} printed no result")
        try:
            document = json.loads(lines[-1])
        except json.JSONDecodeError as error:
            raise ProtocolError(
                f"veritor-zk-host {args[0]} printed no JSON result"
            ) from error
        if type(document) is not dict:
            raise ProtocolError("veritor-zk-host result is not an object")
        return document

    def info(self) -> dict[str, object]:
        """The guest's identity: ELF digest and verifying-key hash."""

        return self._run("info")

    # -- the three modes ------------------------------------------------------

    @staticmethod
    def _counters(document: Mapping[str, object], key: str) -> dict[str, int]:
        table = document.get(key, {})
        if not isinstance(table, Mapping):
            raise ProtocolError(f"veritor-zk-host {key} is not an object")
        return {str(name): int(str(value)) for name, value in table.items()}

    @classmethod
    def _execution(cls, document: Mapping[str, object]) -> ExecutionReport:
        gas = document.get("gas")
        return ExecutionReport(
            verdict=bool(document["verdict"]),
            statement_digest=bytes.fromhex(str(document["statement_digest"])),
            total_cycles=int(str(document["total_cycles"])),
            cycle_tracker=cls._counters(document, "cycle_tracker"),
            gas=None if gas is None else int(str(gas)),
            syscalls=cls._counters(document, "syscalls"),
            execute_seconds=float(str(document.get("execute_seconds", 0.0))),
        )

    def execute(self, statement: Statement, witness: Witness) -> ExecutionReport:
        """Run the guest without proving: the verdict and exact cycle counts."""

        witness.for_statement(statement)
        with TemporaryDirectory(prefix="veritor-sp1-") as directory:
            root = Path(directory)
            statement_path = root / "statement.bin"
            witness_path = root / "witness.bin"
            statement_path.write_bytes(encode_statement(statement))
            witness_path.write_bytes(encode_witness(witness))
            document = self._run(
                "execute",
                "--statement",
                str(statement_path),
                "--witness",
                str(witness_path),
            )
        report = self._execution(document)
        if report.statement_digest != statement_digest(statement):
            raise ProtocolError("the guest committed a digest of another statement")
        self.last_report = report
        return report

    def prove(self, statement: Statement, witness: Witness) -> bytes:
        witness.for_statement(statement)
        with TemporaryDirectory(prefix="veritor-sp1-") as directory:
            root = Path(directory)
            statement_path = root / "statement.bin"
            witness_path = root / "witness.bin"
            proof_path = root / "proof.bin"
            statement_path.write_bytes(encode_statement(statement))
            witness_path.write_bytes(encode_witness(witness))
            document = self._run(
                "prove",
                "--statement",
                str(statement_path),
                "--witness",
                str(witness_path),
                "--out",
                str(proof_path),
                "--mode",
                self.mode,
            )
            proof = proof_path.read_bytes()
        execution = self._execution(document)
        if not execution.verdict:
            raise ProtocolError(
                "the guest rejected the batch; there is nothing to prove"
            )
        vk_hash = str(document["vk_hash"])
        if self.vk_hash is not None and vk_hash != self.vk_hash:
            raise ProtocolError(
                f"the host's verifying key {vk_hash} is not the pinned {self.vk_hash}"
            )
        self.last_report = ProofReport(
            execution=execution,
            mode=str(document["mode"]),
            proof_bytes=int(str(document["proof_bytes"])),
            shards=int(str(document["shards"])),
            vk_hash=vk_hash,
            setup_seconds=float(str(document["setup_seconds"])),
            prove_seconds=float(str(document["prove_seconds"])),
            verify_seconds=float(str(document["verify_seconds"])),
        )
        return proof

    def verify(self, statement: Statement, proof: bytes) -> bool:
        if type(proof) is not bytes or not proof:
            return False
        with TemporaryDirectory(prefix="veritor-sp1-") as directory:
            root = Path(directory)
            statement_path = root / "statement.bin"
            proof_path = root / "proof.bin"
            statement_path.write_bytes(encode_statement(statement))
            proof_path.write_bytes(proof)
            try:
                document = self._run(
                    "verify",
                    "--proof",
                    str(proof_path),
                    "--statement",
                    str(statement_path),
                )
            except ProtocolError:
                # An unloadable proof file is a rejected proof, not a broken verifier.
                return False
        if self.vk_hash is not None and str(document.get("vk_hash")) != self.vk_hash:
            return False
        return (
            bool(document.get("ok"))
            and bool(document.get("statement_match"))
            and bool(document.get("verdict"))
        )


def describe_batch(statement: Statement) -> dict[str, int]:
    """Size counters of a batch, for pairing with an :class:`ExecutionReport`."""

    obligations = statement.obligations
    positions = sum(len(item.positions) for item in obligations)
    gates = sum(statement.program(item.kind).size for item in obligations)
    levels = sum(
        _depth(item.commitments[ref.commitment].count)
        for item in obligations
        for ref in item.positions
    )
    return {
        "obligations": len(obligations),
        "positions": positions,
        "gates": gates,
        "merkle_levels": levels,
        "statement_bytes": len(encode_statement(statement)),
    }


def _depth(count: int) -> int:
    return 0 if count <= 1 else (count - 1).bit_length()


def gate_histogram(statement: Statement) -> dict[str, int]:
    """How many gates of each op the batch checks (source gates included)."""

    histogram: dict[str, int] = {}
    for obligation in statement.obligations:
        program = statement.program(obligation.kind)
        for gate in program.gates:
            histogram[gate.op] = histogram.get(gate.op, 0) + 1
    return histogram


def openings_bytes(witness: Witness) -> int:
    return len(encode_witness(witness))


__all__ = [
    "DEFAULT_WORKSPACE",
    "SP1_BACKEND",
    "ExecutionReport",
    "ProofReport",
    "SP1Backend",
    "SP1Unavailable",
    "describe_batch",
    "gate_histogram",
    "openings_bytes",
    "sp1_toolchain_available",
]
