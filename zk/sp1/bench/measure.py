"""Microbenchmarks for the SP1 checker guest: the constants behind ``proofs/costs.py``.

Synthetic but *valid* batches (real Merkle trees with our framing, real gate
semantics) isolate one cost at a time; every number is an exact cycle count
from ``veritor-zk-host execute`` (no proving).  Run from the repo root::

    .venv/bin/python zk/sp1/bench/measure.py [--json out.json]

Fits (all by two-point slopes, which is exact for these linear costs):

* per gate, by op: the ``gates`` tracker of one obligation whose kind is a
  chain of ``N`` gates of that op, ``N = 8`` vs ``N = 136``;
* per Merkle level: the ``merkle`` tracker of one obligation with a fixed
  number of positions in a domain of depth 4 vs depth 12;
* per opened position (leaf hash + decode, no levels): the ``merkle`` slope
  per position at fixed depth minus depth * per-level;
* per obligation: the slopes of every tracker in the number of obligations
  (1 vs 65 one-gate obligations);
* per batch fixed: the floor of a one-obligation, one-gate, depth-1 batch and
  the intercepts of the per-obligation fit;
* parse per byte: the ``parse`` slope against statement + witness bytes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from veritor.core import Position, make_isa_gate_set
from veritor.core.indexed import ExplicitIndexedDomain
from veritor.protocol import CommitmentDomain, MerkleTree
from veritor.protocol.proofs import (
    CommitmentRef,
    GateOp,
    KindProgram,
    Obligation,
    PositionRef,
    SP1Backend,
    Statement,
    Witness,
    encode_statement,
    encode_witness,
    make_statement,
)
from veritor.protocol.proofs.derive import statement_width

GATE_SET = make_isa_gate_set(16)
WIDTH = statement_width(GATE_SET)
SCHEMA = f"u{WIDTH}"
SESSION = b"\x11" * 32
COMPILED = b"\x22" * 32
BINDING = b"\x33" * 32
IDENTITY = b"\x44" * 32


def word(value: int) -> bytes:
    return value.to_bytes(WIDTH // 8, "big")


def chain_program(op: str, gates: int, kind_seed: int) -> KindProgram:
    """A kind with two ports and ``gates`` gates: g0 = op(p0, p1), gj = op(p0, g(j-1))."""

    ops = tuple(
        GateOp(
            op,
            (("port", 0), ("port", 1)) if j == 0 else (("port", 0), ("local", j - 1)),
        )
        for j in range(gates)
    )
    return KindProgram(kind_seed.to_bytes(32, "big"), gates, (0, 1), ops)


def evaluate_chain(op: str, gates: int, a: int, b: int) -> list[int]:
    gate = GATE_SET[op]
    values: list[int] = []
    previous = b
    for _ in range(gates):
        previous = gate.evaluate((a, previous))
        values.append(previous)
    return values


def build_batch(
    op: str, gates: int, obligations: int, depth: int, *, a: int = 3, b: int = 5
) -> tuple[Statement, Witness]:
    """``obligations`` copies of one chain kind in one commitment of ``2**depth`` positions."""

    per_copy = gates + 2
    needed = obligations * per_copy
    count = max(1 << depth, needed)
    if count != 1 << depth and depth:
        raise ValueError(f"depth {depth} holds {1 << depth} positions, need {needed}")
    domain = CommitmentDomain(
        BINDING, 0, ExplicitIndexedDomain(Position(i) for i in range(count))
    )
    values: dict[int, bytes] = {}
    for copy in range(obligations):
        base = copy * per_copy
        values[base] = word(a)
        values[base + 1] = word(b)
        for j, value in enumerate(evaluate_chain(op, gates, a, b)):
            values[base + 2 + j] = word(value)
    for position in range(needed, count):
        values[position] = word(0)
    tree = MerkleTree(domain, values, lambda _p: SCHEMA)
    ref = CommitmentRef(0, domain.domain_id, tree.commitment.root, count)
    program = chain_program(op, gates, kind_seed=1)
    items: list[Obligation] = []
    witness: list[tuple[tuple[bytes, tuple[bytes, ...]], ...]] = []
    for copy in range(obligations):
        base = copy * per_copy
        positions = tuple(
            PositionRef(0, base + k, base + k, SCHEMA) for k in range(per_copy)
        )
        items.append(
            Obligation(
                SESSION,
                COMPILED,
                copy,
                0,
                program.kind,
                (ref,),
                positions,
                (0, 1),
                tuple(range(2, per_copy)),
            )
        )
        witness.append(
            tuple(
                (opening.value, opening.path)
                for opening in (tree.open(base + k) for k in range(per_copy))
            )
        )
    statement = make_statement(
        GATE_SET.id, bytes.fromhex(GATE_SET.digest), WIDTH, [program], items
    )
    return statement, Witness(tuple(witness))


class Meter:
    def __init__(self, backend: SP1Backend) -> None:
        self.backend = backend
        self.runs: list[dict[str, object]] = []

    def run(self, label: str, statement: Statement, witness: Witness) -> dict[str, int]:
        started = time.time()
        report = self.backend.execute(statement, witness)
        if not report.verdict:
            raise RuntimeError(
                f"{label}: the guest rejected a batch that should verify"
            )
        row = {
            "label": label,
            "obligations": len(statement.obligations),
            "positions": sum(len(item.positions) for item in statement.obligations),
            "gate_count": sum(len(item.gates) for item in statement.obligations),
            "statement_bytes": len(encode_statement(statement)),
            "witness_bytes": len(encode_witness(witness)),
            "total": report.total_cycles,
            **{
                phase: report.cycle_tracker.get(phase, 0)
                for phase in ("io", "digest", "parse", "merkle", "gates")
            },
            "sha_compress": report.syscalls.get("SHA_COMPRESS", 0),
            "wall_seconds": round(time.time() - started, 2),
        }
        self.runs.append(row)
        print(json.dumps(row), file=sys.stderr)
        return {k: int(v) for k, v in row.items() if isinstance(v, int)}


def slope(low: dict[str, int], high: dict[str, int], key: str, delta: int) -> float:
    return (high[key] - low[key]) / delta


def measure(meter: Meter) -> dict[str, object]:
    build: Callable[..., tuple[Statement, Witness]] = build_batch
    results: dict[str, object] = {}

    # -- per gate, by op (depth 8: 256 positions hold 138) ----------------------
    per_gate: dict[str, dict[str, float]] = {}
    for op in ("add", "mul", "sub", "lt", "eq", "shr"):
        a, b = (3, 5) if op != "shr" else (40000, 1)
        low = meter.run(f"{op}x8", *build(op, 8, 1, 8, a=a, b=b))
        high = meter.run(f"{op}x136", *build(op, 136, 1, 8, a=a, b=b))
        per_gate[op] = {
            "gates_phase": slope(low, high, "gates", 128),
            "merkle_phase": slope(low, high, "merkle", 128),
            "parse_phase": slope(low, high, "parse", 128),
            "total": slope(low, high, "total", 128),
        }
    results["per_gate"] = per_gate

    # -- per Merkle level (one obligation, 18 positions, depth 5 vs 13) ---------
    shallow = meter.run("add16@d5", *build("add", 16, 1, 5))
    deep = meter.run("add16@d13", *build("add", 16, 1, 13))
    positions = 18
    per_level = slope(shallow, deep, "merkle", 8 * positions)
    results["per_merkle_level"] = {
        "merkle_phase": per_level,
        "total": slope(shallow, deep, "total", 8 * positions),
        "sha_compress_per_level": slope(shallow, deep, "sha_compress", 8 * positions),
    }

    # -- per opened position: leaf hash + decode, at depth 8 --------------------
    per_position_at_8 = per_gate["add"]["merkle_phase"]
    results["per_position"] = {
        "merkle_phase_at_depth_8": per_position_at_8,
        "leaf_only": per_position_at_8 - 8 * per_level,
        "parse_phase": per_gate["add"]["parse_phase"],
    }

    # -- per obligation (1-gate kind, depth 8: 256 positions hold 65 x 3) -------
    one = meter.run("1ob", *build("add", 1, 1, 8))
    many = meter.run("65ob", *build("add", 1, 65, 8))
    per_obligation = {
        key: slope(one, many, key, 64)
        for key in ("total", "parse", "merkle", "gates", "digest")
    }
    per_obligation["statement_bytes"] = slope(one, many, "statement_bytes", 64)
    per_obligation["witness_bytes"] = slope(one, many, "witness_bytes", 64)
    results["per_obligation_1_gate_depth_8"] = per_obligation

    # -- per batch fixed --------------------------------------------------------
    floor = meter.run("floor", *build("add", 1, 1, 2))
    intercept = {
        key: one[key] - per_obligation[key]
        for key in ("total", "parse", "merkle", "gates")
    }
    results["per_batch"] = {
        "floor_total_1ob_1gate_depth2": floor["total"],
        "floor_phases": {
            k: floor[k] for k in ("io", "digest", "parse", "merkle", "gates")
        },
        "untracked": floor["total"]
        - sum(floor[k] for k in ("io", "digest", "parse", "merkle", "gates")),
        "intercept_from_obligation_fit": intercept,
    }

    # -- parse per byte ---------------------------------------------------------
    bytes_delta = (many["statement_bytes"] + many["witness_bytes"]) - (
        one["statement_bytes"] + one["witness_bytes"]
    )
    results["parse_per_byte"] = slope(one, many, "parse", bytes_delta)
    results["digest_per_statement_byte"] = slope(
        one, many, "digest", many["statement_bytes"] - one["statement_bytes"]
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--json", type=Path, help="write the raw runs and fitted constants here"
    )
    arguments = parser.parse_args()
    backend = SP1Backend()
    info = backend.info()
    meter = Meter(backend)
    results = measure(meter)
    document = {"guest": info, "constants": results, "runs": meter.runs}
    print(json.dumps(document["constants"], indent=1, sort_keys=True))
    if arguments.json:
        arguments.json.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
