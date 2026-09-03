"""Cross-language test vectors: generated here from the Python reference, asserted in Rust.

``zk/sp1/common/tests/vectors.json`` pins the hash framing (`merkle.py`,
`identity.py`), the gate-set digests and semantics (`core/gates.py`) and the
statement/witness codec plus the checker's verdict on a real batch.  The Rust
crate's ``tests/vectors.rs`` reads the file; this test regenerates it and fails
if the committed file is stale.  Set ``VERITOR_WRITE_VECTORS=1`` to rewrite it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from veritor.core import GateSet, make_isa_gate_set, make_word_gate_set
from veritor.core.identity import tagged_sha256
from veritor.protocol import CommitmentDomain, ProverSession, VerifierSession
from veritor.protocol.merkle import _hash, _uint
from veritor.protocol.proofs import (
    Statement,
    TransparentBackend,
    Witness,
    encode_statement,
    encode_witness,
    statement_digest,
)

from .conftest import RECORDING_BACKEND, RecordingBackend

VECTORS = Path(__file__).resolve().parents[4] / "zk" / "sp1" / "common" / "tests" / "vectors.json"

type Openings = tuple[tuple[bytes, tuple[bytes, ...]], ...]


def framing_vectors() -> list[dict[str, object]]:
    cases = [
        (b"leaf", []),
        (b"node", [b"", b"x"]),
        (b"domain", [bytes(range(32)), b"\x01", b"\xff" * 32, b"\x01\x00"]),
        (b"t", [b"a" * 100, b"b" * 300]),
    ]
    return [
        {
            "tag": tag.hex(),
            "parts": [part.hex() for part in parts],
            "digest": _hash(tag, *parts).hex(),
        }
        for tag, parts in cases
    ]


def uint_vectors() -> list[dict[str, object]]:
    return [
        {"value": value, "bytes": _uint(value).hex()}
        for value in (0, 1, 255, 256, 65535, 1 << 40)
    ]


def tagged_vectors() -> list[dict[str, object]]:
    cases = (("veritor/gate-set/v1", b"{}"), ("x", b""), ("veritor/test", b"payload"))
    return [
        {"tag": tag, "payload": payload.hex(), "digest": tagged_sha256(tag, payload)}
        for tag, payload in cases
    ]


def domain_vectors(verifier: VerifierSession) -> list[dict[str, object]]:
    out = []
    for owner, (domain, commitment) in sorted(verifier._commitments.items()):
        assert isinstance(domain, CommitmentDomain)
        # an empty domain (a replay unit with no interior) still has a domain id and empty root
        rank = max(0, min(1, domain.count - 1))
        position = int(domain.positions.unrank(rank)) if domain.count else 0
        value = b"\x00\x2a"
        left, right = bytes(32), bytes([1]) * 32
        out.append(
            {
                "owner": owner,
                "binding": domain.binding.hex(),
                "identity": domain.positions.identity_digest,
                "count": domain.count,
                "domain_id": domain.domain_id.hex(),
                "leaf": {
                    "rank": rank,
                    "position": position,
                    "schema": "u16",
                    "value": value.hex(),
                    "digest": domain.leaf(rank, position, "u16", value).hex(),
                },
                "node": {
                    "level": 3,
                    "index": 5,
                    "left": left.hex(),
                    "right": right.hex(),
                    "digest": domain.node(3, 5, left, right).hex(),
                },
                "empty_root": domain.empty_root().hex(),
                "root": commitment.root.hex(),
            }
        )
    return out


PAIRS = (
    (0, 0),
    (1, 2),
    (65535, 1),
    (40000, 40000),
    (7, 3),
    (3, 7),
    (65535, 65535),
    (1, 16),
    (1, 15),
    (65535, 17),
    (200, 100),
)


def gate_set_vector(gate_set: GateSet) -> dict[str, object]:
    width = {gate.width for gate in gate_set}.pop()
    evaluations = [
        {"op": gate.name, "a": a, "b": b, "out": gate.evaluate((a, b))}
        for gate in gate_set
        if gate.source is None
        for a, b in PAIRS
        if max(a, b) < 1 << width
    ]
    return {
        "id": gate_set.id,
        "width": width,
        "manifest": json.dumps(gate_set.manifest, sort_keys=True, separators=(",", ":")),
        "digest": gate_set.digest,
        "evaluations": evaluations,
    }


def with_opening(witness: Witness, target: tuple[int, int], opening: tuple[bytes, tuple[bytes, ...]]) -> Witness:
    return Witness(
        tuple(
            tuple(opening if (i, j) == target else item for j, item in enumerate(items))
            for i, items in enumerate(witness.obligations)
        )
    )


def batch_vector(statement: Statement, witness: Witness) -> dict[str, object]:
    statement_bytes = encode_statement(statement)
    witness_bytes = encode_witness(witness)
    value, path = witness.obligations[0][0]
    flipped_value = with_opening(witness, (0, 0), (bytes(b ^ 1 for b in value), path))
    target = next(
        (i, j)
        for i, items in enumerate(witness.obligations)
        for j, (_, p) in enumerate(items)
        if p
    )
    value, path = witness.obligations[target[0]][target[1]]
    broken_path = with_opening(witness, target, (value, (bytes(32), *path[1:])))
    return {
        "statement": statement_bytes.hex(),
        "witness": witness_bytes.hex(),
        "statement_digest": statement_digest(statement).hex(),
        "obligations": len(statement.obligations),
        "kinds": len(statement.kinds),
        "gate_set_id": statement.gate_set_id,
        "width": statement.width,
        "verdict": True,
        "rejected_witnesses": [
            {"name": "flipped-value", "witness": encode_witness(flipped_value).hex()},
            {"name": "broken-path", "witness": encode_witness(broken_path).hex()},
            {"name": "truncated", "witness": witness_bytes[:-1].hex()},
        ],
    }


def generate(compiled, expect, honest_values, model_weights) -> dict[str, object]:
    recording = RecordingBackend(TransparentBackend(compiled.circuit.gate_set, compiled))
    verifier = VerifierSession(
        expect(backend=RECORDING_BACKEND, session_id=b"vectors"), compiled, backend=recording
    )
    prover = ProverSession(
        compiled, verifier.header, honest_values, weight_tree=model_weights[1], backend=recording
    )
    replay = verifier.receive_boundary(prover.boundary())
    sample = verifier.receive_interiors(prover.interiors(replay))
    assert verifier.receive_evidence(prover.evidence(sample)).accepted
    ((statement, witness),) = recording.proved
    return {
        "_comment": "Generated by tests/veritor/protocol/proofs/test_vectors.py; do not edit.",
        "framing": framing_vectors(),
        "uint": uint_vectors(),
        "tagged_sha256": tagged_vectors(),
        "domains": domain_vectors(verifier),
        "gate_sets": [
            gate_set_vector(gate_set)
            for gate_set in (make_word_gate_set(16), make_isa_gate_set(16), make_word_gate_set(8))
        ],
        "batch": batch_vector(statement, witness),
    }


def test_vectors_are_current(compiled, expect, honest_values, model_weights) -> None:
    document = generate(compiled, expect, honest_values, model_weights)
    generated = json.dumps(document, indent=1, sort_keys=True) + "\n"
    if os.environ.get("VERITOR_WRITE_VECTORS") == "1":
        VECTORS.parent.mkdir(parents=True, exist_ok=True)
        VECTORS.write_text(generated)
    if not VECTORS.exists():
        pytest.fail(f"{VECTORS} is missing; run with VERITOR_WRITE_VECTORS=1")
    assert VECTORS.read_text() == generated, (
        "zk/sp1/common/tests/vectors.json is stale; regenerate with VERITOR_WRITE_VECTORS=1"
    )
