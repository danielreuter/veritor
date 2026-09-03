"""Component 7: compile determinism and canonical encoding.

The same description bytes give the same digest and the same ``(C, I)``;
non-canonical description bytes are rejected before any compile work; the
description limits bound the verifier's compile work; marks are part of the
digest; transcripts are canonical bytes (NONCANONICAL_TRANSCRIPT).

Compile-work bounds already covered elsewhere and cited here rather than
repeated: ``tests/veritor/compile/test_out_runs.py::
test_interfaces_resolving_to_too_many_runs_are_rejected_without_doing_the_work``,
``::test_the_total_number_of_runs_over_a_description_is_capped``,
``::test_admission_does_not_scale_with_the_input_count`` and
``tests/veritor/compile/test_description.py::test_canonical_encoding_is_enforced``,
``::test_limits_bound_every_summary_without_unrolling``.
"""

from __future__ import annotations

import json
import time

import pytest

from veritor.compile import CompileError, Compiler
from veritor.compile.description import parse_description
from veritor.constructors import Tracer
from veritor.core import CompilationLimits, make_word_gate_set
from veritor.protocol import (
    ProtocolError,
    ProverSession,
    VerificationCode,
    decode_transcript,
)


def test_same_description_bytes_give_the_same_digest_and_layout(sec):
    gate_set, description = sec.chain_description(2, 2)
    first = Compiler(gate_set).compile(description, [0, 0])
    second = Compiler(make_word_gate_set(8)).compile(bytes(description), [0, 0])
    assert first.digest == second.digest
    assert [
        (first.circuit[a].op, first.circuit[a].args, first.circuit[a].width)
        for a in range(first.circuit.n)
    ] == [
        (second.circuit[a].op, second.circuit[a].args, second.circuit[a].width)
        for a in range(second.circuit.n)
    ]
    assert first.index.kinds() == second.index.kinds()
    assert first.index.digest == second.index.digest
    # the digest binds the gate set as well: the same bytes over 4-bit words are another (C, I)
    narrow = Compiler(make_word_gate_set(4)).compile(description, [0, 0])
    assert narrow.digest != first.digest
    # and it is the digest of the canonical bytes (with the gate set), nothing environmental
    assert (
        parse_description(description, gate_set).digest
        == parse_description(description, gate_set).digest
    )


def test_reencoded_description_bytes_are_rejected_before_any_compile_work(sec):
    gate_set, description = sec.chain_description(2, 2)
    document = json.loads(description)
    pretty = json.dumps(document, indent=1).encode()
    with pytest.raises(CompileError, match="canonically serialized"):
        parse_description(pretty, gate_set)
    reordered = json.dumps(document, separators=(",", ":"), sort_keys=False).encode()
    swapped = json.dumps(
        {k: document[k] for k in reversed(list(document))}, separators=(",", ":")
    ).encode()
    assert swapped != description
    with pytest.raises(CompileError, match="canonically serialized"):
        parse_description(swapped, gate_set)
    if reordered != description:
        with pytest.raises(CompileError, match="canonically serialized"):
            parse_description(reordered, gate_set)
    with pytest.raises(CompileError, match="duplicate JSON key"):
        parse_description(description[:-1] + b',"version":2}', gate_set)
    with pytest.raises(CompileError, match="not valid UTF-8 JSON"):
        parse_description(b"\xff" + description, gate_set)
    # oversize bytes are refused before parsing
    with pytest.raises(CompileError, match="max_description_bytes"):
        parse_description(
            description,
            gate_set,
            CompilationLimits(max_description_bytes=len(description) - 1),
        )


def test_changing_a_mark_changes_the_digest_and_the_header(sec):
    """The same gates at the same addresses under other units: another ``(C, I)``."""

    fine, wide = sec.Model(2, 2), sec.Model(2, 2, wide_units=True)
    assert [
        (fine.circuit[a].op, fine.circuit[a].args) for a in range(fine.circuit.n)
    ] == [(wide.circuit[a].op, wide.circuit[a].args) for a in range(wide.circuit.n)]
    assert fine.index.verification_unit_count != wide.index.verification_unit_count
    assert fine.compiled.digest != wide.compiled.digest
    assert fine.index.digest != wide.index.digest
    fine_header = fine.header(fine.expectation())
    wide_header = wide.header(wide.expectation())
    assert fine_header.digest != wide_header.digest
    # a prover holding the other marking cannot even start under this header ...
    with pytest.raises(ProtocolError, match="different compiled circuit"):
        ProverSession(wide.compiled, fine_header, wide.values, weight_tree=wide.tree)
    # ... and a transcript naming the other compiled digest is EXPECTATION_MISMATCH
    expectation = fine.expectation()
    run = fine.run(expectation, fine.values)

    def rename(document: dict) -> None:
        document["header"]["compiled_digest"] = wide.compiled.digest

    report = fine.verify(sec.mutate_transcript(run.transcript, rename), expectation)
    assert report.code == VerificationCode.EXPECTATION_MISMATCH


@pytest.mark.parametrize(
    ("label", "rewrite", "code"),
    [
        (
            "whitespace",
            lambda data: data.replace(b",", b", "),
            VerificationCode.NONCANONICAL_TRANSCRIPT,
        ),
        ("uppercase-hex", None, VerificationCode.NONCANONICAL_TRANSCRIPT),
        ("unreduced-fraction", None, VerificationCode.NONCANONICAL_TRANSCRIPT),
        ("unsorted-keys", None, VerificationCode.NONCANONICAL_TRANSCRIPT),
        ("float", None, VerificationCode.MALFORMED_TRANSCRIPT),
        ("extra-key", None, VerificationCode.MALFORMED_TRANSCRIPT),
        ("missing-key", None, VerificationCode.MALFORMED_TRANSCRIPT),
        ("negative-count", None, VerificationCode.MALFORMED_TRANSCRIPT),
        ("other-version", None, VerificationCode.MALFORMED_TRANSCRIPT),
        ("not-json", lambda data: data[:-1], VerificationCode.MALFORMED_TRANSCRIPT),
    ],
)
def test_transcript_with_a_noncanonical_or_malformed_encoding_is_rejected(
    honest_run, model, sec, label, rewrite, code
):
    run, expectation = honest_run
    canonical = sec.encode_transcript(run.transcript)
    assert model.verify(canonical, expectation).accepted
    if rewrite is not None:
        data = rewrite(canonical)
    else:
        document = json.loads(canonical)
        if label == "uppercase-hex":
            document["header"]["session_id"] = document["header"]["session_id"].upper()
        elif label == "unreduced-fraction":
            document["header"]["policy"]["q"] = [2, 2]
        elif label == "unsorted-keys":
            document = {key: document[key] for key in reversed(list(document))}
        elif label == "float":
            document["boundary"]["commitment"]["count"] = 6.0
        elif label == "extra-key":
            document["header"]["note"] = "x"
        elif label == "missing-key":
            del document["header"]["eta"]
        elif label == "negative-count":
            document["boundary"]["commitment"]["count"] = -6
        elif label == "other-version":
            document["version"] = f"{document['version']}-other"
        data = json.dumps(
            document, separators=(",", ":"), sort_keys=label != "unsorted-keys"
        ).encode()
    assert data != canonical
    report = model.verify(data, expectation)
    assert report.code == code, label
    # the interactive verifier is unaffected: the canonical bytes still verify
    assert model.verify(canonical, expectation).accepted


def test_canonical_bytes_are_unique_per_transcript(honest_run, sec):
    """Decode(encode(t)) == t and encode is a function of the transcript object alone."""

    run, _ = honest_run
    canonical = sec.encode_transcript(run.transcript)
    decoded = decode_transcript(canonical)
    assert decoded == run.transcript and sec.encode_transcript(decoded) == canonical


def test_a_description_of_a_trillion_gates_compiles_in_bounded_time():
    """The index is lazy: compile work is proportional to the description, not the circuit."""

    gate_set = make_word_gate_set(8)
    tracer = Tracer(gate_set)
    add = tracer.gate("add")

    @tracer.definition(input_count=1, key="double", role="verification")
    def double(v):
        return add(v[0], v[0])

    @tracer.definition(input_count=0, key="huge", role="replay")
    def huge(_v):
        x = tracer.inputs(1)
        return tracer.repeat(10**12, double, x[0])

    started = time.perf_counter()
    compiled = Compiler(gate_set).compile(tracer.serialize(huge), [1])
    assert time.perf_counter() - started < 1.0
    assert compiled.circuit.n == 10**12 + 1
    assert compiled.index.verification_unit_count == 10**12 + 1


def test_nesting_deeper_than_the_limit_is_a_compile_error():
    gate_set = make_word_gate_set(8)
    tracer = Tracer(gate_set)
    add = tracer.gate("add")
    depth = CompilationLimits().max_depth + 2

    current = tracer.definition(input_count=1, key=("level", 0), role="verification")(
        lambda v: add(v[0], v[0])
    )
    for level in range(1, depth):
        previous = current
        current = tracer.definition(input_count=1, key=("level", level))(
            lambda v, p=previous: p(v[0])
        )
    deepest = current

    @tracer.definition(input_count=0, key="deep-root", role="replay")
    def root(_v):
        return deepest(tracer.inputs(1)[0])

    started = time.perf_counter()
    with pytest.raises(CompileError, match="nesting depth"):
        Compiler(gate_set).compile(tracer.serialize(root), [1])
    assert time.perf_counter() - started < 1.0
