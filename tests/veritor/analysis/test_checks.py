"""Check outputs carry no capacity: the fold, the reference cuts and the brute-forced union agree."""

from __future__ import annotations

import math
from fractions import Fraction

import pytest

from veritor.analysis import bound
from veritor.analysis.reference import (
    accepted_outputs,
    ancestor_bits,
    check_addresses,
    cover_bits,
    cut_bits,
    error_sets,
    out_bits,
    reach_bits,
    transcript_outputs,
)
from veritor.compile import Compiler
from veritor.constructors import Tracer
from veritor.core import Compiled, VerificationPolicy, make_word_gate_set
from veritor.research import Capacity, Compile

WIDTH = 2
GATES = make_word_gate_set(WIDTH)
POLICIES = [
    (VerificationPolicy(Fraction(1, 2), Fraction(1, 2)), Fraction(1, 4)),
    (VerificationPolicy(1, Fraction(1, 2)), Fraction(1, 8)),
    (VerificationPolicy(0, 1), Fraction(1, 2)),
    (VerificationPolicy(1, 1), Fraction(0)),
]


def fanin(checked: str) -> Compiled:
    """The paper's fan-in circuit over 2-bit cells with ``checked`` of its two outputs marked.

    ``h(u, v, w) = (u+v, v+w, u+w)`` twice, then ``tail`` folds the second
    application; the outputs are ``tail`` (the sum of all three inputs, times
    two) and ``b``, the first ``h``'s middle word.  On inputs ``1, 2, 3``
    those are ``0`` (``12 mod 4``) and ``1``.
    """

    tracer = Tracer(GATES)
    add = tracer.gate("add")

    @tracer.definition(input_count=3, key="h", role="verification")
    def h(v):
        return add(v[0], v[1]), add(v[1], v[2]), add(v[0], v[2])

    @tracer.definition(input_count=3, key="tail", role="verification")
    def tail(v):
        t = add(v[0], v[1])
        return add(t, v[2])

    @tracer.definition(input_count=0, key="first", role="replay")
    def first(_v):
        x = tracer.inputs(3)
        return h(x[0], x[1], x[2])

    @tracer.definition(input_count=3, key="rest", role="replay")
    def rest(v):
        p, q, r = h(v[0], v[1], v[2])
        return tail(p, q, r)

    @tracer.definition(input_count=0, key=("root", checked))
    def root(_v):
        a, b, c = first()
        folded = rest(a, b, c)
        if "tail" in checked:
            tracer.check(folded, 0)
        if "b" in checked:
            tracer.check(b, 1)
        return folded, b

    return Compiler(GATES).compile(tracer.serialize(root), [1, 2, 3])


@pytest.mark.parametrize("checked", ["", "tail", "b", "tail b"])
def test_the_fold_bounds_the_union_and_agrees_with_the_reference_cuts(
    checked: str,
) -> None:
    compiled = fanin(checked)
    circuit, index = compiled.circuit, compiled.index
    fixed = check_addresses(compiled)
    assert len(fixed) == len(checked.split())
    outputs = transcript_outputs(compiled, [1, 2, 3])
    checks = list(compiled.check_values())
    rows = {row.kind: row for row in compiled.kind_table().rows}
    root = index.root
    assert (
        rows[root.kind].out_bits
        == out_bits(circuit, root, fixed)
        == WIDTH * (2 - len(fixed))
    )
    for unit in range(index.verification_unit_count):
        node = index.verification_unit(unit)
        row = rows[node.kind]
        assert reach_bits(circuit, node, fixed) <= row.reach_bits
        assert ancestor_bits(circuit, node, fixed) <= row.ancestor_bits
    for errors in error_sets(index):
        assert (
            cut_bits(compiled, errors)
            <= cover_bits(compiled, errors)
            <= rows[root.kind].out_bits
        )
    for policy, eta in POLICIES:
        union = len(accepted_outputs(outputs, policy, eta, checks))
        result = bound(compiled, policy, eta)
        assert math.log2(union) <= result.bits + 1e-6
        assert result.bits <= WIDTH * (2 - len(fixed)) + 1e-6
    # every accepted output holds the constants at the check positions
    everything = accepted_outputs(
        outputs, VerificationPolicy(0, 1), Fraction(1, 2), checks
    )
    assert len(everything) == 1 << (WIDTH * (2 - len(fixed)))
    assert all(
        output[ordinal] == value for output in everything for ordinal, value in checks
    )


def test_a_circuit_whose_outputs_are_all_checks_bounds_to_the_advice_alone() -> None:
    compiled = fanin("tail b")
    for policy, eta in POLICIES:
        assert bound(compiled, policy, eta).bits == 0.0
    for row in compiled.kind_table().rows:
        assert row.cut_bits == 0 and row.reach_bits == 0 and row.ancestor_bits == 0

    class G:
        digest = "ab" * 32

        def __call__(self, x: object, a: bytes) -> tuple[bytes, tuple[int, ...]]:
            tracer = Tracer(GATES)
            add = tracer.gate("add")
            double = tracer.definition(
                input_count=1, key="double", role="verification"
            )(lambda v: add(v[0], v[0]))

            @tracer.definition(input_count=0, key="unit", role="replay")
            def unit(_v):
                return double(tracer.inputs(1))

            @tracer.definition(input_count=0, key="root")
            def root(_v):
                return tracer.check(unit(), 2)

            return tracer.serialize(root), (1,)

        def advice_bits(self, x: object, a: bytes) -> int:
            return 5

    compilation = Compile(G(), None, b"\xa8", GATES, max_advice_bits=5)
    assert compilation.advice_bits == 5 and compilation.compiled.checks
    assert Capacity(compilation, VerificationPolicy(0, 1), Fraction(1, 2)) == 5.0
    assert Capacity(compilation, VerificationPolicy(1, 1), 0) == 5.0
