import pytest

from veritor.core import (
    Gate,
    GateSet,
    InvalidArtifact,
    decode_value,
    encode_value,
    make_isa_gate_set,
    make_word_gate_set,
)
from veritor.core.gates import namespaced, union_gate_set


def test_word_gate_set_declares_add_and_mul_with_costs():
    gates = make_word_gate_set(8)

    assert [gate.name for gate in gates] == ["add", "in", "mul", "weight"]
    assert gates["add"].arity == 2
    assert gates["add"].width == 8
    assert (gates["add"].replay_cost, gates["mul"].replay_cost) == (1, 2)
    assert gates["mul"].evaluate((16, 17)) == (16 * 17) & 255
    assert gates["add"].check((200, 100), 44)
    assert not gates["add"].check((200, 100), 45)
    assert "add" in gates and "sub" not in gates
    assert gates.id == "veritor.word-arithmetic@2"
    with pytest.raises(InvalidArtifact, match="unknown gate"):
        gates["sub"]


def test_word_gate_set_has_an_input_and_a_weight_source_gate():
    gates = make_word_gate_set(8)

    assert (gates.input_gates, gates.weight_gates) == (("in",), ("weight",))
    for name, source in (("in", "input"), ("weight", "weight")):
        gate = gates[name]
        assert (gate.arity, gate.width, gate.source) == (0, 8, source)
        assert (gate.replay_cost, gate.proof_cost) == (
            0,
            1,
        )  # free to replay, cheapest to prove
        assert gate.manifest["source"] == source
        with pytest.raises(InvalidArtifact, match="is a source gate"):
            gate.evaluate(())
        with pytest.raises(InvalidArtifact, match="is a source gate"):
            gate.check((), 0)
    assert gates["add"].source is None and gates["add"].manifest["source"] is None
    assert make_word_gate_set(8).digest == gates.digest
    assert make_word_gate_set(16).digest != gates.digest


def test_toy_isa_extends_the_word_arithmetic_with_what_a_decoder_needs():
    gates = make_isa_gate_set(16)
    top = (1 << 16) - 1

    assert [gate.name for gate in gates] == [
        "add",
        "eq",
        "in",
        "lt",
        "mul",
        "shr",
        "sub",
        "weight",
    ]
    assert gates.id == "veritor.toy-isa@1"
    assert gates.digest != make_word_gate_set(16).digest
    assert make_isa_gate_set(16).digest == gates.digest
    assert (gates.input_gates, gates.weight_gates) == (("in",), ("weight",))
    assert all(gate.width == 16 for gate in gates)
    assert {gate.name: gate.replay_cost for gate in gates} == {
        "add": 1,
        "sub": 1,
        "mul": 2,
        "lt": 1,
        "eq": 1,
        "shr": 1,
        "in": 0,
        "weight": 0,
    }
    assert all(
        gate.proof_cost == gate.replay_cost for gate in gates if gate.source is None
    )
    assert gates["add"].evaluate((top, 2)) == 1
    assert gates["sub"].evaluate((0, 1)) == top and gates["sub"].evaluate((5, 3)) == 2
    assert gates["mul"].evaluate((1 << 8, 1 << 8)) == 0
    assert gates["lt"].evaluate((3, 4)) == 1 and gates["lt"].evaluate((4, 4)) == 0
    assert gates["lt"].evaluate((4, 3)) == 0
    assert gates["eq"].evaluate((4, 4)) == 1 and gates["eq"].evaluate((4, 5)) == 0
    assert gates["shr"].evaluate((top, 4)) == top >> 4
    assert gates["shr"].evaluate((top, 15)) == 1
    assert (
        gates["shr"].evaluate((top, 16)) == 0 and gates["shr"].evaluate((top, top)) == 0
    )
    assert make_isa_gate_set(4)["shr"].evaluate((15, 4)) == 0
    with pytest.raises(ValueError, match="positive bit count"):
        make_isa_gate_set(0)


def test_source_gates_are_exactly_the_zero_arity_gates():
    with pytest.raises(ValueError, match="only source gates have arity 0"):
        Gate("zero", 0, 8, replay_cost=0, proof_cost=1, evaluate=lambda args: 0)
    with pytest.raises(ValueError, match="only source gates have arity 0"):
        Gate("in", 1, 8, replay_cost=0, proof_cost=1, source="input")
    with pytest.raises(ValueError, match="gate source must be None or one of"):
        Gate("in", 0, 8, replay_cost=0, proof_cost=1, source="advice")
    with pytest.raises(TypeError, match="source gates have no executable relation"):
        Gate(
            "in",
            0,
            8,
            replay_cost=0,
            proof_cost=1,
            source="input",
            evaluate=lambda args: 0,
        )
    with pytest.raises(TypeError, match="source gates have no executable relation"):
        Gate(
            "in",
            0,
            8,
            replay_cost=0,
            proof_cost=1,
            source="input",
            check=lambda a, o: True,
        )
    plain = GateSet(
        (
            Gate(
                "add",
                2,
                8,
                replay_cost=1,
                proof_cost=1,
                evaluate=lambda a: (a[0] + a[1]) & 255,
            ),
        ),
        name="tests.plain",
        version="1",
    )
    assert plain.input_gates == () and plain.weight_gates == ()
    two = GateSet(
        (
            Gate("x", 0, 8, replay_cost=0, proof_cost=1, source="input"),
            Gate("y", 0, 8, replay_cost=0, proof_cost=1, source="input"),
            Gate("w", 0, 8, replay_cost=0, proof_cost=1, source="weight"),
        ),
        name="tests.sources",
        version="1",
    )
    assert two.input_gates == ("x", "y") and two.weight_gates == ("w",)
    # the source is part of the identity
    assert (
        GateSet(
            (Gate("s", 0, 8, replay_cost=0, proof_cost=1, source="input"),),
            name="t",
            version="1",
        ).digest
        != GateSet(
            (Gate("s", 0, 8, replay_cost=0, proof_cost=1, source="weight"),),
            name="t",
            version="1",
        ).digest
    )


def test_gate_validates_arity_and_value_widths():
    add = make_word_gate_set(4)["add"]

    with pytest.raises(InvalidArtifact, match="expects 2 arguments"):
        add.evaluate((1,))
    with pytest.raises(InvalidArtifact, match="argument 1 is not a 4-bit value"):
        add.evaluate((1, 16))
    with pytest.raises(InvalidArtifact, match="output is not a 4-bit value"):
        add.check((1, 2), 16)


def test_gate_arg_widths_default_to_the_output_width_and_may_differ_per_argument():
    """``arg_widths`` is ``(width,) * arity`` unless declared; a declaration is validated per argument and enters the manifest."""

    add = make_word_gate_set(4)["add"]
    assert add.arg_widths == (4, 4) and "arg_widths" not in add.manifest

    widen = Gate(
        "widen_add",
        2,
        32,
        replay_cost=1,
        proof_cost=1,
        evaluate=lambda args: (args[0] << 16) + args[1],
        arg_widths=(16, 32),
    )
    assert widen.arg_widths == (16, 32)
    assert widen.manifest["arg_widths"] == [16, 32]
    assert widen.evaluate((0xFFFF, 1)) == 0xFFFF0001
    assert widen.check((0xFFFF, 1), 0xFFFF0001)
    with pytest.raises(InvalidArtifact, match="argument 0 is not a 16-bit value"):
        widen.evaluate((0x10000, 1))
    with pytest.raises(InvalidArtifact, match="argument 1 is not a 32-bit value"):
        widen.evaluate((1, 1 << 32))
    with pytest.raises(InvalidArtifact, match="output is not a 32-bit value"):
        widen.check((1, 1), 1 << 32)
    with pytest.raises(ValueError, match="arg_widths"):
        Gate(
            "bad",
            2,
            8,
            replay_cost=1,
            proof_cost=1,
            evaluate=lambda args: 0,
            arg_widths=(8,),
        )
    # the same declaration written out explicitly is the same gate: no digest change for single-width sets
    explicit = Gate(
        "add",
        2,
        4,
        replay_cost=add.replay_cost,
        proof_cost=add.proof_cost,
        evaluate=lambda args: 0,
        arg_widths=(4, 4),
    )
    assert explicit.manifest == add.manifest


def test_custom_check_relation_is_used_when_given():
    gate = Gate(
        "lt",
        2,
        1,
        replay_cost=1,
        proof_cost=1,
        evaluate=lambda args: int(args[0] < args[1]),
        check=lambda args, out: out == 1,
    )

    assert gate.check((1, 0), 1)
    assert not gate.check((0, 1), 0)


def test_gate_set_identity_binds_declarations_not_callables():
    def build(evaluate, *, version="1", width=8, cost=1):
        return GateSet(
            (
                Gate(
                    "add",
                    2,
                    width,
                    replay_cost=cost,
                    proof_cost=cost,
                    evaluate=evaluate,
                ),
            ),
            name="tests.gates",
            version=version,
        )

    base = build(lambda args: (args[0] + args[1]) & 255)
    assert base.digest == build(lambda args: (args[0] - args[1]) & 255).digest
    assert base.digest != build(lambda args: 0, version="2").digest
    assert base.digest != build(lambda args: 0, width=16).digest
    assert base.digest != build(lambda args: 0, cost=3).digest
    assert base.id == "tests.gates@1"
    with pytest.raises(ValueError, match="twice"):
        GateSet((base["add"], base["add"]), name="x", version="1")


def test_a_union_namespaces_the_operators_and_shares_the_sources():
    """Two members of the toy ISA: every operator twice under its namespace, one ``in``, one ``weight``."""

    isa = make_isa_gate_set(16)
    fleet = union_gate_set({"sm80": isa, "sm90": isa}, name="tests.fleet", version="1")

    assert len(fleet) == 2 * (len(isa) - 2) + 2
    assert (fleet.input_gates, fleet.weight_gates) == (("in",), ("weight",))
    assert "add@sm80" in fleet and "add@sm90" in fleet and "add" not in fleet
    for name in ("add", "sub", "mul", "lt", "eq", "shr"):
        for namespace in ("sm80", "sm90"):
            copy = fleet[namespaced(name, namespace)]
            assert (copy.arity, copy.width, copy.replay_cost, copy.proof_cost) == (
                isa[name].arity,
                isa[name].width,
                isa[name].replay_cost,
                isa[name].proof_cost,
            )
            assert copy.evaluate((7, 3)) == isa[name].evaluate((7, 3))
    assert fleet.id == "tests.fleet@1"
    assert fleet.digest != isa.digest
    assert (
        fleet.digest
        == union_gate_set(
            {"sm90": isa, "sm80": isa}, name="tests.fleet", version="1"
        ).digest
    )
    with pytest.raises(InvalidArtifact, match="unknown gate"):
        fleet["add@sm70"]


def test_a_union_checks_its_namespaces_and_its_members_sources():
    isa, narrow = make_isa_gate_set(16), make_isa_gate_set(8)

    with pytest.raises(ValueError, match="at least one member"):
        union_gate_set({}, name="x", version="1")
    with pytest.raises(ValueError, match="contain no '@'"):
        union_gate_set({"a@b": isa}, name="x", version="1")
    with pytest.raises(ValueError, match="contain no '@'"):
        union_gate_set({"": isa}, name="x", version="1")
    with pytest.raises(ValueError, match="disagree on the source gate"):
        union_gate_set({"wide": isa, "narrow": narrow}, name="x", version="1")
    with pytest.raises(TypeError, match="are GateSets"):
        union_gate_set({"a": object()}, name="x", version="1")  # type: ignore[dict-item]
    # a union of one member is that member with its operators renamed
    solo = union_gate_set({"only": isa}, name="x", version="1")
    assert [gate.name for gate in solo] == sorted(
        [*(f"{g.name}@only" for g in isa if g.source is None), "in", "weight"]
    )


def test_value_codec_is_fixed_width_big_endian():
    assert encode_value(8, 255) == b"\xff"
    assert encode_value(12, 0x123) == b"\x01\x23"
    assert decode_value(12, b"\x01\x23") == 0x123
    with pytest.raises(InvalidArtifact, match="exactly 2 bytes"):
        decode_value(12, b"\x01")
    with pytest.raises(InvalidArtifact, match="12-bit value"):
        decode_value(12, b"\xff\xff")
    with pytest.raises(InvalidArtifact, match="8-bit value"):
        encode_value(8, 256)
