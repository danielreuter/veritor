import pytest

from veritor.core import (
    Gate,
    GateSet,
    InvalidArtifact,
    decode_value,
    encode_value,
    make_word_gate_set,
)


def test_word_gate_set_declares_add_and_mul_with_costs():
    gates = make_word_gate_set(8)

    assert [gate.name for gate in gates] == ["add", "mul"]
    assert gates["add"].arity == 2
    assert gates["add"].width == 8
    assert (gates["add"].replay_cost, gates["mul"].replay_cost) == (1, 2)
    assert gates["mul"].evaluate((16, 17)) == (16 * 17) & 255
    assert gates["add"].check((200, 100), 44)
    assert not gates["add"].check((200, 100), 45)
    assert "add" in gates and "sub" not in gates
    with pytest.raises(InvalidArtifact, match="unknown gate"):
        gates["sub"]


def test_gate_validates_arity_and_value_widths():
    add = make_word_gate_set(4)["add"]

    with pytest.raises(InvalidArtifact, match="expects 2 arguments"):
        add.evaluate((1,))
    with pytest.raises(InvalidArtifact, match="argument 1 is not a 4-bit value"):
        add.evaluate((1, 16))
    with pytest.raises(InvalidArtifact, match="output is not a 4-bit value"):
        add.check((1, 2), 16)


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
