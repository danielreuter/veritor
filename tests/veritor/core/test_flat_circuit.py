import pytest

from veritor.core import (
    Circuit,
    FlatCircuit,
    Gate,
    GateRef,
    GateSet,
    InvalidArtifact,
)

WIDTH = 16
MASK = (1 << WIDTH) - 1

PAPER_GATES = GateSet(
    (
        Gate("in", 0, WIDTH, replay_cost=0, proof_cost=1, source="input"),
        Gate("w", 0, WIDTH, replay_cost=0, proof_cost=1, source="weight"),
        Gate(
            "+",
            2,
            WIDTH,
            replay_cost=1,
            proof_cost=1,
            evaluate=lambda a: (a[0] + a[1]) & MASK,
        ),
        Gate(
            "x2",
            1,
            WIDTH,
            replay_cost=2,
            proof_cost=3,
            evaluate=lambda a: (a[0] * a[0]) & MASK,
        ),
        Gate(
            "*",
            2,
            WIDTH,
            replay_cost=2,
            proof_cost=3,
            evaluate=lambda a: (a[0] * a[1]) & MASK,
        ),
    ),
    name="tests.paper",
    version="1",
)
IN = GateRef("in", (), WIDTH)
W = GateRef("w", (), WIDTH)


def paper_example() -> FlatCircuit:
    """The eight-address example: in, in, +, x², in, in, +, × with R1 = 0..3, R2 = 4..7.

    Verification units are the pairs {0,1}, {2,3}, {4,5}, {6,7}.  The input
    gates are zero-arity gates inside the units like any other gate.
    """

    return FlatCircuit(
        (
            IN,
            IN,
            GateRef("+", (0, 1), WIDTH),
            GateRef("x2", (2,), WIDTH),
            IN,
            IN,
            GateRef("+", (4, 5), WIDTH),
            GateRef("*", (3, 6), WIDTH),
        ),
        (7,),
        PAPER_GATES,
    )


R1, R2 = range(4), range(4, 8)
V = [range(2), range(2, 4), range(4, 6), range(6, 8)]


def test_paper_example_interfaces_sizes_and_costs():
    c = paper_example()

    assert isinstance(c, Circuit)
    assert (c.n, c.input_count, c.inputs, c.outputs) == (8, 4, (0, 1, 4, 5), (7,))
    assert (c.weight_count, c.weights) == (0, ())
    assert c[3] == GateRef("x2", (2,), WIDTH) and c[0].is_input and not c[0].is_weight
    assert (
        c[0] == GateRef("in", (), WIDTH, "input") and c[0].is_source
    )  # completed with its source
    assert [c.input_rank(a) for a in c.inputs] == [0, 1, 2, 3]
    with pytest.raises(KeyError):
        c.input_rank(2)
    with pytest.raises(KeyError):
        c.weight_rank(0)
    assert c.In(R1) == () and c.Out(R1) == (3,)
    assert c.In(R2) == (3,) and c.Out(R2) == (7,)
    assert [c.In(v) for v in V] == [(), (0, 1), (), (3, 4, 5)]
    assert [c.Out(v) for v in V] == [(0, 1), (3,), (4, 5), (7,)]
    assert [c.Size(v) for v in V] == [2, 2, 2, 2] and c.Size(R1) == 4
    assert (c.Cost(R1), c.Cost(R2)) == (3, 3)  # source gates replay for free
    assert (c.Cost(R1, "proof"), c.Cost(R2, "proof")) == (6, 6)
    assert c.Cost(range(8)) == 6
    with pytest.raises(ValueError, match="unknown cost kind"):
        c.Cost(R1, "time")  # type: ignore[arg-type]


def test_paper_example_semantics():
    c = paper_example()

    tape = c.evaluate((3, 4, 5, 6))
    assert tape == (3, 4, 7, 49, 5, 6, 11, 49 * 11)
    assert c.evaluate_gate(7, (49, 11)) == 539
    assert c.check_gate(3, (7,), 49) and not c.check_gate(3, (7,), 48)
    assert c.encode(7, 539) == (539).to_bytes(2, "big")
    assert c.decode(7, b"\x02\x1b") == 539
    with pytest.raises(InvalidArtifact, match="is a source gate"):
        c.evaluate_gate(0, ())
    with pytest.raises(InvalidArtifact, match="is a source gate"):
        c.check_gate(0, (), 3)
    with pytest.raises(InvalidArtifact, match="expected 4 inputs"):
        c.evaluate((1, 2))
    with pytest.raises(InvalidArtifact, match="expected 0 weights"):
        c.evaluate((1, 2, 3, 4), (5,))
    with pytest.raises(IndexError):
        c[8]


def test_weights_are_assigned_by_rank_between_the_other_gates():
    # in, w, *, w, in, +, * : x0 * w0 + (w1 + x1) ... both sources interleaved with gates
    c = FlatCircuit(
        (
            IN,
            W,
            GateRef("*", (0, 1), WIDTH),
            W,
            IN,
            GateRef("+", (3, 4), WIDTH),
            GateRef("*", (2, 5), WIDTH),
        ),
        (6,),
        PAPER_GATES,
    )

    assert (c.inputs, c.weights) == ((0, 4), (1, 3))
    assert (c.input_count, c.weight_count) == (2, 2)
    assert [c.weight_rank(a) for a in c.weights] == [0, 1] and c.input_rank(4) == 1
    assert c.evaluate((3, 5), (7, 11)) == (3, 7, 21, 11, 5, 16, 21 * 16)
    assert c.Out(range(2)) == (0, 1) and c.In(range(2, 4)) == (0, 1)
    assert c.Cost(range(7)) == 5 and c.Cost(range(7), "proof") == 4 + 3 + 1 + 3


def test_flat_circuit_rejects_ill_formed_gate_lists():
    with pytest.raises(InvalidArtifact, match="reads a later address"):
        FlatCircuit((IN, GateRef("+", (0, 2), WIDTH), IN), (1,), PAPER_GATES)
    with pytest.raises(InvalidArtifact, match="wrong arity"):
        FlatCircuit((IN, GateRef("+", (0,), WIDTH)), (1,), PAPER_GATES)
    with pytest.raises(InvalidArtifact, match="unknown gate"):
        FlatCircuit((IN, GateRef("-", (0, 0), WIDTH)), (1,), PAPER_GATES)
    with pytest.raises(InvalidArtifact, match="wrong arity"):
        FlatCircuit((IN, GateRef("in", (0,), WIDTH)), (1,), PAPER_GATES)
    with pytest.raises(InvalidArtifact, match="misstates the source"):
        FlatCircuit((GateRef("in", (), WIDTH, "weight"),), (0,), PAPER_GATES)
    with pytest.raises(InvalidArtifact, match="misstates the source"):
        FlatCircuit((IN, GateRef("+", (0, 0), WIDTH, "input")), (1,), PAPER_GATES)
    with pytest.raises(InvalidArtifact, match="outside the circuit"):
        FlatCircuit((IN,), (1,), PAPER_GATES)
    with pytest.raises(TypeError, match="index node or an address interval"):
        paper_example().In((0, 1))
