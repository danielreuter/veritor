"""The serving model writes the table the compiler profiles, and every partition tiles.

At toy dimensions the ``request/row`` table equals the profile of
:class:`RequestsG` compiled, and the ``step/row`` table the profile of
:class:`ClusterG` on identical requests under FCFS: the same rows (up to
the names of the kinds) and hence the same ``Bound``, ``Cost`` and
``expected_work`` at every policy.  The other partitions are the same
gates under other marks: every gate in one replay unit and one
verification unit, and at the ``cell`` level no unit wider than a word.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from veritor.analysis import bound, cost
from veritor.compile import Compiler
from veritor.constructors import ClusterG, LMShape, Request, RequestsG, schedule_fcfs
from veritor.core import KindSummary, KindTable, VerificationPolicy, make_isa_gate_set
from veritor.core.description import REPLAY, VERIFICATION
from veritor.evaluation import ServingShape, partitions, serving_table
from veritor.protocol.parameters import expected_work

GATES = make_isa_gate_set(16)
POLICIES = (
    VerificationPolicy(1, 1),
    VerificationPolicy(Fraction(1, 2), Fraction(1, 2)),
    VerificationPolicy(Fraction(1, 8), 1),
    VerificationPolicy(Fraction(1, 3), Fraction(1, 5)),
)
SHAPES = (
    (
        LMShape(vocab=8, d_model=4, heads=2, layers=1, context=6, width=16),
        ServingShape(vocab=8, d_model=4, heads=2, layers=1, prompt=2, generated=3, requests=4, batch=2),
    ),
    (
        LMShape(vocab=6, d_model=6, heads=3, layers=2, context=7, width=16),
        ServingShape(vocab=6, d_model=6, heads=3, layers=2, prompt=3, generated=4, requests=6, batch=3),
    ),
)
TOY = SHAPES[0][1]


def rows_of(table: KindTable) -> list[tuple[object, ...]]:
    """The rows without their kind names, sorted: what two tables must share to be equal."""

    return sorted(
        (
            row.role or "",
            row.copies,
            row.size,
            row.out_count,
            row.out_bits,
            row.reach_bits,
            row.ancestor_bits,
            row.input_count,
            row.verification_units,
            row.replay_cost,
            row.proof_cost,
            row.source_inputs,
            row.source_weights,
            row.min_depth,
            row.max_depth,
            row.closed,
            tuple(sorted(count for _, count in row.children)),
            tuple(sorted(count for _, count in row.verification_kinds)),
        )
        for row in table.rows
    )


def assert_same_profile(compiled: KindTable, model: KindTable, io_count: int) -> None:
    assert (compiled.n, compiled.input_count, compiled.weight_count, compiled.replay_unit_count) == (
        model.n,
        model.input_count,
        model.weight_count,
        model.replay_unit_count,
    )
    assert rows_of(compiled) == rows_of(model)
    for policy in POLICIES:
        for eta in (Fraction(1, 2), Fraction(1, 100)):
            assert bound(compiled, policy, eta).bits == pytest.approx(bound(model, policy, eta).bits, abs=1e-9)
        assert cost(compiled, policy) == cost(model, policy)
        assert expected_work(compiled, policy, io_count) == expected_work(model, policy, io_count)


@pytest.mark.parametrize("lm, serving", SHAPES, ids=("toy", "deeper"))
def test_request_row_is_the_profile_of_requests_g(lm: LMShape, serving: ServingShape) -> None:
    requests = tuple(Request(tuple(range(serving.prompt)), serving.generated) for _ in range(serving.requests))
    description, inputs = RequestsG(lm)(requests, b"")
    compiled = Compiler(GATES).compile(description, inputs)

    model = serving_table(serving, "request", "row")

    assert_same_profile(compiled.kind_table(), model, serving.input_count + serving.output_count)
    assert compiled.kind_table().digest == compiled.digest != model.digest


@pytest.mark.parametrize("lm, serving", SHAPES, ids=("toy", "deeper"))
def test_step_row_is_the_profile_of_cluster_g_on_identical_requests(lm: LMShape, serving: ServingShape) -> None:
    requests = tuple(Request(tuple(range(serving.prompt)), serving.generated) for _ in range(serving.requests))
    waves = serving.requests // serving.batch
    steps = serving.generated * waves
    constructor = ClusterG(lm, pods=1, slots=serving.batch, steps=steps)
    schedule = schedule_fcfs(requests, 1, serving.batch, steps)
    description, inputs = constructor(requests, schedule.encode())
    compiled = Compiler(GATES).compile(description, inputs)

    model = serving_table(serving, "step", "row")

    assert_same_profile(compiled.kind_table(), model, serving.input_count + serving.output_count)


def marks_below(rows: dict[str, KindSummary], kind: str) -> set[str]:
    """The roles of the kinds strictly below ``kind``."""

    found: set[str] = set()
    for child, _ in rows[kind].children:
        if rows[child].role is not None:
            found.add(rows[child].role)
        found |= marks_below(rows, child)
    return found


def test_every_partition_tiles_the_same_gates() -> None:
    tables = {levels: serving_table(TOY, *levels) for levels in partitions()}

    sizes = {table.n for table in tables.values()}
    assert len(sizes) == 1
    for table in tables.values():
        for role in (REPLAY, VERIFICATION):
            assert sum(row.copies * row.size for row in table.rows if row.role == role) == table.n
        assert sum(row.copies for row in table.rows if row.role == VERIFICATION) == sum(
            row.copies * row.verification_units for row in table.rows if row.role == REPLAY
        )
        # a verification unit never contains a mark; a replay unit never contains a replay unit
        rows = {row.kind: row for row in table.rows}
        for row in table.rows:
            if row.role == VERIFICATION:
                assert marks_below(rows, row.kind) == set()
            if row.role == REPLAY:
                assert REPLAY not in marks_below(rows, row.kind)


def test_finer_replay_levels_have_more_units_and_the_cell_level_one_word_each() -> None:
    counts = {levels: serving_table(TOY, *levels).replay_unit_count for levels in partitions()}

    assert len(counts) == 17
    assert counts["request", "row"] < counts["step", "row"] < counts["layer", "row"]
    assert counts["layer", "row"] < counts["matvec", "row"] < counts["row", "gate"] < counts["cell", "gate"]
    # the verification level does not move the replay units
    for replay in ("request", "step", "layer", "matvec", "row"):
        assert len({count for (ru, _), count in counts.items() if ru == replay}) == 1
    widest = max(row.out_bits for row in serving_table(TOY, "cell", "gate").rows if row.role == REPLAY)
    assert widest == TOY.width


def test_verification_levels_refine() -> None:
    tables = [serving_table(TOY, "request", level) for level in ("layer", "row", "cell", "gate")]

    units = [sum(r.copies for r in table.rows if r.role == VERIFICATION) for table in tables]
    assert units[0] < units[1] < units[2] < units[3] == tables[-1].n
    # ``cell`` verification units are what ``cell`` replay units are: dots and single gates, plus the
    # source cells, which at the ``cell`` replay level sit in the weights and prompt units instead
    cell = serving_table(TOY, "cell", "gate")
    sources = TOY.requests * TOY.prompt + TOY.weight_count
    prompts = TOY.requests  # one ``prompt`` unit per request, and the one weights unit
    assert units[2] == cell.replay_unit_count - 1 - prompts + sources
    assert max(row.out_bits for row in tables[2].rows if row.role == VERIFICATION) == TOY.width


@pytest.mark.parametrize(
    "replay, verification",
    [
        ("row", "row"),
        ("layer", "layer"),
        ("cell", "row"),
        ("cell", "cell"),
        ("matvec", "layer"),
        ("nope", "row"),
        ("row", "nope"),
    ],
)
def test_levels_must_be_admissible(replay: str, verification: str) -> None:
    with pytest.raises(ValueError):
        serving_table(TOY, replay, verification)  # type: ignore[arg-type]


def test_closed_kinds_are_the_weights_the_root_the_requests_and_the_prefills() -> None:
    """At every partition the same kinds are closed: those handed the weights and nothing else."""

    for levels in partitions():
        table = serving_table(TOY, *levels)
        rows = {row.kind: row for row in table.rows}
        assert rows[table.root].closed
        closed = {row.kind for row in table.rows if row.closed and row.input_count}
        # a request or a prefill step, a prefill: their ports are the weights
        expected = {"('prefill', 2)"} | (
            {"('prefill_step', 2)"} if levels[0] == "step" else {"('request', 2, 3)"}
        )
        assert closed == expected, levels
        # the replay units that are closed are those (when marked) and the source units
        replay = {row.kind for row in table.rows if row.role == REPLAY and row.closed}
        assert replay == {row.kind for row in table.rows if row.role == REPLAY and not row.input_count} | (
            closed & {row.kind for row in table.rows if row.role == REPLAY}
        )
        # everything reading an activation, a token or the cache is open
        assert not any(row.closed for row in table.rows if row.kind.startswith("('decode"))
        assert not any(row.closed for row in table.rows if row.kind.startswith("('matvec"))


def test_reach_is_the_requests_tokens_or_the_rest_of_the_wave() -> None:
    """At every partition a kind reaches its request's tokens, or its wave's from its step on."""

    tokens = TOY.generated * TOY.width
    wave = TOY.batch * tokens
    wide_partitions = set()
    for levels in partitions():
        table = serving_table(TOY, *levels)
        rows = {row.kind: row for row in table.rows}
        root = rows[table.root]
        assert root.reach_bits == root.out_bits == TOY.requests * tokens
        assert all(0 <= row.reach_bits <= root.out_bits for row in table.rows)
        # the weights RU and its cells are read by everything; the prompt cells by their request or wave
        weights = (table.root, "('weights',)", "('source', 'weight')")
        assert all(rows[kind].reach_bits == root.out_bits for kind in weights)
        below = [row for row in table.rows if row.kind not in weights]
        if levels[0] == "step":
            # the steps of a wave are chained: the step at context ``c`` reaches the tokens of the
            # ``batch`` requests from ``c`` on, the prefill step the whole wave's
            for c in range(TOY.prompt + 1, TOY.prompt + TOY.generated):
                step = rows[f"('decode_step', {c}, {TOY.batch})"]
                assert step.reach_bits == TOY.batch * (TOY.prompt + TOY.generated - c) * TOY.width < step.out_bits
            assert rows[f"('prefill_step', {TOY.batch})"].reach_bits == wave
            assert all(row.reach_bits <= wave for row in below)
            assert max(row.reach_bits for row in below) == wave
        else:
            assert all(row.reach_bits == tokens for row in below), levels
        # wide interfaces inside a request or a step are charged the tokens, cells their word
        wide = [row for row in below if row.role and row.out_bits > wave]
        wide_partitions.update({levels} if wide else set())
        assert all(min(row.out_bits, row.reach_bits) <= wave for row in wide)
        assert all(min(row.out_bits, row.reach_bits) == row.out_bits for row in below if row.out_bits <= TOY.width)
    assert {("request", "row"), ("step", "row"), ("layer", "row"), ("request", "layer")} <= wide_partitions
    assert ("cell", "gate") not in wide_partitions  # no RU or VU wider than a word there


def test_the_shape_is_checked() -> None:
    with pytest.raises(ValueError, match="multiple of heads"):
        ServingShape(vocab=8, d_model=6, heads=4, layers=1, prompt=1, generated=1, requests=1)
    with pytest.raises(ValueError, match="multiple of batch"):
        ServingShape(vocab=8, d_model=4, heads=2, layers=1, prompt=1, generated=1, requests=3, batch=2)
    with pytest.raises(ValueError, match="positive integer"):
        ServingShape(vocab=8, d_model=4, heads=2, layers=0, prompt=1, generated=1, requests=1)


def test_a_frontier_sized_table_is_written_in_milliseconds() -> None:
    shape = ServingShape(
        vocab=32768, d_model=8192, heads=64, layers=80, prompt=512, generated=512, requests=2048, batch=32
    )

    table = serving_table(shape, "cell", "gate")

    assert table.n > 10**17 and table.replay_unit_count > 10**12
    assert len(table.rows) < 4000
