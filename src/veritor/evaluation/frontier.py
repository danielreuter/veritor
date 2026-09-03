"""The honest server's frontier: capacity against prover overhead and verifier work.

For a serving run (:mod:`.serving`), every partition an honest server might
mark and every policy on a grid is priced three ways with the protocol's own
functions: ``Bound`` (the capacity the verifier will certify at its ``eta``),
``Cost`` (the prover's expected cost: the boundary commitment, the
recomputation the sampled replay units force, their interior commitments,
the proofs) and ``expected_work`` (the verifier's).  Both costs are reported
relative to the honest computation itself -- the replay cost of the whole
circuit -- so ``0.05`` means five percent of the serving run.  The
recomputation term is what separates the partitions: a request is closed
(its ports are the weights) and is replayed for ``q`` of its cost, while a
step, a layer, a matvec or a cell reads activations or the cache the honest
server does not keep, so sampling any of the millions inside a request
re-executes the request (see :mod:`veritor.analysis.cost`).

:func:`certify` then answers the calibration question: given what an honest
server will pay and what the verifier can do, what is the smallest capacity
some partition and policy certify?  That number is the ``U_max`` a verifier
can demand without turning honest servers away, and the partition and
policy behind it are what an honest server would choose.

The bound is the Laplace fold alone (:class:`BoundOptions` with ``knapsack``
off) on a fine cost grid: at these scales a single wrong unit costs far less
than ``Lambda / 2048`` and the knapsack's grid would round it to nothing.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

from veritor.analysis import BoundOptions, CostParameters, PolicyGrid, bound, cost
from veritor.core import KindTable, VerificationPolicy, exact_fraction
from veritor.core.policy import ProbabilityInput
from veritor.protocol.parameters import expected_work

from .serving import (
    ReplayLevel,
    ServingShape,
    VerificationLevel,
    serving_table,
)

FRONTIER_OPTIONS = BoundOptions(knapsack=False, max_buckets=1 << 22)
"""Laplace only, with a grid fine enough that no admissible cost rounds to zero."""

DEFAULT_ETAS: tuple[Fraction, ...] = (Fraction(1, 2), Fraction(1, 100), Fraction(1, 10**6))
DEFAULT_GRID = PolicyGrid(
    q=tuple(Fraction(1, 2**k) for k in (1, 3, 5, 7, 9, 11, 13)),
    s=tuple(Fraction(1, 2**k) for k in (0, 3, 6, 9)),
)
DEFAULT_PARTITIONS: tuple[tuple[ReplayLevel, VerificationLevel], ...] = (
    ("request", "row"),
    ("request", "cell"),
    ("request", "gate"),
    ("step", "row"),
    ("step", "cell"),
    ("layer", "row"),
    ("layer", "cell"),
    ("matvec", "row"),
    ("matvec", "cell"),
    ("row", "cell"),
    ("row", "gate"),
    ("cell", "gate"),
)

FRONTIER_SHAPE = ServingShape(
    vocab=32768,
    d_model=8192,
    heads=64,
    layers=80,
    prompt=512,
    generated=512,
    requests=2048,
    batch=32,
    hidden_multiplier=4,
)
"""A 70B-class dense decoder serving 2048 requests of 512 + 512 tokens: 2.7e17 gates."""


@dataclass(frozen=True, slots=True)
class Point:
    """One partition and policy, priced.

    ``bits`` is ``Bound`` at ``eta``; ``overhead`` the prover's expected cost
    and ``work`` the verifier's, both divided by the honest computation's
    replay cost; ``recompute`` the recomputation term of that cost alone, on
    the same scale (``1`` means the sampled units force the whole run to be
    re-executed); ``seconds`` what the bound took to compute.
    """

    replay: str
    verification: str
    q: Fraction
    s: Fraction
    eta: Fraction
    bits: float
    out_bits: int
    overhead: Fraction
    work: Fraction
    seconds: float
    recompute: Fraction = Fraction(0)

    @property
    def policy(self) -> VerificationPolicy:
        return VerificationPolicy(self.q, self.s)

    @property
    def fraction(self) -> float:
        """``bits`` as a fraction of the circuit's output bits."""

        return self.bits / self.out_bits

    def to_json(self) -> dict[str, object]:
        record = asdict(self)
        for name in ("q", "s", "eta", "overhead", "work", "recompute"):
            record[name] = str(record[name])
        return record

    @classmethod
    def from_json(cls, record: dict[str, object]) -> Point:
        return cls(
            replay=str(record["replay"]),
            verification=str(record["verification"]),
            q=Fraction(str(record["q"])),
            s=Fraction(str(record["s"])),
            eta=Fraction(str(record["eta"])),
            bits=float(record["bits"]),  # type: ignore[arg-type]
            out_bits=int(record["out_bits"]),  # type: ignore[arg-type]
            overhead=Fraction(str(record["overhead"])),
            work=Fraction(str(record["work"])),
            seconds=float(record["seconds"]),  # type: ignore[arg-type]
            recompute=Fraction(str(record.get("recompute", 0))),
        )


def honest_cost(table: KindTable) -> int:
    """The replay cost of the whole circuit: what the honest computation costs in the same units."""

    return next(row.replay_cost for row in table.rows if row.kind == table.root)


def price(
    table: KindTable,
    shape: ServingShape,
    replay: str,
    verification: str,
    policy: VerificationPolicy,
    eta: ProbabilityInput,
    *,
    parameters: CostParameters | None = None,
    options: BoundOptions = FRONTIER_OPTIONS,
) -> Point:
    """Bound, prover cost and verifier work of ``policy`` on ``table``."""

    base = honest_cost(table)
    started = time.perf_counter()
    result = bound(table, policy, eta, options)
    seconds = time.perf_counter() - started
    expected = cost(table, policy, parameters)
    work = expected_work(table, policy, shape.input_count + shape.output_count)
    return Point(
        replay=replay,
        verification=verification,
        q=policy.q,
        s=policy.s,
        eta=exact_fraction(eta, name="eta"),
        bits=result.bits,
        out_bits=result.out_bits,
        overhead=expected.total / base,
        work=work / base,
        seconds=seconds,
        recompute=expected.recompute / base,
    )


def sweep(
    shape: ServingShape,
    *,
    etas: Sequence[ProbabilityInput] = DEFAULT_ETAS,
    grid: PolicyGrid = DEFAULT_GRID,
    levels: Iterable[tuple[ReplayLevel, VerificationLevel]] = DEFAULT_PARTITIONS,
    parameters: CostParameters | None = None,
    options: BoundOptions = FRONTIER_OPTIONS,
    log: Callable[[Point], None] | None = None,
) -> list[Point]:
    """Price every partition in ``levels`` at every grid policy and every ``eta``."""

    points: list[Point] = []
    for replay, verification in levels:
        table = serving_table(shape, replay, verification)
        for policy in grid.policies():
            for eta in etas:
                point = price(
                    table, shape, replay, verification, policy, eta, parameters=parameters, options=options
                )
                points.append(point)
                if log is not None:
                    log(point)
    return points


def certify(
    points: Iterable[Point],
    *,
    eta: ProbabilityInput,
    max_overhead: ProbabilityInput | None = None,
    max_work: ProbabilityInput | None = None,
) -> Point | None:
    """The smallest capacity some point certifies at ``eta`` within the budgets.

    Ties break towards the cheaper prover, then the cheaper verifier.
    ``None`` when no point is within budget.
    """

    eta = exact_fraction(eta, name="eta")
    overhead = None if max_overhead is None else exact_fraction(max_overhead, name="max_overhead")
    work = None if max_work is None else exact_fraction(max_work, name="max_work")
    best: Point | None = None
    for point in points:
        if point.eta != eta:
            continue
        if overhead is not None and point.overhead > overhead:
            continue
        if work is not None and point.work > work:
            continue
        if best is None or (point.bits, point.overhead, point.work) < (best.bits, best.overhead, best.work):
            best = point
    return best


# -- persistence and reporting --------------------------------------------------------


def save(
    points: Sequence[Point], shape: ServingShape, path: Path, manifest: Mapping[str, object] | None = None
) -> None:
    """Write the points and the shape, and the run's ``manifest`` when there is one.

    The manifest (see :func:`veritor.evaluation.sweep.manifest`) records
    where the points came from: the commit, the shape, the bound options,
    the grid, the partitions and the wall time.  Files written without one
    are the same as before it existed and :func:`load` reads both.
    """

    record: dict[str, object] = {"shape": shape.manifest}
    if manifest is not None:
        record["manifest"] = dict(manifest)
    record["points"] = [point.to_json() for point in points]
    path.write_text(json.dumps(record, indent=1))


def load(path: Path) -> tuple[ServingShape, list[Point]]:
    """The shape and the points of a file written by :func:`save`, with or without a manifest."""

    record = json.loads(path.read_text())
    shape = ServingShape(**record["shape"])
    return shape, [Point.from_json(item) for item in record["points"]]


def load_manifest(path: Path) -> dict[str, object] | None:
    """The manifest of a file written by :func:`save`, or ``None`` for a file written without one."""

    record = json.loads(path.read_text())
    manifest = record.get("manifest")
    return None if manifest is None else dict(manifest)


def _bits(value: float) -> str:
    if not math.isfinite(value):
        return "inf"
    for unit, scale in (("Gbit", 2**30), ("Mbit", 2**20), ("kbit", 2**10)):
        if value >= scale:
            return f"{value / scale:.3g} {unit}"
    return f"{value:.3g} bit"


def _percent(value: float) -> str:
    if value >= 0.1:
        return f"{100 * value:.0f}%"
    if value >= 0.001:
        return f"{100 * value:.2g}%"
    return f"{100 * value:.1g}%"


def calibration_table(
    points: Sequence[Point],
    *,
    eta: ProbabilityInput,
    overheads: Sequence[ProbabilityInput],
    works: Sequence[ProbabilityInput],
) -> str:
    """A Markdown table: rows are verifier work budgets, columns prover overhead budgets.

    Each cell is the smallest certified capacity within both budgets with the
    partition and policy that achieve it, or a dash.
    """

    header = "| verifier work \\ prover overhead | " + " | ".join(_percent(float(o)) for o in overheads) + " |"
    rule = "|---|" + "|".join("---" for _ in overheads) + "|"
    lines = [header, rule]
    for work in works:
        cells = []
        for overhead in overheads:
            best = certify(points, eta=eta, max_overhead=overhead, max_work=work)
            if best is None:
                cells.append("--")
            else:
                cells.append(
                    f"{_bits(best.bits)} ({_percent(best.fraction)}) `{best.replay}/{best.verification}` "
                    f"q={best.q} s={best.s}"
                )
        lines.append(f"| {_percent(float(work))} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def partition_table(points: Sequence[Point], *, eta: ProbabilityInput, max_work: ProbabilityInput) -> str:
    """Per partition, the smallest capacity within the verifier budget and what it costs the prover."""

    eta = exact_fraction(eta, name="eta")
    lines = [
        "| partition | U | of output | q | s | prover overhead | of which recompute | verifier work |",
        "|---|---|---|---|---|---|---|---|",
    ]
    seen: dict[tuple[str, str], Point] = {}
    for point in points:
        if point.eta != eta or point.work > exact_fraction(max_work, name="max_work"):
            continue
        key = (point.replay, point.verification)
        current = seen.get(key)
        if current is None or (point.bits, point.overhead) < (current.bits, current.overhead):
            seen[key] = point
    for (replay, verification), best in seen.items():
        lines.append(
            f"| `{replay}/{verification}` | {_bits(best.bits)} | {_percent(best.fraction)} | {best.q} | {best.s} "
            f"| {_percent(float(best.overhead))} | {_percent(float(best.recompute))} | {_percent(float(best.work))} |"
        )
    return "\n".join(lines)


__all__ = [
    "DEFAULT_ETAS",
    "DEFAULT_GRID",
    "DEFAULT_PARTITIONS",
    "FRONTIER_OPTIONS",
    "FRONTIER_SHAPE",
    "Point",
    "calibration_table",
    "certify",
    "honest_cost",
    "load",
    "load_manifest",
    "partition_table",
    "price",
    "save",
    "sweep",
]
