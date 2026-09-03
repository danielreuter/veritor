"""The headline estimate: what 1% of a year's compute, spent on Verity, buys.

The abstract's claim -- allocating 1% of global compute to proofs limits the
covert exfiltration capacity of a year's observed outputs to roughly a
terabyte -- is computed here from explicit inputs, so every number can be
disputed one at a time.  ``python -m veritor.evaluation.global_estimate``
writes ``docs/global-estimate.md`` and ``docs/data/global-estimate.json``.

Method.  The year is one circuit: every request served is a replay unit
(RU), every dot product (and every other row-sized operation) inside it a
verification unit (VU), exactly the honest server's partition of
:mod:`veritor.evaluation.frontier`.  Its kind table is
:func:`veritor.evaluation.serving.serving_table` for a 70B-class dense
decoder with as many requests as the year holds; the table is built in
milliseconds whatever the request count, because the analysis never
enumerates copies.  The capacity is the closed-form
:func:`veritor.analysis.rate.rate` (proved directly, within 0.4% of
``Bound``'s fold where the whole-RU channel binds) evaluated at the policy
``(q, s)`` that a fixed budget admits, and ``U = rho * lambda + log2 e``.

The budget constraint is the prover's work as a fraction of the honest
computation, in the kind table's own cost units (the toy ISA charges a
multiply 2 and an add 1, so a multiply-accumulate is ``UNITS_PER_MAC = 3``)::

    overhead(q, s) = q * (1 + h * P / C)      replay the opened RUs and commit their interiors
                   + q * s * alpha            prove the sampled VUs
                   + h * B / (R * C)          commit the boundary once (negligible)

with ``C`` the honest cost of one RU, ``P`` the positions its interior
commitment covers (the declared outputs of its VUs, one per dot product;
or every gate at gate granularity), ``B`` the boundary positions, ``h`` the
cost of committing one position and ``alpha`` the proving factor, the cost
of proving a VU over the cost of computing it.  For a given ``s`` the
budget fixes ``q``; the estimate is the least ``U`` over ``s``.  The
constraint is linear in ``q`` because replaying anything finer than a
request re-executes the request (:mod:`veritor.analysis.cost`), so ``q``
is the fraction of the year's requests the server replays.

Inputs and their sources are the fields of :class:`Inputs`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

from veritor.analysis.rate import LOG2E, RateResult, rate
from veritor.core import VerificationPolicy
from veritor.evaluation.frontier import FRONTIER_SHAPE
from veritor.evaluation.serving import ServingShape, serving_table
from veritor.protocol.proofs.costs import alpha_dot

UNITS_PER_MAC = 3
"""Cost units per multiply-accumulate in the toy ISA (``mul`` 2 + ``add`` 1)."""

Granularity = Literal["vu", "gate"]


@dataclass(frozen=True, slots=True)
class Inputs:
    """Every assumption of the estimate, with its source.

    ``tokens_per_year``: tokens served globally.  Google reported 1.3e15
    tokens/month (Oct 2025) and OpenAI's API 6e9/min (~3e15/yr); the whole
    industry is taken as twice Google, 3e16/yr (``docs/notes/datacenter-realities.md``
    section 12).  It enters only through ``log2`` of the VU count.

    ``shape``: the representative model and request: a 70B-class dense
    decoder (``d_model`` 8192, 80 layers, 64 heads) serving requests of 512
    prompt + 512 generated tokens; the request is the RU, its dot products
    and other row-sized operations the VUs (``FRONTIER_SHAPE``).

    ``alpha``: proving cost over native cost for the dot relations, from the
    measured zkVM rates in :mod:`veritor.protocol.proofs.costs`: 7.1e7 for
    OpenVM with the ``TC_MATMUL`` precompile on an RTX 4090 against a native
    rate of 1.8e14 fp8 MAC/s (1.5e8 for ``TC_DOT``, 1.8e8 for the SP1 fork).

    ``hash_macs``: multiply-accumulates the serving hardware could do in the
    time of one SHA-256 compression: ~1e15 bf16 MAC/s (H100 class) against
    ~1e10 compressions/s on a GPU hashing kernel, so 1e5.

    ``values_per_leaf``: 16-bit committed values per 64-byte leaf, the block
    a compression consumes: 32.  The protocol today hashes one value per
    leaf; packing is a commitment-layout change, not a semantic one.

    ``interior``: what an opened RU's interior commitment covers: ``"vu"``,
    the declared outputs of its VUs that are not the RU's own outputs (the
    values other VUs read; the VU's own gates are re-executed by the check;
    the RU's outputs are boundary positions), which is what the protocol
    commits (:attr:`~veritor.core.KindSummary.interior_count`), or
    ``"gate"``, every non-source gate of the RU, as the prototype committed
    before the interior moved to VU-output granularity.

    ``budget``: the prover's work over the honest computation.  ``lam``:
    the security parameter ``lambda = log2 (1 / eta)``.
    """

    tokens_per_year: float = 3e16
    shape: ServingShape = FRONTIER_SHAPE
    alpha: float = field(default_factory=alpha_dot)
    hash_macs: float = 1e5
    values_per_leaf: int = 32
    interior: Granularity = "vu"
    budget: float = 0.01
    lam: float = 40.0

    @property
    def tokens_per_request(self) -> int:
        return self.shape.prompt + self.shape.generated

    @property
    def requests(self) -> int:
        """Requests in the year, rounded to whole batches."""

        batches = max(
            1, round(self.tokens_per_year / self.tokens_per_request / self.shape.batch)
        )
        return batches * self.shape.batch

    @property
    def hash_units(self) -> float:
        """``h``: committing one position, in cost units."""

        return self.hash_macs * UNITS_PER_MAC / self.values_per_leaf


@dataclass(frozen=True, slots=True)
class Estimate:
    """The estimate at the best policy the budget admits, with its parts."""

    inputs: Inputs
    q: float
    s: float
    rho: float
    capacity_bits: float
    replay_overhead: float
    commit_overhead: float
    proof_overhead: float
    boundary_overhead: float
    ru_cost: int
    ru_positions: int
    boundary_positions: int
    verification_units: int
    rate: RateResult

    @property
    def capacity_terabytes(self) -> float:
        return self.capacity_bits / 8 / 1e12

    @property
    def overhead(self) -> float:
        return (
            self.replay_overhead
            + self.commit_overhead
            + self.proof_overhead
            + self.boundary_overhead
        )

    def record(self) -> dict[str, object]:
        inputs = {
            "tokens_per_year": self.inputs.tokens_per_year,
            "tokens_per_request": self.inputs.tokens_per_request,
            "requests": self.inputs.requests,
            "model": self.inputs.shape.manifest,
            "alpha": self.inputs.alpha,
            "hash_macs": self.inputs.hash_macs,
            "values_per_leaf": self.inputs.values_per_leaf,
            "interior": self.inputs.interior,
            "budget": self.inputs.budget,
            "lambda": self.inputs.lam,
        }
        return {
            "inputs": inputs,
            "q": self.q,
            "s": self.s,
            "rho": self.rho,
            "channel": {
                "binding_errors": self.rate.binding,
                "scattered": self.rate.scattered,
                "whole_ru": self.rate.whole,
                "W_R": self.rate.replay_bits,
                "W_V": self.rate.verification_bits,
                "vus_per_ru": self.rate.verification_units,
            },
            "capacity_bits": self.capacity_bits,
            "capacity_terabytes": self.capacity_terabytes,
            "overhead": {
                "replay": self.replay_overhead,
                "commit_interior": self.commit_overhead,
                "proofs": self.proof_overhead,
                "boundary": self.boundary_overhead,
                "total": self.overhead,
            },
            "ru_cost_units": self.ru_cost,
            "ru_interior_positions": self.ru_positions,
            "boundary_positions": self.boundary_positions,
            "verification_units": self.verification_units,
        }


def estimate(inputs: Inputs | None = None, *, grid: int = 240) -> Estimate:
    """The least capacity the budget admits, searching ``s`` on a log grid."""

    inputs = Inputs() if inputs is None else inputs

    shape = replace(inputs.shape, requests=inputs.requests)
    table = serving_table(shape, "request", "cell")
    rows = {row.kind: row for row in table.rows}
    request = next(
        row for row in table.rows if row.role == "replay" and row.replay_cost > 0
    )
    honest = rows[table.root].replay_cost
    ru_cost = request.replay_cost
    if inputs.interior == "vu":
        ru_positions = request.interior_count
    else:
        ru_positions = request.size - request.source_inputs - request.source_weights
    boundary_positions = (
        request.copies * (request.out_count + request.source_inputs)
        + rows[table.root].source_inputs
    )
    verification_units = sum(
        row.copies for row in table.rows if row.role == "verification"
    )

    h = inputs.hash_units
    replay_unit_factor = (
        1 + h * ru_positions / ru_cost
    )  # replay plus interior commitment per opened RU
    boundary = h * boundary_positions / honest
    proving = inputs.budget - boundary
    if proving <= 0:
        raise ValueError("the boundary commitment alone exceeds the budget")
    best: Estimate | None = None
    for k in range(grid + 1):
        s = 10 ** (-12 + 12 * k / grid)  # 1e-12 .. 1
        q = proving / (replay_unit_factor + s * inputs.alpha)
        if q > 1:
            q = 1.0
        policy = VerificationPolicy(Fraction(q), Fraction(s))
        result = rate(table, policy)
        bits = result.rho * inputs.lam + LOG2E
        if best is None or bits < best.capacity_bits:
            best = Estimate(
                inputs=inputs,
                q=q,
                s=s,
                rho=result.rho,
                capacity_bits=bits,
                replay_overhead=q,
                commit_overhead=q * h * ru_positions / ru_cost,
                proof_overhead=q * s * inputs.alpha,
                boundary_overhead=boundary,
                ru_cost=ru_cost,
                ru_positions=ru_positions,
                boundary_positions=boundary_positions,
                verification_units=verification_units,
                rate=result,
            )
    assert best is not None
    return best


# -- sensitivity ---------------------------------------------------------------

SENSITIVITY: tuple[tuple[str, tuple[object, ...]], ...] = (
    (
        "alpha",
        (
            alpha_dot("openvm-tc-matmul"),
            alpha_dot("openvm-tc-dot"),
            alpha_dot("sp1-tc-dot"),
            1e9,
        ),
    ),
    ("budget", (0.001, 0.01, 0.1)),
    ("lam", (30.0, 40.0, 60.0)),
    ("hash_macs", (1e4, 1e5, 1e6)),
    ("values_per_leaf", (1, 32)),
    ("interior", ("vu", "gate")),
    ("tokens_per_year", (1e16, 3e16, 1e17)),
)


def sensitivity(base: Inputs | None = None) -> list[tuple[str, object, Estimate]]:
    """One-at-a-time variations of :data:`SENSITIVITY` around ``base``."""

    base = Inputs() if base is None else base

    out: list[tuple[str, object, Estimate]] = []
    for name, values in SENSITIVITY:
        for value in values:
            changed: dict[str, Any] = {name: value}
            out.append((name, value, estimate(replace(base, **changed))))
    return out


def shapes() -> tuple[tuple[str, ServingShape], ...]:
    """Alternative request shapes: what production traces look like (brief, section 12)."""

    base = FRONTIER_SHAPE
    return (
        ("512 + 512 (frontier shape)", base),
        ("chat: 160 + 340 (ShareGPT means)", replace(base, prompt=160, generated=340)),
        ("coding: 1500 + 16 (Azure medians)", replace(base, prompt=1500, generated=16)),
        ("agent: 8192 + 192 (Mooncake)", replace(base, prompt=8192, generated=192)),
    )


# -- report --------------------------------------------------------------------


def _e(value: float) -> str:
    return f"{value:.2e}"


def _tb(bits: float) -> str:
    return f"{bits / 8 / 1e12:.2f} TB"


def render(
    base: Estimate,
    rows: list[tuple[str, object, Estimate]],
    by_shape: list[tuple[str, Estimate]],
) -> str:
    i = base.inputs
    lines = [
        "# The headline estimate",
        "",
        "Generated by `python -m veritor.evaluation.global_estimate`; the method and every input are",
        "documented in `veritor.evaluation.global_estimate`.  The year's serving is one circuit:",
        f"{i.requests:.3e} requests (RUs) of {i.tokens_per_request} tokens, {base.verification_units:.3e} VUs, on a",
        f"70B-class dense decoder; the prover spends {i.budget:.1%} of the honest computation on the protocol.",
        "",
        "## Result",
        "",
        f"**U(lambda = {i.lam:g}) = {_e(base.capacity_bits)} bits = {_tb(base.capacity_bits)}** per year of observed outputs,",
        f"at q = {_e(base.q)} (fraction of requests replayed) and s = {_e(base.s)} (fraction of their VUs proved),",
        f"rho = {_e(base.rho)} bits per bit of threshold.",
        "",
        "| overhead term | fraction of honest compute |",
        "|---|---:|",
        f"| replay opened RUs (q) | {base.replay_overhead:.2%} |",
        f"| commit their interiors (q h P / C) | {base.commit_overhead:.2%} |",
        f"| prove sampled VUs (q s alpha) | {base.proof_overhead:.2%} |",
        f"| commit the boundary | {base.boundary_overhead:.2e} |",
        f"| total | {base.overhead:.2%} |",
        "",
        f"Binding channel: {base.rate.binding} error(s) per RU (scattered rho {_e(base.rate.scattered)},",
        f"whole-RU rho {_e(base.rate.whole)}); W_R = {base.rate.replay_bits} bits, W_V = {base.rate.verification_bits} bits,",
        f"{base.rate.verification_units:.3e} fallible VUs per RU; interior positions per RU {base.ru_positions:.3e}",
        f"({i.interior} granularity), RU cost {base.ru_cost:.3e} units.",
        "",
        "Reading: with the scattered channel binding, U is close to",
        "lambda * (W_V + log2(#VUs) + 2) * alpha * ln 2 / budget: the capacity is set by the proving factor",
        "and the budget, and depends on the size of the year's computation only through log2(#VUs).",
        "",
        "## Sensitivity (one input at a time)",
        "",
        "| input | value | q | s | U | overhead split (replay / commit / proofs) |",
        "|---|---|---:|---:|---:|---|",
    ]
    for name, value, est in rows:
        shown = (
            _e(value)
            if isinstance(value, float)
            and value < 1e-2
            or (isinstance(value, float) and value >= 1e4)
            else str(value)
        )
        lines.append(
            f"| {name} | {shown} | {_e(est.q)} | {_e(est.s)} | {_tb(est.capacity_bits)} |"
            f" {est.replay_overhead:.2%} / {est.commit_overhead:.2%} / {est.proof_overhead:.2%} |"
        )
    lines += [
        "",
        "## Request shape",
        "",
        "| request | tokens | RUs | W_R | q | s | U |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, est in by_shape:
        lines.append(
            f"| {name} | {est.inputs.tokens_per_request} | {est.inputs.requests:.2e} | {est.rate.replay_bits} |"
            f" {_e(est.q)} | {_e(est.s)} | {_tb(est.capacity_bits)} |"
        )
    lines += [
        "",
        "## Caveats",
        "",
        "- `alpha` is measured on an RTX 4090 for the dot relations of an fp8 tile; the elementwise gates",
        "  (LayerNorm, softmax, GELU) are not in the measurement and are a small fraction of the compute.",
        "- The cost model is the recompute-honest one: replaying any part of a request re-executes the request.",
        '- `interior = "vu"` is what the protocol commits (VU outputs); the `gate` row prices the interior at',
        "  gate granularity, as the prototype did before. `values_per_leaf = 32` is the packed commitment",
        "  layout the real-scale path needs; the prototype hashes one value per leaf (the `values_per_leaf = 1` row).",
        "- W_V = 16 takes a VU's declared output to be its 16-bit activation; if the fp32 accumulator of a",
        "  dot product is the declared output instead, W_V = 32 and U grows by (32 + log2 #VUs + 2) / (16 + log2 #VUs + 2),",
        "  about 17%.",
        "- The capacity is per year and per verdict (one epoch); the threshold `eta = 2^-lambda` is the",
        "  probability that a strategy outside the certified set is accepted.",
    ]
    return "\n".join(lines) + "\n"


def main(out_dir: Path = Path("docs")) -> None:
    base = estimate()
    rows = sensitivity()
    by_shape = [
        (name, estimate(replace(Inputs(), shape=shape))) for name, shape in shapes()
    ]
    (out_dir / "global-estimate.md").write_text(render(base, rows, by_shape))
    payload = {
        "base": base.record(),
        "sensitivity": [
            {"input": name, "value": value, **est.record()} for name, value, est in rows
        ],
        "shapes": [{"request": name, **est.record()} for name, est in by_shape],
    }
    (out_dir / "data" / "global-estimate.json").write_text(
        json.dumps(payload, indent=1, default=str) + "\n"
    )
    print(
        f"U = {base.capacity_bits:.3e} bits = {base.capacity_terabytes:.2f} TB at q = {base.q:.2e}, s = {base.s:.2e}"
    )


if __name__ == "__main__":
    main()


__all__ = [
    "UNITS_PER_MAC",
    "Estimate",
    "Inputs",
    "estimate",
    "main",
    "render",
    "sensitivity",
    "shapes",
]
