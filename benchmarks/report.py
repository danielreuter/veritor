"""``python -m benchmarks.report docs/data/benchmarks.json -o docs/benchmarks.md``: render the results.

One section per benchmark, one table per sweep (wide sweeps are split into
column groups), the fitted exponents under each table, and a *Bottlenecks*
section whose numbers are pulled from the same JSON so the prose cannot drift
from the measurements.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# -- formatting ---------------------------------------------------------------------


def _sig(value: float, digits: int = 3) -> str:
    if value == 0:
        return "0"
    magnitude = math.floor(math.log10(abs(value)))
    decimals = max(0, digits - 1 - magnitude)
    text = f"{value:.{decimals}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def fmt_time(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    if seconds == 0:
        return "0"
    if seconds < 1e-6:
        return f"{_sig(seconds * 1e9)} ns"
    if seconds < 1e-3:
        return f"{_sig(seconds * 1e6)} µs"
    if seconds < 1:
        return f"{_sig(seconds * 1e3)} ms"
    if seconds < 120:
        return f"{_sig(seconds)} s"
    if seconds < 7200:
        return f"{_sig(seconds / 60)} min"
    if seconds < 172800:
        return f"{_sig(seconds / 3600)} h"
    return f"{_sig(seconds / 86400)} days"


def fmt_bytes(value: float | None) -> str:
    if value is None:
        return "—"
    for unit, scale in (
        ("PB", 1 << 50),
        ("TB", 1 << 40),
        ("GB", 1 << 30),
        ("MB", 1 << 20),
        ("KB", 1 << 10),
    ):
        if value >= scale:
            return f"{_sig(value / scale)} {unit}"
    return f"{_sig(value)} B"


def fmt_count(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, str):
        return value
    if isinstance(value, int) or float(value).is_integer():
        value = int(value)
        return f"{value:,}" if abs(value) < 10**9 else f"{value:.3g}"
    if abs(value) >= 10**9 or abs(value) < 0.01:
        return f"{value:.3g}"
    if abs(value) >= 100:
        return f"{value:,.0f}"
    return _sig(float(value))


def fmt_rate(value: float | None) -> str:
    if value is None:
        return "—"
    for unit, scale in (("G", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(value) >= scale:
            return f"{_sig(value / scale)}{unit}/s"
    return f"{_sig(value)}/s"


def fmt_micro(value: float | None) -> str:
    return "—" if value is None else fmt_time(value * 1e-6)


def fmt_column(name: str, value: Any, x_label: str = "") -> str:
    if value is None:
        return "—"
    if name == "x":
        if "bytes" in x_label:
            return fmt_bytes(value)
        if x_label in ("q", "s"):
            return fmt_count(value) if isinstance(value, str) else _fraction(value)
        return fmt_count(value)
    if name.endswith("_per_s"):
        return fmt_rate(value)
    if "_us_per_" in name:
        return fmt_micro(value)
    if name.endswith("_s") and isinstance(value, (int, float)):
        return fmt_time(value)
    if "bytes" in name or name.endswith("_per_leaf"):
        return fmt_bytes(value)
    return fmt_count(value)


def _fraction(value: float) -> str:
    if value >= 1:
        return fmt_count(value)
    inverse = 1 / value
    return (
        f"1/{round(inverse)}" if abs(inverse - round(inverse)) < 1e-9 else _sig(value)
    )


HEADERS = {
    "time_s": "time",
    "peak_bytes": "peak mem",
    "repeats": "reps",
    "n": "n",
    "kinds_s": "`kinds()`",
    "trace_s": "trace",
    "compile_total_s": "`Compile` (research API)",
    "description_bytes": "description",
    "definitions": "definitions",
    "replay_units": "RUs",
    "verification_units": "VUs",
    "advice_bytes": "advice",
    "sequential_s": "sequential",
    "strided_s": "strided",
    "owner_s": "`owner`",
    "unit_s": "`unit`",
    "verification_unit_s": "`verification_unit`",
    "verification_units_s": "`verification_units`",
    "boundary_build_s": "`boundary()`",
    "boundary_rank_s": "∂ `rank`",
    "boundary_unrank_s": "∂ `unrank`",
    "boundary_contains_s": "∂ `contains`",
    "boundary_contains_miss_s": "∂ `contains` (miss)",
    "interior_build_s": "`interior(u)`",
    "interior_rank_s": "int. `rank`",
    "interior_unrank_s": "int. `unrank`",
    "interior_contains_s": "int. `contains`",
    "boundary_count": "#∂",
    "interior_count": "#interior",
    "unit_size": "RU size",
    "out_count": "#Out",
    "build_s": "`serving_table`",
    "knapsack_s": "knapsack",
    "knapsack_bits": "knapsack bits",
    "laplace_bits": "Laplace bits",
    "cost_s": "`cost`",
    "work_s": "`expected_work`",
    "rows": "rows",
    "replay_kinds": "replay kinds",
    "buckets": "buckets",
    "buckets_used": "buckets used",
    "bits": "bits",
    "steps": "steps",
    "evaluated": "policies",
    "selected": "selected",
    "expected": "expected",
    "per_selected_s": "per selected",
    "denominator_bits": "denominator bits",
    "sample_s": "`derive_sample_selection`",
    "selected_replay_units": "#J",
    "sampled_verification_units": "#T",
    "values_per_s": "values/s",
    "hashes_per_s": "hashes/s",
    "bytes_per_leaf": "retained/leaf",
    "commit_weights_s": "`commit_weights`",
    "depth": "depth",
    "verify_s": "`verify_opening`",
    "proof_bytes": "proof",
    "evaluate_s": "evaluate",
    "gates_per_s": "gates/s",
    "kappa_w_s": "`kappa_W`",
    "verifier_admit_s": "V admit",
    "prover_setup_s": "P setup",
    "prover_commit_boundary_s": "P commit ∂",
    "verifier_boundary_s": "V ∂ + `J`",
    "prover_replay_s": "P replay",
    "prover_commit_interior_s": "P commit interiors",
    "verifier_interiors_s": "V interiors + `T`",
    "prover_prove_s": "P openings",
    "verifier_merkle_s": "V Merkle",
    "verifier_recompute_s": "V recompute",
    "prover_total_s": "P total",
    "verifier_total_s": "V total",
    "replayed_gates": "replayed gates",
    "interior_positions": "interior positions",
    "openings": "openings",
    "boundary_bytes": "∂ msg",
    "interiors_bytes": "interiors msg",
    "evidence_bytes": "evidence msg",
    "transcript_bytes": "transcript",
    "replay_us_per_gate": "replay / gate",
    "replay_us_per_position": "replay / position",
    "commit_us_per_position": "commit / position",
    "prove_us_per_opening": "open / opening",
    "verify_merkle_us_per_opening": "V Merkle / opening",
    "verify_recompute_us_per_opening": "V recompute / opening",
    "evidence_bytes_per_opening": "bytes / opening",
    "transient_ports_s": "`transient_ports`",
    "parse_s": "`parse_description`",
    "root_steps": "root steps",
    "case": "case",
    "shape": "shape",
    "layout": "layout",
    "weights": "weights",
    "gates": "n",
    "k": "k",
    "seeds": "seeds",
}

# Column groups per (benchmark, series) -- everything else falls back to chunks of the
# recorded columns.  Each group is one table.
GROUPS: dict[tuple[str, str], Sequence[Sequence[str]]] = {
    ("protocol", "*"): (
        (
            "case",
            "n",
            "replay_units",
            "verification_units",
            "evaluate_s",
            "gates_per_s",
            "kappa_w_s",
            "time_s",
            "verifier_total_s",
            "transcript_bytes",
        ),
        (
            "prover_setup_s",
            "prover_commit_boundary_s",
            "prover_replay_s",
            "prover_commit_interior_s",
            "prover_prove_s",
            "verifier_admit_s",
            "verifier_boundary_s",
            "verifier_interiors_s",
            "verifier_merkle_s",
            "verifier_recompute_s",
        ),
        (
            "boundary_count",
            "selected_replay_units",
            "replayed_gates",
            "interior_positions",
            "sampled_verification_units",
            "openings",
            "boundary_bytes",
            "interiors_bytes",
            "evidence_bytes",
        ),
        (
            "replay_us_per_gate",
            "commit_us_per_position",
            "prove_us_per_opening",
            "verify_merkle_us_per_opening",
            "verify_recompute_us_per_opening",
            "evidence_bytes_per_opening",
        ),
    ),
    ("kinds", "address_sets_vs_units_repeat"): (
        (
            "n",
            "boundary_count",
            "boundary_build_s",
            "time_s",
            "boundary_unrank_s",
            "boundary_contains_s",
        ),
        (
            "interior_build_s",
            "interior_unrank_s",
            "interior_contains_s",
            "unit_s",
            "owner_s",
            "verification_unit_s",
            "verification_units_s",
        ),
    ),
    ("kinds", "address_sets_vs_units_unrolled"): (
        (
            "n",
            "boundary_count",
            "boundary_build_s",
            "time_s",
            "boundary_unrank_s",
            "boundary_contains_s",
        ),
        (
            "interior_build_s",
            "interior_unrank_s",
            "interior_contains_s",
            "unit_s",
            "owner_s",
            "verification_unit_s",
            "verification_units_s",
        ),
    ),
    ("kinds", "vs_output_runs"): (
        (
            "unit_size",
            "out_count",
            "interior_count",
            "time_s",
            "boundary_rank_s",
            "boundary_unrank_s",
            "boundary_contains_miss_s",
        ),
        (
            "interior_build_s",
            "interior_rank_s",
            "interior_unrank_s",
            "interior_contains_s",
        ),
    ),
    ("analysis", "serving_*"): (
        (
            "shape",
            "rows",
            "replay_kinds",
            "n",
            "build_s",
            "time_s",
            "laplace_bits",
            "peak_bytes",
        ),
        ("knapsack_s", "knapsack_bits", "buckets", "cost_s", "work_s"),
    ),
}

MAX_COLUMNS = 9
HIDDEN = frozenset({"gates", "seeds", "openings_verified"})
"""Columns recorded for cross-checks but not worth a table column (`gates` duplicates `n`)."""


def _groups(bench: str, series: dict[str, Any]) -> list[list[str]]:
    name = series["name"]
    for (b, pattern), groups in GROUPS.items():
        if b != bench:
            continue
        if (
            pattern == "*"
            or pattern == name
            or (pattern.endswith("*") and name.startswith(pattern[:-1]))
        ):
            present = set(series["columns"]) | {"time_s", "peak_bytes"}
            return [[c for c in group if c in present] for group in groups]
    columns = [
        "time_s",
        "peak_bytes",
        *[c for c in series["columns"] if c not in ("time_s", "peak_bytes")],
    ]
    columns = [
        c for c in columns if any(p.get(c) is not None for p in series["points"])
    ]
    return [columns[i : i + MAX_COLUMNS] for i in range(0, len(columns), MAX_COLUMNS)]


def _header(name: str) -> str:
    return HEADERS.get(name, name.replace("_", " "))


def render_table(series: dict[str, Any], columns: Sequence[str]) -> list[str]:
    x_label = series["x_label"]
    points = series["points"]
    columns = [
        c
        for c in columns
        if c not in HIDDEN
        and any(p.get(c) is not None for p in points)
        and not all(p.get(c) == p["x"] for p in points)  # a copy of the x column
    ]
    if not columns:
        return []
    head = [x_label, *[_header(c) for c in columns]]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join("---:" for _ in head) + "|"]
    for point in series["points"]:
        cells = [fmt_column("x", point["x"], x_label)]
        for column in columns:
            cells.append(fmt_column(column, point.get(column)))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def render_fits(series: dict[str, Any]) -> str:
    parts = []
    for column, fit in series["fits"].items():
        if fit is None:
            continue
        parts.append(
            f"{_header(column)} ∝ x^{fit['exponent']:.2f} (R² {fit['r2']:.2f}, {fit['points']} pts)"
        )
    return "Fitted exponents: " + "; ".join(parts) + "." if parts else ""


def render_series(bench: str, series: dict[str, Any]) -> list[str]:
    out = [f"#### `{series['name']}`", ""]
    if series["note"]:
        out += [series["note"], ""]
    for group in _groups(bench, series):
        table = render_table(series, group)
        if table:
            out += table + [""]
    fits = render_fits(series)
    if fits:
        out += [fits, ""]
    return out


ORDER = (
    "compile",
    "lookup",
    "kinds",
    "analysis",
    "challenge",
    "merkle",
    "protocol",
    "reach",
)


def render_manifest(manifest: dict[str, Any]) -> list[str]:
    total = manifest.get("seconds")
    return [
        f"- commit `{manifest.get('git_commit', '?')[:12]}` on `{manifest.get('git_branch', '?')}`"
        + (" (dirty tree)" if manifest.get("git_dirty") else ""),
        (
            f"- {manifest.get('implementation', 'Python')} {manifest.get('python', '?')} on "
            f"{manifest.get('platform', '?')} ({manifest.get('processor') or manifest.get('machine', '?')})"
        ),
        f"- {manifest.get('timestamp', '?')}, "
        + ("quick mode, " if manifest.get("quick") else "")
        + f"{manifest.get('repeats', '?')} timed repeats per point (median reported), "
        + (f"{fmt_time(total)} in total" if total else ""),
        (
            "- times are single-threaded CPython wall clock; peak memory is the `tracemalloc` peak of one "
            "extra run, above the baseline at the call"
        ),
    ]


# -- bottlenecks -----------------------------------------------------------------------


class Data:
    """Safe accessors into the JSON document for the prose."""

    def __init__(self, document: dict[str, Any]) -> None:
        self.benchmarks = document["benchmarks"]

    def series(self, bench: str, name: str) -> dict[str, Any] | None:
        for series in self.benchmarks.get(bench, {}).get("series", []):
            if series["name"] == name:
                return series
        return None

    def points(self, bench: str, name: str) -> list[dict[str, Any]]:
        series = self.series(bench, name)
        return series["points"] if series else []

    def last(self, bench: str, name: str, column: str = "time_s") -> Any:
        for point in reversed(self.points(bench, name)):
            if point.get(column) is not None:
                return point[column]
        return None

    def first(self, bench: str, name: str, column: str = "time_s") -> Any:
        for point in self.points(bench, name):
            if point.get(column) is not None:
                return point[column]
        return None

    def largest(self, bench: str, name: str) -> dict[str, Any] | None:
        points = self.points(bench, name)
        return points[-1] if points else None

    def exponent(self, bench: str, name: str, column: str = "time_s") -> float | None:
        series = self.series(bench, name)
        if not series:
            return None
        fit = series["fits"].get(column)
        return fit["exponent"] if fit else None

    def mean(self, bench: str, name: str, column: str) -> float | None:
        values = [
            p[column]
            for p in self.points(bench, name)
            if isinstance(p.get(column), (int, float))
        ]
        return sum(values) / len(values) if values else None

    def slope(self, bench: str, name: str, column: str = "time_s") -> float | None:
        """Least-squares slope of ``column`` against ``x`` (per-unit cost for linear sweeps)."""

        pairs = [
            (p["x"], p[column])
            for p in self.points(bench, name)
            if isinstance(p.get(column), (int, float))
        ]
        if len(pairs) < 2:
            return None
        mx = sum(x for x, _ in pairs) / len(pairs)
        my = sum(y for _, y in pairs) / len(pairs)
        sxx = sum((x - mx) ** 2 for x, _ in pairs)
        return None if sxx == 0 else sum((x - mx) * (y - my) for x, y in pairs) / sxx


def _exp(value: float | None) -> str:
    return "?" if value is None else f"{value:.2f}"


def _fit(series: dict[str, Any], column: str) -> float | None:
    fit = series.get("fits", {}).get(column)
    return fit["exponent"] if fit else None


def _last_with(series: dict[str, Any], column: str) -> dict[str, Any] | None:
    """The largest point of ``series`` that recorded ``column``."""

    for point in reversed(series["points"]):
        if point.get(column) is not None:
            return point
    return None


def _ratio(a: float | None, b: float | None) -> str:
    return "?" if not a or not b else f"{a / b:.1f}×"


def bottlenecks(document: dict[str, Any]) -> list[str]:
    d = Data(document)
    out: list[str] = ["## Bottlenecks", ""]
    out += [
        (
            "What limits each component asymptotically, the constant this CPython prototype measures for it, "
            "and what the constant means for a deployment.  Every number below is read from the tables above; "
            "the deployment figures use the stated conversion rates and nothing else."
        ),
        "",
    ]

    # -- compile
    deep = d.largest("compile", "deep_repeat")
    defs = d.series("compile", "definitions")
    dmodel = d.series("compile", "cluster_d_model")
    if deep or defs or dmodel:
        out += ["### Compile", ""]
        if deep:
            out.append(
                f"- **Description-bound, never n-bound.**  `Compile` of a `repeat` tower with `n = {fmt_count(deep['x'])}` "
                f"gates takes {fmt_time(deep['time_s'])} and {fmt_bytes(deep['peak_bytes'])} "
                f"(`{deep.get('description_bytes', 0):,}`-byte description, {deep.get('definitions')} definitions); "
                f"time ∝ n^{_exp(d.exponent('compile', 'deep_repeat'))} across the sweep.  The compiler never "
                "materializes a gate: parse, validate, and build the per-definition index frames."
            )
        if defs:
            last = defs["points"][-1]
            per_byte = (
                last["time_s"] / last["description_bytes"]
                if last.get("description_bytes")
                else None
            )
            out.append(
                f"- **Linear in the description.**  Over the `definitions` sweep time ∝ definitions^"
                f"{_exp(d.exponent('compile', 'definitions'))}; at {fmt_count(last['x'])} definitions "
                f"({fmt_bytes(last['description_bytes'])}) compiling takes {fmt_time(last['time_s'])}, i.e. "
                f"{fmt_rate(1 / per_byte) if per_byte else '?'} of description, {fmt_time(last['time_s'] / last['x'])} per definition."
            )
        if dmodel:
            last = dmodel["points"][-1]
            out.append(
                f"- **The cluster ladder confirms it.**  From `d_model = {fmt_count(dmodel['points'][0]['x'])}` to "
                f"`{fmt_count(last['x'])}` the circuit grows ∝ d_model^{_exp(d.exponent('compile', 'cluster_d_model', 'n'))} "
                f"(to n = {fmt_count(last['n'])}) while compile time goes ∝ d_model^{_exp(d.exponent('compile', 'cluster_d_model'))}: "
                f"{fmt_time(last['time_s'])} at the top, of which the tracer's own `trace_s` is {fmt_time(last.get('trace_s'))}.  "
                "Compile is not a deployment bottleneck at any model size; the description size (schedule length × layers) is what matters."
            )
        out.append("")

    # -- lookup
    vs_n = d.series("lookup", "vs_n")
    vs_depth = d.series("lookup", "vs_depth")
    if vs_n or vs_depth:
        out += ["### Lazy gate lookup", ""]
        if vs_n:
            first, last = vs_n["points"][0], vs_n["points"][-1]
            out.append(
                f"- **O(depth), flat in n.**  Random `circuit[i]` costs {fmt_time(first['time_s'])} at n = {fmt_count(first['x'])} "
                f"and {fmt_time(last['time_s'])} at n = {fmt_count(last['x'])} (ratio {_ratio(last['time_s'], first['time_s'])}, "
                f"exponent {_exp(d.exponent('lookup', 'vs_n'))}); sequential access "
                f"{fmt_time(last.get('sequential_s'))}, strided {fmt_time(last.get('strided_s'))}, `owner` "
                f"{fmt_time(last.get('owner_s'))}, `unit` {fmt_time(last.get('unit_s'))} at the top."
            )
        if vs_depth:
            slope = d.slope("lookup", "vs_depth")
            last = vs_depth["points"][-1]
            out.append(
                f"- **Per level: {fmt_time(slope) if slope else '?'}.**  Along the depth sweep time ∝ depth^"
                f"{_exp(d.exponent('lookup', 'vs_depth'))}; at depth {fmt_count(last['x'])} a lookup is {fmt_time(last['time_s'])}.  "
                "Each level is a `_LazyAddresses` descent (an integer division to pick the copy plus a frame allocation); "
                "the Python constant is a few microseconds per level.  A gate-by-gate replay through `circuit[i]` therefore "
                f"runs at ~{fmt_rate(1 / last['time_s']) if last['time_s'] else '?'} gates at this depth, which is why the sessions "
                "use `replay_unit` (one frame walk per RU) rather than indexing gates one by one."
            )
        out.append("")

    # -- kinds
    kt = d.series("kinds", "kind_table_vs_definitions")
    runs = d.series("kinds", "vs_output_runs")
    rep = d.series("kinds", "address_sets_vs_units_repeat")
    unrolled = d.series("kinds", "address_sets_vs_units_unrolled")
    if kt or runs or rep:
        out += ["### Kind table and address sets", ""]
        if kt:
            last = kt["points"][-1]
            out.append(
                f"- **`kind_table()` is linear in the definitions**: ∝ kinds^{_exp(d.exponent('kinds', 'kind_table_vs_definitions'))}, "
                f"{fmt_time(last['time_s'])} for {fmt_count(last['x'])} kinds "
                f"({fmt_time(last['time_s'] / last['x'])} per kind; `kinds()` alone {fmt_time(last.get('kinds_s'))}), "
                f"with n = {fmt_count(last.get('n'))} gates behind it.  It never touches n."
            )
        if runs:
            last = runs["points"][-1]
            out.append(
                f"- **The output runs of one RU are the only per-call factor.**  With {fmt_count(last['x'])} runs in one RU's `Out`, "
                f"the boundary barely moves (`∂.rank` {fmt_time(last.get('boundary_rank_s'))}, ∝ runs^"
                f"{_exp(d.exponent('kinds', 'vs_output_runs', 'boundary_rank_s'))}; `∂.unrank` {fmt_time(last.get('boundary_unrank_s'))}, "
                f"∝ runs^{_exp(d.exponent('kinds', 'vs_output_runs', 'boundary_unrank_s'))}) while the interior grows with them: "
                f"`interior(u)` {fmt_time(last.get('interior_build_s'))} (∝ runs^{_exp(d.exponent('kinds', 'vs_output_runs', 'interior_build_s'))}), "
                f"its `unrank` {fmt_time(last.get('interior_unrank_s'))} (∝ runs^{_exp(d.exponent('kinds', 'vs_output_runs', 'interior_unrank_s'))}), "
                f"its `contains` {fmt_time(last.get('interior_contains_s'))}.  The interior (`_Interior`: the VU outputs of the RU minus "
                "its own `Out`) descends to the VU in `O(depth)` and then subtracts the RU outputs below the address run by run, and "
                "`unrank` repeats that subtraction at each bisection probe; `max_output_runs = 256` caps this at a few hundred "
                "microseconds per call, so it is bounded, not asymptotic, but a hot loop over interior positions should iterate the "
                "domain rather than `unrank` each position."
            )
        if rep and unrolled:
            r_last, u_last = rep["points"][-1], unrolled["points"][-1]
            out.append(
                f"- **Flat in the RU count.**  `∂.unrank`/`contains`, `interior(u)`, `unit(k)`, `owner(a)` and `verification_unit(k)` "
                f"do not move from {fmt_count(rep['points'][0]['x'])} to {fmt_count(r_last['x'])} RUs under a `repeat` "
                f"(exponents {_exp(d.exponent('kinds', 'address_sets_vs_units_repeat', 'boundary_unrank_s'))}, "
                f"{_exp(d.exponent('kinds', 'address_sets_vs_units_repeat', 'interior_build_s'))}, "
                f"{_exp(d.exponent('kinds', 'address_sets_vs_units_repeat', 'unit_s'))}) nor across {fmt_count(u_last['x'])} "
                f"unrolled `call` steps (`owner` {fmt_time(u_last.get('owner_s'))}, exponent "
                f"{_exp(d.exponent('kinds', 'address_sets_vs_units_unrolled', 'owner_s'))}): the unrolled root binary-searches its "
                "step table, the `repeat` divides.  `boundary()` itself is O(#definitions) and cached."
            )
        out.append("")

    # -- analysis
    serving = [
        s
        for s in d.benchmarks.get("analysis", {}).get("series", [])
        if s["name"].startswith("serving_")
    ]
    synthetic = d.series("analysis", "synthetic_vs_replay_kinds")
    buckets = d.series("analysis", "knapsack_vs_buckets")
    grid = d.series("analysis", "optimize_vs_grid")
    if serving or synthetic:
        out += ["### Analysis", ""]
        for series in serving:
            last = series["points"][-1]
            knap = last.get("knapsack_s")
            out.append(
                f"- **`{series['name']}`**: the largest shape (`{last.get('shape')}`, {fmt_count(last.get('rows'))} rows, "
                f"{fmt_count(last.get('replay_kinds'))} replay kinds, n = {fmt_count(last.get('n'))}, "
                f"{fmt_count(last.get('ru_gates'))} gates and {fmt_count(last.get('ru_positions'))} interior positions per RU) folds in "
                f"{fmt_time(last['time_s'])} (Laplace only; ∝ kinds^{_exp(_fit(series, 'time_s'))}), "
                + (
                    f"knapsack {fmt_time(knap)} (∝ kinds^{_exp(_fit(series, 'knapsack_s'))}), "
                    if knap
                    else "knapsack skipped at this size, "
                )
                + f"`cost` {fmt_time(last.get('cost_s'))}, `serving_table` itself {fmt_time(last.get('build_s'))}."
            )
        if synthetic:
            last = synthetic["points"][-1]
            knap = _last_with(synthetic, "knapsack_s") or last
            out.append(
                f"- **Linear in the kinds; the knapsack is the constant.**  On synthetic tables the Laplace fold is "
                f"∝ kinds^{_exp(d.exponent('analysis', 'synthetic_vs_replay_kinds'))} "
                f"({fmt_time(last['time_s'] / last['x'])} per replay kind at {fmt_count(last['x'])} kinds) and the knapsack "
                f"∝ kinds^{_exp(d.exponent('analysis', 'synthetic_vs_replay_kinds', 'knapsack_s'))} "
                f"({fmt_time(knap.get('knapsack_s'))} vs {fmt_time(knap['time_s'])} at {fmt_count(knap['x'])} kinds, "
                f"{_ratio(knap.get('knapsack_s'), knap['time_s'])}"
                + (
                    f"; not run at {fmt_count(last['x'])} kinds, where it would take about a minute)."
                    if knap is not last
                    else ")."
                )
            )
        if buckets:
            last = buckets["points"][-1]
            mem = _last_with(buckets, "peak_bytes") or last
            out.append(
                f"- **Buckets**: `max_buckets` from {fmt_count(buckets['points'][0]['x'])} to {fmt_count(last['x'])} moves the knapsack "
                f"∝ buckets^{_exp(d.exponent('analysis', 'knapsack_vs_buckets'))} in time and ∝ buckets^"
                f"{_exp(d.exponent('analysis', 'knapsack_vs_buckets', 'peak_bytes'))} in memory ({fmt_time(last['time_s'])} at the top, "
                f"{fmt_bytes(mem['peak_bytes'])} at {fmt_count(mem['x'])} buckets); the knapsack term goes from "
                f"{fmt_count(buckets['points'][0].get('knapsack_bits'))} to {fmt_count(last.get('knapsack_bits'))} bits over that range "
                f"and the reported bound stays {fmt_count(last.get('bits'))} bits (the Laplace term is the minimum here)."
            )
        if grid:
            last = grid["points"][-1]
            out.append(
                f"- **`optimize` is one fold per grid point**: ∝ points^{_exp(d.exponent('analysis', 'optimize_vs_grid'))}, "
                f"{fmt_time(last['time_s'] / last['x'])} per policy on the shape used, {fmt_time(last['time_s'])} for "
                f"{fmt_count(last['x'])} policies.  A `(q, s)` grid of 10^4 points is minutes; a coarse-to-fine search is the "
                "obvious deployment move."
            )
        out.append("")

    # -- challenge
    vs_n_c = d.series("challenge", "vs_N_fixed_K")
    vs_k = d.series("challenge", "vs_K_fixed_N")
    derive = d.series("challenge", "derive_selections_vs_units")
    if vs_n_c or vs_k:
        out += ["### Challenge sampling", ""]
        if vs_n_c:
            first, last = vs_n_c["points"][0], vs_n_c["points"][-1]
            out.append(
                f"- **Sublinear in N, linear in K.**  Drawing K ≈ {fmt_count(last.get('expected'))} of N candidates takes "
                f"{fmt_time(first['time_s'])} at N = {fmt_count(first['x'])} and {fmt_time(last['time_s'])} at N = {fmt_count(last['x'])} "
                f"(∝ N^{_exp(d.exponent('challenge', 'vs_N_fixed_K'))}): the binomial count is an O(log N)-bit inversion and Floyd's "
                f"subset costs O(K) draws; {fmt_time(last.get('per_selected_s'))} per selected element."
            )
        if vs_k:
            last = vs_k["points"][-1]
            out.append(
                f"- Along K the cost is ∝ K^{_exp(d.exponent('challenge', 'vs_K_fixed_N'))} ({fmt_time(last['time_s'])} for "
                f"K ≈ {fmt_count(last.get('expected'))}), i.e. ~{fmt_time(last.get('per_selected_s'))} per selected element: "
                "one SHA-256 counter-mode draw plus a set insertion.  The verifier's challenge derivation is never the bottleneck: "
                "even a 10^6-RU selection is a second."
            )
        if derive:
            last = derive["points"][-1]
            out.append(
                f"- **`derive_replay_selection` is flat in the RU count** (∝ RUs^{_exp(d.exponent('challenge', 'derive_selections_vs_units'))}, "
                f"{fmt_time(last['time_s'])} at {fmt_count(last['x'])} RUs) and `derive_sample_selection` "
                f"(∝ RUs^{_exp(d.exponent('challenge', 'derive_selections_vs_units', 'sample_s'))}, {fmt_time(last.get('sample_s'))}) "
                "depends only on |J| and the VUs per RU: the candidate sets are lazy `Units` domains, never lists."
            )
        out.append("")

    # -- merkle
    build = d.series("merkle", "build_vs_leaves")
    open_verify = d.series("merkle", "open_verify_vs_leaves")
    if build:
        last = build["points"][-1]
        values_per_s = last.get("values_per_s")
        hashes_per_s = last.get("hashes_per_s")
        out += ["### Merkle commitments", ""]
        out.append(
            f"- **Build is linear: {fmt_rate(values_per_s)} values, {fmt_rate(hashes_per_s)} domain-separated SHA-256 calls** "
            f"at {fmt_count(last['x'])} leaves (∝ leaves^{_exp(d.exponent('merkle', 'build_vs_leaves'))}), retaining "
            f"{fmt_bytes(last.get('bytes_per_leaf'))} per leaf (∝ leaves^{_exp(d.exponent('merkle', 'build_vs_leaves', 'peak_bytes'))}; "
            f"{fmt_bytes(last['peak_bytes'])} at the top).  `commit_weights` runs at the same rate ({fmt_time(last.get('commit_weights_s'))} "
            f"for {fmt_count(last['x'])} weights).  The 32-byte digests are Python `bytes` objects and every level is a list: the "
            "memory constant is ~3× the digests themselves."
        )
        if open_verify:
            last = open_verify["points"][-1]
            out.append(
                f"- **Open and verify are O(log L)**: `open` {fmt_time(last['time_s'])}, `verify_opening` {fmt_time(last.get('verify_s'))}, "
                f"proof {fmt_bytes(last.get('proof_bytes'))} at {fmt_count(last['x'])} leaves (depth {last.get('depth')}; exponents "
                f"{_exp(d.exponent('merkle', 'open_verify_vs_leaves'))}, {_exp(d.exponent('merkle', 'open_verify_vs_leaves', 'verify_s'))})."
            )
        if hashes_per_s:
            label, step = _frontier_ru(d, "serving_step_row")
            _, cell = _frontier_ru(d, "serving_cell_gate")
            per_leaf = last_bytes_per_leaf(build)
            gpu_step = 2 * step / GPU_HASHES_PER_S if step else None
            verdict = (
                "far longer than the decode step it belongs to, so a per-value hash over a whole step cannot hide behind "
                "the computation at any hashing rate"
                if gpu_step and gpu_step > 1
                else "within a decode step at this shape; at the 70B-class shape (`frontier-70B` in the full barrage, ~10^13 "
                "positions per step) it is hours"
            )
            shape_note = (
                f"`{label}`, the 70B-class shape of `docs/frontier-report.md`"
                if label == "frontier-70B"
                else f"`{label}`; the full barrage reaches `frontier-70B`, the shape of `docs/frontier-report.md`"
            )
            out.append(
                f"- **Deployment.**  Committing the interior of a sampled RU costs one leaf hash per position plus the tree "
                f"(2 hashes per position).  The positions are the declared outputs of the RU's verification units (not its own "
                f"outputs), so the VU marks set the count: a `step` RU of the largest "
                f"serving shape measured ({shape_note}) has "
                f"{fmt_count(step)} positions, a `cell` RU {fmt_count(cell)}.  At the measured {fmt_rate(hashes_per_s)} one `step` "
                f"interior is {fmt_time(2 * step / hashes_per_s) if step else '?'} in this prototype; at {fmt_rate(GPU_HASHES_PER_S)} "
                f"(a GPU-class SHA-256 rate, taken as 10^9/s here) it is {fmt_time(gpu_step)}, {verdict}; a `cell` interior is "
                f"{fmt_time(2 * cell / GPU_HASHES_PER_S) if cell else '?'}.  This is the `h` per committed value that the frontier "
                f"report's cost model charges.  Memory is the other constraint: at {fmt_bytes(per_leaf)} per leaf the Python tree for a "
                f"`step` interior would be {fmt_bytes(step * per_leaf) if step else '?'}; a packed tree is 64 B per leaf."
            )
        out.append("")

    # -- protocol
    cluster = d.series("protocol", "cluster_vs_n")
    requests = d.series("protocol", "requests_vs_n")
    vs_q = d.series("protocol", "cluster_vs_q")
    vs_s = d.series("protocol", "cluster_vs_s")
    if cluster:
        last = cluster["points"][-1]
        out += ["### Protocol end to end", ""]
        replay_us = d.mean("protocol", "cluster_vs_n", "replay_us_per_gate")
        commit_us = d.mean("protocol", "cluster_vs_n", "commit_us_per_position")
        positions_per_gate = _ratio_of_sums(
            d, "protocol", "cluster_vs_n", "interior_positions", "replayed_gates"
        )
        prove_us = d.mean("protocol", "cluster_vs_n", "prove_us_per_opening")
        vmerkle_us = d.mean("protocol", "cluster_vs_n", "verify_merkle_us_per_opening")
        vrecompute_us = d.mean(
            "protocol", "cluster_vs_n", "verify_recompute_us_per_opening"
        )
        bytes_per_opening = d.mean(
            "protocol", "cluster_vs_n", "evidence_bytes_per_opening"
        )
        gates_per_s = last.get("gates_per_s")
        out.append(
            f"- **Everything is linear in what the policy selects, and the constants are per position.**  On `ClusterG` up to "
            f"n = {fmt_count(last['n'])} (`{last.get('case')}`, q = {last.get('q')}, s = {last.get('s')}) the prover's total is ∝ n^"
            f"{_exp(d.exponent('protocol', 'cluster_vs_n'))} and the verifier's ∝ n^{_exp(d.exponent('protocol', 'cluster_vs_n', 'verifier_total_s'))}; "
            f"the honest evaluation through the lazy circuit runs at {fmt_rate(gates_per_s)} gates "
            f"({fmt_time(last.get('evaluate_s'))} at the top), the prover then spends {fmt_time(last['time_s'])} "
            f"({_ratio(last['time_s'], last.get('evaluate_s'))} the evaluation) and the verifier {fmt_time(last.get('verifier_total_s'))} "
            f"({_ratio(last.get('verifier_total_s'), last.get('evaluate_s'))})."
        )
        first = cluster["points"][0]
        drift = (
            f"  The constants drift along the ladder ({fmt_micro(first.get('replay_us_per_gate'))} → "
            f"{fmt_micro(last.get('replay_us_per_gate'))} replay, {fmt_micro(first.get('commit_us_per_position'))} → "
            f"{fmt_micro(last.get('commit_us_per_position'))} commit) as the per-RU value dictionaries and the transcript strings grow: "
            "allocator and cache pressure, not an asymptotic term."
            if first.get("replay_us_per_gate") and last.get("replay_us_per_gate")
            else ""
        )
        per_gate = (replay_us or 0) + (commit_us or 0) * (positions_per_gate or 0)
        out.append(
            f"- **Prover constants** (mean over the ladder): replay {fmt_micro(replay_us)} per replayed gate "
            f"(`replay_unit`: one lazy frame walk and one gate evaluation for every non-source gate of a selected RU), interior "
            f"commitment {fmt_micro(commit_us)} per committed position (value encoding plus the Merkle build), openings "
            f"{fmt_micro(prove_us)} per opening.  The interior is committed at VU-output granularity -- the declared outputs of the "
            f"RU's verification units that are not its own outputs -- so it holds {_pct(positions_per_gate)} of the replayed gates "
            f"on this ladder ({fmt_count(last.get('interior_positions'))} positions for {fmt_count(last.get('replayed_gates'))} gates at "
            f"the top) and replay + commit together are ~{fmt_micro(per_gate)} per replayed gate; the prover's marginal cost is "
            f"q × Σ_selected (|R| × replay + |interior| × commit), independent of s.{drift}"
        )
        out.append(
            f"- **Verifier constants**: {fmt_micro(vmerkle_us)} per opening for the Merkle path ({fmt_count(last.get('openings'))} openings "
            f"at the top, ~{fmt_count(last.get('sampled_verification_units'))} VUs) and {fmt_micro(vrecompute_us)} per opening to recompute "
            f"the gate and compare (the gate lookup dominates); the boundary check and the two challenge derivations are "
            f"{fmt_time((last.get('verifier_boundary_s') or 0) + (last.get('verifier_interiors_s') or 0))}, i.e. noise."
        )
        out.append(
            f"- **Bytes**: {fmt_bytes(bytes_per_opening)} per opening as canonical JSON (hex digests, one path per opening, no "
            f"sibling sharing), {fmt_bytes(last.get('transcript_bytes'))} for the whole transcript at the top; the boundary and "
            f"interior commitment messages are {fmt_bytes(last.get('boundary_bytes'))} and {fmt_bytes(last.get('interiors_bytes'))}.  "
            "Evidence dominates and it is the one message that grows with q·s·|VU|; batching the openings of one VU into a multiproof "
            "would cut it by the shared prefix, about `depth - log2(positions per VU)` hashes per opening."
        )
        if requests:
            r_last = requests["points"][-1]
            out.append(
                f"- **`RequestsG` vs `ClusterG`**: per-request RUs make the boundary the prompts and tokens only "
                f"(|∂| = {fmt_count(r_last.get('boundary_count'))} vs {fmt_count(last.get('boundary_count'))} at the same n), so the "
                f"boundary commitment is {fmt_time(r_last.get('prover_commit_boundary_s'))} vs {fmt_time(last.get('prover_commit_boundary_s'))}; "
                f"the per-gate constants are the same ({fmt_micro(d.mean('protocol', 'requests_vs_n', 'replay_us_per_gate'))} replay)."
            )
        if vs_q and vs_s:
            q_last = vs_q["points"][-1]
            realized = ", ".join(
                f"{fmt_count(p.get('selected_replay_units'))} at q = {_fraction(p['x'])}"
                for p in vs_q["points"]
            )
            out.append(
                f"- **Along q** the replay and interior-commitment phases scale with the realized |J| (∝ q^"
                f"{_exp(d.exponent('protocol', 'cluster_vs_q', 'prover_replay_s'))} on `{q_last.get('case')}`, "
                f"{fmt_count(q_last.get('replay_units'))} RUs; sublinear because |J| is a binomial draw, mean over the seeds "
                f"{realized}, and the fixed costs show at small q); **along s** the openings and the verifier's evidence check scale ∝ s^"
                f"{_exp(d.exponent('protocol', 'cluster_vs_s', 'prover_prove_s'))} and ∝ s^"
                f"{_exp(d.exponent('protocol', 'cluster_vs_s', 'verifier_recompute_s'))} while replay stays put "
                f"(∝ s^{_exp(d.exponent('protocol', 'cluster_vs_s', 'prover_replay_s'))})."
            )
        if replay_us and commit_us and gates_per_s and vmerkle_us and vrecompute_us:
            per_gate = (replay_us + commit_us * (positions_per_gate or 0)) * 1e-6
            per_opening = (vmerkle_us + vrecompute_us) * 1e-6
            label, step = _frontier_ru(d, "serving_step_row")
            out.append(
                f"- **Deployment.**  The prover's marginal cost is q × Σ_selected (|R| × {fmt_micro(replay_us)} + |interior| × "
                f"{fmt_micro(commit_us)} here), {per_gate * gates_per_s:.1f}× the honest per-gate evaluation in the same interpreter at this "
                "ladder's interior density — the ratio is the meaningful number: the prover re-runs the selected RUs, so it pays the honest "
                "computation once more (at the model's own rate on real hardware), and hashes one value per VU output rather than per gate.  "
                "The hash is the part that does not shrink with a faster evaluator, and the VU marks decide how many there are: "
                f"at {fmt_rate(GPU_HASHES_PER_S)} a `step` RU of the `{label}` serving shape ({fmt_count(step)} VU-output positions) commits in "
                f"{fmt_time(2 * step / GPU_HASHES_PER_S) if step else '?'}, so a deployment either marks VUs coarse enough that the interiors "
                "it must hash are affordable (the `cell`/`request` trade-off the frontier report prices) or commits to a compressed digest of the "
                f"interior.  The verifier's work is q·s·|VU| × (inputs + outputs per VU) × ({fmt_micro(vmerkle_us + vrecompute_us)} per opening here, "
                f"{1 / per_opening:,.0f} openings/s): 10^6 openings a step is {fmt_time(1e6 * per_opening)} in this prototype and a few "
                "seconds compiled, and the evidence for them is "
                f"{fmt_bytes(1e6 * (bytes_per_opening or 0))} as JSON ({fmt_bytes(1e6 * 32 * 20)} as packed 20-deep paths)."
            )
        out.append("")

    # -- reach
    chain = d.series("reach", "chain_vs_steps")
    indep = d.series("reach", "independent_vs_steps")
    if chain:
        last, first = chain["points"][-1], chain["points"][0]
        out += ["### `output_reach`", ""]
        if _chain_is_super_quadratic(d):
            out.append(
                f"- **Super-quadratic in the steps of one definition when the steps form a chain.**  `output_reach` on a root with "
                f"{fmt_count(last['x'])} sequential `call` steps takes {fmt_time(last['time_s'])} and {fmt_bytes(last['peak_bytes'])} "
                f"(∝ steps^{_exp(d.exponent('reach', 'chain_vs_steps'))} in time, ∝ steps^{_exp(d.exponent('reach', 'chain_vs_steps', 'peak_bytes'))} "
                f"in memory over the sweep, steepening as the bigint terms take over) against {fmt_time(last.get('parse_s'))} for the parse and "
                f"{fmt_time(last.get('transient_ports_s'))} for `transient_ports`.  `Index.kinds()` inherits it ({fmt_time(last.get('kinds_s'))}).  "
                "See *Performance bugs* below."
            )
        else:
            out.append(
                f"- **Linear up to a logarithm in the steps of one definition, whatever their dependency structure.**  `output_reach` on a "
                f"root with {fmt_count(last['x'])} sequential `call` steps (a decode chain, the closure `Down(j)` of every step being every "
                f"later step) takes {fmt_time(last['time_s'])} and {fmt_bytes(last['peak_bytes'])} (∝ steps^{_exp(d.exponent('reach', 'chain_vs_steps'))} "
                f"in time, ∝ steps^{_exp(d.exponent('reach', 'chain_vs_steps', 'peak_bytes'))} in memory) against {fmt_time(last.get('parse_s'))} "
                f"for the parse and {fmt_time(last.get('transient_ports_s'))} for `transient_ports`; `Index.kinds()` is {fmt_time(last.get('kinds_s'))}.  "
                "The closure is swept as intervals of steps over a segment tree (`_step_reach`; the comment above `_segment_bits` in "
                "`src/veritor/core/index.py`), `O((S + R) · log S)` for `S` steps and `R` recorded argument ranges on a chain or on "
                "siblings reading one step, so the 10^6-step `max_steps_per_definition` limit is seconds, like the parse."
            )
        if indep:
            i_last = indep["points"][-1]
            out.append(
                f"- The same number of independent steps costs {fmt_time(i_last['time_s'])} (∝ steps^{_exp(d.exponent('reach', 'independent_vs_steps'))}), "
                f"many distinct definitions {fmt_time(d.last('reach', 'definitions_vs_count'))} for {fmt_count(d.last('reach', 'definitions_vs_count', 'x'))} "
                f"(∝ ^{_exp(d.exponent('reach', 'definitions_vs_count'))}), and `repeat` nesting is flat in n: the pass is linear in the "
                + (
                    "description everywhere except along a dependency chain inside one definition."
                    if _chain_is_super_quadratic(d)
                    else "description, up to a logarithm of the longest step list, everywhere."
                )
            )
        out.append("")

    out += performance_bugs(d)
    return out


def last_bytes_per_leaf(build: dict[str, Any] | None) -> float:
    if not build:
        return 0.0
    return float(build["points"][-1].get("bytes_per_leaf") or 0.0)


GPU_HASHES_PER_S = 1e9
"""The SHA-256 rate the deployment figures assume; single GPUs are quoted in the 10^9-10^10/s range."""


def _ratio_of_sums(
    d: Data, benchmark: str, series: str, numerator: str, denominator: str
) -> float | None:
    """``sum(numerator) / sum(denominator)`` over the points of a series (``None`` without data)."""

    points = d.points(benchmark, series)
    below = sum(float(p.get(denominator) or 0) for p in points)
    if not below:
        return None
    return sum(float(p.get(numerator) or 0) for p in points) / below


def _pct(value: float | None) -> str:
    return "?" if value is None else f"{100 * value:.0f}%"


def _frontier_ru(d: Data, series: str) -> tuple[str, float | None]:
    """Label and mean interior positions per RU of the largest serving shape measured under ``series``."""

    for point in reversed(d.points("analysis", series)):
        if point.get("ru_positions"):
            return str(point.get("shape", "?")), float(point["ru_positions"])
    return "?", None


def _chain_is_super_quadratic(d: Data) -> bool:
    """Whether the `chain_vs_steps` sweep still shows the bitmask closure's Θ(S³ / w): a fitted exponent past 1.5."""

    exponent = d.exponent("reach", "chain_vs_steps")
    return exponent is not None and exponent > 1.5


def performance_bugs(d: Data) -> list[str]:
    chain = d.series("reach", "chain_vs_steps")
    out = ["## Performance bugs", ""]
    if not chain:
        return out + ["None observed in this run.", ""]
    if not _chain_is_super_quadratic(d):
        last = chain["points"][-1]
        return out + [
            "None observed in this run.",
            "",
            (
                f"- **Fixed: `output_reach` was Θ(S²) Python iterations on Θ(S)-bit integers for a chain of S steps** (14.3 s at "
                f"S = 8,192, extrapolating to ~35 days and ~116 GB of bitmasks at the 10^6-step limit).  `_step_reach` "
                "(`src/veritor/core/index.py`) now records reads as ranges of steps and sweeps the closure `Down` as intervals over a "
                f"segment tree: `chain_vs_steps` is ∝ steps^{_exp(d.exponent('reach', 'chain_vs_steps'))} in this run, "
                f"{fmt_time(last['time_s'])} at S = {fmt_count(last['x'])}.  A strided argument run over more than 64 steps and a closure "
                "of more than 64 maximal intervals are recorded as hulls, which only enlarge a closure (every reach stays a downstream "
                "cut) and are exact on every definition of at most 64 steps; `tests/veritor/core/test_reach.py` checks the sweep against "
                "the bitmask closure it replaced."
            ),
            "",
        ]
    points = chain["points"]
    rows = ", ".join(
        f"S = {fmt_count(p['x'])}: {fmt_time(p['time_s'])}" for p in points
    )
    exponent = d.exponent("reach", "chain_vs_steps")
    last = points[-1]
    # extrapolate with the local exponent between the last two points, floored at 2
    if len(points) >= 2 and points[-2]["time_s"]:
        local = math.log(last["time_s"] / points[-2]["time_s"]) / math.log(
            last["x"] / points[-2]["x"]
        )
    else:
        local = exponent or 2.0
    local = max(local, 2.0)
    projected = last["time_s"] * (1_000_000 / last["x"]) ** local
    out += [
        "### `output_reach` is Θ(S²) Python iterations on Θ(S)-bit integers for a chain of S steps",
        "",
        (
            "- **Where**: `src/veritor/core/index.py`, `_step_reach` (the loop `for j in reversed(range(count))`, in particular the "
            "inner `while rest and bits < total:` bit-iteration), called from `output_reach`, called from `Index.kinds()`, hence "
            "from `Index.kind_table()`, `Compile` (research API) and the verifier's admission."
        ),
        (
            "- **What**: `Down(j)` is kept as a bitmask over the steps of the definition.  For a chain (step `k` reads step `k - 1`) "
            "`Down(j)` is every later step, so `mask` has `S - j` bits and the inner loop pops them one at a time; each pop "
            "(`rest & -rest`, `bit_length`, `rest ^= low`) is an O(S / w) big-integer operation.  The early exit `bits < total` never "
            "fires when the outputs sit in the last step, because `out[i]` is zero for every other step.  Total: Θ(S²) iterations and "
            "Θ(S³ / w) word operations, plus Θ(S² / 8) bytes for the `down` masks."
        ),
        (
            f"- **Measured** (`benchmarks/reach.py`, `chain_vs_steps`): {rows}; fitted exponent {_exp(exponent)}, local exponent "
            f"{local:.2f} between the last two sizes.  `parse_description` on the same descriptions is linear "
            f"({fmt_time(last.get('parse_s'))} at S = {fmt_count(last['x'])})."
        ),
        (
            "- **Reproduction**: `python -c 'from benchmarks._synthetic import chain_steps, GATE_SET; from veritor.compile.description import "
            "parse_description; from veritor.core.index import output_reach; import time; root = parse_description(chain_steps(8192), "
            "GATE_SET).root; t = time.perf_counter(); output_reach(root); print(time.perf_counter() - t)'`."
        ),
        (
            f"- **Impact**: `CompilationLimits.max_steps_per_definition` is 10^6.  Extrapolating with the local exponent, a root with 10^6 "
            f"chained steps (a decode loop written as calls rather than a `repeat`, or a long unrolled schedule) would take "
            f"~{fmt_time(projected)} in `Index.kinds()` and ~{fmt_bytes(1e12 / 8)} of bitmasks, while parsing it takes seconds.  "
            "`ClusterG` roots have tens of steps, so the toy constructors do not hit it; a real schedule with thousands of decode steps "
            "in one definition does (S = 8192 is already the dominant cost of admission)."
        ),
        (
            "- **Fix**: keep `Down` as intervals of steps and sweep them with a segment tree over the step positions (the interval "
            "sweep of `_step_reach`, `O((S + R) · log S)` on a chain); this run still shows the bitmask closure's exponent, so the "
            "data predates it or the sweep regressed."
        ),
        "",
    ]
    return out


# -- document ------------------------------------------------------------------------------


def render(document: dict[str, Any]) -> str:
    manifest = document["manifest"]
    benchmarks = document["benchmarks"]
    lines = [
        "# Scale benchmarks",
        "",
        (
            "How each component of `veritor` behaves as its size parameter grows, measured by `python -m benchmarks.run` "
            "and rendered by `python -m benchmarks.report` from `docs/data/benchmarks.json`.  Each sweep records the median wall "
            "time of a few repeats, the `tracemalloc` peak and the relevant sizes, and fits `y = a · x^b` in log-log space; "
            "`b` is the exponent reported under each table.  Read *Bottlenecks* at the end for what the numbers mean."
        ),
        "",
        "## Manifest",
        "",
        *render_manifest(manifest),
        "",
        "## Contents",
        "",
    ]
    ordered = [name for name in ORDER if name in benchmarks] + [
        n for n in benchmarks if n not in ORDER
    ]
    for index, name in enumerate(ordered, 1):
        lines.append(
            f"{index}. [{benchmarks[name]['title']}](#{index}-{_anchor(benchmarks[name]['title'])})"
        )
    lines += [
        f"{len(ordered) + 1}. [Bottlenecks](#bottlenecks)",
        f"{len(ordered) + 2}. [Performance bugs](#performance-bugs)",
        "",
    ]
    for index, name in enumerate(ordered, 1):
        bench = benchmarks[name]
        lines += [f"## {index}. {bench['title']}", "", bench["description"], ""]
        lines.append(f"`{name}` ran in {fmt_time(bench.get('seconds'))}.")
        lines.append("")
        for series in bench["series"]:
            lines += render_series(name, series)
    lines += bottlenecks(document)
    return "\n".join(lines).rstrip() + "\n"


def _anchor(title: str) -> str:
    keep = []
    for char in title.lower():
        if char.isalnum():
            keep.append(char)
        elif char in " -":
            keep.append("-")
    return "".join(keep).strip("-")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.report", description=__doc__
    )
    parser.add_argument("json", help="results written by benchmarks.run")
    parser.add_argument(
        "-o", "--out", default="docs/benchmarks.md", help="markdown to write"
    )
    args = parser.parse_args(argv)
    document = json.loads(Path(args.json).read_text())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(document))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
