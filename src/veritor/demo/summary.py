"""The numbers the demo produces, as plain dataclasses (JSON-dumpable, testable)."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ArrivalRecord:
    index: int
    time: float
    prompt_length: int
    max_new: int
    request: int | None
    """The request's index in ``x``; ``None`` if it was still queued when the window closed."""


@dataclass(frozen=True, slots=True)
class AttemptRecord:
    pod: int
    step: int
    slot: int
    request: int
    length: int
    outcome: str
    streamed: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class FailureRecord:
    pod: int
    step: int
    aborted: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class WorkloadSummary:
    pods: int
    slots: int
    steps: int
    step_seconds: float
    load: float
    arrivals: int
    admitted: int
    unserved: int
    joins: int
    restarts: int
    failures: tuple[FailureRecord, ...]
    eos_token: int
    eos_stops: int
    completed: int
    cut_by_run_end: int
    tokens: int
    """Tokens streamed to users: the circuit's outputs."""
    token_steps: int
    """Occupant-steps executed (the sum of the joins' lengths), recomputation included."""
    utilization: float
    advice_bytes: int
    advice_bits: int
    arrival_records: tuple[ArrivalRecord, ...]
    attempts: tuple[AttemptRecord, ...]
    occupancy: tuple[tuple[int, ...], ...]
    """Per pod, per step: occupied slots, or ``-1`` while the pod is down."""
    responses: tuple[tuple[int, ...], ...]
    matches_reference: bool
    seconds: float


@dataclass(frozen=True, slots=True)
class ModelSummary:
    vocab: int
    d_model: int
    heads: int
    layers: int
    context: int
    width: int
    sampling: bool
    vocab_bits: int
    random_bits: int
    weights: int
    parameters_seed: int
    weight_root: str


@dataclass(frozen=True, slots=True)
class CompileSummary:
    constructor: str
    compiled_digest: str
    n: int
    kinds: int
    replay_units: int
    verification_units: int
    weight_gates: int
    input_gates: int
    outputs: int
    out_bits: int
    description_bytes: int
    advice_bytes: int
    compile_seconds: float
    evaluate_seconds: float
    gates_per_token_step: float
    """Non-weight gates per occupant-step."""
    vus_per_token_step: float
    head_kind: str
    head_vu_gates: int
    head_vu_cut_bits: int
    """``kappa`` of a head VU: what ``Bound`` charges for corrupting one token decision."""
    boundary_positions: int
    interior_positions: int
    W_R: float
    """Mean boundary interface of a replay unit, in bits (the width of its ``Out``)."""
    W_V: float
    """Mean ``kappa = min(out_bits, reach_bits)`` of a verification unit, in bits."""
    positions_per_vu: float
    """Mean positions the verifier opens for a sampled VU (its gates and declared inputs)."""


@dataclass(frozen=True, slots=True)
class PolicySummary:
    q: str
    s: str
    rule: str
    expected_work: float
    """The verifier's expected work ``W`` for the run's policy, in operations."""
    work_budget: int
    io_count: int
    optimize_q: str | None
    optimize_s: str | None
    optimize_bits: float | None
    optimize_cost: float | None
    grid_points: int
    grid_evaluated: int


@dataclass(frozen=True, slots=True)
class BoundSummary:
    eta: str
    bits: float
    capped: bool
    out_bits: int
    knapsack_bits: float
    laplace_bits: float
    budget_nats: float
    """``Lambda = ln(1 / eta)``."""
    unit_cost_nats: float
    """``c(1) = -ln(1 - q s)``: what one corrupted VU in its own RU costs the adversary."""
    saturation_cost_nats: float
    """``-ln(1 - q)``: what corrupting a whole RU costs."""
    vus_to_eta: int
    """Corrupted head VUs in distinct RUs at which survival falls to ``eta``."""
    bits_realized_to_eta: int
    bits_charged_to_eta: int
    seconds: float


@dataclass(frozen=True, slots=True)
class CostSummary:
    boundary: float
    recompute: float
    commit_interior: float
    proof: float
    total: float
    weights_per_epoch: float
    verifier_expected_work: float


@dataclass(frozen=True, slots=True)
class HonestRunSummary:
    accepted: bool
    code: str
    detail: str
    session_id: str
    q_seed: str
    s_seed: str
    weight_root: str
    boundary_root: str
    boundary_positions: int
    interior_roots: tuple[str, ...]
    replay_units_opened: int
    verification_units_opened: int
    openings: int
    prover_seconds: float
    verifier_seconds: float
    message_bytes: dict[str, int]
    transcript_bytes: int
    max_capacity: int


@dataclass(frozen=True, slots=True)
class AttackRow:
    bits: int
    vus_corrupted: int
    replay_units_touched: int
    errors_per_replay_unit: tuple[int, ...]
    predicted_survival: float
    trials: int
    escaped: int
    observed_survival: float
    sigma: float
    """Standard deviation of the observed rate under the prediction."""
    deviation_sigmas: float
    protocol_trials: int
    protocol_accepted: int
    secret: str
    decoded: str
    honest_tokens_unchanged: bool


@dataclass(frozen=True, slots=True)
class AdversarySummary:
    channel: str
    bits_per_vu: int
    kappa_per_vu: int
    tolerance_sigmas: float
    rows: tuple[AttackRow, ...]
    seconds: float

    @property
    def decoded(self) -> bool:
        return all(row.decoded == row.secret for row in self.rows)

    @property
    def within_tolerance(self) -> bool:
        return all(row.deviation_sigmas <= self.tolerance_sigmas for row in self.rows)


@dataclass(frozen=True, slots=True)
class Summary:
    scale: str
    seed: int
    workload: WorkloadSummary
    model: ModelSummary
    compile: CompileSummary
    policy: PolicySummary
    bound: BoundSummary
    cost: CostSummary
    honest: HonestRunSummary
    adversary: AdversarySummary
    notes: tuple[str, ...]
    total_seconds: float

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(dataclasses.asdict(self), **kwargs)


__all__ = [
    "AdversarySummary",
    "ArrivalRecord",
    "AttackRow",
    "AttemptRecord",
    "BoundSummary",
    "CompileSummary",
    "CostSummary",
    "FailureRecord",
    "HonestRunSummary",
    "ModelSummary",
    "PolicySummary",
    "Summary",
    "WorkloadSummary",
]
