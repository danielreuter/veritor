"""First end-to-end ``Compile`` / ``Verify`` / ``Bound`` research prototype.

The prototype deliberately uses the existing exact toy word machine.  It
connects three previously separate pieces:

* the memoized constructor-call DAG and trusted kernel from ``call_dag``;
* the staged two-stage commitment protocol from ``staged_replay``; and
* the explicit downstream-cut capacity oracle from ``circuit_cut_analysis``.

Verification units are singleton computed gates in this first integration.
That keeps the error pattern used by verification identical to the source set
queried by the structural capacity oracle.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Protocol

from circuit_cut_analysis import (
    CircuitDAG,
    ExplicitCircuitCapacityOracle,
    Gate,
    LogCardinality,
)
from prototypes.call_dag import (
    BatchInput,
    DemoG,
    Kernel,
    KernelReject,
    OccurrencePath,
    ReplayPlan,
    ValidatedDefinition,
    construct,
    make_demo_request,
    trusted_word_gates,
)
from prototypes.staged_replay import (
    StagedClient,
    StagedTranscript,
    StagedVerifier,
    evaluate_assignment,
    public_io_positions,
)

Constructor = Callable[[object, bytes], bytes]
ProbabilityInput = Fraction | int | float


class RandomSource(Protocol):
    def randrange(self, stop: int) -> int: ...


def _probability_fraction(name: str, value: ProbabilityInput) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (Fraction, int, float)):
        raise TypeError(f"{name} must be a rational probability")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return Fraction(str(value))
    return Fraction(value)


def _draw_bernoulli(probability: Fraction, rng: RandomSource) -> bool:
    """Draw an exact Bernoulli coin for an arbitrary rational probability."""

    if probability <= 0:
        return False
    if probability >= 1:
        return True
    return rng.randrange(probability.denominator) < probability.numerator


@dataclass(frozen=True, init=False)
class VerificationPolicy:
    """Public parameters for the two-stage verifier and thresholded bound."""

    replay_probability: Fraction
    within_unit_probability: Fraction
    acceptance_threshold: Fraction

    def __init__(
        self,
        replay_probability: ProbabilityInput,
        within_unit_probability: ProbabilityInput,
        acceptance_threshold: ProbabilityInput,
    ) -> None:
        q = _probability_fraction("replay_probability", replay_probability)
        s = _probability_fraction(
            "within_unit_probability",
            within_unit_probability,
        )
        eta = _probability_fraction(
            "acceptance_threshold",
            acceptance_threshold,
        )
        if not 0 <= q <= 1:
            raise ValueError("replay_probability must lie in [0, 1]")
        if not 0 <= s <= 1:
            raise ValueError("within_unit_probability must lie in [0, 1]")
        if not 0 <= eta < 1:
            raise ValueError("acceptance_threshold must lie in [0, 1)")
        object.__setattr__(self, "replay_probability", q)
        object.__setattr__(self, "within_unit_probability", s)
        object.__setattr__(self, "acceptance_threshold", eta)


@dataclass(frozen=True)
class VerificationUnit:
    """One singleton local check nested inside one replay unit."""

    gate_ordinal: int
    replay_unit_index: int


@dataclass(frozen=True)
class VerificationPartition:
    """A complete ordered singleton partition of the computed gates."""

    units: tuple[VerificationUnit, ...]

    @classmethod
    def singleton(cls, replay_partition: ReplayPlan) -> VerificationPartition:
        return cls(
            tuple(
                VerificationUnit(
                    gate_ordinal=gate_ordinal,
                    replay_unit_index=replay_partition.unit_index_for_gate(
                        gate_ordinal
                    ),
                )
                for gate_ordinal in range(replay_partition.root_gate_count)
            )
        )

    def validate(self, replay_partition: ReplayPlan) -> None:
        expected_ordinals = tuple(range(replay_partition.root_gate_count))
        actual_ordinals = tuple(unit.gate_ordinal for unit in self.units)
        if actual_ordinals != expected_ordinals:
            raise ValueError(
                "verification units must be singleton gates in ordinal order"
            )
        for unit in self.units:
            expected_replay_unit = replay_partition.unit_index_for_gate(
                unit.gate_ordinal
            )
            if unit.replay_unit_index != expected_replay_unit:
                raise ValueError(
                    "every verification unit must lie inside its declared replay unit"
                )


@dataclass(frozen=True)
class BoundResult:
    """Maximum structural capacity among error patterns surviving above eta."""

    capacity: LogCardinality
    witness_error_units: tuple[int, ...]
    witness_survival_probability: Fraction
    evaluated_pattern_count: int
    admissible_pattern_count: int

    @property
    def capacity_bits(self) -> int | float:
        return self.capacity.width_bits


def top_level_replay_paths(
    kernel: Kernel,
    circuit: ValidatedDefinition,
) -> tuple[OccurrencePath, ...]:
    """Choose each positive-size top-level occurrence as one replay unit."""

    paths = []
    for step_index in range(len(circuit.steps)):
        path = (step_index,)
        if kernel.occurrence_summary(circuit, path).gate_count > 0:
            paths.append(path)
    return tuple(paths)


def compile_circuit(
    kernel: Kernel,
    constructor: Constructor,
    x: object,
    a: bytes,
    *,
    input_cells: Sequence[int],
    advice_bound_bits: int,
    replay_paths: Sequence[OccurrencePath] | None = None,
) -> tuple[ValidatedDefinition, ReplayPlan, VerificationPartition]:
    """Compile constructor output and deterministically derive both partitions."""

    construction = construct(
        kernel,
        constructor,
        x,
        a,
        input_cells=input_cells,
        advice_bound_bits=advice_bound_bits,
    )
    circuit = construction.load.root
    selected_paths = (
        top_level_replay_paths(kernel, circuit)
        if replay_paths is None
        else tuple(replay_paths)
    )
    replay_partition = kernel.derive_replay_plan(circuit, selected_paths)
    verification_partition = VerificationPartition.singleton(replay_partition)
    verification_partition.validate(replay_partition)
    return circuit, replay_partition, verification_partition


def _validate_compiled_tuple(
    kernel: Kernel,
    circuit: ValidatedDefinition,
    replay_partition: ReplayPlan,
    verification_partition: VerificationPartition,
) -> None:
    try:
        kernel.validate_replay_plan(circuit, replay_partition)
    except KernelReject as error:
        raise ValueError("replay partition does not match the circuit") from error
    verification_partition.validate(replay_partition)


def verify(
    kernel: Kernel,
    circuit: ValidatedDefinition,
    replay_partition: ReplayPlan,
    verification_partition: VerificationPartition,
    policy: VerificationPolicy,
    assignment: Sequence[int],
    *,
    inputs: Sequence[int],
    claimed_outputs: Sequence[int],
    rng: RandomSource | None = None,
) -> StagedTranscript:
    """Run the concrete boundary -> units -> interiors -> checks protocol."""

    _validate_compiled_tuple(
        kernel,
        circuit,
        replay_partition,
        verification_partition,
    )
    client = StagedClient(
        kernel=kernel,
        root=circuit,
        plan=replay_partition,
        assignment=assignment,
    )
    verifier = StagedVerifier(
        kernel=kernel,
        root=circuit,
        plan=replay_partition,
        inputs=inputs,
        claimed_outputs=claimed_outputs,
    )

    boundary_commitment = client.commit_boundary()
    verifier.receive_boundary_commitment(boundary_commitment)
    transcript = StagedTranscript(boundary_commitment=boundary_commitment)
    io_openings = {
        position: client.open_boundary(position)
        for position in public_io_positions(replay_partition)
    }
    transcript.checks.extend(verifier.check_public_io(io_openings))

    source = rng if rng is not None else random.SystemRandom()
    challenged_units = verifier.lock_replay_units(
        tuple(
            unit_index
            for unit_index in range(len(replay_partition.units))
            if _draw_bernoulli(policy.replay_probability, source)
        )
    )
    transcript.challenged_units = challenged_units
    unit_commitments = client.commit_units(challenged_units)
    verifier.receive_unit_commitments(unit_commitments)
    transcript.unit_commitments = unit_commitments

    sampled_gates = verifier.lock_within_replay_unit_sample(
        tuple(
            gate_ordinal
            for unit_index in challenged_units
            for gate_ordinal in range(
                replay_partition.units[unit_index].gate_start,
                replay_partition.units[unit_index].gate_stop,
            )
            if _draw_bernoulli(policy.within_unit_probability, source)
        )
    )
    transcript.sampled_gates = sampled_gates
    for gate_ordinal in sampled_gates:
        transcript.checks.append(verifier.check_gate(client.open_gate(gate_ordinal)))
    transcript.checks.append(verifier.finalize())
    return transcript


def _value_gate_id(position: int) -> str:
    return f"value/{position}"


def to_explicit_circuit(
    kernel: Kernel,
    circuit: ValidatedDefinition,
) -> CircuitDAG:
    """Adapt the executable word circuit to the exact structural-cut API."""

    flat = kernel.flatten(circuit)
    gates = [
        Gate(_value_gate_id(position), kernel.cell_bits, op="input")
        for position in range(flat.input_count)
    ]
    gates.extend(
        Gate(_value_gate_id(gate.write), kernel.cell_bits, op=gate.function)
        for gate in flat.gates
    )
    edges = [
        (_value_gate_id(read), _value_gate_id(gate.write))
        for gate in flat.gates
        for read in gate.reads
    ]
    outputs = [_value_gate_id(position) for position in flat.outputs]
    return CircuitDAG(gates, edges, outputs)


def survival_probability(
    replay_partition: ReplayPlan,
    verification_partition: VerificationPartition,
    policy: VerificationPolicy,
    error_units: Sequence[int],
) -> Fraction:
    """Evaluate the verifier's exact survival profile on one error pattern."""

    verification_partition.validate(replay_partition)
    pattern = tuple(error_units)
    if (
        any(
            type(unit_index) is not int
            or unit_index < 0
            or unit_index >= len(verification_partition.units)
            for unit_index in pattern
        )
        or tuple(sorted(set(pattern))) != pattern
    ):
        raise ValueError("error units must be sorted, unique, and in range")

    counts = [0] * len(replay_partition.units)
    for unit_index in pattern:
        verification_unit = verification_partition.units[unit_index]
        counts[verification_unit.replay_unit_index] += 1
    q = policy.replay_probability
    s = policy.within_unit_probability
    probability = Fraction(1)
    for count in counts:
        probability *= 1 - q + q * (1 - s) ** count
    return probability


def bound(
    kernel: Kernel,
    circuit: ValidatedDefinition,
    replay_partition: ReplayPlan,
    verification_partition: VerificationPartition,
    policy: VerificationPolicy,
    *,
    max_enumerated_units: int = 20,
) -> BoundResult:
    """Compute the exact finite-game bound by enumerating error patterns.

    For each hypothetical set ``E`` of incorrect singleton verification units,
    this computes the two-stage survival probability and the exact downstream
    cut capacity ``A_C(E)``.  The returned ``U`` is

    ``max A_C(E)`` over patterns with ``p_theta(E) > eta``.

    Enumeration is intentionally limited to tiny circuits.  Large-model
    profiles use the scalable counted optimizers in ``circuit_cut_analysis``
    instead.
    """

    _validate_compiled_tuple(
        kernel,
        circuit,
        replay_partition,
        verification_partition,
    )
    unit_count = len(verification_partition.units)
    if (
        type(max_enumerated_units) is not int
        or max_enumerated_units < 0
        or unit_count > max_enumerated_units
    ):
        raise ValueError(
            f"exact enumeration supports at most {max_enumerated_units} "
            f"verification units, got {unit_count}"
        )

    structural_circuit = to_explicit_circuit(kernel, circuit)
    oracle = ExplicitCircuitCapacityOracle(structural_circuit)
    best_capacity = LogCardinality.zero()
    best_pattern: tuple[int, ...] = ()
    best_probability = Fraction(1)
    admissible_count = 0
    pattern_count = 1 << unit_count

    for mask in range(pattern_count):
        pattern = tuple(index for index in range(unit_count) if mask & (1 << index))
        probability = survival_probability(
            replay_partition,
            verification_partition,
            policy,
            pattern,
        )
        if probability <= policy.acceptance_threshold:
            continue
        admissible_count += 1
        if pattern:
            attacked_gates = frozenset(
                _value_gate_id(
                    circuit.input_count
                    + verification_partition.units[unit_index].gate_ordinal
                )
                for unit_index in pattern
            )
            capacity = oracle.evaluate(attacked_gates).upper_bound
        else:
            capacity = LogCardinality.zero()
        if capacity > best_capacity:
            best_capacity = capacity
            best_pattern = pattern
            best_probability = probability

    return BoundResult(
        capacity=best_capacity,
        witness_error_units=best_pattern,
        witness_survival_probability=best_probability,
        evaluated_pattern_count=pattern_count,
        admissible_pattern_count=admissible_count,
    )


def claimed_outputs(
    replay_partition: ReplayPlan,
    assignment: Sequence[int],
) -> tuple[int, ...]:
    """Read the ordered public output tuple from one claimed assignment."""

    return tuple(assignment[position] for position in replay_partition.root_outputs)


def run_demo() -> Mapping[str, object]:
    """Exercise all three interfaces on two independent dot products."""

    cell_bits = 8
    kernel = Kernel(
        cell_bits=cell_bits,
        gates=trusted_word_gates(cell_bits),
    )
    constructor = DemoG(cell_bits)
    batch = BatchInput(
        (
            make_demo_request(1, 1, cell_bits),
            make_demo_request(2, 2, cell_bits),
        )
    )
    circuit, replay_partition, verification_partition = compile_circuit(
        kernel,
        constructor,
        batch,
        b"",
        input_cells=batch.cells(),
        advice_bound_bits=0,
    )
    assignment = evaluate_assignment(kernel, circuit, batch.cells())
    policy = VerificationPolicy(
        replay_probability=0.5,
        within_unit_probability=0.5,
        acceptance_threshold=0.4,
    )
    transcript = verify(
        kernel,
        circuit,
        replay_partition,
        verification_partition,
        policy,
        assignment,
        inputs=batch.cells(),
        claimed_outputs=claimed_outputs(replay_partition, assignment),
        rng=random.Random(7),
    )
    result = bound(
        kernel,
        circuit,
        replay_partition,
        verification_partition,
        policy,
    )
    return {
        "gate_count": circuit.gate_count,
        "replay_unit_count": len(replay_partition.units),
        "verification_unit_count": len(verification_partition.units),
        "accepted": transcript.accepted,
        "challenged_units": transcript.challenged_units,
        "sampled_gates": transcript.sampled_gates,
        "bound_bits": result.capacity_bits,
        "bound_witness_error_units": result.witness_error_units,
        "bound_witness_survival_probability": float(
            result.witness_survival_probability
        ),
        "evaluated_error_patterns": result.evaluated_pattern_count,
    }


if __name__ == "__main__":
    print(json.dumps(run_demo(), indent=2, sort_keys=True))
