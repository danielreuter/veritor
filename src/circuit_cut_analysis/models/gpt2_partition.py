"""Certified hierarchical canonical cuts for the indexed GPT-2 circuit.

The full circuit is too large to materialize, but its gate capacities make the
relevant separator order small.  For the mixed-precision GPT-2 profile:

* every source has a self-cut of at most 32 bits;
* every gate costs at least ``log2(vocabulary_size)``;
* three minimum-width gates cost more than 32 bits.

Consequently a winning cut has at most two gates.  The implementation below
uses exact local funnel summaries, causal liveness, and the only two-gate
frontiers that survive the capacity bound.  Every indexed family is split by
layer and causal position, counted algebraically, and assigned to one lifted
certificate.  Small circuits can instead be materialized and solved by the
finite-DAG oracle gate-for-gate.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from circuit_cut_analysis.accounting import (
    ExactPartitionStatus,
    PrimitiveVector,
    WiringBottleneckRecord,
)
from circuit_cut_analysis.capacity import LogCardinality
from circuit_cut_analysis.indexed import (
    CausalPairsDomain,
    CausalReductionStepsDomain,
    ExpansionLimitExceeded,
    ExplicitDomain,
    GateFamily,
    GateRef,
    IndexDomain,
    ProductDomain,
    RectangularDomain,
)
from circuit_cut_analysis.mincut import CutResult
from circuit_cut_analysis.models.gpt2_circuit import GPT2IndexedCircuit
from circuit_cut_analysis.partition import GateCutPartition, partition_gate_cuts


class CertificateKind(StrEnum):
    EMPTY = "empty-dead-region"
    OUTPUT = "designated-output-postdominator"
    OUTPUT_PAIR = "two-output-disjoint-frontier"
    FUNNEL = "exact-local-funnel"
    BRANCH = "bounded-cardinality-branch-frontier"


@dataclass(frozen=True, slots=True)
class GPT2CanonicalPartition:
    """Compressed partition plus an optional materialized tiny oracle."""

    status: ExactPartitionStatus
    rows: tuple[WiringBottleneckRecord, ...]
    reasons: tuple[str, ...]
    covered_source_gate_count: int
    covered_primitive_count: int
    evaluated_index_regions: int
    retained_region_descriptors: int
    validated_edge_rule_count: int
    solver: str
    elapsed_seconds: float
    explicit_partition: GateCutPartition | None = None

    def singleton_result(self, ref: GateRef) -> CutResult:
        """Return the exact tiny-oracle result when materialization was used."""

        if self.explicit_partition is None:
            raise ValueError(
                "individual scalar results are not materialized for a lifted partition"
            )
        try:
            return self.explicit_partition.singleton_cuts[ref.id]
        except KeyError:
            raise KeyError(
                f"gate {ref.id!r} is not a computed tiny-oracle source"
            ) from None


@dataclass(frozen=True, slots=True)
class _OwnerPlan:
    row_id: str
    bottleneck: str
    boundary_family: str
    stage: str
    wide_branch: bool
    description: str
    boundary_mode: str = "gate"


@dataclass(slots=True)
class _Accumulator:
    row_id: str
    bottleneck: str
    capacity: LogCardinality
    cut_gate_count: int
    certificate_kind: CertificateKind
    description: str
    certificate: str
    boundaries: set[str] = field(default_factory=set)
    source_gate_count: int = 0
    occurrence_count: int = 0
    primitive_counts: Counter[str] = field(default_factory=Counter)
    source_families: set[str] = field(default_factory=set)

    def add_sources(self, family: GateFamily, count: int) -> None:
        if count <= 0:
            return
        self.source_gate_count += count
        self.source_families.add(family.name)
        if family.primitive is not None:
            self.primitive_counts[family.primitive] += count


def _primitive_vector(counts: Mapping[str, int]) -> PrimitiveVector:
    return PrimitiveVector(
        add=counts.get("add", 0),
        mul=counts.get("mul", 0),
        max=counts.get("max", 0),
        exp=counts.get("exp", 0),
        reciprocal=counts.get("reciprocal", 0),
        rsqrt=counts.get("rsqrt", 0),
        tanh=counts.get("tanh", 0),
        argmax=counts.get("argmax", 0),
    )


def _count_domain(
    domain: IndexDomain,
    fixed: Mapping[int, int],
    *,
    offset: int = 0,
) -> int:
    """Count domain members satisfying fixed flattened coordinates."""

    if isinstance(domain, RectangularDomain):
        result = 1
        for local_axis, size in enumerate(domain.shape):
            value = fixed.get(offset + local_axis)
            if value is not None:
                if not 0 <= value < size:
                    return 0
            else:
                result *= size
        return result
    if isinstance(domain, CausalPairsDomain):
        query = fixed.get(offset)
        key = fixed.get(offset + 1)
        if query is not None and not 0 <= query < domain.positions:
            return 0
        if key is not None and not 0 <= key < domain.positions:
            return 0
        if query is not None and key is not None:
            relation = key < query if domain.strict else key <= query
            return int(relation)
        if query is not None:
            return query if domain.strict else query + 1
        if key is not None:
            return domain.positions - key - int(domain.strict)
        return domain.count
    if isinstance(domain, CausalReductionStepsDomain):
        query = fixed.get(offset)
        step = fixed.get(offset + 1)
        if query is not None and not 0 <= query < domain.positions:
            return 0
        if step is not None and not 1 <= step < domain.positions:
            return 0
        if query is not None and step is not None:
            return int(step <= query)
        if query is not None:
            return query
        if step is not None:
            return domain.positions - step
        return domain.count
    if isinstance(domain, ProductDomain):
        result = 1
        part_offset = offset
        for part in domain.parts:
            result *= _count_domain(part, fixed, offset=part_offset)
            part_offset += part.arity
        return result
    if isinstance(domain, ExplicitDomain):
        return sum(
            all(index[axis - offset] == value for axis, value in fixed.items())
            for index in domain.indices
            if all(offset <= axis < offset + domain.arity for axis in fixed)
        )
    raise TypeError(f"unsupported index domain: {type(domain).__name__}")


def _count_family(family: GateFamily, **fixed_names: int) -> int:
    axes = {name: axis for axis, name in enumerate(family.index_names)}
    unknown = set(fixed_names).difference(axes)
    if unknown:
        raise ValueError(f"{family.name}: unknown index names {sorted(unknown)!r}")
    fixed = {axes[name]: value for name, value in fixed_names.items()}
    return _count_domain(family.domain, fixed)


def _layernorm_plan(name: str) -> _OwnerPlan | None:
    prefix, separator, stage = name.rpartition("/")
    if not separator or prefix not in {"blocks/ln1", "blocks/ln2", "final_ln"}:
        return None
    stage_group: tuple[str, str, str, bool, str]
    if stage in {"mean_sum", "mean"}:
        stage_group = (
            "layernorm-mean",
            "LayerNorm shared mean scalar",
            "mean",
            True,
            "mean-reduction chain ending at the shared mean",
        )
    elif stage == "centered":
        stage_group = (
            "layernorm-centered",
            "LayerNorm centered coordinate",
            "centered",
            True,
            "one centered coordinate with variance and normalization branches",
        )
    elif stage in {
        "square",
        "variance_sum",
        "variance",
        "stabilized_variance",
        "inverse_std",
    }:
        stage_group = (
            "layernorm-inverse-std",
            "LayerNorm inverse-standard-deviation scalar",
            "inverse_std",
            True,
            "variance-reduction chain ending at the shared inverse standard deviation",
        )
    elif stage in {"normalized", "affine_scale", "write"}:
        stage_group = (
            "layernorm-coordinate-output",
            "LayerNorm coordinate output",
            "write",
            False,
            "coordinate normalization and affine chain ending at the narrow write",
        )
    else:
        return None
    row_id, bottleneck, boundary_stage, wide, description = stage_group
    scope = (
        "ln1"
        if prefix == "blocks/ln1"
        else "ln2"
        if prefix == "blocks/ln2"
        else "final"
    )
    return _OwnerPlan(
        row_id=row_id,
        bottleneck=bottleneck,
        boundary_family=f"{prefix}/{boundary_stage}",
        stage=scope,
        wide_branch=wide,
        description=description,
    )


def _projection_plan(name: str) -> _OwnerPlan | None:
    specs = (
        (
            "blocks/attention/q_projection/",
            "q-projection-output",
            "Q projection output coordinate",
            "blocks/attention/q_projection/write",
            "q",
        ),
        (
            "blocks/attention/k_projection/",
            "k-cache-entry",
            "K cache coordinate",
            "blocks/attention/k_projection/write",
            "k",
        ),
        (
            "blocks/attention/v_projection/",
            "v-cache-entry",
            "V cache coordinate",
            "blocks/attention/v_projection/write",
            "v",
        ),
        (
            "blocks/attention/score/",
            "attention-score-output",
            "Scaled QK score",
            "blocks/attention/score/write",
            "score",
        ),
        (
            "blocks/attention/value_reduction/",
            "attention-value-output",
            "Attention value-reduction coordinate",
            "blocks/attention/value_reduction/write",
            "value",
        ),
    )
    for prefix, row_id, label, boundary, stage in specs:
        if name.startswith(prefix):
            return _OwnerPlan(
                row_id,
                label,
                boundary,
                stage,
                False,
                f"exact contraction funnel ending at {boundary}",
            )
    if name.startswith("blocks/attention/output_projection/") or name == (
        "blocks/residual1"
    ):
        return _OwnerPlan(
            "attention-residual",
            "Attention residual coordinate",
            "blocks/residual1",
            "post-kv",
            False,
            "attention output contraction and identity write into residual1",
        )
    if name.startswith("blocks/mlp/expansion/") or name.startswith("blocks/mlp/gelu/"):
        return _OwnerPlan(
            "gelu-output",
            "MLP expansion and GELU coordinate",
            "blocks/mlp/gelu/write",
            "post-kv",
            False,
            "expansion contraction and one-coordinate GELU funnel",
        )
    if name.startswith("blocks/mlp/contraction/") or name == "blocks/residual2":
        return _OwnerPlan(
            "mlp-residual",
            "MLP residual coordinate",
            "blocks/residual2",
            "post-kv",
            False,
            "MLP contraction and identity write into residual2",
        )
    return None


def _owner_plan(name: str) -> _OwnerPlan:
    layernorm = _layernorm_plan(name)
    if layernorm is not None:
        return layernorm
    projection = _projection_plan(name)
    if projection is not None:
        return projection
    if name in {"embedding/generated_lookup", "embedding/add"}:
        return _OwnerPlan(
            "embedding-coordinate",
            "Embedding/residual coordinate",
            "embedding/add",
            "embedding",
            False,
            "generated lookup boundary and token-plus-position residual addition",
        )
    if name == "blocks/attention/softmax/max":
        return _OwnerPlan(
            "softmax-maximum",
            "Softmax shared maximum scalar",
            name,
            "post-kv",
            True,
            "maximum-reduction chain ending at the final maximum",
            boundary_mode="causal-final",
        )
    if name in {
        "blocks/attention/softmax/shifted",
        "blocks/attention/softmax/exp",
    }:
        return _OwnerPlan(
            "softmax-exponential",
            "Softmax exponential value",
            "blocks/attention/softmax/exp",
            "post-kv",
            True,
            "shift and exponential ending at a branching exponential",
        )
    if name in {
        "blocks/attention/softmax/denominator",
        "blocks/attention/softmax/reciprocal",
    }:
        return _OwnerPlan(
            "softmax-reciprocal",
            "Softmax reciprocal denominator",
            "blocks/attention/softmax/reciprocal",
            "post-kv",
            True,
            "denominator reduction ending at the shared reciprocal",
        )
    if name == "blocks/attention/softmax/probability":
        return _OwnerPlan(
            "softmax-probability",
            "Softmax probability",
            name,
            "post-kv",
            False,
            "one narrow normalized probability",
        )
    if name.startswith("lm_head/") or name == "output/argmax":
        return _OwnerPlan(
            "token-output",
            "Atomic generated token output",
            "output/argmax",
            "output",
            False,
            "LM-head computation ending at its designated atomic argmax",
        )
    raise ValueError(f"unclassified computed family: {name!r}")


def _has_future_bypass(plan: _OwnerPlan, layer: int | None, layers: int) -> bool:
    if plan.stage in {"embedding", "ln1", "k", "v"}:
        return True
    if plan.stage in {"final", "output"}:
        return False
    if layer is None:
        raise AssertionError(f"{plan.row_id}: block stage has no layer")
    return layer + 1 < layers


def _capacity_for_plan(
    indexed: GPT2IndexedCircuit,
    plan: _OwnerPlan,
) -> LogCardinality:
    return indexed.circuit.families[plan.boundary_family].capacity.log_value


def _support_reasons(indexed: GPT2IndexedCircuit) -> tuple[str, ...]:
    circuit = indexed.circuit
    token = circuit.families["output/argmax"].capacity.log_value
    non_input = [
        family.capacity.log_value
        for family in circuit.families.values()
        if family.op != "input"
    ]
    non_token = [
        capacity
        for name, family in circuit.families.items()
        if family.op != "input" and name != "output/argmax"
        for capacity in (family.capacity.log_value,)
    ]
    reasons: list[str] = []
    if any(token > capacity for capacity in non_input):
        reasons.append("the atomic token is not a minimum-capacity gate")
    if non_token and token.scale(2) <= min(non_token):
        reasons.append(
            "two token-capacity gates do not exceed every narrow local boundary"
        )
    if non_input and token.scale(3) <= max(non_input):
        reasons.append(
            "three token-capacity gates do not exceed the widest source self-cut"
        )
    config = indexed.config
    if (
        min(
            config.hidden_size,
            config.head_size,
            config.intermediate_size,
            config.vocabulary_size,
        )
        < 3
    ):
        reasons.append(
            "a dense or vector fan-out has fewer than three independent coordinates"
        )
    if indexed.processed_positions < 3:
        reasons.append("fewer than three causal positions need a degenerate summary")
    return tuple(reasons)


def _validate_owner_topology(indexed: GPT2IndexedCircuit) -> tuple[str, ...]:
    """Verify local summaries against the actual family-level wire graph."""

    circuit = indexed.circuit
    plans = {
        name: _owner_plan(name)
        for name, family in circuit.families.items()
        if family.op != "input"
    }
    reasons: list[str] = []
    for rule in circuit.edge_rules:
        source_family = circuit.families[rule.source_family]
        if source_family.op == "input":
            continue
        source_plan = plans[rule.source_family]
        target_plan = plans[rule.target_family]
        source_owner = (source_plan.row_id, source_plan.boundary_family)
        target_owner = (target_plan.row_id, target_plan.boundary_family)
        if source_owner == target_owner:
            continue
        if rule.source_family != source_plan.boundary_family:
            reasons.append(
                f"wire rule {rule.name!r} exits owner {source_owner!r} from "
                f"non-boundary family {rule.source_family!r}"
            )
    for owner in set((plan.row_id, plan.boundary_family) for plan in plans.values()):
        boundary = owner[1]
        if boundary not in circuit.families:
            reasons.append(f"owner {owner!r} names missing boundary {boundary!r}")
    return tuple(reasons)


def _new_accumulator(
    accumulators: dict[str, _Accumulator],
    *,
    row_id: str,
    bottleneck: str,
    capacity: LogCardinality,
    cut_gate_count: int,
    certificate_kind: CertificateKind,
    description: str,
    certificate: str,
    boundaries: tuple[str, ...],
) -> _Accumulator:
    existing = accumulators.get(row_id)
    if existing is not None:
        if (
            existing.capacity != capacity
            or existing.cut_gate_count != cut_gate_count
            or existing.certificate_kind is not certificate_kind
        ):
            raise AssertionError(f"inconsistent aggregate row {row_id!r}")
        existing.boundaries.update(boundaries)
        return existing
    result = _Accumulator(
        row_id=row_id,
        bottleneck=bottleneck,
        capacity=capacity,
        cut_gate_count=cut_gate_count,
        certificate_kind=certificate_kind,
        description=description,
        certificate=certificate,
        boundaries=set(boundaries),
    )
    accumulators[row_id] = result
    return result


def _local_accumulator(
    accumulators: dict[str, _Accumulator],
    indexed: GPT2IndexedCircuit,
    plan: _OwnerPlan,
) -> _Accumulator:
    kind = CertificateKind.BRANCH if plan.wide_branch else CertificateKind.FUNNEL
    certificate = (
        "The indexed edge rules preserve the occurrence key through the local "
        "owner region and every exit crosses the named boundary. The bounded "
        "cut-order proof excludes a cheaper later frontier; equal-width choices "
        "use the maximal residual source side, so this is downstream-most."
    )
    return _new_accumulator(
        accumulators,
        row_id=plan.row_id,
        bottleneck=plan.bottleneck,
        capacity=_capacity_for_plan(indexed, plan),
        cut_gate_count=1,
        certificate_kind=kind,
        description=plan.description,
        certificate=certificate,
        boundaries=(plan.boundary_family,),
    )


def _special_accumulator(
    accumulators: dict[str, _Accumulator],
    indexed: GPT2IndexedCircuit,
    row_id: str,
) -> _Accumulator:
    circuit = indexed.circuit
    token = circuit.families["output/argmax"].capacity.log_value
    probability = circuit.families[
        "blocks/attention/softmax/probability"
    ].capacity.log_value
    if row_id == "dead-empty-cut":
        return _new_accumulator(
            accumulators,
            row_id=row_id,
            bottleneck="Dead computed region",
            capacity=LogCardinality.zero(),
            cut_gate_count=0,
            certificate_kind=CertificateKind.EMPTY,
            description="computed gates with no path to any designated output",
            certificate=(
                "Causal liveness on the indexed rules reaches no y1..yG output, "
                "so the exact canonical cut is empty."
            ),
            boundaries=("∅",),
        )
    if row_id in {"token-output", "final-token-output"}:
        label = (
            "Terminal generated token output"
            if row_id == "token-output"
            else "Final generated token output"
        )
        return _new_accumulator(
            accumulators,
            row_id=row_id,
            bottleneck=label,
            capacity=token,
            cut_gate_count=1,
            certificate_kind=CertificateKind.OUTPUT,
            description=(
                "all owned paths pass through the same designated atomic argmax"
            ),
            certificate=(
                "The designated output is a postdominator, including its zero-edge "
                "self path. Its exact token cardinality is the least positive gate "
                "capacity, proving global minimality and uniqueness."
            ),
            boundaries=("output/argmax",),
        )
    if row_id == "penultimate-output-pair":
        return _new_accumulator(
            accumulators,
            row_id=row_id,
            bottleneck="Penultimate/final token pair",
            capacity=token.scale(2),
            cut_gate_count=2,
            certificate_kind=CertificateKind.OUTPUT_PAIR,
            description=(
                "wide branching sources reaching exactly the final two outputs"
            ),
            certificate=(
                "The two designated outputs cut every suffix path. Two internally "
                "disjoint output paths prove the exact lower bound, and the output "
                "frontier is the downstream-most minimum."
            ),
            boundaries=("output/argmax", "output/argmax"),
        )
    if row_id == "length-one-attention-probability":
        return _new_accumulator(
            accumulators,
            row_id=row_id,
            bottleneck="Length-one attention probability",
            capacity=probability,
            cut_gate_count=1,
            certificate_kind=CertificateKind.FUNNEL,
            description=(
                "Q, score, and degenerate one-key softmax work converging at one probability"
            ),
            certificate=(
                "At query zero the causal domain contains one key. Both numerator "
                "and denominator paths reconverge at the sole probability gate; "
                "exact rule substitution proves it is the downstream-most narrow cut."
            ),
            boundaries=("blocks/attention/softmax/probability",),
        )
    if row_id == "length-two-softmax-probability-pair":
        return _new_accumulator(
            accumulators,
            row_id=row_id,
            bottleneck="Length-two softmax probability pair",
            capacity=probability.scale(2),
            cut_gate_count=2,
            certificate_kind=CertificateKind.BRANCH,
            description=("wide two-key softmax state ending at both probability gates"),
            certificate=(
                "For query one the exact causal fan-out has two probability "
                "branches. They are a cut and give two disjoint witnesses; their "
                "capacity ties the FP32 source and is downstream-most."
            ),
            boundaries=(
                "blocks/attention/softmax/probability",
                "blocks/attention/softmax/probability",
            ),
        )
    raise AssertionError(f"unknown special row {row_id!r}")


def _assignment_row(
    indexed: GPT2IndexedCircuit,
    plan: _OwnerPlan,
    *,
    position: int,
    layer: int | None,
) -> str:
    first_prediction = indexed.prompt_tokens - 1
    generation = position - first_prediction
    if plan.stage == "output":
        return (
            "final-token-output"
            if generation == indexed.generated_tokens - 1
            else "token-output"
        )

    bypass = _has_future_bypass(plan, layer, indexed.config.layers)
    if not bypass:
        if 0 <= generation < indexed.generated_tokens:
            return (
                "final-token-output"
                if generation == indexed.generated_tokens - 1
                else "token-output"
            )
        return "dead-empty-cut"

    first_reachable = max(0, generation)
    reachable_count = indexed.generated_tokens - first_reachable
    if reachable_count <= 0:
        raise AssertionError("a bypass stage must reach a generated output")
    if reachable_count == 1:
        return "final-token-output"

    token = indexed.circuit.families["output/argmax"].capacity.log_value
    local = _capacity_for_plan(indexed, plan)
    output_pair = token.scale(2)
    has_output_pair = reachable_count == 2
    probability = indexed.circuit.families[
        "blocks/attention/softmax/probability"
    ].capacity.log_value
    if (
        position == 0
        and plan.stage in {"q", "score"}
        and probability <= local
        and (not has_output_pair or probability < output_pair)
    ):
        return "length-one-attention-probability"
    if (
        position == 0
        and plan.row_id
        in {
            "softmax-exponential",
            "softmax-reciprocal",
            "softmax-probability",
        }
        and probability <= local
        and (not has_output_pair or probability < output_pair)
    ):
        return "length-one-attention-probability"
    probability_pair = probability.scale(2)
    if (
        position == 1
        and plan.wide_branch
        and plan.row_id
        in {
            "softmax-maximum",
            "softmax-exponential",
            "softmax-reciprocal",
        }
        and probability_pair <= local
        and (not has_output_pair or probability_pair < output_pair)
    ):
        return "length-two-softmax-probability-pair"
    if has_output_pair and output_pair <= local:
        return "penultimate-output-pair"
    return plan.row_id


def _boundary_occurrences(
    family: GateFamily,
    plan: _OwnerPlan,
    *,
    layer: int | None,
    position: int,
    source_count: int,
) -> int:
    if family.name != plan.boundary_family:
        return 0
    if plan.boundary_mode != "causal-final":
        return source_count
    if position <= 0:
        return 0
    fixed: dict[str, int] = {"query": position, "key": position}
    if layer is not None:
        fixed["layer"] = layer
    return _count_family(family, **fixed)


def _iter_family_chunks(
    indexed: GPT2IndexedCircuit,
    family: GateFamily,
) -> tuple[tuple[int | None, int, int], ...]:
    """Return ``(layer, semantic_position, count)`` descriptor chunks."""

    layer_values: tuple[int | None, ...]
    if "layer" in family.index_names:
        layer_values = tuple(range(indexed.config.layers))
    else:
        layer_values = (None,)
    if "position" in family.index_names:
        time_name = "position"
        times = range(indexed.processed_positions)
        position_offset = 0
    elif "query" in family.index_names:
        time_name = "query"
        times = range(indexed.processed_positions)
        position_offset = 0
    elif "generation" in family.index_names:
        time_name = "generation"
        times = range(
            indexed.generated_tokens
            if family.name.startswith("lm_head/") or family.name == "output/argmax"
            else indexed.generated_tokens - 1
        )
        position_offset = (
            indexed.prompt_tokens - 1
            if family.name.startswith("lm_head/") or family.name == "output/argmax"
            else indexed.prompt_tokens
        )
    else:
        raise ValueError(f"{family.name}: no schedule coordinate")

    chunks: list[tuple[int | None, int, int]] = []
    for layer in layer_values:
        for time_coordinate in times:
            fixed = {time_name: time_coordinate}
            if layer is not None:
                fixed["layer"] = layer
            count = _count_family(family, **fixed)
            if count:
                chunks.append((layer, position_offset + time_coordinate, count))
    return tuple(chunks)


def _rows_from_accumulators(
    accumulators: Mapping[str, _Accumulator],
) -> tuple[WiringBottleneckRecord, ...]:
    rows: list[WiringBottleneckRecord] = []
    preferred_order = (
        "dead-empty-cut",
        "embedding-coordinate",
        "layernorm-mean",
        "layernorm-centered",
        "layernorm-inverse-std",
        "layernorm-coordinate-output",
        "q-projection-output",
        "k-cache-entry",
        "v-cache-entry",
        "attention-score-output",
        "softmax-maximum",
        "softmax-exponential",
        "softmax-reciprocal",
        "softmax-probability",
        "length-one-attention-probability",
        "length-two-softmax-probability-pair",
        "attention-value-output",
        "attention-residual",
        "gelu-output",
        "mlp-residual",
        "penultimate-output-pair",
        "token-output",
        "final-token-output",
    )
    rank = {row_id: index for index, row_id in enumerate(preferred_order)}
    for accumulator in sorted(
        accumulators.values(),
        key=lambda item: (rank.get(item.row_id, len(rank)), item.row_id),
    ):
        primitives = _primitive_vector(accumulator.primitive_counts)
        if primitives.total == 0:
            continue
        if accumulator.occurrence_count <= 0:
            raise AssertionError(f"{accumulator.row_id}: no concrete cut occurrences")
        width = float(accumulator.capacity.width_bits)
        rows.append(
            WiringBottleneckRecord(
                row_id=accumulator.row_id,
                bottleneck=accumulator.bottleneck,
                boundary_families=tuple(sorted(accumulator.boundaries)),
                cut_width_expression_bits=accumulator.capacity.expression,
                cut_width_min_bits=width,
                cut_width_max_bits=width,
                occurrence_count=accumulator.occurrence_count,
                represented_primitives=primitives,
                upstream_operations_per_cut=accumulator.description,
                cut_certificate=accumulator.certificate,
                global_minimum_status=(
                    "Globally minimum and downstream-most under the lifted certificate."
                ),
                source_gate_count=accumulator.source_gate_count,
                cut_gate_count=accumulator.cut_gate_count,
                certificate_kind=accumulator.certificate_kind.value,
            )
        )
    return tuple(rows)


def _lifted_partition(indexed: GPT2IndexedCircuit) -> GPT2CanonicalPartition:
    started = time.perf_counter()
    circuit = indexed.circuit
    unsupported = _support_reasons(indexed)
    if unsupported:
        return GPT2CanonicalPartition(
            status=ExactPartitionStatus.UNSUPPORTED,
            rows=(),
            reasons=unsupported,
            covered_source_gate_count=0,
            covered_primitive_count=0,
            evaluated_index_regions=0,
            retained_region_descriptors=0,
            validated_edge_rule_count=0,
            solver="bounded-cut-order hierarchical lift",
            elapsed_seconds=time.perf_counter() - started,
        )

    topology_reasons = _validate_owner_topology(indexed)
    if topology_reasons:
        return GPT2CanonicalPartition(
            status=ExactPartitionStatus.UNSUPPORTED,
            rows=(),
            reasons=topology_reasons,
            covered_source_gate_count=0,
            covered_primitive_count=0,
            evaluated_index_regions=0,
            retained_region_descriptors=0,
            validated_edge_rule_count=0,
            solver="bounded-cut-order hierarchical lift",
            elapsed_seconds=time.perf_counter() - started,
        )

    accumulators: dict[str, _Accumulator] = {}
    retained_descriptors: set[tuple[str, str]] = set()
    evaluated_regions = 0
    family_coverage: Counter[str] = Counter()

    for family in circuit.families.values():
        if family.op == "input":
            continue
        plan = _owner_plan(family.name)
        for layer, position, source_count in _iter_family_chunks(indexed, family):
            evaluated_regions += 1
            row_id = _assignment_row(
                indexed,
                plan,
                position=position,
                layer=layer,
            )
            is_local = row_id == plan.row_id and plan.stage != "output"
            if is_local:
                accumulator = _local_accumulator(accumulators, indexed, plan)
            else:
                accumulator = _special_accumulator(accumulators, indexed, row_id)
            accumulator.add_sources(family, source_count)
            retained_descriptors.add((row_id, family.name))
            family_coverage[family.name] += source_count

            if is_local:
                accumulator.occurrence_count += _boundary_occurrences(
                    family,
                    plan,
                    layer=layer,
                    position=position,
                    source_count=source_count,
                )
            elif (
                row_id == "length-one-attention-probability"
                and family.name == "blocks/attention/softmax/probability"
            ):
                accumulator.occurrence_count += source_count
            elif (
                row_id == "length-two-softmax-probability-pair"
                and family.name == "blocks/attention/softmax/reciprocal"
            ):
                accumulator.occurrence_count += source_count

    uncovered = {
        name: family.count - family_coverage[name]
        for name, family in circuit.families.items()
        if family.op != "input" and family_coverage[name] != family.count
    }
    if uncovered:
        raise AssertionError(f"family region coverage mismatch: {uncovered}")

    if "dead-empty-cut" in accumulators:
        accumulators["dead-empty-cut"].occurrence_count = 1
    if "penultimate-output-pair" in accumulators:
        accumulators["penultimate-output-pair"].occurrence_count = 1
    if "token-output" in accumulators:
        accumulators["token-output"].occurrence_count = max(
            indexed.generated_tokens - 1,
            1,
        )
    if "final-token-output" in accumulators:
        accumulators["final-token-output"].occurrence_count = 1

    rows = _rows_from_accumulators(accumulators)
    covered_sources = sum(
        accumulator.source_gate_count for accumulator in accumulators.values()
    )
    covered_primitives = sum(row.represented_primitives.total for row in rows)
    if covered_sources != circuit.computed_gate_count:
        raise AssertionError(
            f"lift covers {covered_sources:,} sources, expected "
            f"{circuit.computed_gate_count:,}"
        )
    if covered_primitives != circuit.primitive_gate_count:
        raise AssertionError(
            f"lift owns {covered_primitives:,} primitive gates, expected "
            f"{circuit.primitive_gate_count:,}"
        )

    reasons = (
        (
            "Every local owner is closed against the actual indexed edge-rule graph "
            f"except at its named boundary; all {len(circuit.edge_rules)} wire rules "
            "were checked before lifting."
        ),
        (
            "Every non-input indexed family is partitioned into exact layer/causal "
            "regions; algebraic region counts equal the full computed gate count."
        ),
        (
            "The capacity-order certificate limits any winning singleton-source "
            "cut to at most two gates; local funnels, branch frontiers, token "
            "postdominators, and causal dead regions exhaust those cases."
        ),
        (
            "Tiny materialized GPT configurations use the same finite-DAG oracle "
            "to validate source-most and downstream-most extremal cuts gate-for-gate."
        ),
    )
    return GPT2CanonicalPartition(
        status=ExactPartitionStatus.COMPUTED,
        rows=rows,
        reasons=reasons,
        covered_source_gate_count=covered_sources,
        covered_primitive_count=covered_primitives,
        evaluated_index_regions=evaluated_regions,
        retained_region_descriptors=len(retained_descriptors),
        validated_edge_rule_count=len(circuit.edge_rules),
        solver="bounded-cut-order hierarchical lift",
        elapsed_seconds=time.perf_counter() - started,
    )


def _explicit_rows(
    indexed: GPT2IndexedCircuit,
    partition: GateCutPartition,
) -> tuple[WiringBottleneckRecord, ...]:
    circuit = indexed.circuit
    rows: list[WiringBottleneckRecord] = []
    for group_index, group in enumerate(
        sorted(partition.groups, key=lambda item: (len(item.cut), sorted(item.cut)))
    ):
        counts: Counter[str] = Counter()
        source_families: set[str] = set()
        for gate_id in group.source_gates:
            ref = circuit.ref_from_id(gate_id)
            family = circuit.families[ref.family]
            source_families.add(ref.family)
            if family.primitive is not None:
                counts[family.primitive] += 1
        primitives = _primitive_vector(counts)
        if primitives.total == 0:
            continue
        boundary_families = tuple(
            sorted({circuit.ref_from_id(gate_id).family for gate_id in group.cut})
        ) or ("∅",)
        rows.append(
            WiringBottleneckRecord(
                row_id=f"explicit-cut-{group_index:04d}",
                bottleneck=(
                    "Empty dead cut" if not group.cut else " / ".join(boundary_families)
                ),
                boundary_families=boundary_families,
                cut_width_expression_bits=group.exact_capacity.expression,
                cut_width_min_bits=float(group.width_bits),
                cut_width_max_bits=float(group.width_bits),
                occurrence_count=1,
                represented_primitives=primitives,
                upstream_operations_per_cut=(
                    f"{len(group.source_gates)} concrete source gates in "
                    f"{len(source_families)} indexed families"
                ),
                cut_certificate=(
                    "Materialized exact vertex-split max flow; the group cut was "
                    "also checked jointly against all owned sources."
                ),
                global_minimum_status="Exact finite-DAG oracle result.",
                source_gate_count=len(group.source_gates),
                cut_gate_count=len(group.cut),
                certificate_kind="materialized-max-flow",
            )
        )
    return tuple(rows)


def _ref_schedule(
    indexed: GPT2IndexedCircuit,
    ref: GateRef,
) -> tuple[int | None, int]:
    family = indexed.circuit.families[ref.family]
    coordinates = dict(zip(family.index_names, ref.index, strict=True))
    layer = coordinates.get("layer")
    if "position" in coordinates:
        return layer, coordinates["position"]
    if "query" in coordinates:
        return layer, coordinates["query"]
    generation = coordinates.get("generation")
    if generation is None:
        raise ValueError(f"{ref.id}: source has no schedule coordinate")
    if ref.family.startswith("lm_head/") or ref.family == "output/argmax":
        return layer, indexed.prompt_tokens - 1 + generation
    return layer, indexed.prompt_tokens + generation


def _owner_boundary_ref(
    indexed: GPT2IndexedCircuit,
    ref: GateRef,
    plan: _OwnerPlan,
) -> GateRef:
    family = indexed.circuit.families[ref.family]
    coordinates = dict(zip(family.index_names, ref.index, strict=True))
    layer = coordinates.get("layer")
    position = coordinates.get("position", coordinates.get("query"))

    if plan.stage == "embedding":
        coordinate = coordinates["coordinate"]
        if ref.family == "embedding/generated_lookup":
            position = indexed.prompt_tokens + coordinates["generation"]
        assert position is not None
        return GateRef(plan.boundary_family, (position, coordinate))
    if plan.stage in {"ln1", "ln2", "final"}:
        assert position is not None
        outer = (position,) if layer is None else (layer, position)
        if plan.row_id == "layernorm-coordinate-output":
            return GateRef(plan.boundary_family, (*outer, coordinates["coordinate"]))
        if plan.row_id == "layernorm-centered":
            return GateRef(plan.boundary_family, (*outer, coordinates["coordinate"]))
        return GateRef(plan.boundary_family, outer)
    if plan.stage in {"q", "k", "v"}:
        assert layer is not None and position is not None
        output = coordinates["output"]
        return GateRef(plan.boundary_family, (layer, position, output))
    if plan.stage == "score":
        return GateRef(
            plan.boundary_family,
            (
                coordinates["layer"],
                coordinates["query"],
                coordinates["key"],
                coordinates["head"],
            ),
        )
    if plan.stage == "value":
        if ref.family == plan.boundary_family:
            return ref
        flat_coordinate = (
            coordinates["head"] * indexed.config.head_size + coordinates["coordinate"]
        )
        return GateRef(
            plan.boundary_family,
            (coordinates["layer"], coordinates["query"], flat_coordinate),
        )
    if plan.row_id == "attention-residual":
        if ref.family == "blocks/residual1":
            return ref
        return GateRef(
            plan.boundary_family,
            (coordinates["layer"], coordinates["position"], coordinates["output"]),
        )
    if plan.row_id == "gelu-output":
        gelu_coordinate: int | None = coordinates.get(
            "coordinate",
            coordinates.get("output"),
        )
        assert gelu_coordinate is not None
        return GateRef(
            plan.boundary_family,
            (coordinates["layer"], coordinates["position"], gelu_coordinate),
        )
    if plan.row_id == "mlp-residual":
        if ref.family == "blocks/residual2":
            return ref
        return GateRef(
            plan.boundary_family,
            (coordinates["layer"], coordinates["position"], coordinates["output"]),
        )
    if plan.row_id == "softmax-maximum":
        return GateRef(
            plan.boundary_family,
            (
                coordinates["layer"],
                coordinates["query"],
                coordinates["query"],
                coordinates["head"],
            ),
        )
    if plan.row_id == "softmax-exponential":
        return GateRef(
            plan.boundary_family,
            (
                coordinates["layer"],
                coordinates["query"],
                coordinates["key"],
                coordinates["head"],
            ),
        )
    if plan.row_id == "softmax-reciprocal":
        return GateRef(
            plan.boundary_family,
            (
                coordinates["layer"],
                coordinates["query"],
                coordinates["head"],
            ),
        )
    if plan.row_id == "softmax-probability":
        return ref
    raise AssertionError(f"no boundary projection for {ref.id}")


def lifted_certificate_reasons(indexed: GPT2IndexedCircuit) -> tuple[str, ...]:
    """Return why the lifted certificates do not apply, or empty when they do.

    ``lifted_downstream_cut`` is exact only when this returns no reasons: the
    capacity-order argument must hold and every local owner must be closed
    against the actual indexed wire rules.
    """

    return (*_support_reasons(indexed), *_validate_owner_topology(indexed))


def lifted_downstream_cut(
    indexed: GPT2IndexedCircuit,
    ref: GateRef,
) -> frozenset[GateRef]:
    """Lift one scalar source to its certified downstream-most cut.

    This is intended for tiny-oracle validation and spot certificates.  The
    full partition counts regions algebraically and does not call it billions
    of times.
    """

    family = indexed.circuit.require_ref(ref)
    if family.op == "input":
        raise ValueError("the canonical GPT-2 partition selects computed gates only")
    plan = _owner_plan(ref.family)
    layer, position = _ref_schedule(indexed, ref)
    row_id = _assignment_row(
        indexed,
        plan,
        position=position,
        layer=layer,
    )
    output_family = "output/argmax"
    if row_id == "dead-empty-cut":
        return frozenset()
    if row_id == "token-output":
        generation = position - (indexed.prompt_tokens - 1)
        return frozenset((GateRef(output_family, (generation,)),))
    if row_id == "final-token-output":
        return frozenset((GateRef(output_family, (indexed.generated_tokens - 1,)),))
    if row_id == "penultimate-output-pair":
        return frozenset(
            (
                GateRef(output_family, (indexed.generated_tokens - 2,)),
                GateRef(output_family, (indexed.generated_tokens - 1,)),
            )
        )

    coordinates = dict(zip(family.index_names, ref.index, strict=True))
    if row_id == "length-one-attention-probability":
        if plan.stage == "q":
            head = coordinates["output"] // indexed.config.head_size
        else:
            head = coordinates["head"]
        selected_layer = coordinates.get("layer", layer)
        assert selected_layer is not None
        return frozenset(
            (
                GateRef(
                    "blocks/attention/softmax/probability",
                    (selected_layer, 0, 0, head),
                ),
            )
        )
    if row_id == "length-two-softmax-probability-pair":
        selected_layer = coordinates.get("layer", layer)
        assert selected_layer is not None
        head = coordinates["head"]
        return frozenset(
            GateRef(
                "blocks/attention/softmax/probability",
                (selected_layer, 1, key, head),
            )
            for key in range(2)
        )
    return frozenset((_owner_boundary_ref(indexed, ref, plan),))


def compute_gpt2_canonical_partition(
    indexed: GPT2IndexedCircuit,
    *,
    explicit_gate_limit: int = 2_000,
    force_explicit: bool = False,
) -> GPT2CanonicalPartition:
    """Compute the exact tiny oracle or the full hierarchical lifted partition."""

    circuit = indexed.circuit
    if force_explicit and circuit.gate_count > explicit_gate_limit:
        raise ExpansionLimitExceeded(
            f"explicit GPT-2 oracle has {circuit.gate_count:,} gates; "
            f"limit is {explicit_gate_limit:,}"
        )
    use_explicit = force_explicit or circuit.gate_count <= explicit_gate_limit
    if use_explicit:
        started = time.perf_counter()
        materialized = circuit.materialize(
            max_gates=circuit.gate_count,
            max_edges=max(circuit.gate_count * 20, 1_000),
        )
        partition = partition_gate_cuts(materialized)
        rows = _explicit_rows(indexed, partition)
        represented = sum(row.represented_primitives.total for row in rows)
        if represented != circuit.primitive_gate_count:
            raise AssertionError(
                f"explicit groups own {represented:,} primitives, expected "
                f"{circuit.primitive_gate_count:,}"
            )
        return GPT2CanonicalPartition(
            status=ExactPartitionStatus.COMPUTED,
            rows=rows,
            reasons=(
                "The complete indexed graph was materialized within the tiny-oracle limit.",
                "Every singleton and grouped joint cut was solved by exact max flow.",
            ),
            covered_source_gate_count=circuit.computed_gate_count,
            covered_primitive_count=represented,
            evaluated_index_regions=circuit.computed_gate_count,
            retained_region_descriptors=len(partition.groups),
            validated_edge_rule_count=len(circuit.edge_rules),
            solver="materialized exact finite-DAG oracle",
            elapsed_seconds=time.perf_counter() - started,
            explicit_partition=partition,
        )
    return _lifted_partition(indexed)
