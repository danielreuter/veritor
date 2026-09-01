"""Edge-complete computed-source projection for GPT-2 greedy generation.

The graph is stored as gate families and bidirectional index relations.  It
contains every declared scalar arithmetic gate, explicit narrow write-out
boundaries after contractions, LayerNorm fan-in/fan-out, causal K/V reuse,
generated-token feedback, complete LM heads, and atomic argmax outputs.

Fixed weights and constants are literals in gate definitions rather than
vertices. This is therefore not the full parameter-input circuit. It is an
edge-complete projection for paths by which an incorrect *computed* gate can
affect an output, which is the default source relation used by the cut analysis.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from circuit_cut_analysis.indexed import (
    CausalPairsDomain,
    CausalReductionStepsDomain,
    EdgeRule,
    GateCapacity,
    GateFamily,
    GateRef,
    Index,
    IndexDomain,
    IndexedCircuit,
    ProductDomain,
    RectangularDomain,
)
from circuit_cut_analysis.models.gpt2 import GPT2Config
from circuit_cut_analysis.profiles import VLLM_FP16_REFERENCE, ServingProfile

SourceForCoordinate = Callable[[Index, int], Index | None]
CoordinatesForSource = Callable[[Index], Iterable[tuple[Index, int]]]


@dataclass(frozen=True, slots=True)
class VectorInputBinding:
    """One source family feeding a subset of vector occurrences."""

    source_family: str
    source_for_coordinate: SourceForCoordinate
    coordinates_for_source: CoordinatesForSource


@dataclass(frozen=True, slots=True)
class ProjectionResult:
    mul_family: str
    accumulator_family: str | None
    output_family: str


@dataclass(frozen=True, slots=True)
class LayerNormResult:
    output_family: str
    mean_family: str
    inverse_std_family: str


@dataclass(frozen=True, slots=True)
class GPT2IndexedCircuit:
    """Indexed graph plus the schedule needed to interpret its coordinates."""

    circuit: IndexedCircuit
    config: GPT2Config
    profile: ServingProfile
    prompt_tokens: int
    generated_tokens: int
    processed_positions: int
    prediction_positions: tuple[int, ...]

    @property
    def output_refs(self) -> tuple[GateRef, ...]:
        return tuple(sorted(self.circuit.outputs))


class _Builder:
    def __init__(self) -> None:
        self.families: list[GateFamily] = []
        self.rules: list[EdgeRule] = []
        self._family_names: set[str] = set()
        self._rule_names: set[str] = set()

    def family(
        self,
        name: str,
        domain: IndexDomain,
        index_names: tuple[str, ...],
        capacity: GateCapacity,
        op: str,
        primitive: str | None = None,
        *,
        tags: tuple[str, ...] = (),
    ) -> str:
        if name in self._family_names:
            raise ValueError(f"duplicate family during construction: {name!r}")
        self._family_names.add(name)
        self.families.append(
            GateFamily(
                name=name,
                domain=domain,
                index_names=index_names,
                capacity=capacity,
                op=op,
                primitive=primitive,
                tags=tags,
            )
        )
        return name

    def rule(
        self,
        name: str,
        source_family: str,
        target_family: str,
        sources_for_target: Callable[[Index], Iterable[Index]],
        targets_for_source: Callable[[Index], Iterable[Index]],
    ) -> None:
        if name in self._rule_names:
            raise ValueError(f"duplicate rule during construction: {name!r}")
        self._rule_names.add(name)
        self.rules.append(
            EdgeRule(
                name=name,
                source_family=source_family,
                target_family=target_family,
                sources_for_target=sources_for_target,
                targets_for_source=targets_for_source,
            )
        )

    def identity_rule(self, name: str, source_family: str, target_family: str) -> None:
        self.rule(
            name,
            source_family,
            target_family,
            lambda target: (target,),
            lambda source: (source,),
        )

    def finish(self, outputs: Iterable[GateRef]) -> IndexedCircuit:
        return IndexedCircuit(self.families, self.rules, outputs)


def _with_suffix(domain: IndexDomain, *sizes: int) -> ProductDomain:
    return ProductDomain((domain, RectangularDomain(tuple(sizes))))


def _add_biased_projection(
    builder: _Builder,
    *,
    name: str,
    outer_domain: IndexDomain,
    outer_names: tuple[str, ...],
    input_family: str,
    input_size: int,
    output_size: int,
    input_index_for: Callable[[Index, int], Index],
    input_uses_for: Callable[[Index], Iterable[tuple[Index, int]]],
    accumulator_capacity: GateCapacity,
    output_capacity: GateCapacity,
    tags: tuple[str, ...],
) -> ProjectionResult:
    """Add ``output_size`` biased dot products for each outer index."""

    mul = builder.family(
        f"{name}/mul",
        _with_suffix(outer_domain, output_size, input_size),
        (*outer_names, "output", "term"),
        accumulator_capacity,
        "mul",
        "mul",
        tags=tags,
    )
    accumulator = builder.family(
        f"{name}/accumulator",
        _with_suffix(outer_domain, output_size, input_size),
        (*outer_names, "output", "step"),
        accumulator_capacity,
        "add",
        "add",
        tags=tags,
    )
    output = builder.family(
        f"{name}/write",
        _with_suffix(outer_domain, output_size),
        (*outer_names, "output"),
        output_capacity,
        "write",
        tags=(*tags, "inner-product-output", "boundary"),
    )
    outer_arity = outer_domain.arity

    def input_sources(target: Index) -> Iterable[Index]:
        outer = target[:outer_arity]
        term = target[-1]
        return (input_index_for(outer, term),)

    def input_targets(source: Index) -> Iterable[Index]:
        for outer, term in input_uses_for(source):
            for output_coordinate in range(output_size):
                yield (*outer, output_coordinate, term)

    builder.rule(
        f"{name}:input-to-mul",
        input_family,
        mul,
        input_sources,
        input_targets,
    )
    builder.identity_rule(f"{name}:mul-to-accumulator", mul, accumulator)

    def previous_accumulator(target: Index) -> Iterable[Index]:
        step = target[-1]
        if step == 0:
            return ()
        return ((*target[:-1], step - 1),)

    def next_accumulator(source: Index) -> Iterable[Index]:
        step = source[-1]
        if step + 1 >= input_size:
            return ()
        return ((*source[:-1], step + 1),)

    builder.rule(
        f"{name}:accumulator-chain",
        accumulator,
        accumulator,
        previous_accumulator,
        next_accumulator,
    )

    def final_accumulator(target: Index) -> Iterable[Index]:
        return ((*target, input_size - 1),)

    def accumulator_write(source: Index) -> Iterable[Index]:
        if source[-1] != input_size - 1:
            return ()
        return (source[:-1],)

    builder.rule(
        f"{name}:accumulator-to-write",
        accumulator,
        output,
        final_accumulator,
        accumulator_write,
    )
    return ProjectionResult(mul, accumulator, output)


def _add_unbiased_projection(
    builder: _Builder,
    *,
    name: str,
    outer_domain: IndexDomain,
    outer_names: tuple[str, ...],
    input_family: str,
    input_size: int,
    output_size: int,
    input_index_for: Callable[[Index, int], Index],
    input_uses_for: Callable[[Index], Iterable[tuple[Index, int]]],
    accumulator_capacity: GateCapacity,
    output_capacity: GateCapacity,
    tags: tuple[str, ...],
) -> ProjectionResult:
    """Add ``output_size`` unbiased dot products for each outer index."""

    mul = builder.family(
        f"{name}/mul",
        _with_suffix(outer_domain, output_size, input_size),
        (*outer_names, "output", "term"),
        accumulator_capacity,
        "mul",
        "mul",
        tags=tags,
    )
    accumulator: str | None = None
    if input_size > 1:
        accumulator = builder.family(
            f"{name}/accumulator",
            _with_suffix(outer_domain, output_size, input_size - 1),
            (*outer_names, "output", "step"),
            accumulator_capacity,
            "add",
            "add",
            tags=tags,
        )
    output = builder.family(
        f"{name}/write",
        _with_suffix(outer_domain, output_size),
        (*outer_names, "output"),
        output_capacity,
        "write",
        tags=(*tags, "inner-product-output", "boundary"),
    )
    outer_arity = outer_domain.arity

    def input_sources(target: Index) -> Iterable[Index]:
        outer = target[:outer_arity]
        term = target[-1]
        return (input_index_for(outer, term),)

    def input_targets(source: Index) -> Iterable[Index]:
        for outer, term in input_uses_for(source):
            for output_coordinate in range(output_size):
                yield (*outer, output_coordinate, term)

    builder.rule(
        f"{name}:input-to-mul",
        input_family,
        mul,
        input_sources,
        input_targets,
    )

    if accumulator is None:
        builder.rule(
            f"{name}:mul-to-write",
            mul,
            output,
            lambda target: ((*target, 0),),
            lambda source: (source[:-1],) if source[-1] == 0 else (),
        )
        return ProjectionResult(mul, None, output)

    def product_sources(target: Index) -> Iterable[Index]:
        step = target[-1]
        if step == 0:
            return ((*target[:-1], 0), (*target[:-1], 1))
        return ((*target[:-1], step + 1),)

    def product_targets(source: Index) -> Iterable[Index]:
        term = source[-1]
        if term == 0:
            return ((*source[:-1], 0),)
        return ((*source[:-1], term - 1),)

    builder.rule(
        f"{name}:mul-to-accumulator",
        mul,
        accumulator,
        product_sources,
        product_targets,
    )

    def previous_accumulator(target: Index) -> Iterable[Index]:
        step = target[-1]
        if step == 0:
            return ()
        return ((*target[:-1], step - 1),)

    def next_accumulator(source: Index) -> Iterable[Index]:
        step = source[-1]
        if step + 1 >= input_size - 1:
            return ()
        return ((*source[:-1], step + 1),)

    builder.rule(
        f"{name}:accumulator-chain",
        accumulator,
        accumulator,
        previous_accumulator,
        next_accumulator,
    )
    builder.rule(
        f"{name}:accumulator-to-write",
        accumulator,
        output,
        lambda target: ((*target, input_size - 2),),
        lambda source: (source[:-1],) if source[-1] == input_size - 2 else (),
    )
    return ProjectionResult(mul, accumulator, output)


def _mean_target_for_coordinate(
    outer: Index,
    coordinate: int,
    size: int,
) -> Index:
    if size == 1:
        return outer
    step = 0 if coordinate <= 1 else coordinate - 1
    return (*outer, step)


def _add_layer_norm(
    builder: _Builder,
    *,
    name: str,
    outer_domain: IndexDomain,
    outer_names: tuple[str, ...],
    size: int,
    inputs: tuple[VectorInputBinding, ...],
    input_capacity: GateCapacity,
    internal_capacity: GateCapacity,
    output_capacity: GateCapacity,
    tags: tuple[str, ...],
) -> LayerNormResult:
    """Add an exact two-pass LayerNorm with shared statistics."""

    del input_capacity  # The source families carry their own exact capacities.
    reduction_domain = _with_suffix(outer_domain, size - 1)
    vector_domain = _with_suffix(outer_domain, size)
    mean_sum: str | None = None
    if size > 1:
        mean_sum = builder.family(
            f"{name}/mean_sum",
            reduction_domain,
            (*outer_names, "step"),
            internal_capacity,
            "add",
            "add",
            tags=tags,
        )
    mean = builder.family(
        f"{name}/mean",
        outer_domain,
        outer_names,
        internal_capacity,
        "mul",
        "mul",
        tags=(*tags, "shared-statistic"),
    )
    centered = builder.family(
        f"{name}/centered",
        vector_domain,
        (*outer_names, "coordinate"),
        internal_capacity,
        "add",
        "add",
        tags=tags,
    )
    square = builder.family(
        f"{name}/square",
        vector_domain,
        (*outer_names, "coordinate"),
        internal_capacity,
        "mul",
        "mul",
        tags=tags,
    )
    variance_sum: str | None = None
    if size > 1:
        variance_sum = builder.family(
            f"{name}/variance_sum",
            reduction_domain,
            (*outer_names, "step"),
            internal_capacity,
            "add",
            "add",
            tags=tags,
        )
    variance = builder.family(
        f"{name}/variance",
        outer_domain,
        outer_names,
        internal_capacity,
        "mul",
        "mul",
        tags=tags,
    )
    stabilized = builder.family(
        f"{name}/stabilized_variance",
        outer_domain,
        outer_names,
        internal_capacity,
        "add",
        "add",
        tags=tags,
    )
    inverse_std = builder.family(
        f"{name}/inverse_std",
        outer_domain,
        outer_names,
        internal_capacity,
        "rsqrt",
        "rsqrt",
        tags=(*tags, "shared-statistic"),
    )
    normalized = builder.family(
        f"{name}/normalized",
        vector_domain,
        (*outer_names, "coordinate"),
        internal_capacity,
        "mul",
        "mul",
        tags=tags,
    )
    affine_scale = builder.family(
        f"{name}/affine_scale",
        vector_domain,
        (*outer_names, "coordinate"),
        internal_capacity,
        "mul",
        "mul",
        tags=tags,
    )
    output = builder.family(
        f"{name}/write",
        vector_domain,
        (*outer_names, "coordinate"),
        output_capacity,
        "add",
        "add",
        tags=(*tags, "layernorm-output", "boundary"),
    )
    outer_arity = outer_domain.arity

    for binding_index, binding in enumerate(inputs):
        rule_prefix = f"{name}:input-{binding_index}"

        def centered_sources(
            target: Index,
            *,
            selected: VectorInputBinding = binding,
        ) -> Iterable[Index]:
            outer = target[:outer_arity]
            coordinate = target[-1]
            source = selected.source_for_coordinate(outer, coordinate)
            return () if source is None else (source,)

        def centered_targets(
            source: Index,
            *,
            selected: VectorInputBinding = binding,
        ) -> Iterable[Index]:
            for outer, coordinate in selected.coordinates_for_source(source):
                yield (*outer, coordinate)

        builder.rule(
            f"{rule_prefix}-to-centered",
            binding.source_family,
            centered,
            centered_sources,
            centered_targets,
        )

        if mean_sum is None:

            def mean_sources(
                target: Index,
                *,
                selected: VectorInputBinding = binding,
            ) -> Iterable[Index]:
                source = selected.source_for_coordinate(target, 0)
                return () if source is None else (source,)

            def mean_targets(
                source: Index,
                *,
                selected: VectorInputBinding = binding,
            ) -> Iterable[Index]:
                for outer, coordinate in selected.coordinates_for_source(source):
                    if coordinate == 0:
                        yield outer

            builder.rule(
                f"{rule_prefix}-to-mean",
                binding.source_family,
                mean,
                mean_sources,
                mean_targets,
            )
        else:

            def sum_sources(
                target: Index,
                *,
                selected: VectorInputBinding = binding,
            ) -> Iterable[Index]:
                outer = target[:outer_arity]
                step = target[-1]
                coordinates = (0, 1) if step == 0 else (step + 1,)
                for coordinate in coordinates:
                    source = selected.source_for_coordinate(outer, coordinate)
                    if source is not None:
                        yield source

            def sum_targets(
                source: Index,
                *,
                selected: VectorInputBinding = binding,
            ) -> Iterable[Index]:
                for outer, coordinate in selected.coordinates_for_source(source):
                    yield _mean_target_for_coordinate(outer, coordinate, size)

            builder.rule(
                f"{rule_prefix}-to-mean-sum",
                binding.source_family,
                mean_sum,
                sum_sources,
                sum_targets,
            )

    if mean_sum is not None:
        builder.rule(
            f"{name}:mean-sum-chain",
            mean_sum,
            mean_sum,
            lambda target: ((*target[:-1], target[-1] - 1),) if target[-1] > 0 else (),
            lambda source: (
                ((*source[:-1], source[-1] + 1),) if source[-1] + 1 < size - 1 else ()
            ),
        )
        builder.rule(
            f"{name}:mean-sum-to-mean",
            mean_sum,
            mean,
            lambda target: ((*target, size - 2),),
            lambda source: (source[:-1],) if source[-1] == size - 2 else (),
        )

    builder.rule(
        f"{name}:mean-to-centered",
        mean,
        centered,
        lambda target: (target[:-1],),
        lambda source: ((*source, coordinate) for coordinate in range(size)),
    )
    builder.identity_rule(f"{name}:centered-to-square", centered, square)
    builder.identity_rule(f"{name}:centered-to-normalized", centered, normalized)

    if variance_sum is None:
        builder.rule(
            f"{name}:square-to-variance",
            square,
            variance,
            lambda target: ((*target, 0),),
            lambda source: (source[:-1],) if source[-1] == 0 else (),
        )
    else:
        builder.rule(
            f"{name}:square-to-variance-sum",
            square,
            variance_sum,
            lambda target: (
                ((*target[:-1], 0), (*target[:-1], 1))
                if target[-1] == 0
                else ((*target[:-1], target[-1] + 1),)
            ),
            lambda source: (
                (_mean_target_for_coordinate(source[:-1], source[-1], size),)
            ),
        )
        builder.rule(
            f"{name}:variance-sum-chain",
            variance_sum,
            variance_sum,
            lambda target: ((*target[:-1], target[-1] - 1),) if target[-1] > 0 else (),
            lambda source: (
                ((*source[:-1], source[-1] + 1),) if source[-1] + 1 < size - 1 else ()
            ),
        )
        builder.rule(
            f"{name}:variance-sum-to-variance",
            variance_sum,
            variance,
            lambda target: ((*target, size - 2),),
            lambda source: (source[:-1],) if source[-1] == size - 2 else (),
        )

    builder.identity_rule(f"{name}:variance-to-stabilized", variance, stabilized)
    builder.identity_rule(
        f"{name}:stabilized-to-inverse-std",
        stabilized,
        inverse_std,
    )
    builder.rule(
        f"{name}:inverse-std-to-normalized",
        inverse_std,
        normalized,
        lambda target: (target[:-1],),
        lambda source: ((*source, coordinate) for coordinate in range(size)),
    )
    builder.identity_rule(
        f"{name}:normalized-to-affine-scale",
        normalized,
        affine_scale,
    )
    builder.identity_rule(f"{name}:affine-scale-to-write", affine_scale, output)
    return LayerNormResult(output, mean, inverse_std)


def _add_gelu(
    builder: _Builder,
    *,
    name: str,
    domain: IndexDomain,
    index_names: tuple[str, ...],
    input_family: str,
    internal_capacity: GateCapacity,
    output_capacity: GateCapacity,
) -> str:
    tags = ("mlp", "gelu")

    def family(stage: str, op: str, primitive: str, capacity: GateCapacity) -> str:
        return builder.family(
            f"{name}/{stage}",
            domain,
            index_names,
            capacity,
            op,
            primitive,
            tags=tags,
        )

    x2 = family("x2", "mul", "mul", internal_capacity)
    x3 = family("x3", "mul", "mul", internal_capacity)
    cubic = family("cubic", "mul", "mul", internal_capacity)
    inner = family("inner", "add", "add", internal_capacity)
    tanh_input = family("tanh_input", "mul", "mul", internal_capacity)
    tanh = family("tanh", "tanh", "tanh", internal_capacity)
    shifted = family("one_plus_tanh", "add", "add", internal_capacity)
    gated = family("gated", "mul", "mul", internal_capacity)
    output = family("write", "mul", "mul", output_capacity)

    builder.identity_rule(f"{name}:input-to-x2", input_family, x2)
    builder.identity_rule(f"{name}:x2-to-x3", x2, x3)
    builder.identity_rule(f"{name}:input-to-x3", input_family, x3)
    builder.identity_rule(f"{name}:x3-to-cubic", x3, cubic)
    builder.identity_rule(f"{name}:input-to-inner", input_family, inner)
    builder.identity_rule(f"{name}:cubic-to-inner", cubic, inner)
    builder.identity_rule(f"{name}:inner-to-tanh-input", inner, tanh_input)
    builder.identity_rule(f"{name}:tanh-input-to-tanh", tanh_input, tanh)
    builder.identity_rule(f"{name}:tanh-to-shifted", tanh, shifted)
    builder.identity_rule(f"{name}:input-to-gated", input_family, gated)
    builder.identity_rule(f"{name}:shifted-to-gated", shifted, gated)
    builder.identity_rule(f"{name}:gated-to-write", gated, output)
    return output


def _all_positions_vector_uses(source: Index) -> Iterable[tuple[Index, int]]:
    yield (source[:-1], source[-1])


def build_gpt2_indexed_circuit(
    prompt_tokens: int = 100,
    generated_tokens: int = 100,
    *,
    config: GPT2Config,
    profile: ServingProfile = VLLM_FP16_REFERENCE,
) -> GPT2IndexedCircuit:
    """Construct the exact indexed graph for one fixed-prompt greedy execution."""

    if prompt_tokens <= 0 or generated_tokens <= 0:
        raise ValueError("prompt and generated token counts must be positive")
    positions = prompt_tokens + generated_tokens - 1
    if positions > config.max_context:
        raise ValueError(
            "last processed position must fit the declared context window: "
            f"{positions} > {config.max_context}"
        )

    layers = config.layers
    hidden = config.hidden_size
    heads = config.heads
    head_size = config.head_size
    intermediate = config.intermediate_size
    vocabulary = config.vocabulary_size
    predictions = generated_tokens
    decode_inputs = generated_tokens - 1

    activation = GateCapacity.bits(profile.activation_boundary_bits)
    residual_capacity = GateCapacity.bits(profile.residual_bits)
    accumulator_capacity = GateCapacity.bits(profile.accumulator_bits)
    reduction_capacity = GateCapacity.bits(profile.reduction_bits)
    nonlinear_capacity = GateCapacity.bits(profile.nonlinear_internal_bits)
    probability_capacity = GateCapacity.bits(profile.probability_boundary_bits)
    kv_capacity = GateCapacity.bits(profile.kv_cache_bits)
    logit_capacity = GateCapacity.bits(profile.logit_bits)

    builder = _Builder()
    position_vector = RectangularDomain((positions, hidden))
    block_vector = RectangularDomain((layers, positions, hidden))
    block_outer = RectangularDomain((layers, positions))
    prediction_outer = RectangularDomain((predictions,))

    prompt_lookup = builder.family(
        "embedding/prompt_lookup",
        RectangularDomain((prompt_tokens, hidden)),
        ("position", "coordinate"),
        activation,
        "input",
        tags=("embedding", "fixed-prompt"),
    )
    generated_lookup = builder.family(
        "embedding/generated_lookup",
        RectangularDomain((decode_inputs, hidden)),
        ("generation", "coordinate"),
        activation,
        "embedding_lookup",
        tags=("embedding", "token-feedback"),
    )
    embedding = builder.family(
        "embedding/add",
        position_vector,
        ("position", "coordinate"),
        residual_capacity,
        "add",
        "add",
        tags=("embedding", "residual"),
    )
    builder.rule(
        "embedding:prompt-lookup-to-add",
        prompt_lookup,
        embedding,
        lambda target: (target,) if target[0] < prompt_tokens else (),
        lambda source: (source,),
    )
    builder.rule(
        "embedding:generated-lookup-to-add",
        generated_lookup,
        embedding,
        lambda target: (
            ((target[0] - prompt_tokens, target[1]),)
            if target[0] >= prompt_tokens
            else ()
        ),
        lambda source: ((prompt_tokens + source[0], source[1]),),
    )

    ln1_inputs = (
        VectorInputBinding(
            embedding,
            lambda outer, coordinate: (outer[1], coordinate) if outer[0] == 0 else None,
            lambda source: (((0, source[0]), source[1]),),
        ),
    )
    # The previous layer binding is added after residual2's family is known.

    # Reserve the block computations in dependency order. LayerNorm 1 needs a
    # later residual family as an input for layers > 0, which edge rules may
    # safely reference after every family has been registered.
    ln1 = _add_layer_norm(
        builder,
        name="blocks/ln1",
        outer_domain=block_outer,
        outer_names=("layer", "position"),
        size=hidden,
        inputs=ln1_inputs,
        input_capacity=residual_capacity,
        internal_capacity=reduction_capacity,
        output_capacity=activation,
        tags=("block", "layernorm", "ln1"),
    )

    def same_block_input(outer: Index, term: int) -> Index:
        return (*outer, term)

    same_block_uses = _all_positions_vector_uses
    q_projection = _add_biased_projection(
        builder,
        name="blocks/attention/q_projection",
        outer_domain=block_outer,
        outer_names=("layer", "position"),
        input_family=ln1.output_family,
        input_size=hidden,
        output_size=hidden,
        input_index_for=same_block_input,
        input_uses_for=same_block_uses,
        accumulator_capacity=accumulator_capacity,
        output_capacity=activation,
        tags=("block", "attention", "q-projection"),
    )
    k_projection = _add_biased_projection(
        builder,
        name="blocks/attention/k_projection",
        outer_domain=block_outer,
        outer_names=("layer", "position"),
        input_family=ln1.output_family,
        input_size=hidden,
        output_size=hidden,
        input_index_for=same_block_input,
        input_uses_for=same_block_uses,
        accumulator_capacity=accumulator_capacity,
        output_capacity=kv_capacity,
        tags=("block", "attention", "k-projection", "kv-cache"),
    )
    v_projection = _add_biased_projection(
        builder,
        name="blocks/attention/v_projection",
        outer_domain=block_outer,
        outer_names=("layer", "position"),
        input_family=ln1.output_family,
        input_size=hidden,
        output_size=hidden,
        input_index_for=same_block_input,
        input_uses_for=same_block_uses,
        accumulator_capacity=accumulator_capacity,
        output_capacity=kv_capacity,
        tags=("block", "attention", "v-projection", "kv-cache"),
    )

    causal = CausalPairsDomain(positions)
    reduction_steps = CausalReductionStepsDomain(positions)
    attention_outer = ProductDomain(
        (RectangularDomain((layers,)), causal, RectangularDomain((heads,)))
    )
    strict_attention_outer = ProductDomain(
        (
            RectangularDomain((layers,)),
            reduction_steps,
            RectangularDomain((heads,)),
        )
    )
    attention_names = ("layer", "query", "key", "head")
    score_mul = builder.family(
        "blocks/attention/score/mul",
        _with_suffix(attention_outer, head_size),
        (*attention_names, "term"),
        accumulator_capacity,
        "mul",
        "mul",
        tags=("block", "attention", "score"),
    )
    score_accumulator: str | None = None
    if head_size > 1:
        score_accumulator = builder.family(
            "blocks/attention/score/accumulator",
            _with_suffix(attention_outer, head_size - 1),
            (*attention_names, "step"),
            accumulator_capacity,
            "add",
            "add",
            tags=("block", "attention", "score"),
        )
    score_scale = builder.family(
        "blocks/attention/score/scale",
        attention_outer,
        attention_names,
        accumulator_capacity,
        "mul",
        "mul",
        tags=("block", "attention", "score"),
    )
    score_write = builder.family(
        "blocks/attention/score/write",
        attention_outer,
        attention_names,
        activation,
        "write",
        tags=(
            "block",
            "attention",
            "score",
            "inner-product-output",
            "boundary",
        ),
    )

    def q_source_for_score(target: Index) -> Iterable[Index]:
        layer, query, _key, head, term = target
        return ((layer, query, head * head_size + term),)

    def q_score_targets(source: Index) -> Iterable[Index]:
        layer, query, coordinate = source
        head, term = divmod(coordinate, head_size)
        for key in range(query + 1):
            yield (layer, query, key, head, term)

    builder.rule(
        "attention-score:q-to-mul",
        q_projection.output_family,
        score_mul,
        q_source_for_score,
        q_score_targets,
    )

    def k_source_for_score(target: Index) -> Iterable[Index]:
        layer, _query, key, head, term = target
        return ((layer, key, head * head_size + term),)

    def k_score_targets(source: Index) -> Iterable[Index]:
        layer, key, coordinate = source
        head, term = divmod(coordinate, head_size)
        for query in range(key, positions):
            yield (layer, query, key, head, term)

    builder.rule(
        "attention-score:k-to-mul",
        k_projection.output_family,
        score_mul,
        k_source_for_score,
        k_score_targets,
    )

    if score_accumulator is None:
        builder.rule(
            "attention-score:mul-to-scale",
            score_mul,
            score_scale,
            lambda target: ((*target, 0),),
            lambda source: (source[:-1],) if source[-1] == 0 else (),
        )
    else:
        builder.rule(
            "attention-score:mul-to-accumulator",
            score_mul,
            score_accumulator,
            lambda target: (
                ((*target[:-1], 0), (*target[:-1], 1))
                if target[-1] == 0
                else ((*target[:-1], target[-1] + 1),)
            ),
            lambda source: (
                ((*source[:-1], 0),)
                if source[-1] == 0
                else ((*source[:-1], source[-1] - 1),)
            ),
        )
        builder.rule(
            "attention-score:accumulator-chain",
            score_accumulator,
            score_accumulator,
            lambda target: ((*target[:-1], target[-1] - 1),) if target[-1] > 0 else (),
            lambda source: (
                ((*source[:-1], source[-1] + 1),)
                if source[-1] + 1 < head_size - 1
                else ()
            ),
        )
        builder.rule(
            "attention-score:accumulator-to-scale",
            score_accumulator,
            score_scale,
            lambda target: ((*target, head_size - 2),),
            lambda source: (source[:-1],) if source[-1] == head_size - 2 else (),
        )
    builder.identity_rule("attention-score:scale-to-write", score_scale, score_write)

    softmax_max = builder.family(
        "blocks/attention/softmax/max",
        strict_attention_outer,
        attention_names,
        reduction_capacity,
        "max",
        "max",
        tags=("block", "attention", "softmax", "shared-statistic"),
    )
    shifted = builder.family(
        "blocks/attention/softmax/shifted",
        attention_outer,
        attention_names,
        nonlinear_capacity,
        "add",
        "add",
        tags=("block", "attention", "softmax"),
    )
    exponential = builder.family(
        "blocks/attention/softmax/exp",
        attention_outer,
        attention_names,
        nonlinear_capacity,
        "exp",
        "exp",
        tags=("block", "attention", "softmax"),
    )
    denominator = builder.family(
        "blocks/attention/softmax/denominator",
        strict_attention_outer,
        attention_names,
        reduction_capacity,
        "add",
        "add",
        tags=("block", "attention", "softmax"),
    )
    reciprocal = builder.family(
        "blocks/attention/softmax/reciprocal",
        RectangularDomain((layers, positions, heads)),
        ("layer", "query", "head"),
        reduction_capacity,
        "reciprocal",
        "reciprocal",
        tags=("block", "attention", "softmax", "shared-statistic"),
    )
    probability = builder.family(
        "blocks/attention/softmax/probability",
        attention_outer,
        attention_names,
        probability_capacity,
        "mul",
        "mul",
        tags=("block", "attention", "softmax", "boundary"),
    )

    builder.rule(
        "softmax:score-to-max",
        score_write,
        softmax_max,
        lambda target: (
            ((target[0], target[1], 0, target[3]), target)
            if target[2] == 1
            else (target,)
        ),
        lambda source: (
            ((source[0], source[1], 1, source[3]),)
            if source[2] == 0 and source[1] > 0
            else ((source,) if source[2] > 0 else ())
        ),
    )
    builder.rule(
        "softmax:max-chain",
        softmax_max,
        softmax_max,
        lambda target: (
            ((target[0], target[1], target[2] - 1, target[3]),) if target[2] > 1 else ()
        ),
        lambda source: (
            ((source[0], source[1], source[2] + 1, source[3]),)
            if source[2] < source[1]
            else ()
        ),
    )
    builder.identity_rule("softmax:score-to-shifted", score_write, shifted)
    builder.rule(
        "softmax:max-to-shifted",
        softmax_max,
        shifted,
        lambda target: (
            ((target[0], target[1], target[1], target[3]),) if target[1] > 0 else ()
        ),
        lambda source: (
            ((source[0], source[1], key, source[3]) for key in range(source[1] + 1))
            if source[2] == source[1]
            else ()
        ),
    )
    builder.identity_rule("softmax:shifted-to-exp", shifted, exponential)
    builder.rule(
        "softmax:exp-to-denominator",
        exponential,
        denominator,
        lambda target: (
            ((target[0], target[1], 0, target[3]), target)
            if target[2] == 1
            else (target,)
        ),
        lambda source: (
            ((source[0], source[1], 1, source[3]),)
            if source[2] == 0 and source[1] > 0
            else ((source,) if source[2] > 0 else ())
        ),
    )
    builder.rule(
        "softmax:denominator-chain",
        denominator,
        denominator,
        lambda target: (
            ((target[0], target[1], target[2] - 1, target[3]),) if target[2] > 1 else ()
        ),
        lambda source: (
            ((source[0], source[1], source[2] + 1, source[3]),)
            if source[2] < source[1]
            else ()
        ),
    )
    builder.rule(
        "softmax:exp-to-reciprocal",
        exponential,
        reciprocal,
        lambda target: (
            ((target[0], target[1], 0, target[2]),) if target[1] == 0 else ()
        ),
        lambda source: (
            ((source[0], source[1], source[3]),)
            if source[1] == 0 and source[2] == 0
            else ()
        ),
    )
    builder.rule(
        "softmax:denominator-to-reciprocal",
        denominator,
        reciprocal,
        lambda target: (
            ((target[0], target[1], target[1], target[2]),) if target[1] > 0 else ()
        ),
        lambda source: (
            ((source[0], source[1], source[3]),) if source[2] == source[1] else ()
        ),
    )
    builder.identity_rule("softmax:exp-to-probability", exponential, probability)
    builder.rule(
        "softmax:reciprocal-to-probability",
        reciprocal,
        probability,
        lambda target: ((target[0], target[1], target[3]),),
        lambda source: (
            (source[0], source[1], key, source[2]) for key in range(source[1] + 1)
        ),
    )

    value_term_domain = ProductDomain(
        (
            RectangularDomain((layers,)),
            causal,
            RectangularDomain((heads, head_size)),
        )
    )
    strict_value_domain = ProductDomain(
        (
            RectangularDomain((layers,)),
            reduction_steps,
            RectangularDomain((heads, head_size)),
        )
    )
    value_names = ("layer", "query", "key", "head", "coordinate")
    value_mul = builder.family(
        "blocks/attention/value_reduction/mul",
        value_term_domain,
        value_names,
        accumulator_capacity,
        "mul",
        "mul",
        tags=("block", "attention", "value-reduction"),
    )
    value_accumulator = builder.family(
        "blocks/attention/value_reduction/accumulator",
        strict_value_domain,
        value_names,
        accumulator_capacity,
        "add",
        "add",
        tags=("block", "attention", "value-reduction"),
    )
    value_write = builder.family(
        "blocks/attention/value_reduction/write",
        block_vector,
        ("layer", "query", "coordinate"),
        activation,
        "write",
        tags=(
            "block",
            "attention",
            "value-reduction",
            "inner-product-output",
            "boundary",
        ),
    )
    builder.rule(
        "attention-value:probability-to-mul",
        probability,
        value_mul,
        lambda target: ((target[0], target[1], target[2], target[3]),),
        lambda source: ((*source, coordinate) for coordinate in range(head_size)),
    )
    builder.rule(
        "attention-value:v-to-mul",
        v_projection.output_family,
        value_mul,
        lambda target: (
            (
                target[0],
                target[2],
                target[3] * head_size + target[4],
            ),
        ),
        lambda source: (
            (
                source[0],
                query,
                source[1],
                source[2] // head_size,
                source[2] % head_size,
            )
            for query in range(source[1], positions)
        ),
    )
    builder.rule(
        "attention-value:mul-to-accumulator",
        value_mul,
        value_accumulator,
        lambda target: (
            (
                (target[0], target[1], 0, target[3], target[4]),
                target,
            )
            if target[2] == 1
            else (target,)
        ),
        lambda source: (
            ((source[0], source[1], 1, source[3], source[4]),)
            if source[2] == 0 and source[1] > 0
            else ((source,) if source[2] > 0 else ())
        ),
    )
    builder.rule(
        "attention-value:accumulator-chain",
        value_accumulator,
        value_accumulator,
        lambda target: (
            (
                (
                    target[0],
                    target[1],
                    target[2] - 1,
                    target[3],
                    target[4],
                ),
            )
            if target[2] > 1
            else ()
        ),
        lambda source: (
            (
                (
                    source[0],
                    source[1],
                    source[2] + 1,
                    source[3],
                    source[4],
                ),
            )
            if source[2] < source[1]
            else ()
        ),
    )
    builder.rule(
        "attention-value:mul-to-write",
        value_mul,
        value_write,
        lambda target: (
            (
                (
                    target[0],
                    target[1],
                    0,
                    target[2] // head_size,
                    target[2] % head_size,
                ),
            )
            if target[1] == 0
            else ()
        ),
        lambda source: (
            (
                (
                    source[0],
                    source[1],
                    source[3] * head_size + source[4],
                ),
            )
            if source[1] == 0 and source[2] == 0
            else ()
        ),
    )
    builder.rule(
        "attention-value:accumulator-to-write",
        value_accumulator,
        value_write,
        lambda target: (
            (
                (
                    target[0],
                    target[1],
                    target[1],
                    target[2] // head_size,
                    target[2] % head_size,
                ),
            )
            if target[1] > 0
            else ()
        ),
        lambda source: (
            (
                (
                    source[0],
                    source[1],
                    source[3] * head_size + source[4],
                ),
            )
            if source[2] == source[1]
            else ()
        ),
    )

    attention_output = _add_biased_projection(
        builder,
        name="blocks/attention/output_projection",
        outer_domain=block_outer,
        outer_names=("layer", "position"),
        input_family=value_write,
        input_size=hidden,
        output_size=hidden,
        input_index_for=same_block_input,
        input_uses_for=same_block_uses,
        accumulator_capacity=accumulator_capacity,
        output_capacity=activation,
        tags=("block", "attention", "output-projection"),
    )
    residual1 = builder.family(
        "blocks/residual1",
        block_vector,
        ("layer", "position", "coordinate"),
        residual_capacity,
        "add",
        "add",
        tags=("block", "residual", "attention-residual"),
    )
    builder.identity_rule(
        "attention-output-to-residual1",
        attention_output.output_family,
        residual1,
    )
    builder.rule(
        "embedding-to-first-residual1",
        embedding,
        residual1,
        lambda target: (target[1:],) if target[0] == 0 else (),
        lambda source: ((0, *source),),
    )

    ln2 = _add_layer_norm(
        builder,
        name="blocks/ln2",
        outer_domain=block_outer,
        outer_names=("layer", "position"),
        size=hidden,
        inputs=(
            VectorInputBinding(
                residual1,
                lambda outer, coordinate: (*outer, coordinate),
                _all_positions_vector_uses,
            ),
        ),
        input_capacity=residual_capacity,
        internal_capacity=reduction_capacity,
        output_capacity=activation,
        tags=("block", "layernorm", "ln2"),
    )
    mlp_expansion = _add_biased_projection(
        builder,
        name="blocks/mlp/expansion",
        outer_domain=block_outer,
        outer_names=("layer", "position"),
        input_family=ln2.output_family,
        input_size=hidden,
        output_size=intermediate,
        input_index_for=same_block_input,
        input_uses_for=same_block_uses,
        accumulator_capacity=accumulator_capacity,
        output_capacity=activation,
        tags=("block", "mlp", "expansion"),
    )
    gelu_output = _add_gelu(
        builder,
        name="blocks/mlp/gelu",
        domain=RectangularDomain((layers, positions, intermediate)),
        index_names=("layer", "position", "coordinate"),
        input_family=mlp_expansion.output_family,
        internal_capacity=nonlinear_capacity,
        output_capacity=activation,
    )
    mlp_contraction = _add_biased_projection(
        builder,
        name="blocks/mlp/contraction",
        outer_domain=block_outer,
        outer_names=("layer", "position"),
        input_family=gelu_output,
        input_size=intermediate,
        output_size=hidden,
        input_index_for=same_block_input,
        input_uses_for=same_block_uses,
        accumulator_capacity=accumulator_capacity,
        output_capacity=activation,
        tags=("block", "mlp", "contraction"),
    )
    residual2 = builder.family(
        "blocks/residual2",
        block_vector,
        ("layer", "position", "coordinate"),
        residual_capacity,
        "add",
        "add",
        tags=("block", "residual", "mlp-residual"),
    )
    builder.identity_rule(
        "mlp-contraction-to-residual2",
        mlp_contraction.output_family,
        residual2,
    )
    builder.identity_rule("residual1-to-residual2", residual1, residual2)

    # Complete the cross-layer residual edges omitted when LN1 was created.
    # They target LN1's mean reduction and centered gates. Reusing the exact
    # helper with a second binding would duplicate all internal families, so
    # these two relations are added directly.
    ln1_mean_sum = "blocks/ln1/mean_sum" if hidden > 1 else None
    ln1_centered = "blocks/ln1/centered"
    builder.rule(
        "blocks-ln1:previous-residual-to-centered",
        residual2,
        ln1_centered,
        lambda target: (
            ((target[0] - 1, target[1], target[2]),) if target[0] > 0 else ()
        ),
        lambda source: (
            ((source[0] + 1, source[1], source[2]),) if source[0] + 1 < layers else ()
        ),
    )
    if ln1_mean_sum is None:
        builder.rule(
            "blocks-ln1:previous-residual-to-mean",
            residual2,
            ln1.mean_family,
            lambda target: ((target[0] - 1, target[1], 0),) if target[0] > 0 else (),
            lambda source: (
                ((source[0] + 1, source[1]),)
                if source[0] + 1 < layers and source[2] == 0
                else ()
            ),
        )
    else:
        builder.rule(
            "blocks-ln1:previous-residual-to-mean-sum",
            residual2,
            ln1_mean_sum,
            lambda target: (
                (
                    (target[0] - 1, target[1], 0),
                    (target[0] - 1, target[1], 1),
                )
                if target[0] > 0 and target[2] == 0
                else (
                    ((target[0] - 1, target[1], target[2] + 1),)
                    if target[0] > 0
                    else ()
                )
            ),
            lambda source: (
                (
                    (
                        source[0] + 1,
                        source[1],
                        0 if source[2] <= 1 else source[2] - 1,
                    ),
                )
                if source[0] + 1 < layers
                else ()
            ),
        )
    builder.rule(
        "previous-residual-to-next-residual1",
        residual2,
        residual1,
        lambda target: (
            ((target[0] - 1, target[1], target[2]),) if target[0] > 0 else ()
        ),
        lambda source: (
            ((source[0] + 1, source[1], source[2]),) if source[0] + 1 < layers else ()
        ),
    )

    final_ln = _add_layer_norm(
        builder,
        name="final_ln",
        outer_domain=RectangularDomain((positions,)),
        outer_names=("position",),
        size=hidden,
        inputs=(
            VectorInputBinding(
                residual2,
                lambda outer, coordinate: ((layers - 1, outer[0], coordinate)),
                lambda source: (
                    (((source[1],), source[2]),) if source[0] == layers - 1 else ()
                ),
            ),
        ),
        input_capacity=residual_capacity,
        internal_capacity=reduction_capacity,
        output_capacity=activation,
        tags=("final-layernorm", "layernorm"),
    )

    first_prediction_position = prompt_tokens - 1

    def lm_input_index(outer: Index, term: int) -> Index:
        generation = outer[0]
        return (first_prediction_position + generation, term)

    def lm_input_uses(source: Index) -> Iterable[tuple[Index, int]]:
        position, coordinate = source
        generation = position - first_prediction_position
        if 0 <= generation < predictions:
            yield ((generation,), coordinate)

    lm_head = _add_unbiased_projection(
        builder,
        name="lm_head",
        outer_domain=prediction_outer,
        outer_names=("generation",),
        input_family=final_ln.output_family,
        input_size=hidden,
        output_size=vocabulary,
        input_index_for=lm_input_index,
        input_uses_for=lm_input_uses,
        accumulator_capacity=accumulator_capacity,
        output_capacity=logit_capacity,
        tags=("output", "lm-head"),
    )
    argmax = builder.family(
        "output/argmax",
        prediction_outer,
        ("generation",),
        GateCapacity.values(vocabulary),
        "argmax",
        "argmax",
        tags=("output", "token"),
    )
    builder.rule(
        "lm-head-to-argmax",
        lm_head.output_family,
        argmax,
        lambda target: (
            (target[0], vocabulary_index) for vocabulary_index in range(vocabulary)
        ),
        lambda source: ((source[0],),),
    )
    builder.rule(
        "argmax-to-generated-embedding",
        argmax,
        generated_lookup,
        lambda target: ((target[0],),),
        lambda source: (
            ((source[0], coordinate) for coordinate in range(hidden))
            if source[0] < decode_inputs
            else ()
        ),
    )

    outputs = tuple(GateRef(argmax, (generation,)) for generation in range(predictions))
    circuit = builder.finish(outputs)
    return GPT2IndexedCircuit(
        circuit=circuit,
        config=config,
        profile=profile,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        processed_positions=positions,
        prediction_positions=tuple(
            range(first_prediction_position, first_prediction_position + predictions)
        ),
    )
