"""Assumption-light capacity profiles for models without exact indexed circuits.

A :class:`ModelCapacityProfile` describes one fixed inference execution
(fixed prompt length, generated-token count, numerical profile) of a model at
the level of architectural region units.  It certifies much weaker structure
than the GPT-2 indexed circuit: only

* exact computed-gate counts per region (from closed-form architecture
  formulas), and
* the **self-cut** capacity upper bound per region: the corrupted gates
  themselves always form a valid downstream vertex cut, so the sum of their
  own declared value widths, capped by the designated-output frontier, upper
  bounds any attack's structural output capacity with no wiring analysis.

That is sufficient to run the certified verification-sampling study.  The
bounds are looser than canonical-cut bounds, but loose only in the sound
direction for the verifier.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from circuit_cut_analysis.sampling_study import RegionUnit


@dataclass(frozen=True, slots=True)
class CapacityRegion:
    """One disjoint architectural region of computed scalar gates.

    ``gate_count`` is exact for the declared execution semantics.
    ``self_cut_bits_per_gate`` is the declared value width of each gate in
    the region under the chosen numerical profile; ``gate_count x width``
    is the region's certified self-cut capacity.  ``description`` must state
    what operations the region contains so the accounting is auditable.
    ``value_cardinality_upper_bound`` optionally preserves an exact
    non-power-of-two alphabet bound instead of recovering it from a float.
    """

    id: str
    description: str
    gate_count: int
    self_cut_bits_per_gate: float
    value_cardinality_upper_bound: int | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("region id must be non-empty")
        if not isinstance(self.gate_count, int) or isinstance(
            self.gate_count,
            bool,
        ):
            raise ValueError(f"{self.id}: gate count must be an integer")
        if self.gate_count < 0:
            raise ValueError(f"{self.id}: gate count cannot be negative")
        if not math.isfinite(self.self_cut_bits_per_gate):
            raise ValueError(f"{self.id}: gate width must be finite")
        cardinality = self.value_cardinality_upper_bound
        if cardinality is not None:
            if not isinstance(cardinality, int) or isinstance(cardinality, bool):
                raise ValueError(f"{self.id}: value cardinality must be an integer")
            if cardinality < 1:
                raise ValueError(f"{self.id}: value cardinality must be positive")
            if not math.isclose(
                self.self_cut_bits_per_gate,
                math.log2(cardinality),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"{self.id}: gate width must equal log2(value cardinality)"
                )
        if (
            self.self_cut_bits_per_gate <= 0
            and self.gate_count > 0
            and cardinality != 1
        ):
            raise ValueError(f"{self.id}: gate width must be positive")

    @property
    def self_cut_bits(self) -> float:
        return self.gate_count * self.self_cut_bits_per_gate


@dataclass(frozen=True, slots=True)
class ModelCapacityProfile:
    """One fixed execution's exact gate accounting and certified widths."""

    model_id: str
    prompt_tokens: int
    generated_tokens: int
    logical_vocabulary_size: int
    numerical_profile_id: str
    regions: tuple[CapacityRegion, ...]
    assumptions: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.prompt_tokens <= 0 or self.generated_tokens <= 0:
            raise ValueError("prompt and generated token counts must be positive")
        if self.logical_vocabulary_size < 2:
            raise ValueError("vocabulary must contain at least two tokens")
        seen: set[str] = set()
        for region in self.regions:
            if region.id in seen:
                raise ValueError(f"duplicate region id {region.id!r}")
            seen.add(region.id)

    @property
    def token_bits(self) -> float:
        return math.log2(self.logical_vocabulary_size)

    @property
    def output_frontier_bits(self) -> float:
        """Capacity of the always-valid designated-output cut."""

        return self.generated_tokens * self.token_bits

    @property
    def total_gate_count(self) -> int:
        return sum(region.gate_count for region in self.regions)

    def region_units(self) -> tuple[RegionUnit, ...]:
        """Convert regions into sampling-study units with self-cut bounds."""

        return tuple(
            RegionUnit(
                id=region.id,
                row_ids=(region.id,),
                checked_gate_count=region.gate_count,
                capacity_upper_bits=region.self_cut_bits,
                max_single_cut_bits=region.self_cut_bits_per_gate,
            )
            for region in self.regions
            if region.gate_count > 0
        )


def tiled_region_units(
    profile: ModelCapacityProfile,
    *,
    target_bits: float,
    max_units: int = 4096,
) -> tuple[RegionUnit, ...]:
    """Split regions into near-equal tiles with exactly preserved bounds.

    Self-cut capacity is linear in the gates a unit contains (every gate in a
    region has the same declared width), so partitioning a region into
    arbitrary gate subsets keeps each part's certified bound exact.  That
    linearity is what makes tiling sound here; it does **not** hold for
    canonical-cut units, whose capacity depends on shared boundary structure.

    Each region is split into ``ceil(capacity / effective_target)`` tiles of
    near-equal gate count, where ``effective_target`` is raised just enough to
    keep the total unit count near ``max_units``.
    """

    if target_bits <= 0:
        raise ValueError("target tile size must be positive")
    if max_units < len(profile.regions):
        raise ValueError("max_units cannot be below the region count")
    total_capacity = sum(region.self_cut_bits for region in profile.regions)
    effective_target = max(target_bits, total_capacity / max_units)

    units: list[RegionUnit] = []
    for region in profile.regions:
        if region.gate_count == 0:
            continue
        tile_count = min(
            region.gate_count,
            max(1, math.ceil(region.self_cut_bits / effective_target)),
        )
        base, extra = divmod(region.gate_count, tile_count)
        for index in range(tile_count):
            gates = base + (1 if index < extra else 0)
            units.append(
                RegionUnit(
                    id=f"{region.id}/tile-{index}",
                    row_ids=(region.id,),
                    checked_gate_count=gates,
                    capacity_upper_bits=gates * region.self_cut_bits_per_gate,
                    max_single_cut_bits=region.self_cut_bits_per_gate,
                )
            )
    return tuple(units)
