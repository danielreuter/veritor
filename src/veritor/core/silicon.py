"""Silicon-level semantics of tensor-core multiply-accumulate steps.

A tensor-core ``mma`` instruction computes, for each output element,

    D = tc_dot(C, A[0..K-1], B[0..K-1])

where the K products are formed exactly, then summed together with the
incoming FP32 accumulator in fixed-size *groups*: all terms of a group are
aligned to the group's maximum exponent by truncating right shifts, added
exactly in a wide integer adder, and the sum is normalised to a fixed
internal significand width by truncation (round toward zero).  The result of
one group is the accumulator of the next.  No IEEE rounding happens anywhere;
this is what NVIDIA's silicon does and it is *not* what a sequence of FP32
FMAs would do.  The model is the one recovered by Hawkeye (Badash, Boneh,
Komargodski, Srivastava, "Hawkeye: Reproducing GPU-Level Non-Determinism",
MLSys 2026, arXiv 2603.20421) and this module is a bit-exact pure-Python port
of the reference simulator at https://github.com/badasherez/gpu-simulator
(``src/Ampere_simulator.cpp``, ``src/Hopper_simulator.cpp``,
``src/Hopper_fp8_simulator.cpp``, ``utils/utils.cpp``, commit
``30703fcb309c943a6df5eee0277cb81815deb8f4``), generalised so that the group
structure, internal width and exponent floor are parameters.

Everything here is integer arithmetic on raw encodings: BF16 and E4M3 operands
are 16- and 8-bit words, the accumulator and the result are 32-bit words
holding IEEE-754 binary32 bit patterns.  The :class:`Pipeline` instances at
the bottom of the module are the ones validated against hardware; see
``docs/hardware-semantics.md`` for the measurements.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .errors import InvalidArtifact
from .gates import INPUT_SOURCE, WEIGHT_SOURCE, Gate, GateSet

__all__ = [
    "ADA_BF16_M16N8K16",
    "ADA_E4M3_M16N8K32",
    "AMPERE_BF16_M16N8K16",
    "HAWKEYE_AMPERE_GROUPSUM_E4M3_V0",
    "HOPPER_BF16_M16N8K16",
    "HOPPER_E4M3_K32",
    "PIPELINES",
    "Pipeline",
    "Term",
    "bf16_product",
    "e4m3_product",
    "fp32_term",
    "group_sum",
    "make_tensor_core_gate_set",
    "pack_fp32",
    "tc_dot",
    "tc_dot_chain",
]

FP32_SIGNIFICAND_WIDTH = 24
FP32_MIN_NORMAL_EXPONENT = -126
FP32_MAX_EXPONENT = 127

# Products are carried with this many significand bits before alignment, so
# that a 25-bit-wide adder (Ampere) sees them unshifted.  BF16: 8x8 -> 16 bits,
# shifted by 9.  E4M3: 4x4 -> 8 bits, shifted by 17.  This matches Hawkeye.
PRODUCT_WIDTH = 25


@dataclass(frozen=True, slots=True)
class Term:
    """An unnormalised signed binary term ``(-1)^negative * magnitude * 2^(exponent-23)``.

    ``magnitude == 0`` is the (unsigned) zero; its ``exponent`` is ignored.
    The exponent is that of bit 23, so a normal FP32 value with biased
    exponent ``E`` has ``exponent = E - 127`` and a 24-bit magnitude.
    """

    negative: bool
    exponent: int
    magnitude: int


ZERO = Term(False, 0, 0)


def bf16_product(a: int, b: int) -> Term:
    """Exact product of two BF16 words as a 25-bit-scaled :class:`Term`.

    Subnormal inputs keep their unnormalised significand at exponent ``-126``
    (``utils.cpp::multiply_bfloat16_to_gfloat``).  Non-finite operands are
    outside the modelled domain and raise.
    """

    ea, eb = (a >> 7) & 0xFF, (b >> 7) & 0xFF
    if ea == 0xFF or eb == 0xFF:
        raise InvalidArtifact("non-finite BF16 operand")
    sa = (a & 0x7F) | (0x80 if ea else 0)
    sb = (b & 0x7F) | (0x80 if eb else 0)
    magnitude = (sa * sb) << 9
    if magnitude == 0:
        return ZERO
    return Term(((a ^ b) >> 15) & 1 == 1, (ea or 1) + (eb or 1) - 254, magnitude)


def e4m3_product(a: int, b: int) -> Term:
    """Exact product of two OCP E4M3 bytes as a 25-bit-scaled :class:`Term`.

    E4M3 has no infinities; ``0x7f``/``0xff`` are NaN and raise.  Subnormals
    keep their unnormalised significand at exponent ``-6``
    (``utils.cpp::multiply_fp8e4m3_to_gfloat``).
    """

    if (a & 0x7F) == 0x7F or (b & 0x7F) == 0x7F:
        raise InvalidArtifact("E4M3 NaN operand")
    ea, eb = (a >> 3) & 0xF, (b >> 3) & 0xF
    sa = (a & 0x7) | (0x8 if ea else 0)
    sb = (b & 0x7) | (0x8 if eb else 0)
    magnitude = (sa * sb) << 17
    if magnitude == 0:
        return ZERO
    return Term(((a ^ b) >> 7) & 1 == 1, (ea or 1) + (eb or 1) - 14, magnitude)


def fp32_term(bits: int) -> Term:
    """Decode FP32 bits into a 24-bit :class:`Term`; infinities and NaN raise."""

    exponent_field = (bits >> 23) & 0xFF
    if exponent_field == 0xFF:
        raise InvalidArtifact("non-finite FP32 accumulator")
    mantissa = bits & 0x7F_FFFF
    negative = bits >> 31 == 1
    if exponent_field == 0:
        if mantissa == 0:
            return ZERO
        return Term(negative, FP32_MIN_NORMAL_EXPONENT, mantissa)
    return Term(negative, exponent_field - 127, (1 << 23) | mantissa)


def pack_fp32(term: Term) -> int:
    """Encode a 24-bit :class:`Term` as FP32 bits (zero is ``+0``; overflow is ±inf).

    Mirrors ``Gfloat::operator float``: a magnitude without bit 23 set at
    exponent ``-126`` is a subnormal.
    """

    if term.magnitude == 0:
        return 0
    if term.exponent > FP32_MAX_EXPONENT:
        return (int(term.negative) << 31) | 0x7F80_0000
    exponent_field = term.exponent + 127
    if term.magnitude & (1 << 23) == 0:
        exponent_field -= 1
    return (
        (int(term.negative) << 31)
        | (exponent_field << 23)
        | (term.magnitude & 0x7F_FFFF)
    )


def group_sum(terms: Sequence[Term], *, width: int, zero_exponent: int) -> Term:
    """One grouped, truncating accumulation step (``*_simulator.cpp::group_sum``).

    ``width`` is the internal significand width of the adder (25 on Ampere,
    26 on Hopper, 14 on Hopper FP8) and ``zero_exponent`` the exponent floor
    (the maximum exponent used when every term is zero).  Every term is
    rescaled from the 24-bit FP32 grid to ``width`` bits, aligned to the
    largest exponent by a truncating right shift, summed exactly, and the
    result is normalised to ``width`` bits by truncation, denormalised if it
    is below the FP32 normal range, and truncated back to 24 bits.
    """

    rescale = width - FP32_SIGNIFICAND_WIDTH
    max_exponent = zero_exponent
    for term in terms:
        if term.magnitude and term.exponent > max_exponent:
            max_exponent = term.exponent
    total = 0
    for term in terms:
        if not term.magnitude:
            continue
        scaled = (
            term.magnitude << rescale if rescale >= 0 else term.magnitude >> -rescale
        )
        aligned = scaled >> (max_exponent - term.exponent)
        total += -aligned if term.negative else aligned
    if total == 0:
        return ZERO
    negative = total < 0
    magnitude = -total if negative else total
    bit_length = magnitude.bit_length()
    exponent = max_exponent + bit_length - width
    if bit_length > width:
        magnitude >>= bit_length - width
    else:
        magnitude <<= width - bit_length
    if exponent < FP32_MIN_NORMAL_EXPONENT:
        magnitude >>= FP32_MIN_NORMAL_EXPONENT - exponent
        exponent = FP32_MIN_NORMAL_EXPONENT
    magnitude = magnitude >> rescale if rescale >= 0 else magnitude << -rescale
    if magnitude == 0:
        return ZERO
    return Term(negative, exponent, magnitude)


@dataclass(frozen=True, slots=True)
class Pipeline:
    """The silicon parameters of one ``mma`` shape on one architecture.

    ``groups`` lists, in k order, how many products each grouped sum takes;
    the incoming accumulator joins the first group and each group's result
    joins the next.  ``operand_bits`` is 16 for BF16 and 8 for E4M3.
    """

    name: str
    arch: str
    dtype: str
    operand_bits: int
    groups: tuple[int, ...]
    width: int
    zero_exponent: int
    validated: str

    @property
    def k(self) -> int:
        return sum(self.groups)

    def product(self, a: int, b: int) -> Term:
        return bf16_product(a, b) if self.dtype == "bf16" else e4m3_product(a, b)


def tc_dot(
    pipeline: Pipeline, acc_bits: int, a: Sequence[int], b: Sequence[int]
) -> int:
    """``D = acc + sum_k a[k] * b[k]`` exactly as ``pipeline``'s silicon computes it.

    ``acc_bits`` is an FP32 word, ``a`` and ``b`` are ``pipeline.k`` operand
    words each; the result is an FP32 word.
    """

    if len(a) != pipeline.k or len(b) != pipeline.k:
        raise InvalidArtifact(f"{pipeline.name} takes {pipeline.k} operand pairs")
    acc = fp32_term(acc_bits)
    start = 0
    for size in pipeline.groups:
        products = [pipeline.product(a[i], b[i]) for i in range(start, start + size)]
        acc = group_sum(
            [acc, *products], width=pipeline.width, zero_exponent=pipeline.zero_exponent
        )
        if acc.magnitude and acc.exponent > FP32_MAX_EXPONENT:
            # A group whose result leaves the FP32 range saturates to an
            # infinity of its own sign, and that infinity is sticky through
            # the remaining groups (measured on Ada; not part of Hawkeye's
            # model, whose simulator never reaches this range).
            return pack_fp32(acc)
        start += size
    return pack_fp32(acc)


def tc_dot_chain(
    pipeline: Pipeline, acc_bits: int, a: Sequence[int], b: Sequence[int]
) -> int:
    """Fold :func:`tc_dot` over ``a``/``b`` in k-chunks of ``pipeline.k`` (a GEMM output element)."""

    k = pipeline.k
    if len(a) != len(b) or len(a) % k:
        raise InvalidArtifact(f"chain length must be a multiple of {k}")
    for start in range(0, len(a), k):
        acc_bits = tc_dot(
            pipeline, acc_bits, a[start : start + k], b[start : start + k]
        )
    return acc_bits


# --- Validated pipelines ----------------------------------------------------

AMPERE_BF16_M16N8K16 = Pipeline(
    name="ampere_bf16_m16n8k16",
    arch="sm_80",
    dtype="bf16",
    operand_bits=16,
    groups=(8, 8),
    width=25,
    zero_exponent=-132,
    validated="Hawkeye (A100), not re-measured here",
)

HOPPER_BF16_M16N8K16 = Pipeline(
    name="hopper_bf16_m16n8k16",
    arch="sm_90",
    dtype="bf16",
    operand_bits=16,
    groups=(16,),
    width=26,
    zero_exponent=-133,
    validated="Hawkeye (H100), not re-measured here",
)

HOPPER_E4M3_K32 = Pipeline(
    name="hopper_e4m3_wgmma_k32",
    arch="sm_90",
    dtype="e4m3",
    operand_bits=8,
    groups=(32,),
    width=14,
    zero_exponent=-139,
    validated="Hawkeye (H100, zero accumulator only), not re-measured here",
)

# The synthetic contract of the openvm-tc-bench spike: Ampere's BF16 GroupSum
# applied to E4M3 products, two groups of eight.  Kept so that the golden
# vectors of that spike remain checkable; it is not a hardware pipeline.
HAWKEYE_AMPERE_GROUPSUM_E4M3_V0 = Pipeline(
    name="hawkeye_ampere_groupsum_fp8e4m3_v0",
    arch="synthetic",
    dtype="e4m3",
    operand_bits=8,
    groups=(8, 8),
    width=25,
    zero_exponent=-132,
    validated="synthetic; see docs/hardware-semantics.md for the Ada comparison",
)

# Ada Lovelace (RTX 4090, sm_89) as measured in this repository with one
# ``mma.sync`` per record; parameters recovered by
# gpu/tensor-core-semantics/characterize.py and validated by validate_tiles.py
# (results under gpu/tensor-core-semantics/results/).
ADA_BF16_M16N8K16 = Pipeline(
    name="ada_bf16_m16n8k16",
    arch="sm_89",
    dtype="bf16",
    operand_bits=16,
    groups=(8, 8),
    width=25,
    zero_exponent=-132,
    validated=(
        "RTX 4090 (sm_89, driver 580.159.03) mma.sync.m16n8k16.bf16: "
        "24,898,304/24,898,304 elements bit-exact over 194,518 tiles "
        "(random, subnormal, cancellation, mixed-magnitude, near-overflow, "
        "nonzero accumulators); identical parameters to Hawkeye's Ampere model"
    ),
)

ADA_E4M3_M16N8K32 = Pipeline(
    name="ada_e4m3_m16n8k32",
    arch="sm_89",
    dtype="e4m3",
    operand_bits=8,
    groups=(16, 16),
    width=14,
    zero_exponent=-139,
    validated=(
        "RTX 4090 (sm_89, driver 580.159.03) mma.sync.m16n8k32.e4m3: "
        "24,898,304/24,898,304 elements bit-exact over 194,518 tiles; "
        "Hopper's 14-bit FP8 adder but two groups of 16 instead of one of 32"
    ),
)

PIPELINES: dict[str, Pipeline] = {
    pipeline.name: pipeline
    for pipeline in (
        AMPERE_BF16_M16N8K16,
        HOPPER_BF16_M16N8K16,
        HOPPER_E4M3_K32,
        HAWKEYE_AMPERE_GROUPSUM_E4M3_V0,
        ADA_BF16_M16N8K16,
        ADA_E4M3_M16N8K32,
    )
}


def make_tensor_core_gate_set(arch: str, dtype: str) -> GateSet:
    """The gate set of one tensor-core pipeline: ``tc_dot`` plus the two sources.

    The gate has arity ``1 + 2k``: the FP32 accumulator word followed by the
    ``k`` A words and the ``k`` B words.  ``Gate`` currently validates every
    argument against the gate's single ``width`` (32 bits here), so operand
    words are additionally range-checked by the evaluator; a per-argument
    width on ``Gate`` would make that check structural.
    """

    matches = [p for p in PIPELINES.values() if p.arch == arch and p.dtype == dtype]
    if len(matches) != 1:
        raise InvalidArtifact(f"no unique pipeline for arch={arch!r} dtype={dtype!r}")
    pipeline = matches[0]
    k = pipeline.k
    operand_limit = 1 << pipeline.operand_bits

    def split(args: tuple[int, ...]) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
        acc, a, b = args[0], args[1 : 1 + k], args[1 + k :]
        for word in (*a, *b):
            if word >= operand_limit:
                raise InvalidArtifact(
                    f"operand word exceeds {pipeline.operand_bits} bits"
                )
        return acc, a, b

    def evaluate(args: tuple[int, ...]) -> int:
        return tc_dot(pipeline, *split(args))

    def check(args: tuple[int, ...], out: int) -> bool:
        try:
            return evaluate(args) == out
        except InvalidArtifact:
            return False

    return GateSet(
        (
            Gate(
                f"tc_dot{k}",
                1 + 2 * k,
                32,
                replay_cost=k,
                proof_cost=k,
                evaluate=evaluate,
                check=check,
            ),
            Gate("in", 0, 32, replay_cost=0, proof_cost=1, source=INPUT_SOURCE),
            Gate("weight", 0, 32, replay_cost=0, proof_cost=1, source=WEIGHT_SOURCE),
        ),
        name=f"veritor.tensor-core.{pipeline.name}",
        version="1",
    )
