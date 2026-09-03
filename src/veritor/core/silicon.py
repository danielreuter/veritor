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

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

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
    "F32Array",
    "Pipeline",
    "Term",
    "bf16_product",
    "bf16_to_f32",
    "e4m3_product",
    "f32_exp",
    "f32_max",
    "f32_tanh",
    "f32_to_bf16",
    "fp32_term",
    "gelu_tanh",
    "group_sum",
    "is_nan_word",
    "ln_rstd",
    "make_pinned_gate_set",
    "make_tensor_core_gate_set",
    "pack_fp32",
    "pipeline_for",
    "tc_dot",
    "tc_dot_chain",
    "tensor_core_evaluators",
    "tensor_core_gates",
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


def pipeline_for(arch: str, dtype: str) -> Pipeline:
    """The unique validated pipeline of ``arch``/``dtype``."""

    matches = [p for p in PIPELINES.values() if p.arch == arch and p.dtype == dtype]
    if len(matches) != 1:
        raise InvalidArtifact(f"no unique pipeline for arch={arch!r} dtype={dtype!r}")
    return matches[0]


def tensor_core_evaluators(
    pipeline: Pipeline,
) -> tuple[Callable[[tuple[int, ...]], int], Callable[[tuple[int, ...]], int]]:
    """The evaluators of ``tc_dot{k}`` (accumulator, A words, B words) and ``tc_dot{k}_0`` (A words, B words)."""

    k = pipeline.k

    def evaluate(args: tuple[int, ...]) -> int:
        return tc_dot(pipeline, args[0], args[1 : 1 + k], args[1 + k :])

    def evaluate_zero(args: tuple[int, ...]) -> int:
        return tc_dot(pipeline, 0, args[:k], args[k:])

    return evaluate, evaluate_zero


def tensor_core_gates(pipeline: Pipeline) -> tuple[Gate, Gate]:
    """``tc_dot{k}`` and ``tc_dot{k}_0``: one ``mma`` step with, and without, an incoming accumulator.

    ``tc_dot{k}`` has arity ``1 + 2k``: the FP32 accumulator word, then the
    ``k`` A words and the ``k`` B words (``operand_bits`` wide each, declared
    through ``arg_widths``).  ``tc_dot{k}_0`` is the same step with the
    accumulator pinned to ``+0``: the first step of a reduction without a
    bias (attention scores, the value mix, a tied LM head).
    """

    k = pipeline.k
    ob = pipeline.operand_bits
    evaluate, evaluate_zero = tensor_core_evaluators(pipeline)
    return (
        Gate(
            f"tc_dot{k}",
            1 + 2 * k,
            32,
            replay_cost=k,
            proof_cost=k,
            evaluate=evaluate,
            arg_widths=(32, *([ob] * (2 * k))),
        ),
        Gate(
            f"tc_dot{k}_0",
            2 * k,
            32,
            replay_cost=k,
            proof_cost=k,
            evaluate=evaluate_zero,
            arg_widths=(ob,) * (2 * k),
        ),
    )


def make_tensor_core_gate_set(arch: str, dtype: str) -> GateSet:
    """The gate set of one tensor-core pipeline: the two ``tc_dot`` steps plus the two sources.

    See :func:`tensor_core_gates`; the operand words are ``operand_bits``
    wide and validated structurally through ``Gate.arg_widths``.
    """

    pipeline = pipeline_for(arch, dtype)
    return GateSet(
        (
            *tensor_core_gates(pipeline),
            Gate("in", 0, 32, replay_cost=0, proof_cost=1, source=INPUT_SOURCE),
            Gate("weight", 0, 32, replay_cost=0, proof_cost=1, source=WEIGHT_SOURCE),
        ),
        name=f"veritor.tensor-core.{pipeline.name}",
        version="2",
    )


# --- CUDA-core fp32 semantics ----------------------------------------------------
#
# The elementwise operations of a transformer run on the CUDA cores, and their
# result is fixed by IEEE-754 binary32 once fused multiply-add contraction and
# fast-math approximations are off (``nvcc -fmad=false -prec-div=true
# -prec-sqrt=true -ftz=false``): ``+ - * /`` and ``sqrt`` are then correctly
# rounded to nearest-even on the GPU exactly as on any IEEE CPU.  Every gate
# below is an explicit sequence of such operations (plus exact bit
# manipulation), written once over numpy ``float32`` values -- which are
# IEEE-exact for these operations -- so that the same code evaluates one gate
# (a 0-d array) and a whole tensor of them.  The transcendental functions are
# *not* library calls: ``f32_exp`` is a Cody-Waite range reduction with a
# degree-5 polynomial (Cephes ``expf`` coefficients) evaluated by explicit
# multiplies and adds, ``f32_tanh`` is ``1 - 2 / (exp(2|x|) + 1)`` with the
# sign restored, and the reciprocal standard deviation is ``1 / sqrt(v + eps)``
# with the two IEEE operations.  The GPU kernels in ``gpu/gpt2/`` are the
# same sequences written in CUDA C; ``tests/veritor/core/golden/`` holds their
# outputs on the RTX 4090.

F32Array = np.ndarray[tuple[int, ...], np.dtype[np.float32]]

_F32 = np.float32


def _f32_from_bits(bits: int) -> np.float32:
    return np.uint32(bits).view(np.float32)


F32_ZERO = _F32(0.0)
F32_ONE = _F32(1.0)
F32_TWO = _F32(2.0)
F32_HALF = _F32(0.5)
F32_INF = _F32(np.inf)
F32_NAN_BITS = 0x7FC00000
F32_LOG2E = _f32_from_bits(0x3FB8AA3B)  # 1.44269502
F32_RINT_MAGIC = _f32_from_bits(
    0x4B400000
)  # 1.5 * 2**23: (t + M) - M rounds t to an integer
F32_LN2_HI = _f32_from_bits(
    0x3F317200
)  # 0.693145751953125 (12 significant bits: k * LN2_HI is exact)
F32_LN2_LO = _f32_from_bits(0x35BFBE8E)  # 1.42860677e-06
F32_EXP_POLY = tuple(
    _f32_from_bits(bits)
    for bits in (0x39506967, 0x3AB743CE, 0x3C088908, 0x3D2AA9C1, 0x3E2AAAAA, 0x3F000000)
)  # Cephes expf: 1.9875691500E-4 .. 5.0000001201E-1, highest degree first
F32_EXP_LO = _F32(-86.5)  # below: exp is 0 (the result would be subnormal)
F32_EXP_HI = _F32(88.0)  # above: exp is +inf
F32_TANH_SAT = _F32(9.0)  # |x| >= 9: tanh is +-1
F32_GELU_C0 = _f32_from_bits(0x3F4C422A)  # sqrt(2 / pi) = 0.79788456
F32_GELU_C1 = _f32_from_bits(0x3D372713)  # 0.044715
F32_LN_EPS = _f32_from_bits(0x3727C5AC)  # 1e-5
BF16_ONE = 0x3F80


def f32_exp(x: F32Array) -> F32Array:
    """``exp(x)`` as the pinned fp32 sequence (see the module comment).

    ``t = x * log2e``; ``k = rint(t)`` by the magic-number add/subtract;
    ``r = x - k * ln2_hi - k * ln2_lo`` (four operations, the first product
    exact); ``p`` is the degree-5 Horner polynomial in ``r`` (five ``mul``,
    five ``add``); ``y = ((p * r) * r + r) + 1``; the result is ``y * 2**k``
    with ``2**k`` built from its bit pattern.  ``x < -86.5`` gives ``+0``,
    ``x > 88`` gives ``+inf`` and NaN propagates.
    """

    with np.errstate(all="ignore"):
        t = x * F32_LOG2E
        kf = (t + F32_RINT_MAGIC) - F32_RINT_MAGIC
        r = x - kf * F32_LN2_HI
        r = r - kf * F32_LN2_LO
        p = np.full_like(r, F32_EXP_POLY[0])
        for coefficient in F32_EXP_POLY[1:]:
            p = p * r + coefficient
        y = p * r
        y = y * r
        y = y + r
        y = y + F32_ONE
        ki = np.clip(kf, -126, 127).astype(np.int32)
        scale = ((ki + 127) << 23).astype(np.uint32).view(np.float32)
        y = y * scale
        y = np.where(x < F32_EXP_LO, F32_ZERO, y)
        y = np.where(x > F32_EXP_HI, F32_INF, y)
        y = np.where(np.isnan(x), x, y)
        return y.astype(np.float32, copy=False)


def f32_tanh(x: F32Array) -> F32Array:
    """``tanh(x) = sign(x) * (1 - 2 / (exp(2|x|) + 1))`` over :func:`f32_exp`; ``|x| >= 9`` saturates to ``+-1``."""

    with np.errstate(all="ignore"):
        a = np.abs(x)
        e = f32_exp(a + a)
        r = F32_ONE - F32_TWO / (e + F32_ONE)
        r = np.where(a >= F32_TANH_SAT, F32_ONE, r)
        r = np.copysign(r, x)
        r = np.where(np.isnan(x), x, r)
        return r.astype(np.float32, copy=False)


def gelu_tanh(x: F32Array) -> F32Array:
    """``0.5 x (1 + tanh(c0 (x + c1 x^3)))``: ``x3 = (x x) x``, ``inner = x + c1 x3``, ``z = c0 inner``, ``(0.5 x) (1 + tanh z)``."""

    with np.errstate(all="ignore"):
        x2 = x * x
        x3 = x2 * x
        inner = x + F32_GELU_C1 * x3
        t = f32_tanh(F32_GELU_C0 * inner)
        y = (F32_HALF * x) * (F32_ONE + t)
        return y.astype(np.float32, copy=False)


def ln_rstd(variance: F32Array) -> F32Array:
    """``1 / sqrt(v + 1e-5)``: two correctly rounded IEEE operations after the add."""

    with np.errstate(all="ignore"):
        r = F32_ONE / np.sqrt(variance + F32_LN_EPS)
        return r.astype(np.float32, copy=False)


def f32_max(a: F32Array, b: F32Array) -> F32Array:
    """``b if b > a else a``: ties (and NaN comparisons) keep ``a``."""

    return np.where(b > a, b, a).astype(np.float32, copy=False)


def f32_to_bf16(
    bits: np.ndarray[tuple[int, ...], np.dtype[np.uint32]],
) -> np.ndarray[tuple[int, ...], np.dtype[np.uint16]]:
    """Round fp32 words to BF16 words to nearest, ties to even; NaN becomes ``0x7FC0``."""

    bits = bits.astype(np.uint32, copy=False)
    with np.errstate(over="ignore"):
        rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))) >> 16
    exponent_all_ones = (bits & np.uint32(0x7F800000)) == np.uint32(0x7F800000)
    is_nan = exponent_all_ones & ((bits & np.uint32(0x007FFFFF)) != 0)
    return np.where(is_nan, np.uint32(0x7FC0), rounded).astype(np.uint16)


def bf16_to_f32(
    words: np.ndarray[tuple[int, ...], np.dtype[np.uint16]],
) -> np.ndarray[tuple[int, ...], np.dtype[np.uint32]]:
    """Widen BF16 words to fp32 words: the 16 low bits are zero."""

    return (words.astype(np.uint32) << np.uint32(16)).astype(np.uint32)


def is_nan_word(width: int, value: int) -> bool:
    """Whether a 32-bit (binary32) or 16-bit (BF16) word is a NaN pattern."""

    if width == 32:
        return (value & 0x7F800000) == 0x7F800000 and (value & 0x007FFFFF) != 0
    return (value & 0x7F80) == 0x7F80 and (value & 0x007F) != 0


def _bits(value: int) -> F32Array:
    return np.array(value, dtype=np.uint32).view(np.float32)


def _word(value: F32Array) -> int:
    result = int(np.asarray(value, dtype=np.float32).view(np.uint32))
    return F32_NAN_BITS if is_nan_word(32, result) else result


def _unary(fn: Callable[[F32Array], F32Array]) -> Callable[[tuple[int, ...]], int]:
    return lambda args: _word(fn(_bits(args[0])))


def _binary(
    fn: Callable[[F32Array, F32Array], F32Array],
) -> Callable[[tuple[int, ...]], int]:
    return lambda args: _word(fn(_bits(args[0]), _bits(args[1])))


def _ieee(
    op: Callable[[F32Array, F32Array], F32Array],
) -> Callable[[F32Array, F32Array], F32Array]:
    def apply(a: F32Array, b: F32Array) -> F32Array:
        with np.errstate(all="ignore"):
            return op(a, b).astype(np.float32, copy=False)

    return apply


def _pinned(
    name: str,
    width: int,
    cost: int,
    evaluate: Callable[[tuple[int, ...]], int],
    arg_widths: tuple[int, ...],
    *,
    floats: tuple[bool, ...] | None = None,
    float_out: bool = True,
) -> Gate:
    """A gate whose ``check`` is the relation and the absence of NaN among its floating-point words.

    ``floats`` says which arguments are floating-point words (all by default);
    ``float_out`` whether the output is one.  Token words are never NaN-checked.
    """

    is_float = (True,) * len(arg_widths) if floats is None else floats

    def check(args: tuple[int, ...], out: int) -> bool:
        for flag, arg_width, value in zip(is_float, arg_widths, args, strict=True):
            if flag and is_nan_word(arg_width, value):
                return False
        if float_out and is_nan_word(width, out):
            return False
        return evaluate(args) == out

    return Gate(
        name,
        len(arg_widths),
        width,
        replay_cost=cost,
        proof_cost=cost,
        evaluate=evaluate,
        check=check,
        arg_widths=arg_widths,
    )


def make_pinned_gate_set(arch: str = "sm_89", dtype: str = "bf16") -> GateSet:
    """The pinned gate set of a transformer forward pass on ``arch``: tensor-core steps and CUDA-core fp32 ops.

    Values are words: 32-bit words hold IEEE binary32 bit patterns, 16-bit
    words hold BF16 bit patterns or token ids (which of the two a word is
    follows from the gate that consumes it).  The gates:

    * ``tc_dot16``, ``tc_dot16_0``: one tensor-core step (:func:`tensor_core_gates`);
    * ``bf16_to_f32`` (16 -> 32) and ``f32_to_bf16`` (32 -> 16, round to
      nearest even, NaN to ``0x7FC0``);
    * ``f32_add``, ``f32_sub``, ``f32_mul``, ``f32_div``: correctly rounded
      IEEE binary32 operations; ``f32_max``: ``b if b > a else a``;
    * ``f32_exp``, ``f32_tanh``, ``gelu_tanh``, ``ln_rstd``: the fp32
      sequences of the functions of the same names in this module;
    * ``argmax_select(la, lb, ia, ib)`` (32, 32, 16, 16 -> 16): ``ib if lb >
      la else ia`` (with ``f32_max`` on the logits this is one tournament node
      whose ties keep the earlier index);
    * ``token_eq(t, j)`` (16, 16 -> 16): BF16 ``1.0`` (``0x3F80``) if ``t ==
      j`` else ``0``, the one-hot of a token against the token table;
    * the sources ``in`` and ``weight``, 16-bit words (token ids; BF16
      weights and constants).

    ``check`` of a floating-point gate is ``False`` whenever an fp32 or BF16
    argument or the output is a NaN: an honest run has none, and NaN payloads
    are the one thing IEEE leaves to the implementation.  The costs count
    the fp32 operations of a gate's sequence.
    """

    pipeline = pipeline_for(arch, dtype)
    if pipeline.operand_bits != 16:
        raise InvalidArtifact("the pinned gate set is defined over BF16 operands")
    k = pipeline.k
    step, step_zero = tensor_core_evaluators(pipeline)
    f32 = (32,)
    return GateSet(
        (
            _pinned(f"tc_dot{k}", 32, k, step, (32, *([16] * (2 * k)))),
            _pinned(f"tc_dot{k}_0", 32, k, step_zero, (16,) * (2 * k)),
            _pinned("bf16_to_f32", 32, 1, lambda args: args[0] << 16, (16,)),
            _pinned(
                "f32_to_bf16",
                16,
                1,
                lambda args: int(f32_to_bf16(np.array(args[0], dtype=np.uint32))),
                f32,
            ),
            _pinned("f32_add", 32, 1, _binary(_ieee(np.add)), f32 * 2),
            _pinned("f32_sub", 32, 1, _binary(_ieee(np.subtract)), f32 * 2),
            _pinned("f32_mul", 32, 1, _binary(_ieee(np.multiply)), f32 * 2),
            _pinned("f32_div", 32, 1, _binary(_ieee(np.divide)), f32 * 2),
            _pinned("f32_max", 32, 1, _binary(f32_max), f32 * 2),
            _pinned("f32_exp", 32, 24, _unary(f32_exp), f32),
            _pinned("f32_tanh", 32, 30, _unary(f32_tanh), f32),
            _pinned("gelu_tanh", 32, 38, _unary(gelu_tanh), f32),
            _pinned("ln_rstd", 32, 3, _unary(ln_rstd), f32),
            _pinned(
                "argmax_select",
                16,
                1,
                lambda args: args[3] if _bits(args[1]) > _bits(args[0]) else args[2],
                (32, 32, 16, 16),
                floats=(True, True, False, False),
                float_out=False,
            ),
            _pinned(
                "token_eq",
                16,
                1,
                lambda args: BF16_ONE if args[0] == args[1] else 0,
                (16, 16),
                floats=(False, False),
                float_out=False,
            ),
            Gate("in", 0, 16, replay_cost=0, proof_cost=1, source=INPUT_SOURCE),
            Gate("weight", 0, 16, replay_cost=0, proof_cost=1, source=WEIGHT_SOURCE),
        ),
        name=f"veritor.pinned.{pipeline.name}",
        version="1",
    )
