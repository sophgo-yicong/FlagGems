import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)

BLOCK_SIZE = 4096
MAX_GRID = 64

# isnan(x): x is NaN  <=>  exponent bits all-ones AND mantissa != 0.
# We MUST NOT use the `x != x` self-inequality pattern: this TPU returns True
# for `nan == nan` (so `nan != nan` yields False), which would misclassify
# every NaN as non-NaN (see isinf.py). The integer exponent+mantissa bit test
# is the only reliable detector here.
#
# Raw grid-cap kernels cover the common fp16/bf16/fp32 contiguous case; fp64
# and non-contiguous inputs fall back to the generic pointwise_dynamic op.

_EXP_FP32 = 0x7F800000
_MANT_FP32 = 0x007FFFFF
_EXP_FP16 = 0x7C00
_MANT_FP16 = 0x03FF
_EXP_BF16 = 0x7F80
_MANT_BF16 = 0x007F


@triton.jit
def _isnan_fp32_fast(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr, TPB: tl.constexpr):
    EXP: tl.constexpr = 0x7F800000
    MANT: tl.constexpr = 0x007FFFFF
    pid = tl.program_id(0)
    for t in range(TPB):
        offs = (pid + t * tl.num_programs(0)) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        bits = tl.load(x_ptr + offs).to(tl.int32, bitcast=True)
        is_exp = (bits & EXP) == EXP
        has_mant = (bits & MANT) != 0
        tl.store(out_ptr + offs, is_exp & has_mant)


@triton.jit
def _isnan_fp32_masked(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr, TPB: tl.constexpr):
    EXP: tl.constexpr = 0x7F800000
    MANT: tl.constexpr = 0x007FFFFF
    pid = tl.program_id(0)
    for t in range(TPB):
        offs = (pid + t * tl.num_programs(0)) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n
        bits = tl.load(x_ptr + offs, mask=mask).to(tl.int32, bitcast=True)
        is_exp = (bits & EXP) == EXP
        has_mant = (bits & MANT) != 0
        tl.store(out_ptr + offs, is_exp & has_mant, mask=mask)


@triton.jit
def _isnan_i16_fast(
    x_ptr,
    out_ptr,
    n,
    EXP: tl.constexpr,
    MANT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TPB: tl.constexpr,
):
    pid = tl.program_id(0)
    for t in range(TPB):
        offs = (pid + t * tl.num_programs(0)) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        bits = tl.load(x_ptr + offs).to(tl.int16, bitcast=True)
        is_exp = (bits & EXP) == EXP
        has_mant = (bits & MANT) != 0
        tl.store(out_ptr + offs, is_exp & has_mant)


@triton.jit
def _isnan_i16_masked(
    x_ptr,
    out_ptr,
    n,
    EXP: tl.constexpr,
    MANT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TPB: tl.constexpr,
):
    pid = tl.program_id(0)
    for t in range(TPB):
        offs = (pid + t * tl.num_programs(0)) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n
        bits = tl.load(x_ptr + offs, mask=mask).to(tl.int16, bitcast=True)
        is_exp = (bits & EXP) == EXP
        has_mant = (bits & MANT) != 0
        tl.store(out_ptr + offs, is_exp & has_mant, mask=mask)


@pointwise_dynamic(promotion_methods=[(0, "ALWAYS_BOOL")])
@triton.jit
def isnan_func(x):
    # Generic fallback: exponent all-ones AND mantissa != 0, per dtype.
    if x.dtype.is_fp64():
        bits = x.to(tl.int64, bitcast=True)
        exp = 0x7FF0000000000000
        mant = 0x000FFFFFFFFFFFFF
        return ((bits & exp) == exp) & ((bits & mant) != 0)
    elif x.dtype.is_fp32():
        bits = x.to(tl.int32, bitcast=True)
        return ((bits & 0x7F800000) == 0x7F800000) & ((bits & 0x007FFFFF) != 0)
    elif x.dtype.is_fp16():
        bits = x.to(tl.int16, bitcast=True)
        return ((bits & 0x7C00) == 0x7C00) & ((bits & 0x03FF) != 0)
    elif x.dtype.is_bf16():
        bits = x.to(tl.int16, bitcast=True)
        return ((bits & 0x7F80) == 0x7F80) & ((bits & 0x007F) != 0)
    else:
        x_fp32 = x.to(tl.float32)
        bits = x_fp32.to(tl.int32, bitcast=True)
        return ((bits & 0x7F800000) == 0x7F800000) & ((bits & 0x007FFFFF) != 0)


def _can_fast(A):
    return (
        isinstance(A, torch.Tensor)
        and A.is_contiguous()
        and A.dtype in (torch.float32, torch.float16, torch.bfloat16)
    )


def isnan(A):
    logger.debug("GEMS ISNAN (sophgo_tpu)")
    if isinstance(A, torch.Tensor) and not A.is_floating_point():
        return torch.zeros(A.shape, dtype=torch.bool, device=A.device)
    if not _can_fast(A):
        return isnan_func(A)

    # fp16/bf16 -> fp32 is a lossless widening (NaN stays NaN), so we always run
    # the fp32 exponent+mantissa bit test. The int16 bitcast path was unreliable
    # on this TPU (correct compile, wrong result), so it is not used.
    src = A if A.dtype == torch.float32 else A.float()
    out = torch.empty(A.shape, dtype=torch.bool, device=A.device)
    n = src.numel()
    num_tiles = math.ceil(n / BLOCK_SIZE)
    grid = min(num_tiles, MAX_GRID)
    tpb = math.ceil(num_tiles / grid)
    divisible = n % BLOCK_SIZE == 0
    k = _isnan_fp32_fast if divisible else _isnan_fp32_masked
    k[(grid,)](src, out, n, BLOCK_SIZE=BLOCK_SIZE, TPB=tpb)
    return out
