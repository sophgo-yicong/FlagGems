import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)

BLOCK_SIZE = 4096
MAX_GRID = 64

# isfinite(x): x is neither +/-inf nor NaN. Detected by testing the IEEE
# exponent bits: a value is finite iff its exponent field is NOT all-ones.
# We deliberately avoid float compares (`x == x`, `x < inf`): this TPU returns
# True for `nan == nan` and has unreliable NaN ordering (see isinf.py), so only
# the integer exponent-mask test is trustworthy here.
#
# The raw grid-cap kernels below cover the common fp16/bf16/fp32 contiguous
# case; fp64 and anything non-contiguous fall back to the generic
# pointwise_dynamic op which keeps the same bit-logic for every dtype.

_EXP_MASK_FP32 = 0x7F800000
_EXP_MASK_FP16 = 0x7C00
_EXP_MASK_BF16 = 0x7F80


@triton.jit
def _isfinite_fp32_fast(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr, TPB: tl.constexpr):
    EXP: tl.constexpr = 0x7F800000
    pid = tl.program_id(0)
    for t in range(TPB):
        offs = (pid + t * tl.num_programs(0)) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        bits = tl.load(x_ptr + offs).to(tl.int32, bitcast=True)
        tl.store(out_ptr + offs, (bits & EXP) != EXP)


@triton.jit
def _isfinite_fp32_masked(
    x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr, TPB: tl.constexpr
):
    EXP: tl.constexpr = 0x7F800000
    pid = tl.program_id(0)
    for t in range(TPB):
        offs = (pid + t * tl.num_programs(0)) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n
        bits = tl.load(x_ptr + offs, mask=mask).to(tl.int32, bitcast=True)
        tl.store(out_ptr + offs, (bits & EXP) != EXP, mask=mask)


@triton.jit
def _isfinite_i16_fast(
    x_ptr,
    out_ptr,
    n,
    EXP_MASK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TPB: tl.constexpr,
):
    pid = tl.program_id(0)
    for t in range(TPB):
        offs = (pid + t * tl.num_programs(0)) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        bits = tl.load(x_ptr + offs).to(tl.int16, bitcast=True)
        tl.store(out_ptr + offs, (bits & EXP_MASK) != EXP_MASK)


@triton.jit
def _isfinite_i16_masked(
    x_ptr,
    out_ptr,
    n,
    EXP_MASK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TPB: tl.constexpr,
):
    pid = tl.program_id(0)
    for t in range(TPB):
        offs = (pid + t * tl.num_programs(0)) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n
        bits = tl.load(x_ptr + offs, mask=mask).to(tl.int16, bitcast=True)
        tl.store(out_ptr + offs, (bits & EXP_MASK) != EXP_MASK, mask=mask)


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "ALWAYS_BOOL")])
@triton.jit
def isfinite_func(x):
    # Generic fallback: same exponent-mask logic for every dtype.
    if x.dtype.is_fp64():
        int_bits = x.to(tl.int64, bitcast=True)
        exp_mask = 0x7FF0000000000000
        return (int_bits & exp_mask) != exp_mask
    elif x.dtype.is_fp32():
        int_bits = x.to(tl.int32, bitcast=True)
        return (int_bits & 0x7F800000) != 0x7F800000
    elif x.dtype.is_fp16():
        int_bits = x.to(tl.int16, bitcast=True)
        return (int_bits & 0x7C00) != 0x7C00
    elif x.dtype.is_bf16():
        int_bits = x.to(tl.int16, bitcast=True)
        return (int_bits & 0x7F80) != 0x7F80
    else:
        x_fp32 = x.to(tl.float32)
        int_bits = x_fp32.to(tl.int32, bitcast=True)
        return (int_bits & 0x7F800000) != 0x7F800000


def _can_fast(A):
    return A.is_contiguous() and A.dtype in (
        torch.float32,
        torch.float16,
        torch.bfloat16,
    )


def isfinite(
    A: torch.Tensor,
) -> torch.Tensor:
    logger.debug("GEMS ISFINITE (sophgo_tpu)")
    if not A.is_floating_point():
        return torch.full(A.shape, True, dtype=torch.bool, device=A.device)
    if not _can_fast(A):
        return isfinite_func(A)

    # fp16/bf16 -> fp32 is a lossless widening (inf/NaN preserved), so we always
    # run the fp32 exponent bit test. The int16 bitcast path was unreliable on
    # this TPU (correct compile, wrong result on fp16/bf16), so it is not used.
    src = A if A.dtype == torch.float32 else A.float()
    out = torch.empty(A.shape, dtype=torch.bool, device=A.device)
    n = src.numel()
    num_tiles = math.ceil(n / BLOCK_SIZE)
    grid = min(num_tiles, MAX_GRID)
    tpb = math.ceil(num_tiles / grid)
    divisible = n % BLOCK_SIZE == 0
    k = _isfinite_fp32_fast if divisible else _isfinite_fp32_masked
    k[(grid,)](src, out, n, BLOCK_SIZE=BLOCK_SIZE, TPB=tpb)
    return out
