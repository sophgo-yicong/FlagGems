import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.ops.reciprocal import reciprocal as _generic_reciprocal

logger = logging.getLogger(__name__)

BLOCK_SIZE = 4096
MAX_GRID = 64

# Raw-kernel sophgo fast path (grid-cap + no-mask) for the common case:
# contiguous floating-point tensor elementwise reciprocal (1/x). Non-contiguous
# or integer input falls back to the generic pointwise_dynamic op (which handles
# INT_TO_FLOAT promotion). Compute is in fp32 (store downcasts).


def _dispatch(n):
    num_tiles = math.ceil(n / BLOCK_SIZE)
    grid = min(num_tiles, MAX_GRID)
    tpb = math.ceil(num_tiles / grid)
    return grid, tpb


def _can_fast(t):
    return isinstance(t, torch.Tensor) and t.is_contiguous() and t.is_floating_point()


@triton.jit
def reciprocal_kernel_fast(
    x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr, TPB: tl.constexpr
):
    pid = tl.program_id(0)
    for t in range(TPB):
        offs = (pid + t * tl.num_programs(0)) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offs).to(tl.float32)
        tl.store(out_ptr + offs, 1.0 / x)


@triton.jit
def reciprocal_kernel_masked(
    x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr, TPB: tl.constexpr
):
    pid = tl.program_id(0)
    for t in range(TPB):
        offs = (pid + t * tl.num_programs(0)) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
        tl.store(out_ptr + offs, 1.0 / x, mask=mask)


def _run(a, out):
    n = a.numel()
    grid, tpb = _dispatch(n)
    k = reciprocal_kernel_fast if n % BLOCK_SIZE == 0 else reciprocal_kernel_masked
    k[(grid,)](a, out, n, BLOCK_SIZE=BLOCK_SIZE, TPB=tpb)
    return out


def reciprocal(A):
    logger.debug("GEMS RECIPROCAL (sophgo_tpu)")
    if _can_fast(A):
        return _run(A, torch.empty_like(A))
    return _generic_reciprocal(A)


def reciprocal_(A):
    logger.debug("GEMS RECIPROCAL_ (sophgo_tpu)")
    if _can_fast(A):
        return _run(A, A)
    return _generic_reciprocal(A)
