import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.ops.minimum import minimum as _generic_minimum

logger = logging.getLogger(__name__)

BLOCK_SIZE = 4096
MAX_GRID = 64

# Raw-kernel sophgo fast path (grid-cap + no-mask) for the common case:
# same-shape contiguous floating-point tensor vs tensor elementwise minimum.
# Everything else (scalar operand, broadcasting, non-contiguous, integer)
# falls back to the generic pointwise_dynamic op. Compute is in fp32
# (store downcasts).


def _dispatch(n):
    num_tiles = math.ceil(n / BLOCK_SIZE)
    grid = min(num_tiles, MAX_GRID)
    tpb = math.ceil(num_tiles / grid)
    return grid, tpb


def _can_fast(*tensors):
    ref = tensors[0]
    return all(
        isinstance(t, torch.Tensor)
        and t.is_contiguous()
        and t.shape == ref.shape
        and t.is_floating_point()
        for t in tensors
    )


@triton.jit
def minimum_tt_kernel_fast(
    x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr, TPB: tl.constexpr
):
    pid = tl.program_id(0)
    for t in range(TPB):
        offs = (pid + t * tl.num_programs(0)) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offs).to(tl.float32)
        y = tl.load(y_ptr + offs).to(tl.float32)
        tl.store(out_ptr + offs, tl.minimum(x, y))


@triton.jit
def minimum_tt_kernel_masked(
    x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr, TPB: tl.constexpr
):
    pid = tl.program_id(0)
    for t in range(TPB):
        offs = (pid + t * tl.num_programs(0)) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
        y = tl.load(y_ptr + offs, mask=mask).to(tl.float32)
        tl.store(out_ptr + offs, tl.minimum(x, y), mask=mask)


def _run_tt(a, b, out):
    n = a.numel()
    grid, tpb = _dispatch(n)
    k = minimum_tt_kernel_fast if n % BLOCK_SIZE == 0 else minimum_tt_kernel_masked
    k[(grid,)](a, b, out, n, BLOCK_SIZE=BLOCK_SIZE, TPB=tpb)
    return out


def minimum(X, Y):
    logger.debug("GEMS MINIMUM (sophgo_tpu)")
    if isinstance(X, torch.Tensor) and isinstance(Y, torch.Tensor) and _can_fast(X, Y):
        return _run_tt(X, Y, torch.empty_like(X))
    return _generic_minimum(X, Y)
