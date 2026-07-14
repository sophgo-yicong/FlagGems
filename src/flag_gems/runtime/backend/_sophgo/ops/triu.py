import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import triton_lang_extension as tle

# sophgo: hand-tuned block sizes (no libentry/libtuner autotune wrapper).
# Both the 2D and the batched paths use a regular (row, col) tile with a plain
# `row + diag <= col` triangular test — no per-element integer div/mod, which
# is what crippled the old flattened batch kernel (~3 GB/s vs ~50 GB/s for 2D).
# Tile bytes stay well under the 64KB/CTA local-mem budget (the kernel also
# holds the where() result + index compares, so live footprint is a few tiles).
_M_BLOCK_SIZE = 1
_N_BLOCK_SIZE = 2048
_M_GRID_CAP = 64
_NUM_WARPS = 2


@triton.jit(do_not_specialize=["diagonal"])
def triu_kernel(
    X,
    Y,
    M,
    N,
    diagonal,
    M_BLOCK_SIZE: tl.constexpr,
    N_BLOCK_SIZE: tl.constexpr,
):
    row_start = tle.program_id(0) * M_BLOCK_SIZE
    row_step = tle.num_programs(0) * M_BLOCK_SIZE
    for row_off in range(row_start, M, row_step):
        row = row_off + tl.arange(0, M_BLOCK_SIZE)[:, None]
        m_mask = row < M
        Xr = X + row * N
        Yr = Y + row * N

        for n_offset in range(0, N, N_BLOCK_SIZE):
            cols = n_offset + tl.arange(0, N_BLOCK_SIZE)[None, :]
            n_mask = cols < N
            mask = m_mask and n_mask

            x = tl.load(Xr + cols, mask, other=0.0)
            y = tl.where(row + diagonal <= cols, x, 0.0)
            tl.store(Yr + cols, y, mask=mask)


@triton.jit(do_not_specialize=["diagonal"])
def triu_batch_kernel(
    X,
    Y,
    batch,
    M,
    N,
    diagonal,
    M_BLOCK_SIZE: tl.constexpr,
    N_BLOCK_SIZE: tl.constexpr,
):
    # Treat the batched triu as `batch` independent (M, N) matrices. The row
    # index is the *in-matrix* row (regular arange) and the column is a regular
    # arange too, so the triangular test is a plain `row + diag <= col` — no
    # per-element //N / %N (integer div/mod is very slow on this TPU and was
    # the reason the old flattened kernel only hit ~3 GB/s vs ~50 GB/s for 2D).
    batch_id = tle.program_id(0)
    MN = M * N
    Xb = X + batch_id * MN
    Yb = Y + batch_id * MN

    row_start = tle.program_id(1) * M_BLOCK_SIZE
    row_step = tle.num_programs(1) * M_BLOCK_SIZE
    for row_off in range(row_start, M, row_step):
        row = row_off + tl.arange(0, M_BLOCK_SIZE)[:, None]
        m_mask = row < M
        Xr = Xb + row * N
        Yr = Yb + row * N
        for n_offset in range(0, N, N_BLOCK_SIZE):
            cols = n_offset + tl.arange(0, N_BLOCK_SIZE)[None, :]
            n_mask = cols < N
            mask = m_mask and n_mask
            x = tl.load(Xr + cols, mask, other=0.0)
            y = tl.where(row + diagonal <= cols, x, 0.0)
            tl.store(Yr + cols, y, mask=mask)


INT32_MAX = torch.iinfo(torch.int32).max


def triu(A, diagonal=0):
    logging.debug("SOPHGO GEMS TRIU")
    A = A.contiguous()
    out = torch.empty_like(A)
    assert len(A.shape) > 1, "Input tensor must have at least 2 dimensions"
    M, N = A.shape[-2:]
    with torch_device_fn.device(A.device):
        if len(A.shape) == 2:
            grid = (min(triton.cdiv(M, _M_BLOCK_SIZE), _M_GRID_CAP),)
            triu_kernel[grid](
                A,
                out,
                M,
                N,
                diagonal,
                M_BLOCK_SIZE=_M_BLOCK_SIZE,
                N_BLOCK_SIZE=_N_BLOCK_SIZE,
                num_warps=_NUM_WARPS,
            )
        else:
            batch = int(torch.numel(A) / M / N)
            B = A.view(batch, M, N)
            out3 = out.view(batch, M, N)
            grid = (
                batch,
                min(triton.cdiv(M, _M_BLOCK_SIZE), _M_GRID_CAP),
            )
            triu_batch_kernel[grid](
                B,
                out3,
                batch,
                M,
                N,
                diagonal,
                M_BLOCK_SIZE=_M_BLOCK_SIZE,
                N_BLOCK_SIZE=_N_BLOCK_SIZE,
                num_warps=_NUM_WARPS,
            )
            out = out.view(A.shape)
    return out
