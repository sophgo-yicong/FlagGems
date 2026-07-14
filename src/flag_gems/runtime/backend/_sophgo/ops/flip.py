import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry, tl_extra_shim
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

# sophgo: hand-tuned inner block (no libentry/libtuner on the fast paths).
# flip is a gather (read at flipped coords, write in order). The 2D/3D fast
# paths below use a regular multi-dim grid so every index is a plain arange —
# no per-element integer div/mod, which crippled the old flattened rank5 kernel
# (~3 GB/s on triu's identical pattern). The inner N axis is the load/store
# vector; BLOCK_N stays under the 64KB/CTA local-mem budget (the kernel holds
# the index vector, the gather offsets, and the values).
_BLOCK_N = 1024
_BLOCK_N_1D = 4096
_M_GRID_CAP = 64


@triton.jit
def flip_1d_kernel(
    inp_ptr,
    out_ptr,
    N,
    BLOCK_N: tl.constexpr,
):
    # 1D full flip (dims must contain the only axis, else clone short-circuits).
    # Read a FORWARD block from the mirrored position, reverse it in-register,
    # write to the ascending destination — so the DMA load is forward/contiguous
    # (a naive src=N-1-n reads backwards and on this TPU hits ~0.5 GB/s for 1G).
    pid = tle.program_id(0)
    step = tle.num_programs(0)
    num_blocks = tl.cdiv(N, BLOCK_N)
    r = tl.arange(0, BLOCK_N)
    for blk in range(pid, num_blocks, step):
        dst_start = blk * BLOCK_N
        dst_offs = dst_start + r
        src_offs = (
            N - dst_start - BLOCK_N
        ) + r  # mirrored, but ascending -> forward read
        dst_mask = dst_offs < N
        src_mask = (src_offs >= 0) & (src_offs < N)
        x = tl.load(inp_ptr + src_offs, mask=src_mask, other=0.0)
        x_rev = tl_extra_shim.flip(x, 0)
        tl.store(out_ptr + dst_offs, x_rev, mask=dst_mask)


@triton.jit
def flip_2d_kernel(
    inp_ptr,
    out_ptr,
    M,
    N,
    stride_m,
    stride_n,
    flip_m,
    BLOCK_N: tl.constexpr,
):
    # Last axis NOT flipped (flip_n == 0): src_n = n stays in [0, N) for in-range
    # lanes, so there is no negative gather offset. Out-of-range lanes (n >= N)
    # go past the end of the tensor, which the device tolerates under mask.
    m_start = tle.program_id(0)
    m_step = tle.num_programs(0)
    n = tle.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n < N

    for m_off in range(m_start, M, m_step):
        src_m = m_off + flip_m * (M - 1 - 2 * m_off)
        in_off = src_m * stride_m + n * stride_n
        vals = tl.load(inp_ptr + in_off, mask=n_mask, other=0.0)
        out_off = m_off * N + n
        tl.store(out_ptr + out_off, vals, mask=n_mask)


@triton.jit
def flip_2d_inner_kernel(
    inp_ptr,
    out_ptr,
    M,
    N,
    stride_m,
    stride_n,
    flip_m,
    BLOCK_N: tl.constexpr,
):
    # Last axis IS flipped (flip_n == 1). A naive src_n = N-1-n reads backwards
    # and goes negative for n >= N (the device rejects negative masked loads;
    # clamping with tl.minimum fixed correctness but shattered the arange DMA
    # pattern and cost ~90x). Instead mirror the block: read a FORWARD block
    # from (N - dst_start - BLOCK) + arange — still an arange offset, so the DMA
    # stays a fast contiguous load — then reverse in-register and write ascending.
    m_start = tle.program_id(0)
    m_step = tle.num_programs(0)
    dst_start = tle.program_id(1) * BLOCK_N
    r = tl.arange(0, BLOCK_N)
    dst_offs = dst_start + r
    src_offs = (N - dst_start - BLOCK_N) + r
    dst_mask = dst_offs < N
    src_mask = (src_offs >= 0) & (src_offs < N)

    for m_off in range(m_start, M, m_step):
        src_m = m_off + flip_m * (M - 1 - 2 * m_off)
        in_off = src_m * stride_m + src_offs * stride_n
        x = tl.load(inp_ptr + in_off, mask=src_mask, other=0.0)
        x_rev = tl_extra_shim.flip(x, 0)
        out_off = m_off * N + dst_offs
        tl.store(out_ptr + out_off, x_rev, mask=dst_mask)


@triton.jit
def flip_3d_kernel(
    inp_ptr,
    out_ptr,
    B,
    M,
    N,
    stride_b,
    stride_m,
    stride_n,
    flip_b,
    flip_m,
    BLOCK_N: tl.constexpr,
):
    # Last axis NOT flipped (flip_n == 0): src_n = n is non-negative in-range.
    pid_b = tle.program_id(0)
    m_start = tle.program_id(1)
    m_step = tle.num_programs(1)
    n = tle.program_id(2) * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n < N

    src_b = pid_b + flip_b * (B - 1 - 2 * pid_b)

    for m_off in range(m_start, M, m_step):
        src_m = m_off + flip_m * (M - 1 - 2 * m_off)
        in_off = src_b * stride_b + src_m * stride_m + n * stride_n
        vals = tl.load(inp_ptr + in_off, mask=n_mask, other=0.0)
        out_off = pid_b * M * N + m_off * N + n
        tl.store(out_ptr + out_off, vals, mask=n_mask)


@triton.jit
def flip_3d_inner_kernel(
    inp_ptr,
    out_ptr,
    B,
    M,
    N,
    stride_b,
    stride_m,
    stride_n,
    flip_b,
    flip_m,
    BLOCK_N: tl.constexpr,
):
    # Last axis IS flipped (flip_n == 1): mirror-block read + in-register reverse
    # (see flip_2d_inner_kernel). Avoids the negative backward read.
    pid_b = tle.program_id(0)
    m_start = tle.program_id(1)
    m_step = tle.num_programs(1)
    dst_start = tle.program_id(2) * BLOCK_N
    r = tl.arange(0, BLOCK_N)
    dst_offs = dst_start + r
    src_offs = (N - dst_start - BLOCK_N) + r
    dst_mask = dst_offs < N
    src_mask = (src_offs >= 0) & (src_offs < N)

    src_b = pid_b + flip_b * (B - 1 - 2 * pid_b)

    for m_off in range(m_start, M, m_step):
        src_m = m_off + flip_m * (M - 1 - 2 * m_off)
        in_off = src_b * stride_b + src_m * stride_m + src_offs * stride_n
        x = tl.load(inp_ptr + in_off, mask=src_mask, other=0.0)
        x_rev = tl_extra_shim.flip(x, 0)
        out_off = pid_b * M * N + m_off * N + dst_offs
        tl.store(out_ptr + out_off, x_rev, mask=dst_mask)


@libentry()
@triton.jit
def flip_generic_rank5_kernel(
    inp_ptr,
    out_ptr,
    outer_num,
    size0,
    size1,
    size2,
    size3,
    size4,
    inp_stride0,
    inp_stride1,
    inp_stride2,
    inp_stride3,
    inp_stride4,
    flip0,
    flip1,
    flip2,
    flip3,
    flip4,
    BLOCK_OUTER: tl.constexpr,
    BLOCK_INNER: tl.constexpr,
):
    # Fallback for ndim >= 4 (rare). Still uses the flattened-outer div/mod to
    # recover multi-dim coords — the 2D/3D fast paths above avoid it.
    outer = tle.program_id(0) * BLOCK_OUTER + tl.arange(0, BLOCK_OUTER)
    inner = tle.program_id(1) * BLOCK_INNER + tl.arange(0, BLOCK_INNER)
    outer_mask = outer < outer_num
    inner_mask = inner < size4

    tmp = outer
    coord3 = tmp % size3
    tmp = tmp // size3
    coord2 = tmp % size2
    tmp = tmp // size2
    coord1 = tmp % size1
    tmp = tmp // size1
    coord0 = tmp % size0

    src0 = tl.where(flip0 != 0, size0 - 1 - coord0, coord0)
    src1 = tl.where(flip1 != 0, size1 - 1 - coord1, coord1)
    src2 = tl.where(flip2 != 0, size2 - 1 - coord2, coord2)
    src3 = tl.where(flip3 != 0, size3 - 1 - coord3, coord3)
    src4 = tl.where(flip4 != 0, size4 - 1 - inner, inner)

    src_offs = (
        src0[:, None] * inp_stride0
        + src1[:, None] * inp_stride1
        + src2[:, None] * inp_stride2
        + src3[:, None] * inp_stride3
        + src4[None, :] * inp_stride4
    )
    mask = outer_mask[:, None] & inner_mask[None, :]
    vals = tl.load(inp_ptr + src_offs, mask=mask, other=0)
    out_offs = outer[:, None] * size4 + inner[None, :]
    tl.store(out_ptr + out_offs, vals, mask=mask)


def flip(A: torch.Tensor, dims) -> torch.Tensor:
    logging.debug("GEMS FLIP")
    normalized_dims = []
    for dim in dims:
        assert (
            dim >= -A.dim() and dim < A.dim()
        ), "Dimension out of range (expected to be in range of [{}, {}], but got {})".format(
            -A.dim(), A.dim() - 1, dim
        )
        normalized_dim = dim % A.dim()
        assert (
            normalized_dim not in normalized_dims
        ), "dim {} appears multiple times in the list of dims".format(dim)
        normalized_dims.append(normalized_dim)

    if A.numel() <= 1 or all(A.size(dim) <= 1 for dim in normalized_dims):
        return A.clone()

    flip_set = set(normalized_dims)

    # 1D flip: dims must contain the only axis (else clone short-circuited above),
    # so it is a full reverse. Use the mirror-block path (forward DMA load +
    # in-register reverse) — a naive src=N-1-n reads backwards and is very slow.
    if A.is_contiguous() and A.ndim == 1:
        N = A.shape[0]
        out = torch.empty_like(A)
        grid = (min(triton.cdiv(N, _BLOCK_N_1D), _M_GRID_CAP),)
        flip_1d_kernel[grid](
            A,
            out,
            N,
            BLOCK_N=_BLOCK_N_1D,
        )
        return out

    # Fast paths: a regular multi-dim grid where every index is a plain arange,
    # so source coords are computed with O(1) arithmetic — no per-element integer
    # div/mod. Handles non-contiguous inputs too (strides are passed in; the
    # output is a fresh contiguous buffer). If the LAST axis is flipped, use the
    # mirror-block kernel (forward DMA + in-register reverse) to avoid the
    # negative backward read that the device rejects; otherwise the plain kernel.
    if A.ndim == 2:
        M, N = A.shape
        out = torch.empty_like(A)
        grid = (min(triton.cdiv(M, 1), _M_GRID_CAP), triton.cdiv(N, _BLOCK_N))
        if 1 in flip_set:
            flip_2d_inner_kernel[grid](
                A,
                out,
                M,
                N,
                A.stride(0),
                A.stride(1),
                int(0 in flip_set),
                BLOCK_N=_BLOCK_N,
            )
        else:
            flip_2d_kernel[grid](
                A,
                out,
                M,
                N,
                A.stride(0),
                A.stride(1),
                int(0 in flip_set),
                BLOCK_N=_BLOCK_N,
            )
        return out

    if A.ndim == 3:
        B, M, N = A.shape
        out = torch.empty_like(A)
        grid = (
            B,
            min(triton.cdiv(M, 1), _M_GRID_CAP),
            triton.cdiv(N, _BLOCK_N),
        )
        if 2 in flip_set:
            flip_3d_inner_kernel[grid](
                A,
                out,
                B,
                M,
                N,
                A.stride(0),
                A.stride(1),
                A.stride(2),
                int(0 in flip_set),
                int(1 in flip_set),
                BLOCK_N=_BLOCK_N,
            )
        else:
            flip_3d_kernel[grid](
                A,
                out,
                B,
                M,
                N,
                A.stride(0),
                A.stride(1),
                A.stride(2),
                int(0 in flip_set),
                int(1 in flip_set),
                BLOCK_N=_BLOCK_N,
            )
        return out

    if A.ndim <= 5:
        padded_shape = (1,) * (5 - A.ndim) + tuple(A.shape)
        padded_strides = (0,) * (5 - A.ndim) + tuple(A.stride())
        flip_flags = [0] * 5
        dim_offset = 5 - A.ndim
        for dim in normalized_dims:
            flip_flags[dim_offset + dim] = 1

        out = torch.empty(A.shape, device=A.device, dtype=A.dtype)
        outer_num = A.numel() // padded_shape[4]
        grid = (triton.cdiv(outer_num, 16), triton.cdiv(padded_shape[4], 64))
        flip_generic_rank5_kernel[grid](
            A,
            out,
            outer_num,
            padded_shape[0],
            padded_shape[1],
            padded_shape[2],
            padded_shape[3],
            padded_shape[4],
            padded_strides[0],
            padded_strides[1],
            padded_strides[2],
            padded_strides[3],
            padded_strides[4],
            flip_flags[0],
            flip_flags[1],
            flip_flags[2],
            flip_flags[3],
            flip_flags[4],
            BLOCK_OUTER=16,
            BLOCK_INNER=64,
        )
        return out

    # ndim >= 6: rare. Same per-dim structure as the original sophgo_backend
    # flip, but the reversed index is built with sophgo's own arange and the
    # gather with sophgo's own index_select — no torch.arange / torch.index_select.
    from flag_gems.runtime.backend._sophgo.ops.arange import (
        arange_start as _sophgo_arange_start,
    )
    from flag_gems.runtime.backend._sophgo.ops.index_select import (
        index_select as _sophgo_index_select,
    )

    out = A
    for dim in normalized_dims:
        if out.size(dim) <= 1:
            continue
        n = out.size(dim)
        index = _sophgo_arange_start(
            n - 1, -1, -1, dtype=torch.int32, device=out.device
        )
        out = _sophgo_index_select(out, dim, index)
    return out
