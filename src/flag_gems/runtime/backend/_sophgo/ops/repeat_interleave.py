import logging

import torch
import triton
from triton import language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.codegen_config_utils import CodeGenConfig, get_codegen_config
from flag_gems.utils.pointwise_dynamic import pointwise_dynamic
from flag_gems.utils.shape_utils import c_contiguous_stride
from flag_gems.utils.tensor_wrapper import StridedBuffer

from .index_select import index_select

logger = logging.getLogger(__name__)

# Larger tile + wide grid for the copy path, matching the tuned gelu/where
# overrides (default SOPHGO config is only 1024 / grid (512,1,1)).
_base = get_codegen_config()
_config = CodeGenConfig(
    max_tile_size=2048,
    max_grid_size=(65536, 1, 1),
    max_num_warps_per_cta=_base.max_num_warps_per_cta,
    prefer_block_pointer=_base.prefer_block_pointer,
    prefer_1d_tile=_base.prefer_1d_tile,
)


@pointwise_dynamic(num_inputs=1, promotion_methods=[(0, "DEFAULT")], config=_config)
@triton.jit
def copy_func(x):
    return x


# repeat_interleave_self_int fast paths.
#
# The generic codegen broadcast-copy flattens the (A, K, R, B) view to a 1D
# offset and recovers the coords with per-element //shape %shape — the TPU slow
# path (~3 GB/s on the same pattern in triu/flip). For contiguous inp we view
# it as (P, B) with P = A*K and the output as (P*R, B), so
#     out[p*R + r, b] = inp[p, b]
# and use a 2D grid over (p, b): the coordinates come straight from the
# program ids, no divmod. R is a constexpr so the r-axis is materialized with
# tl.arange; the p-axis is grid-strided (cap RI_MAX_GRID programs) to keep
# launch count bounded on huge flattened tensors (the self_int benchmark
# flattens 512^3 -> 134M elements, R=3 -> 402M output).
RI_MAX_GRID = 65536
RI_BLOCK = 1024  # p-axis tile for the B==1 / flattened path
RI_BLOCK_R = 4  # r-axis tile (covers R<=4 directly; loop otherwise)


# B == 1 (dim is last, or tensor was flattened to 1D): out[p*R + r] = inp[p].
# One 2D store of (BLOCK, BLOCK_R) per p-tile; consecutive (p, r) in row-major
# land in consecutive output addresses, so the store is a contiguous run.
@libentry()
@triton.jit
def repeat_interleave_1d_kernel(
    inp_ptr,
    out_ptr,
    N,
    R: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_R: tl.constexpr,
    TPB,
):
    pid = tle.program_id(0)
    nprog = tle.num_programs(0)
    rr = tl.arange(0, BLOCK_R)
    r_mask = rr < R
    for t in range(TPB):
        p = (pid + t * nprog) * BLOCK + tl.arange(0, BLOCK)
        p_mask = p < N
        val = tl.load(inp_ptr + p, mask=p_mask, other=0.0)
        out_off = p.to(tl.int64)[:, None] * R + rr.to(tl.int64)[None, :]
        val_exp = tl.broadcast_to(val[:, None], (BLOCK, BLOCK_R))
        tl.store(out_ptr + out_off, val_exp, mask=p_mask[:, None] & r_mask[None, :])


# B > 1: out[p*R + r, b] = inp[p, b]. Grid over (p, b); the r-axis is unrolled
# (R is constexpr and small — we fall back to codegen for R > RI_MAX_R) so each
# r is a plain 2D (BLOCK_P, BLOCK_B) store, no 3D tile in local mem.
RI_MAX_R = 64
RI_BLOCK_P = 64
RI_BLOCK_B = 64


@libentry()
@triton.jit
def repeat_interleave_2d_kernel(
    inp_ptr,
    out_ptr,
    P,
    B,
    R: tl.constexpr,
    RB,
    BLOCK_P: tl.constexpr,
    BLOCK_B: tl.constexpr,
    TPB,
):
    pid_p = tle.program_id(0)
    pid_b = tle.program_id(1)
    nprog = tle.num_programs(0)
    b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    b_mask = b < B
    for t in range(TPB):
        p = (pid_p + t * nprog) * BLOCK_P + tl.arange(0, BLOCK_P)
        p_mask = p < P
        val = tl.load(
            inp_ptr + p.to(tl.int64)[:, None] * B + b.to(tl.int64)[None, :],
            mask=p_mask[:, None] & b_mask[None, :],
            other=0.0,
        )
        # r-axis unrolled: one 2D store per r. R is constexpr so this is a
        # compile-time unroll, not a runtime loop.
        for r in range(R):
            out_off = p.to(tl.int64)[:, None] * RB + r * B + b.to(tl.int64)[None, :]
            tl.store(out_ptr + out_off, val, mask=p_mask[:, None] & b_mask[None, :])


# repeat_interleave self_tensor dim==0 row gather (rewritten from baseline).
#
# self_tensor with dim==0 is a pure row gather: out[i, :] = inp[index[i], :]
# over a contiguous (rows, c_dim) view — i.e. index_len independent whole-row
# memcpys (input row random, output row sequential).
#
# Every earlier attempt used a 2D (BLOCK_M, BLOCK_C) tile, which forces the
# compiler to materialize 2D offset tensors (inp_off + out_off). Those offsets
# dominate local mem, so the tile had to shrink and the grid exploded into
# thousands of launch-bound programs on the big benchmark shapes.
#
# This version drops to a 1D row-copy: each program owns a set of output rows
# (grid-strided) and copies each row with a single wide 1D DMA. The only live
# buffers are a BLOCK_C value vector and its 1D column offset — local-mem
# footprint is a handful of KB regardless of how many rows we process, so
# BLOCK_C can span the whole row (one wide descriptor per row) and the grid is
# pinned at the core count. The row index is loaded as a scalar, so the copy is
# a plain contiguous load/store, not a gather tile.
@libentry()
@triton.jit
def ri_rowcopy_dim0_kernel(
    inp,
    out,
    index,
    index_len,
    c_dim,
    BLOCK_C: tl.constexpr,
    ROWS,
):
    pid = tle.program_id(0)
    nprog = tle.num_programs(0)
    col = tl.arange(0, BLOCK_C)
    for t in range(ROWS):
        i = pid + t * nprog
        if i < index_len:
            src = tl.load(index + i).to(tl.int32)
            base_in = src * c_dim
            base_out = i * c_dim
            # whole row in one DMA when BLOCK_C >= c_dim; otherwise stride the
            # row in BLOCK_C-wide contiguous chunks.
            for c0 in range(0, c_dim, BLOCK_C):
                cc = c0 + col
                m = cc < c_dim
                v = tl.load(inp + base_in + cc, mask=m, other=0)
                tl.store(out + base_out + cc, v, mask=m)


_INT32_MAX = 2**31 - 1
RI_ROW_GRID = 64
RI_ROWCOPY_MAX_BLOCK_C = 4096


def repeat_interleave_self_int(inp, repeats, dim=None, *, output_size=None):
    logger.debug("GEMS REPEAT_INTERLEAVE_SELF_INT")
    if dim is None:
        inp = inp.flatten()
        dim = 0
    else:
        if (dim < -inp.ndim) or (dim >= inp.ndim):
            raise IndexError(
                "Dimension out of range (expected to be in range of [{}, {}], but got {})".format(
                    -inp.ndim, inp.ndim - 1, dim
                )
            )
    inp_shape = list(inp.shape)
    output_shape = list(inp.shape)

    if dim < 0:
        dim = dim + len(inp_shape)

    output_shape[dim] *= repeats

    if output_size is not None and output_size != output_shape[dim]:
        raise RuntimeError(
            "repeat_interleave: Invalid output_size, expected {} but got {}".format(
                output_shape[dim], output_size
            )
        )

    output = torch.empty(output_shape, dtype=inp.dtype, device=inp.device)

    if repeats == 0:
        return output

    # Fast path: contiguous inp, small R. View as (P, B) over the dim so
    # out[p*R + r, b] = inp[p, b] with a 2D grid (no divmod).
    if inp.is_contiguous() and repeats <= RI_MAX_R:
        A = 1
        for d in range(dim):
            A *= inp_shape[d]
        K = inp_shape[dim]
        B = 1
        for d in range(dim + 1, len(inp_shape)):
            B *= inp_shape[d]
        P = A * K
        if B == 1:
            # 1D / flattened / last-dim: contiguous (BLOCK, BLOCK_R) store.
            BLOCK = RI_BLOCK
            BLOCK_R = min(triton.next_power_of_2(repeats), 64)
            if BLOCK_R < 1:
                BLOCK_R = 1
            n_tiles = triton.cdiv(P, BLOCK)
            grid = (min(n_tiles, RI_MAX_GRID),)
            tpb = triton.cdiv(n_tiles, grid[0])
            with torch_device_fn.device(inp.device):
                repeat_interleave_1d_kernel[grid](
                    inp,
                    output,
                    P,
                    repeats,
                    BLOCK=BLOCK,
                    BLOCK_R=BLOCK_R,
                    TPB=tpb,
                )
            return output
        else:
            BLOCK_P = RI_BLOCK_P
            BLOCK_B = min(triton.next_power_of_2(B), RI_BLOCK_B) if B > 1 else 1
            if BLOCK_B < 1:
                BLOCK_B = 1
            n_tiles = triton.cdiv(P, BLOCK_P)
            grid = (min(n_tiles, RI_MAX_GRID), triton.cdiv(B, BLOCK_B))
            tpb = triton.cdiv(n_tiles, grid[0])
            with torch_device_fn.device(inp.device):
                repeat_interleave_2d_kernel[grid](
                    inp,
                    output,
                    P,
                    B,
                    repeats,
                    repeats * B,
                    BLOCK_P=BLOCK_P,
                    BLOCK_B=BLOCK_B,
                    TPB=tpb,
                )
            return output

    # Fallback: generic codegen broadcast-copy (non-contiguous or huge R).
    inp_stride = list(inp.stride())
    in_view_stride = inp_stride[: dim + 1] + [0] + inp_stride[dim + 1 :]
    out_view_shape = inp_shape[: dim + 1] + [repeats] + inp_shape[dim + 1 :]
    out_view_stride = c_contiguous_stride(out_view_shape)

    in_view = StridedBuffer(inp, out_view_shape, in_view_stride)
    out_view = StridedBuffer(output, out_view_shape, out_view_stride)
    ndim = len(out_view_shape)
    copy_func.instantiate(ndim)(in_view, out0=out_view)
    return output


@triton.jit
def repeat_interleave_tensor_kernel(
    repeats_ptr, cumsum_ptr, out_ptr, size, BLOCK_SIZE: tl.constexpr
):
    pid = tle.program_id(0)
    mask = pid < size
    cumsum = tl.load(cumsum_ptr + pid, mask, other=0)
    repeats = tl.load(repeats_ptr + pid, mask, other=0)
    out_offset = cumsum - repeats

    tl.device_assert(repeats >= 0, "repeats can not be negative")

    out_ptr += out_offset
    for start_k in range(0, repeats, BLOCK_SIZE):
        offsets_k = start_k + tl.arange(0, BLOCK_SIZE)
        mask_k = offsets_k < repeats
        tl.store(out_ptr + offsets_k, pid, mask=mask_k)


def repeat_interleave_tensor(repeats, *, output_size=None):
    logger.debug("GEMS REPEAT_INTERLEAVE_TENSOR")

    assert repeats.ndim == 1, "repeat_interleave only accept 1D vector as repeat"

    cumsum = repeats.cumsum(axis=0)
    result_size = cumsum[-1].item()

    assert result_size >= 0, "repeats can not be negative"

    out = torch.empty((result_size,), dtype=repeats.dtype, device=repeats.device)
    size = repeats.size(0)

    grid = (size,)
    BLOCK_SIZE = 32
    repeat_interleave_tensor_kernel[grid](
        repeats,
        cumsum,
        out,
        size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=1,
    )
    return out


def repeat_interleave_self_tensor(inp, repeats, dim=None, *, output_size=None):
    logger.debug("GEMS REPEAT_INTERLEAVE_SELF_TENSOR")

    if dim is None:
        inp = inp.flatten()
        dim = 0
    else:
        if (dim < -inp.ndim) or (dim >= inp.ndim):
            raise IndexError(
                "Dimension out of range (expected to be in range of [{}, {}], but got {})".format(
                    -inp.ndim, inp.ndim - 1, dim
                )
            )

    if repeats.ndim == 0 or (repeats.ndim == 1 and repeats.size(0) == 1):
        return repeat_interleave_self_int(
            inp, repeats.item(), dim=dim, output_size=output_size
        )
    elif repeats.ndim > 1:
        raise RuntimeError("repeats must be 0-dim or 1-dim tensor")

    inp_shape = list(inp.shape)
    if dim < 0:
        dim = dim + len(inp_shape)

    if repeats.size(0) != inp_shape[dim]:
        raise RuntimeError(
            "repeats must have the same size as input along dim, but got \
                repeats.size(0) = {} and input.size({}) = {}".format(
                repeats.size(0), dim, inp_shape[dim]
            )
        )

    indices = repeat_interleave_tensor(repeats)

    # dim==0 contiguous is a row gather: out[i, :] = inp[index[i], :]. Copy each
    # output row with a single wide 1D DMA; grid pinned at the core count.
    if dim == 0 and inp.is_contiguous():
        c_dim = 1
        for d in range(1, inp.ndim):
            c_dim *= inp_shape[d]
        index_len = indices.numel()
        use_i32 = (
            c_dim > 0
            and (index_len * c_dim) <= _INT32_MAX
            and (inp_shape[0] * c_dim) <= _INT32_MAX
        )
        if use_i32:
            inp_2d = inp.reshape(inp_shape[0], c_dim)
            out_2d = torch.empty((index_len, c_dim), dtype=inp.dtype, device=inp.device)
            indices_c = indices.contiguous()

            # BLOCK_C spans the whole row when it fits the per-row DMA cap; wider
            # rows are strided inside the kernel. Only a single BLOCK_C vector is
            # live, so this never pressures local mem.
            block_c = min(triton.next_power_of_2(c_dim), RI_ROWCOPY_MAX_BLOCK_C)
            if block_c < 1:
                block_c = 1
            grid_row = min(index_len, RI_ROW_GRID)
            rows = triton.cdiv(index_len, grid_row)
            grid = (grid_row,)
            with torch_device_fn.device(inp.device):
                ri_rowcopy_dim0_kernel[grid](
                    inp_2d,
                    out_2d,
                    indices_c,
                    index_len,
                    c_dim,
                    BLOCK_C=block_c,
                    ROWS=rows,
                )
            return out_2d.reshape([index_len] + inp_shape[1:])

    return index_select(inp, dim, indices)
