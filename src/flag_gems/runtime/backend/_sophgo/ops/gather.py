import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

# sophgo gather fast path for the common 2D dim==1 case:
#   out[i, j] = inp[i, index[i, j]]
#
# The generic gather codegen flattens the output to 1D and recovers the per-
# element coordinates with `cur_offset % index_shape{k}` / `// index_shape{k}`
# — the TPU integer-divmod slow path (~3 GB/s, see the triu/flip case). Here a
# 2D grid supplies (i, j) straight from the program ids, so there is no divmod:
# the row index i and column position j are affine, only the gathered input
# column comes from `index`. The row is contiguous per output tile; the input
# read is a per-element indexed load along the columns (unavoidable — that is
# what gather is), but killing the divmod and keeping wide row/col tiles is the
# win over the flattened generic kernel.

_GATHER_GRID_CAP = 64


# 1D per-row formulation (same lesson as repeat_interleave_self_tensor): a 2D
# (BLOCK_M, BLOCK_N) tile forces the indexed input load to materialize a full
# 2D offset tensor (rows*stride + idx), which — with the value and index copies
# PPL keeps live — overflows local mem even at a 2048-value tile. Dropping to
# one row per program-iteration makes the row base a scalar, so only 1D column
# vectors (index, gathered value, per-row offset) are live: local-mem footprint
# is a few KB regardless of M, and BLOCK_N can span the whole row.
@libentry()
@triton.jit
def _gather_dim1_rowwise_kernel(
    inp_ptr,
    index_ptr,
    out_ptr,
    M,
    N,  # index/out column count
    inp_row_stride,  # = inp.shape[1] (elements per input row)
    BLOCK_N: tl.constexpr,
    ROWS_PER_PROG,
):
    pid = tle.program_id(0)
    nprog = tle.num_programs(0)
    col = tl.arange(0, BLOCK_N)
    for t in range(ROWS_PER_PROG):
        i = pid + t * nprog
        if i < M:
            row_out_base = i * N  # index/out are contiguous (M, N)
            row_in_base = i * inp_row_stride
            for c0 in range(0, N, BLOCK_N):
                cols = c0 + col
                m = cols < N
                idx = tl.load(index_ptr + row_out_base + cols, mask=m, other=0).to(
                    tl.int32
                )
                val = tl.load(inp_ptr + row_in_base + idx, mask=m, other=0)
                tl.store(out_ptr + row_out_base + cols, val, mask=m)


_GATHER_MAX_BLOCK_N = 4096


def _select_gather_block_n(N):
    return min(triton.next_power_of_2(N), _GATHER_MAX_BLOCK_N)


def _generic_gather(inp, dim, index, out=None, sparse_grad=False):
    from flag_gems.ops.gather import gather as generic_gather

    return generic_gather(inp, dim, index, out=out, sparse_grad=sparse_grad)


def gather(inp, dim, index, out=None, sparse_grad=False):
    logger.debug("GEMS_SOPHGO_TPU GATHER")

    dim = dim % inp.ndim
    # Fast path: 2D contiguous inp/index, gather along dim==1, int32/int64 index.
    if (
        inp.ndim == 2
        and index.ndim == 2
        and dim == 1
        and inp.is_contiguous()
        and index.is_contiguous()
        and index.dtype in (torch.int32, torch.int64)
        and out is None
        # dim==1 gather shares the row index i directly: out[i,j] =
        # inp[i, index[i,j]], so index may cover fewer rows than inp.
        and index.shape[0] <= inp.shape[0]
    ):
        M, N = index.shape
        if M > 0 and N > 0:
            output = torch.empty((M, N), dtype=inp.dtype, device=inp.device)
            block_n = _select_gather_block_n(N)
            grid_size = min(M, _GATHER_GRID_CAP)
            rows_per_prog = triton.cdiv(M, grid_size)
            with torch_device_fn.device(inp.device):
                _gather_dim1_rowwise_kernel[(grid_size,)](
                    inp,
                    index,
                    output,
                    M,
                    N,
                    inp.shape[1],
                    BLOCK_N=block_n,
                    ROWS_PER_PROG=rows_per_prog,
                    num_warps=4,
                )
            return output

    return _generic_gather(inp, dim, index, out=out, sparse_grad=sparse_grad)
