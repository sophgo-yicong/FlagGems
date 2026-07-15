import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, pointwise_dynamic
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.codegen_config_utils import CodeGenConfig
from flag_gems.utils.shape_utils import MemOverlap, has_internal_overlapping
from flag_gems.utils.tensor_wrapper import StridedBuffer

from ..ops.copy import copy

logger = logging.getLogger(__name__)

# sophgo slice_scatter fast paths.
#   1. The dominant inp -> out full-tensor copy is replaced by a hand-tuned flat
#      grid-stride memcpy (grid capped at the core count) instead of the generic
#      pointwise_dynamic `copy` codegen.
#   2. When the slice dim is the last dim and inp/src are contiguous, the strided
#      src -> out-slice write uses a 2D rowwise kernel (no integer divmod). Other
#      dim/stride shapes fall back to the original copy_func.
_SS_GRID_CAP = 64
_SS_BLOCK = 4096

config_ = CodeGenConfig(
    64,
    (512, 1, 1),
    32,
    False,
    prefer_1d_tile=int(triton.__version__[0]) < 3,
)


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")], config=config_)
@triton.jit
def copy_func(x):
    return x


@libentry()
@triton.jit
def _flat_copy_kernel(dst_ptr, src_ptr, n, BLOCK: tl.constexpr, CHUNKS):
    pid = tle.program_id(0)
    nprog = tle.num_programs(0)
    for c in range(CHUNKS):
        chunk = pid + c * nprog
        off = chunk * BLOCK + tl.arange(0, BLOCK)
        m = off < n
        tl.store(dst_ptr + off, tl.load(src_ptr + off, mask=m, other=0), mask=m)


@libentry()
@triton.jit
def _strided_slice_write_lastdim_kernel(
    out_ptr,
    src_ptr,
    M,  # number of rows (product of all leading dims)
    K,  # src last-dim size
    out_row_stride,  # elements between consecutive out rows
    start,
    step,
    BLOCK: tl.constexpr,
    ROWS_PER_PROG,
):
    pid = tle.program_id(0)
    nprog = tle.num_programs(0)
    col = tl.arange(0, BLOCK)
    for t in range(ROWS_PER_PROG):
        r = pid + t * nprog
        if r < M:
            for c0 in range(0, K, BLOCK):
                k = c0 + col
                m = k < K
                src_val = tl.load(src_ptr + r * K + k, mask=m, other=0)
                tl.store(
                    out_ptr + r * out_row_stride + start + k * step, src_val, mask=m
                )


def _flat_copy(dst, src):
    n = src.numel()
    chunks = math.ceil(n / _SS_BLOCK)
    grid = min(chunks, _SS_GRID_CAP) if chunks > 0 else 1
    chunks_per_prog = math.ceil(chunks / grid) if grid > 0 else 1
    with torch_device_fn.device(dst.device):
        _flat_copy_kernel[(grid,)](dst, src, n, BLOCK=_SS_BLOCK, CHUNKS=chunks_per_prog)


def _strided_slice_write_lastdim(out, src, start, step):
    ndim = src.ndim
    leading = src.shape[:-1]
    M = 1
    for d in leading:
        M *= d
    K = src.shape[-1]
    out_row_stride = out.stride()[-2] if ndim >= 2 else 0
    # out is contiguous in the leading dims; last-dim stride is 1, so the write
    # position for src[r, k] is r * out_row_stride + start + k * step.
    grid = min(M, _SS_GRID_CAP) if M > 0 else 1
    rows_per_prog = math.ceil(M / grid) if grid > 0 else 1
    with torch_device_fn.device(out.device):
        _strided_slice_write_lastdim_kernel[(grid,)](
            out,
            src,
            M,
            K,
            out_row_stride,
            start,
            step,
            BLOCK=_SS_BLOCK,
            ROWS_PER_PROG=rows_per_prog,
        )


def slice_scatter(inp, src, dim=0, start=None, end=None, step=1):
    logging.debug("GEMS SLICE_SCATTER (sophgo_tpu)")
    assert src.device == inp.device, "inp and src reside on different devices."
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert step > 0, "slice step must be positive"
    dim = dim % inp.ndim

    start = start or 0
    end = end or inp.size(dim)
    if end < 0:
        end = end % inp.size(dim)

    valid_shape = list(inp.shape)
    valid_shape[dim] = triton.cdiv(end - start, step)
    assert (
        list(src.shape) == valid_shape
    ), "Expected src to have a size equal to the slice of self"

    if has_internal_overlapping(inp) == MemOverlap.Yes:
        out = torch.empty(inp.size(), dtype=inp.dtype, device=inp.device)
    else:
        out = torch.empty_strided(
            inp.size(), inp.stride(), dtype=inp.dtype, device=inp.device
        )

    ndim = inp.ndim

    # Fast path 1: inp -> out full copy via tuned flat memcpy when contiguous.
    if inp.is_contiguous() and out.is_contiguous():
        _flat_copy(out, inp)
    else:
        copy(inp, out0=out)

    # Fast path 2: strided src -> out-slice write when the slice dim is the last
    # dim and both out and src are contiguous (no divmod needed). Otherwise the
    # generic codegen copy_func handles arbitrary strides.
    if dim == ndim - 1 and out.is_contiguous() and src.is_contiguous() and ndim >= 1:
        _strided_slice_write_lastdim(out, src, start, step)
    else:
        new_strides = list(out.stride())
        new_strides[dim] *= step
        out_slice = StridedBuffer(
            out,
            shape=src.shape,
            strides=new_strides,
            offset=start * out.stride(dim),
        )
        copy_func.instantiate(ndim)(src, out0=out_slice)

    return out
