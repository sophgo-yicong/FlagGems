import itertools
import logging
import math
from typing import List, Tuple, Union

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.codegen_config_utils import CodeGenConfig
from flag_gems.utils.pointwise_dynamic import pointwise_dynamic
from flag_gems.utils.tensor_wrapper import StridedBuffer

logger = logging.getLogger(__name__)

# sophgo cat fast paths for the common contiguous cases.
#   dim == 0:  out is the back-to-back flat concatenation of each (contiguous)
#              input; fuse all T per-tensor memcpys into ONE grid-stride launch.
#   dim == last: each input row r maps to out row r at a per-tensor last-dim
#              offset; fuse all T per-tensor row copies into ONE launch.
# Both kill the T separate pointwise_dynamic launches (one per input) and cap the
# grid at the core count. Generic / non-contiguous / middle-dim cases fall back
# to the original per-tensor copy_func.
_CAT_GRID_CAP = 64
_CAT_BLOCK = 4096
_CAT_MAX_T = 8  # unrolled pointer args; larger T falls back to generic

config_ = CodeGenConfig(
    32,
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
def _cat_dim0_kernel(
    out_ptr,
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    n0,
    n1,
    n2,
    n3,
    n4,
    n5,
    n6,
    n7,
    o0,
    o1,
    o2,
    o3,
    o4,
    o5,
    o6,
    o7,
    T: tl.constexpr,
    BLOCK: tl.constexpr,
    CHUNKS,
):
    pid = tle.program_id(0)
    nprog = tle.num_programs(0)
    for c in range(CHUNKS):
        chunk = pid + c * nprog
        off = chunk * BLOCK + tl.arange(0, BLOCK)
        if T > 0:
            m = off < n0
            tl.store(out_ptr + o0 + off, tl.load(p0 + off, mask=m, other=0), mask=m)
        if T > 1:
            m = off < n1
            tl.store(out_ptr + o1 + off, tl.load(p1 + off, mask=m, other=0), mask=m)
        if T > 2:
            m = off < n2
            tl.store(out_ptr + o2 + off, tl.load(p2 + off, mask=m, other=0), mask=m)
        if T > 3:
            m = off < n3
            tl.store(out_ptr + o3 + off, tl.load(p3 + off, mask=m, other=0), mask=m)
        if T > 4:
            m = off < n4
            tl.store(out_ptr + o4 + off, tl.load(p4 + off, mask=m, other=0), mask=m)
        if T > 5:
            m = off < n5
            tl.store(out_ptr + o5 + off, tl.load(p5 + off, mask=m, other=0), mask=m)
        if T > 6:
            m = off < n6
            tl.store(out_ptr + o6 + off, tl.load(p6 + off, mask=m, other=0), mask=m)
        if T > 7:
            m = off < n7
            tl.store(out_ptr + o7 + off, tl.load(p7 + off, mask=m, other=0), mask=m)


@libentry()
@triton.jit
def _cat_dimlast_kernel(
    out_ptr,
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    o0,
    o1,
    o2,
    o3,
    o4,
    o5,
    o6,
    o7,
    M,
    S,
    T: tl.constexpr,
    BLOCK: tl.constexpr,
    ROWS_PER_PROG,
):
    pid = tle.program_id(0)
    nprog = tle.num_programs(0)
    col = tl.arange(0, BLOCK)
    for t in range(ROWS_PER_PROG):
        r = pid + t * nprog
        if r < M:
            out_row = r * S
            if T > 0:
                m = col < s0
                tl.store(
                    out_ptr + out_row + o0 + col,
                    tl.load(p0 + r * s0 + col, mask=m, other=0),
                    mask=m,
                )
            if T > 1:
                m = col < s1
                tl.store(
                    out_ptr + out_row + o1 + col,
                    tl.load(p1 + r * s1 + col, mask=m, other=0),
                    mask=m,
                )
            if T > 2:
                m = col < s2
                tl.store(
                    out_ptr + out_row + o2 + col,
                    tl.load(p2 + r * s2 + col, mask=m, other=0),
                    mask=m,
                )
            if T > 3:
                m = col < s3
                tl.store(
                    out_ptr + out_row + o3 + col,
                    tl.load(p3 + r * s3 + col, mask=m, other=0),
                    mask=m,
                )
            if T > 4:
                m = col < s4
                tl.store(
                    out_ptr + out_row + o4 + col,
                    tl.load(p4 + r * s4 + col, mask=m, other=0),
                    mask=m,
                )
            if T > 5:
                m = col < s5
                tl.store(
                    out_ptr + out_row + o5 + col,
                    tl.load(p5 + r * s5 + col, mask=m, other=0),
                    mask=m,
                )
            if T > 6:
                m = col < s6
                tl.store(
                    out_ptr + out_row + o6 + col,
                    tl.load(p6 + r * s6 + col, mask=m, other=0),
                    mask=m,
                )
            if T > 7:
                m = col < s7
                tl.store(
                    out_ptr + out_row + o7 + col,
                    tl.load(p7 + r * s7 + col, mask=m, other=0),
                    mask=m,
                )


def _pad_ptrs(tensors):
    return list(tensors) + [tensors[0]] * (_CAT_MAX_T - len(tensors))


def _cat_dim0_fast(A, out0):
    T = len(A)
    numels = [a.numel() for a in A]
    offsets = list(itertools.accumulate(numels[:-1], initial=0))
    ptrs = _pad_ptrs(list(A))
    ns = numels + [numels[0]] * (_CAT_MAX_T - T)
    os_ = offsets + [0] * (_CAT_MAX_T - T)
    max_n = max(numels) if numels else 1
    chunks = math.ceil(max_n / _CAT_BLOCK)
    grid = min(chunks, _CAT_GRID_CAP) if chunks > 0 else 1
    chunks_per_prog = math.ceil(chunks / grid) if grid > 0 else 1
    with torch_device_fn.device(out0.device):
        _cat_dim0_kernel[(grid,)](
            out0,
            *ptrs,
            *ns,
            *os_,
            T,
            BLOCK=_CAT_BLOCK,
            CHUNKS=chunks_per_prog,
        )


def _cat_dimlast_fast(A, out0, dim):
    T = len(A)
    leading_shape = A[0].shape[:dim]
    M = 1
    for d in leading_shape:
        M *= d
    sizes = [a.shape[dim] for a in A]  # last-dim size per tensor
    S = sum(sizes)
    offsets = list(itertools.accumulate(sizes[:-1], initial=0))
    ptrs = _pad_ptrs(list(A))
    ss = sizes + [sizes[0]] * (_CAT_MAX_T - T)
    os_ = offsets + [0] * (_CAT_MAX_T - T)
    grid = min(M, _CAT_GRID_CAP) if M > 0 else 1
    rows_per_prog = math.ceil(M / grid) if grid > 0 else 1
    with torch_device_fn.device(out0.device):
        _cat_dimlast_kernel[(grid,)](
            out0,
            *ptrs,
            *ss,
            *os_,
            M,
            S,
            T,
            BLOCK=_CAT_BLOCK,
            ROWS_PER_PROG=rows_per_prog,
        )


def cat(
    A: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]], dim: int = 0
) -> torch.Tensor:
    logging.debug("GEMS CAT (sophgo_tpu)")

    if len(A) == 0:
        raise RuntimeError("torch.cat(): expected a non-empty list of Tensors")
    if len(A) == 1:
        return A[0]

    # Check if only one tensor is non-empty
    non_empty_tensors = [a for a in A if a.shape[dim] != 0]
    if len(non_empty_tensors) == 1:
        # If only one non-empty tensor, return it directly
        return non_empty_tensors[0]

    assert dim >= -A[0].ndim and dim < A[0].ndim, f"Invalid dim: {dim}"
    # Convert negative dim to positive
    dim = dim % A[0].ndim

    # Same rank check
    inp_shapes = [list(_.shape) for _ in A]
    inp0_shape = inp_shapes[0]
    for s in inp_shapes[1:]:
        if len(s) != len(inp0_shape):
            raise RuntimeError(
                f"Tensors must have same number of dimensions: got {len(inp0_shape)} and {len(s)}"
            )
    # Same size check
    for tensor_idx, inp_shape in enumerate(inp_shapes):
        for idx, (common_length, length) in enumerate(zip(inp0_shape, inp_shape)):
            if idx == dim:
                continue
            elif length != common_length:
                raise RuntimeError(
                    f"Sizes of tensors must match except in dimension {dim}. "
                    f"Expected size {common_length} but got size {length} for tensor number "
                    f"{tensor_idx} in the list"
                )

    out_shape = list(inp0_shape)
    out_shape[dim] = sum(s[dim] for s in inp_shapes)
    out0 = torch.empty(out_shape, dtype=A[0].dtype, device=A[0].device)

    # sophgo fast paths: all contiguous, small T, dim at an end.
    T = len(A)
    all_contig = all(a.is_contiguous() for a in A)
    if all_contig and T <= _CAT_MAX_T:
        if dim == 0:
            _cat_dim0_fast(A, out0)
            return out0
        if dim == A[0].ndim - 1:
            _cat_dimlast_fast(A, out0, dim)
            return out0

    # Generic fallback: per-tensor strided copy.
    out0_strides = out0.stride()
    out0_offsets = list(
        itertools.accumulate(
            [s[dim] * out0_strides[dim] for s in inp_shapes[:-1]], initial=0
        )
    )

    for a, out0_offset in zip(A, out0_offsets):
        in_view = StridedBuffer(a, a.shape, a.stride())
        out_view = StridedBuffer(out0, a.shape, out0.stride(), offset=out0_offset)
        copy_func.instantiate(a.ndim)(in_view, out0=out_view)
    return out0
