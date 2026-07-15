import logging
from typing import List, Tuple, Union

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

# sophgo stack fast paths.
#
# The benchmark stacks 3 contiguous tensors along dim=0 (and dim=-1 under the
# comprehensive level), on 2D (1024, 2^i) and 3D (64, 64, 2^i) shapes. The prior
# sophgo fast path only fired for exactly 4 tensors of ndim==2, so the benchmark
# missed it entirely and fell back to the generic per-tensor pointwise copy
# (one kernel launch + strided view per input, T separate launches).
#
# stack(dim=0) of T contiguous tensors is just T back-to-back whole-tensor
# memcpys into out(T, *shape): out[t*numel : (t+1)*numel] = tensors[t].flat.
# All T copies are fused into ONE kernel — a flat 1D copy with the grid capped
# at the core count, each program grid-striding over BLOCK chunks of every
# tensor. No divmod, no strided views, one launch.
#
# stack(dim=last) adds a new trailing axis: out(*inp_shape, T) with element
# (r, t) at out[r*T + t] — contiguous input read, stride-T store, still one
# fused launch.

_STACK_GRID_CAP = 64
_STACK_BLOCK = 4096
_STACK_MAX_T = 8  # unrolled pointer args; larger T falls back to generic


@libentry()
@triton.jit
def _stack_dim0_flat_kernel(
    out_ptr,
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    numel,  # elements per input tensor
    T: tl.constexpr,
    BLOCK: tl.constexpr,
    CHUNKS,
):
    pid = tle.program_id(0)
    nprog = tle.num_programs(0)
    for c in range(CHUNKS):
        chunk = pid + c * nprog
        off = chunk * BLOCK + tl.arange(0, BLOCK)
        mask = off < numel
        # each tensor's block copied to its own slice out[t*numel + off].
        if T > 0:
            tl.store(
                out_ptr + 0 * numel + off,
                tl.load(p0 + off, mask=mask, other=0),
                mask=mask,
            )
        if T > 1:
            tl.store(
                out_ptr + 1 * numel + off,
                tl.load(p1 + off, mask=mask, other=0),
                mask=mask,
            )
        if T > 2:
            tl.store(
                out_ptr + 2 * numel + off,
                tl.load(p2 + off, mask=mask, other=0),
                mask=mask,
            )
        if T > 3:
            tl.store(
                out_ptr + 3 * numel + off,
                tl.load(p3 + off, mask=mask, other=0),
                mask=mask,
            )
        if T > 4:
            tl.store(
                out_ptr + 4 * numel + off,
                tl.load(p4 + off, mask=mask, other=0),
                mask=mask,
            )
        if T > 5:
            tl.store(
                out_ptr + 5 * numel + off,
                tl.load(p5 + off, mask=mask, other=0),
                mask=mask,
            )
        if T > 6:
            tl.store(
                out_ptr + 6 * numel + off,
                tl.load(p6 + off, mask=mask, other=0),
                mask=mask,
            )
        if T > 7:
            tl.store(
                out_ptr + 7 * numel + off,
                tl.load(p7 + off, mask=mask, other=0),
                mask=mask,
            )


@libentry()
@triton.jit
def _stack_dimlast_kernel(
    out_ptr,
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    numel,  # elements per input tensor
    T: tl.constexpr,
    BLOCK: tl.constexpr,
    CHUNKS,
):
    pid = tle.program_id(0)
    nprog = tle.num_programs(0)
    for c in range(CHUNKS):
        chunk = pid + c * nprog
        off = chunk * BLOCK + tl.arange(0, BLOCK)
        mask = off < numel
        # element r of tensor t lands at out[r*T + t]: contiguous read, stride-T store.
        base = off * T
        if T > 0:
            tl.store(
                out_ptr + base + 0, tl.load(p0 + off, mask=mask, other=0), mask=mask
            )
        if T > 1:
            tl.store(
                out_ptr + base + 1, tl.load(p1 + off, mask=mask, other=0), mask=mask
            )
        if T > 2:
            tl.store(
                out_ptr + base + 2, tl.load(p2 + off, mask=mask, other=0), mask=mask
            )
        if T > 3:
            tl.store(
                out_ptr + base + 3, tl.load(p3 + off, mask=mask, other=0), mask=mask
            )
        if T > 4:
            tl.store(
                out_ptr + base + 4, tl.load(p4 + off, mask=mask, other=0), mask=mask
            )
        if T > 5:
            tl.store(
                out_ptr + base + 5, tl.load(p5 + off, mask=mask, other=0), mask=mask
            )
        if T > 6:
            tl.store(
                out_ptr + base + 6, tl.load(p6 + off, mask=mask, other=0), mask=mask
            )
        if T > 7:
            tl.store(
                out_ptr + base + 7, tl.load(p7 + off, mask=mask, other=0), mask=mask
            )


def _pad_ptrs(tensors):
    return list(tensors) + [tensors[0]] * (_STACK_MAX_T - len(tensors))


def stack(
    tensors: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]], dim: int = 0
) -> torch.Tensor:
    logging.debug("GEMS_SOPHGO_TPU STACK")

    def _generic_stack():
        from flag_gems.ops.stack import stack as generic_stack

        return generic_stack(tensors, dim)

    if len(tensors) == 0:
        raise RuntimeError("stack expected a non-empty TensorList")

    inp_shapes = [list(_.shape) for _ in tensors]
    inp0_shape = inp_shapes[0]
    for i, s in enumerate(inp_shapes[1:]):
        if (dim < -tensors[i + 1].dim() - 1) or (dim > tensors[i + 1].dim()):
            raise IndexError(
                "Dimension out of range (expected to be in range of [{}, {}], but got {})".format(
                    -tensors[i + 1].dim() - 1, tensors[i + 1].dim(), dim
                )
            )
        if s != inp0_shape:
            raise RuntimeError(
                f"stack expects each tensor to be equal size, but got {inp0_shape} at entry 0 and {s} at entry {i+1}"
            )

    ndim = len(inp0_shape)
    if dim < 0:
        dim = dim + ndim + 1

    T = len(tensors)
    all_contig = all(t.is_contiguous() for t in tensors)
    same_dtype = all(t.dtype == tensors[0].dtype for t in tensors)

    if T <= _STACK_MAX_T and all_contig and same_dtype and (dim == 0 or dim == ndim):
        numel = tensors[0].numel()
        if numel > 0:
            out_shape = inp0_shape[:dim] + [T] + inp0_shape[dim:]
            out = torch.empty(
                out_shape, dtype=tensors[0].dtype, device=tensors[0].device
            )
            chunks_total = triton.cdiv(numel, _STACK_BLOCK)
            grid_size = min(chunks_total, _STACK_GRID_CAP)
            chunks = triton.cdiv(chunks_total, grid_size)
            ptrs = _pad_ptrs(tensors)
            kernel = _stack_dim0_flat_kernel if dim == 0 else _stack_dimlast_kernel
            with torch_device_fn.device(tensors[0].device):
                kernel[(grid_size,)](
                    out,
                    *ptrs,
                    numel,
                    T=T,
                    BLOCK=_STACK_BLOCK,
                    CHUNKS=chunks,
                    num_warps=4,
                )
            return out

    return _generic_stack()
