import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

_SOPHGO_GRID_CAP = 64
_SMALL_BLOCK_SIZE = 4096
# 1-in/2-in elementwise bitwise kernels compile up to 16384 elems/block here
# (verified against ppl-compile); the bigger tile halves the grid-stride trip
# count on the bandwidth-bound large shapes.
_LARGE_BLOCK_SIZE = 16384
_INTEGER_FAST_DTYPES = {
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
    torch.int64,
}
# int64 load/store miscompiles on this backend (even a plain copy is wrong), so
# we never widen into it; int32 is the chip's native lane width.
_MAX_WIDEN_BYTES = 4


@libentry()
@triton.jit
def _bitwise_or_tensor_contig_kernel(
    a,
    b,
    out,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    num_jobs = tle.num_programs(0)
    block_start = (pid * BLOCK_SIZE).to(tl.int64)
    step = num_jobs * BLOCK_SIZE
    for block_start_offset in range(block_start, n_elements, step):
        offsets = block_start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(a + offsets, mask=mask)
        y = tl.load(b + offsets, mask=mask)
        tl.store(out + offsets, x | y, mask=mask)


@libentry()
@triton.jit
def _bitwise_or_tensor_contig_nomask_kernel(
    a,
    b,
    out,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    num_jobs = tle.num_programs(0)
    block_start = (pid * BLOCK_SIZE).to(tl.int64)
    step = num_jobs * BLOCK_SIZE
    for block_start_offset in range(block_start, n_elements, step):
        offsets = block_start_offset + tl.arange(0, BLOCK_SIZE)
        x = tl.load(a + offsets)
        y = tl.load(b + offsets)
        tl.store(out + offsets, x | y)


@libentry()
@triton.jit(do_not_specialize=["value"])
def _bitwise_or_scalar_contig_kernel(
    a,
    value,
    out,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    num_jobs = tle.num_programs(0)
    block_start = (pid * BLOCK_SIZE).to(tl.int64)
    step = num_jobs * BLOCK_SIZE
    for block_start_offset in range(block_start, n_elements, step):
        offsets = block_start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(a + offsets, mask=mask)
        tl.store(out + offsets, x | value, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["value"])
def _bitwise_or_scalar_contig_nomask_kernel(
    a,
    value,
    out,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    num_jobs = tle.num_programs(0)
    block_start = (pid * BLOCK_SIZE).to(tl.int64)
    step = num_jobs * BLOCK_SIZE
    for block_start_offset in range(block_start, n_elements, step):
        offsets = block_start_offset + tl.arange(0, BLOCK_SIZE)
        x = tl.load(a + offsets)
        tl.store(out + offsets, x | value)


def _widen_view(flat):
    """Reinterpret a contiguous 1-D integer buffer as the widest allowed dtype.

    Bitwise and/or are bit-for-bit independent, so packing several narrow lanes
    into one wider element gives an identical result while cutting the element
    count (and DMA descriptors); int16 -> int32 halves the work.
    """
    nbytes = flat.numel() * flat.element_size()
    if _MAX_WIDEN_BYTES >= 4 and nbytes % 4 == 0:
        return flat.view(torch.int32)
    if nbytes % 2 == 0:
        return flat.view(torch.int16)
    return flat


def _pack_scalar(value, src_dtype, wide_dtype):
    """Replicate a narrow-dtype scalar across the packed wide lanes.

    After widening the tensor, `x | value` must apply per original lane. Bitwise
    ops are per-bit, so replicating the low src_bits across every src_bits-wide
    slot of the wide element reproduces the per-lane operation exactly (verified
    on CPU incl. negative / boundary values).
    """
    if wide_dtype == src_dtype:
        return int(value)
    src_bits = torch.iinfo(src_dtype).bits
    wide_bits = torch.iinfo(wide_dtype).bits
    v = int(value) & ((1 << src_bits) - 1)
    out = 0
    for shift in range(0, wide_bits, src_bits):
        out |= v << shift
    if out >= (1 << (wide_bits - 1)):
        out -= 1 << wide_bits
    return out


def _choose_block_size(n_elements):
    return _SMALL_BLOCK_SIZE if n_elements <= _SMALL_BLOCK_SIZE else _LARGE_BLOCK_SIZE


def _launch_grid(n_elements, block_size):
    return (min(triton.cdiv(n_elements, block_size), _SOPHGO_GRID_CAP),)


def _launch_tensor(a, b, out):
    if out.numel() == 0:
        return out
    aw = _widen_view(a.reshape(-1))
    bw = _widen_view(b.reshape(-1))
    ow = _widen_view(out.reshape(-1))
    n_elements = ow.numel()
    block_size = _choose_block_size(n_elements)
    grid = _launch_grid(n_elements, block_size)
    if n_elements % block_size == 0:
        _bitwise_or_tensor_contig_nomask_kernel[grid](
            aw, bw, ow, n_elements, BLOCK_SIZE=block_size
        )
    else:
        _bitwise_or_tensor_contig_kernel[grid](
            aw, bw, ow, n_elements, BLOCK_SIZE=block_size
        )
    return out


def _launch_scalar(a, value, out):
    if out.numel() == 0:
        return out
    aw = _widen_view(a.reshape(-1))
    ow = _widen_view(out.reshape(-1))
    packed = _pack_scalar(int(value), a.dtype, aw.dtype)
    n_elements = ow.numel()
    block_size = _choose_block_size(n_elements)
    grid = _launch_grid(n_elements, block_size)
    if n_elements % block_size == 0:
        _bitwise_or_scalar_contig_nomask_kernel[grid](
            aw, packed, ow, n_elements, BLOCK_SIZE=block_size
        )
    else:
        _bitwise_or_scalar_contig_kernel[grid](
            aw, packed, ow, n_elements, BLOCK_SIZE=block_size
        )
    return out


def _launch_bool_tensor(a, b, out):
    # bool is stored as 0/1 bytes; `x | y` on the int8 view is exactly the
    # logical OR torch.bitwise_or defines for bool. Fully self-contained (no
    # logical_* / generic delegation) so it can never recurse under use_gems().
    if out.numel() == 0:
        return out
    a = a.contiguous()
    b = b.contiguous()
    ai, bi, oi = a.view(torch.int8), b.view(torch.int8), out.view(torch.int8)
    n = oi.numel()
    block_size = _choose_block_size(n)
    grid = _launch_grid(n, block_size)
    if n % block_size == 0:
        _bitwise_or_tensor_contig_nomask_kernel[grid](
            ai, bi, oi, n, BLOCK_SIZE=block_size
        )
    else:
        _bitwise_or_tensor_contig_kernel[grid](ai, bi, oi, n, BLOCK_SIZE=block_size)
    return out


def _launch_bool_scalar(a, value, out):
    if out.numel() == 0:
        return out
    ai, oi = a.view(torch.int8), out.view(torch.int8)
    n = oi.numel()
    block_size = _choose_block_size(n)
    grid = _launch_grid(n, block_size)
    v = 1 if bool(value) else 0
    if n % block_size == 0:
        _bitwise_or_scalar_contig_nomask_kernel[grid](
            ai, v, oi, n, BLOCK_SIZE=block_size
        )
    else:
        _bitwise_or_scalar_contig_kernel[grid](ai, v, oi, n, BLOCK_SIZE=block_size)
    return out


# ---------------------------------------------------------------------------
# General TPU triton paths. Every TPU tensor -- regardless of contiguity,
# broadcasting or mixed integer dtype -- is normalised and run through a triton
# kernel here. We never fall back to torch.* / .bitwise_or_ for a TPU tensor,
# because that would silently execute the torch_tpu native op instead of triton.
# torch.* is used ONLY for genuinely non-TPU (e.g. CPU) tensors, where it is the
# correct, non-hijacked implementation.
# ---------------------------------------------------------------------------


def _run_tensor_oop(A, B):
    rdtype = torch.result_type(A, B)
    A2 = A if A.dtype == rdtype else A.to(rdtype)
    B2 = B if B.dtype == rdtype else B.to(rdtype)
    if A2.shape != B2.shape:
        A2, B2 = torch.broadcast_tensors(A2, B2)
    A2 = A2.contiguous()
    B2 = B2.contiguous()
    out = torch.empty_like(A2)
    if rdtype == torch.bool:
        return _launch_bool_tensor(A2, B2, out)
    return _launch_tensor(A2, B2, out)


def _run_tensor_inplace(A, B):
    B2 = B if B.dtype == A.dtype else B.to(A.dtype)
    if B2.shape != A.shape:
        B2 = B2.broadcast_to(A.shape)
    B2 = B2.contiguous()
    Ac = A if A.is_contiguous() else A.contiguous()
    if A.dtype == torch.bool:
        _launch_bool_tensor(Ac, B2, Ac)
    else:
        _launch_tensor(Ac, B2, Ac)
    if Ac is not A:
        A.copy_(Ac)
    return A


def _run_scalar_oop(A, value):
    Ac = A.contiguous()
    out = torch.empty_like(Ac)
    if A.dtype == torch.bool:
        return _launch_bool_scalar(Ac, value, out)
    return _launch_scalar(Ac, int(value), out)


def _run_scalar_inplace(A, value):
    Ac = A if A.is_contiguous() else A.contiguous()
    if A.dtype == torch.bool:
        _launch_bool_scalar(Ac, value, Ac)
    else:
        _launch_scalar(Ac, int(value), Ac)
    if Ac is not A:
        A.copy_(Ac)
    return A


def _to_int_scalar(v):
    if isinstance(v, torch.Tensor):
        return int(v.item())
    return int(v)


# These ops are registered under use_gems() for the TPU dispatch key, so the
# operands are always TPU tensors by the time we get here -- no device check is
# needed and we never fall back to the torch_tpu native op.


def bitwise_or_tensor(A, B):
    logger.debug("SOPHGO GEMS BITWISE OR")
    return _run_tensor_oop(A, B)


def bitwise_or_tensor_(A, B):
    logger.debug("SOPHGO GEMS BITWISE OR_")
    return _run_tensor_inplace(A, B)


def bitwise_or_scalar(A, B):
    logger.debug("SOPHGO GEMS BITWISE OR SCALAR")
    return _run_scalar_oop(A, _to_int_scalar(B))


def bitwise_or_scalar_(A, B):
    logger.debug("SOPHGO GEMS BITWISE OR_ SCALAR")
    return _run_scalar_inplace(A, _to_int_scalar(B))


def bitwise_or_scalar_tensor(A, B):
    logger.debug("SOPHGO GEMS BITWISE OR SCALAR TENSOR")
    return _run_scalar_oop(B, _to_int_scalar(A))
