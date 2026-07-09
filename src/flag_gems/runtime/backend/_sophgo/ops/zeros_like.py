import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle


@libentry()
@triton.jit
def zeros_like_fill_kernel(
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # dtype-correct zero fill: tl.zeros picks the output element type instead of
    # storing a hard-coded f32 0.0 literal (the generic zeros_like used the
    # latter, relying on implicit conversion for int/f16 outputs).
    zeros = tl.zeros([BLOCK_SIZE], dtype=output_ptr.type.element_ty)
    tl.store(output_ptr + offsets, zeros, mask=mask)


@libentry()
@triton.jit
def zeros_like_fill_nomask_kernel(
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # no-mask fast path (n_elements % BLOCK_SIZE == 0); same safe increment the
    # full/fill overrides use. Skips the per-element mask compare. dtype-correct
    # like the masked kernel, so it stays valid for all output dtypes.
    pid = tle.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    zeros = tl.zeros([BLOCK_SIZE], dtype=output_ptr.type.element_ty)
    tl.store(output_ptr + offsets, zeros)


def zeros_like(
    x, *, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None
):
    logging.debug("GEMS ZEROS_LIKE SOPHGO")
    if device is None:
        device = x.device
    if dtype is None:
        dtype = x.dtype
    out = torch.empty_like(x, device=device, dtype=dtype)
    n_elements = out.numel()
    if n_elements == 0:
        return out
    # BLOCK_SIZE 4096 matches the tuned sophgo zeros override (was 1024 in the
    # generic path). Pure bandwidth-bound fill; 4096 is already validated safe
    # by the zeros override.
    block_size = 4096
    grid = (triton.cdiv(n_elements, block_size),)
    with torch_device_fn.device(out.device):
        if n_elements % block_size == 0:
            zeros_like_fill_nomask_kernel[grid](
                out,
                n_elements,
                BLOCK_SIZE=block_size,
            )
        else:
            zeros_like_fill_kernel[grid](
                out,
                n_elements,
                BLOCK_SIZE=block_size,
            )
    return out
