import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle


@libentry()
@triton.jit
def arange_kernel(y_ptr, start, step, size, BLOCK_SIZE: tl.constexpr):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size
    vals = start + offsets * step
    tl.store(y_ptr + offsets, vals, mask=mask)


def arange_start(
    start, end, step=1, *, dtype=None, layout=None, device=None, pin_memory=None
):
    logging.debug("GEMS ARANGE SOPHGO")
    if not all(float(v).is_integer() for v in (start, end, step)):
        from flag_gems.ops.arange import arange_start as generic_arange_start

        return generic_arange_start(
            start,
            end,
            step,
            dtype=dtype,
            layout=layout,
            device=device,
            pin_memory=pin_memory,
        )

    if dtype is torch.int64:
        sgn = (step > 0) - (step < 0)
        size = (end - start + step - sgn) // step
    else:
        size = math.ceil((end - start) / step)

    if dtype is None:
        dtype = torch.int64
    if pin_memory is None:
        pin_memory = False
    if device is None:
        device = runtime.device.name

    result = torch.empty((size,), device=device, dtype=dtype, pin_memory=pin_memory)
    grid = (triton.cdiv(size, 4096),)
    arange_kernel[grid](result, start, step, size, BLOCK_SIZE=4096)
    return result


def arange(end, *, dtype=None, layout=None, device=None, pin_memory=None):
    return arange_start(
        0, end, 1, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )
