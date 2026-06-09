import logging
import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry


def dot_block_size(N):
    if N >= 1024:
        return 1024
    if N >= 256:
        return 256
    return triton.next_power_of_2(N)


@libentry()
@triton.jit
def dot_reduce_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for off in range(0, N, BLOCK_SIZE):
        idx = off + offsets
        mask = idx < N
        a = tl.load(a_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        acc += a * b

    total = tl.sum(acc[None, :], axis=1)
    out_idx = tl.arange(0, 1)
    tl.store(out_ptr + out_idx, total, mask=out_idx < 1)


@libentry()
@triton.jit
def dot_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    N,
    stride_an,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Specialized matmul kernel for dot product: [1, N] @ [N, 1] = [1, 1]
    
    Direct copy from mm.py's matmul_kernel, adapted for M=1, N=1, K=N case.
    """
    # Initialize accumulator (1x1 matrix)
    accumulator = tl.zeros((1, 1), dtype=tl.float32)
    
    # Create base pointers - exactly like mm.py
    offs_k = tl.arange(0, BLOCK_SIZE)
    # For A: [1, N], we need pointers [0, k] where k varies
    a_ptrs = a_ptr + offs_k[None, :] * stride_an  # Shape: [1, BLOCK_SIZE]
    # For B: [N, 1], we need pointers [k, 0] where k varies  
    b_ptrs = b_ptr + offs_k[:, None] * stride_bn  # Shape: [BLOCK_SIZE, 1]
    
    # Process in blocks along the K dimension (which is N in our case)
    for k in range(0, tl.cdiv(N, BLOCK_SIZE)):
        # Load blocks with masks - exactly like mm.py
        a = tl.load(a_ptrs, mask=offs_k[None, :] < N - k * BLOCK_SIZE, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < N - k * BLOCK_SIZE, other=0.0)
        
        # Convert to float32 for tl.dot (tl.dot may not support float64)
        a = a.to(tl.float32)
        b = b.to(tl.float32)
        
        # Accumulate: [1, BLOCK_SIZE] @ [BLOCK_SIZE, 1] = [1, 1]
        accumulator += tl.dot(a, b)
        
        # Advance pointers - exactly like mm.py
        a_ptrs += BLOCK_SIZE * stride_an
        b_ptrs += BLOCK_SIZE * stride_bn
    
    # Convert to float32
    c = accumulator.to(tl.float32)
    
    # Write back the result - exactly like mm.py does
    offs_cm = tl.arange(0, 1)
    offs_cn = tl.arange(0, 1)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < 1) & (offs_cn[None, :] < 1)
    tl.store(c_ptrs, c, mask=c_mask)


def dot(x, y):
    """
    Compute dot product of two 1D tensors.
    
    Implementation: Treat as matrix multiplication [1, N] @ [N, 1] = [1, 1]
    Uses a specialized kernel based on mm.py's matmul_kernel.
    
    Args:
        x: 1D tensor of shape [N]
        y: 1D tensor of shape [N]
        
    Returns:
        Scalar tensor (dot product result)
    """
    logging.debug("GEMS DOT (Sophgo TPU - standalone implementation)")
    
    assert x.shape == y.shape, "Input vectors must have the same shape"
    assert x.dim() == 1, "Input must be 1D tensors"
    
    N = x.shape[0]
    device = x.device
    original_dtype = x.dtype
    
    # Convert float64 to float32 if needed (TPU doesn't support float64 well)
    if x.dtype == torch.float64:
        x = x.to(torch.float32)
        y = y.to(torch.float32)
    
    x = x.contiguous()
    y = y.contiguous()
    c = torch.empty((1,), dtype=torch.float32, device=device)
    block_size = dot_block_size(N)

    with torch_device_fn.device(device):
        dot_reduce_kernel[(1, 1, 1)](
            x,
            y,
            c,
            N,
            BLOCK_SIZE=block_size,
        )
    
    # Squeeze to scalar and convert dtype if needed
    result = c[0]
    if result.dtype != original_dtype:
        result = result.to(original_dtype)
    
    return result
