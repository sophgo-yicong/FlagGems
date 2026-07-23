import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def _fused_mlp_kernel(
    x_ptr,
    w_gate_ptr,
    w_up_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_x_m,
    stride_x_k,
    stride_wg_n,
    stride_wg_k,
    stride_wu_n,
    stride_wu_k,
    stride_o_m,
    stride_o_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fully-fused gate+up MLP: silu(x@w_gate.T) * (x@w_up.T)

    TWO tl.dot in the same K-loop (one for gate, one for up), then
    post-loop silu computation.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_x_m + offs_k[None, :] * stride_x_k
    wg_ptrs = w_gate_ptr + offs_k[:, None] * stride_wg_k + offs_n[None, :] * stride_wg_n
    wu_ptrs = w_up_ptr + offs_k[:, None] * stride_wu_k + offs_n[None, :] * stride_wu_n

    acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_K)
        w_mask = (offs_k[:, None] < K - k * BLOCK_K) & (offs_n[None, :] < N)

        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        wg = tl.load(wg_ptrs, mask=w_mask, other=0.0)
        wu = tl.load(wu_ptrs, mask=w_mask, other=0.0)

        acc_gate += tl.dot(x, wg)
        acc_up += tl.dot(x, wu)

        x_ptrs += BLOCK_K * stride_x_k
        wg_ptrs += BLOCK_K * stride_wg_k
        wu_ptrs += BLOCK_K * stride_wu_k

    # Post-loop: silu(gate) * up
    silu_gate = acc_gate * tl.sigmoid(acc_gate)
    result = silu_gate * acc_up

    o_ptrs = out_ptr + offs_m[:, None] * stride_o_m + offs_n[None, :] * stride_o_n
    o_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(o_ptrs, result.to(out_ptr.dtype.element_ty), mask=o_mask)


def llama_mlp_gate_up(x, w_gate, w_up, act_out=None):
    """Fully-fused gate+up projection with SwiGLU activation.

    Computes: silu(x @ w_gate.T) * (x @ w_up.T)
    Single kernel with two tl.dot in the same loop + post-loop silu.

    Args:
        x: input tensor [M, K]
        w_gate: gate weight [N, K]
        w_up: up weight [N, K]
        act_out: optional pre-allocated output tensor [M, N]

    Returns:
        silu(x @ w_gate.T) * (x @ w_up.T)  shape [M, N]
    """
    logger.debug("GEMS LLAMA MLP GATE_UP FORWARD (fully-fused)")

    M, K = x.shape
    N = w_gate.shape[0]
    assert w_gate.shape == (N, K)
    assert w_up.shape == (N, K)

    out = act_out if act_out is not None else torch.empty(
        (M, N), device=x.device, dtype=x.dtype)

    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 32

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    with torch_device_fn.device(x.device):
        _fused_mlp_kernel[grid](
            x, w_gate, w_up, out,
            M, N, K,
            x.stride(0), x.stride(1),
            w_gate.stride(0), w_gate.stride(1),
            w_up.stride(0), w_up.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )

    return out
