import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.runtime import torch_device_fn


def count_nonzero_dim1_block_m(M):
    if M >= 1024:
        return 16
    if M >= 256:
        return 8
    return 4


def count_nonzero_dim1_block_n(N):
    if N >= 512:
        return 128
    if N >= 256:
        return 64
    return 32


def count_nonzero_dim0_block_m(M):
    if M >= 1024:
        return 64
    if M >= 256:
        return 32
    return 16


def count_nonzero_dim0_block_n(N):
    if N >= 512:
        return 64
    if N >= 128:
        return 32
    return 16


@libentry()
@triton.jit
def count_nonzero_kernel_1(x_ptr, mid_ptr, numel, BLOCK_SIZE: tl.constexpr):
    """
    第一阶段：计算每个 block 内的非零元素数量
    使用 2D tensor 模式避免标量操作（参考 mean 算子修复）

    问题：原始实现中 tl.sum() 返回标量后存储会触发 ppl.get_value，
    导致直接指针解引用在 TPU 上失败。

    修复：使用 2D tensor 操作，保持所有中间结果为 tensor 形式。
    """
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < numel

    # 加载数据并转为非零 mask [BLOCK_SIZE]
    x = tl.load(x_ptr + offset, mask=mask, other=0)
    is_nonzero = (x != 0).to(tl.float32)  # 使用 FP32（PPL avgPool2D 不支持 INT32）

    # 转换为 2D tensor [1, BLOCK_SIZE]
    is_nonzero_2d = is_nonzero[None, :]

    # 在 axis=1 上 sum，结果是 [1] tensor（非标量）
    # 这避免了标量操作触发 ppl.get_value
    local_count = tl.sum(is_nonzero_2d, axis=1)  # shape: [1]

    # 使用 1-element tensor 存储模式
    # 构造存储地址为 tensor（避免标量存储触发 ppl.get_value）
    store_offset = tl.arange(0, 1)  # [0]
    store_mask = store_offset < 1   # [True]
    store_addr = mid_ptr + pid + store_offset

    # 存储 [1] tensor
    tl.store(store_addr, local_count, mask=store_mask)


@libentry()
@triton.jit
def count_nonzero_kernel_2(mid_ptr, out_ptr, mid_size, BLOCK_MID: tl.constexpr):
    """
    第二阶段：汇总中间结果
    使用 2D tensor 模式避免标量操作
    """
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < mid_size

    # 加载中间结果 [BLOCK_MID]
    mid_val = tl.load(mid_ptr + offset, mask=mask, other=0.0)

    # 转换为 2D tensor [1, BLOCK_MID]
    mid_val_2d = mid_val[None, :]

    # 在 axis=1 上 sum，结果是 [1] tensor
    total_count = tl.sum(mid_val_2d, axis=1)  # shape: [1]

    # 1-element tensor 存储
    store_offset = tl.arange(0, 1)
    store_mask = store_offset < 1
    tl.store(out_ptr + store_offset, total_count, mask=store_mask)


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("count_nonzero_dim"), key=["M", "N"])
@triton.jit
def count_nonzero_dim_kernel(X, Out, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """
    TPU 适配版本：按维度计算非零元素
    使用 2D tensor 模式避免标量操作（参考 mean_dim_kernel）

    问题：原始 count_nonzero_kernel 使用 tl.sum() 返回标量，
    导致 tl.store() 触发 ppl.get_value 在 TPU 上失败。

    修复：使用 [BLOCK_M, BLOCK_N] 的 2D 累加器，
    tl.sum(axis=1) 返回 [BLOCK_M] tensor 而非标量。
    """
    # 处理 BLOCK_M 行（2D 模式）
    pid = tle.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    X = X + pid * N
    Out = Out + pid
    row_mask = pid < M

    # 2D 累加器 [BLOCK_M, BLOCK_N]
    counts = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        x = tl.load(X + cols, mask=mask, other=0.0)
        is_nonzero = (x != 0).to(tl.float32)
        counts += is_nonzero

    # axis=1 方向 sum → [BLOCK_M] tensor（非标量）
    row_counts = tl.sum(counts, axis=1)
    # 转换为 2D 用于存储 [BLOCK_M, 1]
    row_counts = row_counts[:, None]
    tl.store(Out, row_counts.to(tl.int32), row_mask)


@libentry()
@triton.jit
def count_nonzero_dim1_2d_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tle.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    row_mask = pid_m < M

    counts = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask
        x = tl.load(x_ptr + pid_m * stride_xm + cols * stride_xn, mask=mask, other=0.0)
        counts += (x != 0).to(tl.float32)

    row_counts = tl.sum(counts, axis=1)[:, None]
    tl.store(out_ptr + pid_m, row_counts.to(tl.int32), row_mask)


@libentry()
@triton.jit
def count_nonzero_dim0_2d_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tle.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    col_mask = pid_n < N

    counts = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, M, BLOCK_M):
        rows = off + tl.arange(0, BLOCK_M)[:, None]
        row_mask = rows < M
        mask = row_mask and col_mask
        x = tl.load(x_ptr + rows * stride_xm + pid_n * stride_xn, mask=mask, other=0.0)
        counts += (x != 0).to(tl.float32)

    col_counts = tl.sum(counts, axis=0)[None, :]
    tl.store(out_ptr + pid_n, col_counts.to(tl.int32), col_mask)


def count_nonzero(x, dim=None):
    """
    TPU 适配版本：
    - dim=None: 使用两阶段 kernel + 2D tensor 模式，完全在 Triton 中完成
    - dim=N: 使用 count_nonzero_dim_kernel + 2D tensor 模式（参考 mean_dim）
    """
    logging.debug("GEMS_SOPHGO_TPU COUNT NONZERO")
    if dim is not None:
        # 使用 2D tensor 模式进行维度约减（类似 mean_dim）
        assert dim >= -x.ndim and dim < x.ndim, "Invalid dim"
        dim = dim % x.ndim

        if x.ndim == 2 and dim in (0, 1):
            x = x.contiguous()
            M, N = x.shape
            out_shape = [N] if dim == 0 else [M]
            out = torch.empty(out_shape, dtype=torch.int32, device=x.device)

            with torch_device_fn.device(x.device):
                if dim == 1:
                    block_m = count_nonzero_dim1_block_m(M)
                    block_n = count_nonzero_dim1_block_n(N)
                    grid = (triton.cdiv(M, block_m),)
                    count_nonzero_dim1_2d_kernel[grid](
                        x,
                        out,
                        M,
                        N,
                        x.stride(0),
                        x.stride(1),
                        BLOCK_M=block_m,
                        BLOCK_N=block_n,
                    )
                else:
                    block_m = count_nonzero_dim0_block_m(M)
                    block_n = count_nonzero_dim0_block_n(N)
                    grid = (triton.cdiv(N, block_n),)
                    count_nonzero_dim0_2d_kernel[grid](
                        x,
                        out,
                        M,
                        N,
                        x.stride(0),
                        x.stride(1),
                        BLOCK_M=block_m,
                        BLOCK_N=block_n,
                    )
            return out

        shape = list(x.shape)
        x = dim_compress(x, dim)
        N = shape[dim]
        M = x.numel() // N

        out_shape = list(shape)
        del out_shape[dim]
        out = torch.empty(out_shape, dtype=torch.int32, device=x.device)

        grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)
        with torch_device_fn.device(x.device):
            count_nonzero_dim_kernel[grid](x.flatten(), out, M, N)
        return out
    else:
        # 全张量情况：两阶段 reduction（完全在 Triton 中完成）
        # 参考 mean 算子的修复方式
        x = x.contiguous().flatten()
        numel = x.numel()

        # 计算 block 大小和 program 数量（与 mean 算子相同）
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(numel)))
        mid_size = triton.cdiv(numel, block_size)
        block_mid = triton.next_power_of_2(mid_size)

        # 分配中间结果和输出
        mid = torch.empty((mid_size,), dtype=torch.float32, device=x.device)
        out = torch.empty([], dtype=torch.float32, device=x.device)

        with torch_device_fn.device(x.device):
            # 第一阶段：计算各 block 的非零计数
            count_nonzero_kernel_1[(mid_size, 1, 1)](x, mid, numel, block_size)

            # 第二阶段：汇总所有中间结果
            count_nonzero_kernel_2[(1, 1, 1)](mid, out, mid_size, block_mid)

        return out.to(torch.int32)
