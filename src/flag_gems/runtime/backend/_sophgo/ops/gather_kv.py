"""Paged-KV gather for decode, fused into a single triton kernel.

Companion to fused_rope_scatter_k (rope.py): that op scatters post-RoPE K
directly into the paged cache; this op gathers K/V back out of the paged
cache for decode attention.

Why this exists: on the sophgo CMODEL emulator, the torch-native gather
(`kv_cache_k[:num_blocks].clone()` + `torch.tpu.synchronize()`, or
`torch.index_select` over the live paged cache) intermittently HARD-ABORTS
the process (exit 255, no Python traceback) at the clone DMA once emulator
state accumulates. Reading the cache inside a PPL kernel — the same device
queue that wrote it via fused_rope_scatter_k — avoids the torch DMA read of
the live paged cache entirely, and also replaces the whole per-batch
slice/clone/reshape/pad/cat/stack/permute op chain (each a separate DMA
hazard) with one kernel launch per tensor.

Semantics match `_gather_kv_for_decode` exactly: output is zero-padded to
max_ctx per batch.
"""
import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def gather_kv_paged_kernel(
    CACHE_ptr,      # paged cache flat: [num_blocks*block_size, nkv, hd]
    BT_ptr,         # block_tables [B, max_num_blocks], int32
    CTX_ptr,        # total context length per batch [B], int32
    OUT_ptr,        # output [B, nkv, max_ctx, hd], zero-initialized
    num_kv_heads,
    head_dim,
    block_size,
    max_num_blocks,
    max_ctx,
    TOTAL_ELEMS,
    BLOCK_SIZE: tl.constexpr,
):
    """Gather one paged K/V cache into a dense [B, nkv, max_ctx, hd] tensor.

    Output offset decomposition (out layout [B, nkv, max_ctx, hd]):
      col     = offs %  head_dim
      t1      = offs // head_dim                        # = (b*nkv + h)*max_ctx + pos
      pos     = t1 %  max_ctx
      t2      = t1 // max_ctx                           # = b*nkv + h
      kv_head = t2 %  num_kv_heads
      b       = t2 // num_kv_heads

    Input slot for a (b, pos):
      block_id = block_tables[b, pos // block_size]
      slot     = block_id * block_size + pos % block_size
      in_offs  = slot * (num_kv_heads*head_dim) + kv_head*head_dim + col

    Positions with pos >= ctx_len[b] read nothing and store 0 (matches the
    zero-padding of the torch path).

    All index arithmetic is int32: the sophgo TPU backend has broken int64,
    and the flat cache offset (num_slots*nkv*hd) fits in int32.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < TOTAL_ELEMS

    col = offs % head_dim
    t1 = offs // head_dim
    pos = t1 % max_ctx
    t2 = t1 // max_ctx
    kv_head = t2 % num_kv_heads
    b = t2 // num_kv_heads

    ctx_len = tl.load(CTX_ptr + b, mask=mask, other=0).to(tl.int32)
    valid = mask & (pos < ctx_len)

    block_idx = pos // block_size
    block_id = tl.load(BT_ptr + b * max_num_blocks + block_idx,
                       mask=valid, other=0).to(tl.int32)
    slot = block_id * block_size + (pos % block_size)
    in_offs = (slot * (num_kv_heads * head_dim)
               + kv_head * head_dim
               + col)
    val = tl.load(CACHE_ptr + in_offs, mask=valid, other=0.0)
    tl.store(OUT_ptr + offs, val, mask=mask)


def gather_kv_paged(kv_cache_kv, block_tables, total_context,
                    num_kv_heads, head_dim):
    """Gather K or V from a paged cache into dense [B, nkv, max_ctx, hd].

    Args:
        kv_cache_kv: paged cache for K or V,
            [num_blocks, block_size, num_kv_heads, head_dim].
        block_tables: [B, max_num_blocks] int32 block ids.
        total_context: [B] int32 total context length (cache_len + input_len)
            per sequence.
        num_kv_heads: number of KV heads.
        head_dim: head dimension.

    Returns:
        Dense tensor [B, num_kv_heads, max_ctx, head_dim] on the same device,
        zero-padded past each sequence's context length.
    """
    logger.debug("GEMS GATHER KV PAGED")
    assert kv_cache_kv.is_contiguous(), "paged cache must be contiguous"
    block_size = kv_cache_kv.shape[1]
    B, max_num_blocks = block_tables.shape
    max_ctx = int(total_context.max().item())

    out = torch.zeros((B, num_kv_heads, max_ctx, head_dim),
                      dtype=kv_cache_kv.dtype, device=kv_cache_kv.device)
    total_elems = out.numel()

    bt_i32 = block_tables.contiguous().to(torch.int32)
    ctx_i32 = total_context.contiguous().to(torch.int32)
    cache_flat = kv_cache_kv.reshape(-1)

    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(total_elems, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(kv_cache_kv.device):
        gather_kv_paged_kernel[grid](
            cache_flat, bt_i32, ctx_i32, out,
            num_kv_heads, head_dim, block_size, max_num_blocks, max_ctx,
            total_elems,
            BLOCK_SIZE,
        )
    return out
