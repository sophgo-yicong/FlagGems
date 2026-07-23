import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def rope_kernel(
    X_ptr,
    Y_ptr,  # separate output pointer for out-of-place operation
    COS_ptr,
    SIN_ptr,
    head_dim,
    half_dim,
    seq_len,
    stride_x_seq,
    n_rows,
    cos_stride_seq,
    cos_stride_d,
    sin_stride_seq,
    sin_stride_d,
    TOTAL_ELEMS,
    BLOCK_SIZE: tl.constexpr,
):
    """Apply rotary position embedding out-of-place.

    For each 2D pair (d, d+half_dim):
      y_d          = x_d * cos[pos, d] - x_{d+half} * sin[pos, d]
      y_{d+half}   = x_{d+half} * cos[pos, d] + x_d * sin[pos, d]

    x: (n_rows, head_dim) f32/bf16 contiguous, where n_rows = batch*num_heads*seq_len
    y: (n_rows, head_dim) output — separate from x, never aliases input
    cos/sin: (seq_len, head_dim) — only seq_len affects which cos row is read.
    stride_x_seq gives the stride between consecutive (seq_len, head_dim) groups.

    Out-of-place avoids CMODEL DMA races between a PPL kernel write and
    subsequent torch-native reads: the input tensor is not modified.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < TOTAL_ELEMS

    # Logical position: row in [0, n_rows), col in [0, head_dim)
    row = offs // head_dim
    col = offs % head_dim

    is_first = col < half_dim
    half_col = tl.where(is_first, col, col - half_dim)

    # Position index: seq_pos = row % seq_len  (cos/sin are per seq position)
    seq_pos = row % seq_len

    cos_idx = seq_pos * cos_stride_seq + half_col * cos_stride_d
    sin_idx = seq_pos * sin_stride_seq + half_col * sin_stride_d

    c = tl.load(COS_ptr + cos_idx, mask=mask, other=1.0).to(tl.float32)
    s = tl.load(SIN_ptr + sin_idx, mask=mask, other=0.0).to(tl.float32)

    x_self = tl.load(X_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    pair_offs = tl.where(is_first, offs + half_dim, offs - half_dim)
    x_pair = tl.load(X_ptr + pair_offs, mask=mask, other=0.0).to(tl.float32)

    # rotate_half: [x1, x2] -> [-x2, x1]
    rotated = tl.where(is_first, -x_pair, x_pair)
    y = x_self * c + rotated * s

    # Store to output pointer — never aliases the input
    tl.store(Y_ptr + offs, y.to(Y_ptr.dtype.element_ty), mask=mask)


def apply_rotary_emb(x, cos, sin):
    """Apply rotary position embedding out-of-place.

    Reads from x, writes to a freshly-allocated output tensor.
    The input x is NOT modified (non-in-place), so there is no DMA hazard
    between the PPL kernel and subsequent torch-native operations on x.

    Args:
        x: [batch, num_heads, seq_len, head_dim] — f32 or bf16 contiguous
        cos: [seq_len, *broadcast*, head_dim] — cos table
        sin: [seq_len, *broadcast*, head_dim] — sin table

    Returns:
        A new tensor with the same shape/dtype as x, containing RoPE-transformed
        values.  x is unchanged.
    """
    logger.debug("GEMS ROPE FORWARD (out-of-place)")
    x = x.contiguous()
    head_dim = x.shape[-1]
    seq_len = x.shape[-2]
    assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"
    half_dim = head_dim // 2

    # Flatten batch, heads dims so x becomes (batch*num_heads*seq_len, head_dim)
    n_rows = x.numel() // head_dim
    x_view = x.view(n_rows, head_dim)

    # Allocate fresh output tensor — never aliases the input
    y = torch.empty_like(x_view)
    y_view = y.view(n_rows, head_dim)

    # Reshape cos/sin to (seq_len, head_dim) — one cos row per seq position
    cos = cos.contiguous()
    sin = sin.contiguous()
    cos_2d = cos.view(seq_len, head_dim)
    sin_2d = sin.view(seq_len, head_dim)

    total_elems = x.numel()
    BLOCK_SIZE = 256

    grid = lambda meta: (triton.cdiv(total_elems, meta["BLOCK_SIZE"]),)

    with torch_device_fn.device(x.device):
        rope_kernel[grid](
            x_view, y_view,
            cos_2d, sin_2d,
            head_dim, half_dim,
            seq_len,
            x.stride(1),  # stride between rows in x_view
            n_rows,
            cos_2d.stride(0), cos_2d.stride(1),
            sin_2d.stride(0), sin_2d.stride(1),
            total_elems,
            BLOCK_SIZE,
        )
    # Restore original shape: y was flattened to (n_rows, head_dim) for
    # the kernel; reshape back to the caller's 4D layout.
    return y.view(x.shape)


@triton.jit
def fused_rope_scatter_k_kernel(
    KEY_ptr,        # pre-RoPE key, contiguous [M, nkv, hd] -> flat [M*nkv, hd]
    CACHE_ptr,      # kv_cache_k flat: [num_slots, nkv, hd] -> flat [num_slots*nkv*hd]
    COS_ptr,
    SIN_ptr,
    SLOT_ptr,       # slot_mapping [M], int32 (flat slot per token)
    head_dim,
    half_dim,
    seq_len,
    num_kv_heads,
    cos_stride_seq,
    cos_stride_d,
    sin_stride_seq,
    sin_stride_d,
    TOTAL_ELEMS,
    BLOCK_SIZE: tl.constexpr,
):
    """Apply RoPE to K and scatter the result directly into the paged KV cache.

    Reads the *pre-RoPE* key (a plain torch buffer, NOT a PPL-kernel output),
    applies rotary embedding, and stores the result DIRECTLY into the paged K
    cache at slot_mapping[token]. This fuses what would otherwise be two steps
    (apply_rotary_emb -> a torch-native transpose/reshape scatter). Fusing
    eliminates the torch-native strided read of the RoPE PPL-kernel output
    buffer (`k.transpose(1,2).contiguous()`), which on the sophgo CMODEL
    emulator non-deterministically corrupts the K cache (write-time region
    stomp): with the fused kernel the K cache is bit-exact every run.

    Layouts:
      key: contiguous [M, num_kv_heads, head_dim] -> flat [M*nkv, hd]
        row       = flat // head_dim   (index into [M*nkv, hd])
        col       = flat %  head_dim
        token_idx = row  // num_kv_heads   in [0, M)
        kv_head   = row  %  num_kv_heads
        seq_pos   = token_idx % seq_len    (which cos/sin row to read)
      cache: [num_slots, num_kv_heads, head_dim] flat, written at
        CACHE[slot*nkv*hd + kv_head*hd + col] = rope(key)[flat],
        where slot = slot_mapping[token_idx].

    The RoPE math is identical to rope_kernel above (2D-pair rotate_half).

    All index arithmetic is int32: the sophgo TPU backend has limited int64
    support, and the flat cache offset (num_slots*nkv*hd) fits in int32.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < TOTAL_ELEMS

    row = offs // head_dim
    col = offs % head_dim
    token_idx = row // num_kv_heads
    kv_head = row % num_kv_heads
    seq_pos = token_idx % seq_len

    is_first = col < half_dim
    half_col = tl.where(is_first, col, col - half_dim)

    cos_idx = seq_pos * cos_stride_seq + half_col * cos_stride_d
    sin_idx = seq_pos * sin_stride_seq + half_col * sin_stride_d
    c = tl.load(COS_ptr + cos_idx, mask=mask, other=1.0).to(tl.float32)
    s = tl.load(SIN_ptr + sin_idx, mask=mask, other=0.0).to(tl.float32)

    x_self = tl.load(KEY_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    pair_offs = tl.where(is_first, offs + half_dim, offs - half_dim)
    x_pair = tl.load(KEY_ptr + pair_offs, mask=mask, other=0.0).to(tl.float32)

    # rotate_half: [x1, x2] -> [-x2, x1]
    rotated = tl.where(is_first, -x_pair, x_pair)
    y = x_self * c + rotated * s

    # Scatter directly into the paged cache. slot is data-dependent, so this
    # is a true scatter store (the only op that ever writes the K cache).
    slot = tl.load(SLOT_ptr + token_idx, mask=mask, other=0).to(tl.int32)
    out_offs = (slot * (num_kv_heads * head_dim)
                + kv_head * head_dim
                + col)
    tl.store(CACHE_ptr + out_offs, y.to(CACHE_ptr.dtype.element_ty), mask=mask)


def fused_rope_scatter_k(key, cos, sin, slot_mapping, kv_cache_k,
                         seq_len, num_kv_heads, head_dim):
    """Apply RoPE to K and write it into the paged KV cache in one kernel.

    This replaces the (apply_rotary_emb -> torch transpose/reshape scatter)
    pair for the K-cache write. Reads the *pre-RoPE* key, so nothing on this
    path ever does a torch-native read of a PPL-kernel output buffer.

    Args:
        key: pre-RoPE key, [M, num_kv_heads, head_dim] (plain torch buffer).
        cos: RoPE cos table, numel == seq_len * head_dim.
        sin: RoPE sin table, numel == seq_len * head_dim.
        slot_mapping: [M] flat slot index per token (int).
        kv_cache_k: paged cache [num_blocks, block_size, num_kv_heads, head_dim].
        seq_len: number of sequence positions (M == batch * seq_len).
        num_kv_heads: number of KV heads.
        head_dim: head dimension (must be even).

    Returns:
        None. kv_cache_k is modified in place at the given slots.
    """
    logger.debug("GEMS FUSED ROPE+SCATTER K")
    assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"
    key_c = key.contiguous()
    half_dim = head_dim // 2
    total_elems = key_c.numel()

    cos_2d = cos.contiguous().view(seq_len, head_dim)
    sin_2d = sin.contiguous().view(seq_len, head_dim)
    cache_flat = kv_cache_k.reshape(-1)
    slot = slot_mapping.to(torch.int32)

    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(total_elems, meta["BLOCK_SIZE"]),)

    with torch_device_fn.device(key_c.device):
        fused_rope_scatter_k_kernel[grid](
            key_c, cache_flat,
            cos_2d, sin_2d, slot,
            head_dim, half_dim, seq_len, num_kv_heads,
            cos_2d.stride(0), cos_2d.stride(1),
            sin_2d.stride(0), sin_2d.stride(1),
            total_elems,
            BLOCK_SIZE,
        )
