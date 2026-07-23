import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


# Sophgo TPU flash-attention kernel.
#
# One kernel invocation handles ONE (batch, head, M-block) triple.
# Grid = (1, batch*heads*num_m_blocks, 1).
# The Python launcher loops nothing; the grid dim 1 covers everything.
#
# PPL compiler constraints:
# - No two calls to inner function (crashes with "Malformed type storage")
# - No if/else on constexpr with tl.dot in both branches (same crash)
# - No constexpr-dependent loop unrolling with multiple dots
# → Single unified code path with causal_bound runtime trick.


@triton.jit
def _soph_attn_fwd(
    Q, K, V,
    sm_scale,
    Out,
    stride_q_outer,      # stride for Q along the (batch*q_head) dim
    stride_q_seqlen,     # stride for Q along seq_len dim (= HEAD_DIM)
    stride_kv_outer,     # stride for K/V along the (batch*kv_head) dim
    stride_kv_seqlen,    # stride for K/V along seq_len dim (= HEAD_DIM)
    stride_o_outer,
    stride_o_seqlen,
    kv_numhead_ratio,    # q_numhead // kv_numhead
    Q_CTX, KV_CTX,
    CAUSAL_BOUND,
    num_m_blocks,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
):
    # grid = (1, batch * q_numhead * num_m_blocks, 1)
    pid = tl.program_id(1)
    start_m = pid % num_m_blocks
    off_hz = pid // num_m_blocks

    # Map q head to kv head for GQA
    kv_hz = off_hz // kv_numhead_ratio if kv_numhead_ratio > 1 else off_hz

    q_base = off_hz * stride_q_outer
    kv_base = kv_hz * stride_kv_outer
    o_base = off_hz * stride_o_outer

    offs_headsize = tl.arange(0, HEAD_DIM)
    offs_n = tl.arange(0, BLOCK_N)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    q_load_mask = offs_m < Q_CTX

    Q_block_ptr = Q + q_base + offs_m[:, None] * stride_q_seqlen + offs_headsize[None, :]
    O_block_ptr = Out + o_base + offs_m[:, None] * stride_o_seqlen + offs_headsize[None, :]

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    qk_scale = sm_scale
    q = tl.load(Q_block_ptr, mask=q_load_mask[:, None], other=0.0).to(tl.float32)

    # K layout: [HD, BLOCK_N] (transposed for dot)
    K_block_ptr = K + kv_base + offs_n[None, :] * stride_kv_seqlen + offs_headsize[:, None]
    V_block_ptr = V + kv_base + offs_n[:, None] * stride_kv_seqlen + offs_headsize[None, :]

    for start_n in range(0, KV_CTX, BLOCK_N):
        kv_load_mask = (start_n + offs_n) < KV_CTX

        k = tl.load(K_block_ptr, mask=kv_load_mask[None, :], other=0.0).to(tl.float32)
        if PRE_LOAD_V:
            v = tl.load(V_block_ptr, mask=kv_load_mask[:, None], other=0.0).to(tl.float32)

        qk = tl.dot(q, k)
        qk = tl.where(kv_load_mask[None, :], qk, -float("inf"))

        # Unified causal mask:
        # causal:     CAUSAL_BOUND = Q_CTX → offs_m >= start_n + offs_n
        # non-causal: CAUSAL_BOUND = KV_CTX + Q_CTX → always true
        causal_mask = (offs_m[:, None] + CAUSAL_BOUND) >= (start_n + offs_n[None, :] + Q_CTX)
        qk = qk * qk_scale
        qk = tl.where(causal_mask, qk, -float("inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]

        p = tl.exp(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij

        if not PRE_LOAD_V:
            v = tl.load(V_block_ptr, mask=kv_load_mask[:, None], other=0.0).to(tl.float32)

        # Separate scaled_acc and dot to prevent PPL fusion bug
        scaled_acc = acc * alpha[:, None]
        delta = tl.dot(p.to(v.dtype), v, allow_tf32=False)
        zero_barrier = tl.math.erf(tl.zeros(delta.shape, dtype=tl.float32))
        acc = scaled_acc + zero_barrier + delta
        m_i = m_ij

        K_block_ptr += BLOCK_N * stride_kv_seqlen
        V_block_ptr += BLOCK_N * stride_kv_seqlen

    result = acc / l_i[:, None]
    tl.store(O_block_ptr, result.to(Out.dtype.element_ty), mask=q_load_mask[:, None])


def scaled_dot_product_attention(
    query, key, value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
):
    """FlashAttention2-style scaled dot-product attention for sophgo TPU.

    Args:
        query: [batch, num_heads, seq_len, head_dim]
        key: [batch, num_kv_heads, kv_len, head_dim]
        value: [batch, num_kv_heads, kv_len, head_dim]
        is_causal: apply causal masking
        scale: softmax scale, defaults to 1/sqrt(head_dim)

    Returns:
        [batch, num_heads, seq_len, head_dim]
    """
    logger.debug("GEMS SOPHGO SDPA FORWARD")
    HEAD_DIM_Q, HEAD_DIM_K = query.shape[-1], key.shape[-1]
    HEAD_DIM_V = value.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K == HEAD_DIM_V
    assert dropout_p == 0.0, "Dropout not supported yet"
    assert attn_mask is None, "Attention mask not supported yet"

    input_dtype = query.dtype
    sm_scale = scale if scale is not None else 1.0 / (HEAD_DIM_K ** 0.5)

    batch, q_numhead, Q_CTX, _ = query.shape
    _, kv_numhead, KV_CTX, _ = key.shape

    o = torch.empty_like(query, dtype=value.dtype)

    BLOCK_M, BLOCK_N = 16, 16
    num_m_blocks = triton.cdiv(Q_CTX, BLOCK_M)
    kv_numhead_ratio = q_numhead // kv_numhead

    if is_causal:
        causal_bound = Q_CTX
    else:
        causal_bound = KV_CTX + Q_CTX

    # Reshape to [batch*heads, seq_len, head_dim] for simplified strides.
    # Q and K/V may have different seq_len (decode: Q_CTX=1, KV_CTX=N),
    # so they get separate stride parameters.
    q_3d = query.reshape(batch * q_numhead, Q_CTX, HEAD_DIM_K).contiguous()
    k_3d = key.reshape(batch * kv_numhead, KV_CTX, HEAD_DIM_K).contiguous()
    v_3d = value.reshape(batch * kv_numhead, KV_CTX, HEAD_DIM_V).contiguous()
    o_3d = o.reshape(batch * q_numhead, Q_CTX, HEAD_DIM_K)

    # grid = (1, batch * q_numhead * num_m_blocks, 1)
    total_programs = batch * q_numhead * num_m_blocks
    grid = (1, total_programs, 1)

    PRE_LOAD_V = False

    with torch_device_fn.device(query.device):
        _soph_attn_fwd[grid](
            q_3d, k_3d, v_3d,
            sm_scale, o_3d,
            q_3d.stride(0), q_3d.stride(1),
            k_3d.stride(0), k_3d.stride(1),
            o_3d.stride(0), o_3d.stride(1),
            kv_numhead_ratio,
            Q_CTX, KV_CTX,
            causal_bound,
            num_m_blocks,
            HEAD_DIM_K,
            BLOCK_M, BLOCK_N,
            PRE_LOAD_V=PRE_LOAD_V,
        )

    o = o_3d.reshape(batch, q_numhead, Q_CTX, HEAD_DIM_K)
    if o.dtype != input_dtype:
        o = o.to(input_dtype)
    return o
