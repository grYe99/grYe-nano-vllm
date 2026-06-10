"""Unified paged attention kernel supporting FP16 and INT8 per-token-head KV cache.

Port of vLLM's ``kernel_unified_attention`` (2D mode only), stripped to essentials:
- Causal masking only
- No ALiBi, softcap, QQ bias, sinks, sliding window, mm_prefix
- No 3D segment parallelism (always 2D: one tile loop per program)
- KV_QUANT_MODE: 0 = FP16 (no quant), 2 = INT8 per-token-head
"""

import torch
import triton
import triton.language as tl


# ===========================================================================
# Helpers (ported from vllm/v1/attention/ops/triton_attention_helpers.py)
# ===========================================================================


@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def find_seq_idx(query_start_len_ptr, target_idx, num_seqs, BLOCK_Q: tl.constexpr):
    """Binary search: find which sequence ``target_idx`` belongs to."""
    left: tl.int32 = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = tl.load(query_start_len_ptr + mid)
        mid_val = val // BLOCK_Q + mid
        if mid_val <= target_idx:
            left = mid + 1
        else:
            right = mid
    return left - 1


@triton.jit
def resolve_seq_and_query_len(
    query_start_len_ptr, seq_lens_ptr, q_block_global_idx, num_seqs, BLOCK_Q: tl.constexpr,
):
    seq_idx = find_seq_idx(query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q)
    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx
    q_block_local_idx = q_block_global_idx - q_block_start_idx
    cur_start = tl.load(query_start_len_ptr + seq_idx)
    cur_stop = tl.load(query_start_len_ptr + seq_idx + 1)
    cur_batch_query_len = cur_stop - cur_start
    seq_len = tl.load(seq_lens_ptr + seq_idx)
    return seq_idx, q_block_local_idx, cur_start, cur_batch_query_len, seq_len


@triton.jit
def softmax_step(S, M, L):
    """Online softmax step. Returns (M_new, L_new, P, alpha)."""
    m_j = tl.maximum(M, tl.max(S, axis=1))
    m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
    P = tl.exp(S - m_j[:, None])
    l_j = tl.sum(P, axis=1)
    alpha = tl.exp(M - m_j)
    L_new = L * alpha + l_j
    return m_j, L_new, P, alpha


# ===========================================================================
# Main kernel
# ===========================================================================


@triton.jit
def paged_attention_kernel(
    # Output
    output_ptr,
    # Inputs
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    k_scale_cache_ptr,
    v_scale_cache_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    query_start_len_ptr,
    # Scalars
    scale,
    num_queries_per_kv: tl.constexpr,
    num_query_heads: tl.constexpr,
    block_table_stride: tl.int64,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,
    # K cache strides: [num_blocks, block_size, num_kv_heads, head_dim]
    stride_kb: tl.int64,  # stride(0) between physical blocks
    stride_ks: tl.int64,  # stride(1) between slots within a block
    stride_kh: tl.int64,  # stride(2) between kv heads
    stride_kd: tl.constexpr,  # stride(3) within head_dim (=1)
    # V cache strides
    stride_vb: tl.int64,
    stride_vs: tl.int64,
    stride_vh: tl.int64,
    stride_vd: tl.constexpr,
    # Scale cache strides: [num_blocks, block_size, num_kv_heads]
    stride_ksb: tl.int64,  # scale, stride(0) between blocks
    stride_kss: tl.int64,  # scale, stride(1) between slots
    stride_ksh: tl.int64,  # scale, stride(2) between heads
    stride_vsb: tl.int64,
    stride_vss: tl.int64,
    stride_vsh: tl.int64,
    # Constants
    KV_QUANT_MODE: tl.constexpr,  # 0 = none, 2 = int8 per-token-head
    BLOCK_SIZE: tl.constexpr,  # KV cache block size (e.g. 16)
    TILE_SIZE: tl.constexpr,  # attention tile size
    HEAD_SIZE: tl.constexpr,  # head dimension (e.g. 96)
    HEAD_SIZE_PADDED: tl.constexpr,  # next power of 2 (e.g. 128)
    BLOCK_Q: tl.constexpr,  # query tokens per block
    BLOCK_M: tl.constexpr,  # = BLOCK_Q * num_queries_per_kv
    num_seqs: tl.int32,
):
    USE_PER_TOKEN_HEAD_SCALES: tl.constexpr = KV_QUANT_MODE >= 2

    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    # === [paged] 序列解析：这个 q-block 属于哪个序列？ ===
    seq_idx, q_block_local_idx, cur_start, cur_batch_query_len, seq_len = (
        resolve_seq_and_query_len(
            query_start_len_ptr, seq_lens_ptr, q_block_global_idx, num_seqs, BLOCK_Q
        )
    )

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    # Offsets
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)

    # === [attention] 加载 Q 矩阵 [BLOCK_M, HEAD_SIZE_PADDED] ===
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_start + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv

    # Masks
    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    # Load Q: [BLOCK_M, HEAD_SIZE_PADDED]
    q_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )
    Q = tl.load(
        query_ptr + q_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    block_table_offset = seq_idx * block_table_stride
    context_len = seq_len - cur_batch_query_len

    # === [attention] online softmax 初始化 ===
    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    # === [attention] causal bound: 只迭代到最后一个 query 能 attend 到的 KV 位置 ===
    max_query_pos = (BLOCK_M - 1) // num_queries_per_kv
    max_seq_prefix_len = tl.minimum(seq_len, context_len + q_block_local_idx * BLOCK_Q + max_query_pos + 1)
    num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)

    for j in range(num_tiles):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len

        # === [paged] 通过 block_table 查物理地址 ===
        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
        ).to(tl.int64)

        # === [paged] 用物理地址加载 K tile [HEAD_SIZE_PADDED, TILE_SIZE] ===
        k_offset = (
            physical_block_idx[None, :] * stride_kb
            + kv_head_idx * stride_kh
            + offs_d[:, None] * stride_kd
            + (seq_offset % BLOCK_SIZE)[None, :] * stride_ks
        )
        K_load = tl.load(
            key_cache_ptr + k_offset,
            mask=dim_mask[:, None] & tile_mask[None, :],
            other=0.0,
        )

        # === [paged] 用物理地址加载 V tile [TILE_SIZE, HEAD_SIZE_PADDED] ===
        v_offset = (
            physical_block_idx[:, None] * stride_vb
            + kv_head_idx * stride_vh
            + offs_d[None, :] * stride_vd
            + (seq_offset % BLOCK_SIZE)[:, None] * stride_vs
        )
        V_load = tl.load(
            value_cache_ptr + v_offset,
            mask=dim_mask[None, :] & tile_mask[:, None],
            other=0.0,
        )

        # === [attention] K/V 转型 (INT8→FP16 纯 cast, scale 融合到 S/P) ===
        K = K_load.to(Q.dtype)
        V = V_load.to(Q.dtype)

        # === [paged] 加载 per-token-head scale（仅 INT8） ===
        if USE_PER_TOKEN_HEAD_SCALES:
            scale_idx = (
                physical_block_idx * stride_ksb
                + (seq_offset % BLOCK_SIZE) * stride_kss
                + kv_head_idx * stride_ksh
            )
            k_token_head_scales = tl.load(
                k_scale_cache_ptr + scale_idx, mask=tile_mask, other=1.0
            )
            v_scale_idx = (
                physical_block_idx * stride_vsb
                + (seq_offset % BLOCK_SIZE) * stride_vss
                + kv_head_idx * stride_vsh
            )
            v_token_head_scales = tl.load(
                v_scale_cache_ptr + v_scale_idx, mask=tile_mask, other=1.0
            )

        # === [attention] causal mask ===
        query_abs_pos = context_len + query_pos[:, None]
        seq_mask = seq_offset[None, :] <= query_abs_pos

        # === [attention] score: S = Q @ K * scale (+ per-token-head scale) ===
        S = tl.zeros([BLOCK_M, TILE_SIZE], dtype=tl.float32)
        if USE_PER_TOKEN_HEAD_SCALES:
            S += tl.dot(Q, K) * (scale * k_token_head_scales[None, :])
        else:
            S += scale * tl.dot(Q, K)

        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask,
            S,
            float("-inf"),
        )

        # === [attention] online softmax ===
        M, L, P, alpha = softmax_step(S, M, L)
        acc = acc * alpha[:, None]

        # === [attention] 加权求和 ===
        if USE_PER_TOKEN_HEAD_SCALES:
            P_scaled = (P * v_token_head_scales[None, :]).to(V.dtype)
            acc += tl.dot(P_scaled, V)
        else:
            acc += tl.dot(P.to(V.dtype), V)

    # === [attention] epilogue: 归一化 ===
    acc = acc / L[:, None]

    # Write output
    output_offset = (
        query_offset_0[:, None] * output_stride_0
        + query_offset_1[:, None] * output_stride_1
        + offs_d[None, :]
    )
    tl.store(
        output_ptr + output_offset,
        acc,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )


# ===========================================================================
# Python wrapper
# ===========================================================================


def paged_attention(
    q: torch.Tensor,           # [total_tokens, num_query_heads, head_size]
    k_cache: torch.Tensor,     # [num_blocks, block_size, num_kv_heads, head_size]
    v_cache: torch.Tensor,     # same
    k_scale_cache: torch.Tensor | None,  # [num_blocks, block_size, num_kv_heads] fp16, or None
    v_scale_cache: torch.Tensor | None,  # same
    block_table: torch.Tensor,  # [num_seqs, max_num_blocks] int32
    seq_lens: torch.Tensor,     # [num_seqs] int32 — total length of each seq (KV + query)
    cu_seqlens_q: torch.Tensor,  # [num_seqs + 1] int32 — cumulative query lengths
    softmax_scale: float,
    kv_quant_mode: int = 0,     # 0 = none, 2 = int8_per_token_head
    is_prefill: bool | None = None,  # None = auto-detect from cu_seqlens_q
) -> torch.Tensor:
    """Unified paged attention: single kernel for prefill and decode.

    Supports FP16 cache (``kv_quant_mode=0``) and INT8 per-token-head
    quantized cache (``kv_quant_mode=2``).
    """
    num_seqs = len(seq_lens)
    num_query_heads = q.shape[1]
    num_kv_heads = k_cache.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = q.shape[2]
    head_size_padded = triton.next_power_of_2(head_size)
    block_size = k_cache.shape[1]

    BLOCK_M = 16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    BLOCK_Q = BLOCK_M // num_queries_per_kv

    # Grid: enough q-blocks to cover all query tokens
    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs

    use_per_token_head_scales = kv_quant_mode >= 2
    if use_per_token_head_scales:
        assert k_scale_cache is not None and v_scale_cache is not None
        ks_strides = k_scale_cache.stride()
        vs_strides = v_scale_cache.stride()
    else:
        # Dummy strides — never dereferenced (DCE'd by constexpr)
        ks_strides = (0, 0, 0)
        vs_strides = (0, 0, 0)

    # Decode uses smaller tiles (16), prefill uses larger (32)
    # is_prefill is passed explicitly when called from attention layers to
    # avoid GPU tensor reads in CUDA graph capture. Fallback for standalone use.
    if is_prefill is None:
        is_prefill = cu_seqlens_q[-1] > num_seqs
    tile_size = 32 if is_prefill else 16

    output = torch.empty_like(q)

    grid = (total_num_q_blocks, num_kv_heads)
    paged_attention_kernel[grid](
        output_ptr=output,
        query_ptr=q,
        key_cache_ptr=k_cache,
        value_cache_ptr=v_cache,
        k_scale_cache_ptr=k_scale_cache if use_per_token_head_scales else k_cache,
        v_scale_cache_ptr=v_scale_cache if use_per_token_head_scales else v_cache,
        block_tables_ptr=block_table,
        seq_lens_ptr=seq_lens,
        query_start_len_ptr=cu_seqlens_q,
        scale=softmax_scale,
        num_queries_per_kv=num_queries_per_kv,
        num_query_heads=num_query_heads,
        block_table_stride=block_table.stride(0),
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        output_stride_0=output.stride(0),
        output_stride_1=output.stride(1),
        stride_kb=k_cache.stride(0),
        stride_ks=k_cache.stride(1),
        stride_kh=k_cache.stride(2),
        stride_kd=k_cache.stride(3),
        stride_vb=v_cache.stride(0),
        stride_vs=v_cache.stride(1),
        stride_vh=v_cache.stride(2),
        stride_vd=v_cache.stride(3),
        stride_ksb=ks_strides[0],
        stride_kss=ks_strides[1],
        stride_ksh=ks_strides[2],
        stride_vsb=vs_strides[0],
        stride_vss=vs_strides[1],
        stride_vsh=vs_strides[2],
        KV_QUANT_MODE=kv_quant_mode,
        BLOCK_SIZE=block_size,
        TILE_SIZE=tile_size,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=head_size_padded,
        BLOCK_Q=BLOCK_Q,
        BLOCK_M=BLOCK_M,
        num_seqs=num_seqs,
    )

    return output
