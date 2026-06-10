import torch
import triton
import triton.language as tl


@triton.jit
def store_kvcache_int8_kernel(
    key_ptr, key_stride_tok, key_stride_head,
    value_ptr, value_stride_tok, value_stride_head,
    k_cache_ptr, k_cache_slot_stride, k_cache_head_stride,
    v_cache_ptr, v_cache_slot_stride, v_cache_head_stride,
    k_scale_ptr, k_scale_stride,
    v_scale_ptr, v_scale_stride,
    slot_mapping_ptr,
    head_dim: tl.constexpr,
    HEAD_DIM_PADDED: tl.constexpr,
):
    """
    One program per (token, head).

    Quantizes K/V from FP16 to INT8 using symmetric per-token-head quantization.
    scale = max(|val|) / 127.0

    ``HEAD_DIM_PADDED`` must be the next power of 2 >= head_dim (Triton requires
    power-of-2 for ``tl.arange``). The extra lanes are masked out.
    """
    idx = tl.program_id(0)  # token index
    head_id = tl.program_id(1)  # head index

    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return

    offs_d = tl.arange(0, HEAD_DIM_PADDED)
    dim_mask = offs_d < head_dim

    # Load K: head_dim FP16 elements
    k_offsets = idx * key_stride_tok + head_id * key_stride_head + offs_d
    k_fp16 = tl.load(key_ptr + k_offsets, mask=dim_mask, other=0.0)

    # Load V
    v_offsets = idx * value_stride_tok + head_id * value_stride_head + offs_d
    v_fp16 = tl.load(value_ptr + v_offsets, mask=dim_mask, other=0.0)

    # Symmetric quantization: scale = max(|val|) / 127.0
    k_absmax = tl.max(tl.abs(k_fp16), axis=0)
    k_scale = tl.where(k_absmax > 0, k_absmax / 127.0, 1.0)
    k_float = k_fp16 / k_scale
    k_float = tl.clamp(k_float, -128.0, 127.0)
    k_int8 = tl.cast(k_float, tl.int8)

    v_absmax = tl.max(tl.abs(v_fp16), axis=0)
    v_scale = tl.where(v_absmax > 0, v_absmax / 127.0, 1.0)
    v_float = v_fp16 / v_scale
    v_float = tl.clamp(v_float, -128.0, 127.0)
    v_int8 = tl.cast(v_float, tl.int8)

    # Store INT8 data
    k_cache_offsets = slot * k_cache_slot_stride + head_id * k_cache_head_stride + offs_d
    v_cache_offsets = slot * v_cache_slot_stride + head_id * v_cache_head_stride + offs_d
    tl.store(k_cache_ptr + k_cache_offsets, k_int8, mask=dim_mask)
    tl.store(v_cache_ptr + v_cache_offsets, v_int8, mask=dim_mask)

    # Store FP16 scales
    tl.store(k_scale_ptr + slot * k_scale_stride + head_id, k_scale)
    tl.store(v_scale_ptr + slot * v_scale_stride + head_id, v_scale)


def store_kvcache_int8(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    v_scale_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    N, num_kv_heads, head_dim = key.shape
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert k_cache.dtype == torch.int8 and v_cache.dtype == torch.int8
    assert k_scale_cache.dtype == torch.float16 and v_scale_cache.dtype == torch.float16
    assert k_cache.shape[-2] == num_kv_heads and k_cache.shape[-1] == head_dim
    assert slot_mapping.numel() == N

    # Cache shape is [num_blocks, block_size, num_kv_heads, head_dim]
    # stride(1) = num_kv_heads * head_dim (token stride within a block)
    # stride(2) = head_dim (head stride)
    # slot = block_id * block_size + tok_id
    # offset = slot * stride(1) + head_id * stride(2) + dim_offset
    head_dim_padded = triton.next_power_of_2(head_dim)
    grid = (N, num_kv_heads)
    store_kvcache_int8_kernel[grid](
        key, key.stride(0), key.stride(1),
        value, value.stride(0), value.stride(1),
        k_cache, k_cache.stride(1), k_cache.stride(2),
        v_cache, v_cache.stride(1), v_cache.stride(2),
        k_scale_cache, k_scale_cache.stride(1),
        v_scale_cache, v_scale_cache.stride(1),
        slot_mapping,
        head_dim,
        head_dim_padded,
    )
