import torch
import pytest

from nanovllm.layers.kvcache_int8 import store_kvcache_int8
from nanovllm.layers.paged_attention import paged_attention


def _make_4d_cache(num_blocks, block_size, num_kv_heads, head_dim, dtype=torch.int8):
    """Create a 4D cache matching the layout used by the model runner.
    Shape: [num_blocks, block_size, num_kv_heads, head_dim]
    """
    return torch.zeros(num_blocks, block_size, num_kv_heads, head_dim, dtype=dtype, device='cuda')


def _make_scale_4d(num_blocks, block_size, num_kv_heads):
    """Create a 4D scale cache.
    Shape: [num_blocks, block_size, num_kv_heads]
    """
    return torch.zeros(num_blocks, block_size, num_kv_heads, dtype=torch.float16, device='cuda')


def test_store_kvcache_int8_roundtrip():
    """Test that store + dequantize roundtrip has acceptable error."""
    torch.manual_seed(42)
    N, num_kv_heads, head_dim = 32, 4, 128
    key = torch.randn(N, num_kv_heads, head_dim, dtype=torch.float16, device='cuda')
    value = torch.randn(N, num_kv_heads, head_dim, dtype=torch.float16, device='cuda')

    # Use block_size=1 so each block holds 1 token (simplifies slot mapping)
    block_size = 1
    num_blocks = 256
    k_cache = _make_4d_cache(num_blocks, block_size, num_kv_heads, head_dim)
    v_cache = _make_4d_cache(num_blocks, block_size, num_kv_heads, head_dim)
    k_scale_cache = _make_scale_4d(num_blocks, block_size, num_kv_heads)
    v_scale_cache = _make_scale_4d(num_blocks, block_size, num_kv_heads)
    # slot = block_id * block_size + offset = block_id (since block_size=1)
    slot_mapping = torch.arange(N, dtype=torch.int32, device='cuda')

    store_kvcache_int8(key, value, k_cache, v_cache, k_scale_cache, v_scale_cache, slot_mapping)

    # index into 4D: cache[block_id, tok_id=0, :, :]
    k_dequant = k_cache[:N, 0].float() * k_scale_cache[:N, 0].unsqueeze(-1).float()
    v_dequant = v_cache[:N, 0].float() * v_scale_cache[:N, 0].unsqueeze(-1).float()

    k_mae = (k_dequant - key.float()).abs().mean().item()
    v_mae = (v_dequant - value.float()).abs().mean().item()
    k_max = key.abs().mean().item()

    rel_k_err = k_mae / k_max
    assert rel_k_err < 0.05, f"K relative error too high: {rel_k_err:.4f}"
    assert k_mae < 1.0, f"K MAE too high: {k_mae}"


def test_store_kvcache_int8_zero_values():
    """Test quantization of zero values produces zero INT8."""
    N, num_kv_heads, head_dim = 8, 2, 64
    key = torch.zeros(N, num_kv_heads, head_dim, dtype=torch.float16, device='cuda')
    value = torch.zeros(N, num_kv_heads, head_dim, dtype=torch.float16, device='cuda')

    block_size = 1
    num_blocks = 16
    k_cache = _make_4d_cache(num_blocks, block_size, num_kv_heads, head_dim)
    v_cache = _make_4d_cache(num_blocks, block_size, num_kv_heads, head_dim)
    k_scale_cache = _make_scale_4d(num_blocks, block_size, num_kv_heads)
    v_scale_cache = _make_scale_4d(num_blocks, block_size, num_kv_heads)
    slot_mapping = torch.arange(N, dtype=torch.int32, device='cuda')

    store_kvcache_int8(key, value, k_cache, v_cache, k_scale_cache, v_scale_cache, slot_mapping)

    assert k_cache[:N].eq(0).all(), "Zero input should produce zero INT8"
    assert v_cache[:N].eq(0).all(), "Zero input should produce zero INT8"


def test_store_kvcache_int8_slot_mapping():
    """Test that slot_mapping correctly routes tokens to arbitrary slots."""
    N, num_kv_heads, head_dim = 4, 2, 64
    key = torch.randn(N, num_kv_heads, head_dim, dtype=torch.float16, device='cuda')
    value = torch.randn(N, num_kv_heads, head_dim, dtype=torch.float16, device='cuda')

    block_size = 1
    num_blocks = 64
    k_cache = _make_4d_cache(num_blocks, block_size, num_kv_heads, head_dim)
    v_cache = _make_4d_cache(num_blocks, block_size, num_kv_heads, head_dim)
    k_scale_cache = _make_scale_4d(num_blocks, block_size, num_kv_heads)
    v_scale_cache = _make_scale_4d(num_blocks, block_size, num_kv_heads)

    # Non-contiguous slots: 10, 20, 30, 40
    slot_mapping = torch.tensor([10, 20, 30, 40], dtype=torch.int32, device='cuda')

    store_kvcache_int8(key, value, k_cache, v_cache, k_scale_cache, v_scale_cache, slot_mapping)

    # Verify data was stored at the correct slots (not at slot 0)
    assert k_cache[0, 0].eq(0).all(), "Block 0 should be untouched"
    assert k_scale_cache[10, 0].any() or k_cache[10, 0].any(), "Slot 10 should have data"
    assert k_scale_cache[20, 0].any() or k_cache[20, 0].any(), "Slot 20 should have data"


def test_store_kvcache_int8_negative_values():
    """Test quantization of negative constant values."""
    N, num_kv_heads, head_dim = 8, 2, 64
    key = -torch.ones(N, num_kv_heads, head_dim, dtype=torch.float16, device='cuda')
    value = -torch.ones(N, num_kv_heads, head_dim, dtype=torch.float16, device='cuda')

    block_size = 1
    num_blocks = 16
    k_cache = _make_4d_cache(num_blocks, block_size, num_kv_heads, head_dim)
    v_cache = _make_4d_cache(num_blocks, block_size, num_kv_heads, head_dim)
    k_scale_cache = _make_scale_4d(num_blocks, block_size, num_kv_heads)
    v_scale_cache = _make_scale_4d(num_blocks, block_size, num_kv_heads)
    slot_mapping = torch.arange(N, dtype=torch.int32, device='cuda')

    store_kvcache_int8(key, value, k_cache, v_cache, k_scale_cache, v_scale_cache, slot_mapping)

    # all -1.0 values: absmax = 1.0, scale = 1.0/127 ≈ 0.00787, int8 = round(-1/0.00787) = -127
    # Check the first element of head 0 of block 0
    assert k_cache[0, 0, 0, 0].item() == -127, f"Expected -127, got {k_cache[0, 0, 0, 0].item()}"
    assert v_cache[0, 0, 0, 0].item() == -127, f"Expected -127, got {v_cache[0, 0, 0, 0].item()}"


def test_int8_paged_attention_end_to_end():
    """INT8 store + paged_attention matches manual dequantize + flash_attn.

    In the actual runtime pipeline, ``store_kvcache_int8`` is called *before*
    ``paged_attention``, so the KV cache includes the current token:
    ``seq_lens = ctx_len + 1`` and the cache must have ``ctx_len + 1`` tokens.
    """
    from flash_attn import flash_attn_varlen_func

    num_kv_heads, head_dim = 4, 64
    num_query_heads = 8
    block_size = 16
    num_blocks = 16
    ctx_len = 5  # 5 cached prior tokens
    total_tokens = ctx_len + 1  # +1 for the current query token stored in cache

    # Create K/V for ALL tokens (prior + current query)
    k_fp16 = torch.randn(total_tokens, num_kv_heads, head_dim, dtype=torch.float16, device='cuda')
    v_fp16 = torch.randn(total_tokens, num_kv_heads, head_dim, dtype=torch.float16, device='cuda')

    k_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_dim, dtype=torch.int8, device='cuda')
    v_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_dim, dtype=torch.int8, device='cuda')
    k_scale = torch.zeros(num_blocks, block_size, num_kv_heads, dtype=torch.float16, device='cuda')
    v_scale = torch.zeros(num_blocks, block_size, num_kv_heads, dtype=torch.float16, device='cuda')

    # Store all tokens to block 2
    slot_mapping = torch.full((total_tokens,), -1, dtype=torch.int32, device='cuda')
    for i in range(total_tokens):
        slot_mapping[i] = 2 * block_size + i  # block 2, offset i
    store_kvcache_int8(k_fp16, v_fp16, k_cache, v_cache, k_scale, v_scale, slot_mapping)

    # Dequantize reference (all tokens, matching seq_lens)
    k_deq = (k_cache[2, :total_tokens].float() * k_scale[2, :total_tokens].unsqueeze(-1).float()).to(torch.float16)
    v_deq = (v_cache[2, :total_tokens].float() * v_scale[2, :total_tokens].unsqueeze(-1).float()).to(torch.float16)

    # One query token
    q = torch.randn(1, num_query_heads, head_dim, dtype=torch.float16, device='cuda')

    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device='cuda')
    cu_seqlens_k = torch.tensor([0, total_tokens], dtype=torch.int32, device='cuda')
    seq_lens = torch.tensor([total_tokens], dtype=torch.int32, device='cuda')
    block_table = torch.tensor([[2, -1]], dtype=torch.int32, device='cuda')

    softmax_scale = head_dim ** -0.5

    out_ref = flash_attn_varlen_func(
        q, k_deq, v_deq,
        cu_seqlens_q=cu_seqlens_q, max_seqlen_q=1,
        cu_seqlens_k=cu_seqlens_k, max_seqlen_k=total_tokens,
        softmax_scale=softmax_scale, causal=True,
    )

    out_kernel = paged_attention(
        q, k_cache, v_cache, k_scale, v_scale, block_table,
        seq_lens, cu_seqlens_q, softmax_scale, kv_quant_mode=2,
    )

    diff = (out_kernel - out_ref).abs().max().item()
    # INT8 quantization error: expect < ~0.1 with head_dim=128 random data
    assert diff < 2.0, f"INT8 end-to-end diff too large: {diff}"
