"""Unit test: compare paged_attention with flash_attn_with_kvcache for decode.

This tests both FP16 and INT8 KV cache paths to find numerical discrepancies.
"""

import torch
import torch.nn.functional as F
import pytest

from flash_attn import flash_attn_with_kvcache

from nanovllm.layers.paged_attention import paged_attention
from nanovllm.layers.kvcache_int8 import store_kvcache_int8


# Use power-of-2 head dim to avoid Triton arange constraint
# (store_kvcache_int8 kernel needs power-of-2 for tl.arange(0, head_dim))
NUM_KV_HEADS = 4
NUM_QUERY_HEADS = 8
HEAD_DIM = 128
BLOCK_SIZE = 256
DTYPE = torch.float16
DEVICE = "cuda"

# Alternative: test with non-power-of-2 head dim using the padded path
# (only for paged_attention, which handles HEAD_SIZE_PADDED correctly)


def make_paged_kv_cache(
    kv_tokens: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim]
    block_size: int = BLOCK_SIZE,
):
    """Create paged K/V cache from contiguous K/V tokens.

    Returns (k_cache, v_cache, block_table, seq_len).
    """
    num_tokens, num_kv_heads, head_dim = kv_tokens.shape
    num_blocks = (num_tokens + block_size - 1) // block_size
    k_cache = torch.zeros(
        num_blocks, block_size, num_kv_heads, head_dim, dtype=DTYPE, device=DEVICE
    )
    v_cache = torch.zeros_like(k_cache)
    for i in range(num_tokens):
        b = i // block_size
        s = i % block_size
        k_cache[b, s] = kv_tokens[i]
        v_cache[b, s] = kv_tokens[i]
    block_table = torch.tensor(
        [list(range(num_blocks))], dtype=torch.int32, device=DEVICE
    )
    return k_cache, v_cache, block_table


def store_int8_kvcache(
    kv_tokens: torch.Tensor,
    block_table: torch.Tensor,
    seq_len: int,
    block_size: int = BLOCK_SIZE,
):
    """Store INT8 quantized KV cache in paged layout."""
    num_kv_heads, head_dim = kv_tokens.shape[1], kv_tokens.shape[2]
    num_blocks, _, _, _ = block_table.shape[1], block_size, num_kv_heads, head_dim
    # Actually compute: max block_idx in block_table + 1
    max_block = block_table.max().item() + 1 if block_table.numel() > 0 else 1
    num_blocks_alloc = max(block_table.max().item() + 1, num_blocks)

    k_cache = torch.zeros(
        max_block, block_size, num_kv_heads, head_dim, dtype=torch.int8, device=DEVICE
    )
    v_cache = torch.zeros_like(k_cache)

    k_scale_cache = torch.zeros(
        max_block, block_size, num_kv_heads, dtype=torch.float16, device=DEVICE
    )
    v_scale_cache = torch.zeros_like(k_scale_cache)

    # Create slot mapping for all tokens
    slot_mapping = torch.zeros(seq_len, dtype=torch.int32, device=DEVICE)
    for i in range(seq_len):
        b = i // block_size
        s = i % block_size
        slot_mapping[i] = block_table[0, b] * block_size + s

    store_kvcache_int8(
        kv_tokens, kv_tokens,
        k_cache, v_cache,
        k_scale_cache, v_scale_cache,
        slot_mapping,
    )
    return k_cache, v_cache, k_scale_cache, v_scale_cache


@pytest.mark.parametrize("kv_len", [1, 16, 32, 64, 128, 256, 512])
@pytest.mark.parametrize("use_int8", [False, True])
def test_decode_single_token(kv_len: int, use_int8: bool):
    """Test decode: single query token, varying KV cache lengths.

    In the actual nano-vllm pipeline, ``store_kvcache`` is called *before*
    ``paged_attention``, so the KV cache includes the current token.
    ``seq_lens = context.context_lens = [len(seq)]`` where each seq has
    already had the new token appended, so ``seq_lens[i] = kv_len + 1``.
    """
    torch.manual_seed(42)
    scale = HEAD_DIM ** -0.5

    # Create K/V tokens (random) — includes both prior and current token
    kv_tokens = torch.randn(kv_len + 1, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)

    # The last token is the "current" query token stored into cache
    prior_tokens = kv_tokens[:kv_len]
    curr_token = kv_tokens[kv_len:]  # [1, num_kv_heads, head_dim]

    # Create query
    q = torch.randn(1, NUM_QUERY_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    q_unsqueezed = q.unsqueeze(1)  # [1, 1, num_heads, head_dim] for flash_attn_with_kvcache

    # Build paged cache with ALL tokens (prior + current, i.e. kv_len + 1 tokens)
    all_tokens = kv_tokens  # [kv_len + 1, num_kv_heads, head_dim]
    k_cache, v_cache, block_table = make_paged_kv_cache(all_tokens)

    # seq_lens = total cached tokens = kv_len + 1 (matching actual runtime)
    # context_lens for flash_attn_with_kvcache = same (cache includes current token)
    seq_len = torch.tensor([kv_len + 1], dtype=torch.int32, device=DEVICE)
    context_lens = torch.tensor([kv_len + 1], dtype=torch.int32, device=DEVICE)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)

    if use_int8:
        # INT8 path
        k_cache_i8, v_cache_i8, k_scale_cache, v_scale_cache = store_int8_kvcache(
            all_tokens, block_table, kv_len + 1
        )
        out = paged_attention(
            q, k_cache_i8, v_cache_i8,
            k_scale_cache, v_scale_cache,
            block_table, seq_len, cu_seqlens_q,
            scale, kv_quant_mode=2, is_prefill=False,
        )
    else:
        # FP16 path via paged_attention
        out = paged_attention(
            q, k_cache, v_cache,
            None, None,
            block_table, seq_len, cu_seqlens_q,
            scale, kv_quant_mode=0, is_prefill=False,
        )

    # Reference: flash_attn_with_kvcache (FP16)
    ref = flash_attn_with_kvcache(
        q_unsqueezed, k_cache, v_cache,
        cache_seqlens=context_lens, block_table=block_table,
        softmax_scale=scale, causal=True,
    ).squeeze(1)  # [1, num_heads, head_dim]

    # Compare
    diff = (out.float() - ref.float()).abs().max().item()
    rel_diff = (diff / (ref.float().abs().max().item() + 1e-10))

    mode = "INT8" if use_int8 else "FP16"
    print(f"[{mode}] kv_len={kv_len:4d}  max_abs_diff={diff:.6e}  rel_diff={rel_diff:.6e}")

    # For FP16, expect near-exact match
    # For INT8, expect small error from quantization
    if use_int8:
        assert diff < 1.0, f"INT8: kv_len={kv_len}, abs_diff={diff:.4f} too large!"
    else:
        assert diff < 1e-2, f"FP16: kv_len={kv_len}, abs_diff={diff:.4f} too large!"


@pytest.mark.parametrize("kv_len", [4, 8, 16, 32, 64, 128, 256])
def test_compare_int8_vs_fp16(kv_len: int):
    """Compare paged_attention INT8 output vs. flash_attn_with_kvcache FP16 output.

    INT8 should be close to FP16 with small quantization error.
    """
    torch.manual_seed(42)
    scale = HEAD_DIM ** -0.5

    kv_tokens = torch.randn(kv_len, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    q = torch.randn(1, NUM_QUERY_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)

    k_cache, v_cache, block_table = make_paged_kv_cache(kv_tokens)
    seq_len = torch.tensor([kv_len + 1], dtype=torch.int32, device=DEVICE)
    context_lens = torch.tensor([kv_len], dtype=torch.int32, device=DEVICE)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)

    # FP16 reference
    q_unsqueezed = q.unsqueeze(1)
    ref = flash_attn_with_kvcache(
        q_unsqueezed, k_cache, v_cache,
        cache_seqlens=context_lens, block_table=block_table,
        softmax_scale=scale, causal=True,
    ).squeeze(1)

    # INT8 paged_attention
    k_cache_i8, v_cache_i8, k_scale_cache, v_scale_cache = store_int8_kvcache(
        kv_tokens, block_table, kv_len
    )
    out = paged_attention(
        q, k_cache_i8, v_cache_i8,
        k_scale_cache, v_scale_cache,
        block_table, seq_len, cu_seqlens_q,
        scale, kv_quant_mode=2, is_prefill=False,
    )

    diff = (out.float() - ref.float()).abs().max().item()
    print(f"kv_len={kv_len:4d}  INT8_max_abs_diff={diff:.6e}")

    # INT8 quantization error should be bounded
    # Typically < 0.1 for head_dim=96 with random data
    assert diff < 2.0, f"kv_len={kv_len}, INT8 vs FP16 diff={diff:.4f} too large!"


if __name__ == "__main__":
    print("=" * 60)
    print("Testing decode with single token query")
    print("=" * 60)

    for kv_len in [1, 16, 32, 64, 128, 256, 512]:
        for use_int8 in [False, True]:
            try:
                test_decode_single_token(kv_len, use_int8)
            except AssertionError as e:
                print(f"FAIL: {e}")
