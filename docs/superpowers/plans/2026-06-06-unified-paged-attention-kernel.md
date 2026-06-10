# Unified Paged Attention Kernel (INT8 + FP16) Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan.

**Goal:** Replace the Python-level gather+dequantize + `flash_attn_varlen_func` approach with a single unified Triton paged attention kernel that handles INT8 dequantization inline.

**Architecture:** Port a simplified version of vLLM's `kernel_unified_attention` (2D mode only, no 3D/segments) that loads INT8 K/V cache tiles via `block_table`, applies per-token-head scales fused into the attention score computation, and handles both prefill and decode with a single kernel path.

**Tech Stack:** Triton, PyTorch

---

### File Structure

| File | Responsibility |
|------|---------------|
| `nanovllm/layers/paged_attention.py` (NEW) | Unified Triton paged attention kernel + Python wrapper |
| `nanovllm/layers/kvcache_int8.py` (unchanged) | Store kernel — already correct |
| `nanovllm/layers/attention.py` (MODIFY) | Replace `_prefill_int8`/`_decode_int8` with unified kernel call |
| `tests/test_paged_attention.py` (NEW) | Correctness tests against FP16 baseline |

### Task 1: Write paged attention Triton kernel

**Files:**
- Create: `nanovllm/layers/paged_attention.py`

This is a stripped-down port of vLLM's `kernel_unified_attention` with:
- No 3D mode (`IS_3D=False`, `NUM_SEGMENTS_PER_SEQ=1` always)
- No ALiBi, no softcap, no QQ bias, no sinks
- No sliding window, no chunked attention
- No mm_prefix (bidirectional ranges)
- No tensor descriptors (`USE_TD=False`)
- No FP8 output (`USE_FP8=False`)
- No per-tensor FP8 dequant (`KV_QUANT_MODE` only 0 or 2)
- Single `num_queries_per_kv=1` (GQA not needed for Qwen3-0.6B — but keep it general)

The kernel takes:
```python
@triton.jit
def paged_attention_kernel(
    output_ptr,
    query_ptr,
    key_cache_ptr, value_cache_ptr,
    k_scale_cache_ptr, v_scale_cache_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    query_start_len_ptr,
    scale,
    num_queries_per_kv: tl.constexpr,
    block_table_stride: tl.int64,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,
    stride_kb: tl.int64, stride_ks: tl.int64, stride_kh: tl.int64, stride_kd: tl.constexpr,
    stride_vb: tl.int64, stride_vs: tl.int64, stride_vh: tl.int64, stride_vd: tl.constexpr,
    stride_ksb: tl.int64, stride_kss: tl.int64, stride_ksh: tl.int64,
    stride_vsb: tl.int64, stride_vss: tl.int64, stride_vsh: tl.int64,
    KV_QUANT_MODE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_M: tl.constexpr,
    num_seqs: tl.int32,
):
```

The Python wrapper:
```python
def paged_attention(
    q,              # [total_tokens, num_query_heads, head_size]
    k_cache,        # [num_blocks, block_size, num_kv_heads, head_size] int8 or fp16
    v_cache,        # same
    k_scale_cache,  # [num_blocks, block_size, num_kv_heads] or None
    v_scale_cache,  # same
    block_table,    # [num_seqs, max_num_blocks]
    seq_lens,       # [num_seqs] — total len of each seq (context + query)
    cu_seqlens_q,   # [num_seqs + 1] — cumulative query lengths
    softmax_scale,
    kv_quant_mode=0,  # 0=none, 2=int8_per_token_head
) -> torch.Tensor:   # [total_tokens, num_query_heads, head_size]
```

**Key helpers ported from vLLM (embedded in same file):**
- `resolve_seq_and_query_len` — binary search over `query_start_len_ptr` to resolve `(seq_idx, q_block_local_idx, cur_start, cur_batch_query_len, seq_len)`
- `find_seq_idx` — binary search helper
- `compute_tile_loop_bounds` — simplified (no sliding window, no 3D, no mm_prefix): `loop_lo=0, loop_hi=num_tiles`
- `compute_kv_seq_mask` — simplified: just `seq_offset[None, :] <= query_abs_pos` (causal)
- `softmax_step` — online softmax: `m_j = max(M, max(S))`, `P = exp(S - m_j)`, `alpha = exp(M - m_j)`, `L = L*alpha + sum(P)`

**Kernel algorithm (from vLLM, stripped):**
1. `pid = program_id(0)`, `kv_head_idx = program_id(1)`
2. `resolve_seq_and_query_len` → `(seq_idx, q_block_local_idx, cur_start, cur_batch_query_len, seq_len)`
3. Early return if `q_block_local_idx * BLOCK_Q >= cur_batch_query_len`
4. Load Q tile `[BLOCK_M, HEAD_SIZE_PADDED]`
5. `context_len = seq_len - cur_batch_query_len`
6. `M = -inf`, `L = 1.0`, `acc = 0`
7. Tile loop over K/V:
   a. `seq_offset = j * TILE_SIZE + offs_t`
   b. `physical_block_idx = block_tables_ptr[seq_idx * block_table_stride + seq_offset // BLOCK_SIZE]`
   c. Load K tile (pointer arithmetic via physical_block_idx), cast via `_cast_kv_tile`
   d. Load V tile (pointer arithmetic via physical_block_idx), cast via `_cast_kv_tile`
   e. If `KV_QUANT_MODE >= 2`: load `k_token_head_scales`, `v_token_head_scales` from scale caches
   f. `S = dot(Q, K) * (scale * k_scale[None, :])` if INT8, else `S = scale * dot(Q, K)`
   g. Apply causal mask
   h. `softmax_step(S, M, L)` → `(M_new, L_new, P, alpha)`
   i. `acc = acc * alpha[:, None]`
   j. `acc += dot(P * v_scale[None, :], V)` if INT8, else `acc += dot(P.to(V.dtype), V)`
8. Epilogue: `acc = acc / L[:, None]`, write to `output_ptr`

### Task 2: Integrate into Attention.forward

**Files:**
- Modify: `nanovllm/layers/attention.py`

Replace the `_prefill_int8` / `_decode_int8` paths with a single call to `paged_attention`. The forward method simplifies to:

```python
def forward(self, q, k, v):
    context = get_context()
    if self.k_cache.numel():
        if self.k_scale_cache is not None:
            store_kvcache_int8(k, v, ...)
        else:
            store_kvcache(k, v, ...)

    return paged_attention(
        q, self.k_cache, self.v_cache,
        self.k_scale_cache, self.v_scale_cache,
        context.block_tables,
        context.seq_lens,    # NEW field: per-seq total length
        context.cu_seqlens_q,
        self.scale,
        kv_quant_mode=2 if self.k_scale_cache is not None else 0,
    )
```

**Context changes needed:**
- Add `seq_lens` field to context (set during `prepare_prefill`/`prepare_decode`)
- `seq_lens[i]` = total length of sequence i (context_len + query_len for prefill, context_len + 1 for decode)

### Task 3: Write correctness tests

**Files:**
- Create: `tests/test_paged_attention.py`

Tests:
1. `test_fp16_paged_attention_vs_ref` — FP16 flash_attn_varlen_func as reference, compare output (prefill with prefix cache, decode)
2. `test_int8_paged_attention_vs_ref` — INT8, compare against current A2 implementation
3. `test_paged_attention_causal_mask` — verify causal masking is correct (token i can only attend to tokens <= i)

### Task 4: Run PPL validation

- Run `python profile/ppl_eval.py auto` — baseline must not regress
- Run `python profile/ppl_eval.py int8` — PPL must match current A2 result (~18.88)
- Run `python profile/compare_cache.py int8` — block count must match (321)
