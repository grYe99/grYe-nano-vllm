# KV Cache INT8_PER_TOKEN_HEAD Quantization Design

## Overview

INT8 per-token per-head symmetric quantization for both K and V caches in nano-vllm.

## Data Layout

### INT8 data cache (`kv_cache`)
```
Shape:  [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
Dtype:  torch.int8
```
- `[0]` = K cache, `[1]` = V cache

### Scale cache (`kv_scale_cache`)
```
Shape:  [2, num_layers, num_blocks, block_size, num_kv_heads]
Dtype:  torch.float16
```
- One scalar scale per (token, head) per K or V
- Symmetric quantization: `scale = max(|val|) / 127.0`
- Dequantize: `val_fp16 = val_int8 * scale`

### Memory savings
- FP16: 2 bytes per element
- INT8: 1 byte (data) + 2/head_dim bytes (scale) per element
- For head_dim=128: effective = 1 + 2/128 = 1.0156 bytes per element (~49% of FP16)

## Configuration

### `nanovllm/config.py`

Add `kvcache_dtype: str = "auto"` field to `Config`:
- `"auto"`: uses model `torch_dtype` (current FP16/BF16 behavior)
- `"int8_per_token_head"`: uses INT8 with per-token-head scales

## Implementation

### 1. Cache Allocation (`nanovllm/engine/model_runner.py`)

When `kvcache_dtype == "int8_per_token_head"`:
- `kv_cache` → `torch.int8`
- `kv_scale_cache` → `torch.float16`, same leading dims as `kv_cache` but without `head_dim`
- Both allocated as single flat tensors, sliced per layer to each `Attention` module

`gpu_memory_utilization` calculation accounts for both tensors.

### 2. Store Kernel — Triton (`nanovllm/layers/attention.py`)

New kernel `store_kvcache_int8_kernel`:
- Input: FP16 K, V tensors from the linear projection
- Quantizes each (token, head) independently:
  - `scale = max(|val|) / 127.0` (symmetric)
  - `val_int8 = clamp(round(val / scale), -128, 127).to(torch.int8)`
  - Stores `val_int8` to `kv_cache[slot, head, :]`
  - Stores `scale` to `kv_scale_cache[slot, head]`
- Slot mapping same as current: `block_id * block_size + offset_within_block`

### 3. Decode Forward (`nanovllm/layers/attention.py`)

**Phase 1 (A2 — simple, this PR):**
- Gather all past K/V tokens per sequence from INT8 cache + scale cache
- Dequantize each to FP16: `val_fp16 = val_int8 * scale.unsqueeze(-1)`
- Concatenate across sequences into flat K, V tensors
- Use `flash_attn_varlen_func` instead of `flash_attn_with_kvcache`
- The `Context` object supplies `cu_seqlens_k` for decode

- **TODO(optimize)**: 改用按需反量化 + block_table 重映射方案（A1），避免每次 decode 重新 gather + concat：
  - 收集所有序列 decode 需要的唯一 block ID
  - 用 Triton kernel 批量反量化这些 block 到临时 FP16 buffer
  - 构建重映射的 block_table 指向临时 buffer
  - 仍可调用 `flash_attn_with_kvcache`

- **TODO(optimize)**: 探索 fused INT8 attention Triton kernel，将反量化与 attention 计算融合，消除中间 buffer（选项 B）

### 4. Prefill with Prefix Cache

When block_table has cached tokens:
- Read cached K/V from INT8 cache
- Dequantize: `val * scale.unsqueeze(-1)`
- Concatenate with non-cached tokens' FP16 K/V
- Feed to `flash_attn_varlen_func`

### 5. Layer Interface

`Attention` module receives two new references:
- `self.k_cache_int8`
- `self.v_cache_int8`
- `self.k_scale_cache`
- `self.v_scale_cache`

The original `self.k_cache` and `self.v_cache` are `None` when in INT8 mode.

## Error Tolerance

Symmetric INT8 quantization with per-token-head scale:
- Expected error: ~0.5% relative to FP16 for typical KV distributions
- Verification: end-to-end logit difference vs FP16 baseline < 1.0 at temperature=0

## Files Changed

| File | Changes |
|------|---------|
| `nanovllm/config.py` | Add `kvcache_dtype` field |
| `nanovllm/engine/model_runner.py` | INT8 cache allocation, slot mapping for INT8 |
| `nanovllm/layers/attention.py` | New Triton store kernel, decode path with dequant + flash_attn_varlen_func, prefill path with dequant |
| `nanovllm/utils/context.py` | May need `cu_seqlens_k` for decode path |
| `tests/test_kvcache_int8.py` | Roundtrip, end-to-end, scale correctness tests |
