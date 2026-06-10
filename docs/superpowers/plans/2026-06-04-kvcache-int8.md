# KV Cache INT8_PER_TOKEN_HEAD Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add INT8 per-token per-head symmetric quantization for KV cache in nano-vllm.

**Architecture:** Three layers of change — (1) Config flag to enable INT8 cache, (2) ModelRunner allocates INT8 cache + scale cache, (3) Attention layer uses new Triton store kernel and dequantizes before flash_attn_varlen_func.

**Tech Stack:** Python, PyTorch, Triton, flash-attn

---

### Task 1: Add `kvcache_dtype` config option

**Files:**
- Modify: `nanovllm/config.py:7-27`

- [ ] **1.1: Add field to Config dataclass**

Add `kvcache_dtype: str = "auto"` after `num_kvcache_blocks`.

- [ ] **1.2: Add validation in `__post_init__`**

Add assert: `self.kvcache_dtype in ("auto", "int8_per_token_head")`


### Task 2: Add INT8 cache allocation in ModelRunner

**Files:**
- Modify: `nanovllm/engine/model_runner.py:106-124`

- [ ] **2.1: Set up INT8 cache dtype and scale cache dtype**

In `allocate_kv_cache`, compute:
```python
use_int8 = config.kvcache_dtype == "int8_per_token_head"
cache_dtype = torch.int8 if use_int8 else hf_config.torch_dtype
scale_dtype = torch.float16 if use_int8 else None
```

- [ ] **2.2: Adjust block_bytes calculation for INT8**

When INT8, factor in both the INT8 data (1 byte) and the FP16 scale (2 bytes / head_dim per element):
```python
block_data_bytes = 2 * num_hidden_layers * block_size * num_kv_heads * head_dim * cache_dtype.itemsize
block_scale_bytes = 2 * num_hidden_layers * block_size * num_kv_heads * scale_dtype.itemsize if use_int8 else 0
block_bytes = block_data_bytes + block_scale_bytes
```

- [ ] **2.3: Allocate INT8 kv_cache and scale_cache**

```python
self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim, dtype=cache_dtype)
if use_int8:
    self.kv_scale_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, dtype=scale_dtype)
else:
    self.kv_scale_cache = None
```

- [ ] **2.4: Assign per-layer slices including scale cache**

In the loop assigning module.k_cache / v_cache, also assign k_scale_cache / v_scale_cache when INT8:
```python
module.k_cache = self.kv_cache[0, layer_id]
module.v_cache = self.kv_cache[1, layer_id]
if use_int8:
    module.k_scale_cache = self.kv_scale_cache[0, layer_id]
    module.v_scale_cache = self.kv_scale_cache[1, layer_id]
```


### Task 3: Implement INT8 Triton store kernel

**Files:**
- Add: `nanovllm/layers/kvcache_int8.py` (new file)
- Modify: `nanovllm/layers/attention.py` (integrate new kernel)

- [ ] **3.1: Create `nanovllm/layers/kvcache_int8.py` with Triton store kernel**

New kernel `store_kvcache_int8_kernel`:
```
input:  key [N, num_kv_heads, head_dim] FP16
        value [N, num_kv_heads, head_dim] FP16
        k_cache [num_slots, num_kv_heads, head_dim] INT8
        v_cache [num_slots, num_kv_heads, head_dim] INT8
        k_scale_cache [num_slots, num_kv_heads] FP16
        v_scale_cache [num_slots, num_kv_heads] FP16
        slot_mapping [N] INT32

for each token idx in 0..N-1:
    slot = slot_mapping[idx]
    if slot == -1: skip
    for each head h in 0..num_kv_heads-1:
        load k_fp16 = key[idx, h, :]   # head_dim elements
        load v_fp16 = value[idx, h, :]

        # symmetric quantization per (token, head)
        k_absmax = max(abs(k_fp16))
        k_scale = k_absmax / 127.0 if k_absmax > 0 else 1.0
        k_int8 = clamp(round(k_fp16 / k_scale), -128, 127)

        v_absmax = max(abs(v_fp16))
        v_scale = v_absmax / 127.0 if v_absmax > 0 else 1.0
        v_int8 = clamp(round(v_fp16 / v_scale), -128, 127)

        store k_int8 -> k_cache[slot, h, :]
        store v_int8 -> v_cache[slot, h, :]
        store k_scale -> k_scale_cache[slot, h]
        store v_scale -> v_scale_cache[slot, h]
```

Implementation details:
- Triton program_id(0) = token index
- Use `tl.load`/`tl.store`
- For absmax: use `tl.max(tl.abs_(val), axis=1)`
- For rounding: `tl.math.round(val / scale)`
- Guard against division by zero with `tl.where(k_absmax > 0, k_absmax / 127.0, 1.0)`
- Clamp: `tl.clamp(rounded, -128, 127)`
- Each Triton program handles one (token, head) pair → grid = (N, num_kv_heads)

- [ ] **3.2: Create host function `store_kvcache_int8`**

```python
def store_kvcache_int8(key, value, k_cache, v_cache, k_scale_cache, v_scale_cache, slot_mapping):
    N, num_kv_heads, head_dim = key.shape
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert k_cache.dtype == torch.int8
    assert k_scale_cache.dtype == torch.float16
    grid = (N, num_kv_heads)
    store_kvcache_int8_kernel[grid](key, value, k_cache, v_cache, k_scale_cache, v_scale_cache, slot_mapping, head_dim)
```


### Task 4: Modify Attention forward for INT8 decode path

**Files:**
- Modify: `nanovllm/layers/attention.py`

- [ ] **4.1: Initialize INT8 cache references in `__init__`**

```python
self.k_cache = self.v_cache = torch.tensor([])
self.k_scale_cache = self.v_scale_cache = None  # INT8 mode: torch.tensor
```

- [ ] **4.2: Modify forward — use INT8 store kernel when applicable**

In forward, replace `store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)` with:
```python
if self.k_scale_cache is not None:
    store_kvcache_int8(k, v, k_cache, v_cache, self.k_scale_cache, self.v_scale_cache, context.slot_mapping)
else:
    store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
```

- [ ] **4.3: Modify forward — decode path for INT8**

Replace the decode branch (lines 71-74):

```python
else:  # decode
    if self.k_scale_cache is not None:
        # INT8: dequantize per sequence, then varlen prefill-style attention
        # TODO(optimize): 改用按需反量化 + block_table 重映射方案（A1），
        # 避免每次 decode 重新 gather + concat：
        #   1. 收集所有序列 decode 需要的唯一 block ID
        #   2. 用 Triton kernel 批量反量化这些 block 到临时 FP16 buffer
        #   3. 构建重映射的 block_table 指向临时 buffer
        #   4. 仍可调用 flash_attn_with_kvcache
        # TODO(optimize): 探索 fused INT8 attention Triton kernel，
        # 将反量化与 attention 计算融合，消除中间 buffer（选项 B）

        block_tables = context.block_tables  # [num_seqs, max_num_blocks]
        context_lens = context.context_lens  # [num_seqs]
        num_seqs = block_tables.size(0)
        k_list, v_list = [], []
        cu_seqlens_k = [0]
        for i in range(num_seqs):
            ctx_len = context_lens[i].item()
            total_kv_tokens = ctx_len
            # Gather all K/V for this sequence from INT8 cache
            # block_table stores the block_ids for this sequence
            k_fp16 = gather_dequantize_kv(
                k_cache, context_lens[i], block_tables[i], self.block_size,
                self.num_kv_heads, self.head_dim, self.k_scale_cache
            )
            v_fp16 = gather_dequantize_kv(
                v_cache, context_lens[i], block_tables[i], self.block_size,
                self.num_kv_heads, self.head_dim, self.v_scale_cache
            )
            k_list.append(k_fp16)
            v_list.append(v_fp16)
            cu_seqlens_k.append(cu_seqlens_k[-1] + total_kv_tokens)

        k_all = torch.cat(k_list, dim=0)  # [total_kv, num_kv_heads, head_dim]
        v_all = torch.cat(v_list, dim=0)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, device=q.device)

        # Use cu_seqlens_q from context (decode: each seq has exactly 1 query token)
        cu_seqlens_q = torch.arange(0, num_seqs + 1, dtype=torch.int32, device=q.device)
        max_seqlen_q = 1
        max_seqlen_k = context_lens.max().item()

        o = flash_attn_varlen_func(
            q, k_all, v_all,
            cu_seqlens_q=cu_seqlens_q, max_seqlen_q=max_seqlen_q,
            cu_seqlens_k=cu_seqlens_k, max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale, causal=True
        )
    else:
        o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                    cache_seqlens=context.context_lens, block_table=context.block_tables,
                                    softmax_scale=self.scale, causal=True)
```

- [ ] **4.4: Create helper function `gather_dequantize_kv`**

In `attention.py`, add:
```python
def gather_dequantize_kv(cache, ctx_len, block_table, block_size, num_kv_heads, head_dim, scale_cache):
    \"\"\"Gather K/V for one sequence from INT8 cache and dequantize to FP16.\"\"\"
    num_blocks = (ctx_len + block_size - 1) // block_size
    tokens = []
    for i in range(num_blocks):
        block_id = block_table[i].item()
        if block_id == -1:
            break
        start = 0
        end = block_size
        if i == num_blocks - 1:
            end = ctx_len - i * block_size
        block_cache = cache[block_id, start:end]       # [num_tokens, num_kv_heads, head_dim]
        block_scale = scale_cache[block_id, start:end]  # [num_tokens, num_kv_heads]
        # Dequantize
        tokens.append((block_cache.float() * block_scale.unsqueeze(-1)).half())
    return torch.cat(tokens, dim=0) if tokens else torch.empty(0, num_kv_heads, head_dim, dtype=torch.float16, device=cache.device)
```

Wait — cache shape is [num_blocks, block_size, num_kv_heads, head_dim]. Let me verify this matches.

The cache per-layer shape is `self.kv_cache[0, layer_id]` which has shape `[num_blocks, block_size, num_kv_heads, head_dim]`. Access by block_id: `cache[block_id]` → `[block_size, num_kv_heads, head_dim]`, then slice by block_size offset.

- [ ] **4.5: Modify forward — prefill path for INT8**

When INT8 and block_tables is not None (prefix cache), dequantize cached tokens:

```python
if context.is_prefill:
    if context.block_tables is not None:  # prefix cache
        if self.k_scale_cache is not None:
            # INT8: dequantize cached K/V before varlen prefill
            # ... dequantize logic similar to decode but simpler
            k_cached, v_cached = dequantize_cached_kv(...)
            k, v = k_cached, v_cached
        else:
            k, v = k_cache, v_cache
    o = flash_attn_varlen_func(q, k, v, ...)
```

Actually, looking at the current code more carefully:

```python
if context.is_prefill:
    if context.block_tables is not None:    # prefix cache
        k, v = k_cache, v_cache
    o = flash_attn_varlen_func(q, k, v, ...)
```

When prefix caching is active, the entire `k_cache` and `v_cache` (all blocks, all tokens) are passed — and `flash_attn_varlen_func` uses `block_table` to index into them internally. But since we use INT8, we need to dequantize.

For the prefill path with INT8, the simplest approach is to replicate what `flash_attn_with_kvcache`/`flash_attn_varlen_func` does with block_table: gather the cached blocks according to block_table and dequantize them, then concat with the new tokens.

Actually, let me think about this more carefully. The current prefill code for prefix cache does:
```python
k, v = k_cache, v_cache  # Pass the full cache tensors
o = flash_attn_varlen_func(q, k, v, ..., block_table=context.block_tables)
```
Where `flash_attn_varlen_func`'s `block_table` parameter tells it which blocks to look up in k/v.

For INT8, we can't pass the full cache directly. We need to either:
(a) Dequantize the entire cache first (wasteful), or
(b) Gather only needed blocks per sequence and dequantize.

Let me use option (b) for the prefill path as well, consistent with the A2 approach.

But actually, the prefill path with prefix caching is a more complex case. Let me think about this differently.

In the current code for prefill with block_table (prefix cache):
- The full k_cache, v_cache is passed
- block_table tells which blocks to use for each sequence
- new tokens' K/V are stored to cache first (via store_kvcache), then the full cache is passed

With INT8:
- The full k_cache, v_cache are INT8
- We need FP16 for flash_attn_varlen_func
- We could dequantize the specific blocks needed and concatenate them

But the flash_attn_varlen_func with block_table internals expects the k/v to have shape where it can index by block_table * block_size. So we need to dequantize into a FP16 cache that has the same layout.

Let me simplify: for the initial A2 approach, I'll dequantize the needed blocks per sequence.

Let me revise the plan to be more concrete.

For prefill with INT8 + prefix cache:
```python
if context.block_tables is not None:
    if self.k_scale_cache is not None:
        # Gather all cached K/V tokens according to block_table and dequantize
        # This replaces passing the full k_cache/v_cache
        k_cached = self._gather_dequantize_cached(context)
        v_cached = self._gather_dequantize_cached(context)  # for V
        # Concat cached (dequantized) + new tokens' K/V (already FP16 from projection)
        k = torch.cat([k_cached, k], dim=0)  # Hmm, this doesn't match...
        v = torch.cat([v_cached, v], dim=0)
```

Actually, I think for prefill, the current code does `store_kvcache` first, then passes the entire cache. But it doesn't concat k from projection + cache. Let me re-read the code:

```python
if k_cache.numel() and v_cache.numel():
    store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
if context.is_prefill:
    if context.block_tables is not None:    # prefix cache
        k, v = k_cache, v_cache
    o = flash_attn_varlen_func(q, k, v, ...)
```

After store_kvcache, the new K/V are in the cache already. Then `k, v = k_cache, v_cache` replaces the projected K/V with the full cache. The block_table param tells flash_attn_varlen_func where each sequence's tokens are.

For INT8 prefill with prefix cache, the simplest A2 approach:
1. Store new K/V to INT8 cache (quantized)
2. Gather all needed K/V from INT8 cache per sequence
3. Dequantize to FP16
4. Concat across sequences into flat tensors
5. Pass to flash_attn_varlen_func WITHOUT block_table (since we've already resolved to flat tensors)

This is similar to the decode approach but happens once during prefill. It's straightforward for A2.

OK, let me finalize the plan and write it out properly.


### Task 5: Test

**Files:**
- Create: `tests/test_kvcache_int8.py`

- [ ] **5.1: Write test for store + dequantize roundtrip accuracy**

- [ ] **5.2: Write test for INT8 vs FP16 end-to-end output similarity**

- [ ] **5.3: Run tests and verify**
