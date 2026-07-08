# vLLM 量化知识笔记

## 校准（Calibration）在 vLLM 中的位置

### 核心结论：vLLM 推理引擎内部没有"用数据集校准"的流程

校准发生在**模型离线量化阶段**（外部工具），不在推理引擎中。

### 三种 scale 来源

| 来源 | 方式 | 适用场景 |
|------|------|---------|
| **checkpoint 加载**（推荐） | llm-compressor 用数据集算出 scale → save_pretrained → vLLM 加载 | FP8 per-tensor KV cache |
| **动态计算**（`calculate_kv_scales=True`，已废弃） | 第一个 forward batch 的 `max\|val\| / range` | FP8 per-tensor（v0.19 移除） |
| **默认 fallback** | 无 checkpoint scale → `k_scale=v_scale=q_scale=1.0` | 无校准时的兜底 |

### llm-compressor 校准流程

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

recipe = QuantizationModifier(
    targets="Linear",
    kv_cache_scheme=QuantizationArgs(num_bits=8, type="float", strategy="tensor"),
)
oneshot(model, dataset=calibration_dataset, recipe=recipe)
model.save_pretrained(save_dir, save_compressed=True)
```

校准后 checkpoint 中新增 `q_scale/k_scale/v_scale` 参数（每层一个标量）。

---

## 不同量化方法需要不同 kernel 的原因

### 不能统一的本质差异

| 差异 | AWQ | GPTQ | FP8 |
|------|-----|------|-----|
| **Pack 维度** | 沿 N (output) 维 | 沿 K (input) 维 | 8-bit 无需 pack |
| **Bit 顺序** | `[0,4,1,5,2,6,3,7]` | `[0,1,2,3,4,5,6,7]` | N/A |
| **Zero point** | 有 | 有 | 无 |
| **g_idx (desc_act)** | 无 | 有 | 无 |
| **Dequant 公式** | `(val - zp) * scale` | `(val - zp) * scale` | `val * scale` |

### 可以统一的部分（历史包袱，非算法要求）

- **Pack 维度**：AWQ 沿 N 维 pack 只是实现选择，vLLM 的 `_convert_awq_to_standard_format()` 可以转成沿 K 维 pack
- **Bit 顺序**：AWQ 的乱序也是早期实现选择，不是算法要求的

### 关键代码

```python
# awq_marlin.py:96-137 — AWQ → GPTQ 标准格式转换
# AWQ: qweight[K, N//8]  packed_dim=1
# ↓ 转换后 ↓
# Standard: qweight[K//8, N]  packed_dim=0
```

---

## KV Cache INT8/FP8 量化

### Per-token-head 动态量化（INT8 和 FP8 都同此模式）

K/V 存为 INT8/FP8，scale 按 (token, head) 在写 cache 时实时计算：

```python
# kvcache_int8.py — nano-vllm 实现
scale = max(|val|) / 127.0  # INT8 对称量化，无 zero point

# vLLM per-token-head 也走同样逻辑
# kv_cache.py:57-69 中 q_scale/k_scale/v_scale 都设为 1.0
# 实际 scale 在 reshape-and-cache kernel 中动态算
```

**Q 不量化**（`q_descale = None`），保持 FP16，attention 计算用 FP16 Tensor Core。

### FP8 per-tensor 量化

K/V 存为 FP8，Q 也被量化为 FP8（在 attention forward 之前，用 `QuantFP8` 做静态量化）：

```python
# attention.py:461-471
query, _ = self.query_quant(query, self._q_scale)
# _q_scale 来自 checkpoint（建议）或 calculate_kv_scales 或 1.0
```

Q 量化是一个**独立的 CUDA kernel**（`ops.scaled_fp8_quant`），增加了访存开销，但 flash attention 3 内部用 FP8 Tensor Core 做 attention 计算，整体收益 > 开销。

### 为什么 per-token-head 不量化 Q

1. **Q 不存 cache，量化不省显存**
2. **Per-(token, head) 的 Q 量化与硬件原语不兼容**：`_scaled_mm` 只能接受 per-tensor/per-row/per-block 的 descale，不能处理 (token, head) 二维交叉的 scale 粒度
3. **Per-token-head 只用 Triton 后端**，没有 flash attention 3 的 FP8 Tensor Core 加速
4. **每步 forward 都要重新算**，比写 cache 时顺便算 K/V scale 多出额外开销

代码中的显式排除（`attention.py:424`）：
```python
and not self.kv_cache_dtype.endswith("per_token_head")
```

---

## 三种方案的架构对比

| 方案 | K/V cache 存储格式 | Q 格式 | attention 计算 | 适用后端 |
|------|-------------------|--------|---------------|---------|
| INT8 per-token-head | INT8 + fp16 scale[token, head] | FP16 | FP16 Tensor Core | Triton |
| FP8 per-token-head | FP8 + fp16 scale[token, head] | FP16 | FP16 Tensor Core | Triton |
| FP8 per-tensor | FP8 + per-tensor scale | FP8 | FP8 Tensor Core | FlashAttn3 |

---

## 参考文件

- `vllm/model_executor/layers/quantization/kv_cache.py` — KV cache scale 加载逻辑
- `vllm/model_executor/layers/attention/attention.py` — Q 量化 gate 和 calc_kv_scales
- `vllm/model_executor/layers/quantization/input_quant_fp8.py` — QuantFP8 实现
- `vllm/v1/attention/backends/triton_attn.py` — per-token-head 实现
- `vllm/v1/attention/ops/triton_unified_attention.py` — Triton kernel 中的量化分支
- `vllm/v1/attention/backends/flash_attn.py` — flash attention 3 的 FP8 per-tensor 实现
- `vllm/model_executor/layers/quantization/awq_marlin.py` — AWQ→GPTQ 格式转换
- `vllm/model_executor/layers/quantization/awq.py` — AWQ 权重量化
- `vllm/model_executor/layers/quantization/auto_gptq.py` — GPTQ 权重量化
