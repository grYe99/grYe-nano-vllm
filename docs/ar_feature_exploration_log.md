# AR Feature 探索记录

> 日期: 2026-06-11
> 分支: feat/async_ar
> 硬件: 2× RTX 4090 (PCIe Gen4, no P2P)
> 模型: Qwen3-0.6B / Qwen3-8B

## 背景

nano-vllm 是一个轻量级推理引擎，支持 Tensor Parallelism (TP=2) 在 2 卡上推理。
`feat/async_ar` 分支旨在探索 AllReduce 通信的计算重叠优化。

## 尝试的方案

### 方案 1: 单层内 Chunked async AllReduce ❌ [已移除]

**思路**: 将一次 `F.linear(x, w)` 按 token 维度切分成多个 chunk，第 i 块 GEMM → async AR → 第 i+1 块 GEMM，使 GEMM 与 AR 在 GPU stream 上并发。

**代码**: `RowParallelLinear._forward_chunked()` (nanovllm/layers/linear.py)

**问题**:
- AllReduce 是昂贵的操作，切分 GEMM 降低单次计算量但不降低通信量
- 2× async AR 的总 NCCL 开销 > 1× sync AR
- Qwen3-0.6B 上慢 19-42%，Qwen3-8B 上仅在 ≥2048 tokens 长 prefill 时有微弱收益

**结论**: 设计错误。分块导致通信次数加倍而计算量减半，净亏损。已整体 revert。

### 方案 2: FusedAllReduceRMSNorm ❌ [已移除]

**思路**: 将 all_reduce 与 RMSNorm 融合，减少 HBM 访存。配套 FusedAddRMSNorm 在 async chunked 场景下省掉去重 AR。

**代码**: `FusedAllReduceRMSNorm`, `FusedAddRMSNorm` (nanovllm/layers/layernorm.py)

**问题**: NCCL 的 all_reduce 本身就走 HBM，fuse norm 不额外省带宽。且 async chunked 方案本身错误，配套 fuse 随之失效。

**结论**: 无实际收益。已整体移除。

### 方案 3: 跨层 async all_reduce (CUDA Stream Pipeline) ⚠️ [保留为 POC]

**思路**: 在 NCCL stream 上 async 提交 all_reduce，GPU 默认 stream 可以提前 launch 下一层 kernel，等待时机推迟到结果真正被使用时。

**代码**: `AsyncPipeline` (nanovllm/utils/async_tp.py), `Config(ar_async=True)`

**结果**: 
- Qwen3-0.6B: 比 sync 慢 4-17% (overhead 主导)
- Qwen3-8B, 1024 tokens: 比 sync 快 ~14%（但方差内，不显著）
- 其他长度: ±10% 以内

**分析**: dense transformer 的数据依赖严格串行，o_proj/down_proj 的 all_reduce 结果被紧跟着的 layernorm 立即使用，没有独立计算路径可以重叠。async 的 "overlap" 仅发生在 CPU kernel launch 间隙，收益被 PCIe 传输延迟吞没。

**结论**: 在 PCIe + dense transformer 上无实际收益，保留为学习参考。

### 方案 4: token-level AsyncTP (类 vLLM) 🔬 [调研未实现]

**调研**: 阅读 vLLM `collective_fusion.py` 和 `_symmetric_memory` 源码。

**核心发现**: 
- vLLM AsyncTP 是 token-level parallelism（不是 Megatron 的 hidden dim split）
- Weight 完整复制，batch 在 rank 间切分
- 使用 `fused_all_gather_matmul` kernel：all_gather 与 matmul 在同一个 CUDA kernel 内交织
- 依赖 `_symmetric_memory` + NVLink P2P 实现 GPU 直接读远端显存

**我们的硬件限制**:
- `can_device_access_peer(0, 1) = False` (RTX 4090 PCIe 无 P2P)
- `_symmetric_memory.rendezvous()` 失败：`CUDASymmetricMemoryAllocator` 需要 P2P
- 纯 NCCL all_gather + two-phase matmul 可实现，但 PCIe 传输延迟 (10-30μs) >> GEMM 时间，重叠收益天花板 ~30%

**结论**: 算法正确，硬件限制。token-level TP 在 PCIe 上无实际收益。

## 发现的 Bug 与修复

### Bug 1: Sequence pickle 丢失 token_ids 🔧 [已修复]

**文件**: `nanovllm/engine/sequence.py`

**问题**: `__getstate__` 对 `num_completion_tokens > 0` 的 sequence 不保存 `token_ids`，只保存 `last_token`。当 preempt 踢回的 sequence 以 prefill 模式再调度时，子进程拿到无 `token_ids` 的 Sequence → `AttributeError`。

**修复**: `__getstate__` 总是保存 `token_ids + last_token`，去掉条件分支。

**影响**: 修复了多 batch 场景下任意 TP 组合都可能崩溃的 bug。

### Bug 2: ar_mode 未在子进程配置 🔧 [已修复]

**文件**: `nanovllm/engine/model_runner.py`

**问题**: `Config.__post_init__` 调用 `ar_mode.configure()`。子进程通过 spawn 创建，pickle 越过 `__post_init__`，导致子进程永远用默认 ar_mode。原来导致 chunked async 场景下 NCCL 操作不匹配死锁。

**修复**: `ModelRunner.__init__` 开头调用 `ar_mode.configure(...)`。

### Bug 3: atexit 重复调用 crash 🔧 [已修复]

**文件**: `nanovllm/engine/llm_engine.py`

**问题**: `atexit.register(self.exit)` 在手动 `llm.exit()` 后触发（此时 `self.model_runner` 已被 del）。

**修复**: `getattr(self, "model_runner", None)` guard。

## 实验数据摘要

| 方案 | Qwen3-0.6B | Qwen3-8B | 结论 |
|------|:----------:|:--------:|------|
| Chunked async AR | -19~-42% | -21~+26% | 设计错误，移除 |
| 跨层 async pipeline | -4~-17% | -14~+13% | 噪声范围内，保留 POC |
| token-level AsyncTP | 未实现 (P2P 不可用) | | 需 NVLink 硬件 |

## 当前代码状态

| 组件 | 状态 |
|------|------|
| `RowParallelLinear.forward` | 标准 sync all_reduce（原始设计） |
| `layernorm.py` | 标准 `RMSNorm`（无 fused 变体） |
| `nanovllm/utils/async_tp.py` | `AsyncPipeline` POC 保留 |
| `Config(ar_async=True)` | 可启用 async pipeline（不建议用于生产） |
| `test_ar_combos.py` | 已删除（专为废弃的 chunked 方案写的） |
| `bench_async_tp.py` | 对比脚本保留 |

## 参考

- vLLM AsyncTP: `vllm/compilation/passes/fusion/collective_fusion.py`
- `torch.distributed._symmetric_memory`: PyTorch 的 symmetric memory 实现
- Megatron-LM: 标准 TP 范式 (ColumnParallel + RowParallel + all_reduce)
