# AR Mode: 正交优化标志

把 `NANOVLLM_AR_MODE` 从三选一的 enum 重构为两个独立的 boolean 标志。

## 设计

两个正交维度：

| 优化 | 标志 | 含义 |
|------|------|------|
| 分块异步 all_reduce | `ar_async_chunked` | RowParallelLinear 做 chunked async all_reduce |
| 融合归一化 | `ar_fused_norm` | RMSNorm 用 Triton fused kernel（含或不含 AR） |

通过 `LLM(path, ar_async_chunked=True, ar_fused_norm=True)` 控制。

## 四种组合

| ar_async_chunked | ar_fused_norm | RowParallelLinear | RMSNorm |
|---|---|---|---|
| False | False | `_forward_sync` (F.linear + sync AR) | RMSNorm (plain, @torch.compile) |
| False | True | `_forward_pass` (F.linear only) | FusedAllReduceRMSNorm (AR + Triton add+norm) |
| True | False | `_forward_chunked` (chunked async AR) | RMSNorm (plain, @torch.compile) |
| True | True | `_forward_chunked` (chunked async AR) | FusedAddRMSNorm (Triton add+norm, NO AR) |

关键组合是 `(True, True)`：RowParallelLinear 做完 chunked async all_reduce 后，FusedAddRMSNorm 只需做 add+rmsnorm，不需要重复 all_reduce。

### 在 TP=1 时

四种组合行为完全一致——`tp_size <= 1` 的 guard 直接跳过所有 all_reduce，只是 `F.linear` 而已。

---

## 一句话总结

把 `RowParallelLinear` 的 token 维度**分块**，让第 i+1 块的 GEMM 与第 i 块的 all_reduce **在 GPU 上并发执行**，从而隐藏通信延迟。

---

## 为什么能重叠

GPU 上有两类独立的执行单元：

| 单元 | 负责 | 对应操作 |
|------|------|---------|
| **Tensor Core** | 矩阵乘法 | `F.linear(chunk, weight)` |
| **NCCL** | 跨卡通信 | `dist.all_reduce(tensor)` |

它们运行在不同的 CUDA stream 上。只要数据没有依赖关系，就可以同时跑：

```
时间 ────────────────────────────────────────────→

Naive（同步等）:
  |←── F.linear(chunk) ──→|←── all_reduce ──→|
                                                   GPU 空闲
  |←── F.linear(chunk) ──→|←── all_reduce ──→|

分块异步（重叠）:
  |← F.linear(c0) →|← F.linear(c1) →|← concat →|
                    |← all_reduce(c0)→|           ← 藏在 GEMM 后面
                                       |← all_reduce(c1)→|
```

---

## 代码实现

```python
def _forward_chunked(self, x):
    # 单卡或小输入直接退化为同步
    if self.tp_size <= 1:
        return F.linear(x, self.weight, self.bias)
    if x.size(0) < get_ar_min_tokens():  # 默认 1024
        return self._forward_sync(x)

    chunks = x.chunk(get_ar_num_chunks())  # 默认 2 块
    handles, outputs = [], []

    for i, chunk in enumerate(chunks):
        # Step A: 计算当前块的 GEMM（在 default stream 上）
        y = F.linear(chunk, self.weight, None)

        # Step B: 启动当前块的异步 all_reduce（在 NCCL stream 上）
        h = dist.all_reduce(y, async_op=True)

        # Step C: 等待上一块的 all_reduce 完成（此时上一块数据已就绪）
        if i > 0:
            handles[i - 1].wait()

        handles.append(h)
        outputs.append(y)

    # Step D: 等待最后一块
    handles[-1].wait()
    return torch.cat(outputs, dim=0)
```

### 执行流拆解

以 2 块为例，`num_chunks=2`：

```
Iteration 0:
  default stream: F.linear(chunk0)        ──→ 产生 y0
  nccl stream:                             开始 all_reduce(y0)  ← 与下面重叠
                                                    ↓
Iteration 1:
  default stream: F.linear(chunk1)   ←─── concurrent ──→  ← 这里 GEMM 和通信同时跑
  nccl stream:                          all_reduce(y0) 完成
                   ↑
             此时 wait(handle0) 几乎不阻塞，因为 y0 已经归约好了

  default stream: wait(handle0) → 几乎立即返回
  default stream: wait(handle1) → 等 y1 归约完
  default stream: torch.cat([y0, y1])
```

### 为什么不使用 bias

每块 GEMM 都不传 bias：`F.linear(chunk, weight, None)`。

因为 bias 是完整向量（不是分块的），逐块加 bias 会在 cat 后重复加 `num_chunks` 次。所以 bias 在 cat 之后统一加一次：

```python
y = torch.cat(output_chunks, dim=0)
if self.bias is not None and self.tp_rank == 0:
    y = y + self.bias
```

---

## 为什么需要 min_tokens 阈值

分块异步不是没有代价的：

| 开销 | 原因 |
|------|------|
| 额外 kernel launch | 从 1 次 `F.linear` 变成 `num_chunks` 次 |
| chunk 非对齐 | GEMM 在非最优 shape 上效率略低 |
| 同步开销 | `handle.wait()` 有 CPU-GPU 同步成本 |

当 token 数很少时（decode 阶段，通常 1~256 tokens），这些开销可能超过收益。所以默认阈值设 `1024` tokens 以下走同步路径：

```python
min_tokens = 1024  # NANOVLLM_AR_MIN_TOKENS
```

这意味着：
- **Prefill 阶段**（几千 tokens）→ 走分块异步，收益明显
- **Decode 阶段**（1 token）→ 退化为同步，没有额外开销

---

## 权威来源

分块异步 all_reduce 不是 nano-vllm 发明的，而是分布式训练/推理中**经典的通信计算重叠模式**。

### 1. Lina (USENIX ATC '23)

发表在顶会的论文，核心贡献就是 tensor partitioning + pipelining：

> "Breaking large allreduce tensors into equal-sized small chunks (micro-ops) so that communication can be interleaved with computation at finer granularity."

—— Lina: Breaking the Memory Wall for MoE Inference, USENIX ATC '23

这和我们方案的核心思想完全一致：把张量切成小块，交错安排通信和计算，避免 GPU 空闲等待。

### 2. NVIDIA NeMo Megatron Bridge Pipelined Overlap

NVIDIA 官方提供的 Megatron Bridge 文档描述了 **Pipelined Overlap** 策略：

> "TP communications with direct computation dependency are overlapped in pipelined fashion — All-gather is replaced with multiple steps of input P2P ring exchanges, Reduce-scatter is replaced with multiple steps of GEMM output P2P ring exchanges."

—— NeMo Megatron Bridge Communication Overlap 文档

这个做法在概念上是一致的：把一次集体通信拆成多个小步骤，与计算流水线化。

### 3. NVIDIA Apex DistributedDataParallel

Apex DDP 的 `num_allreduce_streams` 参数，在多 stream 上重叠梯度 all_reduce 与 backward 计算。场景不同（gradient vs forward），但底层的 **CUDA stream pipelining 技术完全一样**——用独立的 stream 跑通信，用 event 控制依赖，让通信和计算并发。

### 4. 通用 CUDA 编程模式

用多个 CUDA stream + event 实现并发，是 CUDA 编程手册中明确支持的标准模式。NCCL 的 `ncclAllReduce` 本身接受 `cudaStream_t` 参数，可以提交到任意 stream。这个方案本质上就是利用了 NCCL 和 CUDA 的原生能力，没有任何 hack。

---

## 与 vLLM 的对比

### vLLM 没有分块异步 all_reduce

vLLM 当前的 `RowParallelLinear.forward` 就是简单的同步 all_reduce（`vllm/model_executor/layers/linear.py:1558`）：

```python
if self.reduce_results and self.tp_size > 1:
    output = tensor_model_parallel_all_reduce(output_parallel)
```

没有分块、没有 async_op、没有 token 数阈值。

### vLLM 的 TP 优化栈

| 方案 | 类型 | 硬件要求 | 是否重叠 | 是否存在于 nano-vllm |
|------|------|---------|---------|-------------------|
| **NCCL AllReduce** | 同步通信 | 任意 GPU | ❌ | ✅ 组合 (F,F) |
| **CustomAllReduce** | 更快 AR kernel | NVLink full-mesh | ❌ | ❌ |
| **FlashInfer AR** | 更快 AR kernel | sm90+ (H100) | ❌ | ❌ |
| **AllReduceFusionPass** | 融合算子 | sm90+ (H100) | ❌ | ✅ 组合 (F,T) (Triton) |
| **SymmMem** | 对称内存 AR | H100 + NVLink | ❌ | ❌ |
| **AsyncTP (symm_mem)** | GEMM+通信融合 | H100 + NVLink 4.0 | ✅ | ❌ |
| **PushBasedAllReduce** | 更快 AR kernel | NVLink | ❌ | ❌ |
| **分块异步 (nano-vllm)** | 分块+管道 | **任意 GPU** | ✅ | **✅ ar_async_chunked** |

核心发现：**vLLM 没有通用的、能跑在任何 GPU 上的通信计算重叠手段。** 每个加速方案都有硬件门槛。

### vLLM 各方案详解

#### CustomAllReduce

用途：在 NVLink 上替代 NCCL 的 all_reduce，通过 GPU P2P 直接读写对方显存完成归约，延迟更低。

```python
# custom_all_reduce.py:232
def should_custom_ar(self, inp):
    if self.world_size == 2 or self.fully_connected:  # NVLink full-mesh
        return inp_size < self.max_size
    return False  # PCIe → 禁用
```

注释也很直白：

> "for 4 or more non NVLink-capable GPUs, custom allreduce provides **little performance improvement over NCCL**."

**它不是重叠方案。** 只是让 all_reduce 本身跑得更快（利用 NVLink P2P），计算和通信仍然是串行的。

#### AsyncTP (symm_mem)

这是 vLLM **唯一真正的通信计算重叠方案**。但它做的不是「分块 async all_reduce」，而是 **GEMM 和通信的算子级融合**：

```
AsyncTP pattern 1: AllGather + GEMM → torch.ops.symm_mem.fused_all_gather_matmul
AsyncTP pattern 2: GEMM + ReduceScatter → torch.ops.symm_mem.fused_matmul_reduce_scatter
```

这依赖 PyTorch 的 `_symmetric_memory` 扩展（sm90+ NVLink 专属），通过 `torch.compile` Inductor pass 自动模式匹配替换。

**和 ar_async_chunked 的核心区别：**

| 维度 | vLLM AsyncTP | nano-vllm ar_async_chunked |
|------|-------------|------------------|
| 重叠方式 | 同一个 kernel 内 GEMM 和通信融合 | 不同 kernel 在不同 stream 上并发 |
| 硬件要求 | H100 + NVLink 4.0 | **任意 GPU** |
| 实现方式 | 编译器 Inductor pass | 手動 stream pipelining |
| CUDA Graph | ✅ | ❌（async_op 不兼容） |
| 粒度 | 算子级（整个 GEMM + 整个 RS） | token 级（分块） |

### vLLM 的 PCIe fallback

在 PCIe 或非 NVLink 环境中，vLLM 的 TP 通信走 `cuda_communicator.py:284-298`：

```python
pynccl_comm = self.pynccl_comm
if pynccl_comm is None or pynccl_comm.disabled:
    out = input_.clone()
    torch.distributed.all_reduce(out, group=self.device_group)  # ← 纯同步 NCCL
    return out
# 或者：
out = pynccl_comm.all_reduce(input_)  # ← pynccl 调 ncclAllReduce，也是同步
```

**PCIe 上 vLLM 的 TP 就是「GEMM → 同步等 all_reduce → 下一层」，没有重叠、没有优化、没有任何特殊处理。**

这也是合理的——vLLM 的目标部署环境是 H100 集群，PCIe 场景不在优化范围内。

### 我们的定位

```
vLLM 的优化栈（NVLink-only）:
  NCCL AR → CustomAR → FlashInfer → SymmMem → AsyncTP
  ↑ PCIe 用户到这里就停了

nano-vllm ar_async_chunked:
  在 PCIe 上也能做通信计算重叠
  不依赖特殊硬件，用纯 Python + torch.distributed 实现
```

**在 PCIe 场景下，ar_async_chunked=True 反而是有实际价值的优化——这是 vLLM 覆盖不到的领域。**

---

## 四种组合的适用场景

| ar_async_chunked | ar_fused_norm | 适用场景 |
|---|---|---|
| False | False | Baseline，用于验证正确性和对比性能。对齐 vLLM 标准 TP。 |
| False | True | 融合算子优化。适合 decode 为主的场景（token 少，chunked 无收益）。 |
| True | False | 纯通信计算重叠。长 prefill（≥1024 tokens）效果显著，但 norm 用纯 Python 实现。 |
| True | True | **推荐**。同时享受通信计算重叠 + Triton 融合归一化，四种组合中性能最优。 |

### 在 TP=1 时

四种组合行为完全一致——`tp_size <= 1` 的 guard 直接跳过所有 all_reduce，只是 `F.linear` 而已。

---

## 验证

```python
from nanovllm import LLM

# 四种组合都应产生相同的 token_ids
llm = LLM(path, tensor_parallel_size=2, ar_async_chunked=False, ar_fused_norm=False)
llm = LLM(path, tensor_parallel_size=2, ar_async_chunked=False, ar_fused_norm=True)
llm = LLM(path, tensor_parallel_size=2, ar_async_chunked=True, ar_fused_norm=False)
llm = LLM(path, tensor_parallel_size=2, ar_async_chunked=True, ar_fused_norm=True)
```

Env var 向后兼容：
```bash
NANOVLLM_AR_MODE=0 python bench.py   # ar_async_chunked=False, ar_fused_norm=False
NANOVLLM_AR_MODE=1 python bench.py   # ar_async_chunked=False, ar_fused_norm=True
NANOVLLM_AR_MODE=2 python bench.py   # ar_async_chunked=True,  ar_fused_norm=True
```

---

## 常见问题

### Q: 为什么 vLLM 不做分块异步 all_reduce？

A: 因为 vLLM 有条件走更优的路线。在 H100+NVLink 环境下，symm_mem 算子级融合（GEMM + ReduceScatter 在同一个 CUDA op 内完成）比分块异步更高效、更优雅。vLLM 不做不是因为分块异步不对，而是一个工程取舍——他们有 H100 集群，没必要在纯 Python 层做手动管道。

### Q: 分块异步的收益确定性如何？

A: 收益取决于 token 数和 chunk 数的比例。chunk 数固定为 2 时，token 越多，GEMM 的计算量越大，掩盖 all_reduce 延迟的效果越好。理论最大收益是将 all_reduce 延迟完全隐藏在 GEMM 中，但受限于 NCCL stream 和 default stream 的调度公平性，实际收益可能低于理论值。这也是学界和工业界仍在研究的课题。

### Q: 这个方案有学术或工业界的认可吗？

A: 有。USENIX ATC '23 的 Lina 论文、NVIDIA NeMo Megatron Bridge 的 Pipelined Overlap、NVIDIA Apex DDP 的 multi-stream DDP，都是同一种技术（tensor chunking + stream pipelining）的不同变体。分块异步 all_reduce 本身不是新发明，而是一个经典的 CUDA stream 编程模式在 TP 推理场景中的应用。
