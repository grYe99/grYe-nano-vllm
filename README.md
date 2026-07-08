# GrYe-Nano-vLLM

基于 [Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) 的个人优化项目，针对 LLM 推理瓶颈（Linear、Attention）进行量化加速。

## 背景与动机

通过 PyTorch Profiler 对 Qwen3-8B 推理过程采样，定位主要耗时算子：

| 类别 | 包含 Kernel | CUDA Time | 占比 |
|:----:|------------|:---------:|:----:|
| **Linear** | `aten::mm`, `gemv` | ~277 ms | **~58%** |
| **Attention** | `flash_fwd_splitkv`, `combine`, `varlen` | ~118 ms | **~25%** |
| 其他 | RMS norm, element-wise, KV cache store 等 | ~80 ms | ~17% |

**Linear + Attention 合计占总 CUDA 时间的 ~83%**，是主要推理瓶颈。参考 vLLM 的量化方案，从两个方向进行优化：

1. **KV Cache INT8 量化** — 减少 Attention 阶段 KV Cache 的显存读写开销，同时支持更大的 batch size
2. **AWQ 量化模型推理** — 加载 4bit 权重量化模型，减少 Linear 层的显存访问和计算量

<details>
<summary>Profile 原始数据 (PyTorch Profiler, RTX 4090)</summary>

```
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
## Call CompiledFxGraph ...                                  0.00%       0.000us         0.00%       0.000us       0.000us     224.782ms        47.32%     224.782ms      51.793us          4340
                                           aten::linear         1.18%      83.193ms        13.79%     973.416ms      55.220us       0.000us         0.00%     146.725ms       8.323us         17628
                                               aten::mm         6.68%     471.768ms         9.44%     666.134ms      37.788us     146.153ms        30.77%     146.725ms       8.323us         17628
       std::enable_if<...>::type internal::gemvx...            0.00%       0.000us         0.00%       0.000us       0.000us      90.369ms        19.02%      90.369ms      10.227us          8836
       void flash::flash_fwd_splitkv_kernel<...>              0.00%       0.000us         0.00%       0.000us       0.000us      71.535ms        15.06%      71.535ms      16.483us          4340
       std::enable_if<...>::type internal::gemvx...            0.00%       0.000us         0.00%       0.000us       0.000us      40.180ms         8.46%      40.180ms       4.629us          8680
       void flash::flash_fwd_splitkv_combine_kernel<...>      0.00%       0.000us         0.00%       0.000us       0.000us      32.901ms         6.93%      32.901ms       7.581us          4340
                                   fused_add_rms_kernel         0.00%       0.000us         0.00%       0.000us       0.000us      14.188ms         2.99%      14.188ms       1.624us          8736
                 flash_attn::_flash_attn_varlen_forward         0.02%       1.277ms         0.04%       2.496ms      89.154us      13.573ms         2.86%      13.573ms     484.735us            28
                                   store_kvcache_kernel         0.00%       0.000us         0.00%       0.000us       0.000us       5.426ms         1.14%       5.426ms       1.242us          4368
```

`Self CUDA` 为 GPU kernel 实际耗时，排除 `allreduce`（单卡无实际通信开销）。

</details>

详细技术笔记见 [`docs/quantization.md`](docs/quantization.md)。

## Key Features

* 🚀 **KV Cache INT8 Per-Token-Head 在线量化** — 参考vLLM的Triton Kernel 实现：
  - 对称量化 + KV Cache Store
  - 反量化 + PagedAttention 融合
  - Qwen3-8B Throughput **提升 51.5%**，GSM8K 准确率下降 <1%

* 📖 **AWQ 4bit 权重量化推理** — 实现三种反量化 + GEMM 融合算子并对比选型：
  - [Triton](docs/quantization.md) / [awq_gemm (CUDA)](docs/awq_cuda_kernel.md) / [Marlin](docs/marlin_kernel.md)
  - 通过 MicroBench 对比不同 `num_tokens` 下的性能，最终默认采用 **Marlin Kernel**
  - KV Cache Block 分配数量翻倍，Qwen3-8B Throughput **提升 120%**

* 📊 **接入 lm-eval** — 实现适配器，支持通过 lm-eval 评估模型准确率

## Installation

```bash
uv pip install -e . --no-build-isolation
```

## Model Download
通过huggingface下载模型，并重定向~/huggingface/
```bash
hf download Qwen/Qwen3-8B --local-dir ~/huggingface/Qwen3-8B
hf download Qwen/Qwen3-8B-AWQ --local-dir ~/huggingface/Qwen3-8B-AWQ
```

## 实验数据

**实验环境**

| 配置 | 参数 |
|------|------|
| Hardware | RTX 4090 (24GB) |
| Model | Qwen3-8B |
| Total Requests | 256 sequences |
| Input Length | Randomly sampled 100–1024 tokens |
| Output Length | Randomly sampled 100–1024 tokens |

---

### KV Cache INT8 量化

开启方式：

```python
llm = LLM(path, enforce_eager=False, max_model_len=4096, kv_cache_dtype="int8_per_token_head")
```

#### 性能对比

| 配置 | Throughput | TTFT (avg) | TPOT (avg) |
|:----:|:----------:|:----------:|:----------:|
| baseline (fp16) | 856.37 tok/s | 62612.1 ms | 28.53 ms |
| INT8 量化 | **1297.45 tok/s** (+51.5%) | **33724.1 ms** | 35.49 ms |

> **吞吐提升的原因分析：**
> - TTFT 下降 46%（62612 → 33724 ms）是 throughput 提升的主要驱动。Prefill 计算量在 fp16/INT8 下完全一致（INT8 只影响 KV cache 存储），TTFT 下降来自 **block 数量翻倍后的排队时间缩短** — 更多序列可一次性从 waiting 调度到 running，无需等待
> - TPOT 上升（28.53 → 35.49 ms）是连续批处理下并发度增大的固有折衷：更多序列同时 decode → 每步耗时增加。INT8 反量化也带来少量额外开销（详见[此处](#关于小模型的说明)）
> - 总体吞吐 856.37→1297.45 tok/s（+51.5%），说明 TTFT 缩短节省的时间远大于 TPOT 增加带来的开销

> **TPOT 为什么上升？** 有两个因素共同作用：
> 1. **KV Cache 内存减半 → 可分配的 Block 数量翻倍 → Scheduler 能容纳更多并发序列** → 每步 decode 处理更多序列 → 单步耗时增加。这是 Continuous Batching 下的固有现象：并发度增大时 TPOT 上升、TTFT 下降、总体吞吐提升（[TensorRT-LLM benchmark](https://meesho.github.io/BharatMLStack/blog/llm-inference-optimization-sub-sec-latency/) 同样显示并发 1→256 时 ITL 从 9ms 升至 59ms，吞吐提升 40×）
> 2. **INT8 反量化开销** — 每次 Attention 读取 KV Cache 时需将 INT8 转回 FP16，增加少量计算

#### 准确率 (GSM8K)

| 配置 | flexible-extract | strict-match |
|:----:|:----------------:|:------------:|
| baseline (fp16) | 0.8832 | 0.8787 |
| INT8 量化 | 0.8749 (-0.8%) | 0.8704 (-0.9%) |

准确率下降 <1%，无明显劣化。

运行方式：

```bash
python eval_gsm8k.py
```

---

### AWQ 量化

加载 AWQ 模型：

```python
path = os.path.expanduser("~/huggingface/Qwen3-8B-AWQ/")
```

#### 性能对比

| 配置 | Throughput | TTFT (avg) | TPOT (avg) |
|:----:|:----------:|:----------:|:----------:|
| baseline (bf16) | 862.78 tok/s | 62132.2 ms | 28.32 ms |
| AWQ (Marlin, 默认) | **1896.69 tok/s** (+120%) | **24498.8 ms** | 33.56 ms |
| AWQ (awq_gemm dispatch) | 1619.73 tok/s (+88%) | 23889.6 ms | 54.37 ms |

Marlin 为默认方案；awq_gemm dispatch 策略关闭 `awq_use_marlin` 后生效：

```python
llm = LLM(path, enforce_eager=False, max_model_len=4096, awq_use_marlin=False)
```

dispatch 策略：`M <= 512` 时 awq_gemm 更优，`M > 512` 时 dequant + cuBLAS 更优。

> **吞吐提升的原因分析：**
> - AWQ 将权重从 bf16 (~16GB) 压缩到 4bit (~5GB)，释放约 11GB 显存用于 KV Cache Block
> - Block 数量大幅增长 → prefill 排队时间缩短 → TTFT 下降 60%（62132 → 24498 ms），是 throughput 提升的主要驱动
> - Marlin Kernel 自身的 GEMM 加速也有贡献（详见下方 MicroBench）


> AWQ 是 weight-only 量化，对精度影响极小，实测与 hf baseline 无显著差异。

#### 算子 MicroBench

K=4096, N=4096, group_size=128，单位: TFLOPs

<p align="center">
<img width="800" src="profile/awq_bench.svg">
</p>

<div align="center">

|  M   | awq_gemm | Triton | deq+cuBLAS | Marlin | best |
|:----:|:--------:|:------:|:----------:|:------:|:----:|
|  1   |   25.1   |  98.0  |   114.6    |  24.2  | Marlin |
|  2   |   24.5   |  98.7  |   117.1    |  23.9  | Marlin |
|  4   |   25.0   | 120.3  |   115.6    |  22.2  | Marlin |
|  8   |   22.8   | 101.9  |   116.4    |  22.7  | Marlin |
|  16  |   23.3   | 101.6  |   116.0    |  23.3  | Marlin |
|  32  |   24.4   | 104.3  |   119.0    |  23.4  | Marlin |
|  64  |   35.9   | 104.1  |   160.0    |  38.2  | awq_gemm |
| 128  |   62.1   | 101.1  |   112.3    |  36.4  | Marlin |
| 256  |  111.0   | 148.1  |   112.0    |  62.3  | Marlin |
| 512  |  207.9   | 287.9  |   150.5    | 113.2  | Marlin |
| 1024 |  416.2   | 602.1  |   261.7    | 227.8  | Marlin |
| 2048 |  914.8   | 1299.6 |   466.5    | 438.5  | Marlin |
| 4096 | 1779.3   | 2636.5 |   870.5    | 832.8  | Marlin |

</div>

运行方式：

```bash
python profile/microbench_awq.py
```

## 深入阅读

- [`docs/quantization.md`](docs/quantization.md) — 量化技术笔记（含 vLLM 量化架构分析、SmoothQuant、AWQ、GPTQ、FP8 等）
- [`docs/awq_cuda_kernel.md`](docs/awq_cuda_kernel.md) — AWQ CUDA Kernel 实现详解
- [`docs/marlin_kernel.md`](docs/marlin_kernel.md) — Marlin Kernel 实现详解
- [`docs/model-runner-deep-dive.md`](docs/model-runner-deep-dive.md) — Model Runner 源码分析

### 关于小模型的说明

原始 Nano-vLLM 使用 Qwen3-0.6B 测试，本项目的量化优化主要针对 **Qwen3-8B** 规模。小模型上存在以下问题：

1. **KV Cache 量化后准确率急剧下降**（GSM8K: 45% → 5%）— 小模型参数少，对量化误差更敏感
2. **AWQ 模型吞吐反而劣化约 14%** — 小模型参数量小，量化节省的访存不足以抵消反量化带来的额外开销

关于 1：小模型因为模型参数少，对量化参数更敏感。[相关研究](https://arxiv.org/abs/2406.10251)指出 *"Since LLM abilities emerge with scale, smaller LLMs are more sensitive to quantization"*（大模型能力随规模涌现，小模型对量化更敏感）。[IJCAI 2025](https://www.ijcai.org/proceedings/2025/902) 的实验也表明，4bit 量化下小模型（1B）精度下降严重，而 70B 级模型仍能保持稳定。

关于 2：AWQ (W4A16) 量化本身不节省计算——矩阵乘仍以 FP16 精度运行，因此优化收益来自**减少 weight 访存**。但反量化（INT4→FP16）会引入额外指令开销，有两种实现选择：单独的反量化 kernel，或反量化+GEMM 融合算子，两者都会带来性能损耗。小模型参数量小，量化减少的访存收益不足以抵消反量化开销，因此整体吞吐反而劣化。[vLLM 社区](https://discuss.vllm.ai/t/no-throughput-improvement-for-quantized-qwen-2-5-7b-instruct/1901/2)和 [GPUStack 性能报告](https://docs.gpustack.ai/2.1/performance-lab/references/the-impact-of-quantization-on-vllm-inference-performance/)均观察到类似现象。

详见 [`docs/quantization.md`](docs/quantization.md) 中的分析。
