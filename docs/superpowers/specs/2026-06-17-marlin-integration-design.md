# Marlin Kernel Integration for nano-vllm AWQ

## 目标

将 vllm 的 Marlin kernel（IST-DASLab 衍生版）引入 nano-vllm，替换现有的 AWQ fused CUDA kernel + dequant+cuBLAS 二路 dispatch，实现全 M 范围的统一 Marlin 推理。

同时保留现有 kernel 代码（注释掉），作为性能对比基线。

## 设计原则

- **最小改动**：只复制 Marlin 所需文件，不引入 vllm 的 MPLinearKernel 框架
- **一行切换**：`awq.py` 顶部 `_USE_MARLIN = True/False` 控制
- **只覆盖 AWQ 场景**：fp16×int4, group_size=128, has_zero_point
- **只覆盖标准 Linear**：ColumnParallel, RowParallel, MergedColumnParallel, QKVParallel

## 文件改动

### 新增 CUDA 文件（从 vllm 复制）

```
nanovllm/csrc/
├── core/
│   ├── scalar_type.hpp           ← [必需] vllm 的 ScalarType 系统 (~360 lines)
│   └── registration.h            ← [必需] TORCH_LIBRARY_IMPL_EXPAND 宏 (~28 lines)
│
└── marlin/
    ├── kernel.h                  ← Marlin kernel 声明 + MARLIN_KERNEL_PARAMS
    ├── marlin.cuh                ← 公共头文件（常量、辅助函数）
    ├── marlin_dtypes.cuh         ← ScalarType 特化（依赖 scalar_type.hpp）
    ├── marlin_mma.h              ← MMA 指令封装
    ├── marlin_template.h         ← 核心 template kernel (~2081 lines)
    ├── marlin.cu                 ← host wrapper + torch custom op 绑定
    ├── awq_marlin_repack.cu      ← AWQ 格式 → Marlin 格式 weight 重排
    ├── kernel_selector.h         ← [从 vllm 复制] 模板选择器 (由 generate 生成但直接拿来)
    └── sm80_kernel_float16_u4_float16.cu  ← [从 vllm 复制] 预生成的 fp16×u4 实例
```

注意事项：
- `kernel_selector.h` 和 `sm80_kernel_float16_u4_float16.cu` **直接从 vllm 复制**，不运行 `generate_kernels.py`（该脚本会删除目录下所有 kernel 文件，且依赖 jinja2）
- `scalar_type.hpp` 需要将 `namespace vllm` 改为 `namespace nanovllm`

不需要复制的 vllm 文件：GPTQ repack、FP8 kernel、BF16 kernel、INT8 activation kernel、所有其他 sm80/sm89 kernel 文件。

### 修改 Python 文件

```
nanovllm/layers/quantization/awq.py
├── 新增 _USE_MARLIN 标志
├── __init__: 按需分配 workspace buffer (int32[num_SMs])
├── weight_loader: repack qweight + qzeros
└── forward: marlin_gemm（所有 M）

profile/microbench_awq.py
└── 新增 Marlin kernel 的 benchmark 项
```

### 修改构建文件

```
setup.py
├── 添加 marlin/ 目录下的 .cu 文件和 scalar_type.hpp 依赖
├── 编译选项添加 -DMARLIN_NAMESPACE_NAME=nanovllm_marlin
```

Marlin 作为**独立的 CUDAExtension**（`nanovllm._C_marlin`），与现有的 `nanovllm._C` 并列。原因：
- Marlin 的模板实例化大量、编译时间长，分离后不影响现有 AWQ kernel 的快速迭代
- Marlin 使用 `cudaFuncSetAttribute` 动态设置 shared memory，与现有 kernel 架构不同
- 构建错误隔离，方便调试

### 注释掉的代码（保留作对比）

```
nanovllm/csrc/awq/gemm_kernels.cu  ← 保留，不参与编译
awq.py 中旧的 forward dispatch        ← if not _USE_MARLIN 分支
```

## 依赖项处理

### scalar_type.hpp（关键依赖）

从 `../vllm/csrc/core/scalar_type.hpp` 复制，改：
- `namespace vllm` → `namespace nanovllm`
- 只保留 `kFloat16`、`kU4` 两个 type（其余 BF16/FP8 等不需要）

### registration.h

从 `../vllm/csrc/core/registration.h` 复制，改：
- `namespace vllm` → `namespace nanovllm`

### 命名空间避免冲突

`kernel.h` 中使用 `MARLIN_NAMESPACE_NAME` 宏（默认 `marlin`）。通过编译选项隔离：
```
-DMARLIN_NAMESPACE_NAME=nanovllm_marlin
```

所有 kernel template 的显式实例化代码（`sm80_kernel_*.cu`、`kernel_selector.h`）中的 `vllm::` 前缀需要批量替换为 `nanovllm::`。

## 数据流

### 加载时（weight_loader）

```
AWQ checkpoint:
  qweight: int32[IC, OC//8]   ← AWQ pack order [0,4,1,5,2,6,3,7]
  qzeros:  int32[G, OC//8]    ← 同上
  scales:  fp16[G, OC]

     │ awq_marlin_repack(qweight, IC, OC, 4, is_a_8bit=False)
     ▼
  marlin_qweight: int32[IC//16, OC*2]   ← Marlin tile layout
                                               (16×16 tile interleaved)

     │ awq_marlin_repack(qzeros, G, OC, 4, is_a_8bit=False)
     ▼
  marlin_qzeros: int32[G//16, OC*2]

  scales: fp16[G, OC]  ← 形状不变，Marlin 直接使用
```

### 推理时（forward）

```
input: fp16[M, IC]

    │ marlin_gemm(
    │   input, None, marlin_qweight, bias, scales,
    │   None, None, marlin_qzeros,
    │   None, None, workspace, uint4,
    │   M, OC, IC, is_k_full=True,
    │   use_atomic_add=False, use_fp32_reduce=False, is_zp_float=False)
    ▼
output: fp16[M, OC]
```

Marlin kernel 内部：
1. `cp.async` 加载 int4 weight → SMEM（int4 压缩态）
2. 从 SMEM ldmatrix int4 → 寄存器
3. 寄存器内 dequant（sub zero × scale）→ fp16
4. `mma.sync` Tensor Core matmul
5. Double buffer + pipeline overlap

## API 签名

### marlin_gemm（host wrapper，暴露给 Python）

```python
def marlin_gemm(
    a: fp16[M, K],
    c: None,                    # partial sum buffer, AWQ 不需要
    b_q_weight: int32[K//16, N*2],  # repacked weights
    b_bias: fp16[N] | None,
    b_scales: fp16[G, N],
    a_scales: None,             # W4A16 不需要
    global_scale: None,         # 不需要
    b_zeros: int32[G//16, N*2],    # repacked zeros
    g_idx: None,                # AWQ 不需要 group index（无 act_order）
    perm: None,                 # AWQ 不需要 permutation
    workspace: int32[num_SMs],  # 大小 = GPU SM 数量 (RTX4090=128)
    b_q_type: uint4,
    size_m: int, size_n: int, size_k: int,
    is_k_full: bool = True,     # AWQ 无 act_order，始终 True
    use_atomic_add: bool = False,
    use_fp32_reduce: bool = False,
    is_zp_float: bool = False,  # AWQ zero point 是 int4，不是 fp16
) -> fp16[M, N]
```

### awq_marlin_repack

```python
def awq_marlin_repack(
    b_q_weight: int32[K, N//8],  # AWQ format
    size_k: int, size_n: int,
    num_bits: int = 4,
    is_a_8bit: bool = False,
) -> int32[K//16, N*2]          # Marlin format
```

## workspace buffer

```python
num_sms = torch.cuda.get_device_properties(0).multi_processor_count
workspace = torch.zeros(num_sms, dtype=torch.int32, device='cuda')
```

Marlin 内部使用 workspace 做 SM 级别的 atomic 同步。大小只需 `num_SMs` 个 int32。

## 性能验证（microbench_awq.py 更新）

在 `profile/microbench_awq.py` 中新增 Marlin 的 benchmark 项：

```python
# 新增 benchmark 项目
marlin_latency = benchmark_marlin_gemm(M_values, K, N)

# 输出表格扩展一列
# M  | CUDA fused | dequant+cuBLAS | Triton fused | Marlin (new) | 最佳
```

benchmark 覆盖 M=[1, 16, 32, 64, 128, 256, 512, 1024, 4096]，K=N 取实际模型值（如 4096/7168/18432）。

## 性能预期

基于社区数据（Llama 3.1-8B, H100）和 Marlin paper：

| 场景 | 当前 kernel | Marlin | 依据 |
|------|-----------|--------|------|
| Decode (M=1) | CUDA fused OK | ~1.5-2x | 更高 occupancy + double buffer |
| Small batch (M=16) | CUDA fused OK | ~1.5-2x | 更大 tile + 无 bank conflict |
| Medium (M=128) | CUDA fused 开始退化 | ~2x | SMEM int4 省带宽 + cp.async |
| Prefill (M=1024) | dequant+cuBLAS 2.25x 访存 | ~1.3-1.9x | 省去 fp16 物化开销 |

Marlin 在超大 M（>4096）时也会退化到接近 cuBLAS，但**不会像 dequant+cuBLAS 那样有 2.25x 额外访存**。

## 实现步骤

1. **复制依赖文件**：从 vllm 复制 `core/scalar_type.hpp`、`core/registration.h`，改 namespace
2. **复制 Marlin 核心文件**：复制 `kernel.h`, `marlin.cuh`, `marlin_dtypes.cuh`, `marlin_mma.h`, `marlin_template.h`, `marlin.cu`, `awq_marlin_repack.cu`
3. **复制预生成文件**：复制 `sm80_kernel_float16_u4_float16.cu`、`kernel_selector.h`（从 vllm 拿，不运行 generate）
4. **批量替换 namespace**：所有复制来的文件中 `vllm::` → `nanovllm::`
5. **修改 marlin.cu 中 torch op 注册名**：改 namespace 为 `nanovllm`，op 名为 `torch.ops.nanovllm.marlin_gemm` / `torch.ops.nanovllm.awq_marlin_repack`
6. **修改 setup.py**：添加 Marlin 作为第二个 CUDAExtension，编译选项加 `-DMARLIN_NAMESPACE_NAME=nanovllm_marlin`
7. **编译验证**：`python setup.py build_ext --inplace`
8. **修改 awq.py**：添加 `_USE_MARLIN` 标志 + workspace 分配 + weight_loader repack + forward dispatch
9. **修改 microbench_awq.py**：新增 Marlin benchmark 项
10. **正确性验证**：对比 Marlin 输出与 dequant+cuBLAS 基准（max_diff < 1e-2）
11. **性能验证**：microbench 对比各 M 值的 latency

## 风险与缓解

| 风险 | 影响 | 缓解 |
|------|------|------|
| **编译时间**：~130 个模板实例 + 2081 lines template | 首次编译 2-5 分钟 | 独立 CUDAExtension，不影响现有代码迭代 |
| **scalar_type.hpp 适配**：namespace 替换遗漏 | 编译报错 | 系统性地 `sed` 替换所有 `vllm::` 为 `nanovllm::` |
| **qzeros repack**：shape 与 qweight 不同（G×OC vs IC×OC） | repack 后形状错误 | 验证 repack 输出 shape：`G//16 × OC*2` |
| **TP 兼容**：repack 发生在 TP shard 之后 | 每 rank 独立 repack，shape 需对齐 | repack 的输入已经是 shard 后的 weight |
| **Double buffer shared memory 占用** | 可能超出 SMEM 上限 | Marlin 有 stages 参数控制（`stages=4` 是 Ampere 默认） |
| **Marlin 小 M 退化**：`m_block_size_8` 路径性能可能不如预期 | microbench 验证 | 如果退化严重，可 short-circuit 小 M 到 CUDA fused |
