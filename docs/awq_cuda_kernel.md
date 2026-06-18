# AWQ 4-bit Quantization & CUDA GEMM Kernel

## 目录

- [1. AWQ 量化原理](#1-awq-量化原理)
- [2. AWQ 数据格式与 Packing](#2-awq-数据格式与-packing)
- [3. Kernel 总体架构](#3-kernel-总体架构)
- [4. 分块并行策略 (Tiling)](#4-分块并行策略-tiling)
- [5. 数据加载与流水线](#5-数据加载与流水线)
- [6. Tensor Core 指令详解](#6-tensor-core-指令详解)
- [7. 去量化 (Dequantize) 技术](#7-去量化-dequantize-技术)
- [8. Split-K 并行](#8-split-k-并行)
- [9. 三路 Dispatch 策略](#9-三路-dispatch-策略)
- [10. 与 vllm 的差异对比](#10-与-vllm-的差异对比)
- [11. 与通用 TensorCore GEMM 优化路线对比](#11-与通用-tensorcore-gemm-优化路线对比)

---

## 1. AWQ 量化原理

### 核心思想

AWQ (Activation-aware Weight Quantization) 是一种面向 LLM 的 4-bit 权重量化方法，来自 MIT-HAN 实验室（Lin et al., 2023）。

关键洞察：**权重中约 1% 的 "salient channels" 对模型精度起决定性作用**。这些 channels 的特征是 activation 幅度大。AWQ 不直接保留这些 channel 为高精度，而是通过对 scaling factor 进行 channel-wise 的 reparameterization 来保护它们。

### 量化流程

1. 将权重矩阵 `W` 按 group（通常 128 列）分组
2. 对每个 group 独立的 4-bit 量化：`q = clamp(round(W / scale + zero), 0, 15)`
3. 对 activation 幅度大的 channel，使用较小的 scale（等价于保留更多精度）
4. 存储 `qweight`（int32[K, M//8]）、`scales`（fp16[G, M]）、`qzeros`（int32[G, M//8]）

### 反量化公式

```
dequantized_w = (qweight - qzeros) * scales
```

这里的操作是 per-group、per-output-channel 的广播运算。

---

## 2. AWQ 数据格式与 Packing

### 4-bit 打包

每个 int32（32 bit）包含 8 个 int4 值。AWQ 的特殊之处在于其 pack order 不是顺序的：

```
int32 中 bit 分布（8 个 4-bit 值）:
  [b0 b1 b2 b3 | b4 b5 b6 b7 | ... | b28 b29 b30 b31]
```

**AWQ 的 pack order**（反序 indexing）：

```
slot index:   0   1   2   3   4   5   6   7
实际位置:     0   4   1   5   2   6   3   7
shift (bits): 0  16   4  20   8  24  12  28
```

即 `AWQ_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]`，`AWQ_SHIFTS = [s*4 for s in AWQ_ORDER]`。

因此解包时：`(packed >> shifts[i]) & 0xF` 得到第 i 个 int4 值。

### 为什么是这种顺序？

这不是 PTX 指令的硬性要求，而是 **算法与反量化函数的 co-design**。

`dequantize_s4_to_fp16x2`（`dequantize.cuh`）是一个手写的 `__device__` 函数，用 4 条 inline PTX 指令（`lop3.b32`, `sub.f16x2`, `fma.rn.f16x2`）完成 int4→fp16x2 的转换，不是标准的 PTX 指令。它用 `BOTTOM_MASK(0x000f000f)` 和 `TOP_MASK(0x00f000f0)` 配合 `>>8` 移位来提取 nibble。

AWQ order 使得反量化函数自然产生**相邻 output 位置**的 fp16x2 对：

```
对原始 int32:
  BOTTOM_MASK(0x000f000f) → 提取 bits 0-3 和 bits 16-19 → fp16x2(val[0], val[1])  ✅ 相邻
  TOP_MASK(0x00f000f0)    → 提取 bits 4-7 和 bits 20-23 → fp16x2(val[2], val[3])  ✅ 相邻

对 i4s >> 8:
  BOTTOM_MASK             → 提取 bits 8-11 和 bits 24-27 → fp16x2(val[4], val[5])  ✅ 相邻
  TOP_MASK                → 提取 bits 12-15 和 bits 28-31 → fp16x2(val[6], val[7])  ✅ 相邻
```

如果是顺序 packing `[0..7]`，同样的 mask 会产生跨 4 位的配对，需要额外 shuffle 指令重排。

**仍然可以用 `lop3.b32` 实现顺序 packing**，只要换不同的 mask 模式即可。当前方案只是最简洁的写法——4 条指令，无需额外 shuffle。后续所有实现（vllm、我们）都沿用这个约定。

### 张量形状约定

```
qweight: int32[IC, OC//8]    -- IC=in_features, OC=out_features
scales:  fp16[G, OC]         -- G=IC/group_size
qzeros:  int32[G, OC//8]     -- 也是 int4 packed

每个 int32 在 qweight 中沿 OC（output）维度 pack 8 个值。
```

> **注意命名混淆**：本文在不同上下文中使用 "M" 指代不同含义。
> Kernel 参数中 `M` = batch_size（token 数），`IC` = in_features，`OC` = out_features。
> 而 AWQ 权重形状 `qweight[IC, OC//8]` 中沿 OC 维打包——有些文档写作 `qweight[K, M//8]`（K=IC, M=OC）。
> 两个 M 含义不同，阅读时需注意区分。

### Qwen3-8B 实例

以 Qwen3-8B 的 q_proj 层为例（`hidden_size=7168`）：

```
GEMM:  C[M, OC] = A[M, IC] @ dequantize(B)[IC, OC]

A = activation (hidden states):   float16[M, IC=7168]
B = qweight (AWQ 打包权重):       int32[IC=7168,    OC//8=7168//8=896]
dequantize(B) = 反量化后权重:     float16[7168,     7168]
C = 输出:                         float16[M=bsz,    OC=7168]

说明:
  IC = in_features  = hidden_size  = 7168  (GEMM 的 K 维/归约维)
  OC = out_features = hidden_size  = 7168  (GEMM 的 N 维)
  M  = batch_size/tokens                  (GEMM 的 M 维)
```

对 gate_proj/up_proj（`intermediate_size=18432`）：

```
GEMM:  C[M, 18432] = A[M, 7168] @ dequantize(B)[7168, 18432]
IC = 7168 OC = 18432
```

对 down_proj：

```
GEMM: C[M, 7168] = A[M, 18432] @ dequantize(B)[18432, 7168]
IC = 18432 OC = 7168
```

### 为何沿 OC 维打包？

这是出于 GEMM 访存模式的优化：GEMM `C[M,OC] = A[M,IC] * B[IC,OC]` 中，B 矩阵沿 IC 维连续存放。AWQ 沿 OC（output）维打包，使得：

- 在 fused kernel 中，每个线程加载的 B 元素来自同一 IC 行、连续的 OC 列——这些列刚好打包在同一个 int32 中
- 同时调用 `dequantize_s4_to_fp16x2` 把 int32 展开成 8 个 fp16，直接参与 Tensor Core 计算

---

## 3. Kernel 总体架构

### 文件结构

```
nanovllm/csrc/awq/
├── gemm_kernels.cu    # 融合 AWQ GEMM CUDA kernel + host wrapper
├── dequantize.cuh     # PTX dequantize device function
└── torch_bindings.cpp # TORCH_LIBRARY / PYBIND11 绑定
```

### 调用链

```
Python (torch.ops.nanovllm.awq_gemm)
  → C++ host wrapper (awq_gemm 函数, gemm_kernels.cu:273)
    → CUDA kernel launch (gemm_forward_4bit_cuda_m16nXk32<N>)
      → dequantize_s4_to_fp16x2 (PTX inline asm)
      → ldmatrix / mma.sync (PTX inline asm)
```

### CPU Host Wrapper

```
torch::Tensor awq_gemm(
    _in_feats [M, IC],     // fp16 activations
    _kernel   [IC, OC//8], // int32 packed weights
    _scaling_factors [G, OC], // fp16 scales
    _zeros    [G, OC//8],  // int32 packed zeros
    split_k_iters          // 沿 K 维的分区数（通常 8）
)
```

**关键逻辑：**

1. **形状校验**：检查 IC 能被 group_size 整除、OC 是 64/128 的倍数
2. **输出分配**：`[split_k_iters, M, OC]` 的 fp16 张量
3. **Kernel 选择**：OC%128==0 用 N=128 kernel，否则 N=64
4. **Grid 计算**：`(M+15)//16 * j_factors * split_k_iters` 个 block
5. **分块归约**：`_out_feats.sum(0)` 在 split_k_iters 维上求和

### Kernel 签名

```cpp
template<int N>  // N = tile size along OC: 64 or 128
__global__ void __launch_bounds__(64)
gemm_forward_4bit_cuda_m16nXk32(
    int G,              // group_size
    int split_k_iters,  // K 分区数
    half* A,            // [M, IC] activations
    int* B,             // [IC, OC//8] packed weights
    half* scaling_factors, // [IC//G, OC]
    int* zeros,         // [IC//G, OC//8]
    int M, int IC, int OC,  // 矩阵维度
    half* C             // [split_k_iters, M, OC] output partial sums
);
```

---

## 4. 分块并行策略 (Tiling)

### 三维 Grid 设计（原生 3D）

Kernel 使用 CUDA 原生 3D grid，`blockIdx` 的三个分量对应三个并行维度：

```
grid = dim3((M + 15) / 16,    // blockIdx.x = M-tile 索引
             j_factors,        // blockIdx.y = OC-tile 索引
             split_k_iters)    // blockIdx.z = split-K 分区索引
```

原始版本把所有维度编码到 `blockIdx.x` 一维中手动 decode（`blockIdx.x % (m_tiles * j_factors)` → M+OC, `blockIdx.x / (m_tiles * j_factors)` → split-K）。3D grid 后 kernel 内不再需要 decode 逻辑，语义等价。

### Tile 大小

| 维度 | Tile 大小 | 说明 |
|------|-----------|------|
| M（行） | 16 | 对应 `m16n8k16` Tensor Core 的 16 行 |
| OC（列） | 64 或 128 | 由模板参数 N 决定 |
| IC（K） | 32 | 每个 inner loop iteration 处理 32 列 |

### 为什么 M-tile 是 16？

Tensor Core `mma.sync.aligned.m16n8k16` 指令一次处理 **16 行 x 8 列** 的输出。每个 warp 内：

- 每个 warp 有 32 个线程，分成两组（threadIdx.x 0-15 为一组）
- 每组加载 8 个 A 值，配合 `ldmatrix` 加载 16x16 的 A 矩阵 tile
- 两个 `m16n8k16` 指令在 N 维拼接，形成一个 warp 的 16x16 输出

### 线程块配置

```cpp
dim3 threads_per_block(32, 2);  // 64 线程 = 2 warps
```

2 个 warp 的分工：

- `threadIdx.y = 0`：处理输出的前 `N/2` 列
- `threadIdx.y = 1`：处理输出的后 `N/2` 列
- `threadIdx.x = 0..31`：warp 内线程，负责加载数据、做 mma

### 线程内寄存器分配

- `C_warp[32]`：32 个 float 累加器（每个 warp 覆盖 `16x16` 输出区域）
- `A_shared_warp[8]`：加载经过 `ldmatrix` 转置后的 A tile
- `B_shared_warp[N/4]`：加载 B tile 的寄存器

### 4.3 Warp 级计算组织

Block 配置 `dim3(32, 2) = 64 线程 = 2 warps`。两个 warp 沿 OC 维度分工：

| Warp | threadIdx.y | 负责行范围 | 负责列范围 |
|------|-------------|-----------|-----------|
| 0 | 0 | M-tile 行 0-7 | OC-tile 前 N/2 列 |
| 1 | 1 | M-tile 行 8-15 | OC-tile 后 N/2 列 |

**MMa 计算流水线**（每个 warp 的内部循环）：

```
k_0_0 外循环（每次 32 列 K）:
  │  加载 A_shared[16×32] + B_shared[32×(N+8)]
  │  __syncthreads()
  │
  └── k_0_1 内循环 × 2（每次 16 列 K）:
        for j_0_4 = 0 .. N/32-1:
          ldmatrix A_shared_warp[4 regs]    ← 16×16 A tile
          ldmatrix B_shared_warp[2 regs]    ← 16×8  B tile
          mma.sync.m16n8k16                 ← 输出 16×8
          ldmatrix B_shared_warp[2 regs]    ← 下一组 16×8
          mma.sync.m16n8k16                 ← 输出 16×8 (拼接成 16×16)
```

**每 warp 的 mma 数量**：

| 循环层级 | 迭代次数 | 每次 mma 数 | 说明 |
|---------|---------|------------|------|
| `k_0_0` 外循环 | `k_bound` 次 | — | split-K 分块 |
| `k_0_1` 内循环 | 2 次 | `2 × N/32` | 每次 16 K 列 |
| **总计 per warp** | — | **`N/8`** | **= 2 × 2 × N/32** |

以 kernel 函数名 `m16nXk32` 的完整含义：
- **m16**: 每个 warp 处理 16 行 M
- **nX**: N 是模板参数（64/128），由两个 warp 各负责 N/2
- **k32**: 每次外循环处理 32 列 K（拆成 2 次 k_0_1 各 16 列适配 m16n8k16）

示例 N=128（Qwen3-8B gate_proj, OC=18432, 一个 block 负责 N=128）：
- 每 warp: `N/8 = 16` 次 mma
- 每 block: 32 次 mma
- 整个 OC 维: `OC/N = 18432/128 = 144` 个 block

**线程内寄存器布局**：

每个 warp 的 `C_warp[32]` 存放 `16 × N/2` 个 f32 累加值。以 N=128 为例：
- 每个 warp 覆盖 `16 × 64 = 1024` 个输出元素
- 32 个线程平分，每个线程持有 32 个 f32
- 每个 mma 指令输出 4 个 f32 到每个线程，分散在 C_warp 的不同 slot 中

**C_ptr 输出偏移公式**：

```cpp
half* C_ptr = C
    + blockIdx.z * M * OC       // split-K 偏移
    + blockIdx.y * N            // OC-tile 偏移
    + threadIdx.y * (N / 2)     // warp 0 → 前 N/2, warp 1 → 后 N/2
    + (threadIdx.x % 4) * 2;    // 线程在 16×N/2 内的列偏移
```

---

## 5. 数据加载与流水线

### 共享内存布局

```
A_shared: [16 x (32 + 8)]  fp16   -- 16 行，每行预留 8 个 padding 避免 bank conflict
B_shared: [32 x (N + 8)]   fp16   -- 32 行，每行预留 8 个 padding
```

### 两级流水线

Kernel 采用两级循环来实现计算与数据加载的 overlap：

```
外层循环: k_0_0（主循环，每次处理 split-k 中的 32 个 K）
  │
  ├─ 从 global memory 加载 A tile → A_shared  (uint4 load)
  ├─ 从 global memory 加载 B tile → dequant → B_shared  (uint4 load + dequant)
  ├─ __syncthreads()
  │
  └─ 内层循环: k_0_1（2 次，每次处理 16 个 K）
      │
      ├─ ldmatrix A_shared → A_shared_warp (寄存器)
      ├─ ldmatrix B_shared → B_shared_warp (寄存器)
      └─ 2 次 mma.sync 累加 → C_warp
```

### Global → Shared 加载模式

**A 矩阵加载（3D grid 版本）：**

```cpp
half* A_ptr = A + ((int)blockIdx.x * 16 +
                   ((int)threadIdx.y) * row_stride_warp +
                   ((int)threadIdx.x) / (32 / 8)) * IC +
               (((int)threadIdx.x) % (32 / 8)) * 8;
```

A_ptr 由两部分组成：

- **行偏移** `(...) * IC`：跳到目标行的起始地址
  - `blockIdx.x * 16` — M-tile 的起始行
  - `threadIdx.y * row_stride_warp` — warp 0 处理行 0-7，warp 1 处理行 8-15
  - `threadIdx.x / 4` — 每个 warp 内每 4 个连续线程共享同一行（32 列由 4 个线程平分，每个 8 列）
- **列偏移** `(...) * 8`：`threadIdx.x % 4 * 8` — 4 个列组的选择

64 个线程如何覆盖 16×32 的 A tile：

```
warp 0 (threadIdx.y=0) → 行 0 ~ 行 7：
                  ┌─ 列 0-7 ─┬─ 列 8-15 ─┬─ 列 16-23 ─┬─ 列 24-31 ─┐
  行0  (t.x=0-3)  │ thr 0    │ thr 1     │ thr 2      │ thr 3       │
  行1  (t.x=4-7)  │ thr 4    │ thr 5     │ thr 6      │ thr 7       │
  行2  (t.x=8-11) │ thr 8    │ thr 9     │ thr 10     │ thr 11      │
  ...             │          │           │            │             │
  行7  (t.x=28-31)│ thr 28   │ thr 29    │ thr 30     │ thr 31      │
                  └──────────┴───────────┴────────────┴─────────────┘

warp 1 (threadIdx.y=1) → 行 8 ~ 行 15：
                  ┌─ 列 0-7 ─┬─ 列 8-15 ─┬─ 列 16-23 ─┬─ 列 24-31 ─┐
  行8  (t.x=0-3)  │ thr 0    │ thr 1     │ thr 2      │ thr 3       │
  ...             │          │           │            │             │
  行15 (t.x=28-31)│ thr 28   │ thr 29    │ thr 30     │ thr 31      │
                  └──────────┴───────────┴────────────┴─────────────┘
```

每个线程用 `uint4`（128 bit = 8 × fp16）一次加载 **8 个连续 fp16**。64 线程 × 8 元素 = 512 元素 = 16行 × 32列的 A tile 全覆盖。

示例（M=47, IC=4096, blockIdx.x=2）：

```python
threadIdx.y=0, threadIdx.x=3:
  row = blockIdx.x*16 + threadIdx.y*row_stride_warp + threadIdx.x//4
      = 32 + 0 + 0 = 32
  col = (threadIdx.x % 4) * 8 = 24
  加载 = A[32, 24 .. 32)  ← 8 个 fp16
```

**B 矩阵加载 + 去量化（3D grid 版本）：**

```cpp
int* B_ptr = B + ((int)threadIdx.y) * (OC / 8) * (256 / N) +
             (((int)threadIdx.x) / (N / 8)) * (OC / 8) +
             (int)blockIdx.y * (N / 8) +
             (((int)threadIdx.x) % (N / 8));
```

每个线程加载 **1 个 int32**（包含 8 个 int4），然后调用 `dequantize_s4_to_fp16x2` 立即反量化为 8 个 fp16，存入 `B_shared`。

### Shared → Register (ldmatrix)

`ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16` 指令从共享内存加载 4 个 8x8 的 fp16 矩阵（即一个 16x16 区域），并：

- **x4**：4 个 8x8 子矩阵拼接成 16x16
- **.trans**：转置加载（将 shared memory 中的行主序转为列主序，适配 Tensor Core 输入格式）

### 写入输出

```cpp
for (int ax1_0_1 = 0; ax1_0_1 < N/32; ++ax1_0_1) {
    for (int local_id = 0; local_id < 8; ++local_id) {
        int row_offset = blockIdx_y / j_factors1 * 16
                       + threadIdx.x / 4 + (local_id % 4) / 2 * 8;
        if (row_offset < M) {  // 边界检查
            C_ptr + ax1_0_1 * 16 + row_offset * OC + ... = __float2half(C_warp[...]);
        }
    }
}
```

`row_offset` 覆盖 block 内的 16 行：通过 `threadIdx.x` 的低 2 bit 做 `local_id` 循环实现。

---

## 6. Tensor Core 指令详解

### mma.sync.aligned.m16n8k16

这是 Ampere 架构（SM >= 80）引入的矩阵乘累加指令：

```
D = A * B + C
  D: f32[16, 8]    -- 累加器（4 个 8 元素片段）
  A: f16[16, 16]   -- 从 A_shared_warp[8] 加载
  B: f16[16, 8]    -- 从 B_shared_warp[8] 加载
```

在我们的 kernel 中，每个 warp 每轮执行 **2 次** mma 指令来覆盖完整的 16x16 输出：

```
第 1 次 mma: C_warp[0:8]  = A @ B[:, 0:8]
第 2 次 mma: C_warp[8:16] = A @ B[:, 8:16]
```

这 2 次 mma 在内层循环 `j_0_4 < N/32` 中完成（N=128 时 N/32=4，即 4 组 2 次 mma，覆盖 64 列；2 个 warp 覆盖全部 128 列）。

### inline PTX vs CUDA C

Kernel 中所有关键操作都是用 **inline PTX assembly** 书写的：

```c
__asm__ __volatile__(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
    : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])     // outputs
    : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),          // A (fp16 as uint32)
      "r"(B[0]), "r"(B[1]),                                  // B (fp16 as uint32)
      "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));          // D (f32)
```

原因：CUDA C 编译器对 Tensor Core 指令的生成不可控，inline PTX 可以精确控制寄存器分配和指令调度。

---

## 7. 去量化 (Dequantize) 技术

### `dequantize_s4_to_fp16x2` 函数

这是 kernel 最精巧的部分。函数签名：

```cpp
__device__ uint4 dequantize_s4_to_fp16x2(uint32_t const& source);
```

**输入**：1 个 `uint32_t`，包含 8 个 int4 值（packed as `[n3 n2 n1 n0]`，每个 ni 是 4 个 4-bit 值的 pack）

**输出**：1 个 `uint4`，即 4 个 `uint32_t`，每个低 16 bit 存一个 fp16 值（共 8 个 fp16 值）

**实现原理**（利用 FP16 格式的特性）：

FP16 格式（IEEE 754）：
```
[1 sign | 5 exponent | 10 mantissa]
```

关键 trick：**int4 编码值 `0..15` 和 FP16 数值之间可以通过 magic number 转换**。

```
1. 将 int4 值放入 FP16 的低 10 bit（尾数域），同时设置 exponent = 0x19（即 25）

   原始 int4（0-15）→ FP16 位模式 = 0x6400 | (int4 << 0)   （对于低 4 bit）
                                = 0x6400 | (int4 << 4)   （对于高 4 bit）
   magic: 0x64006400 是 FP16 中表示 0x19 指数的值 = 2^6 × 2^0 = 2^6

2. 减去 0x6400（"zero point" correction）
   → 0x6400 + int4 - 0x6400 = int4（以 FP16 格式）

3. 对于"高 4 bit"部分还需乘以 1/16
   → (0x6400 + int4×16 - 0x6400) × 1/16 = int4
   → 通过 fma 完成：result × 1/16 + (-64)
   其中 1/16 = 0x2c00（FP16 的 2^-4）
         -64 = 0xd400（FP16 的 -64，用于抵消上一步 magic number 的 16 倍）
```

**关键操作序列**：

```asm
// Step 1: 将低 4 bit 和高 4 bit 分别扩展为 FP16（使用 lop3.b32 进行位操作）
// lop3.b32 是 3 输入的布尔运算，用 immLut 指定的真值表

// Step 2: 转化为实际 int4 值（减去 magic number）
sub.f16x2 h0, h0, 0x64006400

// Step 3: 缩放（对高半部分除以 16）
fma.rn.f16x2 h1, h1, 0x2c002c00, 0xd400d400
```

使用 `lop3.b32` 代替 `&` + `|` 两条指令——一条指令完成 AND+OR 组合。

### 魔术数原理详解

`lop3.b32` 是一条 PTX 三元逻辑指令：`lop3.b32 d, a, b, c, immLut`。对每个 bit 位置，取 a/b/c 的对应 bit，用 immLut 的 8 bit 真值表查找输出。`immLut = 0xEA` 实现的是 `(a & b) | c`：

```
(a,b,c)  → lookup immLut
 111 → bit 7=1  110 → bit 6=1  101 → bit 5=1  100 → bit 4=0
 011 → bit 3=1  010 → bit 2=0  001 → bit 1=1  000 → bit 0=0
```

所以 `lop3.b32 h[0], i4s, 0x000f000f, 0x64006400, 0xEA` 等价于 `h[0] = (i4s & 0x000f000f) | 0x64006400`。

**FP16 magic number 0x6400 的秘密**：

FP16 格式：`[1 sign | 5 exponent(bias=15) | 10 mantissa]`

```
0x6400 = 0110 0100 0000 0000
       = sign=0, exponent=11001(=25), mantissa=0000000000
       = 1.0 × 2^(25-15) = 1.0 × 2^10 = 1024
```

将 int4 值 v(0-15) 写入 mantissa 的低 4 bit，FP16 值变为 `(1 + v/1024) × 1024 = 1024 + v`。减去 1024 即可还原 v。

**四路展开的数据流**：

输入 int32 `i4s` 的 8 个 int4 值按 AWQ order packing：

```
bit 31  28 27  24 23  20 19  16 15  12 11   8 7    4 3    0
├─ v7 ─┤├─ v5 ─┤├─ v3 ─┤├─ v1 ─┤├─ v6 ─┤├─ v4 ─┤├─ v2 ─┤├─ v0 ─┤
   byte 3         byte 2         byte 1         byte 0
```

| 操作 | 输入 | Mask | Magic | 提取的值 | 含意 |
|------|------|------|-------|---------|------|
| h[0] | i4s | 0x000f000f | 0x64006400 | (v0, v1) | byte0&2 的低半字节 |
| h[1] | i4s | 0x00f000f0 | 0x64006400 | (v2×16, v3×16) | byte0&2 的高半字节 |
| h[2] | i4s>>8 | 0x000f000f | 0x64006400 | (v4, v5) | byte1&3 的低半字节 |
| h[3] | i4s>>8 | 0x00f000f0 | 0x64006400 | (v6×16, v7×16) | byte1&3 的高半字节 |

然后做减法/缩放还原真实值：

```
h[0]: sub.f16x2  h[0], 0x64006400          → (v0, v1)
h[1]: fma.rn.f16x2 h[1], 0x2c00, 0xd400   → (v2, v3)
  // 0x2c00 = 1/16, 0xd400 = -64
  // (1024 + v×16) × 1/16 + (-64) = 64+v-64 = v
h[2]: sub.f16x2  h[2], 0x64006400          → (v4, v5)
h[3]: fma.rn.f16x2 h[3], 0x2c00, 0xd400   → (v6, v7)
```

`0xd400` 在 fp16 中是 `-64`（sign=1, exponent=10101=21, 2^(21-15)=64）。高 nibble 的 int4 被嵌入到 mantissa 第 4-7 bit，相当于乘以 16，所以需要用 `×1/16` 缩回，同时修正 magic 偏移：`(1024 + v×16) × 1/16 + (-64) = v`。

最终 uint4 输出 = `{v0, v1, v2, v3, v4, v5, v6, v7}` 共 8 个 fp16。8 条 PTX 指令（4×lop3 + 2×sub + 2×fma）完成 int32→8×fp16 的完整转换。

### 为什么要有独立的 Dequantize 步骤？

因为 AWQ 使用对称量化（只是减去 zero point），计算模式是：
```
B_dequantized = (B_raw >> shifts & 0xF - zeros >> shifts & 0xF) × scales
```

这不只是简单的 int4→fp16 转换，还需要：
1. 减去 zero point（group-wise）
2. 乘以 scaling factor（group-wise）

在 fused kernel 中，这一系列操作在 B 加载到 shared memory 之前完成，
使得 B_shared 中已经存储了完整的 fp16 权重，可以直接给 Tensor Core 使用。

---

## 8. Split-K 并行

### 动机

标准的 GEMM 是 2D grid（M-dim, OC-dim）。Split-K 增加了第三个并行维度（IC/K dim）：

```
标准: grid(M-tiles × OC-tiles, 1)
Split-K: grid(M-tiles × OC-tiles × split_k_iters, 1)
```

### 工作方式

1. 每个 split-K 分区独立执行 `C_partial[split_id] = A × B_partial`，只处理 IC 维的一个分片
2. 所有 split 完成后，在 IC（K）维上求和归约：`C = sum(C_partial, dim=0)`

### 优势

- 增加并行度（GPU 利用率更高），特别是 M 小时
- 每个分区处理更少的 K，减少 shared memory 压力
- 更好的 cache 局部性

### 代价

- 3D 输出缓冲：`[split_k_iters, M, N]`，内存开销 `split_k_iters ×`
- 最后的 sum 归约是一个额外的 kernel launch

### K 维循环与交错访问

kernel 内部通过 `k_bound` 和 `k_0_0` 控制每个 split-K 分区的工作量：

```cpp
int k_bound = (IC / 32 + split_k_iters - 1) / split_k_iters;  // ceil(IC/32 / split_k_iters)
if ((k_bound - 1) * split_k_iters * 32 + blockIdx.z * 32 >= IC) k_bound -= 1;
for (int _k_0_0 = 0; _k_0_0 < k_bound; ++_k_0_0) {
    int k_0_0 = _k_0_0 * split_k_iters + blockIdx.z;  // 交错索引
    ...
}
```

每个迭代加载 32 个 K 元素，**每个分区不是取一段连续的 K**，而是**交错**访问：

```
IC=4096, split_k_iters=8, 每个迭代 32 列

             ┌─── 分区 0 的迭代: k=0, 8, 16, ... ──┐
             │   ┌── 分区 1 的迭代: k=1, 9, 17, ... │
             │   │   ┌── ...                       │
             ▼   ▼   ▼                              ▼
K 维: ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬───
      │ 0  │ 1  │ 2  │ 3  │ 4  │ 5  │ 6  │ 7  │ 8  │ 9  │...
      └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴───
       stride = split_k_iters = 8
```

示例（IC=4096 即 128 个 tile，split_k_iters=8）：

```
k_bound = ceil(128/8) = 16 个迭代/分区

分区 4 (blockIdx.z=4):
  _k_0_0=0  → k_0_0 = 0*8+4 = 4   → 加载 K[128..160)
  _k_0_0=1  → k_0_0 = 1*8+4 = 12  → 加载 K[384..416)
  ...
  _k_0_0=15 → k_0_0 = 15*8+4 = 124 → 加载 K[3968..4000)
```

交错（而非 contiguous 分区）是负载均衡的考虑——如果某些 K 行有稀疏性，交错可以让计算压力更均匀地分布在各个分区。

### 在 vllm / 本项目中的配置

`split_k_iters = 8`。沿 IC（in_features，GEMM 的 K 维）交错分 8 份。所有情况下硬编码为 8，不做 M 动态调整。

---

## 9. 三路 Dispatch 策略

```python
def forward(self, x):
    M = x.size(-2)
    if M < 512:              # 解码阶段（小批量）
        y = torch.ops.nanovllm.awq_gemm(x, qweight, scales, qzeros, 8)
    else:                    # 预填充阶段（大批量）
        weight = awq_dequantize(qweight, scales, qzeros)  # @torch.compile
        y = F.linear(x, weight.t())
```

### M < 512: CUDA 融合 Kernel

- `awq_gemm` 在一个 kernel 中完成：加载 activation → 加载 int4 weight → 反量化 → Tensor Core GEMM
- 避免 materialize 全精度权重矩阵
- 访存密集场景下优势明显（访存量降低到 1/4）
- M 较小时，vectorized 加载的额外开销可以忽略

### M ≥ 512: Dequant + cuBLAS

- 先用 `@torch.compile` 编译的纯 PyTorch kernel 反量化权重
- 然后 `F.linear` 调用 cuBLAS 做标准 fp16 GEMM
- cuBLAS 在大 GEMM 上的优势（tiling、寄存器调度、memory pipeline）远超手写 kernel
- 反量化的额外写入/读取 2×K×M 的开销被大 GEMM 的计算密集特性摊平

### 阈值 M=512 的由来

Microbenchmark 数据（K=4096, N=4096, fp16）：

```
M       | CUDA fused  | dequant+cuBLAS | 最佳
--------|-------------|----------------|------
  1     |    44.9 us  |   339.0 us     | CUDA
  64    |   136.1 us  |   336.9 us     | CUDA
  128   |   285.4 us  |   394.2 us     | CUDA
  256   |   516.7 us  |   618.6 us     | CUDA
  512   |  1201.4 us  |   988.0 us     | cuBLAS ← 反转点
  4096  | 10227.7 us  |  5615.1 us     | cuBLAS
```

CUDA fused kernel 保持优势到 M=512，比 vllm 的阈值（M=256）更宽松。
`@torch.compile` + cuBLAS 在大 M 时优势扩大（M=4096 时 1.8x）。

---

## 10. 与 vllm 的差异对比

### 综合对比

| 方面 | grYe-nano-vllm | vllm |
|------|----------------|------|
| **CUDA kernel 来源** | mit-han-lab/llm-awq → 手改 | mit-han-lab/llm-awq → 重构 |
| **M 限制** | ~~M≤16（已解除）~~ 无 | 无（从开始就无限制） |
| **M 阈值** | M < 512 | M < 256 |
| **CUDA gemm kernel 模板** | `m16nXk32<N>`, N=64/128 | `m16nXk32<N>`, N=64/128 |
| **Dequant 方式** | `@torch.compile` 纯 PyTorch | 自定义 CUDA kernel（`ops.awq_dequantize`） |
| **Triton 路径** | 保留 awq_gemm_triton 作对比 | 仅 `VLLM_USE_TRITON_AWQ=True` 时使用（默认关） |
| **Marlin 升级** | 无 | 自动 AWQ → Marlin 升级 |
| **VLLM_BATCH_INVARIANT** | 无 | 强制走 dequant+matmul 路径 |

### 四种路径的比较

```
grYe-nano-vllm:
  M < 512  → CUDA fused op (Tensor Core)
  M ≥ 512  → @torch.compile dequant + F.linear (cuBLAS)

vllm legacy path:
  M < 256  → ops.awq_gemm (CUDA fused)
  M ≥ 256  → ops.awq_dequantize (CUDA) + torch.matmul

vllm marlin path (默认):
  所有 M   → ops.marlin_gemm (单个 kernel 处理所有大小)

vllm triton path (VLLM_USE_TRITON_AWQ=True):
  M < 256  → awq_gemm_triton
  M ≥ 256  → awq_dequantize_triton + torch.matmul
```

### Dequant 实现的差异

我们的 `@torch.compile awq_dequantize` 使用纯 PyTorch 操作：

```python
@torch.compile
def awq_dequantize(qweight, scales, qzeros, group_size=128, awq_order=None):
    # vectorized 展开: [K, M//8] → [K, M//8, 8] → [K, M]
    w = (qweight.unsqueeze(-1) >> (order * 4)) & 0xF
    w = w.to(torch.float16).reshape(K, M)
    # 同样的操作处理 qzeros
    z = (qzeros.unsqueeze(-1) >> (order * 4)) & 0xF
    z = z.to(torch.float16).reshape(G, M)
    # (w - z) * s 的 group-wise 操作
    return (w.reshape(G, GS, M) - z.unsqueeze(1)) * scales.unsqueeze(1)
```

`@torch.compile`（TorchInductor）将其编译为高效的融合 Triton kernel——移位、mask、减法、乘法全部融合到一个 kernel 中。性能接近手写 CUDA dequant kernel。

vllm 的 CUDA dequant kernel 在同一个 .cu 文件中，也是 inline PTX 实现，但不需要 `@torch.compile` 的 JIT 编译开销。

### Kernel 模板的差异

vllm 在 PR #2723 中把两个独立的 kernel（`m16n128k32` 和 `m16n64k32`）重构为单个模板 `m16nXk32<N>`，主要改动：

- 所有硬编码的 `128` 替换为 `N`（如 `B_shared[32 * (N + 8)]`）
- 所有循环次数使用 `N/32`、`N/16` 等表达式
- 我们 clone 的是重构后的版本

### 线程配置差异

| 方面 | grYe-nano-vllm | vllm |
|------|----------------|------|
| threads_per_block | `(32, 2)` = 64 线程 | `(32, 2)` = 64 线程 |
| blockDim.x | 32（warp 内） | 32 |
| blockDim.y | 2（2 warps） | 2 |
| 每个 block 输出 | `16 × N` | `16 × N` |
| grid 计算 | `(M+15)/16 × j_factors × split_k` | 相同 |
| split_k_iters | 硬编码 8 | 调用者传入（固定 8） |

实际上 CUDA kernel 本身完全一致，区别只在 host wrapper 检查和 dispatch 策略。

---

## 11. 与通用 TensorCore GEMM 优化路线对比

### 标准优化路线

从 naive TensorCore 实现到接近 cuBLAS 性能，社区积累了一套逐步优化的流程。下表列出各阶段优化技术，对照本 kernel 的使用情况。

参考来源：
- [一步步优化 GEMM by Tensorcore](https://zhuanlan.zhihu.com/p/638522893) — FP16 MMA 逐步优化路线
- [Bruce-Lee-LY/cuda_hgemm](https://github.com/Bruce-Lee-LY/cuda_hgemm) — 多版本 HGEMM 实现，含 double buffer / swizzle / cp.async
- [CUDA 实战 hgemm — 超越 cuBLAS](https://blog.csdn.net/WingEdge777/article/details/160992845) — 4 个 kernel 版本从 naive 到超 cuBLAS

### 已使用的优化

| # | 优化技术 | 本 kernel 中的体现 | 收益来源 |
|---|---------|------------------|---------|
| **1** | **Tile 层级划分** (CTA→Warp→MMA) | `grid(×,×,split_k)` → 2 warp → `m16n8k16` | 分治基础，没有这个无从谈起 |
| **2** | **每 block 多 warp 共享 SMEM** | 2 warp/block (dim3(32,2)=64线程)，共享同一块 A_shared | A 的 global→shared 流量减半 |
| **3** | **每 warp 处理更多列 (放大 warp tile)** | N=128 时 warp tile = 16×64，`j_0_4` 循环 N/32=4 次复用 A_shared_warp | ldmatrix/mma 比从 1 提升到 1.6+ |
| **4** | **K 维参与 Grid 划分 (Split-K)** | `grid(,, split_k_iters=8)`，第三维沿 K 交错分区 | M 小时增加并行度，喂饱 GPU |
| **5** | **向量化访存** | A: `*(uint4*)` 128-bit, B: `*(uint32_t*)`, C: 未合并 | A 带宽效率 8x，B 4x |
| **6** | **Shared memory bank conflict 避免 (padding)** | `A_shared[16×(32+8)]`, `B_shared[32×(N+8)]` + stride padding | 消除 bank conflict，SMEM 带宽不降 |
| **7** | **Global memory 访存合并** | A_ptr/B_ptr 公式：连续 threadId → 连续 global 地址 | 充分利用显存带宽 |
| **8** | **累加器在寄存器** | `C_warp[32]` 每个线程 32 个 f32 | **最大跳升**（优化路线中 3.44% → 35.39%） |
| **9** | **ldmatrix.sync 专用指令** | `ldmatrix.sync.aligned.m8n8.x4.trans` | 1 条指令完成 SMEM→RF 重排+转置 |
| **10** | **On-the-fly dequant (global→SMEM 路径)** | B 的 int32 加载后立即反量化再写 SMEM | Global B 带宽降到 1/4 |
| **11** | **预取 scales/zeros** | 在 `ax0_ax1_fused_0` 循环之前加载 | 每个 k-step 少读 N/16 次 global |
| **12** | **编译期常量 + 模板参数** | `template<int N>`, `constexpr row_stride` | 所有除法编译期算好，无 idiv |
| **13** | **Zero-fill 替代分支** | `ld_A_flag ? load : make_uint4(0)` | 避免 mma/write-back 路径的分支 |
| **14** | **f16x2 打包 SIMD 指令** | `sub.f16x2` / `fma.rn.f16x2` | 指令量减半 |
| **15** | **`__restrict__` 指针** | `half* __restrict__ A, int* __restrict__ B` | 消除 aliasing 保护指令 |
| **16** | **Tile shape 适配硬件约束** | N=64/128 (OC tile)，与 OC 对齐 | 保证 ldmatrix 与 mma 的最佳匹配 |

### 未使用的常见优化

| # | 优化技术 | 为什么本 kernel 没做 |
|---|---------|-------------------|
| **X1** | **XOR Swizzle** (代替 padding 消 bank conflict) | Padding 更简单，`8/136=5.9%` SMEM 浪费可接受。Swizzle 需要额外 XOR 和复杂地址计算 |
| **X2** | **Async Copy `cp.async`** (global→SMEM 异步拷贝) | B 路径插了 dequant（int32→fp16），`cp.async` 是纯拷贝不能用。A 路径纯拷贝可以做，但单独为 A 做 double buffer 会使同步逻辑复杂化 |
| **X3** | **Double Buffer (ping-pong SMEM)** (计算与加载 overlap) | K tile 仅 32（内循环 2 次），overlap 窗口太小；SMEM 翻倍降低 occupancy，得不偿失 |
| **X4** | **Multi-stage Pipeline (3+ 级流水线)** | 同理 K tile 太小。Ampere 上 stage>2 反而因 SMEM 占用高导致 -28% 性能（PyGPUkit 实测） |
| **X5** | **ThreadBlock Swizzle** (L2 cache locality) | M 小（decode）时 block 数不多，L2 优化收益有限 |
| **X6** | **Warp Specialization** (producer/consumer) | Hopper SM90+ 特性，Ampere 不支持 |
| **X7** | **C 输出向量化写回** (`uint4` 合并写) | `row_offset < M` 边界检查导致无法简单合并 |

### 路线图对照

```
Naive MMA baseline                              ~3% cuBLAS
  │
  ├─ 向量化访存 + 访存合并                        ~5%
  │   本 kernel: ✅ (A: uint4, B: uint32)
  │
  ├─ Bank conflict 消除 (padding/swizzle)        ~5%
  │   本 kernel: ✅ (padding)
  │
  ├─ cp.async 异步拷贝                           ~5%
  │   本 kernel: ❌ (B 路径有 dequant)
  │
  ├─ 累加器从 SMEM 移到寄存器                     ~35%  ← 最大跳升
  │   本 kernel: ✅ (C_warp[32])
  │
  ├─ Double buffer (load/compute overlap)        ~39%
  │   本 kernel: ❌ (K tile 太小)
  │
  ├─ Tile shape tuning (多 warp + 大 warp tile)  ~62%
  │   本 kernel: ✅ (2 warp, 16×N/2 per warp)
  │
  ├─ ThreadBlock swizzle (L2 locality)           ~68%
  │   本 kernel: ❌
  │
  └─ ldmatrix 指令                               ~74%
      本 kernel: ✅ (ldmatrix.sync.x4.trans)
```

### 这个 kernel 的取舍逻辑

```
B 的 global→shared 路径中插了 dequant（int32 → fp16 转换）
  → 无法用 cp.async（纯拷贝指令）
  → 不用 cp.async 则 double buffer 的 overlap 效果有限
  → K tile 太小（32），pipeline 收益微乎其微
  → 放弃 double buffer，节省 shared memory 提升 occupancy
  → 走"简单直接"路线：单 buffer + sync + 算
```

最终定位：**AWQ 量化场景下针对 decode 阶段（M<<512）够用的性能**。计算密集场景（prefill）交给 dequant+cuBLAS。在访存带宽受限的 decode 场景中，int4 weight 的 4 倍带宽节省＞所有 pipeline 优化的总和。
