# Marlin 学习笔记

> 本文档针对 fp16×u4 AWQ (group_size=128, 无 act_order) 场景，梳理 Marlin kernel 的核心逻辑、数据流、与 AWQ GEMM 的对比，以及数据格式的差异。

---

## 一、Marlin Kernel 总体架构

### 1.1 宏观数据流

```
A(fp16, m×k) ──→ SMEM ──→ 寄存器 ──┐
                                    ├─── Tensor Core MMA ──→ 累加器 ──→ C(fp16, m×n)
B(u4, k×n)   ──→ SMEM ──→ 寄存器(dequant→fp16) ──┘
S(fp16)       ──→ SMEM ──→ 寄存器 ──→ scale
ZP(u4)        ──→ SMEM ──→ 寄存器 ──→ sub zp
```

### 1.2 文件结构

| 文件 | 行数 | 作用 |
|------|------|------|
| `marlin_template.h` | 2080 | 核心 kernel 模板 `Marlin<>`，所有 tile 配置共用 |
| `marlin.cu` | ~300 | 模板实例化 + torch binding + launch 逻辑 |
| `dequant.h` | 40 | u4→fp16 单指令 dequant (lop3 查表法) |
| `marlin.cuh` | ~50 | 类型定义、MarlinScalarType、Vec 等基础设施 |
| `marlin_mma.h` | 外部 | Tensor Core mma 指令封装 |
| `awq_marlin_repack.cu` | ~200 | AWQ 格式 → Marlin 格式的 repack kernel |

### 1.3 核心 Pipeline 循环 (main loop, marlin_template.h:1788-2075)

Marlin 的核心理念是一个**两阶段软件流水线**：

```
while (slice_iters) {           // 遍历 K 方向的所有 tile
  for (pipe = 0; pipe < stages; ) {  // stages=4, SMEM pipeline
    for (k = 0; k < b_sh_wr_iters; k++) {  // 寄存器级循环
      // 阶段①: SMEM → 寄存器 (含 dequant)
      fetch_to_registers(k+1, pipe%stages)
      fetch_scales_to_registers(k+1, pipe)
      fetch_zp_to_registers(k+1, pipe)

      if (k == 倒数第二步) {
        // 阶段②: 发起下一轮 global → SMEM 异步拷贝 (cp.async)
        fetch_to_shared(...)
        pipe++;
        // 阶段③: 等待 SMEM 就绪
        wait_for_stage()
        init_same_group(...)
      }

      // 阶段④: Tensor Core 矩阵乘法
      matmul(k, pipe - (是否已切换pipe ? 1 : 0))
    }
    slice_iters--;
  }
  // 一个 stripe 完成 → reduce → write
}
```

**关键概念：**
- **stages=4**: SMEM 环形缓冲区的深度，`cp.async` 可同时有 `stages-2=2` 个 outstanding 请求
- **双缓冲**: `frag_a[2]`, `frag_b_quant[2]`, `frag_s[2]` 用 `k % 2` 切换读/写寄存器组
- **cp.async**: 异步 global→SMEM 拷贝，计算与搬运完全重叠

---

## 二、marlin_template.h 逐段解析

### 2.1 基础设施 (L1-175)

| 行号 | 内容 | 说明 |
|------|------|------|
| 22-36 | 命名空间、头文件引用 | `marlin.cuh`, `dequant.h`, `marlin_mma.h` |
| 38-74 | Sm75 回退 | 在 __CUDA_ARCH__ < 750 上定义空 kernel，直接 return |
| 80-100 | `ldsm` | 用 `ldmatrix` PTX 指令从 SMEM 加载 A 矩阵到寄存器（Tensor Core 原生布局） |
| 105-115 | `scale` | 将 scale 乘到 dequant 后的 frag_b 上 (`__hmul2`) |
| 117-128 | `scale_and_sub` | FMA 融合：`frag_b * scale - zp * scale` |
| 130-140 | `sub_zp` | 减 zero-point (`__hsub2`) |
| 142-163 | `scale4` | act_order 专用：4 个独立 scale 分别乘到 4 个 quarter 上 |
| 166-173 | `scale_float` | fp32 版本的 scale 操作 |
| 176-222 | 同步原语 | `barrier_acquire`, `barrier_release`, `wait_negative_and_add` |

### 2.2 Kernel 参数和推导 (L224-360)

| 行号 | 内容 | 说明 |
|------|------|------|
| 242-269 | Kernel 参数 | A/B/C 矩阵指针、scale/zp/g_idx 指针、维度信息、同步锁 |
| 282-303 | 架构检查 | Sm75 (`__CUDA_ARCH__ == 750`) 和 Sm89 的兼容性控制 |
| 304-319 | 类型别名 | `FragA`, `FragB`, `FragC`, `FragS`, `FragZP` 等 |
| 338-352 | 编译期常量推导 | `is_a_8bit=false`, `has_zp=true`, `dequant_skip_flop=true` |
| 360 | `has_act_order` | `= group_blocks == 0` → 对 group_size=128 是 false |

### 2.3 Shared Memory 布局 (L363-764)

| 行号 | 内容 | 说明 |
|------|------|------|
| 363 | `extern __shared__ int4 sh[]` | 动态共享内存基址 |
| 364-365 | `sh_a_s`, `sh_new` | A 的 scale 指针（仅 is_a_8bit 用） |
| 531-545 | A 的 stride 计算 | `a_sh_stage`, `a_gl_rd_delta_o`, `a_sh_wr_iters` |
| 548-558 | B 的 stride 计算 | `b_gl_stride`, `b_sh_stage`, `b_sh_wr_iters` |
| 560-568 | Scale 的 stride 计算 | `s_gl_stride`, `s_sh_stage`, `s_tb_groups` |
| 582-589 | ZP 的 stride 计算 | `zp_gl_stride`, `zp_sh_stage` |
| 744-764 | SMEM 分区分配 | `sh_b` → `sh_red` → `sh_bias` → `sh_g_idx` → `sh_zp` → `sh_s` → `sh_a` |

**SMEM 分配顺序 (从低地址到高地址)：**
```
sh_bias | sh_g_idx(act_order) | sh_zp | sh_s | sh_a
```

### 2.4 寄存器分配和索引 (L767-791)

| 行号 | 内容 | 说明 |
|------|------|------|
| 767 | `frag_a[2][thread_m_blocks]` | A 矩阵寄存器双缓冲 |
| 768 | `frag_b_quant[2][1]` | B 矩阵 u4 打包数据双缓冲 |
| 769 | `frag_c[thread_m_blocks][4][2]` | 累加器 (fp32) |
| 771 | `frag_s[2][4]` | Scale 双缓冲 (不适用于 act_order) |
| 774 | `frag_qzp[2][1]` | 打包后的 u4 zero-point |
| 775-776 | `frag_zp`, `frag_zpf` | 已 dequant 的 zp (fp16) |

### 2.5 Pipeline Lambda 函数 (L841-1160)

| 行号 | 函数 | 说明 |
|------|------|------|
| 841-908 | `fetch_to_shared` | `cp.async` 发起下一轮 A/B/S/ZP 的 global→SMEM 拷贝。A 用 XOR 布局避免 bank conflict，B 跨 stride 拷贝。S/ZP 只在开启新 group 时拷贝 |
| 923-930 | `wait_for_stage` | `cp_async_wait<stages-2>()` + `__syncthreads()` |
| 934-947 | `fetch_to_registers` | SMEM → 寄存器。A 用 `ldmatrix` (Tensor Core 直接布局)，B 直接 cast 读 int4 打包数据 |
| 952-965 | `init_same_group` | act_order 专用：检查当前 tile 是否在同一个 group 内 |
| 967-1022 | `fetch_scales_to_registers` | Scale 从 SMEM 加载到寄存器。分 group_blocks=-1/>=thread_k_blocks/< 三种情况 |
| 1089-1160 | `fetch_zp_to_registers` | ZP 从 SMEM 加载到 `frag_qzp` 寄存器 |

### 2.6 计算核心 (L1176-1292)

| 行号 | 内容 | 说明 |
|------|------|------|
| 1176-1292 | `matmul` | 核心计算函数 |
| 1186-1203 | ZP dequant | 对 `frag_qzp` 做 `dequant_data` 得到 fp16 的 `frag_zp` |
| 1212-1220 | FP8 scale dequant | `dequant_fp8_scales` — 对 fp16×u4 是死代码 |
| 1225-1291 | **主循环 (j=0..3)** | 详见下文 |

**matmul 内部 (j=0..3) 每个 quarter 的处理流程：**

```cpp
for (int j = 0; j < 4; j++) {
  // 1. 从 frag_b_quant 解包两个 int4
  b_quant_0 = frag_b_quant[k%2][0][j];   // 32位，含8个4-bit值
  b_quant_1 = b_quant_0 >> 8;              // 取高4个值

  // 2. dequant: u4→fp16 (lop3 单指令查表)
  dequant_data(b_quant_0, &frag_b0);       // → 4个fp16
  dequant_data(b_quant_1, &frag_b1);

  // 3. dequant_skip_flop=true: 分两步做
  sub_zp(frag_b0, frag_zp[j], 0);          // -= zp
  sub_zp(frag_b1, frag_zp[j], 1);
  scale(frag_b0, frag_s[k%2][j], 0);       // *= s
  scale(frag_b1, frag_s[k%2][j], 1);

  // 4. Tensor Core MMA
  for (int i = 0; i < thread_m_blocks; i++) {
    mma(frag_a[k%2][i], frag_b0, frag_c[i][j][0]);
    mma(frag_a[k%2][i], frag_b1, frag_c[i][j][1]);
  }
}
```

### 2.7 Reduce 和 Write (L1405-1744)

| 行号 | 函数 | 说明 |
|------|------|------|
| 1405-1464 | `thread_block_reduce` | 线程块内 32 个线程的局部 reduce |
| 1466-1580 | `global_reduce_fp16` | 跨 SM 的 fp16 全局 reduce (atomicAdd) |
| 1581-1624 | `global_reduce_fp32` | 跨 SM 的 fp32 全局 reduce |
| 1626-1745 | `write_result` | 累加器 → fp16 → SMEM reorder → 全局写入。per-column scale 在这里做 |

### 2.8 Pipeline 启动和主循环 (L1748-2075)

| 行号 | 内容 | 说明 |
|------|------|------|
| 1748-1785 | `start_pipes` | 冷启动 pipeline：预填充 stages-1 轮 SMEM 数据 |
| 1788-2075 | **主循环** | 上文 1.3 节的 pipeline 循环 |
| 1846-2073 | 列 slice 切换 | 当前 stripe 完成后，处理 column reduce、scale、bias、write_result，然后切到下一列 |

---

## 三、Marlin vs AWQ GEMM 对比

### 3.1 宏观对比

| 方面 | AWQ GEMM | Marlin |
|------|----------|--------|
| **dequant 方法** | 逐值 `(val-zp)*scale` 计算 | `lop3` 单指令查表，4 个值一次性 dequant |
| **数据搬运** | 无特殊优化，直接 global load | `cp.async` 异步 pipeline，计算与搬运完全重叠 |
| **寄存器布局** | 无特殊要求 | Marlin 格式特殊排列，需 repack |
| **Tensor Core** | `m16n8k16` (H100+) | `m16n16k16` (sm75+) |
| **线程块划分** | 简单 1D grid | 3D-like stripe 划分，减少 cross-SM reduce |
| **group size 处理** | 累加器不受影响 | `group_blocks` 控制 scale 复用频率 |
| **双缓冲** | 无 | `frag_a[2]`, `frag_b_quant[2]`, `frag_s[2]` |
| **计算/搬运重叠** | 无重叠 | `stages=4` pipeline, `cp.async` 完全重叠 |
| **SMEM bank conflict** | 无特殊处理 | A 矩阵 XOR 布局保证 bank conflict free |
| **对 act_order 支持** | 简单 | 复杂 (g_idx 查表 + 跨 group 边界) |

### 3.2 微观 Kernel 结构对比

```
AWQ GEMM:                    Marlin:
┌─────────────────┐          ┌──────────────────────────────┐
│ load A(tile)    │          │ start_pipes (prefetch)       │
│ load B(tile)    │          ├──────────────────────────────┤
│ for k in K_tile:│          │ while slice_iters:           │
│   for n in N:   │          │   for pipe in stages:        │
│     dequant     │          │     for k in b_sh_wr_iters:  │
│     compute     │          │       fetch_to_registers    │
│   write_result  │          │       fetch_scales          │
└─────────────────┘          │       fetch_zp              │
                             │       cp_async (next tile)  │
                             │       matmul                │
                             │   thread_block_reduce        │
                             │   global_reduce + write      │
                             └──────────────────────────────┘
```

### 3.3 性能差异的本质原因

Marlin 更快主要有三个原因：

1. **计算/搬运重叠**: AWQ GEMM 是「先 load 再 compute」的串行模式，Marlin 通过 `cp.async` 让 global memory 搬运和 Tensor Core 计算完全重叠，显存带宽利用率更高
2. **dequant 效率**: AWQ 每次读一个 int32 然后逐半字节处理；Marlin 的 `lop3` 单指令可以一次处理 4 个 u4→fp16 的转换
3. **Stripe 划分**: 对宽矩阵（large N），Marlin 的 stripe 划分减少了跨 SM 同步的次数

---

## 四、AWQ 数据格式 vs Marlin 数据格式

### 4.1 为什么需要不同的数据格式？

**Marlin 格式的目的：**
- **SMEM bank conflict free**: A 矩阵用 XOR 映射，B 矩阵用固定的 tile 大小 (16×64) 和特定的 interleave 模式，确保 SMEM 读写不冲突
- **合并的 global memory 访问**: B 矩阵的 tile 布局确保 warp 内的线程访问连续的 global 地址
- **减少寄存器压力**: 特定的 packing 顺序使 Tensor Core MMA 指令可以直接消费寄存器数据

**AWQ 格式的目的：**
- **简单的 column-wise packing**: 方便 Python 层处理和 ONNX 导出
- **直观的 dequant**: 每个 int32 独立打包，不需要 tile 上下文

### 4.2 权重格式对比

| 属性 | AWQ | Marlin |
|------|-----|--------|
| **Tensor shape** | `[K, M/8]` | `[K/16, M*2]` |
| **int32 内 slot 顺序** | `[0,4,1,5,2,6,3,7]` | `[0,2,4,6,1,3,5,7]` |
| **维度含义** | 每行 K, 每列 8 个输出通道 | 每 16 个 K 行, 每列 2 组 64 输出通道 |
| **访问模式** | 按 K 行连续 (row-major) | 按 16×64 tile 分块 (tiled) |
| **数据来源** | AWQ 量化直接产出 | AWQ 格式 repack 得到 |

**AWQ int32 内布局：**
```
bit:  0  4  8  12  16  20  24  28
     ┌──┬──┬──┬──┬──┬──┬──┬──┐
     │w0│w2│w4│w6│w1│w3│w5│w7│
     └──┴──┴──┴──┴──┴──┴──┴──┘
     slot order: [0,4,1,5,2,6,3,7] → 实际位置: [0,1,2,3,4,5,6,7]
```

**Marlin int32 内布局：**
```
bit:  0  4  8  12  16  20  24  28
     ┌──┬──┬──┬──┬──┬──┬──┬──┐
     │w0│w1│w2│w3│w4│w5│w6│w7│
     └──┴──┴──┴──┴──┴──┴──┴──┘
     slot order: [0,2,4,6,1,3,5,7] → 实际位置: [0,1,2,3,4,5,6,7]
```

**Repack 过程** (awq_marlin_repack.cu):
```
1. undo AWQ packing: 用 undo_pack=[0,4,1,5,2,6,3,7] 恢复自然顺序
2. re-pack Marlin: 用 pack_idx=[0,2,4,6,1,3,5,7] 按 Marlin 顺序重排
3. tile 变换: 从 [K, M/8] 变为 [K/16, M*2] (16×64 tile)
   ┌─────────────┐        ┌──────────────────────┐
   │ K x M/8     │  ──→   │ K/16 x M*2          │
   │ (contiguous)│        │ (16x64 tiled)        │
   └─────────────┘        └──────────────────────┘
```

### 4.3 Scale 格式对比

| 属性 | AWQ | Marlin |
|------|-----|--------|
| **Tensor shape** | `[G, M]` | `[G, M]` (同 shape, 但 permuted) |
| **元素类型** | fp16 | fp16 |
| **排列方式** | 自然顺序 `[group][channel]` | 经 `scale_perm` (8×8 矩阵转置) 重排 |
| **目的** | 直观的 group×channel 索引 | 匹配 SMEM 读取模式，避免 bank conflict |

**scale_perm 详解：**

`scale_perm` 是一个 64 元素的排列 (awq.py:34-42)：

```
scale_perm[64] = [0, 8, 16, 24, 32, 40, 48, 56,
                  1, 9, 17, 25, 33, 41, 49, 57,
                  2, 10, 18, 26, 34, 42, 50, 58,
                  3, 11, 19, 27, 35, 43, 51, 59,
                  4, 12, 20, 28, 36, 44, 52, 60,
                  5, 13, 21, 29, 37, 45, 53, 61,
                  6, 14, 22, 30, 38, 46, 54, 62,
                  7, 15, 23, 31, 39, 47, 55, 63]
```

本质上是将 `[group][channel]` 重新排列为 `[channel%8][group][channel/8]`：
- 原始的 scale 是 `[g0c0, g0c1, ..., g0c63, g1c0, ...]`
- permute 后变为 `[g0c0, g1c0, g2c0, ..., g7c0, g0c1, g1c1, ...]`

这样 Marlin kernel 在读取 scale 时，同一个 warp 的线程读取连续的 group，地址对齐没有 gap。

### 4.4 Zero-Point 格式对比

| 属性 | AWQ | Marlin |
|------|-----|--------|
| **Tensor shape** | `[G, M/8]` | `[G, M/8]` (同 shape) |
| **int32 内 slot 顺序** | `[0,4,1,5,2,6,3,7]` (同 AWQ 权重) | `[0,2,4,6,1,3,5,7]` (同 Marlin 权重) |
| **scale_perm 应用** | 无 | 有 (同 scale) |
| **数值范围** | 0-15 (u4) | 0-15 (u4) |

**ZP 转换流程** (awq.py:96-128)：

```
AWQ zp [G, M/8] (packed, AWQ interleave)
    │
    ├─ 1. unpack_cols: int32 → 8×u4 per element
    ├─ 2. undo AWQ interleave
    ├─ 3. apply scale_perm (reshape + permute + ravel)
    ├─ 4. re-pack with Marlin interleave [0,2,4,6,1,3,5,7]
    │
    └─ Marlin zp [G, M/8] (packed, Marlin interleave)
```

### 4.5 格式对比总结

```
AWQ 格式:                            Marlin 格式:

weights:                           weights:
┌─────────────┐                    ┌──────────────────────┐
│ K x M/8     │  ──repack──→      │ (K/16) x (M*2)       │
│ [0,4,1,5,..]│                    │ [0,2,4,6,1,3,5,7]   │
│ row-major   │                    │ 16x64 tiled          │
└─────────────┘                    └──────────────────────┘

scales:                            scales:
┌─────────────┐                    ┌──────────────────────┐
│ G x M       │  ──permute──→     │ G x M                │
│ group-major │                    │ scale_perm rearranged│
│ natural order                   │ group-interleaved    │
└─────────────┘                    └──────────────────────┘

zero-points:                       zero-points:
┌─────────────┐                    ┌──────────────────────┐
│ G x M/8     │  ──转换────→      │ G x M/8              │
│ AWQ interlv │                    │ Marlin interleave    │
│ no permute  │                    │ + scale_perm         │
└─────────────┘                    └──────────────────────┘
```

**关键区别的本质原因：**

1. **Weight 格式** — Marlin 的 tile layout 是为了 **SMEM 无 bank conflict** 和 **Tensor Core 直接消费**。AWQ 的简单 row-major 格式在 Marlin kernel 中会导致 SMEM 访问冲突和低效的 ldmatrix 使用

2. **Scale 排列** — Marlin 的 `scale_perm` 是为了匹配 **SMEM 读取模式**：kernel 内一个 warp 的 32 个线程需要同时读取 32 个 scale 值，如果按 `[group][channel]` 连续存储，相邻线程会访问相同的 bank。`scale_perm` 按 `channel % 8` 交错存储解决了这个问题

3. **Zero-point 排列** — 需要同时匹配 scale 的 `scale_perm` 和 Marlin 的 packing 顺序，因为 zp 和 scale 在 kernel 内被**配对使用**

---

## 五、Marlin 的关键参数

### 5.1 Tile 配置 (marlin.cu)

```
small_batch (M <= 16):        large_batch (M > 16):
  thread_k  thread_n  threads    thread_k  thread_n  threads
  128       128       256         64        256       256
  64        128       128         64        128       128
  128       64        128        128        64        128
```

`thread_k` 和 `thread_n` 决定每个线程块覆盖的 K/N 范围。线程数 128/256 对应 4/8 warps。

### 5.2 `group_blocks`

```
group_blocks = group_size / 16

group_size=128 → group_blocks=8
group_size=64  → group_blocks=4
group_size=32  → group_blocks=2
group_size=-1  → group_blocks=-1 (per-column quantization, 每列独立 scale)
```

控制多少个 K-tile 共享一组 scale/zp。在 `fetch_to_shared` 中，每 `pipe % div_ceil(group_blocks, thread_k_blocks) == 0` 才加载新的 scale/zp。

### 5.3 `dequant_skip_flop`

对于 fp16×u4+zp 场景为 `true`。表示 dequant 时跳过完整的 `(val - zp) * scale` 计算，改为：
1. dequant: 简单地将 u4 映射到 fp16 (用 0x6400 magic)
2. sub_zp: 减去 dequant 后的 zp
3. scale: 乘以 scale

三步分离的好处是：lop3 单指令可以完成 dequant，sub 和 scale 分别由 `__hsub2` 和 `__hmul2` 在后续完成。

### 5.4 `stages`

固定为 4 (sm80+)。控制 `cp.async` ring buffer 深度。`cp_async_wait<stages-2>()` = `cp_async_wait<2>()`，即可同时有 2 个 outstanding 拷贝请求。

### 5.5 `has_act_order`

`= group_blocks == 0`。对 group_size=128, group_blocks=8, 为 false。此路径下 `g_idx` 不会被使用。

---

## 六、与 AWQ GEMM 的数据格式转换

### 6.1 转换对 (awq.py)

```
weight:  _marlin_repack()            — CUDA kernel (awq_marlin_repack.cu)
scales:  _marlin_permute_scales()    — PyTorch tensor permute
zp:      _awq_to_marlin_zero_points() — PyTorch unpack→permute→repack
```

### 6.2 加载时机

转换在模型加载后立即执行 (`loader.py`)：

```python
# 对每个 linear 层:
qweight = marlin_repack(qweight)        # 权重格式转换
scales  = marlin_permute_scales(scales)  # scale 排列
qzeros  = awq_to_marlin_zp(qzeros)      # zero-point 转换
```

转换后的 tensor 替换原始 AWQ tensor，后续 forward 直接使用 Marlin 格式。

### 6.3 转换前后 Tensor 形状变化

```
qweight:  [K, M/8]     →  [K/16, M*2]     (形状变化)
scales:   [G, M]       →  [G, M]           (形状不变, 值重排)
qzeros:   [G, M/8]     →  [G, M/8]         (形状不变, 值重排+重打包)
```

---

## 七、阅读建议

1. **先看 marlin.cu** (~300 行): 了解 kernel 如何被选中和 launch
2. **然后看 dequant.h** (40 行): 理解 u4→fp16 的核心机制 (lop3 魔术)
3. **再看 marlin_template.h 的 main loop** (L1788-2075): 理解 pipeline 的宏观流程
4. **最后看 matmul 函数** (L1176-1292): 理解 dequant+zp+scale+mma 的微观计算流程
5. **对比 awq.py 的转换函数**: 理解 AWQ 和 Marlin 格式之间的映射关系
