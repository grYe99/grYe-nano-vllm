# nano-vllm AWQ 量化支持方案

> 借鉴 vllm 对 AWQ 的支持，分析 nano-vllm 加载 AWQ 量化模型所需的模型层面和算子层面改动。

---

## 1. AWQ 量化原理与数学公式

### 1.1 量化公式

AWQ 的核心量化公式：

```
quantize:   w_int4 = clamp(round(w_fp16 / scale) + zero, 0, 15)
dequantize: w_fp16 = (w_int4 - zero) × scale
```

其中：
- **`w_fp16`**：原始的 fp16 权重（每层的参数矩阵，如 `Q_proj.weight`）
- **`scale`**：缩放因子，fp16，**一组的元素共享一个 scale**
- **`zero`**：零点偏移，int4（0~15），**一组的元素共享一个 zero**
- **`w_int4`**：4-bit 无符号整数（0~15）

### 1.2 关键参数及其作用

以 `model.layers.0.self_attn.q_proj` 为例，假设 `in_features=4096`, `out_features=4096`, `group_size=128`：

```python
num_groups = in_features // group_size  # = 4096 ÷ 128 = 32
pack_factor = 32 // 4                   # = 8（一个 int32 能 pack 8 个 4-bit 值）
```

**`qweight`** — Packed 4-bit 量化权重：
```
形状: (out_features, in_features // 8) = (4096, 512)  → dtype int32
大小: 4096 × 512 × 4B = 8 MB
对比: 原始 fp16 weight 大小 = 4096 × 4096 × 2B = 32 MB
节省: 压缩比 4×
```
每个 int32 元素中 pack 了 8 个连续的 4-bit 权重值。**注意**：这个 int32 不存储 int4 值的"含义"（它不是一个真正的 int32 整数），它只是一个 32-bit 的容器。

**`scales`** — 浮点缩放因子：
```
形状: (num_groups, out_features) = (32, 4096)  → dtype fp16
大小: 32 × 4096 × 2B = 256 KB
```
每组（128 个输入特征）对应一个 scale 向量（长度 = out_features）。每个 scale 的作用是将该组中所有 int4 权重的值"恢复"到正确的 fp16 范围。

**`qzeros`** — Packed 4-bit zero points：
```
形状: (num_groups, out_features // 8) = (32, 512)  → dtype int32
大小: 32 × 512 × 4B = 64 KB
```
和 `qweight` 一样，8 个 4-bit zero 值 pack 到一个 int32 中。每个 zero point 的作用是在反量化时做对称平移。

**总结：原始 fp16 weight 32 MB → qweight 8 MB + scales 0.25 MB + qzeros 0.06 MB = 8.3 MB，净节省约 3.8×。**

### 1.3 Weight Unpack 的位操作细节

理解从 int32 中提取 4-bit 值需要精确的位操作。假设一个 int32 的 32 个 bit 为 `[b31 b30 ... b1 b0]`：

**GPTQ 标准 unpack**（8 个 int4 按 `[0,1,2,3,4,5,6,7]` 顺序排列）：
```
int32 [b31 b30 b29 b28 | b27 b26 b25 b24 | ... | b3 b2 b1 b0]
  idx    7               6                       0

提取 idx=0: (int32 >> 0)  & 0xF  → 取 b3 b2 b1 b0
提取 idx=1: (int32 >> 4)  & 0xF  → 取 b7 b6 b5 b4
提取 idx=2: (int32 >> 8)  & 0xF  → 取 b11 b10 b9 b8
...
提取 idx=7: (int32 >> 28) & 0xF  → 取 b31 b30 b29 b28
```

**AWQ unpack**（8 个 int4 按 `[0,4,1,5,2,6,3,7]` 顺序排列）：
```
int32 [b31 b30 b29 b28 | b27 b26 b25 b24 | ... | b3 b2 b1 b0]
  idx    7                       0

提取 idx=0: (int32 >> 0)  & 0xF  → 取 b3 b2 b1 b0  (与标准相同)
提取 idx=1: (int32 >> 16) & 0xF  → 取 b19 b18 b17 b16  ← 不同！
提取 idx=2: (int32 >> 4)  & 0xF  → 取 b7 b6 b5 b4   ← 不同！
...
```

vllm Triton kernel 中构造 reverse 顺序张量的代码：
```python
# 生成 [0, 4, 1, 5, 2, 6, 3, 7]
reverse_awq_order = (tl.arange(0, 2) * 4)[None, :] + tl.arange(0, 4)[:, None]
reverse_awq_order = reverse_awq_order.reshape(8)  # → [0, 4, 1, 5, 2, 6, 3, 7]

# 转为 bit shift 值： [0, 16, 4, 20, 8, 24, 12, 28]
shifts = reverse_awq_order * 4
```

然后 `(iweights >> shifts) & 0xF` 一步完成 unpack + reorder。

### 1.4 完整反量化计算示例

假设 `group_size=4`（简化示例），用 Python 伪代码展示反量化全程：

```python
# 一个 group 有 4 个权重值：w[0], w[1], w[2], w[3]
# 这 4 个权重被量化后 pack 到一个 int32 中（实际是 8 个，这里简化为 4）

# 从检查点加载的原始数据
qweight_packed = 0x1234  # 举例用的 int32
scale = 0.5               # fp16 值
zero_packed = 0xAB        # pack 了 4 个 4-bit zero 的 int32

# Step 1: Unpack qweight
# AWQ 顺序: idx [0]在 bit[0:4], idx[1]在 bit[16:20], idx[2]在 bit[4:8], idx[3]在 bit[20:24]
w0 = (qweight_packed >> 0)  & 0xF
w1 = (qweight_packed >> 16) & 0xF
w2 = (qweight_packed >> 4)  & 0xF
w3 = (qweight_packed >> 20) & 0xF
# 结果: w_int4 = [w0, w1, w2, w3]，每个值范围 0~15

# Step 2: Unpack zeros
z0 = (zero_packed >> 0)  & 0xF
z1 = (zero_packed >> 16) & 0xF
z2 = (zero_packed >> 4)  & 0xF
z3 = (zero_packed >> 20) & 0xF

# Step 3: Dequantize（这个 group 内的所有 4 个值共享同一个 scale）
w_fp16_0 = (w0 - z0) * scale
w_fp16_1 = (w1 - z1) * scale
w_fp16_2 = (w2 - z2) * scale
w_fp16_3 = (w3 - z3) * scale
```

### 1.5 内存布局：为什么 AWQ 的格式是"非标准"的

假设权重矩阵的形状为 `[N, K]`（N = out_features, K = in_features），要 pack 到 `[N, K/8]` 的 int32 矩阵中：

**标准布局（GPTQ 风格）** — 沿 K 维度 pack：
```
原始 fp16 矩阵 W[N, K]:
  row 0: [w00 w01 w02 ... w0,K-1]
  row 1: [w10 w11 w12 ... w1,K-1]
  ...

pack 后 int32 矩阵 Q[N, K/8]:
  row 0: [pack(w00..w07), pack(w08..w0,15), ...]
  row 1: [pack(w10..w17), pack(w18..w1,15), ...]
  ```

**AWQ 布局** — 沿 N 维度 pack（注意这是非标准的）：
```
原始 fp16 矩阵 W[N, K]:
  col 0: [w00 w10 w20 ... wN-1,0]
  col 1: [w01 w11 w21 ... wN-1,1]
  ...

pack 后 int32 矩阵 Q[N/8, K]:
  col 0: [pack(w00..w70), pack(w80..w15,0), ...]
  col 1: [pack(w01..w71), pack(w81..w15,1), ...]
```

两种布局的 shape 物理上相同（`N×K/8` 和 `N/8×K` 元素数相等），但**连续的 8 个 4-bit 值来自不同的维度**。

这意味着 AWQ 的 `qweight[i][j]` 在显存中相邻的 8 个 int4 值对应的是**同一 input 特征下 8 个不同的 output neuron**，而不是同一 output neuron 的 8 个 input 特征。对 kernel 设计的影响：

- **standard packing**：`A[M,K] @ B[K,N]` 时，B tile 按 K 维度加载，连续 int32 正好是连续的 K 值，cache 友好
- **AWQ packing**：连续 int32 里的 8 个 int4 值跨度是 N 维度，K 维度的连续性被打乱了。AWQ kernel 需要做额外的 shuffle

### 1.6 config.json 中的量化配置

AWQ 模型的 `config.json` 会包含：

```json
{
  "quantization_config": {
    "quant_method": "awq",
    "bits": 4,
    "group_size": 128,
    "zero_point": true
  }
}
```

当前 nano-vllm 的 `Config` 会通过 `AutoConfig.from_pretrained()` 读取这个 config，但完全不处理其中的 `quantization_config` 字段。

---

## 2. AWQ 推理的两种实现路径

### 2.1 核心矛盾：带宽 vs 计算

AWQ 推理的核心问题是：**权重是 int4 的，但 GPU 的 Tensor Core 只支持 fp16/bf16 的输入**。所以无论怎么做，最终给 Tensor Core 的矩阵乘法的操作数都必须以 fp16 形式存在。问题在于**什么时候做这个类型转换**：

- **路径 A**：一次性把整个 int4 权重矩阵反量化成 fp16，然后走标准 matmul
  - 代价：显存中多了一个完整的 fp16 副本，多了 4× 的 HBM 读取量
  - 优势：反量化后的 matmul 可以使用高度优化的 cuBLAS

- **路径 B**：在 matrix multiply 的过程中，按 tile 逐块反量化
  - 代价：需要自己写 kernel，不能直接用 cuBLAS
  - 优势：权重始终以 int4 形式存放在 HBM 中，反量化发生在寄存器/tile 层面，**省了 4× 的 HBM 读取**

选择依据是 **M 的大小（batch token 数）**：

| | Prefill (M 大) | Decode (M 小) |
|---|---|---|
| 瓶颈 | **Compute-bound** — GPU 算力打满 | **Memory-bound** — HBM 带宽打满 |
| 最贵的操作 | 矩阵乘法本身 | 从 HBM 读权重 |
| 路径选择 | **路径 A** — 反量化代价被大 GEMM 摊薄 | **路径 B** — 省 4× HBM 读取最关键 |

### 2.2 路径 A：Dequantize + FP16 Matmul

```
输入 fp16 x [M, K]
    ┌─────────────────────────────────────────┐
    │ awq_dequantize kernel:                   │
    │   1. 按 tile 加载 qweight int32 [N, K/8] │
    │   2. 逐 int32 unpack 出 8 个 int4 值     │
    │   3. 按 group 查找 scales/qzeros 并反量化 │
    │   4. 写回 fp16 weight [N, K]             │
    └─────────────────────────────────────────┘
    ↓
torch.matmul(x, weight)  ← cuBLAS dispatch:
     ├─ M ≥ 阈值 → GEMM kernel (Tensor Core)
     └─ M < 阈值 → GEMV kernel (memory-optimized)
```

`awq_dequantize` 的输出是一个临时的 fp16 weight。vllm 中 `awq_dequantize` kernel（Triton 版）的 block 划分：

```
grid: (X_blocks, Y_blocks)
  X_blocks = num_cols // BLOCK_SIZE_X    # qweight 的列数方向
  Y_blocks = num_rows // BLOCK_SIZE_Y    # qweight 的行数方向

每个 block 处理:
  qweight[Y*BLOCK_SIZE_Y : (Y+1)*BLOCK_SIZE_Y,
          X*BLOCK_SIZE_X : (X+1)*BLOCK_SIZE_X]  # int32 tile
  → 反量化后对应
  weight[Y*BLOCK_SIZE_Y : (Y+1)*BLOCK_SIZE_Y,
         X*BLOCK_SIZE_X*8 : (X+1)*BLOCK_SIZE_X*8]  # fp16 tile
```

一个 BLOCK_SIZE_X=32, BLOCK_SIZE_Y=32 的 block 读取 32×32=1024 个 int32（4KB），反量化为 32×256=8192 个 fp16（16KB），然后写回。

**优点：** 实现简单，仅需要 awq_dequantize 一个 kernel，matmul 交给 cuBLAS
**缺点：** 每层 forward 都要展开一次权重；展开后的 fp16 weight 存在**全局显存**中，写回和后续读取都消耗 HBM 带宽

### 2.3 路径 B：Fused AWQ GEMM

fused 的思路是在一个 kernel 内部完成以下流程：

```
对 K 维度的每个 tile:
  1. 加载 A tile (fp16):          A[pid_m*BM : (pid_m+1)*BM, k : k+BK]
  2. 加载 B tile (int4 packed):   B[k : k+BK, pid_n*(BN/8) : (pid_n+1)*(BN/8)]
  3. 在寄存器/Shared Memory 内反量化 B:
     unpack int4 → 减去 zero → 乘以 scale
  4. 用 dequant 后的 fp16 B tile 和 A tile 做 tl.dot() 累加
```

**关键差异：** B tile 以 int4 形式从 HBM 加载（减少 4× 带宽），反量化在寄存器/shmem 中完成，结果直接进算子流水线，不写回全局显存。整层 forward 只需要这一个 kernel。

fused kernel 具体实现细节见第 4 节。

### 2.4 路径 B 的三种子方案

| 方案 | 实现 | SM 要求 | 性能 | 开发难度 |
|------|------|---------|------|---------|
| **Triton awq_gemm** | Triton `@jit` kernel | 不限 (Triton 支持) | 基准 | 低 |
| **CUDA awq_gemm** | 自定义 CUDA kernel m16nXk32 | SM75+ | ~1.5× vs Triton | 高 |
| **Marlin** | 自定义 CUDA + PTX + repack | SM75+ | ~2× vs Triton | 最高（需 repack） |

---

## 3. nano-vllm 需要修改的文件

### 3.1 配置层：检测 AWQ

**文件：** `nanovllm/config.py`

新增字段：
```python
@dataclass
class Config:
    ...
    quant_method: str | None = None   # "awq", None=不量化

    def __post_init__(self):
        ...
        # 从 hf_config 的 quantization_config 中检测
        qc = getattr(self.hf_config, "quantization_config", None)
        if qc and qc.get("quant_method") == "awq":
            self.quant_method = "awq"
```

### 3.2 线性层：替换 forward + weight_loader

**文件：** `nanovllm/layers/linear.py`（或新建 `nanovllm/layers/quantization/`）

需要对以下线性层实现 AWQ 版本（每个都要替换 `weight` 参数为 `qweight/qzeros/scales`，并实现新的 `weight_loader` 和 `forward`）：

| 原线性层 | AWQ 版本 | 说明 |
|-----------|----------|------|
| `QKVParallelLinear` | `AWQQKVParallelLinear` | q_proj, k_proj, v_proj 各有一组 qweight/qzeros/scales |
| `MergedColumnParallelLinear` | `AWQMergedColumnParallelLinear` | gate_proj, up_proj 各有一组 |
| `RowParallelLinear` | `AWQRowParallelLinear` | o_proj, down_proj |
| `ParallelLMHead` | 可选 | lm_head 通常不用量化 |

**核心逻辑示例（以简单的 `ColumnParallelLinear` 为例）：**

```python
class AWQColumnParallelLinear(nn.Module):
    def __init__(self, input_size, output_size, group_size=128, bias=False):
        super().__init__()
        self.group_size = group_size
        self.num_groups = input_size // group_size
        self.pack_factor = 8  # 32/4

        # 三个量化参数
        self.qweight = nn.Parameter(torch.empty(output_size, input_size // self.pack_factor), dtype=torch.int32)
        self.qzeros = nn.Parameter(torch.empty(self.num_groups, output_size // self.pack_factor), dtype=torch.int32)
        self.scales = nn.Parameter(torch.empty(self.num_groups, output_size), dtype=torch.float16)

    def weight_loader(self, param, loaded_weight, shard_id=None):
        # 根据 shard_id 做 TP 分片后 copy
        ...

    def forward(self, x):
        if x.size(0) >= 256 or use_dequant_path:
            # 路径 A: 反量化后 matmul
            weight = awq_dequantize(self.qweight, self.scales, self.qzeros)
            return F.linear(x, weight, self.bias)
        else:
            # 路径 B: fused awq_gemm
            return awq_gemm(x, self.qweight, self.scales, self.qzeros)
```

**关键设计决策：** AWQ 的 packed 参数**不包含原始 fp16 weight**。这意味着只有路径 B 能真正利用量化加速。路径 A 虽然写起来简单（反量化后仍然是 fp16 matmul），但失去了 kernel 层面的加速优势。一个折中是**在 weight_loader 时就反量化好**，这样运行时仍是标准 fp16 matmul，但显存节省不再存在。

### 3.3 权重加载器：支持 AWQ 参数名

**文件：** `nanovllm/utils/loader.py`

当前 loader 通过 `packed_modules_mapping` 将 HF 的 `q_proj`/`k_proj`/`v_proj` 映射到模型的 `qkv_proj` 参数。AWQ 检查点中每个子模块有三个文件：

```
q_proj.qweight   → qkv_proj 的 qweight 参数，附带 shard_id="q"
q_proj.qzeros    → qkv_proj 的 qzeros 参数，附带 shard_id="q"
q_proj.scales    → qkv_proj 的 scales 参数，附带 shard_id="q"
k_proj.qweight   → qkv_proj 的 qweight 参数，附带 shard_id="k"
...
```

需要扩展 `load_model` 函数，使其能识别 `qweight`/`qzeros`/`scales` 后缀，并正确设置 `shard_id` 和 param 类型。关键改动在 `packed_modules_mapping`：

```python
# 当前只映射主干名称
packed_modules_mapping = {
    "q_proj": ("qkv_proj", "q"),
    "k_proj": ("qkv_proj", "k"),
    ...
}
# 需要扩展为三个后缀
# q_proj.qweight → qkv_proj.qweight, shard_id="q"
# q_proj.qzeros  → qkv_proj.qzeros,  shard_id="q"
# q_proj.scales  → qkv_proj.scales,   shard_id="q"
```

### 3.4 模型构建：条件选择量化层

**文件：** `nanovllm/models/qwen3.py`

根据 `config.quant_method` 选择使用原始线性层还是量化版本：

```python
class Qwen3Attention(nn.Module):
    def __init__(self, ..., quant_method=None):
        if quant_method == "awq":
            self.qkv_proj = AWQQKVParallelLinear(...)
            self.o_proj = AWQRowParallelLinear(...)
        else:
            self.qkv_proj = QKVParallelLinear(...)
            self.o_proj = RowParallelLinear(...)
```

### 3.5 KV Cache 显存计算

**文件：** `nanovllm/engine/model_runner.py`

AWQ 量化后 weight 显存显著减少，`allocate_kv_cache` 中 `torch.cuda.mem_get_info()` 反映的可用显存会更大，从而自动分配更多 KV cache block。理论上不需要额外改动，但如果需要对量化后的 KV cache 做特殊处理，需要考虑。

---

## 4. AWQ Kernel 实现深度解析

### 4.1 `awq_dequantize` Triton Kernel

**目的：** 将 AWQ packed 权重展开为 fp16，用于路径 A（dequant + matmul）。

**输入/输出：**
```
qweight: int32[K, N/8]     ← AWQ packed: [0,4,1,5,2,6,3,7] order, dim=0 packing
scales:  fp16[K/G, N]      ← G = group_size (通常是 128)
qzeros:  int32[K/G, N/8]   ← zero points, 也是 AWQ packed
输出:    fp16[K, N]         ← 反量化后的完整权重矩阵
```

**Grid 划分（2D）：**
```
pid_x (列方向): X_blocks = ceil(N/8 / BLOCK_SIZE_X)
pid_y (行方向): Y_blocks = ceil(K / BLOCK_SIZE_Y)

每个 block 处理:
  输入: qweight[pid_y*BY : (pid_y+1)*BY, pid_x*BX : (pid_x+1)*BX]  → int32 tile [BY, BX]
  输出: result[pid_y*BY : (pid_y+1)*BY, pid_x*BX*8 : (pid_x+1)*BX*8]  → fp16 tile [BY, BX*8]
```

**Kernel 内部流程（一个 block）：**

```
Step 1: 加载 qweight tile [BY, BX] 到寄存器
        iweights = tl.load(qweight + offsets)
        ↓
Step 2: 从 [BY, BX] interleave 到 [BY, BX*8]
        iweights = tl.interleave(iweights, iweights) × 3
        # 这是 Triton 技巧：用 interleave 把一个 int32 "展开"为 8 份
        # 配合下面的 shift 操作，一次展开 8 个 4-bit 值
        ↓
Step 3: 构建 AWQ reorder shifts
        reverse_awq_order = [0, 4, 1, 5, 2, 6, 3, 7]
        shifts = reverse_awq_order * 4  → [0, 16, 4, 20, 8, 24, 12, 28]
        shifts → broadcast 到 [BY, BX*8]
        ↓
Step 4: Unpack + reorder（一步完成）
        iweights = (iweights >> shifts) & 0xF
        # 每个 4-bit 值被提取出来并重新排序为 [0,1,2,3,4,5,6,7]
        ↓
Step 5: 加载 qzeros tile（形状 [BY/G, BX] → broadcast 到 [BY, BX*8]）
        zeros = tl.load(zeros_ptr + zero_offsets)
        zeros = tl.interleave(zeros, zeros) × 3  # 和 qweight 同样的展开
        zeros = (zeros >> shifts) & 0xF           # 同样 unpack + reorder
        zeros = tl.broadcast_to(zeros, [BY, BX*8])
        ↓
Step 6: 加载 scales tile（形状 [BY/G, BX*8] → broadcast 到 [BY, BX*8]）
        scales = tl.load(scales_ptr + scale_offsets)
        scales = tl.broadcast_to(scales, [BY, BX*8])
        ↓
Step 7: 反量化
        iweights = (iweights - zeros) * scales
        iweights = iweights.to(fp16)
        ↓
Step 8: 写回 result 的对应区域
        tl.store(result_ptr + result_offsets, iweights)
```

**interleave 技巧详解：**

Triton 的 `tl.interleave(x, x)` 将输入沿最后一维交替排列：
```
输入 [BY, BX]:     [a0 a1 a2 ... a{BX-1}]
interleave([a, a]): [a0 a0 a1 a1 a2 a2 ... a{BX-1} a{BX-1}]
```

调用 3 次 interleave 后，`[BY, BX]` → `[BY, BX*8]`，每个原始 int32 元素被复制了 8 次。然后 `>> shifts` 从每次复制中提取不同的 4-bit 槽位，最终得到一个 `[BY, BX*8]` 的 fp16 tile，其中每个元素对应一个解包后的 4-bit 权重值。

---

### 4.2 `awq_gemm` Triton Fused Kernel

**目的：** 在一个 kernel 内完成 dequantize + matmul，无需中间写回 fp16 权重。

**输入/输出：**
```
input:   fp16[M, K]        ← 激活输入
qweight: int32[K, N/8]     ← AWQ packed 权重
qzeros:  int32[K/G, N/8]   ← AWQ packed zero points
scales:  fp16[K/G, N]      ← 浮点 scales
输出:    fp16[M, N]         ← 矩阵乘结果（累加后 sum）
```

**Grid 划分（3D）：**
```
                                 ┌─────────────────────┐
                                 │    M 方向 (pid_m)    │
                                 │  ceil(M/BLOCK_SIZE_M)│
                                 ├─────────────────────┤
  grid shape: (pid_m * pid_n,    │    N 方向 (pid_n)    │
                split_k_iters)   │  ceil(N/BLOCK_SIZE_N)│
                                 ├─────────────────────┤
                                 │  K 方向 (pid_z)      │
                                 │  split_k_iters       │
                                 └─────────────────────┘
```

3D grid 的原因：split-k 并行策略。

#### Split-K 并行策略

**问题：** 对 M 和 N 的 tiling 是自然的（每个 block 处理输出 C 的一部分），但 K 维度不好切——K 维度的累加是**归约**操作。

**Split-K 解决方式：**
- 将 K 维度分成 `split_k_iters` 份
- 每个 split 独立计算部分和 `C_partial[M, N]`，写入 `result[split_k, M, N]`
- 所有 split 完成后，在 HOST 侧做 `result.sum(0)` 得到最终结果

```
K 维度 = hidden_size (如 4096)
                 ↓ split_k_iters = 4

split 0: K[0:1024]  →  C_partial_0[M, N]   ┐
split 1: K[1024:2048] → C_partial_1[M, N]   ├→ result.sum(0) → C[M, N]
split 2: K[2048:3072] → C_partial_2[M, N]   │
split 3: K[3072:4096] → C_partial_3[M, N]   ┘
```

这样增加了并行度（更多 block 同时运行），代价是额外的显存（`split_k_iters * M * N` 个临时结果）。

**Kernel 内部流程（一个 block）：**

```
pid = blockIdx.x
pid_z = blockIdx.y  ← 哪个 split

pid_m = pid // num_pid_n
pid_n = pid % num_pid_n

accumulator = zeros([BLOCK_SIZE_M, BLOCK_SIZE_N])

# 沿 K 维度循环
for k = 0; k < K / (BLOCK_SIZE_K * SPLIT_K); k++:
    # 计算这个迭代的 K 偏移
    k_offset = k * BLOCK_SIZE_K * SPLIT_K + pid_z * BLOCK_SIZE_K

    # 加载 A tile: [BLOCK_SIZE_M, BLOCK_SIZE_K], fp16
    a = tl.load(A + (pid_m*BM + arange(BM)) * K + (k_offset + arange(BK)))

    # 加载 B tile: [BLOCK_SIZE_K, BLOCK_SIZE_N//8], int32 packed
    b = tl.load(B + (k_offset + arange(BK)) * (N//8) + (pid_n*(BN//8) + arange(BN//8)))

    # interleave 展开 int32 → [BLOCK_SIZE_K, BLOCK_SIZE_N]
    b = tl.interleave(b, b) × 3

    # 加载并解包 qzeros
    g_idx = k_offset // group_size
    zeros = tl.load(qzeros + g_idx * (N//8) + pid_n*(BN//8) + arange(BN//8))
    zeros = tl.interleave(zeros, zeros) × 3
    zeros = (zeros >> shifts) & 0xF
    zeros = tl.broadcast_to(zeros, [BK, BN])

    # 加载 scales
    scales = tl.load(scales + g_idx * N + pid_n * BN + arange(BN))
    scales = tl.broadcast_to(scales, [BK, BN])

    # Dequantize B tile
    b = (b >> shifts) & 0xF
    b = (b - zeros) * scales
    b = b.to(fp16)

    # Fused matmul: accumulate += A × B_dequant
    accumulator = tl.dot(a, b, accumulator)

# 写回 C_partial[split, pid_m*BM : (pid_m+1)*BM, pid_n*BN : (pid_n+1)*BN]
C[pid_z, pid_m*BM : (pid_m+1)*BM, pid_n*BN : (pid_n+1)*BN] = accumulator
```

**Tiling 参数：**
```
BLOCK_SIZE_M = 32    # M 方向 tile 大小
BLOCK_SIZE_N = 32    # N 方向 tile 大小（反量化后的 fp16 维度）
BLOCK_SIZE_K = 32    # K 方向 tile 大小（对应 K//8 中的 8×）
split_k_iters = 8    # K 方向切分份数（需为 2 的幂，≤32）

对每个 block:
  读取: A tile 32×32×2B = 2KB (fp16)
        B tile 32×4×4B = 0.5KB (int32 packed)
        一个 qzeros int32
        一个 scales 向量   ← 总共 ~3KB/block
  计算: 32×32×32 = 32768 次乘加运算
```

---

### 4.3 CUDA AWQ GEMM Kernel（`gemm_forward_4bit_cuda_m16nXk32`）

CUDA 版本的 AWQ kernel 来自 MIT-HAN-Lab 的原始 AWQ 实现，采用**固定 tile size 设计**：

#### Tiling 策略

```
Kernel 名: m16nXk32
  m16 = M-tile = 16（固定）
  nX  = N-tile = 64 或 128（模板参数 N）
  k32 = K-tile = 32（固定）

Thread block: 64 线程
  WARP 组织: 每个 block 2 个 warp（threadIdx.y = warp_id, 0/1）
             每个 warp 32 线程（threadIdx.x = 0..31）
  Warp 分工:
    warp 0: 覆盖输出的第 1 行
    warp 1: 覆盖输出的第 9 行
    每个 warp 内部覆盖 N 维度的 4 个连续值

Shared Memory:
  A_shared: 16 × (32 + 8) = 640 fp16  ← M-tile × (K-tile + padding)
  B_shared: 32 × (64 + 8) = 2304 fp16 ← K-tile × (N-tile + padding)
  (padding 8 用于 bank conflict 避免)
```

#### 循环流程

```
for each K-block:
  1. __syncthreads()
  2. 加载 A tile [16, 32] 到 A_shared（从 HBM）
  3. 加载 B tile（int32 packed）到 B_shared
     先在寄存器中 dequantize:
       解包 int32 → 8 × int4
       减去 zero point
       乘以 scale
     结果写回 B_shared（此时已是 fp16 [32, 64]）
  4. __syncthreads()
  5. 每个 warp 从 shared memory 读取 A 和 B tile
     做 16×64 的部分矩阵乘累加
  6. 移动到下一个 K-block
```

**CUDA 版本与 Triton 版本的关键区别：**

| 维度 | CUDA m16nXk32 | Triton |
|------|---------------|--------|
| M-tile | 固定 16 | 可配置（默认 32） |
| N-tile | 64 或 128（模板参数） | 可配置（默认 32） |
| K-tile | 固定 32 | 可配置（默认 32） |
| Split-K | 支持（blockIdx_z） | 支持（pid_z） |
| 反量化位置 | Shared Memory（写 B_shared 前） | 寄存器（直接在寄存器做 >> &0xF） |
| Block 大小 | 64 线程 | Triton 自动选择 |
| Tensor Core | 无（使用 CUDA core fma） | 有（tl.dot 使用 Tensor Core） |

**为什么 CUDA 版本用 16×64 tile size？**
- M-tile=16：对应 64 线程（2 warp × 32 线程），不用 Tensor Core 时这个并行度最佳
- N-tile=64/128：对应一个 warp 在一个 K-tile 后能产生足够的输出，隐藏流水线延迟

---

### 4.4 Triton 方案推荐理由

对于 nano-vllm，推荐优先实现 **Triton 版本的 awq_dequantize 和 awq_gemm**：

1. **无需编译** — Triton 是 Python 级别的 JIT，不依赖 CUDA toolkit 版本
2. **可读性好** — 上述 kernel 实现仅 ~100 行 Python
3. **可调试** — 支持 `TRITON_INTERPRETER=1` 模式下逐行调试
4. **Tensor Core 利用** — `tl.dot` 自动映射到 Tensor Core 指令
5. **渐进式开发** — 先实现 dequantize-only 版本验证正确性，再实现 fused gemm

---

## 5. 总体实现计划

### 第一阶段：基础支持（最小可行）

目标：能成功加载 AWQ 模型并得到正确推理结果。

1. **config.py** — 添加 `quant_method` 字段，从 `hf_config.quantization_config` 自动检测
2. **定义 AWQ 参数结构** — 在 AWQ 线性层中创建 `qweight/qzeros/scales` 三个参数
3. **loader.py** — 扩展 `load_model` 支持以 `.qweight/.qzeros/.scales` 结尾的权重名称
4. **实现 `awq_dequantize`（Triton）** — 写一个 Triton kernel 将 AWQ weight 反量化回 fp16
5. **AWQ 线性层（dequant 路径）** — forward 中先 dequantize 再调 `F.linear`
6. **模型构建** — `Qwen3ForCausalLM` 根据 `quant_method` 选择 AWQ 版本的线性层

### 第二阶段：性能优化

目标：用融合 kernel 替换 dequant+matmul 两阶段。

1. **实现 `awq_gemm`（Triton）** — fused dequantize + matmul
2. **heuristic 选择** — token 数多时走 dequant+matmul，少时走 fused gemm
3. **可选的 Marlin repack** — 将 AWQ weight 转换为 Marlin 格式后使用 marlin_gemm

### 第三阶段：高级特性（可选）

1. AWQ MoE 支持（如果后续需要）
2. 更多量化方法支持（GPTQ, FP8）

---

## 6. 参考文件清单

### nano-vllm 侧（需要修改）

| 文件 | 改动内容 |
|------|---------|
| `nanovllm/config.py` | 添加 `quant_method` 检测 |
| `nanovllm/layers/linear.py` | 新增 AWQXXXXLinear 类 |
| `nanovllm/utils/loader.py` | 支持 `.qweight/.qzeros/.scales` 后缀的权重名 |
| `nanovllm/models/qwen3.py` | 根据 `quant_method` 选择量化层 |
| `nanovllm/engine/model_runner.py` | 需要将 `quant_method` 传递给 model 构建 |
| 新建 `nanovllm/layers/quantization/awq.py` | AWQ 线性层定义 |
| 新建 `nanovllm/layers/quantization/awq_triton.py` | Triton AWQ kernel |

### vllm 侧（参考）

| 文件 | 参考价值 |
|------|---------|
| `vllm/model_executor/layers/quantization/awq.py` | AWQ config + linear method 定义 |
| `vllm/model_executor/layers/quantization/awq_triton.py` | **Triton AWQ kernel 实现（可直接参考）** |
| `vllm/model_executor/layers/quantization/awq_marlin.py` | AWQ→Marlin 转换（性能优化参考） |
| `vllm/model_executor/layers/quantization/base_config.py` | 量化框架基类 |
| `vllm/model_executor/layers/quantization/__init__.py` | 量化注册机制 |
| `vllm/_custom_ops.py` | awq_gemm / awq_dequantize 的 Python 入口 |
| `csrc/libtorch_stable/quantization/awq/gemm_kernels.cu` | CUDA AWQ kernel |

---

## 7. 验证方法

1. **正确性验证：** 加载同一个 AWQ 量化模型，用相同 prompt 推理，对比 nano-vllm 输出与 HuggingFace reference 实现的 logits 差异
2. **性能验证：** 对比量化前后的 throughput（tokens/s）和显存占用
3. **端到端测试：** `python -m nanovllm.llm --model <awq-model-path> --prompt "Hello"`

---

## 附：AWQ 检查点文件布局示例

```
model-00001-of-00002.safetensors:
  model.layers.0.self_attn.q_proj.qweight    # int32  [out_feat, in_feat/8]
  model.layers.0.self_attn.q_proj.qzeros     # int32  [num_groups, out_feat/8]
  model.layers.0.self_attn.q_proj.scales     # fp16   [num_groups, out_feat]
  model.layers.0.self_attn.k_proj.qweight    # int32  [out_feat, in_feat/8]
  model.layers.0.self_attn.k_proj.qzeros     # int32  [num_groups, out_feat/8]
  model.layers.0.self_attn.k_proj.scales     # fp16   [num_groups, out_feat]
  model.layers.0.self_attn.v_proj.qweight
  model.layers.0.self_attn.v_proj.qzeros
  model.layers.0.self_attn.v_proj.scales
  model.layers.0.self_attn.o_proj.qweight
  model.layers.0.self_attn.o_proj.qzeros
  model.layers.0.self_attn.o_proj.scales
  model.layers.0.mlp.gate_proj.qweight
  model.layers.0.mlp.gate_proj.qzeros
  model.layers.0.mlp.gate_proj.scales
  model.layers.0.mlp.up_proj.qweight
  ...
  model.layers.0.mlp.down_proj.qweight
  ...
  model.embed_tokens.weight                # fp16 [vocab, hidden] (不量化)
  model.norm.weight                        # fp16 (不量化)
  lm_head.weight                           # fp16 (通常不量化)
```

---

## 8. 参考文献与优化技术

### 8.1 核心论文

| 论文 | 链接 | 与 AWQ/nano-vllm 的关系 |
|------|------|------------------------|
| **AWQ**: Activation-aware Weight Quantization (Lin et al., 2024) | [arXiv 2306.00978](https://arxiv.org/abs/2306.00978) | AWQ 量化方法本身，理解量化原理 |
| **MARLIN**: Mixed-precision Auto-Regressive LINear Kernels (Frantar et al., PPoPP 2025) | [arXiv 2408.11743](https://arxiv.org/abs/2408.11743) | 高性能 4-bit GEMM kernel 设计，vllm 默认将 AWQ 转换为 Marlin 格式执行 |
| **GPTQ**: Accurate Post-Training Quantization for Generative Pre-trained Transformers (Frantar et al., 2023) | [arXiv 2210.17323](https://arxiv.org/abs/2210.17323) | Marlin kernel 最初为之设计，与 AWQ 共享相同的 kernel 后端 |

### 8.2 Marlin Kernel 的关键优化技术

Marlin 论文 (PPoPP 2025) 中通过 ablation study 验证了以下微优化的重要性，忽略其中任何一个可导致性能腰斩：

1. **Asynchronous memory access** — 使用 `cp.async` 指令异步加载权重到 shared memory，与计算流水线重叠
2. **Pipelining** — 4 级流水线设计，深藏访存延迟
3. **Conflict-free shared memory** — 通过 XOR 操作避免 bank conflict，确保 shared memory 带宽利用率
4. **L2 cache pollution avoidance** — 使用 `evict_first` cache hint，防止权重数据污染 L2 cache（这对 LLM 推理很关键，因为后续的 KV cache 访问也依赖 L2 cache）
5. **Tile 设计** — 16×64 tile size (Marlin tile)，256 线程，4 级流水线

### 8.3 vllm 中 Marlin 的扩展

vllm 在原始 Marlin (GPTQ-only) 基础上做了重要扩展：

- **AWQ→Marlin 适配**: 需要 `awq_marlin_repack` CUDA kernel，处理 AWQ 的非标准 pack 顺序和 output-dim packing
- **Zero point 支持**: 原始 Marlin 只支持对称量化 (`uint4b8`)，vllm 扩展为支持非对称 (`uint4`)
- **SM 版本特化**: 为 SM80 (Ampere)、SM89 (Ada)、SM90 (Hopper) 分别生成不同的 kernel 特化版本

### 8.4 nano-vllm 推荐路线

```
第一阶段 (当前目标)         第二阶段 (性能优化)          第三阶段 (可选)
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│ Triton AWQ      │  →   │ Triton awq_gemm │  →   │ Marlin repack   │
│ dequantize      │       │ fused kernel    │       │ + marlin_gemm   │
│ + F.linear      │       │                 │       │ (CUDA, 最高性能)│
│ [Python/Triton] │       │ [Triton]        │       │ [CUDA]          │
└─────────────────┘       └─────────────────┘       └─────────────────┘
     正确性优先              性能提升 ~2x              极致性能 ~3-4x
     无需 CUDA 编译          需安装 Triton             需 CUDA toolkit
```

---

## 9. vllm INT4 量化策略全景

### 9.1 策略总览

vllm 中涉及 INT4 权重量化的策略多达 **10+ 种**，每种都有对应的 GEMM kernel。以下按权重来源分类：

| 量化策略 | 对应 Kernel(s) | 典型模型来源 |
|----------|---------------|-------------|
| **AWQ** | `ops.awq_gemm` (CUDA) / Triton AWQ | AWQ 量化模型 (如 `Qwen2.5-7B-AWQ`) |
| **AWQ + Marlin** | Machete → Marlin → Conch → Exllama (自动选择) | 同上，但自动转换到高性能 kernel |
| **GPTQ** | Machete → Marlin → Conch → Exllama (自动选择) | GPTQ 量化模型 (如 `Llama-2-7B-GPTQ`) |
| **GPTQ + Marlin** | 同上 | 同上 |
| **Compressed-Tensors** | Machete → Marlin → Conch → Exllama | 通过 nm-vllm 或 llm-compressor 量化的模型 |
| **BitsAndBytes (NF4)** | `bitsandbytes.matmul_4bit` | HuggingFace 4-bit 加载 (QLoRA) |
| **GGUF** | llama.cpp kernel | GGUF 格式模型 |
| **TorchAO** | torchao 原生 kernel | PyTorch 量化生态 |
| **Humming** | humming 库自带 kernel | Humming 量化模型 |
| **INC** | XPU int4_gemm / oneDNN | Intel 生态 |
| **Quark** | 同 AWQ/GPTQ (委托给已有 kernel) | AMD Quark 工具量化 |
| **CPU AWQ** | 纯 C++ kernel (无 GPU) | AWQ 模型在 CPU 上推理 |

### 9.2 Kernel 与量化策略的映射关系

vllm 有一个**两层间接**的调度架构：

```
量化策略 (quant_method)
    ↓  get_quant_method() 决定 LinearMethod
LinearMethod (如 AWQMarlinLinearMethod)
    ↓  choose_mp_linear_kernel() 选择 GEMM kernel
MPLinearKernel (如 MarlinLinearKernel, MacheteLinearKernel...)
    ↓  apply_weights() 调用实际的底层算子
底层算子 (ops.marlin_gemm, ops.machete_mm, ops.gptq_gemm...)
```

关键点：**多个量化策略可以共享同一个 GEMM kernel**。例如 `AWQMarlin` 和 `GPTQMarlin` 最终都走 `MarlinLinearKernel`，区别在于前者的 weight 是 AWQ 格式，需要在 `process_weights_after_loading()` 中多一步 AWQ→GPTQ 格式转换。

### 9.3 所有 INT4 GEMM Kernel 详细对比

#### 按 HW 平台分组

**Hopper (SM90) — 性能从高到低：**

| 排名 | Kernel | 类型 | 激活精度 | 说明 |
|------|--------|------|---------|------|
| 1 | **CutlassW4A8** | CUTLASS 3.x (WGMMA) | FP8 | 使用 Hopper FP8 tensor core，吞吐最高；权重为 signed int4 |
| 2 | **Machete** | CUTLASS 3.x (WGMMA) | FP16/BF16 | 使用 Hopper WGMMA 指令，W4A16 中最快 |
| 3 | **Marlin** | 自定义 CUDA + PTX | FP16 | 经典方案，在 Hopper 上比 Machete 略慢 |
| 4 | **Conch** | Triton | FP16/BF16 | Triton 实现，性能尚可 |
| 5 | **Exllama** | 自定义 CUDA | FP16 only | 兼容性最广，性能最低 |

**Ampere (SM80-89) — 性能从高到低：**

| 排名 | Kernel | 类型 | 说明 |
|------|--------|------|------|
| 1 | **Marlin** | 自定义 CUDA + PTX | Ampere 上 W4A16 的最优选择 |
| 2 | **Conch** | Triton | 竞争力接近 Marlin |
| 3 | **AllSpark** | 自定义 CUDA | W8A16 方案（非 int4），精度更高但加速比小 |
| 4 | **Exllama** | 自定义 CUDA | 最通用的 fallback |

**Turing (SM75):** Marlin (最佳) → Exllama (fallback)

**ROCm (AMD):** TritonW4A16 (最佳) → Conch → Exllama

#### 按功能特性对比

| Kernel | SM 下限 | Zero Pt | g_idx (act_order) | Fused GEMM | 离线 Repack | 激活 dtype | Notes |
|--------|---------|---------|-------------------|-----------|------------|-----------|-------|
| AWQ (classic) | SM75 | 是 | 否 | 混合* | 否 | fp16 | *小 batch fused, 大 batch dequant+matmul |
| Marlin | SM75 | 是 | 是 | 是 | 是 | fp16/bf16 | 最成熟的高性能 4-bit kernel |
| Machete | SM90 | 是** | 是*** | 是 | 是 | fp16/bf16 | **zero point 预合并到 scale；***不支持 TP |
| Conch | SM80 | 是 | 否 | 是 | 最小 | fp16/bf16 | Triton 方案，ROCm 友好 |
| Exllama | SM60 | 是 | 是 | 是 | 是 | fp16 only | 最广泛的硬件兼容性 |
| CutlassW4A8 | SM90 | 否 | 否 | 是 | 是 | FP8 | W4A8，使用 Hopper FP8 tensor core |
| TritonW4A16 | ROCm | 是 | 否 | 是 | 最小 | fp16/bf16 | ROCm (AMD GPU) 专用 |

### 9.4 Weight Packing 格式差异

**GPTQ 标准 packing** (Marlin, Machete, Conch, Exllama, TritonW4A16 共用)：
```
int32 的 8 个 4-bit 槽位: [0:3] [4:7] [8:11] [12:15] [16:19] [20:23] [24:27] [28:31]
对应 index:                0      1      2       3       4       5       6       7
qweight shape: (K//8, N)  — 沿 input 维度 pack
```

**AWQ packing** (仅 AWQ 格式的 checkpoint 使用)：
```
int32 的 8 个 4-bit 槽位: [0:3] [4:7] [8:11] [12:15] [16:19] [20:23] [24:27] [28:31]
对应 index:                0      4      1       5       2       6       3       7
qweight shape: (K, N//8)  — 沿 output 维度 pack
```

**核心差异两点**：pack 顺序不同（AWQ 的 `[0,4,1,5,2,6,3,7]` vs 标准的 `[0,1,2,3,4,5,6,7]`）**和** packing 维度不同（AWQ 沿 output vs 标准沿 input）。这是 AWQ 需要额外 repack 的原因。

**Marlin 内部 tile 格式**：repack 后，权重被重组为 16-output-neuron 的 tile 格式 (GPTQ_MARLIN_TILE=16)，min thread N=64，min thread K=128。

**Machete 内部格式**：repack 后，权重被重组为 CUTLASS WGMMA 友好的 layout。Zero point 被吸收到 scale 中（`dequant: (w * scale) - pre_applied_zp`）。

### 9.5 对应关系总结

**每个量化策略都有对应的 GEMM kernel 吗？** 是的，但一个 kernel 可以服务多个策略：

| Checkpoint 格式 | 加载后的量化策略 | 实际使用的 GEMM kernel |
|----------------|----------------|----------------------|
| AWQ | `awq` (classic) | `ops.awq_gemm` 或 `ops.awq_dequantize` + matmul |
| AWQ | `awq_marlin` (自动升级) | choose_mp_linear_kernel() → Machete/Marlin/Conch/Exllama |
| GPTQ | `gptq` / `gptq_marlin` | choose_mp_linear_kernel() → Machete/Marlin/Conch/Exllama |
| CompressedTensors | `compressed-tensors` | choose_mp_linear_kernel() → Marlin/Conch/Exllama |
| bitsandbytes (NF4) | `bitsandbytes` | bitsandbytes 自有 kernel |
| GGUF | `gguf` | llama.cpp kernel |

**所以对 nano-vllm 来说，支持 AWQ 的正确路径是：**

```
加载 AWQ checkpoint → 保持 AWQ 格式 → 用 Triton awq_gemm kernel 做 fused GEMM
```

不需要实现 Marlin repack 或 Machete 等复杂 kernel。**先把一条路径走通，再考虑优化。**

### 9.6 对于 nano-vllm 的选型建议

| 层级 | 推荐选择 | 理由 |
|------|---------|------|
| Kernel 方案 | **Triton AWQ (非对称, 带 zero point)** | 无需 CUDA 编译，代码可读性好，性能是 CUDA 的 80-90% |
| Weight 格式 | **保持 AWQ 原生格式** | 不需要 repack，减少代码复杂度 |
| 推理路径 | **先 dequantize + matmul，后 fused gemm** | 分步实现，每步可验证正确性 |
| 后续优化 | **Marlin repack (可选)** | 如果 Triton 性能不够，再考虑 CUDA |
```
