# Phase 3: AWQ CUDA Custom Op

## 目标

将 AWQ fused GEMM 从 Triton kernel 替换为 C++ CUDA custom op，通过 torch 注册为 `torch.ops.nanovllm.awq_gemm`，在面试中展示手写 CUDA kernel + torch binding 能力。

## 改动范围

| 文件 | 操作 | 说明 |
|------|------|------|
| `setup.py` | 新增 | CUDAExtension 编译 .cu |
| `nanovllm/csrc/__init__.py` | 新增 | 导入编译后的 `.so`，触发 TORCH_LIBRARY 注册 |
| `nanovllm/csrc/awq/dequantize.cuh` | 新增 | s4→fp16x2 device 函数 |
| `nanovllm/csrc/awq/gemm_kernels.cu` | 新增 | CUDA kernel + host wrapper + TORCH_LIBRARY |
| `nanovllm/layers/quantization/awq.py` | 修改 | ColumnParallel 和 RowParallel 的 forward 各加三级路径 |
| `pyproject.toml` | 修改 | build 依赖添加 `ninja` |

## 实现方案

### CUDA Kernel

直接移植 vllm 的 CUDA kernel (`gemm_forward_4bit_cuda_m16nXk32`)：

- 模板参数 N ∈ {64, 128}，在 host wrapper 中根据 OC 选择
- 使用 `ldmatrix.sync.aligned.m8n8.x4` 快速加载 shared memory → register
- 使用 `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` Tensor Core 做矩阵乘
- 在 shared memory 中完成 dequantize（inline PTX `lop3.b32` 解包，`sub.f16x2` 减 zero，`fma.rn.f16x2` 乘 scale）
- split-k 并行：多个 block 各自计算部分 K 范围，host 侧 `sum(0)` 归约
- 限制：M ≤ 16，OC 须为 64 或 128 的倍数

### Torch Binding

`.cu` 文件末尾用 `TORCH_LIBRARY` 注册 op：

```cpp
TORCH_LIBRARY(nanovllm, m) {
    m.def("awq_gemm(Tensor in_feats, Tensor kernel, "
           "Tensor scaling_factors, Tensor zeros, "
           "int split_k_iters) -> Tensor");
}
```

扩展模块名：`nanovllm._C`（即 `TORCH_EXTENSION_NAME` 设为 `nanovllm._C`）。

**Python 侧导入**：在 `nanovllm/csrc/__init__.py` 中 import：
```python
from nanovllm import _C  # 触发 TORCH_LIBRARY 注册
```
然后在 `nanovllm/__init__.py` 中 import `csrc` 包。

### Build

`setup.py` 用 `torch.utils.cpp_extension.CUDAExtension`：

```python
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    ext_modules=[
        CUDAExtension(
            "nanovllm._C",
            ["nanovllm/csrc/awq/gemm_kernels.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3", "-std=c++17",
                    "--generate-code=arch=compute_89,code=sm_89",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
```

`pyproject.toml` 中 `[build-system]` 添加 `ninja`：
```toml
requires = ["setuptools>=61", "ninja"]
```

`pip install -e .` 会自动编译 `.cu` → `.so`，输出到 `nanovllm/_C.so`。

### Python 调用（三级路径）

**AWQColumnParallelLinear.forward**（及其子类 `AWQMergedColumnParallelLinear`、`AWQQKVParallelLinear`）：

```python
def forward(self, x):
    M = x.size(-2)
    if M <= 16:
        return torch.ops.nanovllm.awq_gemm(
            x, self.qweight, self.scales, self.qzeros, 8)
    elif M < 256:
        return awq_gemm_triton(x, self.qweight, self.scales,
                               self.qzeros, self.group_size, split_k_iters=8)
    else:
        weight = self._dequantize_weight().t().to(x.dtype)
        return F.linear(x, weight, self.bias)
```

**AWQRowParallelLinear.forward**（独立于 Column，单独修改）：

```python
def forward(self, x):
    x_shard = x.narrow(-1, self.tp_rank * (x.size(-1) // self.tp_size),
                       x.size(-1) // self.tp_size)
    M = x_shard.size(-2)
    if M <= 16:
        y = torch.ops.nanovllm.awq_gemm(
            x_shard, self.qweight, self.scales, self.qzeros, 8)
    elif M < 256:
        y = awq_gemm_triton(x_shard, self.qweight, self.scales,
                            self.qzeros, self.group_size, split_k_iters=8)
    else:
        weight = self._dequantize_weight().t().to(x.dtype)
        y = F.linear(x_shard, weight, self.bias if self.tp_rank == 0 else None)
    if self.tp_size > 1:
        dist.all_reduce(y)
    return y
```

路径选择逻辑：

```
M ≤ 16      ──→ CUDA custom op (PTX Tensor Core)
16 < M < 256 ──→ Triton fused gemm (保留现有实现)
M ≥ 256     ──→ dequant + cuBLAS matmul
```

## 验收标准

1. **正确性**：`torch.ops.nanovllm.awq_gemm` 输出与 `awq_gemm_triton` 的 **每元素最大绝对差（max per-element absolute diff）** < 0.125
2. **性能**：bench.py throughput 不低于 Triton 版本（~800 tok/s）
3. **M=1 单条 decode 场景** CUDA 版本不慢于 Triton 版本
4. **编译**：`pip install -e .` 自动编译 CUDA extension，无报错
5. **M > 16 decode 场景**降级到 Triton kernel，功能正常
