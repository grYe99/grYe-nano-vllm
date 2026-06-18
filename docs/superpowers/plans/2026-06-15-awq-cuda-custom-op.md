# AWQ CUDA Custom Op Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Triton AWQ fused GEMM with a C++ CUDA custom op (`torch.ops.nanovllm.awq_gemm`) for M ≤ 16 decode, using vllm's PTX-optimized kernel.

**Architecture:** Port vllm's `gemm_forward_4bit_cuda_m16nXk32` CUDA kernel with inline PTX Tensor Core instructions, register via `TORCH_LIBRARY` as a standard torch custom op. Three-tier dispatch in Python: M≤16 → CUDA, 16<M<256 → Triton, M≥256 → dequant+cuBLAS.

**Tech Stack:** CUDA C++ (nvcc), inline PTX (`mma.sync`, `ldmatrix`), Torch C++ API (`TORCH_LIBRARY`, `torch::Tensor`), setuptools (`CUDAExtension`)

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `setup.py` | Create | `CUDAExtension("nanovllm._C", ...)` + `BuildExtension` |
| `pyproject.toml` | Modify | Add `ninja` to build-system requires |
| `nanovllm/csrc/awq/dequantize.cuh` | Create | `dequantize_s4_to_fp16x2` device function (PTX) |
| `nanovllm/csrc/awq/gemm_kernels.cu` | Create | GEMM kernel + host `awq_gemm` wrapper + `TORCH_LIBRARY` |
| `nanovllm/__init__.py` | Modify | `import nanovllm._C` to trigger op registration |
| `nanovllm/layers/quantization/awq.py` | Modify | Three-tier dispatch in `AWQColumnParallelLinear.forward` and `AWQRowParallelLinear.forward` |

---

### Task 1: Build system setup

**Files:**
- Create: `setup.py`
- Modify: `pyproject.toml` (line 3: add ninja)

- [ ] **Step 1: Create setup.py**

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
                    "-O3",
                    "-std=c++17",
                    "--generate-code=arch=compute_89,code=sm_89",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
```

- [ ] **Step 2: Modify pyproject.toml**

Change the `[build-system] requires` list to add `ninja`:

```toml
[build-system]
requires = ["setuptools>=61", "ninja"]
build-backend = "setuptools.build_meta"
```

- [ ] **Step 3: Reinstall and verify compilation**

```bash
pip install -e .
```

Expected output: nvcc compiles `gemm_kernels.cu`, produces `nanovllm/_C.so`.

- [ ] **Step 4: Commit**

```bash
git add setup.py pyproject.toml
git commit -m "build: add CUDAExtension for AWQ custom op"
```

---

### Task 2: Dequantize CUDA header

**Files:**
- Create: `nanovllm/csrc/awq/dequantize.cuh`

- [ ] **Step 1: Create dequantize.cuh**

Ported from vllm `csrc/libtorch_stable/quantization/awq/dequantize.cuh`. Only change: remove `vllm::awq` namespace, use `nanovllm::awq` namespace.

```cuda
/*
Adapted from https://github.com/mit-han-lab/llm-awq
*/
#pragma once

#include <cuda_fp16.h>

namespace nanovllm {
namespace awq {

__device__ uint4 dequantize_s4_to_fp16x2(uint32_t const& source) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750
  assert(false);
#else
  uint4 result;
  uint32_t* h = reinterpret_cast<uint32_t*>(&result);
  uint32_t const i4s = reinterpret_cast<uint32_t const&>(source);

  static constexpr uint32_t immLut = (0xf0 & 0xcc) | 0xaa;
  static constexpr uint32_t BOTTOM_MASK = 0x000f000f;
  static constexpr uint32_t TOP_MASK = 0x00f000f0;
  static constexpr uint32_t I4s_TO_F16s_MAGIC_NUM = 0x64006400;

  const uint32_t top_i4s = i4s >> 8;

  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[0])
               : "r"(i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM),
                 "n"(immLut));
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[1])
               : "r"(i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM),
                 "n"(immLut));
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[2])
               : "r"(top_i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM),
                 "n"(immLut));
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[3])
               : "r"(top_i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM),
                 "n"(immLut));

  static constexpr uint32_t FP16_TOP_MAGIC_NUM = 0x64006400;
  static constexpr uint32_t ONE_SIXTEENTH = 0x2c002c00;
  static constexpr uint32_t NEG_64 = 0xd400d400;

  asm volatile("sub.f16x2 %0, %1, %2;\n"
               : "=r"(h[0])
               : "r"(h[0]), "r"(FP16_TOP_MAGIC_NUM));
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
               : "=r"(h[1])
               : "r"(h[1]), "r"(ONE_SIXTEENTH), "r"(NEG_64));
  asm volatile("sub.f16x2 %0, %1, %2;\n"
               : "=r"(h[2])
               : "r"(h[2]), "r"(FP16_TOP_MAGIC_NUM));
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
               : "=r"(h[3])
               : "r"(h[3]), "r"(ONE_SIXTEENTH), "r"(NEG_64));

  return result;
#endif
  __builtin_unreachable();
}

}  // namespace awq
}  // namespace nanovllm
```

- [ ] **Step 2: Create csrc/awq directory**

```bash
mkdir -p nanovllm/csrc/awq
```

Then write the file above.

---

### Task 3: GEMM kernel + torch binding

**Files:**
- Create: `nanovllm/csrc/awq/gemm_kernels.cu`

This is the core task. Port vllm's `gemm_kernels.cu` from `../vllm/csrc/libtorch_stable/quantization/awq/gemm_kernels.cu` with the following adaptations:
- Replace `torch::stable::Tensor` → `torch::Tensor`
- Replace `torch::stable::empty(...)` → `torch::empty(...)`
- Replace `vllm::awq` → `nanovllm::awq`
- Replace `STABLE_TORCH_LIBRARY_FRAGMENT(_C, ...)` → `TORCH_LIBRARY(nanovllm, ...)`
- Replace `torch::stable::sum(_out_feats, 0)` → `_out_feats.sum(0)`
- Use `getCurrentCUDAStream()` instead of `get_current_cuda_stream()`
- Add input validation (check device == cuda, dtype == half)
- **Important**: Remove the `#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 750` (Turing) code path. We target SM89 only, so only keep the Ampere+ (`#else`) branch with `mma.sync.aligned.m16n8k16`.

- [ ] **Step 1: Write gemm_kernels.cu**

The file has three parts:
1. `gemm_forward_4bit_cuda_m16nXk32<N>` kernel (identical to vllm, only namespace change)
2. `awq_gemm` host function (torch::Tensor API, input validation, kernel launch)
3. `TORCH_LIBRARY(nanovllm, ...)` registration

Part 1 — kernel (copy from vllm, change namespace):

```cuda
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include "dequantize.cuh"

namespace nanovllm {
namespace awq {

template <int N>
__global__ void __launch_bounds__(64)
    gemm_forward_4bit_cuda_m16nXk32(int G, int split_k_iters,
                                    half* __restrict__ A, int* __restrict__ B,
                                    half* __restrict__ scaling_factors,
                                    int* __restrict__ zeros, int M, int IC,
                                    int OC, half* __restrict__ C) {
  // --- identical to vllm's kernel, only namespace change ---
  // For brevity: copy the 300 lines from vllm verbatim.
  // Everything inside this function is identical.
}
```

Part 2 — host wrapper:

```cuda
torch::Tensor awq_gemm(torch::Tensor _in_feats,
                       torch::Tensor _kernel,
                       torch::Tensor _scaling_factors,
                       torch::Tensor _zeros,
                       int64_t split_k_iters) {
  // Input validation
  TORCH_CHECK(_in_feats.is_cuda(), "in_feats must be CUDA tensor");
  TORCH_CHECK(_in_feats.dtype() == torch::kHalf, "in_feats must be float16");
  // ... launch kernel based on OC % 128 or % 64
}
```

Part 3 — registration (note: requires both `def` for schema and `TORCH_LIBRARY_IMPL` for kernel dispatch):

```cuda
TORCH_LIBRARY(nanovllm, m) {
    m.def("awq_gemm(Tensor in_feats, Tensor kernel, "
           "Tensor scaling_factors, Tensor zeros, "
           "int split_k_iters) -> Tensor");
}

TORCH_LIBRARY_IMPL(nanovllm, CUDA, m) {
    m.impl("awq_gemm", &awq_gemm);
}
```

Full content (port directly from vllm):

```cuda
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include "dequantize.cuh"

namespace nanovllm {
namespace awq {

template <int N>
__global__ void __launch_bounds__(64)
    gemm_forward_4bit_cuda_m16nXk32(int G, int split_k_iters,
                                    half* __restrict__ A, int* __restrict__ B,
                                    half* __restrict__ scaling_factors,
                                    int* __restrict__ zeros, int M, int IC,
                                    int OC, half* __restrict__ C) {
  assert(N == 64 || N == 128);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750
  assert(false);
#else
  static constexpr uint32_t ZERO = 0x0;
  float C_warp[32];
  __shared__ half A_shared[16 * (32 + 8)];
  __shared__ half B_shared[32 * (N + 8)];

  int j_factors1 = ((OC + N - 1) / N);
  int blockIdx_y = blockIdx.x % ((M + 16 - 1) / 16 * j_factors1);
  int blockIdx_z = blockIdx.x / ((M + 16 - 1) / 16 * j_factors1);

  half A_shared_warp[8];
  half B_shared_warp[N / 4];
  for (int j_0_4_init = 0; j_0_4_init < N / 32; ++j_0_4_init) {
    for (int i = 0; i < 8; ++i) {
      C_warp[(j_0_4_init * 8) + i] = 0.0;
    }
  }

  static constexpr int row_stride_warp = 32 * 8 / 32;
  static constexpr int row_stride = 2 * 32 * 8 / N;
  bool ld_A_flag =
      (blockIdx_y / j_factors1 * 16 + threadIdx.y * row_stride_warp +
       threadIdx.x * 8 / 32) < M;

  half* A_ptr =
      A +
      (((int)blockIdx_y) / j_factors1 * 16 +
       (((int)threadIdx.y) * row_stride_warp) + ((int)threadIdx.x) / (32 / 8)) *
          IC +
      (((int)threadIdx.x) % (32 / 8)) * 8;

  int* B_ptr = B + ((int)threadIdx.y) * (OC / 8) * (256 / N) +
               (((int)threadIdx.x) / (N / 8)) * (OC / 8) +
               (((int)blockIdx_y) % j_factors1) * (N / 8) +
               (((int)threadIdx.x) % (N / 8)) * 1;

  half* A_shared_ptr = A_shared +
                       ((int)threadIdx.y) * row_stride_warp * (32 + 8) +
                       (((int)threadIdx.x) / (32 / 8)) * (32 + 8) +
                       (((int)threadIdx.x) % (32 / 8)) * 8;

  half* B_shared_ptr = B_shared +
                       ((int)threadIdx.y) * (row_stride / 2) * (N + 8) +
                       (((int)threadIdx.x) / (N / 8)) * (N + 8) +
                       (((int)threadIdx.x) % (N / 8)) * 8;

  int* zeros_ptr = zeros + (((int)blockIdx_y) % j_factors1) * (N / 8) +
                   ((int)threadIdx.x) % (N / 8);

  half* scaling_factors_ptr = scaling_factors +
                              (((int)blockIdx_y) % j_factors1) * N +
                              (((int)threadIdx.x) % (N / 8)) * 8;

  half* C_ptr =
      C +
      static_cast<long long>(blockIdx_z) * M * OC
      + (((int)blockIdx_y) % j_factors1) * N + ((int)threadIdx.y) * (N / 2) +
      (((int)threadIdx.x) % 4) * 2;

  int k_bound = (IC / 32 + split_k_iters - 1) / split_k_iters;
  if ((k_bound - 1) * split_k_iters * 32 + blockIdx_z * 32 >= IC) k_bound -= 1;
  for (int _k_0_0 = 0; _k_0_0 < k_bound; ++_k_0_0) {
    int k_0_0 = _k_0_0 * split_k_iters + blockIdx_z;
    __syncthreads();
    if (ld_A_flag) {
      *(uint4*)(A_shared_ptr) = *(uint4*)(A_ptr + (k_0_0 * 32));
    } else {
      *(uint4*)(A_shared_ptr) = make_uint4(0, 0, 0, 0);
    }

    uint32_t zeros_loaded = *(uint32_t*)(zeros_ptr + k_0_0 * 32 / G * (OC / 8));
    uint4 B_loaded_zero = dequantize_s4_to_fp16x2(zeros_loaded);
    uint4 B_loaded_scale =
        *(uint4*)(scaling_factors_ptr + k_0_0 * 32 / G * (OC));
    int* B_ptr_local = B_ptr + k_0_0 * 32 * (OC / 8);

    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < N / 16; ++ax0_ax1_fused_0) {
      uint32_t B_loaded =
          *(uint32_t*)(B_ptr_local + ax0_ax1_fused_0 * row_stride * (OC / 8));
      uint4 B_loaded_fp16 = dequantize_s4_to_fp16x2(B_loaded);

      // Dequantize: (B - zero) * scale
      asm volatile("sub.f16x2 %0, %1, %2;\n"
                   : "=r"(B_loaded_fp16.x)
                   : "r"(B_loaded_fp16.x), "r"(B_loaded_zero.x));
      asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
                   : "=r"(B_loaded_fp16.x)
                   : "r"(B_loaded_fp16.x), "r"(B_loaded_scale.x), "r"(ZERO));
      asm volatile("sub.f16x2 %0, %1, %2;\n"
                   : "=r"(B_loaded_fp16.y)
                   : "r"(B_loaded_fp16.y), "r"(B_loaded_zero.y));
      asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
                   : "=r"(B_loaded_fp16.y)
                   : "r"(B_loaded_fp16.y), "r"(B_loaded_scale.y), "r"(ZERO));
      asm volatile("sub.f16x2 %0, %1, %2;\n"
                   : "=r"(B_loaded_fp16.z)
                   : "r"(B_loaded_fp16.z), "r"(B_loaded_zero.z));
      asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
                   : "=r"(B_loaded_fp16.z)
                   : "r"(B_loaded_fp16.z), "r"(B_loaded_scale.z), "r"(ZERO));
      asm volatile("sub.f16x2 %0, %1, %2;\n"
                   : "=r"(B_loaded_fp16.w)
                   : "r"(B_loaded_fp16.w), "r"(B_loaded_zero.w));
      asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
                   : "=r"(B_loaded_fp16.w)
                   : "r"(B_loaded_fp16.w), "r"(B_loaded_scale.w), "r"(ZERO));

      *(uint4*)(B_shared_ptr + ax0_ax1_fused_0 * row_stride * (N + 8)) =
          B_loaded_fp16;
    }
    __syncthreads();

    for (int k_0_1 = 0; k_0_1 < 2; ++k_0_1) {
      // Load A from shared memory using ldmatrix
      {
        unsigned int addr;
        __asm__ __volatile__(
            "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, "
            "addr; }\n"
            : "=r"(addr)
            : "l"((void*)((&(A_shared[(k_0_1 * 16)])) +
                          (((((int)threadIdx.x) & 15) * 40) +
                           ((((int)threadIdx.x) >> 4) * 8)))));

        __asm__ __volatile__(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
            "{%0, %1, %2, %3}, [%4];\n"
            : "=r"(((unsigned*)(A_shared_warp + 0))[0]),
              "=r"(((unsigned*)(A_shared_warp + 0))[1]),
              "=r"(((unsigned*)(A_shared_warp + 0))[2]),
              "=r"(((unsigned*)(A_shared_warp + 0))[3])
            : "r"(addr));
      }

      for (int ax1_0 = 0; ax1_0 < N / 32; ++ax1_0) {
        // Load B from shared memory using ldmatrix.trans
        {
          unsigned int addr;
          __asm__ __volatile__(
              "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, "
              "addr; }\n"
              : "=r"(addr)
              : "l"((void*)((&(B_shared[(((k_0_1 * (N * 16 + 128)) +
                                          (((int)threadIdx.y) * (N / 2))) +
                                         (ax1_0 * 16))])) +
                            (((((int)threadIdx.x) & 15) * (N + 8)) +
                             ((((int)threadIdx.x) >> 4) * 8)))));
          __asm__ __volatile__(
              "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
              "{%0, %1, %2, %3}, [%4];\n"
              : "=r"(((unsigned*)(B_shared_warp + (ax1_0 * 8)))[0]),
                "=r"(((unsigned*)(B_shared_warp + (ax1_0 * 8)))[1]),
                "=r"(((unsigned*)(B_shared_warp + (ax1_0 * 8)))[2]),
                "=r"(((unsigned*)(B_shared_warp + (ax1_0 * 8)))[3])
              : "r"(addr));
        }
      }

      for (int j_0_4 = 0; j_0_4 < N / 32; ++j_0_4) {
        // Ampere+ (SM89): mma.sync.aligned.m16n8k16
        // First half of C tile
        __asm__ __volatile__(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, "
            "%13};\n"
            : "=f"(((float*)(C_warp + (j_0_4 * 8)))[0]),
              "=f"(((float*)(C_warp + (j_0_4 * 8)))[1]),
              "=f"(((float*)(C_warp + (j_0_4 * 8)))[2]),
              "=f"(((float*)(C_warp + (j_0_4 * 8)))[3])
            : "r"(((unsigned*)(A_shared_warp + 0))[0]),
              "r"(((unsigned*)(A_shared_warp + 0))[1]),
              "r"(((unsigned*)(A_shared_warp + 0))[2]),
              "r"(((unsigned*)(A_shared_warp + 0))[3]),
              "r"(((unsigned*)(B_shared_warp + (j_0_4 * 8)))[0]),
              "r"(((unsigned*)(B_shared_warp + (j_0_4 * 8)))[1]),
              "f"(((float*)(C_warp + (j_0_4 * 8)))[0]),
              "f"(((float*)(C_warp + (j_0_4 * 8)))[1]),
              "f"(((float*)(C_warp + (j_0_4 * 8)))[2]),
              "f"(((float*)(C_warp + (j_0_4 * 8)))[3]));
        // Second half of C tile
        __asm__ __volatile__(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, "
            "%13};\n"
            : "=f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[0]),
              "=f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[1]),
              "=f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[2]),
              "=f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[3])
            : "r"(((unsigned*)(A_shared_warp + 0))[0]),
              "r"(((unsigned*)(A_shared_warp + 0))[1]),
              "r"(((unsigned*)(A_shared_warp + 0))[2]),
              "r"(((unsigned*)(A_shared_warp + 0))[3]),
              "r"(((unsigned*)(B_shared_warp + ((j_0_4 * 8) + 4)))[0]),
              "r"(((unsigned*)(B_shared_warp + ((j_0_4 * 8) + 4)))[1]),
              "f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[0]),
              "f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[1]),
              "f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[2]),
              "f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[3]));
      }
    }
  }

  // Store results
  for (int ax1_0_1 = 0; ax1_0_1 < (N / 32); ++ax1_0_1) {
    for (int local_id = 0; local_id < 8; ++local_id) {
      int row_offset = (((int)blockIdx_y) / j_factors1) * 16 +
                       ((int)threadIdx.x) / 4 + (local_id % 4) / 2 * 8;
      if (row_offset < M) {
        *(C_ptr + ax1_0_1 * 16 + row_offset * OC + (local_id / 4) * 8 +
          local_id % 2) = __float2half(C_warp[(ax1_0_1 * 8) + local_id]);
      }
    }
  }
#endif
}

}  // namespace awq
}  // namespace nanovllm

torch::Tensor awq_gemm(torch::Tensor _in_feats,
                       torch::Tensor _kernel,
                       torch::Tensor _scaling_factors,
                       torch::Tensor _zeros,
                       int64_t split_k_iters) {
  TORCH_CHECK(_in_feats.is_cuda(), "in_feats must be CUDA tensor");
  TORCH_CHECK(_in_feats.dtype() == torch::kHalf, "in_feats must be float16");
  TORCH_CHECK(_kernel.is_cuda(), "kernel must be CUDA tensor");
  TORCH_CHECK(_scaling_factors.is_cuda(), "scaling_factors must be CUDA tensor");
  TORCH_CHECK(_zeros.is_cuda(), "zeros must be CUDA tensor");

  int num_in_feats = _in_feats.size(0);
  int num_in_channels = _in_feats.size(1);
  auto _out_feats = torch::empty(
      {split_k_iters, num_in_feats, _kernel.size(1) * 8},
      _in_feats.options());
  int num_out_channels = _out_feats.size(-1);

  TORCH_CHECK(num_out_channels % 64 == 0,
              "OC must be multiple of 64, got ", num_out_channels);
  TORCH_CHECK(num_in_feats <= 16,
              "M must be <= 16 for CUDA kernel, got ", num_in_feats);

  auto in_feats = reinterpret_cast<half*>(_in_feats.mutable_data_ptr<at::Half>());
  auto kernel = reinterpret_cast<int*>(_kernel.mutable_data_ptr<int>());
  auto out_feats = reinterpret_cast<half*>(_out_feats.mutable_data_ptr<at::Half>());
  auto scaling_factors = reinterpret_cast<half*>(_scaling_factors.mutable_data_ptr<at::Half>());
  auto zeros = reinterpret_cast<int*>(_zeros.mutable_data_ptr<int>());
  int group_size = num_in_channels / _scaling_factors.size(0);

  auto stream = c10::cuda::getCurrentCUDAStream();

  if (num_out_channels % 128 == 0) {
    int j_factors1 = num_out_channels / 128 / 1;
    dim3 num_blocks((num_in_feats + 16 - 1) / 16 * j_factors1 * split_k_iters);
    dim3 threads_per_block(32, 2);
    nanovllm::awq::gemm_forward_4bit_cuda_m16nXk32<128>
        <<<num_blocks, threads_per_block, 0, stream>>>(
            group_size, split_k_iters, in_feats, kernel, scaling_factors, zeros,
            num_in_feats, num_in_channels, num_out_channels, out_feats);
  } else {
    int j_factors1 = num_out_channels / 64 / 1;
    dim3 num_blocks(1 * (num_in_feats + 16 - 1) / 16 * j_factors1 *
                    split_k_iters);
    dim3 threads_per_block(32, 2);
    nanovllm::awq::gemm_forward_4bit_cuda_m16nXk32<64>
        <<<num_blocks, threads_per_block, 0, stream>>>(
            group_size, split_k_iters, in_feats, kernel, scaling_factors, zeros,
            num_in_feats, num_in_channels, num_out_channels, out_feats);
  }
  return _out_feats.sum(0);
}

TORCH_LIBRARY(nanovllm, m) {
    m.def("awq_gemm(Tensor in_feats, Tensor kernel, "
           "Tensor scaling_factors, Tensor zeros, "
           "int split_k_iters) -> Tensor");
}

TORCH_LIBRARY_IMPL(nanovllm, CUDA, m) {
    m.impl("awq_gemm", &awq_gemm);
}
```

- [ ] **Step 2: Build and verify compilation**

```bash
pip install -e .
```

Expected: nvcc compiles successfully, produces `nanovllm/_C*`.so in site-packages.

- [ ] **Step 3: Verify op is registered**

```bash
python -c "print(torch.ops.nanovllm.awq_gemm)"
```

Expected: `<torch._ops.OpOverloadPacket at 0x...>`

---

### Task 4: Python integration

**Files:**
- Modify: `nanovllm/__init__.py`
- Modify: `nanovllm/layers/quantization/awq.py`

- [ ] **Step 1: Modify nanovllm/__init__.py**

Add import of the CUDA extension to trigger `TORCH_LIBRARY` registration:

```python
from nanovllm.llm import LLM
from nanovllm.sampling_params import SamplingParams
import nanovllm._C  # trigger TORCH_LIBRARY registration for AWQ CUDA op
```

- [ ] **Step 2: Modify AWQColumnParallelLinear.forward**

Replace the two-tier dispatch (line 112-121) with three-tier:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    M = x.size(-2)
    if M <= 16:
        return torch.ops.nanovllm.awq_gemm(
            x, self.qweight, self.scales, self.qzeros, 8)
    elif M < 256:
        return awq_gemm_triton(x, self.qweight, self.scales,
                               self.qzeros, self.group_size, split_k_iters=8)
    weight = self._dequantize_weight()
    weight = weight.t().to(x.dtype)
    return F.linear(x, weight, self.bias)
```

- [ ] **Step 3: Modify AWQRowParallelLinear.forward**

Replace the two-tier dispatch (line 293-304) with three-tier:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
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

- [ ] **Step 4: Verify basic import and forward pass**

```bash
python -c "
import torch
# Verify op is registered
print('Op:', torch.ops.nanovllm.awq_gemm)
# Verify module loads
from nanovllm import LLM
print('LLM import OK')
"
```

---

### Task 5: Correctness verification

- [ ] **Step 1: Compare CUDA op vs Triton kernel output**

```bash
python -c "
import torch
from nanovllm.layers.quantization.awq_triton import awq_gemm_triton

torch.manual_seed(42)
K, M, N, G = 4096, 4, 2048, 128
num_groups = K // G

qweight = torch.randint(0, 2**31, (K, N // 8), dtype=torch.int32, device='cuda')
scales = torch.randn(num_groups, N, dtype=torch.float16, device='cuda') * 0.1
qzeros = torch.randint(0, 2**31, (num_groups, N // 8), dtype=torch.int32, device='cuda')
act = torch.randn(M, K, dtype=torch.float16, device='cuda')

ref = awq_gemm_triton(act, qweight, scales, qzeros, G, split_k_iters=8)
out = torch.ops.nanovllm.awq_gemm(act, qweight, scales, qzeros, 8)

diff = (ref - out).abs().max().item()
print(f'M={M}: max diff = {diff:.6f}')
assert diff < 0.125, f'Max diff {diff} exceeds threshold'
print('PASS')
"
```

Run for M=1, 4, 8, 16:

```
python -c "for m in [1, 4, 8, 16]: ..."
```

- [ ] **Step 2: Run end-to-end inference**

```bash
python example.py
```

Expected: model generates normally, similar output quality.

- [ ] **Step 3: Run benchmark**

```bash
python bench.py
```

Expected: throughput comparable to or better than Triton-only baseline (~800 tok/s).

---

### Task 6: Cleanup and commit

- [ ] **Step 1: Commit all changes**

```bash
git add setup.py nanovllm/csrc/ nanovllm/__init__.py nanovllm/layers/quantization/awq.py pyproject.toml
git commit -m "feat: CUDA AWQ GEMM custom op via torch TORCH_LIBRARY

- Port vllm's PTX-optimized gemm_forward_4bit_cuda_m16nXk32 kernel
- Register as torch.ops.nanovllm.awq_gemm via TORCH_LIBRARY
- Three-tier dispatch: M<=16 CUDA, 16<M<256 Triton, M>=256 dequant+cuBLAS
- setup.py with CUDAExtension for automated compilation"
```

---

## Rollback Plan

If the CUDA extension fails to compile:
```bash
# Revert all CUDA-specific changes
git checkout -- setup.py pyproject.toml nanovllm/__init__.py nanovllm/layers/quantization/awq.py
rm -rf nanovllm/csrc/
pip install -e .  # Rebuild without CUDA extension
```

This restores the Triton-only state (Phase 2), which is fully functional.
