# Marlin Kernel Integration Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate vllm's Marlin kernel as a separate CUDA extension (`nanovllm._C_marlin`) to replace the current AWQ fused CUDA kernel + dequant+cuBLAS two-way dispatch with a unified Marlin GEMM for all M values.

**Architecture:** Copy Marlin CUDA files from `../vllm/`, rename `vllm::` namespace to `nanovllm::`, build as a second CUDAExtension alongside the existing `nanovllm._C`. Add `_USE_MARLIN = True/False` flag in `awq.py` to switch between legacy and Marlin paths at one line.

**Tech Stack:** CUDA C++ (Marlin template kernel), PyTorch TORCH_LIBRARY, Triton, cuBLAS

---

## File Structure

### New files (created under nanovllm/csrc from copies of ../vllm/csrc/)

```
nanovllm/csrc/core/
├── scalar_type.hpp          ← copy from ../vllm/csrc/core/, s/vllm/nanovllm/, strip to kFloat16+kU4 only
└── registration.h           ← copy from ../vllm/csrc/core/, s/vllm/nanovllm/

nanovllm/csrc/marlin/
├── kernel.h                 ← copy from ../vllm/csrc/quantization/marlin/kernel.h, s/vllm::/nanovllm::/
├── marlin.cuh               ← copy from ../vllm/csrc/quantization/marlin/marlin.cuh (namespace is MARLIN_NAMESPACE_NAME, no vllm::)
├── marlin_dtypes.cuh        ← copy from ../vllm/csrc/quantization/marlin/marlin_dtypes.cuh, s/vllm::/nanovllm::/
├── marlin_mma.h             ← copy from ../vllm/csrc/quantization/marlin/marlin_mma.h (no vllm:: namespace)
├── marlin_template.h        ← copy from ../vllm/csrc/quantization/marlin/marlin_template.h, s/vllm::/nanovllm::/
├── marlin.cu                ← copy from ../vllm/csrc/quantization/marlin/marlin.cu, s/vllm::/nanovllm::/ + s/namespace marlin/namespace nanovllm_marlin/
├── awq_marlin_repack.cu     ← copy from ../vllm/csrc/quantization/marlin/awq_marlin_repack.cu, s/namespace marlin/namespace nanovllm_marlin/
├── kernel_selector.h        ← copy from ../vllm/csrc/quantization/marlin/kernel_selector.h, s/vllm::/nanovllm::/ everywhere
└── sm80_kernel_float16_u4_float16.cu  ← copy from ../vllm/csrc/quantization/marlin/, s/vllm::/nanovllm::/
```

### Modified files

| File | Change |
|------|--------|
| `setup.py` | Add second CUDAExtension `nanovllm._C_marlin` with marlin .cu files |
| `nanovllm/layers/quantization/awq.py` | Add `_USE_MARLIN` flag, workspace buffer, weight repack, marlin_gemm forward |
| `profile/microbench_awq.py` | Add Marlin benchmark column |

---

### Task 1: Create core dependency files (scalar_type.hpp, registration.h)

**Files:**
- Create: `nanovllm/csrc/core/scalar_type.hpp`
- Create: `nanovllm/csrc/core/registration.h`

- [ ] **Create scalar_type.hpp from vllm version**

Read `../vllm/csrc/core/scalar_type.hpp`, copy to `nanovllm/csrc/core/scalar_type.hpp` with:
- `namespace vllm` → `namespace nanovllm`
- Remove all type definitions except `kFloat16` and `kU4` (remove kBFloat16, kS8, kFE4M3fn, kFE2M1f, kU4B8, kU8B128, kS4, kS8, kFE8M0fnu)

Keep all type definitions — only change `namespace vllm` → `namespace nanovllm`. Even though AWQ only uses fp16×u4, the `marlin.cu` host wrapper code references `kBFloat16`, `kFE4M3fn`, `kS8`, `kFE2M1f`, `kFE8M0fnu`, `kU4B8`, `kU8B128`, `kS4` in its `marlin_gemm()` dispatch function, and the compiler needs these symbols to exist. The overhead is negligible (a few dozen constexpr variables in a header).

- [ ] **Create registration.h from vllm version**

Copy `../vllm/csrc/core/registration.h` → `nanovllm/csrc/core/registration.h` with:
- No namespace changes needed (no vllm namespace in this file)
- Just a direct copy

### Task 2: Copy Marlin CUDA files with namespace replacement

**Files:**
- Create: `nanovllm/csrc/marlin/kernel.h`
- Create: `nanovllm/csrc/marlin/marlin.cuh`
- Create: `nanovllm/csrc/marlin/marlin_dtypes.cuh`
- Create: `nanovllm/csrc/marlin/marlin_mma.h`
- Create: `nanovllm/csrc/marlin/marlin_template.h`
- Create: `nanovllm/csrc/marlin/marlin.cu`
- Create: `nanovllm/csrc/marlin/awq_marlin_repack.cu`
- Create: `nanovllm/csrc/marlin/kernel_selector.h`
- Create: `nanovllm/csrc/marlin/sm80_kernel_float16_u4_float16.cu`

- [ ] **Copy files with `vllm::` → `nanovllm::` replacement**

For each file in the list, copy from `../vllm/csrc/quantization/marlin/` to `nanovllm/csrc/marlin/` and apply namespace transformations:

| Source file | Changes needed |
|-------------|---------------|
| `kernel.h` | s/vllm::ScalarType/nanovllm::ScalarType/ (x5), include path `core/scalar_type.hpp` stays same (it's resolved relative to include path) |
| `marlin.cuh` | No `vllm::` refs; `MARLIN_NAMESPACE_NAME` handles namespace |
| `marlin_dtypes.cuh` | s/vllm::/nanovllm::/ for ScalarType includes |
| `marlin_mma.h` | s/vllm::/nanovllm::/ — has `vllm::ScalarTypeId` |
| `marlin_template.h` | s/vllm::/nanovllm::/ for all ScalarType ops |
| `marlin.cu` | s/vllm::/nanovllm::/ for all ScalarType refs, s/namespace marlin /namespace nanovllm_marlin /, keep `TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, m)` as-is |
| `awq_marlin_repack.cu` | s/namespace marlin /namespace nanovllm_marlin / (x2), keep `TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, m)` as-is |
| `kernel_selector.h` | Replace ALL `vllm::` → `nanovllm::` (this file is ~390KB with many template instantiations) |
| `sm80_kernel_float16_u4_float16.cu` | Replace ALL `vllm::` → `nanovllm::` (130 lines, 48 instantiations) |

Use `sed` for bulk replacements:
```bash
# For kernel_selector.h (large file)
sed -i 's/vllm::/nanovllm::/g' nanovllm/csrc/marlin/kernel_selector.h

# For other files with vllm:: references
sed -i 's/vllm::/nanovllm::/g' nanovllm/csrc/marlin/kernel.h
sed -i 's/vllm::/nanovllm::/g' nanovllm/csrc/marlin/marlin_dtypes.cuh
sed -i 's/vllm::/nanovllm::/g' nanovllm/csrc/marlin/marlin_template.h
sed -i 's/vllm::/nanovllm::/g' nanovllm/csrc/marlin/marlin_mma.h
sed -i 's/vllm::/nanovllm::/g' nanovllm/csrc/marlin/marlin.cu
sed -i 's/vllm::/nanovllm::/g' nanovllm/csrc/marlin/sm80_kernel_float16_u4_float16.cu
```

For namespace changes in marlin.cu and awq_marlin_repack.cu:
```bash
# marlin.cu: line 34 "namespace marlin {" → "namespace nanovllm_marlin {"
sed -i 's/^namespace marlin {/namespace nanovllm_marlin {/' nanovllm/csrc/marlin/marlin.cu

# awq_marlin_repack.cu: line 5 "namespace marlin {" → "namespace nanovllm_marlin {"
# and line 207 "}  // namespace marlin" stays as-is
sed -i 's/^namespace marlin {/namespace nanovllm_marlin {/' nanovllm/csrc/marlin/awq_marlin_repack.cu
```

No namespace changes needed for `marlin.cuh` — it uses `MARLIN_NAMESPACE_NAME` macro.

### Task 3: Modify setup.py for Marlin CUDAExtension

**Files:**
- Modify: `setup.py`

- [ ] **Add marlin CUDAExtension and include path**

Add a second CUDAExtension `nanovllm._C_marlin` alongside the existing one:

```python
CUDAExtension(
    "nanovllm._C_marlin",
    [
        "nanovllm/csrc/marlin/marlin.cu",
        "nanovllm/csrc/marlin/awq_marlin_repack.cu",
        "nanovllm/csrc/marlin/sm80_kernel_float16_u4_float16.cu",
    ],
    extra_compile_args={
        "cxx": ["-O3", "-std=c++17"],
        "nvcc": [
            "-O3",
            "-std=c++17",
            "--generate-code=arch=compute_89,code=sm_89",
            "-DMARLIN_NAMESPACE_NAME=nanovllm_marlin",
            "--use_fast_math",
        ],
    },
    include_dirs=[
        # So that #include "core/scalar_type.hpp" resolves correctly
        "nanovllm/csrc",
    ],
),
```

Note: `marlin_template.h` and `kernel_selector.h` are headers included transitively, not compiled directly.

- [ ] **Build and check compilation**

```bash
python setup.py build_ext --inplace 2>&1 | tail -30
```

Expected: compilation succeeds, `nanovllm/_C_marlin*.so` is created.
If compilation fails, fix `vllm::` remnants and retry.

### Task 4: Modify awq.py with Marlin support

**Files:**
- Modify: `nanovllm/layers/quantization/awq.py`

- [ ] **Add _USE_MARLIN flag and workspace buffer**

At the top of `awq.py`, after imports:

```python
_USE_MARLIN = True  # Set False to use legacy AWQ CUDA kernel

if _USE_MARLIN:
    import nanovllm._C_marlin  # noqa: F401 — registers torch.ops.nanovllm._C_marlin.*
    _num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    _workspace = torch.zeros(_num_sms, dtype=torch.int32, device="cuda")
```

- [ ] **Add Marlin repack buffer + post-load repack**

**Key insight:** `AWQMergedColumnParallelLinear` and `AWQQKVParallelLinear` override `weight_loader` without calling `super()`. `loader.py` calls `param.weight_loader()` directly per-parameter — it never calls `nn.Module.load_state_dict()`. So neither the per-`weight_loader` approach nor a `load_state_dict` override works for subclasses.

**Solution:** Define `_marlin_repack()` on `_AWQBase`. `load_model` in `loader.py` calls it on all submodules after loading weights.

In `_AWQBase.__init__`, register no buffers — they're created in subclass `__init__` where shapes are known.

In each layer class `__init__` (`AWQColumnParallelLinear`, `AWQRowParallelLinear`), after existing parameter definitions:

```python
if _USE_MARLIN:
    # Marlin format buffers. Populated by _marlin_repack() after weight loading.
    # qweight [K, N//8] → [K//16, N*2]; qzeros [G, N//8] → [G//16, N*2]
    self.register_buffer("marlin_qweight", torch.empty(
        self.qweight.size(0) // 16, self.qweight.size(1) * 2,
        dtype=torch.int32))
    self.register_buffer("marlin_qzeros", torch.empty(
        self.qzeros.size(0) // 16, self.qzeros.size(1) * 2,
        dtype=torch.int32))
```

Add `_marlin_repack` to `_AWQBase`:

```python
def _marlin_repack(self):
    """Repack qweight/qzeros → Marlin format. Called after weight loading."""
    if not _USE_MARLIN:
        return
    if not hasattr(self, 'marlin_qweight'):
        return
    ic = self.qweight.size(0)
    oc = self.qweight.size(1) * 8  # pack_factor = 8
    self.marlin_qweight.copy_(
        torch.ops.nanovllm._C_marlin.awq_marlin_repack(
            self.qweight.data, ic, oc, 4, False))
    g = self.qzeros.size(0)
    oc_zp = self.qzeros.size(1) * 8
    self.marlin_qzeros.copy_(
        torch.ops.nanovllm._C_marlin.awq_marlin_repack(
            self.qzeros.data, g, oc_zp, 4, False))
```

- [ ] **Modify loader.py to call marlin_repack post-load**

In `loader.py`, at end of `load_model()`:

```python
from nanovllm.layers.quantization.awq import _USE_MARLIN

def load_model(model: nn.Module, path: str):
    # ... existing code ...
    # Post-load repack for Marlin weight format conversion
    if _USE_MARLIN:
        for module in model.modules():
            if hasattr(module, '_marlin_repack'):
                module._marlin_repack()
```

- [ ] **Modify forward to use marlin_gemm**

In `AWQColumnParallelLinear.forward`, replace entire forward body:

```python
if _USE_MARLIN:
    orig_dtype = x.dtype
    if x.dtype != torch.float16:
        x = x.to(torch.float16)
    y = torch.ops.nanovllm._C_marlin.marlin_gemm(
        x,                  # a: fp16[M, K]
        None,               # c: partial sum buffer (not needed)
        self.marlin_qweight, # b_q_weight: int32[K//16, N*2]
        self.bias if self.bias is not None else None,  # b_bias
        self.scales,        # b_scales: fp16[G, N]
        None,               # a_scales (W4A16 doesn't need)
        None,               # global_scale (not needed)
        self.marlin_qzeros, # b_zeros: int32[G//16, N*2]
        None,               # g_idx (no act_order)
        None,               # perm (no act_order)
        _workspace,         # int32[num_SMs]
        2,                  # b_q_type: kU4.id() = 2
        x.size(0),          # size_m
        y.size(1),          # size_n  -- wait, we don't have y yet
    )
```

Actually, looking at the python-side vllm to understand the ScalarType IDs:
- `kFloat16.id() = 0`
- `kU4.id() = 2`

But we stripped scalar_type.hpp. Let's verify: in the original scalar_type.hpp, the types are likely numbered in declaration order. I'll need to check the actual IDs.

Let me check:
```cpp
inline constexpr ScalarType kFloat16(0, ...);   // id = 0
inline constexpr ScalarType kU4(2, ...);         // id = 2 (assuming BF16 is 1)
```

Actually wait, the ScalarType constructor takes an id as first parameter. Let me read the full scalar_type.hpp to see the declarations.

I'll need to read it in the plan execution phase. For now, I'll note that we need to verify the IDs.

For the forward method, the simplest approach: replace the entire `forward` body for `_USE_MARLIN`:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    if _USE_MARLIN:
        orig_dtype = x.dtype
        if x.dtype != torch.float16:
            x = x.to(torch.float16)
        y = torch.ops.nanovllm._C_marlin.marlin_gemm(
            x, None, self.marlin_qweight,
            self.bias, self.scales,
            None, None, self.marlin_qzeros,
            None, None, _workspace,
            2,  # kU4.id()
            x.size(0), self.scales.size(1), x.size(1),
            True,  # is_k_full (AWQ has no act_order)
            False,  # use_atomic_add
            False,  # use_fp32_reduce
            False,  # is_zp_float (AWQ zero point is int4)
        )
        if orig_dtype != torch.float16:
            y = y.to(orig_dtype)
        return y

    # Original dispatch (when _USE_MARLIN = False)
    M = x.size(-2)
    if M < 512:
        ...
```

Similarly for `AWQRowParallelLinear.forward`, with the same marlin_gemm call but on `x_shard`.

- [ ] **Similar modifications for AWQRowParallelLinear**

For `AWQRowParallelLinear.forward`, same Marlin GEMM call but on `x_shard`:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    x_shard = x.narrow(-1, self.tp_rank * (x.size(-1) // self.tp_size),
                       x.size(-1) // self.tp_size)

    if _USE_MARLIN:
        orig_dtype = x_shard.dtype
        if x_shard.dtype != torch.float16:
            x_shard = x_shard.to(torch.float16)
        y = torch.ops.nanovllm._C_marlin.marlin_gemm(
            x_shard, None, self.marlin_qweight,
            self.bias if self.tp_rank == 0 else None,
            self.scales, None, None, self.marlin_qzeros,
            None, None, _workspace,
            2,
            x_shard.size(0), self.scales.size(1), x_shard.size(1),
            True, False, False, False,
        )
        if orig_dtype != torch.float16:
            y = y.to(orig_dtype)
    else:
        # Original dispatch
        ...

    if self.tp_size > 1:
        dist.all_reduce(y)
    return y
```

### Task 5: Update microbench_awq.py

**Files:**
- Modify: `profile/microbench_awq.py`

- [ ] **Add Marlin benchmark column**

Changes:
1. Add `import nanovllm._C_marlin` at top
2. Create marlin-formatted weights during setup
3. Add Marlin benchmark function and column in output

```python
# After imports
_USE_MARLIN = True
if _USE_MARLIN:
    import nanovllm._C_marlin  # noqa: F401
    _num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    _workspace = torch.zeros(_num_sms, dtype=torch.int32, device='cuda')
```

In `create_awq_weights`, add Marlin repack step:
```python
marlin_qweight = torch.ops.nanovllm._C_marlin.awq_marlin_repack(qweight, K, N, 4, False)
marlin_qzeros = torch.ops.nanovllm._C_marlin.awq_marlin_repack(qzeros, NUM_GROUPS, N, 4, False)
```

Update benchmark loop:
```python
# 4. Marlin GEMM
def fn_marlin(a=act, w=marlin_qweight, s=scales, z=marlin_qzeros):
    return torch.ops.nanovllm._C_marlin.marlin_gemm(
        a, None, w, None, s, None, None, z,
        None, None, _workspace, 2,
        M, N, K, True, False, False, False,
    )
t_marlin = benchmark(fn_marlin, "Marlin", M)

# Update header and best calculation
times = {"CUDA": t_cuda, "Triton": t_triton, "deq+cuB": t_deq, "Marlin": t_marlin}
```

### Task 6: Build and verify compilation

- [ ] **Build both extensions**

```bash
cd /home/yeguorong/code/grYe-nano-vllm
python setup.py build_ext --inplace 2>&1
```

Expected: both `nanovllm._C` and `nanovllm._C_marlin` compile successfully.
If compilation fails:
- Check for missed `vllm::` → `nanovllm::` replacements
- Verify `scalar_type.hpp` has the correct includes
- Check that `MARLIN_NAMESPACE_NAME` is correctly defined via compiler flag
- Check `#include` paths resolve correctly (they use `core/scalar_type.hpp` relative to include_dirs)

- [ ] **Verify import works**

```bash
python -c "import nanovllm._C_marlin; print(torch.ops.nanovllm._C_marlin.marlin_gemm)"
python -c "import nanovllm._C_marlin; print(torch.ops.nanovllm._C_marlin.awq_marlin_repack)"
```

Expected: both ops are registered and accessible.

### Task 7: Correctness verification

**Files:**
- Test: `example.py` or a dedicated correctness script

- [ ] **Write and run correctness test**

Create quick test comparing Marlin output vs dequant+cuBLAS baseline:

```python
import torch
import nanovllm._C_marlin

K, N = 4096, 4096
GROUP_SIZE = 128
NUM_GROUPS = K // GROUP_SIZE

# Create random AWQ weights
qweight = torch.randint(0, 2**31, (K, N // 8), dtype=torch.int32, device='cuda')
qzeros = torch.randint(0, 2**31, (NUM_GROUPS, N // 8), dtype=torch.int32, device='cuda')
scales = torch.randn(NUM_GROUPS, N, dtype=torch.float16, device='cuda') * 0.1

# Repack to Marlin format
marlin_qweight = torch.ops.nanovllm._C_marlin.awq_marlin_repack(qweight, K, N, 4, False)
marlin_qzeros = torch.ops.nanovllm._C_marlin.awq_marlin_repack(qzeros, NUM_GROUPS, N, 4, False)

# Baseline: dequant + cuBLAS
act = torch.randn(128, K, dtype=torch.float16, device='cuda')

# Marlin result
workspace = torch.zeros(torch.cuda.get_device_properties(0).multi_processor_count, dtype=torch.int32, device='cuda')
out_marlin = torch.ops.nanovllm._C_marlin.marlin_gemm(
    act, None, marlin_qweight, None, scales, None, None, marlin_qzeros,
    None, None, workspace, 2, 128, N, K, True, False, False, False,
)

# Baseline (dequant + matmul)
from nanovllm.layers.quantization.awq_triton import awq_dequantize
deq = awq_dequantize(qweight, scales, qzeros, GROUP_SIZE)
out_baseline = act @ deq.t()

max_diff = (out_marlin - out_baseline).abs().max().item()
print(f"Max diff: {max_diff:.6f}")
assert max_diff < 1e-2, f"Max diff too large: {max_diff}"
```

Test across multiple M values: `[1, 16, 32, 64, 128, 256, 512, 1024, 4096]`.

- [ ] **Run end-to-end test with Qwen3-0.6B-AWQ**

```bash
python example.py
```

Expected: model loads and generates coherent output.

### Task 8: Performance benchmark

**Files:**
- Test: `profile/microbench_awq.py`

- [ ] **Run microbenchmark and compare**

```bash
python profile/microbench_awq.py
```

Expected output table with 4 columns: CUDA op, Triton, dequant+cuBLAS, Marlin.
Analyze which kernel wins for each M value.

- [ ] **Run end-to-end benchmark**

```bash
python bench.py
```

Compare throughput before and after Marlin integration (toggle `_USE_MARLIN`).

### Task 9: Cleanup and commit

- [ ] **Remove temporary debug prints and clean up code**
- [ ] **Commit all changes**

```bash
git add -A
git commit -m "feat: integrate Marlin kernel for AWQ inference

Add Marlin kernel from vllm as a separate CUDA extension (nanovllm._C_marlin).
Replace legacy CUDA fused + dequant+cuBLAS two-way dispatch with unified
Marlin GEMM for all M values. Controllable via _USE_MARLIN flag in awq.py.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```
