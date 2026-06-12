# PyNccl: Bypassing torch.distributed Overhead for Tensor Parallel AllReduce

## 1. Background: TP and the all_reduce Bottleneck

### Why TP Needs all_reduce

Tensor parallelism (TP) splits each model layer's weight matrices across GPUs. At each `RowParallelLinear` boundary, each GPU holds a partial result (a column shard of the output). These partial results must be summed across all GPUs to reconstruct the correct output:

```
Input: x                     [batch, hidden_dim]
  ↓
Weight: W = [W_0 | W_1]      Each GPU has half the columns
  ↓
Each GPU: y_i = x @ W_i      [batch, hidden_dim/2] — partial result
  ↓
all_reduce(y_i) → y          Sum across GPUs → [batch, hidden_dim]
```

Without `all_reduce`, each GPU's output would represent only its local shard, yielding incorrect logits.

### Decode Step: 48 all_reduces per Step

In a 24-layer transformer with TP, each decode step executes:

| Location | all_reduce calls | Per layer | Total |
|----------|-----------------|-----------|-------|
| Attention output projection (o_proj) | RowParallelLinear | 1 | 24 |
| MLP down projection (down_proj) | RowParallelLinear | 1 | 24 |
| **Total** | | **2** | **48** |

48 `all_reduce` calls per decode step. Each call must complete before the subsequent layer can proceed.

### Raw Profiler Data (2x RTX 4090, PCIe Gen4, eager mode)

Profiling with NVIDIA Nsight Systems on Qwen3-0.6B with TP=2:

| Metric | Count | Avg CPU time | Avg GPU time | Total CPU/step |
|--------|-------|-------------|-------------|----------------|
| `c10d::allreduce_` | 48/step | 96.8 us | — | 4.65 ms |
| NCCL kernel (actual) | 48/step | — | 7.95 us | 0.38 ms |
| Ratio (CPU:GPU) | | **12.2x** | | |

The CPU spends **12x more time** dispatching `all_reduce` than the GPU spends actually communicating. The GPU is idle, waiting for the next kernel launch.

> **Key insight:** On PCIe, this is not a GPU bandwidth problem. It is a **CPU dispatch overhead** problem.

---

## 2. torch.distributed all_reduce Call Chain

### What Happens When You Call `dist.all_reduce(tensor)`

```
Python:   dist.all_reduce(tensor)
            ↓
C++:      ProcessGroupNCCL::all_reduce
            ├── record_param_comms (profiler hook)
            ├── NCCL communicator lookup
            ├── Synchronization bookkeeping
            ├── cuLaunchKernel(ncclAllReduce, ..., stream)
            ↓
GPU:      ncclDevKernel_AllReduce  ← actual communication
```

### Per-Call Cost Breakdown (from 2x3080Ti Trace)

Using Nsight Systems trace on 2x RTX 3080Ti (TP=2, eager mode):

| Event | Count | Avg Duration | P50 | % of c10d::allreduce_ |
|-------|-------|:---:|:---:|:---:|
| `c10d::allreduce_` | 14,592 | 103.1 us | 91.3 us | 100% |
| `record_param_comms` (nested inside) | 14,848 | 85.8 us | 74.2 us | **84.6%** |
| c10d setup + teardown (excl. comms) | — | 17.0 us | 17.0 us | **15.4%** |
| NCCL GPU kernel | — | ~22 us (GPU) | — | (GPU, not CPU) |

**Key finding: `record_param_comms` is nested INSIDE `c10d::allreduce_` and wraps the `ncclAllReduce` call itself.** It is not "extra overhead on top" — it is the profiler scope that contains the actual kernel launch. Removing it saves the profiler recording cost but the remaining c10d setup/teardown is only ~17 us.

**Timeline of one call (from trace):**
```
c10d::allreduce_ [199.9us] ←── 整个函数
  ├── c10d setup (validation, lookup): 35.2us
  ├── record_param_comms [161.0us] ←── 包含 ncclAllReduce
  │     ├── profiler setup + cuLaunchKernel: 83.7us
  │     │     └── nccl kernel GPU exec 22.2us ← 在此期间
  │     └── profiler teardown + sync: 55.2us
  └── c10d teardown: 3.7us
```

### Why torch.distributed Adds So Much Overhead

1. **Profiler hooks (`record_param_comms`):** Every collective operation records events for `torch.profiler`. This allocates memory, acquires locks, and writes trace data — even when no profiler is active.

2. **General-purpose design:** `ProcessGroupNCCL` handles dynamic communicator management, multiple process group backends, async operations, error handling, and compatibility with PyTorch's autograd and distributed data parallel (DDP) frameworks. TP inference doesn't need any of this.

3. **GIL interactions:** The C++ dispatcher must coordinate with the Python GIL for thread safety, adding synchronization overhead.

4. **NCCL communicator lookup:** Each call resolves the NCCL communicator pointer from a map, even though TP inference uses a single static communicator.

5. **CUDA stream synchronization:** `torch.distributed` inserts additional stream synchronization to ensure correctness across composable distributed strategies.

---

## 3. PyNccl Design

### Architecture

PyNccl bypasses the entire `torch.distributed` stack by calling NCCL directly via Python's `ctypes`:

```
Before (torch.distributed):
  Python → C++ (c10d) → cuLaunchKernel → NCCL GPU kernel
            ↑ 97 us CPU overhead

After (PyNccl):
  Python → cuLaunchKernel → NCCL GPU kernel
            ↑ ~17 us CPU overhead
```

### Key Components

**NCCL binding via ctypes:**

```python
import ctypes
libnccl = ctypes.CDLL("libnccl.so")
libnccl.ncclAllReduce.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
    ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
]
```

**Direct all_reduce:**

```python
def all_reduce(self, tensor):
    stream = torch.cuda.current_stream()
    self._lib.ncclAllReduce(
        tensor.data_ptr(),    # send buffer
        tensor.data_ptr(),    # recv buffer (in-place)
        tensor.numel(),
        dtype_map[tensor.dtype],
        ncclSum,
        self._comm,
        stream.cuda_stream,   # raw CUDA stream handle
    )
```

### What's Eliminated

| Overhead | Source | Eliminated? |
|----------|--------|-------------|
| `record_param_comms` | c10d profiler hook | Yes |
| C++ dispatcher setup | ProcessGroupNCCL | Yes |
| Communicator lookup | Dynamic map lookup | Yes (static handle) |
| Stream sync | c10d bookkeeping | Yes |
| GIL coordination | Python/C++ boundary | Partially |
| cuLaunchKernel | CUDA driver | Still present |
| NCCL GPU kernel | NCCL library | Still present |

### Remaining Cost

| Component | Time |
|-----------|------|
| ctypes function call | ~1 us |
| cuLaunchKernel | ~10 us |
| NCCL GPU kernel | ~8 us |
| **Total per call** | **~19 us** |

Target: ~17-20 us per call vs ~97 us with torch.distributed — a **~5x reduction** in CPU overhead.

---

## 4. CUDA Graph Compatibility

### What Are CUDA Graphs?

CUDA graphs capture a sequence of GPU kernel launches and replay them without CPU involvement. This eliminates kernel launch overhead during inference.

### How CUDA Graphs Interact with NCCL

**During graph capture** (`torch.cuda.graph`):

1. `model.forward()` calls `dist.all_reduce(tensor)` or `pynccl.all_reduce(tensor)`
2. `ncclAllReduce` is launched on the CUDA capture stream
3. `torch.cuda.graph` captures the NCCL kernel as part of the graph

**During replay:**

1. `graph.replay()` replays all captured kernels from GPU instruction buffer
2. The NCCL kernel is replayed from the graph — **no cuLaunchKernel, no host code**
3. All CPU-side code (c10d, record_param_comms, ctypes calls) is **completely bypassed**

### NCCL 2.9+: The Capture Mode Split

Before NCCL 2.9, `ncclAllReduce` used CUDA API calls like `cudaEventCreate`/`cudaEventRecord` for internal synchronization — these are **forbidden during stream capture** (`"operation not permitted when stream is capturing"`).

NCCL 2.9+ restructured the launch into three phases:

```
ncclAllReduce(stream):
  ├── Phase 1: Before_NoUncapturedCuda
  │     └── 使用预分配 event, 只做 capture-safe 的 CUDA 操作
  ├── Phase 2: LaunchKernel ←── 被录进 graph
  │     └── cuLaunchKernel(nccl_kernel, ..., stream)
  └── Phase 3: After_NoCuda ←── capture stream 之外执行
        └── 通过 IPC/socket 提交 proxy op (延迟到 capture 结束后)
```

Before 2.9, `dist.all_reduce` inside `torch.cuda.graph()` would **throw an error**. After 2.9, NCCL's GPU kernels can be captured, while proxy operations still run outside the capture block (unchanged from before).

### Important: record_param_comms

`record_param_comms` is a Kineto profiler hook in `ProcessGroupNCCL` that runs **even when no profiler is active** — it accounts for **84.6%** of `c10d::allreduce_` CPU time. This is because Kineto is always-on to support on-demand profiling. There is no official API to disable it.

This overhead is only relevant in **eager mode**: during CG capture it runs once (one-time cost), and during CG replay it is completely bypassed.

### CG Replay: Transport-Dependent Behavior

Whether CG replay truly eliminates all CPU overhead depends on NCCL's underlying transport:

| Transport | CG replay behavior | CPU overhead during replay |
|:---------|:-------------------|:---:|
| **P2P** (NVLink/A100 PCIe) | GPU kernel captured → replay pure GPU | **0** |
| **SHM** (consumer GPU, P2P locked) | GPU kernel captured, **proxy thread still runs outside graph** | **proxy CPU overhead remains** |

For SHM transport, the CPU proxy thread polls head/tail counters in host memory and issues `cudaMemcpyAsync` calls to forward data between GPUs. This **cannot be captured** by CUDA Graph. The proxy thread runs independently before/after graph segments, using CUDA events for inter-stream signaling with the captured graph.

```
CG replay with SHM:
  graph.replay()
    ├── GPU kernel_A (captured) → 写 SHM, 更新 head
    ├── cudaStreamWaitEvent(event) ← 等 proxy 转发完
    │     ←── proxy thread: polling → cudaMemcpyAsync → event signal
    └── GPU kernel_B (captured) → 读转发后的数据
```

So on consumer GPUs (RTX 4090/3080Ti), **CG does not eliminate all all_reduce CPU overhead** — the proxy thread still consumes CPU cycles. This is a fundamental difference from enterprise GPUs with P2P.

---

## 5. NCCL Transport: P2P vs SHM

### How NCCL Selects a Transport

NCCL registers transports with priority: **P2P > SHM > NET > CollNet**. When `canConnect` returns true, that transport is bound.

### P2P Transport (Enterprise GPUs: A100/H100)

On NVLink or PCIe with P2P enabled:
- GPU can **directly LD/ST peer GPU's memory** via NVLink or BAR1 mapping
- No CPU involvement in data path — GPU reads/writes peer memory directly in kernel code
- Works with both NVLink and PCIe P2P (when driver/BIOS permit)

```
GPU0 kernel:  peer[1].buffer = my_data;  // GPU writes directly to GPU1's HBM
              __threadfence();
              sum = peer[1].buffer + my_data;
```

### SHM Transport (Consumer GPUs: RTX 4090/3080Ti)

When P2P is **locked by NVIDIA driver** (consumer SKU restriction):
- NCCL falls back to **SHM (Shared Memory)** transport
- Data path: `GPU0 → PCIe write → Host Memory → PCIe read → GPU1`
- A **CPU proxy thread** handles data forwarding:

```
┌─────────────┐     PCIe write      ┌──────────────┐     PCIe read      ┌─────────────┐
│   GPU 0     │ ──────────────────→ │  Host SHM    │ ──────────────────→ │   GPU 1     │
│  (kernel)   │                     │  (pinned)    │                     │  (kernel)   │
└─────────────┘                     └──────┬───────┘                     └─────────────┘
                                           │
                                    CPU Proxy Thread
                                    (poll head/tail,
                                     cudaMemcpyAsync)
```

**Pseudo-code of SHM data flow:**

```python
# === GPU kernel (写端) ===
__global__ void allreduce_step1(data, shm_buf, head):
    slot = head % NCCL_STEPS
    shm_buf[slot] = data
    __threadfence_system()           # 保证 host 侧可见
    head += 1                        # 通知 proxy: 数据就绪

# === CPU Proxy Thread (后台独立线程) ===
def proxy_progress():
    while True:
        if proxy_head > proxy_tail:  # GPU 有新数据?
            cudaMemcpyAsync(...)     # 转发到另一张 GPU
            proxy_tail += 1
        sleep_or_poll()

# === GPU kernel (读端) ===
__global__ void allreduce_step2(data, shm_buf, tail):
    while tail < expected:           # 等 proxy 转发完
        poll()
    data = shm_buf[slot]
```

### Why "PCIe Requires CPU" Is Inaccurate

The accurate statement depends on GPU class:

| GPU | P2P Available? | Transport | Data Path Needs CPU? |
|-----|:---:|:---------|:---:|
| A100/H100 PCIe | ✅ (driver allows) | P2P | ❌ GPU LD/ST peer BAR1 |
| A100/H100 NVLink | ✅ (NVLink) | P2P | ❌ GPU LD/ST peer HBM |
| **RTX 4090/3080Ti** | **❌ (driver locked)** | **SHM** | **✅ CPU proxy thread** |

**PCIe protocol itself supports GPU-initiated P2P** (BAR1 mapping → GPU memory controller generates PCIe TLP packets autonomously). The limitation is an **NVIDIA driver lock on consumer SKUs**, not a PCIe hardware limitation.

### Why AsyncTP Is Impossible on Consumer GPUs

AsyncTP (vLLM's compute-communication overlap) relies on **kernel-level P2P writes** to peer GPU memory — which requires NVLink or unlocked PCIe P2P. On consumer GPUs:

1. P2P is driver-locked → cannot get peer BAR1 address
2. Fallback to SHM → requires CPU proxy for data forwarding
3. Fused custom kernel impossible → must use separate NCCL all_reduce

The best optimization is therefore reducing CPU dispatch overhead per NCCL call — exactly what PyNccl does.

---

## 6. Comparison: torch.distributed vs PyNccl

| Metric | torch.distributed | PyNccl | Improvement |
|--------|-------------------|--------|-------------|
| CPU time per call (eager) | ~103 us | ~12 us | **~8.6x** |
| GPU time per call | ~22 us | ~22 us | Same |
| Calls per decode step | 48 | 48 | Same |
| CPU time per step (eager) | ~4.9 ms | ~0.6 ms | **~4.3 ms saved** |
| CUDA Graph capture | ✅ (NCCL 2.9+) | ✅ (NCCL 2.9+) | Same |
| CG replay (P2P transport) | 0 CPU path | 0 CPU path | Same |
| CG replay (SHM transport) | Proxy CPU remains | Proxy CPU remains | Same |
| Dependencies | torch.distributed | libnccl.so (ctypes) | Fewer |
| Code complexity | Opaque C++ | ~40 lines Python | Simpler |

### Expected Benefit

In **eager mode** (or when `dist.all_reduce` CG capture fails on consumer GPUs):
- Decode step latency reduces by ~4.3 ms
- Throughput improves proportionally

In **CUDA graph mode with P2P transport** (NVLink/A100):
- Warmup/capture phase is faster (PyNccl adds less overhead during model warmup)
- Replay phase is identical (zero CPU path in both cases)
- **PyNccl has zero per-step benefit**

In **CUDA graph mode with SHM transport** (consumer GPU):
- Proxy thread CPU overhead persists in both cases
- PyNccl's eager-mode savings still apply

---

## Summary

PyNccl is a targeted optimization for tensor parallel inference on consumer GPUs:

1. **`record_param_comms`** (Kineto profiler hook) accounts for **84.6%** of `c10d::allreduce_` CPU time (~86 us/call). PyNccl bypasses it entirely.
2. **PyNccl** reduces CPU overhead from ~103 us to ~12 us per call (~91 us saved).
3. For **48 all_reduces per decode step**, the total saving is **~4.4 ms per step** in eager mode.
4. **On P2P-enabled systems** (NVLink/A100 PCIe): CG replay truly has zero CPU overhead — PyNccl has no per-step benefit.
5. **On consumer GPUs** (RTX with P2P locked): NCCL falls back to **SHM transport** with a CPU proxy thread. The proxy's polling and data forwarding overhead persists even in CG replay, making PyNccl's eager-mode savings applicable in **both** eager and CG modes.
6. PyNccl can also **fix CG capture failures** — `dist.all_reduce` sometimes fails to capture (CUDA event creation during capture), while direct `ncclAllReduce` avoids this.
