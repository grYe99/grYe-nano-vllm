# PyNccl: Bypassing torch.distributed Overhead for Tensor Parallel AllReduce

## Background

### The Problem

In the nano-vllm project, tensor parallelism (TP) uses `torch.distributed.all_reduce` to synchronize gradients/hidden states across GPUs. Profiling on 2x RTX 4090 (PCIe Gen4) reveals a severe CPU overhead issue:

| Metric | torch.distributed | NCCL kernel (actual) | Ratio |
|--------|-------------------|---------------------|-------|
| `c10d::allreduce_` | 96.8us CPU avg | 7.95us GPU avg | **12.2x** |
| `record_param_comms` | 81.8us CPU avg | — | — |
| cuLaunchKernel | ~10us | — | — |

Each decode step executes 48 all_reduce calls (2 per layer x 24 layers). The CPU dispatch overhead totals **4.7ms/step**, while the actual GPU communication takes only **0.4ms/step**. The CPU is the bottleneck — GPUs idle waiting for the next kernel launch.

### Why PCIe Makes This Worse

Unlike NVLink, PCIe cannot perform GPU-initiated peer-to-peer transfers. Every NCCL all_reduce on PCIe must:
1. CPU launches NCCL kernel (cuLaunchKernel)
2. NCCL kernel copies data to peer via CPU-mediated DMA
3. NCCL kernel synchronizes with peer(s)

This inherently requires CPU involvement. NVLink-native custom all-reduce kernels (as used by vLLM) are not possible on PCIe.

### Why Not Fix torch.distributed?

The overhead comes from `torch.distributed`'s general-purpose design:
- Python GIL handling
- C++ dispatcher setup and teardown
- Profiler event recording (`record_param_comms`)
- NCCL communicator lookup
- Synchronization bookkeeping

These are useful for PyTorch's distributed training use case but unnecessary for TP inference where the communicator is static and calls are synchronous.

## Solution: PyNccl Communicator

### Architecture

```
Before (torch.distributed):
  Python: dist.all_reduce(tensor)
    → c10d ProcessGroupNCCL (C++)
      → record_param_comms (profiler)
        → ncclComm lookup
          → cuLaunchKernel(ncclAllReduce)
            → ncclDevKernel_AllReduce (GPU)

After (PyNccl):
  Python: pynccl.all_reduce(tensor)
    → ctypes ncclAllReduce (direct)
      → cuLaunchKernel(ncclAllReduce)
        → ncclDevKernel_AllReduce (GPU)
```

Savings per call: ~70-80us CPU time by eliminating the c10d wrapper and profiler recording.

### NCCL Unique ID Bootstrap

NCCL requires a unique ID (`ncclUniqueId`) shared across all ranks in a communicator. The bootstrap mechanism uses `torch.distributed.broadcast_object_list`:

1. **Rank 0** calls `ncclGetUniqueId()` via ctypes to generate a 128-byte unique ID
2. **Rank 0** wraps it in a list: `[bytes(uid)]`; other ranks prepare `[None]`
3. `dist.broadcast_object_list(unique_id, src=0)` distributes the ID over the already-initialized NCCL process group
4. **All ranks** create `NcclCommunicator(world_size, rank, unique_id[0])`

```python
# In ModelRunner.__init__(), after dist.init_process_group:
if rank == 0:
    libnccl = ctypes.CDLL("libnccl.so")
    uid = (ctypes.c_uint8 * 128)()
    libnccl.ncclGetUniqueId(uid)
    unique_id = [bytes(uid)]
else:
    unique_id = [None]
dist.broadcast_object_list(unique_id, src=0)
init_communicator(world_size, rank, unique_id[0])
```

This avoids the chicken-and-egg problem of needing shared memory (which is set up after model creation/warmup in the current code). `dist.broadcast_object_list` works over the existing NCCL process group without additional infrastructure.

### Components

```
nanovllm/utils/pynccl.py
  └── class NcclCommunicator
        ├── __init__(world_size, rank)
        │     ├── Load libnccl.so via ctypes
        │     ├── ncclGetUniqueId (rank 0) / receive ID (other ranks)
        │     └── ncclCommInitRank → create NCCL communicator
        │
        └── all_reduce(tensor)
              ├── Get current CUDA stream
              ├── ncclAllReduce(tensor.data_ptr, tensor.data_ptr,
              │     numel, dtype, ncclSum, comm, stream.cuda_stream)
              └── Return tensor (in-place)
```

### Stream Management for CUDA Graph

```python
def all_reduce(self, tensor: torch.Tensor) -> None:
    stream = torch.cuda.current_stream()
    stream_ptr = ctypes.c_void_p(stream.cuda_stream)
    self._lib.ncclAllReduce(
        tensor.data_ptr(),
        tensor.data_ptr(),
        tensor.numel(),
        self._dtype_map[tensor.dtype],
        self._nccl_sum,
        self._comm,
        stream_ptr
    )
```

- Uses `torch.cuda.current_stream().cuda_stream` to get the raw CUDA stream handle as a Python int
- During `torch.cuda.graph` capture, this stream is the capture stream
- During replay, the graph replays the captured NCCL kernel on the same stream
- NCCL 2.19+ is CUDA graph compatible; user has NCCL 2.27.7

### NCCL Type Mapping

```python
_dtype_map = {
    torch.float32: self._nccl_float,
    torch.float16: self._nccl_half,
    torch.bfloat16: self._nccl_bfloat16,
}
```

## Files Changed

| File | Change |
|------|--------|
| `nanovllm/utils/pynccl.py` | **New**: NcclCommunicator class + `init_communicator`/`get_communicator` singleton |
| `nanovllm/layers/linear.py` | RowParallelLinear: replace `dist.all_reduce` with `pynccl.get_communicator().all_reduce()` |
| `nanovllm/layers/embed_head.py` | VocabParallelEmbedding: same replacement |
| `nanovllm/engine/model_runner.py` | Call `pynccl.init_communicator()` after creating NCCL process group |

### Communicator Access Pattern

Module-level singleton, **not dependency injection**. This avoids threading a communicator reference through the entire module hierarchy (model → decoder layer → attention/MLP → linear).

```python
# nanovllm/utils/pynccl.py — singleton pattern
_NCCL_COMM: NcclCommunicator | None = None

def init_communicator(world_size: int, rank: int, unique_id: bytes) -> None:
    global _NCCL_COMM
    _NCCL_COMM = NcclCommunicator(world_size, rank, unique_id)

def get_communicator() -> NcclCommunicator:
    assert _NCCL_COMM is not None
    return _NCCL_COMM
```

**Initialization flow in ModelRunner**:
1. `dist.init_process_group("nccl", ...)` — keep for process group infrastructure
2. Bootstrap NCCL unique ID via existing shm IPC (as described above)
3. `pynccl.init_communicator(world_size, rank, unique_id)` — initialize PyNccl
4. Create model, load weights, etc.

**Usage in layers**:
```python
# RowParallelLinear.forward:
y = F.linear(x, self.weight, ...)
if self.tp_size > 1:
    pynccl.get_communicator().all_reduce(y)  # replaces dist.all_reduce(y)
return y
```

## CUDA Graph Integration

The existing `capture_cudagraph()` in `ModelRunner` captures `self.model(input_ids, positions)`. With PyNccl:

- **During warmup/capture**: `model.forward()` calls `pynccl.all_reduce()` → `ncclAllReduce` launches NCCL kernel on capture stream → CUDA graph captures it
- **During replay**: `graph.replay()` replays all captured kernels including NCCL → no CPU path executed

No changes to the CUDA graph capture code itself.

## Expected Improvement

| Mode | Before (CPU/step) | After (CPU/step) | Improvement |
|------|-------------------|-------------------|-------------|
| Eager, TP=2 | ~4.7ms (48×97us) | ~0.8ms (48×~17us) | **~4ms saved** |
| CUDA Graph | ~4.7ms (capture only) | ~0.8ms (capture only) | Faster init |
| CUDA Graph replay | 0us (no CPU path) | 0us (no CPU path) | Same |

In eager mode (the default when `torch.compile` rewrites the graph), this translates to a measurable latency reduction per decode step.

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| `libnccl.so` not found | Fallback to `torch.distributed.all_reduce` |
| CUDA graph incompatibility | NCCL 2.27.7 supports graphs; test with `NCCL_ALGO=Ring` |
| NCCL communicator creation fails | Already handled by existing `dist.init_process_group`; PyNccl is secondary |
| Different NCCL dtype mapping | Explicit mapping table verified at init |

## Learning Document

A separate document `docs/pynccl_learning.md` will cover:

1. TP inference and the all_reduce bottleneck
2. torch.distributed all_reduce call chain (Python → C++ → NCCL)
3. Why each layer adds overhead (GIL, profiling, dispatch)
4. PyNccl design: ctypes wrapper, stream management, graph compatibility
5. PCIe vs NVLink: why custom AR kernels don't work on PCIe
6. Quantitative comparison with profiler data from this project
