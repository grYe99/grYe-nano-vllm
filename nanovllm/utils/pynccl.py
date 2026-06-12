"""PyNccl communicator — raw NCCL all_reduce via ctypes.

Uses the system NCCL library (libnccl.so.2 → /usr/lib/x86_64-linux-gnu/
libnccl.so.2.25.1) which correctly establishes ring connections on this
platform.  The conda-bundled NCCL (nvidia-nccl-cu12) produces broken
communicators where ncclAllReduce corrupts tensor data.
"""

import ctypes
import functools
import os

import torch
from torch.distributed import ReduceOp

# === ctypes type aliases ===
ncclResult_t = ctypes.c_int
ncclComm_t = ctypes.c_void_p


class ncclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


# === NCCL data-type / reduce-op helpers (same numbering as nccl.h) ===

class ncclDataTypeEnum:
    ncclFloat16 = 6
    ncclFloat32 = 7
    ncclBfloat16 = 9

    _map: dict[torch.dtype, int] = {
        torch.float16: ncclFloat16,
        torch.float32: ncclFloat32,
        torch.bfloat16: ncclBfloat16,
    }

    @classmethod
    def from_torch(cls, dtype: torch.dtype) -> int:
        v = cls._map.get(dtype)
        if v is None:
            raise TypeError(f"Unsupported dtype {dtype} for NCCL all_reduce")
        return v


class ncclRedOpTypeEnum:
    ncclSum = 0

    @classmethod
    def from_torch(cls, op: ReduceOp) -> int:
        assert op == ReduceOp.SUM, "only SUM is supported"
        return cls.ncclSum


# === NCCL library loader (system lib, NOT conda-bundled) ===

_INTERNAL: dict[str, dict[str, ctypes.CDLL | dict]] = {}
"""Cache: so_file → {lib: CDLL, funcs: {name: ctypes function}}."""


def _find_system_nccl() -> str:
    """Return path to the system libnccl.so.2 (never the conda-bundled one)."""
    return "libnccl.so.2"  # resolves via ldconfig to /usr/lib/.../libnccl.so.2.25.1


def _nccl_lib() -> dict[str, object]:
    so = _find_system_nccl()
    if so not in _INTERNAL:
        lib = ctypes.CDLL(so)
        funcs: dict[str, object] = {}
        for name, restype, argtypes in [
            ("ncclCommInitRank", ncclResult_t,
             [ctypes.POINTER(ncclComm_t), ctypes.c_int, ncclUniqueId, ctypes.c_int]),
            ("ncclAllReduce", ncclResult_t,
             [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
              ctypes.c_int, ctypes.c_int, ncclComm_t, ctypes.c_void_p]),
            ("ncclCommDestroy", ncclResult_t, [ncclComm_t]),
            ("ncclGetUniqueId", ncclResult_t, [ctypes.POINTER(ncclUniqueId)]),
            ("ncclCommCount", ncclResult_t, [ncclComm_t, ctypes.POINTER(ctypes.c_int)]),
        ]:
            f = getattr(lib, name)
            f.restype = restype
            f.argtypes = argtypes
            funcs[name] = f
        _INTERNAL[so] = {"lib": lib, "funcs": funcs}
    return _INTERNAL[so]


# === public API ===

_NCCL_COMM: "NcclCommunicator | None" = None


def init_communicator(world_size: int, rank: int, unique_id: bytes) -> None:
    global _NCCL_COMM
    _NCCL_COMM = NcclCommunicator(world_size, rank, unique_id)


def destroy_communicator() -> None:
    global _NCCL_COMM
    if _NCCL_COMM is not None:
        _NCCL_COMM.destroy()
        _NCCL_COMM = None


def get_communicator() -> "NcclCommunicator":
    assert _NCCL_COMM is not None, "NcclCommunicator not initialized"
    return _NCCL_COMM


def is_initialized() -> bool:
    return _NCCL_COMM is not None


class NcclCommunicator:

    def __init__(self, world_size: int, rank: int, unique_id: bytes) -> None:
        self.world_size = world_size
        self.rank = rank
        self._funcs = _nccl_lib()["funcs"]

        uid = ncclUniqueId()
        ctypes.memmove(ctypes.addressof(uid.internal), unique_id, 128)

        with torch.accelerator.device_index(rank):
            comm = ncclComm_t()
            self._funcs["ncclCommInitRank"](
                ctypes.byref(comm), world_size, uid, rank,
            )
            self.comm = comm

            # Warmup: small all_reduce triggers ring setup
            data = torch.zeros(1, device=f"cuda:{rank}")
            self.all_reduce(data)
            torch.cuda.synchronize()
            del data

    def all_reduce(self, tensor: torch.Tensor) -> None:
        assert tensor.is_cuda, f"all_reduce requires CUDA tensor, got {tensor.device}"
        stream = torch.cuda.current_stream()
        self._funcs["ncclAllReduce"](
            tensor.data_ptr(),
            tensor.data_ptr(),
            tensor.numel(),
            ncclDataTypeEnum.from_torch(tensor.dtype),
            ncclRedOpTypeEnum.from_torch(ReduceOp.SUM),
            self.comm,
            stream.cuda_stream,
        )

    def destroy(self) -> None:
        if hasattr(self, "comm") and self.comm is not None:
            try:
                self._funcs["ncclCommDestroy"](self.comm)
            except Exception:
                pass
            self.comm = None

    def __del__(self) -> None:
        self.destroy()
