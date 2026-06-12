import ctypes
import os

import torch

_NCCL_COMM = None
# Global reference to loaded NCCL library, used by _check for error strings
_nccl_lib = None

# ncclResult_t constants
_NCCL_OK = 0
_nccl_sum = 0
_nccl_float = 0
_nccl_float16 = 1
_nccl_bfloat16 = 2
_nccl_int32 = 3


# ncclUniqueId struct (128-byte opaque buffer)
class NcclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_uint8 * 128)]


def _check(result: int, context: str = "") -> None:
    if result != _NCCL_OK:
        msg = f"NCCL error code: {result}"
        if _nccl_lib is not None:
            try:
                error_str = _nccl_lib.ncclGetErrorString(result)
                msg += f" ({error_str})"
            except Exception:
                pass
        if context:
            msg = f"{context}: {msg}"
        raise RuntimeError(msg)


def _find_libnccl() -> str:
    """Locate libnccl.so.

    Priority:
    1. PYNCCL_SO_PATH environment variable
    2. PyTorch's bundled nvidia NCCL package
    3. System paths (libnccl.so, libnccl.so.2)
    """
    so_path = os.environ.get("PYNCCL_SO_PATH")
    if so_path:
        return so_path
    # Check PyTorch's bundled nvidia NCCL package
    torch_dir = os.path.dirname(torch.__file__)
    nv_nccl = os.path.join(torch_dir, "..", "nvidia", "nccl", "lib", "libnccl.so.2")
    nv_nccl = os.path.normpath(nv_nccl)
    if os.path.exists(nv_nccl):
        return nv_nccl
    # Fallback to system paths
    for name in ["libnccl.so", "libnccl.so.2"]:
        try:
            ctypes.CDLL(name, mode=ctypes.RTLD_NOLOAD)
            return name
        except OSError:
            continue
    msg = (
        "libnccl.so not found. Install NCCL via: "
        "pip install nvidia-nccl-cu12"
    )
    raise OSError(msg)


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

    def __init__(self, world_size: int, rank: int, unique_id: bytes):
        if len(unique_id) != 128:
            raise ValueError(
                f"unique_id must be exactly 128 bytes, got {len(unique_id)}"
            )
        self.world_size = world_size
        self.rank = rank
        self._lib = ctypes.CDLL(_find_libnccl())
        global _nccl_lib
        _nccl_lib = self._lib
        self._init_nccl_types()
        self._setup_argtypes()
        # Parse unique_id bytes into ncclUniqueId struct
        uid = NcclUniqueId()
        ctypes.memmove(ctypes.byref(uid), unique_id, 128)
        comm = ctypes.c_void_p()
        _check(
            self._lib.ncclCommInitRank(
                ctypes.byref(comm),
                ctypes.c_int(world_size),
                uid,
                ctypes.c_int(rank),
            ),
            context="ncclCommInitRank",
        )
        self._comm = comm

    def _init_nccl_types(self) -> None:
        self._dtype_map: dict[torch.dtype, int] = {
            torch.float32: _nccl_float,
            torch.float16: _nccl_float16,
            torch.bfloat16: _nccl_bfloat16,
            torch.int32: _nccl_int32,
        }

    def _setup_argtypes(self) -> None:
        # ncclAllReduce: sendbuff, recvbuff, count, datatype, op, comm, stream
        self._lib.ncclAllReduce.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self._lib.ncclAllReduce.restype = ctypes.c_int
        # ncclGetErrorString: ncclResult_t → const char*
        self._lib.ncclGetErrorString.argtypes = [ctypes.c_int]
        self._lib.ncclGetErrorString.restype = ctypes.c_char_p
        # ncclCommDestroy: comm
        self._lib.ncclCommDestroy.argtypes = [ctypes.c_void_p]
        self._lib.ncclCommDestroy.restype = ctypes.c_int

    def all_reduce(self, tensor: torch.Tensor) -> None:
        if not tensor.is_cuda:
            raise ValueError(
                f"all_reduce requires a CUDA tensor, got {tensor.device}"
            )
        stream = torch.cuda.current_stream()
        stream_ptr = ctypes.c_void_p(stream.cuda_stream)
        nccl_dtype = self._dtype_map.get(tensor.dtype)
        if nccl_dtype is None:
            raise TypeError(
                f"Unsupported dtype {tensor.dtype} for all_reduce. "
                f"Supported dtypes: {list(self._dtype_map.keys())}"
            )
        _check(
            self._lib.ncclAllReduce(
                tensor.data_ptr(),
                tensor.data_ptr(),
                ctypes.c_size_t(tensor.numel()),
                ctypes.c_int(nccl_dtype),
                ctypes.c_int(_nccl_sum),
                self._comm,
                stream_ptr,
            ),
            context="ncclAllReduce",
        )

    def destroy(self) -> None:
        if hasattr(self, "_comm") and self._comm and hasattr(self, "_lib"):
            self._lib.ncclCommDestroy(self._comm)
            self._comm = None

    def __del__(self) -> None:
        try:
            self.destroy()
        except Exception:
            pass
