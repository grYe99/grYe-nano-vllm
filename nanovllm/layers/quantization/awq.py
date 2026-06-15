"""AWQ 4-bit quantized linear layers with tensor parallelism support."""

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.layers.quantization.awq_triton import awq_dequantize, awq_gemm_triton

GROUP_SIZE = 128
PACK_FACTOR = 8  # 32 bits / 4 bits

# AWQ pack order: reverse order maps [0,4,1,5,2,6,3,7] → [0,1,2,3,4,5,6,7]
AWQ_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


def _attach_weight_loader(param, loader_fn):
    """Attach a weight_loader function to a Parameter."""
    param.weight_loader = loader_fn


class _AWQBase(nn.Module):
    """Mixin-like base with helpers for AWQ weight loaders."""

    def __init__(self):
        super().__init__()
        # Pre-compute AWQ order tensor as a buffer so it's not created during
        # CUDA Graph capture (torch.tensor(..., device='cuda') is not
        # permitted when a stream is capturing).
        self.register_buffer(
            "_awq_order",
            torch.tensor(AWQ_ORDER, dtype=torch.int32),
            persistent=False,
        )

    @staticmethod
    def _shard_on_dim(param_data, loaded_weight, dim, tp_rank, tp_size):
        """Slice loaded_weight along dim by tp_rank and copy into param_data."""
        shard_size = loaded_weight.size(dim) // tp_size
        start = tp_rank * shard_size
        loaded_shard = loaded_weight.narrow(dim, start, shard_size)
        param_data.copy_(loaded_shard)

    @staticmethod
    def _shard_on_dim_inplace(param_data, loaded_weight, dim, tp_rank, tp_size, offset, size):
        """Like _shard_on_dim but writes into a slice of param_data."""
        shard = loaded_weight.chunk(tp_size, dim)[tp_rank]
        target = param_data.narrow(dim, offset, size)
        target.copy_(shard)


class AWQColumnParallelLinear(_AWQBase, nn.Module):
    """Column-parallel AWQ linear: splits output dim across TP ranks.

    AWQ qweight convention: qweight[in_features, out_features // 8]
    Packed along the output dimension — 8 consecutive int4 values
    along the output dim are packed into one int32.

    Parameters (TP-sharded along the output dimension):
        qweight: int32 [in_features, out_features_per_rank // 8]
        qzeros:  int32 [num_groups, out_features_per_rank // 8]
        scales:  fp16  [num_groups, out_features_per_rank]
    """

    def __init__(self, input_size: int, output_size: int, bias: bool = False,
                 group_size: int = GROUP_SIZE):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.group_size = group_size
        self.num_groups = input_size // group_size
        output_size_per_rank = divide(output_size, self.tp_size)

        # qweight: [in_features, out_features_per_rank // 8]
        self.qweight = nn.Parameter(
            torch.empty(input_size, output_size_per_rank // PACK_FACTOR, dtype=torch.int32),
            requires_grad=False)
        self.qzeros = nn.Parameter(
            torch.empty(self.num_groups, output_size_per_rank // PACK_FACTOR, dtype=torch.int32),
            requires_grad=False)
        self.scales = nn.Parameter(
            torch.empty(self.num_groups, output_size_per_rank, dtype=torch.float16),
            requires_grad=False)

        _attach_weight_loader(self.qweight, self.weight_loader)
        _attach_weight_loader(self.qzeros, self.weight_loader)
        _attach_weight_loader(self.scales, self.weight_loader)

        if bias:
            self.bias = nn.Parameter(torch.empty(output_size_per_rank), requires_grad=False)
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor,
                      loaded_shard_id: int | str | None = None):
        if param is self.qweight:
            # loaded_weight: [in_features, out_features // 8]
            # shard on dim 1 (packed output dim)
            self._shard_on_dim(param.data, loaded_weight, 1, self.tp_rank, self.tp_size)
        elif param is self.qzeros:
            # loaded_weight: [num_groups, out_features // 8]
            # shard on dim 1 (packed output dim)
            self._shard_on_dim(param.data, loaded_weight, 1, self.tp_rank, self.tp_size)
        elif param is self.scales:
            # loaded_weight: [num_groups, out_features]
            # shard on dim 1 (output dim)
            self._shard_on_dim(param.data, loaded_weight, 1, self.tp_rank, self.tp_size)
        elif param is self.bias:
            self._shard_on_dim(param.data, loaded_weight, 0, self.tp_rank, self.tp_size)
        else:
            raise ValueError(f"Unknown parameter: {param.shape}")

    def _dequantize_weight(self) -> torch.Tensor:
        # Returns: [in_features, out_features_per_rank]
        return awq_dequantize(self.qweight, self.scales, self.qzeros,
                              self.group_size, awq_order=self._awq_order)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [M, in_features]
        M = x.size(-2)
        if M <= 16:
            # Small decode: CUDA Tensor Core custom op (M <= 16 constraint)
            orig_dtype = x.dtype
            if x.dtype != torch.float16:
                x = x.to(torch.float16)
            y = torch.ops.nanovllm.awq_gemm(
                x, self.qweight, self.scales, self.qzeros, 8)
            if orig_dtype != torch.float16:
                y = y.to(orig_dtype)
            return y
        elif M < 256:
            # Medium batch (decode phase): fused gemm avoids materializing
            # the full weight matrix, saving bandwidth.
            return awq_gemm_triton(x, self.qweight, self.scales, self.qzeros,
                                   self.group_size, split_k_iters=8)
        # Large batch (prefill phase): cuBLAS GEMM is efficient enough that
        # dequant overhead is well amortized.
        weight = self._dequantize_weight()
        weight = weight.t().to(x.dtype)
        return F.linear(x, weight, self.bias)


class AWQMergedColumnParallelLinear(AWQColumnParallelLinear):
    """Merged column-parallel AWQ linear (e.g. gate_proj + up_proj).

    Checkpoint has separate qweight/qzeros/scales for each sub-module.
    They are concatenated along the **output** dimension (dim 1 of qweight).
    weight_loader uses loaded_shard_id (int) to place each sub-module slice.
    """

    def __init__(self, input_size: int, output_sizes: list[int],
                 bias: bool = False, group_size: int = GROUP_SIZE):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias, group_size)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor,
                      loaded_shard_id: int):
        """loaded_shard_id: index into output_sizes (0 for gate, 1 for up)."""
        # Number of output elements this sub-module occupies per TP rank
        out_size_per_rank = self.output_sizes[loaded_shard_id] // self.tp_size
        out_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size

        if param is self.qweight:
            offset = out_offset // PACK_FACTOR
            size = out_size_per_rank // PACK_FACTOR
            # loaded_weight: [in_features, out_size // 8]
            shard = loaded_weight.chunk(self.tp_size, 1)[self.tp_rank]
            param.data.narrow(1, offset, size).copy_(shard)
        elif param is self.qzeros:
            offset = out_offset // PACK_FACTOR
            size = out_size_per_rank // PACK_FACTOR
            shard = loaded_weight.chunk(self.tp_size, 1)[self.tp_rank]
            param.data.narrow(1, offset, size).copy_(shard)
        elif param is self.scales:
            offset = out_offset
            size = out_size_per_rank
            shard = loaded_weight.chunk(self.tp_size, 1)[self.tp_rank]
            param.data.narrow(1, offset, size).copy_(shard)
        elif param is self.bias:
            offset = out_offset
            size = out_size_per_rank
            shard = loaded_weight.chunk(self.tp_size, 0)[self.tp_rank]
            param.data.narrow(0, offset, size).copy_(shard)
        else:
            raise ValueError(f"Unknown parameter: {param.shape}")


class AWQQKVParallelLinear(AWQColumnParallelLinear):
    """QKV column-parallel AWQ linear.

    Loads separate q/k/v slices into one concatenated output via shard_id str.
    """

    def __init__(self, hidden_size: int, head_size: int,
                 total_num_heads: int, total_num_kv_heads: int | None = None,
                 bias: bool = False, group_size: int = GROUP_SIZE):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * head_size
        super().__init__(hidden_size, output_size, bias, group_size)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor,
                      loaded_shard_id: str):
        """loaded_shard_id: "q", "k", or "v".

        For qweight: the packed output dimension is dim 1.
        Each head group occupies (num_heads * head_size // 8) packed int32s.
        """
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            offset = self.num_heads * self.head_size
        elif loaded_shard_id == "v":
            shard_size = self.num_kv_heads * self.head_size
            offset = (self.num_heads + self.num_kv_heads) * self.head_size
        else:
            raise ValueError(f"Unknown shard_id: {loaded_shard_id}")

        if param is self.qweight:
            # loaded_weight: [hidden_size, total_out // 8], shard on dim 1
            loaded_shard = loaded_weight.chunk(self.tp_size, 1)[self.tp_rank]
            param.data.narrow(1, offset // PACK_FACTOR, shard_size // PACK_FACTOR).copy_(loaded_shard)
        elif param is self.qzeros:
            loaded_shard = loaded_weight.chunk(self.tp_size, 1)[self.tp_rank]
            param.data.narrow(1, offset // PACK_FACTOR, shard_size // PACK_FACTOR).copy_(loaded_shard)
        elif param is self.scales:
            loaded_shard = loaded_weight.chunk(self.tp_size, 1)[self.tp_rank]
            param.data.narrow(1, offset, shard_size).copy_(loaded_shard)
        elif param is self.bias:
            loaded_shard = loaded_weight.chunk(self.tp_size, 0)[self.tp_rank]
            param.data.narrow(0, offset, shard_size).copy_(loaded_shard)
        else:
            raise ValueError(f"Unknown parameter: {param.shape}")


class AWQRowParallelLinear(_AWQBase, nn.Module):
    """Row-parallel AWQ linear: splits input dim across TP ranks.

    AWQ qweight convention: qweight[in_features, out_features // 8].
    Row-parallel splits the **input** dimension, so each rank gets
    a slice of dim 0 of qweight.

    Parameters (TP-sharded along input dimension):
        qweight: int32 [in_features_per_rank, out_features // 8]
        qzeros:  int32 [num_groups_per_rank, out_features // 8]
        scales:  fp16  [num_groups_per_rank, out_features]
    """

    def __init__(self, input_size: int, output_size: int, bias: bool = False,
                 group_size: int = GROUP_SIZE):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.group_size = group_size

        input_size_per_rank = divide(input_size, self.tp_size)
        num_groups_per_rank = input_size_per_rank // group_size

        # qweight: [in_features_per_rank, out_features // 8]
        self.qweight = nn.Parameter(
            torch.empty(input_size_per_rank, output_size // PACK_FACTOR, dtype=torch.int32),
            requires_grad=False)
        self.qzeros = nn.Parameter(
            torch.empty(num_groups_per_rank, output_size // PACK_FACTOR, dtype=torch.int32),
            requires_grad=False)
        self.scales = nn.Parameter(
            torch.empty(num_groups_per_rank, output_size, dtype=torch.float16),
            requires_grad=False)

        _attach_weight_loader(self.qweight, self.weight_loader)
        _attach_weight_loader(self.qzeros, self.weight_loader)
        _attach_weight_loader(self.scales, self.weight_loader)

        if bias:
            self.bias = nn.Parameter(torch.empty(output_size), requires_grad=False)
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor,
                      loaded_shard_id: int | str | None = None):
        if param is self.qweight:
            # loaded_weight: [in_features, out_features // 8]
            # shard on dim 0 (input dim)
            self._shard_on_dim(param.data, loaded_weight, 0, self.tp_rank, self.tp_size)
        elif param is self.qzeros:
            # loaded_weight: [num_groups, out_features // 8]
            # shard on dim 0 (groups = in_features // group_size)
            self._shard_on_dim(param.data, loaded_weight, 0, self.tp_rank, self.tp_size)
        elif param is self.scales:
            # loaded_weight: [num_groups, out_features]
            # shard on dim 0 (groups)
            self._shard_on_dim(param.data, loaded_weight, 0, self.tp_rank, self.tp_size)
        elif param is self.bias:
            param.data.copy_(loaded_weight)
        else:
            raise ValueError(f"Unknown parameter: {param.shape}")

    def _dequantize_weight(self) -> torch.Tensor:
        # Returns: [in_features_per_rank, out_features]
        return awq_dequantize(self.qweight, self.scales, self.qzeros,
                              self.group_size, awq_order=self._awq_order)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [M, in_features] — need to slice input for this rank
        x_shard = x.narrow(-1, self.tp_rank * (x.size(-1) // self.tp_size),
                           x.size(-1) // self.tp_size)
        M = x_shard.size(-2)

        if M <= 16:
            # Small decode: CUDA Tensor Core custom op (M <= 16 constraint)
            orig_dtype = x_shard.dtype
            if x_shard.dtype != torch.float16:
                x_shard = x_shard.to(torch.float16)
            y = torch.ops.nanovllm.awq_gemm(
                x_shard, self.qweight, self.scales, self.qzeros, 8)
            if orig_dtype != torch.float16:
                y = y.to(orig_dtype)
        elif M < 256:
            # Medium batch: fused gemm
            y = awq_gemm_triton(x_shard, self.qweight, self.scales, self.qzeros,
                                self.group_size, split_k_iters=8)
        else:
            # Large batch: dequant + cuBLAS matmul
            weight = self._dequantize_weight().t().to(x.dtype)
            y = F.linear(x_shard, weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
