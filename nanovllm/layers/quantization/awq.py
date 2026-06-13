"""AWQ 4-bit quantized linear layers with tensor parallelism support."""

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.layers.quantization.awq_triton import awq_dequantize

GROUP_SIZE = 128
PACK_FACTOR = 8  # 32 bits / 4 bits


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


def _attach_weight_loader(param, loader_fn):
    """Attach a weight_loader function to a Parameter."""
    param.weight_loader = loader_fn


class _AWQBase(nn.Module):
    """Mixin-like base with helpers for AWQ weight loaders."""

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

    Parameters loaded from checkpoint (TP-sharded during loading):
        qweight: int32 [out_features // tp_size, in_features // 8]
        qzeros:  int32 [num_groups, out_features // tp_size // 8]
        scales:  fp16  [num_groups, out_features // tp_size]
    """

    def __init__(self, input_size: int, output_size: int, bias: bool = False,
                 group_size: int = GROUP_SIZE):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.group_size = group_size
        self.num_groups = input_size // group_size
        output_size_per_rank = divide(output_size, self.tp_size)

        self.qweight = nn.Parameter(
            torch.empty(output_size_per_rank, input_size // PACK_FACTOR, dtype=torch.int32),
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
            # loaded_weight shape: [out_features, in_features // 8]
            # shard on dim 0 (output dim)
            self._shard_on_dim(param.data, loaded_weight, 0, self.tp_rank, self.tp_size)
        elif param is self.qzeros:
            # loaded_weight shape: [num_groups, out_features // 8]
            # shard on dim 1 (packed output dim)
            self._shard_on_dim(param.data, loaded_weight, 1, self.tp_rank, self.tp_size)
        elif param is self.scales:
            # loaded_weight shape: [num_groups, out_features]
            # shard on dim 1 (output dim)
            self._shard_on_dim(param.data, loaded_weight, 1, self.tp_rank, self.tp_size)
        elif param is self.bias:
            self._shard_on_dim(param.data, loaded_weight, 0, self.tp_rank, self.tp_size)
        else:
            raise ValueError(f"Unknown parameter: {param.shape}")

    def _dequantize_weight(self) -> torch.Tensor:
        return awq_dequantize(self.qweight, self.scales, self.qzeros, self.group_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._dequantize_weight()
        return F.linear(x, weight, self.bias)


class AWQMergedColumnParallelLinear(AWQColumnParallelLinear):
    """Merged column-parallel AWQ linear (e.g. gate_proj + up_proj).

    output_sizes: list of output dimensions for each sub-module.
    weight_loader uses loaded_shard_id (int) to place each sub-module slice.
    """

    def __init__(self, input_size: int, output_sizes: list[int],
                 bias: bool = False, group_size: int = GROUP_SIZE):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias, group_size)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor,
                      loaded_shard_id: int):
        """loaded_shard_id: index into output_sizes (0 for gate, 1 for up)."""
        if param is self.qweight:
            # Sub-module portion in the packed output rows
            offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
            size = self.output_sizes[loaded_shard_id] // self.tp_size
            shard = loaded_weight.chunk(self.tp_size, 0)[self.tp_rank]
            target = param.data.narrow(0, offset, size)
            target.copy_(shard)
        elif param is self.qzeros:
            offset = sum(self.output_sizes[:loaded_shard_id]) // PACK_FACTOR // self.tp_size
            size = self.output_sizes[loaded_shard_id] // PACK_FACTOR // self.tp_size
            shard = loaded_weight.chunk(self.tp_size, 1)[self.tp_rank]
            target = param.data.narrow(1, offset, size)
            target.copy_(shard)
        elif param is self.scales:
            offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
            size = self.output_sizes[loaded_shard_id] // self.tp_size
            shard = loaded_weight.chunk(self.tp_size, 1)[self.tp_rank]
            target = param.data.narrow(1, offset, size)
            target.copy_(shard)
        elif param is self.bias:
            offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
            size = self.output_sizes[loaded_shard_id] // self.tp_size
            shard = loaded_weight.chunk(self.tp_size, 0)[self.tp_rank]
            target = param.data.narrow(0, offset, size)
            target.copy_(shard)
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
        """loaded_shard_id: "q", "k", or "v"."""
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
            loaded_shard = loaded_weight.chunk(self.tp_size, 0)[self.tp_rank]
            param.data.narrow(0, offset, shard_size).copy_(loaded_shard)
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

    Parameters (TP-sharded along input dimension):
        qweight: int32 [out_features, in_features // tp_size // 8]
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

        self.qweight = nn.Parameter(
            torch.empty(output_size, input_size_per_rank // PACK_FACTOR, dtype=torch.int32),
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
            # loaded_weight: [out_features, in_features // 8]
            # shard on dim 1 (packed input dim)
            self._shard_on_dim(param.data, loaded_weight, 1, self.tp_rank, self.tp_size)
        elif param is self.qzeros:
            # loaded_weight: [num_groups, out_features // 8]
            # shard on dim 0 (groups)
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
        return awq_dequantize(self.qweight, self.scales, self.qzeros, self.group_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._dequantize_weight()
        y = F.linear(x, weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
