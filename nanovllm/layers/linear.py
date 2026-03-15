import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from nanovllm.layers.quant_kernel import w4a16_gemm, pack_weight_to_int4


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y


class W4A16Linear(nn.Module):
    """
    Weight-only INT4 quantized linear (single GPU / replicated).

    Weights are stored as packed int4 (half-split format, group_size=128).
    Activations remain FP16. Forward uses Triton fused dequant+GEMM kernel.
    """

    def __init__(self, input_size: int, output_size: int, group_size: int = 128):
        super().__init__()
        assert input_size % group_size == 0, \
            f"input_size={input_size} must be divisible by group_size={group_size}"
        self.input_size = input_size
        self.output_size = output_size
        self.group_size = group_size
        num_groups = input_size // group_size

        self.register_buffer("weight_packed",
                             torch.empty(input_size // 2, output_size, dtype=torch.uint8))
        self.register_buffer("scales",
                             torch.empty(num_groups, output_size, dtype=torch.float16))
        self.register_buffer("zeros",
                             torch.empty(num_groups, output_size, dtype=torch.float16))

    def pack_weights(self, weight: torch.Tensor):
        """Quantize and pack FP16 weight [output_size, input_size]."""
        w_packed, scales, zeros = pack_weight_to_int4(weight, self.group_size)
        self.weight_packed.copy_(w_packed)
        self.scales.copy_(scales)
        self.zeros.copy_(zeros)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.input_size)
        if not x_2d.is_contiguous():
            x_2d = x_2d.contiguous()
        y = w4a16_gemm(x_2d, self.weight_packed, self.scales, self.zeros, self.group_size)
        return y.view(*orig_shape[:-1], self.output_size)


class W4A16ColumnParallelLinear(W4A16Linear):
    """Column-parallel W4A16 linear (output dimension is already TP-sharded).
    No all_reduce needed: output is a TP shard, consumer handles aggregation."""
    pass


class W4A16RowParallelLinear(W4A16Linear):
    """Row-parallel W4A16 linear (input dimension is TP-sharded, output needs all_reduce)."""

    def __init__(self, input_size: int, output_size: int, group_size: int = 128):
        super().__init__(input_size, output_size, group_size)
        self.tp_size = dist.get_world_size() if dist.is_initialized() else 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y


def quantize_model(model: nn.Module, group_size: int = 128) -> None:
    """
    In-place replace ColumnParallelLinear/RowParallelLinear with W4A16 variants.

    Must be called after FP16 weights are loaded. Each rank quantizes its own
    TP shard independently (sharding is already done by weight_loader).

    QKVParallelLinear and MergedColumnParallelLinear are intentionally skipped
    (complex multi-shard weight_loader logic; not yet supported).
    """
    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, (QKVParallelLinear, MergedColumnParallelLinear)):
            continue
        if isinstance(module, ColumnParallelLinear):
            replacements.append((name, module, "column"))
        elif isinstance(module, RowParallelLinear):
            replacements.append((name, module, "row"))

    for name, module, kind in replacements:
        out_f, in_f = module.weight.shape  # LinearBase stores [out, in]
        if kind == "column":
            new_layer = W4A16ColumnParallelLinear(in_f, out_f, group_size)
        else:
            new_layer = W4A16RowParallelLinear(in_f, out_f, group_size)
        new_layer.to(module.weight.device)
        new_layer.pack_weights(module.weight.data)

        # Navigate to parent and set attribute
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            parent = model.get_submodule(parts[0])
            setattr(parent, parts[1], new_layer)
        else:
            setattr(model, name, new_layer)
