import torch
from torch import nn
import torch.distributed as dist
import triton
import triton.language as tl


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)


@triton.jit
def fused_add_rms_kernel(
    x_ptr, residual_ptr, weight_ptr, output_ptr, residual_out_ptr,
    stride_x_row, stride_residual_row, stride_output_row, stride_residual_out_row,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    x_offset = row * stride_x_row
    residual_offset = row * stride_residual_row
    output_offset = row * stride_output_row
    residual_out_offset = row * stride_residual_out_row

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x = tl.load(x_ptr + x_offset + cols, mask=mask).to(tl.float32)
    residual = tl.load(residual_ptr + residual_offset + cols, mask=mask).to(tl.float32)

    z = x + residual
    tl.store(residual_out_ptr + residual_out_offset + cols, z.to(x_ptr.dtype.element_ty), mask=mask)

    var = tl.sum(z * z, axis=0) / N
    rstd = tl.rsqrt(var + eps)
    z = z * rstd

    weight = tl.load(weight_ptr + cols, mask=mask).to(tl.float32)
    z = z * weight

    tl.store(output_ptr + output_offset + cols, z.to(x_ptr.dtype.element_ty), mask=mask)


def fused_add_rms_triton(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.is_contiguous() and residual.is_contiguous()
    assert x.dim() == 2 and residual.dim() == 2
    assert x.shape == residual.shape
    M, N = x.shape
    output = torch.empty_like(x)
    residual_out = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    fused_add_rms_kernel[grid](
        x, residual, weight, output, residual_out,
        x.stride(0), residual.stride(0), output.stride(0), residual_out.stride(0),
        N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output, residual_out


@triton.jit
def rms_forward_kernel(
    x_ptr, weight_ptr, output_ptr,
    stride_x_row, stride_output_row,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    x_offset = row * stride_x_row
    output_offset = row * stride_output_row

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x = tl.load(x_ptr + x_offset + cols, mask=mask).to(tl.float32)

    var = tl.sum(x * x, axis=0) / N
    rstd = tl.rsqrt(var + eps)
    x = x * rstd

    weight = tl.load(weight_ptr + cols, mask=mask).to(tl.float32)
    x = x * weight

    tl.store(output_ptr + output_offset + cols, x.to(x_ptr.dtype.element_ty), mask=mask)


def rms_forward_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    assert x.is_contiguous()
    assert x.dim() == 2
    M, N = x.shape
    output = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    rms_forward_kernel[grid](
        x, weight, output,
        x.stride(0), output.stride(0),
        N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


class FusedAllReduceRMSNorm(nn.Module):
    """Fused all_reduce + (residual add) + RMSNorm.

    In tensor-parallel inference, RowParallelLinear produces a local result
    that must be all-reduced across GPUs before the subsequent layernorm.
    This class fuses all_reduce with add_rms_forward (or plain rms_forward)
    to reduce global memory round-trips.

    - When `residual is None`: pure RMSNorm without all_reduce (used by the
      first decoder layer where input comes from embeddings).
    - When `residual is not None`: all_reduce(x), then compute
      z = x + residual, followed by RMSNorm(z). Used in all other cases.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            # First decoder layer: x is from embeddings, no all_reduce needed
            return rms_forward_triton(x, self.weight, self.eps), x
        # All-reduce x (in-place) before fused add + rmsnorm
        tp_size = dist.get_world_size() if dist.is_initialized() else 1
        if tp_size > 1:
            dist.all_reduce(x)
        return fused_add_rms_triton(x, residual, self.weight, self.eps)


class FusedAddRMSNorm(nn.Module):
    """Fused (residual add) + RMSNorm WITHOUT all_reduce.

    Used when RowParallelLinear already does the all_reduce (chunked async),
    so the layernorm only needs to add residual and apply RMSNorm.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            # First decoder layer: x is from embeddings, no residual add needed
            return rms_forward_triton(x, self.weight, self.eps), x
        # Direct fused add + rmsnorm, no all_reduce (already done by linear)
        return fused_add_rms_triton(x, residual, self.weight, self.eps)


def create_rmsnorm(hidden_size: int, eps: float = 1e-6) -> "RMSNorm | FusedAllReduceRMSNorm | FusedAddRMSNorm":
    """Factory: dispatches based on ar_fused_norm and ar_async_chunked flags."""
    from nanovllm.utils.ar_mode import get_ar_fused_norm, get_ar_async_chunked

    if not get_ar_fused_norm():
        return RMSNorm(hidden_size, eps)
    if get_ar_async_chunked():
        # Linear already does chunked async AR; norm only needs add+rmsnorm
        return FusedAddRMSNorm(hidden_size, eps)
    return FusedAllReduceRMSNorm(hidden_size, eps)
