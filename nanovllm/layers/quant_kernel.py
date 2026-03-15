import torch
import triton
import triton.language as tl


@triton.jit
def _w4a16_gemm_kernel(
    x_ptr, w_ptr, s_ptr, z_ptr, y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_sg, stride_sn,
    stride_ym, stride_yn,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    DTYPE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k_lo = k + tl.arange(0, BLOCK_K // 2)
        offs_k_hi = k + (BLOCK_K // 2) + tl.arange(0, BLOCK_K // 2)

        x_lo = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k_lo[None, :] * stride_xk,
            mask=(offs_m[:, None] < M) & (offs_k_lo[None, :] < K),
            other=0.0,
        )
        x_hi = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k_hi[None, :] * stride_xk,
            mask=(offs_m[:, None] < M) & (offs_k_hi[None, :] < K),
            other=0.0,
        )

        # packed row index: each packed row stores one lo+hi pair from the same k-group
        offs_k_packed = (k // 2) + tl.arange(0, BLOCK_K // 2)
        w_packed = tl.load(
            w_ptr + offs_k_packed[:, None] * stride_wk + offs_n[None, :] * stride_wn,
            mask=(offs_k_packed[:, None] < K // 2) & (offs_n[None, :] < N),
            other=0,
        )

        w_lo = (w_packed & 0xF).to(DTYPE)
        w_hi = ((w_packed >> 4) & 0xF).to(DTYPE)

        group_idx = k // GROUP_SIZE
        s = tl.load(
            s_ptr + group_idx * stride_sg + offs_n * stride_sn,
            mask=offs_n < N, other=1.0,
        ).to(DTYPE)
        z = tl.load(
            z_ptr + group_idx * stride_sg + offs_n * stride_sn,
            mask=offs_n < N, other=0.0,
        ).to(DTYPE)

        w_lo_dq = (w_lo - z[None, :]) * s[None, :]
        w_hi_dq = (w_hi - z[None, :]) * s[None, :]

        acc = tl.dot(x_lo, w_lo_dq, acc=acc, out_dtype=tl.float32)
        acc = tl.dot(x_hi, w_hi_dq, acc=acc, out_dtype=tl.float32)

    y = acc.to(DTYPE)
    tl.store(
        y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
        y,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def w4a16_gemm(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    assert x.dtype in (torch.float16, torch.bfloat16), f"x must be float16 or bfloat16, got {x.dtype}"
    assert x.is_contiguous() and w_packed.is_contiguous()
    assert scales.is_contiguous() and zeros.is_contiguous()
    assert group_size == 128, f"Only group_size=128 supported (BLOCK_K hardcoded to group_size)"
    M, K = x.shape
    _, N = w_packed.shape
    assert K == w_packed.shape[0] * 2
    assert K % group_size == 0
    assert scales.shape == (K // group_size, N), f"scales shape should be [{K//group_size}, {N}], got {scales.shape}"
    assert zeros.shape == (K // group_size, N), f"zeros shape should be [{K//group_size}, {N}], got {zeros.shape}"

    # scales/zeros must match x dtype for tl.dot type consistency
    scales = scales.to(x.dtype)
    zeros = zeros.to(x.dtype)

    TRITON_DTYPE = tl.float16 if x.dtype == torch.float16 else tl.bfloat16
    y = torch.empty(M, N, dtype=x.dtype, device=x.device)

    BLOCK_M = 16
    BLOCK_N = 64
    BLOCK_K = group_size

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _w4a16_gemm_kernel[grid](
        x, w_packed, scales, zeros, y,
        M, N, K,
        x.stride(0), x.stride(1),
        w_packed.stride(0), w_packed.stride(1),
        scales.stride(0), scales.stride(1),
        y.stride(0), y.stride(1),
        GROUP_SIZE=group_size,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        DTYPE=TRITON_DTYPE,
    )
    return y


def pack_weight_to_int4(
    weight: torch.Tensor,
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    N, K = weight.shape
    assert K % group_size == 0

    w = weight.T.contiguous().float()               # [K, N]
    num_groups = K // group_size
    w_3d = w.reshape(num_groups, group_size, N)     # [G, 128, N]

    w_min = w_3d.amin(dim=1)                        # [G, N]
    w_max = w_3d.amax(dim=1)                        # [G, N]
    scales = ((w_max - w_min) / 15.0).clamp(min=1e-8)
    zeros = -w_min / scales

    w_quant = ((w_3d - w_min[:, None, :]) / scales[:, None, :]).round().clamp(0, 15).to(torch.uint8)

    # Half-split packing: within each group of `group_size` K elements,
    # the first half (rows 0..63) is packed into the lo nibble (bits 0-3),
    # and the second half (rows 64..127) into the hi nibble (bits 4-7).
    # This allows the kernel to load two contiguous x sub-tiles that map
    # directly to w_lo and w_hi without scatter/gather.
    half = group_size // 2
    w_lo = w_quant[:, :half, :]
    w_hi = w_quant[:, half:, :]
    w_packed = (w_lo | (w_hi << 4)).reshape(K // 2, N)

    return w_packed, scales.to(torch.float16), zeros.to(torch.float16)
