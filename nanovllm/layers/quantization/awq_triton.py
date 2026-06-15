"""
AWQ quantization kernels: dequantize and fused gemm.

AWQ packs 4-bit weights into int32 with a non-standard order:
  within each int32 [b0 b1 ... b31], the 8 int4 values are stored as
  indices [0, 4, 1, 5, 2, 6, 3, 7].

AWQ packs along the **output** dimension — 8 consecutive int4 values
along the output dim are packed into one int32.

Parameter shapes (K=in_features, M=out_features, group_size=128):
    qweight: int32 [K, M // 8]         — packed 4-bit weights
    scales:  fp16  [K // 128, M]       — per-group per-row scale
    qzeros:  int32 [K // 128, M // 8]  — packed 4-bit zero points

Output: fp16[K, M] — dequantized weight in (in_features, out_features) layout.
This is the transpose of what F.linear expects, so the caller must .t() before use.
"""
import torch
import triton
import triton.language as tl

# AWQ pack order: reverse order maps [0,4,1,5,2,6,3,7] → [0,1,2,3,4,5,6,7]
# shift amounts for each slot: [0, 16, 4, 20, 8, 24, 12, 28]
AWQ_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]
AWQ_SHIFTS = [s * 4 for s in AWQ_ORDER]  # [0, 16, 4, 20, 8, 24, 12, 28]


def awq_dequantize(qweight: torch.Tensor, scales: torch.Tensor,
                   qzeros: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    """Pure-PyTorch AWQ dequantize — reference implementation.

    qweight: int32[K, M//8] (K=in_features, M=out_features)
    scales:  fp16[num_groups, M]
    qzeros:  int32[num_groups, M//8]

    Returns fp16 weight of shape [K, M] = [in_features, out_features].
    """
    K, M8 = qweight.shape
    M = M8 * 8
    num_groups = K // group_size
    G = num_groups

    order = torch.tensor(AWQ_ORDER, device=qweight.device)

    # Unpack qweight: [K, M/8] with each int32 containing 8 int4 values
    # Expand: [K, M/8] → [K, M/8, 8], then shift and mask, then reshape
    w = (qweight.unsqueeze(-1) >> (order * 4)) & 0xF         # [K, M/8, 8]
    w = w.to(scales.dtype).reshape(K, M)                      # [K, M]

    # Unpack qzeros: [G, M/8] → [G, M/8, 8] → [G, M]
    z = (qzeros.unsqueeze(-1) >> (order * 4)) & 0xF           # [G, M/8, 8]
    z = z.to(scales.dtype).reshape(G, M)                      # [G, M]

    # Reshape for group-wise dequantize
    # w: [K, M] → [G, group_size, M]
    w = w.reshape(G, group_size, M)                            # [G, GS, M]
    z = z.unsqueeze(1)                                         # [G, 1,  M]
    s = scales.unsqueeze(1)                                    # [G, 1,  M]

    w = (w - z) * s                                            # [G, GS, M]
    w = w.reshape(K, M)                                        # [K, M]
    return w


@triton.jit
def _awq_dequantize_kernel(
    qweight_ptr, scales_ptr, qzeros_ptr, output_ptr,
    K, M, num_groups, group_size,
    stride_qw_k, stride_qw_m8,
    stride_s_g, stride_s_m,
    stride_qz_g, stride_qz_m8,
    stride_out_k, stride_out_m,
    BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr,
):
    """Triton kernel: dequantize one tile [BLOCK_K, BLOCK_M] of the weight matrix.

    Each program handles a (K_tile, M_tile) of the output [K, M].
    """
    pid_k = tl.program_id(0)
    pid_m = tl.program_id(1)

    off_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    k_mask = off_k < K
    m_mask = off_m < M

    # Which group each K element belongs to
    group_id = off_k // group_size                             # [BLOCK_K]
    k_in_group = off_k % group_size                            # [BLOCK_K]

    # The packed column in qweight for this K element:
    # qweight column = (group_size // 8) * group_id + k_in_group // 8
    qw_m8_off = (group_size // 8) * group_id + k_in_group // 8  # [BLOCK_K]

    # Load qweight tile: access qweight[off_k, qw_m8_off]
    # qweight layout: [K, M//8]
    qw_ptrs = qweight_ptr + off_k[:, None] * stride_qw_k + qw_m8_off[:, None] * stride_qw_m8
    qw = tl.load(qw_ptrs, mask=k_mask[:, None] & m_mask[None, :], other=0)

    # Unpack: the 8 int4 slots within each int32 are at AWQ order
    # slot = k_in_group % 8, shift = AWQ_SHIFTS[slot]
    shift = tl.where(k_in_group % 8 == 0, 0,
             tl.where(k_in_group % 8 == 1, 16,
             tl.where(k_in_group % 8 == 2, 4,
             tl.where(k_in_group % 8 == 3, 20,
             tl.where(k_in_group % 8 == 4, 8,
             tl.where(k_in_group % 8 == 5, 24,
             tl.where(k_in_group % 8 == 6, 12, 28)))))))
    w_int4 = (qw >> shift[:, None]) & 0xF                     # [BLOCK_K, BLOCK_M]

    # Load scales: scales[group_id, off_m]
    s_ptrs = scales_ptr + group_id[:, None] * stride_s_g + off_m[None, :] * stride_s_m
    s = tl.load(s_ptrs, mask=k_mask[:, None] & m_mask[None, :])

    # Load qzeros: qzeros[group_id, off_m // 8]
    z_m8 = off_m // 8                                          # [BLOCK_M]
    z_ptrs = qzeros_ptr + group_id[:, None] * stride_qz_g + z_m8[None, :] * stride_qz_m8
    z_packed = tl.load(z_ptrs, mask=k_mask[:, None] & m_mask[None, :], other=0)

    # Unpack zeros: slot = off_m % 8
    z_shift = tl.where(off_m % 8 == 0, 0,
               tl.where(off_m % 8 == 1, 16,
               tl.where(off_m % 8 == 2, 4,
               tl.where(off_m % 8 == 3, 20,
               tl.where(off_m % 8 == 4, 8,
               tl.where(off_m % 8 == 5, 24,
               tl.where(off_m % 8 == 6, 12, 28)))))))
    z_val = (z_packed >> z_shift[None, :]) & 0xF               # [BLOCK_K, BLOCK_M]

    # Dequantize
    w_fp16 = (w_int4.to(tl.float16) - z_val.to(tl.float16)) * s

    # Store output [K, M]
    out_ptrs = output_ptr + off_k[:, None] * stride_out_k + off_m[None, :] * stride_out_m
    tl.store(out_ptrs, w_fp16, mask=k_mask[:, None] & m_mask[None, :])


def triton_awq_dequantize(qweight, scales, qzeros, group_size=128):
    """Triton-accelerated AWQ dequantize.

    qweight: int32[K, M//8] → output fp16[K, M]
    Same interface as awq_dequantize().
    """
    K, M8 = qweight.shape
    M = M8 * 8
    num_groups = K // group_size
    output = torch.empty(K, M, dtype=scales.dtype, device=qweight.device)

    BLOCK_K = 64
    BLOCK_M = 128
    grid = (triton.cdiv(K, BLOCK_K), triton.cdiv(M, BLOCK_M))

    _awq_dequantize_kernel[grid](
        qweight, scales, qzeros, output,
        K, M, num_groups, group_size,
        qweight.stride(0), qweight.stride(1),
        scales.stride(0), scales.stride(1),
        qzeros.stride(0), qzeros.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_K=BLOCK_K, BLOCK_M=BLOCK_M,
    )
    return output


# ---------------------------------------------------------------------------
# Fused AWQ GEMM: activation @ (dequantize(qweight, scales, qzeros))
# Ported from vllm: uses tl.interleave for packed-dim expansion and
# interleaved split-K for better load balancing.
# ---------------------------------------------------------------------------

@triton.jit
def _awq_gemm_kernel(
    a_ptr, b_ptr, scales_ptr, zeros_ptr, c_ptr,
    M, N, K, group_size,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, SPLIT_K: tl.constexpr,
):
    """Fused AWQ GEMM Triton kernel.

    C[M, N] = A[M, K] @ dequantize(B[K, N//8], scales, zeros)
    with split-K parallelism (SPLIT_K partitions along K).

    C is stored as a 3D partial-sum buffer (SPLIT_K, M, N).
    """
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)

    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    accumulator_dtype = c_ptr.type.element_ty

    # Precompute AWQ reverse-order shifts for unpacking.
    # AWQ stores 8 int4 values per int32 in slot order [0, 4, 1, 5, 2, 6, 3, 7].
    # Shift amounts = slot * 4.
    reverse_awq_order = ((tl.arange(0, 2) * 4)[None, :] + tl.arange(0, 4)[:, None]).reshape(8)
    shifts = reverse_awq_order * 4
    shifts = tl.broadcast_to(shifts[None, :], (BLOCK_SIZE_K * (BLOCK_SIZE_N // 8), 8))
    shifts = tl.reshape(shifts, (BLOCK_SIZE_K, BLOCK_SIZE_N))

    # M, N offsets
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * (BLOCK_SIZE_N // 8) + tl.arange(0, BLOCK_SIZE_N // 8)
    offs_zn = pid_n * (BLOCK_SIZE_N // 8) + tl.arange(0, BLOCK_SIZE_N // 8)
    offs_sn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    masks_am = offs_am < M
    masks_bn = offs_bn < N // 8
    masks_zn = offs_zn < N // 8
    masks_sn = offs_sn < N

    # K offsets for this split-K partition (interleaved)
    offs_k = pid_z * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    # Pointer setup
    a_ptrs = a_ptr + K * offs_am[:, None] + offs_k[None, :]
    b_ptrs = b_ptr + (N // 8) * offs_k[:, None] + offs_bn[None, :]

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=accumulator_dtype)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        masks_k = offs_k < K
        masks_a = masks_am[:, None] & masks_k[None, :]
        a = tl.load(a_ptrs, mask=masks_a, other=0.0)

        masks_b = masks_k[:, None] & masks_bn[None, :]
        b = tl.load(b_ptrs, mask=masks_b, other=0)
        # Expand packed dim: [BLOCK_K, BLOCK_N//8] → [BLOCK_K, BLOCK_N]
        b = tl.interleave(b, b)
        b = tl.interleave(b, b)
        b = tl.interleave(b, b)

        # Group index for this K tile (all K in a block belong to same group
        # since BLOCK_SIZE_K=32 < group_size=128).
        szk = (BLOCK_SIZE_K * SPLIT_K * k + pid_z * BLOCK_SIZE_K) // group_size
        offs_szk = szk + tl.arange(0, 1)

        # Load zeros
        offs_z = (N // 8) * offs_szk[:, None] + offs_zn[None, :]
        masks_zk = offs_szk < K // group_size
        masks_z = masks_zk[:, None] & masks_zn[None, :]
        zeros = tl.load(zeros_ptr + offs_z, mask=masks_z, other=0)
        zeros = tl.interleave(zeros, zeros)
        zeros = tl.interleave(zeros, zeros)
        zeros = tl.interleave(zeros, zeros)
        zeros = tl.broadcast_to(zeros, (BLOCK_SIZE_K, BLOCK_SIZE_N))

        # Load scales
        offs_s = N * offs_szk[:, None] + offs_sn[None, :]
        masks_sk = offs_szk < K // group_size
        masks_s = masks_sk[:, None] & masks_sn[None, :]
        scales = tl.load(scales_ptr + offs_s, mask=masks_s, other=0.0)
        scales = tl.broadcast_to(scales, (BLOCK_SIZE_K, BLOCK_SIZE_N))

        # Dequantize
        b = (b >> shifts) & 0xF
        zeros = (zeros >> shifts) & 0xF
        b = (b - zeros) * scales
        b = b.to(c_ptr.type.element_ty)

        # Accumulate
        accumulator = tl.dot(a, b, accumulator, out_dtype=accumulator_dtype)

        offs_k += BLOCK_SIZE_K * SPLIT_K
        a_ptrs += BLOCK_SIZE_K * SPLIT_K
        b_ptrs += BLOCK_SIZE_K * SPLIT_K * (N // 8)

    c = accumulator.to(c_ptr.type.element_ty)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + pid_z * N * M + N * offs_cm[:, None] + offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def awq_gemm_triton(
    activations: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    group_size: int = 128,
    split_k_iters: int = 8,
) -> torch.Tensor:
    """Fused AWQ GEMM: activation @ dequantize(qweight, scales, qzeros).

    This avoids materializing the full weight matrix, saving bandwidth
    especially for small-batch (decode) scenarios.

    Ported from vllm's awq_gemm_triton with identical kernel logic.

    Args:
        activations: fp16/bf16[M, K] — input activations.
        qweight:  int32[K, N // 8] — packed 4-bit weights.
        scales:   fp16[K // G, N] — per-group per-output scales.
        qzeros:   int32[K // G, N // 8] — packed zero points.
        group_size: AWQ group size (default 128).
        split_k_iters: Number of K-partitions for split-K (default 8).
                       Must be a power of 2 and <= 32.

    Returns:
        Tensor of same dtype as input with shape [M, N].
    """
    M, K = activations.shape
    N = qweight.shape[1] * 8

    assert split_k_iters & (split_k_iters - 1) == 0, \
        f"split_k_iters must be power of 2, got {split_k_iters}"
    assert group_size <= K
    assert scales.shape[0] == K // group_size
    assert qzeros.shape[0] == K // group_size

    # Cast activations from bf16 to fp16 if needed (AWQ weights are fp16, and tl.dot
    # requires both operands to have the same dtype).  Preserve original
    # dtype to cast the output back.
    orig_dtype = activations.dtype
    if activations.dtype != torch.float16:
        activations = activations.to(torch.float16)

    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32

    grid = (
        triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),
        split_k_iters,
    )

    # 3D partial-result buffer for split-K reduction
    result = torch.zeros(
        (split_k_iters, M, N), dtype=torch.float16, device=activations.device,
    )

    _awq_gemm_kernel[grid](
        activations, qweight, scales, qzeros, result,
        M, N, K, group_size,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        SPLIT_K=split_k_iters,
    )

    result = result.sum(0)
    if orig_dtype != torch.float16:
        result = result.to(orig_dtype)
    return result
