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
