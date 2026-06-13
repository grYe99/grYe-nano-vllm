"""
AWQ quantization kernels: dequantize and fused gemm.

AWQ packs 4-bit weights into int32 with a non-standard order:
  within each int32 [b0 b1 ... b31], the 8 int4 values are stored as
  indices [0, 4, 1, 5, 2, 6, 3, 7].

Parameter shapes (M=out_features, K=in_features, group_size=128):
    qweight: int32 [M, K // 8]         — packed 4-bit weights
    scales:  fp16  [K // 128, M]       — per-group per-row scale
    qzeros:  int32 [K // 128, M // 8]  — packed 4-bit zero points
"""
import torch
import triton
import triton.language as tl

AWQ_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]


def awq_dequantize(qweight: torch.Tensor, scales: torch.Tensor,
                   qzeros: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    """Pure-PyTorch AWQ dequantize — reference implementation.

    Returns fp16 weight of shape [M, K].
    """
    M, K8 = qweight.shape
    K = K8 * 8
    num_groups = K // group_size

    order = qweight.new_tensor(AWQ_ORDER)  # [8]

    # Unpack qweight: [M, K/8] → [M, K/8, 8] → [M, K]
    unpacked = (qweight.unsqueeze(-1) >> (order * 4)) & 0xF       # [M, K/8, 8]
    unpacked = unpacked.to(scales.dtype).reshape(M, K)             # [M, K]

    # Unpack qzeros: [num_groups, M/8] → [num_groups, M/8, 8] → [num_groups, M]
    z_unpacked = (qzeros.unsqueeze(-1) >> (order * 4)) & 0xF
    z_unpacked = z_unpacked.to(scales.dtype).reshape(num_groups, M)  # [num_groups, M]

    # Weight 3d: [M, num_groups, group_size]
    w3 = unpacked.reshape(M, num_groups, group_size).permute(1, 0, 2)  # [num_groups, M, group_size]
    z3 = z_unpacked.unsqueeze(-1)   # [num_groups, M, 1]
    s3 = scales.unsqueeze(-1)       # [num_groups, M, 1]

    w3 = (w3 - z3) * s3             # [num_groups, M, group_size]
    w3 = w3.permute(1, 0, 2)        # [M, num_groups, group_size]
    return w3.reshape(M, K)         # [M, K]


@triton.jit
def _awq_dequantize_kernel(
    qweight_ptr, scales_ptr, qzeros_ptr, output_ptr,
    M, K, num_groups, group_size,
    stride_qw_m, stride_qw_k8,
    stride_s_g, stride_s_m,
    stride_qz_g, stride_qz_m8,
    stride_out_m, stride_out_k,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Triton kernel: dequantize one tile [BLOCK_M, BLOCK_K] of the weight matrix.

    Each program handles a tile of the output.
    """
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    m_mask = off_m < M
    k_mask = off_k < K

    # Compute which groups each k belongs to
    group_ids = off_k // group_size            # [BLOCK_K]
    k_within_group = off_k % group_size        # [BLOCK_K]
    # Packed column index within the group's section of qweight
    pack_col_in_group = k_within_group // 8    # [BLOCK_K]
    pack_slot = k_within_group % 8             # [BLOCK_K]

    # qweight column = group_id * (group_size // 8) + pack_col_in_group
    qw_k8_off = group_ids * (group_size // 8) + pack_col_in_group  # [BLOCK_K]

    # Load qweight tile: [BLOCK_M, BLOCK_K] mapping through qw_k8_off
    # We need a 2D load from qweight; the column indices vary per row of the tile.
    # Use a mask to guard out-of-bounds accesses.
    qw_ptrs = qweight_ptr + off_m[:, None] * stride_qw_m + qw_k8_off[None, :] * stride_qw_k8
    qw_packed = tl.load(qw_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0)  # [BLOCK_M, BLOCK_K]

    # AWQ unpack: slot order within int32 is [0,4,1,5,2,6,3,7]
    # shift amounts = [0, 16, 4, 20, 8, 24, 12, 28]
    shift = tl.where(pack_slot == 0, 0,
             tl.where(pack_slot == 1, 16,
             tl.where(pack_slot == 2, 4,
             tl.where(pack_slot == 3, 20,
             tl.where(pack_slot == 4, 8,
             tl.where(pack_slot == 5, 24,
             tl.where(pack_slot == 6, 12, 28)))))))  # [BLOCK_K]
    int4_val = (qw_packed >> shift[None, :]) & 0xF  # [BLOCK_M, BLOCK_K]

    # Load scales: scales[group_id, off_m] → shape [BLOCK_M, BLOCK_K] via broadcasting
    s_ptrs = scales_ptr + group_ids[None, :] * stride_s_g + off_m[:, None] * stride_s_m
    s = tl.load(s_ptrs, mask=m_mask[:, None] & k_mask[None, :])  # [BLOCK_M, BLOCK_K]

    # Load qzeros: qzeros[group_id, off_m // 8]
    z_m8_off = off_m // 8                                                         # [BLOCK_M]
    z_ptrs = qzeros_ptr + group_ids[None, :] * stride_qz_g + z_m8_off[:, None] * stride_qz_m8
    z_packed = tl.load(z_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0)  # [BLOCK_M, BLOCK_K]

    # Unpack zero points: slot = off_m % 8
    z_slot = off_m % 8                                              # [BLOCK_M]
    z_shift = tl.where(z_slot == 0, 0,
              tl.where(z_slot == 1, 16,
              tl.where(z_slot == 2, 4,
              tl.where(z_slot == 3, 20,
              tl.where(z_slot == 4, 8,
              tl.where(z_slot == 5, 24,
              tl.where(z_slot == 6, 12, 28)))))))                  # [BLOCK_M]
    z_val = (z_packed >> z_shift[:, None]) & 0xF                   # [BLOCK_M, BLOCK_K]

    # Dequantize
    weight_fp16 = (int4_val.to(tl.float16) - z_val.to(tl.float16)) * s

    # Store
    out_ptrs = output_ptr + off_m[:, None] * stride_out_m + off_k[None, :] * stride_out_k
    tl.store(out_ptrs, weight_fp16, mask=m_mask[:, None] & k_mask[None, :])


def triton_awq_dequantize(qweight, scales, qzeros, group_size=128):
    """Triton-accelerated AWQ dequantize.

    Same interface as awq_dequantize().
    """
    M, K8 = qweight.shape
    K = K8 * 8
    num_groups = K // group_size
    output = torch.empty(M, K, dtype=scales.dtype, device=qweight.device)

    # Block sizes
    BLOCK_M = 64
    BLOCK_K = 128
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_K))

    _awq_dequantize_kernel[grid](
        qweight, scales, qzeros, output,
        M, K, num_groups, group_size,
        qweight.stride(0), qweight.stride(1),
        scales.stride(0), scales.stride(1),
        qzeros.stride(0), qzeros.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
    )
    return output
