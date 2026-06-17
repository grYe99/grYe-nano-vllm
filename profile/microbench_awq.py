"""Microbenchmark: compare AWQ kernel strategies across different M values.

Tests:
  1. CUDA custom op (torch.ops.nanovllm.awq_gemm) — all M
  2. Triton fused GEMM (awq_gemm_triton) — kept for comparison
  3. dequant + cuBLAS (F.linear)

Usage:
    python profile/microbench_awq.py
"""
import time
import torch
from nanovllm.layers.quantization.awq_triton import awq_dequantize, awq_gemm_triton
from nanovllm.layers.quantization.awq import (
    _awq_to_marlin_zero_points, _marlin_permute_scales,
)

# Marlin setup
import nanovllm._C_marlin  # noqa: F401 — registers marlin ops under torch.ops.nanovllm
_num_sms = torch.cuda.get_device_properties(0).multi_processor_count
_workspace = torch.zeros(_num_sms, dtype=torch.int32, device='cuda')
# kU4 ScalarType ID: uint(4)=ScalarType(0,4,false,0,false,NAN_IEEE_754=1)
# Packed: mantissa(4)<<8 | nan_repr(1)<<50 = 1125899906843648
_MARLIN_KU4_ID = 1125899906843648

torch.set_default_device("cuda")

K = 4096   # in_features
N = 4096   # out_features
GROUP_SIZE = 128
PACK_FACTOR = 8
NUM_GROUPS = K // GROUP_SIZE

M_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
WARMUP = 10
ITERS = 50


def create_awq_weights():
    """Create random AWQ weights with realistic distribution."""
    qweight = torch.randint(0, 2**31, (K, N // PACK_FACTOR), dtype=torch.int32)
    qzeros = torch.randint(0, 2**31, (NUM_GROUPS, N // PACK_FACTOR), dtype=torch.int32)
    scales = torch.randn(NUM_GROUPS, N, dtype=torch.float16) * 0.1
    # Marlin repack (qweight, qzeros, scales all converted)
    marlin_qweight = torch.ops.nanovllm.awq_marlin_repack(qweight, K, N, 4, False)
    marlin_qzeros = _awq_to_marlin_zero_points(qzeros, NUM_GROUPS, N, 4)
    marlin_scales = _marlin_permute_scales(scales, K, N, GROUP_SIZE)
    return qweight, scales, qzeros, marlin_qweight, marlin_qzeros, marlin_scales


def benchmark(fn, label, M):
    """Run benchmark and return median latency in microseconds."""
    # Warmup
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()

    # Timed runs
    start = time.perf_counter()
    for _ in range(ITERS):
        fn()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1e6 / ITERS  # us
    return elapsed


def main():
    qweight, scales, qzeros, marlin_qweight, marlin_qzeros, marlin_scales = create_awq_weights()

    print(f"K={K}, N={N}, group_size={GROUP_SIZE}")
    print(f"{'M':>6} | {'CUDA op':>10} | {'Triton':>12} | {'deq+cuB':>12} | {'Marlin':>12} | {'best':>8}")
    print("-" * 70)

    for M in M_VALUES:
        act = torch.randn(M, K, dtype=torch.float16)

        # 1. CUDA custom op (all M)
        def fn_cuda(a=act, w=qweight, s=scales, z=qzeros):
            return torch.ops.nanovllm.awq_gemm(a, w, s, z, 8)
        t_cuda = benchmark(fn_cuda, "CUDA op", M)

        # 2. Triton fused GEMM
        def fn_triton(a=act, w=qweight, s=scales, z=qzeros):
            return awq_gemm_triton(a, w, s, z, GROUP_SIZE, split_k_iters=8)
        t_triton = benchmark(fn_triton, "Triton", M)

        # 3. dequant + cuBLAS
        def fn_dequant(a=act, w=qweight, s=scales, z=qzeros):
            deq = awq_dequantize(w, s, z, GROUP_SIZE)
            return a @ deq.t()
        t_deq = benchmark(fn_dequant, "dequant+cuBLAS", M)

        # 4. Marlin GEMM (correct format conversion)
        def fn_marlin(a=act, w=marlin_qweight, ms=marlin_scales, mz=marlin_qzeros):
            return torch.ops.nanovllm.marlin_gemm(
                a, None, w, None, ms, None, None, mz,
                None, None, _workspace, _MARLIN_KU4_ID, M, N, K,
                True, False, False, False,
            )
        t_marlin = benchmark(fn_marlin, "Marlin", M)

        # Determine best
        times = {"CUDA": t_cuda, "Triton": t_triton, "deq+cuB": t_deq, "Marlin": t_marlin}
        best_label = min(times, key=times.get)

        def fmt(t):
            return f"{t:>8.1f}" if t != float("inf") else f"{'N/A':>10}"

        print(f"{M:>6} | {fmt(t_cuda)} | {fmt(t_triton)} | {fmt(t_deq)} | {fmt(t_marlin)} | {best_label:>8}")


if __name__ == "__main__":
    main()
