"""Microbenchmark: compare AWQ kernel strategies across different M values.

Tests:
  1. CUDA custom op (torch.ops.nanovllm.awq_gemm) — M ≤ 16 only
  2. Triton fused GEMM (awq_gemm_triton)
  3. dequant + cuBLAS (F.linear)

Usage:
    python profile/microbench_awq.py
"""
import time
import torch
from nanovllm.layers.quantization.awq_triton import awq_dequantize, awq_gemm_triton

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
    return qweight, scales, qzeros


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
    qweight, scales, qzeros = create_awq_weights()

    print(f"K={K}, N={N}, group_size={GROUP_SIZE}")
    print(f"{'M':>6} | {'CUDA op':>10} | {'Triton fused':>13} | {'dequant+cuBLAS':>16} | {'best':>8}")
    print("-" * 65)

    for M in M_VALUES:
        act = torch.randn(M, K, dtype=torch.float16)

        # 1. CUDA custom op (M ≤ 16 only)
        if M <= 16:
            def fn_cuda(a=act, w=qweight, s=scales, z=qzeros):
                return torch.ops.nanovllm.awq_gemm(a, w, s, z, 8)
            t_cuda = benchmark(fn_cuda, "CUDA op", M)
        else:
            t_cuda = float("inf")

        # 2. Triton fused GEMM
        def fn_triton(a=act, w=qweight, s=scales, z=qzeros):
            return awq_gemm_triton(a, w, s, z, GROUP_SIZE, split_k_iters=8)
        t_triton = benchmark(fn_triton, "Triton", M)

        # 3. dequant + cuBLAS
        # Warm up @torch.compile once for the first call
        def fn_dequant(a=act, w=qweight, s=scales, z=qzeros):
            deq = awq_dequantize(w, s, z, GROUP_SIZE)
            return a @ deq.t()
        t_deq = benchmark(fn_dequant, "dequant+cuBLAS", M)

        # Determine best
        times = {"CUDA": t_cuda, "Triton": t_triton, "deq+cuB": t_deq}
        best_label = min(times, key=times.get)

        def fmt(t):
            return f"{t:>8.1f}" if t != float("inf") else f"{'N/A':>10}"

        print(f"{M:>6} | {fmt(t_cuda)} | {fmt(t_triton)} | {fmt(t_deq)} | {best_label:>8}")


if __name__ == "__main__":
    main()
