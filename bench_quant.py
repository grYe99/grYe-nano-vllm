# bench_quant.py
"""
W4A16 fused kernel benchmark: 对比 FP16 / unfused dequant+matmul / fused kernel

用法:
    python bench_quant.py

示例输出:
    GPU: NVIDIA GeForce RTX 4090

    M=  1, K=4096, N=4096 | fp16: 0.082ms | unfused: 0.061ms | fused: 0.028ms | speedup vs fp16: 2.93x
    M=  4, K=4096, N=4096 | fp16: 0.085ms | unfused: 0.064ms | fused: 0.031ms | speedup vs fp16: 2.74x
    ...
"""
import torch
from nanovllm.layers.quant_kernel import w4a16_gemm, pack_weight_to_int4


def dequant_to_fp16(w_packed: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor,
                    K: int, group_size: int = 128) -> torch.Tensor:
    """Unfused path: fully dequantize int4 weight to FP16 tensor."""
    N = w_packed.shape[1]
    num_groups = K // group_size
    # Unpack half-split int4
    w_packed_3d = w_packed.reshape(num_groups, group_size // 2, N)
    w_lo = (w_packed_3d & 0xF).float()                # [G, 64, N]
    w_hi = ((w_packed_3d >> 4) & 0xF).float()         # [G, 64, N]
    half = group_size // 2
    w_quant = torch.empty(num_groups, group_size, N, dtype=torch.float32, device=w_packed.device)
    w_quant[:, :half, :] = w_lo
    w_quant[:, half:, :] = w_hi
    # Dequantize
    s = scales.float().unsqueeze(1)   # [G, 1, N]
    z = zeros.float().unsqueeze(1)    # [G, 1, N]
    w_fp = (w_quant - z) * s          # [G, 128, N]
    return w_fp.reshape(K, N).T.contiguous().half()   # [N, K]


def benchmark(fn, warmup: int = 20, rep: int = 200) -> float:
    """Return median latency in milliseconds."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    # Time
    import time
    times = []
    for _ in range(rep):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2] * 1000  # median, ms


def run_bench(M: int, K: int, N: int, group_size: int = 128) -> None:
    x = torch.randn(M, K, dtype=torch.float16, device="cuda")
    w_fp16 = torch.randn(N, K, dtype=torch.float16, device="cuda")
    w_packed, scales, zeros = pack_weight_to_int4(w_fp16, group_size)

    # FP16 baseline
    t_fp16 = benchmark(lambda: x @ w_fp16.T)

    # Unfused: dequant to FP16 first, then matmul
    w_dq = dequant_to_fp16(w_packed, scales, zeros, K, group_size)
    t_unfused = benchmark(lambda: x @ w_dq.T)

    # Fused kernel
    t_fused = benchmark(lambda: w4a16_gemm(x, w_packed, scales, zeros, group_size))

    speedup = t_fp16 / t_fused
    print(f"M={M:3d}, K={K}, N={N} | "
          f"fp16: {t_fp16:.3f}ms | unfused: {t_unfused:.3f}ms | fused: {t_fused:.3f}ms | "
          f"speedup vs fp16: {speedup:.2f}x")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("No GPU available, skipping benchmark.")
        raise SystemExit(0)

    print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    print("--- Typical linear layer sizes (Qwen3-0.6B / 7B style) ---")
    for M in [1, 4, 16, 32]:
        run_bench(M, K=4096, N=4096)

    print()
    print("--- Small layer sizes ---")
    for M in [1, 4, 16, 32]:
        run_bench(M, K=1024, N=1024)
