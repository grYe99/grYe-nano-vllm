"""
bench_chunked_prefill.py - Chunked Prefill Benchmark

验证混合批次路径是否触发：
  python bench_chunked_prefill.py --chunk-size 0
  python bench_chunked_prefill.py --chunk-size 256

预期：chunk-size>0 时，tqdm 应显示 Prefill 和 Decode 同时非零（混合批次触发的标志）
"""
import argparse
import time

MODEL = "./assets/Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--chunk-size", type=int, default=0,
                        help="prefill_chunk_size (0=disabled)")
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    from nanovllm import LLM, SamplingParams

    # 短 prompt：会快速完成 prefill，立刻进入 decode，制造"在途 decode 请求"
    short_prompts = [
        "The capital of France is",
        "Python is a programming language that",
        "The best way to learn is",
        "In the future, artificial intelligence will",
        "The most important thing in life is",
        "Scientists recently discovered that",
        "The history of computing began when",
        "To solve this problem, we need to",
    ]

    # 长 prompt：~1024 tokens，在 chunk_size=256 时需要 4 个 step 完成 prefill
    # 期间短 prompt 已在 decode，形成真正的混合批次
    long_prompts = [
        "hello " * 512,    # ~1024 tokens
        "world " * 512,
    ]

    prompts = short_prompts + long_prompts
    sp = SamplingParams(max_tokens=args.max_tokens, temperature=1.0)

    print(f"\nchunk_size={args.chunk_size}")
    print(f"workload: {len(short_prompts)} short prompts + {len(long_prompts)} long prompts (~1024 tokens each)")
    print(f"max_tokens={args.max_tokens}")
    print(f"预期：chunk_size>0 时 tqdm 同时显示 Prefill 和 Decode 非零")
    print()

    llm = LLM(args.model, prefill_chunk_size=args.chunk_size, enforce_eager=True)

    t_start = time.perf_counter()
    outputs = llm.generate(prompts, sp, use_tqdm=True)
    total_time = time.perf_counter() - t_start

    total_output_tokens = sum(len(o["token_ids"]) for o in outputs)
    print(f"\n结果: {len(outputs)} 个请求完成")
    print(f"总时间: {total_time:.2f}s")
    print(f"输出 token 总数: {total_output_tokens}")
    print(f"吞吐量: {total_output_tokens / total_time:.1f} tok/s")

if __name__ == "__main__":
    main()
