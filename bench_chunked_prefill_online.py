"""
bench_chunked_prefill_online.py - Chunked Prefill Online Arrival Benchmark

模拟在线到达场景：短 prompt 先进入 decode，再注入长 prompt，
观察长 prompt 的 prefill 是否阻塞短 prompt 的 decode（ITL 毛刺）。

用法：
  python bench_chunked_prefill_online.py --chunk-size 0    # 原始调度器
  python bench_chunked_prefill_online.py --chunk-size 256  # chunked prefill

预期：
  chunk-size=0  : itl_p99 远高于 itl_p50（长 prefill 阻塞 decode，出现毛刺）
  chunk-size=256: itl_p99 接近 itl_p50（decode 与 prefill 交替，无毛刺）
"""
import argparse
from nanovllm import LLM, SamplingParams
from nanovllm.utils.metrics import MetricsCollector

MODEL = "./assets/Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"

# 短 prompt：每个 ~10 tokens，prefill 极快，大部分时间在 decode
SHORT_PROMPTS = [
    "The capital of France is",
    "Python is a programming language that",
    "The best way to learn is",
    "In the future, artificial intelligence will",
    "The most important thing in life is",
    "Scientists recently discovered that",
    "The history of computing began when",
    "To solve this problem, we need to",
]

# 长 prompt：~1024 tokens（"hello " 每次约 2 tokens，512 次 ≈ 1024 tokens）
# 在 chunk_size=256 时需要 2 步完成 prefill；chunk_size=0 时一步完成但耗时长
LONG_PROMPTS = [
    "hello " * 512,
    "world " * 512,
]


def run(model: str, chunk_size: int, max_tokens: int):
    print(f"\n{'='*50}")
    print(f"chunk_size={chunk_size}, max_tokens={max_tokens}")
    print(f"短 prompt: {len(SHORT_PROMPTS)} 个，长 prompt: {len(LONG_PROMPTS)} 个 (~1024 tokens)")
    print(f"场景: 短 prompt 先进入 decode，长 prompt 后注入")
    print(f"{'='*50}")

    sp_short = SamplingParams(max_tokens=max_tokens, temperature=1.0)
    sp_long  = SamplingParams(max_tokens=32, temperature=1.0)  # 长 prompt 少生成

    metrics = MetricsCollector()
    llm = LLM(model, prefill_chunk_size=chunk_size, enforce_eager=True, metrics=metrics)

    # --- Phase 1: 提交短 prompt ---
    for p in SHORT_PROMPTS:
        llm.add_request(p, sp_short)

    metrics.on_generate_start()
    outputs = {}

    # --- Phase 1 loop: 运行直到至少有一个短 prompt 完成 prefill 并进入 decode ---
    long_injected = False
    while not llm.is_finished():
        step_outputs, num_tokens = llm.step()
        for seq_id, token_ids in step_outputs:
            outputs[seq_id] = token_ids

        # 注入时机：running 中存在已完成 prefill 的 seq（即有 seq 在 decode）
        # 且长 prompt 尚未注入
        if not long_injected:
            if any(seq.is_prefill_done for seq in llm.scheduler.running):
                # 此时短 prompt 正在 decode，注入长 prompt 触发阻塞场景
                for p in LONG_PROMPTS:
                    llm.add_request(p, sp_long)
                long_injected = True
                print(f"[注入长 prompt] 共 {len(llm.scheduler.running)} 个 seq 在 running，"
                      f"{sum(1 for s in llm.scheduler.running if s.is_prefill_done)} 个在 decode")
                # 长 prompt 此时在 waiting 队列，下一个 step 调度器会将其与
                # 正在 decode 的短 prompt 一起调度，从而触发 prefill-vs-decode 竞争

    metrics.on_generate_end()

    if not long_injected:
        print("警告：长 prompt 未成功注入（所有短 prompt 已完成），场景未触发")

    metrics.print_summary()

    total_tokens = sum(len(v) for v in outputs.values())
    print(f"\n完成请求数: {len(outputs)}")
    print(f"输出 token 总数: {total_tokens}")
    print()

    # 返回关键指标供对比
    s = metrics.summary()
    return {
        "chunk_size": chunk_size,
        "itl_p50_ms": s["itl_p50"] * 1000,
        "itl_p99_ms": s["itl_p99"] * 1000,
        "ttft_p99_ms": s["ttft_p99"] * 1000,
        "throughput": s["throughput_tok_s"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--chunk-size", type=int, default=256,
                        help="prefill_chunk_size (0=disabled, 256=recommended)")
    parser.add_argument("--max-tokens", type=int, default=128,
                        help="短 prompt 的最大生成 token 数，越大 decode 阶段越长，效果越明显")
    parser.add_argument("--compare", action="store_true",
                        help="连续运行 chunk_size=0 和 chunk_size=256 并打印对比表")
    args = parser.parse_args()

    if args.compare:
        results = []
        for cs in [0, 256]:
            r = run(args.model, cs, args.max_tokens)
            results.append(r)

        print("\n" + "="*60)
        print("对比结果（在线到达场景）")
        print("="*60)
        print(f"{'指标':<20} {'chunk=0':>12} {'chunk=256':>12} {'改善':>10}")
        print("-"*60)
        r0, r256 = results[0], results[1]
        itl_improve = (r0["itl_p99_ms"] - r256["itl_p99_ms"]) / (r0["itl_p99_ms"] or 1) * 100
        print(f"{'ITL p99 (ms)':<20} {r0['itl_p99_ms']:>12.1f} {r256['itl_p99_ms']:>12.1f} {itl_improve:>+9.1f}%")
        print(f"{'ITL p50 (ms)':<20} {r0['itl_p50_ms']:>12.1f} {r256['itl_p50_ms']:>12.1f}")
        itl_ratio_0   = r0['itl_p99_ms'] / (r0['itl_p50_ms'] or 1)
        itl_ratio_256 = r256['itl_p99_ms'] / (r256['itl_p50_ms'] or 1)
        print(f"{'ITL p99/p50 比值':<20} {itl_ratio_0:>12.2f} {itl_ratio_256:>12.2f}  (越接近1越稳定)")
        print(f"{'TTFT p99 (ms)':<20} {r0['ttft_p99_ms']:>12.1f} {r256['ttft_p99_ms']:>12.1f}")
        tput_change = (r256['throughput'] - r0['throughput']) / (r0['throughput'] or 1) * 100
        print(f"{'Throughput (tok/s)':<20} {r0['throughput']:>12.1f} {r256['throughput']:>12.1f} {tput_change:>+9.1f}%")
        print("="*60)
        print("\n解读：")
        print(f"  ITL p99/p50 比值: chunk=0 时为 {itl_ratio_0:.2f}x，chunk=256 时为 {itl_ratio_256:.2f}x")
        print(f"  比值越接近 1.0 说明 ITL 越稳定（无毛刺）")
        if itl_ratio_256 < itl_ratio_0 * 0.8:
            print(f"  ✓ chunked prefill 有效：ITL 稳定性提升 {(1-itl_ratio_256/(itl_ratio_0 or 1))*100:.0f}%")
        else:
            print(f"  ✗ 改善不明显，尝试增大 --max-tokens 或使用更长的 long prompt")
    else:
        run(args.model, args.chunk_size, args.max_tokens)


if __name__ == "__main__":
    main()
