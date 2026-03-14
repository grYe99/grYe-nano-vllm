"""
bench_metrics.py - 展示 MetricsCollector 收集的 TTFT/ITL/E2E 指标

用法：
  python bench_metrics.py

预期输出：tqdm 进度条 + 结束后打印 Metrics Summary
"""
from nanovllm import LLM, SamplingParams
from nanovllm.utils.metrics import MetricsCollector

MODEL = "./assets/Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"

prompts = [
    "The capital of France is",
    "Python is a programming language that",
    "The best way to learn is",
    "In the future, artificial intelligence will",
    "The most important thing in life is",
    "Scientists recently discovered that",
    "The history of computing began when",
    "To solve this problem, we need to",
    "a " * 200,   # ~200 token prompt，测试较长 prefill 的 TTFT
    "b " * 200,
]

sp = SamplingParams(max_tokens=32, temperature=1.0)
metrics = MetricsCollector()
llm = LLM(MODEL, enforce_eager=True, metrics=metrics)
outputs = llm.generate(prompts, sp, use_tqdm=True)

print(f"\n生成了 {len(outputs)} 个响应")
