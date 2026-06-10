#!/usr/bin/env python3
"""Verify all 4 ar_async_chunked / ar_fused_norm combos produce identical token_ids."""

import sys
from nanovllm import LLM
from nanovllm.sampling_params import SamplingParams


def main():
    MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "~/huggingface/Qwen3-0.6B/"

    # All 4 combinations
    COMBOS = [
        dict(ar_async_chunked=False, ar_fused_norm=False, label="sync + plain"),
        dict(ar_async_chunked=False, ar_fused_norm=True,  label="sync + fused"),
        dict(ar_async_chunked=True,  ar_fused_norm=False, label="chunked + plain"),
        dict(ar_async_chunked=True,  ar_fused_norm=True,  label="chunked + fused"),
    ]

    prompts = ["The capital of France is", "Today I feel", "def fibonacci(n):"]

    print(f"Model: {MODEL_PATH}")
    print(f"Prompts: {prompts}")
    print()

    results = {}
    for combo in COMBOS:
        print(f"--- Testing: {combo['label']} ---")
        llm = LLM(MODEL_PATH, tensor_parallel_size=2, **{k: combo[k] for k in ("ar_async_chunked", "ar_fused_norm")})
        output = llm.generate(prompts, SamplingParams(max_tokens=32))
        token_ids = tuple(tuple(o["token_ids"]) for o in output)
        results[combo["label"]] = token_ids
        for o in output:
            print(f"  {o['text'][:80]}")
        print()

    baseline = results["sync + plain"]
    print("=== Correctness Check ===")
    all_ok = True
    for label, token_ids in results.items():
        match = "✓" if token_ids == baseline else "✗ MISMATCH"
        if match != "✓":
            all_ok = False
        print(f"  {label:20s} {match}")

    print()
    if all_ok:
        print("PASS: All combinations produce identical token_ids!")
    else:
        print("FAIL: Some combinations differ from baseline.")
        sys.exit(1)


if __name__ == "__main__":
    main()
