#!/usr/bin/env python3
"""Verify all 4 ar_async_chunked / ar_fused_norm combos produce identical token_ids.

Each combo runs in a separate subprocess to avoid CUDA memory / process group conflicts.
"""

import sys
import os
import subprocess
import json
import tempfile
from pathlib import Path

MODEL_PATH = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/huggingface/Qwen3-0.6B/")
COMBOS = [
    dict(ar_async_chunked=False, ar_fused_norm=False, label="sync + plain"),
    dict(ar_async_chunked=False, ar_fused_norm=True,  label="sync + fused"),
    dict(ar_async_chunked=True,  ar_fused_norm=False, label="chunked + plain"),
    dict(ar_async_chunked=True,  ar_fused_norm=True,  label="chunked + fused"),
]

# Each combo saves its token_ids to a temp file
results = {}
for combo in COMBOS:
    label = combo["label"]
    print(f"--- Testing: {label} ---")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        out_path = f.name

    script = f"""
import sys, json
sys.path.insert(0, "{Path(__file__).parent}")
from nanovllm import LLM
from nanovllm.sampling_params import SamplingParams
llm = LLM("{MODEL_PATH}", tensor_parallel_size=2,
          ar_async_chunked={combo["ar_async_chunked"]},
          ar_fused_norm={combo["ar_fused_norm"]})
output = llm.generate(
    ["The capital of France is", "Today I feel", "def fibonacci(n):"],
    SamplingParams(max_tokens=32),
    use_tqdm=False,
)
json.dump([o["token_ids"] for o in output], open("{out_path}", "w"))
for o in output:
    print("  ", o["text"][:80])
llm.exit()
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=120,
    )
    print(result.stdout)
    if result.stderr:
        print("stderr:", result.stderr[:500])
    if result.returncode != 0:
        print(f"FAILED ({result.returncode})")
        sys.exit(1)
    with open(out_path) as f:
        results[label] = tuple(tuple(t) for t in json.load(f))
    Path(out_path).unlink()
    print()

# Compare against baseline
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
