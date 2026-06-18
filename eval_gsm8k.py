"""Run GSM8K evaluation with nano-vllm."""
import os
import lm_eval
from nanovllm.lm_eval_adapter import NanoVLLM

model = NanoVLLM(os.path.expanduser("~/huggingface/Qwen3-0.6B/"), kvcache_dtype="int8_per_token_head")
results = lm_eval.simple_evaluate(
    model=model,
    tasks=["gsm8k"],
    num_fewshot=5,
    batch_size="auto",
)
print(lm_eval.utils.make_table(results))
