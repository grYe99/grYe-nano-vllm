import csv
import os
from torch.profiler import profile, ProfilerActivity
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
tokenizer = AutoTokenizer.from_pretrained(path)
llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = [
    "introduce yourself",
    "list all prime numbers within 100",
]
prompts = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    for prompt in prompts
]

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    outputs = llm.generate(prompts, sampling_params)

os.makedirs("profile", exist_ok=True)
prof.export_chrome_trace("profile/trace.json")

# print to console
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=50))

# export full CSV
# events = prof.key_averages()
# with open("profile/profiler_results.csv", "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["Name", "Self CPU (us)", "CPU Total (us)", "CPU Time Avg (us)",
#                       "CUDA Total (us)", "CUDA Time Avg (us)", "# of Calls"])
#     for e in events:
#         writer.writerow([
#             e.key,
#             e.self_cpu_time_total,
#             e.cpu_time_total,
#             e.cpu_time_total / max(e.count, 1),
#             e.device_time * e.count,
#             e.device_time,
#             e.count,
#         ])

# print("CSV exported to profile/profiler_results.csv")
