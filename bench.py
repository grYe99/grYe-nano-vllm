import os
import time
import statistics
from random import randint, seed
from nanovllm import LLM, SamplingParams
# from vllm import LLM, SamplingParams


def main():
    seed(0)
    num_seqs = 256
    max_input_len = 1024
    max_ouput_len = 1024

    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    llm = LLM(path, enforce_eager=False, max_model_len=4096)

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]
    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    llm.generate(["Benchmark: "], SamplingParams())
    t = time.time()
    outputs = llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = (time.time() - t)
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t

    ttfts = [o["ttft_ms"] for o in outputs]
    tpots = [o["tpot_ms"] for o in outputs]
    # Flatten all per-step ITLs from all requests
    itls = [itl for o in outputs for itl in o["itl_ms"]]

    avg_ttft = statistics.mean(ttfts) if ttfts else 0
    med_ttft = statistics.median(ttfts) if ttfts else 0
    p99_ttft = sorted(ttfts)[int(len(ttfts) * 0.99)] if ttfts else 0

    avg_tpot = statistics.mean(tpots) if tpots else 0
    med_tpot = statistics.median(tpots) if tpots else 0
    p99_tpot = sorted(tpots)[int(len(tpots) * 0.99)] if tpots else 0

    avg_itl = statistics.mean(itls) if itls else 0
    med_itl = statistics.median(itls) if itls else 0
    p99_itl = sorted(itls)[int(len(itls) * 0.99)] if itls else 0

    print(f"============ Serving Benchmark Result ============")
    print(f"Successful requests:                     {num_seqs}       ")
    print(f"Benchmark duration (s):                  {t:.2f}     ")
    print(f"Total generated tokens:                  {total_tokens}     ")
    print(f"Output token throughput (tok/s):         {throughput:.2f}    ")
    print(f"---------------Time to First Token----------------")
    print(f"Mean TTFT (ms):                          {avg_ttft:.2f}  ")
    print(f"Median TTFT (ms):                        {med_ttft:.2f}  ")
    print(f"P99 TTFT (ms):                           {p99_ttft:.2f}  ")
    print(f"-----Time per Output Token (excl. 1st token)------")
    print(f"Mean TPOT (ms):                          {avg_tpot:.2f}  ")
    print(f"Median TPOT (ms):                        {med_tpot:.2f}  ")
    print(f"P99 TPOT (ms):                           {p99_tpot:.2f}  ")
    print(f"---------------Inter-token Latency----------------")
    print(f"Mean ITL (ms):                           {avg_itl:.2f}  ")
    print(f"Median ITL (ms):                         {med_itl:.2f}  ")
    print(f"P99 ITL (ms):                            {p99_itl:.2f}  ")
    print(f"==================================================")


if __name__ == "__main__":
    main()
