"""Compare FP16 vs INT8 KV cache generation token-by-token on GSM8K.

Usage:
    python profile/compare_cache.py [--samples N] [--max-tokens M]
"""
import gc
import os
import sys
import json
import argparse

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nanovllm import LLM, SamplingParams

MODEL_PATH = os.path.expanduser("~/huggingface/Qwen3-0.6B/")

# Few-shot examples for GSM8K (matching lm_eval's base gsm8k.yaml format)
# Format: "Question: {{question}}\nAnswer: {{answer_with_####_stripped}}\n\n"
# We use the first 5 training examples (consistent across runs for reproducibility)
FEW_SHOT = [
    (
        "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. #### 6",
    ),
    (
        "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. #### 5",
    ),
    (
        "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. #### 39",
    ),
    (
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. #### 8",
    ),
    (
        "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. #### 9",
    ),
]

# 20 test questions from GSM8K test set
TEST_QUESTIONS = [
    "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
    "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
    "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
    "A robe takes 2 1/2 yards of fabric and 3/8 yards of trim. How many yards of fabric and trim together are needed to make 12 robes?",
    "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins with four every day. She sells the rest at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make daily at the farmers' market?",
    "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables. How many cups of food do the chickens eat in one week?",
    "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
    "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
    "Faye is planning on renting a car while on vacation. The rental company charges a flat fee of $50 plus $0.50 per mile driven. The total cost of renting the car and driving it 150 miles is $125. How much would it cost to rent the car and drive it 250 miles?",
    "A baker bought 4 bags of flour. Each bag weighs 5 kg. If he uses 1/2 kg of flour for each cake, how many cakes can he make?",
    "A copy machine makes 32 copies per minute. How many copies does it make in 4 minutes and 15 seconds?",
    "Eddie's Earnings: Eddie worked 30 hours at his part-time job. He gets paid $15 per hour. He also received $20 in tips. How much did Eddie earn in total?",
    "The pen costs $1.50. The notebook costs $2.25. How much change should you get if you pay with a $10 bill?",
    "At a restaurant, you order a meal that costs $12.50. You also order a drink that costs $2.25. If the sales tax is 8%, what is the total cost of your meal including tax?",
    "A rectangular garden has a length of 15 feet and a width of 10 feet. What is the perimeter of the garden in feet?",
    "A store sold 36 pairs of shoes in one day. Each pair costs $45. If each customer bought exactly 3 pairs, how many customers bought shoes that day?",
    "Sarah has 24 apples. She wants to put them into bags with 6 apples each. How many bags does she need?",
    "Tom has 5 red marbles and 3 blue marbles. He gives 2 red marbles to his friend. How many red marbles does Tom have now?",
    "A train travels at 60 miles per hour. How far will it travel in 2.5 hours?",
    "There are 8 rows of chairs in a school auditorium. Each row has 12 chairs. If 30 chairs are occupied, how many chairs are empty?",
]


def format_gsm8k_prompt(question: str) -> str:
    """Format a question with 5 few-shot examples matching lm_eval's gsm8k template."""
    prefix = ""
    for q, a in FEW_SHOT:
        prefix += f"Question: {q}\nAnswer: {a}\n\n"
    prefix += f"Question: {question}\nAnswer:"
    return prefix


def run_and_collect(dtype: str, questions: list[str], max_tokens: int = 300):
    """Initialize LLM with given KV cache dtype, generate all samples, return tokens."""
    print(f"\n{'='*60}")
    print(f"  Running with kvcache_dtype={dtype}")
    print(f"{'='*60}")

    llm = LLM(MODEL_PATH, kvcache_dtype=dtype, max_num_seqs=1,
              enforce_eager=True, gpu_memory_utilization=0.5)

    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)

    prompts = [format_gsm8k_prompt(q) for q in questions]

    try:
        results = llm.generate(prompts, sp, use_tqdm=True)
    finally:
        try:
            llm.exit()
        except Exception:
            pass
        del llm
        gc.collect()
        torch.cuda.empty_cache()

    return [r["token_ids"] for r in results]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=300)
    args = parser.parse_args()

    questions = TEST_QUESTIONS[:min(args.samples, len(TEST_QUESTIONS))]
    n = len(questions)

    # Run FP16
    fp16_tokens = run_and_collect("auto", questions, args.max_tokens)

    # Run INT8
    int8_tokens = run_and_collect("int8_per_token_head", questions, args.max_tokens)

    # Compare
    print(f"\n{'='*60}")
    print(f"  Divergence Analysis ({n} samples)")
    print(f"{'='*60}")

    divergence_stats = {}
    for i in range(n):
        fp = fp16_tokens[i]
        int8 = int8_tokens[i]
        min_len = min(len(fp), len(int8))
        div_point = None
        for j in range(min_len):
            if fp[j] != int8[j]:
                div_point = j
                break
        divergence_stats[i] = {
            "fp16_len": len(fp),
            "int8_len": len(int8),
            "min_len": min_len,
            "first_divergence": div_point,
        }
        if div_point is None:
            if len(fp) == len(int8):
                print(f"  Sample {i:2d}: IDENTICAL ({min_len} tokens)")
            else:
                print(f"  Sample {i:2d}: FIRST {min_len} tokens SAME, "
                      f"FP16={len(fp)} INT8={len(int8)} diff={len(fp)-len(int8)}")
        else:
            first_fp = fp[div_point] if div_point < len(fp) else None
            first_int8 = int8[div_point] if div_point < len(int8) else None
            print(f"  Sample {i:2d}: DIVERGE at token {div_point:3d} "
                  f"(FP16={first_fp}, INT8={first_int8}) "
                  f"[FP16_len={len(fp)}, INT8_len={len(int8)}]")

    # Summary
    diverged = [v for v in divergence_stats.values() if v["first_divergence"] is not None]
    identical = [v for v in divergence_stats.values() if v["first_divergence"] is None]

    print(f"\n  Summary:")
    print(f"    Total samples: {n}")
    print(f"    Identical:     {len(identical)}")
    print(f"    Diverged:      {len(diverged)}")

    if diverged:
        div_tokens = [v["first_divergence"] for v in diverged]
        print(f"    Divergence at token:")
        print(f"      min   = {min(div_tokens)}")
        print(f"      max   = {max(div_tokens)}")
        print(f"      mean  = {sum(div_tokens)/len(div_tokens):.1f}")
        print(f"      median= {sorted(div_tokens)[len(div_tokens)//2]}")
        for threshold in [1, 2, 5, 10, 20, 50, 100]:
            count = sum(1 for t in div_tokens if t >= threshold)
            print(f"      >= {threshold:3d} tokens: {count}/{len(diverged)}")


if __name__ == "__main__":
    main()
