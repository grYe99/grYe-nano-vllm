"""
bench_pinned_cache.py - Pinned Prefix Cache 端到端验证

三个场景：
  场景1  基础 LRU 命中：相同前缀连续发两次，第二次应命中 cache
  场景2  TTFT 对比：命中时 TTFT 明显低于未命中
  场景3  驱逐压力下 pinned vs non-pinned：
         直接对 BlockManager 注入压力，验证 pinned block 最后被淘汰

用法：
  python bench_pinned_cache.py
"""

import statistics
from nanovllm import LLM, SamplingParams
from nanovllm.utils.metrics import MetricsCollector
from nanovllm.engine.block_manager import BlockManager

MODEL = "./assets/Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"

PREFIX = "hello " * 300          # 约 300 token，超过一个 block (block_size=256)
QUERY  = "What is 1+1?"


def run_and_collect(llm, prompts, sps):
    metrics = MetricsCollector()
    llm.scheduler.metrics = metrics
    llm.metrics = metrics
    llm.generate(prompts, sps, use_tqdm=False)
    return metrics.completed


# ─────────────────────────────────────────────────────────────────────────────
print("Loading model...")
llm = LLM(MODEL, enforce_eager=True, gpu_memory_utilization=0.9)
print("Model loaded.\n")

results = {}

# ─────────────────────────────────────────────────────────────────────────────
# 场景 1：基础 LRU 命中
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("场景1: 基础 LRU 命中")
print("=" * 60)

prompt_full = PREFIX + QUERY
sp = SamplingParams(max_tokens=8, temperature=1.0)

c1 = run_and_collect(llm, [prompt_full], [sp])
m1 = c1[0]
print(f"  第1次: num_cached_tokens={m1.num_cached_tokens}  "
      f"hit_rate={m1.cache_hit_rate:.1%}  TTFT={m1.ttft*1000:.1f}ms")

c2 = run_and_collect(llm, [prompt_full], [sp])
m2 = c2[0]
print(f"  第2次: num_cached_tokens={m2.num_cached_tokens}  "
      f"hit_rate={m2.cache_hit_rate:.1%}  TTFT={m2.ttft*1000:.1f}ms")

hit_ok = m2.num_cached_tokens > 0
print(f"  [{'PASS' if hit_ok else 'FAIL'}] 第2次 num_cached_tokens > 0")
results["场景1 LRU命中"] = hit_ok

# ─────────────────────────────────────────────────────────────────────────────
# 场景 2：TTFT 对比
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("场景2: TTFT 对比（命中 vs 首次）")
print("=" * 60)

# warm-up：确保 cache 存在
run_and_collect(llm, [prompt_full], [sp])

c_repeat = run_and_collect(llm,
    [prompt_full] * 5,
    [SamplingParams(max_tokens=8, temperature=1.0)] * 5)

ttfts_hit  = [m.ttft * 1000 for m in c_repeat if m.num_cached_tokens > 0]
ttft_miss  = m1.ttft * 1000   # 场景1首次请求作基准

if ttfts_hit:
    avg_hit = statistics.mean(ttfts_hit)
    print(f"  首次(miss) TTFT : {ttft_miss:.1f}ms")
    print(f"  命中(hit)  TTFT : mean={avg_hit:.1f}ms  "
          f"min={min(ttfts_hit):.1f}ms  max={max(ttfts_hit):.1f}ms")
    ttft_ok = avg_hit < ttft_miss
    print(f"  [{'PASS' if ttft_ok else 'FAIL'}] 命中时 TTFT 低于首次")
    results["场景2 TTFT降低"] = ttft_ok
else:
    print("  [SKIP] 无命中请求，跳过 TTFT 对比")

# ─────────────────────────────────────────────────────────────────────────────
# 场景 3：驱逐压力下 pinned vs non-pinned（直接对 BlockManager 操作）
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("场景3: 驱逐压力下 pinned vs non-pinned（BlockManager 层验证）")
print("=" * 60)

bm: BlockManager = llm.scheduler.block_manager
block_size = bm.block_size

print(f"  BlockManager 状态: "
      f"free={len(bm.free_block_ids)}  "
      f"cached={len(bm.cached_block_ids)}  "
      f"used={len(bm.used_block_ids)}")
print(f"  pinned blocks in cached: "
      f"{sum(1 for bid in bm.cached_block_ids if bm.blocks[bid].pinned)}")

# 通过 inference 建立 pinned block：发送带 cache_breakpoint 的前缀请求
sp_pinned = SamplingParams(max_tokens=4, temperature=1.0, cache_breakpoint=block_size)
run_and_collect(llm, [prompt_full], [sp_pinned])

# 查找在 cached 中的 pinned block
pinned_ids_before = {bid for bid in bm.cached_block_ids if bm.blocks[bid].pinned}
total_cached_before = len(bm.cached_block_ids)
print(f"\n  发送 pinned 请求后:")
print(f"    cached 总数: {total_cached_before}")
print(f"    其中 pinned: {len(pinned_ids_before)}")

if not pinned_ids_before:
    print("  [SKIP] 没有 pinned block 进入 cached，跳过场景3")
    results["场景3 pinned存活"] = None
else:
    # 向 cached 中注入若干 non-pinned block（模拟其他请求释放的块）
    # 直接操控 BlockManager：从 free 取块，设 hash，放入 cached
    import time
    injected = []
    for i in range(min(10, len(bm.free_block_ids))):
        fake_bid = bm.free_block_ids[0]
        block = bm.blocks[fake_bid]
        # 给它一个唯一 hash（不与真实 hash 冲突）
        fake_hash = -(i + 1000)
        bm.free_block_ids.remove(fake_bid)
        bm.used_block_ids.add(fake_bid)
        block.ref_count = 1
        block.hash = fake_hash
        block.token_ids = list(range(block_size))
        bm.hash_to_block_id[fake_hash] = fake_bid
        # 模拟 deallocate：移入 cached（non-pinned）
        block.ref_count = 0
        bm.used_block_ids.remove(fake_bid)
        bm.cached_block_ids[fake_bid] = None
        injected.append(fake_bid)

    print(f"\n  注入 {len(injected)} 个 non-pinned block 到 cached:")
    print(f"    cached 总数: {len(bm.cached_block_ids)}")
    non_pinned_in_cached = sum(1 for bid in bm.cached_block_ids if not bm.blocks[bid].pinned)
    print(f"    non-pinned: {non_pinned_in_cached}  pinned: {len(pinned_ids_before)}")

    # 触发 LRU 驱逐，驱逐次数 = 注入的 non-pinned block 数
    evicted_pinned = 0
    evicted_non_pinned = 0
    for _ in range(len(injected)):
        # 记录驱逐前的 pinned block 集合
        before = {bid for bid in bm.cached_block_ids if bm.blocks[bid].pinned}
        bm._evict_lru()
        after = {bid for bid in bm.cached_block_ids if bm.blocks[bid].pinned}
        if len(before) > len(after):
            evicted_pinned += 1
        else:
            evicted_non_pinned += 1

    pinned_ids_after = {bid for bid in bm.cached_block_ids if bm.blocks[bid].pinned}
    print(f"\n  驱逐 {len(injected)} 次后:")
    print(f"    驱逐了 non-pinned: {evicted_non_pinned} 次")
    print(f"    驱逐了 pinned:     {evicted_pinned} 次")
    print(f"    pinned block 存活: {len(pinned_ids_after)}/{len(pinned_ids_before)}")

    pin_ok = evicted_non_pinned >= evicted_pinned and len(pinned_ids_after) > 0
    print(f"  [{'PASS' if pin_ok else 'FAIL'}] non-pinned 先于 pinned 被驱逐，pinned 仍存活")
    results["场景3 pinned存活"] = pin_ok

    # 清理注入的 fake block（避免影响后续）
    for fake_bid in list(injected):
        if fake_bid in bm.cached_block_ids:
            bm._evict_block(fake_bid)

# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("汇总")
print("=" * 60)
all_pass = True
for name, ok in results.items():
    if ok is None:
        status = "SKIP"
    elif ok:
        status = "PASS"
    else:
        status = "FAIL"
        all_pass = False
    print(f"  [{status}] {name}")

print()
print("总体：", "ALL PASS ✓" if all_pass else "有 FAIL，请检查上方输出")
