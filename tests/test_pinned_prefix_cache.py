"""
Pinned Prefix Cache correctness tests.

Verifications (from design doc Step 0):
  1. pinned block survives eviction pressure
  2. LRU eviction order: non-pinned before pinned, oldest before newest
  3. cache_hit_rate improves for repeated prefixes
  4. can_allocate counts free + cached blocks
  5. deallocate moves hashed blocks to cached (not free)
"""

import pytest
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


BLOCK_SIZE = 4  # 小 block 便于测试


def make_seq(token_ids, cache_breakpoint=0):
    Sequence.block_size = BLOCK_SIZE
    sp = SamplingParams(cache_breakpoint=cache_breakpoint)
    return Sequence(token_ids, sp)


def make_bm(num_blocks):
    return BlockManager(num_blocks=num_blocks, block_size=BLOCK_SIZE)


# ---------------------------------------------------------------------------
# 1. deallocate 把有 hash 的 block 移入 cached（不直接归还 free）
# ---------------------------------------------------------------------------

def test_deallocate_moves_hashed_block_to_cached():
    bm = make_bm(4)
    # 一个满 block 的 seq（block_size=4，恰好 4 token）
    seq = make_seq(list(range(BLOCK_SIZE)))
    bm.allocate(seq)
    assert len(bm.used_block_ids) == 1
    assert len(bm.free_block_ids) == 3

    bm.deallocate(seq)
    # 满块（有 hash）应进 cached，不进 free
    assert len(bm.cached_block_ids) == 1
    assert len(bm.free_block_ids) == 3   # 未满块（最后 block 若不满）此时可能回 free
    assert len(bm.used_block_ids) == 0


# ---------------------------------------------------------------------------
# 2. can_allocate 计入 cached 块
# ---------------------------------------------------------------------------

def test_can_allocate_counts_cached():
    bm = make_bm(2)
    # 先用完所有 free blocks
    seq1 = make_seq(list(range(BLOCK_SIZE * 2)))   # 2 blocks
    bm.allocate(seq1)
    assert len(bm.free_block_ids) == 0

    # 释放 seq1，2 个 hashed block 进 cached
    bm.deallocate(seq1)
    assert len(bm.cached_block_ids) == 2
    assert len(bm.free_block_ids) == 0

    # can_allocate 应该仍然返回 True
    seq2 = make_seq(list(range(BLOCK_SIZE * 2, BLOCK_SIZE * 4)))  # 不同内容
    assert bm.can_allocate(seq2)


# ---------------------------------------------------------------------------
# 3. allocate 命中 cached block（cache hit via LRU pool）
# ---------------------------------------------------------------------------

def test_cache_hit_via_cached_pool():
    bm = make_bm(4)
    tokens = list(range(BLOCK_SIZE))
    seq1 = make_seq(tokens)
    bm.allocate(seq1)
    bm.deallocate(seq1)
    assert len(bm.cached_block_ids) == 1

    # 再次用相同前缀
    seq2 = make_seq(tokens)
    bm.allocate(seq2)
    assert seq2.num_cached_tokens == BLOCK_SIZE
    # block 应从 cached 移入 used
    assert len(bm.cached_block_ids) == 0
    assert len(bm.used_block_ids) == 1


# ---------------------------------------------------------------------------
# 4. LRU 驱逐顺序：non-pinned 先于 pinned，最旧的先于最新的
# ---------------------------------------------------------------------------

def test_eviction_non_pinned_before_pinned():
    # 3 blocks：2 cached（1 pinned，1 not），1 free
    bm = make_bm(3)

    # seq_a: 1 block, pinned（cache_breakpoint 覆盖该 block）
    tokens_a = list(range(BLOCK_SIZE))
    seq_a = make_seq(tokens_a, cache_breakpoint=BLOCK_SIZE)
    bm.allocate(seq_a)
    bm.deallocate(seq_a)
    pinned_block_id = next(iter(bm.cached_block_ids))
    assert bm.blocks[pinned_block_id].pinned

    # seq_b: 1 block, non-pinned
    tokens_b = list(range(BLOCK_SIZE, BLOCK_SIZE * 2))
    seq_b = make_seq(tokens_b, cache_breakpoint=0)
    bm.allocate(seq_b)
    bm.deallocate(seq_b)
    assert len(bm.cached_block_ids) == 2

    # 触发 eviction（需要 1 个 free block，但 free 已空）
    bm._evict_lru()
    # non-pinned block 应先被驱逐，pinned 仍在
    assert len(bm.cached_block_ids) == 1
    remaining_id = next(iter(bm.cached_block_ids))
    assert bm.blocks[remaining_id].pinned


def test_eviction_lru_order_among_non_pinned():
    bm = make_bm(3)

    tokens_old = list(range(BLOCK_SIZE))
    tokens_new = list(range(BLOCK_SIZE, BLOCK_SIZE * 2))

    seq_old = make_seq(tokens_old)
    bm.allocate(seq_old)
    bm.deallocate(seq_old)   # 先进 cached（最旧）

    seq_new = make_seq(tokens_new)
    bm.allocate(seq_new)
    bm.deallocate(seq_new)   # 后进 cached（最新）

    old_block_id = list(bm.cached_block_ids.keys())[0]
    new_block_id = list(bm.cached_block_ids.keys())[1]

    bm._evict_lru()
    # 最旧的应先被驱逐
    assert old_block_id not in bm.cached_block_ids
    assert new_block_id in bm.cached_block_ids


# ---------------------------------------------------------------------------
# 5. pinned block 在显存充足时不被驱逐
# ---------------------------------------------------------------------------

def test_pinned_block_survives_when_eviction_has_alternative():
    bm = make_bm(3)

    # pinned block
    tokens_pinned = list(range(BLOCK_SIZE))
    seq_p = make_seq(tokens_pinned, cache_breakpoint=BLOCK_SIZE)
    bm.allocate(seq_p)
    bm.deallocate(seq_p)
    pinned_id = next(iter(bm.cached_block_ids))

    # non-pinned block（后加入，较新但无 pin）
    tokens_np = list(range(BLOCK_SIZE, BLOCK_SIZE * 2))
    seq_np = make_seq(tokens_np)
    bm.allocate(seq_np)
    bm.deallocate(seq_np)

    assert len(bm.cached_block_ids) == 2

    bm._evict_lru()

    # pinned block 应仍在
    assert pinned_id in bm.cached_block_ids
    assert bm.blocks[pinned_id].pinned


# ---------------------------------------------------------------------------
# 6. 当所有 cached 都是 pinned 时，才驱逐最旧的 pinned
# ---------------------------------------------------------------------------

def test_evict_pinned_when_no_non_pinned():
    bm = make_bm(3)

    tokens_a = list(range(BLOCK_SIZE))
    tokens_b = list(range(BLOCK_SIZE, BLOCK_SIZE * 2))

    seq_a = make_seq(tokens_a, cache_breakpoint=BLOCK_SIZE)
    bm.allocate(seq_a)
    bm.deallocate(seq_a)
    old_pinned_id = next(iter(bm.cached_block_ids))

    seq_b = make_seq(tokens_b, cache_breakpoint=BLOCK_SIZE)
    bm.allocate(seq_b)
    bm.deallocate(seq_b)

    assert all(bm.blocks[bid].pinned for bid in bm.cached_block_ids)

    bm._evict_lru()
    # 最旧的 pinned 被驱逐
    assert old_pinned_id not in bm.cached_block_ids
    assert len(bm.cached_block_ids) == 1


# ---------------------------------------------------------------------------
# 7. cache_hit_rate：相同前缀第二次请求命中
# ---------------------------------------------------------------------------

def test_cache_hit_rate_improves():
    bm = make_bm(8)
    prefix_len = BLOCK_SIZE * 2   # 2 满块
    tokens = list(range(prefix_len + 2))  # +2 使最后一块不满

    # 第一次请求
    seq1 = make_seq(tokens)
    bm.allocate(seq1)
    assert seq1.num_cached_tokens == 0
    bm.deallocate(seq1)

    # 第二次请求（相同 tokens）
    seq2 = make_seq(tokens)
    bm.allocate(seq2)
    assert seq2.num_cached_tokens == prefix_len  # 2 块命中
    bm.deallocate(seq2)


# ---------------------------------------------------------------------------
# 8. eviction 触发：free 耗尽时从 cached 驱逐
# ---------------------------------------------------------------------------

def test_eviction_triggered_on_allocation():
    num_blocks = 2
    bm = make_bm(num_blocks)

    # 填满 cached（1 block）
    tokens_a = list(range(BLOCK_SIZE))
    seq_a = make_seq(tokens_a)
    bm.allocate(seq_a)
    bm.deallocate(seq_a)
    # 此时 cached=1, free=1

    # 申请 2 blocks（超过 free，需要驱逐 cached 1 block）
    tokens_b = list(range(BLOCK_SIZE * 2, BLOCK_SIZE * 4))
    seq_b = make_seq(tokens_b)
    assert bm.can_allocate(seq_b)
    bm.allocate(seq_b)
    assert len(bm.used_block_ids) == 2
    assert len(bm.cached_block_ids) == 0
