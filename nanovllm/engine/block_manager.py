from collections import deque, OrderedDict
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0  # 被几个sequence使用
        self.hash = -1
        self.token_ids = []
        self.pinned = False  # pinned block 最后被 LRU 淘汰

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []
        self.pinned = False


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
        # ref_count=0 且有有效 hash 的 block，按访问时间排序（末尾=最近使用），供 LRU 淘汰
        self.cached_block_ids: OrderedDict[int, None] = OrderedDict()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))  # 考虑前缀的hash
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return block

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def _evict_lru(self):
        """从 cached_block_ids 中驱逐一个 block，优先驱逐 non-pinned，再驱逐最旧的 pinned。"""
        assert self.cached_block_ids, "No cached blocks to evict"
        # 优先驱逐非 pinned block（从最旧开始）
        for block_id in self.cached_block_ids:
            if not self.blocks[block_id].pinned:
                self._evict_block(block_id)
                return
        # 全为 pinned，驱逐最旧的
        block_id = next(iter(self.cached_block_ids))
        self._evict_block(block_id)

    def _evict_block(self, block_id: int):
        block = self.blocks[block_id]
        del self.hash_to_block_id[block.hash]
        del self.cached_block_ids[block_id]
        block.pinned = False
        block.hash = -1
        block.token_ids = []
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        available = len(self.free_block_ids) + len(self.cached_block_ids)
        return available >= seq.num_blocks

    def allocate(self, seq: Sequence):
        assert not seq.block_table
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                if not self.free_block_ids:
                    self._evict_lru()
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    # 正在被其他 seq 使用，共享
                    block = self.blocks[block_id]
                    block.ref_count += 1
                elif block_id in self.cached_block_ids:
                    # 在 LRU 缓存中，取出复用
                    block = self.blocks[block_id]
                    del self.cached_block_ids[block_id]
                    self.used_block_ids.add(block_id)
                    block.ref_count = 1
                else:
                    # 不应出现（hashed block 只在 used 或 cached 中），fallback
                    if not self.free_block_ids:
                        self._evict_lru()
                    block_id = self.free_block_ids[0]
                    block = self._allocate_block(block_id)
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
                # 如果该 block 完全落在 cache_breakpoint 范围内，标记为 pinned
                if (i + 1) * self.block_size <= seq.cache_breakpoint:
                    block.pinned = True
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                if block.hash != -1:
                    # 有效缓存块：加入 LRU 队列（末尾=最近使用）
                    self.used_block_ids.remove(block_id)
                    self.cached_block_ids[block_id] = None
                else:
                    # 无效块（未写满的最后一块）：直接归还
                    self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        available = len(self.free_block_ids) + len(self.cached_block_ids)
        return available >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:  # block第一个，说明上个block满了，要新建block
            assert last_block.hash != -1
            if not self.free_block_ids:
                self._evict_lru()
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:  # block最后一个，刚好满了，计算hash供prefix cache复用
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks - 1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            assert last_block.hash == -1
