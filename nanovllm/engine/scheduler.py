from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.prefill_chunk_size = config.prefill_chunk_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.metrics = None  # set by LLMEngine after init

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], list[Sequence]]:
        if self.prefill_chunk_size > 0:
            return self._schedule_chunked()
        return self._schedule_original()

    def _schedule_original(self) -> tuple[list[Sequence], list[Sequence]]:
        """原始调度逻辑，整批 prefill 优先。返回 (prefill_seqs, []) 或 ([], decode_seqs)。"""
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        # prefill
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, []
        # decode
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return [], scheduled_seqs

    def _schedule_chunked(self) -> tuple[list[Sequence], list[Sequence]]:
        """
        Chunked prefill 调度：
        - Phase 1: 尽量多地从 waiting/running(prefill未完成) 中取 seq，每个分配一个 chunk
          - chunk token 总预算 = max_num_batched_tokens
          - 每个 seq 的 chunk_len = min(num_tokens_to_prefill, prefill_chunk_size, budget_remaining)
        - Phase 2: 调度所有已完成 prefill 的 running seq 进行 decode
        """
        prefill_seqs = []
        decode_seqs = []
        chunk_token_budget = self.max_num_batched_tokens
        num_seqs = len(self.running)

        # 清零上一 step 的 chunk_len（防止 budget 耗尽时跳过的 seq 残留旧值）
        for seq in self.running:
            if not seq.is_prefill_done:
                seq.current_chunk_len = 0

        # Phase 1a: 继续正在 prefill 的 running seq（is_prefill_done=False）
        for seq in list(self.running):
            if seq.is_prefill_done:
                continue
            if chunk_token_budget <= 0:
                break
            chunk_len = min(seq.num_tokens_to_prefill, self.prefill_chunk_size, chunk_token_budget)
            seq.current_chunk_len = chunk_len
            prefill_seqs.append(seq)
            chunk_token_budget -= chunk_len

        # Phase 1b: 从 waiting 启动新 seq
        while self.waiting and chunk_token_budget > 0 and num_seqs < self.max_num_seqs:
            candidate = self.waiting[0]
            if not self.block_manager.can_allocate(candidate):
                break
            self.block_manager.allocate(candidate)
            candidate.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(candidate)
            num_seqs += 1
            chunk_len = min(candidate.num_tokens_to_prefill, self.prefill_chunk_size, chunk_token_budget)
            candidate.current_chunk_len = chunk_len
            prefill_seqs.append(candidate)
            chunk_token_budget -= chunk_len

        # Phase 2: 调度所有完成 prefill 的 running seq 进行 decode
        for seq in list(self.running):
            if not seq.is_prefill_done:
                continue
            while not self.block_manager.can_append(seq):
                if self.running:
                    victim = self.running.pop()
                    if victim is not seq:
                        # 单次抢占后 free_block_ids 必然 +1，满足 can_append 的 >=1 要求
                        self.preempt(victim)
                        break
                    else:
                        self.preempt(seq)
                        seq = None
                        break
                else:
                    self.preempt(seq)
                    seq = None
                    break
            if seq is not None and seq.is_prefill_done:
                self.block_manager.may_append(seq)
                decode_seqs.append(seq)

        assert prefill_seqs or decode_seqs, "_schedule_chunked: no sequences scheduled (memory exhausted?)"
        return prefill_seqs, decode_seqs

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.num_computed_tokens = 0  # reset partial prefill state
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)
        if self.metrics:
            self.metrics.on_preempt(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if self.metrics:
                self.metrics.on_token_generated(seq)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
                if self.metrics:
                    self.metrics.on_request_finished(seq)
