from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.metrics = None  # set by LLMEngine after init

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            # num_cached_tokens是prefix cache命中的token数量
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft() # 队列头
            while not self.block_manager.can_append(seq): #不能添加，没显存了
                if self.running:
                    self.preempt(self.running.pop()) # 抢占，驱逐队列尾的prompt到waiting的front，并释放block
                else:
                    self.preempt(seq) #抢占自己，让自己回到waiting的front，并释放block
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq) # append后，当前block写满，记录hash；上个block写满，新建block
                scheduled_seqs.append(seq)
        # assert失败情况：抢占了所有running的seq，并且自己也抢占了，因此为empty，意味着就算自己独占所有block也跑不起来（超长序列）
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
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

    def postprocess_multi(self, seqs: list, tokens_per_seq: list) -> None:
        """tokens_per_seq[i] is the list of accepted tokens for seqs[i] (1 to k+1 tokens)."""
        for seq, token_list in zip(seqs, tokens_per_seq):
            for token_id in token_list:
                seq.append_token(token_id)
                if self.metrics:
                    self.metrics.on_token_generated(seq)
                if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager.deallocate(seq)
                    self.running.remove(seq)
                    if self.metrics:
                        self.metrics.on_request_finished(seq)
                    break
