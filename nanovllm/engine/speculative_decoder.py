"""
SpeculativeDecoder: holds draft ModelRunner + draft BlockManager,
implements the 3-phase speculative decoding logic:
  1. _draft():   draft model auto-regressively generates k candidate tokens
  2. _verify():  target model parallel-verifies k+1 tokens (manually built inputs)
  3. _reject():  rejection sampling, returns accepted token list per seq
"""
import torch
from nanovllm.config import Config
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.sequence import Sequence
from nanovllm.utils.context import set_context, reset_context


class SpeculativeDecoder:

    def __init__(self, draft_model: str, target_config: Config):
        draft_config = Config(
            draft_model,
            enforce_eager=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_num_batched_tokens=target_config.max_num_batched_tokens,
            max_num_seqs=target_config.max_num_seqs,
            max_model_len=target_config.max_model_len,
        )
        self.draft_runner = ModelRunner(
            draft_config, rank=0, event=[],
            store_name="nanovllm_draft_store"
        )
        self.draft_block_manager = BlockManager(
            draft_config.num_kvcache_blocks,
            draft_config.kvcache_block_size
        )
        self.block_size = draft_config.kvcache_block_size
        self.draft_seqs: dict[int, Sequence] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def prefill_draft(self, seqs: list[Sequence]) -> None:
        """Call after target prefill to initialize draft KV cache for these seqs."""
        new_draft_seqs = []
        for seq in seqs:
            draft_seq = Sequence.__new__(Sequence)
            draft_seq.seq_id = seq.seq_id
            draft_seq.status = seq.status
            draft_seq.token_ids = seq.token_ids      # shared reference
            draft_seq.last_token = seq.last_token
            draft_seq.num_tokens = seq.num_tokens
            draft_seq.num_prompt_tokens = seq.num_prompt_tokens
            draft_seq.num_cached_tokens = 0
            draft_seq.block_table = []
            draft_seq.temperature = seq.temperature
            draft_seq.max_tokens = seq.max_tokens
            draft_seq.ignore_eos = seq.ignore_eos
            self.draft_block_manager.allocate(draft_seq)
            self.draft_seqs[seq.seq_id] = draft_seq
            new_draft_seqs.append(draft_seq)
        # Run draft prefill to populate KV cache (discard sampled tokens)
        self.draft_runner.run(new_draft_seqs, is_prefill=True)

    def cleanup(self, seq: Sequence) -> None:
        """Call when seq finishes to free draft-side blocks."""
        if seq.seq_id in self.draft_seqs:
            self.draft_block_manager.deallocate(self.draft_seqs[seq.seq_id])
            del self.draft_seqs[seq.seq_id]

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def step(self, seqs: list[Sequence], target_runner: ModelRunner, k: int) -> list[list[int]]:
        draft_results = self._draft(seqs, k)
        target_logits_per_seq = self._verify(seqs, target_runner, draft_results, k)
        tokens_per_seq = self._reject(seqs, draft_results, target_logits_per_seq, k)
        return tokens_per_seq

    def sync_after_rejection(self, seqs: list[Sequence], tokens_per_seq: list[list[int]],
                              draft_results: list[list[tuple]]) -> None:
        """Rollback draft seqs to pre-draft state, then append accepted tokens."""
        for seq, accepted_tokens, dr in zip(seqs, tokens_per_seq, draft_results):
            if seq.seq_id not in self.draft_seqs:
                continue
            draft_seq = self.draft_seqs[seq.seq_id]
            k_actual = len(dr)

            # Rollback: draft seq's num_tokens was advanced by k_actual during _draft()
            pre_draft_num_tokens = draft_seq.num_tokens - k_actual
            pre_draft_num_blocks = (pre_draft_num_tokens + self.block_size - 1) // self.block_size
            # Free any extra blocks allocated during draft
            while len(draft_seq.block_table) > pre_draft_num_blocks:
                freed = draft_seq.block_table.pop()
                self.draft_block_manager.free_blocks.append(freed)
            # Roll back token count (token_ids is shared ref, don't modify it)
            draft_seq.num_tokens = pre_draft_num_tokens
            draft_seq.last_token = draft_seq.token_ids[pre_draft_num_tokens - 1]

            # Append accepted tokens
            for tok in accepted_tokens:
                draft_seq.last_token = tok
                draft_seq.num_tokens += 1
                if self.draft_block_manager.can_append(draft_seq):
                    self.draft_block_manager.may_append(draft_seq)

            if seq.is_finished:
                self.cleanup(seq)

    # ------------------------------------------------------------------
    # 3 phases
    # ------------------------------------------------------------------

    def _draft(self, seqs: list[Sequence], k: int) -> list[list[tuple]]:
        """
        Run draft model k autoregressive steps.
        Returns results[i] = [(token_id, p_draft_scalar, probs_vector), ...] length k.
        IMPORTANT: only updates ds.num_tokens and ds.last_token, NOT ds.token_ids
        (token_ids is shared with target seq which hasn't accepted these tokens yet).
        """
        results: list[list[tuple]] = [[] for _ in seqs]
        draft_seq_list = [self.draft_seqs[s.seq_id] for s in seqs]

        for step in range(k):
            token_ids, probs = self.draft_runner.run_with_probs(draft_seq_list, is_prefill=False)
            for i, (ds, tid) in enumerate(zip(draft_seq_list, token_ids)):
                p_scalar = probs[i, tid].item()
                results[i].append((tid, p_scalar, probs[i].clone()))
                # Only update position tracking, NOT token_ids list
                ds.last_token = tid
                ds.num_tokens += 1
                if self.draft_block_manager.can_append(ds):
                    self.draft_block_manager.may_append(ds)

        return results

    def _verify(self, seqs: list[Sequence], target_runner: ModelRunner,
                draft_results: list[list[tuple]], k: int) -> list[torch.Tensor]:
        """
        Run target model on k+1 new tokens per seq (last accepted + k drafts).
        Manually build varlen prefill inputs. Returns logits_per_seq[i]: [k+1, vocab].
        """
        input_ids_list = []
        positions_list = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        slot_mapping_list = []
        block_tables_list = []
        seqlen_q = k + 1
        max_seqlen_q = seqlen_q
        max_seqlen_k = 0

        bs = target_runner.block_size

        for seq, dr in zip(seqs, draft_results):
            orig_len = seq.num_tokens  # before any new tokens appended
            # k+1 input tokens: last accepted token + k draft tokens
            new_token_ids = [seq.last_token] + [t for t, _, _ in dr]
            start_pos = orig_len - 1
            new_positions = list(range(start_pos, start_pos + k + 1))

            input_ids_list.extend(new_token_ids)
            positions_list.extend(new_positions)

            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            seqlen_k = orig_len + k  # full context length for KV
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_k = max(max_seqlen_k, seqlen_k)

            # slot_mapping: physical slots for each of the k+1 new positions
            for pos_offset in range(k + 1):
                abs_pos = start_pos + pos_offset
                blk_idx = abs_pos // bs
                blk_offset = abs_pos % bs
                if blk_idx >= len(seq.block_table):
                    raise RuntimeError(
                        f"block_table too short for seq {seq.seq_id}: "
                        f"blk_idx={blk_idx}, len={len(seq.block_table)}. "
                        f"Call pre_allocate_target_blocks before _verify."
                    )
                phys_slot = seq.block_table[blk_idx] * bs + blk_offset
                slot_mapping_list.append(phys_slot)

            block_tables_list.append(list(seq.block_table))

        # Pad block_tables to same length
        max_bt_len = max(len(bt) for bt in block_tables_list)
        block_tables_padded = [bt + [-1] * (max_bt_len - len(bt)) for bt in block_tables_list]

        input_ids = torch.tensor(input_ids_list, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions  = torch.tensor(positions_list, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q_t = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k_t = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping_t  = torch.tensor(slot_mapping_list, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables_t  = torch.tensor(block_tables_padded, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        set_context(True, cu_seqlens_q_t, cu_seqlens_k_t,
                    max_seqlen_q, max_seqlen_k,
                    slot_mapping_t, None, block_tables_t)

        with torch.inference_mode():
            # Get ALL token hidden states (not just last per seq as compute_logits would do in prefill mode)
            import torch.nn.functional as F
            hidden_states = target_runner.model(input_ids, positions)  # [total_q_tokens, hidden_size]
            logits = F.linear(hidden_states, target_runner.model.lm_head.weight)  # [total_q_tokens, vocab]
        reset_context()

        # Split back to per-seq [k+1, vocab]
        logits_per_seq = []
        offset = 0
        for _ in seqs:
            logits_per_seq.append(logits[offset: offset + seqlen_q])
            offset += seqlen_q

        return logits_per_seq

    def _reject(self, seqs: list[Sequence], draft_results: list[list[tuple]],
                target_logits_per_seq: list[torch.Tensor], k: int) -> list[list[int]]:
        """
        Rejection sampling. Returns 1 to k+1 accepted tokens per seq.
        Guarantees output distribution == pure target sampling.
        """
        all_accepted = []
        for seq, dr, target_logits in zip(seqs, draft_results, target_logits_per_seq):
            temperature = max(seq.temperature, 1e-7)
            target_probs = torch.softmax(target_logits.float() / temperature, dim=-1)  # [k+1, vocab]

            accepted = []
            all_accepted_flag = True
            for i, (draft_tok, p_draft, prob_draft_vec) in enumerate(dr):
                p_target = target_probs[i, draft_tok].item()
                r = torch.rand(1).item()
                if p_draft > 0 and r <= p_target / p_draft:
                    accepted.append(draft_tok)
                else:
                    # Correction: sample from (p_target - p_draft).clamp(0)
                    correction = (target_probs[i] - prob_draft_vec.to(target_probs.device)).clamp(min=0)
                    s = correction.sum()
                    if s > 1e-9:
                        correction = correction / s
                        correction_tok = torch.multinomial(correction, 1).item()
                    else:
                        # Fallback: argmax of target (handles temperature≈0 case)
                        correction_tok = target_probs[i].argmax().item()
                    accepted.append(correction_tok)
                    all_accepted_flag = False
                    break

            if all_accepted_flag:
                # Bonus token from target_probs[k] (the position after all k drafts)
                bonus_probs = target_probs[k]
                bonus_tok = torch.multinomial(bonus_probs, 1).item()
                accepted.append(bonus_tok)

            all_accepted.append(accepted)

        return all_accepted

    def pre_allocate_target_blocks(self, seqs: list[Sequence], k: int,
                                    target_block_manager: BlockManager) -> None:
        """
        Pre-allocate k additional block slots in target's block manager per seq.
        schedule() already called may_append once; we need k more for k+1 total new tokens.
        """
        for seq in seqs:
            for _ in range(k):
                if target_block_manager.can_append(seq):
                    target_block_manager.may_append(seq)
