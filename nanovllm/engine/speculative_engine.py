"""
SpeculativeLLMEngine: subclass of LLMEngine, overrides step() for speculative decode.
Prefill: same as normal (target model), then seeds draft KV cache.
Decode: executes 3-phase speculative decode.
"""
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.speculative_decoder import SpeculativeDecoder


class SpeculativeLLMEngine(LLMEngine):

    def __init__(self, target_model: str, draft_model: str,
                 num_speculative_tokens: int = 4, **kwargs):
        # Force eager mode to avoid two ModelRunners competing for CUDA Graph memory
        kwargs["enforce_eager"] = True
        # Lower gpu_memory_utilization default: target takes less KV cache to leave room for draft
        kwargs.setdefault("gpu_memory_utilization", 0.7)
        # Smaller warmup batch reduces peak activation memory, freeing more room for draft KV
        kwargs.setdefault("max_num_batched_tokens", 4096)
        super().__init__(target_model, **kwargs)
        self.k = num_speculative_tokens
        try:
            self.spec_decoder = SpeculativeDecoder(draft_model, self.model_runner.config)
        except Exception:
            self.exit()  # release target GPU memory if draft init fails
            raise

    def exit(self):
        if hasattr(self, "spec_decoder") and hasattr(self.spec_decoder, "draft_runner"):
            self.spec_decoder.draft_runner.exit()
        super().exit()

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()

        if is_prefill:
            # Standard target prefill
            token_ids = self.model_runner.call("run", seqs, is_prefill)
            if self.metrics:
                for seq in seqs:
                    self.metrics.on_prefill_done(seq)
            self.scheduler.postprocess(seqs, token_ids)
            # Seed draft KV cache
            self.spec_decoder.prefill_draft(seqs)
            outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
            num_tokens = sum(len(seq) for seq in seqs)
        else:
            # Pre-allocate k extra block slots in target's block manager
            self.spec_decoder.pre_allocate_target_blocks(
                seqs, self.k, self.scheduler.block_manager
            )
            # 3 phases
            draft_results = self.spec_decoder._draft(seqs, self.k)
            target_logits = self.spec_decoder._verify(seqs, self.model_runner, draft_results, self.k)
            tokens_per_seq = self.spec_decoder._reject(seqs, draft_results, target_logits, self.k)

            # Update target seqs
            self.scheduler.postprocess_multi(seqs, tokens_per_seq)

            # Sync draft seqs
            self.spec_decoder.sync_after_rejection(seqs, tokens_per_seq, draft_results)

            outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
            total_accepted = sum(len(t) for t in tokens_per_seq)
            num_tokens = -total_accepted  # negative = decode (matches LLMEngine convention)

        return outputs, num_tokens
