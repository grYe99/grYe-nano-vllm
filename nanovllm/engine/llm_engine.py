import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.utils.metrics import MetricsCollector


class LLMEngine:

    def __init__(self, model, **kwargs):
        self.metrics = kwargs.pop("metrics", None)
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        self.scheduler.metrics = self.metrics
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        if self.metrics:
            self.metrics.on_request_added(seq)
        self.scheduler.add(seq)

    def step(self):
        prefill_seqs, decode_seqs = self.scheduler.schedule()

        if prefill_seqs and not decode_seqs:
            if self.scheduler.prefill_chunk_size > 0:
                # chunked mode: must use run_mixed (even with empty decode list) so that
                # prepare_chunked_prefill correctly handles current_chunk_len / num_computed_tokens
                token_ids = self.model_runner.call("run_mixed", prefill_seqs, decode_seqs)
                outputs = []
                for i, seq in enumerate(prefill_seqs):
                    seq.num_computed_tokens += seq.current_chunk_len
                    if seq.is_prefill_done:
                        if self.metrics:
                            self.metrics.on_prefill_done(seq)
                        self.scheduler.postprocess([seq], [token_ids[i]])
                        if seq.is_finished:
                            outputs.append((seq.seq_id, seq.completion_token_ids))
                num_tokens = sum(seq.current_chunk_len for seq in prefill_seqs)
                return outputs, num_tokens
            else:
                # non-chunked: original whole-prefill path
                token_ids = self.model_runner.call("run", prefill_seqs, True)
                if self.metrics:
                    for seq in prefill_seqs:
                        self.metrics.on_prefill_done(seq)
                self.scheduler.postprocess(prefill_seqs, token_ids)
                outputs = [(seq.seq_id, seq.completion_token_ids) for seq in prefill_seqs if seq.is_finished]
                num_tokens = sum(len(seq) - seq.num_cached_tokens for seq in prefill_seqs)
                return outputs, num_tokens

        elif not prefill_seqs and decode_seqs:
            # 纯 decode
            token_ids = self.model_runner.call("run", decode_seqs, False)
            self.scheduler.postprocess(decode_seqs, token_ids)
            outputs = [(seq.seq_id, seq.completion_token_ids) for seq in decode_seqs if seq.is_finished]
            num_tokens = -len(decode_seqs)
            return outputs, num_tokens

        else:
            # 混合批次（chunked prefill chunk + decode）
            token_ids = self.model_runner.call("run_mixed", prefill_seqs, decode_seqs)
            outputs = []

            # 处理 prefill seqs：更新 num_computed_tokens，只在最后一个 chunk 时 postprocess
            for i, seq in enumerate(prefill_seqs):
                seq.num_computed_tokens += seq.current_chunk_len
                if seq.is_prefill_done:
                    # 最后一个 chunk 完成，token_ids[i] 是该 seq 的第一个 completion token
                    if self.metrics:
                        self.metrics.on_prefill_done(seq)
                    self.scheduler.postprocess([seq], [token_ids[i]])
                    if seq.is_finished:
                        outputs.append((seq.seq_id, seq.completion_token_ids))

            # 处理 decode seqs
            decode_token_ids = token_ids[len(prefill_seqs):]
            self.scheduler.postprocess(decode_seqs, decode_token_ids)
            for seq in decode_seqs:
                if seq.is_finished:
                    outputs.append((seq.seq_id, seq.completion_token_ids))

            # 混合步骤：num_tokens 记为 -len(decode_seqs)（主要贡献是 decode throughput）
            num_tokens = -len(decode_seqs)
            return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        if self.metrics:
            self.metrics.on_generate_start()
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                elif num_tokens < 0:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                # num_tokens == 0: 中间 chunk 步，不更新吞吐量
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())] # map -> list
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        if self.metrics:
            self.metrics.on_generate_end()
            self.metrics.print_summary()
        return outputs
