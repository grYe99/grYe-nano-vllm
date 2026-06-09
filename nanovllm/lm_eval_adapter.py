"""lm-eval adapter for nano-vllm.

Usage:
    from nanovllm.lm_eval_adapter import NanoVLLM

    model = NanoVLLM("~/huggingface/Qwen3-0.6B/")
    import lm_eval
    results = lm_eval.simple_evaluate(
        model=model, tasks=["hellaswag"], num_fewshot=0, limit=10,
    )
"""

from lm_eval.api.model import LM
from lm_eval.api.instance import Instance

from nanovllm import LLM, SamplingParams


class NanoVLLM(LM):
    """lm-eval compatible model adapter wrapping nano-vllm's LLM."""

    def __init__(self, model_path: str, **kwargs):
        super().__init__()
        self._device = "cuda"
        self.llm = LLM(model_path, **kwargs)
        self.tokenizer = self.llm.tokenizer

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def max_length(self) -> int:
        return self.llm.config.max_model_len

    @property
    def batch_size(self) -> int:
        return self.llm.config.max_num_seqs

    def tok_encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def tok_decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        """Compute loglikelihood for each (context, continuation) pair."""
        prompts = []
        for req in requests:
            ctx_str, cont_str = req.args
            ctx_tokens = self.tok_encode(ctx_str)
            cont_tokens = self.tok_encode(cont_str)
            prompts.append((ctx_tokens, cont_tokens))
        return self.llm.compute_logprobs(prompts)

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[float]:
        """Rolling loglikelihood (not implemented)."""
        raise NotImplementedError(
            "loglikelihood_rolling is not supported by nano-vllm"
        )

    def generate_until(self, requests: list[Instance]) -> list[str]:
        """Generate text for each (context, gen_kwargs) pair."""
        results = []
        for req in requests:
            context, gen_kwargs = req.args
            max_gen_toks = gen_kwargs.get("max_gen_toks", 256)
            temperature = gen_kwargs.get("temperature", 0.0)
            until = gen_kwargs.get("until", [])

            sp = SamplingParams(
                temperature=max(temperature, 0.0),
                max_tokens=max_gen_toks,
                ignore_eos=True,
            )

            # Encode context without special tokens and generate
            ctx_tokens = self.tok_encode(context)
            output = self.llm.generate(
                [ctx_tokens], sp, use_tqdm=False
            )
            text = output[0]["text"]

            # Strip the first matching stop sequence
            min_idx = len(text)
            for stop in until:
                if stop:
                    idx = text.find(stop)
                    if idx != -1 and idx < min_idx:
                        min_idx = idx
            text = text[:min_idx]

            results.append(text)

        return results
