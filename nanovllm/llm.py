from nanovllm.engine.llm_engine import LLMEngine


class LLM(LLMEngine):
    pass


from nanovllm.engine.speculative_engine import SpeculativeLLMEngine


class SpeculativeLLM(SpeculativeLLMEngine):
    def generate(self, prompts, sampling_params, use_tqdm=True):
        return super().generate(prompts, sampling_params, use_tqdm)
