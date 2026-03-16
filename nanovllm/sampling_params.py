from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
    cache_breakpoint: int = 0  # pin blocks covering the first N prompt tokens against LRU eviction

    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
