import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        # 这里可以实现top-k，top-p，过滤长尾词

        # 随机采样，torch.empty_like(probs).exponential_(1)为torch提供的操作，vllm里有seed可控制产生相同随机数序列，以进行对比实验
        # Gumbel-max trick（exponential_ + div_ + argmax）
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
