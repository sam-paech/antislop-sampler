# File: ai/antislop-sampler/src/logits_processor_slop.py

import torch
from transformers import LogitsProcessor
from typing import Set

class SlopTokenBanLogitsProcessor(LogitsProcessor):
    """
    A LogitsProcessor that sets banned token logits to -inf, so they won't be sampled.
    """
    def __init__(self, banned_token_ids: Set[int]):
        super().__init__()
        self.banned_token_ids = banned_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if len(self.banned_token_ids) == 0:
            return scores
        
        # For each batch item, set banned token logits to -inf.
        for b in range(scores.shape[0]):
            scores[b, list(self.banned_token_ids)] = float('-inf')

        return scores
