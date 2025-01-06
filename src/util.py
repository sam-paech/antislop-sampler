from typing import List, Dict, Tuple, Generator, Set, Union
from transformers import PreTrainedTokenizer

def precompute_starting_tokens(
    tokenizer: PreTrainedTokenizer, slop_phrase_prob_adjustments: Dict[str, float], ban_slop_first_tokens: bool
) -> Dict[Tuple[int, ...], Set[int]]:
    starting_tokens_lookup = {}

    for slop_phrase in slop_phrase_prob_adjustments.keys():
        starting_tokens = set()
        variants = [
            slop_phrase.lower(),
            slop_phrase.capitalize(),
            slop_phrase.upper(),
            f" {slop_phrase.lower()}",
            f" {slop_phrase.capitalize()}",
            f" {slop_phrase.upper()}",
        ]

        for variant in variants:
            token_ids = tokenizer.encode(variant, add_special_tokens=False)
            if token_ids:
                starting_tokens.add(token_ids[0])

                if not ban_slop_first_tokens:
                    first_token_decoded = tokenizer.decode(token_ids[0], skip_special_tokens=True)

                    for i in range(len(first_token_decoded) - 1):
                        prefix = first_token_decoded[:-(i + 1)]
                        if prefix == ' ':
                            continue
                        encoded_prefix = tokenizer.encode(prefix, add_special_tokens=False)
                        if encoded_prefix:
                            starting_tokens.add(encoded_prefix[0])

        starting_tokens_lookup[slop_phrase] = starting_tokens

    return starting_tokens_lookup