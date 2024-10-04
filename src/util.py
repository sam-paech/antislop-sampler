from typing import List, Dict, Tuple, Generator, Set, Union
from transformers import PreTrainedTokenizer

def precompute_starting_tokens(
    tokenizer: PreTrainedTokenizer, slop_phrase_prob_adjustments: Dict[str, float]
) -> Dict[Tuple[int, ...], Set[int]]:
    starting_tokens_lookup = {}

    for word in slop_phrase_prob_adjustments.keys():
        variants = [
            word.lower(),
            word.capitalize(),
            word.upper(),
            f" {word.lower()}",
            f" {word.capitalize()}",
            f" {word.upper()}",
        ]

        for variant in variants:
            token_ids = tokenizer.encode(variant, add_special_tokens=False)
            starting_tokens = set()
            if token_ids:
                starting_tokens.add(token_ids[0])
                first_token_decoded = tokenizer.decode(token_ids[0], skip_special_tokens=True)

                for i in range(len(first_token_decoded) - 1):
                    prefix = first_token_decoded[:-(i + 1)]
                    if prefix == ' ':
                        continue
                    encoded_prefix = tokenizer.encode(prefix, add_special_tokens=False)
                    if encoded_prefix:
                        starting_tokens.add(encoded_prefix[0])

                starting_tokens_lookup[tuple(token_ids)] = starting_tokens

    return starting_tokens_lookup