# File: ai/antislop-sampler/src/validator_regex.py

import re
import time
import torch
from transformers import StoppingCriteria
from IPython.display import clear_output, display, HTML
from typing import List, Dict, Tuple, Optional

# This implements banning of sequences using regex matching.
# If the inference text matches one of the specified regex expressions, 
# we backtrack to the position of the first match, ban that token, then
# continue inference.
class RegexValidator:
    def __init__(self, tokenizer, regex_bans: List[str], slow_debug, debug_delay, debug_output, probs_cache_longrange):
        self.tokenizer = tokenizer
        self.regex_bans = [re.compile(pattern) for pattern in regex_bans]
        self.slow_debug = slow_debug
        self.debug_delay = debug_delay
        self.debug_output = debug_output
        self.probs_cache_longrange = probs_cache_longrange

        self.last_downregulated_token = 0

    def validate_regex_matches(self, generated_sequence, prompt_length, probs_cache):
        result = self._validate_regex_matches(generated_sequence, prompt_length, probs_cache)        
        if result is not False:
            generated_sequence, problematic_token, invalid_match, reason = result
            if self.slow_debug:
                problematic_token_decoded = self.tokenizer.decode(problematic_token, skip_special_tokens=True)
                debug_info = f"Regex violation detected:\n{reason}\nProblematic match: '{invalid_match}' in token '{problematic_token_decoded}'"
                self._display_debug(debug_info)
                time.sleep(self.debug_delay)
            
            problematic_pos = len(generated_sequence)
            
            # Clear subsequent logit cache
            to_del = [key for key in probs_cache if key > problematic_pos]
            for key in to_del:
                del probs_cache[key]
            
            # Flag positions to keep in the logit cache
            self.probs_cache_longrange[problematic_pos] = True
            
            return generated_sequence

        return False

    def _validate_regex_matches(self, generated_sequence, prompt_length, probs_cache=None, validate_only=False):
        # Decode only the newly generated tokens
        generated_text = self.tokenizer.decode(generated_sequence[prompt_length:], skip_special_tokens=True)

        for pattern in self.regex_bans:
            match = pattern.search(generated_text)
            if match:
                # Find the position in tokens where the match starts
                match_start_char_pos = match.start()

                # We prepend a char because some tokenisers won't add the initial space
                # for the first token which would otherwise be there if there was a token
                # in front of it. If we don't have this we'll sometimes match to the wrong
                # token position.
                prepended_char = self.tokenizer.encode('|', add_special_tokens=False)
                problematic_token_pos = None

                for start_pos in range(len(generated_sequence)-1, prompt_length-1, -1):
                    test_str = self.tokenizer.decode(prepended_char + generated_sequence[start_pos:], skip_special_tokens=True)

                    if  len(generated_text) - (len(test_str) - 1) <= match_start_char_pos: # -1 is to account for the prepended char
                        problematic_token_pos = start_pos                        
                        break
                if problematic_token_pos == None:
                    print('!! failed to get problematic token pos')
                    return False

                problematic_token = generated_sequence[problematic_token_pos]
                reason = f"Regex pattern '{pattern.pattern}' matched."
                
                if validate_only:
                    return True

                # Adjust the generated sequence and cache
                generated_sequence, problematic_token = self._backtrack_and_adjust(generated_sequence, problematic_token_pos, upregulate_safe_continuation=False, probs_cache=probs_cache)
                return generated_sequence, problematic_token, match.group(), reason

        return False

    def _create_char_to_token_mapping(self, tokens, prompt_length):
        char_to_token_pos = {}
        char_pos = 0
        for token_pos, token in enumerate(tokens):
            token_text = self.tokenizer.decode([token], skip_special_tokens=True)
            for _ in range(len(token_text)):
                char_to_token_pos[char_pos] = token_pos + prompt_length
                char_pos += 1
        return char_to_token_pos

    def _backtrack_and_adjust(self, generated_sequence, problematic_pos, upregulate_safe_continuation=False, probs_cache=None):
        # Identify the problematic token and backtrack
        problematic_token = generated_sequence[problematic_pos]

        # Downregulate the problematic token        
        self.last_downregulated_token = problematic_token
        probs_cache[problematic_pos][:, problematic_token] *= 0.0001

        # Flag positions to keep in the logit cache
        self.probs_cache_longrange[problematic_pos] = True

        # Clear the probs_cache ahead of start_pos since we've backtracked
        to_del = [key for key in probs_cache if key > problematic_pos]
        for key in to_del:
            del probs_cache[key]

        return generated_sequence[:problematic_pos], problematic_token
    
        

    def _display_debug(self, message: str):
        """
        Displays debug information in the debug_output widget.
        """
        if self.debug_output:
            with self.debug_output:
                self.debug_output.clear_output(wait=True)
                display(HTML(f"<pre>{message}</pre>"))

class RegexValidationStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, regex_validator, prompt_length):
        self.tokenizer = tokenizer
        self.regex_validator = regex_validator
        self.prompt_length = prompt_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        previous_tokens = input_ids[0].tolist()
        
        # Check if any regex pattern matches the generated sequence
        result = self.regex_validator._validate_regex_matches(previous_tokens, self.prompt_length, validate_only=True)

        return result
