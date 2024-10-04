import time
import torch
from transformers import StoppingCriteria
from IPython.display import clear_output, display, HTML

class JSONValidator:
    def __init__(self, tokenizer, slow_debug, debug_delay, debug_output, probs_cache_longrange):
        self.tokenizer = tokenizer
        self.slow_debug = slow_debug
        self.debug_delay = debug_delay
        self.debug_output = debug_output
        self.probs_cache_longrange = probs_cache_longrange
        
        # Escaped tokens lookup for common characters that need escaping in JSON strings
        self.escaped_tokens_lookup = {
            '\n': self.tokenizer.encode('\\n', add_special_tokens=False),
            '\t': self.tokenizer.encode('\\t', add_special_tokens=False),
            '\r': self.tokenizer.encode('\\r', add_special_tokens=False),
            '"': self.tokenizer.encode('\\"', add_special_tokens=False),
            ' "': self.tokenizer.encode(' \\"', add_special_tokens=False),
        }

        self.last_downregulated_token = 0

    def validate_json_string(self, generated_sequence, prompt_length, probs_cache):
        result = self._validate_json_string(generated_sequence, prompt_length, probs_cache)        
        if result is not False:
            generated_sequence, problematic_token, invalid_char, reason = result
            if self.slow_debug:
                problematic_token_decoded = self.tokenizer.decode(problematic_token, skip_special_tokens=True)
                debug_info = f"JSON structure violation detected:\n{reason}\nProblematic char: {[invalid_char]} in token {[problematic_token_decoded]}"
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

    # String validation checks for unescaped chars inside a (partial) json generation.
    # It works better to do this with with long-range constraints (at least for double
    # quotes) because we aren't forcing a termination of the string if that's not what
    # the model intended.
    #
    # To do this, if we are in a json string and see a ", we wait for more tokens to
    # see if the continuation looks like it was intended to be a string terminator,
    # or meant to be a quotation mark within the string. If it looks like it was an
    # accidentally unescaped quote, we downregulate this token and upregulate the
    # escaped token at that position (then backtrack & resample)
    #
    # It's not foolproof, but it does fix most of the json parsing fails that occur.
    def _validate_json_string(self, generated_sequence, prompt_length, probs_cache=None, validate_only=False):
        # Get only the generated text
        generated_text = self.tokenizer.decode(generated_sequence[prompt_length:])

        # Create character to token position mapping
        char_to_token_pos = self._create_char_to_token_mapping(generated_sequence[prompt_length:], prompt_length)

        in_json = False
        in_string = False
        escape_next = False
        validating_string_end = False
        string_end_char = -1
        
        # Tracks the nesting of JSON objects and arrays
        brace_stack = []
        expected_tokens = []

        for i, char in enumerate(generated_text):
            if not in_json:
                if char in '{[':
                    in_json = True
                    # Start tracking the JSON structure
                    brace_stack.append(char)
                    expected_tokens = self._get_expected_tokens(brace_stack)
                continue

            if escape_next:
                escape_next = False
                continue

            if validating_string_end:
                if char in ['\n', '\t', '\r', ' ']:
                    continue
                elif char in ',:]}[{':
                    # Valid string termination
                    in_string = False
                    validating_string_end = False
                    expected_tokens = self._get_expected_tokens(brace_stack)
                else:
                    # Invalid string termination, backtrack
                    if validate_only:
                        return True

                    generated_sequence, problematic_token = self._backtrack_and_adjust(generated_sequence, char_to_token_pos[string_end_char], upregulate_safe_continuation=True, probs_cache=probs_cache)
                    return generated_sequence, problematic_token, char, "Unexpected string termination"

            if char == '\\':
                escape_next = True
            elif char == '"':
                if in_string:
                    # End of string, validate its termination
                    string_end_char = i
                    validating_string_end = True
                else:
                    # Start of string
                    in_string = True
                    expected_tokens = ['"']
            elif in_string:
                if char in '\n\r\t':
                    # These characters should be escaped in JSON strings
                    if validate_only:
                        return True
                    generated_sequence, problematic_token = self._backtrack_and_adjust(generated_sequence, char_to_token_pos[i], upregulate_safe_continuation=True, probs_cache=probs_cache)
                    return generated_sequence, problematic_token, char, "Unterminated character in string"
                
            elif char in '{[':
                brace_stack.append(char)
                expected_tokens = self._get_expected_tokens(brace_stack)
            elif char in '}]':
                if not brace_stack or (char == '}' and brace_stack[-1] != '{') or (char == ']' and brace_stack[-1] != '['):
                    if validate_only:
                        return True
                    generated_sequence, problematic_token = self._backtrack_and_adjust(generated_sequence, char_to_token_pos[i], probs_cache=probs_cache)
                    return generated_sequence, problematic_token, char, "Unexpected '}' or ']'"
                brace_stack.pop()
                expected_tokens = self._get_expected_tokens(brace_stack)
            elif char == ',':
                if not brace_stack or brace_stack[-1] not in '{[':
                    if validate_only:
                        return True
                    generated_sequence, problematic_token = self._backtrack_and_adjust(generated_sequence, char_to_token_pos[i], probs_cache=probs_cache)                    
                    return generated_sequence, problematic_token, char, "Unexpected ','"
                expected_tokens = self._get_expected_tokens(brace_stack)
            elif char == ':':
                if not brace_stack or brace_stack[-1] != '{':
                    if validate_only:
                        return True
                    generated_sequence, problematic_token = self._backtrack_and_adjust(generated_sequence, char_to_token_pos[i], probs_cache=probs_cache)
                    return generated_sequence, problematic_token, char, "Unexpected ':'"
                expected_tokens = self._get_expected_tokens(brace_stack)
            elif char not in ' \n\t\r':
                if char not in expected_tokens:
                    if validate_only:
                        return True
                    generated_sequence, problematic_token = self._backtrack_and_adjust(generated_sequence, char_to_token_pos[i], probs_cache=probs_cache)
                    return generated_sequence, problematic_token, char, "Unexpected char (expecting whitespace)"

            # Check if we have closed all JSON structures
            if not brace_stack:
                in_json = False

        return False

    def _get_expected_tokens(self, brace_stack):
        # Determine expected tokens based on current JSON structure
        if not brace_stack:
            return ['{', '[']
        elif brace_stack[-1] == '{':
            return ['"', '}', ' ', '\n', '\t', '\r']
        elif brace_stack[-1] == '[':
            return ['{', '[', ']', '"', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 't', 'f', 'n', ' ', '\n', '\t', '\r']
        else:
            return [',', '}', ']', ' ', '\n', '\t', '\r']

    def _create_char_to_token_mapping(self, tokens, prompt_length):
        char_to_token_pos = {}
        char_pos = 0
        for token_pos, token in enumerate(tokens):
            token_text = self.tokenizer.decode(token)
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

        # Upregulate the properly escaped version
        if upregulate_safe_continuation:
            self._upregulate_safe_continuation(problematic_pos, problematic_token, probs_cache)
        
        return generated_sequence[:problematic_pos], problematic_token

    # We need to account for cases where the problematic token is more than 1 char
    # let's take this approach:
    # - first, check if the token is in the lookup
    #   - if so, we upregulate the escaped lookup value
    # - if not, we check if the 1st char of the decoded token is on of our problematic tokens
    #   - if so, upregulate just that (escaped) token
    # - else, we extract the substring up to (and not including) the first problematic char
    #   - then upregulate the first token of this encoded string, as the intended continuation
    def _upregulate_safe_continuation(self, problematic_pos, problematic_token, probs_cache):
        # this function is specifically to handle unescaped chars inside strings

        problematic_token_decoded = self.tokenizer.decode(problematic_token, skip_special_tokens=True)

        debug=False
        
        # Check if the token is in the lookup
        if problematic_token_decoded in self.escaped_tokens_lookup:
            if debug:
                print('upregulating1', [self.tokenizer.decode(self.escaped_tokens_lookup[problematic_token_decoded][0])])
            probs_cache[problematic_pos][:, self.escaped_tokens_lookup[problematic_token_decoded][0]] *= 2
            # normalise probs
            probs_cache[problematic_pos].div_(torch.sum(probs_cache[problematic_pos]))

        elif problematic_token_decoded[0] in self.escaped_tokens_lookup:            
            encoded_escaped_tok = self.escaped_tokens_lookup[problematic_token_decoded[0]][0]
            if debug:
                print('upregulating2', [self.tokenizer.decode(encoded_escaped_tok)])

            probs_cache[problematic_pos][:, encoded_escaped_tok] *= 2
            # normalise probs
            probs_cache[problematic_pos].div_(torch.sum(probs_cache[problematic_pos]))
        else:
            # Find the first problematic character
            first_problematic_index = next((i for i, char in enumerate(problematic_token_decoded) if char in self.escaped_tokens_lookup), None)
            
            if first_problematic_index is not None:
                # Extract the substring up to the first problematic character
                safe_substring = problematic_token_decoded[:first_problematic_index]
                
                # Encode the safe substring
                encoded_safe_substring = self.tokenizer.encode(safe_substring)
                
                # Upregulate the first token of the encoded safe substring
                if encoded_safe_substring:
                    if debug:
                        print('upregulating3', [safe_substring])
                    probs_cache[problematic_pos][:, encoded_safe_substring[0]] *= 2
                    # normalise probs
                    probs_cache[problematic_pos].div_(torch.sum(probs_cache[problematic_pos]))


    def _display_debug(self, message: str):
        """
        Displays debug information in the debug_output widget.
        """
        if self.debug_output:
            with self.debug_output:
                self.debug_output.clear_output(wait=True)
                display(HTML(f"<pre>{message}</pre>"))
        #else:
        #    print(message)


class JSONValidationStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, json_validator, prompt_length, previous_tokens = None):
        self.tokenizer = tokenizer
        self.json_validator = json_validator
        self.prompt_length = prompt_length
        self.previous_tokens = previous_tokens or []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Combine previous tokens with newly generated tokens
        #print(self.tokenizer.decode(input_ids[0]))
        self.previous_tokens = input_ids[0].tolist()
        
        # Check if the generated sequence is valid JSON
        result = self.json_validator._validate_json_string(self.previous_tokens, self.prompt_length, validate_only=True)

        #if result:
        #    print('! json invalid: stopping generation')
        
        return result

    def update_previous_tokens(self, new_tokens):
        self.previous_tokens = new_tokens