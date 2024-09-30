import torch
from transformers import PreTrainedTokenizer, PreTrainedModel, StoppingCriteria, StoppingCriteriaList
from typing import List, Dict, Tuple, Generator, Set, Union
import time

class SlopPhraseStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer: PreTrainedTokenizer, token_sequences: Dict[Tuple[int, ...], float], max_sequence_length: int):
        self.tokenizer = tokenizer
        self.token_sequences = token_sequences
        self.max_sequence_length = max_sequence_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        generated_sequence = input_ids[0].tolist()
        matched_sequence, _ = self._detect_disallowed_sequence(generated_sequence)
        return matched_sequence is not None

    def _detect_disallowed_sequence(self, generated_sequence: List[int]) -> Tuple[Tuple[int, ...], int]:
        for seq_length in range(self.max_sequence_length, 0, -1):            
            if len(generated_sequence) < seq_length:
                continue
            candidate_sequence = tuple(generated_sequence[-seq_length:])
            if candidate_sequence in self.token_sequences:
                start_pos = len(generated_sequence) - seq_length
                return candidate_sequence, start_pos
        return None, -1

class AdvancedCustomWordSampler:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        slop_phrase_prob_adjustments: Dict[str, float],
        starting_tokens_lookup: Dict[Tuple[int, ...], Set[int]],
        adjustment_strength: float = 1.0,
        device: torch.device = torch.device('cuda'),
        slow_debug: bool = False,
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.slop_phrase_prob_adjustments = slop_phrase_prob_adjustments
        self.starting_tokens_lookup = starting_tokens_lookup
        self.adjustment_strength = adjustment_strength
        self.device = device
        self.slow_debug = slow_debug

        self.token_sequences = self._prepare_token_sequences()
        self.max_sequence_length = max(len(seq) for seq in self.token_sequences.keys())

        self.stopping_criteria = SlopPhraseStoppingCriteria(tokenizer, self.token_sequences, self.max_sequence_length)
        self.output_every_n_tokens = 1

        self.downregulated_positions = {}  # Key: position, Value: set of sequences

        self.logit_cache = {}

    def _prepare_token_sequences(self) -> Dict[Tuple[int, ...], float]:
        token_sequences = {}
        for word, prob_adjustment_factor in self.slop_phrase_prob_adjustments.items():
            variants = [
                word.lower(),
                word.capitalize(),
                word.upper(),
                f" {word.lower()}",
                f" {word.capitalize()}",
                f" {word.upper()}",
            ]
            for variant in variants:
                token_ids = tuple(self.tokenizer.encode(variant, add_special_tokens=False))
                if token_ids:
                    token_sequences[token_ids] = prob_adjustment_factor
        return token_sequences

    @torch.no_grad()
    def generate_stream(
        self,
        prompt: str,
        max_length: int,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
        min_p: float = None,
    ) -> Generator[List[int], None, None]:
        """
        Generates text in a streaming fashion with custom downregulation and backtracking.

        Args:
            prompt (str): The initial text prompt.
            max_length (int): The maximum length of the generated text.
            temperature (float): Sampling temperature.
            top_k (int): Top-k filtering.
            top_p (float): Top-p (nucleus) filtering.
            min_p (float): Minimum probability filtering.

        Yields:
            Generator[List[int], None, None]: Yields generated token sequences.
        """
        
        # Encode the prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        generated_sequence = input_ids[0].tolist()
        current_position = len(generated_sequence)  # Tracks the current position in the sequence
        output_tokens_counter = 0
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

        while len(generated_sequence) < max_length:
            current_input_ids = torch.tensor([generated_sequence], device=self.device)

            regenerating = False
            if current_position in self.logit_cache:
                # We backtracked and want to use the cached logits
                next_token_logits = self.logit_cache[current_position]
                regenerating = True
            else:
                outputs = self.model.generate(
                    current_input_ids,
                    attention_mask = torch.ones_like(current_input_ids),
                    max_new_tokens=1,
                    do_sample=False,
                    temperature = 1, # we apply temp ourselves after this
                    pad_token_id = pad_token_id,
                    num_return_sequences=1,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                next_token_logits = outputs.scores[0]
                self.logit_cache[current_position] = next_token_logits.clone()

                # Apply temperature
                # note: we don't want to re-apply temp if using cached logits
                next_token_logits = next_token_logits / temperature

            # Apply min_p, top-k and top-p filtering
            filtered_logits = self._filter_logits(next_token_logits, top_k, top_p, min_p)

            # Sample the next token
            probs = torch.softmax(filtered_logits, dim=-1)
            next_token_index = torch.multinomial(probs, num_samples=1)
            next_token = next_token_index.item()

            if regenerating and self.slow_debug:
                alt_token = self.tokenizer.decode(next_token, skip_special_tokens=True)
                debug_info = f"Alternate token: {alt_token}"
                self._display_debug(debug_info)
                
            # Append the new token to the sequence
            generated_sequence.append(next_token)
            current_position += 1
            output_tokens_counter += 1

            # Clean up the logits cache
            to_del = [key for key in self.logit_cache if key < current_position - self.max_sequence_length - 5]
            for key in to_del:
                del self.logit_cache[key]

            # Yield the current text chunk            
            if output_tokens_counter >= self.output_every_n_tokens:
                output_tokens_counter = 0
                yield generated_sequence  # Yield the generated token sequence

            # Check for end-of-sequence token
            if next_token == self.tokenizer.eos_token_id:
                break

            # After adding the token, check for disallowed sequences
            matched_sequence, start_pos = self.stopping_criteria._detect_disallowed_sequence(generated_sequence)

            if matched_sequence:
                # Downregulate the relevant tokens at the start_pos
                adjustment = self.token_sequences[matched_sequence]
                word = self.tokenizer.decode(torch.tensor(matched_sequence))

                # Display debug information
                if self.slow_debug:
                    debug_info = f"Replacing '{word}'"
                    self._display_debug(debug_info)          

                # Identify starting tokens to downregulate
                starting_tokens = self.starting_tokens_lookup.get(matched_sequence, set())

                for token_id in starting_tokens:
                    self.logit_cache[start_pos][:, token_id] *= adjustment ** self.adjustment_strength

                # Record that this sequence has been downregulated at this position
                if start_pos not in self.downregulated_positions:
                    self.downregulated_positions[start_pos] = set()
                self.downregulated_positions[start_pos].add(matched_sequence)

                # Check if the starting token would still be selected after downregulation
                slop_phrase_starting_token = generated_sequence[start_pos]
                if torch.argmax(self.logit_cache[start_pos]).item() == slop_phrase_starting_token:
                    if self.slow_debug:
                        debug_info = f"[INFO] Slop phrase '{self.tokenizer.decode(matched_sequence)}' prob was downregulated {round(1/adjustment, 2)}x but still selected."
                        self._display_debug(debug_info)
                        time.sleep(self.debug_delay)
                    continue

                # Backtrack: remove tokens from the generated_sequence that are part of the disallowed sequence
                for _ in range(len(matched_sequence)):
                    generated_sequence.pop()
                    current_position -= 1

                # Clear the logit_cache ahead of start_pos since we've backtracked
                to_del = [key for key in self.logit_cache if key > start_pos]
                for key in to_del:
                    del self.logit_cache[key]

                continue  # Continue to the next iteration

        yield generated_sequence

        # Clear variables to free up memory
        del next_token_logits, filtered_logits

    def _filter_logits(self, logits: torch.FloatTensor, top_k: int, top_p: float, min_p: float) -> torch.FloatTensor:
        # Apply min_p filtering
        if min_p is not None:
            probs = torch.softmax(logits, dim=-1)
            top_prob, _ = torch.max(probs, dim=-1)
            scaled_min_p = min_p * top_prob
            logits = torch.where(probs < scaled_min_p, float('-inf'), logits)

        if top_k is not None and top_k > 0:
            top_k = min(top_k, logits.size(-1))
            top_k_logits, _ = torch.topk(logits, top_k)
            min_top_k = top_k_logits[:, -1].unsqueeze(-1)
            logits = torch.where(logits < min_top_k, float('-inf'), logits)

        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits = logits.masked_fill(indices_to_remove, float('-inf'))

        return logits

    def _display_debug(self, debug_info):
        print(debug_info)

def chat_antislop(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    messages: List[Dict[str, str]],
    max_length: int,
    temperature: float = 1.0,
    top_k: int = None,
    top_p: float = None,
    min_p: float = None,
    slop_phrase_prob_adjustments: Dict[str, float] = None,
    adjustment_strength: float = 1.0, # 1.0 is no change from the provided adjustment factors. 0-10 are reasonable values.
    device: torch.device = torch.device('cuda'),
    streaming: bool = False,
) -> Union[Generator[str, None, None], List[int]]:
    """
    Generates a chat response while avoiding overrepresented phrases (slop).

    Args:
        model (PreTrainedModel): The language model.
        tokenizer (PreTrainedTokenizer): The tokenizer.
        messages (List[Dict[str, str]]): The list of messages in the conversation.
        max_length (int): The maximum length of the generated text (including the prompt).
        temperature (float): Sampling temperature.
        top_k (int): Top-k filtering.
        top_p (float): Top-p (nucleus) filtering.
        slop_phrase_prob_adjustments (Dict[str, float]): Dictionary of target words with their respective probability adjustment factor.
        adjustment_strength (float): Strength of the downregulation adjustment.
        device (torch.device): Device to run the model on.
        streaming (bool): Whether to yield tokens as they are generated.

    Returns:
        Union[Generator[str, None, None], List[int]]: 
            If streaming is True, yields generated text chunks.
            If streaming is False, returns a list of generated token IDs.
    """
    # Build the prompt using the tokenizer's chat template
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return generate_antislop(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        slop_phrase_prob_adjustments=slop_phrase_prob_adjustments,
        adjustment_strength=adjustment_strength,
        device=device,
        streaming=streaming,
    )


def generate_antislop(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_length: int,
    temperature: float = 1.0,
    top_k: int = None,
    top_p: float = None,
    min_p: float = None,
    slop_phrase_prob_adjustments: Dict[str, float] = None,
    adjustment_strength: float = 1.0,
    device: torch.device = torch.device('cuda'),
    streaming: bool = False,
) -> Union[Generator[str, None, None], List[int]]:
    """
    Wrapper function for generate_antislop that handles both streaming and non-streaming modes.
    """
    # Type checking
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    if not isinstance(max_length, int):
        raise TypeError("max_length must be an integer")
    if not isinstance(temperature, (int, float)):
        raise TypeError("temperature must be a float")
    if top_k is not None and not isinstance(top_k, int):
        raise TypeError("top_k must be an integer or None")
    if top_p is not None and not isinstance(top_p, float):
        raise TypeError("top_p must be a float or None")
    if min_p is not None and not isinstance(min_p, float):
        raise TypeError("min_p must be a float or None")
    if slop_phrase_prob_adjustments is not None and not isinstance(slop_phrase_prob_adjustments, dict):
        raise TypeError("slop_phrase_prob_adjustments must be a dictionary or None")
    if not isinstance(adjustment_strength, (int, float)):
        raise TypeError("adjustment_strength must be a float")
    if not isinstance(device, torch.device):
        raise TypeError("device must be an instance of torch.device")
    if not isinstance(streaming, bool):
        raise TypeError("streaming must be a boolean")

    # Value validation
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive")
    if top_p is not None and (top_p <= 0 or top_p > 1):
        raise ValueError("top_p must be in the range (0, 1]")
    if min_p is not None and (min_p <= 0 or min_p > 1):
        raise ValueError("min_p must be in the range (0, 1]")
    if adjustment_strength < 0:
        raise ValueError("adjustment_strength must be non-negative")

    if slop_phrase_prob_adjustments:
        for phrase, adjustment in slop_phrase_prob_adjustments.items():
            if not isinstance(phrase, str):
                raise TypeError("All keys in slop_phrase_prob_adjustments must be strings")
            if not isinstance(adjustment, (int, float)):
                raise TypeError("All values in slop_phrase_prob_adjustments must be floats")

    if streaming:
        return _generate_antislop(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            slop_phrase_prob_adjustments=slop_phrase_prob_adjustments,
            adjustment_strength=adjustment_strength,
            device=device,
            streaming=True
        )
    else:
        generated_tokens = []
        for token in _generate_antislop(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            slop_phrase_prob_adjustments=slop_phrase_prob_adjustments,
            adjustment_strength=adjustment_strength,
            device=device,
            streaming=True  # We always use streaming internally
        ):
            generated_tokens.append(token)
        return generated_tokens

# The original generate_antislop function remains unchanged
def _generate_antislop(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_length: int,
    temperature: float = 1.0,
    top_k: int = None,
    top_p: float = None,
    min_p: float = None,
    slop_phrase_prob_adjustments: Dict[str, float] = None,
    adjustment_strength: float = 1.0,
    device: torch.device = torch.device('cuda'),
    streaming: bool = False,
) -> Generator[int, None, None]:
    """
    Generates text while avoiding overrepresented phrases (slop).
    This function is now always a generator.
    """

    if slop_phrase_prob_adjustments is None:
        slop_phrase_prob_adjustments = [['kaleidoscope', 0.5], ['symphony', 0.5], ['testament to', 0.5], ['elara', 0.5], ['moth to a flame', 0.5], ['canvas', 0.5], ['eyes glinted', 0.5], ['camaraderie', 0.5], ['humble abode', 0.5], ['cold and calculating', 0.5], ['eyes never leaving', 0.5], ['tapestry', 0.5], ['barely above a whisper', 0.5], ['body and soul', 0.5], ['orchestra', 0.5], ['depths', 0.5], ['a dance of', 0.5], ['chuckles darkly', 0.5], ['maybe, just maybe', 0.5], ['maybe that was enough', 0.5], ['with a mixture of', 0.5], ['air was filled with anticipation', 0.5], ['cacophony', 0.5], ['bore silent witness to', 0.5], ['eyes sparkling with mischief', 0.5], ['was only just beginning', 0.5], ['practiced ease', 0.5], ['ready for the challenges', 0.5], ['only just getting started', 0.5], ['once upon a time', 0.5], ['nestled deep within', 0.5], ['ethereal beauty', 0.5], ['life would never be the same again.', 0.5], ["it's important to remember", 0.5], ['for what seemed like an eternity', 0.5], ['feel a sense of pride and accomplishment', 0.5], ['little did he know', 0.5], ['ball is in your court', 0.5], ['game is on', 0.5], ['choice is yours', 0.5], ['feels like an electric shock', 0.5], ['threatens to consume', 0.5], ['meticulous', 0.5], ['meticulously', 0.5], ['navigating', 0.5], ['complexities', 0.5], ['realm', 0.5], ['understanding', 0.5], ['dive into', 0.5], ['shall', 0.5], ['tailored', 0.5], ['towards', 0.5], ['underpins', 0.5], ['everchanging', 0.5], ['ever-evolving', 0.5], ['world of', 0.5], ['not only', 0.5], ['alright', 0.5], ['embark', 0.5], ['journey', 0.5], ["today's digital age", 0.5], ['game changer', 0.5], ['designed to enhance', 0.5], ['it is advisable', 0.5], ['daunting', 0.5], ['when it comes to', 0.5], ['in the realm of', 0.5], ['amongst', 0.5], ['unlock the secrets', 0.5], ['unveil the secrets', 0.5], ['and robust', 0.5], ['diving', 0.5], ['elevate', 0.5], ['unleash', 0.5], ['cutting-edge', 0.5], ['rapidly', 0.5], ['expanding', 0.5], ['mastering', 0.5], ['excels', 0.5], ['harness', 0.5], ["it's important to note", 0.5], ['delve into', 0.5], ['bustling', 0.5], ['in summary', 0.5], ['remember that', 0.5], ['take a dive into', 0.5], ['landscape', 0.5], ['in the world of', 0.5], ['vibrant', 0.5], ['metropolis', 0.5], ['firstly', 0.5], ['moreover', 0.5], ['crucial', 0.5], ['to consider', 0.5], ['essential', 0.5], ['there are a few considerations', 0.5], ['ensure', 0.5], ["it's essential to", 0.5], ['furthermore', 0.5], ['vital', 0.5], ['keen', 0.5], ['fancy', 0.5], ['as a professional', 0.5], ['however', 0.5], ['therefore', 0.5], ['additionally', 0.5], ['specifically', 0.5], ['generally', 0.5], ['consequently', 0.5], ['importantly', 0.5], ['indeed', 0.5], ['thus', 0.5], ['alternatively', 0.5], ['notably', 0.5], ['as well as', 0.5], ['despite', 0.5], ['essentially', 0.5], ['even though', 0.5], ['in contrast', 0.5], ['in order to', 0.5], ['due to', 0.5], ['even if', 0.5], ['given that', 0.5], ['arguably', 0.5], ['you may want to', 0.5], ['on the other hand', 0.5], ['as previously mentioned', 0.5], ["it's worth noting that", 0.5], ['to summarize', 0.5], ['ultimately', 0.5], ['to put it simply', 0.5], ["in today's digital era", 0.5], ['reverberate', 0.5], ['enhance', 0.5], ['emphasize', 0.5], ['revolutionize', 0.5], ['foster', 0.5], ['remnant', 0.5], ['subsequently', 0.5], ['nestled', 0.5], ['labyrinth', 0.5], ['gossamer', 0.5], ['enigma', 0.5], ['whispering', 0.5], ['sights unseen', 0.5], ['sounds unheard', 0.5], ['indelible', 0.5], ['my friend', 0.5], ['in conclusion', 0.5], ['technopolis', 0.5], ['was soft and gentle', 0.5], ['shivers down', 0.5], ['shivers up', 0.5], ['leaving trails of fire', 0.5], ['ministrations', 0.5], ['audible pop', 0.5], ['rivulets of', 0.5], ['despite herself', 0.5], ['reckless abandon', 0.5], ['torn between', 0.5], ['fiery red hair', 0.5], ['long lashes', 0.5], ['propriety be damned', 0.5], ['world narrows', 0.5], ['chestnut eyes', 0.5], ['cheeks flaming', 0.5], ['cheeks hollowing', 0.5], ['understandingly', 0.5], ['paperbound', 0.5], ['hesitantly', 0.5], ['piqued', 0.5], ['delved', 0.5], ['curveballs', 0.5], ['marveled', 0.5], ['inclusivity', 0.5], ['birdwatcher', 0.5], ['newfound', 0.5031423922762257], ['marveling', 0.5055622891781474], ["hiroshi's", 0.506870969939047], ['greentech', 0.5095092042816856], ['thoughtfully', 0.510153898156777], ['intently', 0.5153227374075411], ['birdwatching', 0.5157928537951464], ['amidst', 0.5161190296674488], ['cherishing', 0.5165772000484282], ['attentively', 0.5169695157301188], ['interjected', 0.5208671011920856], ['serendipitous', 0.5219535186850968], ["marianne's", 0.5220118279910801], ["maya's", 0.5229467776607973], ['excitedly', 0.5235248665614571], ['steepled', 0.5235772300889154], ['engrossed', 0.5236764398055735], ['fostering', 0.5259281627970829], ['brainstormed', 0.5274863713437], ['furrowed', 0.5280860997212533], ['nodded', 0.528640180937889], ['contemplatively', 0.5293698584415747], ['jotted', 0.5300819077932343], ["mia's", 0.5311706933553655]];
        slop_phrase_prob_adjustments = dict(slop_phrase_prob_adjustments)

    starting_tokens_lookup = precompute_starting_tokens(tokenizer, slop_phrase_prob_adjustments)

    sampler = AdvancedCustomWordSampler(
        model=model,
        tokenizer=tokenizer,
        slop_phrase_prob_adjustments=slop_phrase_prob_adjustments,
        starting_tokens_lookup=starting_tokens_lookup,
        adjustment_strength=adjustment_strength,
        device=device,
        slow_debug=False
    )

    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    buffer_size = sampler.max_sequence_length + 5
    tokens_to_wait = buffer_size
    last_released_position = len(prompt_tokens) - 1
    last_sequence_length = len(prompt_tokens)

    token_stream = sampler.generate_stream(
        prompt=prompt,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p
    )

    for generated_sequence in token_stream:
        current_length = len(generated_sequence)
        
        if current_length <= last_sequence_length:
            tokens_to_wait += last_sequence_length - current_length
        else:
            if tokens_to_wait > 0:
                tokens_to_wait -= 1
            else:                    
                last_released_position += 1
                token_to_release = generated_sequence[last_released_position]
                yield token_to_release
        
        last_sequence_length = current_length
    
    # Release any remaining tokens after generation is complete
    if last_released_position < len(generated_sequence)-1:
        for tok in generated_sequence[last_released_position+1:]:
            yield tok

# The precompute_starting_tokens function remains unchanged
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
                    encoded_prefix = tokenizer.encode(prefix, add_special_tokens=False)
                    if encoded_prefix:
                        starting_tokens.add(encoded_prefix[0])

                starting_tokens_lookup[tuple(token_ids)] = starting_tokens

    return starting_tokens_lookup
