import time
from typing import List, Dict, Tuple, Generator, Set, Union

import torch
from transformers import (
    PreTrainedTokenizer,
    StoppingCriteria,
)
from IPython.display import display, HTML
from ipywidgets import Output

class SlopPhraseHandler:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        slop_phrase_prob_adjustments: Dict[str, float],
        starting_tokens_lookup: Dict[Tuple[int, ...], Set[int]],
        adjustment_strength: float,
        slow_debug: bool,
        inference_output: Output | None,
        debug_output: Output | None,
        debug_delay: float,
    ):
        self.tokenizer = tokenizer
        self.slop_phrase_prob_adjustments = slop_phrase_prob_adjustments
        self.starting_tokens_lookup = starting_tokens_lookup
        self.adjustment_strength = adjustment_strength
        self.slow_debug = slow_debug
        self.inference_output = inference_output
        self.debug_output = debug_output
        self.debug_delay = debug_delay

        self.slop_phrase_sequences = self._prepare_slop_phrase_sequences()
        self.max_slop_phrase_length = max(len(seq) for seq in self.slop_phrase_sequences.keys()) if self.slop_phrase_sequences else 0
        self.stopping_criteria = SlopPhraseStoppingCriteria(tokenizer, self.slop_phrase_sequences, self.max_slop_phrase_length)
        self.downregulated_positions = {}  # Key: position, Value: set of sequences
        self.probs_cache = {}
        self.probs_cache_longrange = {}  # flags which positions in the logit cache we ignore during cleanup, as we want to keep some positions for long range constraint checks        

    
    def _prepare_slop_phrase_sequences(self) -> Dict[Tuple[int, ...], float]:
        slop_phrase_sequences = {}
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
                    slop_phrase_sequences[token_ids] = prob_adjustment_factor
        return slop_phrase_sequences


    def _handle_disallowed_sequence(
        self,
        matched_sequence: Tuple[int, ...],
        start_pos: int,
        generated_sequence: List[int],
        probs_cache: Dict[int, torch.FloatTensor],
        adjustment_strength: float,
        slow_debug: bool,
        tokenizer: PreTrainedTokenizer,
        inference_output: Output,
        debug_output: Output,
        debug_delay: float,
    ) -> List[int]:
        # Downregulate the relevant tokens at the start_pos
        adjustment = self.slop_phrase_sequences[matched_sequence]
        matched_phrase = self.tokenizer.decode(torch.tensor(matched_sequence))

        if slow_debug and False:
            current_text = self.tokenizer.decode(generated_sequence[start_pos:start_pos + len(matched_sequence)])
            # Add HTML formatting to display the matched_phrase in red
            highlighted_text = f"{current_text}<span style='color: red;'>{matched_phrase}</span>"
            
            with inference_output:
                inference_output.clear_output(wait=True)
                display(HTML(f"<div style='white-space: pre-wrap;'>{highlighted_text}</div>"))

        # Display debug information
        debug_info = f"Replacing '{matched_phrase}'"
        self._display_debug(debug_info)

        if slow_debug:
            time.sleep(debug_delay)
            if debug_output:
                with debug_output:
                    debug_output.clear_output(wait=True)                        

        # Identify starting tokens to downregulate
        starting_tokens = self.starting_tokens_lookup.get(matched_sequence, set())

        for token_id in starting_tokens:
            self.probs_cache[start_pos][:, token_id] *= adjustment ** adjustment_strength

        # Record that this sequence has been downregulated at this position
        if start_pos not in self.downregulated_positions:
            self.downregulated_positions[start_pos] = set()
        self.downregulated_positions[start_pos].add(matched_sequence)

        # Check if the starting token would still be selected after downregulation
        slop_phrase_starting_token = generated_sequence[start_pos]
        if torch.argmax(self.probs_cache[start_pos]).item() == slop_phrase_starting_token:
            if slow_debug:
                debug_info = f"Slop phrase '{self.tokenizer.decode(matched_sequence)}' prob was downregulated {round(1/(adjustment**adjustment_strength), 2)}x but still selected."
                self._display_debug(debug_info)
                #time.sleep(debug_delay)
            return generated_sequence

        # Backtrack: remove tokens from the generated_sequence that are part of the disallowed sequence
        for _ in range(len(matched_sequence)):
            generated_sequence.pop()

        if slow_debug and False:
            current_text = self.tokenizer.decode(generated_sequence[self.prompt_length:])
            with inference_output:
                inference_output.clear_output(wait=True)
                display(HTML(f"<div style='white-space: pre-wrap;'>{current_text}</div>"))
            #time.sleep(debug_delay)

        # Clear the probs_cache ahead of start_pos since we've backtracked
        to_del = [key for key in self.probs_cache if key > start_pos]
        for key in to_del:
            del self.probs_cache[key]

        return generated_sequence

    def deslop(self, generated_sequence, prompt_length):
        self.prompt_length = prompt_length
        # After adding the token(s), check for disallowed sequences
        matched_sequence, start_pos = self.stopping_criteria._detect_disallowed_sequence(generated_sequence)

        if matched_sequence:
            matched_phrase = self.tokenizer.decode(torch.tensor(matched_sequence))

            if self.slow_debug:
                current_text = self.tokenizer.decode(generated_sequence[prompt_length:-len(matched_sequence)])
                # Add HTML formatting to display the matched_phrase in red
                highlighted_text = f"{current_text}<span style='color: red;'>{matched_phrase}</span>"
                
                with self.inference_output:
                    self.inference_output.clear_output(wait=True)
                    display(HTML(f"<div style='white-space: pre-wrap;'>{highlighted_text}</div>"))

            # Display debug information
            debug_info = f"Replacing '{matched_phrase}'"
            self._display_debug(debug_info)

            if self.slow_debug:
                time.sleep(self.debug_delay)
                if self.debug_output:
                    with self.debug_output:
                        self.debug_output.clear_output(wait=True)                        

            # Handle the disallowed sequence using SlopPhraseHandler
            generated_sequence = self._handle_disallowed_sequence(
                matched_sequence=matched_sequence,
                start_pos=start_pos,
                generated_sequence=generated_sequence,
                probs_cache=self.probs_cache,
                adjustment_strength=self.adjustment_strength,
                slow_debug=self.slow_debug,
                tokenizer=self.tokenizer,
                inference_output=self.inference_output,
                debug_output=self.debug_output,
                debug_delay=self.debug_delay,                
            )

            return generated_sequence
        return False

    def _display_debug(self, message: str):
        """
        Displays debug information in the debug_output widget.
        """
        if self.debug_output:
            with self.debug_output:
                self.debug_output.clear_output(wait=True)
                display(HTML(f"<pre>{message}</pre>"))


class SlopPhraseStoppingCriteria:
    def __init__(self, tokenizer: PreTrainedTokenizer, slop_phrase_sequences: Dict[Tuple[int, ...], float], max_slop_phrase_length: int):
        self.tokenizer = tokenizer
        self.slop_phrase_sequences = slop_phrase_sequences
        self.max_slop_phrase_length = max_slop_phrase_length

    def _detect_disallowed_sequence(self, generated_sequence: List[int]) -> Tuple[Tuple[int, ...], int]:
        for seq_length in range(self.max_slop_phrase_length, 0, -1):            
            if len(generated_sequence) < seq_length:
                continue
            candidate_sequence = tuple(generated_sequence[-seq_length:])
            if candidate_sequence in self.slop_phrase_sequences:
                start_pos = len(generated_sequence) - seq_length
                return candidate_sequence, start_pos
        return None, -1


class CustomSlopPhraseStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, slop_phrase_sequences, max_slop_phrase_length, previous_tokens=None):
        self.tokenizer = tokenizer
        self.slop_phrase_sequences = slop_phrase_sequences
        self.max_slop_phrase_length = max_slop_phrase_length
        self.previous_tokens = previous_tokens or []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Combine previous tokens with newly generated tokens
        self.previous_tokens = input_ids[0].tolist()
        # Check only the last max_sequence_length tokens
        for seq_length in range(self.max_slop_phrase_length, 0, -1):
            if len(self.previous_tokens) < seq_length:
                continue
            candidate_sequence = tuple(self.previous_tokens[-seq_length:])
            if candidate_sequence in self.slop_phrase_sequences:
                #print('hit! stopping early')
                return True
        return False

    def update_previous_tokens(self, new_tokens):
        self.previous_tokens = new_tokens