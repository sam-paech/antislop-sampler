import time
from typing import List, Dict, Tuple, Generator, Set, Union
from threading import Thread
import threading
from queue import Queue
import torch
from transformers import (
    PreTrainedTokenizer,
    PreTrainedModel,
    StoppingCriteriaList,
    TextIteratorStreamer
)
from IPython.display import display, HTML
from ipywidgets import Output
from src.validator_slop import SlopPhraseHandler, CustomSlopPhraseStoppingCriteria
from src.validator_json import JSONValidator, JSONValidationStoppingCriteria
from src.util import precompute_starting_tokens

class AntiSlopSampler:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        slop_phrase_prob_adjustments: Dict[str, float],
        starting_tokens_lookup: Dict[Tuple[int, ...], Set[int]],
        adjustment_strength: float = 1.0,
        device: torch.device = torch.device('cuda'),
        slow_debug: bool = False,
        output_every_n_tokens: int = 1,
        debug_delay: float = 2.0,
        inference_output=None,
        debug_output=None,
		enforce_json: bool = False,
        antislop_enabled: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.slop_phrase_prob_adjustments = slop_phrase_prob_adjustments
        self.starting_tokens_lookup = starting_tokens_lookup
        self.adjustment_strength = adjustment_strength
        self.device = device
        self.slow_debug = slow_debug
        self.output_every_n_tokens = output_every_n_tokens
        self.debug_delay = debug_delay        
        self.downregulated_positions = {}  # Key: position, Value: set of sequences
        self.enforce_json = enforce_json
        self.antislop_enabled = antislop_enabled

        # Output widgets
        self.inference_output = inference_output
        self.debug_output = debug_output

        # Escaped toks used for lookups in json string repair
        self.escaped_tokens_lookup = {
            '\n': self.tokenizer.encode('\\n', add_special_tokens=False),
            '\t': self.tokenizer.encode('\\t', add_special_tokens=False),
            '\r': self.tokenizer.encode('\\r', add_special_tokens=False),
            '"': self.tokenizer.encode('\\"', add_special_tokens=False),
            ' "': self.tokenizer.encode(' \\"', add_special_tokens=False),
        }

        # Initialize Slop Phrase Handler
        self.slop_phrase_handler = SlopPhraseHandler(
            tokenizer=tokenizer,
            slop_phrase_prob_adjustments=slop_phrase_prob_adjustments,
            starting_tokens_lookup=starting_tokens_lookup,
            adjustment_strength=adjustment_strength,
            slow_debug=slow_debug,
            inference_output=inference_output,
            debug_output=debug_output,
            debug_delay=debug_delay
        )
        self.json_validator = JSONValidator(tokenizer, slow_debug, debug_delay, debug_output, self.slop_phrase_handler.probs_cache_longrange)
        self.streamer_retval = None

    def _generate_streaming(self, current_input_ids, new_toks_to_generate, temperature, min_p, top_k, top_p, pad_token_id, stopping_criteria_args):
        streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=False)

        generation_kwargs = dict(
            input_ids=current_input_ids,
            attention_mask=torch.ones_like(current_input_ids),
            max_new_tokens=new_toks_to_generate,
            do_sample=True,
            temperature=temperature,
            min_p=min_p,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=pad_token_id,
            num_return_sequences=1,
            return_dict_in_generate=True,
            output_logits=True,
            streamer=streamer,
            **stopping_criteria_args
        )

        # Create an Event to signal thread termination
        stop_event = threading.Event()

        # Create a Queue to store the generation output or errors
        output_queue = Queue()

        # Define a function to run generation and put the result in the queue
        def generate_and_queue():            
            try:
                output = self.model.generate(**generation_kwargs)
                if not stop_event.is_set():
                    output_queue.put((output, None))  # None means no exception occurred
            except Exception as e:
                print(f"Exception during generation: {e}")  # Debug print
                if not stop_event.is_set():
                    output_queue.put((None, e))  # Put the exception in the queue
                stop_event.set()

        # Start the generation in a separate thread
        thread = Thread(target=generate_and_queue)
        thread.start()

        try:
            for new_text in streamer:
                yield new_text
        except Exception as e:
            print(f"Exception during streaming: {e}")  # Debug print
            # Add the exception to the output queue so it is propagated to the caller
            if not stop_event.is_set():
                output_queue.put((None, e))  # Handle exception during streaming
        #finally:
        #    stop_event.set()  # Signal the thread to stop

        # Wait for the generation to complete or for the thread to be terminated
        thread.join()

        # Initialize default empty lists for output variables
        generated_sequence = []
        new_logits = []
        error = None  # Default error to None

        # Check if there's any output in the queue
        if not output_queue.empty():
            generation_output, error = output_queue.get()

            # Check if an error occurred during generation or streaming
            if error:
                print(f"Generation or streaming failed: {error}")
            else:
                # Extract logits and sequence from the generation output
                new_logits = generation_output.logits
                generated_sequence = generation_output.sequences[0].tolist()

        # Add final debug information for empty output
        if not generated_sequence:
            print("Warning: Generated sequence is empty.")
        if not new_logits:
            print("Warning: Logits are empty.")

        # Return the generated sequence, logits, and any error
        self.streamer_retval = (generated_sequence, new_logits, error)



    @torch.no_grad()
    def generate_stream(
        self,
        prompt: str,
        max_length: int = None,
        max_new_tokens: int = None,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
        min_p: float = None,
    ) -> Generator[List[int], None, None]:
        """
        Generates text in a streaming fashion with custom downregulation and backtracking.

        Args:
            prompt (str): The initial text prompt.
            max_length (int, optional): The maximum length of the generated text.
            max_new_tokens (int, optional): The maximum number of new tokens to generate.
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

        # If the prompt already came with a bos token, we don't want to add it again
        if self.tokenizer.bos_token and \
                prompt.startswith(self.tokenizer.bos_token) and \
                not prompt.startswith(self.tokenizer.bos_token * 2) and \
                generated_sequence[0] == self.tokenizer.bos_token_id and \
                generated_sequence[1] == self.tokenizer.bos_token_id:
            generated_sequence = generated_sequence[1:]

        self.prompt_length = len(generated_sequence)
        self.prompt_length_chars = len(prompt)
        current_position = len(generated_sequence)  # Tracks the current position in the sequence
        output_tokens_counter = 0
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        next_token_logits = None
        filtered_logits = None

        if max_length != None:
            this_max_new_tokens = max_length - self.prompt_length
            if this_max_new_tokens < 0:
                this_max_new_tokens = 0
            if max_new_tokens == None or this_max_new_tokens < max_new_tokens:
                max_new_tokens = this_max_new_tokens
        else:
            if max_new_tokens == None:
                max_new_tokens = 8096

        stopping_criteria_args = {}
        self.stopping_criteria = []
        
        if self.enforce_json:
            json_stopping_criteria = JSONValidationStoppingCriteria(
                tokenizer=self.tokenizer,
                json_validator=self.json_validator,
                prompt_length=self.prompt_length
            )
            self.stopping_criteria.append(json_stopping_criteria)

        if self.antislop_enabled:
            antislop_stopping_criteria = CustomSlopPhraseStoppingCriteria(
                tokenizer=self.tokenizer,
                slop_phrase_sequences=self.slop_phrase_handler.slop_phrase_sequences,
                max_slop_phrase_length=self.slop_phrase_handler.max_slop_phrase_length,
                previous_tokens=[]  # Initially empty
            )
            self.stopping_criteria.append(antislop_stopping_criteria)

        if self.stopping_criteria:
            stopping_criteria_args = {
                "stopping_criteria": StoppingCriteriaList(self.stopping_criteria)
            }
        

        while True:            
            if max_new_tokens is not None and len(generated_sequence) - self.prompt_length >= max_new_tokens:
                #print('max_new_tokens reached')
                break

            new_toks_to_generate = max_new_tokens - (len(generated_sequence) - self.prompt_length)            

            current_input_ids = torch.tensor([generated_sequence], device=self.device)

            regenerating = False

            if current_position in self.slop_phrase_handler.probs_cache:
                # We backtracked and want to use the cached logits
                next_token_probs = self.slop_phrase_handler.probs_cache[current_position]
                regenerating = True
            else:
                context = ""
                #print(new_toks_to_generate)
                for new_text in self._generate_streaming(
                    current_input_ids,
                    new_toks_to_generate,
                    temperature,
                    min_p,
                    top_k,
                    top_p,
                    pad_token_id,
                    stopping_criteria_args
                ):
                    context += new_text
                    output_tokens_counter += 1

                    # sometimes model.generate adds an extra bos token so we'll manually clip it off.
                    # otherwise we have conflicts with the originally calculated prompt_length
                    if self.tokenizer.bos_token and \
                            prompt.startswith(self.tokenizer.bos_token) and \
                            not prompt.startswith(self.tokenizer.bos_token * 2) and \
                            context.startswith(self.tokenizer.bos_token * 2):
                        context = context[len(self.tokenizer.bos_token):]
                    
                    if output_tokens_counter >= self.output_every_n_tokens:
                        output_tokens_counter = 0

                        if self.inference_output:
                            with self.inference_output:
                                self.inference_output.clear_output(wait=True)
                                
                                display(HTML(f"<div style='white-space: pre-wrap;'>{context[self.prompt_length_chars:]}</div>"))
                        
                        yield self.tokenizer.encode(context, add_special_tokens=False)

                torch.cuda.empty_cache()

                # sync with the returned vals in case the streaming came thru out of order
                # (not sure if this is necessary)
                if self.streamer_retval:                    
                    generated_sequence, new_logits, error = self.streamer_retval
                    if error:
                        yield []
                        return
                    #if not generated_sequence:
                        # Model failed to return any tokens; likely an error so we'll return an empty list.
                    #    yield []
                    #    return
                    
                    # sometimes model.generate adds an extra bos token so we'll manually clip it off.
                    # otherwise we have conflicts with the originally calculated prompt_length
                    if self.tokenizer.bos_token and \
                            prompt.startswith(self.tokenizer.bos_token) and \
                            not prompt.startswith(self.tokenizer.bos_token * 2) and \
                            generated_sequence[0] == self.tokenizer.bos_token_id and \
                            generated_sequence[1] == self.tokenizer.bos_token_id:
                        generated_sequence = generated_sequence[1:]
  
                    self.streamer_retval = None
                else:
                    print('!! error missing retval from streamer')
                    yield []
                    return

                if self.stopping_criteria:
                    for criteria in self.stopping_criteria:
                        criteria.update_previous_tokens(generated_sequence)
                #print(len(new_logits))
                for i, logit in enumerate(new_logits):
                    self.slop_phrase_handler.probs_cache[current_position + i] = torch.softmax(logit.clone(), dim=-1)

                next_token = generated_sequence[-1]
                current_position = len(generated_sequence)


            if regenerating:
                # Apply min_p, top-k and top-p filtering
                filtered_probs = self._filter_probs(next_token_probs, top_k, top_p, min_p)
                # Sample the next token                
                next_token_index = torch.multinomial(filtered_probs, num_samples=1)

                next_token = next_token_index.item()
                # Append the new token to the sequence
                generated_sequence.append(next_token)                
                output_tokens_counter += 1

                if self.stopping_criteria:
                    for criteria in self.stopping_criteria:
                        criteria.update_previous_tokens(generated_sequence)

                if output_tokens_counter >= self.output_every_n_tokens:
                    output_tokens_counter = 0
                    current_text = self.tokenizer.decode(generated_sequence[self.prompt_length:])
                    if self.inference_output:
                        with self.inference_output:
                            self.inference_output.clear_output(wait=True)
                            display(HTML(f"<div style='white-space: pre-wrap;'>{current_text}</div>"))
                    yield generated_sequence  # Yield the generated token sequence
                #print('downregulated token after reselection', self.slop_phrase_handler.probs_cache[current_position][:, self.json_validator.last_downregulated_token])
                current_position = len(generated_sequence)

            if regenerating and self.slow_debug:
                alt_token = self.tokenizer.decode(next_token, skip_special_tokens=True)
                debug_info = f"Alternate token: {[alt_token]}"

                self._display_debug(debug_info)
                if self.slow_debug:
                    time.sleep(self.debug_delay)


            # Clean up the probs cache
            if not self.enforce_json:
                # json validation needs to keep the long range dependencies
                # although we can probably delete the ones that aren't flagged in self.probs_cache_longrange.

                to_del = [key for key in self.slop_phrase_handler.probs_cache if key < current_position - self.slop_phrase_handler.max_slop_phrase_length - 5 and not self.slop_phrase_handler.probs_cache_longrange.get(key, False)]                
                for key in to_del:
                    if key not in self.slop_phrase_handler.probs_cache_longrange:
                        del self.slop_phrase_handler.probs_cache[key]


            # Check for end-of-sequence token
            if next_token == self.tokenizer.eos_token_id:
                break

            # JSON validation
            if self.enforce_json:
                result = self.json_validator.validate_json_string(generated_sequence, self.prompt_length, self.slop_phrase_handler.probs_cache)
                if result != False:
                    generated_sequence = result
                    current_position = len(generated_sequence)
                    continue  # Skip the rest of this iteration and start over

            # After adding the token, check for disallowed sequences
            if self.antislop_enabled:
                antislop_result = self.slop_phrase_handler.deslop(generated_sequence, self.prompt_length)
                if antislop_result != False:
                    generated_sequence = antislop_result
                    current_position = len(generated_sequence)
                    continue



        # Final display of the generated text
        final_text = self.tokenizer.decode(generated_sequence[self.prompt_length:], skip_special_tokens=False)
        if self.inference_output:
            with self.inference_output:
                self.inference_output.clear_output(wait=True)
                display(HTML(f"<div style='white-space: pre-wrap;'>{final_text}</div>"))
        yield generated_sequence

        # Clear variables to free up memory
        del next_token_logits, filtered_logits
        torch.cuda.empty_cache()


    def _filter_probs(self, probs: torch.FloatTensor, top_k: int, top_p: float, min_p: float) -> torch.FloatTensor:
        # Make a copy of the probabilities to ensure we do not modify the original tensor
        probs = probs.clone()

        # Apply min_p filtering
        if min_p is not None:
            top_prob, _ = torch.max(probs, dim=-1)
            scaled_min_p = min_p * top_prob
            probs = torch.where(probs < scaled_min_p, 0, probs)

        if top_k is not None and top_k > 0:
            top_k = min(top_k, probs.size(-1))
            top_k_probs, _ = torch.topk(probs, top_k)
            min_top_k = top_k_probs[:, -1].unsqueeze(-1)
            probs = torch.where(probs < min_top_k, 0, probs)

        if top_p is not None and top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=1, index=sorted_indices, src=sorted_indices_to_remove
            )
            probs = probs.masked_fill(indices_to_remove, 0)

        return probs

    def _display_debug(self, message: str):
        """
        Displays debug information in the debug_output widget.
        """
        if self.debug_output:
            with self.debug_output:
                self.debug_output.clear_output(wait=True)
                display(HTML(f"<pre>{message}</pre>"))
        else:
            print(message)

    def _clear_gpu_memory_async(self):
        def clear_gpu_memory():        
            torch.cuda.empty_cache()            

        # Create and start the daemon thread
        cleaner_thread = threading.Thread(target=clear_gpu_memory, daemon=True)
        cleaner_thread.start()

        # Return immediately without waiting for the thread
        return

def chat_antislop(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    messages: List[Dict[str, str]],
    max_length: int = None,
    max_new_tokens: int = None,
    temperature: float = 1.0,
    top_k: int = None,
    top_p: float = None,
    min_p: float = None,
    slop_phrase_prob_adjustments: Dict[str, float] = None,
    adjustment_strength: float = 1.0,  # 1.0 is no change from the provided adjustment factors.
    device: torch.device = torch.device('cuda'),
    streaming: bool = False,
    slow_debug: bool = False,  # Add slow_debug argument for debugging
    output_every_n_tokens: int = 1,  # Control how frequently the output is updated
    debug_delay: float = 2.0,  # Delay for slow debugging mode
    inference_output: Output = None,  # For visualization during generation
    debug_output: Output = None,  # For visualization of debug information
    enforce_json: bool = False,
    antislop_enabled: bool = True,
):
    """
    Generates a chat response while avoiding overrepresented phrases (slop) with debugging features.
    This method creates a generator or a non-streamed output, depending on the streaming flag.

    Args:
        model (PreTrainedModel): The language model.
        tokenizer (PreTrainedTokenizer): The tokenizer.
        messages (List[Dict[str, str]]): The list of messages in the conversation.
        max_length (int, optional): The maximum length of the generated text (including the prompt).
        max_new_tokens (int, optional): The maximum number of new tokens to generate.
        temperature (float): Sampling temperature.
        top_k (int): Top-k filtering.
        top_p (float): Top-p (nucleus) filtering.
        min_p (float): Minimum probability filtering.
        slop_phrase_prob_adjustments (Dict[str, float], optional): Dictionary of target words with their respective probability adjustment factor.
        adjustment_strength (float, optional): Strength of the downregulation adjustment.
        device (torch.device, optional): The device to run the model on.
        streaming (bool, optional): Whether to yield tokens as they are generated.
        slow_debug (bool, optional): Enables slow debug mode when set to True.
        output_every_n_tokens (int, optional): Frequency of updating the inference output display.
        debug_delay (float, optional): Time in seconds to pause during slow debug steps.
        inference_output (Output, optional): For visualization during generation.
        debug_output (Output, optional): For visualization of debug information.

    Returns:
        Union[Generator[str, None, None], List[int]]:
            If streaming is True, yields generated text chunks.
            If streaming is False, returns a list of generated token IDs.
    """

    # Build the prompt using the provided messages
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return generate_antislop(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        slop_phrase_prob_adjustments=slop_phrase_prob_adjustments,
        adjustment_strength=adjustment_strength,
        device=device,
        slow_debug=slow_debug,
        output_every_n_tokens=output_every_n_tokens,
        debug_delay=debug_delay,
        inference_output=inference_output,
        debug_output=debug_output,
        antislop_enabled=antislop_enabled,
        enforce_json=enforce_json,
        streaming=streaming,
    )
    

def generate_antislop(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_length: int = None,
    max_new_tokens: int = None,
    temperature: float = 1.0,
    top_k: int = None,
    top_p: float = None,
    min_p: float = None,
    slop_phrase_prob_adjustments: Dict[str, float] = None,
    adjustment_strength: float = 1.0,
    device: torch.device = torch.device('cuda'),
    streaming: bool = False,
    slow_debug: bool = False,  # Added slow_debug
    output_every_n_tokens: int = 1,
    debug_delay: float = 2.0,
    inference_output: Output = None,
    debug_output: Output = None,
    enforce_json: bool = False,
    antislop_enabled: bool = True,
) -> Union[Generator[str, None, None], List[int]]:
    """
    Wrapper function for generate_antislop that handles both streaming and non-streaming modes.
    """
    # Type checking and validation of input arguments
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    if max_length is not None and not isinstance(max_length, int):
        raise TypeError("max_length must be an integer or None")
    if max_new_tokens is not None and not isinstance(max_new_tokens, int):
        raise TypeError("max_new_tokens must be an integer or None")
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
    if max_length is not None and max_length <= 0:
        raise ValueError("max_length must be positive")
    if max_new_tokens is not None and max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if top_k is not None and top_k < 0:
        raise ValueError("top_k must be positive")
    if top_p is not None and (top_p < 0 or top_p > 1):
        raise ValueError("top_p must be in the range (0, 1]")
    if min_p is not None and (min_p < 0 or min_p > 1):
        print(min_p)
        raise ValueError("min_p must be in the range (0, 1]")
    if adjustment_strength < 0:
        raise ValueError("adjustment_strength must be non-negative")
    
    if not debug_output or not inference_output:
        debug_delay = 0
        slow_debug = False

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
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            slop_phrase_prob_adjustments=slop_phrase_prob_adjustments,
            adjustment_strength=adjustment_strength,
            device=device,
            slow_debug=slow_debug,  # Pass slow_debug to support detailed debug output
            output_every_n_tokens=output_every_n_tokens,
            debug_delay=debug_delay,
            inference_output=inference_output,
            debug_output=debug_output,
            enforce_json=enforce_json,
            antislop_enabled=antislop_enabled,
            streaming=True
        )
    else:
        generated_tokens = []
        for token in _generate_antislop(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            slop_phrase_prob_adjustments=slop_phrase_prob_adjustments,
            adjustment_strength=adjustment_strength,
            device=device,
            slow_debug=slow_debug,  # Pass slow_debug to support detailed debug output
            output_every_n_tokens=output_every_n_tokens,
            debug_delay=debug_delay,
            inference_output=inference_output,
            debug_output=debug_output,
            enforce_json=enforce_json,
            antislop_enabled=antislop_enabled,
            streaming=True  # Always stream internally
        ):
            generated_tokens.append(token)
        return generated_tokens



def _generate_antislop(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_length: int = None,
    max_new_tokens: int = None,
    temperature: float = 1.0,
    top_k: int = None,
    top_p: float = None,
    min_p: float = None,
    slop_phrase_prob_adjustments: Dict[str, float] = None,
    adjustment_strength: float = 1.0,
    device: torch.device = torch.device('cuda'),
    slow_debug: bool = False,  # Added slow_debug
    output_every_n_tokens: int = 1,
    debug_delay: float = 2.0,
    inference_output: 'Output' = None,  # Assuming Output is defined elsewhere
    debug_output: 'Output' = None,
    streaming: bool = False,
    enforce_json: bool = False,
    antislop_enabled: bool = True,
) -> Generator[int, None, None]:
    """
    Generates text while avoiding overrepresented phrases (slop).
    This function is now always a generator with temporal buffering.
    """
    # Precompute starting tokens for the slop phrases
    starting_tokens_lookup = precompute_starting_tokens(tokenizer, slop_phrase_prob_adjustments or {})

    # Initialize the sampler
    sampler = AntiSlopSampler(
        model=model,
        tokenizer=tokenizer,
        slop_phrase_prob_adjustments=slop_phrase_prob_adjustments or {},
        starting_tokens_lookup=starting_tokens_lookup,
        adjustment_strength=adjustment_strength,
        device=device,
        slow_debug=slow_debug,  # Enable slow debugging
        output_every_n_tokens=output_every_n_tokens,
        debug_delay=debug_delay,
        inference_output=inference_output,
        debug_output=debug_output,
        enforce_json=enforce_json,
        antislop_enabled=antislop_enabled,
    )

    # Generate token stream
    token_stream = sampler.generate_stream(
        prompt=prompt,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p
    )

    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    if len(prompt_tokens) == 0:
        print('! prompt is empty')
        return
    buffer_size = sampler.slop_phrase_handler.max_slop_phrase_length + 5

    last_released_position = len(prompt_tokens) - 1

    # Temporal buffering parameters
    ema_alpha = 0.2  # Smoothing factor for EMA
    average_token_time = 0.05  # Initial average token generation time in seconds
    n_temporal_buffer_tokens = 8  # Number of tokens for temporal buffer
    temporal_buffer_time = n_temporal_buffer_tokens * average_token_time  # Total time for temporal buffer
    last_generation_time = time.time()
    token_times = [last_generation_time] * len(prompt_tokens)

    for generated_sequence in token_stream:
        current_time = time.time()

        # update token times
        if len(generated_sequence) <= len(token_times):
            # we backtracked
            token_times = token_times[:len(generated_sequence)-1]
            #token_times.append(time.time())
        #else:
        #    n_new_tokens = len(generated_sequence) - len(token_times)
        #    token_times += [time.time()] * n_new_tokens

        

        if False:
            # Identify new tokens beyond the last released position
            if len(generated_sequence) > last_released_position + 1:
                new_tokens = generated_sequence[last_released_position + 1:]
                num_new_tokens = len(new_tokens)

                # Estimate generation time per token
                generation_time = current_time - last_generation_time
                token_time = generation_time / num_new_tokens if num_new_tokens > 0 else 0

                # Update moving average
                average_token_time = ema_alpha * token_time + (1 - ema_alpha) * average_token_time
                temporal_buffer_time = n_temporal_buffer_tokens * average_token_time
                #print(temporal_buffer_time)

                # Update last_generation_time
                last_generation_time = current_time

            # Handle backtracking (sequence becoming shorter)
            elif len(generated_sequence) < last_released_position + 1:
                # Calculate how many tokens were removed from the buffer
                # Note: last_released_position remains unchanged
                removed_tokens = (last_released_position + 1) - len(generated_sequence)


        # Release tokens based on buffer_size and temporal buffer
        sequence_len = len(generated_sequence)
        while sequence_len - last_released_position > buffer_size:
            # calculate simple moving avg of last n token times
            adjusted_last_released_pos = last_released_position - len(prompt_tokens)
            sma_tokens = token_times[len(prompt_tokens):][adjusted_last_released_pos-n_temporal_buffer_tokens:adjusted_last_released_pos]
            if len(sma_tokens) > 0:
                sma_token_time = (time.time() - sma_tokens[0]) / len(sma_tokens)
            else:
                sma_token_time = 0
            temporal_buffer_time = n_temporal_buffer_tokens * sma_token_time
            print(sma_token_time)

            last_released_position += 1
            token_to_release = generated_sequence[last_released_position]

            # Determine if buffer is over or under the temporal buffer
            n_buffered_tokens = sequence_len - last_released_position - n_temporal_buffer_tokens
            print(n_buffered_tokens)
            buffer_time = n_buffered_tokens * sma_token_time
            print(buffer_time, temporal_buffer_time)
            if buffer_time > temporal_buffer_time:
            #if sequence_len - last_released_position > buffer_size + n_temporal_buffer_tokens:
                # Buffer is over the temporal buffer, release at full rate with min delay
                sleep_time = 0.02
            else:
                print('under buffer', temporal_buffer_time - buffer_time)
                # Buffer is under the temporal buffer, calculate additional delay
                # to maintain the temporal buffer
                # Calculate required time to reach temporal buffer
                required_time = temporal_buffer_time - buffer_time
                # Distribute the required_time over the remaining buffered tokens
                sleep_time = required_time / max(n_buffered_tokens, 1)

                # Ensure that sleep_time is not negative
                sleep_time = max(sleep_time, 0.0)
                #sleep_time = sma_token_time * 2

            # Sleep to smooth the output
            if sleep_time > 0:
                time.sleep(sleep_time)

            token_times.append(time.time())

            # Yield the token
            yield token_to_release
            

            # Update buffered tokens as one token is released
            n_buffered_tokens = sequence_len - last_released_position - n_temporal_buffer_tokens

    # Release any remaining tokens after generation is complete
    if last_released_position < len(generated_sequence) - 1:
        print(len(generated_sequence) - last_released_position, 'to release')
        for tok in generated_sequence[last_released_position + 1:]:
            # Release remaining tokens at full rate with constant delay
            yield tok
            time.sleep(0.02)  # Constant delay as per user's instruction

    del sampler
