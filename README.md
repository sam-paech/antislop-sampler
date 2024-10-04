# AntiSlop Sampler

## Overview

The AntiSlop sampler uses a backtracking mechanism to go back and revise when it encounters a disallowed word or phrase. No more testaments or tapestries or other gpt-slop.

Try the sampler here: [https://colab.research.google.com/drive/11TjqWQCZ8OJBV6Yi2XI1CsOLx0OTtu0t?usp=sharing](https://colab.research.google.com/drive/1Rd3V4AN31cDytfmY9u80rzHXPD_dS6x9?usp=sharing)


Here it is in action (in slow mode so you can see its backtracking & revisions):

https://github.com/user-attachments/assets/aafe267e-adf1-43e6-9622-5e68b08f7fb3

### 2024-10-05 Update

Refactored the code, lots of fixes.

- Now using model.generate with stopping conditions, to generate for multiple tokens instead of just 1 at a time. This is much faster.
- Added a basic JSON validator + enforcement to demonstrate how the sampler can enforce long-range constraints.
- Switch to probs from logits for the cached values, so that down/upregulation works as expected (this was a mistake in the previous implementation).
- Refactored the code for better organisation.

Quick blurb on the JSON validator:

It uses the same backtracking mechanism to retry invalid JSON output. It checks for unintended unescaped quotes in strings, and encourages the model to choose a valid continuation. This is a very common fail mode for JSON outputs. Other kinds of per-token JSON grammars will just terminate the string if they see an unescaped quote, sadly ending the profound thought the LLM was in the middle of expressing. This is better. You can also use it with high temps.

<details>
<summary>### 2024-10-01 Update</summary>

- Squashed vram leaks, fixed bugs. It should work with any transformers model now.
- Support min_p
- Now using slop_phrase_prob_adjustments.json by default, which has a more intuitive probability adjustment per slop phrase (1 == no change; < 1 means probability is reduced by that factor). It looks like this:
```
[
    ["kaleidoscope", 0.5],
    ["symphony", 0.5],
    ["testament to", 0.5],
    ["elara", 0.5],
    ...
]
```
- I discovered the sampler can squash an annoying habit of LLM writing: overuse of antitheses, e.g. `...not x, but y`, simply by downregulating the string `", not"`. Yay! I think there will be a lot of interesting life hacks to be found like this.
- I've made some generate functions (found in `antislop_generate.py`) that you can import to deploy the sampler in your code:

</details>

### chat_antislop
```python
# Chat generation with streaming
messages = [
    {"role": "user", "content": prompt}
]
for token in chat_antislop(
    model=model,
    tokenizer=tokenizer,
    messages=messages,
    max_new_tokens=400,
    temperature=1,
    min_p=0.1,
    # The adjustment_strength param scales how strongly the probability adjustments are applied.
    # A value of 1 means the values in slop_phrase_prob_adjustments (or the defaults) are used unmodified.
    # Reasonable values are 0 (disabled) thru 100+ (effectively banning the list).
    adjustment_strength=100.0,
    # Optional: Provide a list of slop phrases and probability adjustments
    slop_phrase_prob_adjustments=slop_phrase_prob_adjustments,
    enforce_json=False,
    antislop_enabled=True,
    streaming=True
):
    print(tokenizer.decode(token), end='', flush=True)
```

### generate_antislop
```python
# generate without streaming
prompt_with_template = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
generated_text = generate_antislop(
    model=model,
    tokenizer=tokenizer,
    prompt=prompt,
    max_length=300,
    temperature=1,
    min_p=0.1,
    adjustment_strength=100.0,
    slop_phrase_prob_adjustments=slop_phrase_prob_adjustments,
    enforce_json=False,
    antislop_enabled=True,
    streaming=False
)        
print(tokenizer.decode(generated_text))
```

## What this does:

You can give it a list of words & phrases to avoid like "a tapestry of", "a testament to", etc., and it will backtrack and try something else if it hits that phrase. It can handle 1000s of slop phrases since the lookups are fast. The phrases and downregulation amounts are user configurable. Previous approaches have done this with per-token logit biasing; but that's quite ineffective since most slop words & phrases are more than one token, and it impairs output quality if we downregulate all those partial-word tokens. So instead, we wait for the whole phrase to appear in the output, then backtrack and downregulate all the tokens that could have produced the slop phrase, and continue from there.

For the default slop list, we computed a large list of words that are over-represented in LLM output compared to normal human writing. This list is supplemented by a list of over-used phrases that are pet peeves of LLM enthusiasts. During generation, if any of the words & phrases in this list are generated, the sampler reduces the probability of the starting tokens that can lead to that phrase, by the factor specified in the config. This way you can lightly de-slop or strongly de-slop. You can of course also specify your own phrase list & weights.

## Why it's interesting:

Samplers typically work at the token level -- but that doesn't work if want to avoid words/phrases that tokenise to >1 tokens. Elara might tokenise to ["El", "ara"], and we don't want to reduce the probs of everything beginning with "El". So, this approach waits for the whole phrase to appear, then backtracks and reduces the probabilities of all the likely tokens that will lead to that phrase being output. Nobody afaik has tried this before. It should produce better results than instructing the model to avoid words & phrases in the prompt.

* Disclaimers: This is only implemented in Transformers thus far. It is not well optimised. The code has come together over a few days so expect research grade code & possibly bugs.


## What you need to implement this

If you'd like to implement this sampler in something other than transformers, here's what you need:

- A loop to manage the state of the sampler, as it backtracks and needs to refer to past logits that it's cached
- Per-token continuation generation (think: completions, not chat.completions)
- Raw logits
- Ability to bias logits when generating

Unfortunately that rules out most commercial APIs since few let you specify logit biases. For inferencing engines, they will likely be a mixed bag in terms of ease of integration, as most/all samplers work per token without this weird backtracking stuff we're doing here.

If you do implement this sampler in your thing, please let me know about it!

<details>
<summary>GPT-generated details</summary>

The **AntiSlop Language Model Sampler** is an advanced text generation tool designed to enhance the quality and diversity of outputs from language models. It addresses the issue of overused words and phrases (referred to as "GPT slop") that are commonly generated by language models due to their training data biases.

By integrating a custom sampling strategy with dynamic adjustments and backtracking mechanisms, the sampler actively downregulates the probability of generating specified overrepresented words or phrases. This results in more varied and engaging text generation that avoids common clichés and repetitive language patterns.

## Functional Explanation

### Motivation

Language models are trained on vast corpora of text, which often leads them to overproduce certain words or phrases that are statistically more frequent in the training data. This can result in outputs that are:

- **Repetitive**: Frequently using the same expressions.
- **Predictable**: Lacking originality due to overreliance on common phrases.
- **Less Engaging**: Failing to capture the reader's interest with fresh language.

The AntiSlop sampler tackles this problem by implementing a dynamic token adjustment system that:

- Monitors the generated tokens in real-time.
- Detects when an overrepresented word or phrase is about to be generated.
- Adjusts the model's output probabilities to discourage the generation of these overused expressions.
- Allows for controlled backtracking to revise the output when necessary.

### Core Components

#### 1. Overrepresented Words List

A JSON file (`over_represented_words.json`) contains a list of words and phrases identified as overrepresented, along with their respective ratios indicating the degree of overrepresentation.

Example format:

```
[
  ["word1", penalty],
  ["phrase two", penalty],
  ...
]
```

#### 2. Token Sequence Preparation

The sampler preprocesses the passed slop phrases dict to:

- Generate multiple variants (e.g., lowercase, uppercase, capitalized, with leading spaces).
- Tokenize each variant using the model's tokenizer.
- Map the token sequences to adjustment factors found in the dict, which are specified per slop phrase.

#### 3. Starting Tokens Lookup

To efficiently detect when an overrepresented word is being generated, the sampler:

- Precomputes a lookup table of starting token IDs for each token sequence.
- Includes all possible prefixes of the first token to account for subword tokenizations.

#### 4. Dynamic Logit Adjustment

During generation:

- The sampler monitors the tokens being generated.
- If a sequence matching a disallowed word is detected, it:

  - **Checks** to see if we would select the starting token of the disallowed word anyway after its probabilities were downregulated. If not:
  - **Backtracks**: Removes the tokens associated with the overrepresented word from the generated sequence.
  - **Adjusts Logits**: Modifies the model's logits (pre-softmax output probabilities) to downregulate the probability of generating the overrepresented word in subsequent attempts.
  - **Resamples**: Continues generation from the backtracked position, encouraging the model to choose alternative words.

- This process can repeat if the model generates another disallowed word at the same position, resulting in the logit cache at that position being downregulated for additional tokens.

#### 5. Backtracking Mechanism

- When backtracking, the logit cache is used instead of regenerating new logits for the backtracked position, as we may have backtracked to that position before and want to retain the previously downregulated probabilities.

### Technical Workflow

1. **Initialization**:

   - Load the language model and tokenizer.
   - Read the overrepresented words and their penalties.
   - Prepare token sequences and starting tokens lookup.

2. **Generation Loop**:

   - **Token Prediction**:

     - Use the model to predict the next token.
     - Apply temperature scaling and filtering strategies (e.g., top-k, top-p).

   - **Token Sampling**:

     - Sample the next token based on the adjusted probabilities.

   - **Sequence Update**:

     - Append the sampled token to the generated sequence.
     - Update the current position and token counters.

   - **Overrepresented Word Detection**:

     - Check if the recent tokens form a sequence matching any overrepresented word.
     - Utilize the precomputed maximum sequence length for efficient detection.

   - **Adjustment and Backtracking** (if overrepresented word detected):

     - Retrieve the adjustment factor for the detected sequence.
     - Adjust the logits at the position where the sequence started.
     - Backtrack by removing the overrepresented tokens from the sequence.
     - Update the logit cache.
     - Record the downregulation to avoid redundant adjustments.

   - **Termination Conditions**:

     - Check for end-of-sequence tokens.
     - Continue until the maximum length is reached or other stopping criteria are met.

3. **Output**:

   - The final generated text is returned after applying all adjustments.
   - Optional streaming output can display intermediate results during generation.

### Important Notes

- **Overrepresented Words List**: Customize the `slop_phrase_prob_adjustments.json` file to target specific words or phrases relevant to your use case.
- **Adjusting Parameters**:

  - Increasing `adjustment_strength` will more aggressively downregulate overrepresented words but may affect generation fluency.

- **Performance Considerations**:

  - The sampler introduces additional computational overhead due to dynamic adjustments and backtracking.
  - Ensure that your environment has sufficient resources, especially when using large models.

## Technical Details

- **Logit Adjustment**:

  - The adjustment is applied in the logit space before the softmax function.
  - Adjusted logits are calculated as:

    adjusted_logits = logits * (adjustment_factor ** adjustment_strength)

  - This method allows for fine-grained control over the probability distribution without outright masking tokens.

- **Caching Mechanism**:

  - The sampler caches the model's outputs (`logits`) to avoid redundant computations during backtracking and to ensure state is maintained for previously downregulated logits.

- **Tokenization Considerations**:

  - The tokenizer's behavior (e.g., subword tokenization) is accounted for by precomputing all possible prefixes.
  - This ensures that partial matches of overrepresented words are detected early in the generation process.

</details>


## How to Cite

A paper is in the works, hang tight.

```
@misc{paech2024antislop,
      title={antislop-sampler},
      author={Samuel J. Paech},
      year={2024},
      howpublished={\url{https://github.com/sam-paech/antislop-sampler}},
      note={GitHub repository}
}
```