import json
import os
import argparse
from typing import List, Dict, Union, Optional, Any, AsyncGenerator, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn
import logging
import threading

import torch
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig  # Ensure this is imported if you use quantization
)

# Set up logging
logging.basicConfig(level=logging.INFO)  # Changed to INFO for cleaner logs
logger = logging.getLogger(__name__)

# Import your custom antislop_generate module
from src.antislop_generate import chat_antislop, generate_antislop

app = FastAPI(title="AntiSlop OpenAI-Compatible API")

# Global variables to hold the model and tokenizer
model: Optional[PreTrainedModel] = None
tokenizer: Optional[PreTrainedTokenizer] = None
DEFAULT_SLOP_ADJUSTMENTS: Dict[str, float] = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define Pydantic models for request and response schemas

class CompletionRequest(BaseModel):
    model: Optional[str] = Field(default=None, description="Model to use for completion")
    prompt: Union[str, List[str]]
    max_tokens: Optional[int] = Field(default=16, ge=1, description="Maximum number of tokens to generate")
    temperature: Optional[float] = Field(default=1.0, ge=0.0, description="Sampling temperature")
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0, description="Nucleus sampling probability")
    top_k: Optional[int] = Field(default=50, ge=0, description="Top-K sampling")
    min_p: Optional[float] = Field(default=0.0, ge=0.0, le=1.0, description="Minimum probability threshold")
    stream: Optional[bool] = Field(default=False, description="Whether to stream back partial progress")
    slop_phrases: Optional[List[Tuple[str, float]]] = Field(
        default=None,
        description="List of slop phrases and their adjustment values, e.g., [['a testament to', 0.3], ['tapestry of', 0.1]]"
    )
    adjustment_strength: Optional[float] = Field(default=1.0, ge=0.0, description="Strength of adjustments")
    enforce_json: Optional[bool] = Field(default=False, description="Enforce JSON formatting")
    antislop_enabled: Optional[bool] = Field(default=True, description="Enable AntiSlop functionality")


class ChatCompletionMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = Field(default=None, description="Model to use for completion")
    messages: List[ChatCompletionMessage]
    max_tokens: Optional[int] = Field(default=16, ge=1, description="Maximum number of tokens to generate")
    temperature: Optional[float] = Field(default=1.0, ge=0.0, description="Sampling temperature")
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0, description="Nucleus sampling probability")
    top_k: Optional[int] = Field(default=50, ge=0, description="Top-K sampling")
    min_p: Optional[float] = Field(default=0.0, ge=0.0, le=1.0, description="Minimum probability threshold")
    stream: Optional[bool] = Field(default=False, description="Whether to stream back partial progress")
    slop_phrases: Optional[List[Tuple[str, float]]] = Field(
        default=None,
        description="List of slop phrases and their adjustment values, e.g., [['a testament to', 0.3], ['tapestry of', 0.1]]"
    )
    adjustment_strength: Optional[float] = Field(default=1.0, ge=0.0, description="Strength of adjustments")
    enforce_json: Optional[bool] = Field(default=False, description="Enforce JSON formatting")
    antislop_enabled: Optional[bool] = Field(default=True, description="Enable AntiSlop functionality")


class CompletionChoice(BaseModel):
    text: str
    index: int
    logprobs: Optional[Any] = None
    finish_reason: Optional[str] = None


class ChatCompletionChoice(BaseModel):
    message: ChatCompletionMessage
    index: int
    finish_reason: Optional[str] = None


class CompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Dict[str, int]


class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Dict[str, int]


# Utility functions

import uuid
import time
import asyncio
import queue  # Import queue for thread-safe communication

def generate_id() -> str:
    return str(uuid.uuid4())


def current_timestamp() -> int:
    return int(time.time())


def load_slop_adjustments(file_path: Optional[str]) -> Dict[str, float]:
    if file_path is None:
        return {}
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Slop phrase adjustments file not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            adjustments = json.load(f)
            if not isinstance(adjustments, dict):
                raise ValueError("Slop phrase adjustments file must contain a JSON object (dictionary).")
            # Ensure all values are floats
            for key, value in adjustments.items():
                adjustments[key] = float(value)
            return adjustments
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from slop phrase adjustments file: {e}")


# Startup event to load model and tokenizer
@app.on_event("startup")
async def load_model_and_tokenizer():
    global model, tokenizer, DEFAULT_SLOP_ADJUSTMENTS, device

    # Load configuration from environment variables
    model_name = os.getenv("MODEL_NAME", "gpt2")
    load_in_4bit = os.getenv("LOAD_IN_4BIT", "false").lower() == "true"
    load_in_8bit = os.getenv("LOAD_IN_8BIT", "false").lower() == "true"
    slop_adjustments_file = os.getenv("SLOP_ADJUSTMENTS_FILE", None)

    # Validate mutually exclusive flags
    if load_in_4bit and load_in_8bit:
        logger.error("Cannot set both LOAD_IN_4BIT and LOAD_IN_8BIT. Choose one.")
        raise ValueError("Cannot set both LOAD_IN_4BIT and LOAD_IN_8BIT. Choose one.")

    # Load slop phrase adjustments from file if provided
    DEFAULT_SLOP_ADJUSTMENTS = load_slop_adjustments(slop_adjustments_file)

    logger.info(f"Using device: {device}")

    # Load tokenizer
    logger.info(f"Loading tokenizer for model '{model_name}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token  # Ensure pad_token is set
        logger.info("Tokenizer loaded.")
    except Exception as e:
        logger.error(f"Error loading tokenizer: {e}")
        raise e

    # Load model with appropriate precision
    logger.info(f"Loading model '{model_name}'...")
    try:
        if load_in_4bit:
            # Configure 4-bit loading
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.float16
            )
            try:
                import bitsandbytes  # Ensure bitsandbytes is installed
            except ImportError:
                logger.error("bitsandbytes is required for 4-bit loading. Install it via 'pip install bitsandbytes'.")
                raise ImportError("bitsandbytes is required for 4-bit loading. Install it via 'pip install bitsandbytes'.")

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto"
            )
            logger.info("Model loaded in 4-bit precision.")
        elif load_in_8bit:
            # Configure 8-bit loading
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
            try:
                import bitsandbytes  # Ensure bitsandbytes is installed
            except ImportError:
                logger.error("bitsandbytes is required for 8-bit loading. Install it via 'pip install bitsandbytes'.")
                raise ImportError("bitsandbytes is required for 8-bit loading. Install it via 'pip install bitsandbytes'.")

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto"
            )
            logger.info("Model loaded in 8-bit precision.")
        else:
            # Load model normally
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
            logger.info("Model loaded in 16-bit precision.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise e  # Let FastAPI handle the startup failure

    logger.info("Model and tokenizer loaded successfully.")


# Utility function for streaming responses
async def stream_tokens_sync(generator: Any, is_chat: bool = False) -> AsyncGenerator[str, None]:
    """
    Converts a synchronous generator to an asynchronous generator for streaming responses.
    Formats the output to match OpenAI's streaming response format.
    """
    q = queue.Queue()

    def generator_thread():
        try:
            for token in generator:
                q.put(token)
            q.put(None)  # Signal completion
        except Exception as e:
            q.put(e)  # Signal exception

    # Start the generator in a separate daemon thread
    thread = threading.Thread(target=generator_thread, daemon=True)
    thread.start()

    while True:
        try:
            # Retrieve the next token from the queue in a non-blocking way
            token = await asyncio.to_thread(q.get)
            if token is None:
                # Send final finish_reason to indicate the end of the stream
                finish_data = {
                    "choices": [
                        {
                            "delta": {},
                            "index": 0,
                            "finish_reason": "stop"
                        }
                    ]
                }
                yield f"data: {json.dumps(finish_data)}\n\n"
                break
            if isinstance(token, Exception):
                # Handle exceptions by sending a finish_reason with 'error'
                error_data = {
                    "choices": [
                        {
                            "delta": {},
                            "index": 0,
                            "finish_reason": "error"
                        }
                    ]
                }
                yield f"data: {json.dumps(error_data)}\n\n"
                raise token  # Re-raise the exception after notifying the client

            # Decode the token to text
            #text = tokenizer.decode([token], skip_special_tokens=True)
            text = tokenizer.decode([token], skip_special_tokens=False)

            # Prepare the data in OpenAI's streaming format
            data = {
                "choices": [
                    {
                        "delta": {"content": text},
                        "index": 0,
                        "finish_reason": None
                    }
                ]
            }

            # Yield the formatted data as a Server-Sent Event (SSE)
            yield f"data: {json.dumps(data)}\n\n"

            # Yield control back to the event loop
            await asyncio.sleep(0)
        except Exception as e:
            logger.error(f"Error in stream_tokens_sync: {e}")
            # Optionally, notify the client about the error
            error_data = {
                "choices": [
                    {
                        "delta": {},
                        "index": 0,
                        "finish_reason": "error"
                    }
                ]
            }
            yield f"data: {json.dumps(error_data)}\n\n"
            break  # Exit the loop after handling the error


# Endpoint: /v1/completions
@app.post("/v1/completions", response_model=CompletionResponse)
async def completions(request: CompletionRequest):
    global model, tokenizer
    try:
        logger.debug(f"Received completion request: {request}")
        logger.debug(f"Current model: {model}")
        logger.debug(f"Current tokenizer: {tokenizer}")

        if model is None or tokenizer is None:
            raise HTTPException(status_code=500, detail="Model and tokenizer are not loaded.")

        # Use the model specified in the request or default
        used_model = request.model if request.model else model.config.name_or_path

        # Handle prompt as string or list
        if isinstance(request.prompt, list):
            prompt = "\n".join(request.prompt)
        else:
            prompt = request.prompt

        # Process slop_phrases parameter
        if request.slop_phrases is not None:
            # Convert list of tuples to dictionary
            slop_adjustments = dict(request.slop_phrases)
        else:
            # Use default slop_phrase_prob_adjustments
            slop_adjustments = DEFAULT_SLOP_ADJUSTMENTS.copy()

        # Call the generate_antislop function
        if request.stream:
            # Streaming response
            generator = generate_antislop(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                min_p=request.min_p,
                slop_phrase_prob_adjustments=slop_adjustments,
                adjustment_strength=request.adjustment_strength,
                device=device,
                streaming=True,
                slow_debug=False,  # Adjust as needed
                output_every_n_tokens=1,
                debug_delay=0.0,
                inference_output=None,
                debug_output=None,
                enforce_json=request.enforce_json,
                antislop_enabled=request.antislop_enabled,
            )

            return StreamingResponse(
                stream_tokens_sync(generator, is_chat=False),
                media_type="text/event-stream"
            )

        else:
            # Non-streaming response
            generated_tokens = generate_antislop(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                min_p=request.min_p,
                slop_phrase_prob_adjustments=slop_adjustments,
                adjustment_strength=request.adjustment_strength,
                device=device,
                streaming=False,
                slow_debug=False,
                output_every_n_tokens=5,
                debug_delay=0.0,
                inference_output=None,
                debug_output=None,
                enforce_json=request.enforce_json,
                antislop_enabled=request.antislop_enabled,
            )

            # Decode the tokens
            text = tokenizer.decode(generated_tokens, skip_special_tokens=False)

            # Create the response
            response = CompletionResponse(
                id=generate_id(),
                object="text_completion",
                created=current_timestamp(),
                model=used_model,
                choices=[
                    CompletionChoice(
                        text=text,
                        index=0,
                        logprobs=None,
                        finish_reason="length" if request.max_tokens else "stop"
                    )
                ],
                usage={
                    "prompt_tokens": len(tokenizer.encode(prompt)),
                    "completion_tokens": len(generated_tokens),
                    "total_tokens": len(tokenizer.encode(prompt)) + len(generated_tokens),
                }
            )
            return response

    except Exception as e:
        logger.error(f"Error in /v1/completions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint: /v1/chat/completions
@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    global model, tokenizer
    try:
        logger.debug(f"Received chat completion request: {request}")

        if model is None or tokenizer is None:
            raise HTTPException(status_code=500, detail="Model and tokenizer are not loaded.")

        # Use the model specified in the request or default
        used_model = request.model if request.model else model.config.name_or_path

        # Build the prompt from chat messages
        # Assuming 'apply_chat_template' is not available, construct prompt manually
        prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in request.messages])

        # Process slop_phrases parameter
        if request.slop_phrases is not None:
            # Convert list of tuples to dictionary
            slop_adjustments = dict(request.slop_phrases)
        else:
            # Use default slop_phrase_prob_adjustments
            slop_adjustments = DEFAULT_SLOP_ADJUSTMENTS.copy()

        if request.stream:
            print('streaming')
            # Streaming response
            generator = chat_antislop(
                model=model,
                tokenizer=tokenizer,
                messages=[msg.dict() for msg in request.messages],
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                min_p=request.min_p,
                slop_phrase_prob_adjustments=slop_adjustments,
                adjustment_strength=request.adjustment_strength,
                device=device,
                streaming=True,
                slow_debug=False,  # Adjust as needed
                output_every_n_tokens=1,
                debug_delay=0.0,
                inference_output=None,
                debug_output=None,
                enforce_json=request.enforce_json,
                antislop_enabled=request.antislop_enabled,
            )

            return StreamingResponse(
                stream_tokens_sync(generator, is_chat=True),
                media_type="text/event-stream"
            )

        else:
            print('not streaming')
            # Non-streaming response
            generated_tokens = chat_antislop(
                model=model,
                tokenizer=tokenizer,
                messages=[msg.dict() for msg in request.messages],
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                min_p=request.min_p,
                slop_phrase_prob_adjustments=slop_adjustments,
                adjustment_strength=request.adjustment_strength,
                device=device,
                streaming=False,
                slow_debug=False,
                output_every_n_tokens=5,
                debug_delay=0.0,
                inference_output=None,
                debug_output=None,
                enforce_json=request.enforce_json,
                antislop_enabled=request.antislop_enabled,
            )

            # Decode the tokens
            text = tokenizer.decode(generated_tokens, skip_special_tokens=False)

            # Create the response
            response = ChatCompletionResponse(
                id=generate_id(),
                object="chat.completion",
                created=current_timestamp(),
                model=used_model,
                choices=[
                    ChatCompletionChoice(
                        message=ChatCompletionMessage(role="assistant", content=text),
                        index=0,
                        finish_reason="length" if request.max_tokens else "stop"
                    )
                ],
                usage={
                    "prompt_tokens": len(tokenizer.encode(prompt)),
                    "completion_tokens": len(generated_tokens),
                    "total_tokens": len(tokenizer.encode(prompt)) + len(generated_tokens),
                }
            )
            return response

    except Exception as e:
        logger.error(f"Error in /v1/chat/completions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Main function to parse arguments and start Uvicorn
def main():
    parser = argparse.ArgumentParser(description="Launch the AntiSlop OpenAI-Compatible API server.")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the model directory or HuggingFace model ID (e.g., 'gpt2')."
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load the model in 4-bit precision (requires appropriate support)."
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load the model in 8-bit precision (requires appropriate support)."
    )
    parser.add_argument(
        "--slop_adjustments_file",
        type=str,
        default=None,
        help="Path to the JSON file containing slop phrase probability adjustments."
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind the server to."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port number to bind the server to."
    )
    parser.add_argument(
        "--no_reload",
        action="store_true",
        help="Disable auto-reloading (useful for production)."
    )

    args = parser.parse_args()

    # Set environment variables based on parsed arguments
    os.environ["MODEL_NAME"] = args.model
    os.environ["LOAD_IN_4BIT"] = str(args.load_in_4bit)
    os.environ["LOAD_IN_8BIT"] = str(args.load_in_8bit)
    if args.slop_adjustments_file:
        os.environ["SLOP_ADJUSTMENTS_FILE"] = args.slop_adjustments_file

    # Run the app using Uvicorn
    uvicorn.run(
        "run_api:app",
        host=args.host,
        port=args.port,
        reload=not args.no_reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()
