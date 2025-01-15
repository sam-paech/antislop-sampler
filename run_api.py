import json
import os
import argparse
from typing import List, Dict, Union, Optional, Any, AsyncGenerator, Tuple

from fastapi import FastAPI, HTTPException, Request
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
logging.basicConfig(level=logging.INFO)  # Set to DEBUG for more detailed logs
logger = logging.getLogger(__name__)

# Import your custom antislop_generate module
from src.antislop_generate import chat_antislop, generate_antislop

app = FastAPI(title="AntiSlop OpenAI-Compatible API")

# Global variables to hold the model and tokenizer
model: Optional[PreTrainedModel] = None
tokenizer: Optional[PreTrainedTokenizer] = None
DEFAULT_SLOP_ADJUSTMENTS: Dict[str, float] = {}
device: Optional[torch.device] = None  # Modified to allow dynamic setting

# Variables to store model metadata
model_loaded_time: Optional[int] = None
model_name_loaded: Optional[str] = None

# Define a global asyncio.Lock to enforce single concurrency
import asyncio
lock = asyncio.Lock()

# Define Pydantic models for request and response schemas

class CompletionRequest(BaseModel):
    model: Optional[str] = Field(default=None, description="Model to use for completion")
    prompt: Union[str, List[str]]
    max_tokens: Optional[int] = Field(default=None, ge=1, description="Maximum number of tokens to generate")
    temperature: Optional[float] = Field(default=1.0, ge=0.0, description="Sampling temperature")
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Nucleus sampling probability")
    top_k: Optional[int] = Field(default=None, ge=0, description="Top-K sampling")
    min_p: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Minimum probability threshold")
    stream: Optional[bool] = Field(default=False, description="Whether to stream back partial progress")
    slop_phrases: Optional[List[Tuple[str, float]]] = Field(
        default=None,
        description="List of slop phrases and their adjustment values, e.g., [['a testament to', 0.3], ['tapestry of', 0.1]]"
    )
    adjustment_strength: Optional[float] = Field(default=20.0, ge=0.0, description="Strength of adjustments")
    enforce_json: Optional[bool] = Field(default=False, description="Enforce JSON formatting")
    antislop_enabled: Optional[bool] = Field(default=True, description="Enable AntiSlop functionality")
    regex_bans: Optional[List[str]] = Field(default=False, description="Ban strings matching these regex expressions")


class ChatCompletionMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = Field(default=None, description="Model to use for completion")
    messages: List[ChatCompletionMessage]
    max_tokens: Optional[int] = Field(default=None, ge=1, description="Maximum number of tokens to generate")
    temperature: Optional[float] = Field(default=1.0, ge=0.0, description="Sampling temperature")
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Nucleus sampling probability")
    top_k: Optional[int] = Field(default=None, ge=0, description="Top-K sampling")
    min_p: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Minimum probability threshold")
    stream: Optional[bool] = Field(default=False, description="Whether to stream back partial progress")
    slop_phrases: Optional[List[Tuple[str, float]]] = Field(
        default=None,
        description="List of slop phrases and their adjustment values, e.g., [['a testament to', 0.3], ['tapestry of', 0.1]]"
    )
    adjustment_strength: Optional[float] = Field(default=20.0, ge=0.0, description="Strength of adjustments")
    enforce_json: Optional[bool] = Field(default=False, description="Enforce JSON formatting")
    antislop_enabled: Optional[bool] = Field(default=True, description="Enable AntiSlop functionality")
    regex_bans: Optional[List[str]] = Field(default=False, description="Ban strings matching these regex expressions")


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


# New Pydantic models for /v1/models endpoint

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str
    permission: List[Any] = []
    root: str
    parent: Optional[str] = None


class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


# Utility functions

import uuid
import time
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
            adjustments = dict(json.load(f))
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
    global model_loaded_time, model_name_loaded

    # Set device based on GPU_ID environment variable
    gpu_id = os.getenv("GPU_ID")
    if gpu_id is not None:
        try:
            gpu_id_int = int(gpu_id)
            if torch.cuda.is_available() and gpu_id_int < torch.cuda.device_count():
                device = torch.device(f"cuda:{gpu_id_int}")
            else:
                logger.warning(f"Specified GPU ID {gpu_id_int} is not available. Falling back to default GPU or CPU.")
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except ValueError:
            logger.warning(f"Invalid GPU ID '{gpu_id}'. Falling back to default GPU or CPU.")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load configuration from environment variables
    model_name = os.getenv("MODEL_NAME", "")
    load_in_4bit = os.getenv("LOAD_IN_4BIT", "false").lower() == "true"
    load_in_8bit = os.getenv("LOAD_IN_8BIT", "false").lower() == "true"
    slop_adjustments_file = os.getenv("SLOP_ADJUSTMENTS_FILE", None)

    # Validate mutually exclusive flags
    if load_in_4bit and load_in_8bit:
        logger.error("Cannot set both LOAD_IN_4BIT and LOAD_IN_8BIT. Choose one.")
        raise ValueError("Cannot set both LOAD_IN_4BIT and LOAD_IN_8BIT. Choose one.")

    # Load slop phrase adjustments from file if provided
    try:
        DEFAULT_SLOP_ADJUSTMENTS = load_slop_adjustments(slop_adjustments_file)
        logger.info(f"Loaded {len(DEFAULT_SLOP_ADJUSTMENTS)} slop phrase adjustments.")
    except Exception as e:
        logger.error(f"Failed to load slop adjustments: {e}")
        raise ValueError("Slop adjustments file could not be loaded. Make sure you have the right file path and file structure.")

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
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
            try:
                model.to(device)
            except Exception as e:
                # if an already quantised model is loaded, the model.to(device) will
                # throw a benign error that we can ignore.
                print(e)
            logger.info("Model loaded in 16-bit precision.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise e  # Let FastAPI handle the startup failure

    logger.info("Model and tokenizer loaded successfully.")

    # Store model metadata
    model_loaded_time = current_timestamp()
    model_name_loaded = model_name


# Utility function for streaming responses

def generate_id() -> str:
    return str(uuid.uuid4())


def current_timestamp() -> int:
    return int(time.time())


async def stream_tokens_sync(generator: Any, is_chat: bool = False) -> AsyncGenerator[str, None]:
    """
    Converts a synchronous generator to an asynchronous generator for streaming responses.
    Formats the output to match OpenAI's streaming response format.
    """
    q = queue.Queue()

    def generator_thread():
        try:
            logger.debug("Generator thread started.")
            for token in generator:
                q.put(token)
                logger.debug(f"Token put into queue: {token}")
            q.put(None)  # Signal completion
            logger.debug("Generator thread completed.")
        except Exception as e:
            logger.error(f"Exception in generator_thread: {e}")
            q.put(e)  # Signal exception

    # Start the generator in a separate daemon thread
    thread = threading.Thread(target=generator_thread, daemon=True)
    thread.start()
    logger.debug("Generator thread initiated.")

    try:
        tokens = []
        text = ''
        while True:
            token = await asyncio.to_thread(q.get)
            logger.debug(f"Token retrieved from queue: {token}")

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
                logger.debug("Finished streaming tokens.")
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
                logger.error(f"Exception during token streaming: {token}")
                break  # Exit the loop after handling the error

            # Decode the token to text
            # Note: This is inefficient; we're decoding the whole sequence every time a 
            # new token comes in. This is to handle cases where the tokeniser doesn't 
            # prepend spaces to words. There's probably a better way to do this.
            tokens.append(token)
            full_text = tokenizer.decode(tokens, skip_special_tokens=True)
            new_text = full_text[len(text):]
            text = full_text
            logger.debug(f"Decoded token: {new_text}")

            # Prepare the data in OpenAI's streaming format
            data = {
                "choices": [
                    {
                        "delta": {"content": new_text},
                        "index": 0,
                        "finish_reason": None
                    }
                ]
            }

            # Yield the formatted data as a Server-Sent Event (SSE)
            yield f"data: {json.dumps(data)}\n\n"
            logger.debug("Yielded token to client.")

            # Yield control back to the event loop
            await asyncio.sleep(0)

    except asyncio.CancelledError:
        logger.warning("Streaming task was cancelled by the client.")
    except Exception as e:
        logger.error(f"Unexpected error in stream_tokens_sync: {e}")
    finally:
        logger.debug("Exiting stream_tokens_sync.")


# Endpoint: /v1/completions
@app.post("/v1/completions", response_model=CompletionResponse)
async def completions(request: CompletionRequest, req: Request):
    logger.info("Completion request received, waiting for processing...")

    if request.stream and request.regex_bans:
        raise HTTPException(status_code=500, detail="Streaming cannot be enabled when using regex bans.")
    
    try:
        if model is None or tokenizer is None:
            logger.error("Model and tokenizer are not loaded.")
            raise HTTPException(status_code=500, detail="Model and tokenizer are not loaded.")

        # Use the model specified in the request or default
        used_model = request.model if request.model else model_name_loaded

        # Handle prompt as string or list
        if isinstance(request.prompt, list):
            prompt = "\n".join(request.prompt)
        else:
            prompt = request.prompt

        # Process slop_phrases parameter
        if request.slop_phrases is not None:
            # Convert list of tuples to dictionary
            slop_adjustments = dict(request.slop_phrases)
            logger.debug(f"Slop adjustments provided: {slop_adjustments}")
        else:
            # Use default slop_phrase_prob_adjustments
            slop_adjustments = DEFAULT_SLOP_ADJUSTMENTS.copy()
            logger.debug(f"Using default slop adjustments with {len(slop_adjustments)} entries.")

        if request.stream:
            logger.info("Streaming completion request started.")
            # Streaming response
            generator_source = generate_antislop(
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
                antislop_enabled=True, #request.antislop_enabled,
                ban_slop_first_tokens=False,
                regex_bans=request.regex_bans
            )

            async def streaming_generator():
                async with lock:
                    logger.info("Lock acquired for streaming completion request.")
                    try:
                        async for token in stream_tokens_sync(generator_source, is_chat=False):
                            yield token
                    except Exception as e:
                        logger.error(f"Exception in streaming_generator: {e}")
                    finally:
                        logger.info("Streaming generator completed and lock released.")

            return StreamingResponse(
                streaming_generator(),
                media_type="text/event-stream"
            )

        else:
            logger.info("Non-streaming completion request started.")
            async with lock:
                logger.info("Lock acquired for non-streaming completion request.")
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
                    antislop_enabled=True, #request.antislop_enabled,
                    ban_slop_first_tokens=False,
                    regex_bans=request.regex_bans
                )

            # Decode the tokens
            text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            logger.debug(f"Generated text: {text}")

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
            logger.info("Completion request processing completed.")
            return response

    except Exception as e:
        logger.error(f"Error during completion processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        logger.debug("Exiting /v1/completions endpoint.")


# Endpoint: /v1/chat/completions
@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest, req: Request):
    logger.info("Chat completion request received, waiting for processing...")

    if request.stream and request.regex_bans:
        raise HTTPException(status_code=500, detail="Streaming cannot be enabled when using regex bans.")
    
    try:
        if model is None or tokenizer is None:
            logger.error("Model and tokenizer are not loaded.")
            raise HTTPException(status_code=500, detail="Model and tokenizer are not loaded.")

        # Use the model specified in the request or default
        used_model = request.model if request.model else model_name_loaded

        # Build the prompt from chat messages
        prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in request.messages])
        logger.debug(f"Constructed prompt from messages: {prompt}")

        # Process slop_phrases parameter
        if request.slop_phrases is not None:
            # Convert list of tuples to dictionary
            slop_adjustments = dict(request.slop_phrases)
            logger.debug(f"Slop adjustments provided: {slop_adjustments}")
        else:
            # Use default slop_phrase_prob_adjustments
            slop_adjustments = DEFAULT_SLOP_ADJUSTMENTS.copy()
            logger.debug(f"Using default slop adjustments with {len(slop_adjustments)} entries.")

        if request.stream:
            logger.info("Streaming chat completion request started.")
            # Streaming response
            generator_source = chat_antislop(
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
                antislop_enabled=True, #request.antislop_enabled,
                ban_slop_first_tokens=False,
                regex_bans=request.regex_bans
            )

            async def streaming_generator():
                async with lock:
                    logger.info("Lock acquired for streaming chat completion request.")
                    try:
                        async for token in stream_tokens_sync(generator_source, is_chat=True):
                            yield token
                    except Exception as e:
                        logger.error(f"Exception in streaming_generator: {e}")
                    finally:
                        logger.info("Streaming generator completed and lock released.")

            return StreamingResponse(
                streaming_generator(),
                media_type="text/event-stream"
            )

        else:
            logger.info("Non-streaming chat completion request started.")
            async with lock:
                logger.info("Lock acquired for non-streaming chat completion request.")
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
                    antislop_enabled=True, #request.antislop_enabled,
                    ban_slop_first_tokens=False,
                    regex_bans=request.regex_bans
                )

            # Decode the tokens
            text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            logger.debug(f"Generated chat text: {text}")

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
            logger.info("Chat completion request processing completed.")
            return response

    except Exception as e:
        logger.error(f"Error during chat completion processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        logger.debug("Exiting /v1/chat/completions endpoint.")


# New Endpoint: /v1/models
@app.get("/v1/models", response_model=ModelsResponse)
async def get_models():
    logger.info("Models request received.")
    try:
        if model is None or model_name_loaded is None or model_loaded_time is None:
            logger.error("Model is not loaded.")
            raise HTTPException(status_code=500, detail="Model is not loaded.")

        model_info = ModelInfo(
            id=model_name_loaded,
            created=model_loaded_time,
            owned_by="user",  # Adjust as needed
            permission=[],    # Can be populated with actual permissions if available
            root=model_name_loaded,
            parent=None
        )

        response = ModelsResponse(
            data=[model_info]
        )

        logger.info("Models response prepared successfully.")
        return response

    except Exception as e:
        logger.error(f"Error during models processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        logger.debug("Exiting /v1/models endpoint.")


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
        "--gpu",
        type=int,
        default=None,
        help="GPU ID to load the model on (e.g., 0, 1). Optional."
    )

    args = parser.parse_args()

    # Set environment variables based on parsed arguments
    os.environ["MODEL_NAME"] = args.model
    os.environ["LOAD_IN_4BIT"] = str(args.load_in_4bit)
    os.environ["LOAD_IN_8BIT"] = str(args.load_in_8bit)
    if args.slop_adjustments_file:
        os.environ["SLOP_ADJUSTMENTS_FILE"] = args.slop_adjustments_file
    if args.gpu is not None:
        os.environ["GPU_ID"] = str(args.gpu)

    # Run the app using Uvicorn with single worker and single thread
    uvicorn.run(
        "run_api:app",  # Ensure this matches the filename if different
        host=args.host,
        port=args.port,
        reload=False,
        log_level="info",  # Set to DEBUG for more detailed logs
        timeout_keep_alive=600,  # 10 minutes
        workers=1,  # Single worker to enforce global lock
        loop="asyncio",  # Ensure using asyncio loop
    )


if __name__ == "__main__":
    main()
