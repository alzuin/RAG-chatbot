# services/openrouter.py
import os
import json
import logging
import httpx
from typing import Dict, List, Any

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = os.getenv("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions")
DEFAULT_MODEL = os.getenv("MODEL_ID", "anthropic/claude-3-haiku")
DEFAULT_SITE = os.getenv("OPENROUTER_SITE", "https://propmatchiq.com")

logger = logging.getLogger(__name__)


async def call_openrouter(
        messages: List[Dict[str, str]],
        model: str = DEFAULT_MODEL,
        temperature: float = 0.7,
        max_tokens: int = 1024
) -> str:
    """
    Sends a prompt to the OpenRouter API and returns the assistant's generated response.

    This function uses HTTPX to make an asynchronous POST request to OpenRouter,
    providing a list of messages in chat format, and returns the assistant's reply.

    Args:
        messages (List[Dict[str, str]]): A list of messages, each with "role" and "content",
            formatted for chat-style interaction (e.g., user/assistant/system).
        model (str, optional): The model to use (default is Claude Haiku or other `DEFAULT_MODEL`).
        temperature (float, optional): Sampling temperature to control randomness (0.0 = deterministic,
            higher = more creative). Defaults to 0.7.
        max_tokens (int, optional): Maximum number of tokens to generate in the reply. Defaults to 1024.

    Returns:
        str: The assistant's generated text response.

    Raises:
        ValueError: If the `OPENROUTER_API_KEY` is not set.
        httpx.HTTPStatusError: If the API response indicates an error.
        Exception: For any other unexpected error during the request.
    """
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")

    logger.info(f"Requested model: {model}")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": DEFAULT_SITE,
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    logger.info(f"API request payload: {json.dumps(payload, indent=2)}")


    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            actual_model = result.get("model")

            logger.info(f"OpenRouter API call successful with model: {actual_model}")

            # Extract the assistant's message content
            return result["choices"][0]["message"]["content"].strip()

    except Exception as e:
        logger.exception(f"Error calling OpenRouter API: {e}")
        raise
