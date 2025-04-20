# services/bedrock_client.py
import os
import json
import logging
from utils.aws_clients import get_bedrock_runtime_client

LLM_MODEL_ID = os.getenv("MODEL_ID", "amazon.titan-text-lite-v1")
EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0")
REGION = os.getenv("AWS_REGION", "eu-west-2")

logger = logging.getLogger(__name__)

async def call_bedrock(prompt: str) -> str:
    """
    Sends a prompt to the configured Bedrock-hosted LLM and returns the generated response.

    This function prepares the request body using the Anthropic message format,
    invokes the model asynchronously, and parses the text response.

    Args:
        prompt (str): The user input to be sent to the LLM.

    Returns:
        str: The generated text response from the LLM.

    Raises:
        Exception: Propagates any exception encountered during model invocation.
    """
    async with await get_bedrock_runtime_client() as client:
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1024,
            "temperature": 0.7
        }

        try:
            response = await client.invoke_model(
                modelId=LLM_MODEL_ID,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json"
            )
            response_body = await response["body"].read()
            parsed = json.loads(response_body)
            return parsed["content"][0]["text"].strip()

        except Exception as e:
            logger.exception(f"Error calling Bedrock model: {e}")
            raise

async def get_embedding(text: str) -> list[float]:
    """
    Generates an embedding vector for the given text using a Bedrock-hosted embedding model.

    This function asynchronously sends the input text to the embedding model and
    returns the resulting vector representation.

    Args:
        text (str): The input text to be embedded.

    Returns:
        list[float]: The embedding vector representing the input text.

    Raises:
        Exception: Propagates any exception encountered during the embedding request.
    """
    async with await get_bedrock_runtime_client() as client:
        body = json.dumps({"inputText": text})
        try:
            response = await client.invoke_model(
                modelId=EMBED_MODEL_ID,
                body=body,
                contentType="application/json",
                accept="application/json"
            )
            response_body = await response["body"].read()
            parsed = json.loads(response_body)
            return parsed["embedding"]
        except Exception as e:
            logger.exception("Error fetching embedding from Bedrock")
            raise