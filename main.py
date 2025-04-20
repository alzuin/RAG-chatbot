# main.py
import json
import os
import logging
import asyncio
import time

from services.bedrock_client import get_embedding
from services.qdrant_client import get_similar_items
from services.history import save_message, load_history
from services.validate_metadata import extract_metadata_from_user_message
from services.session_metadata import update_and_save_metadata, load_metadata
from services.content_manager_or import build_prompt_or
from services.openrouter import call_openrouter

from utils.helpers import parse_event, make_response, utc_now, has_valid_value
from utils.safety import is_reply_grounded
from utils.aws_clients import get_dynamodb_client

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Track cold start
LAMBDA_COLD_START_TIME = time.time()
COLD_START = True

async def async_handler(event, context):
    """
    AWS Lambda async handler for processing user messages in a real estate chat assistant.

    This function orchestrates the full flow:
    - Logs cold start time (once per container).
    - Parses user input from the event.
    - Loads history and metadata in parallel.
    - Extracts new metadata and computes message embedding in parallel.
    - Updates metadata and determines if listings should be shown.
    - Searches for similar properties via vector similarity.
    - Builds a contextual prompt including preferences and listings.
    - Calls the OpenRouter LLM for a response.
    - Checks for hallucinations in the assistant’s reply.
    - Persists both user and assistant messages to DynamoDB.
    - Returns the final reply in a format compatible with API Gateway.

    Returns:
        dict: JSON response containing the assistant's reply or an error message.
    """
    global COLD_START

    if COLD_START:
        cold_start_duration = time.time() - LAMBDA_COLD_START_TIME
        logger.info(f"❄️ Cold start detected — duration: {cold_start_duration:.2f} seconds")
        COLD_START = False

    logger.info(f"Received event: {json.dumps(event)}")

    lambda_start = asyncio.get_event_loop().time()  # Start full Lambda timer

    try:
        user_id, message = parse_event(event)
        timestamp = utc_now()

        # Measure performance
        timings = {}
        start = asyncio.get_event_loop().time()

        # Start saving user message without waiting
        t0 = asyncio.get_event_loop().time()
        save_user_task = asyncio.create_task(save_message(user_id, message, role="user", timestamp=timestamp))

        # Load history and metadata in parallel
        t0 = asyncio.get_event_loop().time()
        client = await get_dynamodb_client()
        history_task = load_history(user_id)  # This doesn't need DynamoDB client
        metadata_task = load_metadata(user_id, client=client)

        history, session_metadata = await asyncio.gather(history_task, metadata_task)
        timings["load_history_and_metadata"] = asyncio.get_event_loop().time() - t0

        # Get previous similar items from history
        previous_similar_items = []
        for item in reversed(history):
            if item.get("role", {}).get("S") == "assistant":
                try:
                    message_data = json.loads(item.get("message", {}).get("S", "{}"))
                    if "similar_items" in message_data:
                        previous_similar_items = message_data["similar_items"]
                        break
                except json.JSONDecodeError:
                    continue

        # Start metadata extraction and embedding in parallel
        t0 = asyncio.get_event_loop().time()
        new_metadata_task = asyncio.create_task(
            extract_metadata_from_user_message(
                message,
                previous_metadata=session_metadata,
                similar_items=previous_similar_items
            )
        )
        embedding_task = asyncio.create_task(get_embedding(message))

        new_metadata_raw, embedding = await asyncio.gather(new_metadata_task, embedding_task)
        timings["metadata_extraction"] = asyncio.get_event_loop().time() - t0

        # Update and reload metadata
        await update_and_save_metadata(user_id, new_metadata_raw)
        session_metadata = await load_metadata(user_id)
        logger.info(f"Metadata used for injection decision: {json.dumps(session_metadata, indent=2)}")

        inject_similar_items = (
                has_valid_value(session_metadata.get("location")) and
                has_valid_value(session_metadata.get("urgency")) and
                has_valid_value(session_metadata.get("budget"))
        )

        # Get similar items
        t0 = asyncio.get_event_loop().time()
        current_similar_items = await get_similar_items(embedding)
        similar_items = current_similar_items if current_similar_items else previous_similar_items
        timings["retrieval"] = asyncio.get_event_loop().time() - t0

        similar_items_for_prompt = similar_items if inject_similar_items else []
        if not inject_similar_items:
            logger.info("Skipping listing injection into prompt: missing required metadata")

        # Build prompt
        t0 = asyncio.get_event_loop().time()
        prompt = build_prompt_or(message, history, similar_items_for_prompt, session_metadata)
        timings["build_prompt"] = asyncio.get_event_loop().time() - t0
        logger.info("Prompt built for OpenRouter.")

        # LLM response
        t0 = asyncio.get_event_loop().time()
        reply = await call_openrouter(prompt)
        timings["llm_response"] = asyncio.get_event_loop().time() - t0

        if not is_reply_grounded(reply, similar_items):
            logger.warning("⚠️ Assistant reply may contain hallucinated listing!")
            logger.warning(f"Reply: {reply}")
            reply = (
                "I’m sorry — I wasn’t able to find any additional listings that match your request. "
                "Would you like to adjust your preferences?"
            )
        else:
            logger.info(f"Reply: {reply}")

        # Save assistant message
        t0 = asyncio.get_event_loop().time()
        message_data = {
            "reply": reply,
            "similar_items": similar_items
        }
        await asyncio.gather(
            save_user_task,  # ensure user message is saved
            save_message(user_id, json.dumps(message_data), role="assistant")
        )
        timings["save_reply"] = asyncio.get_event_loop().time() - t0

        logger.info(f"⏱️ Timings breakdown (in seconds): {json.dumps(timings, indent=2)}")
        logger.info(f"🚀 Total handler duration (seconds): {(asyncio.get_event_loop().time() - lambda_start):.2f}")

        return make_response(200, {"reply": reply})

    except Exception as e:
        logger.exception("Lambda handler failed")
        return make_response(500, {"error": str(e)})

# Lambda entrypoint

def handler(event, context):
    """
    Lambda entrypoint for synchronous execution.

    Delegates to `async_handler` using asyncio.

    Args:
        event (dict): The Lambda event.
        context (LambdaContext): Runtime context for the invocation.

    Returns:
        dict: JSON-encoded API Gateway-compatible response.
    """
    return asyncio.run(async_handler(event, context))
