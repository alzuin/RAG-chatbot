# services/content_manager_or.py

from typing import List, Dict, Any
import re
import json
import os
from services.prompt_helpers import format_metadata_context_prompt
from services.prompt_loader import load_and_render_prompt_from_s3

PROMPT_DOMAIN = os.getenv("PROMPT_DOMAIN", "general_information")

def format_history_or(history: List[Dict]) -> List[Dict[str, str]]:
    """
    Converts DynamoDB-stored chat history into the OpenRouter-compatible message format.

    Each history item may include metadata in DynamoDB's attribute format and may contain
    JSON blobs. This function extracts the actual message content, cleans up any trailing
    metadata blocks, and formats it into a list of messages with "role" and "content".

    Args:
        history (List[Dict]): A list of message entries from DynamoDB, where each entry
            may follow the DynamoDB attribute format (e.g., {"S": "value"}).

    Returns:
        List[Dict[str, str]]: A list of message objects formatted for OpenRouter, with keys:
            - "role": One of "user", "assistant", or "system"
            - "content": The cleaned message content
    """
    messages = []

    for item in history:
        # Extract role and message, handling DynamoDB attribute format
        role = item.get("role", {}).get("S", "user") if isinstance(item.get("role"), dict) else str(
            item.get("role", "user"))
        raw_message = item.get("message", {}).get("S", "") if isinstance(item.get("message"), dict) else str(item.get("message", ""))

        # If the message is a JSON blob, try to extract the "reply" field only
        try:
            message_data = json.loads(raw_message)
            content = message_data.get("reply", raw_message)
        except json.JSONDecodeError:
            content = raw_message

        # Strip any lingering inline metadata blocks (just in case)
        content = re.sub(r"\{\s*\"session_id\".*?\}\s*$", "", content, flags=re.DOTALL)

        # If the message includes a JSON block at the end, strip it
        content = re.sub(r"\{\s*\"session_id\".*?\}\s*$", "", content, flags=re.DOTALL)

        # Map 'user' and 'assistant' roles to OpenRouter format
        if role in ["user", "assistant", "system"]:
            messages.append({"role": role, "content": content})

    return messages


def format_similar_items_or(similar_items: List[Dict], field_schema: Dict) -> str:
    """
    Formats a list of similar property items into a readable message for OpenRouter-compatible LLMs.

    The function constructs a user-friendly list of property summaries, using display configuration
    from the schema to determine which fields to include, how to label them, and how to format numbers.
    A final instruction line is added to guide the assistant not to hallucinate other listings.

    Args:
        similar_items (List[Dict]): List of property items, each containing a 'payload' dict with metadata.
        field_schema (Dict): A schema definition containing the "display_fields" list, where each field includes:
            - "key": Field name in the payload
            - "label": Display label
            - "prefix" (optional): String to prepend
            - "suffix" (optional): String to append
            - "format" (optional): Set to "number" to apply number formatting

    Returns:
        str: A formatted string describing each similar item, ready to be injected into an LLM prompt.
    """
    display_fields = field_schema.get("display_fields", [])

    if not similar_items:
        return "There are currently no matching listings."

    lines = [f"There {'is' if len(similar_items) == 1 else 'are'} {len(similar_items)} matching listing{'s' if len(similar_items) > 1 else ''} available:", ""]

    for idx, item in enumerate(similar_items, start=1):
        payload = item.get("payload", {})
        lines.append(f"Property {idx} of {len(similar_items)}:")
        for field in display_fields:
            key = field["key"]
            value = payload.get(key, "Unknown")
            if value != "Unknown" and field.get("format") == "number":
                try:
                    value = f"{int(value):,}"
                except:
                    pass
            if value != "Unknown":
                value = f"{field.get('prefix', '')}{value}{field.get('suffix', '')}"
            lines.append(f"- {field['label']}: {value}")
        lines.append("")

    lines.append("⚠️ The assistant must ONLY reference these listings. Do not invent additional options.")
    return "\n".join(lines)

def build_prompt_or(
        current_message: str,
        history: List[Dict],
        similar_items: List[Dict],
        session_metadata: Dict[str, any] = None
) -> List[Dict[str, str]]:
    """
    Constructs a complete OpenRouter-compatible prompt, combining system context,
    chat history, similar property listings, and the current user input.

    This function is responsible for composing the full message sequence expected
    by the LLM, including:
      - A base system prompt loaded from S3
      - Prior conversation history (excluding previous system prompts)
      - Optional session metadata (e.g. preferences or filters)
      - A formatted list of similar property listings
      - The latest user message

    Args:
        current_message (str): The most recent message from the user.
        history (List[Dict]): Full chat history, typically retrieved from storage (e.g. DynamoDB).
        similar_items (List[Dict]): Listings returned by a vector search for similar properties.
        session_metadata (Dict[str, any], optional): Additional contextual metadata from previous user inputs.

    Returns:
        List[Dict[str, str]]: A list of messages formatted for OpenRouter, preserving order and context.
    """

    # Start building prompt
    messages = []

    system_prompt = load_and_render_prompt_from_s3(domain=PROMPT_DOMAIN, prompt_name="llm_prompt", context_name='llm_context')
    messages.append({"role": "system", "content": system_prompt.strip()})

    # Add past chat history (excluding prior system prompts)
    for msg in format_history_or(history):
        if msg["role"] != "system":
            messages.append(msg)

    # Inject metadata as invisible context to improve assistant response
    if session_metadata:
        grounding = format_metadata_context_prompt(session_metadata)
        messages.append({
            "role": "user",
            "content": grounding
        })

    if similar_items:
        listings_text = format_similar_items_or(similar_items)
        messages.append({
            "role": "user",
            "content": (
                "⚠️ Below is the ONLY set of listings available. You MUST use only these listings when generating your reply. "
                "Do not invent or speculate. If none are suitable, say so clearly and suggest adjusting filters.\n\n"
                f"{listings_text}"
            )
        })
    else:
        messages.append({
            "role": "user",
            "content": "There are currently no listings available that match the user's query. You must not invent any new listings. Wait for further input."
        })

    # Add current user input
    if not messages or messages[-1]["role"] != "user" or messages[-1]["content"] != current_message:
        messages.append({"role": "user", "content": current_message})

    return messages