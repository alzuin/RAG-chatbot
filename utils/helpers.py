# utils/helpers.py

import json
from datetime import datetime, timezone
from typing import Tuple

def utc_now() -> str:
    """
    Returns the current UTC time as an ISO 8601 formatted string.

    Returns:
        str: Current UTC timestamp (e.g., "2025-04-20T12:34:56+00:00").
    """
    return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()

def parse_event(event: dict) -> Tuple[str, str]:
    """
    Extracts the `user_id` and `message` fields from an AWS Lambda event body.

    Args:
        event (dict): The Lambda event, typically passed to a handler function.

    Returns:
        Tuple[str, str]: A tuple containing the user ID and message.

    Raises:
        ValueError: If required fields are missing or the body is not valid JSON.
    """
    try:
        body = json.loads(event.get("body", "{}"))
        user_id = body.get("user_id")
        message = body.get("message")

        if not user_id or not message:
            raise ValueError("Missing user_id or message in request")

        return user_id, message
    except Exception as e:
        raise ValueError(f"Invalid request format: {e}")

def make_response(status_code: int, body: dict) -> dict:
    """
    Formats a standard API Gateway-compatible response.

    Args:
        status_code (int): HTTP status code to return.
        body (dict): Response body as a dictionary.

    Returns:
        dict: Formatted response including headers and serialized JSON body.
    """
    return {
        "statusCode": status_code,
        "body": json.dumps(body),
        "headers": {
            "Content-Type": "application/json"
        }
    }

def has_valid_value(value):
    """
    Checks whether a value is a non-empty, non-whitespace string.

    Args:
        value: Any input to validate.

    Returns:
        bool: True if the value is a valid string with content, False otherwise.
    """
    return isinstance(value, str) and value.strip()
