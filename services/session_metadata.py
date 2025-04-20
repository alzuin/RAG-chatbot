# services/session_metadata.py

import os
from typing import Dict, Optional, Any
from utils.helpers import utc_now
from services.validate_metadata import validate_metadata, classify_lead
from utils.aws_clients import get_dynamodb_client
from services.prompt_loader import load_field_schema

TABLE_NAME = os.getenv("DDB_METADATA_TABLE", "chat-session-metadata")
PROMPT_DOMAIN = os.getenv("PROMPT_DOMAIN", "general_information")

async def update_and_save_metadata(user_id: str, new_raw_metadata: Dict[str, Any]):
    """
    Validates and merges new metadata with existing user data, reclassifies the lead,
    and saves the updated metadata.

    This function ensures that new user metadata is properly cleaned, merged with
    existing records (giving preference to new values), and re-evaluated for lead
    classification using a schema-driven scoring system.

    Args:
        user_id (str): Unique identifier for the user/session.
        new_raw_metadata (Dict[str, Any]): Raw incoming metadata (e.g., from an LLM or user input)
            that needs validation and integration.

    Side Effects:
        - Writes the updated metadata back to persistent storage.
        - Updates the "lead_classification" field based on the latest metadata and schema.

    Raises:
        Any exceptions from validation, schema loading, or persistence will propagate.
    """
    new_metadata = validate_metadata(new_raw_metadata)
    existing_metadata = await load_metadata(user_id)

    # Merge and prefer new values
    merged = existing_metadata.copy()
    for key, value in new_metadata.items():
        if value is None:
            continue
        if isinstance(value, list):
            merged[key] = list(set(merged.get(key, []) + value))
        else:
            merged[key] = value

    # Recalculate classification
    merged["lead_classification"] = classify_lead(merged,load_field_schema(PROMPT_DOMAIN))

    await save_metadata(user_id, merged)

async def save_metadata(user_id: str, new_metadata: Dict[str, Any]):
    """
    Merges and saves user metadata into DynamoDB, ensuring existing values are preserved
    and new values take precedence.

    This function:
    - Loads the existing metadata for the user.
    - Merges it with the new metadata (lists are merged uniquely, strings overwrite).
    - Cleans up empty/null values.
    - Formats the data to match DynamoDB attribute types.
    - Persists the updated metadata back to the table.

    Args:
        user_id (str): Unique identifier for the user/session.
        new_metadata (Dict[str, Any]): New metadata values to store.

    Side Effects:
        - Writes a merged metadata record to DynamoDB.
        - Overwrites previous record for the same user ID.

    Raises:
        Exception: Any errors from DynamoDB operations will propagate.
    """
    async with await get_dynamodb_client() as client:
        existing = await load_metadata(user_id, client=client)
        merged = existing.copy()

        for key, value in new_metadata.items():
            if value is None or (isinstance(value, str) and value.strip() == ""):
                continue
            if isinstance(value, list):
                merged[key] = list(set(existing.get(key, []) + value))
            else:
                merged[key] = str(value)

        item = {
            "user_id": {"S": user_id},
            "timestamp": {"S": utc_now()}
        }

        for key, value in merged.items():
            if isinstance(value, list) and value:
                item[key] = {"SS": list(map(str, value))}
            elif isinstance(value, str) and value.strip():
                item[key] = {"S": value.strip()}

        await client.put_item(TableName=TABLE_NAME, Item=item)

async def load_metadata(user_id: str, client: Optional[Any] = None) -> Optional[Dict[str, Any]]:
    """
    Loads the most recent metadata record for a given user from DynamoDB.

    This function retrieves the latest metadata entry for the user ID, skipping control fields
    like `user_id` and `timestamp`, and unpacks DynamoDB attribute types into plain Python types.

    Args:
        user_id (str): The unique identifier for the user/session.
        client (Optional[Any]): Optional existing DynamoDB client. If not provided,
            a new one will be created using `get_dynamodb_client()`.

    Returns:
        Optional[Dict[str, Any]]: A dictionary of user metadata with strings and string lists,
        or an empty dictionary if no record is found.

    Raises:
        botocore.exceptions.ClientError: If the query fails.
    """
    async with (client or await get_dynamodb_client()) as client:
        response = await client.query(
            TableName=TABLE_NAME,
            KeyConditionExpression="user_id = :uid",
            ExpressionAttributeValues={":uid": {"S": user_id}},
            Limit=1,
            ScanIndexForward=False
        )
        items = response.get("Items", [])
        if not items:
            return {}

        item = items[0]
        result = {}
        for key, value in item.items():
            if key in ("user_id", "timestamp"):
                continue
            if "S" in value:
                result[key] = value["S"]
            elif "SS" in value:
                result[key] = value["SS"]

        return result