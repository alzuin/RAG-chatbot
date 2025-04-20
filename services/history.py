import os
import logging
from utils.aws_clients import get_dynamodb_client
from utils.helpers import utc_now

TABLE_NAME = os.getenv("DDB_TABLE", "chat-history")
logger = logging.getLogger(__name__)

async def save_message(user_id: str, message: str, role: str = "user", timestamp: str = None):
    """
    Saves a chat message to DynamoDB for a given user, with optional role and timestamp.

    This function wraps the message and metadata in the DynamoDB attribute format
    and asynchronously persists it to the configured table.

    Args:
        user_id (str): Unique identifier for the user/session.
        message (str): The content of the message to be stored.
        role (str, optional): The role of the sender (e.g., "user", "assistant", or "system").
            Defaults to "user".
        timestamp (str, optional): A UTC timestamp in ISO format. If not provided,
            the current UTC time is used.

    Returns:
        None

    Side Effects:
        - Writes an item to DynamoDB.
        - Logs the save operation.
    """
    timestamp = timestamp or utc_now()
    item = {
        "user_id": {"S": user_id},
        "timestamp": {"S": timestamp},
        "role": {"S": role},
        "message": {"S": message}
    }

    async with await get_dynamodb_client() as client:
        await client.put_item(TableName=TABLE_NAME, Item=item)
        logger.info(f"Saved message for {user_id} at {timestamp}")

async def load_history(user_id: str, limit: int = 10):
    """
    Loads the most recent chat history for a given user from DynamoDB.

    This function queries messages associated with a specific user ID,
    sorted by ascending timestamp (oldest to newest), and returns up to the
    specified number of results.

    Args:
        user_id (str): Unique identifier for the user/session.
        limit (int, optional): Maximum number of messages to retrieve. Defaults to 10.

    Returns:
        List[Dict]: A list of DynamoDB items representing the user's message history.

    Side Effects:
        - Logs the number of messages retrieved.
    """
    async with await get_dynamodb_client() as client:
        response = await client.query(
            TableName=TABLE_NAME,
            KeyConditionExpression="user_id = :uid",
            ExpressionAttributeValues={":uid": {"S": user_id}},
            Limit=limit,
            ScanIndexForward=True
        )
        logger.info(f"Loaded {len(response['Items'])} messages for {user_id}")
        return response["Items"]