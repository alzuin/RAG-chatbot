# utils/aws_clients.py
import os
import aioboto3
from aiobotocore.session import get_session as get_aiobotocore_session

AWS_REGION = os.getenv("AWS_REGION", "eu-west-2")

_aioboto3_session = aioboto3.Session()
_aiobotocore_session = get_aiobotocore_session()

def get_dynamodb_session():
    """
    Returns the pre-initialized aioboto3 session for DynamoDB access.

    Returns:
        aioboto3.Session: The shared asynchronous session instance.
    """
    return _aioboto3_session

def get_bedrock_session():
    """
    Returns the pre-initialized aiobotocore session for Bedrock access.

    Returns:
        aiobotocore.AioSession: The shared asynchronous session instance.
    """
    return _aiobotocore_session

async def get_dynamodb_client():
    """
    Asynchronously creates a DynamoDB client from the configured aioboto3 session.

    Returns:
        aioboto3.client: An async DynamoDB client scoped to the configured region.
    """
    session = get_dynamodb_session()
    return session.client("dynamodb", region_name=AWS_REGION)

async def get_bedrock_runtime_client():
    """
    Asynchronously creates a Bedrock Runtime client from the configured aiobotocore session.

    Returns:
        aiobotocore.client: An async Bedrock Runtime client scoped to the configured region.
    """
    session = get_bedrock_session()
    return session.create_client("bedrock-runtime", region_name=AWS_REGION)
