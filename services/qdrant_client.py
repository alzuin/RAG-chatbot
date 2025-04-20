# services/qdrant_client.py
import os
import asyncio
import httpx
import logging
from typing import List, Dict, Optional

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant.internal:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "items")
DEFAULT_SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.2"))

logger = logging.getLogger(__name__)

async def get_similar_items(
        vector: List[float],
        top_k: int = 5,
        filter_payload: Optional[Dict] = None,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> List[Dict]:
    """
    Performs a semantic vector similarity search in Qdrant and returns the top matches.

    This function queries the Qdrant vector database using the provided embedding vector.
    Optionally applies a payload filter and a minimum relevance score threshold.

    Args:
        vector (List[float]): The embedding vector to search with.
        top_k (int, optional): The maximum number of similar items to return. Defaults to 5.
        filter_payload (Optional[Dict], optional): Qdrant-compatible filter to narrow search results.
            Defaults to None.
        score_threshold (float, optional): Minimum similarity score required to include a result.
            Defaults to `DEFAULT_SCORE_THRESHOLD`.

    Returns:
        List[Dict]: A list of matched items, including their metadata (`payload`) and similarity scores.

    Raises:
        httpx.HTTPStatusError: If the request to Qdrant fails or returns an error status.
    """
    payload = {
        "vector": vector,
        "top": top_k,
        "score_threshold": score_threshold,
        "with_payload": True,
    }

    if filter_payload:
        payload["filter"] = filter_payload

    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.post(f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/search", json=payload)
        response.raise_for_status()
        result = response.json()
        logger.info(f"Retrieved similar items: {result.get('result', [])}")
        return result.get("result", [])