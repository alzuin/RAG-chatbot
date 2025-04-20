from typing import List, Dict

def is_reply_grounded(reply_text: str, similar_items: List[Dict], must_match: bool = False) -> bool:
    """
    Determines whether an assistant's reply is grounded in known listings.

    This function checks if the reply text references any of the listings provided
    in `similar_items`, either by address or external ID. It also considers certain
    phrases (e.g., clarifying questions) as safe by default.

    Args:
        reply_text (str): The full text of the assistant's response.
        similar_items (List[Dict]): A list of known listings, each with a 'payload'
            containing at least 'external_id' and 'address'.
        must_match (bool, optional): If True, the reply must explicitly reference at
            least one known listing to be considered valid. Defaults to False.

    Returns:
        bool: True if the reply is grounded in known data or is safely neutral;
              False if it appears to hallucinate listings.
    """
    if not similar_items:
        return True  # Nothing to check

    known_ids = {h["payload"]["external_id"].lower() for h in similar_items}
    known_addresses = {h["payload"]["address"].lower() for h in similar_items}

    reply_lower = reply_text.lower()
    # Early check: if it's just a clarifying question, it's safe
    if "would a" in reply_lower and "also work for you" in reply_lower:
        return True

    mentions_known_items = any(addr in reply_lower for addr in known_addresses) or \
                           any(ext_id in reply_lower for ext_id in known_ids)

    # 🔍 Only flag as hallucination if it talks about a listing without referencing a known one
    mentions_possible_listing = any(keyword in reply_lower for keyword in ["listing", "property", "address", "option", "match"])

    if mentions_known_items:
        return True
    elif mentions_possible_listing:
        return not must_match  # allow if we're not forcing strict match
    else:
        return True  # nothing suggests a listing was referenced