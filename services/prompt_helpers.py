from typing import Dict, Any
from services.prompt_loader import load_field_schema

def format_metadata_context_prompt(metadata: Dict[str, Any], domain: str = "real_estate") -> str:
    """
    Formats session metadata into a user-friendly, LLM-readable context prompt.

    This prompt is intended to "ground" the assistant by listing previously confirmed
    user preferences (e.g., budget, location, property type). It helps ensure the assistant
    does not contradict or override known preferences unless explicitly told to do so.

    Args:
        metadata (Dict[str, Any]): A dictionary of session metadata (e.g. filters or preferences).
        domain (str, optional): The domain to load the field schema from (defaults to "real_estate").

    Returns:
        str: A formatted string summarizing the user's confirmed preferences, suitable for
             injection as a user message in the LLM prompt.
    """
    if not metadata:
        return ""

    field_schema = load_field_schema(domain)
    label_mapping = field_schema.get("context_labels", {})

    lines = [
        "📌 The user has already confirmed the following preferences in earlier messages. Do NOT change or ignore these unless the user explicitly says otherwise."
    ]

    for key, label in label_mapping.items():
        value = metadata.get(key)
        if value:
            if isinstance(value, list):
                value = ", ".join(value)
            lines.append(f"- {label}: {value}")

    lines.append("\n👉 Continue the conversation using this context and only update fields if the user clearly expresses a change.")
    return "\n".join(lines)