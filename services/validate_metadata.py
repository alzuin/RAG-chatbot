import json
import re
import logging
import os
from typing import Dict, Any, List, Optional

from services.openrouter import call_openrouter
from services.prompt_helpers import format_metadata_context_prompt
from services.prompt_loader import load_field_schema, load_and_render_prompt_from_s3

EXTRACTION_MODEL = os.getenv("EXTRACTION_MODEL", "mistralai/mistral-7b-instruct")
PROMPT_DOMAIN = os.getenv("PROMPT_DOMAIN", "general_information")

logger = logging.getLogger(__name__)

# --- Sanitizers ---
def sanitize_range_number(value: Any) -> Optional[str]:
    """
    Sanitizes and formats a string or numeric input into a human-readable currency range.

    This function extracts numeric values (with at least 3 digits), strips commas,
    and formats them into a GBP (£) price string. If two values are found, it returns
    a range (e.g. "£250,000–£300,000"). If one value is found, it returns a single amount
    (e.g. "£250,000"). If no valid numbers are found, returns None.

    Args:
        value (Any): A raw input value, potentially containing a numeric price or range.

    Returns:
        Optional[str]: A formatted price string or None if parsing fails.
    """
    if not value:
        return None
    try:
        str_val = str(value).replace(",", "")
        match = re.findall(r"\d{3,}", str_val)
        if len(match) == 2:
            return f"£{int(match[0]):,}–£{int(match[1]):,}"
        elif len(match) == 1:
            return f"£{int(match[0]):,}"
    except Exception:
        pass
    return None

def sanitize_list(value: Any) -> List[str]:
    """
    Sanitizes input into a list of clean, trimmed strings.

    This utility handles cases where the input may be:
    - A comma-separated string (e.g., "garden, balcony ,garage")
    - An actual list (e.g., ["garden", "balcony", None])
    - Any other type (which results in an empty list)

    Args:
        value (Any): The input to sanitize. Can be a string, list, or other.

    Returns:
        List[str]: A list of non-empty, trimmed string values.
    """
    if isinstance(value, str):
        return [v.strip() for v in value.split(",")]
    elif isinstance(value, list):
        return [str(v).strip() for v in value if v]
    return []

def sanitize_type(value: Any) -> Optional[str]:
    """
    Sanitizes an input representing a single type or category, returning a lowercase string.

    This function handles both string and list input:
    - If a non-empty list is provided, it returns the first item (trimmed and lowercased).
    - If a string is provided, it trims and lowercases it directly.
    - For any other input or empty values, returns None.

    Args:
        value (Any): Input value to sanitize (can be a string or list of strings).

    Returns:
        Optional[str]: A lowercase, trimmed string or None if input is invalid.
    """
    if isinstance(value, list) and value:
        return value[0].strip().lower()
    if isinstance(value, str):
        return value.strip().lower()
    return None

def normalize_choice(value: Any, valid_choices: List[str]) -> Optional[str]:
    """
    Normalizes a string input to match one of the valid choices, case-insensitively.

    This function is useful when user inputs may vary in casing or formatting but should
    be mapped to a predefined list of accepted values.

    Args:
        value (Any): The input value to normalize.
        valid_choices (List[str]): A list of accepted strings to match against.

    Returns:
        Optional[str]: The matching choice from `valid_choices` (with original casing),
                       or None if no match is found.
    """
    if isinstance(value, str):
        for choice in valid_choices:
            if value.strip().lower() == choice.lower():
                return choice
    return None

# Registry of sanitizer functions by field type
FIELD_SANITIZERS = {
    "range_number": sanitize_range_number,
    "list": sanitize_list,
    "type": sanitize_type,
    "string": lambda v: str(v).strip() if isinstance(v, str) else None
}

def validate_metadata(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates and sanitizes raw metadata input based on the configured schema.

    This function:
    - Loads the field schema for the current domain.
    - Applies schema-defined sanitizers for each field type.
    - Optionally normalizes fields with predefined choices.
    - Logs and falls back to raw values if no sanitizer is defined.

    Args:
        raw (Dict[str, Any]): Raw metadata input (e.g., from user input or LLM extraction).

    Returns:
        Dict[str, Any]: Cleaned and validated metadata dictionary ready for persistence or scoring.
    """
    field_schema = load_field_schema(PROMPT_DOMAIN)
    metadata_fields = field_schema.get("metadata_fields", {})

    metadata = {"session_id": raw.get("session_id")}

    for field, config in metadata_fields.items():
        value = raw.get(field)
        field_type = config.get("type", "string")

        if "choices" in config:
            metadata[field] = normalize_choice(value, config["choices"])
        else:
            sanitizer = FIELD_SANITIZERS.get(field_type)
            if sanitizer:
                metadata[field] = sanitizer(value)
            else:
                logger.warning(f"Unknown field type for '{field}': {field_type}")
                metadata[field] = value

    return metadata

def classify_lead(metadata: Dict[str, Any], schema: Dict[str, Any]) -> str:
    """
    Classifies a lead as "Hot", "Warm", or "Cold" based on weighted scoring rules defined in the schema.

    The classification is driven by `metadata_fields` in the schema, where each field may define:
    - Static weights for categorical options
    - Thresholds and scores for numeric ranges (e.g., budgets)

    The function calculates a total score by evaluating all relevant metadata fields, and compares
    it to `lead_score_thresholds` to determine the final classification.

    Args:
        metadata (Dict[str, Any]): Cleaned and validated user metadata (e.g., preferences, budget, etc.).
        schema (Dict[str, Any]): A schema defining field weights, thresholds, and classification cutoffs.

    Returns:
        str: One of "Hot", "Warm", or "Cold", based on the computed score.
    """
    score = 0
    fields = schema.get("metadata_fields", {})

    for key, field_config in fields.items():
        value = metadata.get(key)
        if not value:
            continue

        if "weights" in field_config:
            weights = field_config["weights"]
            if field_config.get("type") == "range_number" and "thresholds" in weights:
                try:
                    match = re.findall(r"\d{3,}", str(value).replace(",", ""))
                    if match:
                        max_val = int(match[-1])
                        for threshold, s in zip(weights["thresholds"], weights["scores"]):
                            if max_val >= threshold:
                                score += s
                except Exception:
                    continue
            else:
                for option, weight in weights.items():
                    if str(value).strip().lower() == option.strip().lower():
                        score += weight
                        break

    thresholds = schema.get("lead_score_thresholds", {"Hot": 3, "Warm": 1, "Cold": 0})
    if score >= thresholds.get("Hot", 3):
        return "Hot"
    elif score >= thresholds.get("Warm", 1):
        return "Warm"
    return "Cold"

async def extract_metadata_from_user_message(
        user_message: str,
        previous_metadata: Dict[str, Any] = None,
        similar_items: List[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Uses an LLM to extract updated metadata from the user's message, with context from
    previous preferences and displayed listings.

    This function:
    - Loads an extraction prompt template from S3
    - Injects previous metadata as grounding
    - Lists shown property IDs to prevent reprocessing
    - Sends the assembled prompt to the LLM and parses the structured output

    Args:
        user_message (str): The current message from the user.
        previous_metadata (Dict[str, Any], optional): Previously confirmed metadata (used for grounding).
        similar_items (List[Dict[str, Any]], optional): List of recently shown property items,
            used to pass their external IDs as context.

    Returns:
        Dict[str, Any]: Extracted metadata fields in structured form, or an empty dictionary on failure.
    """
    grounding = format_metadata_context_prompt(previous_metadata or {})
    shown_ids = [item["payload"]["external_id"] for item in similar_items or []]
    id_list = "\n".join(f"- {id}" for id in shown_ids)

    prompt_template = load_and_render_prompt_from_s3(PROMPT_DOMAIN, "extract_prompt", context_name="extract_context")

    prompt = f"""{prompt_template}

Previously confirmed preferences:
{grounding}

Property IDs shown to the user:
{id_list}

Now extract any updated metadata from the following user message:

User message:
\"\"\"{user_message}\"\"\"
"""
    try:
        raw_json = await call_openrouter(
            messages=[{"role": "system", "content": prompt.strip()}],
            model=EXTRACTION_MODEL,
            temperature=0,
            max_tokens=512
        )

        if not raw_json.strip():
            logger.warning("🛑 Metadata extractor returned empty string.")
            logger.warning(f"Prompt:\n{prompt}")
            return {}

        return json.loads(raw_json)
    except Exception as e:
        logger.warning(f"Failed to extract metadata from user message: {e}")
        return {}
