# services/prompt_loader.py

import os
import boto3
import json
from typing import Dict
from jinja2 import Template

s3 = boto3.client("s3")
BUCKET = os.getenv("PROMPT_S3_BUCKET")

def load_and_render_prompt_from_s3(domain: str, prompt_name: str, context_name: str) -> str:
    """
    Loads a Jinja2 prompt template and its rendering context from S3, then returns the rendered result.

    This function is typically used to load a system prompt (e.g., for an LLM) with dynamic content.
    It expects a Jinja2 `.j2` template and a matching JSON context file in the same domain folder.

    Args:
        domain (str): The domain name, used to locate the files under `domains/{domain}/`.
        prompt_name (str): The base filename (without extension) of the Jinja2 template.
        context_name (str): The base filename (without extension) of the JSON context.

    Returns:
        str: The fully rendered prompt, ready to be injected into an LLM request.

    Raises:
        botocore.exceptions.ClientError: If S3 fails to fetch either object.
        json.JSONDecodeError: If the context file is not valid JSON.
        jinja2.TemplateError: If the template fails to render.
    """
    key_template = f"domains/{domain}/{prompt_name}.j2"
    key_context = f"domains/{domain}/{context_name}.json"

    template_str = s3.get_object(Bucket=BUCKET, Key=key_template)["Body"].read().decode("utf-8")
    context_json = s3.get_object(Bucket=BUCKET, Key=key_context)["Body"].read().decode("utf-8")
    context = json.loads(context_json)

    return Template(template_str).render(**context)

def load_field_schema(domain: str) -> Dict:
    """
    Loads the field schema for a given domain from S3.

    This schema typically defines metadata field labels, display settings,
    classification weights, and other configuration used to drive prompt rendering
    and user metadata handling.

    Args:
        domain (str): The domain name (e.g., "real_estate") used to locate the schema file.

    Returns:
        Dict: A parsed JSON dictionary representing the field schema.

    Raises:
        botocore.exceptions.ClientError: If the schema file does not exist or cannot be fetched.
        json.JSONDecodeError: If the JSON file content is invalid.
    """
    response = s3.get_object(
        Bucket=os.environ["PROMPT_S3_BUCKET"],
        Key=f"domains/{domain}/fields.json"
    )
    return json.loads(response["Body"].read().decode("utf-8"))

