"""
json_parser.py — Robust extraction of JSON from LLM responses.

WHY IS THIS NEEDED?
  Even with a perfect system prompt, LLMs occasionally:
    - Wrap JSON in markdown code fences (```json ... ```)
    - Add a preamble like "Here is the extracted data:"
    - Add a postamble like "Let me know if you need anything else."
    - Return slightly malformed JSON (trailing commas, single quotes)
    - Return only some of the required fields

  This module handles ALL of those cases gracefully.
  The API must NEVER crash because of LLM formatting quirks.

PARSING STRATEGY (in order of attempt):
  1. Direct parse      — try json.loads() on the full response (happy path)
  2. Fence stripping   — remove ```json / ``` markers, retry
  3. Regex extraction  — find the first { ... } block in the string
  4. Fallback          — return all-null fields with 0.0 confidence

IMPORTANT:
  This parser NEVER raises exceptions to the caller.
  It always returns a dict — either the parsed data or a safe fallback.
"""

import json
import re
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# REQUIRED FIELDS — used to fill in any missing fields after parsing
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = [
    "vendor_name",
    "amount",
    "currency",
    "date",
    "category",
    "description",
    "invoice_id",
]


def _make_null_field() -> dict:
    """
    Returns a null/zero-confidence field structure.
    Used as the default when a field is missing from LLM output.
    """
    return {"value": None, "confidence": 0.0}


def _make_fallback_response() -> dict:
    """
    Returns a fully null response for all required fields.
    Used when JSON parsing fails completely.
    """
    return {field: _make_null_field() for field in REQUIRED_FIELDS}


def _strip_markdown_fences(text: str) -> str:
    """
    Removes markdown code block markers from LLM output.

    Handles:
      ```json { ... } ```
      ``` { ... } ```
      `{ ... }`

    Args:
        text: Raw LLM response string

    Returns:
        Cleaned string with fences removed
    """
    # Remove ```json ... ``` or ``` ... ``` blocks
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text)
    # Remove single backtick wrapping
    text = re.sub(r"`([^`]*)`", r"\1", text)
    return text.strip()


def _extract_json_block(text: str) -> str | None:
    """
    Uses regex to find the first valid JSON object in a string.

    This handles cases like:
      "Here is the result: { ... } Hope that helps!"
    
    We find the first '{' and the last '}' to extract the JSON block.
    This is intentionally simple — deeply nested extraction isn't needed
    because our LLM output is a single flat/shallow object.

    Args:
        text: Text that may contain a JSON object somewhere inside it

    Returns:
        The extracted JSON substring, or None if no block found
    """
    # Find the first '{' and last '}' — handles the common LLM preamble case
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return None

    return text[start: end + 1]


def _validate_and_normalise_field(field_data: Any, field_name: str) -> dict:
    """
    Validates a single extracted field and normalises it to the expected shape.

    Handles edge cases:
      - LLM returned a string instead of a dict: {"vendor_name": "Swiggy"}
      - LLM omitted confidence: {"value": "Swiggy"}
      - LLM returned confidence as a string: {"confidence": "0.95"}
      - LLM returned confidence > 1.0 or < 0.0: clip to valid range

    Args:
        field_data: Whatever the LLM returned for this field
        field_name: Used for logging only

    Returns:
        A normalised dict with "value" and "confidence" keys
    """
    # If the LLM returned a raw string instead of a dict, wrap it
    if isinstance(field_data, str):
        logger.warning(f"Field '{field_name}' was a string, not a dict — wrapping")
        return {"value": field_data, "confidence": 0.5}  # moderate confidence for implied value

    # If it's not a dict at all, treat as null
    if not isinstance(field_data, dict):
        logger.warning(f"Field '{field_name}' is unexpected type {type(field_data)} — nulling")
        return _make_null_field()

    # Extract value — default to None
    value = field_data.get("value", None)

    # Normalise value: empty string → None
    if isinstance(value, str) and value.strip() == "":
        value = None

    # Convert non-string values to string (e.g. amount as number: 1250 → "1250")
    if value is not None and not isinstance(value, str):
        value = str(value)

    # Extract and normalise confidence
    raw_confidence = field_data.get("confidence", 0.0)

    try:
        confidence = float(raw_confidence)
    except (TypeError, ValueError):
        logger.warning(f"Field '{field_name}' has unparseable confidence '{raw_confidence}' — defaulting to 0.0")
        confidence = 0.0

    # Clip to valid range [0.0, 1.0]
    confidence = max(0.0, min(1.0, confidence))

    # If value is null but confidence is > 0, that's contradictory — zero it out
    if value is None and confidence > 0.0:
        confidence = 0.0

    return {"value": value, "confidence": confidence}


def parse_llm_response(raw_response: str) -> dict:
    """
    Main entry point. Parses the LLM's raw text response into a clean dict.

    This function NEVER raises an exception. All errors are caught and
    result in a null fallback response.

    Parsing pipeline:
      1. Try direct json.loads() (fastest, works for clean responses)
      2. Strip markdown fences, retry json.loads()
      3. Extract first JSON block with regex, retry json.loads()
      4. Return all-null fallback

    After successful parse:
      - Validates each field's structure
      - Fills in any missing fields with null
      - Ensures all required fields are present

    Args:
        raw_response: The raw string returned by the LLM

    Returns:
        A dict with all REQUIRED_FIELDS, each being {"value": ..., "confidence": ...}
    """
    if not raw_response or not raw_response.strip():
        logger.error("LLM returned empty response — using fallback")
        return _make_fallback_response()

    parsed: dict | None = None

    # ── Attempt 1: Direct parse ──────────────────────────────────────────
    try:
        parsed = json.loads(raw_response)
        logger.debug("JSON parsed directly on first attempt")
    except json.JSONDecodeError:
        pass

    # ── Attempt 2: Strip markdown fences ────────────────────────────────
    if parsed is None:
        cleaned = _strip_markdown_fences(raw_response)
        try:
            parsed = json.loads(cleaned)
            logger.debug("JSON parsed after stripping markdown fences")
        except json.JSONDecodeError:
            pass

    # ── Attempt 3: Regex extraction ──────────────────────────────────────
    if parsed is None:
        json_block = _extract_json_block(raw_response)
        if json_block:
            try:
                parsed = json.loads(json_block)
                logger.debug("JSON parsed via regex block extraction")
            except json.JSONDecodeError:
                pass

    # ── All attempts failed → fallback ───────────────────────────────────
    if parsed is None:
        logger.error(
            f"All JSON parsing attempts failed. "
            f"Raw response snippet: {raw_response[:200]!r}"
        )
        return _make_fallback_response()

    # ── Validate and normalise each field ─────────────────────────────────
    result: dict = {}

    for field in REQUIRED_FIELDS:
        if field in parsed:
            result[field] = _validate_and_normalise_field(parsed[field], field)
        else:
            # LLM omitted this field entirely — fill with null
            logger.warning(f"Field '{field}' missing from LLM response — using null")
            result[field] = _make_null_field()

    return result
