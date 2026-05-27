"""
extractor.py — Orchestrates the full extraction pipeline.

THIS IS THE CORE SERVICE. It:
  1. Takes raw text
  2. Builds the LLM prompt
  3. Calls the Groq API
  4. Parses the JSON response
  5. Applies confidence thresholds
  6. Constructs the final Pydantic response model

SINGLE RESPONSIBILITY PATTERN:
  This service ONLY orchestrates — it doesn't do the work itself.
  - Prompts live in app/prompts/extraction.py
  - API call lives in app/services/groq_client.py
  - JSON parsing lives in app/parsers/json_parser.py
  - Schemas live in app/models/schemas.py

  This makes each component independently testable and replaceable.
  (e.g. swap Groq for OpenAI by only changing groq_client.py)
"""

import logging
from app.models.schemas import (
    ExtractionResponse,
    ExtractionFields,
    ExtractedField,
)
from app.prompts.extraction import get_extraction_messages
from app.parsers.json_parser import parse_llm_response
from app.services.groq_client import get_groq_client, GroqClientError

logger = logging.getLogger(__name__)

# ── Confidence threshold ───────────────────────────────────────────────────
# Any field with confidence BELOW this value will be flagged for human review.
# Assessment requirement: 0.75
CONFIDENCE_THRESHOLD = 0.75


def _build_extracted_field(field_data: dict) -> ExtractedField:
    """
    Converts a raw parsed field dict into a typed ExtractedField Pydantic model.

    IMPORTANT: needs_review is computed HERE, not by the LLM.
    We deliberately do NOT ask the LLM to set needs_review.
    Reason: the LLM might set it inconsistently or game the threshold.
    We enforce the rule ourselves, deterministically.

    Args:
        field_data: {"value": ..., "confidence": ...} dict from parser

    Returns:
        A validated ExtractedField instance
    """
    confidence = field_data["confidence"]
    needs_review = confidence < CONFIDENCE_THRESHOLD

    return ExtractedField(
        value=field_data["value"],
        confidence=confidence,
        needs_review=needs_review,
    )


def _build_response(parsed_fields: dict) -> ExtractionResponse:
    """
    Takes the fully parsed + normalised field dict and builds
    the final ExtractionResponse Pydantic model.

    REVIEW_REQUIRED LOGIC:
      We check ALL seven fields. If even ONE has needs_review=True,
      the top-level review_required flag is set to True.
      This is intentionally conservative — human review is cheap
      compared to missing an extraction error.

    Args:
        parsed_fields: Dict with all 7 field names as keys

    Returns:
        A fully validated ExtractionResponse
    """
    # Build each field
    extracted = {
        field_name: _build_extracted_field(field_data)
        for field_name, field_data in parsed_fields.items()
    }

    # Wrap in ExtractionFields model
    fields = ExtractionFields(**extracted)

    # Determine top-level review flag
    review_required = any(
        field.needs_review
        for field in [
            fields.vendor_name,
            fields.amount,
            fields.currency,
            fields.date,
            fields.category,
            fields.description,
            fields.invoice_id,
        ]
    )

    return ExtractionResponse(
        review_required=review_required,
        fields=fields,
    )


async def extract_fields(raw_text: str) -> ExtractionResponse:
    """
    Main entry point for the extraction pipeline.

    ASYNC WHY?
      FastAPI is async-first. Making this async allows the server to
      handle other requests while waiting for the Groq API to respond.
      The Groq SDK is synchronous, but we keep the function signature
      async so it integrates cleanly with FastAPI's async route handlers.

    ERROR HANDLING PHILOSOPHY:
      - GroqClientError is re-raised — the route handler converts it to HTTP 503
      - All other errors are caught and logged — the route handler returns HTTP 500
      - We NEVER let the API return an unstructured Python traceback

    Args:
        raw_text: The cleaned, validated raw text from the request

    Returns:
        ExtractionResponse with all fields, confidence scores, and review flags

    Raises:
        GroqClientError: If the Groq API call fails
        Exception: For unexpected errors (caught by route handler)
    """
    logger.info(f"Starting extraction for text of length {len(raw_text)} chars")

    # Step 1: Build the prompt messages
    messages = get_extraction_messages(raw_text)

    # Step 2: Call the Groq API
    # Note: get_groq_client() returns a singleton — no new connection per request
    client = get_groq_client()

    logger.debug("Sending prompt to Groq API...")
    raw_llm_response = client.complete(messages)
    logger.debug(f"LLM raw response (first 300 chars): {raw_llm_response[:300]!r}")

    # Step 3: Parse the LLM's response into a clean dict
    # This NEVER raises — returns a null fallback on any parse failure
    parsed_fields = parse_llm_response(raw_llm_response)

    # Step 4: Apply confidence thresholds and build the response model
    response = _build_response(parsed_fields)

    # Summarise results for logging (useful for monitoring)
    low_confidence_fields = [
        name for name, field in response.fields.model_dump().items()
        if field["needs_review"]
    ]

    logger.info(
        f"Extraction complete. "
        f"review_required={response.review_required}. "
        f"Low-confidence fields: {low_confidence_fields or 'none'}"
    )

    return response
