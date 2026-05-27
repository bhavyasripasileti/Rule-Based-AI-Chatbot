"""
extract.py — Route handler for POST /extract.

ROUTE HANDLER RESPONSIBILITIES (and ONLY these):
  1. Receive the HTTP request
  2. Validate input (Pydantic does this automatically via the type hint)
  3. Call the service layer
  4. Handle service-layer errors and convert them to HTTP responses
  5. Return the HTTP response

WHAT THE ROUTE HANDLER DOES NOT DO:
  - Business logic (that's in services/extractor.py)
  - Prompt building (that's in prompts/extraction.py)
  - JSON parsing (that's in parsers/json_parser.py)

This separation means you can unit-test the extractor service
completely independently of FastAPI.

HTTP STATUS CODES USED:
  200 — Successful extraction (even if all fields are null/low-confidence)
  422 — Pydantic validation failure (malformed request body)
  503 — Groq API unavailable or failed
  500 — Unexpected internal error
"""

import logging
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.models.schemas import ExtractionRequest, ExtractionResponse, ErrorResponse
from app.services.extractor import extract_fields
from app.services.groq_client import GroqClientError

logger = logging.getLogger(__name__)

# APIRouter lets us split routes across multiple files and mount them
# in main.py with a prefix (e.g. /api/v1)
router = APIRouter()


@router.post(
    "/extract",
    response_model=ExtractionResponse,
    responses={
        200: {"description": "Successful extraction with confidence scores"},
        422: {"description": "Invalid request body", "model": ErrorResponse},
        503: {"description": "LLM service unavailable", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse},
    },
    summary="Extract structured fields from unstructured text",
    description=(
        "Accepts raw unstructured text and uses an LLM to extract "
        "structured fields with per-field confidence scores. "
        "Fields below the 0.75 confidence threshold are flagged for human review."
    ),
)
async def extract_endpoint(request: ExtractionRequest) -> ExtractionResponse:
    """
    POST /extract — Main extraction endpoint.

    FastAPI automatically:
      - Parses the JSON request body
      - Validates it against ExtractionRequest (Pydantic)
      - Returns 422 with a clear error if validation fails
      - Serialises the ExtractionResponse to JSON on success

    We only need to handle the service-layer errors here.
    """
    try:
        # Call the extraction pipeline
        # request.text is already stripped and validated by Pydantic
        result = await extract_fields(request.text)
        return result

    except GroqClientError as e:
        # The LLM service is down or misconfigured
        # 503 = "Service Unavailable" — appropriate for upstream failures
        logger.error(f"Groq API error on /extract: {e}")
        return JSONResponse(
            status_code=503,
            content=ErrorResponse(
                error="LLM service unavailable",
                detail=str(e),
            ).model_dump(),
        )

    except Exception as e:
        # Catch-all: something unexpected happened
        # Log with full traceback for debugging, but return a clean message to client
        logger.error(f"Unexpected error on /extract: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="Internal server error",
                detail="An unexpected error occurred. Please try again.",
            ).model_dump(),
        )
