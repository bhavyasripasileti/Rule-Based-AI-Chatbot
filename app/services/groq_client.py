"""
groq_client.py — Wrapper around the official Groq SDK.

WHY A WRAPPER INSTEAD OF USING GROQ DIRECTLY?
  Single Responsibility: the extractor service shouldn't know
  about HTTP errors, rate limits, or API-specific exceptions.
  This wrapper translates all Groq-specific errors into our own
  clean exceptions, making the rest of the code cleaner.

  It also makes testing easier — we can mock this wrapper
  without mocking the entire Groq SDK.

CONFIGURATION:
  - Model: llama-3.3-70b-versatile (best balance of speed + capability)
  - Temperature: 0.0 — we want DETERMINISTIC extraction, not creativity
  - Max tokens: 1024 — enough for 7 fields with confidence scores
  - Timeout: 30s — fail fast rather than hanging indefinitely
"""

import logging
import os
from groq import Groq, APIConnectionError, APIStatusError, RateLimitError

logger = logging.getLogger(__name__)

# ── Model configuration ────────────────────────────────────────────────────

MODEL_NAME    = "llama-3.3-70b-versatile"
TEMPERATURE   = 0.0    # deterministic — same input = same output
MAX_TOKENS    = 1024   # sufficient for our JSON response shape
REQUEST_TIMEOUT = 30   # seconds — fail fast, don't hang


class GroqClientError(Exception):
    """
    Raised when the Groq API call fails for any reason.
    
    Wrapping SDK exceptions in our own type means the rest of the
    application only needs to handle one error type, regardless of
    which specific Groq error occurred.
    """
    pass


class GroqExtractionClient:
    """
    Thin wrapper around the Groq SDK for our extraction use case.

    USAGE:
        client = GroqExtractionClient()
        response_text = client.complete(messages)

    THREAD SAFETY:
        The Groq client is thread-safe and can be shared.
        We create one instance at module load time (singleton pattern).
    """

    def __init__(self) -> None:
        """
        Initialise the Groq client.

        WHY READ API KEY HERE AND NOT AT IMPORT TIME?
          Reading at __init__ time means the error only triggers when
          the service is actually instantiated, not at module import.
          This allows tests to import the module without needing a key.
        """
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise GroqClientError(
                "GROQ_API_KEY environment variable is not set. "
                "Add it to your .env file or Render environment."
            )

        # The Groq client handles connection pooling and retries internally
        self._client = Groq(api_key=api_key)
        logger.info(f"Groq client initialised with model: {MODEL_NAME}")

    def complete(self, messages: list[dict]) -> str:
        """
        Send messages to the Groq API and return the assistant's response text.

        EDGE CASES HANDLED:
          - Network errors      → GroqClientError
          - Rate limit errors   → GroqClientError (with rate limit message)
          - API errors (4xx/5xx)→ GroqClientError
          - Empty response      → GroqClientError
          - Unexpected format   → GroqClientError

        Args:
            messages: List of {"role": ..., "content": ...} dicts

        Returns:
            The raw text content of the assistant's response

        Raises:
            GroqClientError: If the API call fails for any reason
        """
        try:
            logger.debug(f"Sending {len(messages)} messages to Groq API")

            response = self._client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                # stop=None means let the model decide when to stop
            )

            # Validate the response structure
            if not response.choices:
                raise GroqClientError("Groq API returned a response with no choices")

            content = response.choices[0].message.content

            if content is None:
                raise GroqClientError("Groq API returned a null content in message")

            logger.debug(
                f"Groq response received. "
                f"Tokens used: {response.usage.total_tokens if response.usage else 'unknown'}"
            )

            return content

        except RateLimitError as e:
            logger.error(f"Groq rate limit exceeded: {e}")
            raise GroqClientError(
                "Groq API rate limit exceeded. Please try again in a moment."
            ) from e

        except APIConnectionError as e:
            logger.error(f"Groq connection error: {e}")
            raise GroqClientError(
                "Could not connect to Groq API. Check your internet connection."
            ) from e

        except APIStatusError as e:
            logger.error(f"Groq API status error {e.status_code}: {e.message}")
            raise GroqClientError(
                f"Groq API returned an error: {e.status_code} — {e.message}"
            ) from e

        except GroqClientError:
            # Re-raise our own errors without wrapping them again
            raise

        except Exception as e:
            # Catch-all: log and wrap unknown errors
            logger.error(f"Unexpected error calling Groq API: {e}", exc_info=True)
            raise GroqClientError(f"Unexpected error from Groq API: {str(e)}") from e


# ── Module-level singleton ─────────────────────────────────────────────────
# Created once when the module is first imported.
# FastAPI's dependency injection will use this shared instance.
# WHY SINGLETON: avoids reconnecting on every request; Groq client is thread-safe.

_groq_client: GroqExtractionClient | None = None


def get_groq_client() -> GroqExtractionClient:
    """
    Returns the module-level Groq client singleton.
    Creates it on first call (lazy initialisation).

    Using lazy init means startup doesn't fail if GROQ_API_KEY
    isn't set yet (useful during local development / testing).
    """
    global _groq_client
    if _groq_client is None:
        _groq_client = GroqExtractionClient()
    return _groq_client
