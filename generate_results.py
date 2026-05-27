"""
generate_results.py — Runs all sample inputs through the extraction pipeline
and saves the results to results.json.

HOW TO USE:
  1. Make sure your .env file has GROQ_API_KEY set
  2. Run: python generate_results.py
  3. Check results.json in the project root

WHY THIS EXISTS:
  The assessment requires a results.json file showing the API
  working on real inputs. This script:
    - Calls the extraction service DIRECTLY (no HTTP — faster, no server needed)
    - Processes all 5 sample inputs
    - Saves results with metadata (input text, output, sample ID)
    - Handles errors gracefully — one failed sample won't stop the rest

IMPORTANT:
  This script loads the same .env and uses the same pipeline as the live API.
  Results here are authentic — not hand-crafted.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure the project root is on the Python path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from app.services.extractor import extract_fields
from app.services.groq_client import GroqClientError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

SAMPLES_PATH = Path("sample_inputs/samples.json")
OUTPUT_PATH  = Path("results.json")


async def process_sample(sample: dict) -> dict:
    """
    Process one sample and return the result dict.

    Args:
        sample: A dict with id, description, and text keys

    Returns:
        A result dict including the input metadata and extraction output
    """
    sample_id   = sample["id"]
    description = sample["description"]
    text        = sample["text"]

    logger.info(f"Processing: {sample_id}")

    try:
        response = await extract_fields(text)

        return {
            "sample_id":   sample_id,
            "description": description,
            "input_text":  text,
            "output":      response.model_dump(),
            "status":      "success",
        }

    except GroqClientError as e:
        logger.error(f"Groq error on sample '{sample_id}': {e}")
        return {
            "sample_id":   sample_id,
            "description": description,
            "input_text":  text,
            "output":      None,
            "status":      "error",
            "error":       str(e),
        }

    except Exception as e:
        logger.error(f"Unexpected error on sample '{sample_id}': {e}", exc_info=True)
        return {
            "sample_id":   sample_id,
            "description": description,
            "input_text":  text,
            "output":      None,
            "status":      "error",
            "error":       f"Unexpected error: {str(e)}",
        }


async def main() -> None:
    """
    Main async runner — processes all samples and writes results.json.
    """
    # Validate setup
    if not os.getenv("GROQ_API_KEY"):
        logger.error("GROQ_API_KEY is not set. Cannot run without it.")
        sys.exit(1)

    if not SAMPLES_PATH.exists():
        logger.error(f"Samples file not found: {SAMPLES_PATH}")
        sys.exit(1)

    # Load samples
    samples = json.loads(SAMPLES_PATH.read_text())
    logger.info(f"Loaded {len(samples)} samples from {SAMPLES_PATH}")

    # Process all samples (sequentially to avoid rate limits)
    results = []
    for sample in samples:
        result = await process_sample(sample)
        results.append(result)

        # Small delay between API calls to be respectful of rate limits
        await asyncio.sleep(1)

    # Build final output
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model":        "llama-3.3-70b-versatile",
        "provider":     "Groq",
        "total_samples": len(results),
        "successful":   sum(1 for r in results if r["status"] == "success"),
        "failed":       sum(1 for r in results if r["status"] == "error"),
        "results":      results,
    }

    # Write to file
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    logger.info(f"✓ Results written to {OUTPUT_PATH}")
    logger.info(f"  Successful: {output['successful']}/{output['total_samples']}")


if __name__ == "__main__":
    asyncio.run(main())
