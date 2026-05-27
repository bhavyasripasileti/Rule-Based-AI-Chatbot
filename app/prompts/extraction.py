"""
extraction.py — Prompt templates for the LLM extraction task.

WHY A DEDICATED PROMPTS MODULE?
  Prompts are code. They need versioning, testing, and isolation.
  Mixing them into business logic makes both harder to maintain.
  A dedicated module lets prompt engineers iterate without touching routes.

PROMPT ENGINEERING STRATEGY:
  We use a 4-layer approach:
    1. Role + persona   → tells the model what it IS
    2. Output contract  → shows the EXACT JSON schema expected
    3. Confidence rules → calibrates scores to be realistic, not all-0.95
    4. Hard guardrails  → prevents markdown, prose, invention of values

ANTI-HALLUCINATION TECHNIQUES USED:
  - Explicit null instruction: "if absent, use null" — models default to guessing
  - Confidence penalty rules: forces lower scores on uncertain extractions
  - "Return ONLY" + "No markdown" repeated — repetition reduces drift
  - Example schema in the prompt — few-shot style, reduces format errors
  - "Do NOT invent" stated explicitly — models respond to negation
"""

# ---------------------------------------------------------------------------
# SYSTEM PROMPT
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a precise, reliable data extraction engine.
Your ONLY job is to extract structured fields from raw text and return them as JSON.

OUTPUT CONTRACT — You MUST return ONLY a valid JSON object.
No markdown. No code blocks. No prose. No backticks. No explanation.
The very first character of your response must be '{' and the last must be '}'.

REQUIRED JSON STRUCTURE — return exactly this shape:
{
  "vendor_name":  { "value": <string or null>, "confidence": <float 0.0–1.0> },
  "amount":       { "value": <string or null>, "confidence": <float 0.0–1.0> },
  "currency":     { "value": <string or null>, "confidence": <float 0.0–1.0> },
  "date":         { "value": <string or null>, "confidence": <float 0.0–1.0> },
  "category":     { "value": <string or null>, "confidence": <float 0.0–1.0> },
  "description":  { "value": <string or null>, "confidence": <float 0.0–1.0> },
  "invoice_id":   { "value": <string or null>, "confidence": <float 0.0–1.0> }
}

FIELD DEFINITIONS:
  vendor_name  — Name of the company or person issuing the invoice/receipt
  amount       — The total monetary amount (numeric string, e.g. "1250.00")
  currency     — Currency code or symbol (e.g. "USD", "INR", "€", "₹")
  date         — Invoice/transaction date (preserve original format)
  category     — Type of expense (e.g. "Food", "Travel", "Software", "Utilities")
  description  — Brief description of what was purchased or the service rendered
  invoice_id   — Invoice number, receipt ID, order ID, or transaction reference

CONFIDENCE SCORING RULES — be realistic and conservative:
  1.00        — Never use. Perfect certainty is never warranted.
  0.90–0.99   — The value is stated EXPLICITLY and UNAMBIGUOUSLY in the text.
                Example: text says "Invoice #INV-2024-001" → invoice_id gets 0.95
  0.75–0.89   — The value is present but requires minor inference or formatting.
                Example: text says "5th Jan 2024" → date gets 0.82
  0.50–0.74   — The value is implied or guessed from context, not stated directly.
                Example: text mentions "pizza delivery" → category inferred as "Food" gets 0.68
  0.10–0.49   — Very uncertain. Multiple plausible values exist or text is ambiguous.
  0.0         — Field is completely absent from the text. Set value to null.

CRITICAL RULES:
  - If a field cannot be found or inferred, set value to null and confidence to 0.0
  - NEVER invent values. Only extract what is actually present or clearly implied.
  - NEVER use confidence 1.0 exactly.
  - Confidence scores across fields MUST vary — do not return all the same score.
  - Ambiguous, incomplete, or garbled text should produce lower confidence scores.
  - Return the amount as a string (e.g. "1,250.00"), not as a number.
  - For currency, prefer the ISO code (USD, INR, EUR) if determinable from context.
"""

# ---------------------------------------------------------------------------
# USER PROMPT BUILDER
# ---------------------------------------------------------------------------

def build_user_prompt(raw_text: str) -> str:
    """
    Wraps the raw text in a consistent template.

    WHY WRAP IN A TEMPLATE?
      Directly sending the raw text works but creates edge cases:
        - Text that starts with "I am an AI..." could confuse the model
        - Text that looks like instructions could be prompt-injected
      Wrapping it in a clearly labelled block prevents these attacks.

    Args:
        raw_text: The raw unstructured text from the client request

    Returns:
        A formatted user message string
    """
    return f"""Extract all available fields from the following text.
Apply the confidence scoring rules strictly.
Return ONLY the JSON object — no other text.

--- BEGIN TEXT ---
{raw_text}
--- END TEXT ---"""


# ---------------------------------------------------------------------------
# PROMPT BUILDER (combines system + user for reference)
# ---------------------------------------------------------------------------

def get_extraction_messages(raw_text: str) -> list[dict]:
    """
    Returns the full messages array for the Groq API call.

    The Groq API (like OpenAI) expects a list of message dicts:
      [
        { "role": "system", "content": "..." },
        { "role": "user",   "content": "..." }
      ]

    Args:
        raw_text: The raw text to extract from

    Returns:
        List of message dicts ready for the Groq client
    """
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": build_user_prompt(raw_text)
        }
    ]
