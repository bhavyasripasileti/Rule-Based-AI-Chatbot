# LLM Field Extraction API

A production-quality REST API that extracts structured fields from unstructured text using a Large Language Model. Built with **FastAPI**, **Groq** (llama-3.3-70b-versatile), and **Pydantic** — deployed on Render.

[![Deploy Status](https://img.shields.io/badge/deployment-Render-46E3B7)](https://render.com)
[![Python](https://img.shields.io/badge/python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688)](https://fastapi.tiangolo.com)

---

## Live Demo

| Endpoint | URL |
|---|---|
| API Base | `https://YOUR-APP-NAME.onrender.com` |
| Interactive Docs | `https://YOUR-APP-NAME.onrender.com/docs` |
| Health Check | `https://YOUR-APP-NAME.onrender.com/health` |

---

## What It Does

Accepts any raw unstructured text (invoice, receipt, email, note) and returns:

- **7 structured fields** extracted by an LLM
- **Per-field confidence scores** (0.0 – 1.0)
- **Automatic review flags** for fields below the 0.75 confidence threshold
- **Consistent JSON responses** — even for garbage input

```bash
curl -X POST https://YOUR-APP.onrender.com/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "Invoice from Swiggy. Order #INV-2024-001. Amount: ₹850. Date: Nov 15 2024."}'
```

```json
{
  "review_required": false,
  "fields": {
    "vendor_name": { "value": "Swiggy",      "confidence": 0.95, "needs_review": false },
    "amount":      { "value": "850",         "confidence": 0.92, "needs_review": false },
    "currency":    { "value": "INR",         "confidence": 0.88, "needs_review": false },
    "date":        { "value": "Nov 15 2024", "confidence": 0.90, "needs_review": false },
    "category":    { "value": "Food",        "confidence": 0.78, "needs_review": false },
    "description": { "value": "Food delivery order", "confidence": 0.82, "needs_review": false },
    "invoice_id":  { "value": "INV-2024-001","confidence": 0.95, "needs_review": false }
  }
}
```

---

## Architecture

```
app/
├── main.py              # FastAPI app, CORS, lifespan startup
├── routes/
│   └── extract.py       # POST /extract — HTTP layer only
├── services/
│   ├── groq_client.py   # Groq SDK wrapper with error handling
│   └── extractor.py     # Orchestration: prompt → LLM → parse → score
├── prompts/
│   └── extraction.py    # All prompt templates (isolated for easy tuning)
├── parsers/
│   └── json_parser.py   # Robust JSON extraction from LLM output
└── models/
    └── schemas.py       # Pydantic request/response contracts
```

**Design principles used:**
- **Single Responsibility** — each file has exactly one job
- **Dependency Inversion** — services depend on abstractions, not the Groq SDK directly
- **Fail-safe parsing** — the JSON parser never raises; it always returns a safe fallback
- **Input validation at the boundary** — Pydantic rejects bad requests before the LLM is called

---

## Prompt Engineering Strategy

The prompt is the most critical component. A well-engineered prompt is the difference between reliable extraction and random output.

### 4-Layer Prompt Design

**Layer 1 — Role definition**
```
"You are a precise, reliable data extraction engine."
```
Giving the model a role anchors its behaviour. Models respond consistently when they have a clear persona to maintain.

**Layer 2 — Explicit output contract**

The prompt shows the *exact* JSON schema the model must return, with types specified. This is similar to function-calling but works with any model — including open-source ones on Groq:
```
"You MUST return ONLY a valid JSON object. The very first character must be '{' and the last '}'."
```

**Layer 3 — Calibrated confidence rules**

Rather than letting the model pick arbitrary confidence scores, we define explicit rules:
```
0.90–0.99 → Value is stated EXPLICITLY and UNAMBIGUOUSLY
0.75–0.89 → Value is present but requires minor inference
0.50–0.74 → Value is implied from context, not stated directly
0.10–0.49 → Very uncertain, multiple plausible interpretations
0.0       → Field is completely absent
```
This produces *realistic* variance in scores — not all 0.95 across every field.

**Layer 4 — Hard anti-hallucination guardrails**
```
"If a field cannot be found, set value to null and confidence to 0.0."
"NEVER invent values."
"Return ONLY raw JSON — no markdown, no prose, no explanation."
```
Explicit negation ("NEVER", "ONLY") measurably reduces LLM hallucination rates.

### Why `temperature=0.0`?

Extraction is not a creative task. We want the *same* output for the *same* input, every time. Temperature 0 forces the model to always choose the highest-probability token — maximising determinism.

---

## Confidence Scoring Strategy

Confidence scores come from the LLM — not from heuristics or regex patterns. This is intentional:

**Why LLM-generated confidence?**
- The LLM has full context of the text when assigning scores
- A regex-based approach couldn't handle paraphrasing, ambiguity, or language variation
- The LLM can reason about uncertainty: "the text says 'around 2000' — that's low confidence for amount"

**How we enforce the threshold:**
The `needs_review` and `review_required` flags are computed by the application code — *not* by the LLM. This ensures:
- The threshold is applied consistently (always exactly 0.75)
- The LLM can't manipulate its own review flags
- The flag logic is testable independently of the LLM

```python
# In extractor.py — deterministic, always correct
needs_review = confidence < CONFIDENCE_THRESHOLD   # 0.75
review_required = any(field.needs_review for field in all_fields)
```

---

## Handling Ambiguous & Garbage Input

The API never crashes on bad input. Here's how each failure mode is handled:

| Input type | What happens |
|---|---|
| Empty string `""` | Rejected by Pydantic at the HTTP layer (422) |
| Whitespace only `"   "` | Rejected by custom validator (422) |
| Garbage text `"asdfgh !!!!"` | LLM returns null fields with 0.0 confidence (200) |
| LLM returns markdown-wrapped JSON | Parser strips fences and retries (200) |
| LLM returns JSON with preamble | Parser extracts the JSON block via regex (200) |
| LLM returns malformed JSON | Parser returns all-null fallback (200) |
| Groq API is down | Returns 503 with structured error message |
| Unexpected crash | Returns 500 with clean message (no traceback exposed) |

---

## Local Setup

### Prerequisites
- Python 3.11+
- A Groq API key ([get one free here](https://console.groq.com/keys))

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/llm-extraction-api.git
cd llm-extraction-api

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# 5. Run the server
uvicorn app.main:app --reload --port 8000
```

### Test it
```bash
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "Receipt from Amazon. Order #AMZ-2024-9981. Total: $29.99. Date: December 1, 2024."}'
```

---

## Running Tests

```bash
pip install pytest pytest-asyncio httpx
pytest tests/ -v
```

All tests use mocked Groq responses — no API key required for testing.

---

## Generating results.json

```bash
python generate_results.py
```

This runs all 5 sample inputs through the live pipeline and saves results to `results.json`. Requires a valid `GROQ_API_KEY` in `.env`.

---

## Deploying to Render

### Step-by-step

1. **Push to GitHub**
   ```bash
   git init && git add . && git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/llm-extraction-api.git
   git push -u origin main
   ```

2. **Create a Render Web Service**
   - Go to [render.com](https://render.com) → New → Web Service
   - Connect your GitHub repo
   - Render auto-detects `render.yaml` — no manual config needed

3. **Add environment variable**
   - In Render dashboard → Environment → Add Variable
   - Key: `GROQ_API_KEY`
   - Value: your Groq API key

4. **Deploy**
   - Click "Deploy" — Render runs `pip install -r requirements.txt` then starts the server
   - Wait ~2 minutes for the first deploy
   - Your API is live at `https://YOUR-APP-NAME.onrender.com`

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | ✅ Yes | Your Groq API key from console.groq.com |

---

## API Reference

### `POST /extract`

Extract structured fields from unstructured text.

**Request body:**
```json
{
  "text": "string (1–10,000 characters)"
}
```

**Response (200):**
```json
{
  "review_required": "boolean",
  "fields": {
    "vendor_name":  { "value": "string|null", "confidence": "float", "needs_review": "boolean" },
    "amount":       { "value": "string|null", "confidence": "float", "needs_review": "boolean" },
    "currency":     { "value": "string|null", "confidence": "float", "needs_review": "boolean" },
    "date":         { "value": "string|null", "confidence": "float", "needs_review": "boolean" },
    "category":     { "value": "string|null", "confidence": "float", "needs_review": "boolean" },
    "description":  { "value": "string|null", "confidence": "float", "needs_review": "boolean" },
    "invoice_id":   { "value": "string|null", "confidence": "float", "needs_review": "boolean" }
  }
}
```

**Error responses:**
- `422` — Request validation failed (empty text, missing body)
- `503` — Groq API unavailable
- `500` — Unexpected internal error

### `GET /health`
Returns `{"status": "ok"}` — used by Render for uptime monitoring.

### `GET /docs`
Interactive Swagger UI for exploring and testing the API.

---

## Tech Stack

| Technology | Version | Purpose |
|---|---|---|
| Python | 3.11 | Language |
| FastAPI | 0.115 | Web framework |
| Uvicorn | 0.32 | ASGI server |
| Pydantic | 2.10 | Data validation |
| Groq SDK | 0.13 | LLM API client |
| python-dotenv | 1.0 | Environment config |
| Render | — | Cloud deployment |

---

## Key Technical Decisions

**Why Groq over OpenAI?**
Groq's LPU hardware delivers ~10x faster inference than GPU-based providers for the same model. For an extraction API where latency matters, this is a significant advantage.

**Why llama-3.3-70b-versatile?**
It's the best open-source model for instruction-following at Groq's current offering — reliably returns structured JSON and follows complex prompt constraints.

**Why not use structured outputs / function calling?**
Groq's API supports JSON mode. We use prompt engineering instead because:
1. It works identically across any model or provider
2. It's more portable if we switch providers
3. The robust parser handles the rare formatting failures

**Why is confidence generated by the LLM rather than computed post-hoc?**
The LLM has full semantic context when generating the score. A regex or rule-based system can only check if a field matches a pattern — it can't reason about whether an extracted value is plausible given surrounding context.
