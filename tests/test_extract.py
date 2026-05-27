"""
test_extract.py — Tests for the extraction pipeline.

TESTING STRATEGY:
  We test the parsers and services in isolation (unit tests),
  and the API endpoint with FastAPI's TestClient (integration tests).

  We deliberately DO NOT test the Groq API itself in automated tests —
  that would be an expensive, flaky external dependency.
  Instead, we mock the Groq client and test everything around it.

HOW TO RUN:
  pip install pytest pytest-asyncio httpx
  pytest tests/ -v
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.parsers.json_parser import parse_llm_response, _make_fallback_response
from app.main import app


# ── JSON Parser Tests ──────────────────────────────────────────────────────

class TestJsonParser:
    """Unit tests for the JSON parser — the most critical component."""

    def test_parse_clean_json(self):
        """Happy path: clean JSON string from LLM."""
        raw = json.dumps({
            "vendor_name": {"value": "Swiggy", "confidence": 0.95},
            "amount":      {"value": "850.00", "confidence": 0.92},
            "currency":    {"value": "INR",    "confidence": 0.90},
            "date":        {"value": "2024-11-15", "confidence": 0.88},
            "category":    {"value": "Food",   "confidence": 0.85},
            "description": {"value": "Food delivery", "confidence": 0.80},
            "invoice_id":  {"value": "INV-001", "confidence": 0.93},
        })
        result = parse_llm_response(raw)
        assert result["vendor_name"]["value"] == "Swiggy"
        assert result["vendor_name"]["confidence"] == 0.95

    def test_parse_json_with_markdown_fences(self):
        """LLM wrapped response in ```json ... ``` — should be handled."""
        raw = '```json\n{"vendor_name": {"value": "TestCo", "confidence": 0.9}, "amount": {"value": null, "confidence": 0.0}, "currency": {"value": null, "confidence": 0.0}, "date": {"value": null, "confidence": 0.0}, "category": {"value": null, "confidence": 0.0}, "description": {"value": null, "confidence": 0.0}, "invoice_id": {"value": null, "confidence": 0.0}}\n```'
        result = parse_llm_response(raw)
        assert result["vendor_name"]["value"] == "TestCo"

    def test_parse_json_with_preamble(self):
        """LLM added a preamble sentence — should still extract the JSON."""
        raw = 'Here is the extracted data: {"vendor_name": {"value": "ACME", "confidence": 0.88}, "amount": {"value": null, "confidence": 0.0}, "currency": {"value": null, "confidence": 0.0}, "date": {"value": null, "confidence": 0.0}, "category": {"value": null, "confidence": 0.0}, "description": {"value": null, "confidence": 0.0}, "invoice_id": {"value": null, "confidence": 0.0}}'
        result = parse_llm_response(raw)
        assert result["vendor_name"]["value"] == "ACME"

    def test_parse_empty_string(self):
        """Empty LLM response — should return fallback (no crash)."""
        result = parse_llm_response("")
        assert result == _make_fallback_response()

    def test_parse_garbage_input(self):
        """Complete garbage — should return fallback (no crash)."""
        result = parse_llm_response("asdfjkl; nothing here !@#$%")
        assert result == _make_fallback_response()

    def test_missing_fields_filled_with_null(self):
        """LLM returned only 2 of 7 fields — missing ones should be null/0.0."""
        raw = json.dumps({
            "vendor_name": {"value": "TechCo", "confidence": 0.9},
            "amount":      {"value": "100.00", "confidence": 0.85},
            # 5 fields missing
        })
        result = parse_llm_response(raw)
        # All 7 fields must exist
        assert "currency" in result
        assert "date" in result
        assert "category" in result
        assert "description" in result
        assert "invoice_id" in result
        # Missing fields should be null
        assert result["currency"]["value"] is None
        assert result["currency"]["confidence"] == 0.0

    def test_confidence_clipped_to_range(self):
        """LLM returned confidence > 1.0 — should be clipped to 1.0."""
        raw = json.dumps({
            "vendor_name": {"value": "Corp", "confidence": 1.5},
            "amount":      {"value": None, "confidence": -0.3},
            "currency":    {"value": None, "confidence": 0.0},
            "date":        {"value": None, "confidence": 0.0},
            "category":    {"value": None, "confidence": 0.0},
            "description": {"value": None, "confidence": 0.0},
            "invoice_id":  {"value": None, "confidence": 0.0},
        })
        result = parse_llm_response(raw)
        assert result["vendor_name"]["confidence"] == 1.0   # clipped from 1.5
        assert result["amount"]["confidence"] == 0.0         # clipped from -0.3

    def test_empty_string_value_becomes_none(self):
        """LLM returned '' for a value — should be normalised to None."""
        raw = json.dumps({
            "vendor_name": {"value": "", "confidence": 0.5},
            "amount":      {"value": None, "confidence": 0.0},
            "currency":    {"value": None, "confidence": 0.0},
            "date":        {"value": None, "confidence": 0.0},
            "category":    {"value": None, "confidence": 0.0},
            "description": {"value": None, "confidence": 0.0},
            "invoice_id":  {"value": None, "confidence": 0.0},
        })
        result = parse_llm_response(raw)
        assert result["vendor_name"]["value"] is None
        # When value is null, confidence should be zeroed
        assert result["vendor_name"]["confidence"] == 0.0


# ── API Integration Tests ──────────────────────────────────────────────────

class TestExtractEndpoint:
    """Integration tests using FastAPI's TestClient with a mocked Groq client."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    @pytest.fixture
    def mock_groq_response(self):
        """A realistic LLM response for a clear invoice."""
        return json.dumps({
            "vendor_name": {"value": "Swiggy", "confidence": 0.95},
            "amount":      {"value": "850.00", "confidence": 0.92},
            "currency":    {"value": "INR",    "confidence": 0.90},
            "date":        {"value": "2024-11-15", "confidence": 0.88},
            "category":    {"value": "Food",   "confidence": 0.85},
            "description": {"value": "Food delivery", "confidence": 0.80},
            "invoice_id":  {"value": "INV-001", "confidence": 0.93},
        })

    def test_health_check(self, client):
        """Health endpoint should always return 200."""
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_extract_success(self, client, mock_groq_response):
        """Successful extraction returns correct structure."""
        with patch("app.services.extractor.get_groq_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.complete.return_value = mock_groq_response
            mock_get_client.return_value = mock_client

            resp = client.post("/extract", json={"text": "Invoice from Swiggy INV-001"})

        assert resp.status_code == 200
        data = resp.json()
        assert "review_required" in data
        assert "fields" in data
        assert "vendor_name" in data["fields"]
        assert data["fields"]["vendor_name"]["value"] == "Swiggy"
        assert data["fields"]["vendor_name"]["confidence"] == 0.95
        assert data["fields"]["vendor_name"]["needs_review"] == False

    def test_extract_low_confidence_flags_review(self, client):
        """Low-confidence fields should be flagged and review_required=True."""
        low_conf_response = json.dumps({
            "vendor_name": {"value": "Unknown Shop", "confidence": 0.55},
            "amount":      {"value": "2000",          "confidence": 0.60},
            "currency":    {"value": None,             "confidence": 0.0},
            "date":        {"value": None,             "confidence": 0.0},
            "category":    {"value": "Food",          "confidence": 0.65},
            "description": {"value": "team dinner",  "confidence": 0.70},
            "invoice_id":  {"value": None,             "confidence": 0.0},
        })
        with patch("app.services.extractor.get_groq_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.complete.return_value = low_conf_response
            mock_get_client.return_value = mock_client

            resp = client.post("/extract", json={"text": "team dinner ~2000 bucks"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["review_required"] == True
        assert data["fields"]["vendor_name"]["needs_review"] == True

    def test_extract_empty_text_rejected(self, client):
        """Empty text should be rejected with 422 before reaching the LLM."""
        resp = client.post("/extract", json={"text": ""})
        assert resp.status_code == 422

    def test_extract_whitespace_only_rejected(self, client):
        """Whitespace-only text should be rejected with 422."""
        resp = client.post("/extract", json={"text": "   "})
        assert resp.status_code == 422

    def test_extract_missing_text_field_rejected(self, client):
        """Missing 'text' field should be rejected with 422."""
        resp = client.post("/extract", json={"wrong_field": "hello"})
        assert resp.status_code == 422

    def test_extract_groq_error_returns_503(self, client):
        """Groq API failure should return 503, not crash the server."""
        from app.services.groq_client import GroqClientError
        with patch("app.services.extractor.get_groq_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.complete.side_effect = GroqClientError("Rate limit exceeded")
            mock_get_client.return_value = mock_client

            resp = client.post("/extract", json={"text": "Some invoice text"})

        assert resp.status_code == 503
        data = resp.json()
        assert "error" in data

    def test_all_required_fields_present_in_response(self, client):
        """All 7 required fields must always be present, even if null."""
        null_response = json.dumps({
            field: {"value": None, "confidence": 0.0}
            for field in ["vendor_name", "amount", "currency", "date",
                         "category", "description", "invoice_id"]
        })
        with patch("app.services.extractor.get_groq_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.complete.return_value = null_response
            mock_get_client.return_value = mock_client

            resp = client.post("/extract", json={"text": "asdfgh garbage"})

        assert resp.status_code == 200
        fields = resp.json()["fields"]
        for required_field in ["vendor_name", "amount", "currency", "date",
                                "category", "description", "invoice_id"]:
            assert required_field in fields, f"Missing field: {required_field}"
