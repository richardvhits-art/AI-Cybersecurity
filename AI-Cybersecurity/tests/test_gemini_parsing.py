import pytest

from ai.gemini_client import GeminiUnavailableError, _parse_response, sanitize_alert_for_ai


def test_sanitize_alert_strips_disallowed_fields():
    alert = {
        "id": 1, "prediction": "anomaly", "anomaly_score": 0.8,
        "created_at": "2024-01-01", "status": "open",
    }
    sanitized = sanitize_alert_for_ai(alert)
    assert "id" not in sanitized
    assert "created_at" not in sanitized
    assert sanitized["prediction"] == "anomaly"


def test_parse_response_valid_json():
    raw = """{
        "summary": "test", "why_suspicious": ["a"], "possible_category": "unknown",
        "severity_explanation": "test", "recommended_investigation": ["b"],
        "confidence_note": "note"
    }"""
    parsed = _parse_response(raw)
    assert parsed["summary"] == "test"


def test_parse_response_strips_markdown_fences():
    raw = """```json
    {
        "summary": "test", "why_suspicious": [], "possible_category": "unknown",
        "severity_explanation": "test", "recommended_investigation": [],
        "confidence_note": "note"
    }
    ```"""
    parsed = _parse_response(raw)
    assert parsed["summary"] == "test"


def test_parse_response_invalid_json_raises():
    with pytest.raises(GeminiUnavailableError):
        _parse_response("not valid json {{{")


def test_parse_response_missing_fields_raises():
    raw = '{"summary": "test"}'
    with pytest.raises(GeminiUnavailableError):
        _parse_response(raw)
