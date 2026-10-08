"""
Module 7b: Gemini AI Explanation Client
-------------------------------------------
Wraps the official google-genai SDK. Gemini receives ONLY structured,
sanitized alert evidence -- never raw sensitive data, never the ability
to override ML predictions.

Fails gracefully: missing API key or API errors never crash the app or
hide the underlying ML alert.
"""

from __future__ import annotations

import json
import logging
import os

from ai.prompts import SYSTEM_PROMPT, build_user_prompt

logger = logging.getLogger("ai.gemini_client")

# Fields Gemini is allowed to see. Anything else (e.g. internal DB ids not
# needed for explanation) is stripped before the request is sent.
ALLOWED_ALERT_FIELDS = [
    "prediction", "anomaly_score", "threat_category", "confidence",
    "risk_score", "severity", "failed_logins", "unique_dest_ports",
    "connection_frequency", "destination_port", "protocol",
    "bytes_sent", "bytes_received", "packet_count", "flow_duration",
]


class GeminiUnavailableError(Exception):
    """Raised when the Gemini explanation cannot be produced, for any reason."""


def sanitize_alert_for_ai(alert_data: dict) -> dict:
    """Keep only fields Gemini actually needs to write an explanation."""
    return {k: v for k, v in alert_data.items() if k in ALLOWED_ALERT_FIELDS and v is not None}


def _get_client():
    """Lazily construct the Gemini client. Import is local so the rest of the
    app can run even if google-genai isn't installed in a minimal environment."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise GeminiUnavailableError(
            "Gemini AI explanation is unavailable because GEMINI_API_KEY is not configured."
        )
    try:
        from google import genai
    except ImportError as e:
        raise GeminiUnavailableError(
            "google-genai package is not installed. Run: pip install google-genai"
        ) from e

    return genai.Client(api_key=api_key)


def generate_security_explanation(alert_data: dict) -> dict:
    """
    Generate a structured explanation for an already-detected alert.
    alert_data: dict of ML-derived evidence (see ALLOWED_ALERT_FIELDS).

    Returns a dict matching the schema in prompts.SYSTEM_PROMPT.
    Raises GeminiUnavailableError on any failure -- callers must catch this
    and still display the underlying ML alert without the explanation.
    """
    sanitized = sanitize_alert_for_ai(alert_data)
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    try:
        client = _get_client()
        user_prompt = build_user_prompt(sanitized)

        response = client.models.generate_content(
            model=model_name,
            contents=user_prompt,
            config={
                "system_instruction": SYSTEM_PROMPT,
                "response_mime_type": "application/json",
            },
        )
        raw_text = response.text
    except GeminiUnavailableError:
        raise
    except Exception as e:  # noqa: BLE001 - any SDK/network error must not crash the app
        logger.error("Gemini API call failed: %s", e)
        raise GeminiUnavailableError(
            "Gemini AI explanation is temporarily unavailable. The ML alert is still valid."
        ) from e

    return _parse_response(raw_text)


def _parse_response(raw_text: str) -> dict:
    """Parse Gemini's JSON response defensively -- strip markdown fences if present."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise GeminiUnavailableError(
            f"Gemini returned a response that could not be parsed as JSON: {e}"
        )

    required_keys = [
        "summary", "why_suspicious", "possible_category",
        "severity_explanation", "recommended_investigation", "confidence_note",
    ]
    missing = [k for k in required_keys if k not in parsed]
    if missing:
        raise GeminiUnavailableError(
            f"Gemini response is missing required fields: {missing}"
        )

    return parsed
