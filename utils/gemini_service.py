"""Google Gemini integration using the official `google-genai` SDK.

The model is asked to transcribe only what is visibly printed and to return
null for anything it cannot read. Responses are validated before use.
"""
from __future__ import annotations

import json
import re
from typing import Any

import config
from utils.ocr_service import FIELDS, NOT_DETECTED, STATUS_GEMINI, STATUS_MISSING, empty_fields

PROMPT = """You are reading the packaging of a medicine from a photo.
Transcribe ONLY information that is clearly visible in the image.
Rules:
- Do not guess, infer, or use outside knowledge. If a value is not visible or not legible, use null.
- Copy values exactly as printed (keep units and date formats).
- "raw_text" must contain all legible text you can see, line by line.
Return a single JSON object with exactly these keys:
medicine_name, composition, strength, manufacturer, batch_number, mfg_date,
expiry_date, mrp, pack_quantity, warnings, raw_text, illegible_fields
where illegible_fields is a list of field names that are present but unreadable."""

FIELD_KEYS = [k for k, _ in FIELDS]


def is_configured() -> bool:
    return config.get_gemini_api_key() is not None


def sdk_available() -> bool:
    try:
        from google import genai  # noqa: F401
        return True
    except ImportError:
        return False


def _strip_fences(text: str) -> str:
    return re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()


def parse_gemini_json(text: str) -> dict[str, Any]:
    """Validate a Gemini JSON reply and convert it to the shared field format."""
    data = json.loads(_strip_fences(text))
    if not isinstance(data, dict):
        raise ValueError("Gemini returned JSON that is not an object.")
    fields = empty_fields()
    illegible = set(data.get("illegible_fields") or [])
    for key in FIELD_KEYS:
        val = data.get(key)
        if isinstance(val, list):
            val = "; ".join(str(v) for v in val if v)
        val = (str(val).strip() if val is not None else "")
        if val and val.lower() not in ("null", "none", "n/a", "not visible", "not detected"):
            fields[key].update(value=val, status=STATUS_GEMINI)
        elif key in illegible:
            fields[key].update(value=NOT_DETECTED, status="Present but illegible")
        else:
            fields[key].update(value=NOT_DETECTED, status=STATUS_MISSING)
    raw = data.get("raw_text") or ""
    if isinstance(raw, list):
        raw = "\n".join(map(str, raw))
    return {"fields": fields, "raw_text": str(raw).strip()}


def friendly_error(exc: Exception) -> str:
    msg = str(exc)
    low = msg.lower()
    if "429" in msg or "quota" in low or "resource_exhausted" in low:
        return "Gemini quota exceeded or rate-limited. Wait a minute or use Tesseract in Settings."
    if "401" in msg or "403" in msg or "api key" in low or "permission" in low:
        return "Gemini rejected the API key. Check GEMINI_API_KEY in your .env file."
    if "timeout" in low or "deadline" in low:
        return "Gemini timed out. Check your internet connection and try again."
    if "connect" in low or "network" in low or "resolve" in low:
        return "Could not reach Gemini. Check your internet connection."
    if "404" in msg or "not found" in low:
        return "The configured Gemini model was not found. Change the model name in Settings."
    return f"Gemini request failed: {msg[:200]}"


def extract_medicine_details(image_bytes: bytes, mime_type: str, model: str | None = None) -> dict[str, Any]:
    key = config.get_gemini_api_key()
    if not key:
        return {"ok": False, "error": "Gemini API key is not configured."}
    if not sdk_available():
        return {"ok": False, "error": "google-genai is not installed. Run: python -m pip install google-genai"}
    from google import genai
    from google.genai import types

    import time

    model = model or config.DEFAULT_GEMINI_MODEL
    models = [model]

    if model != "gemini-3.1-flash-lite":
        models.append("gemini-3.1-flash-lite")

    client = genai.Client(
        api_key=key,
        http_options=types.HttpOptions(
            timeout=35000
        )
    )

    text = ""
    selected_model = model
    errors = []

    for current_model in models:
        for attempt in range(1):
            try:
                response = client.models.generate_content(
                    model=current_model,
                    contents=[
                        types.Part.from_bytes(
                            data=image_bytes,
                            mime_type=mime_type
                        ),
                        PROMPT
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.0
                    ),
                )

                text = response.text or ""
                selected_model = current_model
                break

            except Exception as exc:
                error = str(exc)
                errors.append(
                    f"{current_model} attempt {attempt + 1}: "
                    f"{friendly_error(exc)}"
                )

                retryable = (
                    "503" in error
                    or "429" in error
                    or "UNAVAILABLE" in error
                    or "RESOURCE_EXHAUSTED" in error
                    or "timeout" in error.lower()
                    or "timed out" in error.lower()
                    or "deadline_exceeded" in error.lower()
                )

                if not retryable:
                    return {
                        "ok": False,
                        "error": friendly_error(exc)
                    }

                if attempt == 0:
                    time.sleep(2)

        if text.strip():
            break

    if not text.strip() and errors:
        return {
            "ok": False,
            "error": "Gemini models unavailable: " + " | ".join(errors)
        }

    model = selected_model
    if not text.strip():
        return {"ok": False, "error": "Gemini returned an empty response (the image may have been blocked)."}
    try:
        parsed = parse_gemini_json(text)
    except (json.JSONDecodeError, ValueError) as exc:
        return {"ok": False, "error": f"Gemini returned an invalid response: {exc}"}
    return {"ok": True, "provider": f"Google Gemini ({model})", "page_confidence": None,
            "note": "Gemini does not provide calibrated per-field confidence; fields show detection status only.",
            **parsed}
