"""OCR extraction: Gemini (primary) with Tesseract as offline fallback.

Every field carries a *status* describing how it was obtained. Numeric confidence
is only reported where the engine genuinely provides one (Tesseract word
confidence); Gemini does not return calibrated per-field confidences, so its
fields are labelled by detection status only.
"""
from __future__ import annotations

import re
import shutil
from typing import Any

from PIL import Image

import config
from utils.image_processing import enhance_for_ocr

NOT_DETECTED = "Not detected"
FIELDS = [
    ("medicine_name", "Medicine name"),
    ("composition", "Generic / composition"),
    ("strength", "Strength"),
    ("manufacturer", "Manufacturer"),
    ("batch_number", "Batch number"),
    ("mfg_date", "Manufacturing date"),
    ("expiry_date", "Expiry date"),
    ("mrp", "MRP"),
    ("pack_quantity", "Pack quantity"),
    ("warnings", "Warnings"),
]

STATUS_GEMINI = "Visible text (Gemini)"
STATUS_TESSERACT = "Pattern match (Tesseract)"
STATUS_EDITED = "Manually edited"
STATUS_MANUAL = "Manually entered"
STATUS_MISSING = "Not detected"

_DATE = r"((?:0?[1-9]|1[0-2])[\/\-\.\s](?:19|20)?\d{2}|[A-Z]{3}[\.\-\s]*'?(?:19|20)?\d{2}|\d{1,2}[\/\-\.](?:0?[1-9]|1[0-2])[\/\-\.](?:19|20)?\d{2})"
PATTERNS = {
    "batch_number": r"(?:B\.?\s*No\.?|Batch\s*(?:No\.?|Number)?|Lot\s*(?:No\.?)?)\s*[:.\-]?\s*([A-Z0-9][A-Z0-9\-\/]{2,15})",
    "mfg_date": r"(?:Mfg\.?|MFD\.?|Mfd\.?\s*Date|Manufactured\s*on|Mfg\.?\s*Date|DOM)\s*[:.\-]?\s*" + _DATE,
    "expiry_date": r"(?:Exp\.?|EXP\.?|Expiry|Expiry\s*Date|Use\s*before)\s*(?:Date)?\s*[:.\-]?\s*" + _DATE,
    "mrp": r"(?:M\.?R\.?P\.?)[^0-9₹]{0,25}(?:₹|Rs\.?|INR)?\s*([0-9]{1,5}(?:\.[0-9]{1,2})?)",
    "strength": r"\b(\d+(?:\.\d+)?\s?(?:mg|mcg|µg|g|ml|mL|IU|%)(?:\s*\+\s*\d+(?:\.\d+)?\s?(?:mg|mcg|µg|g|ml|mL|IU|%))*)\b",
    "pack_quantity": r"\b(\d{1,4}\s*(?:x\s*\d{1,3}\s*)?(?:Tablets?|Capsules?|Caps|Tabs|Strips?|ml|mL|Sachets?))\b",
    "manufacturer": r"(?:Mfd\.?\s*by|Manufactured\s*by|Mkt\.?\s*by|Marketed\s*by)\s*[:.\-]?\s*([A-Z][A-Za-z&\.'\s]{2,60}?(?:Ltd\.?|Limited|Pvt\.?|Inc\.?|Laboratories|Labs|Pharma\w*)(?:\s+(?:Pvt\.?\s*)?(?:Limited|Ltd\.?))?)",
    "composition": r"(?:Each\s+(?:film[\s\-]coated\s+|uncoated\s+|hard\s+gelatin\s+)?(?:tablet|capsule|ml|5\s*ml)\s+contains)\s*[:.\-]?\s*(.{3,120}?)(?:\n|$|Excipients|Colour)",
}
WARNING_HINTS = ("keep out of reach", "store", "warning", "caution", "schedule h", "do not", "prescription")


def empty_fields() -> dict[str, dict[str, Any]]:
    return {k: {"label": lbl, "value": NOT_DETECTED, "status": STATUS_MISSING, "confidence": None}
            for k, lbl in FIELDS}


def parse_packaging_text(text: str) -> dict[str, dict[str, Any]]:
    """Extract structured fields from raw OCR text using conservative regexes."""
    fields = empty_fields()
    if not text or not text.strip():
        return fields
    flat = re.sub(r"[ \t]+", " ", text)
    for key, pattern in PATTERNS.items():
        m = re.search(pattern, flat, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            value = m.group(1).strip(" :.-\n")
            if value:
                fields[key].update(value=value, status=STATUS_TESSERACT)
    warnings = [ln.strip() for ln in text.splitlines()
                if any(h in ln.lower() for h in WARNING_HINTS) and len(ln.strip()) > 6]
    if warnings:
        fields["warnings"].update(value="; ".join(warnings[:4]), status=STATUS_TESSERACT)
    # Medicine name heuristic: first prominent line that is mostly letters (uncertain)
    for ln in text.splitlines():
        s = ln.strip()
        if 3 <= len(s) <= 40 and sum(c.isalpha() for c in s) >= max(3, len(s) * 0.5) \
                and not re.search(r"mfg|exp|batch|mrp|tablet|contains|store|keep", s, re.I):
            fields["medicine_name"].update(value=s, status=STATUS_TESSERACT + " — heuristic, please confirm")
            break
    return fields


def tesseract_available() -> bool:
    if shutil.which("tesseract") is None:
        return False
    try:
        import pytesseract  # noqa: F401
        return True
    except ImportError:
        return False


def run_tesseract(img: Image.Image) -> dict[str, Any]:
    if not tesseract_available():
        return {"ok": False, "error": "Tesseract is not installed. On macOS run: brew install tesseract"}
    import pytesseract
    try:
        prepared = enhance_for_ocr(img)
        data = pytesseract.image_to_data(prepared, output_type=pytesseract.Output.DICT, config="--psm 6")
        text = pytesseract.image_to_string(prepared, config="--psm 6")
    except Exception as exc:  # pytesseract raises a variety of errors
        return {"ok": False, "error": f"Tesseract failed: {exc}"}
    confs = [float(c) for c, w in zip(data.get("conf", []), data.get("text", [])) if w.strip() and float(c) >= 0]
    mean_conf = round(sum(confs) / len(confs), 1) if confs else None
    fields = parse_packaging_text(text)
    for f in fields.values():
        if f["status"] != STATUS_MISSING:
            f["confidence"] = None  # per-field confidence is not measured; only page-level below
    return {"ok": True, "provider": "Tesseract OCR", "fields": fields, "raw_text": text.strip(),
            "page_confidence": mean_conf,
            "note": "Page-level mean word confidence from Tesseract; fields were found with text patterns."}


def extract(img: Image.Image, image_bytes: bytes, mime: str, provider: str = "auto",
            gemini_consent: bool = False, gemini_model: str | None = None) -> dict[str, Any]:
    """Run OCR with the selected provider. Never returns invented values."""
    from utils import gemini_service

    attempts = []
    use_gemini = provider in ("auto", "gemini") and gemini_service.is_configured()
    if use_gemini and not gemini_consent:
        attempts.append("Gemini skipped: consent to send the image to Google was not given.")
        use_gemini = False
    if use_gemini:
        res = gemini_service.extract_medicine_details(image_bytes, mime, model=gemini_model)
        if res["ok"]:
            res["attempts"] = attempts
            return res
        attempts.append(f"Gemini: {res['error']}")
        if provider == "gemini":
            return {"ok": False, "error": res["error"], "attempts": attempts}
    elif provider == "gemini" and not gemini_service.is_configured():
        return {"ok": False, "error": f"Gemini is not configured. Add {config.GEMINI_API_KEY_ENV} to your .env file.",
                "attempts": attempts}
    if provider in ("auto", "tesseract"):
        res = run_tesseract(img)
        res["attempts"] = attempts
        return res
    return {"ok": False, "error": "OCR is turned off in Settings.", "attempts": attempts}
