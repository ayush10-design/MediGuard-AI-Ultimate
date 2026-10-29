import json

from utils import gemini_service, ocr_service

SAMPLE = """DOLO-650
Paracetamol Tablets IP 650 mg
Each uncoated tablet contains:
Paracetamol IP 650 mg
B.No. DOBS3975
Mfg. Date: 03/2025
Exp. Date: 02/2028
M.R.P. Rs. 33.60 incl. of all taxes
15 Tablets
Mfd. by: Micro Labs Limited
Keep out of reach of children. Store below 25°C.
"""


def test_tesseract_text_parser_extracts_fields():
    f = ocr_service.parse_packaging_text(SAMPLE)
    assert f["batch_number"]["value"] == "DOBS3975"
    assert f["mfg_date"]["value"] == "03/2025"
    assert f["expiry_date"]["value"] == "02/2028"
    assert f["mrp"]["value"] == "33.60"
    assert f["strength"]["value"].replace(" ", "") == "650mg"
    assert "15 Tablets" in f["pack_quantity"]["value"]
    assert "Micro Labs Limited" in f["manufacturer"]["value"]
    assert "Keep out of reach" in f["warnings"]["value"]
    assert f["medicine_name"]["value"] == "DOLO-650"
    assert "heuristic" in f["medicine_name"]["status"]


def test_parser_never_invents_missing_fields():
    f = ocr_service.parse_packaging_text("Some random words without details")
    assert f["batch_number"]["value"] == ocr_service.NOT_DETECTED
    assert f["expiry_date"]["status"] == ocr_service.STATUS_MISSING
    assert all(v["confidence"] is None for v in f.values())


def test_gemini_json_validation():
    payload = {"medicine_name": "Pan 40", "composition": None, "strength": "40 mg", "manufacturer": "Alkem",
               "batch_number": "null", "mfg_date": None, "expiry_date": "AUG 2027", "mrp": None,
               "pack_quantity": None, "warnings": ["Store in a cool place"], "raw_text": "PAN 40\nAUG 2027",
               "illegible_fields": ["mrp"]}
    out = gemini_service.parse_gemini_json("```json\n" + json.dumps(payload) + "\n```")
    f = out["fields"]
    assert f["medicine_name"]["value"] == "Pan 40" and f["medicine_name"]["status"] == ocr_service.STATUS_GEMINI
    assert f["batch_number"]["value"] == ocr_service.NOT_DETECTED
    assert f["mrp"]["status"] == "Present but illegible"
    assert f["warnings"]["value"] == "Store in a cool place"
    assert out["raw_text"].startswith("PAN 40")


def test_gemini_error_messages():
    assert "quota" in gemini_service.friendly_error(Exception("429 RESOURCE_EXHAUSTED")).lower()
    assert "api key" in gemini_service.friendly_error(Exception("403 permission denied")).lower()


def test_extract_without_any_provider(monkeypatch):
    from PIL import Image
    monkeypatch.setattr(gemini_service, "is_configured", lambda: False)
    res = ocr_service.extract(Image.new("RGB", (50, 50)), b"", "image/jpeg", provider="gemini")
    assert res["ok"] is False and "not configured" in res["error"]
