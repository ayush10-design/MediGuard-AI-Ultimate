from utils import history_db, ocr_service
from utils import report_generator as rg


def _report(classified=True):
    fields = ocr_service.parse_packaging_text("B.No. AB123\nExp. 01/2020\nMfd. by: Cipla Limited")
    cls = ({"ok": True, "predicted_class": "Fake", "scores": {"Fake": 0.82, "Real": 0.18}, "model_name": "MobileNetV2",
            "model_version": "test", "timestamp": "2026-01-01T10:00:00"} if classified
           else {"ok": False, "error": "Model Not Trained"})
    return rg.new_report("image", "pack.jpg", fields, {"provider": "Tesseract OCR", "raw_text": "B.No. AB123"}, cls)


def test_history_roundtrip(tmp_path):
    db = tmp_path / "h.db"
    rid = history_db.add_record(_report(), db_path=db)
    df = history_db.list_records(db_path=db)
    assert len(df) == 1 and df.iloc[0]["predicted_class"] == "Fake"
    assert abs(df.iloc[0]["fake_score"] - 0.82) < 1e-9
    assert history_db.get_report(rid, db_path=db)["filename"] == "pack.jpg"
    history_db.add_record(_report(False), db_path=db)
    assert history_db.list_records(predicted="Fake", db_path=db).shape[0] == 1
    assert history_db.list_records(search="pack", db_path=db).shape[0] == 2
    history_db.delete_record(rid, db_path=db)
    assert history_db.stats(db_path=db)["total"] == 1


def test_untrained_report_has_no_scores(tmp_path):
    r = _report(False)
    rows = dict((a, b) for a, b, _ in rg.to_rows(r))
    assert rows["Classification"] == "Not performed"
    assert "Fake class score" not in rows
    html = rg.to_html(r, {"report_include_metrics": True, "report_include_ml_explanation": True})
    assert "No classification was performed" in html


def test_exports_and_expiry_observation():
    r = _report()
    obs = rg.packaging_observations(r)
    assert any("in the past" in o for o in obs["observations"])
    assert any("expired" in x.lower() for x in rg.recommendations(r))
    assert b"Fake class score" in rg.to_csv(r)
    html = rg.to_html(r)
    for section in ("A. Medicine identification", "B. AI classification", "F. Safety and recommendation"):
        assert section in html
    pdf = rg.to_pdf(r, {"report_include_metrics": True, "report_include_ml_explanation": True})
    assert pdf[:4] == b"%PDF" and len(pdf) > 3000


def test_date_parsing():
    assert rg.parse_month_year("02/2028").year == 2028
    assert rg.parse_month_year("AUG.2027").month == 8
    assert rg.parse_month_year("12-08-2026").month == 8
    assert rg.parse_month_year("Not detected") is None
