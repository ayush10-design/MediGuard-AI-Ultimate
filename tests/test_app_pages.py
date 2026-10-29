"""Headless Streamlit run of every page (no model, no dataset, no Gemini key)."""
from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
ROOT = Path(__file__).resolve().parent.parent
PAGES = sorted(p.name for p in (ROOT / "pages").glob("*.py"))


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exceptions(page, monkeypatch, tmp_path):
    import config
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "h.db")
    at = st_testing.AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    at.switch_page(f"pages/{page}")
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_manual_entry_generates_report_without_classification(monkeypatch, tmp_path):
    import config
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "h.db")
    at = st_testing.AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.run()
    at.switch_page("pages/3_Manual_Entry.py")
    at.run()
    at.selectbox(key="m_name").set_value("Dolo 650").run()
    assert not at.exception
    gen = [b for b in at.button if b.label == "Generate information report"][0]
    gen.click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert "manual_report" in at.session_state
    rep = at.session_state["manual_report"]
    assert rep["classification"] is None
    assert rep["fields"]["medicine_name"]["value"] == "Dolo 650"
