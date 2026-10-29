import streamlit as st

import config
from utils import gemini_service, ui
from utils.ocr_service import tesseract_available

ui.page_header("Settings", "Choose how text is read, how the interface behaves and what reports include.", "settings")
s = dict(ui.get_settings())

a, b = st.columns(2, gap="large")
with a:
    st.markdown("#### Text extraction")
    s["ocr_provider"] = st.radio(
        "OCR provider", ["auto", "gemini", "tesseract", "none"], index=["auto", "gemini", "tesseract", "none"].index(s["ocr_provider"]),
        format_func={"auto": "Automatic — Gemini, then Tesseract", "gemini": "Gemini only", "tesseract": "Tesseract only (offline)",
                     "none": "Off"}.get)
    key = config.get_gemini_api_key()
    st.markdown(f"{ui.chip('Gemini key configured' if key else 'Gemini key missing', 'ok' if key else 'warn')} "
                f"{ui.chip('google-genai installed' if gemini_service.sdk_available() else 'google-genai missing', 'ok' if gemini_service.sdk_available() else 'bad')} "
                f"{ui.chip('Tesseract found' if tesseract_available() else 'Tesseract not found', 'ok' if tesseract_available() else 'warn')}",
                unsafe_allow_html=True)
    st.caption(f"API key: {config.mask_secret(key)} — set {config.GEMINI_API_KEY_ENV} in the .env file and restart the app.")
    s["gemini_model"] = st.text_input("Gemini model", s["gemini_model"], help="For example gemini-2.5-flash")
    if st.button("Test Gemini connection", disabled=not key):
        from PIL import Image
        from utils.image_processing import to_bytes
        with st.spinner("Contacting Gemini…"):
            res = gemini_service.extract_medicine_details(to_bytes(Image.new("RGB", (64, 64), "white")), "image/jpeg", s["gemini_model"])
        if res["ok"]:
            st.success("Gemini responded correctly.")
        else:
            st.error(res["error"])

    st.markdown("#### History")
    s["save_history"] = st.toggle("Save analyses to local history", s["save_history"])

with b:
    st.markdown("#### Interface")
    s["animations"] = st.toggle("Animations", s["animations"], help="Turns off the scan line, reveals and transitions.")
    s["custom_cursor"] = st.toggle("Syringe cursor (desktop only)", s["custom_cursor"],
                                   help="Never shown on touch screens or when your system asks for reduced motion (glow only).")
    st.markdown("#### Reports")
    s["report_include_ml_explanation"] = st.toggle("Include machine-learning explanation", s["report_include_ml_explanation"])
    s["report_include_metrics"] = st.toggle("Include model performance metrics", s["report_include_metrics"])
    s["report_author"] = st.text_input("Prepared by", s["report_author"], placeholder="Your name")
    s["report_institution"] = st.text_input("Institution", s["report_institution"], placeholder="College / department")

if st.button("Save settings", type="primary"):
    config.save_settings(s)
    st.session_state["settings"] = config.load_settings()
    st.success("Settings saved.")
    st.rerun()
