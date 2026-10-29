import html
import time
import hashlib

import streamlit as st
from streamlit_cropper import st_cropper

import config
from utils import history_db, image_processing as ip, ocr_service, ui
from utils import gemini_service
from utils.model_utils import model_status, predict
from utils import combined_verification as verification  # COMBINED_VERIFICATION_UPGRADE_V1
from utils.report_generator import new_report
from utils.report_view import render_report

S = st.session_state
settings = ui.get_settings()
ui.page_header("Detection Lab", "Prepare a medicine photo, read its label and run the trained classifier.", "lab",
               ui.chip("Model ready" if model_status()["trained"] else "Model not trained",
                       "ok" if model_status()["trained"] else "warn"))


def prepare_ocr_image(img):
    """Make a compact JPEG for remote OCR; never modify the original PIL image."""
    from io import BytesIO
    from PIL import Image, ImageOps

    image = ImageOps.exif_transpose(img).convert("RGB")
    image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=85, optimize=True)
    return buffer.getvalue()


def reset_analysis():
    for k in ("lab_fields", "lab_ocr", "lab_report"):
        S.pop(k, None)


left, right = st.columns([1.05, 1], gap="large")

# ------------------------------------------------------------------ LEFT: image + OCR
with left:
    st.markdown("#### 1. Image")
    phone = S.get("phone_image")
    sources = ["Upload image"] + (["Photo from phone"] if phone else [])
    src = st.radio("Image source", sources, horizontal=True, label_visibility="collapsed",
                   index=1 if phone and S.get("prefer_phone") else 0)
    raw, filename = None, None
    if src == "Upload image":
        up = st.file_uploader("Upload a medicine packaging photo", type=["jpg", "jpeg", "png", "webp"],
                              help="JPG, JPEG, PNG or WEBP, up to 15 MB.", on_change=reset_analysis)
        if up is not None:
            raw, filename = up.getvalue(), up.name
    else:
        raw, filename = phone["bytes"], phone["filename"]
        st.caption(f"Received from phone: {phone['filename']}")

    img = None
    if not raw and S.get("lab_image_id"):
        reset_analysis()
        S.pop("lab_image_id", None)
        S.pop("lab_upload_id", None)
    if raw:
        try:
            original = ip.load_image(raw, filename)
        except ip.ImageValidationError as exc:
            st.error(str(exc))
            original = None
        if original is not None:
            # Never re-use extracted text/results from a different upload or crop.
            uploaded_id = hashlib.sha256(raw).hexdigest()
            if S.get("lab_upload_id") != uploaded_id:
                reset_analysis()
                S["lab_upload_id"] = uploaded_id
                S.pop("lab_image_id", None)

            with st.expander("Rotate and crop", expanded=True):
                rotation = st.select_slider("Rotate image", options=[0, 90, 180, 270], value=0,
                                            format_func=lambda d: f"{d}°")
                rotated = ip.rotate(original, rotation)
                enable_crop = st.checkbox("Enable manual cropping", value=False)
                if enable_crop:
                    img = st_cropper(rotated, realtime_update=True,
                                     box_color="#1769E0", aspect_ratio=None,
                                     return_type="image", key=f"med_crop_{uploaded_id[:12]}_{rotation}")
                    if img is None:
                        img = rotated
                else:
                    img = rotated

            # Re-encode the processed PIL image to avoid browser preview issues.
            img = img.convert("RGB")
            image_id = hashlib.sha256(ip.to_bytes(img)).hexdigest()
            if S.get("lab_image_id") != image_id:
                reset_analysis()
                S["lab_image_id"] = image_id
            import base64
            preview = base64.b64encode(ip.to_bytes(img)).decode("ascii")
            st.markdown(
                f'<img src="data:image/jpeg;base64,{preview}" '
                f'style="width:100%;height:auto;border-radius:12px;" '
                f'alt="Medicine packaging preview">',
                unsafe_allow_html=True
            )
            st.caption("Final image for OCR and classification")
            st.markdown(f"{ui.chip(f'{img.width} × {img.height} px', 'info')} {ui.chip('Uploaded', 'ok')} "
                        f"{ui.chip(html.escape(filename), '')}", unsafe_allow_html=True)
            for n in ip.image_quality_notes(img):
                ui.note(n, "warn")
    else:
        ui.note("Upload a clear, well-lit photo of the pack — or use <b>Live Camera Scan</b> to send one from your phone.")

    st.markdown("#### 2. Read the label")
    provider = settings.get("ocr_provider", "auto")
    gem_ok = gemini_service.is_configured()
    consent = False
    if provider in ("auto", "gemini") and gem_ok:
        consent = st.checkbox("Send this image to Google Gemini for text extraction",
                              value=S.get("gemini_consent", False), key="gemini_consent",
                              help="The image leaves your laptop and is processed by Google's API. Leave unchecked to use offline Tesseract.")
    elif provider in ("auto", "gemini"):
        st.caption("Gemini is not configured — Tesseract (offline) will be used if installed.")
    if st.button("Extract packaging details", disabled=img is None, use_container_width=True):
        with st.spinner("Reading the packaging…"):
            mime = "image/jpeg"
            res = ocr_service.extract(img, prepare_ocr_image(img), mime, provider, consent, settings.get("gemini_model"))
        S["lab_ocr"] = res
        if res.get("ok"):
            S["lab_fields"] = res["fields"]
            st.success(f"Text extracted with {res['provider']}.")
        else:
            st.error(res.get("error", "Extraction failed."))
        for a in res.get("attempts") or []:
            st.caption(a)

    if S.get("lab_fields"):
        st.markdown("##### Extracted details — correct anything misread")
        with st.form("edit_fields"):
            edited = {}
            for key, f in S["lab_fields"].items():
                edited[key] = st.text_input(f["label"], value="" if f["value"] == ocr_service.NOT_DETECTED else f["value"],
                                            placeholder="Not detected", help=f"Source: {f['status']}")
            if st.form_submit_button("Save corrections"):
                for key, val in edited.items():
                    f = S["lab_fields"][key]
                    old = "" if f["value"] == ocr_service.NOT_DETECTED else f["value"]
                    if val.strip() != old:
                        f.update(value=val.strip() or ocr_service.NOT_DETECTED,
                                 status=ocr_service.STATUS_EDITED if val.strip() else ocr_service.STATUS_MISSING)
                st.success("Corrections saved. They will be marked as manually edited in the report.")
        raw_text = (S.get("lab_ocr") or {}).get("raw_text")
        if raw_text:
            with st.expander("Raw detected text"):
                st.text(raw_text)

# ------------------------------------------------------------------ RIGHT: model + analysis
with right:
    st.markdown("#### 3. Classify")
    ms = model_status()
    if ms["trained"]:
        st.markdown(f'<div class="mg-card"><h3>{config.MODEL_NAME}</h3><p>Version {ms.get("version") or "—"} · '
                    f'classes {", ".join(ms["classes"])} · input {config.IMG_SIZE[0]}×{config.IMG_SIZE[1]}</p></div>',
                    unsafe_allow_html=True)
        if not ms["tensorflow"]:
            ui.note(ms["message"], "bad")
    else:
        st.markdown('<div class="mg-card"><h3>Model Not Trained</h3><p>No classifier is available, so no Fake/Real scores '
                    'will be shown. Add images to <code>data/train/Fake</code> and <code>data/train/Real</code>, then run '
                    '<code>python train_model.py</code>. You can still extract details and create a report.</p></div>',
                    unsafe_allow_html=True)

    stage = st.empty()
    online_consent = st.checkbox("Include online product-information lookup via Tavily (shares medicine name and search terms)", value=False, key="online_lookup_consent")
    go = st.button("Analyze Medicine", type="primary", disabled=img is None, use_container_width=True)
    if go and img is not None:
        uri = ui.image_data_uri(ip.to_bytes(img.copy().resize((min(img.width, 900), int(img.height * min(img.width, 900) / img.width)))))
        states = ["pending"] * 4
        anim = settings.get("animations", True)

        def show(i, state, detail=""):
            states[i] = state
            stage.markdown(ui.scan_html(uri, states, detail), unsafe_allow_html=True)
            if anim and state == "active":
                time.sleep(0.35)  # brief pause so each real stage is visible

        # Stage 1 — extraction (reuse earlier extraction if present)
        show(0, "active")
        if S.get("lab_fields"):
            show(0, "done")
        elif settings.get("ocr_provider") != "none" and (ocr_service.tesseract_available() or (gemini_service.is_configured() and consent)):
            res = ocr_service.extract(img, prepare_ocr_image(img), "image/jpeg", settings.get("ocr_provider", "auto"), consent,
                                      settings.get("gemini_model"))
            S["lab_ocr"] = res
            if res.get("ok"):
                S["lab_fields"] = res["fields"]; show(0, "done")
            else:
                show(0, "failed", res.get("error", ""))
        else:
            show(0, "skipped")
        # Stage 2 — preprocessing
        show(1, "active")
        ip.prepare_for_model(img)
        show(1, "done")
        # Stage 3 — inference
        show(2, "active")
        if ms["trained"]:
            cls = predict(img)
            show(2, "done" if cls.get("ok") else "failed", "" if cls.get("ok") else cls.get("error", ""))
        else:
            cls = {"ok": False, "error": "Model Not Trained — run python train_model.py to enable classification."}
            show(2, "skipped")
        # Stage 4 — report
        show(3, "active")
        fields = S.get("lab_fields") or ocr_service.empty_fields()
        report = new_report("image", filename, fields, S.get("lab_ocr"), cls,
                            {"width": img.width, "height": img.height}, ip.image_quality_notes(img))
        # Compare against local reference photos and optionally search public product information.
        with st.spinner("Building combined evidence review…"):
            local_review = verification.check_local(img)
            online_review = (verification.check_online(fields, model=settings.get("gemini_model") or "gemini-3.5-flash")
                             if online_consent else
                             {"status": "not_checked", "summary": "Not checked (online consent not enabled).", "sources": []})
            combined_review = verification.combined_assessment(fields, cls, local_review, online_review)
        report["combined_verification"] = combined_review
        S["lab_report"] = report
        S["last_report"] = report
        if settings.get("save_history", True):
            report["history_id"] = history_db.add_record(report)
        show(3, "done")

    report = S.get("lab_report")
    if report:
        cls = report.get("classification") or {}
        if cls.get("ok"):
            tone = "fake" if cls["predicted_class"].lower() == "fake" else "real"
            st.markdown(f'<div class="mg-result {tone} mg-reveal"><span class="mg-chip info">Model classification</span>'
                        f'<div class="verdict">{html.escape(cls["predicted_class"])}</div>{ui.prob_bars_html(cls["scores"])}'
                        f'<p class="mg-note">The model compares visual patterns with its training photos. '
                        f'This is not a chemical or regulatory test.</p></div>', unsafe_allow_html=True)
            top = max(cls["scores"].values())
            if top < 0.75:
                ui.note(f"Low certainty: the top score is {top*100:.1f}%. Treat this result as inconclusive.", "warn")
        else:
            ui.note("<b>No classification.</b> " + html.escape(cls.get("error", "")), "warn")
        st.markdown('<div class="mg-note"><b>Three different things:</b> OCR reads the printed text · the model scores visual '
                    'similarity to Fake/Real photos · only a pharmacist, the manufacturer or a lab can verify authenticity.</div>',
                    unsafe_allow_html=True)

if S.get("lab_report"):
    st.divider()
    st.markdown("## Medicine Screening Results")
    with st.expander(
        "View detailed AI analysis and model performance",
        expanded=False
    ):
        render_report(S["lab_report"], key="lab")
    # Evidence-based general medicine guide.
    from utils.medicine_guide import get_guide

    guide = get_guide(S["lab_report"].get("fields") or {})
    st.markdown("### Medicine Information & Usage Guide")

    if guide:
        st.caption(
            "General educational information about the ingredient, "
            "not a personalised prescription or pack authentication."
        )
        st.markdown(f"**Ingredient:** {guide['ingredient']}")
        st.markdown("**What is it used for?**")
        st.write(guide["uses"])

        st.markdown("**How does it work?**")
        st.write(guide["working"])

        st.markdown("**When is it taken?**")
        st.write(guide["when"])

        st.markdown("**How is it taken?**")
        st.write(guide["how"])

        st.warning(guide["urgent"])

        with st.expander("Side effects and detailed safety precautions"):
            st.markdown("**Possible side effects**")
            st.write(guide["side_effects"])
            st.markdown("**Precautions and interactions**")
            st.write(guide["precautions"])

        with st.expander("Medical references"):
            for label, url in guide["sources"]:
                st.markdown(f"- [{label}]({url})")
    else:
        st.info(
            "A reviewed medicine guide is not available for this "
            "ingredient yet. Please consult a pharmacist or "
            "authoritative medicine reference."
        )

    # Combined evidence assessment: not an authenticity certificate.
    combined = S["lab_report"].get("combined_verification")
    if combined:
        st.markdown("### Combined evidence review")
        st.warning(f"**{combined['outcome']}** — {combined['reason']}")
        st.markdown("**Local dataset comparison (image similarity only)**")
        st.caption(combined['local'].get('note', 'Not available'))
        for hit in combined['local'].get('nearest', []):
            st.write(f"Reference label: {hit['dataset_label']} | image-hash distance: {hit['distance']} | {hit['filename']}")
        st.markdown("**Online product-information lookup**")

        online = combined.get("online") or {}
        comparison = online.get("comparison") or {}

        if comparison:
            labels = {
                "medicine_name": "Medicine name",
                "strength": "Strength",
                "composition": "Composition",
                "manufacturer": "Manufacturer",
            }

            rows = []
            for field, label in labels.items():
                status = comparison.get(field, "Not confirmed")
                rows.append({
                    "Medicine detail": label,
                    "Online evidence": status,
                })

            st.table(rows)

        st.caption(
            "Matches are based on public product information. "
            "They do not confirm this physical medicine pack, "
            "its batch number, or authenticity."
        )

        st.write(online.get("summary", "Not checked"))

        sources = online.get("sources") or []
        if sources:
            st.markdown("**Online references**")
            for hit in sources:
                st.markdown(
                    f"- [{hit['title']}]({hit['url']})"
                )
        else:
            st.info("No online references available.")
        for flag in combined.get('flags', []):
            st.warning(flag)
        st.caption(combined['disclaimer'])
        try:
            pdf_bytes = verification.combined_pdf(S['lab_report'], combined)
            st.download_button("Download combined screening PDF", pdf_bytes,
                               f"mediguard_combined_{S['lab_report']['id']}.pdf",
                               mime="application/pdf", key='lab_combined_pdf')
        except Exception as exc:
            st.error(f"Combined PDF unavailable: {exc}")

