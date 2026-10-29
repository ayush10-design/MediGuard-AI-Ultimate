import html
from datetime import date

import streamlit as st

from utils import autocomplete as ac, history_db, ocr_service, ui
from utils.report_generator import new_report
from utils.report_view import render_report

S = st.session_state
ui.page_header("Manual Medicine Entry", "Type the details printed on the pack. Suggestions speed things up; nothing is auto-verified.", "form")
ui.note("Suggestions come from a small reference catalogue. A name appearing in it does <b>not</b> mean your pack is genuine. "
        "Manual entries never receive a Fake/Real classification — that needs a photo.", "warn")


def combo(label: str, options, key: str, placeholder: str, help_text: str = "") -> str:
    """Searchable select that also accepts new values (true type-ahead autocomplete)."""
    try:
        val = st.selectbox(label, options=list(options), index=None, key=key, placeholder=placeholder,
                           accept_new_options=True, help=help_text)
    except TypeError:  # Streamlit < 1.45 fallback: text input + clickable suggestions
        val = st.text_input(label, key=key, placeholder=placeholder, help=help_text)
        sugg = ac.suggest(val or "", options, 5) if val else []
        if sugg and val not in sugg:
            cols = st.columns(len(sugg))
            for c, s in zip(cols, sugg):
                c.button(s, key=f"{key}_{s}", on_click=lambda k=key, v=s: S.__setitem__(k, v))
    return (val or "").strip()


c1, c2 = st.columns(2)
with c1:
    medicine = combo("Medicine name", [e.medicine for e in ac.load_catalogue()], "m_name",
                     "Start typing, e.g. Dolo", "Type to search. Press Enter to keep a name that is not in the list.")
    entry = ac.lookup_medicine(medicine) if medicine else None
with c2:
    mf_opts = list(ac.load_manufacturers())
    manufacturer = combo("Manufacturer", mf_opts, "m_mfr", "Start typing, e.g. Man → Mankind Pharma")

# Dependent suggestion — offered, never applied silently
if entry:
    diff = [(k, getattr(entry, k)) for k in ("manufacturer", "composition", "strength")
            if getattr(entry, k) and S.get(f"m_{k[:4]}" if k != "manufacturer" else "m_mfr") != getattr(entry, k)]
    if diff:
        st.markdown(f'<div class="mg-note">Catalogue lists <b>{html.escape(entry.medicine)}</b> as '
                    f'{html.escape(entry.manufacturer)} · {html.escape(entry.composition)} · {html.escape(entry.strength)}. '
                    'Use these values only if they match what is printed on your pack.</div>', unsafe_allow_html=True)
        def _apply(e=entry):
            S["m_mfr"], S["m_comp"], S["m_stre"] = e.manufacturer, e.composition, e.strength
        st.button("Apply catalogue details", on_click=_apply)

c3, c4 = st.columns(2)
with c3:
    composition = combo("Composition", sorted({e.composition for e in ac.load_catalogue()}), "m_comp", "e.g. Paracetamol")
    batch = st.text_input("Batch number", key="m_batch", placeholder="As printed, e.g. DOBS3975")
    mfg = st.text_input("Manufacturing date", key="m_mfg", placeholder="e.g. 03/2025")
    mrp = st.number_input("MRP (₹)", min_value=0.0, step=0.5, format="%.2f", key="m_mrp")
    barcode = st.text_input("Barcode / QR data", key="m_code", placeholder="Digits or text from the code")
with c4:
    strength = st.text_input("Strength / dosage", key="m_stre", placeholder="e.g. 650 mg")
    pack = st.text_input("Pack size", key="m_pack", placeholder="e.g. 15 tablets")
    exp = st.text_input("Expiry date", key="m_exp", placeholder="e.g. 02/2028")
    dosage = st.text_input("Dosage instructions", key="m_dose", placeholder="e.g. As directed by the physician")
    notes = st.text_area("Additional notes", key="m_notes", height=96)

if mf_opts and manufacturer and medicine and entry is None:
    others = ac.medicines_for_manufacturer(manufacturer)
    if others:
        st.caption(f"Catalogue products from {manufacturer}: {', '.join(others[:8])}")

if st.button("Generate information report", type="primary", disabled=not medicine):
    vals = {"medicine_name": medicine, "manufacturer": manufacturer, "composition": composition, "strength": strength,
            "batch_number": batch, "mfg_date": mfg, "expiry_date": exp, "mrp": f"₹{mrp:.2f}" if mrp else "",
            "pack_quantity": pack, "warnings": ""}
    fields = ocr_service.empty_fields()
    for k, v in vals.items():
        if v:
            fields[k].update(value=v, status=ocr_service.STATUS_MANUAL)
    extra = {"dosage": dosage, "barcode_or_qr_data": barcode, "notes": notes,
             "catalogue_match": f"Name found in reference catalogue (not a verification)" if entry else "Not in reference catalogue"}
    report = new_report("manual", None, fields, None, None, extra=extra)
    S["manual_report"] = report
    S["last_report"] = report
    if ui.get_settings().get("save_history", True):
        history_db.add_record(report)

if not medicine:
    st.caption("Enter at least the medicine name to generate a report.")

if S.get("manual_report"):
    st.divider()
    st.markdown("## Medicine information report")
    render_report(S["manual_report"], key="manual")
