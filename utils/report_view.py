"""Renders the detailed analysis report inside Streamlit (used by several pages)."""
from __future__ import annotations

import html

import streamlit as st

from utils import report_generator as rg
from utils import ui, visualization as viz


def render_report(report: dict, key: str = "report", show_downloads: bool = True) -> None:
    settings = ui.get_settings()
    cls = report.get("classification") or {}
    head = (f'{ui.chip("Image analysis" if report.get("analysis_type") == "image" else "Manual entry", "info")} '
            f'{ui.chip("Report " + report.get("id", ""), "")} {ui.chip(report.get("created_at", ""), "")}')
    st.markdown(f'<div class="mg-reveal">{head}</div>', unsafe_allow_html=True)

    ui.section_title("A", "Medicine identification")
    extra = report.get("extra") or {}
    st.markdown(ui.fields_table_html(report.get("fields") or {}), unsafe_allow_html=True)
    if any(extra.values()):
        rows = "".join(f'<tr><td>{html.escape(k.replace("_", " ").capitalize())}</td><td>{html.escape(str(v))}</td>'
                       f'<td><span class="mg-src m">Manually entered</span></td></tr>' for k, v in extra.items() if v)
        st.markdown(f'<table class="mg-table">{rows}</table>', unsafe_allow_html=True)

    ui.section_title("B", "AI classification")
    if cls.get("ok"):
        tone = "fake" if cls["predicted_class"].lower() == "fake" else "real"
        c1, c2 = st.columns([1, 1.3])
        with c1:
            st.markdown(f'<div class="mg-result {tone}"><div class="mg-chip info">Model output</div>'
                        f'<div class="verdict">{html.escape(cls["predicted_class"])}</div>'
                        f'{ui.prob_bars_html(cls["scores"])}<p class="mg-note">'
                        f'{html.escape(cls["model_name"])} · version {html.escape(str(cls["model_version"]))}<br>'
                        f'Analysed {html.escape(cls["timestamp"])} · inference completed successfully</p></div>',
                        unsafe_allow_html=True)
        with c2:
            st.plotly_chart(viz.probability_bars(cls["scores"]), use_container_width=True, key=f"{key}_prob")
            ui.note("This is a visual classification learned from labelled photos — not a laboratory or regulatory test.", "warn")
    else:
        ui.note("<b>No classification was performed.</b> " + html.escape(
            cls.get("error") or "Manual-entry reports never include an image classification."), "")

    if settings.get("report_include_ml_explanation", True):
        ui.section_title("C", "Machine learning explanation")
        for a, b in rg.ML_EXPLANATION:
            st.markdown(f"**{a}.** {b}")

    ui.section_title("D", "Model performance")
    m = report.get("metrics") or {}
    if settings.get("report_include_metrics", True) and m.get("test"):
        cols = st.columns(4)
        for col, k in zip(cols, ("accuracy", "precision", "recall", "f1")):
            col.metric(k.capitalize() if k != "f1" else "F1-score", f"{m['test'][k]*100:.1f}%")
        st.caption(f"Measured on the {m.get('evaluated_on', 'test split')}. Positive class for precision/recall: "
                   f"{m.get('positive_class_for_binary_metrics')}.")
        if m.get("confusion_matrix"):
            st.plotly_chart(viz.confusion_matrix(m["confusion_matrix"], m["labels"]), use_container_width=True, key=f"{key}_cm")
    else:
        ui.note("No measured metrics yet. Train the model with <code>python train_model.py</code>.")

    ui.section_title("E", "Packaging observations")
    obs = rg.packaging_observations(report)
    for o in obs["observations"] or ["No specific observations."]:
        st.markdown(f"- {o}")
    st.markdown(f"**Not detected:** {', '.join(obs['missing']) or 'none'}")
    if obs["uncertainty"]:
        with st.expander("Extraction details and uncertainty"):
            for u in obs["uncertainty"]:
                st.markdown(f"- {u}")
            raw = (report.get("ocr") or {}).get("raw_text")
            if raw:
                st.text_area("Raw detected text", raw, height=160, disabled=True, key=f"{key}_raw")

    ui.section_title("F", "Safety and recommendation")
    for r in rg.recommendations(report):
        st.markdown(f"- {r}")
    ui.note(html.escape(rg.DISCLAIMER), "warn")

    if show_downloads:
        st.markdown("#### Download report")
        c1, c2, c3 = st.columns(3)
        try:
            pdf = rg.to_pdf(report, settings)
            c1.download_button("Download PDF", pdf, f"mediguard_report_{report['id']}.pdf", "application/pdf",
                               use_container_width=True, type="primary", key=f"{key}_pdf")
        except Exception as exc:
            c1.error(f"PDF export failed: {exc}")
        c2.download_button("Download CSV", rg.to_csv(report), f"mediguard_report_{report['id']}.csv", "text/csv",
                           use_container_width=True, key=f"{key}_csv")
        c3.download_button("Printable HTML", rg.to_html(report, settings).encode(), f"mediguard_report_{report['id']}.html",
                           "text/html", use_container_width=True, key=f"{key}_html")
