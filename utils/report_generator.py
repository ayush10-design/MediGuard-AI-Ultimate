"""Builds the detailed medicine analysis report (on-screen, HTML, PDF, CSV)."""
from __future__ import annotations

import csv
import html
import io
import re
import uuid
from datetime import date, datetime
from typing import Any

import config
from utils.ocr_service import FIELDS, NOT_DETECTED, STATUS_MISSING

DISCLAIMER = ("Educational prototype. The Fake/Real result is a visual classification learned from a labelled "
              "image dataset. It is not laboratory testing, regulatory verification or proof of authenticity or "
              "chemical safety. Always confirm with a pharmacist, the manufacturer or the drug regulator.")

ML_EXPLANATION = [
    ("Model", "MobileNetV2, a convolutional neural network pre-trained on ImageNet, with a new two-class "
              "classification head trained on labelled Fake/Real medicine photos."),
    ("Why MobileNetV2", "It is small (about 3.5 million parameters) and fast on a laptop CPU, yet accurate on "
                         "images thanks to depthwise-separable convolutions and inverted residual blocks — a "
                         "good fit for a live demonstration on a MacBook."),
    ("Transfer learning", "The ImageNet-trained layers already detect edges, textures, shapes and print patterns. "
                          "These layers are first frozen while the new head learns; optionally the top layers are "
                          "then fine-tuned with a small learning rate."),
    ("Preprocessing", f"Each image is corrected for phone orientation, converted to RGB, resized to "
                      f"{config.IMG_SIZE[0]}×{config.IMG_SIZE[1]} pixels and scaled to the [-1, 1] range inside "
                      "the saved model, so training and inference use identical steps."),
    ("What it learns", "Visual patterns that separated the two classes in the training photos — for example "
                       "print quality, colour, layout, typography and packaging texture."),
    ("Limitations", "The model cannot read chemistry, check batch records or see hidden features such as holograms "
                    "under UV light. It may also learn photo-specific shortcuts (background, lighting, camera) from "
                    "the dataset. No pixel-level explanation (such as Grad-CAM) was computed for this prediction, "
                    "so we do not claim which regions drove it."),
]

_MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def parse_month_year(text: str) -> date | None:
    """Parse common Indian packaging date formats (e.g. 08/2026, AUG.2026, 12-08-2026)."""
    if not text or text == NOT_DETECTED:
        return None
    t = text.strip().lower()
    m = re.search(r"\b(\d{1,2})[\/\-\. ](\d{1,2})[\/\-\. ]((?:19|20)?\d{2})\b", t)
    if m:
        mo, yr = int(m.group(2)), int(m.group(3))
    else:
        m = re.search(r"\b(\d{1,2})[\/\-\. ]+((?:19|20)?\d{2})\b", t)
        if m:
            mo, yr = int(m.group(1)), int(m.group(2))
        else:
            m = re.search(r"\b([a-z]{3})[a-z]*[\.\-\s']*((?:19|20)?\d{2})\b", t)
            if not m or m.group(1) not in _MONTHS:
                return None
            mo, yr = _MONTHS[m.group(1)], int(m.group(2))
    if yr < 100:
        yr += 2000
    if not 1 <= mo <= 12 or not 1990 <= yr <= 2100:
        return None
    return date(yr, mo, 1)


def new_report(analysis_type: str, filename: str | None, fields: dict, ocr: dict | None,
               classification: dict | None, image_info: dict | None = None,
               quality_notes: list[str] | None = None, extra: dict | None = None) -> dict[str, Any]:
    from utils.model_utils import load_metrics, load_training_config
    metrics = load_metrics()
    return {
        "id": uuid.uuid4().hex[:10],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "analysis_type": analysis_type,
        "filename": filename,
        "image_info": image_info or {},
        "fields": fields,
        "ocr": ocr,
        "classification": classification,
        "quality_notes": quality_notes or [],
        "metrics": metrics,
        "training_config": load_training_config(),
        "extra": extra or {},
        "app_version": config.APP_VERSION,
    }


def packaging_observations(report: dict) -> dict[str, list[str]]:
    fields = report.get("fields") or {}
    found = [f["label"] for f in fields.values() if f.get("value") not in (None, "", NOT_DETECTED)]
    missing = [f["label"] for f in fields.values() if f.get("value") in (None, "", NOT_DETECTED)]
    obs, uncertainty = [], []
    exp = parse_month_year((fields.get("expiry_date") or {}).get("value", ""))
    mfg = parse_month_year((fields.get("mfg_date") or {}).get("value", ""))
    if exp:
        today = date.today().replace(day=1)
        if exp < today:
            obs.append(f"The expiry date as read ({fields['expiry_date']['value']}) appears to be in the past. "
                       "Do not use an expired medicine.")
        else:
            obs.append(f"The expiry date as read ({fields['expiry_date']['value']}) is not in the past.")
    if exp and mfg and mfg > exp:
        obs.append("The manufacturing date as read is later than the expiry date — re-check the reading or the pack.")
    for key in ("batch_number", "expiry_date", "mfg_date", "mrp", "manufacturer"):
        f = fields.get(key) or {}
        if f.get("value") in (None, "", NOT_DETECTED):
            obs.append(f"{f.get('label', key)} was not found in the image or entry. Genuine Indian packs normally print it.")
    ocr = report.get("ocr") or {}
    if ocr.get("provider"):
        uncertainty.append(f"Text extracted by {ocr['provider']}.")
    if ocr.get("page_confidence") is not None:
        uncertainty.append(f"Mean Tesseract word confidence for the page: {ocr['page_confidence']}%.")
    if ocr.get("note"):
        uncertainty.append(ocr["note"])
    if any("heuristic" in (f.get("status") or "") for f in fields.values()):
        uncertainty.append("The medicine name was guessed from the most prominent text line; confirm it manually.")
    edited = [f["label"] for f in fields.values() if f.get("status") == "Manually edited"]
    if edited:
        uncertainty.append("Edited by user: " + ", ".join(edited) + ".")
    for e in ocr.get("attempts") or []:
        uncertainty.append(e)
    uncertainty += report.get("quality_notes") or []
    if report.get("analysis_type") == "manual":
        uncertainty.append("All details were typed by the user; no image or OCR was used.")
    return {"found": found, "missing": missing, "observations": obs, "uncertainty": uncertainty}


def recommendations(report: dict) -> list[str]:
    cls = report.get("classification") or {}
    recs = []
    obs = packaging_observations(report)
    if any("in the past" in o.lower() and "not in the past" not in o.lower() for o in obs["observations"]):
        recs.append("The pack appears to be expired. Do not consume it; return it to the pharmacy.")
    if cls.get("ok"):
        pred = (cls.get("predicted_class") or "").lower()
        top = max(cls.get("scores", {}).values() or [0])
        if pred == "fake":
            recs.append("The model's visual classification leans towards the Fake class. Do not use the medicine "
                        "until a pharmacist or the manufacturer has checked it.")
            recs.append("Report suspected substandard or falsified medicines to CDSCO (cdsco.gov.in) or your State "
                        "Drug Controller.")
        else:
            recs.append("The model's visual classification leans towards the Real class. This does not confirm the "
                        "medicine is genuine — packaging can be copied convincingly.")
        if top < 0.75:
            recs.append(f"The top class score is only {top*100:.1f}%, so the model is uncertain. Treat the result as inconclusive.")
    elif report.get("analysis_type") == "image":
        recs.append("No model classification is available for this image. " + (cls.get("error") or ""))
    recs += [
        "Buy medicines only from licensed pharmacies and keep the bill.",
        "Check that the seal, batch number, expiry date and MRP are printed clearly and match the bill.",
        "Many Indian packs carry a QR code or a unique code that can be verified with the manufacturer — scan it with the manufacturer's official app or website.",
        "If you feel unwell after taking a medicine, contact a doctor and report it through the PvPI helpline (1800-180-3024).",
    ]
    return recs


# ------------------------------------------------------------------ exports
def to_rows(report: dict) -> list[tuple[str, str, str]]:
    rows = [("Report ID", report.get("id", ""), ""), ("Created", report.get("created_at", ""), ""),
            ("Analysis type", report.get("analysis_type", ""), ""), ("File", report.get("filename") or "—", "")]
    for key, label in FIELDS:
        f = (report.get("fields") or {}).get(key) or {}
        rows.append((label, f.get("value", NOT_DETECTED), f.get("status", STATUS_MISSING)))
    for k, v in (report.get("extra") or {}).items():
        rows.append((k.replace("_", " ").capitalize(), str(v), "Manually entered"))
    cls = report.get("classification") or {}
    if cls.get("ok"):
        rows.append(("Predicted class", cls["predicted_class"], "Model inference"))
        for c, s in cls["scores"].items():
            rows.append((f"{c} class score", f"{s*100:.2f}%", "Model inference"))
        rows.append(("Model", f"{cls.get('model_name')} v{cls.get('model_version')}", ""))
    else:
        rows.append(("Classification", "Not performed", (cls or {}).get("error", "Manual entry")))
    return rows


def to_csv(report: dict) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Field", "Value", "Source / status"])
    w.writerows(to_rows(report))
    return buf.getvalue().encode("utf-8")


def _esc(v: Any) -> str:
    return html.escape(str(v if v is not None else ""))


def to_html(report: dict, settings: dict | None = None) -> str:
    settings = settings or config.load_settings()
    cls = report.get("classification") or {}
    obs = packaging_observations(report)
    metrics = (report.get("metrics") or {}).get("test") if settings.get("report_include_metrics") else None
    field_rows = "".join(
        f"<tr><td>{_esc(f['label'])}</td><td>{_esc(f['value'])}</td><td><span class='tag'>{_esc(f['status'])}</span></td></tr>"
        for f in (report.get("fields") or {}).values())
    extra_rows = "".join(f"<tr><td>{_esc(k.replace('_',' ').capitalize())}</td><td>{_esc(v)}</td><td><span class='tag'>Manually entered</span></td></tr>"
                         for k, v in (report.get("extra") or {}).items() if v)
    if cls.get("ok"):
        bars = "".join(
            f"<div class='bar'><span>{_esc(c)}</span><div class='track'><div class='fill {'fake' if c.lower()=='fake' else 'real'}' "
            f"style='width:{s*100:.1f}%'></div></div><b>{s*100:.1f}%</b></div>" for c, s in cls["scores"].items())
        cls_html = (f"<p class='verdict'>Predicted class: <b>{_esc(cls['predicted_class'])}</b></p>{bars}"
                    f"<p class='muted'>Model {_esc(cls.get('model_name'))}, version {_esc(cls.get('model_version'))}, "
                    f"analysed {_esc(cls.get('timestamp'))}. Inference completed successfully.</p>")
    else:
        cls_html = f"<p class='muted'>No classification was performed. {_esc(cls.get('error') or 'Manual entry reports do not include an image classification.')}</p>"
    ml_html = "".join(f"<dt>{_esc(a)}</dt><dd>{_esc(b)}</dd>" for a, b in ML_EXPLANATION) if settings.get("report_include_ml_explanation") else ""
    if metrics:
        met_html = "<table><tr><th>Metric (independent test set)</th><th>Value</th></tr>" + "".join(
            f"<tr><td>{k.replace('_',' ').capitalize()}</td><td>{v*100:.2f}%</td></tr>"
            for k, v in metrics.items() if isinstance(v, (int, float))) + "</table>"
        cm = (report.get("metrics") or {}).get("confusion_matrix")
        labels = (report.get("metrics") or {}).get("labels") or []
        if cm:
            met_html += "<table><tr><th></th>" + "".join(f"<th>Pred {_esc(l)}</th>" for l in labels) + "</tr>" + "".join(
                f"<tr><th>True {_esc(labels[i])}</th>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>" for i, row in enumerate(cm)) + "</table>"
    else:
        met_html = "<p class='muted'>No measured model metrics are available (model not trained yet, or metrics excluded in Settings).</p>"
    li = lambda items: "".join(f"<li>{_esc(i)}</li>" for i in items) or "<li>None</li>"
    raw = (report.get("ocr") or {}).get("raw_text") or ""
    author = " · ".join(x for x in (settings.get("report_author"), settings.get("report_institution")) if x)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MediGuard AI report {_esc(report.get('id'))}</title>
<style>
:root{{--blue:#1769E0;--navy:#142B4F;--teal:#0D9488;--soft:#F5F9FF;--line:#E1EAF7}}
*{{box-sizing:border-box}} body{{font-family:Inter,-apple-system,'Segoe UI',sans-serif;color:var(--navy);margin:0;background:#fff;line-height:1.5}}
.wrap{{max-width:860px;margin:0 auto;padding:40px 28px}}
header{{display:flex;justify-content:space-between;align-items:flex-end;border-bottom:3px solid var(--blue);padding-bottom:16px;margin-bottom:24px}}
h1{{margin:0;font-size:26px}} h2{{font-size:18px;margin:32px 0 10px;color:var(--blue)}}
.muted{{color:#5A6E8C;font-size:14px}} table{{width:100%;border-collapse:collapse;margin:8px 0 16px;font-size:14px}}
td,th{{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);vertical-align:top}} th{{background:var(--soft)}}
.tag{{background:var(--soft);border:1px solid var(--line);border-radius:999px;padding:1px 9px;font-size:12px;white-space:nowrap}}
.bar{{display:grid;grid-template-columns:70px 1fr 64px;gap:10px;align-items:center;margin:6px 0}}
.track{{background:var(--soft);border-radius:8px;height:14px;overflow:hidden}} .fill{{height:100%}} .fake{{background:#E05A5A}} .real{{background:#2FA36B}}
.verdict{{font-size:18px}} dt{{font-weight:600;margin-top:10px}} dd{{margin:2px 0 0}}
pre{{background:var(--soft);padding:12px;border-radius:10px;white-space:pre-wrap;font-size:12px}}
.disclaimer{{margin-top:36px;padding:14px 16px;border-left:4px solid #E3A008;background:#FFF8E6;font-size:13px}}
@media print{{.wrap{{padding:0}} h2{{break-after:avoid}} table{{break-inside:avoid}}}}
</style></head><body><div class="wrap">
<header><div><h1>Medicine analysis report</h1><div class="muted">MediGuard AI · {_esc(config.TAGLINE)}</div></div>
<div class="muted">Report {_esc(report.get('id'))}<br>{_esc(report.get('created_at'))}<br>{_esc(author)}</div></header>
<h2>A. Medicine identification</h2><table><tr><th>Field</th><th>Value</th><th>Source</th></tr>{field_rows}{extra_rows}</table>
<h2>B. AI classification</h2>{cls_html}
{'<h2>C. Machine learning explanation</h2><dl>' + ml_html + '</dl>' if ml_html else ''}
<h2>D. Model performance</h2>{met_html}
<h2>E. Packaging observations</h2><ul>{li(obs['observations'])}</ul>
<p class="muted"><b>Not detected:</b> {_esc(', '.join(obs['missing']) or 'none')}</p>
<p class="muted"><b>Extraction notes:</b></p><ul class="muted">{li(obs['uncertainty'])}</ul>
{'<p class="muted"><b>Raw detected text</b></p><pre>' + _esc(raw) + '</pre>' if raw else ''}
<h2>F. Safety and recommendation</h2><ul>{li(recommendations(report))}</ul>
<div class="disclaimer">{_esc(DISCLAIMER)}</div></div></body></html>"""


def to_pdf(report: dict, settings: dict | None = None) -> bytes:
    """Professional PDF using ReportLab (tables + probability chart)."""
    from reportlab.graphics.charts.barcharts import HorizontalBarChart
    from reportlab.graphics.shapes import Drawing
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)

    settings = settings or config.load_settings()
    blue, navy, soft, line = (colors.HexColor(c) for c in ("#1769E0", "#142B4F", "#F5F9FF", "#E1EAF7"))
    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["BodyText"], fontName="Helvetica", fontSize=9.5, leading=13, textColor=navy)
    small = ParagraphStyle("s", parent=body, fontSize=8.5, leading=11, textColor=colors.HexColor("#5A6E8C"))
    h1 = ParagraphStyle("h1", parent=body, fontName="Helvetica-Bold", fontSize=18, leading=22)
    h2 = ParagraphStyle("h2", parent=body, fontName="Helvetica-Bold", fontSize=12, leading=16, textColor=blue, spaceBefore=10, spaceAfter=4)
    P = lambda t, st=body: Paragraph(html.escape(str(t)), st)

    def table(rows, widths, header=True):
        t = Table([[P(c) if not isinstance(c, Paragraph) else c for c in r] for r in rows], colWidths=widths, repeatRows=1 if header else 0)
        style = [("GRID", (0, 0), (-1, -1), 0.4, line), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                 ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]
        if header:
            style.append(("BACKGROUND", (0, 0), (-1, 0), soft))
        t.setStyle(TableStyle(style))
        return t

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
                            title=f"MediGuard AI report {report.get('id')}", author="MediGuard AI")
    W = doc.width
    story = [P("Medicine analysis report", h1),
             P(f"MediGuard AI — {config.TAGLINE}   |   Report {report.get('id')}   |   {report.get('created_at')}", small)]
    author = " · ".join(x for x in (settings.get("report_author"), settings.get("report_institution")) if x)
    if author:
        story.append(P(author, small))
    story += [Spacer(1, 6), P("A. Medicine identification", h2)]
    rows = [["Field", "Value", "Source"]] + [[f["label"], f["value"], f["status"]] for f in (report.get("fields") or {}).values()]
    rows += [[k.replace("_", " ").capitalize(), v, "Manually entered"] for k, v in (report.get("extra") or {}).items() if v]
    story.append(table(rows, [W * 0.28, W * 0.44, W * 0.28]))

    story.append(P("B. AI classification", h2))
    cls = report.get("classification") or {}
    if cls.get("ok"):
        names = list(cls["scores"])
        vals = [round(cls["scores"][n] * 100, 1) for n in names]
        d = Drawing(W, 22 * len(names) + 20)
        ch = HorizontalBarChart()
        ch.x, ch.y, ch.width, ch.height = 50, 10, W - 90, 22 * len(names)
        ch.data = [vals]
        ch.valueAxis.valueMin, ch.valueAxis.valueMax, ch.valueAxis.valueStep = 0, 100, 25
        ch.categoryAxis.categoryNames = names
        ch.bars[0].fillColor = blue
        for i, n in enumerate(names):
            ch.bars[(0, i)].fillColor = colors.HexColor("#E05A5A" if n.lower() == "fake" else "#2FA36B")
        ch.barLabelFormat = "%.1f%%"
        ch.barLabels.nudge = 14
        d.add(ch)
        story += [table([["Predicted class", "Model", "Version", "Timestamp"],
                         [cls["predicted_class"], cls.get("model_name"), cls.get("model_version"), cls.get("timestamp")]],
                        [W * 0.2, W * 0.36, W * 0.2, W * 0.24]),
                  table([["Class", "Score"]] + [[n, f"{v:.2f}%"] for n, v in zip(names, vals)], [W * 0.5, W * 0.5]),
                  d, P("Inference completed successfully.", small)]
    else:
        story.append(P("No classification was performed. " + (cls.get("error") or "Manual entry reports do not include an image classification.")))

    if settings.get("report_include_ml_explanation", True):
        story.append(P("C. Machine learning explanation", h2))
        story.append(table([[Paragraph(f"<b>{html.escape(a)}</b>", body), P(b)] for a, b in ML_EXPLANATION], [W * 0.24, W * 0.76], header=False))

    story.append(P("D. Model performance", h2))
    m = report.get("metrics") or {}
    if settings.get("report_include_metrics", True) and m.get("test"):
        story.append(table([["Metric", "Value", "Evaluated on"]] + [[k.replace("_", " ").capitalize(), f"{v*100:.2f}%", "Independent held-out test set"]
                            for k, v in m["test"].items() if isinstance(v, (int, float))], [W * 0.3, W * 0.2, W * 0.5]))
        if m.get("confusion_matrix"):
            labels = m.get("labels") or []
            story.append(KeepTogether([P("Confusion matrix (test set)", small),
                                       table([[""] + [f"Pred {l}" for l in labels]] + [[f"True {labels[i]}"] + [str(v) for v in r] for i, r in enumerate(m["confusion_matrix"])],
                                             [W / (len(labels) + 1)] * (len(labels) + 1))]))
    else:
        story.append(P("No measured model metrics are available.", small))

    obs = packaging_observations(report)
    story.append(P("E. Packaging observations", h2))
    for o in obs["observations"] or ["No specific observations."]:
        story.append(P("• " + o))
    story.append(P("Not detected: " + (", ".join(obs["missing"]) or "none"), small))
    for u in obs["uncertainty"]:
        story.append(P("– " + u, small))
    raw = (report.get("ocr") or {}).get("raw_text")
    if raw:
        story += [Spacer(1, 4), P("Raw detected text:", small), P(raw[:2500].replace("\n", " / "), small)]

    story.append(P("F. Safety and recommendation", h2))
    for r in recommendations(report):
        story.append(P("• " + r))
    story += [Spacer(1, 10), table([[P(DISCLAIMER, small)]], [W], header=False)]

    def footer(canvas, _doc):
        canvas.saveState()
        canvas.setFillColor(blue)
        canvas.rect(0, A4[1] - 6, A4[0], 6, stroke=0, fill=1)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#5A6E8C"))
        canvas.drawString(18 * mm, 9 * mm, "MediGuard AI — educational prototype, not a substitute for laboratory or regulatory verification")
        canvas.drawRightString(A4[0] - 18 * mm, 9 * mm, f"Page {_doc.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()
