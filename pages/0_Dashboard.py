import pandas as pd
import streamlit as st

import config
from utils import history_db, ui, visualization as viz
from utils.data_loader import scan_dataset
from utils.model_utils import load_metrics, model_status

HERO_ART = """<svg class="mg-hero-art" viewBox="0 0 320 220" aria-hidden="true">
<rect x="40" y="26" width="240" height="168" rx="26" fill="rgba(255,255,255,.12)" stroke="rgba(255,255,255,.35)"/>
<g fill="rgba(255,255,255,.9)">
 <rect x="70" y="54" width="44" height="44" rx="22"/><rect x="138" y="54" width="44" height="44" rx="22"/><rect x="206" y="54" width="44" height="44" rx="22"/>
 <rect x="70" y="122" width="44" height="44" rx="22"/><rect x="138" y="122" width="44" height="44" rx="22"/><rect x="206" y="122" width="44" height="44" rx="22" opacity=".45"/>
</g>
<rect class="mg-pulse" x="28" y="104" width="264" height="3" rx="1.5" fill="#A8D4FF"/>
<path d="M28 34v-14h14M292 34v-14h-14M28 186v14h14M292 186v14h-14" stroke="#A8D4FF" stroke-width="4" fill="none" stroke-linecap="round"/>
</svg>"""


@st.cache_data(ttl=60, show_spinner=False)
def dataset_counts():
    s = scan_dataset(verify=False)
    return s.counts, s.exists


ms = model_status()
counts, ds_exists = dataset_counts()
stats = history_db.stats()
metrics = load_metrics()

chips = [ui.chip("Model trained" if ms["trained"] else "Model not trained", "ok" if ms["trained"] else "warn"),
         ui.chip(f"{sum(counts.values())} dataset images", "info"), ui.chip(f"{stats['total']} analyses saved", "info")]
st.markdown(f"""<div class="mg-hero">{HERO_ART}
<h1>Screen medicine packaging in under a minute</h1>
<p>Photograph a pack, read its printed details, and see how a trained neural network classifies it — with every result
labelled for what it is, and what it is not.</p>
<div class="mg-chiprow">{''.join(c.replace('mg-chip ', 'mg-chip ') for c in chips)}</div></div>""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.markdown(ui.stat("Ready" if ms["trained"] else "Not trained", "Classifier status", "teal"), unsafe_allow_html=True)
c2.markdown(ui.stat(f"{metrics['test']['accuracy']*100:.1f}%" if metrics else "—", "Test accuracy (measured)"), unsafe_allow_html=True)
c3.markdown(ui.stat(sum(counts.values()), "Images in dataset", "green"), unsafe_allow_html=True)
c4.markdown(ui.stat(stats["total"], "Completed analyses", "red"), unsafe_allow_html=True)

if not ms["trained"]:
    ui.note(f"<b>Model not trained.</b> {ms['message']}", "warn")

st.markdown("### Start a screening")
q1, q2, q3, q4 = st.columns(4)
with q1:
    st.markdown(ui.card("Upload a photo", "Analyse a JPG, PNG or WEBP image of the pack."), unsafe_allow_html=True)
    st.page_link("pages/1_Detection_Lab.py", label="Open Detection Lab", icon=":material/biotech:")
with q2:
    st.markdown(ui.card("Use your phone", "Scan a QR code and send a photo from your phone camera."), unsafe_allow_html=True)
    st.page_link("pages/2_Camera_Scan.py", label="Connect phone", icon=":material/photo_camera:")
with q3:
    st.markdown(ui.card("Type the details", "Enter printed details with smart suggestions."), unsafe_allow_html=True)
    st.page_link("pages/3_Manual_Entry.py", label="Manual entry", icon=":material/edit_note:")
with q4:
    st.markdown(ui.card("Review results", "Search past analyses and download reports."), unsafe_allow_html=True)
    st.page_link("pages/6_Prediction_History.py", label="Prediction history", icon=":material/history:")

left, right = st.columns([1, 1.4])
with left:
    st.markdown("### Dataset")
    if counts and sum(counts.values()):
        st.plotly_chart(viz.class_distribution(counts, "Real / Fake images"), use_container_width=True)
    else:
        ui.note("No images found in <code>data/train/</code>. Add <code>Fake/</code> and <code>Real/</code> folders to see the distribution.")
with right:
    st.markdown("### Recent analyses")
    df = history_db.list_records(limit=50)
    if len(df):
        st.plotly_chart(viz.history_timeline(df), use_container_width=True)
        view = df.head(6)[["timestamp", "filename", "predicted_class", "fake_score", "real_score", "medicine_name"]].copy()
        for c in ("fake_score", "real_score"):
            view[c] = view[c].map(lambda v: f"{v*100:.1f}%" if pd.notna(v) else "—")
        view.columns = ["Time", "File", "Prediction", "Fake", "Real", "Medicine"]
        st.dataframe(view, hide_index=True, use_container_width=True)
    else:
        ui.note("No analyses yet. Results appear here after your first screening.")

st.markdown("### How a screening works")
st.markdown("""<div class="mg-steps">
<div class="mg-step"><b>Capture</b><span>Upload, or photograph with your phone</span></div>
<div class="mg-step"><b>Read the label</b><span>Gemini or Tesseract extracts printed text</span></div>
<div class="mg-step"><b>Classify</b><span>MobileNetV2 scores the image Fake vs Real</span></div>
<div class="mg-step"><b>Review</b><span>You correct any misread field</span></div>
<div class="mg-step"><b>Report</b><span>Download PDF, CSV or printable HTML</span></div>
</div>""", unsafe_allow_html=True)
st.caption("MediGuard AI is an educational prototype. A visual classification is not proof that a medicine is genuine or safe.")
