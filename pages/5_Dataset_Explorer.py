import random

import streamlit as st

import config
from utils import ui, visualization as viz
from utils.data_loader import scan_dataset

ui.page_header("Dataset Explorer", "What the classifier learns from: class folders, image counts and samples.", "data")


@st.cache_data(ttl=120, show_spinner="Checking every image…")
def summary():
    s = scan_dataset(verify=True)
    return {"exists": s.exists, "root": str(s.root), "classes": s.classes, "counts": s.counts, "invalid": s.invalid,
            "files": {c: [str(p) for p in v] for c, v in s.files.items()}}


if st.button("Rescan dataset"):
    summary.clear()
s = summary()

if not s["exists"] or not sum(s["counts"].values()):
    ui.note(f"<b>No images found</b> in <code>{s['root']}</code>. Create <code>Fake/</code> and <code>Real/</code> folders "
            "inside it and copy the labelled images there (see README → Dataset).", "warn")
    st.stop()

cols = st.columns(len(s["classes"]) + 2)
for col, c in zip(cols, s["classes"]):
    col.markdown(ui.stat(s["counts"][c], f"{c} images", "red" if c.lower() == "fake" else "green"), unsafe_allow_html=True)
cols[-2].markdown(ui.stat(sum(s["counts"].values()), "Valid images"), unsafe_allow_html=True)
cols[-1].markdown(ui.stat(len(s["invalid"]), "Invalid / unreadable", "teal"), unsafe_allow_html=True)

a, b = st.columns([1, 1.2])
with a:
    st.plotly_chart(viz.class_distribution(s["counts"]), use_container_width=True)
with b:
    st.markdown("#### Source and class mapping")
    st.markdown("""Reference dataset: **Fake vs Real Medicine Dataset (images)** on Kaggle, used by the notebook
*Fake Medicine Detector* (kaggle.com/code/surajkumarjha1/fake-medicine-detector). Download it yourself and place
the class folders under `data/train/`.""")
    st.markdown("Class indices are assigned alphabetically from folder names at training time:")
    st.table({"Index": list(range(len(s["classes"]))), "Class folder": s["classes"]})
    counts = list(s["counts"].values())
    if min(counts) and max(counts) / min(counts) > 1.2:
        ui.note(f"Classes are imbalanced (ratio {max(counts)/min(counts):.1f}:1). Training applies class weights automatically.", "warn")

st.markdown("#### Sample images")
pick = st.segmented_control("Class", s["classes"], default=s["classes"][0]) if hasattr(st, "segmented_control") \
    else st.radio("Class", s["classes"], horizontal=True)
files = s["files"].get(pick or s["classes"][0], [])
if files:
    random.seed(7)
    sample = random.sample(files, min(12, len(files)))
    grid = st.columns(6)
    for i, f in enumerate(sample):
        grid[i % 6].image(f, use_container_width=True)
if s["invalid"]:
    with st.expander(f"{len(s['invalid'])} invalid files (skipped during training)"):
        st.write(s["invalid"])
