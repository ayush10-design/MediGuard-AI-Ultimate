import json

import pandas as pd
import streamlit as st

import config
from utils import ui, visualization as viz
from utils.model_utils import load_history, load_metrics, load_training_config, model_status

ui.page_header("Model Performance", "Measured results of the trained classifier. Nothing here is estimated or simulated.", "chart")
ms, metrics, history, cfg = model_status(), load_metrics(), load_history(), load_training_config()

if not ms["trained"] or not metrics:
    ui.note("<b>Model Not Trained.</b> Performance appears here after you run <code>python train_model.py</code> "
            "with images in <code>data/train/Fake</code> and <code>data/train/Real</code>.", "warn")
    st.stop()

t = metrics["test"]
c = st.columns(4)
for col, (k, label) in zip(c, [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1-score")]):
    col.metric(label, f"{t[k]*100:.2f}%")
st.caption(f"All four metrics are from the **{metrics['evaluated_on']}** "
           f"({metrics['split_sizes']['test']} images). Precision/recall treat **{metrics.get('positive_class_for_binary_metrics')}** "
           f"as the positive class. Validation accuracy: {metrics['validation']['accuracy']*100:.2f}%.")

dups = metrics.get("near_duplicates") or {}
if dups.get("test") or dups.get("validation"):
    ui.note(f"<b>Possible data leakage:</b> {dups.get('test', 0)} test and {dups.get('validation', 0)} validation images are "
            "near-identical to a training image. Product photos taken in the same session (same pack, angle, background) "
            "make test scores look better than real-world performance. Prefer splitting by product or photo session.", "warn")
else:
    ui.note("No near-duplicate images were detected between the training and test splits (perceptual-hash check). "
            "Leakage can still exist if the same product appears in different photos.", "")

a, b = st.columns(2)
with a:
    st.plotly_chart(viz.metric_bars({"Accuracy": t["accuracy"], "Precision": t["precision"], "Recall": t["recall"], "F1": t["f1"]}),
                    use_container_width=True)
with b:
    st.plotly_chart(viz.confusion_matrix(metrics["confusion_matrix"], metrics["labels"]), use_container_width=True)

if history:
    a, b = st.columns(2)
    a.plotly_chart(viz.training_curves(history, "accuracy"), use_container_width=True)
    b.plotly_chart(viz.training_curves(history, "loss"), use_container_width=True)

st.markdown("#### Per-class report (test set)")
rep = metrics.get("classification_report") or {}
rows = [{"Class": k, "Precision": round(v["precision"], 3), "Recall": round(v["recall"], 3), "F1": round(v["f1-score"], 3), "Images": int(v["support"])}
        for k, v in rep.items() if isinstance(v, dict)]
if rows:
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

with st.expander("Training configuration"):
    st.json(cfg or {})
