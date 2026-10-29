import streamlit as st

import config
from utils import ui

ui.page_header("About MediGuard AI", config.TAGLINE, "info")
a, b = st.columns([1.4, 1], gap="large")
with a:
    st.markdown(f"""
**{config.APP_FULL_NAME}** (version {config.APP_VERSION}) is a college project that combines three separate tools
for screening medicine packaging:

1. **Label reading** — Google Gemini or offline Tesseract OCR extracts the printed details, which you can correct.
2. **Visual classification** — a MobileNetV2 network, fine-tuned on a labelled Fake/Real image dataset, scores how
   closely a photo resembles each class.
3. **Reporting** — a structured report separates what was read, what the model predicted, what was measured about
   the model, and what to do next.

#### What it cannot do
- It cannot test the chemical contents or quality of a medicine.
- It cannot confirm a batch with the manufacturer or regulator.
- A model trained on a few hundred photos can be fooled by lighting, angle, background or new packaging designs.
- Catalogue suggestions in manual entry are a typing aid, not a register of approved products.

If you suspect a falsified medicine in India, contact your pharmacist, the manufacturer, or report it to CDSCO.
""")
with b:
    st.markdown(ui.card("Technology", "Python 3.11 · Streamlit · TensorFlow/Keras MobileNetV2 · Google Gen AI SDK · "
                        "Tesseract · OpenCV · Pillow · Plotly · SQLite · Flask · ReportLab"), unsafe_allow_html=True)
    st.markdown(ui.card("Privacy", "Images stay on this laptop unless you tick the Gemini consent box. "
                        "History stores results only, not photos. Phone uploads stay in data/camera/inbox."), unsafe_allow_html=True)
    st.markdown(ui.card("Original reference", "Kaggle notebook “Fake Medicine Detector” by surajkumarjha1, using the "
                        "Fake vs Real Medicine Dataset (images)."), unsafe_allow_html=True)
