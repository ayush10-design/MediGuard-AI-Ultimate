"""MediGuard AI — entry point and navigation.

Run with:  python -m streamlit run app.py
"""
import streamlit as st

import config

config.ensure_directories()
st.set_page_config(page_title="MediGuard AI", page_icon=str(config.ASSETS_DIR / "logo.svg"),
                   layout="wide", initial_sidebar_state="expanded")

from utils import ui  # noqa: E402  (after set_page_config)

ui.apply_theme()
ui.sidebar_brand()

pages = {
    "Screening": [
        st.Page("pages/0_Dashboard.py", title="Dashboard", icon=":material/space_dashboard:", default=True),
        st.Page("pages/1_Detection_Lab.py", title="Detection Lab", icon=":material/biotech:"),
        st.Page("pages/2_Camera_Scan.py", title="Live Camera Scan", icon=":material/photo_camera:"),
        st.Page("pages/3_Manual_Entry.py", title="Manual Medicine Entry", icon=":material/edit_note:"),
    ],
    "Insights": [
        st.Page("pages/4_Model_Performance.py", title="Model Performance", icon=":material/monitoring:"),
        st.Page("pages/5_Dataset_Explorer.py", title="Dataset Explorer", icon=":material/photo_library:"),
        st.Page("pages/6_Prediction_History.py", title="Prediction History", icon=":material/history:"),
        st.Page("pages/7_Reports.py", title="Detailed Reports", icon=":material/description:"),
    ],
    "System": [
        st.Page("pages/8_Settings.py", title="Settings", icon=":material/tune:"),
        st.Page("pages/9_About.py", title="About", icon=":material/info:"),
    ],
}
st.navigation(pages).run()
