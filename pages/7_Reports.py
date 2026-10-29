import streamlit as st

from utils import history_db, ui
from utils.report_view import render_report

S = st.session_state
ui.page_header("Detailed Reports", "Open the latest analysis or any saved one, then download it as PDF, CSV or printable HTML.", "report")

df = history_db.list_records(limit=200)
options = {}
if S.get("last_report"):
    options["Latest analysis (this session)"] = ("session", None)
for _, r in df.iterrows():
    label = f"#{r.id} · {r.timestamp} · {r.medicine_name or r.filename or 'manual entry'} · {r.predicted_class or 'not classified'}"
    options[label] = ("db", int(r.id))

if not options:
    ui.note("No reports yet. Analyse an image in the Detection Lab or create a manual entry first.")
    st.stop()

default = 0
if S.get("selected_report_id"):
    for i, (_, v) in enumerate(options.items()):
        if v == ("db", S["selected_report_id"]):
            default = i
choice = st.selectbox("Report", list(options), index=default)
kind, rid = options[choice]
report = S["last_report"] if kind == "session" else history_db.get_report(rid)
if not report:
    st.error("This report could not be loaded.")
    st.stop()
render_report(report, key=f"rep_{report.get('id')}")
