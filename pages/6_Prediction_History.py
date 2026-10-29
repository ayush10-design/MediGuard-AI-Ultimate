import pandas as pd
import streamlit as st

from utils import history_db, ui

S = st.session_state
ui.page_header("Prediction History", "Every saved analysis on this laptop. Images themselves are not stored.", "history")

c1, c2, c3 = st.columns([2, 1, 1])
search = c1.text_input("Search by file or medicine name", placeholder="e.g. Dolo or phone_2025")
pred = c2.selectbox("Prediction", ["All", "Fake", "Real"])
c3.write(""); c3.write("")
df = history_db.list_records(search=search, predicted=None if pred == "All" else pred)

if df.empty:
    ui.note("No matching analyses. Run a screening in the Detection Lab or create a manual entry.")
    st.stop()

view = df.copy()
for c in ("fake_score", "real_score"):
    view[c] = view[c].map(lambda v: round(v * 100, 2) if pd.notna(v) else None)
view = view.rename(columns={"id": "ID", "timestamp": "Time", "filename": "File", "analysis_type": "Type",
                            "predicted_class": "Prediction", "fake_score": "Fake %", "real_score": "Real %", "medicine_name": "Medicine"})
st.dataframe(view, hide_index=True, use_container_width=True,
             column_config={"Fake %": st.column_config.ProgressColumn("Fake %", min_value=0, max_value=100, format="%.1f%%"),
                            "Real %": st.column_config.ProgressColumn("Real %", min_value=0, max_value=100, format="%.1f%%")})

a, b, c = st.columns([1, 1, 1])
a.download_button("Export CSV", df.to_csv(index=False).encode(), "mediguard_history.csv", "text/csv", use_container_width=True)
rid = b.selectbox("Open report", df["id"].tolist(), format_func=lambda i: f"#{i} · {df.loc[df.id == i, 'timestamp'].iloc[0]}",
                  label_visibility="collapsed")
if b.button("View report", type="primary", use_container_width=True):
    S["selected_report_id"] = int(rid)
    st.switch_page("pages/7_Reports.py")
with c.popover("Delete…", use_container_width=True):
    if st.button(f"Delete record #{rid}"):
        history_db.delete_record(int(rid)); st.rerun()
    if st.button("Delete all history", type="primary"):
        history_db.clear_all(); st.rerun()
