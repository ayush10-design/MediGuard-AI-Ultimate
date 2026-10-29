import io
import subprocess
import sys
import time
import urllib.request
from datetime import datetime

import qrcode
import streamlit as st

import config
from camera import sessions
from utils import ui

S = st.session_state
ui.page_header("Live Camera Scan", "Use your phone as a camera: scan the QR code, photograph the pack and send it here.", "camera")

PORT = config.CAMERA_PORT


def server_alive() -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1.0) as r:
            return r.status == 200
    except Exception:
        return False


def start_server() -> str | None:
    if server_alive():
        return None
    log = open(config.CAMERA_DIR / "server.log", "ab")
    try:
        proc = subprocess.Popen([sys.executable, str(config.BASE_DIR / "camera" / "server.py")], cwd=str(config.BASE_DIR),
                                stdout=log, stderr=subprocess.STDOUT)
    except OSError as exc:
        return f"Could not start the camera server: {exc}"
    S["camera_proc_pid"] = proc.pid
    for _ in range(25):
        time.sleep(0.2)
        if server_alive():
            return None
        if proc.poll() is not None:
            break
    return ("The camera server did not start. Port {p} may be in use — set MEDIGUARD_CAMERA_PORT in .env, "
            "or run it yourself: python camera/server.py (see data/camera/server.log)").format(p=PORT)


alive = server_alive()
session = S.get("camera_session")
if session and session["expires"] < time.time():
    session = None
    S.pop("camera_session", None)

left, right = st.columns([1, 1.1], gap="large")
with left:
    st.markdown("#### Connection")
    st.markdown(f"{ui.chip('Camera server running' if alive else 'Camera server stopped', 'ok' if alive else 'warn')} "
                f"{ui.chip('Session active' if session else 'No session', 'ok' if session else '')}", unsafe_allow_html=True)
    st.write("")
    if st.button("Connect Phone", type="primary", use_container_width=True):
        err = start_server()
        if err:
            st.error(err)
        else:
            S["camera_session"] = sessions.create_session()
            S["camera_seen"] = set()
            st.rerun()
    if session and st.button("End session", use_container_width=True):
        sessions.revoke(session["token"])
        S.pop("camera_session", None)
        st.rerun()

    with st.expander("Phone can't connect?"):
        st.markdown(f"""
- Both devices must be on the **same Wi-Fi** network (guest or office networks often block device-to-device traffic).
- macOS may show *“Do you want the application Python to accept incoming network connections?”* — click **Allow**.
  If you clicked Deny: **System Settings › Network › Firewall › Options**, find Python and set it to *Allow*.
- Check the server: open `http://localhost:{PORT}/health` on the laptop. It should show `"status": "ok"`.
- The camera opens through a normal photo picker (`capture="environment"`), which works over plain local HTTP.
  Live in-browser video would need HTTPS, so it is intentionally not used.
- iPhone HEIC photos are converted to JPEG automatically by Safari when uploading.
- The server listens only on your local network and is not exposed to the internet. Stop it by closing this app or with Ctrl+C.
""")

with right:
    if session:
        ip = sessions.local_ip()
        url = f"http://{ip}:{PORT}/capture?token={session['token']}"
        qr = qrcode.QRCode(border=2, box_size=9, error_correction=qrcode.constants.ERROR_CORRECT_M)
        qr.add_data(url); qr.make(fit=True)
        buf = io.BytesIO()
        qr.make_image(fill_color="#142B4F", back_color="white").save(buf, format="PNG")
        mins = max(0, int((session["expires"] - time.time()) // 60))
        st.markdown('<div class="mg-card" style="text-align:center">', unsafe_allow_html=True)
        import base64
        qr_data = base64.b64encode(buf.getvalue()).decode("ascii")
        st.markdown(
            f'<img src="data:image/png;base64,{qr_data}" '
            f'style="width:260px;max-width:100%;height:auto;" '
            f'alt="Phone connection QR code">',
            unsafe_allow_html=True
        )
        st.markdown(f"**Scan with your phone camera** · link expires in about {mins} min", unsafe_allow_html=True)
        st.code(url, language=None)
        st.markdown("</div>", unsafe_allow_html=True)
        if ip.startswith("127."):
            ui.note("Could not find this laptop's Wi-Fi address. Connect to Wi-Fi and press Connect Phone again.", "bad")
    else:
        st.markdown(ui.card("How it works", "Press <b>Connect Phone</b>. A private QR code appears here. Scan it, take a photo "
                            "of the pack and tap <b>Send to MediGuard AI</b>. The photo arrives on this screen within seconds."),
                    unsafe_allow_html=True)


@st.fragment(run_every=3)
def inbox():
    sess = S.get("camera_session")
    if not sess:
        return
    files = sessions.list_inbox(sess["token"])
    st.markdown("#### Photos received")
    if not files:
        st.markdown(f'<div class="mg-note"><span class="mg-pulse">●</span> Waiting for a photo from your phone… '
                    f'(checked {datetime.now():%H:%M:%S})</div>', unsafe_allow_html=True)
        return
    new = [f for f in files if f.name not in S.setdefault("camera_seen", set())]
    if new:
        st.toast(f"New photo received: {new[0].name}")
        S["camera_seen"].update(f.name for f in new)
    cols = st.columns(4)
    for i, f in enumerate(files[:8]):
        with cols[i % 4]:
            import base64
            import mimetypes
            photo_data = base64.b64encode(f.read_bytes()).decode("ascii")
            mime = mimetypes.guess_type(f.name)[0] or "image/jpeg"
            st.markdown(
                f'<img src="data:{mime};base64,{photo_data}" '
                f'style="width:100%;height:auto;border-radius:10px;" '
                f'alt="Received medicine photo">',
                unsafe_allow_html=True
            )
            st.caption(f"{datetime.fromtimestamp(f.stat().st_mtime):%H:%M:%S}")
            if st.button("Analyse", key=f"use_{f.name}", use_container_width=True):
                S["phone_image"] = {"bytes": f.read_bytes(), "filename": f.name}
                S["prefer_phone"] = True
                for k in ("lab_fields", "lab_ocr", "lab_report"):
                    S.pop(k, None)
                st.switch_page("pages/1_Detection_Lab.py")


inbox()
st.caption("Photos are kept in data/camera/inbox only for this demo session; delete that folder any time.")
