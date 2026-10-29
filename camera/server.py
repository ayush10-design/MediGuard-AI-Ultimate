"""Local phone-to-laptop photo transfer server.

Run:  python camera/server.py
The phone (on the same Wi-Fi) opens http://<laptop-ip>:8765/capture?token=...
Photos are validated and saved to data/camera/inbox/<token>/ where the
Streamlit app picks them up.
"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import Flask, jsonify, render_template, request  # noqa: E402
from PIL import Image, UnidentifiedImageError  # noqa: E402

import config  # noqa: E402
from camera import sessions  # noqa: E402


def create_app(sessions_path: Path | None = None) -> Flask:
    app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"))
    app.config["MAX_CONTENT_LENGTH"] = config.CAMERA_MAX_UPLOAD_MB * 1024 * 1024
    sp = sessions_path

    @app.after_request
    def security_headers(resp):
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Referrer-Policy"] = "no-referrer"
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.get("/health")
    def health():
        return jsonify(status="ok", service="mediguard-camera", time=int(time.time()))

    @app.get("/capture")
    def capture():
        token = request.args.get("token")
        ok, msg = sessions.validate(token, sp)
        return render_template("capture.html", token=token if ok else "", error=None if ok else msg,
                               max_mb=config.CAMERA_MAX_UPLOAD_MB), (200 if ok else 403)

    @app.post("/upload")
    def upload():
        token = request.args.get("token") or request.form.get("token")
        ok, msg = sessions.validate(token, sp)
        if not ok:
            return jsonify(ok=False, error=msg), 403
        f = request.files.get("photo")
        if f is None or not f.filename:
            return jsonify(ok=False, error="No photo received."), 400
        ext = Path(f.filename).suffix.lower()
        if ext not in config.ALLOWED_IMAGE_EXTENSIONS:
            return jsonify(ok=False, error="Only JPG, JPEG, PNG or WEBP photos are accepted. "
                                           "On iPhone, set Camera › Formats to 'Most Compatible' if HEIC is used."), 415
        data = f.read()
        try:
            Image.open(io.BytesIO(data)).verify()
        except (UnidentifiedImageError, OSError, SyntaxError):
            return jsonify(ok=False, error="The file is not a valid image."), 415
        out_dir = sessions.inbox_dir(token)
        out_dir.mkdir(parents=True, exist_ok=True)
        name = f"phone_{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time()*1000) % 1000:03d}{ext}"
        (out_dir / name).write_bytes(data)
        sessions.record_upload(token, sp)
        return jsonify(ok=True, filename=name, message="Photo sent to MediGuard AI.")

    @app.errorhandler(413)
    def too_large(_e):
        return jsonify(ok=False, error=f"Photo is larger than {config.CAMERA_MAX_UPLOAD_MB} MB."), 413

    return app


if __name__ == "__main__":
    config.ensure_directories()
    host = "0.0.0.0"  # reachable from devices on the same local network only (not port-forwarded)
    ip = sessions.local_ip()
    print(f"MediGuard camera server running on http://{ip}:{config.CAMERA_PORT}  (Ctrl+C to stop)")
    print("Only devices on your local Wi-Fi can reach it. macOS may ask to allow incoming connections — click Allow.")
    create_app().run(host=host, port=config.CAMERA_PORT, debug=False, threaded=True)
