import io
import time

import pytest
from PIL import Image

import config
from camera import sessions
from camera.server import create_app


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CAMERA_INBOX_DIR", tmp_path / "inbox")
    sp = tmp_path / "sessions.json"
    app = create_app(sessions_path=sp)
    app.testing = True
    return app.test_client(), sp


def png_bytes():
    b = io.BytesIO(); Image.new("RGB", (40, 40), "blue").save(b, "PNG"); return b.getvalue()


def test_health(env):
    client, _ = env
    assert client.get("/health").get_json()["status"] == "ok"


def test_upload_flow(env):
    client, sp = env
    s = sessions.create_session(path=sp)
    assert client.get(f"/capture?token={s['token']}").status_code == 200
    r = client.post(f"/upload?token={s['token']}", data={"photo": (io.BytesIO(png_bytes()), "pack.png")},
                    content_type="multipart/form-data")
    assert r.status_code == 200 and r.get_json()["ok"]
    assert len(sessions.list_inbox(s["token"])) == 1


def test_rejects_bad_token_type_and_content(env):
    client, sp = env
    assert client.get("/capture?token=nope").status_code == 403
    assert client.post("/upload?token=nope", data={"photo": (io.BytesIO(png_bytes()), "a.png")},
                       content_type="multipart/form-data").status_code == 403
    s = sessions.create_session(path=sp)
    assert client.post(f"/upload?token={s['token']}", data={"photo": (io.BytesIO(b"hello"), "a.exe")},
                       content_type="multipart/form-data").status_code == 415
    assert client.post(f"/upload?token={s['token']}", data={"photo": (io.BytesIO(b"not an image"), "a.jpg")},
                       content_type="multipart/form-data").status_code == 415


def test_expired_token(env, monkeypatch):
    client, sp = env
    s = sessions.create_session(ttl_minutes=1, path=sp)
    real = time.time
    monkeypatch.setattr(time, "time", lambda: real() + 120)
    ok, msg = sessions.validate(s["token"], sp)
    assert not ok and "expired" in msg


def test_size_limit(env):
    client, sp = env
    s = sessions.create_session(path=sp)
    big = b"0" * (config.CAMERA_MAX_UPLOAD_MB * 1024 * 1024 + 10)
    r = client.post(f"/upload?token={s['token']}", data={"photo": (io.BytesIO(big), "a.jpg")},
                    content_type="multipart/form-data")
    assert r.status_code == 413
