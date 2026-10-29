"""Expiring phone-upload session tokens shared between Streamlit and the Flask server.

Tokens live in a small JSON file so the two processes agree without a database.
"""
from __future__ import annotations

import json
import secrets
import socket
import time
from pathlib import Path

import config


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    tmp.replace(path)


def create_session(ttl_minutes: int = config.CAMERA_TOKEN_TTL_MINUTES, path: Path | None = None) -> dict:
    path = path or config.CAMERA_SESSIONS_PATH
    data = {t: s for t, s in _load(path).items() if s.get("expires", 0) > time.time()}  # prune expired
    token = secrets.token_urlsafe(18)
    session = {"token": token, "created": time.time(), "expires": time.time() + ttl_minutes * 60, "uploads": 0}
    data[token] = session
    _save(path, data)
    return session


def validate(token: str | None, path: Path | None = None) -> tuple[bool, str]:
    if not token:
        return False, "Missing session token. Scan the QR code shown on the laptop again."
    s = _load(path or config.CAMERA_SESSIONS_PATH).get(token)
    if not s:
        return False, "Unknown session. Generate a new QR code on the laptop."
    if s["expires"] < time.time():
        return False, "This session has expired. Generate a new QR code on the laptop."
    if s.get("uploads", 0) >= config.CAMERA_MAX_UPLOADS_PER_SESSION:
        return False, "Upload limit for this session reached. Generate a new QR code."
    return True, "ok"


def record_upload(token: str, path: Path | None = None) -> None:
    path = path or config.CAMERA_SESSIONS_PATH
    data = _load(path)
    if token in data:
        data[token]["uploads"] = data[token].get("uploads", 0) + 1
        _save(path, data)


def revoke(token: str, path: Path | None = None) -> None:
    path = path or config.CAMERA_SESSIONS_PATH
    data = _load(path)
    data.pop(token, None)
    _save(path, data)


def inbox_dir(token: str) -> Path:
    safe = "".join(c for c in token if c.isalnum() or c in "-_")
    return config.CAMERA_INBOX_DIR / safe


def list_inbox(token: str) -> list[Path]:
    d = inbox_dir(token)
    if not d.exists():
        return []
    return sorted((p for p in d.iterdir() if p.suffix.lower() in config.ALLOWED_IMAGE_EXTENSIONS),
                  key=lambda p: p.stat().st_mtime, reverse=True)


def local_ip() -> str:
    """Best-effort LAN address (no packets are actually sent)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()
