"""Central configuration for MediGuard AI.

All paths are relative to the project folder so the project can be moved
anywhere on disk. Secrets are read from environment variables or a local
.env file and are never hard-coded.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv is optional at import time
    load_dotenv = None

BASE_DIR = Path(__file__).resolve().parent
if load_dotenv is not None:
    load_dotenv(BASE_DIR / ".env")

APP_NAME = "MediGuard AI"
APP_FULL_NAME = "MediGuard AI — Intelligent Medicine Authentication & Analysis System"
TAGLINE = "Smarter Medicine Screening. Safer Tomorrow."
APP_VERSION = "1.0.0"

# ---------------------------------------------------------------- paths
DATA_DIR = BASE_DIR / "data"
DATASET_DIR = Path(os.getenv("MEDIGUARD_DATASET_DIR", DATA_DIR / "train"))
CATALOGUE_PATH = DATA_DIR / "catalogue.csv"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
ASSETS_DIR = BASE_DIR / "assets"
DB_PATH = Path(os.getenv("MEDIGUARD_DB_PATH", DATA_DIR / "history.db"))
SETTINGS_PATH = DATA_DIR / "settings.json"
CAMERA_DIR = DATA_DIR / "camera"
CAMERA_SESSIONS_PATH = CAMERA_DIR / "sessions.json"
CAMERA_INBOX_DIR = CAMERA_DIR / "inbox"

MODEL_PATH = MODELS_DIR / "mediguard_mobilenetv2.keras"
CLASS_MAP_PATH = MODELS_DIR / "class_mapping.json"
HISTORY_PATH = MODELS_DIR / "training_history.json"
METRICS_PATH = MODELS_DIR / "metrics.json"
TRAIN_CONFIG_PATH = MODELS_DIR / "training_config.json"
CLASS_REPORT_PATH = MODELS_DIR / "classification_report.txt"

# ---------------------------------------------------------------- model
IMG_SIZE = (224, 224)
MODEL_NAME = "MobileNetV2 (transfer learning)"
SEED = 42
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# ---------------------------------------------------------------- Gemini
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_TIMEOUT_SECONDS = 120

# ---------------------------------------------------------------- camera
CAMERA_PORT = int(os.getenv("MEDIGUARD_CAMERA_PORT", "8765"))
CAMERA_TOKEN_TTL_MINUTES = 15
CAMERA_MAX_UPLOAD_MB = 10
CAMERA_MAX_UPLOADS_PER_SESSION = 25

DEFAULT_SETTINGS = {
    "ocr_provider": "auto",          # auto | gemini | tesseract | none
    "gemini_model": DEFAULT_GEMINI_MODEL,
    "animations": True,
    "custom_cursor": True,
    "compact_layout": False,
    "report_include_metrics": True,
    "report_include_ml_explanation": True,
    "report_author": "",
    "report_institution": "",
    "save_history": True,
}


def get_gemini_api_key() -> str | None:
    key = os.getenv(GEMINI_API_KEY_ENV, "").strip()
    return key or None


def mask_secret(value: str | None) -> str:
    if not value:
        return "Not configured"
    if len(value) <= 8:
        return "••••"
    return f"{value[:4]}••••••••{value[-4:]}"


def load_settings() -> dict:
    settings = dict(DEFAULT_SETTINGS)
    try:
        if SETTINGS_PATH.exists():
            stored = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            settings.update({k: v for k, v in stored.items() if k in DEFAULT_SETTINGS})
    except (OSError, json.JSONDecodeError):
        pass
    return settings


def save_settings(settings: dict) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean = {k: settings.get(k, v) for k, v in DEFAULT_SETTINGS.items()}
    SETTINGS_PATH.write_text(json.dumps(clean, indent=2), encoding="utf-8")


def ensure_directories() -> None:
    for d in (DATA_DIR, MODELS_DIR, REPORTS_DIR, CAMERA_DIR, CAMERA_INBOX_DIR):
        d.mkdir(parents=True, exist_ok=True)
