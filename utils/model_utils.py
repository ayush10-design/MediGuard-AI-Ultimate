"""Loading the trained classifier and running real inference.

TensorFlow is imported lazily so the app starts even when TensorFlow or a
trained model is missing.
"""
from __future__ import annotations

import json
from datetime import datetime
from functools import lru_cache
from typing import Any

import numpy as np
from PIL import Image

import config
from utils.image_processing import prepare_for_model


def _read_json(path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def tensorflow_available() -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec("tensorflow") is not None
    except Exception:
        return False


def model_status() -> dict[str, Any]:
    """Describe whether a usable trained model exists, without loading TensorFlow."""
    exists = config.MODEL_PATH.exists() and config.CLASS_MAP_PATH.exists()
    cfg = _read_json(config.TRAIN_CONFIG_PATH) or {}
    mapping = _read_json(config.CLASS_MAP_PATH) or {}
    status = {
        "trained": exists,
        "model_path": config.MODEL_PATH.name,
        "classes": [mapping[k] for k in sorted(mapping, key=int)] if mapping else [],
        "version": cfg.get("model_version"),
        "trained_at": cfg.get("trained_at"),
        "tensorflow": tensorflow_available(),
    }
    if not exists:
        status["message"] = "Model Not Trained — add images to data/train/<Class>/ and run: python train_model.py"
    elif not status["tensorflow"]:
        status["message"] = "A trained model exists but TensorFlow is not installed. Run: python -m pip install -r requirements.txt"
    else:
        status["message"] = "Model ready"
    return status


def load_metrics() -> dict | None:
    return _read_json(config.METRICS_PATH)


def load_history() -> dict | None:
    return _read_json(config.HISTORY_PATH)


def load_training_config() -> dict | None:
    return _read_json(config.TRAIN_CONFIG_PATH)


@lru_cache(maxsize=2)
def _load_model(path: str, mtime: float):
    import tensorflow as tf  # noqa: WPS433 (lazy import by design)
    return tf.keras.models.load_model(path)


def load_model():
    return _load_model(str(config.MODEL_PATH), config.MODEL_PATH.stat().st_mtime)


def predict(img: Image.Image) -> dict[str, Any]:
    """Run the trained model on one image. Returns an error dict instead of fake scores."""
    status = model_status()
    if not status["trained"]:
        return {"ok": False, "error": status["message"]}
    if not status["tensorflow"]:
        return {"ok": False, "error": status["message"]}
    mapping = _read_json(config.CLASS_MAP_PATH)
    try:
        model = load_model()
        batch = prepare_for_model(img)
        probs = np.asarray(model.predict(batch, verbose=0))[0].astype(float)
    except Exception as exc:
        return {"ok": False, "error": f"Inference failed: {exc}"}
    classes = [mapping[k] for k in sorted(mapping, key=int)]
    if probs.shape[0] != len(classes):
        return {"ok": False, "error": "Model output does not match the saved class mapping. Retrain the model."}
    scores = {c: float(p) for c, p in zip(classes, probs)}
    top = max(scores, key=scores.get)
    return {
        "ok": True,
        "predicted_class": top,
        "scores": scores,
        "model_name": config.MODEL_NAME,
        "model_version": status.get("version") or "unknown",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def score_for(scores: dict[str, float], name: str) -> float | None:
    for k, v in scores.items():
        if k.lower() == name.lower():
            return v
    return None
