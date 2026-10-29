"""Image loading, validation, rotation, cropping and OCR pre-processing."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

import config

try:
    import cv2  # type: ignore
except ImportError:  # OpenCV is optional; Pillow fallbacks are used
    cv2 = None

MIME_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}


class ImageValidationError(ValueError):
    pass


def is_allowed_extension(filename: str) -> bool:
    return Path(filename).suffix.lower() in config.ALLOWED_IMAGE_EXTENSIONS


def load_image(data: bytes, filename: str = "image") -> Image.Image:
    """Decode bytes into an RGB image, honouring EXIF orientation from phones."""
    if filename and not is_allowed_extension(filename):
        raise ImageValidationError("Unsupported file type. Use JPG, JPEG, PNG or WEBP.")
    try:
        probe = Image.open(io.BytesIO(data))
        probe.verify()
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ImageValidationError(f"The file could not be read as an image ({exc}).") from exc
    return img.convert("RGB")


def rotate(img: Image.Image, degrees: int) -> Image.Image:
    degrees = int(degrees) % 360
    return img if degrees == 0 else img.rotate(-degrees, expand=True)


def crop_percent(img: Image.Image, left: float, top: float, right: float, bottom: float) -> Image.Image:
    """Crop using percentages trimmed from each edge (0-45 each)."""
    w, h = img.size
    l, t = int(w * left / 100), int(h * top / 100)
    r, b = w - int(w * right / 100), h - int(h * bottom / 100)
    if r - l < 16 or b - t < 16:
        raise ImageValidationError("Crop is too tight; less than 16 px would remain.")
    return img.crop((l, t, r, b))


def to_bytes(img: Image.Image, fmt: str = "JPEG", quality: int = 92) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality) if fmt == "JPEG" else img.save(buf, format=fmt)
    return buf.getvalue()


def prepare_for_model(img: Image.Image, size=config.IMG_SIZE) -> np.ndarray:
    """Resize to the model input size and return a float32 batch (1, H, W, 3) in 0-255.

    Scaling to MobileNetV2's [-1, 1] range happens inside the saved model, so
    training and inference always share identical preprocessing.
    """
    resized = img.convert("RGB").resize(size, Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32)
    return np.expand_dims(arr, 0)


def enhance_for_ocr(img: Image.Image) -> Image.Image:
    """Grayscale, upscale small images and apply adaptive threshold for Tesseract."""
    gray = ImageOps.grayscale(img)
    if max(gray.size) < 1400:
        scale = 1400 / max(gray.size)
        gray = gray.resize((int(gray.width * scale), int(gray.height * scale)), Image.LANCZOS)
    if cv2 is None:
        return ImageOps.autocontrast(gray)
    arr = np.asarray(gray)
    arr = cv2.bilateralFilter(arr, 7, 50, 50)
    arr = cv2.adaptiveThreshold(arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11)
    return Image.fromarray(arr)


def image_quality_notes(img: Image.Image) -> list[str]:
    """Simple, measurable observations about the photo (not about the medicine)."""
    notes = []
    w, h = img.size
    if min(w, h) < 300:
        notes.append(f"Low resolution ({w}×{h}px); small packaging text may be unreadable.")
    gray = np.asarray(ImageOps.grayscale(img), dtype=np.float32)
    mean = float(gray.mean())
    if mean < 50:
        notes.append("Image is very dark; retake in better light.")
    elif mean > 225:
        notes.append("Image is very bright or washed out; glare may hide text.")
    if cv2 is not None:
        sharp = float(cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F).var())
        if sharp < 60:
            notes.append(f"Image appears blurry (sharpness score {sharp:.0f}); hold the camera steady.")
    return notes
