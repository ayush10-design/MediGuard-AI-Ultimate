"""Diagnose the local setup: Python version, packages, Tesseract, Gemini key, dataset and model.

Run:  python scripts/check_setup.py
"""
import importlib.util
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from utils.data_loader import scan_dataset  # noqa: E402
from utils.model_utils import model_status  # noqa: E402

OK, WARN, BAD = "✅", "⚠️ ", "❌"
print(f"{OK if sys.version_info[:2] == (3, 11) else WARN} Python {sys.version.split()[0]} (3.11 recommended)")
for mod, pkg in [("streamlit", "streamlit"), ("tensorflow", "tensorflow"), ("google.genai", "google-genai"),
                 ("cv2", "opencv-python-headless"), ("PIL", "pillow"), ("plotly", "plotly"), ("flask", "flask"),
                 ("qrcode", "qrcode"), ("reportlab", "reportlab"), ("pytesseract", "pytesseract"), ("sklearn", "scikit-learn")]:
    try:
        found = importlib.util.find_spec(mod) is not None
    except ModuleNotFoundError:
        found = False
    print(f"{OK if found else BAD} {pkg}")
print(f"{OK if shutil.which('tesseract') else WARN} Tesseract binary {'found' if shutil.which('tesseract') else 'missing (brew install tesseract)'}")
print(f"{OK if config.get_gemini_api_key() else WARN} Gemini API key: {config.mask_secret(config.get_gemini_api_key())}")
s = scan_dataset(verify=False)
print(f"{OK if s.usable else WARN} Dataset {config.DATASET_DIR}: {s.counts or 'no images'}")
m = model_status()
print(f"{OK if m['trained'] else WARN} Model: {m['message']}")
