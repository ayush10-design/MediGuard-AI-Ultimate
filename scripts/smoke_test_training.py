"""PIPELINE smoke test only — NOT a real model.

Creates throw-away synthetic images in a temporary folder and runs train_model.py for 1 epoch
with random weights, writing into a temporary output folder (never into models/). It proves the
training code runs end-to-end on this machine. The resulting numbers are meaningless.

Run:  python scripts/smoke_test_training.py
"""
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent

with tempfile.TemporaryDirectory() as tmp:
    data, out = Path(tmp) / "train", Path(tmp) / "models"
    for cls, colour in (("Fake", (200, 60, 60)), ("Real", (60, 140, 220))):
        (data / cls).mkdir(parents=True)
        for i in range(14):
            im = Image.new("RGB", (160, 120), colour)
            ImageDraw.Draw(im).rectangle([10 + i * 3, 20, 90 + i * 2, 80], fill=(255, 255, 255))
            im.save(data / cls / f"{cls}_{i}.png")
    cmd = [sys.executable, str(ROOT / "train_model.py"), "--data-dir", str(data), "--output-dir", str(out),
           "--epochs", "1", "--fine-tune-epochs", "1", "--weights", "none", "--batch-size", "4"]
    print("Running:", " ".join(cmd))
    rc = subprocess.call(cmd, cwd=ROOT)
    files = sorted(p.name for p in out.glob("*")) if out.exists() else []
    print("Output files:", files)
    sys.exit(rc)
