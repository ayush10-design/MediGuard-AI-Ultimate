import subprocess
import sys
from pathlib import Path

from PIL import Image

import config
from utils import model_utils
from utils.data_loader import scan_dataset

ROOT = Path(__file__).resolve().parent.parent


def test_missing_model_reports_not_trained(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "none.keras")
    monkeypatch.setattr(config, "CLASS_MAP_PATH", tmp_path / "none.json")
    st = model_utils.model_status()
    assert st["trained"] is False and "Model Not Trained" in st["message"]
    res = model_utils.predict(Image.new("RGB", (100, 100)))
    assert res["ok"] is False and "scores" not in res


def test_dataset_scan_reads_class_names(tmp_path):
    for cls in ("Real", "Fake"):
        (tmp_path / cls).mkdir()
        for i in range(3):
            Image.new("RGB", (30, 30)).save(tmp_path / cls / f"{i}.jpg")
    (tmp_path / "Fake" / "broken.jpg").write_bytes(b"nope")
    s = scan_dataset(tmp_path)
    assert s.classes == ["Fake", "Real"]
    assert s.counts == {"Fake": 3, "Real": 3}
    assert len(s.invalid) == 1


def test_training_fails_clearly_without_dataset(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "train_model.py"), "--data-dir", str(tmp_path / "missing")],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 2
    assert "Dataset folder not found" in r.stderr


def test_training_fails_with_single_class(tmp_path):
    (tmp_path / "Real").mkdir()
    Image.new("RGB", (30, 30)).save(tmp_path / "Real" / "a.jpg")
    r = subprocess.run([sys.executable, str(ROOT / "train_model.py"), "--data-dir", str(tmp_path)],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 2 and "at least two" in r.stderr
