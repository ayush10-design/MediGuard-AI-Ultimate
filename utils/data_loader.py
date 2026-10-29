"""Dataset discovery and validation. Class names are read from folder names,
never assumed from a fixed numeric order."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

import config


@dataclass
class DatasetSummary:
    root: Path
    exists: bool
    classes: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    invalid: list[str] = field(default_factory=list)
    files: dict[str, list[Path]] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def usable(self) -> bool:
        return len(self.classes) >= 2 and all(self.counts.get(c, 0) >= 2 for c in self.classes)


def find_class_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted([d for d in root.iterdir() if d.is_dir() and not d.name.startswith((".", "_"))],
                  key=lambda d: d.name.lower())


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*")
                  if p.is_file() and p.suffix.lower() in config.ALLOWED_IMAGE_EXTENSIONS)


def scan_dataset(root: Path | None = None, verify: bool = True) -> DatasetSummary:
    root = Path(root or config.DATASET_DIR)
    summary = DatasetSummary(root=root, exists=root.exists())
    for d in find_class_dirs(root):
        good: list[Path] = []
        for p in list_images(d):
            if verify:
                try:
                    with Image.open(p) as im:
                        im.verify()
                except Exception:
                    summary.invalid.append(str(p.relative_to(root)))
                    continue
            good.append(p)
        if good or d.name.lower() in ("fake", "real"):
            summary.classes.append(d.name)
            summary.counts[d.name] = len(good)
            summary.files[d.name] = good
    return summary


def average_hash(path: Path, size: int = 8) -> int:
    """Tiny perceptual hash used to flag near-duplicate images across splits."""
    with Image.open(path) as im:
        g = im.convert("L").resize((size, size), Image.BILINEAR)
        px = list(g.getdata())
    avg = sum(px) / len(px)
    bits = 0
    for v in px:
        bits = (bits << 1) | (1 if v >= avg else 0)
    return bits


def count_near_duplicates(train: list[Path], other: list[Path], max_distance: int = 3) -> int:
    train_hashes = []
    for p in train:
        try:
            train_hashes.append(average_hash(p))
        except Exception:
            pass
    hits = 0
    for p in other:
        try:
            h = average_hash(p)
        except Exception:
            continue
        if any(bin(h ^ t).count("1") <= max_distance for t in train_hashes):
            hits += 1
    return hits
