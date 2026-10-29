"""Train the MediGuard AI Fake/Real classifier (MobileNetV2 transfer learning).

Usage:
    python train_model.py                       # defaults
    python train_model.py --epochs 15 --fine-tune-epochs 5
    python train_model.py --data-dir path/to/train --weights imagenet

Expected layout (folder names become class names; order is read, not assumed):
    data/train/Fake/*.jpg
    data/train/Real/*.jpg
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

import config
from utils.data_loader import count_near_duplicates, scan_dataset

EXIT_NO_DATA = 2


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train the MediGuard AI MobileNetV2 classifier.")
    p.add_argument("--data-dir", default=str(config.DATASET_DIR))
    p.add_argument("--epochs", type=int, default=15, help="Epochs for the frozen-base phase")
    p.add_argument("--fine-tune-epochs", type=int, default=5, help="Extra epochs unfreezing the top of MobileNetV2 (0 to skip)")
    p.add_argument("--fine-tune-layers", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-split", type=float, default=0.15)
    p.add_argument("--test-split", type=float, default=0.15)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--weights", default="imagenet", choices=["imagenet", "none"],
                   help="'none' trains from scratch (only for offline smoke tests)")
    p.add_argument("--output-dir", default=str(config.MODELS_DIR))
    return p.parse_args(argv)


def fail(msg: str) -> None:
    print(f"\n[MediGuard] Training cannot start:\n  {msg}\n", file=sys.stderr)
    print("  Put labelled images in data/train/Fake/ and data/train/Real/ (JPG, PNG or WEBP),", file=sys.stderr)
    print("  for example from the Kaggle 'Fake vs Real Medicine Dataset (images)'.\n", file=sys.stderr)
    sys.exit(EXIT_NO_DATA)


def set_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def main(argv=None) -> int:
    args = parse_args(argv)
    data_dir = Path(args.data_dir)
    out = Path(args.output_dir)

    # ---------------- 1. Load original dataset splits
    import hashlib
    from collections import defaultdict

    root = Path(args.data_dir).expanduser()
    if (root / "train").is_dir() and (root / "val").is_dir() and (root / "test").is_dir():
        split_dirs = {
            "train": root / "train",
            "validation": root / "val",
            "test": root / "test",
        }
    else:
        fail(
            "Expected dataset root containing train/, val/, test/. "
            f"Received: {root}"
        )

    summaries = {
        name: scan_dataset(folder)
        for name, folder in split_dirs.items()
    }

    classes = ["Fake", "Real"]

    for name, summary in summaries.items():
        if set(summary.classes) != set(classes):
            fail(
                f"{name}: expected class folders Fake and Real; "
                f"found {summary.classes}"
            )

        for cls in classes:
            if summary.counts.get(cls, 0) < 1:
                fail(f"{name}/{cls} has no valid images")

        if summary.invalid:
            fail(
                f"{name} contains {len(summary.invalid)} invalid images: "
                + ", ".join(summary.invalid[:5])
            )

        print(f"[MediGuard] {name}: {summary.counts}")

    def get_arrays(summary):
        paths, labels = [], []
        for idx, cls in enumerate(classes):
            files = summary.files[cls]
            paths.extend(str(p) for p in files)
            labels.extend([idx] * len(files))
        return np.array(paths), np.array(labels, dtype=np.int32)

    tr_p, tr_y = get_arrays(summaries["train"])
    va_p, va_y = get_arrays(summaries["validation"])
    te_p, te_y = get_arrays(summaries["test"])

    print(
        f"[MediGuard] Original splits: "
        f"train={len(tr_p)}, validation={len(va_p)}, test={len(te_p)}"
    )

    # Check exact duplicates between all three splits.
    hashes = defaultdict(set)

    for name, paths in (
        ("train", tr_p),
        ("validation", va_p),
        ("test", te_p),
    ):
        for path in paths:
            digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            hashes[digest].add(name)

    duplicate_groups = sum(
        len(split_names) > 1 for split_names in hashes.values()
    )

    print(
        f"[MediGuard] Cross-split exact duplicate groups: "
        f"{duplicate_groups}"
    )

    if duplicate_groups:
        fail(
            f"Dataset leakage detected: {duplicate_groups} image groups "
            "appear in multiple splits. Resolve before training."
        )

    set_seeds(args.seed)

    dup_val = count_near_duplicates(
        [Path(p) for p in tr_p],
        [Path(p) for p in va_p]
    )
    dup_test = count_near_duplicates(
        [Path(p) for p in tr_p],
        [Path(p) for p in te_p]
    )

    if dup_val or dup_test:
        print(
            f"[MediGuard] Near-duplicate warning: "
            f"validation={dup_val}, test={dup_test}"
        )

    summary = summaries["train"]
    data_dir = root

    # ---------------- 2. TensorFlow
    try:
        import tensorflow as tf
    except ImportError:
        print("[MediGuard] TensorFlow is not installed. Run: python -m pip install -r requirements.txt", file=sys.stderr)
        return 3
    tf.keras.utils.set_random_seed(args.seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    layers = tf.keras.layers
    H, W = config.IMG_SIZE
    AUTOTUNE = tf.data.AUTOTUNE

    def load(path, label):
        raw = tf.io.read_file(path)
        img = tf.io.decode_image(raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, (H, W), method="bilinear")
        img = tf.cast(img, tf.float32)  # 0-255; scaling to [-1,1] is inside the model
        return img, label

    augment = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.06),
        layers.RandomZoom(0.12),
        layers.RandomContrast(0.15),
        layers.RandomTranslation(0.06, 0.06),
    ], name="train_augmentation")

    def make_ds(p, y, training):
        ds = tf.data.Dataset.from_tensor_slices((p, y))
        if training:
            ds = ds.shuffle(len(p), seed=args.seed, reshuffle_each_iteration=True)
        ds = ds.map(load, num_parallel_calls=AUTOTUNE).batch(args.batch_size)
        if training:  # augmentation for the training set ONLY
            ds = ds.map(lambda x, l: (augment(x, training=True), l), num_parallel_calls=AUTOTUNE)
        return ds.prefetch(AUTOTUNE)

    train_ds, val_ds, test_ds = make_ds(tr_p, tr_y, True), make_ds(va_p, va_y, False), make_ds(te_p, te_y, False)

    weights = None if args.weights == "none" else "imagenet"
    try:
        base = tf.keras.applications.MobileNetV2(input_shape=(H, W, 3), include_top=False, weights=weights)
    except Exception as exc:
        print(f"[MediGuard] Could not load MobileNetV2 ImageNet weights ({exc}). Check your internet connection "
              "(weights download once, ~14 MB).", file=sys.stderr)
        return 4
    base.trainable = False

    inputs = tf.keras.Input(shape=(H, W, 3), name="image")
    x = layers.Rescaling(1 / 127.5, offset=-1, name="mobilenet_v2_scaling")(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(len(classes), activation="softmax", name="class_scores")(x)
    model = tf.keras.Model(inputs, outputs, name="mediguard_mobilenetv2")

    counts = np.bincount(tr_y, minlength=len(classes))
    class_weight = {i: float(len(tr_y) / (len(classes) * c)) for i, c in enumerate(counts) if c}
    imbalance = max(counts) / max(1, min(counts))
    use_cw = bool(imbalance > 1.2)
    print(f"[MediGuard] Training class counts {counts.tolist()} → class weights {'ON' if use_cw else 'OFF'} {class_weight if use_cw else ''}")

    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / "best_checkpoint.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(str(ckpt_path), monitor="val_loss", save_best_only=True),
    ]
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks,
                   class_weight=class_weight if use_cw else None, verbose=2)
    history = {k: [float(v) for v in vals] for k, vals in h1.history.items()}
    ft_start = None

    if args.fine_tune_epochs > 0:
        ft_start = len(history["loss"])
        base.trainable = True
        for layer in base.layers[:-args.fine_tune_layers]:
            layer.trainable = False
        for layer in base.layers:  # keep BatchNorm statistics frozen
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
        model.compile(optimizer=tf.keras.optimizers.Adam(args.lr / 10), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        h2 = model.fit(train_ds, validation_data=val_ds, epochs=ft_start + args.fine_tune_epochs, initial_epoch=ft_start,
                       callbacks=callbacks, class_weight=class_weight if use_cw else None, verbose=2)
        for k, vals in h2.history.items():
            history.setdefault(k, []).extend(float(v) for v in vals)
    history["fine_tune_start_epoch"] = ft_start

    # ---------------- 3. Evaluate on the untouched test split
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
    probs = model.predict(test_ds, verbose=0)
    pred = probs.argmax(axis=1)
    avg = "binary" if len(classes) == 2 else "macro"
    pos = classes.index("Fake") if "Fake" in classes else (1 if len(classes) == 2 else None)
    kw = {"average": avg, "zero_division": 0}
    if avg == "binary":
        kw["pos_label"] = pos
    prec, rec, f1, _ = precision_recall_fscore_support(te_y, pred, **kw)
    test_metrics = {"accuracy": float(accuracy_score(te_y, pred)), "precision": float(prec), "recall": float(rec), "f1": float(f1)}
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    report_txt = classification_report(te_y, pred, target_names=classes, zero_division=0)
    report_dict = classification_report(te_y, pred, target_names=classes, zero_division=0, output_dict=True)
    cm = confusion_matrix(te_y, pred, labels=list(range(len(classes)))).tolist()
    print("\n[MediGuard] Independent test-set results\n" + report_txt)

    version = datetime.now().strftime("%Y%m%d-%H%M")
    model.save(config.MODEL_PATH if out == config.MODELS_DIR else out / config.MODEL_PATH.name)
    if ckpt_path.exists():
        ckpt_path.unlink()  # final model already holds the best weights
    write = lambda name, obj: (out / name).write_text(json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    write(config.CLASS_MAP_PATH.name, {str(i): c for i, c in enumerate(classes)})
    write(config.HISTORY_PATH.name, history)
    write(config.METRICS_PATH.name, {
        "evaluated_on": "independent held-out test split (never used for training or early stopping)",
        "positive_class_for_binary_metrics": classes[pos] if pos is not None else None,
        "test": test_metrics, "validation": {"accuracy": float(val_acc), "loss": float(val_loss)},
        "labels": classes, "confusion_matrix": cm, "classification_report": report_dict,
        "split_sizes": {"train": len(tr_p), "validation": len(va_p), "test": len(te_p)},
        "near_duplicates": {"validation": dup_val, "test": dup_test},
    })
    write(config.TRAIN_CONFIG_PATH.name, {
        "model_version": version, "trained_at": datetime.now().isoformat(timespec="seconds"),
        "architecture": "MobileNetV2 + GAP + Dropout(0.3) + Dense(softmax)", "base_weights": weights or "random",
        "image_size": [H, W], "classes": classes, "class_counts": summary.counts, "invalid_images": len(summary.invalid),
        "epochs_requested": args.epochs, "fine_tune_epochs": args.fine_tune_epochs, "fine_tune_layers": args.fine_tune_layers,
        "epochs_run": len(history["loss"]), "batch_size": args.batch_size, "learning_rate": args.lr, "seed": args.seed,
        "val_split": args.val_split, "test_split": args.test_split, "class_weights_used": use_cw,
        "augmentation": ["RandomFlip(horizontal)", "RandomRotation(0.06)", "RandomZoom(0.12)", "RandomContrast(0.15)", "RandomTranslation(0.06)"],
        "tensorflow_version": tf.__version__, "data_dir": str(data_dir),
    })
    (out / config.CLASS_REPORT_PATH.name).write_text(report_txt, encoding="utf-8")
    print(f"[MediGuard] Saved model and metrics to {out}/ (version {version}). Start the app: python -m streamlit run app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
