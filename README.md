# MediGuard AI — Intelligent Medicine Authentication & Analysis System
*Smarter Medicine Screening. Safer Tomorrow.*

A college-project prototype that reads medicine packaging (Google Gemini or offline Tesseract), classifies
packaging photos as **Fake** or **Real** with a MobileNetV2 transfer-learning model you train yourself, accepts photos
from your phone over local Wi-Fi via a QR code, and produces detailed PDF / CSV / HTML reports.

> **Important.** The Fake/Real output is a *visual classification* learned from labelled photos. It is not laboratory
> testing, regulatory verification, or proof of authenticity or chemical safety.

---

## 1. Open the project on your MacBook

1. Double-click `MediGuard_AI_Ultimate.zip` in Finder — it extracts to a `MediGuard_AI_Ultimate` folder.
2. Open **VS Code › File › Open Folder…** and choose that folder.
3. Open the terminal in VS Code: **Terminal › New Terminal**.

## 2. Install Python 3.11 (once)

```bash
brew install python@3.11          # needs Homebrew: https://brew.sh
brew install tesseract            # optional offline OCR fallback
```

## 3. Create the virtual environment and install packages

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

In VS Code press **⌘⇧P › Python: Select Interpreter** and choose `.venv`.
Optional GPU acceleration on Apple Silicon: `python -m pip install tensorflow-metal==1.1.0`.

Check everything with:

```bash
python scripts/check_setup.py
```

## 4. Configure Gemini (optional, recommended)

```bash
cp .env.example .env
open -e .env        # paste your key after GEMINI_API_KEY=
```

Get a free key at <https://aistudio.google.com/apikey>. The key is read from `.env` and only ever shown masked.
Before any image is sent to Google, the Detection Lab asks for consent. Without a key, Tesseract is used.

## 5. Add the dataset

Download the **Fake vs Real Medicine Dataset (images)** used by the Kaggle notebook
<https://www.kaggle.com/code/surajkumarjha1/fake-medicine-detector> and arrange it like this:

```text
data/train/
├── Fake/   ← images of counterfeit packaging
└── Real/   ← images of genuine packaging
```

Folder names become class names (indices are assigned alphabetically and saved in `models/class_mapping.json`).
If the Kaggle download has extra nesting (e.g. `dataset/train/fake`), move the two class folders up, or train with
`python train_model.py --data-dir path/to/folder-containing-class-folders`.
JPG, JPEG, PNG and WEBP are accepted; unreadable files are skipped and listed in the Dataset Explorer.

## 6. Train the model

```bash
python train_model.py
```

What happens: stratified 70 / 15 / 15 train / validation / test split (seed 42) → augmentation on training images
only → MobileNetV2 (ImageNet weights, frozen) + new head → class weights if imbalanced → early stopping and
checkpointing → optional fine-tuning of the top 30 layers → evaluation on the untouched test split.
The first run downloads the ImageNet weights (~14 MB). Useful options:
`--epochs 20 --fine-tune-epochs 5 --batch-size 16`.

Saved in `models/`: `mediguard_mobilenetv2.keras`, `class_mapping.json`, `training_history.json`,
`metrics.json` (accuracy, precision, recall, F1, confusion matrix, per-class report), `training_config.json`,
`classification_report.txt`.

**Data leakage.** Product datasets often contain several photos of the same pack from the same session. If those land
in both training and test sets, the test score will look better than real-world performance. The trainer runs a
perceptual-hash check and prints a warning (also shown on the Model Performance page). For a fairer test, keep all
photos of one physical product in the same split.

## 7. Run the app

```bash
python -m streamlit run app.py
```

It opens at <http://localhost:8501>. The app works without a model (shows **Model Not Trained**), without Gemini
and without a dataset — it never substitutes simulated results.

## 8. Phone camera (Live Camera Scan)

1. Phone and MacBook on the **same Wi-Fi** (not a guest network that isolates devices).
2. In the app open **Live Camera Scan › Connect Phone**. This starts the local upload server on port 8765.
   macOS may ask whether Python may accept incoming connections — click **Allow**.
3. Scan the QR code with the phone's camera app, take a photo, tap **Send to MediGuard AI**.
4. The photo appears on the laptop within ~3 seconds; click **Analyse** to open it in the Detection Lab.

You can also start the server manually: `python camera/server.py`.
Security: random token per session, expires after 15 minutes, max 25 uploads, JPG/PNG/WEBP only, 10 MB limit, local
network only. The phone uses a standard photo picker (`capture="environment"`), which works over local HTTP; live
in-browser video would require HTTPS and is deliberately not used.

## 9. Tests

```bash
python -m pytest -q
python scripts/smoke_test_training.py   # optional: proves training runs (synthetic data, temp folder, meaningless scores)
```

## 10. Troubleshooting

| Problem | Fix |
|---|---|
| `python3.11: command not found` | `brew install python@3.11`, then reopen the terminal |
| TensorFlow install fails | Make sure the venv uses Python 3.11 (`python --version`); TensorFlow 2.16 does not support 3.13 |
| `Model Not Trained` | Add images to `data/train/Fake` and `data/train/Real`, run `python train_model.py` |
| Training: "Dataset folder not found" / "at least two" | Check the folder layout in section 5 |
| Could not load ImageNet weights | Needs internet once; check the connection or proxy |
| Tesseract not found | `brew install tesseract`, restart the app |
| Gemini quota / 429 | Wait a minute, or choose Tesseract in Settings |
| Gemini model not found | Change the model name in Settings (e.g. `gemini-2.5-flash`) |
| Phone cannot open the link | Same Wi-Fi? Firewall allowed? Try `http://<laptop-ip>:8765/health` on the phone |
| Port 8765 in use | Set `MEDIGUARD_CAMERA_PORT=8766` in `.env` and restart |
| Syringe cursor distracting | Settings › Interface › turn off "Syringe cursor" |

## Project structure

```text
app.py                  navigation & theme          train_model.py   MobileNetV2 training
config.py               paths, settings, secrets    pages/           10 dashboard pages
utils/ui.py             design system, cursor, scan animation
utils/ocr_service.py    Tesseract + text parsing    utils/gemini_service.py  Gemini SDK integration
utils/model_utils.py    model status & inference    utils/report_generator.py PDF / HTML / CSV
utils/history_db.py     SQLite history              utils/autocomplete.py  catalogue search
camera/server.py        Flask phone-upload server   camera/templates/capture.html  mobile page
data/catalogue.csv      reference list (typing aid only — not a register of genuine products)
tests/                  pytest suite                scripts/         setup check, training smoke test
```
