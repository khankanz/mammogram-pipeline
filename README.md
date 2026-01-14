# DICOM Classification Pipeline

**Thursday**: Pitched an idea. Got skeptical looks.
**Friday**: Said "fuck it" and built it anyway.
**Tuesday**: Team of 3 assigned to productionize it.

---

## What is this?

A mammogram classification pipeline that identifies:
- **Biopsy markers** (bright white metallic tools)
- **Magnification views** (horizontal compression plate artifacts)
- **Neither** (normal screening images)

Built because DICOM metadata *should* contain this info but often doesn't. Hospital SOPs vary. Manufacturer implementations differ. The UK's OPTIMUM dataset spans their entire hospital network. Inconsistency is the only consistency.

So I trained a model to just *look* at the images. Wild concept, I know.

## The Numbers

| Metric | Value |
|--------|-------|
| Accuracy | 92% after ~500 labels |
| Inference | 250 images/sec (CPU) |
| Time to classify 40K images | ~2.5 minutes |
| Time to build v1 | 1 afternoon |

## The Stack

No frameworks. No cloud. Just Python.

- **pydicom** — DICOM loading with proper VOI LUT normalization
- **fastai** — ResNet18, active learning loop
- **SQLite** — Labels database (sqlite-utils)
- **FastHTML** — Keyboard-driven labeling UI
- **ONNX Runtime** — Fast CPU inference

## Quick Start

```bash
# Install dependencies
uv sync

# 1. Preprocess DICOMs → thumbnails
uv run python scripts/01_preprocess.py --dicom-dir /path/to/dicoms

# 2. Label images (keyboard-driven: 1=Biopsy, 2=Mag, 3=Neither)
uv run python scripts/02_label_server.py
# Open http://localhost:5001

# 3. Train model
uv run python scripts/03_train.py

# 4. Export to ONNX
uv run python scripts/04_export.py

# 5. Batch inference → study-level CSV
uv run python scripts/05_inference.py --input-dir /path/to/dicoms --output results.csv
```

## The Labeling UI

Single keypress labels AND advances. No mouse clicking. No multi-step forms.

- `1` → Biopsy Tool
- `2` → Mag View
- `3` → Neither
- `s` → Skip
- `u` → Undo

Dark background because I'm staring at medical images. Preloads next 3 images for instant transitions. Labeled ~500 images in under an hour.

## Active Learning Loop

You don't need to label thousands of images upfront.

1. Label a small seed set (~50)
2. Train
3. Model shows you what it's uncertain about
4. Label those
5. Repeat

50 → 100 → 200 → 350 → 500 labels. 67% → 92% accuracy. That's it.

## Output

Study-level CSV. If ANY image in a study has a biopsy marker or mag view, the study gets flagged.

```csv
study_id,has_biopsy_tool,has_mag_view,max_prob_biopsy,max_prob_mag,image_count
study_001,1,0,0.97,0.12,15
study_002,0,1,0.08,0.94,12
study_003,0,0,0.15,0.22,18
```

Conservative by design. We'd rather review false positives than miss real ones.

## Project Structure

```
mammogram-pipeline/
├── scripts/
│   ├── 01_preprocess.py      # DICOM → PNG thumbnails
│   ├── 02_label_server.py    # FastHTML labeling UI
│   ├── 03_train.py           # fastai training
│   ├── 04_export.py          # ONNX export
│   ├── 05_inference.py       # Batch classification
│   ├── 06_gradcam.py         # Model interpretability
│   └── 07_review.py          # Review predictions
├── lib/
│   ├── config.py             # Paths, constants
│   ├── db.py                 # SQLite helpers
│   └── dicom_utils.py        # DICOM normalization
└── content/                  # Blog/tweet storm drafts
```

## Why This Exists

Most ideas die in the "sounds complicated" phase.

Skip that phase. Build the shitty v1. Let reality tell you if you're wrong.

The gap between "interesting idea" and "working prototype" is almost always smaller than it looks. The tools exist. The tutorials exist. The compute is cheap.

What's rare is someone saying "fuck it, I'll find out if this works by 5pm."

---

*Built in a day. Productionized by a team. You can just do shit.*
