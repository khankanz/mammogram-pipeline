# CLAUDE.md — DICOM 3-Class Classification Pipeline

## Project Overview

Build an active learning pipeline for classifying mammogram DICOM images into **three mutually exclusive categories**:
- **Neither**: Normal image (no special features)
- **Biopsy Tool**: Bright white marker visible
- **Mag View**: Compression plates visible

**Key Constraint**: Labels are mutually exclusive — an image is BiopsyTool OR MagView OR Neither, never multiple.

**Output Aggregation**: Final CSV reports at the **study level** (folder). If ANY image in a study folder contains BiopsyTool or MagView, the entire study is flagged.

**Performance Target**: 40,000 images classified in <5 minutes (achieved via batched ONNX inference)

## Philosophy (Opinionated Choices)

These decisions are final. Do not deviate:

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Package Manager | **uv** | Fast, modern Python package manager |
| Model | ResNet18 ImageNet pretrained, **single 3-class** | Simpler than two binary models for mutually exclusive classes |
| Framework | fastai v2 | Medical imaging support, DataBlock, lr_find |
| Database | SQLite via sqlite-utils | Single file, observable, no server |
| UI | FastHTML + HTMX | Server-rendered, session state, keyboard shortcuts |
| DICOM Processing | pydicom + joblib | Parallel preprocessing to PNG cache |
| DICOM Normalization | **DICOM-standard VOI LUT** via pydicom | Proper windowing using apply_modality_lut + apply_voi_lut |
| Multi-frame | **Process ALL frames** | Each frame becomes a separate labeled image |
| Inference | ONNX Runtime (CPU) | Portable, no GPU dependencies required |
| Image Size | 224x224 | Standard ImageNet, fast training |
| GPU | 8GB VRAM | Batch size 32 is safe |

**DO NOT**:
- Use PyTorch Lightning, Hugging Face Trainer, or other abstractions
- Add JavaScript frameworks (React, Vue, etc.)
- Use async/await patterns (FastHTML handles this)
- Over-engineer the model architecture
- Train two separate binary models (use single 3-class model)

## Data Structure

**Flat folder structure** — one level of study folders:

```
data/dicoms/
  study_001/
    img1.dcm
    img2.dcm
  study_002/
    img1.dcm
```

**Test data**: Place sample DICOMs in `/test/` folder for development.

**Linkage preserved**:
- `study_id` = folder name (e.g., "study_001")
- `filename` = "study_id/image_stem" (e.g., "study_001/img1")
- `thumbnail_path` = flattened path (e.g., "study_001_img1.png")

## Project Structure

```
dicom-search/
├── CLAUDE.md                 # This file
├── pyproject.toml            # Dependencies (uv)
├── test/                     # Sample DICOMs for development
├── data/
│   ├── dicoms/              # Raw DICOM files (input, read-only)
│   ├── thumbnails/          # Preprocessed 224x224 PNGs (cached)
│   └── labels.db            # SQLite database
├── models/
│   ├── model.pkl            # fastai learner (single 3-class model)
│   └── model.onnx           # Exported for inference
├── scripts/
│   ├── 01_preprocess.py     # DICOM → PNG thumbnail cache
│   ├── 02_label_server.py   # FastHTML labeling UI
│   ├── 03_train.py          # fastai training (3-class)
│   ├── 04_export.py         # Convert to ONNX
│   └── 05_inference.py      # Batch classify → study-level CSV
└── lib/
    ├── __init__.py
    ├── db.py                # SQLite helpers
    ├── dicom_utils.py       # DICOM loading with VOI LUT
    └── config.py            # Paths, constants
```

## Database Schema

Single table in `labels.db`:

```sql
CREATE TABLE labels (
    id INTEGER PRIMARY KEY,
    filename TEXT UNIQUE NOT NULL,      -- e.g., "study_001/image_01"
    study_id TEXT NOT NULL,             -- Folder name for aggregation
    thumbnail_path TEXT,                 -- e.g., "thumbnails/study_001_image_01.png"
    frame_number INTEGER DEFAULT 0,      -- For multi-frame DICOMs
    has_biopsy_tool INTEGER,            -- NULL=unlabeled, 0=No, 1=Yes
    has_mag_view INTEGER,               -- NULL=unlabeled, 0=No, 1=Yes
    labeled_at TIMESTAMP,
    confidence_biopsy REAL,             -- Model prediction probability
    confidence_mag REAL,
    predicted_at TIMESTAMP
);

CREATE INDEX idx_study ON labels(study_id);
CREATE INDEX idx_unlabeled ON labels(has_biopsy_tool, has_mag_view);
CREATE INDEX idx_confidence ON labels(confidence_biopsy, confidence_mag);
```

**Label encoding** (mutually exclusive):
- BiopsyTool: `has_biopsy_tool=1, has_mag_view=0`
- MagView: `has_biopsy_tool=0, has_mag_view=1`
- Neither: `has_biopsy_tool=0, has_mag_view=0`

## Script Specifications

### 01_preprocess.py

**Purpose**: Convert all DICOMs to normalized 224x224 PNGs, populate database.

**Behavior**:
1. Walk `data/dicoms/` (or `test/` with --test flag) for DICOM files
2. Process ALL frames from multi-frame DICOMs
3. For each frame:
   - Read with pydicom
   - Apply Modality LUT via `apply_modality_lut()`
   - Apply VOI LUT via `apply_voi_lut()` (DICOM-standard windowing)
   - Handle MONOCHROME1 inversion
   - Normalize to 0-255
   - Resize to 224x224 with LANCZOS
   - Save as PNG to `data/thumbnails/`
   - Insert row into database with study_id and frame_number
4. Use `joblib.Parallel(n_jobs=8)` for parallelism
5. Skip files already processed

**CLI**:
```bash
python scripts/01_preprocess.py --dicom-dir data/dicoms
python scripts/01_preprocess.py --test  # Use test/ folder
```

### 02_label_server.py

**Purpose**: FastHTML server for keyboard-driven labeling with 3 mutually exclusive choices.

**UI Requirements**:
- Dark background (#0a0a0a), centered image display
- Image sized to max 80vh, `object-fit: contain`
- Stats bar showing totals and class counts
- Current image filename and study ID displayed
- **Simplified keyboard shortcuts (3 choices)**:
  - `1` = Biopsy Tool
  - `2` = Mag View
  - `3` = Neither
  - `s` = Skip (next image without labeling)
  - `u` = Undo last label
- Single keypress labels and auto-advances (no partial state)
- Prioritize showing images with NULL labels
- Secondary: show low-confidence predictions for review
- Preload next 3 images for instant transitions

**Routes**:
```
GET  /                  → Main labeling interface
GET  /image/<id>        → Serve thumbnail PNG
POST /label             → Record label (choice=biopsy|mag|neither), return next
POST /skip              → Skip current, return next
POST /undo              → Revert last label
GET  /stats             → JSON stats for polling
```

### 03_train.py

**Purpose**: Train single 3-class ResNet18 model.

**Classes**: `["neither", "biopsy_tool", "mag_view"]`

**Behavior**:
1. Query database for fully labeled images
2. Convert labels to 3-class format:
   - `has_biopsy_tool=1` → "biopsy_tool"
   - `has_mag_view=1` → "mag_view"
   - Both 0 → "neither"
3. Split 80/20 train/valid
4. Create fastai DataBlock with augmentations
5. Use `vision_learner(dls, resnet18, metrics=[accuracy, F1Score(average='macro')])`
6. Run `learn.lr_find()`, use valley suggestion
7. `learn.fine_tune(6)`
8. Print confusion matrix and classification report
9. Save single model: `models/model.pkl`
10. Run predictions on unlabeled images, update confidence columns

**CLI**:
```bash
python scripts/03_train.py --db data/labels.db --output-dir models/
# Outputs: models/model.pkl
```

### 04_export.py

**Purpose**: Convert fastai model to ONNX for fast inference.

**Behavior**:
1. Load `model.pkl`
2. Export to ONNX with dynamic batch axis
3. Verify ONNX output matches PyTorch output
4. Print model size

**Output**: Single `model.onnx` with 3-class output `[neither, biopsy_tool, mag_view]`

**CLI**:
```bash
python scripts/04_export.py --input models/model.pkl --output models/model.onnx
```

### 05_inference.py

**Purpose**: Batch classify a directory of DICOMs, output **study-level** CSV.

**Behavior**:
1. Accept input directory containing DICOM files
2. Load ONNX model (CPU inference for portability)
3. Process all images in batches of 32
4. **Aggregate to study level**: If ANY image in a study has BiopsyTool or MagView, flag the study
5. Output CSV with columns:
   - `study_id` — folder name
   - `has_biopsy_tool` — 1 if any image in study has it, else 0
   - `has_mag_view` — 1 if any image in study has it, else 0
   - `max_prob_biopsy` — highest biopsy probability across study images
   - `max_prob_mag` — highest mag probability across study images
   - `image_count` — number of images in study
6. Print timing statistics

**CLI**:
```bash
python scripts/05_inference.py --input-dir data/dicoms --output results.csv
```

**Example Output CSV**:
```csv
study_id,has_biopsy_tool,has_mag_view,max_prob_biopsy,max_prob_mag,image_count
study_001,1,0,0.97,0.12,15
study_002,0,1,0.08,0.94,12
study_003,0,0,0.15,0.22,18
```

## DICOM Normalization (lib/dicom_utils.py)

Uses DICOM-standard LUT functions from pydicom:

```python
from pydicom.pixel_data_handlers.util import apply_voi_lut, apply_modality_lut

def normalize_array(arr: np.ndarray, ds: pydicom.Dataset) -> np.ndarray:
    arr = arr.astype(np.float64)

    # Apply Modality LUT (RescaleSlope/Intercept)
    arr = apply_modality_lut(arr, ds)

    # Apply VOI LUT (DICOM-standard windowing)
    arr = apply_voi_lut(arr, ds)

    # Handle MONOCHROME1 inversion
    if getattr(ds, 'PhotometricInterpretation', '') == 'MONOCHROME1':
        arr = arr.max() - arr

    # Normalize to 0-255
    arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255
    return arr.astype(np.uint8)
```

## Iteration Workflow

1. Run `01_preprocess.py` on new data (or `--test` for development)
2. Open `02_label_server.py`, label images (1=Biopsy, 2=MagView, 3=Neither)
3. Run `03_train.py` to train 3-class model
4. Model auto-predicts on unlabeled data (confidence scores)
5. Return to step 2, labeling UI shows uncertain predictions first
6. Repeat until validation accuracy >95%
7. Run `04_export.py` then `05_inference.py` on full dataset

## Final Deliverable

```bash
# Install dependencies
uv sync

# 1. Preprocess DICOMs
uv run python scripts/01_preprocess.py --dicom-dir /path/to/studies

# 2. Label images (keyboard-driven: 1=Biopsy, 2=MagView, 3=Neither)
uv run python scripts/02_label_server.py
# Open http://localhost:5001

# 3. Train 3-class model
uv run python scripts/03_train.py

# 4. Export for production
uv run python scripts/04_export.py

# 5. Classify all images → study-level CSV
uv run python scripts/05_inference.py --input-dir /path/to/studies --output results.csv
```
