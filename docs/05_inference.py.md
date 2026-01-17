# Production Inference: DICOMs to Study-Level Classifications

This script is the production endpoint. Development is over—you've trained your model, exported to ONNX, now you need to classify thousands of DICOMs you've never seen before.

Key distinction: **this does NOT touch your development database**. It creates a separate results database. Training workflow and production workflow stay completely isolated.

## Design Principles

```python
# Design principles:
# 1. Standalone - no dependency on development database or training workflow
# 2. Resumable - SQLite tracks progress, can restart from interruption  
# 3. Fast - parallel DICOM loading, batched ONNX inference
# 4. Study-level output - radiologists think in studies, not images
```

The output is dual:
- **PRIMARY**: SQLite database (`results.db`) — image-level predictions, resumable
- **DERIVED**: CSV file (`results.csv`) — study-level aggregation for humans/Excel

---

## ONNX Session: Provider Selection

```python
def load_session(model_path):
    "Load ONNX model with best available provider (CUDA > CPU)"
    import onnxruntime as ort
    
    # Try CUDA first, fall back to CPU
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    available = ort.get_available_providers()
    providers = [p for p in providers if p in available]
    
    print(f"ONNX providers: {providers}")
    return ort.InferenceSession(str(model_path), providers=providers)
```

List your preferences, filter to what's available. If CUDA is installed and working, you get GPU inference. Otherwise, CPU. No manual configuration needed.

---

## Image Preprocessing: Speaking the Model's Language

```python
def preprocess(img):
    "PIL Image → normalized numpy array ready for ONNX. Shape: (3, 224, 224)"
    # Step 1: Convert to float 0-1 range
    arr = np.array(img).astype(np.float32) / 255.0
    
    # Step 2: ImageNet normalization
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    
    # Step 3: HWC → CHW
    return arr.transpose(2, 0, 1)
```

Three transformations:

**Step 1: Float conversion (0-255 → 0-1)**
Neural networks train better with small, centered values. Raw pixels 0-255 would blow up gradients in early layers.

**Step 2: ImageNet normalization**
The pretrained model learned features on ImageNet, which has specific per-channel statistics. We normalize our data to match that distribution. Without this, the model sees "out of distribution" inputs and performs worse. It's like speaking the model's native language vs broken phrases.

**Step 3: HWC → CHW**
PIL/numpy store images as (Height, Width, Channels)—rows of RGB pixels. PyTorch/ONNX expect (Channels, Height, Width)—separate R, G, B planes. Same data, different memory layout.

---

## Sigmoid: Logits to Probabilities

```python
def sigmoid(x):
    "Logits → probabilities. Numerically stable version."
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
```

Sigmoid squashes any real number to (0, 1). `sigmoid(20) ≈ 1.0`, `sigmoid(-20) ≈ 0.0`.

The clip prevents overflow: `exp(710)` overflows float64. In practice, `sigmoid(|x| > 20)` is already 0 or 1 for all purposes. The clip is belt-and-suspenders defensive programming.

---

## Batch Inference

```python
def predict_batch(session, images):
    "Run ONNX inference on a batch of preprocessed images. Returns (N, 2) probabilities."
    if not images: return np.array([])
    
    # Stack list of (3, 224, 224) → single (N, 3, 224, 224) batch
    batch = np.stack(images).astype(np.float32)
    
    # session.run(output_names, input_dict)
    # None = return all outputs
    # 'image' = the input name we defined during ONNX export
    logits = session.run(None, {'image': batch})[0]
    
    return sigmoid(logits)
```

Batching is key for throughput. Processing 32 images in one GPU kernel is much faster than 32 separate kernel launches.

---

## Why SQLite for Results?

```python
# Why SQLite instead of JSONL for 500K images?
# 1. Resume: WHERE predicted_at IS NULL instantly finds unprocessed images
# 2. Query: "Show all biopsy=YES" is one SQL statement vs parsing entire file
# 3. Concurrent: WAL mode handles writes safely; JSONL risks corruption
# 4. Speed: ~295K inserts/sec batched; comparable to JSONL append
```

Simon Willison's advice: "If you need local structured data storage, SQLite should be your default."

The schema mirrors the development database pattern—schema + indexes defined together:

```python
_results_schema = dict(
    id=int, filename=str, study_id=str, frame_number=int,
    prob_biopsy=float, prob_mag=float, predicted_at=str
)

_results_indexes = [
    (["filename"], {"unique": True}),   # Fast lookup for resume, prevent duplicates
    (["study_id"], {}),                  # Fast aggregation to study level
    (["prob_biopsy"], {}),               # Query "show me high-confidence biopsy"
    (["prob_mag"], {}),                  # Query "show me high-confidence mag"
    (["predicted_at"], {}),              # Query by time range
]
```

---

## Resume Capability

```python
def is_processed(db, filename):
    "Check if we already predicted this file (for resume capability)"
    return db.execute(
        "SELECT 1 FROM predictions WHERE filename = ? LIMIT 1", [filename]
    ).fetchone() is not None
```

In the main loop:

```python
if resume:
    before = len(dicoms)
    processed = {r["filename"] for r in db["predictions"].rows}
    dicoms = L(dicoms).filter(lambda x: f"{x[1]}/{x[0].stem}" not in processed)
    print(f"  Resuming: {before - len(dicoms)} already processed, {len(dicoms)} remaining")
```

Job crashed at 50%? Run again with `--resume`. It picks up where it left off. No wasted computation.

---

## Study-Level Aggregation

Radiologists think in studies, not individual images. A study might have 15 images, but they want one answer: "does this study contain a biopsy tool?"

```python
def aggregate_studies(db, threshold=0.5):
    """Aggregate image-level predictions to study-level results.
    
    Logic: If ANY image in a study has prob >= threshold, flag the study.
    We use MAX because biopsy markers might only appear in 2 of 15 images.
    Averaging would dilute the signal.
    """
    rows = list(db["predictions"].rows)
    
    # Group by study_id
    studies = defaultdict(list)
    for r in rows:
        studies[r["study_id"]].append((r["prob_biopsy"], r["prob_mag"]))
    
    results = []
    for study_id, preds in studies.items():
        max_biopsy = max(p[0] for p in preds)
        max_mag = max(p[1] for p in preds)
        
        results.append({
            'study_id': study_id,
            'has_biopsy_tool': 1 if max_biopsy >= threshold else 0,
            'has_mag_view': 1 if max_mag >= threshold else 0,
            'max_prob_biopsy': round(max_biopsy, 4),
            'max_prob_mag': round(max_mag, 4),
            'image_count': len(preds),
        })
    
    return sorted(results, key=lambda x: x['study_id'])
```

Why MAX instead of MEAN? A biopsy marker might only appear in 2 of 15 images. Averaging would dilute 0.95 confidence down to 0.13. MAX preserves the signal.

---

## Processing Pipeline

```python
def process_dicom(item, session):
    """Process one DICOM file, return list of (filename, study_id, frame_idx, probs).
    
    Handles multi-frame DICOMs by iterating through all frames.
    Returns empty list on failure (defensive - don't crash the batch).
    """
    dcm_path, study_id = item
    results = []
    
    try:
        ds = pydicom.dcmread(str(dcm_path))
        num_frames = get_frame_count(ds)
        
        for frame_idx in range(num_frames):
            filename = f"{study_id}/{dcm_path.stem}"
            if num_frames > 1:
                filename = f"{filename}_frame{frame_idx:04d}"
            
            img = dicom_to_pil(dcm_path, frame_idx, IMAGE_SIZE)
            if img is None: continue
            
            arr = preprocess(img)
            results.append((filename, study_id, frame_idx, arr))
            
    except Exception as e:
        logger.warning(f"Failed to process {dcm_path}: {e}")
    
    return results
```

Defensive programming: return empty list on failure, don't crash the batch. One corrupted DICOM shouldn't stop a 500K image job.

---

## Main Inference Loop

The loop accumulates preprocessed images into batches, runs inference when the batch is full, and saves results:

```python
batch_inputs = []   # preprocessed arrays
batch_meta = []     # (filename, study_id, frame_idx)

for dcm_path, study_id in progress_bar(dicoms):
    frames = process_dicom((dcm_path, study_id), session)
    
    for filename, sid, frame_idx, arr in frames:
        batch_inputs.append(arr)
        batch_meta.append((filename, sid, frame_idx))
        
        # When batch is full, run inference
        if len(batch_inputs) >= bs:
            probs = predict_batch(session, batch_inputs)
            
            # Save to database
            preds = [
                {
                    "filename": meta[0],
                    "study_id": meta[1],
                    "frame_number": meta[2],
                    "prob_biopsy": float(probs[i, 0]),
                    "prob_mag": float(probs[i, 1]),
                    "predicted_at": now,
                }
                for i, meta in enumerate(batch_meta)
            ]
            save_predictions(db, preds)
            
            # Clear batch
            batch_inputs = []
            batch_meta = []

# Don't forget the final partial batch!
if batch_inputs:
    # ... same logic
```

The final partial batch is easy to forget. If you have 1000 images and batch size 32, that's 31 full batches (992 images) plus one partial batch of 8. Without the final block, you'd lose those 8.

---

## CLI

```python
@call_parse
def main(
    input:Path=None,                         # Directory containing DICOM files
    db:Path=Path("results.db"),              # Results database path (PRIMARY output)
    output:Path=Path("results.csv"),         # Study-level CSV export (derived from db)
    model:Path=MODEL_DIR/"model.onnx",       # ONNX model path
    batch_size:int=BATCH_SIZE,               # Batch size for inference
    threshold:float=0.75,                    # Classification threshold
    test:bool_arg=False,                     # Use test directory
    resume:bool_arg=False,                   # Resume interrupted job
):
```

Usage examples:

```bash
# Basic: classify all DICOMs in a directory
python scripts/05_inference.py --input /path/to/dicoms

# Resume interrupted job
python scripts/05_inference.py --input /path/to/dicoms --resume

# Custom output paths
python scripts/05_inference.py --input /path/to/dicoms --db results.db --output results.csv

# Test mode with sample data
python scripts/05_inference.py --test
```

---

## Output Summary

The script prints a summary at the end:

```
============================================================
SUMMARY
============================================================
Images processed:        12,847
Total time:              45.2s
Throughput:              284.2 images/sec
Studies classified:      1,204
  With biopsy tool:      89
  With mag view:         156
  With both:             23
============================================================
```

Two artifacts:
1. **results.db** — Image-level predictions. Query it, resume from it, it's your source of truth.
2. **results.csv** — Study-level aggregation. Open in Excel, share with radiologists.

---

## The Production/Development Boundary

This script explicitly maintains separation:

```python
# Results database schema - separate from development database!
# This is for tracking inference progress and storing predictions on production data.
```

Your development database (`labels.db`) has your training labels, active learning state, hold-out assignments. Your production database (`results.db`) has predictions on new data.

Never mix them. The development database is for iteration. The production database is for deployment.

---

That's the inference pipeline. ONNX for portability, SQLite for resumability, study-level aggregation for clinical relevance. Standalone, fast, and completely separate from your training workflow.